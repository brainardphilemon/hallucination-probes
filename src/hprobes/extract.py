"""Stage 1-2: generate answers + layer-wise hidden states with an LLM, then BLEURT-20 labels.

Artifacts are written as::

    <artifact-dir>/generation/<domain>/{rows.jsonl, features.npy, manifest.json}
    <artifact-dir>/bleurt/<domain>/{scores.jsonl, manifest.json}

``features.npy`` has shape (rows, num_layers + 1, hidden_size): the hidden state of the last
non-EOS answer token at the embedding output and after every transformer layer. Both stages
resume from partial outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np

from hprobes.domains import DATASET_SPECS, DOMAINS, PRESETS, load_domain_records
from hprobes.io import prepare_manifest, read_jsonl, sha256_json

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_SYSTEM_PROMPT = "Answer the question with only a short factual answer. Do not explain your answer."
DEFAULT_BLEURT_MODEL = "lucadiliello/BLEURT-20"
DEFAULT_BLEURT_REVISION = "e22b9eb071dcb939da9fff08b20a7e746c72a830"


def chat_prompt(tokenizer, question: str, system_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def compute_dtype(torch, device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16


def release_gpu(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generation_manifest(
    domain: str,
    records: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    num_layers: int,
    hidden_size: int,
) -> dict[str, Any]:
    spec = DATASET_SPECS[domain]
    return {
        "schema_version": 1,
        "domain": domain,
        "dataset": spec,
        "selected_ids_sha256": sha256_json([row["id"] for row in records]),
        "selected_count": len(records),
        "model": args.model,
        "model_revision": args.model_revision,
        "four_bit": not args.no_4bit,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "system_prompt": args.system_prompt,
        "feature_definition": "last non-EOS answer token hidden state at embedding output plus every transformer layer",
        "feature_shape": [len(records), num_layers + 1, hidden_size],
        "feature_dtype": "float16",
    }


def trim_terminal_tokens(generated_ids, tokenizer):
    terminal = {value for value in [tokenizer.eos_token_id, tokenizer.pad_token_id] if value is not None}
    end = len(generated_ids)
    while end and int(generated_ids[end - 1]) in terminal:
        end -= 1
    return generated_ids[:end]


def build_prompt(tokenizer, question: str, system_prompt: str) -> str:
    """Chat prompt with the system instruction; templates that reject a system role get it prepended to the user turn."""
    try:
        return chat_prompt(tokenizer, question, system_prompt)
    except Exception:
        return tokenizer.apply_chat_template([{"role": "user", "content": f"{system_prompt}\n\n{question}"}],
                                             tokenize=False, add_generation_prompt=True)


END_OF_TURN_TOKENS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>", "</s>")


def stop_token_ids(tokenizer, model) -> list[int]:
    """Ids that end an answer: the model's configured EOS ids, the tokenizer's EOS / pad, and known end-of-turn
    markers present in the vocabulary (Gemma's <end_of_turn> is often missing from generation_config)."""
    ids: set[int] = set()
    configured = getattr(model.generation_config, "eos_token_id", None)
    for value in ([configured] if isinstance(configured, int) else list(configured or [])):
        ids.add(int(value))
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if value is not None:
            ids.add(int(value))
    convert, unk = getattr(tokenizer, "convert_tokens_to_ids", None), getattr(tokenizer, "unk_token_id", None)
    for name in END_OF_TURN_TOKENS if convert else ():
        value = convert(name)
        if isinstance(value, int) and value >= 0 and value != unk:
            ids.add(value)
    return sorted(ids)


def strip_trailing(ids, drop: set[int]):
    """Remove trailing end / special tokens so the kept hidden state is the answer's last real token."""
    end = len(ids)
    while end and int(ids[end - 1]) in drop:
        end -= 1
    return ids[:end]


def answer_token_ids(generated, tokenizer, stops):
    """Generated ids without trailing stop / special tokens or whitespace-only tokens (Gemma: 'Golf.' '\\n'
    '<end_of_turn>' '<eos>'), so the replayed hidden state belongs to the answer's last real token."""
    drop = set(stops) | {int(i) for i in getattr(tokenizer, "all_special_ids", [])}
    answer_ids = strip_trailing(generated, drop)
    while len(answer_ids) > 1 and not tokenizer.decode(answer_ids[-1:], skip_special_tokens=True).strip():
        answer_ids = answer_ids[:-1]
    return answer_ids


def model_load_settings(args):
    """(revision, from_pretrained kwargs): 4-bit NF4 unless --no-4bit; Gemma gets bfloat16 + eager attention
    (float16 overflows and sdpa lacks its logit soft-capping); a local model folder ignores the hub revision."""
    import torch
    from transformers import AutoConfig, BitsAndBytesConfig

    revision = None if (not args.model_revision or Path(args.model).expanduser().is_dir()) else args.model_revision
    config = AutoConfig.from_pretrained(args.model, revision=revision, cache_dir=getattr(args, "cache_dir", None))
    gemma = str(getattr(config, "model_type", "")).startswith("gemma")
    dtype = torch.bfloat16 if gemma else compute_dtype(torch, args.device)
    kwargs = {"revision": revision, "cache_dir": getattr(args, "cache_dir", None), "torch_dtype": dtype,
              "low_cpu_mem_usage": True}
    if gemma:
        kwargs["attn_implementation"] = "eager"
    if not args.no_4bit:
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True), device_map={"": args.device})
    return revision, kwargs


def stack_last_token_features(hidden_states, torch):
    """Stack embedding/output-layer states at the final replayed answer token."""
    return torch.stack([hidden[0, -1].float().cpu() for hidden in hidden_states], dim=0)


def ensure_generation(
    records_by_domain: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Kaggle GPU accelerator")
    if not args.no_4bit and not str(args.device).startswith("cuda"):
        raise ValueError("4-bit bitsandbytes loading requires CUDA; use --no-4bit for CPU")
    revision, load_kwargs = model_load_settings(args)
    config = AutoConfig.from_pretrained(args.model, revision=revision, cache_dir=args.cache_dir)
    incomplete: list[str] = []
    for domain, records in records_by_domain.items():
        manifest = generation_manifest(
            domain, records, args, config.num_hidden_layers, config.hidden_size
        )
        manifest_path = output_dir / "generation" / domain / "manifest.json"
        prepare_manifest(manifest_path, manifest)
        rows = read_jsonl(output_dir / "generation" / domain / "rows.jsonl")
        expected_ids = {str(row["id"]) for row in records}
        stored_ids = {str(row["id"]) for row in rows}
        if not stored_ids.issubset(expected_ids):
            raise ValueError(f"Stored rows do not belong to the selected {domain} cohort")
        feature_path = output_dir / "generation" / domain / "features.npy"
        if len(rows) < len(records):
            incomplete.append(domain)
        elif len(rows) > len(records):
            raise ValueError(f"Too many generated rows for {domain}")
        elif not feature_path.is_file():
            raise FileNotFoundError(f"Completed row log exists but features are missing: {feature_path}")
    if not incomplete:
        print("Generation/features already complete", flush=True)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=args.cache_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.no_4bit:
        model = model.to(args.device)
    model.eval()
    stops = stop_token_ids(tokenizer, model)
    try:
        for domain in incomplete:
            records = records_by_domain[domain]
            domain_dir = output_dir / "generation" / domain
            rows_path = domain_dir / "rows.jsonl"
            feature_path = domain_dir / "features.npy"
            saved_rows = read_jsonl(rows_path)
            completed = {str(row["id"]) for row in saved_rows}
            expected_ids = {str(row["id"]) for row in records}
            if not completed.issubset(expected_ids):
                raise ValueError(f"Stored rows do not belong to the selected {domain} cohort")
            shape = (len(records), model.config.num_hidden_layers + 1, model.config.hidden_size)
            if feature_path.exists():
                features = np.lib.format.open_memmap(feature_path, mode="r+")
                if features.shape != shape or features.dtype != np.float16:
                    raise ValueError(f"Feature array mismatch for {domain}: {features.shape}/{features.dtype}")
            else:
                if saved_rows:
                    raise FileNotFoundError(f"{feature_path} is missing but resumable rows exist")
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=shape)
            with rows_path.open("a", encoding="utf-8") as stream, torch.inference_mode():
                for position, record in enumerate(records):
                    example_id = str(record["id"])
                    if example_id in completed:
                        continue
                    prompt = build_prompt(tokenizer, record["input_text"], args.system_prompt)
                    # Chat templates already contain their special tokens (Gemma's <bos>); do not add them twice.
                    encoded = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=args.max_input_tokens,
                        add_special_tokens=False,
                    ).to(args.device)
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=stops,
                    )[0, encoded["input_ids"].shape[1]:]
                    answer_ids = answer_token_ids(generated, tokenizer, stops)
                    replay_ids = torch.cat([encoded["input_ids"][0], answer_ids]).unsqueeze(0)
                    replay = model(
                        replay_ids,
                        attention_mask=torch.ones_like(replay_ids),
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    layerwise = stack_last_token_features(replay.hidden_states, torch).numpy().astype(np.float16)
                    features[position] = layerwise
                    features.flush()
                    row = {
                        "schema_version": 1,
                        "id": example_id,
                        "position": position,
                        "domain": domain,
                        "input_text": record["input_text"],
                        "model_input_text": prompt,
                        "output_text": tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
                        "ground_truth_answers": record["ground_truth_answers"],
                        "input_length": int(encoded["input_ids"].shape[1]),
                        "output_length": int(len(answer_ids)),
                        "feature_row": position,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed.add(example_id)
                    del generated, answer_ids, replay_ids, replay, layerwise, encoded
                    print(f"generate {domain} {position + 1}/{len(records)} id={example_id}", flush=True)
            del features
    finally:
        del model, tokenizer
        release_gpu(torch)


class Bleurt20Scorer:
    def __init__(self, args: argparse.Namespace):
        import torch
        from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
        from huggingface_hub import hf_hub_download

        if str(args.bleurt_device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("BLEURT CUDA device requested but unavailable")
        self.torch = torch
        self.device = args.bleurt_device
        self.batch_size = args.bleurt_batch_size
        self.model_name = args.bleurt_model
        self.revision = args.bleurt_revision
        self.tokenizer = BleurtTokenizer.from_pretrained(
            args.bleurt_model, revision=args.bleurt_revision, cache_dir=args.cache_dir
        )
        config = BleurtConfig.from_pretrained(
            args.bleurt_model, revision=args.bleurt_revision, cache_dir=args.cache_dir
        )
        local_model = Path(args.bleurt_model).expanduser()
        if local_model.is_dir():
            weights_path = local_model / "pytorch_model.bin"
        else:
            weights_path = Path(hf_hub_download(
                repo_id=args.bleurt_model,
                filename="pytorch_model.bin",
                revision=args.bleurt_revision,
                cache_dir=args.cache_dir,
            ))
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError("BLEURT-20 requires a PyTorch version with weights_only support") from exc
        self.model = BleurtForSequenceClassification(config)
        self.model.load_state_dict(state_dict, strict=True)
        del state_dict
        self.model.eval()
        if str(self.device).startswith("cuda"):
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

    def close(self) -> None:
        del self.model, self.tokenizer
        release_gpu(self.torch)

    def score_rows(self, rows: Sequence[dict[str, Any]], cache_path: Path) -> list[dict[str, Any]]:
        cached_rows = read_jsonl(cache_path)
        cached = {str(row["id"]): row for row in cached_rows}
        expected = {str(row["id"]) for row in rows}
        if not set(cached).issubset(expected):
            raise ValueError(f"Cached BLEURT rows do not belong to {cache_path}")
        if len(cached) == len(rows):
            return [cached[str(row["id"])] for row in rows]
        references: list[str] = []
        candidates: list[str] = []
        owners: list[tuple[int, int]] = []
        for row_index, row in enumerate(rows):
            if str(row["id"]) in cached:
                continue
            aliases = row.get("ground_truth_answers") or []
            if not aliases:
                raise ValueError(f"Row {row.get('id')} has no references")
            for reference_index, reference in enumerate(aliases):
                references.append(str(reference))
                candidates.append(str(row["output_text"]))
                owners.append((row_index, reference_index))
        last_pair: dict[int, int] = {}
        for pair_index, (row_index, _) in enumerate(owners):
            last_pair[row_index] = pair_index
        best: dict[int, tuple[float, str, int]] = {}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as stream, self.torch.inference_mode():
            for start in range(0, len(references), self.batch_size):
                stop = min(start + self.batch_size, len(references))
                encoded = self.tokenizer(
                    references[start:stop],
                    candidates[start:stop],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                scores = self.model(**encoded).logits.reshape(-1).float().cpu().tolist()
                for pair_index, score_value in enumerate(scores, start):
                    score = float(score_value)
                    row_index, reference_index = owners[pair_index]
                    if not math.isfinite(score):
                        raise ValueError(f"Non-finite BLEURT score for {rows[row_index]['id']}")
                    reference = str(rows[row_index]["ground_truth_answers"][reference_index])
                    current = best.get(row_index)
                    if current is None or score > current[0]:
                        best[row_index] = (score, reference, reference_index)
                    if pair_index == last_pair[row_index]:
                        selected = best[row_index]
                        saved = {
                            "id": str(rows[row_index]["id"]),
                            "bleurt_score": selected[0],
                            "ground_truth_text": selected[1],
                            "ground_truth_reference_index": selected[2],
                            "bleurt_model": self.model_name,
                            "bleurt_revision": self.revision,
                        }
                        stream.write(json.dumps(saved, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        cached[saved["id"]] = saved
                print(f"BLEURT {cache_path.parent.name} pairs {stop}/{len(references)}", flush=True)
        if len(cached) != len(rows):
            raise RuntimeError(f"BLEURT did not score every row for {cache_path}")
        return [cached[str(row["id"])] for row in rows]


def ensure_bleurt(domains: Sequence[str], args: argparse.Namespace, output_dir: Path) -> None:
    pending: list[tuple[str, list[dict[str, Any]]]] = []
    for domain in domains:
        rows = read_jsonl(output_dir / "generation" / domain / "rows.jsonl")
        manifest = {
            "schema_version": 1,
            "domain": domain,
            "cohort_sha256": sha256_json([row["id"] for row in rows]),
            "bleurt_model": args.bleurt_model,
            "bleurt_revision": args.bleurt_revision,
            "precision": "float16" if str(args.bleurt_device).startswith("cuda") else "float32",
            "reference_policy": "maximum score over all aliases",
        }
        prepare_manifest(output_dir / "bleurt" / domain / "manifest.json", manifest)
        scored = read_jsonl(output_dir / "bleurt" / domain / "scores.jsonl")
        expected_ids = {str(row["id"]) for row in rows}
        scored_ids = {str(row["id"]) for row in scored}
        if not scored_ids.issubset(expected_ids):
            raise ValueError(f"BLEURT rows do not belong to the generated {domain} cohort")
        if len(scored) < len(rows):
            pending.append((domain, rows))
        elif len(scored) > len(rows):
            raise ValueError(f"Too many BLEURT rows for {domain}")
    if not pending:
        print("BLEURT scoring already complete", flush=True)
        return
    scorer = Bleurt20Scorer(args)
    try:
        for domain, rows in pending:
            scorer.score_rows(rows, output_dir / "bleurt" / domain / "scores.jsonl")
    finally:
        scorer.close()


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Where generation/ and bleurt/ artifacts are written (resumable)")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS,
                        help="Datasets to process (default: all domains used by --preset)")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Process every domain of a preset")
    parser.add_argument("--stage", choices=("all", "generate", "bleurt"), default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--no-4bit", action="store_true", help="Load the LLM without 4-bit quantization")
    parser.add_argument("--bleurt-model", default=DEFAULT_BLEURT_MODEL)
    parser.add_argument("--bleurt-revision", default=DEFAULT_BLEURT_REVISION)
    parser.add_argument("--samples-per-domain", type=int, default=0,
                        help="Deterministic sample count per domain; 0 uses each full split")
    parser.add_argument("--questions-from", type=Path,
                        help="Take the questions from an existing artifact folder (its generation/<domain>/rows.jsonl, "
                             "first --samples-per-domain rows) instead of downloading the datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bleurt-device", default="cuda:0")
    parser.add_argument("--bleurt-batch-size", type=int, default=16)
    parser.add_argument("--cache-dir")
    return parser


def saved_questions(artifact_dir: Path, domain: str, limit: int = 0) -> list[dict[str, Any]]:
    """Questions (id, input_text, references) in the order they are stored in an existing artifact folder, so a new
    model answers exactly the questions an earlier extraction used; the first ``limit`` rows if limit > 0."""
    rows = read_jsonl(Path(artifact_dir) / "generation" / domain / "rows.jsonl")
    if not rows:
        raise SystemExit(f"No saved questions for {domain} under {artifact_dir}")
    rows = rows[:limit] if limit else rows
    return [{"id": str(r["id"]), "input_text": r["input_text"], "ground_truth_answers": r["ground_truth_answers"]}
            for r in rows]


def run(args: argparse.Namespace) -> None:
    domains = list(args.domains or [])
    if args.preset:
        preset = PRESETS[args.preset]
        domains += [d for d in [*preset["sources"], *preset["targets"]] if d not in domains]
    if not domains:
        raise SystemExit("Choose --domains and/or --preset")
    if args.samples_per_domain < 0 or args.max_new_tokens < 2 or args.max_input_tokens < 1:
        raise SystemExit("Invalid sample count or token limits")
    output_dir = args.artifact_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "generate"):
        if args.questions_from:
            records = {d: saved_questions(args.questions_from, d, args.samples_per_domain) for d in domains}
        else:
            records = {domain: load_domain_records(domain, args) for domain in domains}
        ensure_generation(records, args, output_dir)
    if args.stage in ("all", "bleurt"):
        ensure_bleurt(domains, args, output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
