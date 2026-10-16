"""Generate answers with any Hugging Face chat model and keep its last-answer-token hidden states.

Like ``hprobes extract`` for one model, but robust to chat templates without a system role (Gemma), to several
stop tokens (``<end_of_turn>``) and to models that need bfloat16 / eager attention (Gemma 2). Questions come from a
jsonl with ``id``, ``input_text`` and ``ground_truth_answers``. Writes in ``--output-dir``:
  states.npy    (N, layers + 1, hidden) float16, the last answer token's hidden state of every layer
  records.jsonl one line per question: input, generated_output, actual_output (list), id; read by ``hprobes predict``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hprobes.extract import DEFAULT_SYSTEM_PROMPT, chat_prompt, compute_dtype, release_gpu, stack_last_token_features
from hprobes.predict import read_records


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


def generate_with(model, tokenizer, records: Sequence[dict[str, Any]], device: str, system_prompt: str,
                  max_input_tokens: int = 512, max_new_tokens: int = 32, log_every: int = 1):
    """-> (states (N, layers + 1, hidden) float16, answer texts, the token each hidden state was taken from)."""
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    stops = stop_token_ids(tokenizer, model)
    trailing = set(stops) | {int(i) for i in getattr(tokenizer, "all_special_ids", [])}
    states, answers, last_tokens = [], [], []
    with torch.inference_mode():
        for i, record in enumerate(records):
            prompt = build_prompt(tokenizer, record["input_text"], system_prompt)
            # The template already contains special tokens such as <bos>; do not add them twice.
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens,
                                add_special_tokens=False).to(device)
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                       use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=stops)[0, encoded["input_ids"].shape[1]:]
            answer_ids = strip_trailing(generated, trailing)
            replay_ids = torch.cat([encoded["input_ids"][0], answer_ids]).unsqueeze(0)
            replay = model(replay_ids, attention_mask=torch.ones_like(replay_ids), output_hidden_states=True,
                           use_cache=False, return_dict=True)
            states.append(stack_last_token_features(replay.hidden_states, torch).numpy().astype(np.float16))
            answers.append(tokenizer.decode(answer_ids, skip_special_tokens=True).strip())
            last_tokens.append(tokenizer.decode(replay_ids[0, -1:], skip_special_tokens=False))
            if log_every and (i + 1) % log_every == 0:
                print(f"generate {i + 1}/{len(records)}: {record['input_text'][:60]!r} -> {answers[-1]!r}", flush=True)
    return np.stack(states), answers, last_tokens


def load_model(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    revision = args.model_revision or None
    config = AutoConfig.from_pretrained(args.model, revision=revision)
    gemma = str(getattr(config, "model_type", "")).startswith("gemma")
    dtype = torch.bfloat16 if gemma else compute_dtype(torch, args.device)  # Gemma 2 overflows in float16
    kwargs: dict[str, Any] = {"revision": revision, "torch_dtype": dtype, "low_cpu_mem_usage": True}
    if gemma:
        kwargs["attn_implementation"] = "eager"  # Gemma 2 logit soft-capping is not supported by sdpa
    if not args.no_4bit:
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True), device_map={"": args.device})
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.no_4bit:
        model = model.to(args.device)
    return model.eval(), tokenizer


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hub id or local folder of a causal chat model")
    parser.add_argument("--model-revision", default="", help="Hub revision (empty = default / local folder)")
    parser.add_argument("--questions", type=Path, required=True,
                        help="jsonl with id, input_text, ground_truth_answers per line")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N questions (0 = all)")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def run(args) -> np.ndarray:
    records = read_records(args.questions)
    records = records[:args.limit] if args.limit else records
    model, tokenizer = load_model(args)
    try:
        states, answers, last_tokens = generate_with(model, tokenizer, records, args.device, args.system_prompt,
                                                     args.max_input_tokens, args.max_new_tokens)
    finally:
        del model, tokenizer
        import torch

        release_gpu(torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "states.npy", states)
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record, answer, last_token in zip(records, answers, last_tokens):
            stream.write(json.dumps({"id": record.get("id"), "input": record["input_text"],
                                     "generated_output": answer, "last_token": last_token,
                                     "actual_output": record.get("ground_truth_answers")}, ensure_ascii=False) + "\n")
    print(f"Saved states {states.shape} and records to {args.output_dir}", flush=True)
    return states


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
