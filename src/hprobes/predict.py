"""Inference: score new (hidden states, input, generated output[, actual output]) examples.

``hprobes train --method mlp ...`` writes ``<output-dir>/model.pt`` containing the exact preprocessing
(layer pooling, PCA fitted on source-train), the trained MLP, and the calibrated decision threshold.
The detector reads only the hidden states; texts are for reporting and for the optional BLEURT check
of the generated output against the actual output.

Hidden states must be the last generated-answer-token hidden states of every layer (embedding output
plus all blocks), shape (layers, hidden) per example, from the same LLM the detector was trained on
(Qwen2.5-7B-Instruct: 29 x 3584). Label convention: 1 = correct, 0 = hallucinated.

Python use::

    det = Detector.load("runs/mlp/model.pt")
    out = det.predict(states)               # (N, 29, 3584) or (29, 3584)
    out["probability_correct"], out["hallucinated"]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

BUNDLE_FORMAT = 1
# Accepted field names for the text columns of --records (first match wins).
FIELDS = {
    "input": ("input", "input_text", "question", "prompt"),
    "generated": ("generated_output", "generated", "output_text", "answer"),
    "actual": ("actual_output", "actual", "ground_truth_text", "ground_truth_answers", "reference"),
}


def save_bundle(path: Path, exported: dict, summary: dict) -> None:
    import torch

    bundle = {
        "format": BUNDLE_FORMAT, **exported,
        "threshold": float(summary["threshold"]),
        "label_definition": summary["label_definition"],
        "bleurt_threshold": summary["bleurt_threshold"],
        "sources": summary["sources"], "targets": summary["targets"],
        "loss": summary["config"]["loss"],
        "target_metrics": summary["target_metrics"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


class Detector:
    """Pooling -> per-LLM PCA -> shared residual MLP -> probability that the answer is correct.

    A bundle trained on several LLMs holds one PCA per LLM; ``predict`` picks the PCA whose
    (layers, hidden size) matches the input, or the one named by ``llm=``.
    """

    def __init__(self, bundle: dict, device: str = "cpu"):
        import torch

        from hprobes.features import LayerPCA
        from hprobes.training import residual_mlp

        if bundle.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"Unsupported model bundle format {bundle.get('format')}")
        self.bundle = bundle
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.threshold = float(bundle["threshold"])
        self.adaptive = bundle.get("pca_fit") == "self"  # PCA is recomputed from the input itself
        self.fixed = bundle.get("pca_fit") in ("avg", "dct")  # data-free reduction of each example
        models = bundle.get("models") or {"default": {"n_layers": bundle["n_layers"], "hidden_size": bundle["hidden_size"],
                                                      "pca": bundle["pca"]}}
        self.llms = {name: (int(m["n_layers"]), int(m["hidden_size"]),
                            LayerPCA.from_state(m["pca"]) if m["pca"] else None) for name, m in models.items()}
        self.standardizers = {name: (m["standardize"]["mean"].float(), m["standardize"]["std"].float())
                              for name, m in models.items() if m.get("standardize")}
        head = bundle["head"]
        self.model = residual_mlp(head["input_dim"], head["hidden_dim"], head["hidden_layers"], head["dropout"])
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device).eval()

    @classmethod
    def load(cls, path, device: str = "cpu") -> "Detector":
        import torch

        return cls(torch.load(Path(path), map_location="cpu", weights_only=False), device)

    def resolve(self, shape: tuple[int, int], llm: str | None = None, any_layers: bool = False) -> str:
        if any_layers:  # layer count is adapted by pooling; only the hidden size has to match
            names = [n for n, (_, hidden, _) in self.llms.items() if hidden == shape[1] and llm in (None, n)]
            if len(names) != 1:
                known = {n: f"hidden {h}" for n, (_, h, _) in self.llms.items()}
                raise ValueError(f"No single LLM in this detector has hidden size {shape[1]}; it knows {known}")
            return names[0]
        if llm is not None:
            if llm not in self.llms:
                raise ValueError(f"Unknown LLM {llm!r}; this detector knows {sorted(self.llms)}")
            if self.llms[llm][:2] != tuple(shape):
                raise ValueError(f"{llm!r} expects hidden states of shape (N, {self.llms[llm][0]}, "
                                 f"{self.llms[llm][1]}); got (N, {shape[0]}, {shape[1]})")
            return llm
        matches = [name for name, (layers, hidden, _) in self.llms.items() if (layers, hidden) == tuple(shape)]
        if len(matches) == 1:
            return matches[0]
        known = {name: f"({layers}, {hidden})" for name, (layers, hidden, _) in self.llms.items()}
        if not matches:
            raise ValueError(f"No LLM in this detector has hidden states of shape (layers, hidden) = {tuple(shape)}; "
                             f"it knows {known}")
        raise ValueError(f"Several LLMs share shape {tuple(shape)} ({matches}); pass llm=<name>")

    def features(self, states, batch_size: int = 256, llm: str | None = None, any_layers: bool = False,
                 standardize_on_input: bool = False, return_pooled: bool = False):
        """The MLP's input for these hidden states: pooling -> PCA / reduction (if any) -> standardization (if any).

        standardize_on_input z-scores with the mean / std of this batch itself instead of the stored training
        statistics; use it for an LLM the detector was not trained on (needs many examples).
        """
        import torch

        from hprobes.features import feature_moments, pool_layers, standardize_

        x = torch.as_tensor(np.asarray(states))
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected hidden states of shape (N, layers, hidden) or (layers, hidden); "
                             f"got {tuple(x.shape)}")
        b = self.bundle
        # A detector trained with --pool-layers N pools any layer count down to N, so layers never have to match.
        any_layers = any_layers or bool(b.get("pool_layers", 0))
        pca, out_layers, name = None, b.get("pool_layers", 0), None
        if not (self.adaptive or self.fixed):
            name = self.resolve(tuple(x.shape[1:]), llm, any_layers)
            pca = self.llms[name][2]
        if any_layers:
            # Pool the input's layers adaptively to the count this detector's PCA / MLP was trained on.
            if b["pooling"] == "none":
                raise ValueError("--any-layers needs a detector trained with layer pooling")
            hidden = x.shape[2]
            out_layers = (b["head"]["input_dim"] // hidden if pca is None and not (self.adaptive or self.fixed)
                          else len(pca.bases) if pca and pca.scope == "per-layer"
                          else pca.bases[0].shape[0] // hidden if pca else out_layers)

        def pooled(chunk):
            return pool_layers(chunk.to(self.device).float(), b["pooling"], b["pool_kernel"], b["pool_stride"],
                               out_layers)

        feats, pooled_parts = [], []
        with torch.inference_mode():
            if self.adaptive:
                if len(x) < 2:
                    raise ValueError("This detector computes its PCA from the examples you give it, so it needs "
                                     "a batch of at least 2 (ideally more than the PCA dimension)")
                from hprobes.features import self_pca

                whole = torch.cat([pooled(x[s:s + batch_size]).cpu() for s in range(0, len(x), batch_size)])
                feats = [self_pca(whole, b["pca_dim"], b["pca_scope"], b["pca_seed"], self.device).flatten(1)]
                pooled_parts = [whole]
            else:
                for start in range(0, len(x), batch_size):
                    block = pooled(x[start:start + batch_size])
                    if return_pooled:
                        pooled_parts.append(block.cpu())
                    if self.fixed:
                        from hprobes.features import fixed_reduce

                        f = fixed_reduce(block.cpu(), b["pca_dim"], b["pca_fit"], b["pca_scope"],
                                         b.get("reduce_norm", "rms"), self.device)
                    else:
                        f = pca.transform(block, self.device) if pca else block.cpu()
                    feats.append(f.flatten(1).float().cpu())
            features = torch.cat(feats).clone()
        if standardize_on_input:
            standardize_(features, *feature_moments([features]))
        else:
            stats = self.standardizers.get(name) if name is not None else (
                next(iter(self.standardizers.values())) if len(self.standardizers) == 1 else None)
            if stats is not None:
                standardize_(features, *stats)
        if return_pooled:
            return features, torch.cat(pooled_parts)
        return features

    def logits(self, states, batch_size: int = 256, llm: str | None = None, any_layers: bool = False,
               standardize_on_input: bool = False) -> np.ndarray:
        import torch

        features = self.features(states, batch_size, llm, any_layers, standardize_on_input)
        with torch.inference_mode():
            out = [self.model(features[s:s + batch_size].to(self.device)).float().cpu()
                   for s in range(0, len(features), batch_size)]
        return torch.cat(out).numpy()

    def predict(self, states, batch_size: int = 256, llm: str | None = None, any_layers: bool = False,
                standardize_on_input: bool = False) -> dict[str, np.ndarray]:
        logits = self.logits(states, batch_size, llm, any_layers, standardize_on_input)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        correct = probability >= self.threshold
        return {"raw_logit": logits, "probability_correct": probability,
                "probability_hallucinated": 1.0 - probability,
                "predicted_label": correct.astype(np.int64), "hallucinated": ~correct}


def load_states(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        archive = np.load(path)
        key = "states" if "states" in archive else archive.files[0]
        return archive[key]
    if path.suffix in (".pt", ".pth"):
        import torch

        return torch.load(path, map_location="cpu").numpy()
    raise ValueError(f"Unsupported hidden-state file {path} (use .npy, .npz or .pt)")


def read_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def pick(row: dict[str, Any], kind: str):
    for name in FIELDS[kind]:
        if row.get(name) not in (None, ""):
            return row[name]
    return None


def as_references(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return [value]
        else:
            return [value]
    return [str(v) for v in value]


def bleurt_scores(records: Sequence[dict[str, Any]], args) -> list[float | None]:
    """Best BLEURT-20 score of the generated output against any actual output; None if missing."""
    from hprobes.extract import Bleurt20Scorer

    scorer = Bleurt20Scorer(SimpleNamespace(
        bleurt_device=args.device, bleurt_batch_size=args.bleurt_batch_size, bleurt_model=args.bleurt_model,
        bleurt_revision=args.bleurt_revision, cache_dir=None))
    pairs, owners = [], []
    for i, row in enumerate(records):
        generated = pick(row, "generated")
        for reference in as_references(pick(row, "actual")):
            if generated is not None:
                pairs.append((reference, str(generated)))
                owners.append(i)
    scores: list[float | None] = [None] * len(records)
    with scorer.torch.inference_mode():
        for start in range(0, len(pairs), scorer.batch_size):
            chunk = pairs[start:start + scorer.batch_size]
            encoded = scorer.tokenizer([r for r, _ in chunk], [g for _, g in chunk], padding=True,
                                       truncation=True, max_length=512, return_tensors="pt").to(scorer.device)
            for owner, value in zip(owners[start:start + len(chunk)],
                                    scorer.model(**encoded).logits.reshape(-1).float().cpu().tolist()):
                scores[owner] = value if scores[owner] is None else max(scores[owner], value)
    scorer.close()
    return scores


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    from hprobes.extract import DEFAULT_BLEURT_MODEL, DEFAULT_BLEURT_REVISION

    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="model.pt written by `hprobes train`")
    parser.add_argument("--states", type=Path, required=True,
                        help=".npy/.npz/.pt hidden states, (N, layers, hidden) or (layers, hidden)")
    parser.add_argument("--records", type=Path,
                        help="jsonl/json/csv, one row per example, same order as --states. Fields: "
                             "input (input_text/question), generated_output (output_text), "
                             "actual_output (ground_truth_text / ground_truth_answers)")
    parser.add_argument("--output", type=Path, help="Write predictions to .csv or .jsonl")
    parser.add_argument("--bleurt", action="store_true",
                        help="Also score generated vs actual output with BLEURT-20 and compare labels")
    parser.add_argument("--bleurt-model", default=DEFAULT_BLEURT_MODEL)
    parser.add_argument("--bleurt-revision", default=DEFAULT_BLEURT_REVISION)
    parser.add_argument("--bleurt-batch-size", type=int, default=16)
    parser.add_argument("--llm", help="Which LLM's PCA to use; default: the one matching the shape of --states")
    parser.add_argument("--any-layers", action="store_true",
                        help="Accept a different number of layers than the detector was trained on: they are pooled "
                             "adaptively to its pooled layer count (hidden size must still match)")
    parser.add_argument("--standardize-on-input", action="store_true",
                        help="Standardize the MLP inputs with the statistics of --states itself (for an LLM the detector "
                             "was not trained on); give many examples")
    parser.add_argument("--threshold", type=float, help="Override the saved decision threshold on P(correct)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser


def run(args) -> list[dict[str, Any]]:
    detector = Detector.load(args.model, args.device)
    if args.threshold is not None:
        detector.threshold = args.threshold
    states = load_states(args.states)
    if np.ndim(states) == 2:
        states = np.asarray(states)[None]
    prediction = detector.predict(states, llm=args.llm, any_layers=args.any_layers,
                                  standardize_on_input=args.standardize_on_input)
    count = len(prediction["raw_logit"])
    records = read_records(args.records) if args.records else [{} for _ in range(count)]
    if len(records) != count:
        raise SystemExit(f"{len(records)} records but {count} hidden-state rows")
    bleurt = bleurt_scores(records, args) if args.bleurt else [None] * count
    threshold = float(detector.bundle["bleurt_threshold"])
    rows = []
    for i, record in enumerate(records):
        row = {"index": i, "input": pick(record, "input"), "generated_output": pick(record, "generated"),
               "actual_output": pick(record, "actual"),
               "probability_correct": float(prediction["probability_correct"][i]),
               "probability_hallucinated": float(prediction["probability_hallucinated"][i]),
               "predicted_label": int(prediction["predicted_label"][i]),
               "verdict": "hallucinated" if prediction["hallucinated"][i] else "correct"}
        if bleurt[i] is not None:
            row["bleurt_score"] = bleurt[i]
            row["bleurt_label"] = int(bleurt[i] >= threshold)
            row["detector_agrees_with_bleurt"] = row["bleurt_label"] == row["predicted_label"]
        rows.append(row)
    for row in rows[:20]:
        extra = f"  bleurt={row['bleurt_score']:.3f}" if "bleurt_score" in row else ""
        print(f"[{row['index']}] P(correct)={row['probability_correct']:.3f} -> {row['verdict']}{extra}")
    if len(rows) > 20:
        print(f"... {len(rows) - 20} more rows")
    if "bleurt_score" in rows[0]:
        agree = np.mean([r["detector_agrees_with_bleurt"] for r in rows if "bleurt_score" in r])
        print(f"Agreement with BLEURT labels: {agree:.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".csv":
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                   encoding="utf-8")
        print(f"Saved {len(rows)} predictions to {args.output}")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
