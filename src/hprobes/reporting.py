"""Turn validation / test / target logits into thresholds, metrics and result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hprobes.io import csv_dump_atomic, json_dump_atomic, jsonl_dump_atomic
from hprobes.metrics import binary_metrics, calibrate_threshold


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


def save_results(args, data, method_summary: dict[str, Any], val_logits: np.ndarray,
                 test_logits: np.ndarray, target_logits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Calibrate the threshold on source validation, score source test and every target, write files."""
    y_val, y_test = data.labels("val"), data.labels("test")
    val_scores, test_scores = sigmoid(val_logits), sigmoid(test_logits)
    if args.loss == "logit_adjusted":
        threshold = 0.5
        calibration = {"objective": "raw logit >= 0 (logit-adjusted training, Menon et al. Eq. 10)"}
    else:
        threshold, calibration = calibrate_threshold(val_scores.tolist(), y_val.tolist())
    results, target_metrics = [], {}
    for domain in data.targets:
        logits = target_logits[domain.name]
        scores = sigmoid(logits)
        predictions = (scores >= threshold).astype(np.int64)
        target_metrics[domain.name] = binary_metrics(domain.labels.tolist(), predictions.tolist(), scores.tolist())
        for row, label, logit, score, predicted in zip(domain.rows, domain.labels, logits, scores, predictions):
            judged = domain.bleurt[str(row["id"])]
            results.append({**row, "method": args.method, "source_domains": [d.name for d in data.sources],
                            "target_domain": domain.name, "ground_truth_text": judged["ground_truth_text"],
                            "ground_truth_label": int(label), "bleurt_score": float(judged["bleurt_score"]),
                            "raw_logit": float(logit), "sala_probability": float(score),
                            "predicted_label": int(predicted), "classifier_loss": args.loss,
                            "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "method": args.method,
        "sources": [d.name for d in data.sources],
        "targets": [d.name for d in data.targets],
        "num_layers_including_embedding_output": data.n_layers,
        "hidden_size": data.hidden_size,
        "llm_shapes": {name: list(shape) for name, shape in data.models.items()},
        "label_definition": "1 = BLEURT-correct/non-hallucinated (score >= threshold); 0 = otherwise",
        "bleurt_threshold": args.bleurt_threshold,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **method_summary,
        "threshold": threshold,
        "threshold_calibration": calibration,
        "source_test_metrics": binary_metrics(y_test.tolist(), (test_scores >= threshold).astype(int).tolist(),
                                              test_scores.tolist()),
        "target_metrics": target_metrics,
        "target_supervision_used": False,
        "source_counts": data.source_counts(),
    }
    output_dir = Path(args.output_dir)
    jsonl_dump_atomic(output_dir / "results.jsonl", results)
    csv_dump_atomic(output_dir / "results.csv", results)
    json_dump_atomic(output_dir / "summary.json", summary)
    headline = {name: round(m["auroc"], 4) if m.get("auroc") is not None else None
                for name, m in target_metrics.items()}
    print(json.dumps({"method": args.method, "best_source_val_auroc": method_summary.get("best_source_val_auroc"),
                      "source_test_auroc": summary["source_test_metrics"].get("auroc"),
                      "target_auroc": headline}, indent=2), flush=True)
    print(f"Saved {len(results)} target rows to {output_dir / 'results.csv'}", flush=True)
    return summary
