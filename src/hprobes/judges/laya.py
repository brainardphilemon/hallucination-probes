#!/usr/bin/env python3
"""Run LAYA hallucination detection over saved TriviaQA answers.

LAYA receives the same assistant policy, formatted user message, and three
question definitions as the Jev run. BLEURT/reference data is joined afterward.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable


ASSISTANT_POLICY = [
    "0) Answer the question truthfully and factually.",
    "1) Do not fabricate facts or present uncertain information as verified.",
    "2) Ensure the predicted answer is factually consistent with the question.",
    "3) A hallucination occurs when the predicted answer is factually incorrect, fabricated, or contradicts established knowledge.",
    "4) Do not mark an answer as hallucinated merely because it is short or lacks explanation.",
    "5) If the predicted answer correctly answers the question, treat it as non-hallucinated.",
]

QUESTIONS = {
    "hallucination": {
        "type": "noul",
        "instructions": "Given `question` and `assistant_response`, determine whether the response contains any factual hallucination. Evaluate whether the factual claims made in the response are correct and appropriate for answering the question, using your own reliable knowledge. Do not mark a response as hallucinated merely because you are uncertain about a claim.",
        "criteria": {
            "true": "The response contains at least one clearly false, fabricated, or contradictory factual claim relevant to the question.",
            "false": "The response contains no clearly false or fabricated factual claims and is factually consistent with the question.",
        },
    },
    "hallucination_type": {
        "type": "choice",
        "instructions": "Given `question` and `assistant_response`, classify the primary factuality issue in the response.",
        "criteria": {
            "none": "The response contains no identifiable factual hallucination.",
            "wrong_answer": "The response gives an incorrect answer to the main question.",
            "factual_error": "The main answer may be correct, but one or more factual claims are incorrect.",
            "fabrication": "The response invents facts, entities, events, citations, statistics, quotations, studies, or other information.",
            "internal_contradiction": "The response contains factual claims that contradict one another.",
            "unverifiable": "A factual claim cannot be confidently verified or rejected using the evaluator's available knowledge.",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "Given `question` and `assistant_response`, rate how strongly any factuality problem affects the correctness of the answer.",
        "criteria": [
            "No hallucination: The answer is factually correct with no meaningful hallucination.",
            "Minor: A small factual mistake is present, but the main answer to the question remains correct.",
            "Moderate: A factual error or fabrication materially affects part of the answer.",
            "Major: The main answer to the question is incorrect, fabricated, or substantially misleading.",
        ],
    },
}
CONFIG_ID = hashlib.sha256(json.dumps({"assistant_policy": ASSISTANT_POLICY, "questions": QUESTIONS}, sort_keys=True).encode()).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-dir")
    source.add_argument("--source-csv", help="Saved results.csv with question, Qwen answer, reference, and BLEURT score")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", default="triviaqa")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_saved_sources(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if args.source_csv:
        with Path(args.source_csv).open(encoding="utf-8", newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        required = {"id", "input_text", "output_text", "ground_truth_text", "bleurt_score"}
        if source_rows and not required.issubset(source_rows[0]):
            raise ValueError(f"Saved result CSV lacks columns: {sorted(required - source_rows[0].keys())}")
        # Prior LAYA predictions, if present, are deliberately discarded.
        generated = [{"id": row["id"], "input_text": row["input_text"], "output_text": row["output_text"]} for row in source_rows]
        scored = {str(row["id"]): {"ground_truth_text": row["ground_truth_text"], "bleurt_score": row["bleurt_score"]} for row in source_rows}
        return generated, scored
    artifact_dir = Path(args.artifact_dir)
    generation_path = artifact_dir / "generation" / args.domain / "rows.jsonl"
    bleurt_path = artifact_dir / "bleurt" / args.domain / "scores.jsonl"
    if not generation_path.is_file() or not bleurt_path.is_file():
        raise FileNotFoundError(f"Missing generation or BLEURT file under {artifact_dir}")
    return read_jsonl(generation_path), {str(row["id"]): row for row in read_jsonl(bleurt_path)}


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result file")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "assistant_policy": ASSISTANT_POLICY,
        "user_message": f"Question: {row['input_text']}\n\nPredicted Answer: {row['output_text']}",
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(r["ground_truth_hallucination_label"] == 1 and r["predicted_label"] == 1 for r in rows)
    tn = sum(r["ground_truth_hallucination_label"] == 0 and r["predicted_label"] == 0 for r in rows)
    fp = sum(r["ground_truth_hallucination_label"] == 0 and r["predicted_label"] == 1 for r in rows)
    fn = sum(r["ground_truth_hallucination_label"] == 1 and r["predicted_label"] == 0 for r in rows)
    total = len(rows)
    return {
        "samples": total,
        "accuracy": (tp + tn) / total if total else None,
        "precision_hallucinated": tp / (tp + fp) if tp + fp else None,
        "recall_hallucinated": tp / (tp + fn) if tp + fn else None,
        "f1_hallucinated": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; enable a Kaggle GPU or use --device cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated, scored = read_saved_sources(args)
    if not generated:
        raise ValueError("No saved TriviaQA rows found")
    if args.max_samples:
        generated = generated[: args.max_samples]
    ids = [str(row["id"]) for row in generated]
    if len(ids) != len(set(ids)):
        raise ValueError("Saved source contains duplicate IDs")
    missing = [row_id for row_id in ids if row_id not in scored]
    if missing:
        raise ValueError(f"Missing BLEURT scores for {len(missing)} rows; first={missing[0]}")

    checkpoint_path = output_dir / "laya_triviaqa_rows.jsonl"
    completed_rows = read_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    stale = [row for row in completed_rows if row.get("laya_config_id") != CONFIG_ID or row.get("laya_model") != args.model]
    if stale:
        raise ValueError(
            f"Checkpoint contains {len(stale)} rows from another LAYA task definition; "
            "choose a new --output-dir to avoid mixing configurations"
        )
    completed = {str(row["id"]): row for row in completed_rows}
    if len(completed) != len(completed_rows):
        raise ValueError("Checkpoint contains duplicate IDs")
    pending = [row for row in generated if str(row["id"]) not in completed]
    print(
        f"TriviaQA rows={len(generated)} completed={len(completed_rows)} pending={len(pending)}; "
        f"LAYA device={device} threshold={args.threshold}",
        flush=True,
    )

    if pending:
        import laya

        load_kwargs: dict[str, Any] = {"device": device}
        if args.revision:
            load_kwargs["revision"] = args.revision
        agent = laya.load(args.model, **load_kwargs)

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            states = [build_state(row) for row in batch]
            predictions = agent.predict_batch(
                states,
                QUESTIONS,
                batch_size=len(states),
                sort_by_length=False,
            )
            saved = []
            for source, prediction in zip(batch, predictions):
                row_id = str(source["id"])
                judged = scored[row_id]
                answer = prediction["answers"]["hallucination"]
                issue = prediction["answers"]["hallucination_type"]
                severity = prediction["answers"]["severity"]
                probability = float(answer["noul"])
                # BLEURT >= 0.5 means correct in the original run, hence not hallucinated.
                ground_truth_hallucination = int(float(judged["bleurt_score"]) < 0.5)
                saved.append({
                    "id": row_id,
                    "input_text": str(source["input_text"]),
                    "output_text": str(source["output_text"]),
                    "ground_truth_text": str(judged["ground_truth_text"]),
                    "bleurt_score": float(judged["bleurt_score"]),
                    "ground_truth_hallucination_label": ground_truth_hallucination,
                    "laya_hallucination_noul": probability,
                    "laya_hallucination_type": issue["choice"],
                    "laya_hallucination_type_probabilities": json.dumps(issue["probabilities"], ensure_ascii=False),
                    "laya_severity_score": float(severity["score"]),
                    "laya_severity_probabilities": json.dumps(severity["probabilities"], ensure_ascii=False),
                    "laya_severity_legend": json.dumps(severity["legend"], ensure_ascii=False),
                    "predicted_label": int(probability > args.threshold),
                    "predicted_label_text": "hallucinated" if probability > args.threshold else "not_hallucinated",
                    "threshold": args.threshold,
                    "laya_model": args.model,
                    "laya_config_id": CONFIG_ID,
                    "device": device,
                })
            append_jsonl(checkpoint_path, saved)
            completed.update({str(row["id"]): row for row in saved})
            done = min(start + len(batch), len(pending))
            print(f"LAYA scored {done}/{len(pending)} pending rows ({len(completed)}/{len(generated)} total)", flush=True)

    ordered = [completed[str(row["id"])] for row in generated]
    write_csv(output_dir / "results.csv", ordered)
    summary = {
        "domain": args.domain,
        "model": args.model,
        "device": device,
        "laya_config_id": CONFIG_ID,
        "threshold_rule": f"hallucinated iff laya_hallucination_noul > {args.threshold}",
        "ground_truth_rule": "hallucinated iff BLEURT-20 score < 0.5",
        **metrics(ordered),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
