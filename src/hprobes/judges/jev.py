#!/usr/bin/env python3
"""Evaluate saved TriviaQA answers with Jev, without rerunning Qwen or BLEURT."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


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


def build_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "assistant_policy": ASSISTANT_POLICY,
        "user_message": f"Question: {row['input_text']}\n\nPredicted Answer: {row['output_text']}",
    }


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
        # LAYA-specific predictions are deliberately discarded before Jev inference.
        generated = [{"id": row["id"], "input_text": row["input_text"], "output_text": row["output_text"]} for row in source_rows]
        scored = {str(row["id"]): {"ground_truth_text": row["ground_truth_text"], "bleurt_score": row["bleurt_score"]} for row in source_rows}
        return generated, scored
    artifact_dir = Path(args.artifact_dir)
    generation_path = artifact_dir / "generation" / args.domain / "rows.jsonl"
    bleurt_path = artifact_dir / "bleurt" / args.domain / "scores.jsonl"
    return read_jsonl(generation_path), {str(row["id"]): row for row in read_jsonl(bleurt_path)}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


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
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-dir")
    source.add_argument("--source-csv", help="Saved results.csv with question, Qwen answer, reference, and BLEURT score")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", default="triviaqa")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--max-samples", type=int, default=0, help="0 processes all available rows")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    if not os.environ.get("TYPESAFE_API_KEY"):
        parser.error("Set TYPESAFE_API_KEY (in Kaggle Secrets); never put the key in notebook source")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated, scored = read_saved_sources(args)
    if not generated:
        raise ValueError("No saved TriviaQA rows found")
    if args.max_samples:
        generated = generated[: args.max_samples]
    ids = [str(row["id"]) for row in generated]
    if len(ids) != len(set(ids)):
        raise ValueError("Generation file contains duplicate IDs")
    missing = [row_id for row_id in ids if row_id not in scored]
    if missing:
        raise ValueError(f"Missing BLEURT scores for {len(missing)} generated rows; first={missing[0]}")

    checkpoint = output_dir / "jev_triviaqa_rows.jsonl"
    prior = read_jsonl(checkpoint) if checkpoint.exists() else []
    if any(row.get("jev_config_id") != CONFIG_ID or row.get("jev_requested_model") != args.model for row in prior):
        raise ValueError("Checkpoint uses a different question/state definition or model; use a new output directory")
    completed = {str(row["id"]): row for row in prior}
    if len(completed) != len(prior):
        raise ValueError("Checkpoint contains duplicate IDs")
    pending = [row for row in generated if str(row["id"]) not in completed]
    print(f"TriviaQA rows={len(generated)} completed={len(completed)} pending={len(pending)} model={args.model}", flush=True)

    if pending:
        from typesafe_sdk import TypeSafeClient

        with TypeSafeClient(model=args.model) as client:
            for index, source in enumerate(pending, 1):
                # Ground truth is never sent to Jev.
                answer = client.system_one(state=build_state(source), questions=QUESTIONS)
                probability = float(answer.nouls["hallucination"].noul)
                if not 0 <= probability <= 1:
                    raise ValueError(f"Invalid Jev noul result for row {source['id']}")
                issue = answer.choices["hallucination_type"]
                severity = answer.scores["severity"]
                row_id = str(source["id"])
                reference = scored[row_id]
                label = int(probability > 0.5)
                result = {
                    "id": row_id,
                    "input_text": str(source["input_text"]),
                    "output_text": str(source["output_text"]),
                    "ground_truth_text": str(reference["ground_truth_text"]),
                    "bleurt_score": float(reference["bleurt_score"]),
                    "ground_truth_hallucination_label": int(float(reference["bleurt_score"]) < 0.5),
                    "jev_hallucination_noul": probability,
                    "predicted_label": label,
                    "predicted_label_text": "hallucinated" if label else "not_hallucinated",
                    "jev_hallucination_type": issue.choice,
                    "jev_hallucination_type_probabilities": json.dumps(issue.probabilities, ensure_ascii=False),
                    "jev_severity_score": float(severity.score),
                    "jev_severity_probabilities": json.dumps(severity.probabilities, ensure_ascii=False),
                    "jev_severity_legend": json.dumps(severity.legend, ensure_ascii=False),
                    "jev_model": answer.model,
                    "jev_requested_model": args.model,
                    "jev_input_tokens": answer.usage.input_tokens,
                    "jev_output_tokens": answer.usage.output_tokens,
                    "jev_config_id": CONFIG_ID,
                }
                append_jsonl(checkpoint, result)
                completed[row_id] = result
                if index % 100 == 0 or index == len(pending):
                    print(f"Jev scored {index}/{len(pending)} pending ({len(completed)}/{len(generated)} total)", flush=True)

    ordered = [completed[row_id] for row_id in ids]
    results_path = output_dir / "results.csv"
    temporary = results_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(results_path)
    summary = {
        "domain": args.domain,
        "requested_model": args.model,
        "jev_config_id": CONFIG_ID,
        "predicted_label_rule": "hallucinated iff Jev hallucination noul > 0.5",
        "ground_truth_rule": "hallucinated iff BLEURT-20 score < 0.5",
        "jev_input_tokens": sum(row["jev_input_tokens"] for row in ordered),
        "jev_output_tokens": sum(row["jev_output_tokens"] for row in ordered),
        **metrics(ordered),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
