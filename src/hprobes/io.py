"""Atomic file writers and readers shared by every stage."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np

RESULT_COLUMNS = [
    "method",
    "setting",
    "source_domain",
    "target_domain",
    "id",
    "input_text",
    "output_text",
    "ground_truth_text",
    "ground_truth_label",
    "bleurt_score",
    "raw_logit",
    "sala_probability",
    "predicted_label",
    "classifier_loss",
    "logit_adjustment_tau",
    "training_logit_adjustment",
    "source_train_prior_0",
    "source_train_prior_1",
    "split",
    "sala_threshold",
    "ground_truth_answers",
]


def json_dump_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def npz_dump_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def jsonl_dump_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def csv_dump_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = {key: source.get(key) for key in RESULT_COLUMNS}
            row["ground_truth_answers"] = json.dumps(row["ground_truth_answers"], ensure_ascii=False)
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid/truncated JSON on {path}:{number}") from exc
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def prepare_manifest(path: Path, current: dict[str, Any]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != current:
            raise ValueError(f"Manifest mismatch at {path}; use a new --output-dir for changed settings")
    else:
        json_dump_atomic(path, current)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
