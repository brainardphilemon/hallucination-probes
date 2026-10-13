"""Load generation + BLEURT artifacts for any choice of source and target datasets.

Every source dataset gets the same stratified split as the SALA paper code: 25% source test,
then 10% of the remainder as source validation (random_state 42 / 43), so splits are
identical across methods and runs. Target datasets are used in full and their labels are
only read for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hprobes.io import read_jsonl


@dataclass
class Domain:
    name: str
    rows: list[dict[str, Any]]
    labels: np.ndarray
    features: np.ndarray  # memory-mapped (rows, layers, hidden), float16
    split: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ExperimentData:
    artifact_dir: Path
    sources: list[Domain]
    targets: list[Domain]
    n_layers: int
    hidden_size: int

    def entries(self, split: str) -> list[tuple[int, int, int]]:
        """(source index, row id, label) for a source split: train / val / test."""
        return [(d, int(i), int(domain.labels[i]))
                for d, domain in enumerate(self.sources) for i in domain.split[split]]

    def labels(self, split: str) -> np.ndarray:
        return np.concatenate([domain.labels[domain.split[split]] for domain in self.sources])

    def source_counts(self) -> dict[str, dict[str, int]]:
        return {d.name: {"total": len(d.labels), **{k: len(v) for k, v in d.split.items()}}
                for d in self.sources}


def load_labels(artifact_dir: Path, domain: str, bleurt_threshold: float):
    rows = read_jsonl(artifact_dir / "generation" / domain / "rows.jsonl")
    scored = read_jsonl(artifact_dir / "bleurt" / domain / "scores.jsonl")
    by_id = {str(row["id"]): row for row in scored}
    if not rows or len(by_id) != len(rows):
        raise ValueError(f"Missing or incomplete generation/BLEURT artifacts for {domain}")
    labels = np.asarray([float(by_id[str(row["id"])]["bleurt_score"]) >= bleurt_threshold for row in rows],
                        dtype=np.int64)
    return rows, labels, by_id


def source_split_indices(labels: np.ndarray, domain: str):
    """Independent stratified train / validation / test split of one source dataset."""
    from sklearn.model_selection import StratifiedShuffleSplit

    if len(np.unique(labels)) < 2:
        raise ValueError(f"Source {domain} contains only one BLEURT class")
    indices = np.arange(len(labels))
    outer = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    trainval_idx, test_idx = next(outer.split(indices, labels))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=43)
    train_pos, val_pos = next(inner.split(trainval_idx, labels[trainval_idx]))
    return trainval_idx[train_pos], trainval_idx[val_pos], test_idx


def load_experiment(artifact_dir: Path, sources: Sequence[str], targets: Sequence[str],
                    bleurt_threshold: float = 0.5) -> ExperimentData:
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    overlap = set(sources) & set(targets)
    if overlap:
        raise ValueError(f"Datasets cannot be both source and target: {sorted(overlap)}")
    if not sources or not targets:
        raise ValueError("Need at least one source and one target dataset")
    loaded: dict[str, Domain] = {}
    shape = None
    for name in [*sources, *targets]:
        rows, labels, bleurt = load_labels(artifact_dir, name, bleurt_threshold)
        features = np.load(artifact_dir / "generation" / name / "features.npy", mmap_mode="r")
        if len(features) != len(rows):
            raise ValueError(f"{name}: {len(features)} feature rows but {len(rows)} answers")
        if shape is not None and features.shape[1:] != shape:
            raise ValueError(f"{name}: feature shape {features.shape[1:]} differs from {shape}")
        shape = features.shape[1:]
        domain = Domain(name, rows, labels, features)
        domain.bleurt = bleurt
        loaded[name] = domain
    for name in sources:
        train, val, test = source_split_indices(loaded[name].labels, name)
        loaded[name].split = {"train": train, "val": val, "test": test}
    for name in targets:
        loaded[name].split = {"all": np.arange(len(loaded[name].labels))}
    data = ExperimentData(artifact_dir, [loaded[n] for n in sources], [loaded[n] for n in targets],
                          int(shape[0]), int(shape[1]))
    if len(np.unique(data.labels("train"))) != 2 or len(np.unique(data.labels("val"))) != 2:
        raise ValueError("Source train and validation splits must contain both BLEURT classes")
    return data


class StateDataset:
    """Hidden-state matrices (layers, hidden) as float16 plus labels, read lazily from the memmaps."""

    def __init__(self, domains: Sequence[Domain], split: str):
        self.features = [domain.features for domain in domains]
        self.entries = [(d, int(i), int(domain.labels[i]))
                        for d, domain in enumerate(domains) for i in domain.split[split]]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, position):
        d, row, label = self.entries[position]
        return np.asarray(self.features[d][row], dtype=np.float16).copy(), np.float32(label)
