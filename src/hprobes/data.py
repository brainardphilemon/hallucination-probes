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
    model: str = "default"  # which LLM produced the hidden states


@dataclass
class ExperimentData:
    artifact_dir: Any  # a Path, or {llm name: Path} when several LLMs are used
    sources: list[Domain]
    targets: list[Domain]
    n_layers: int
    hidden_size: int
    models: dict[str, tuple[int, int]] = field(default_factory=dict)  # llm name -> (layers, hidden size)

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


def parse_spec(spec: str, artifacts: dict[str, Path]) -> tuple[str, str]:
    """'tqa' (one LLM) or 'llama:tqa' (several LLMs) -> (llm name, dataset name)."""
    from hprobes.domains import DOMAINS

    model, _, name = spec.rpartition(":")
    if not model:
        if len(artifacts) != 1:
            raise ValueError(f"With several LLMs write datasets as <llm>:<dataset> (got {spec!r}); "
                             f"LLMs: {sorted(artifacts)}")
        model = next(iter(artifacts))
    if model not in artifacts:
        raise ValueError(f"Unknown LLM {model!r} in {spec!r}; LLMs: {sorted(artifacts)}")
    if name not in DOMAINS:
        raise ValueError(f"Unknown dataset {name!r} in {spec!r}; datasets: {list(DOMAINS)}")
    return model, name


def load_experiment(artifact_dir, sources: Sequence[str], targets: Sequence[str],
                    bleurt_threshold: float = 0.5) -> ExperimentData:
    """artifact_dir: one folder (a single LLM) or {llm name: folder}; with several LLMs name datasets <llm>:<dataset>."""
    artifacts = ({str(k): Path(v).expanduser().resolve() for k, v in artifact_dir.items()}
                 if isinstance(artifact_dir, dict) else {"default": Path(artifact_dir).expanduser().resolve()})
    multi = len(artifacts) > 1
    overlap = set(sources) & set(targets)
    if overlap:
        raise ValueError(f"Datasets cannot be both source and target: {sorted(overlap)}")
    if not sources or not targets:
        raise ValueError("Need at least one source and one target dataset")
    loaded: dict[str, Domain] = {}
    shapes: dict[str, tuple[int, int]] = {}
    for spec in [*sources, *targets]:
        model, dataset = parse_spec(spec, artifacts)
        rows, labels, bleurt = load_labels(artifacts[model], dataset, bleurt_threshold)
        features = np.load(artifacts[model] / "generation" / dataset / "features.npy", mmap_mode="r")
        if len(features) != len(rows):
            raise ValueError(f"{spec}: {len(features)} feature rows but {len(rows)} answers")
        shape = (int(features.shape[1]), int(features.shape[2]))
        if shapes.setdefault(model, shape) != shape:
            raise ValueError(f"{spec}: feature shape {shape} differs from {shapes[model]} of {model}")
        domain = Domain(spec if multi else dataset, rows, labels, features, model=model)
        domain.bleurt = bleurt
        loaded[spec] = domain
    for spec in sources:
        train, val, test = source_split_indices(loaded[spec].labels, loaded[spec].name)
        loaded[spec].split = {"train": train, "val": val, "test": test}
    for spec in targets:
        loaded[spec].split = {"all": np.arange(len(loaded[spec].labels))}
    first = next(iter(shapes.values()))
    data = ExperimentData(artifact_dir if multi else next(iter(artifacts.values())),
                          [loaded[n] for n in sources], [loaded[n] for n in targets], first[0], first[1], shapes)
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
