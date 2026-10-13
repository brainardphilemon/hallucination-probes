"""Seconds-long CPU smoke tests on a synthetic artifact directory (no model downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hprobes.cli import main
from hprobes.features import pool_layers, pooled_layer_count

LAYERS, HIDDEN, ROWS = 9, 32, 120


def write_domain(root: Path, name: str, seed: int, shift: float) -> None:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, ROWS)
    features = rng.normal(shift, 1.0, (ROWS, LAYERS, HIDDEN)).astype(np.float16)
    features[:, :, :4] += (2 * labels[:, None, None] - 1).astype(np.float16)
    gen, bleurt = root / "generation" / name, root / "bleurt" / name
    gen.mkdir(parents=True)
    bleurt.mkdir(parents=True)
    np.save(gen / "features.npy", features)
    with (gen / "rows.jsonl").open("w") as f:
        for i in range(ROWS):
            f.write(json.dumps({"id": f"{name}-{i}", "domain": name, "input_text": f"q{i}",
                                "output_text": f"a{i}"}) + "\n")
    with (bleurt / "scores.jsonl").open("w") as f:
        for i, label in enumerate(labels):
            f.write(json.dumps({"id": f"{name}-{i}", "ground_truth_text": "x",
                                "bleurt_score": 0.8 if label else 0.2}) + "\n")


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    for k, name in enumerate(("tqa", "sciq", "popqa", "triviaqa")):
        write_domain(root, name, k, 0.3 * k)
    return root


def train(artifacts, tmp_path, *extra):
    out = tmp_path / "run"
    main(["train", "--artifact-dir", str(artifacts), "--output-dir", str(out), "--device", "cpu",
          "--sources", "tqa", "sciq", "popqa", "--targets", "triviaqa", "--epochs", "2", "--num-workers", "0",
          *extra])
    summary = json.loads((out / "summary.json").read_text())
    assert (out / "results.csv").exists()
    assert summary["target_metrics"]["triviaqa"]["auroc"] is not None
    return summary


@pytest.mark.parametrize("pooling", ["none", "max", "mean", "logsumexp"])
def test_pool_shapes(pooling):
    import torch

    x = torch.randn(3, 29, 8)
    assert pool_layers(x, pooling, 3, 2).shape == (3, pooled_layer_count(29, pooling, 3, 2), 8)
    assert pooled_layer_count(29, "logsumexp", 3, 2) == 15


@pytest.mark.parametrize("extra", [
    [],
    ["--pooling", "max", "--loss", "focal"],
    ["--pooling", "logsumexp", "--pca-dim", "8", "--pca-scope", "per-layer"],
    ["--pca-dim", "16", "--pca-scope", "global", "--loss", "logit_adjusted"],
])
def test_mlp(artifacts, tmp_path, extra):
    train(artifacts, tmp_path, "--method", "mlp", *extra)


@pytest.mark.parametrize("extra", [
    ["--proj-dim", "4"],
    ["--pooling", "logsumexp", "--pca-dim", "8", "--proj-dim", "4", "--loss", "focal"],
])
def test_sala(artifacts, tmp_path, extra):
    summary = train(artifacts, tmp_path, "--method", "sala", "--epochs-proj", "2", "--hidden-dim", "32", *extra)
    assert summary["selected_projection_dimensions"][0] == 4


def test_polysemantic(artifacts, tmp_path):
    train(artifacts, tmp_path, "--method", "polysemantic", "--pooling", "max", "--model-dim", "16",
          "--embedding-dim", "16", "--latents", "4", "--latent-heads", "2", "--heads", "2")


def test_preset_rejects_overlap(artifacts, tmp_path):
    with pytest.raises(ValueError):
        main(["train", "--method", "mlp", "--artifact-dir", str(artifacts), "--output-dir", str(tmp_path),
              "--device", "cpu", "--sources", "tqa", "--targets", "tqa"])
