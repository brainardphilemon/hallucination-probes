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


@pytest.mark.parametrize("extra", [["--pooling", "logsumexp", "--pca-dim", "8"], ["--pooling", "max"]])
def test_predict_reproduces_training_scores(artifacts, tmp_path, extra):
    import csv

    from hprobes.predict import Detector

    train(artifacts, tmp_path, "--method", "mlp", *extra)
    detector = Detector.load(tmp_path / "run" / "model.pt")
    with (tmp_path / "run" / "results.csv").open() as f:
        expected = [float(r["sala_probability"]) for r in csv.DictReader(f)]
    features = np.load(artifacts / "generation" / "triviaqa" / "features.npy")
    out = detector.predict(features)
    assert np.allclose(out["probability_correct"], expected, atol=1e-5)
    assert detector.predict(features[0])["probability_correct"].shape == (1,)
    with pytest.raises(ValueError):
        detector.predict(features[:, :5])


def test_predict_cli(artifacts, tmp_path):
    train(artifacts, tmp_path, "--method", "mlp", "--pooling", "max")
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps({"input": f"q{i}", "generated_output": f"a{i}",
                                           "actual_output": f"a{i}"}) + "\n" for i in range(5)))
    features = str(artifacts / "generation" / "triviaqa" / "features.npy")
    main(["predict", "--model", str(tmp_path / "run" / "model.pt"), "--states", features,
          "--output", str(tmp_path / "p.csv")])
    assert (tmp_path / "p.csv").exists()
    np.save(tmp_path / "five.npy", np.load(features)[:5])
    main(["predict", "--model", str(tmp_path / "run" / "model.pt"), "--states", str(tmp_path / "five.npy"),
          "--records", str(records), "--output", str(tmp_path / "p.jsonl")])
    first = json.loads((tmp_path / "p.jsonl").read_text().splitlines()[0])
    assert first["generated_output"] == "a0" and first["verdict"] in ("correct", "hallucinated")
