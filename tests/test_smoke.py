"""Seconds-long CPU smoke tests on a synthetic artifact directory (no model downloads)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from hprobes.cli import main
from hprobes.features import pool_layers, pooled_layer_count

LAYERS, HIDDEN, ROWS = 9, 32, 120


def write_domain(root: Path, name: str, seed: int, shift: float, layers: int = LAYERS, hidden: int = HIDDEN) -> None:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, ROWS)
    features = rng.normal(shift, 1.0, (ROWS, layers, hidden)).astype(np.float16)
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


def test_import_legacy_roundtrip(artifacts, tmp_path):
    """Simulate an old sala_raw_concat.py run (weights + summary, no PCA) and rebuild the bundle."""
    import torch

    from hprobes.legacy import rebuild
    from hprobes.predict import Detector

    summary = train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp", "--pca-dim", "8")
    run = tmp_path / "run"
    bundle = torch.load(run / "model.pt", weights_only=False)
    old = tmp_path / "old"
    old.mkdir()
    torch.save({"input_dim": bundle["head"]["input_dim"], "hidden_dim": bundle["head"]["hidden_dim"],
                "hidden_layers": bundle["head"]["hidden_layers"], "pooling": "none",
                "model": bundle["state_dict"]}, old / "best_mlp.pt")
    (old / "results.csv").write_text((run / "results.csv").read_text())
    (old / "summary.json").write_text(json.dumps({
        "pooling_method": "logsumexp1d", "pool_kernel_size": 3, "seed": 42, "feature_pca_dim": 8,
        "feature_pca_per_layer": True, "source_domains": ["tqa", "sciq", "popqa"], "target_domain": "triviaqa",
        "num_layers_including_embedding_output": LAYERS, "hidden_size": HIDDEN, "bleurt_threshold": 0.5,
        "sala_threshold": summary["threshold"], "label_definition": summary["label_definition"],
        "target_metrics": summary["target_metrics"]["triviaqa"]}))
    report = rebuild(old, artifacts, tmp_path / "rebuilt.pt", "cpu")
    assert report["max_abs_probability_diff"] < 1e-5
    assert abs(report["target_auroc_rebuilt"] - report["target_auroc_original"]) < 1e-6
    assert Detector.load(tmp_path / "rebuilt.pt").threshold == summary["threshold"]


@pytest.mark.parametrize("pooling", ["logsumexp", "max"])
def test_adaptive_pooling_matches_kernel_pooling_for_qwen(pooling):
    import torch

    x = torch.randn(4, 29, 16)
    assert torch.allclose(pool_layers(x, pooling, 3, 2, 15), pool_layers(x, pooling, 3, 2), atol=1e-6)


@pytest.mark.parametrize("layers", [2, 15, 16, 24, 32, 33, 80])
def test_adaptive_pooling_any_layer_count(layers):
    import torch

    out = pool_layers(torch.randn(2, layers, 8), "logsumexp", 3, 2, 15) if layers >= 2 else None
    assert out.shape == (2, 15, 8) and torch.isfinite(out).all()


def test_pool_layers_option_end_to_end(artifacts, tmp_path):
    summary = train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp", "--pool-layers", "5",
                    "--pca-dim", "8")
    assert summary["mlp_input_dimension"] == 5 * 8


def test_train_on_several_llms_and_autodetect_shape(tmp_path):
    from hprobes.predict import Detector

    a, b = tmp_path / "a", tmp_path / "b"
    for name, seed in (("tqa", 1), ("sciq", 2), ("triviaqa", 3)):
        write_domain(a, name, seed, 0.0, 9, 32)
    for name, seed in (("tqa", 4), ("popqa", 5), ("triviaqa", 6)):
        write_domain(b, name, seed, 0.2, 12, 48)
    out = tmp_path / "multi"
    main(["train", "--method", "mlp", "--artifact-dir", f"qwen={a}", f"llama={b}", "--output-dir", str(out),
          "--device", "cpu", "--sources", "qwen:tqa", "qwen:sciq", "llama:tqa", "llama:popqa",
          "--targets", "qwen:triviaqa", "llama:triviaqa", "--epochs", "2", "--num-workers", "0",
          "--pooling", "logsumexp", "--pool-layers", "5", "--pca-dim", "8"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mlp_input_dimension"] == 5 * 8 and set(summary["target_metrics"]) == {"qwen:triviaqa", "llama:triviaqa"}
    detector = Detector.load(out / "model.pt")
    qwen_states, llama_states = (np.load(d / "generation" / "triviaqa" / "features.npy") for d in (a, b))
    assert detector.predict(qwen_states)["probability_correct"].shape == (ROWS,)
    assert detector.predict(llama_states)["probability_correct"].shape == (ROWS,)
    with pytest.raises(ValueError, match="No LLM"):
        detector.predict(np.zeros((2, 7, 32), dtype=np.float16))
    with pytest.raises(ValueError, match="expects"):
        detector.predict(llama_states, llm="qwen")
