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
    with pytest.raises(ValueError, match="hidden size"):
        detector.predict(np.zeros((2, 7, 40), dtype=np.float16))
    with pytest.raises(ValueError, match="hidden size"):
        detector.predict(llama_states, llm="qwen")


def test_self_fitted_pca_works_for_any_shape(artifacts, tmp_path):
    import torch

    from hprobes.features import self_pca
    from hprobes.predict import Detector

    summary = train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp", "--pool-layers", "5",
                    "--pca-dim", "8", "--pca-fit", "self")
    assert summary["mlp_input_dimension"] == 5 * 8
    detector = Detector.load(tmp_path / "run" / "model.pt")
    with (tmp_path / "run" / "results.csv").open() as f:
        expected = [float(r["sala_probability"]) for r in csv.DictReader(f)]
    features = np.load(artifacts / "generation" / "triviaqa" / "features.npy")
    assert np.allclose(detector.predict(features)["probability_correct"], expected, atol=1e-4)
    rng = np.random.default_rng(0)
    for layers, hidden in ((7, 20), (40, 100), (3, 6)):  # other LLM shapes still run; hidden < PCA dim is zero-padded
        out = detector.predict(rng.normal(size=(30, layers, hidden)).astype(np.float16))
        assert out["probability_correct"].shape == (30,) and np.isfinite(out["raw_logit"]).all()
    with pytest.raises(ValueError, match="at least 2"):
        detector.predict(features[0])
    # signs, scale and row order must not change the result
    x = torch.randn(40, 3, 24)
    z = self_pca(x, 6)
    assert torch.allclose(z, self_pca(x * 5.0, 6), atol=1e-4)
    perm = torch.randperm(40)
    assert torch.allclose(z[perm], self_pca(x[perm], 6), atol=1e-3)


def test_self_pca_across_llms(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for name, seed in (("tqa", 1), ("sciq", 2), ("triviaqa", 3)):
        write_domain(a, name, seed, 0.0, 9, 32)
    for name, seed in (("tqa", 4), ("popqa", 5), ("triviaqa", 6)):
        write_domain(b, name, seed, 0.2, 12, 48)
    main(["train", "--method", "mlp", "--artifact-dir", f"qwen={a}", f"llama={b}", "--output-dir", str(tmp_path / "o"),
          "--device", "cpu", "--sources", "qwen:tqa", "llama:tqa", "llama:popqa", "--targets", "qwen:triviaqa",
          "llama:triviaqa", "--epochs", "2", "--num-workers", "0", "--pooling", "logsumexp", "--pool-layers", "5",
          "--pca-dim", "8", "--pca-fit", "self"])
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["mlp_input_dimension"] == 40


def test_fixed_reductions_are_data_free():
    import torch

    from hprobes.features import dct_basis, fixed_reduce

    x = torch.randn(6, 4, 40)
    basis = dct_basis(40, 40)
    assert torch.allclose(basis.T @ basis, torch.eye(40), atol=1e-5)  # orthonormal
    assert torch.allclose(fixed_reduce(x, 8, "avg", norm="none"), x.reshape(6, 4, 8, 5).mean(-1), atol=1e-5)
    for method in ("avg", "dct"):
        for hidden, dim in ((40, 8), (7, 8), (300, 8)):
            batch = torch.randn(5, 3, hidden)
            z = fixed_reduce(batch, dim, method)
            assert z.shape == (5, 3, dim) and torch.isfinite(z).all()
            assert torch.allclose(z[2:3], fixed_reduce(batch[2:3], dim, method), atol=1e-5)  # per-example, no batch needed
    constant = torch.ones(1, 1, 40)  # DCT of a constant keeps only the DC term
    assert fixed_reduce(constant, 5, "dct", norm="none")[0, 0, 1:].abs().max() < 1e-5


@pytest.mark.parametrize("method", ["avg", "dct"])
def test_train_and_predict_with_fixed_reduction(artifacts, tmp_path, method):
    from hprobes.predict import Detector

    summary = train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp", "--pool-layers", "5",
                    "--pca-dim", "8", "--pca-fit", method)
    assert summary["mlp_input_dimension"] == 40
    detector = Detector.load(tmp_path / "run" / "model.pt")
    with (tmp_path / "run" / "results.csv").open() as f:
        expected = [float(r["sala_probability"]) for r in csv.DictReader(f)]
    features = np.load(artifacts / "generation" / "triviaqa" / "features.npy")
    assert np.allclose(detector.predict(features)["probability_correct"], expected, atol=1e-4)
    assert np.allclose(detector.predict(features[:1])["probability_correct"], expected[:1], atol=1e-4)  # one example works
    other = np.random.default_rng(1).normal(size=(3, 40, 100)).astype(np.float16)  # another LLM's shape
    assert detector.predict(other)["probability_correct"].shape == (3,)


def test_any_layers_pools_to_trained_layer_count(artifacts, tmp_path):
    from hprobes.predict import Detector

    train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp", "--pca-dim", "8")  # 9 layers -> 5
    detector = Detector.load(tmp_path / "run" / "model.pt")
    features = np.load(artifacts / "generation" / "triviaqa" / "features.npy")
    assert np.allclose(detector.predict(features)["raw_logit"],
                       detector.predict(features, any_layers=True)["raw_logit"], atol=1e-4)
    deeper = np.random.default_rng(0).normal(size=(4, 21, HIDDEN)).astype(np.float16)  # another LLM: 21 layers
    assert detector.predict(deeper, any_layers=True)["raw_logit"].shape == (4,)
    with pytest.raises(ValueError):
        detector.predict(deeper)  # without the option the layer count must match
    with pytest.raises(ValueError, match="hidden size"):
        detector.predict(np.zeros((2, 21, HIDDEN + 1), dtype=np.float16), any_layers=True)


def test_generate_handles_templates_without_system_role_and_multiple_stop_tokens():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from hprobes.generate import build_prompt, generate_with, stop_token_ids

    class Tok:
        eos_token_id, pad_token_id = 1, None

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            if messages[0]["role"] == "system":
                raise ValueError("System role not supported")
            return "<bos>" + messages[0]["content"]

        def __call__(self, text, return_tensors=None, truncation=False, max_length=None, add_special_tokens=True):
            assert add_special_tokens is False
            ids = [ord(c) % 50 + 2 for c in text][:max_length if truncation else None]
            return transformers.BatchEncoding({"input_ids": torch.tensor([ids])})

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(int(i) + 40) for i in ids)

    assert build_prompt(Tok(), "q", "sys") == "<bos>sys\n\nq"
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(transformers.Qwen2Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
        num_key_value_heads=2, eos_token_id=[1, 7])).eval()
    assert stop_token_ids(Tok(), model) == [1, 7]
    states, answers, last_tokens = generate_with(model, Tok(), [{"input_text": "hello"}, {"input_text": "hi"}], "cpu",
                                                 "sys", max_new_tokens=4, log_every=0)
    assert states.shape == (2, 4, 16) and len(answers) == 2 and len(last_tokens) == 2


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_loaded_bundle_runs_on_gpu(artifacts, tmp_path, device):
    """A model.pt keeps its PCA on the CPU; predicting on a GPU must move it (failed on Kaggle T4)."""
    import torch

    from hprobes.predict import Detector

    if device == "mps" and not torch.backends.mps.is_available() or device == "cuda" and not torch.cuda.is_available():
        pytest.skip(f"{device} not available")
    train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp", "--pca-dim", "8")
    features = np.load(artifacts / "generation" / "triviaqa" / "features.npy")
    cpu = Detector.load(tmp_path / "run" / "model.pt").predict(features)["probability_correct"]
    gpu_detector = Detector.load(tmp_path / "run" / "model.pt")
    gpu_detector.device = torch.device(device)
    gpu_detector.model.to(device)
    gpu = gpu_detector.predict(features, any_layers=True)["probability_correct"]
    assert np.allclose(cpu, gpu, atol=1e-3)


def test_diagnose_report_matches_detector(artifacts, tmp_path, capsys):
    from hprobes.diagnose import report

    train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp", "--pca-dim", "8")
    features = np.load(artifacts / "generation" / "triviaqa" / "features.npy")
    other = (np.random.default_rng(0).normal(size=(6, 17, HIDDEN)) * 50).astype(np.float16)  # other depth, bigger scale
    results = report(tmp_path / "run" / "model.pt", {"ref": features, "other": other}, reference="ref")
    out = capsys.readouterr().out
    assert "stages reproduce Detector.logits" in out and results["other"]["pooled_shape"] == (6, 5, HIDDEN)
    assert all(r["matches_detector"] < 1e-4 for r in results.values())


def test_answer_end_tokens_are_stripped():
    """Gemma ends with <end_of_turn> <eos>; the kept state must be the answer's last real token (bug seen on Kaggle)."""
    from hprobes.generate import stop_token_ids, strip_trailing

    class Tok:
        eos_token_id, pad_token_id, unk_token_id = 1, 0, 3
        all_special_ids = [0, 1, 2, 3, 107]
        vocab = {"<end_of_turn>": 107}

        def convert_tokens_to_ids(self, name):
            return self.vocab.get(name, self.unk_token_id)

    class Model:
        class generation_config:
            eos_token_id = 1

    stops = stop_token_ids(Tok(), Model())
    assert 107 in stops and 3 not in stops
    assert strip_trailing([50, 51, 52, 107, 1], set(stops)) == [50, 51, 52]
    assert strip_trailing([50, 107, 0, 0], set(stops)) == [50]


def test_import_legacy_without_pca_is_exact_and_diagnosable(artifacts, tmp_path):
    """Old no-PCA runs (e.g. sala-g14-logsumexp-4x128) load exactly; diagnose works without a PCA."""
    import torch

    from hprobes.diagnose import report
    from hprobes.legacy import rebuild

    summary = train(artifacts, tmp_path, "--method", "mlp", "--pooling", "logsumexp")
    run = tmp_path / "run"
    bundle = torch.load(run / "model.pt", weights_only=False)
    old = tmp_path / "old"
    old.mkdir()
    torch.save({"input_dim": bundle["head"]["input_dim"], "hidden_dim": bundle["head"]["hidden_dim"],
                "hidden_layers": bundle["head"]["hidden_layers"], "pooling": "logsumexp1d",
                "model": bundle["state_dict"]}, old / "best_mlp.pt")
    (old / "results.csv").write_text((run / "results.csv").read_text())
    (old / "summary.json").write_text(json.dumps({  # no feature_pca_dim key, like the real older runs
        "pooling_method": "logsumexp1d", "pool_kernel_size": 3, "pool_stride": 2, "seed": 42,
        "source_domains": ["tqa", "sciq", "popqa"], "target_domain": "triviaqa",
        "num_layers_including_embedding_output": LAYERS, "hidden_size": HIDDEN, "bleurt_threshold": 0.5,
        "sala_threshold": summary["threshold"], "label_definition": summary["label_definition"],
        "target_metrics": summary["target_metrics"]["triviaqa"]}))
    out = rebuild(old, artifacts, tmp_path / "nopca.pt", "cpu")
    assert out["max_abs_probability_diff"] < 1e-5
    features = np.load(artifacts / "generation" / "triviaqa" / "features.npy")
    deeper = np.random.default_rng(0).normal(size=(4, 15, HIDDEN)).astype(np.float16)
    results = report(tmp_path / "nopca.pt", {"ref": features, "deeper": deeper}, reference="ref")
    assert results["deeper"]["pca"].shape == (4, 5, HIDDEN) and results["deeper"]["matches_detector"] < 1e-4


def test_generate_drops_trailing_newline_before_end_of_turn():
    """Gemma answers 'Golf.' + '\\n' + <end_of_turn>; the kept token must be '.', not the newline (seen on Kaggle)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from hprobes.generate import generate_with

    NEWLINE, END = 9, 7

    class Tok:
        eos_token_id, pad_token_id = 1, 0
        all_special_ids = [0, 1, END]

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[-1]["content"]

        def __call__(self, text, return_tensors=None, truncation=False, max_length=None, add_special_tokens=True):
            return transformers.BatchEncoding({"input_ids": torch.tensor([[11, 12, 13]])})

        def decode(self, ids, skip_special_tokens=True):
            return "".join("\n" if int(i) == NEWLINE else ("" if int(i) in self.all_special_ids else "x")
                           for i in ids)

    class FakeModel(torch.nn.Module):
        generation_config = type("G", (), {"eos_token_id": 1})()

        def generate(self, input_ids, **kwargs):  # answer tokens 20, 21, then newline, end-of-turn, eos
            return torch.cat([input_ids, torch.tensor([[20, 21, NEWLINE, END, 1]])], dim=1)

        def forward(self, ids, **kwargs):
            seen.append(ids[0].tolist())
            return type("O", (), {"hidden_states": [torch.zeros(1, ids.shape[1], 4)] * 3})()

    seen = []
    _, answers, last = generate_with(FakeModel(), Tok(), [{"input_text": "q"}], "cpu", "sys", log_every=0)
    assert seen[0] == [11, 12, 13, 20, 21]   # replay ends at the answer's last real token
