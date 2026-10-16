"""Import a model trained with the old SALA fork's ``sala_raw_concat.py`` into a ``model.pt`` bundle.

That script saved the MLP (``best_mlp.pt``) and the calibrated threshold (``summary.json``) but not the
per-layer PCA. The PCA is a deterministic function of the source-train features and the seed, so it is
rebuilt here with the same procedure, and the result is checked against the run's own ``results.csv``
(or, failing that, its recorded target AUROC) before the bundle is trusted.

    hprobes import-legacy --run-dir sala_g14_6src_logsumexp_layerpca512_4x128 \
        --artifact-dir /kaggle/input/.../sala_qwen25_g14_6src_triviaqa_v1 --output model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

POOLING_NAMES = {"logsumexp1d": "logsumexp", "max1d": "max", "none": "none"}


def rebuild(run_dir: Path, artifact_dir: Path, output: Path, device: str = "cuda") -> dict:
    import torch

    from hprobes.data import StateDataset, load_experiment
    from hprobes.features import LayerPCA, pooled_layer_count, pooled_matrix
    from hprobes.predict import BUNDLE_FORMAT, Detector
    from hprobes.training import get_device

    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    saved = torch.load(run_dir / "best_mlp.pt", map_location="cpu", weights_only=False)
    method = summary["pooling_method"]
    if method not in POOLING_NAMES:
        raise SystemExit(f"Unsupported legacy pooling {method!r}; supported: {sorted(POOLING_NAMES)}")
    pooling = POOLING_NAMES[method]
    kernel, stride = int(summary.get("pool_kernel_size") or 3), int(summary.get("pool_stride") or 2)
    seed, pca_dim = int(summary["seed"]), int(summary.get("feature_pca_dim") or 0)
    sources, target = summary["source_domains"], summary["target_domain"]
    dev = get_device(device)
    data = load_experiment(artifact_dir, sources, [target], float(summary["bleurt_threshold"]))
    if (data.n_layers, data.hidden_size) != (summary["num_layers_including_embedding_output"],
                                             summary["hidden_size"]):
        raise SystemExit("Feature shape differs from the run's; wrong artifact directory?")

    pca_state = None
    if pca_dim:
        scope = "per-layer" if summary.get("feature_pca_per_layer") else "global"
        pca = LayerPCA(pca_dim, scope, seed)
        pooled_args = SimpleNamespace(pooling=pooling, pool_kernel=kernel, pool_stride=stride, pool_layers=0)
        print(f"Refitting {scope} PCA({pca_dim}) on {len(StateDataset(data.sources, 'train'))} source-train rows",
              flush=True)
        with torch.inference_mode():
            pca.fit_transform(pooled_matrix(StateDataset(data.sources, "train"), pooled_args, dev), dev)
        pca_state = pca.state()

    state = {k: v.cpu() for k, v in saved["model"].items()}
    bundle = {
        "format": BUNDLE_FORMAT, "pooling": pooling, "pool_kernel": kernel, "pool_stride": stride,
        # kernel 3 / stride 2 equals adaptive pooling to the same count (exact for logsumexp and max), which lets
        # the detector take other layer counts; verified against the original run's results below.
        "pool_layers": (pooled_layer_count(data.n_layers, pooling, kernel, stride)
                        if pooling in ("logsumexp", "max") and kernel == 3 and stride == 2 else 0),
        "n_layers": data.n_layers, "hidden_size": data.hidden_size, "pca": pca_state,
        "head": {"input_dim": int(saved["input_dim"]), "hidden_dim": int(saved["hidden_dim"]),
                 "hidden_layers": int(saved["hidden_layers"]), "dropout": 0.2},
        "state_dict": state, "threshold": float(summary["sala_threshold"]),
        "label_definition": summary["label_definition"], "bleurt_threshold": float(summary["bleurt_threshold"]),
        "sources": sources, "targets": [target], "loss": summary.get("classifier_loss", "bce"),
        "target_metrics": {target: summary["target_metrics"]}, "imported_from": str(run_dir),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)

    # Verify against the original run before trusting the rebuilt PCA.
    detector = Detector(bundle, device)
    domain = data.targets[0]
    probability = detector.predict(np.asarray(domain.features))["probability_correct"]
    report = {"bundle": str(output)}
    results = run_dir / "results.csv"
    if results.exists():
        with results.open(encoding="utf-8", newline="") as stream:
            original = {r["id"]: float(r["sala_probability"]) for r in csv.DictReader(stream)}
        ids = [str(r["id"]) for r in domain.rows]
        if set(ids) == set(original):
            diff = np.abs(probability - np.asarray([original[i] for i in ids]))
            report.update(max_abs_probability_diff=float(diff.max()), mean_abs_probability_diff=float(diff.mean()))
    from sklearn.metrics import roc_auc_score

    report.update(target_auroc_rebuilt=float(roc_auc_score(domain.labels, probability)),
                  target_auroc_original=float(summary["target_metrics"]["auroc"]))
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Old run output folder containing best_mlp.pt, summary.json (and results.csv)")
    parser.add_argument("--artifact-dir", type=Path, required=True,
                        help="Saved features the run was trained on (generation/ and bleurt/ folders)")
    parser.add_argument("--output", type=Path, required=True, help="model.pt to write for `hprobes predict`")
    parser.add_argument("--device", default="cuda", help="cuda or cpu (PCA refit; falls back to cpu)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rebuild(args.run_dir, args.artifact_dir, args.output, args.device)


if __name__ == "__main__":
    main()
