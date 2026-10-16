"""Stage-by-stage check of a detector on several sets of hidden states.

Runs the same steps as ``Detector.logits`` (pooling -> PCA -> MLP) but keeps the intermediate values, so you can
see where an input stops looking like the training data:
  * shapes and the pooling windows actually used,
  * hidden-state norms per layer (raw and pooled),
  * the MLP's input features (PCA coordinates, or the pooled states themselves for a detector without PCA)
    compared with a reference set (z-scores against the reference's per-feature mean / std),
  * raw logits and probabilities,
  * a consistency check that these stages reproduce ``Detector.logits`` exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def stages(detector, states, llm: str | None = None) -> dict[str, Any]:
    import torch

    from hprobes.features import adaptive_windows, pool_layers

    x = torch.as_tensor(np.asarray(states, dtype=np.float32))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    b = detector.bundle
    name = detector.resolve(tuple(x.shape[1:]), llm, any_layers=True)
    _, _, pca = detector.llms[name]
    if pca is not None and pca.scope != "per-layer":
        raise ValueError("diagnose supports detectors without PCA or with a per-layer PCA")
    out_layers = len(pca.bases) if pca is not None else b["head"]["input_dim"] // x.shape[2]
    with torch.inference_mode():
        pooled = pool_layers(x.to(detector.device), b["pooling"], b["pool_kernel"], b["pool_stride"], out_layers).cpu()
        z = pca.transform(pooled, detector.device) if pca is not None else pooled  # what the MLP reads
        logits = detector.model(z.flatten(1).to(detector.device)).float().cpu().numpy()
    return {
        "llm": name, "input_shape": tuple(x.shape), "pooled_shape": tuple(pooled.shape),
        "windows": adaptive_windows(x.shape[1], out_layers).tolist(),
        "raw_norm": x.norm(dim=2).median(0).values.numpy(),          # (layers,)
        "pooled_norm": pooled.norm(dim=2).median(0).values.numpy(),   # (pooled layers,)
        "pca": z.numpy(),                                 # MLP input: (n, pooled layers, PCA dim or hidden)
        "logits": logits,
        "matches_detector": float(np.abs(logits - detector.logits(states, llm=llm, any_layers=True)).max()),
    }


def report(model_path, inputs: dict[str, Any], reference: str, device: str = "cpu", llm: str | None = None) -> dict:
    """Print a comparison of every input against ``inputs[reference]`` (normally real data of the training LLM)."""
    from hprobes.predict import Detector

    detector = Detector.load(model_path, device)
    results = {name: stages(detector, states, llm) for name, states in inputs.items()}
    ref = results[reference]["pca"].reshape(len(results[reference]["pca"]), -1)
    mean, std = ref.mean(0), ref.std(0) + 1e-6
    threshold = detector.threshold
    print(f"threshold on P(correct) = {threshold:.4f}")
    for name, r in results.items():
        z = (r["pca"].reshape(len(r["pca"]), -1) - mean) / std
        prob = 1 / (1 + np.exp(-np.clip(r["logits"], -80, 80)))
        layers = len(r["raw_norm"])
        picks = sorted({0, layers // 2, layers - 1})
        print(f"\n== {name}  (LLM slot '{r['llm']}')")
        print(f"  input {r['input_shape']} -> pooled {r['pooled_shape']}; first windows {r['windows'][:3]}, last {r['windows'][-1]}")
        print("  median raw norm at layers " + ", ".join(f"{i}: {r['raw_norm'][i]:.1f}" for i in picks))
        print("  median pooled norm first/mid/last: " + ", ".join(f"{v:.1f}" for v in r["pooled_norm"][[0, len(r['pooled_norm']) // 2, -1]]))
        print(f"  MLP input features vs '{reference}': median |z| = {np.median(np.abs(z)):.2f}, 99th pct |z| = {np.percentile(np.abs(z), 99):.1f}, "
              f"max |z| = {np.abs(z).max():.1f}")
        print(f"  raw logit min / median / max = {r['logits'].min():.2f} / {np.median(r['logits']):.2f} / {r['logits'].max():.2f}; "
              f"P(correct) median = {np.median(prob):.4f}; predicted correct = {int((prob >= threshold).sum())}/{len(prob)}")
        print(f"  stages reproduce Detector.logits: max |diff| = {r['matches_detector']:.2e}")
    return results
