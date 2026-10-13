"""SALA: per-layer invariant subspace projections, concatenated, then a residual MLP.

For every (optionally pooled) layer: standardize with source-train statistics, optionally reduce
with a per-layer PCA, learn an ISR-initialized invariant projection across source domains
(fixed --proj-dim or leave-one-domain-out selection with --lodo-dim), and standardize the
projected features. The per-layer projections are concatenated and fed to the MLP head.
Per-layer results are checkpointed under <output-dir>/sala_layers/ so interrupted runs resume.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hprobes.features import pool_layers, pooled_layer_count
from hprobes.io import npz_dump_atomic, prepare_manifest
from hprobes.losses import make_loss
from hprobes.training import fit, predict, residual_mlp, seed_everything, tensor_loader


def pooled_store(domain, args, device, cache_dir: Path):
    """Pooled (rows, pooled_layers, hidden) float16 array for a domain; the raw memmap if no pooling."""
    import torch

    if args.pooling == "none":
        return domain.features
    path = cache_dir / f"{domain.name}_{args.pooling}_k{args.pool_kernel}_s{args.pool_stride}.npy"
    rows = len(domain.features)
    layers = pooled_layer_count(domain.features.shape[1], args.pooling, args.pool_kernel, args.pool_stride)
    if path.exists():
        store = np.load(path, mmap_mode="r")
        if store.shape == (rows, layers, domain.features.shape[2]):
            return store
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                      shape=(rows, layers, domain.features.shape[2]))
    with torch.inference_mode():
        for start in range(0, rows, 256):
            block = torch.from_numpy(np.asarray(domain.features[start:start + 256], dtype=np.float32)).to(device)
            store[start:start + 256] = pool_layers(block, args.pooling, args.pool_kernel,
                                                   args.pool_stride).cpu().numpy().astype(np.float16)
    store.flush()
    return np.load(path, mmap_mode="r")


def layer_block(store, rows, layer) -> np.ndarray:
    return np.asarray(store[rows, layer, :], dtype=np.float32)


def probe_auc(z_train, y_train, z_val, y_val, args, device, epochs):
    """Small MLP probe used for leave-one-domain-out projection-dimension selection."""
    probe_args = SimpleNamespace(epochs=epochs, lr=1e-3, weight_decay=1e-4, lr_schedule="constant",
                                 warmup_ratio=0.0, amp=False, grad_clip=0.0)
    model = residual_mlp(z_train.shape[1], 1024, 3).to(device)
    train = tensor_loader(z_train, y_train, 512, True, args.seed)
    val = tensor_loader(z_val, y_val, 512, False, args.seed)

    def step(batch):
        return model(batch[0].to(device)), batch[1].to(device)

    from torch import nn

    best, _, _ = fit(model, train, val, y_val, step, nn.BCEWithLogitsLoss(), probe_args, device, log_prefix="    ")
    return best


def select_proj_dim_lodo(x_train, y_train, dom_train, layer, args, device, candidates):
    """Pick the projection dimension maximizing mean - 2 * std of held-out-domain AUROC."""
    from hprobes.sala_core.projections import train_layerwise_invariant_projection

    best_dim, best_score, stats = None, -math.inf, {}
    for k in candidates:
        folds = []
        for held_out in np.unique(dom_train):
            keep, hold = dom_train != held_out, dom_train == held_out
            seed_everything(args.seed + 1000 + (layer + 1) * 10000)
            w = train_layerwise_invariant_projection(
                x_train[keep], y_train[keep], dom_train[keep], proj_dim=int(k), n_epochs=args.epochs_proj_selection,
                lr=1e-3, init_W=None, device=device, verbose=False,
                lam_inv=args.lam_inv, lam_sep=args.lam_sep, lam_reg=args.lam_reg)
            folds.append(probe_auc(x_train[keep] @ w, y_train[keep], x_train[hold] @ w, y_train[hold],
                                   args, device, args.epochs_probe_selection))
        score = float(np.mean(folds) - 2.0 * np.std(folds))
        stats[int(k)] = {"folds": folds, "score": score}
        if score > best_score:
            best_dim, best_score = int(k), score
    return best_dim, best_score, stats


def run(args, data, device):
    import torch

    from hprobes.sala_core.projections import compute_isr_init_W, train_layerwise_invariant_projection

    if args.pca_dim and args.pca_scope == "global":
        raise SystemExit("SALA projects each layer separately; use --pca-scope per-layer with --pca-dim")
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "_pooled_cache"
    stores = {d.name: pooled_store(d, args, device, cache_dir) for d in [*data.sources, *data.targets]}
    layers = pooled_layer_count(data.n_layers, args.pooling, args.pool_kernel, args.pool_stride)
    y_train, y_val = data.labels("train"), data.labels("val")
    dom_train = np.concatenate([np.full(len(d.split["train"]), i) for i, d in enumerate(data.sources)])
    if args.lodo_dim and len(data.sources) < 2:
        raise SystemExit("--lodo-dim needs at least two source datasets")
    width = args.pca_dim or data.hidden_size
    candidates = [k for k in args.proj_dim_candidates if k <= width]
    if args.proj_dim > width or (args.lodo_dim and not candidates):
        raise SystemExit(f"Projection dimension must not exceed the per-layer width {width}")

    def gather(layer, split):
        return np.concatenate([layer_block(stores[d.name], d.split[split], layer) for d in data.sources])

    # Refuse to resume per-layer checkpoints produced with different settings.
    prepare_manifest(output_dir / "sala_layers" / "manifest.json", {
        "sources": [d.name for d in data.sources], "targets": [d.name for d in data.targets],
        **{k: getattr(args, k) for k in ("pooling", "pool_kernel", "pool_stride", "pca_dim", "proj_dim",
                                          "lodo_dim", "proj_dim_candidates", "epochs_proj",
                                          "epochs_proj_selection", "epochs_probe_selection", "lam_inv",
                                          "lam_sep", "lam_reg", "seed", "bleurt_threshold")}})
    parts = {name: [] for name in ["train", "val", "test", *[d.name for d in data.targets]]}
    chosen, lodo = [], []
    for layer in range(layers):
        checkpoint = output_dir / "sala_layers" / f"layer_{layer:02d}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint)
            for name in parts:
                parts[name].append(saved[name])
            chosen.append(int(saved["projection_dim"]))
            print(f"sala: resumed layer {layer + 1}/{layers}", flush=True)
            continue
        x = {"train": gather(layer, "train"), "val": gather(layer, "val"), "test": gather(layer, "test")}
        for d in data.targets:
            x[d.name] = layer_block(stores[d.name], d.split["all"], layer)
        if args.pca_dim:
            # Per-layer PCA fitted on source-train rows of this layer.
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                torch.manual_seed(args.seed + layer)
                train = torch.from_numpy(x["train"]).to(device)
                mean = train.mean(0)
                _, _, basis = torch.svd_lowrank(train - mean, q=args.pca_dim, niter=4)
            x = {k: ((torch.from_numpy(v).to(device) - mean) @ basis).cpu().numpy() for k, v in x.items()}
        mu = x["train"].mean(0, keepdims=True)
        sd = x["train"].std(0, keepdims=True) + 1e-8
        x = {k: (v - mu) / sd for k, v in x.items()}
        dim = args.proj_dim
        if args.lodo_dim:
            dim, score, _ = select_proj_dim_lodo(x["train"], y_train, dom_train, layer, args, device, candidates)
            lodo.append({"layer": layer, "projection_dim": dim, "score": score})
        seed_everything(args.seed + 1000 + (layer + 1) * 10000)
        init_w = compute_isr_init_W(x["train"], y_train, dom_train, proj_dim=dim, device=device)
        w = train_layerwise_invariant_projection(
            x["train"], y_train, dom_train, init_W=init_w, proj_dim=dim, n_epochs=args.epochs_proj, lr=1e-3,
            lam_inv=args.lam_inv, lam_sep=args.lam_sep, lam_reg=args.lam_reg, device=device, verbose=False)
        z_train = x["train"] @ w
        z_mu, z_sd = z_train.mean(0, keepdims=True), z_train.std(0, keepdims=True) + 1e-6
        z = {k: ((v @ w - z_mu) / z_sd).astype(np.float32) for k, v in x.items()}
        npz_dump_atomic(checkpoint, projection_dim=np.asarray(dim), **z)
        for name in parts:
            parts[name].append(z[name])
        chosen.append(int(dim))
        print(f"sala: projected layer {layer + 1}/{layers} dim={dim}", flush=True)

    z = {name: np.concatenate(v, axis=1) for name, v in parts.items()}
    seed_everything(args.seed)
    model = residual_mlp(z["train"].shape[1], args.hidden_dim, args.hidden_layers, args.dropout).to(device)
    loaders = {"train": tensor_loader(z["train"], y_train, args.batch_size, True, args.seed),
               "val": tensor_loader(z["val"], y_val, 256, False, args.seed),
               "test": tensor_loader(z["test"], data.labels("test"), 256, False, args.seed)}
    for d in data.targets:
        loaders[d.name] = tensor_loader(z[d.name], d.labels, 256, False, args.seed)

    def step(batch):
        return model(batch[0].to(device)), batch[1].to(device)

    best_auc, best_epoch, history = fit(model, loaders["train"], loaders["val"], y_val, step,
                                        make_loss(args, y_train), args, device)
    logits = {name: predict(model, loader, step, device, args) for name, loader in loaders.items()}
    summary = {"pooled_layers": layers, "per_layer_pca_dim": args.pca_dim or None,
               "projection_dimension_policy": "lodo" if args.lodo_dim else "fixed",
               "selected_projection_dimensions": chosen, "lodo_selection": lodo,
               "projected_concat_dimension": int(z["train"].shape[1]),
               "best_source_val_auroc": best_auc, "best_epoch": best_epoch, "history": history}
    return summary, logits["val"], logits["test"], {d.name: logits[d.name] for d in data.targets}
