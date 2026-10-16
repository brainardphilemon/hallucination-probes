"""Layer pooling and source-train PCA for (layers x hidden) hidden-state matrices."""

from __future__ import annotations

import numpy as np

POOLING = ("none", "max", "mean", "logsumexp")


def pooled_layer_count(n_layers: int, pooling: str, kernel: int = 3, stride: int = 2, out_layers: int = 0) -> int:
    if pooling == "none":
        return n_layers
    if out_layers:
        return out_layers
    pad = kernel // 2
    return (n_layers + 2 * pad - kernel) // stride + 1


def adaptive_windows(n_layers: int, out_layers: int):
    """Index windows (out_layers, width) that bring any number of layers down to ``out_layers``.

    Window centers are spread evenly over the layers (first and last layer included) and each window
    reaches ceil(spacing / 2) layers to either side; indices beyond the ends repeat the edge layer, which
    is the same as replicate padding. For 29 layers -> 15 this is exactly kernel 3, stride 2.
    """
    import math

    import torch

    if out_layers < 2 or n_layers < 2:
        raise ValueError("Adaptive pooling needs at least 2 input and 2 output layers")
    spacing = (n_layers - 1) / (out_layers - 1)
    half = max(1, math.ceil(spacing / 2)) if spacing > 1 else 0
    centers = torch.round(torch.arange(out_layers) * spacing).long()
    offsets = torch.arange(-half, half + 1)
    return (centers[:, None] + offsets[None]).clamp(0, n_layers - 1)


def pool_layers(states, pooling: str, kernel: int = 3, stride: int = 2, out_layers: int = 0):
    """Pool a (batch, layers, hidden) tensor across the layer axis, independently per hidden unit.

    logsumexp: soft maximum over each window with replicate padding at the ends.
    max / mean: standard 1-D pooling with padding kernel // 2 (mean ignores padding).
    For 29 Qwen2.5-7B layers, kernel 3 and stride 2 give 15 pooled layers.
    out_layers > 0 ignores kernel/stride and always produces ``out_layers`` layers, whatever the number of
    input layers (see adaptive_windows); the hidden size is untouched.
    """
    import torch
    from torch.nn import functional as F

    if pooling == "none":
        return states
    if out_layers:
        windows = states[:, adaptive_windows(states.shape[1], out_layers).to(states.device)]  # (B, out, width, H)
        if pooling == "logsumexp":
            return torch.logsumexp(windows, dim=2)
        if pooling == "max":
            return windows.amax(dim=2)
        if pooling == "mean":
            return windows.mean(dim=2)
        raise ValueError(f"Unknown pooling: {pooling}")
    pad = kernel // 2
    channels_first = states.transpose(1, 2)
    if pooling == "logsumexp":
        windows = F.pad(channels_first, (pad, pad), mode="replicate").unfold(2, kernel, stride)
        pooled = torch.logsumexp(windows, dim=-1)
    elif pooling == "max":
        pooled = F.max_pool1d(channels_first, kernel_size=kernel, stride=stride, padding=pad)
    elif pooling == "mean":
        pooled = F.avg_pool1d(channels_first, kernel_size=kernel, stride=stride, padding=pad,
                              count_include_pad=False)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")
    return pooled.transpose(1, 2)


def pooled_matrix(dataset, args, device, batch_size: int = 256):
    """Pooled features of a dataset as a CPU float32 tensor (rows, pooled_layers, hidden).

    Preallocated on the CPU so unpooled 29-layer features fit even when the GPU cannot hold them.
    """
    import torch
    from torch.utils.data import DataLoader

    out = None
    position = 0
    with torch.inference_mode():
        for xb, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            pooled = pool_layers(xb.to(device).float(), args.pooling, args.pool_kernel, args.pool_stride, args.pool_layers).cpu()
            if out is None:
                out = torch.empty(len(dataset), *pooled.shape[1:])
            out[position:position + len(xb)] = pooled
            position += len(xb)
    return out


class LayerPCA:
    """PCA fitted on source-train rows only: one PCA over all features ('global') or one per layer."""

    def __init__(self, dim: int, scope: str, seed: int):
        self.dim, self.scope, self.seed = dim, scope, seed
        self.means, self.bases, self.explained = [], [], []

    def _groups(self, x):
        return list(range(x.shape[1])) if self.scope == "per-layer" else [None]

    @staticmethod
    def _block(x, group):
        return x[:, group, :] if group is not None else x.reshape(len(x), -1)

    def fit_transform(self, x, device):
        """x: (rows, layers, hidden) CPU tensor, consumed (centered in place on CPU to save memory).
        Returns (rows, groups, dim) CPU tensor."""
        import torch

        parts = []
        with torch.inference_mode(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(self.seed)
            for group in self._groups(x):
                block = self._block(x, group).to(device)
                self.means.append(block.mean(0))
                block -= self.means[-1]
                _, singular, basis = torch.svd_lowrank(block, q=self.dim, niter=4)
                self.bases.append(basis)
                self.explained.append(float((singular.square() / block.square().sum()).sum()))
                parts.append((block @ basis).cpu())
                del block
        return torch.stack(parts, dim=1)

    def transform(self, x, device):
        import torch

        return torch.stack([((self._block(x, g).to(device) - m) @ b).cpu()
                            for g, m, b in zip(self._groups(x), self.means, self.bases)], dim=1)

    def state(self) -> dict:
        return {"dim": self.dim, "scope": self.scope, "seed": self.seed,
                "means": [m.cpu() for m in self.means], "bases": [b.cpu() for b in self.bases]}

    @classmethod
    def from_state(cls, state: dict) -> "LayerPCA":
        pca = cls(state["dim"], state["scope"], state["seed"])
        pca.means, pca.bases = list(state["means"]), list(state["bases"])
        return pca

    def summary(self) -> dict:
        return {"dim": self.dim, "scope": self.scope, "groups": len(self.bases),
                "mean_explained_variance": float(np.mean(self.explained)) if self.explained else None}
