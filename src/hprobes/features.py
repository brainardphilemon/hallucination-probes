"""Layer pooling and source-train PCA for (layers x hidden) hidden-state matrices."""

from __future__ import annotations

import numpy as np

POOLING = ("none", "max", "mean", "logsumexp")


def pooled_layer_count(n_layers: int, pooling: str, kernel: int = 3, stride: int = 2) -> int:
    if pooling == "none":
        return n_layers
    pad = kernel // 2
    return (n_layers + 2 * pad - kernel) // stride + 1


def pool_layers(states, pooling: str, kernel: int = 3, stride: int = 2):
    """Pool a (batch, layers, hidden) tensor across the layer axis, independently per hidden unit.

    logsumexp: soft maximum over each window with replicate padding at the ends.
    max / mean: standard 1-D pooling with padding kernel // 2 (mean ignores padding).
    For 29 Qwen2.5-7B layers, kernel 3 and stride 2 give 15 pooled layers.
    """
    import torch
    from torch.nn import functional as F

    if pooling == "none":
        return states
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
            pooled = pool_layers(xb.to(device).float(), args.pooling, args.pool_kernel, args.pool_stride).cpu()
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

    def summary(self) -> dict:
        return {"dim": self.dim, "scope": self.scope, "groups": len(self.bases),
                "mean_explained_variance": float(np.mean(self.explained)) if self.explained else None}
