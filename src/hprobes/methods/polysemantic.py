"""Polysemantic tokens: one transformer token per hidden unit (its trajectory across layers).

Each hidden unit's (pooled) layer trajectory, standardized with source-train statistics, is
embedded by a shared Linear(layers -> d). Neuron identity comes from a learned per-neuron
position table (--neuron-positions learned) and/or data-derived descriptors (source-train mean,
spread and label correlation per layer; disable with --no-descriptors). Without the position
table the model works for any hidden size. Encoder: a Perceiver bottleneck (--poly-encoder
perceiver; latents cross-attend to the neuron tokens, mean of latents = embedding) or full
self-attention with [CLS] (--poly-encoder full). The embedding feeds the residual MLP head.
"""

from __future__ import annotations

import math

from hprobes.data import StateDataset
from hprobes.features import pool_layers, pooled_layer_count
from hprobes.losses import make_loss
from hprobes.training import fit, predict, residual_mlp, seed_everything


def run(args, data, device):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if args.pca_dim:
        raise SystemExit("--pca-dim is supported by --method mlp and sala")
    layers = pooled_layer_count(data.n_layers, args.pooling, args.pool_kernel, args.pool_stride)
    hidden = data.hidden_size
    seed_everything(args.seed)
    splits = {"train": StateDataset(data.sources, "train"), "val": StateDataset(data.sources, "val"),
              "test": StateDataset(data.sources, "test")}
    splits.update({f"target:{d.name}": StateDataset([d], "all") for d in data.targets})
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {name: DataLoader(ds, batch_size=args.batch_size if name == "train" else 256,
                                shuffle=name == "train", generator=generator if name == "train" else None,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")
               for name, ds in splits.items()}

    def pool(states):
        return pool_layers(states.float(), args.pooling, args.pool_kernel, args.pool_stride)

    total = torch.zeros(layers, hidden, dtype=torch.float64)
    total_sq, total_xy = torch.zeros_like(total), torch.zeros_like(total)
    with torch.inference_mode():
        for xb, yb in DataLoader(splits["train"], batch_size=256):
            pooled = pool(xb.to(device)).cpu().double()
            total += pooled.sum(0)
            total_sq += pooled.square().sum(0)
            total_xy += (pooled * yb.double()[:, None, None]).sum(0)
    n = len(splits["train"])
    mean64 = total / n
    std64 = (total_sq / n - mean64.square()).clamp(min=0).sqrt()
    p = float(data.labels("train").mean())
    correlation = ((total_xy / n - mean64 * p) / (std64.clamp(min=1e-8) * math.sqrt(p * (1 - p)))).clamp(-1, 1)
    descriptors = torch.cat((mean64.sign() * mean64.abs().log1p(), std64.clamp(min=1e-8).log(), correlation),
                            dim=0).T.float()
    descriptors = (descriptors - descriptors.mean(0)) / descriptors.std(0).clamp(min=1e-6)

    class Polysemantic(nn.Module):
        def __init__(self):
            super().__init__()
            d, width = args.model_dim, args.embedding_dim
            self.register_buffer("feature_mean", mean64.float())
            self.register_buffer("feature_std", std64.float().clamp(min=1e-4))
            self.token_embed = nn.Linear(layers, d)
            self.neuron_pos = nn.Parameter(torch.zeros(1, hidden, d)) if args.neuron_positions == "learned" else None
            if args.descriptors:
                self.register_buffer("descriptors", descriptors.clone())
                self.descriptor_embed = nn.Sequential(nn.Linear(descriptors.shape[1], d), nn.GELU(), nn.Linear(d, d))
            self.input_dropout = nn.Dropout(args.dropout)
            if args.poly_encoder == "full":
                self.cls = nn.Parameter(torch.zeros(1, 1, d))
                layer = nn.TransformerEncoderLayer(d, args.heads, 4 * d, args.dropout, activation="gelu",
                                                   batch_first=True, norm_first=True)
                self.blocks = nn.TransformerEncoder(layer, args.fusion_layers, norm=nn.LayerNorm(d),
                                                    enable_nested_tensor=False)
                self.to_embedding = nn.Linear(d, width)
                nn.init.trunc_normal_(self.cls, std=0.02)
            else:
                self.token_norm = nn.LayerNorm(d)
                self.latents = nn.Parameter(torch.zeros(1, args.latents, width))
                self.cross_norm = nn.LayerNorm(width)
                self.cross_attn = nn.MultiheadAttention(width, args.latent_heads, dropout=args.dropout,
                                                        kdim=d, vdim=d, batch_first=True)
                self.cross_ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
                                               nn.Dropout(args.dropout), nn.Linear(2 * width, width))
                layer = nn.TransformerEncoderLayer(width, args.latent_heads, 2 * width, args.dropout,
                                                   activation="gelu", batch_first=True, norm_first=True)
                self.blocks = nn.TransformerEncoder(layer, args.fusion_layers, norm=nn.LayerNorm(width),
                                                    enable_nested_tensor=False)
                nn.init.trunc_normal_(self.latents, std=0.02)
            self.head = residual_mlp(width, args.hidden_dim, args.hidden_layers, args.dropout)
            if self.neuron_pos is not None:
                nn.init.trunc_normal_(self.neuron_pos, std=0.02)

        def forward(self, states):
            x = ((pool(states) - self.feature_mean) / self.feature_std).clamp(-args.feature_clip, args.feature_clip)
            tokens = self.token_embed(x.transpose(1, 2))
            if self.neuron_pos is not None:
                tokens = tokens + self.neuron_pos
            if args.descriptors:
                tokens = tokens + self.descriptor_embed(self.descriptors)
            tokens = self.input_dropout(tokens)
            if args.poly_encoder == "full":
                sequence = torch.cat((self.cls.expand(len(tokens), -1, -1), tokens), dim=1)
                embedding = self.to_embedding(self.blocks(sequence)[:, 0])
            else:
                tokens = self.token_norm(tokens)
                latents = self.latents.expand(len(tokens), -1, -1)
                attended, _ = self.cross_attn(self.cross_norm(latents), tokens, tokens, need_weights=False)
                latents = latents + attended
                latents = latents + self.cross_ffn(latents)
                embedding = self.blocks(latents).mean(1)
            return self.head(embedding).float()

    model = Polysemantic().to(device)

    def step(batch):
        return model(batch[0].to(device, non_blocking=True)), batch[1].to(device, non_blocking=True)

    print(f"polysemantic: encoder={args.poly_encoder}; {hidden} neuron tokens of length {layers}; "
          f"positions={args.neuron_positions}; descriptors={args.descriptors}; "
          f"parameters={sum(q.numel() for q in model.parameters()):,}", flush=True)
    best_auc, best_epoch, history = fit(model, loaders["train"], loaders["val"], data.labels("val"), step,
                                        make_loss(args, data.labels("train")), args, device)
    logits = {name: predict(model, loader, step, device, args) for name, loader in loaders.items() if name != "train"}
    summary = {"encoder": args.poly_encoder, "neuron_tokens": hidden, "token_length": layers,
               "neuron_positions": args.neuron_positions, "descriptors": args.descriptors,
               "embedding_dim": args.embedding_dim, "parameters": sum(q.numel() for q in model.parameters()),
               "best_source_val_auroc": best_auc, "best_epoch": best_epoch, "history": history}
    return summary, logits["val"], logits["test"], {d.name: logits[f"target:{d.name}"] for d in data.targets}
