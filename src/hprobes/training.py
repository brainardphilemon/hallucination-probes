"""Shared pieces for every method: the residual MLP head and one training loop."""

from __future__ import annotations

import math
import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str):
    import torch

    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else
                        name if name == "mps" else "cpu")


def amp_dtype(device):
    import torch

    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def residual_mlp(input_dim: int, hidden_dim: int, hidden_layers: int, dropout: float = 0.2):
    """SALA's StrongMLP pattern with a configurable depth: Linear -> (LayerNorm, Mish, Dropout, Linear) blocks."""
    from torch import nn

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_in = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Mish(), nn.Dropout(dropout),
                              nn.Linear(hidden_dim, hidden_dim))
                for _ in range(hidden_layers - 1)
            ])
            self.out = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            h = self.fc_in(x)
            for block in self.blocks:
                h = h + block(h)
            return self.out(h).squeeze(-1)

    return ResidualMLP()


def lr_lambda(schedule: str, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(warmup_ratio * total_steps))

    def factor(step: int) -> float:
        if schedule == "constant":
            return 1.0
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def fit(model, train_loader, val_loader, y_val, step, loss_fn, args, device, param_groups=None,
        log_prefix: str = ""):
    """Train with AdamW, keep the epoch with the best source-validation AUROC, restore it.

    step(batch) -> (logits, labels) moves a batch to the device and runs the model.
    Returns (best_auc, best_epoch, history).
    """
    import torch
    from sklearn.metrics import roc_auc_score

    dtype = amp_dtype(device) if args.amp else None
    groups = param_groups or [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(args.lr_schedule, len(train_loader) * args.epochs, args.warmup_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    best_auc, best_epoch, best_state, history = -1.0, None, None, []
    for epoch in range(args.epochs):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits, labels = step(batch)
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.item()) * len(labels)
            seen += len(labels)
        val_auc = float(roc_auc_score(y_val, predict(model, val_loader, step, device, args)))
        history.append({"epoch": epoch + 1, "loss": total / max(seen, 1), "source_val_auroc": val_auc})
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{log_prefix}epoch {epoch + 1}/{args.epochs} loss={total / max(seen, 1):.5f} "
              f"source_val_auc={val_auc:.5f} best={best_auc:.5f}", flush=True)
    model.load_state_dict(best_state)
    return best_auc, best_epoch, history


def predict(model, loader, step, device, args):
    import torch

    dtype = amp_dtype(device) if args.amp else None
    model.eval()
    logits = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        for batch in loader:
            logits.append(step(batch)[0].float().cpu().numpy())
    return np.concatenate(logits).reshape(-1)


def tensor_loader(x, y, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TensorDataset(torch.as_tensor(x).float(), torch.as_tensor(y).float()),
                      batch_size=batch_size, shuffle=shuffle, generator=generator)
