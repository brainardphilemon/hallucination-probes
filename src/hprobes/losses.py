"""Binary classification losses: BCE, focal (Lin et al., 2017) and logit-adjusted BCE (Menon et al., 2021)."""

from __future__ import annotations

import numpy as np

LOSSES = ("bce", "focal", "logit_adjusted")


def sigmoid_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0):
    """Mean binary focal loss; alpha weights the positive class (negative alpha disables weighting)."""
    import torch.nn.functional as F

    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean()


def logit_adjustment(labels: np.ndarray, tau: float = 1.0) -> float:
    """tau * log(prior_1 / prior_0), added to the positive logit inside the training loss only."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if np.any(counts == 0) or tau <= 0:
        raise ValueError("Logit adjustment needs both classes and tau > 0")
    priors = counts / counts.sum()
    return float(tau * np.log(priors[1] / priors[0]))


def make_loss(args, train_labels: np.ndarray):
    """Return loss_fn(logits, targets) for args.loss."""
    from torch import nn

    if args.loss == "focal":
        return lambda logits, targets: sigmoid_focal_loss(logits, targets, args.focal_alpha, args.focal_gamma)
    bce = nn.BCEWithLogitsLoss()
    if args.loss == "logit_adjusted":
        shift = logit_adjustment(train_labels, args.logit_adjust_tau)
        return lambda logits, targets: bce(logits + shift, targets)
    return bce
