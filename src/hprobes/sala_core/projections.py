# Adapted from the official SALA implementation (https://github.com/Nellie179/SALA),
# Copyright (c) 2026 The SALA Authors, released under the MIT License (see LICENSE).



import math
import numpy as np
import torch
from hprobes.sala_core.isr import ISR

def compute_isr_init_W(
    X_src_tr_std: np.ndarray,
    y_src_tr: np.ndarray,
    dom_src_tr: np.ndarray,
    proj_dim: int,
    extracted_class: int = 0,
    fit_method: str = "cov",
    l2_reg: float = 0.01,
    device: torch.device = None,
    min_env_size: int = 5,
) -> np.ndarray:
    """Compute an ISR projection matrix to be used as init_W (shape [d, proj_dim])."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_ids = np.unique(dom_src_tr)
    envs = []
    for e in env_ids:
        m = (dom_src_tr == e)
        X_env = X_src_tr_std[m]
        y_env = y_src_tr[m]
        if X_env.shape[0] < min_env_size:
            continue
        envs.append((X_env, y_env))

    d = int(X_src_tr_std.shape[1])
    k = int(proj_dim)

    if len(envs) < 2:
        Q, _ = np.linalg.qr(np.random.randn(d, k))
        return Q.astype(np.float32)

    isr = ISR(
        dim_inv=k,
        fit_method=fit_method,
        l2_reg=l2_reg,
        verbose=False,
        logistic_regression=False,
        device=device,
    )
    P = isr.fit(
        envs,
        extracted_class=extracted_class,
        fit_clf=False,
        return_proj_mat=True,
    )
    P = np.asarray(P, dtype=np.float32)

    if P.shape[0] != d:
        raise ValueError(f"ISR proj_mat first dim {P.shape[0]} != feature dim {d}")

    if P.shape[1] != k:
        if P.shape[1] > k:
            P = P[:, :k]
        else:
            pad = (np.random.randn(d, k - P.shape[1]).astype(np.float32) * 0.01)
            P = np.concatenate([P, pad], axis=1)
    return P

def train_layerwise_invariant_projection(
    X_src_tr_std: np.ndarray,   
    y_src_tr: np.ndarray,       
    dom_src_tr: np.ndarray,     
    proj_dim: int = 16,
    n_epochs: int = 300,
    lr: float = 1e-3,
    lam_inv: float = 1.0,
    lam_sep: float = 0.1,
    lam_reg: float = 0.0,
    init_W: np.ndarray = None,
    device: torch.device = None,
    verbose: bool = True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(X_src_tr_std).float().to(device)  # [N, d]
    y = torch.from_numpy(y_src_tr).long().to(device)       # [N]
    dom = torch.from_numpy(dom_src_tr).long().to(device)   # [N]

    N, d = X.shape
    k = int(proj_dim)
    
    # ---- init W: use init_W if provided, else random ----
    if init_W is not None:
        init_W = np.asarray(init_W, dtype=np.float32)
        assert init_W.shape[0] == d, f"init_W first dim {init_W.shape[0]} must equal d={d}"
        if init_W.shape[1] >= k:
            W_init = init_W[:, :k]
        else:
            pad = (np.random.randn(d, k - init_W.shape[1]).astype(np.float32) * 0.01)
            W_init = np.concatenate([init_W, pad], axis=1)
        W = torch.nn.Parameter(torch.from_numpy(W_init).to(device))
    else:
        W = torch.nn.Parameter(torch.randn(d, k, device=device) * (1.0 / math.sqrt(d)))

    # anchor so lam_reg works (even with random init)
    W_ref = W.detach().clone() if lam_reg > 0.0 else None

    # ---- optimizer (true L2) ----
    optimizer = torch.optim.Adam([W], lr=lr, weight_decay=1e-4)

    env_ids = dom.unique().tolist()
    eps = 1e-12

    for ep in range(1, n_epochs + 1):
        Z = X @ W 

        L_inv = torch.zeros((), device=device)
        pair_cnt = 0
        mean_by_y = {}

        for y_val in [0, 1]:
            class_mus = []
            for e_val in env_ids:
                mask = (y == y_val) & (dom == e_val)
                if mask.sum().item() < 2:
                    continue
                class_mus.append(Z[mask].mean(dim=0))  # [k]

            if len(class_mus) >= 2:
                for i in range(len(class_mus)):
                    for j in range(i + 1, len(class_mus)):
                        diff = class_mus[i] - class_mus[j]
                        L_inv = L_inv + (diff * diff).sum()
                        pair_cnt += 1

                stacked = torch.stack(class_mus, dim=0)
                mean_by_y[y_val] = stacked.mean(dim=0)

        # normalize invariance loss: by pair count and by k
        if pair_cnt > 0:
            L_inv = L_inv / float(pair_cnt)
        L_inv = L_inv / float(max(1, k))

        # separation term (maximize class mean gap), normalized by k
        if 0 in mean_by_y and 1 in mean_by_y:
            diff_cls = mean_by_y[1] - mean_by_y[0]
            L_sep = - (diff_cls * diff_cls).sum() / float(max(1, k))
        else:
            L_sep = torch.zeros((), device=device)

        # anchor reg (optional)
        if lam_reg > 0.0 and W_ref is not None:
            L_reg = ((W - W_ref) ** 2).sum() / (float(d * max(1, k)) + eps)
        else:
            L_reg = torch.zeros((), device=device)

        loss = lam_inv * L_inv + lam_sep * L_sep + lam_reg * L_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (ep % 50 == 0 or ep == 1):
            print(
                f"[InvSub] epoch {ep}/{n_epochs} | "
                f"L_inv={L_inv.item():.4f} | L_sep={L_sep.item():.4f} | L_reg={L_reg.item():.4f}"
            )

    return W.detach().cpu().numpy().astype(np.float32)