"""Normalizing-flow copula baseline.

A minimal Real-NVP-style coupling-flow copula for held-out log-density evaluation.
The density is defined on normal scores $z = \\Phi^{-1}(u)$ and the copula log-density is

    log c(u) = 0.5 (||z||^2 - ||w||^2) + log|det J_f(z)|

where $w = f(z)$ is the flow output with base density $\\mathcal{N}(0, I)$.
This is the standard copula-by-flow construction (Laszkiewicz et al., 2021; Rezende & Mohamed, 2015).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _pseudo_obs_rank_np(x: np.ndarray) -> np.ndarray:
    """Rank-based pseudo-observations in (0, 1), matching the repo convention."""
    ranks = np.apply_along_axis(
        lambda col: _scipy_rankdata(col, method="average"), 0, x
    )
    return ranks / (x.shape[0] + 1.0)


def _scipy_rankdata(values: np.ndarray, method: str = "average") -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return rankdata(values, method=method)
    except Exception:
        # Fallback (used only if scipy import fails): not truly average-rank,
        # but the rank positions on a continuous sample are unique almost surely.
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(values) + 1)
        return ranks


class _CouplingLayer(nn.Module):
    """Real-NVP coupling layer with affine transform on masked-out dimensions."""

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_masked = z * self.mask
        s = self.scale_net(z_masked) * (1.0 - self.mask)
        t = self.shift_net(z_masked) * (1.0 - self.mask)
        w = z_masked + (1.0 - self.mask) * (z * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return w, log_det


class NFCopula(nn.Module):
    """Real-NVP copula over normal-score inputs."""

    def __init__(self, dim: int, n_blocks: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        masks = []
        for i in range(n_blocks):
            m = torch.zeros(dim, dtype=torch.float32)
            if i % 2 == 0:
                m[: dim // 2] = 1.0
            else:
                m[dim - dim // 2 :] = 1.0
            masks.append(m)
        self.layers = nn.ModuleList(
            [_CouplingLayer(dim, hidden_dim, m) for m in masks]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def log_copula_density(self, u: torch.Tensor) -> torch.Tensor:
        """Log copula density evaluated at pseudo-observations $u \\in (0,1)^d$."""
        u_clamped = u.clamp(1e-6, 1.0 - 1e-6)
        z = torch.special.ndtri(u_clamped)
        w, log_det = self.forward(z)
        return 0.5 * ((z ** 2).sum(dim=-1) - (w ** 2).sum(dim=-1)) + log_det


def nf_copula_nll_fit_eval(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    n_epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    n_blocks: int = 4,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> float:
    """Fit a Real-NVP copula on train pseudo-observations and return test copula NLL.

    The NLL is the held-out mean $-\\log c(u)$ in nats, comparable to the other
    copula-NLL baselines in the benchmark suite.
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValueError("x_train and x_test must share a 2D shape (N, d)")
    d = int(x_train.shape[1])

    u_train = _pseudo_obs_rank_np(x_train)
    u_test = _pseudo_obs_rank_np(x_test)

    dev = torch.device(device) if device else torch.device("cpu")
    u_tr = torch.from_numpy(u_train).float().to(dev)
    u_te = torch.from_numpy(u_test).float().to(dev)

    model = NFCopula(dim=d, n_blocks=n_blocks, hidden_dim=hidden_dim).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)

    n_tr = u_tr.shape[0]
    batch = min(batch_size, n_tr)
    for _ in range(int(n_epochs)):
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            log_c = model.log_copula_density(u_tr[idx])
            loss = -log_c.mean()
            if torch.isnan(loss) or torch.isinf(loss):
                # Abort on numerical breakdown; caller will fall back.
                return float("nan")
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        log_c_test = model.log_copula_density(u_te)
        return float(-log_c_test.mean().item())
