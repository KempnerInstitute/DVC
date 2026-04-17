"""Minimal MINE (Mutual Information Neural Estimation) baseline.

Implements the Donsker--Varadhan representation
    MI(X; Y) = sup_T  E_{p(x,y)}[T(x, y)] - log E_{p(x)p(y)}[exp(T(x, y))]
with a small MLP discriminator \\(T_\\theta\\) \\citep{Belghazi2018}.
Used for comparing per-window pairwise mutual-information estimates
against DVC's copula-derived estimates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _MINEStatistic(nn.Module):
    """Small MLP realizing the statistic \\(T_\\theta(x, y)\\)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy).squeeze(-1)


def mine_mi_estimate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    ema: float = 0.01,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> float:
    """Estimate mutual information between two (univariate) samples.

    ``x`` and ``y`` should be 1D arrays of equal length.
    Returns the mean MI estimate over the final $20\\%$ of training epochs
    in nats.
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have equal length")

    n = x.shape[0]
    dev = torch.device(device) if device else torch.device("cpu")
    x_t = torch.from_numpy(x).float().unsqueeze(-1).to(dev)
    y_t = torch.from_numpy(y).float().unsqueeze(-1).to(dev)

    model = _MINEStatistic(input_dim=2, hidden_dim=hidden_dim).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)

    ma_et = 1.0
    batch = min(batch_size, n)
    recent: list[float] = []
    for epoch in range(int(n_epochs)):
        idx = torch.randperm(n, device=dev)
        idx_marg = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            b = idx[i : i + batch]
            bm = idx_marg[i : i + batch]
            joint = torch.cat([x_t[b], y_t[b]], dim=-1)
            marg = torch.cat([x_t[b], y_t[bm]], dim=-1)
            t_joint = model(joint)
            t_marg = model(marg)
            with torch.no_grad():
                ma_et = (1.0 - ema) * ma_et + ema * torch.exp(t_marg).mean().item()
            # Bias-corrected Donsker-Varadhan objective
            ma_et = max(ma_et, 1e-6)
            loss = -(t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8)
                     * (torch.exp(t_marg).mean().detach() / ma_et))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch >= int(0.8 * n_epochs):
            model.eval()
            with torch.no_grad():
                idx_all = torch.arange(n, device=dev)
                idx_marg_all = torch.randperm(n, device=dev)
                joint = torch.cat([x_t, y_t], dim=-1)
                marg = torch.cat([x_t, y_t[idx_marg_all]], dim=-1)
                mi_est = model(joint).mean().item() - np.log(torch.exp(model(marg)).mean().item() + 1e-8)
                recent.append(float(mi_est))
            model.train()

    return float(np.mean(recent)) if recent else float("nan")
