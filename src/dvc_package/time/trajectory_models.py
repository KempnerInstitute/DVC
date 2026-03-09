"""
Reusable temporal trajectory models.

These modules expose a common interface for mapping time -> parameter vectors.
They are intended to be shared by:
- dynamic bandwidth models
- jointly parameterized vine edge trajectories
- latent state-space dynamic dependence models
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeTrajectoryBase(nn.Module):
    """Base class for time -> parameter trajectory models."""

    def __init__(
        self,
        output_dim: int,
        *,
        constraint: str = "identity",
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.constraint = str(constraint)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.register_buffer("_time_min", torch.tensor(0.0), persistent=False)
        self.register_buffer("_time_max", torch.tensor(1.0), persistent=False)
        self.register_buffer("_reference_grid", torch.linspace(0.0, 1.0, 2), persistent=False)

    def set_time_range(self, t_min: float, t_max: float) -> None:
        t_lo = float(min(t_min, t_max))
        t_hi = float(max(t_min, t_max))
        if abs(t_hi - t_lo) < 1e-8:
            t_hi = t_lo + 1.0
        self._time_min = torch.tensor(t_lo, dtype=torch.float32, device=self._time_min.device)
        self._time_max = torch.tensor(t_hi, dtype=torch.float32, device=self._time_max.device)

    def set_reference_time_grid(self, time_points: torch.Tensor) -> None:
        t = torch.as_tensor(time_points, dtype=torch.float32, device=self._time_min.device).reshape(-1)
        if t.numel() < 2:
            t = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self._time_min.device)
        self._reference_grid = t
        self.set_time_range(float(t.min().item()), float(t.max().item()))

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        denom = (self._time_max - self._time_min).clamp_min(1e-8)
        return ((t - self._time_min) / denom).clamp(0.0, 1.0)

    def _apply_constraint(self, raw: torch.Tensor) -> torch.Tensor:
        if self.constraint == "identity":
            return raw
        if self.constraint == "positive":
            return F.softplus(raw) + self.min_value
        if self.constraint == "bounded":
            return self.min_value + (self.max_value - self.min_value) * torch.sigmoid(raw)
        raise ValueError(f"Unknown trajectory constraint: {self.constraint}")

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self._time_min.device)


class BasisTrajectory(TimeTrajectoryBase):
    """Smooth radial-basis trajectory."""

    def __init__(
        self,
        output_dim: int,
        *,
        n_basis: int = 4,
        width_scale: float = 1.5,
        constraint: str = "identity",
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__(
            output_dim,
            constraint=constraint,
            min_value=min_value,
            max_value=max_value,
        )
        self.n_basis = int(max(n_basis, 1))
        self.width_scale = float(max(width_scale, 1e-3))
        self.coefficients = nn.Parameter(torch.zeros(self.n_basis, self.output_dim))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def _basis(self, t01: torch.Tensor) -> torch.Tensor:
        if self.n_basis == 1:
            return torch.ones(t01.shape[0], 1, dtype=t01.dtype, device=t01.device)
        centers = torch.linspace(0.0, 1.0, self.n_basis - 1, device=t01.device, dtype=t01.dtype)
        spacing = 1.0 / max(self.n_basis - 2, 1)
        width = max(self.width_scale * spacing, 1e-3)
        phi = torch.exp(-0.5 * ((t01 - centers[None, :]) / width) ** 2)
        return torch.cat([torch.ones_like(t01[:, :1]), phi], dim=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t01 = self._normalize_time(t.to(dtype=torch.float32))
        basis = self._basis(t01)
        raw = basis @ self.coefficients + self.bias
        return self._apply_constraint(raw)


class MLPTrajectory(TimeTrajectoryBase):
    """Generic MLP time -> parameter trajectory."""

    def __init__(
        self,
        output_dim: int,
        *,
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        constraint: str = "identity",
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__(
            output_dim,
            constraint=constraint,
            min_value=min_value,
            max_value=max_value,
        )
        hidden = [int(v) for v in (hidden_dims or [64, 32])]
        layers = []
        in_dim = 1
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(self._activation(activation))
            if dropout_rate > 0:
                layers.append(nn.Dropout(float(dropout_rate)))
            in_dim = width
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _activation(self, name: str) -> nn.Module:
        if name == "elu":
            return nn.ELU()
        if name == "tanh":
            return nn.Tanh()
        if name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        return nn.ReLU()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        raw = self.network(self._normalize_time(t.to(dtype=torch.float32)))
        return self._apply_constraint(raw)


class StateSpaceTrajectory(TimeTrajectoryBase):
    """
    Low-rank latent state-space trajectory with linear interpolation in time.

    The latent state sequence is defined on a reference time grid. Outputs are
    generated by a shared linear readout and can be regularized by an AR(1)-style
    transition penalty.
    """

    def __init__(
        self,
        output_dim: int,
        *,
        latent_dim: int = 3,
        n_steps: int = 8,
        transition_penalty: float = 1e-2,
        constraint: str = "identity",
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        super().__init__(
            output_dim,
            constraint=constraint,
            min_value=min_value,
            max_value=max_value,
        )
        self.latent_dim = int(max(latent_dim, 1))
        self.n_steps = int(max(n_steps, 2))
        self.transition_penalty = float(max(transition_penalty, 0.0))
        self.latent_states = nn.Parameter(torch.zeros(self.n_steps, self.latent_dim))
        self.readout = nn.Linear(self.latent_dim, self.output_dim)
        self.phi_unconstrained = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        nn.init.normal_(self.latent_states, mean=0.0, std=0.05)

    def set_reference_time_grid(self, time_points: torch.Tensor) -> None:
        super().set_reference_time_grid(time_points)
        n_ref = max(int(torch.as_tensor(time_points).numel()), 2)
        if n_ref != self.n_steps:
            self.n_steps = n_ref
            device = self.latent_states.device
            new_param = nn.Parameter(torch.zeros(self.n_steps, self.latent_dim, device=device))
            nn.init.normal_(new_param, mean=0.0, std=0.05)
            self.latent_states = new_param

    def _interpolate_states(self, t01: torch.Tensor) -> torch.Tensor:
        ref01 = self._normalize_time(self._reference_grid.to(device=t01.device, dtype=t01.dtype))
        query = t01.squeeze(-1)
        idx_hi = torch.searchsorted(ref01, query, right=False).clamp(max=ref01.numel() - 1)
        idx_lo = (idx_hi - 1).clamp(min=0)
        idx_hi = torch.where(idx_hi < idx_lo, idx_lo, idx_hi)

        t_lo = ref01[idx_lo]
        t_hi = ref01[idx_hi]
        denom = (t_hi - t_lo).clamp_min(1e-8)
        w_hi = torch.where(idx_hi == idx_lo, torch.zeros_like(query), (query - t_lo) / denom)
        w_lo = 1.0 - w_hi

        z_lo = self.latent_states[idx_lo]
        z_hi = self.latent_states[idx_hi]
        return w_lo[:, None] * z_lo + w_hi[:, None] * z_hi

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t01 = self._normalize_time(t.to(dtype=torch.float32))
        z = self._interpolate_states(t01)
        raw = self.readout(z)
        return self._apply_constraint(raw)

    def regularization_loss(self) -> torch.Tensor:
        if self.transition_penalty <= 0.0 or self.latent_states.shape[0] < 2:
            return torch.zeros((), dtype=torch.float32, device=self.latent_states.device)
        phi = 0.995 * torch.tanh(self.phi_unconstrained)
        resid = self.latent_states[1:] - phi * self.latent_states[:-1]
        return self.transition_penalty * torch.mean(resid.pow(2))


def create_trajectory_model(
    kind: str,
    output_dim: int,
    **kwargs,
) -> TimeTrajectoryBase:
    """Factory for reusable temporal trajectory models."""
    k = str(kind).lower().strip()
    if k in {"basis", "rbf", "spline"}:
        return BasisTrajectory(output_dim, **kwargs)
    if k in {"state_space", "state-space", "latent"}:
        return StateSpaceTrajectory(output_dim, **kwargs)
    if k in {"mlp", "neural"}:
        return MLPTrajectory(output_dim, **kwargs)
    raise ValueError(f"Unknown trajectory model kind: {kind}")


__all__ = [
    "BasisTrajectory",
    "MLPTrajectory",
    "StateSpaceTrajectory",
    "TimeTrajectoryBase",
    "create_trajectory_model",
]
