"""
Low-rank latent state-space dynamic C-vine.

This estimator builds on the joint dynamic C-vine family-selection step and then
fits a shared latent state trajectory that drives all edge-parameter paths. It
is intended as a more structured dynamic alternative to independent windowed
fits and to per-edge smooth-basis trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.param_copula import copulaccdf
from ..core.vine_factory import create_vine
from .joint_dynamic_cvine import (
    JointDynamicCVine,
    _build_time_basis,
    _tau_from_latent,
    _tau_to_theta,
)
from .regularized_cvine import _as_window_list, _build_cvine_edges, _pseudo_obs_rank, mean_copula_nll
from .trajectory_models import StateSpaceTrajectory


def _tau_from_latent_torch(eta: torch.Tensor, family: str, tau_cap: float = 0.95) -> torch.Tensor:
    fam = str(family).lower().strip()
    if fam == "ind":
        return torch.zeros_like(eta)
    if fam in {"gaussian", "student", "frank"}:
        return float(tau_cap) * torch.tanh(eta)
    if fam in {"clayton", "gumbel", "joe"}:
        return float(tau_cap) * torch.sigmoid(eta)
    return float(tau_cap) * torch.tanh(eta)


@dataclass
class LatentStateEdgeFit:
    level: int
    edge: Tuple[int, int]
    family: str
    target_tau_trajectory: List[float]
    tau_trajectory: List[float]
    theta_trajectory: List[Any]
    latent_loading: List[float]
    bias: float
    selection_score: float
    fit_mse: float
    student_df: Optional[float] = None

    @property
    def edge_key(self) -> Tuple[int, int, int]:
        i, j = self.edge
        return (int(self.level), min(int(i), int(j)), max(int(i), int(j)))


@dataclass
class LatentStateDynamicCVineResult:
    time_points: List[float]
    order: List[int]
    vines_by_time: List[vine_obj_bin]
    edge_fits: List[LatentStateEdgeFit]
    mean_nll_by_time: List[float]
    latent_states: List[List[float]]
    ar_coefficient: float
    config: Dict[str, Any]

    def mean_nlls(self) -> np.ndarray:
        return np.asarray(self.mean_nll_by_time, dtype=np.float64)

    def edge_fit_map(self) -> Dict[Tuple[int, int, int], LatentStateEdgeFit]:
        return {edge.edge_key: edge for edge in self.edge_fits}

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        windows, _ = _as_window_list(data_by_time, time_points=None)
        if len(windows) != len(self.vines_by_time):
            raise ValueError("Evaluation windows must match the fitted time grid")
        out = np.zeros(len(windows), dtype=np.float64)
        for idx, (vine, x) in enumerate(zip(self.vines_by_time, windows)):
            out[idx] = mean_copula_nll(vine, x)
        return out


class _LatentTauLevelModel(torch.nn.Module):
    def __init__(
        self,
        families: Sequence[str],
        *,
        latent_dim: int,
        n_steps: int,
        transition_penalty: float,
        target_tau: np.ndarray,
        time_points: np.ndarray,
    ):
        super().__init__()
        self.families = [str(f) for f in families]
        self.target_tau = torch.tensor(target_tau, dtype=torch.float32)
        self.time_points = torch.tensor(time_points, dtype=torch.float32).unsqueeze(-1)
        self.trajectory = StateSpaceTrajectory(
            output_dim=target_tau.shape[1],
            latent_dim=latent_dim,
            n_steps=n_steps,
            transition_penalty=transition_penalty,
            constraint="identity",
        )
        self.trajectory.set_reference_time_grid(torch.tensor(time_points, dtype=torch.float32))

    def predict_tau(self) -> torch.Tensor:
        eta = self.trajectory(self.time_points)
        cols = []
        for edge_idx, family in enumerate(self.families):
            cols.append(_tau_from_latent_torch(eta[:, edge_idx], family))
        return torch.stack(cols, dim=1).to(self.target_tau.device)

    def loss(self) -> torch.Tensor:
        tau_hat = self.predict_tau()
        target = self.target_tau.to(tau_hat.device)
        mse = torch.mean((tau_hat - target) ** 2)
        return mse + self.trajectory.regularization_loss()


class LatentStateDynamicCVine:
    """
    Dynamic C-vine with a shared latent state trajectory.

    This model first selects globally plausible families using the joint
    smooth-basis estimator, then fits a low-rank latent state-space model that
    drives all edge trajectories within each tree level.
    """

    def __init__(
        self,
        *,
        families: Optional[Sequence[str]] = None,
        order: Optional[Sequence[int]] = None,
        selection_n_basis: int = 4,
        selection_smoothness_penalty: float = 5.0,
        latent_dim: int = 2,
        transition_penalty: float = 1e-2,
        n_epochs: int = 250,
        lr: float = 2e-2,
    ):
        self.families = list(families or ["gaussian", "student", "clayton", "gumbel", "frank", "ind"])
        self.order = [int(v) for v in order] if order is not None else None
        self.selection_n_basis = int(max(selection_n_basis, 1))
        self.selection_smoothness_penalty = float(max(selection_smoothness_penalty, 0.0))
        self.latent_dim = int(max(latent_dim, 1))
        self.transition_penalty = float(max(transition_penalty, 0.0))
        self.n_epochs = int(max(n_epochs, 10))
        self.lr = float(max(lr, 1e-4))
        self.result_: Optional[LatentStateDynamicCVineResult] = None

    def fit(
        self,
        data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> LatentStateDynamicCVineResult:
        windows, times = _as_window_list(data_by_time, time_points)
        selection_model = JointDynamicCVine(
            families=self.families,
            n_basis=self.selection_n_basis,
            smoothness_penalty=self.selection_smoothness_penalty,
            order=self.order,
            maxiter=60,
        )
        selection_result = selection_model.fit(windows, time_points=times)
        order = list(selection_result.order)
        ind_vine = _build_cvine_edges(order)
        d = int(windows[0].shape[1])

        u_state_by_time: List[np.ndarray] = []
        for x in windows:
            n = int(x.shape[0])
            u_state = np.zeros((n, d, d), dtype=np.float32)
            u_state[:, 0, :] = _pseudo_obs_rank(x)
            u_state_by_time.append(u_state)

        time_basis = _build_time_basis(times, n_basis=min(self.selection_n_basis, len(times)))
        edge_fits: List[LatentStateEdgeFit] = []
        level_edge_fits: List[List[LatentStateEdgeFit]] = []
        latest_latent_states: Optional[np.ndarray] = None
        latest_phi = 0.0

        for level, edges in enumerate(ind_vine):
            selected_edges = []
            target_tau_cols = []
            families_level = []
            for edge in edges:
                i, j = int(edge[0]), int(edge[1])
                u_pairs_by_time = [
                    np.column_stack([u_state[:, level, i], u_state[:, level, j]]).astype(np.float32)
                    for u_state in u_state_by_time
                ]
                selected = selection_model._select_edge_fit(
                    level=level,
                    edge=(i, j),
                    basis_matrix=time_basis,
                    u_pairs_by_time=u_pairs_by_time,
                )
                selected_edges.append(selected)
                target_tau_cols.append(np.asarray(selected.tau_trajectory, dtype=np.float64))
                families_level.append(selected.family)

            target_tau = np.stack(target_tau_cols, axis=1)
            latent_model = _LatentTauLevelModel(
                families_level,
                latent_dim=self.latent_dim,
                n_steps=len(times),
                transition_penalty=self.transition_penalty,
                target_tau=target_tau,
                time_points=np.asarray(times, dtype=np.float32),
            )
            opt = torch.optim.Adam(latent_model.parameters(), lr=self.lr)
            for _ in range(self.n_epochs):
                opt.zero_grad(set_to_none=True)
                loss = latent_model.loss()
                loss.backward()
                opt.step()

            eta = latent_model.trajectory(latent_model.time_points).detach().cpu().numpy()
            latent_states = latent_model.trajectory.latent_states.detach().cpu().numpy()
            latest_latent_states = latent_states
            latest_phi = float((0.995 * torch.tanh(latent_model.trajectory.phi_unconstrained)).detach().cpu())

            fits_level: List[LatentStateEdgeFit] = []
            for edge_idx, selected in enumerate(selected_edges):
                tau_hat = _tau_from_latent(eta[:, edge_idx], selected.family)
                theta_hat = [
                    _tau_to_theta(selected.family, tau_t, student_df=selected.student_df)
                    for tau_t in tau_hat
                ]
                fit_mse = float(np.mean((tau_hat - np.asarray(selected.tau_trajectory, dtype=np.float64)) ** 2))
                edge_fit = LatentStateEdgeFit(
                    level=level,
                    edge=selected.edge,
                    family=selected.family,
                    target_tau_trajectory=list(np.asarray(selected.tau_trajectory, dtype=np.float64)),
                    tau_trajectory=list(np.asarray(tau_hat, dtype=np.float64)),
                    theta_trajectory=theta_hat,
                    latent_loading=list(latent_model.trajectory.readout.weight.detach().cpu().numpy()[edge_idx]),
                    bias=float(latent_model.trajectory.readout.bias.detach().cpu().numpy()[edge_idx]),
                    selection_score=float(selected.selection_score),
                    fit_mse=fit_mse,
                    student_df=selected.student_df,
                )
                fits_level.append(edge_fit)
                edge_fits.append(edge_fit)

                i, j = edge_fit.edge
                for t_idx, u_state in enumerate(u_state_by_time):
                    u_pair = np.column_stack([u_state[:, level, i], u_state[:, level, j]]).astype(np.float32)
                    uv = torch.tensor(u_pair, dtype=torch.float32)
                    cobj = cop_par_obj(edge_fit.family, edge_fit.theta_trajectory[t_idx])
                    try:
                        hval = copulaccdf(cobj, uv).clamp(1e-6, 1.0 - 1e-6)
                        h_np = hval.detach().cpu().numpy().astype(np.float32)
                        h_np = np.where(np.isfinite(h_np), h_np, u_pair[:, 1])
                    except Exception:
                        h_np = u_pair[:, 1]
                    if level < d - 1:
                        u_state_by_time[t_idx][:, level + 1, j] = h_np

            level_edge_fits.append(fits_level)

        vines_by_time: List[vine_obj_bin] = []
        mean_nll_by_time: List[float] = []
        for t_idx, x in enumerate(windows):
            vine = create_vine("c-vine", d, families=self.families)
            vine.ind_vine = ind_vine
            vine.variable_order = list(order)
            vine.param = True
            vine.fitted = True
            vine.copulas = [
                [cop_par_obj(edge_fit.family, edge_fit.theta_trajectory[t_idx]) for edge_fit in fits_level]
                for fits_level in level_edge_fits
            ]
            vines_by_time.append(vine)
            mean_nll_by_time.append(float(mean_copula_nll(vine, x)))

        result = LatentStateDynamicCVineResult(
            time_points=[float(v) for v in times],
            order=list(order),
            vines_by_time=vines_by_time,
            edge_fits=edge_fits,
            mean_nll_by_time=mean_nll_by_time,
            latent_states=[] if latest_latent_states is None else latest_latent_states.tolist(),
            ar_coefficient=float(latest_phi),
            config={
                "families": list(self.families),
                "selection_n_basis": int(self.selection_n_basis),
                "selection_smoothness_penalty": float(self.selection_smoothness_penalty),
                "latent_dim": int(self.latent_dim),
                "transition_penalty": float(self.transition_penalty),
                "n_epochs": int(self.n_epochs),
                "lr": float(self.lr),
                "order": list(order),
            },
        )
        self.result_ = result
        return result

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit() must be called before evaluate()")
        return self.result_.evaluate(data_by_time)


__all__ = [
    "LatentStateDynamicCVine",
    "LatentStateDynamicCVineResult",
    "LatentStateEdgeFit",
]
