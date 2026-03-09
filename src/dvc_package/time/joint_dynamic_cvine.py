"""
Jointly parameterized dynamic C-vine fitting.

This module implements a genuinely joint temporal parameterization for a C-vine
with fixed variable order. Rather than fitting a separate copula in each time
window, each edge parameter is represented as a smooth function of time and
estimated jointly across the full sequence.

The estimator is deliberately pragmatic:
- the C-vine order is fixed for the whole sequence
- edge families are selected globally over time
- each edge trajectory is fit once using penalized likelihood over all windows
- higher-tree pseudo-observations are propagated from the jointly fit lower
  levels, so the fitted object remains a full time-dependent vine

This is intended to be the main dynamic estimator, with the existing windowed
and regularized methods retained as controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import kendalltau

from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.param_copula import (
    clayton_kendalltau_to_alpha,
    copulaccdf,
    copulapdf,
    frank_kendalltau_to_theta,
    gumbel_kendalltau_to_theta,
    joe_kendalltau_to_theta,
)
from ..core.vine_factory import create_vine
from .regularized_cvine import (
    _as_window_list,
    _build_cvine_edges,
    _dependence_matrix,
    _greedy_order_from_root,
    _normalize_family_name,
    _pseudo_obs_rank,
    _unique_families,
    mean_copula_nll,
)


SIGNED_FAMILIES = {"gaussian", "student", "frank"}
POSITIVE_FAMILIES = {"clayton", "gumbel", "joe"}


def _time_to_unit_interval(time_points: np.ndarray) -> np.ndarray:
    t = np.asarray(time_points, dtype=np.float64).reshape(-1)
    if t.size == 0:
        raise ValueError("time_points must be non-empty")
    lo = float(np.min(t))
    hi = float(np.max(t))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(t)
    return (t - lo) / (hi - lo)


def _build_time_basis(time_points: np.ndarray, n_basis: int, width_scale: float = 1.5) -> np.ndarray:
    """
    Build a smooth time basis with one intercept plus local RBF features.

    Parameters
    ----------
    time_points:
        1-D absolute time values.
    n_basis:
        Total number of basis functions, including the intercept.
    width_scale:
        Width multiplier for the radial basis functions.
    """
    t01 = _time_to_unit_interval(time_points)
    n_basis = max(int(n_basis), 1)
    if n_basis == 1:
        return np.ones((t01.size, 1), dtype=np.float64)

    n_local = n_basis - 1
    centers = np.linspace(0.0, 1.0, n_local, dtype=np.float64)
    spacing = 1.0 / max(n_local - 1, 1)
    width = max(width_scale * spacing, 1e-3)
    basis = np.exp(-0.5 * ((t01[:, None] - centers[None, :]) / width) ** 2)
    basis = np.concatenate([np.ones((t01.size, 1), dtype=np.float64), basis], axis=1)
    return basis


def _family_supports_signed_tau(family: str) -> bool:
    fam = _normalize_family_name(family)
    return fam in SIGNED_FAMILIES


def _empirical_tau_series(u_pairs_by_time: Sequence[np.ndarray]) -> np.ndarray:
    tau_vals = np.zeros(len(u_pairs_by_time), dtype=np.float64)
    for idx, uv in enumerate(u_pairs_by_time):
        tau, _ = kendalltau(uv[:, 0], uv[:, 1])
        tau_vals[idx] = 0.0 if not np.isfinite(tau) else float(tau)
    return tau_vals


def _tau_from_latent(latent: np.ndarray, family: str, tau_cap: float = 0.95) -> np.ndarray:
    fam = _normalize_family_name(family)
    latent = np.asarray(latent, dtype=np.float64)
    if fam == "ind":
        return np.zeros_like(latent)
    if fam in SIGNED_FAMILIES:
        return float(tau_cap) * np.tanh(latent)
    if fam in POSITIVE_FAMILIES:
        latent_clip = np.clip(latent, -30.0, 30.0)
        return float(tau_cap) / (1.0 + np.exp(-latent_clip))
    return float(tau_cap) * np.tanh(latent)


def _latent_from_tau(tau: np.ndarray, family: str, tau_cap: float = 0.95) -> np.ndarray:
    fam = _normalize_family_name(family)
    tau = np.asarray(tau, dtype=np.float64)
    if fam == "ind":
        return np.zeros_like(tau)

    if fam in SIGNED_FAMILIES:
        scaled = np.clip(tau / float(tau_cap), -0.98, 0.98)
        return np.arctanh(scaled)

    clipped = np.clip(tau, 1e-5, float(tau_cap) - 1e-5)
    p = clipped / float(tau_cap)
    return np.log(p) - np.log1p(-p)


def _tau_to_theta(family: str, tau: float, student_df: Optional[float] = None) -> Any:
    fam = _normalize_family_name(family)
    tau = float(np.clip(tau, -0.95, 0.95))

    if fam == "ind":
        return None
    if fam == "gaussian":
        rho = float(np.sin(0.5 * np.pi * tau))
        return float(np.clip(rho, -0.98, 0.98))
    if fam == "student":
        rho = float(np.sin(0.5 * np.pi * tau))
        nu = float(student_df if student_df is not None else 8.0)
        return (float(np.clip(rho, -0.98, 0.98)), float(np.clip(nu, 2.1, 50.0)))
    if fam == "clayton":
        return float(max(clayton_kendalltau_to_alpha(max(tau, 0.0)), 1e-3))
    if fam == "gumbel":
        return float(max(gumbel_kendalltau_to_theta(max(tau, 0.0)), 1.001))
    if fam == "joe":
        return float(max(joe_kendalltau_to_theta(max(tau, 0.0)), 1.001))
    if fam == "frank":
        theta = float(frank_kendalltau_to_theta(tau))
        if abs(theta) < 1e-3:
            theta = 1e-3 if theta >= 0.0 else -1e-3
        return theta
    raise ValueError(f"Unsupported family for joint dynamic fitting: {family}")


def _initial_coefficients(
    basis_matrix: np.ndarray,
    u_pairs_by_time: Sequence[np.ndarray],
    family: str,
) -> np.ndarray:
    emp_tau = _empirical_tau_series(u_pairs_by_time)
    fam = _normalize_family_name(family)
    if fam in POSITIVE_FAMILIES:
        emp_tau = np.clip(emp_tau, 0.0, None)
    latent = _latent_from_tau(emp_tau, fam)
    beta, *_ = np.linalg.lstsq(basis_matrix, latent, rcond=None)
    return np.asarray(beta, dtype=np.float64)


def _trajectory_roughness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return 0.0
    return float(np.mean(np.diff(values, n=2) ** 2))


def _ridge_penalty(beta: np.ndarray) -> float:
    beta = np.asarray(beta, dtype=np.float64)
    if beta.size <= 1:
        return 0.0
    return float(np.sum(beta[1:] ** 2))


def _sum_edge_nll(family: str, theta_path: Sequence[Any], uv_tensors: Sequence[torch.Tensor]) -> float:
    total = 0.0
    fam = _normalize_family_name(family)
    with torch.no_grad():
        for theta, uv in zip(theta_path, uv_tensors):
            if fam == "ind":
                continue
            cobj = cop_par_obj(fam, theta)
            pdf = copulapdf(cobj, uv).clamp_min(1e-30)
            total -= float(torch.log(pdf).sum().detach().cpu())
    return float(total)


@dataclass
class JointDynamicEdgeFit:
    level: int
    edge: Tuple[int, int]
    family: str
    coefficients: List[float]
    tau_trajectory: List[float]
    theta_trajectory: List[Any]
    raw_nll: float
    penalized_objective: float
    selection_score: float
    roughness: float
    ridge_penalty: float
    n_obs: int
    student_df: Optional[float] = None

    @property
    def edge_key(self) -> Tuple[int, int, int]:
        i, j = self.edge
        return (int(self.level), min(int(i), int(j)), max(int(i), int(j)))


@dataclass
class JointDynamicCVineResult:
    time_points: List[float]
    normalized_time: List[float]
    order: List[int]
    vines_by_time: List[vine_obj_bin]
    edge_fits: List[JointDynamicEdgeFit]
    basis_matrix: np.ndarray
    mean_nll_by_time: List[float]
    config: Dict[str, Any]

    def mean_nlls(self) -> np.ndarray:
        return np.asarray(self.mean_nll_by_time, dtype=np.float64)

    def edge_fit_map(self) -> Dict[Tuple[int, int, int], JointDynamicEdgeFit]:
        return {edge.edge_key: edge for edge in self.edge_fits}

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        windows, _ = _as_window_list(data_by_time, time_points=None)
        if len(windows) != len(self.vines_by_time):
            raise ValueError("Evaluation windows must match the fitted time grid")
        out = np.zeros(len(windows), dtype=np.float64)
        for idx, (vine, x) in enumerate(zip(self.vines_by_time, windows)):
            out[idx] = mean_copula_nll(vine, x)
        return out


class JointDynamicCVine:
    """
    Jointly parameterized dynamic C-vine with a fixed order.

    Parameters
    ----------
    families:
        Candidate copula families. Independence is always added.
    n_basis:
        Number of time basis functions, including the intercept.
    smoothness_penalty:
        Penalty on the second finite difference of the implied Kendall-tau path.
    ridge_penalty:
        Small coefficient penalty to stabilize the joint fit.
    order:
        Optional fixed C-vine variable order. If omitted, a pooled Kendall-tau
        order is estimated once and then reused for the whole sequence.
    student_df_grid:
        Degrees-of-freedom candidates for Student copulas.
    maxiter:
        Maximum number of L-BFGS iterations per edge family fit.
    """

    def __init__(
        self,
        *,
        families: Optional[Sequence[str]] = None,
        n_basis: int = 4,
        smoothness_penalty: float = 5.0,
        ridge_penalty: float = 1e-3,
        order: Optional[Sequence[int]] = None,
        student_df_grid: Sequence[float] = (4.0, 8.0, 16.0),
        maxiter: int = 80,
    ):
        fams = list(families or ["gaussian", "student", "clayton", "gumbel", "frank"])
        self.families = _unique_families(fams + ["ind"])
        self.n_basis = int(max(n_basis, 1))
        self.smoothness_penalty = float(max(smoothness_penalty, 0.0))
        self.ridge_penalty = float(max(ridge_penalty, 0.0))
        self.order = [int(v) for v in order] if order is not None else None
        self.student_df_grid = [float(v) for v in student_df_grid]
        self.maxiter = int(max(maxiter, 10))
        self.result_: Optional[JointDynamicCVineResult] = None

    def _infer_order(self, windows: Sequence[np.ndarray]) -> List[int]:
        if self.order is not None:
            return list(self.order)
        pooled = np.concatenate([_pseudo_obs_rank(x) for x in windows], axis=0)
        dep = _dependence_matrix(pooled)
        root = int(np.argmax(dep.sum(axis=1)))
        return _greedy_order_from_root(dep, root)

    def _edge_objective(
        self,
        beta: np.ndarray,
        *,
        basis_matrix: np.ndarray,
        family: str,
        uv_tensors: Sequence[torch.Tensor],
        student_df: Optional[float],
    ) -> float:
        tau = _tau_from_latent(basis_matrix @ beta, family)
        theta_path = [_tau_to_theta(family, tau_t, student_df=student_df) for tau_t in tau]
        raw_nll = _sum_edge_nll(family, theta_path, uv_tensors)
        roughness = _trajectory_roughness(tau)
        ridge = _ridge_penalty(beta)
        return float(raw_nll + self.smoothness_penalty * roughness + self.ridge_penalty * ridge)

    def _fit_edge_family(
        self,
        *,
        family: str,
        basis_matrix: np.ndarray,
        u_pairs_by_time: Sequence[np.ndarray],
    ) -> JointDynamicEdgeFit:
        fam = _normalize_family_name(family)
        total_obs = int(sum(int(uv.shape[0]) for uv in u_pairs_by_time))
        uv_tensors = [torch.tensor(uv, dtype=torch.float32) for uv in u_pairs_by_time]

        if fam == "ind":
            tau_path = np.zeros(len(u_pairs_by_time), dtype=np.float64)
            theta_path = [None for _ in u_pairs_by_time]
            raw_nll = 0.0
            roughness = 0.0
            ridge = 0.0
            selection = 0.0
            coeffs = np.zeros(basis_matrix.shape[1], dtype=np.float64)
            return JointDynamicEdgeFit(
                level=-1,
                edge=(-1, -1),
                family=fam,
                coefficients=coeffs.tolist(),
                tau_trajectory=tau_path.tolist(),
                theta_trajectory=theta_path,
                raw_nll=float(raw_nll),
                penalized_objective=float(raw_nll),
                selection_score=float(selection),
                roughness=float(roughness),
                ridge_penalty=float(ridge),
                n_obs=total_obs,
                student_df=None,
            )

        beta0 = _initial_coefficients(basis_matrix, u_pairs_by_time, fam)
        df_candidates = self.student_df_grid if fam == "student" else [None]

        best_payload: Optional[Dict[str, Any]] = None
        for student_df in df_candidates:
            objective = lambda beta: self._edge_objective(
                beta,
                basis_matrix=basis_matrix,
                family=fam,
                uv_tensors=uv_tensors,
                student_df=student_df,
            )
            opt = minimize(
                objective,
                beta0,
                method="L-BFGS-B",
                options={"maxiter": self.maxiter},
            )
            beta_candidate = np.asarray(getattr(opt, "x", beta0), dtype=np.float64)
            if beta_candidate.shape != beta0.shape or not np.all(np.isfinite(beta_candidate)):
                beta_candidate = np.asarray(beta0, dtype=np.float64)
            beta_hat = beta_candidate
            tau_hat = _tau_from_latent(basis_matrix @ beta_hat, fam)
            theta_hat = [_tau_to_theta(fam, tau_t, student_df=student_df) for tau_t in tau_hat]
            raw_nll = _sum_edge_nll(fam, theta_hat, uv_tensors)
            roughness = _trajectory_roughness(tau_hat)
            ridge = _ridge_penalty(beta_hat)
            penalized = raw_nll + self.smoothness_penalty * roughness + self.ridge_penalty * ridge
            k_eff = int(beta_hat.size + (1 if student_df is not None else 0))
            selection = float(raw_nll + 0.5 * k_eff * np.log(max(total_obs, 2)))

            candidate = {
                "coefficients": beta_hat,
                "tau": tau_hat,
                "theta": theta_hat,
                "raw_nll": float(raw_nll),
                "penalized": float(penalized),
                "selection": selection,
                "roughness": float(roughness),
                "ridge": float(ridge),
                "student_df": None if student_df is None else float(student_df),
            }
            if best_payload is None or candidate["selection"] < best_payload["selection"] - 1e-9:
                best_payload = candidate

        if best_payload is None:
            raise RuntimeError(f"Failed to fit dynamic edge family {family}")

        return JointDynamicEdgeFit(
            level=-1,
            edge=(-1, -1),
            family=fam,
            coefficients=best_payload["coefficients"].tolist(),
            tau_trajectory=best_payload["tau"].tolist(),
            theta_trajectory=best_payload["theta"],
            raw_nll=float(best_payload["raw_nll"]),
            penalized_objective=float(best_payload["penalized"]),
            selection_score=float(best_payload["selection"]),
            roughness=float(best_payload["roughness"]),
            ridge_penalty=float(best_payload["ridge"]),
            n_obs=total_obs,
            student_df=best_payload["student_df"],
        )

    def _select_edge_fit(
        self,
        *,
        level: int,
        edge: Tuple[int, int],
        basis_matrix: np.ndarray,
        u_pairs_by_time: Sequence[np.ndarray],
    ) -> JointDynamicEdgeFit:
        best_fit: Optional[JointDynamicEdgeFit] = None
        for family in self.families:
            candidate = self._fit_edge_family(
                family=family,
                basis_matrix=basis_matrix,
                u_pairs_by_time=u_pairs_by_time,
            )
            candidate.level = int(level)
            candidate.edge = (int(edge[0]), int(edge[1]))
            if best_fit is None or candidate.selection_score < best_fit.selection_score - 1e-9:
                best_fit = candidate
        if best_fit is None:
            raise RuntimeError(f"Failed to select joint edge fit for edge={edge}")
        return best_fit

    def fit(
        self,
        data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> JointDynamicCVineResult:
        windows, times = _as_window_list(data_by_time, time_points)
        order = self._infer_order(windows)
        ind_vine = _build_cvine_edges(order)
        d = int(windows[0].shape[1])
        t01 = _time_to_unit_interval(times)
        basis_matrix = _build_time_basis(times, n_basis=min(self.n_basis, len(times)))

        u_state_by_time: List[np.ndarray] = []
        for x in windows:
            n = int(x.shape[0])
            u_state = np.zeros((n, d, d), dtype=np.float32)
            u_state[:, 0, :] = _pseudo_obs_rank(x)
            u_state_by_time.append(u_state)

        edge_fits: List[JointDynamicEdgeFit] = []
        level_edge_fits: List[List[JointDynamicEdgeFit]] = []

        for level, edges in enumerate(ind_vine):
            fits_level: List[JointDynamicEdgeFit] = []
            for edge in edges:
                i, j = int(edge[0]), int(edge[1])
                u_pairs_by_time = [
                    np.column_stack([u_state[:, level, i], u_state[:, level, j]]).astype(np.float32)
                    for u_state in u_state_by_time
                ]
                edge_fit = self._select_edge_fit(
                    level=level,
                    edge=(i, j),
                    basis_matrix=basis_matrix,
                    u_pairs_by_time=u_pairs_by_time,
                )
                fits_level.append(edge_fit)
                edge_fits.append(edge_fit)

                for t_idx, u_pair in enumerate(u_pairs_by_time):
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

        result = JointDynamicCVineResult(
            time_points=[float(v) for v in times],
            normalized_time=[float(v) for v in t01],
            order=list(order),
            vines_by_time=vines_by_time,
            edge_fits=edge_fits,
            basis_matrix=np.asarray(basis_matrix, dtype=np.float64),
            mean_nll_by_time=mean_nll_by_time,
            config={
                "families": list(self.families),
                "n_basis": int(basis_matrix.shape[1]),
                "smoothness_penalty": float(self.smoothness_penalty),
                "ridge_penalty": float(self.ridge_penalty),
                "order": list(order),
                "student_df_grid": list(self.student_df_grid),
                "maxiter": int(self.maxiter),
            },
        )
        self.result_ = result
        return result

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit() must be called before evaluate()")
        return self.result_.evaluate(data_by_time)


__all__ = [
    "JointDynamicCVine",
    "JointDynamicCVineResult",
    "JointDynamicEdgeFit",
]
