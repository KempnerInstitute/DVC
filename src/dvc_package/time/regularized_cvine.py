"""
Temporally regularized dynamic C-vine fitting.

This module implements a practical dynamic C-vine estimator built on top of
the existing static fitter:
- root sequences are optimized with dynamic programming under a switch penalty
- per-window edge families are selected with stickiness penalties
- same-family parameters can be shrunk toward the previous window

The implementation is intentionally windowed and reproducible. It does not claim
full joint optimization over all vine states, but it provides a concrete path
from the current static codebase toward a defensible dynamic C-vine method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.stats import kendalltau

from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.param_copula import copulaccdf, copulapdf, parametric_fit
from ..core.vine_factory import create_vine


def _normalize_family_name(family: str) -> str:
    fam = str(family).lower().strip()
    if fam in {"independence", "independent"}:
        return "ind"
    if fam in {"gauss"}:
        return "gaussian"
    if fam in {"student-t", "t"}:
        return "student"
    return fam


def _unique_families(families: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for family in families:
        fam = _normalize_family_name(family)
        if fam not in seen:
            seen.add(fam)
            out.append(fam)
    return out


def _as_window_list(
    data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
    time_points: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    if isinstance(data_by_time, np.ndarray) and data_by_time.ndim == 3:
        windows = [np.asarray(data_by_time[t], dtype=np.float32) for t in range(data_by_time.shape[0])]
    else:
        windows = [np.asarray(x, dtype=np.float32) for x in data_by_time]
    if not windows:
        raise ValueError("data_by_time must be non-empty")

    d = windows[0].shape[1]
    for x in windows:
        if x.ndim != 2:
            raise ValueError(f"Each window must be 2D, got shape={x.shape}")
        if x.shape[1] != d:
            raise ValueError("All windows must have the same number of variables")

    if time_points is None:
        times = np.arange(len(windows), dtype=np.float32)
    else:
        times = np.asarray(time_points, dtype=np.float32).reshape(-1)
        if times.size != len(windows):
            raise ValueError("time_points length must match the number of windows")

    return windows, times


def _pseudo_obs_rank(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    n, d = x.shape
    u = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        col = x[:, j]
        ranks = np.argsort(np.argsort(col, kind="mergesort"), kind="mergesort").astype(np.float64) + 1.0
        u[:, j] = ranks / (n + 1.0)
    return np.clip(u, eps, 1.0 - eps).astype(np.float32)


def _dependence_matrix(u: np.ndarray) -> np.ndarray:
    d = u.shape[1]
    dep = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            tau, _ = kendalltau(u[:, i], u[:, j])
            val = abs(float(tau)) if np.isfinite(tau) else 0.0
            dep[i, j] = dep[j, i] = val
    return dep


def _build_cvine_edges(order: Sequence[int]) -> List[List[List[int]]]:
    order = [int(v) for v in order]
    d = len(order)
    ind_vine: List[List[List[int]]] = []
    for level in range(d - 1):
        root = order[level]
        edges_level = []
        for j in range(level + 1, d):
            edges_level.append([root, order[j]])
        ind_vine.append(edges_level)
    return ind_vine


def _greedy_order_from_root(dep: np.ndarray, root: int) -> List[int]:
    d = dep.shape[0]
    remaining = [i for i in range(d) if i != int(root)]
    order = [int(root)]
    while remaining:
        scores = [float(dep[v, remaining].sum() - dep[v, v]) for v in remaining]
        best_idx = int(np.argmax(scores))
        order.append(int(remaining.pop(best_idx)))
    return order


def _theta_to_array(theta: Any) -> np.ndarray:
    if theta is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(theta, np.ndarray):
        arr = theta.astype(np.float64).reshape(-1)
    elif isinstance(theta, (list, tuple)):
        arr = np.asarray(theta, dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray([theta], dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_theta(family: str, theta: Any) -> Any:
    fam = _normalize_family_name(family)
    if fam == "ind":
        return None

    arr = _theta_to_array(theta)
    if arr.size == 0:
        return None

    if fam == "gaussian":
        return float(np.clip(arr[0], -0.98, 0.98))
    if fam == "student":
        rho = float(np.clip(arr[0], -0.98, 0.98))
        nu = float(np.clip(arr[1] if arr.size > 1 else 4.0, 2.1, 50.0))
        return (rho, nu)
    if fam in {"clayton", "claytonrot90"}:
        return float(max(arr[0], 1e-3))
    if fam in {"gumbel", "joe"}:
        return float(max(arr[0], 1.001))
    if fam == "frank":
        val = float(arr[0])
        if abs(val) < 1e-3:
            val = 1e-3 if val >= 0 else -1e-3
        return val
    return float(arr[0])


def parameter_distance(theta_a: Any, theta_b: Any) -> float:
    arr_a = _theta_to_array(theta_a)
    arr_b = _theta_to_array(theta_b)
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0
    m = min(arr_a.size, arr_b.size)
    return float(np.mean(np.abs(arr_a[:m] - arr_b[:m])))


def _blend_theta(family: str, current_theta: Any, previous_theta: Any, smoothing: float) -> Any:
    if previous_theta is None or current_theta is None or smoothing <= 0.0:
        return _sanitize_theta(family, current_theta)
    arr_cur = _theta_to_array(current_theta)
    arr_prev = _theta_to_array(previous_theta)
    if arr_cur.size == 0 or arr_prev.size == 0 or arr_cur.size != arr_prev.size:
        return _sanitize_theta(family, current_theta)
    blended = (1.0 - float(smoothing)) * arr_cur + float(smoothing) * arr_prev
    return _sanitize_theta(family, blended)


@dataclass(frozen=True)
class EdgeCandidate:
    family: str
    theta: Any
    raw_aic: float


@dataclass
class SelectedEdgeFit:
    family: str
    theta: Any
    raw_aic: float
    adjusted_aic: float
    family_switched: bool
    parameter_distance: float


@dataclass
class EdgeDiagnostics:
    level: int
    edge: Tuple[int, int]
    family: str
    theta: Any
    raw_aic: float
    adjusted_aic: float
    family_switched: bool
    parameter_distance: float

    @property
    def edge_key(self) -> Tuple[int, int, int]:
        i, j = self.edge
        return (int(self.level), min(int(i), int(j)), max(int(i), int(j)))


@dataclass
class DynamicCVineWindowFit:
    time_index: int
    time_value: float
    root: int
    order: List[int]
    vine: vine_obj_bin
    mean_nll: float
    edge_diagnostics: List[EdgeDiagnostics] = field(default_factory=list)

    @property
    def family_switch_count(self) -> int:
        return int(sum(1 for edge in self.edge_diagnostics if edge.family_switched))

    @property
    def parameter_drift_total(self) -> float:
        return float(sum(edge.parameter_distance for edge in self.edge_diagnostics))

    def edge_map(self) -> Dict[Tuple[int, int, int], EdgeDiagnostics]:
        return {edge.edge_key: edge for edge in self.edge_diagnostics}


@dataclass
class RegularizedDynamicCVineResult:
    window_fits: List[DynamicCVineWindowFit]
    root_sequence: List[int]
    orders: List[List[int]]
    time_points: List[float]
    root_local_costs: np.ndarray
    root_objective: float
    config: Dict[str, Any]

    def mean_nlls(self) -> np.ndarray:
        return np.asarray([fit.mean_nll for fit in self.window_fits], dtype=np.float64)

    def total_family_switches(self) -> int:
        return int(sum(fit.family_switch_count for fit in self.window_fits[1:]))

    def total_parameter_drift(self) -> float:
        return float(sum(fit.parameter_drift_total for fit in self.window_fits[1:]))

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        windows, _ = _as_window_list(data_by_time, time_points=None)
        if len(windows) != len(self.window_fits):
            raise ValueError("Evaluation windows must match the number of fitted windows")
        out = np.zeros(len(windows), dtype=np.float64)
        for idx, (fit, x) in enumerate(zip(self.window_fits, windows)):
            out[idx] = mean_copula_nll(fit.vine, x)
        return out


def enumerate_edge_candidates(u_pair: np.ndarray, families: Sequence[str]) -> List[EdgeCandidate]:
    u_pair = np.asarray(u_pair, dtype=np.float32)
    if u_pair.ndim != 2 or u_pair.shape[1] != 2:
        raise ValueError(f"Expected u_pair shape (N,2), got {u_pair.shape}")

    fams = _unique_families(families)
    u3 = u_pair[:, :, None]
    aic2, theta_list, _logp_list = parametric_fit(u3, families=fams, n_cop=1)
    out: List[EdgeCandidate] = []
    for idx, family in enumerate(fams):
        out.append(
            EdgeCandidate(
                family=family,
                theta=_sanitize_theta(family, theta_list[0][idx]),
                raw_aic=float(aic2[0, idx]),
            )
        )
    return out


def select_edge_candidate(
    candidates: Sequence[EdgeCandidate],
    prev_copula: Optional[cop_par_obj] = None,
    *,
    family_switch_penalty: float = 0.0,
    parameter_drift_penalty: float = 0.0,
    parameter_smoothing: float = 0.0,
) -> SelectedEdgeFit:
    if not candidates:
        raise ValueError("candidates must be non-empty")

    prev_family = None
    prev_theta = None
    if prev_copula is not None:
        prev_family = _normalize_family_name(prev_copula.family)
        prev_theta = prev_copula.theta

    best: Optional[SelectedEdgeFit] = None
    best_prefers_prev = False
    for cand in candidates:
        switched = False
        theta = cand.theta
        drift = 0.0

        if prev_family is not None:
            if cand.family != prev_family:
                switched = True
            else:
                theta = _blend_theta(cand.family, cand.theta, prev_theta, parameter_smoothing)
                drift = parameter_distance(theta, prev_theta)

        adjusted = float(cand.raw_aic)
        if switched:
            adjusted += float(family_switch_penalty)
        elif prev_family is not None:
            adjusted += float(parameter_drift_penalty) * drift

        prefers_prev = prev_family is not None and cand.family == prev_family
        is_better = best is None or adjusted < best.adjusted_aic - 1e-12
        ties_best = best is not None and abs(adjusted - best.adjusted_aic) <= 1e-12
        tie_break = ties_best and cand.raw_aic < best.raw_aic - 1e-12
        sticky_break = ties_best and abs(cand.raw_aic - best.raw_aic) <= 1e-12 and prefers_prev and not best_prefers_prev

        if is_better or tie_break or sticky_break:
            best = SelectedEdgeFit(
                family=cand.family,
                theta=theta,
                raw_aic=float(cand.raw_aic),
                adjusted_aic=float(adjusted),
                family_switched=switched,
                parameter_distance=float(drift),
            )
            best_prefers_prev = prefers_prev

    if best is None:
        raise RuntimeError("Edge candidate selection failed unexpectedly")
    return best


def solve_root_sequence(local_costs: np.ndarray, switch_penalty: float) -> Tuple[List[int], float]:
    costs = np.asarray(local_costs, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError(f"Expected local_costs shape (T,d), got {costs.shape}")
    t_steps, n_roots = costs.shape
    if t_steps == 0 or n_roots == 0:
        raise ValueError("local_costs must be non-empty")

    dp = np.full_like(costs, np.inf)
    back = np.full((t_steps, n_roots), -1, dtype=np.int32)
    dp[0] = costs[0]

    for t in range(1, t_steps):
        for root in range(n_roots):
            prev = dp[t - 1] + float(switch_penalty) * (np.arange(n_roots) != root)
            best_prev = int(np.argmin(prev))
            dp[t, root] = costs[t, root] + prev[best_prev]
            back[t, root] = best_prev

    path = np.zeros(t_steps, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    objective = float(dp[-1, path[-1]])
    for t in range(t_steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path.astype(int).tolist(), objective


def mean_copula_nll(vine: vine_obj_bin, x: np.ndarray) -> float:
    u = _pseudo_obs_rank(np.asarray(x, dtype=np.float32))
    n, d = u.shape
    u_state = torch.zeros((n, d, d), dtype=torch.float32)
    u_state[:, 0, :] = torch.tensor(u, dtype=torch.float32)

    log_cop = torch.zeros(n, dtype=torch.float32)
    for level in range(d - 1):
        edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
        cops_now = vine.copulas[level] if level < len(vine.copulas) else []
        for edge_idx, edge in enumerate(edges_now):
            if edge_idx >= len(cops_now):
                continue
            cobj = cops_now[edge_idx]
            i, j = int(edge[0]), int(edge[1])
            uv = torch.stack([u_state[:, level, i], u_state[:, level, j]], dim=1)
            pdf_val = copulapdf(cobj, uv).clamp_min(1e-30)
            log_cop = log_cop + torch.log(pdf_val)
            if level < d - 1:
                try:
                    hval = copulaccdf(cobj, uv).clamp(1e-6, 1.0 - 1e-6)
                    hval = torch.where(torch.isfinite(hval), hval, uv[:, 1])
                    u_state[:, level + 1, j] = hval
                except Exception:
                    u_state[:, level + 1, j] = uv[:, 1]
    return float((-log_cop).mean().detach().cpu())


class RegularizedDynamicCVine:
    """
    Windowed dynamic C-vine with temporal regularization.

    Parameters
    ----------
    families:
        Candidate copula families for each edge. Independence is always added.
    root_switch_penalty:
        Penalty applied when the selected C-vine root changes between windows.
    family_switch_penalty:
        Penalty applied when an edge changes family relative to the previous window.
    parameter_drift_penalty:
        Penalty multiplier on same-family parameter changes.
    parameter_smoothing:
        Convex shrinkage weight toward the previous parameter after selection.
    root_score_method:
        Either ``"aic"`` or ``"kendall_tau"`` for the root-path local cost.
    """

    def __init__(
        self,
        *,
        families: Optional[Sequence[str]] = None,
        root_switch_penalty: float = 0.0,
        family_switch_penalty: float = 0.0,
        parameter_drift_penalty: float = 0.0,
        parameter_smoothing: float = 0.0,
        root_score_method: str = "aic",
    ):
        fams = families or ["gaussian", "student", "clayton", "frank", "gumbel", "joe"]
        self.families = _unique_families(list(fams) + ["ind"])
        self.root_switch_penalty = float(root_switch_penalty)
        self.family_switch_penalty = float(family_switch_penalty)
        self.parameter_drift_penalty = float(parameter_drift_penalty)
        self.parameter_smoothing = float(np.clip(parameter_smoothing, 0.0, 1.0))
        self.root_score_method = str(root_score_method).lower().strip()
        if self.root_score_method not in {"aic", "kendall_tau"}:
            raise ValueError("root_score_method must be 'aic' or 'kendall_tau'")
        self.result_: Optional[RegularizedDynamicCVineResult] = None

    def _pairwise_best_aic_matrix(self, u: np.ndarray) -> np.ndarray:
        d = u.shape[1]
        out = np.zeros((d, d), dtype=np.float64)
        for i in range(d):
            for j in range(i + 1, d):
                candidates = enumerate_edge_candidates(u[:, [i, j]], self.families)
                best_aic = min(float(c.raw_aic) for c in candidates)
                out[i, j] = out[j, i] = best_aic
        return out

    def _local_root_costs(self, dep: np.ndarray, pair_costs: np.ndarray) -> np.ndarray:
        d = dep.shape[0]
        costs = np.zeros(d, dtype=np.float64)
        for root in range(d):
            if self.root_score_method == "kendall_tau":
                costs[root] = -float(np.sum(dep[root]) - dep[root, root])
            else:
                costs[root] = float(np.sum(pair_costs[root]) - pair_costs[root, root])
        return costs

    def _fit_window(
        self,
        x: np.ndarray,
        *,
        time_index: int,
        time_value: float,
        root: int,
        dep: np.ndarray,
        prev_fit: Optional[DynamicCVineWindowFit],
    ) -> DynamicCVineWindowFit:
        x = np.asarray(x, dtype=np.float32)
        u0 = _pseudo_obs_rank(x)
        n, d = u0.shape
        order = _greedy_order_from_root(dep, root)
        ind_vine = _build_cvine_edges(order)

        vine = create_vine("c-vine", d, families=self.families)
        vine.ind_vine = ind_vine
        vine.variable_order = list(order)
        vine.param = True
        vine.fitted = True

        prev_edge_map = prev_fit.edge_map() if prev_fit is not None else {}
        u_state = np.zeros((n, d, d), dtype=np.float32)
        u_state[:, 0, :] = u0
        copulas: List[List[cop_par_obj]] = []
        edge_diags: List[EdgeDiagnostics] = []

        for level in range(d - 1):
            cops_level: List[cop_par_obj] = []
            for edge in ind_vine[level]:
                i, j = int(edge[0]), int(edge[1])
                u_pair = np.column_stack([u_state[:, level, i], u_state[:, level, j]]).astype(np.float32)
                prev_edge = prev_edge_map.get((level, min(i, j), max(i, j)))
                prev_copula = None
                if prev_edge is not None:
                    prev_copula = cop_par_obj(prev_edge.family, prev_edge.theta)
                candidates = enumerate_edge_candidates(u_pair, self.families)
                chosen = select_edge_candidate(
                    candidates,
                    prev_copula=prev_copula,
                    family_switch_penalty=self.family_switch_penalty,
                    parameter_drift_penalty=self.parameter_drift_penalty,
                    parameter_smoothing=self.parameter_smoothing,
                )
                cobj = cop_par_obj(chosen.family, chosen.theta)
                cops_level.append(cobj)
                edge_diags.append(
                    EdgeDiagnostics(
                        level=level,
                        edge=(i, j),
                        family=chosen.family,
                        theta=chosen.theta,
                        raw_aic=float(chosen.raw_aic),
                        adjusted_aic=float(chosen.adjusted_aic),
                        family_switched=bool(chosen.family_switched),
                        parameter_distance=float(chosen.parameter_distance),
                    )
                )

                uv = torch.tensor(u_pair, dtype=torch.float32)
                try:
                    hval = copulaccdf(cobj, uv).clamp(1e-6, 1.0 - 1e-6)
                    h_np = hval.detach().cpu().numpy().astype(np.float32)
                    h_np = np.where(np.isfinite(h_np), h_np, u_pair[:, 1])
                except Exception:
                    h_np = u_pair[:, 1]
                if level < d - 1:
                    u_state[:, level + 1, j] = h_np
            copulas.append(cops_level)

        vine.copulas = copulas
        mean_nll = mean_copula_nll(vine, x)
        return DynamicCVineWindowFit(
            time_index=int(time_index),
            time_value=float(time_value),
            root=int(root),
            order=list(order),
            vine=vine,
            mean_nll=float(mean_nll),
            edge_diagnostics=edge_diags,
        )

    def fit(
        self,
        data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> RegularizedDynamicCVineResult:
        windows, times = _as_window_list(data_by_time, time_points)
        d = windows[0].shape[1]

        dep_mats: List[np.ndarray] = []
        pair_cost_mats: List[np.ndarray] = []
        root_local_costs = np.zeros((len(windows), d), dtype=np.float64)

        for idx, x in enumerate(windows):
            u = _pseudo_obs_rank(x)
            dep = _dependence_matrix(u)
            dep_mats.append(dep)
            if self.root_score_method == "aic":
                pair_costs = self._pairwise_best_aic_matrix(u)
            else:
                pair_costs = np.zeros((d, d), dtype=np.float64)
            pair_cost_mats.append(pair_costs)
            root_local_costs[idx] = self._local_root_costs(dep, pair_costs)

        root_sequence, root_objective = solve_root_sequence(root_local_costs, self.root_switch_penalty)

        window_fits: List[DynamicCVineWindowFit] = []
        orders: List[List[int]] = []
        prev_fit: Optional[DynamicCVineWindowFit] = None
        for idx, (x, t_val, root, dep) in enumerate(zip(windows, times, root_sequence, dep_mats)):
            fit = self._fit_window(
                x,
                time_index=idx,
                time_value=float(t_val),
                root=int(root),
                dep=dep,
                prev_fit=prev_fit,
            )
            window_fits.append(fit)
            orders.append(list(fit.order))
            prev_fit = fit

        result = RegularizedDynamicCVineResult(
            window_fits=window_fits,
            root_sequence=[int(r) for r in root_sequence],
            orders=orders,
            time_points=[float(t) for t in times],
            root_local_costs=root_local_costs,
            root_objective=float(root_objective),
            config={
                "families": list(self.families),
                "root_switch_penalty": float(self.root_switch_penalty),
                "family_switch_penalty": float(self.family_switch_penalty),
                "parameter_drift_penalty": float(self.parameter_drift_penalty),
                "parameter_smoothing": float(self.parameter_smoothing),
                "root_score_method": self.root_score_method,
            },
        )
        self.result_ = result
        return result

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit() must be called before evaluate()")
        return self.result_.evaluate(data_by_time)


__all__ = [
    "DynamicCVineWindowFit",
    "EdgeCandidate",
    "EdgeDiagnostics",
    "RegularizedDynamicCVine",
    "RegularizedDynamicCVineResult",
    "SelectedEdgeFit",
    "enumerate_edge_candidates",
    "mean_copula_nll",
    "parameter_distance",
    "select_edge_candidate",
    "solve_root_sequence",
]
