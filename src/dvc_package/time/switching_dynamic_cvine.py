"""Joint switching dynamic C-vine fitting.

This module complements :mod:`joint_dynamic_cvine`.  The smooth joint model is
well suited to continuously varying edge parameters, but it keeps one family per
edge for the whole sequence.  Here each edge is still fit as one temporal object
over all windows, but the temporal object is a discrete state path:

    family_e(t), theta_e(t) = argmin_path sum_t local_cost_t(state_t)
                              + transition_penalties(state_{t-1}, state_t)

The resulting model is a single time-indexed vine with a fixed C-vine order and
jointly selected edge paths.  It is designed for abrupt changes, on/off
episodes, and family switches while retaining a full vine at every time point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.param_copula import copulaccdf
from ..core.vine_factory import create_vine
from .regularized_cvine import (
    EdgeCandidate,
    _as_window_list,
    _build_cvine_edges,
    _dependence_matrix,
    _greedy_order_from_root,
    _normalize_family_name,
    _pseudo_obs_rank,
    _unique_families,
    enumerate_edge_candidates,
    mean_copula_nll,
    parameter_distance,
)


@dataclass
class SwitchingEdgeState:
    """One selected edge state at one time point."""

    time_index: int
    family: str
    theta: Any
    raw_aic: float
    local_cost: float


@dataclass
class SwitchingDynamicEdgeFit:
    """Temporal state path for one vine edge."""

    level: int
    edge: Tuple[int, int]
    states: List[SwitchingEdgeState]
    objective: float
    family_switches: int
    parameter_drift_total: float
    candidate_families: List[str] = field(default_factory=list)

    @property
    def edge_key(self) -> Tuple[int, int, int]:
        i, j = self.edge
        return (int(self.level), min(int(i), int(j)), max(int(i), int(j)))

    @property
    def family_path(self) -> List[str]:
        return [state.family for state in self.states]

    @property
    def theta_path(self) -> List[Any]:
        return [state.theta for state in self.states]


@dataclass
class SwitchingDynamicCVineResult:
    """Fitted switching dynamic C-vine."""

    time_points: List[float]
    order: List[int]
    vines_by_time: List[vine_obj_bin]
    edge_fits: List[SwitchingDynamicEdgeFit]
    mean_nll_by_time: List[float]
    config: Dict[str, Any]

    def mean_nlls(self) -> np.ndarray:
        return np.asarray(self.mean_nll_by_time, dtype=np.float64)

    def edge_fit_map(self) -> Dict[Tuple[int, int, int], SwitchingDynamicEdgeFit]:
        return {edge.edge_key: edge for edge in self.edge_fits}

    def total_family_switches(self) -> int:
        return int(sum(edge.family_switches for edge in self.edge_fits))

    def total_parameter_drift(self) -> float:
        return float(sum(edge.parameter_drift_total for edge in self.edge_fits))

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        windows, _ = _as_window_list(data_by_time, time_points=None)
        if len(windows) != len(self.vines_by_time):
            raise ValueError("Evaluation windows must match the fitted time grid")
        out = np.zeros(len(windows), dtype=np.float64)
        for idx, (vine, x) in enumerate(zip(self.vines_by_time, windows)):
            out[idx] = mean_copula_nll(vine, x)
        return out


def _candidate_by_family(candidates: Sequence[EdgeCandidate]) -> Dict[str, EdgeCandidate]:
    out: Dict[str, EdgeCandidate] = {}
    for cand in candidates:
        out[_normalize_family_name(cand.family)] = cand
    return out


def _solve_edge_state_path(
    candidates_by_time: Sequence[Dict[str, EdgeCandidate]],
    families: Sequence[str],
    *,
    sample_sizes: Sequence[int],
    family_switch_penalty: float,
    parameter_drift_penalty: float,
    activation_penalty: float,
    normalize_cost: bool,
) -> Tuple[List[SwitchingEdgeState], float, int, float]:
    """Viterbi-style temporal family/path selection for one edge."""

    fams = [_normalize_family_name(f) for f in families]
    t_steps = len(candidates_by_time)
    n_states = len(fams)
    if t_steps == 0 or n_states == 0:
        raise ValueError("Need at least one time point and one family")

    local = np.full((t_steps, n_states), np.inf, dtype=np.float64)
    for t_idx, cand_map in enumerate(candidates_by_time):
        n_t = max(int(sample_sizes[t_idx]), 1)
        for s_idx, fam in enumerate(fams):
            cand = cand_map.get(fam)
            if cand is None:
                continue
            cost = float(cand.raw_aic)
            if normalize_cost:
                cost = cost / float(n_t)
            if fam != "ind":
                cost += float(activation_penalty)
            local[t_idx, s_idx] = cost

    dp = np.full_like(local, np.inf)
    back = np.full((t_steps, n_states), -1, dtype=np.int32)
    dp[0] = local[0]

    for t_idx in range(1, t_steps):
        prev_map = candidates_by_time[t_idx - 1]
        cur_map = candidates_by_time[t_idx]
        for s_idx, fam in enumerate(fams):
            if not np.isfinite(local[t_idx, s_idx]):
                continue
            cur = cur_map.get(fam)
            best_val = np.inf
            best_prev = -1
            for p_idx, prev_fam in enumerate(fams):
                if not np.isfinite(dp[t_idx - 1, p_idx]):
                    continue
                prev = prev_map.get(prev_fam)
                trans = 0.0
                if prev_fam != fam:
                    trans += float(family_switch_penalty)
                elif prev is not None and cur is not None:
                    trans += float(parameter_drift_penalty) * parameter_distance(prev.theta, cur.theta)
                value = dp[t_idx - 1, p_idx] + trans
                if value < best_val:
                    best_val = value
                    best_prev = p_idx
            dp[t_idx, s_idx] = local[t_idx, s_idx] + best_val
            back[t_idx, s_idx] = best_prev

    last = int(np.nanargmin(dp[-1]))
    objective = float(dp[-1, last])
    path = np.zeros(t_steps, dtype=np.int32)
    path[-1] = last
    for t_idx in range(t_steps - 1, 0, -1):
        prev = int(back[t_idx, path[t_idx]])
        path[t_idx - 1] = max(prev, 0)

    states: List[SwitchingEdgeState] = []
    switches = 0
    drift_total = 0.0
    prev_fam: Optional[str] = None
    prev_theta: Any = None
    for t_idx, s_idx in enumerate(path):
        fam = fams[int(s_idx)]
        cand = candidates_by_time[t_idx][fam]
        if prev_fam is not None:
            if prev_fam != fam:
                switches += 1
            else:
                drift_total += parameter_distance(prev_theta, cand.theta)
        states.append(
            SwitchingEdgeState(
                time_index=int(t_idx),
                family=fam,
                theta=cand.theta,
                raw_aic=float(cand.raw_aic),
                local_cost=float(local[t_idx, int(s_idx)]),
            )
        )
        prev_fam = fam
        prev_theta = cand.theta

    return states, objective, int(switches), float(drift_total)


class SwitchingDynamicCVine:
    """Joint dynamic C-vine with temporally switched edge families/states.

    Parameters
    ----------
    families:
        Candidate edge families. Independence is always added.
    order:
        Optional fixed C-vine order. If omitted, one pooled Kendall-tau order is
        inferred and reused across time.
    family_switch_penalty:
        Viterbi transition penalty when an edge changes family/state.
    parameter_drift_penalty:
        Penalty on same-family parameter drift between adjacent time points.
    activation_penalty:
        Per-time penalty added to non-independence states. Useful when the goal
        is conservative change detection.
    normalize_cost:
        Use AIC per observation as the local cost, making penalties comparable
        across windows with different sample sizes.
    """

    def __init__(
        self,
        *,
        families: Optional[Sequence[str]] = None,
        order: Optional[Sequence[int]] = None,
        family_switch_penalty: float = 0.05,
        parameter_drift_penalty: float = 0.0,
        activation_penalty: float = 0.0,
        normalize_cost: bool = True,
    ):
        fams = families or ["gaussian", "student", "clayton", "gumbel", "frank", "joe"]
        self.families = _unique_families(list(fams) + ["ind"])
        self.order = [int(v) for v in order] if order is not None else None
        self.family_switch_penalty = float(max(family_switch_penalty, 0.0))
        self.parameter_drift_penalty = float(max(parameter_drift_penalty, 0.0))
        self.activation_penalty = float(max(activation_penalty, 0.0))
        self.normalize_cost = bool(normalize_cost)
        self.result_: Optional[SwitchingDynamicCVineResult] = None

    def _infer_order(self, windows: Sequence[np.ndarray]) -> List[int]:
        if self.order is not None:
            return list(self.order)
        pooled = np.concatenate([_pseudo_obs_rank(x) for x in windows], axis=0)
        dep = _dependence_matrix(pooled)
        root = int(np.argmax(dep.sum(axis=1)))
        return _greedy_order_from_root(dep, root)

    def fit(
        self,
        data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> SwitchingDynamicCVineResult:
        windows, times = _as_window_list(data_by_time, time_points)
        order = self._infer_order(windows)
        ind_vine = _build_cvine_edges(order)
        d = int(windows[0].shape[1])

        u_state_by_time: List[np.ndarray] = []
        for x in windows:
            n = int(x.shape[0])
            u_state = np.zeros((n, d, d), dtype=np.float32)
            u_state[:, 0, :] = _pseudo_obs_rank(x)
            u_state_by_time.append(u_state)

        edge_fits: List[SwitchingDynamicEdgeFit] = []
        level_edge_fits: List[List[SwitchingDynamicEdgeFit]] = []
        sample_sizes = [int(x.shape[0]) for x in windows]

        for level, edges in enumerate(ind_vine):
            fits_level: List[SwitchingDynamicEdgeFit] = []
            for edge in edges:
                i, j = int(edge[0]), int(edge[1])
                u_pairs_by_time = [
                    np.column_stack([u_state[:, level, i], u_state[:, level, j]]).astype(np.float32)
                    for u_state in u_state_by_time
                ]
                candidate_maps = [
                    _candidate_by_family(enumerate_edge_candidates(u_pair, self.families))
                    for u_pair in u_pairs_by_time
                ]
                states, objective, switches, drift = _solve_edge_state_path(
                    candidate_maps,
                    self.families,
                    sample_sizes=sample_sizes,
                    family_switch_penalty=self.family_switch_penalty,
                    parameter_drift_penalty=self.parameter_drift_penalty,
                    activation_penalty=self.activation_penalty,
                    normalize_cost=self.normalize_cost,
                )
                edge_fit = SwitchingDynamicEdgeFit(
                    level=int(level),
                    edge=(i, j),
                    states=states,
                    objective=float(objective),
                    family_switches=int(switches),
                    parameter_drift_total=float(drift),
                    candidate_families=list(self.families),
                )
                fits_level.append(edge_fit)
                edge_fits.append(edge_fit)

                for t_idx, u_pair in enumerate(u_pairs_by_time):
                    cobj = cop_par_obj(states[t_idx].family, states[t_idx].theta)
                    uv = torch.tensor(u_pair, dtype=torch.float32)
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
                [cop_par_obj(edge_fit.states[t_idx].family, edge_fit.states[t_idx].theta) for edge_fit in fits_level]
                for fits_level in level_edge_fits
            ]
            vines_by_time.append(vine)
            mean_nll_by_time.append(float(mean_copula_nll(vine, x)))

        result = SwitchingDynamicCVineResult(
            time_points=[float(v) for v in times],
            order=list(order),
            vines_by_time=vines_by_time,
            edge_fits=edge_fits,
            mean_nll_by_time=mean_nll_by_time,
            config={
                "families": list(self.families),
                "order": list(order),
                "family_switch_penalty": float(self.family_switch_penalty),
                "parameter_drift_penalty": float(self.parameter_drift_penalty),
                "activation_penalty": float(self.activation_penalty),
                "normalize_cost": bool(self.normalize_cost),
            },
        )
        self.result_ = result
        return result

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit() must be called before evaluate()")
        return self.result_.evaluate(data_by_time)


__all__ = [
    "SwitchingDynamicCVine",
    "SwitchingDynamicCVineResult",
    "SwitchingDynamicEdgeFit",
    "SwitchingEdgeState",
]
