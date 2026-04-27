"""
Dynamic nonparametric vine models.

This module provides two nonparametric time-dependent variants built on top of
the validated static nonparametric fitter:

- WindowedNonparametricCVine: fit a separate static nonparametric vine per time
  window. Despite the historical name, it supports fixed `C`/`D`/`R` vine
  structures and pooled vine-family selection.
- JointDynamicNonparametricCVine: fit time-dependent bandwidth trajectories for
  each edge jointly across all windows over a fixed `C`/`D`/`R` vine structure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..core.nonparametric_vine import (
    _build_edge_input_pairs,
    _build_internal_edge_structure,
    _build_sampling_metadata,
    _ensure_grid_metadata,
    _augment_validation,
    _build_oriented_h_copulas,
    _fit_nonparametric_edge,
    _project_monotone_ccdf_grid,
    _validate_nonparametric_edge,
    build_independence_edge_copula,
    build_nonparametric_edge_copula,
    evaluate_nonparametric_edge_h,
    make_nonparametric_uniform_grid,
    prepare_nonparametric_edge_context,
    prepare_nonparametric_margin_pseudo_obs,
)
from ..core.cop_eval import cdf_grid_fun, normalize_pdf_grid
from ..core.objects import copula_obj
from ..core.utils_bandwidth import check_bound_bw
from ..core.utils_interpolation import interp_regular_nd_grid, nearestInterp2d
from ..core.utils_prob import biv_norm, kernel_cdf
from ..core.vine_factory import create_vine, optimize_vine_type
from ..core.vine_tree import flip_check_all
from .regularized_cvine import (
    _as_window_list,
    _build_cvine_edges,
    _dependence_matrix,
    _greedy_order_from_root,
)
from .trajectory_models import create_trajectory_model
from ..core.nonparametric_vine import _pdf_grid_from_bandwidth


def _time_to_unit_interval(time_points: np.ndarray) -> np.ndarray:
    t = np.asarray(time_points, dtype=np.float64).reshape(-1)
    lo = float(np.min(t))
    hi = float(np.max(t))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(t)
    return (t - lo) / (hi - lo)


def _mean_nonparametric_nll(vine, x: np.ndarray) -> float:
    x_t = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    return float((-vine.logpdf(x_t)).mean().detach().cpu())


def _copula_from_pdf_grid(
    ctx: Any,
    pdf_grid_uv: torch.Tensor,
    opt_bw: torch.Tensor,
    *,
    normalization_iters: int,
    validation_extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a nonparametric edge copula from a pre-smoothed copula density grid."""
    device = ctx.data_s.device
    pdf = torch.as_tensor(pdf_grid_uv, dtype=torch.float32, device=device)
    if pdf.dim() == 2:
        pdf = pdf.unsqueeze(-1)
    pdf = torch.nan_to_num(pdf, nan=1.0, posinf=1e6, neginf=1e-12).clamp_min(1e-12)

    adu11, adu22 = ctx.grid_u.diff()
    adu11 = adu11.to(device)
    adu22 = adu22.to(device)
    x1_s, x2_s = ctx.grid_s.axis()
    norm_ref = biv_norm(x1_s.to(device), x2_s.to(device)).unsqueeze(-1).to(device)
    pdf = normalize_pdf_grid(
        adu11,
        adu22,
        pdf * norm_ref,
        norm_ref,
        1,
        iterations=int(normalization_iters),
    ) / norm_ref.clamp_min(1e-12)
    pdf = torch.nan_to_num(pdf, nan=1.0, posinf=1e6, neginf=1e-12).clamp_min(1e-12)

    ccdf = cdf_grid_fun(pdf, ctx.grid_u.ex.to(device), adu11, adu22, 1)
    ccdf = _project_monotone_ccdf_grid(ccdf)

    cop = copula_obj(torch.as_tensor(opt_bw, dtype=torch.float32).detach().cpu())
    cop.family = "kercop"
    cop.pd_grid_uv = pdf[:, :, 0].detach().cpu()
    cop.ccdf_grid = ccdf[:, :, 0].detach().cpu()
    cop.cdf = cop.ccdf_grid
    cop.normalization_iterations = int(normalization_iters)

    ccdf_train_raw = interp_regular_nd_grid(
        ctx.data_s[:, :, 0],
        ctx.grid_s.min.to(device),
        ctx.grid_s.max.to(device),
        cop.ccdf_grid.to(device),
    ).detach()
    ccdf_train_u = torch.tensor(
        kernel_cdf(
            ccdf_train_raw.cpu().numpy(),
            ccdf_train_raw.cpu().numpy(),
            ctx.grid_u.ex.detach().cpu().numpy(),
        )[0],
        dtype=torch.float32,
        device=device,
    )
    order = torch.argsort(ccdf_train_raw)
    cop.ccdf_train_raw = ccdf_train_raw[order].detach().cpu()
    cop.ccdf_train_u = ccdf_train_u[order].detach().cpu()
    cop.grid_s_min = ctx.grid_s.min.detach().cpu()
    cop.grid_s_max = ctx.grid_s.max.detach().cpu()
    cop.validation = _validate_nonparametric_edge(cop)
    _augment_validation(cop, validation_extra)
    return cop


def _temporal_smooth_edge_copulas(
    edge_copulas: Sequence[Any],
    contexts: Sequence[Any],
    normalized_time: np.ndarray,
    *,
    bandwidth: float,
    normalization_iters: int,
    log_density: bool = True,
) -> List[Any]:
    """Couple nonparametric edge shapes by smoothing fitted density grids over time."""
    bw_time = float(bandwidth)
    if bw_time <= 0.0 or len(edge_copulas) <= 1:
        return [copy.deepcopy(cop) for cop in edge_copulas]

    t = np.asarray(normalized_time, dtype=np.float64).reshape(-1)
    eligible = [
        idx for idx, cop in enumerate(edge_copulas)
        if getattr(cop, "family", "kercop") != "ind" and getattr(cop, "pd_grid_uv", None) is not None
    ]
    if not eligible:
        return [copy.deepcopy(cop) for cop in edge_copulas]

    out: List[Any] = []
    for idx, cop in enumerate(edge_copulas):
        if getattr(cop, "family", "kercop") == "ind":
            out.append(copy.deepcopy(cop))
            continue

        dist = (t[eligible] - t[idx]) / max(bw_time, 1e-8)
        weights = np.exp(-0.5 * dist * dist)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-12:
            out.append(copy.deepcopy(cop))
            continue
        weights = weights / weights_sum

        pdf_terms = []
        for weight, src_idx in zip(weights, eligible):
            src_pdf = torch.as_tensor(edge_copulas[src_idx].pd_grid_uv, dtype=torch.float32)
            if log_density:
                pdf_terms.append(float(weight) * torch.log(src_pdf.clamp_min(1e-12)))
            else:
                pdf_terms.append(float(weight) * src_pdf)
        smoothed = torch.stack(pdf_terms, dim=0).sum(dim=0)
        if log_density:
            smoothed = torch.exp(smoothed)

        out.append(
            _copula_from_pdf_grid(
                contexts[idx],
                smoothed,
                torch.as_tensor(cop.opt_bw, dtype=torch.float32),
                normalization_iters=normalization_iters,
                validation_extra={
                    "dynamic_smoothing": "temporal_log_density" if log_density else "temporal_density",
                    "temporal_smoothing_bandwidth": bw_time,
                    "temporal_smoothing_n_neighbors": int(len(eligible)),
                },
            )
        )
    return out


def _infer_cvine_order(windows: Sequence[np.ndarray], order: Optional[Sequence[int]]) -> List[int]:
    if order is not None:
        return [int(v) for v in order]
    x_cat = np.concatenate([np.asarray(x, dtype=np.float32) for x in windows], axis=0)
    dep = _dependence_matrix(x_cat)
    root = int(np.argmax(np.sum(dep, axis=0)))
    return _greedy_order_from_root(dep, root)


def _normalize_vine_type(vine_type: str) -> str:
    vt = str(vine_type).lower().strip()
    if vt in {"c", "cvine", "c-vine"}:
        return "c-vine"
    if vt in {"d", "dvine", "d-vine"}:
        return "d-vine"
    if vt in {"r", "rvine", "r-vine"}:
        return "r-vine"
    if vt in {"auto", "best"}:
        return "auto"
    raise ValueError(f"Unsupported vine_type: {vine_type!r}")


def _clone_vine_structure(template_vine: Any, knots: int) -> Any:
    d = int(template_vine.n_cop)
    kwargs: Dict[str, Any] = {}
    if getattr(template_vine, "vine_family", None) == "d-vine":
        kwargs["variable_order"] = list(getattr(template_vine, "variable_order", list(range(d))))
    if getattr(template_vine, "vine_family", None) == "r-vine":
        kwargs["r_matrix"] = np.asarray(getattr(template_vine, "r_matrix"), dtype=np.int32).copy()
    vine = create_vine(getattr(template_vine, "vine_family", "c-vine"), d, families=["kercop"], knots=knots, **kwargs)
    vine.ind_vine = [[list(edge) for edge in level] for level in getattr(template_vine, "ind_vine", [])]
    if hasattr(template_vine, "variable_order"):
        vine.variable_order = list(getattr(template_vine, "variable_order"))
    if hasattr(template_vine, "r_matrix") and getattr(template_vine, "r_matrix", None) is not None:
        vine.r_matrix = np.asarray(getattr(template_vine, "r_matrix"), dtype=np.int32).copy()
    return vine


def _select_nonparametric_template_vine(
    windows: Sequence[np.ndarray],
    *,
    vine_type: str,
    order: Optional[Sequence[int]],
    knots: int,
    npc_dict: Optional[Dict[str, Any]] = None,
    optimize_structure: bool = False,
    selection_criterion: str = "aic",
    optimization_method: str = "sequential",
    optimization_criterion: str = "kendall_tau",
    vine_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    pooled = np.concatenate([np.asarray(x, dtype=np.float32) for x in windows], axis=0)
    d = int(pooled.shape[1])
    vt = _normalize_vine_type(vine_type)
    vine_kwargs = dict(vine_kwargs or {})
    fit_kwargs = dict(
        gen_dict={"param": False, "binning": False, "fitted": False},
        npc_dict=dict(npc_dict or {}),
        par_dict={},
        bin_dict={},
        vine_kwargs=dict(vine_kwargs),
        optimize_structure=bool(optimize_structure),
        optimization_method=optimization_method,
        optimization_criterion=optimization_criterion,
    )

    if vt == "auto":
        template = optimize_vine_type(
            pooled,
            vine_types=["c-vine", "d-vine", "r-vine"],
            selection_criterion=selection_criterion,
            **fit_kwargs,
        )
        return template

    if vt == "d-vine" and order is not None and "variable_order" not in vine_kwargs:
        vine_kwargs["variable_order"] = [int(v) for v in order]
    fit_kwargs["vine_kwargs"] = dict(vine_kwargs)

    if optimize_structure:
        return optimize_vine_type(
            pooled,
            vine_types=[vt],
            selection_criterion=selection_criterion,
            **fit_kwargs,
        )

    if vt == "r-vine" and "data" not in vine_kwargs:
        vine_kwargs["data"] = pooled

    template = create_vine(vt, d, families=["kercop"], knots=knots, **vine_kwargs)
    if vt == "c-vine":
        resolved_order = _infer_cvine_order(windows, order)
        template.ind_vine = _build_cvine_edges(resolved_order)
        template.variable_order = list(resolved_order)
    template.fit(
        pooled,
        gen_dict={"param": False, "binning": False, "fitted": False},
        npc_dict=dict(npc_dict or {}),
        par_dict={},
        bin_dict={},
    )
    template.selected_vine_type = vt
    template.selection_criterion = selection_criterion
    return template


def _base_u_state(windows: Sequence[np.ndarray], grid_u_ex: np.ndarray) -> List[np.ndarray]:
    states: List[np.ndarray] = []
    for x in windows:
        x = np.asarray(x, dtype=np.float32)
        n, d = x.shape
        u_state = np.zeros((n, d, d), dtype=np.float32)
        for i in range(d):
            u_state[:, 0, i] = kernel_cdf(x[:, i], x[:, i], grid_u_ex)[0].astype(np.float32)
        states.append(u_state)
    return states


def _build_prefit_nonparametric_vine(
    x_train: np.ndarray,
    template_vine: Any,
    copulas_by_level: Sequence[Sequence[Any]],
    knots: int,
) -> Any:
    vine = _clone_vine_structure(template_vine, knots=knots)
    vine.param = False
    vine.fitted = True
    vine.copulas = [list(level) for level in copulas_by_level]

    device = torch.device("cpu")
    ex_u = make_nonparametric_uniform_grid(knots, device=device)
    from ..core.grid_ops import grid_obj
    from ..core.transformation import Transform

    vine.grid_u = _ensure_grid_metadata(grid_obj(ex_u))
    vine.grid_s = _ensure_grid_metadata(grid_obj(Transform(1).forward_u(ex_u)))
    vine.theta = prepare_nonparametric_margin_pseudo_obs(vine, np.asarray(x_train, dtype=np.float32), ex_u.cpu().numpy(), device)
    vine.theta_flip = torch.zeros_like(vine.theta)
    vine._internal_ind_vine = _build_internal_edge_structure(vine, int(x_train.shape[1]))
    sample_order, sample_r_matrix, sample_nodes = _build_sampling_metadata(vine, int(x_train.shape[1]))
    vine._sample_order = sample_order
    vine._sampling_r_matrix = sample_r_matrix
    vine._sampling_nodes = sample_nodes
    vine.flip_flag = []
    vine.ind_edge_rel = []
    vine._np_h_copulas = []
    for level in range(max(int(x_train.shape[1]) - 1, 0)):
        data_u = _build_edge_input_pairs(
            state=vine.theta,
            state_flip=vine.theta_flip,
            edge_refs=vine._internal_ind_vine,
            level=level,
            device=device,
        )
        flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(vine._internal_ind_vine, level, False, 1)
        vine.flip_flag.append(flip_flag1)
        vine.ind_edge_rel.append(ind_edge_rel1)
        level_cops = vine.copulas[level] if level < len(vine.copulas) else []
        h_level_cops = _build_oriented_h_copulas(
            level_cops,
            data_u,
            flip_flag1,
            ind_edge_rel1,
            grid_u=vine.grid_u,
            grid_s=vine.grid_s,
            bandwidth_method="rule_of_thumb",
            knn_k=10,
            batch_size=3,
            fallback_normalization_iters=50,
        )
        vine._np_h_copulas.append(h_level_cops)
        for j, ind_edge in enumerate(ind_edge_rel1):
            if ind_edge >= len(level_cops):
                continue
            uv = data_u[:, :, ind_edge]
            cop = h_level_cops[j] if j < len(h_level_cops) else level_cops[ind_edge]
            if flip_flag1[j]:
                hval = evaluate_nonparametric_edge_h(cop, uv[:, [1, 0]], vine.grid_s)
                vine.theta_flip[:, level + 1, ind_edge] = hval
            else:
                hval = evaluate_nonparametric_edge_h(cop, uv, vine.grid_s)
                vine.theta[:, level + 1, ind_edge] = hval
    vine.training_data = np.asarray(x_train, dtype=np.float32).copy()
    return vine


@dataclass
class WindowedNonparametricCVineResult:
    time_points: List[float]
    normalized_time: List[float]
    order: List[int]
    vines_by_time: List[Any]
    mean_nll_by_time: List[float]
    config: Dict[str, Any]

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        windows, _ = _as_window_list(data_by_time, time_points=None)
        if len(windows) != len(self.vines_by_time):
            raise ValueError("Evaluation windows must match the fitted time grid")
        out = np.zeros(len(windows), dtype=np.float64)
        for idx, (vine, x) in enumerate(zip(self.vines_by_time, windows)):
            out[idx] = _mean_nonparametric_nll(vine, x)
        return out


class WindowedNonparametricCVine:
    def __init__(
        self,
        *,
        order: Optional[Sequence[int]] = None,
        knots: int = 11,
        npc_dict: Optional[Dict[str, Any]] = None,
        vine_type: str = "c-vine",
        vine_kwargs: Optional[Dict[str, Any]] = None,
        optimize_structure: bool = False,
        selection_criterion: str = "aic",
        optimization_method: str = "sequential",
        optimization_criterion: str = "kendall_tau",
        temporal_smoothing_bandwidth: float = 0.0,
        temporal_smoothing_log_density: bool = True,
        temporal_smoothing_normalization_iters: int = 50,
        bandwidth_method: str = "rule_of_thumb",
        knn_k: int = 10,
    ):
        self.order = None if order is None else [int(v) for v in order]
        self.knots = int(knots)
        self.npc_dict = dict(npc_dict or {})
        self.vine_type = _normalize_vine_type(vine_type)
        self.vine_kwargs = dict(vine_kwargs or {})
        self.optimize_structure = bool(optimize_structure)
        self.selection_criterion = str(selection_criterion)
        self.optimization_method = str(optimization_method)
        self.optimization_criterion = str(optimization_criterion)
        self.temporal_smoothing_bandwidth = float(max(temporal_smoothing_bandwidth, 0.0))
        self.temporal_smoothing_log_density = bool(temporal_smoothing_log_density)
        self.temporal_smoothing_normalization_iters = int(max(temporal_smoothing_normalization_iters, 1))
        self.bandwidth_method = str(bandwidth_method)
        self.knn_k = int(knn_k)
        self.result_: Optional[WindowedNonparametricCVineResult] = None

    def fit(
        self,
        data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> WindowedNonparametricCVineResult:
        windows, times = _as_window_list(data_by_time, time_points)
        template_vine = _select_nonparametric_template_vine(
            windows,
            vine_type=self.vine_type,
            order=self.order,
            knots=self.knots,
            npc_dict=self.npc_dict,
            optimize_structure=self.optimize_structure,
            selection_criterion=self.selection_criterion,
            optimization_method=self.optimization_method,
            optimization_criterion=self.optimization_criterion,
            vine_kwargs=self.vine_kwargs,
        )
        order = list(getattr(template_vine, "_sample_order", getattr(template_vine, "variable_order", list(range(int(windows[0].shape[1]))))))
        ind_vine = [[list(edge) for edge in level] for level in getattr(template_vine, "ind_vine", [])]

        vines_by_time = []
        mean_nll_by_time = []
        for x in windows:
            vine = _clone_vine_structure(template_vine, knots=self.knots)
            vine.ind_vine = [list(map(list, level)) for level in ind_vine]
            vine.fit(
                np.asarray(x, dtype=np.float32),
                gen_dict={"param": False, "binning": False, "fitted": False},
                npc_dict=self.npc_dict,
                par_dict={},
                bin_dict={},
            )
            vines_by_time.append(vine)
            mean_nll_by_time.append(_mean_nonparametric_nll(vine, x))

        if self.temporal_smoothing_bandwidth > 0.0:
            vines_by_time = self._smooth_vines_over_time(
                template_vine=template_vine,
                fitted_vines=vines_by_time,
                windows=windows,
                times=np.asarray(times, dtype=np.float32),
            )
            mean_nll_by_time = [_mean_nonparametric_nll(vine, x) for vine, x in zip(vines_by_time, windows)]

        result = WindowedNonparametricCVineResult(
            time_points=[float(v) for v in times],
            normalized_time=[float(v) for v in _time_to_unit_interval(times)],
            order=list(order),
            vines_by_time=vines_by_time,
            mean_nll_by_time=mean_nll_by_time,
            config={
                "order": list(order),
                "selected_vine_type": getattr(template_vine, "selected_vine_type", getattr(template_vine, "vine_family", self.vine_type)),
                "vine_family": getattr(template_vine, "vine_family", self.vine_type),
                "optimize_structure": self.optimize_structure,
                "selection_criterion": self.selection_criterion,
                "optimization_method": self.optimization_method,
                "optimization_criterion": self.optimization_criterion,
                "vine_kwargs": dict(self.vine_kwargs),
                "knots": int(self.knots),
                "npc_dict": dict(self.npc_dict),
                "temporal_smoothing_bandwidth": float(self.temporal_smoothing_bandwidth),
                "temporal_smoothing_log_density": bool(self.temporal_smoothing_log_density),
                "temporal_smoothing_normalization_iters": int(self.temporal_smoothing_normalization_iters),
                "bandwidth_method": str(self.bandwidth_method),
                "knn_k": int(self.knn_k),
            },
        )
        self.result_ = result
        return result

    def _smooth_vines_over_time(
        self,
        *,
        template_vine: Any,
        fitted_vines: Sequence[Any],
        windows: Sequence[np.ndarray],
        times: np.ndarray,
    ) -> List[Any]:
        d = int(windows[0].shape[1])
        device = torch.device("cpu")
        ex_u = make_nonparametric_uniform_grid(self.knots, device=device)
        from ..core.grid_ops import grid_obj
        from ..core.transformation import Transform

        grid_u = _ensure_grid_metadata(grid_obj(ex_u))
        grid_s = _ensure_grid_metadata(grid_obj(Transform(1).forward_u(ex_u)))
        internal_ind_vine = _build_internal_edge_structure(template_vine, d)
        normalized_time = _time_to_unit_interval(np.asarray(times, dtype=np.float32))

        u_state_by_time = _base_u_state(windows, ex_u.detach().cpu().numpy())
        u_state_flip_by_time = [np.zeros_like(state) for state in u_state_by_time]
        copulas_by_time: List[List[List[Any]]] = [[] for _ in windows]

        for level, edges in enumerate(internal_ind_vine):
            level_pairs_by_time = [
                _build_edge_input_pairs(
                    state=torch.tensor(u_state, dtype=torch.float32, device=device),
                    state_flip=torch.tensor(u_state_flip, dtype=torch.float32, device=device),
                    edge_refs=internal_ind_vine,
                    level=level,
                    device=device,
                ).detach().cpu().numpy()
                for u_state, u_state_flip in zip(u_state_by_time, u_state_flip_by_time)
            ]
            level_copulas_by_time: List[List[Any]] = [[] for _ in windows]
            for edge_idx, _edge in enumerate(edges):
                contexts = [
                    prepare_nonparametric_edge_context(
                        torch.tensor(pairs[:, :, edge_idx].astype(np.float32), dtype=torch.float32, device=device),
                        grid_u=grid_u,
                        grid_s=grid_s,
                        bandwidth_method=self.bandwidth_method,
                        knn_k=self.knn_k,
                    )
                    for pairs in level_pairs_by_time
                ]
                edge_copulas = []
                for vine, ctx in zip(fitted_vines, contexts):
                    if level < len(vine.copulas) and edge_idx < len(vine.copulas[level]):
                        edge_copulas.append(vine.copulas[level][edge_idx])
                    else:
                        cop = build_independence_edge_copula(ctx)
                        _augment_validation(cop, {"dynamic_reason": "missing_static_depth"})
                        edge_copulas.append(cop)
                smoothed_edge_copulas = _temporal_smooth_edge_copulas(
                    edge_copulas,
                    contexts,
                    normalized_time,
                    bandwidth=self.temporal_smoothing_bandwidth,
                    normalization_iters=self.temporal_smoothing_normalization_iters,
                    log_density=self.temporal_smoothing_log_density,
                )
                for t_idx, cop in enumerate(smoothed_edge_copulas):
                    level_copulas_by_time[t_idx].append(cop)

            for t_idx, level_cops in enumerate(level_copulas_by_time):
                while len(copulas_by_time[t_idx]) <= level:
                    copulas_by_time[t_idx].append([])
                copulas_by_time[t_idx][level] = list(level_cops)
                flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(internal_ind_vine, level, False, 1)
                uv_level = level_pairs_by_time[t_idx]
                h_level_cops = _build_oriented_h_copulas(
                    level_cops,
                    torch.tensor(uv_level, dtype=torch.float32, device=device),
                    flip_flag1,
                    ind_edge_rel1,
                    grid_u=grid_u,
                    grid_s=grid_s,
                    bandwidth_method=self.bandwidth_method,
                    knn_k=self.knn_k,
                    batch_size=3,
                    fallback_normalization_iters=self.temporal_smoothing_normalization_iters,
                )
                for j, ind_edge in enumerate(ind_edge_rel1):
                    cop = h_level_cops[j] if j < len(h_level_cops) else level_cops[ind_edge]
                    uv = torch.tensor(uv_level[:, :, ind_edge], dtype=torch.float32, device=device)
                    if flip_flag1[j]:
                        hval = evaluate_nonparametric_edge_h(cop, uv[:, [1, 0]], grid_s).detach().cpu().numpy().astype(np.float32)
                        u_state_flip_by_time[t_idx][:, level + 1, ind_edge] = hval
                    else:
                        hval = evaluate_nonparametric_edge_h(cop, uv, grid_s).detach().cpu().numpy().astype(np.float32)
                        u_state_by_time[t_idx][:, level + 1, ind_edge] = hval

        return [
            _build_prefit_nonparametric_vine(
                np.asarray(x, dtype=np.float32),
                template_vine=template_vine,
                copulas_by_level=copulas_by_time[t_idx],
                knots=self.knots,
            )
            for t_idx, x in enumerate(windows)
        ]

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit() must be called before evaluate()")
        return self.result_.evaluate(data_by_time)


@dataclass
class DynamicNonparametricEdgeFit:
    level: int
    edge: Tuple[int, int]
    trajectory_type: str
    bandwidth_trajectory: List[List[float]]
    base_bandwidth_trajectory: List[List[float]]
    target_bandwidth_trajectory: List[List[float]]
    loss: float
    status: str
    selected_model_trajectory: Optional[List[str]] = None

    @property
    def edge_key(self) -> Tuple[int, int, int]:
        i, j = self.edge
        return (int(self.level), min(int(i), int(j)), max(int(i), int(j)))


@dataclass
class JointDynamicNonparametricCVineResult:
    time_points: List[float]
    normalized_time: List[float]
    order: List[int]
    vines_by_time: List[Any]
    edge_fits: List[DynamicNonparametricEdgeFit]
    mean_nll_by_time: List[float]
    config: Dict[str, Any]

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        windows, _ = _as_window_list(data_by_time, time_points=None)
        if len(windows) != len(self.vines_by_time):
            raise ValueError("Evaluation windows must match the fitted time grid")
        out = np.zeros(len(windows), dtype=np.float64)
        for idx, (vine, x) in enumerate(zip(self.vines_by_time, windows)):
            out[idx] = _mean_nonparametric_nll(vine, x)
        return out


class JointDynamicNonparametricCVine:
    def __init__(
        self,
        *,
        order: Optional[Sequence[int]] = None,
        knots: int = 11,
        vine_type: str = "c-vine",
        vine_kwargs: Optional[Dict[str, Any]] = None,
        optimize_structure: bool = False,
        selection_criterion: str = "aic",
        optimization_method: str = "sequential",
        optimization_criterion: str = "kendall_tau",
        trajectory_type: str = "basis",
        trajectory_kwargs: Optional[Dict[str, Any]] = None,
        bandwidth_method: str = "rule_of_thumb",
        knn_k: int = 10,
        n_epochs: int = 30,
        lr: float = 3e-2,
        smoothness_penalty: float = 1e-2,
        batch_size: int = 2,
        normalization_iters: int = 25,
        final_normalization_iters: int = 50,
        warm_start_epochs: int = 40,
        gradient_clip: float = 5.0,
        density_smoothing_bandwidth: float = 0.0,
        density_smoothing_log_density: bool = True,
    ):
        self.order = None if order is None else [int(v) for v in order]
        self.knots = int(knots)
        self.vine_type = _normalize_vine_type(vine_type)
        self.vine_kwargs = dict(vine_kwargs or {})
        self.optimize_structure = bool(optimize_structure)
        self.selection_criterion = str(selection_criterion)
        self.optimization_method = str(optimization_method)
        self.optimization_criterion = str(optimization_criterion)
        self.trajectory_type = str(trajectory_type)
        self.trajectory_kwargs = dict(trajectory_kwargs or {})
        self.bandwidth_method = str(bandwidth_method)
        self.knn_k = int(knn_k)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.smoothness_penalty = float(smoothness_penalty)
        self.batch_size = int(batch_size)
        self.normalization_iters = int(normalization_iters)
        self.final_normalization_iters = int(final_normalization_iters)
        self.warm_start_epochs = int(max(warm_start_epochs, 0))
        self.gradient_clip = float(max(gradient_clip, 0.0))
        self.density_smoothing_bandwidth = float(max(density_smoothing_bandwidth, 0.0))
        self.density_smoothing_log_density = bool(density_smoothing_log_density)
        self.result_: Optional[JointDynamicNonparametricCVineResult] = None

    def _fit_edge_trajectory(
        self,
        level: int,
        edge: Tuple[int, int],
        time_points: np.ndarray,
        u_pairs_by_time: Sequence[np.ndarray],
        grid_u,
        grid_s,
    ) -> Tuple[DynamicNonparametricEdgeFit, List[Any]]:
        contexts = [
            prepare_nonparametric_edge_context(
                torch.tensor(np.asarray(uv, dtype=np.float32), dtype=torch.float32),
                grid_u=grid_u,
                grid_s=grid_s,
                bandwidth_method=self.bandwidth_method,
                knn_k=self.knn_k,
            )
            for uv in u_pairs_by_time
        ]
        device = contexts[0].data_s.device
        t_tensor = torch.tensor(np.asarray(time_points, dtype=np.float32), dtype=torch.float32, device=device)
        target_bws = []
        target_mults = []
        target_model_labels: List[str] = []
        static_npc_dict = {
            "opt_method": "LL1",
            "batch_size": self.batch_size,
            "max_iter_phase1": 6,
            "max_iter_phase2": 8,
            "lr_phase1": 8e-2,
            "lr_phase2": 3e-2,
            "tol_phase1": 1e-4,
            "tol_phase2": 5e-4,
            "normal_iters_phase1": max(10, self.normalization_iters),
            "normal_iters_phase2": max(20, self.final_normalization_iters),
            "validation_fraction": 0.15,
            "final_normalization_iters": self.final_normalization_iters,
        }
        for ctx in contexts:
            static_cop = _fit_nonparametric_edge(ctx, npc_dict=static_npc_dict, cfg={})
            target_model_labels.append(str(getattr(static_cop, "family", "kercop")).lower().strip())
            target_bw = static_cop.opt_bw.to(device) if torch.is_tensor(static_cop.opt_bw) else torch.tensor(static_cop.opt_bw, dtype=torch.float32, device=device)
            if target_bw.dim() == 1:
                target_bw = target_bw.view(2, 1)
            target_bw = torch.nan_to_num(check_bound_bw(target_bw), nan=1.0, posinf=2.0, neginf=1e-2)
            target_bws.append(target_bw)
            target_mults.append(torch.nan_to_num(check_bound_bw(target_bw / ctx.base_bw), nan=1.0, posinf=2.0, neginf=1e-2))

        traj = create_trajectory_model(
            self.trajectory_type,
            output_dim=2,
            constraint="bounded",
            min_value=0.1,
            max_value=2.0,
            **self.trajectory_kwargs,
        ).to(device)
        traj.set_reference_time_grid(t_tensor)

        opt = torch.optim.Adam(traj.parameters(), lr=self.lr)
        best_state = copy.deepcopy(traj.state_dict())
        best_loss = float("inf")
        status = "optimized"
        target_mult_tensor = torch.stack([tm[:, 0] for tm in target_mults], dim=0)

        def _regularized_loss(mult: torch.Tensor) -> torch.Tensor:
            reg = torch.zeros((), dtype=torch.float32, device=device)
            if mult.shape[0] >= 3:
                reg = reg + self.smoothness_penalty * torch.mean((mult[2:] - 2.0 * mult[1:-1] + mult[:-2]) ** 2)
            if hasattr(traj, "regularization_loss"):
                reg = reg + self.smoothness_penalty * traj.regularization_loss()
            return reg

        for _epoch in range(self.warm_start_epochs):
            opt.zero_grad()
            mult = torch.nan_to_num(traj(t_tensor), nan=1.0, posinf=2.0, neginf=0.1)
            loss = torch.mean((mult - target_mult_tensor) ** 2) + _regularized_loss(mult)
            if not torch.isfinite(loss):
                status = "warm_start_fallback"
                break
            loss.backward()
            if self.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(traj.parameters(), self.gradient_clip)
            bad_grad = any(
                (param.grad is not None) and (not torch.isfinite(param.grad).all())
                for param in traj.parameters()
            )
            if bad_grad:
                status = "warm_start_fallback"
                break
            opt.step()

        for _epoch in range(self.n_epochs):
            opt.zero_grad()
            mult = torch.nan_to_num(traj(t_tensor), nan=1.0, posinf=2.0, neginf=0.1)
            loss = torch.zeros((), dtype=torch.float32, device=device)
            total_weight = 0.0
            for idx, ctx in enumerate(contexts):
                bw = check_bound_bw(mult[idx].view(2, 1) * ctx.base_bw)
                pd_grid_uv = _pdf_grid_from_bandwidth(
                    bw,
                    ctx,
                    batch_size=self.batch_size,
                    normalization_iters=self.normalization_iters,
                )[:, :, 0]
                pd_points = nearestInterp2d(
                    ctx.data_s[:, :, 0],
                    ctx.grid_s.ax1.to(device),
                    ctx.grid_s.ax2.to(device),
                    pd_grid_uv,
                ).clamp_min(1e-12)
                loss = loss + (-torch.log(pd_points).sum())
                total_weight += float(pd_points.shape[0])
            loss = loss / max(total_weight, 1.0)
            loss = loss + _regularized_loss(mult)
            if not torch.isfinite(loss):
                status = "nll_fallback"
                break
            loss.backward()
            if self.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(traj.parameters(), self.gradient_clip)
            bad_grad = any(
                (param.grad is not None) and (not torch.isfinite(param.grad).all())
                for param in traj.parameters()
            )
            if bad_grad:
                status = "nll_fallback"
                break
            opt.step()
            loss_val = float(loss.detach().cpu())
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = copy.deepcopy(traj.state_dict())

        traj.load_state_dict(best_state)
        mult = torch.nan_to_num(traj(t_tensor).detach(), nan=1.0, posinf=2.0, neginf=0.1)
        use_target_bandwidths = (not np.isfinite(best_loss)) or (not torch.isfinite(mult).all())

        edge_copulas = []
        bw_traj: List[List[float]] = []
        base_bw_traj: List[List[float]] = []
        target_bw_traj: List[List[float]] = []
        for idx, ctx in enumerate(contexts):
            if target_model_labels[idx] == "ind":
                cop = build_independence_edge_copula(ctx)
                _augment_validation(cop, {"dynamic_selected_model": "independence", "dynamic_reason": "static_edge_validation"})
                edge_copulas.append(cop)
                bw = target_bws[idx]
            elif use_target_bandwidths:
                bw = target_bws[idx]
                status = "target_bandwidth_fallback"
                cop = build_nonparametric_edge_copula(
                    ctx,
                    bw,
                    batch_size=self.batch_size,
                    normalization_iters=self.final_normalization_iters,
                )
                edge_copulas.append(cop)
            else:
                bw = check_bound_bw(mult[idx].view(2, 1) * ctx.base_bw)
                if not torch.isfinite(bw).all():
                    bw = target_bws[idx]
                    status = "target_bandwidth_fallback"
                cop = build_nonparametric_edge_copula(
                    ctx,
                    bw,
                    batch_size=self.batch_size,
                    normalization_iters=self.final_normalization_iters,
                )
                edge_copulas.append(cop)
            bw_traj.append([float(v) for v in bw[:, 0].detach().cpu().numpy().tolist()])
            base_bw_traj.append([float(v) for v in ctx.base_bw[:, 0].detach().cpu().numpy().tolist()])
            target_bw_traj.append([float(v) for v in target_bws[idx][:, 0].detach().cpu().numpy().tolist()])

        if self.density_smoothing_bandwidth > 0.0:
            edge_copulas = _temporal_smooth_edge_copulas(
                edge_copulas,
                contexts,
                _time_to_unit_interval(np.asarray(time_points, dtype=np.float32)),
                bandwidth=self.density_smoothing_bandwidth,
                normalization_iters=self.final_normalization_iters,
                log_density=self.density_smoothing_log_density,
            )

        edge_fit = DynamicNonparametricEdgeFit(
            level=int(level),
            edge=(int(edge[0]), int(edge[1])),
            trajectory_type=self.trajectory_type,
            bandwidth_trajectory=bw_traj,
            base_bandwidth_trajectory=base_bw_traj,
            target_bandwidth_trajectory=target_bw_traj,
            loss=float(best_loss if np.isfinite(best_loss) else float("nan")),
            status=status,
            selected_model_trajectory=list(target_model_labels),
        )
        return edge_fit, edge_copulas

    def fit(
        self,
        data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> JointDynamicNonparametricCVineResult:
        windows, times = _as_window_list(data_by_time, time_points)
        template_vine = _select_nonparametric_template_vine(
            windows,
            vine_type=self.vine_type,
            order=self.order,
            knots=self.knots,
            npc_dict={
                "opt_method": "LL1",
                "batch_size": self.batch_size,
                "max_iter_phase1": 6,
                "max_iter_phase2": 8,
                "lr_phase1": 8e-2,
                "lr_phase2": 3e-2,
                "tol_phase1": 1e-4,
                "tol_phase2": 5e-4,
                "normal_iters_phase1": max(10, self.normalization_iters),
                "normal_iters_phase2": max(20, self.final_normalization_iters),
                "validation_fraction": 0.15,
                "final_normalization_iters": self.final_normalization_iters,
            },
            optimize_structure=self.optimize_structure,
            selection_criterion=self.selection_criterion,
            optimization_method=self.optimization_method,
            optimization_criterion=self.optimization_criterion,
            vine_kwargs=self.vine_kwargs,
        )
        order = list(getattr(template_vine, "_sample_order", getattr(template_vine, "variable_order", list(range(int(windows[0].shape[1]))))))
        public_ind_vine = [[list(edge) for edge in level] for level in getattr(template_vine, "ind_vine", [])]
        d = int(windows[0].shape[1])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ex_u = make_nonparametric_uniform_grid(self.knots, device=device)
        from ..core.grid_ops import grid_obj
        from ..core.transformation import Transform

        internal_ind_vine = _build_internal_edge_structure(template_vine, d)

        grid_u = _ensure_grid_metadata(grid_obj(ex_u))
        grid_s = _ensure_grid_metadata(grid_obj(Transform(1).forward_u(ex_u)))
        u_state_by_time = _base_u_state(windows, ex_u.detach().cpu().numpy())
        u_state_flip_by_time = [np.zeros_like(state) for state in u_state_by_time]

        edge_fits: List[DynamicNonparametricEdgeFit] = []
        copulas_by_time: List[List[List[Any]]] = [[] for _ in windows]

        for level, edges in enumerate(internal_ind_vine):
            level_pairs_by_time = [
                _build_edge_input_pairs(
                    state=torch.tensor(u_state, dtype=torch.float32, device=device),
                    state_flip=torch.tensor(u_state_flip, dtype=torch.float32, device=device),
                    edge_refs=internal_ind_vine,
                    level=level,
                    device=device,
                ).detach().cpu().numpy()
                for u_state, u_state_flip in zip(u_state_by_time, u_state_flip_by_time)
            ]
            level_copulas_by_time: List[List[Any]] = [[] for _ in windows]
            for edge_idx, edge in enumerate(edges):
                edge_label = public_ind_vine[level][edge_idx]
                u_pairs_by_time = [pairs[:, :, edge_idx].astype(np.float32) for pairs in level_pairs_by_time]
                edge_fit, edge_copulas = self._fit_edge_trajectory(
                    level=level,
                    edge=(int(edge_label[0]), int(edge_label[1])),
                    time_points=np.asarray(times, dtype=np.float32),
                    u_pairs_by_time=u_pairs_by_time,
                    grid_u=grid_u,
                    grid_s=grid_s,
                )
                edge_fits.append(edge_fit)
                for t_idx, cop in enumerate(edge_copulas):
                    level_copulas_by_time[t_idx].append(cop)
            for t_idx, level_cops in enumerate(level_copulas_by_time):
                while len(copulas_by_time[t_idx]) <= level:
                    copulas_by_time[t_idx].append([])
                copulas_by_time[t_idx][level] = list(level_cops)
                flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(internal_ind_vine, level, False, 1)
                uv_level = level_pairs_by_time[t_idx]
                h_level_cops = _build_oriented_h_copulas(
                    level_cops,
                    torch.tensor(uv_level, dtype=torch.float32, device=device),
                    flip_flag1,
                    ind_edge_rel1,
                    grid_u=grid_u,
                    grid_s=grid_s,
                    bandwidth_method=self.bandwidth_method,
                    knn_k=self.knn_k,
                    batch_size=self.batch_size,
                    fallback_normalization_iters=self.final_normalization_iters,
                )
                for j, ind_edge in enumerate(ind_edge_rel1):
                    cop = h_level_cops[j] if j < len(h_level_cops) else level_cops[ind_edge]
                    uv = torch.tensor(uv_level[:, :, ind_edge], dtype=torch.float32, device=device)
                    if flip_flag1[j]:
                        hval = evaluate_nonparametric_edge_h(cop, uv[:, [1, 0]], grid_s).detach().cpu().numpy().astype(np.float32)
                        u_state_flip_by_time[t_idx][:, level + 1, ind_edge] = hval
                    else:
                        hval = evaluate_nonparametric_edge_h(cop, uv, grid_s).detach().cpu().numpy().astype(np.float32)
                        u_state_by_time[t_idx][:, level + 1, ind_edge] = hval

        vines_by_time = []
        mean_nll_by_time = []
        for t_idx, x in enumerate(windows):
            vine = _build_prefit_nonparametric_vine(
                np.asarray(x, dtype=np.float32),
                template_vine=template_vine,
                copulas_by_level=copulas_by_time[t_idx],
                knots=self.knots,
            )
            vines_by_time.append(vine)
            mean_nll_by_time.append(_mean_nonparametric_nll(vine, x))

        result = JointDynamicNonparametricCVineResult(
            time_points=[float(v) for v in times],
            normalized_time=[float(v) for v in _time_to_unit_interval(times)],
            order=list(order),
            vines_by_time=vines_by_time,
            edge_fits=edge_fits,
            mean_nll_by_time=mean_nll_by_time,
            config={
                "order": list(order),
                "selected_vine_type": getattr(template_vine, "selected_vine_type", getattr(template_vine, "vine_family", self.vine_type)),
                "vine_family": getattr(template_vine, "vine_family", self.vine_type),
                "optimize_structure": self.optimize_structure,
                "selection_criterion": self.selection_criterion,
                "optimization_method": self.optimization_method,
                "optimization_criterion": self.optimization_criterion,
                "vine_kwargs": dict(self.vine_kwargs),
                "knots": int(self.knots),
                "trajectory_type": self.trajectory_type,
                "trajectory_kwargs": dict(self.trajectory_kwargs),
                "bandwidth_method": self.bandwidth_method,
                "knn_k": int(self.knn_k),
                "n_epochs": int(self.n_epochs),
                "lr": float(self.lr),
                "smoothness_penalty": float(self.smoothness_penalty),
                "batch_size": int(self.batch_size),
                "normalization_iters": int(self.normalization_iters),
                "final_normalization_iters": int(self.final_normalization_iters),
                "warm_start_epochs": int(self.warm_start_epochs),
                "gradient_clip": float(self.gradient_clip),
                "density_smoothing_bandwidth": float(self.density_smoothing_bandwidth),
                "density_smoothing_log_density": bool(self.density_smoothing_log_density),
            },
        )
        self.result_ = result
        return result

    def evaluate(self, data_by_time: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("fit() must be called before evaluate()")
        return self.result_.evaluate(data_by_time)


WindowedDynamicNonparametricVine = WindowedNonparametricCVine
JointDynamicNonparametricVine = JointDynamicNonparametricCVine


__all__ = [
    "WindowedNonparametricCVine",
    "WindowedNonparametricCVineResult",
    "WindowedDynamicNonparametricVine",
    "JointDynamicNonparametricCVine",
    "JointDynamicNonparametricCVineResult",
    "JointDynamicNonparametricVine",
    "DynamicNonparametricEdgeFit",
]
