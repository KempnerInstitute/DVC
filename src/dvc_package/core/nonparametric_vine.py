"""
Nonparametric vine copula fitting and evaluation.

This module fills the gap between the existing kernel/local-likelihood
utilities and the public vine API. The current implementation focuses on a
usable static nonparametric vine path:

- fit edge bandwidths with a lightweight held-out likelihood objective
- evaluate pair-copula pdf/cdf grids using the TensorFlow-style grid pipeline
- propagate pseudo-observations through the fitted nonparametric h-functions

The dynamic/nonparametric extensions can build on top of these primitives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .cop_eval import normalize_pdf_grid
from .grid_ops import grid_obj
from .objects import copula_obj, vine_obj_bin
from .transformation import Transform
from .utils_bandwidth import (
    bandwidth_knn,
    bandwidth_rule_of_thumb,
    bandwidth_sqrt_cov,
    check_bound_bw,
)
from .utils_interpolation import interp_regular_nd_grid, interp1d_linear_gpu, nearestInterp2d
from .utils_locallik import loclik_batch_eval
from .utils_prob import biv_norm, kernel_cdf
from .vine_tree import parent_var, flip_check_all

logger = logging.getLogger("DVC.nonparametric_vine")


@dataclass
class _EdgeContext:
    data_u: torch.Tensor
    data_s: torch.Tensor
    data_x: torch.Tensor
    grid_u: grid_obj
    grid_s: grid_obj
    grid_x: grid_obj
    transformer: Transform
    base_bw: torch.Tensor


def _project_monotone_ccdf_grid(ccdf_grid: torch.Tensor) -> torch.Tensor:
    """Project each conditional-CDF row onto a monotone grid in [0, 1]."""
    ccdf = torch.nan_to_num(ccdf_grid, nan=0.5, posinf=1.0, neginf=0.0)
    ccdf = torch.clamp(ccdf, 0.0, 1.0)
    ccdf = torch.cummax(ccdf, dim=1).values
    row_min = ccdf[:, :1, :]
    row_max = ccdf[:, -1:, :]
    span = torch.clamp(row_max - row_min, min=1e-8)
    ccdf = (ccdf - row_min) / span
    return torch.clamp(ccdf, 0.0, 1.0)


def _validate_nonparametric_edge(cop: copula_obj) -> Dict[str, float]:
    """Return lightweight diagnostic summaries for a fitted nonparametric edge."""
    pdf = torch.as_tensor(cop.pd_grid_uv, dtype=torch.float32)
    ccdf = torch.as_tensor(cop.ccdf_grid, dtype=torch.float32)
    row_diffs = ccdf[:, 1:] - ccdf[:, :-1]
    return {
        "pdf_min": float(torch.min(pdf).item()),
        "pdf_max": float(torch.max(pdf).item()),
        "ccdf_min": float(torch.min(ccdf).item()),
        "ccdf_max": float(torch.max(ccdf).item()),
        "ccdf_monotone_violation": float(torch.clamp(-row_diffs, min=0.0).max().item() if row_diffs.numel() else 0.0),
    }


def _infer_cvine_order_from_structure(vine: vine_obj_bin, d: int) -> List[int]:
    root_order: List[int] = []
    for level in range(max(d - 1, 0)):
        edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
        if edges_now:
            root_order.append(int(edges_now[0][0]))
    remaining = [v for v in range(d) if v not in root_order]
    order = root_order + remaining
    if len(order) < d:
        order.extend(v for v in range(d) if v not in order)
    return order[:d]


def _infer_dvine_order_from_structure(vine: vine_obj_bin, d: int) -> List[int]:
    order = getattr(vine, "variable_order", None)
    if order is not None and len(order) == d:
        return [int(v) for v in order]

    edges_0 = vine.ind_vine[0] if getattr(vine, "ind_vine", None) else []
    if not edges_0:
        return list(range(d))

    seen: List[int] = []
    for edge in edges_0:
        for node in edge:
            node_i = int(node)
            if node_i not in seen:
                seen.append(node_i)
    for node_i in range(d):
        if node_i not in seen:
            seen.append(node_i)
    return seen[:d]


def _rvine_build_edges(tree_index: np.ndarray):
    n_ind = tree_index.shape[0]
    e_0 = {tree_index[n_ind - 2, n_ind - 2], tree_index[n_ind - 1, n_ind - 2]}
    e_levels = [[e_0]]
    unions = [[e_0]]

    for _ in range(1, n_ind - 1):
        e_levels.append([])
        unions.append([])

    n = n_ind - 1
    for i in range(n - 2, -1, -1):
        e_1 = {tree_index[i, i], tree_index[n, i]}
        e_levels[0].append(e_1)
        unions[0].append(e_1)

        e_new = e_1
        u_new = e_1
        for k in range(1, n_ind - i - 1):
            u = set()
            for uu in range(n - k, n):
                u.add(tree_index[uu, i])
            u.add(tree_index[n, i])

            flag = False
            for hh in range(len(e_levels[k - 1])):
                if unions[k - 1][hh].issubset(u):
                    flag = True
                    e_new1 = [e_new, e_levels[k - 1][hh]]
                    e_levels[k].append(e_new1)
                    u_union1 = u_new.union(unions[k - 1][hh])
                    unions[k].append(u_union1)
            if not flag:
                raise ValueError("The matrix is not a regular vine")
            e_new = e_new1
            u_new = u_union1
    return e_levels


def _rvine_edges_index(e_levels, r_matrix: np.ndarray, tr: int) -> List[List[int]]:
    edges_ind: List[List[int]] = []
    n = r_matrix.shape[0] - 1
    if tr == 0:
        for i in range(n - 1, -1, -1):
            ind = [int(r_matrix[n, i] - 1), int(r_matrix[i, i] - 1)]
            edges_ind.append(ind)
    else:
        for edge in e_levels[tr]:
            ind0 = 0
            ind1 = 0
            for yy in range(len(e_levels[tr - 1])):
                if edge[0] == e_levels[tr - 1][yy]:
                    ind0 = yy
                if edge[1] == e_levels[tr - 1][yy]:
                    ind1 = yy
            edges_ind.append([int(ind1), int(ind0)])
    return edges_ind


def _build_internal_edge_structure(vine: vine_obj_bin, d: int) -> List[List[List[int]]]:
    family = getattr(vine, "vine_family", "c-vine")

    if family == "c-vine":
        order = _infer_cvine_order_from_structure(vine, d)
        refs: List[List[List[int]]] = []
        refs.append([[int(order[0]), int(order[j + 1])] for j in range(d - 1)])
        for level in range(1, d - 1):
            refs.append([[0, j] for j in range(1, d - level)])
        return refs

    if family == "d-vine":
        order = _infer_dvine_order_from_structure(vine, d)
        refs = []
        refs.append([[int(order[j]), int(order[j + 1])] for j in range(d - 1)])
        for level in range(1, d - 1):
            refs.append([[j, j + 1] for j in range(d - level - 1)])
        return refs

    if family == "r-vine":
        r_matrix = getattr(vine, "r_matrix", None)
        if r_matrix is None:
            raise ValueError("R-vine nonparametric fitting requires a valid r_matrix.")
        # The current public factory stores the R-matrix in an upper-triangular
        # convention, while the legacy tree builder expects the corresponding
        # lower-triangular form.
        r_mat = np.flipud(np.fliplr(np.asarray(r_matrix, dtype=np.int32)))
        e_levels = _rvine_build_edges(r_mat)
        return [_rvine_edges_index(e_levels, r_mat, tr) for tr in range(len(e_levels))]

    raise ValueError(f"Unsupported vine family for nonparametric fitting: {family}")


def _build_sampling_metadata(vine: vine_obj_bin, d: int) -> Tuple[List[int], Optional[np.ndarray], Optional[np.ndarray]]:
    family = getattr(vine, "vine_family", "c-vine")

    if family == "c-vine":
        order = _infer_cvine_order_from_structure(vine, d)
        return order, None, None

    if family == "d-vine":
        order = _infer_dvine_order_from_structure(vine, d)
        return order, None, None

    if family == "r-vine":
        r_matrix = getattr(vine, "r_matrix", None)
        if r_matrix is None:
            raise ValueError("R-vine nonparametric fitting requires a valid r_matrix.")
        lower = np.flipud(np.fliplr(np.asarray(r_matrix, dtype=np.int32)))
        nodes = lower.diagonal()[::-1].astype(np.int32)
        order = [int(v - 1) for v in nodes]
        return order, lower, nodes

    raise ValueError(f"Unsupported vine family for nonparametric sampling metadata: {family}")


def _build_edge_input_pairs(
    state: torch.Tensor,
    state_flip: torch.Tensor,
    edge_refs: List[List[List[int]]],
    level: int,
    device: torch.device,
) -> torch.Tensor:
    edges_now = edge_refs[level] if level < len(edge_refs) else []
    n = state.shape[0]
    data_u = torch.zeros((n, 2, len(edges_now)), dtype=torch.float32, device=device)

    for j, edge in enumerate(edges_now):
        if level == 0:
            data_u[:, 0, j] = state[:, level, int(edge[0])]
            data_u[:, 1, j] = state[:, level, int(edge[1])]
        else:
            parent1, _inx1, _inx2 = parent_var(level, edge_refs, edge)
            if edge_refs[level - 1][edge[0]][0] != parent1:
                data_u[:, 0, j] = state_flip[:, level, int(edge[0])]
            else:
                data_u[:, 0, j] = state[:, level, int(edge[0])]
            data_u[:, 1, j] = state[:, level, int(edge[1])]
    return data_u.clamp(1e-6, 1.0 - 1e-6)


def _device_for_tensor_like(x: np.ndarray) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_nonparametric_uniform_grid(knots: int,
                                    device: torch.device,
                                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create the TensorFlow-style uniform grid used by the legacy code."""
    normal = torch.distributions.Normal(0.0, 1.0)
    s_axis = torch.linspace(-3.2, 3.2, int(knots), device=device, dtype=dtype)
    u_axis = normal.cdf(s_axis).clamp(1e-6, 1.0 - 1e-6)
    uu, vv = torch.meshgrid(u_axis, u_axis, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)


def _ensure_grid_metadata(g: grid_obj) -> grid_obj:
    g.axis()
    g.diff()
    g.min_grid()
    g.max_grid()
    return g


def _init_bandwidth(data_x: torch.Tensor,
                    method: str = "rule_of_thumb",
                    knn_k: int = 10) -> torch.Tensor:
    if method == "rule_of_thumb":
        bw = bandwidth_rule_of_thumb(data_x, deg=2, n_cop=data_x.shape[2])
    elif method == "knn":
        bw = bandwidth_knn(data_x, k=int(knn_k))
    elif method == "sqrt_cov":
        bw = bandwidth_sqrt_cov(data_x)
    else:
        raise ValueError(f"Unknown nonparametric bandwidth method: {method}")
    return check_bound_bw(bw)


def prepare_nonparametric_margin_pseudo_obs(vine: vine_obj_bin,
                                            x: np.ndarray,
                                            grid_u_ex: np.ndarray,
                                            device: torch.device) -> torch.Tensor:
    n, d = x.shape
    theta = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    vine.Mar_G = []
    for i in range(d):
        col = np.asarray(x[:, i], dtype=np.float64).reshape(-1)
        if i < len(vine.margin):
            vine.margin[i].ker = col.copy()
        interp_cdf, mar_s1, mar_p1 = kernel_cdf(col, col, grid_u_ex)
        vine.Mar_G.append([mar_s1, mar_p1])
        theta[:, 0, i] = torch.tensor(interp_cdf, dtype=torch.float32, device=device)
    return theta


def prepare_nonparametric_edge_context(u_pair: torch.Tensor,
                                       grid_u: grid_obj,
                                       grid_s: grid_obj,
                                       bandwidth_method: str,
                                       knn_k: int) -> _EdgeContext:
    transformer = Transform(1)
    data_u = u_pair.unsqueeze(-1)
    data_s = transformer.forward_u(data_u)
    data_x = transformer.forward_s(data_s)
    grid_x = _ensure_grid_metadata(grid_obj(transformer.forward_s(grid_s.ex)))
    base_bw = _init_bandwidth(data_x, method=bandwidth_method, knn_k=knn_k)
    return _EdgeContext(
        data_u=data_u,
        data_s=data_s,
        data_x=data_x,
        grid_u=grid_u,
        grid_s=grid_s,
        grid_x=grid_x,
        transformer=transformer,
        base_bw=base_bw,
    )


def _pdf_grid_from_bandwidth(B: torch.Tensor,
                             ctx: _EdgeContext,
                             batch_size: int,
                             normalization_iters: int) -> torch.Tensor:
    n_cop = ctx.data_x.shape[2]
    adu11, adu22 = ctx.grid_u.diff()
    x1_s, x2_s = ctx.grid_s.axis()
    norm_ref = biv_norm(x1_s, x2_s).unsqueeze(-1).repeat(1, 1, n_cop).to(B.device)

    grid_x = ctx.grid_x.ex
    if grid_x.dim() == 2:
        grid_x = grid_x.unsqueeze(-1)
    ker_grid_fin = loclik_batch_eval(
        B,
        ctx.data_s,
        grid_x,
        n_cop=n_cop,
        batch_size=batch_size,
    )
    knots = ctx.grid_u.ax1.shape[0]
    ker_grid_all = ker_grid_fin.reshape(knots, knots, n_cop).permute(1, 0, 2)
    ker_grid_all = ker_grid_all + 1e-15 * norm_ref
    pdf1 = normalize_pdf_grid(
        adu11,
        adu22,
        ker_grid_all,
        norm_ref,
        n_cop,
        iterations=int(normalization_iters),
    )
    return pdf1 / norm_ref


def _fit_bandwidth_multiplier(ctx: _EdgeContext,
                              batch_size: int,
                              opt_method: str,
                              max_iter_phase1: int,
                              lr_phase1: float,
                              tol_phase1: float,
                              max_iter_phase2: int,
                              lr_phase2: float,
                              tol_phase2: float,
                              validation_fraction: float = 0.15,
                              normal_iters_phase1: int = 25,
                              normal_iters_phase2: int = 50) -> torch.Tensor:
    n = ctx.data_s.shape[0]
    if n < 16:
        return ctx.base_bw

    n_val = max(int(round(n * validation_fraction)), 4)
    perm = torch.randperm(n, device=ctx.data_s.device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.numel() < 8:
        return ctx.base_bw

    def subset_context(idx: torch.Tensor) -> _EdgeContext:
        return _EdgeContext(
            data_u=ctx.data_u[idx],
            data_s=ctx.data_s[idx],
            data_x=ctx.data_x[idx],
            grid_u=ctx.grid_u,
            grid_s=ctx.grid_s,
            grid_x=ctx.grid_x,
            transformer=ctx.transformer,
            base_bw=ctx.base_bw,
        )

    train_ctx = subset_context(train_idx)
    val_s = ctx.data_s[val_idx][:, :, 0]

    def objective(multiplier: torch.Tensor, normal_iters: int) -> torch.Tensor:
        bw = check_bound_bw(multiplier * ctx.base_bw)
        pd_grid_uv = _pdf_grid_from_bandwidth(
            bw,
            train_ctx,
            batch_size=max(1, batch_size),
            normalization_iters=normal_iters,
        )[:, :, 0]
        pd_points = nearestInterp2d(val_s, ctx.grid_s.ax1, ctx.grid_s.ax2, pd_grid_uv)
        pd_points = pd_points.clamp_min(1e-12)
        return -torch.log(pd_points).mean()

    mult_shape = (1, 1) if opt_method.upper() == "LL1" else (2, 1)
    mult = torch.full(mult_shape, 1.0, dtype=ctx.base_bw.dtype, device=ctx.base_bw.device, requires_grad=True)

    def run_phase(num_iter: int, lr: float, tol: float, normal_iters: int) -> None:
        opt = torch.optim.Adam([mult], lr=float(lr))
        prev = None
        for _ in range(int(num_iter)):
            opt.zero_grad()
            loss = objective(torch.nn.functional.softplus(mult), normal_iters=normal_iters)
            loss.backward()
            opt.step()
            if prev is not None and abs(float(loss.detach()) - prev) < float(tol):
                break
            prev = float(loss.detach())

    run_phase(max_iter_phase1, lr_phase1, tol_phase1, normal_iters=normal_iters_phase1)
    run_phase(max_iter_phase2, lr_phase2, tol_phase2, normal_iters=normal_iters_phase2)
    return check_bound_bw(torch.nn.functional.softplus(mult.detach()) * ctx.base_bw)


def build_nonparametric_edge_copula(ctx: _EdgeContext,
                                    bw: torch.Tensor,
                                    *,
                                    batch_size: int = 3,
                                    normalization_iters: int = 50) -> copula_obj:
    from .vine_eval import evaluate_fit

    data_dict = {
        "data_s": ctx.data_s,
        "data_x": ctx.data_x,
    }
    grid_dict = {
        "grid_u": ctx.grid_u,
        "grid_s": ctx.grid_s,
        "grid_x": ctx.grid_x.ex.unsqueeze(-1) if ctx.grid_x.ex.dim() == 2 else ctx.grid_x.ex,
    }
    par_dict = {
        "bw": bw,
        "n_cop": 1,
        "batch": max(1, int(batch_size)),
        "normalization_iters": int(normalization_iters),
    }
    pd_grid_uv, ccdf_grid, _theta_unused, _grad_u, _grad_v = evaluate_fit(data_dict, grid_dict, par_dict)

    cop = copula_obj(bw.detach().cpu())
    cop.pd_grid_uv = torch.clamp(pd_grid_uv[:, :, 0], min=1e-12).detach().cpu()
    # `ccdf_grid` is the conditional CDF / h-function grid used for
    # pseudo-observation propagation and inverse h-function sampling.
    cop.ccdf_grid = _project_monotone_ccdf_grid(ccdf_grid[:, :, 0].unsqueeze(-1))[:, :, 0].detach().cpu()
    cop.cdf = cop.ccdf_grid
    cop.normalization_iterations = int(normalization_iters)

    ccdf_train_raw = interp_regular_nd_grid(
        ctx.data_s[:, :, 0],
        ctx.grid_s.min,
        ctx.grid_s.max,
        cop.ccdf_grid.to(ctx.data_s.device),
    ).detach()
    ccdf_train_u = torch.tensor(
        kernel_cdf(
            ccdf_train_raw.cpu().numpy(),
            ccdf_train_raw.cpu().numpy(),
            ctx.grid_u.ex.detach().cpu().numpy(),
        )[0],
        dtype=torch.float32,
        device=ctx.data_s.device,
    )
    order = torch.argsort(ccdf_train_raw)
    cop.ccdf_train_raw = ccdf_train_raw[order].detach().cpu()
    cop.ccdf_train_u = ccdf_train_u[order].detach().cpu()
    cop.grid_s_min = ctx.grid_s.min.detach().cpu()
    cop.grid_s_max = ctx.grid_s.max.detach().cpu()
    cop.validation = _validate_nonparametric_edge(cop)
    return cop


def _fit_nonparametric_edge(ctx: _EdgeContext,
                            npc_dict: dict,
                            cfg: dict) -> copula_obj:
    optimizer_cfg = cfg.get("optimizer", {})
    batch_size = int(npc_dict.get("batch_size", optimizer_cfg.get("batch_size", 3)))
    max_iter_phase1 = int(npc_dict.get("max_iter_phase1", optimizer_cfg.get("max_iter_phase1", 12)))
    lr_phase1 = float(npc_dict.get("lr_phase1", optimizer_cfg.get("lr_phase1", 0.08)))
    tol_phase1 = float(npc_dict.get("tol_phase1", optimizer_cfg.get("tol_phase1", 1e-4)))
    max_iter_phase2 = int(npc_dict.get("max_iter_phase2", optimizer_cfg.get("max_iter_phase2", 16)))
    lr_phase2 = float(npc_dict.get("lr_phase2", optimizer_cfg.get("lr_phase2", 0.03)))
    tol_phase2 = float(npc_dict.get("tol_phase2", optimizer_cfg.get("tol_phase2", 5e-4)))
    opt_method = str(npc_dict.get("opt_method", "LL1")).upper()
    normal_iters_phase1 = int(npc_dict.get("normal_iters_phase1", 25))
    normal_iters_phase2 = int(npc_dict.get("normal_iters_phase2", 50))
    validation_fraction = float(npc_dict.get("validation_fraction", 0.15))

    bw = _fit_bandwidth_multiplier(
        ctx,
        batch_size=batch_size,
        opt_method=opt_method,
        max_iter_phase1=max_iter_phase1,
        lr_phase1=lr_phase1,
        tol_phase1=tol_phase1,
        max_iter_phase2=max_iter_phase2,
        lr_phase2=lr_phase2,
        tol_phase2=tol_phase2,
        validation_fraction=validation_fraction,
        normal_iters_phase1=normal_iters_phase1,
        normal_iters_phase2=normal_iters_phase2,
    )

    return build_nonparametric_edge_copula(
        ctx,
        bw,
        batch_size=batch_size,
        normalization_iters=int(npc_dict.get("final_normalization_iters", 50)),
    )


def evaluate_nonparametric_edge_pdf(cop: copula_obj,
                                    uv: torch.Tensor,
                                    grid_s: grid_obj) -> torch.Tensor:
    points_s = Transform(1).forward_u(uv)
    pd_grid_uv = cop.pd_grid_uv.to(points_s.device)
    return nearestInterp2d(points_s, grid_s.ax1.to(points_s.device), grid_s.ax2.to(points_s.device), pd_grid_uv)


def evaluate_nonparametric_edge_h(cop: copula_obj,
                                  uv: torch.Tensor,
                                  grid_s: grid_obj) -> torch.Tensor:
    points_s = Transform(1).forward_u(uv)
    ccdf_grid = getattr(cop, "ccdf_grid", None)
    if ccdf_grid is None:
        ccdf_grid = cop.cdf
    raw = interp_regular_nd_grid(
        points_s,
        cop.grid_s_min.to(points_s.device),
        cop.grid_s_max.to(points_s.device),
        ccdf_grid.to(points_s.device),
    )
    raw_axis = cop.ccdf_train_raw.to(points_s.device)
    u_axis = cop.ccdf_train_u.to(points_s.device)
    return interp1d_linear_gpu(raw, raw_axis, u_axis).clamp(1e-6, 1.0 - 1e-6)


def fit_nonparametric_vine(
    vine: vine_obj_bin,
    x: np.ndarray,
    gen_dict: dict,
    npc_dict: dict,
    par_dict: dict,
    bin_dict: dict,
    cfg: Optional[dict] = None,
):
    cfg = dict(cfg or {})
    device = _device_for_tensor_like(x)
    x_np = np.asarray(x, dtype=np.float32)
    n, d = x_np.shape

    vine.param = False
    vine.binning = bool(gen_dict.get("binning", False))
    if vine.binning:
        raise NotImplementedError("Binned nonparametric vine fitting is not implemented in the PyTorch path yet.")
    vine.fitted = True
    vine.n_cop = d
    vine.training_data = x_np.copy()
    vine._internal_ind_vine = _build_internal_edge_structure(vine, d)
    sample_order, sample_r_matrix, sample_nodes = _build_sampling_metadata(vine, d)
    vine._sample_order = sample_order
    vine._sampling_r_matrix = sample_r_matrix
    vine._sampling_nodes = sample_nodes

    ex_u = make_nonparametric_uniform_grid(vine.knots, device=device)
    vine.grid_u = _ensure_grid_metadata(grid_obj(ex_u))
    vine.grid_s = _ensure_grid_metadata(grid_obj(Transform(1).forward_u(ex_u)))
    vine.theta = prepare_nonparametric_margin_pseudo_obs(vine, x_np, ex_u.detach().cpu().numpy(), device)
    vine.theta_flip = torch.zeros_like(vine.theta)
    vine.flip_flag = []
    vine.ind_edge_rel = []

    bandwidth_cfg = cfg.get("bandwidth", {})
    bandwidth_method = str(bandwidth_cfg.get("method", "rule_of_thumb"))
    knn_k = int(bandwidth_cfg.get("knn_k", 10))

    vine.copulas = []
    for level in range(d - 1):
        edges_now = vine._internal_ind_vine[level] if level < len(vine._internal_ind_vine) else []
        data_u = _build_edge_input_pairs(
            state=vine.theta,
            state_flip=vine.theta_flip,
            edge_refs=vine._internal_ind_vine,
            level=level,
            device=device,
        )
        level_cops: List[copula_obj] = []
        for edge_idx, _edge in enumerate(edges_now):
            u_pair = data_u[:, :, edge_idx]
            ctx = prepare_nonparametric_edge_context(
                u_pair=u_pair,
                grid_u=vine.grid_u,
                grid_s=vine.grid_s,
                bandwidth_method=bandwidth_method,
                knn_k=knn_k,
            )
            cop = _fit_nonparametric_edge(ctx, npc_dict=npc_dict, cfg=cfg)
            level_cops.append(cop)
        vine.copulas.append(level_cops)

        flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(vine._internal_ind_vine, level, False, 1)
        vine.flip_flag.append(flip_flag1)
        vine.ind_edge_rel.append(ind_edge_rel1)

        for j, ind_edge in enumerate(ind_edge_rel1):
            uv = data_u[:, :, ind_edge]
            if flip_flag1[j]:
                hval = evaluate_nonparametric_edge_h(level_cops[ind_edge], uv, vine.grid_s)
                vine.theta_flip[:, level + 1, ind_edge] = hval
            else:
                hval = evaluate_nonparametric_edge_h(level_cops[ind_edge], uv[:, [1, 0]], vine.grid_s)
                vine.theta[:, level + 1, ind_edge] = hval

    edge_diagnostics = [
        cop.validation
        for level in vine.copulas
        for cop in level
        if getattr(cop, "validation", None)
    ]
    vine.nonparametric_summary = {
        "n_levels": len(vine.copulas),
        "n_edges": sum(len(level) for level in vine.copulas),
        "max_ccdf_monotone_violation": float(
            max((diag["ccdf_monotone_violation"] for diag in edge_diagnostics), default=0.0)
        ),
        "max_pdf_value": float(max((diag["pdf_max"] for diag in edge_diagnostics), default=0.0)),
    }

    return vine


def evaluate_nonparametric_vine(vine: vine_obj_bin, points: torch.Tensor):
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    points = points.float()
    device = points.device
    n, d = points.shape

    edge_refs = getattr(vine, "_internal_ind_vine", None)
    if edge_refs is None:
        edge_refs = _build_internal_edge_structure(vine, d)
        vine._internal_ind_vine = edge_refs

    u_state = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    u_state_flip = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    grid_u_ex = vine.grid_u.ex.detach().cpu().numpy()
    for i in range(d):
        if vine.margin and i < len(vine.margin) and getattr(vine.margin[i], "ker", None) is not None:
            cdf_query, _mar_s, _mar_p = kernel_cdf(
                np.asarray(vine.margin[i].ker),
                points[:, i].detach().cpu().numpy(),
                grid_u_ex,
            )
            u_state[:, 0, i] = torch.tensor(cdf_query, dtype=torch.float32, device=device)
        else:
            u_state[:, 0, i] = points[:, i].clamp(1e-6, 1.0 - 1e-6)

    log_cop = torch.zeros(n, dtype=torch.float32, device=device)
    for level in range(d - 1):
        if not getattr(vine, "flip_flag", None) or level >= len(vine.flip_flag):
            flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(edge_refs, level, False, 1)
        else:
            flip_flag1 = vine.flip_flag[level]
            ind_edge_rel1 = vine.ind_edge_rel[level]

        edges_now = edge_refs[level] if level < len(edge_refs) else []
        cops_now = vine.copulas[level] if level < len(vine.copulas) else []
        point_u = _build_edge_input_pairs(
            state=u_state,
            state_flip=u_state_flip,
            edge_refs=edge_refs,
            level=level,
            device=device,
        )
        for j, ind_edge in enumerate(ind_edge_rel1):
            if ind_edge >= len(cops_now):
                continue
            cop = cops_now[ind_edge]
            uv = point_u[:, :, ind_edge]
            if flip_flag1[j]:
                uv = uv[:, [1, 0]]
            pdf_val = evaluate_nonparametric_edge_pdf(cop, uv, vine.grid_s).clamp_min(1e-12)
            log_cop = log_cop + torch.log(pdf_val)
            if level < d - 1:
                hval = evaluate_nonparametric_edge_h(cop, uv, vine.grid_s)
                if flip_flag1[j]:
                    u_state_flip[:, level + 1, ind_edge] = hval
                else:
                    u_state[:, level + 1, ind_edge] = hval

    p_cop = torch.exp(log_cop)
    log_marg = torch.zeros_like(log_cop)
    return p_cop, p_cop, log_marg


__all__ = [
    "fit_nonparametric_vine",
    "evaluate_nonparametric_vine",
    "make_nonparametric_uniform_grid",
    "prepare_nonparametric_margin_pseudo_obs",
    "prepare_nonparametric_edge_context",
    "build_nonparametric_edge_copula",
    "evaluate_nonparametric_edge_pdf",
    "evaluate_nonparametric_edge_h",
]
