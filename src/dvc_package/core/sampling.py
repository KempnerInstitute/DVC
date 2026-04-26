##################################################
# src/DVC/sampling.py
##################################################

import torch
import numpy as np
from typing import Tuple, Optional

from .utils_prob import kernel_cdf, copulainvccdf, copulaccdf
from .vine_tree import parent_var
from .transformation import Transform
from .preparation import prep_copula
from .utils_interpolation import interp1d_linear_gpu, interp_regular_nd_grid


def kerncopccdfinv(w: torch.Tensor, ccdf_grid: torch.Tensor, 
                   u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """
    Inverse h-function for kernel copulas.
    
    Args:
        w: Random uniform values, shape [N, 2]
        ccdf_grid: Conditional CDF / h-function grid values
        u1, u2: Grid axes
        
    Returns:
        U2 values sampled from the copula
    """
    device = w.device
    len_w = w.shape[0]
    len_ax = u1.shape[0]
    
    # Tile for broadcasting
    u1_tile = u1.repeat(len_w, 1).t()  # [len_ax, len_w]
    w0_tile = w[:, 0].repeat(len_ax, 1)  # [len_ax, len_w]
    
    # Find nearest indices
    m1 = torch.argmin(torch.abs(u1_tile - w0_tile), dim=0)
    
    # Gather conditional-CDF values and project them to a monotone path.
    g = ccdf_grid[m1, :].t()  # [len_ax, len_w]
    g = torch.nan_to_num(g, nan=0.5, posinf=1.0, neginf=0.0)
    g = torch.clamp(g, 0.0, 1.0)
    g = torch.cummax(g, dim=0).values
    
    # Monotone inversion along the u2 axis with linear interpolation.
    target = w[:, 1]
    target_tile = target.repeat(len_ax, 1)
    idx_hi = torch.sum(g < target_tile, dim=0).clamp(max=len_ax - 1)
    idx_lo = (idx_hi - 1).clamp(min=0)

    col_idx = torch.arange(len_w, device=device)
    g_lo = g[idx_lo, col_idx]
    g_hi = g[idx_hi, col_idx]
    u_lo = u2[idx_lo]
    u_hi = u2[idx_hi]

    denom = (g_hi - g_lo).abs().clamp_min(1e-8)
    alpha = ((target - g_lo) / denom).clamp(0.0, 1.0)
    u_interp = u_lo + alpha * (u_hi - u_lo)

    at_lower = idx_hi == 0
    return torch.where(at_lower, u_hi, u_interp).clamp(float(u2.min()), float(u2.max()))


def _inverse_nonparametric_edge_h(cop, conditioning: np.ndarray, uniforms: np.ndarray, u1: torch.Tensor, u2: torch.Tensor) -> np.ndarray:
    """Invert the calibrated nonparametric h-function used by evaluation.

    The fitted model propagates pseudo-observations through a calibrated
    conditional-CDF map, not the raw grid values. Sampling has to invert that
    same calibrated map to preserve uniform margins.
    """
    if getattr(cop, "family", "kercop") == "ind":
        return np.clip(np.asarray(uniforms, dtype=np.float32).reshape(-1), 1e-6, 1.0 - 1e-6)
    device = u1.device
    cond_t = torch.as_tensor(conditioning, dtype=torch.float32, device=device).reshape(-1)
    uni_t = torch.as_tensor(uniforms, dtype=torch.float32, device=device).reshape(-1)
    raw_axis = cop.ccdf_train_raw
    u_axis = cop.ccdf_train_u
    raw_axis_t = raw_axis.to(device) if torch.is_tensor(raw_axis) else torch.tensor(raw_axis, dtype=torch.float32, device=device)
    u_axis_t = u_axis.to(device) if torch.is_tensor(u_axis) else torch.tensor(u_axis, dtype=torch.float32, device=device)
    raw_target = interp1d_linear_gpu(uni_t, u_axis_t, raw_axis_t)

    len_w = cond_t.shape[0]
    len_ax = u1.shape[0]
    u1_tile = u1.repeat(len_w, 1).t()
    cond_tile = cond_t.repeat(len_ax, 1)
    m1 = torch.argmin(torch.abs(u1_tile - cond_tile), dim=0)

    ccdf_grid = getattr(cop, "ccdf_grid", None)
    if ccdf_grid is None:
        ccdf_grid = getattr(cop, "cdf", None)
    if ccdf_grid is None:
        raise ValueError("Nonparametric copula is missing a conditional CDF grid.")
    ccdf_grid_t = ccdf_grid.to(device) if torch.is_tensor(ccdf_grid) else torch.tensor(ccdf_grid, dtype=torch.float32, device=device)
    g = ccdf_grid_t[m1, :].t()
    g = torch.nan_to_num(g, nan=0.5, posinf=1.0, neginf=0.0)
    g = torch.clamp(g, 0.0, 1.0)
    g = torch.cummax(g, dim=0).values

    target_tile = raw_target.repeat(len_ax, 1)
    idx_hi = torch.sum(g < target_tile, dim=0).clamp(max=len_ax - 1)
    idx_lo = (idx_hi - 1).clamp(min=0)
    col_idx = torch.arange(len_w, device=device)
    g_lo = g[idx_lo, col_idx]
    g_hi = g[idx_hi, col_idx]
    u_lo = u2[idx_lo]
    u_hi = u2[idx_hi]
    denom = (g_hi - g_lo).abs().clamp_min(1e-8)
    alpha = ((raw_target - g_lo) / denom).clamp(0.0, 1.0)
    u_interp = u_lo + alpha * (u_hi - u_lo)
    at_lower = idx_hi == 0
    out = torch.where(at_lower, u_hi, u_interp).clamp(float(u2.min()), float(u2.max()))
    return out.detach().cpu().numpy().astype(np.float32)


def _evaluate_nonparametric_edge_h_sampled(cop, uv: np.ndarray, vine) -> np.ndarray:
    """Match the TensorFlow sampler: interpolate raw h-values then rank-calibrate
    them within the current sampled batch to recover approximately uniform
    pseudo-observations."""
    if getattr(cop, "family", "kercop") == "ind":
        return np.clip(np.asarray(uv, dtype=np.float32)[:, 1], 1e-6, 1.0 - 1e-6)
    uv_t = torch.tensor(uv, dtype=torch.float32)
    points_s = Transform(1).forward_u(uv_t)
    ccdf_grid = getattr(cop, "ccdf_grid", None)
    if ccdf_grid is None:
        ccdf_grid = getattr(cop, "cdf", None)
    if ccdf_grid is None:
        raise ValueError("Nonparametric copula is missing a conditional CDF grid.")
    ccdf_grid_t = ccdf_grid if torch.is_tensor(ccdf_grid) else torch.tensor(ccdf_grid, dtype=torch.float32)
    grid_s_min = cop.grid_s_min if torch.is_tensor(cop.grid_s_min) else torch.tensor(cop.grid_s_min, dtype=torch.float32)
    grid_s_max = cop.grid_s_max if torch.is_tensor(cop.grid_s_max) else torch.tensor(cop.grid_s_max, dtype=torch.float32)
    raw = interp_regular_nd_grid(points_s, grid_s_min, grid_s_max, ccdf_grid_t).detach().cpu().numpy()
    ex = vine.grid_u.ex.detach().cpu().numpy() if torch.is_tensor(vine.grid_u.ex) else np.asarray(vine.grid_u.ex)
    calibrated, _mar_s, _mar_p = kernel_cdf(raw, raw, ex)
    return np.clip(calibrated.astype(np.float32), 1e-6, 1.0 - 1e-6)


def vine_copula_sample(vine, cases: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample from a non-parametric vine copula.
    
    Args:
        vine: Vine copula object
        cases: Number of samples to generate
        
    Returns:
        sample1: Samples in original space
        u: Samples in uniform space
        sample_pdf: PDF grid values
        sample_pds: PDS grid values
    """
    edge_refs = getattr(vine, "_internal_ind_vine", getattr(vine, "ind_vine", None))
    if edge_refs is None:
        raise ValueError("Nonparametric sampler requires a fitted vine structure.")

    d = getattr(vine, "n_cop", len(edge_refs) + 1)
    n = d - 1

    # Always draw true uniforms; the grid support is only used for interpolation.
    w = np.random.uniform(1e-6, 1.0 - 1e-6, (cases, d)).astype(np.float32)
    v = np.zeros((cases, d, d), dtype=np.float32)
    v_flip = np.zeros((cases, d, d), dtype=np.float32)
    v[:, 0, 0] = w[:, 0]

    if hasattr(vine, 'grid_u') and vine.grid_u is not None:
        u1 = vine.grid_u.ax1
        u2 = vine.grid_u.ax2
        device = u1.device if torch.is_tensor(u1) else torch.device('cpu')
        u1 = u1 if torch.is_tensor(u1) else torch.tensor(u1, dtype=torch.float32, device=device)
        u2 = u2 if torch.is_tensor(u2) else torch.tensor(u2, dtype=torch.float32, device=device)
    else:
        device = torch.device('cpu')
        u1 = torch.linspace(0, 1, 50, device=device)
        u2 = torch.linspace(0, 1, 50, device=device)

    sample_order = list(getattr(vine, "_sample_order", list(range(d))))
    sampling_r_matrix = getattr(vine, "_sampling_r_matrix", None)
    sampling_nodes = getattr(vine, "_sampling_nodes", None)

    forward_flip_cache = [[[] for _ in range(ii)] for ii in range(1, d - 1)]

    def _resolve_edge_index(tr: int, col: int, flip_flag: Optional[bool] = None) -> int:
        ind_edge_rel = getattr(vine, "ind_edge_rel", None)
        flip_flags = getattr(vine, "flip_flag", None)
        if not ind_edge_rel or tr >= len(ind_edge_rel):
            return int(col)
        candidates = np.where(np.asarray(ind_edge_rel[tr]) == int(col))[0]
        if len(candidates) == 0:
            return int(col)
        if flip_flag is None or not flip_flags or tr >= len(flip_flags):
            return int(candidates[0])
        for cand in candidates:
            if cand < len(flip_flags[tr]) and bool(flip_flags[tr][cand]) == bool(flip_flag):
                return int(cand)
        return int(candidates[0])

    for i in range(1, d):
        v[:, i, i] = w[:, i]

        c = 0
        for k in range(i - 1, -1, -1):
            tr = k
            col = i - k - 1
            ind_now = edge_refs[k][c]
            edge_idx = _resolve_edge_index(tr, col)

            if k == 0:
                if sampling_r_matrix is not None and sampling_nodes is not None:
                    tr1 = n - k
                    col1 = n - i
                    ind1_val = sampling_r_matrix[tr1, col1]
                    ind1_arr = np.where(sampling_nodes == ind1_val)[0]
                    ind1 = int(ind1_arr[0]) if len(ind1_arr) > 0 else 0
                else:
                    ind1 = 0
                v2 = v[:, k, ind1][:, np.newaxis]
                v1 = v[:, k + 1, i][:, np.newaxis]
                vv = np.concatenate((v2, v1), axis=1)
            else:
                parent, _inx1, _inx2 = parent_var(k, edge_refs, ind_now)
                if edge_refs[k - 1][ind_now[0]][0] != parent:
                    v2 = v_flip[:, k, k + ind_now[0]][:, np.newaxis]
                else:
                    v2 = v[:, k, k + ind_now[0]][:, np.newaxis]
                v1 = v[:, k + 1, i][:, np.newaxis]
                vv = np.concatenate((v2, v1), axis=1)

            cop = vine.copulas[tr][edge_idx]
            v[:, k, i] = _inverse_nonparametric_edge_h(
                cop,
                conditioning=vv[:, 0],
                uniforms=vv[:, 1],
                u1=u1,
                u2=u2,
            )
            c += 1

        if i < d - 1:
            cc1 = 0
            for ii in range(1, i + 1):
                cc2 = 0
                for j in range(0, ii):
                    tr = j
                    col = ii - j - 1
                    ind_now = edge_refs[j][ii - 1 - j]
                    ind_sup = edge_refs[j + 1][0] if j == n - 2 else edge_refs[j + 1][i - 1 - j]

                    flip_flag = False
                    parent, inx1, inx2 = parent_var(j + 1, edge_refs, ind_sup)
                    u_edge = {ind_now[0], ind_now[1]}
                    if ((u_edge.issubset(inx1)) or (u_edge.issubset(inx2))) and ind_now[0] != parent:
                        flip_flag = True

                    if i > 1 and flip_flag in forward_flip_cache[cc1][cc2]:
                        cc2 += 1
                        continue
                    forward_flip_cache[cc1][cc2].append(flip_flag)

                    if j == 0:
                        if sampling_r_matrix is not None and sampling_nodes is not None:
                            tr1 = n - j
                            col1 = n - ii
                            ind1_val = sampling_r_matrix[tr1, col1]
                            ind1_arr = np.where(sampling_nodes == ind1_val)[0]
                            ind1 = int(ind1_arr[0]) if len(ind1_arr) > 0 else 0
                        else:
                            ind1 = 0
                        v2 = v[:, j, ind1][:, np.newaxis]
                    else:
                        parent1, _inx1, _inx2 = parent_var(j, edge_refs, ind_now)
                        if edge_refs[j - 1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:, j, j + ind_now[0]][:, np.newaxis]
                        else:
                            v2 = v[:, j, j + ind_now[0]][:, np.newaxis]

                    v1 = v[:, j, ii][:, np.newaxis]
                    data_u = np.concatenate((v2, v1), axis=1) if not flip_flag else np.concatenate((v1, v2), axis=1)

                    edge_idx = _resolve_edge_index(tr, col, flip_flag=flip_flag)
                    cop = vine.copulas[tr][edge_idx]
                    hval = _evaluate_nonparametric_edge_h_sampled(cop, data_u, vine)
                    if not flip_flag:
                        v[:, j + 1, ii] = hval
                    else:
                        v_flip[:, j + 1, ii] = hval
                    cc2 += 1
                cc1 += 1

    u = np.reshape(v[:, 0, :], w.shape)
    u1_out = np.zeros_like(u)
    for c, ind in enumerate(sample_order):
        u1_out[:, int(ind)] = u[:, c]
    u = np.clip(u1_out, 1e-6, 1.0 - 1e-6)
    
    # Add small centered jitter to avoid exact grid values without biasing means.
    if hasattr(vine, 'grid_u'):
        u_ax = vine.grid_u.ax1.cpu().numpy() if torch.is_tensor(vine.grid_u.ax1) else vine.grid_u.ax1
        gr_diff = vine.grid_u.diff1.cpu().numpy() if torch.is_tensor(vine.grid_u.diff1) else np.diff(u_ax)
        
        for i in range(d):
            u_p1 = u[:, i][:, np.newaxis]
            u_ax1 = np.tile(u_ax[:, np.newaxis], [1, u_p1.shape[0]]).T
            u_upd = np.tile(u_p1, [1, u_ax.shape[0]])
            u_diff = np.abs(u_ax1 - u_upd)
            ind1 = np.argmin(u_diff, axis=1)
            diff_val = gr_diff[np.minimum(ind1, len(gr_diff)-1)]
            jitter = diff_val * (np.random.uniform(-0.5, 0.5, diff_val.shape[0]))
            u[:, i] = u[:, i] + jitter
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    
    # Transform to original space using margins
    sample1 = np.zeros([cases, d], u.dtype)
    max_margin_size = 1
    if hasattr(vine, "Mar_G") and vine.Mar_G is not None and len(vine.Mar_G) > 0:
        max_margin_size = max(len(vine.Mar_G[j][0]) for j in range(d))
    sample_pdf = np.zeros([max_margin_size, d], u.dtype)
    sample_pds = np.zeros([max_margin_size, d], u.dtype)
    
    for i in range(d):
        if hasattr(vine, 'Mar_G') and vine.Mar_G is not None and i < len(vine.Mar_G):
            mar_s1, mar_p1 = vine.Mar_G[i]
            mar_s1 = mar_s1 if isinstance(mar_s1, np.ndarray) else mar_s1.cpu().numpy()
            mar_p1 = mar_p1 if isinstance(mar_p1, np.ndarray) else mar_p1.cpu().numpy()
            
            # Interpolate to get samples
            sample1_pro = np.interp(u[:, i], mar_p1, mar_s1)
            sample1[:, i] = prep_copula(sample1_pro, 0)
            margin_size = len(mar_p1)
            sample_pdf[:margin_size, i] = mar_p1
            sample_pds[:margin_size, i] = mar_s1
        else:
            # Use margin distribution if available
            if hasattr(vine, 'margin') and i < len(vine.margin):
                margin = vine.margin[i]
                if margin.dist == 'norm':
                    loc, scale = margin.theta
                    from scipy.stats import norm
                    sample1[:, i] = norm.ppf(u[:, i], loc=loc, scale=scale)
                else:
                    sample1[:, i] = u[:, i]
            else:
                sample1[:, i] = u[:, i]
    
    return sample1, u, sample_pdf, sample_pds


def vine_cop_par_sample(vine, cases: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample from a parametric vine copula.
    
    Args:
        vine: Vine copula object
        cases: Number of samples to generate
        
    Returns:
        sample1: Samples in original space  
        u: Samples in uniform space
        sample_pdf: PDF grid values
        sample_pds: PDS grid values
    """
    from .nonparametric_vine import _build_internal_edge_structure, _build_sampling_metadata

    d = getattr(vine, "n_cop", len(getattr(vine, "margin", [])))
    n = d - 1

    edge_refs = getattr(vine, "_internal_ind_vine", None)
    if edge_refs is None:
        edge_refs = _build_internal_edge_structure(vine, d)
        vine._internal_ind_vine = edge_refs

    sample_order = list(getattr(vine, "_sample_order", []))
    sampling_r_matrix = getattr(vine, "_sampling_r_matrix", None)
    sampling_nodes = getattr(vine, "_sampling_nodes", None)
    if not sample_order:
        sample_order, sampling_r_matrix, sampling_nodes = _build_sampling_metadata(vine, d)
        vine._sample_order = sample_order
        vine._sampling_r_matrix = sampling_r_matrix
        vine._sampling_nodes = sampling_nodes

    mag = 1.0 - 1e-5
    mig = 1e-5
    w = np.random.uniform(mig, mag, (cases, d)).astype(np.float32)
    v = np.zeros((cases, d, d), dtype=np.float32)
    v_flip = np.zeros((cases, d, d), dtype=np.float32)
    v[:, 0, 0] = w[:, 0]

    def _map_var_to_pos(var_id: int) -> int:
        try:
            return int(sample_order.index(int(var_id)))
        except ValueError:
            return int(var_id)

    for i in range(1, d):
        v[:, i, i] = w[:, i]

        c = 0
        for k in range(i - 1, -1, -1):
            tr = k
            col = i - k - 1
            ind_now = edge_refs[k][c]

            if k == 0:
                if sampling_r_matrix is not None and sampling_nodes is not None:
                    tr1 = n - k
                    col1 = n - i
                    ind1_val = sampling_r_matrix[tr1, col1]
                    ind1_arr = np.where(sampling_nodes == ind1_val)[0]
                    ind1 = int(ind1_arr[0]) if len(ind1_arr) > 0 else _map_var_to_pos(ind_now[0])
                else:
                    ind1 = _map_var_to_pos(ind_now[0])

                v2 = v[:, k, ind1][:, np.newaxis]
                v1 = v[:, k + 1, i][:, np.newaxis]
                vv = np.concatenate((v2, v1), axis=1)
                v[:, k, i] = copulainvccdf(vine.copulas[tr][col], torch.from_numpy(vv)).cpu().numpy()
            else:
                parent, _inx1, _inx2 = parent_var(k, edge_refs, ind_now)
                if edge_refs[k - 1][ind_now[0]][0] != parent:
                    v2 = v_flip[:, k, k + ind_now[0]][:, np.newaxis]
                else:
                    v2 = v[:, k, k + ind_now[0]][:, np.newaxis]

                v1 = v[:, k + 1, i][:, np.newaxis]
                vv = np.concatenate((v2, v1), axis=1)
                v[:, k, i] = copulainvccdf(vine.copulas[tr][col], torch.from_numpy(vv)).cpu().numpy()
            c += 1

        if i < d - 1:
            for ii in range(1, i + 1):
                for j in range(0, ii):
                    tr = j
                    col = ii - j - 1
                    ind_now = edge_refs[j][ii - 1 - j]

                    if j == n - 2:
                        ind_sup = edge_refs[j + 1][0]
                    else:
                        ind_sup = edge_refs[j + 1][i - 1 - j]

                    if j == 0:
                        if sampling_r_matrix is not None and sampling_nodes is not None:
                            tr1 = n - j
                            col1 = n - ii
                            ind1_val = sampling_r_matrix[tr1, col1]
                            ind1_arr = np.where(sampling_nodes == ind1_val)[0]
                            ind1 = int(ind1_arr[0]) if len(ind1_arr) > 0 else _map_var_to_pos(ind_now[0])
                        else:
                            ind1 = _map_var_to_pos(ind_now[0])
                        v2 = v[:, j, ind1][:, np.newaxis]
                    else:
                        parent1, _inx1, _inx2 = parent_var(j, edge_refs, ind_now)
                        if edge_refs[j - 1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:, j, j + ind_now[0]][:, np.newaxis]
                        else:
                            v2 = v[:, j, j + ind_now[0]][:, np.newaxis]

                    v1 = v[:, j, ii][:, np.newaxis]
                    vv = np.concatenate((v2, v1), axis=1)

                    parent, inx1, inx2 = parent_var(j + 1, edge_refs, ind_sup)
                    u_edge = {ind_now[0], ind_now[1]}
                    if ((u_edge.issubset(inx1)) or (u_edge.issubset(inx2))) and ind_now[0] != parent:
                        vv_flip = np.concatenate((v1, v2), axis=1)
                        v_flip[:, j + 1, ii] = copulaccdf(vine.copulas[tr][col], torch.from_numpy(vv_flip)).cpu().numpy()
                    else:
                        v[:, j + 1, ii] = copulaccdf(vine.copulas[tr][col], torch.from_numpy(vv)).cpu().numpy()

    u = np.reshape(v[:, 0, :], w.shape)
    u_reordered = np.zeros_like(u)
    for c, ind in enumerate(sample_order):
        u_reordered[:, int(ind)] = u[:, c]
    u = np.clip(u_reordered, 1e-6, 1 - 1e-6)

    sample1 = np.zeros((cases, d), dtype=u.dtype)
    for i in range(d):
        if hasattr(vine, 'margin') and i < len(vine.margin):
            margin = vine.margin[i]
            if margin.dist == 'norm':
                loc, scale = margin.theta
                from scipy.stats import norm
                sample1[:, i] = norm.ppf(u[:, i], loc=loc, scale=scale)
            else:
                sample1[:, i] = u[:, i]
        else:
            sample1[:, i] = u[:, i]

    sample_pdf = np.zeros_like(u)
    sample_pds = np.zeros_like(u)
    return sample1, u, sample_pdf, sample_pds
