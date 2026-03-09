###############################################
# src/DVC/vine_model.py
###############################################
"""
Parametric vine fitting, evaluation, and sampling.

The current parametric path works with `C-vine`, `D-vine`, and `R-vine`
structures through the common `ind_vine` edge representation. The
nonparametric path is delegated to `nonparametric_vine.py`.
"""

import torch
import math
import numpy as np
import logging
from typing import Optional

from .objects import vine_obj_bin, cop_par_obj
from .param_copula import parametric_fit, copulapdf, copulaccdf, copulainvccdf
from .utils_prob import kernel_cdf
from .grid_ops import mk_grid, grid_obj
from .transformation import Transform
from .utils_prob import biv_norm

logger = logging.getLogger("DVC.vine")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")


def _prepare_parametric_internal_structure(vine: vine_obj_bin, d: int) -> None:
    from .nonparametric_vine import _build_internal_edge_structure, _build_sampling_metadata

    vine._internal_ind_vine = _build_internal_edge_structure(vine, d)
    sample_order, sample_r_matrix, sample_nodes = _build_sampling_metadata(vine, d)
    vine._sample_order = sample_order
    vine._sampling_r_matrix = sample_r_matrix
    vine._sampling_nodes = sample_nodes


def _initialize_parametric_margin_state(
    vine: vine_obj_bin,
    x_torch: torch.Tensor,
    device: torch.device,
) -> None:
    n, d = x_torch.shape
    vine.theta = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    vine.theta_flip = torch.zeros_like(vine.theta)
    vine.flip_flag = []
    vine.ind_edge_rel = []
    vine.training_data = x_torch.detach().cpu().numpy().astype(np.float32, copy=True)

    for i in range(d):
        values = x_torch[:, i].contiguous()
        sorted_col = torch.sort(values)[0]
        ranks = torch.searchsorted(sorted_col, values).float() + 1.0
        vine.theta[:, 0, i] = (ranks / (n + 1.0)).clamp(1e-6, 1.0 - 1e-6)
        if i < len(vine.margin):
            vine.margin[i].ker = values.detach().cpu().numpy()


def _evaluate_parametric_margin_state(vine: vine_obj_bin, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = points.device
    n, d = points.shape
    log_marg = torch.zeros(n, device=device)
    u_state = torch.zeros((n, d, d), dtype=torch.float32, device=device)

    for i in range(d):
        if i < len(vine.margin) and vine.margin[i].dist == "norm" and hasattr(vine.margin[i], "theta"):
            loc, scale = vine.margin[i].theta
            dist_i = torch.distributions.Normal(float(loc), float(scale))
        else:
            dist_i = torch.distributions.Normal(0.0, 1.0)
        log_marg = log_marg + dist_i.log_prob(points[:, i])
        u_state[:, 0, i] = dist_i.cdf(points[:, i]).clamp(1e-6, 1.0 - 1e-6)

    return log_marg, u_state


def fit_vine(
    vine: vine_obj_bin,
    x: np.ndarray,
    gen_dict: dict,
    npc_dict: dict,
    par_dict: dict,
    bin_dict: dict,
    cfg: Optional[dict] = None
):
    """Fit a parametric vine using the internal edge-index recursion."""
    if not bool(gen_dict.get("param", False)):
        from .nonparametric_vine import fit_nonparametric_vine
        return fit_nonparametric_vine(vine, x, gen_dict, npc_dict, par_dict, bin_dict, cfg=cfg)

    from .nonparametric_vine import _build_edge_input_pairs
    from .vine_tree import flip_check_all

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)

    vine.param = True
    vine.binning = False
    vine.fitted = True
    d = x.shape[1]
    vine.n_cop = d

    # Build a small grid if needed
    knots = vine.knots if hasattr(vine, 'knots') else 30
    ex_u = mk_grid(knots, dtype=torch.float32).to(device)
    vine.grid_u = grid_obj(ex_u)

    # For reference, we store a normal PDF grid
    transformer = Transform(d)
    vine.grid_s = grid_obj(transformer.forward_u(ex_u))
    x1_s, x2_s = vine.grid_s.axis()
    norm2d = biv_norm(x1_s, x2_s).to(device)
    vine.ref_bivnorm = norm2d

    _prepare_parametric_internal_structure(vine, d)
    _initialize_parametric_margin_state(vine, x_torch, device)

    vine.copulas = []
    families = par_dict.get("param_families", ["ind","gaussian","clayton"])

    logger.info(f"Vine topology (family={vine.vine_family}, method={vine.method}, d={d})")
    for lv, eds in enumerate(getattr(vine, "_internal_ind_vine", vine.ind_vine)):
        logger.info(f"  Level {lv}: {eds}")

    for level in range(d - 1):
        logger.info(f"Fitting level {level}/{d-1}...")
        edges_now = vine._internal_ind_vine[level] if level < len(vine._internal_ind_vine) else []
        data_u = _build_edge_input_pairs(
            state=vine.theta,
            state_flip=vine.theta_flip,
            edge_refs=vine._internal_ind_vine,
            level=level,
            device=device,
        )
        cop_list_level = []

        for edge_idx, edge in enumerate(edges_now):
            _ = edge
            uv = torch.nan_to_num(
                data_u[:, :, edge_idx],
                nan=0.5,
                posinf=1.0 - 1e-6,
                neginf=1e-6,
            ).clamp(1e-6, 1.0 - 1e-6)
            data_np = uv.unsqueeze(-1).detach().cpu().numpy()
            aic2, thetas_list, logp_list = parametric_fit(data_np, families, n_cop=1)
            xvals = data_np[:,0,0]
            yvals = data_np[:,1,0]
            if np.std(xvals)<1e-15 or np.std(yvals)<1e-15:
                emp_corr = 0.0
            else:
                x_centered = xvals - np.mean(xvals)
                y_centered = yvals - np.mean(yvals)
                sx = np.std(x_centered, ddof=0)
                sy = np.std(y_centered, ddof=0)
                c_ = 0.0
                if sx > 1e-12 and sy > 1e-12:
                    c_ = float(np.mean((x_centered / sx) * (y_centered / sy)))
                if not np.isfinite(c_):
                    c_ = 0.0
                emp_corr = abs(c_)
            n_smpl = xvals.shape[0]
            for ff, fam_ in enumerate(families):
                if fam_=='ind':
                    if emp_corr>0.1:
                        aic2[0][ff]+= n_smpl*(emp_corr**2)*10
                    elif emp_corr>0.05:
                        aic2[0][ff]+= n_smpl*(emp_corr**2)*5

            best_idx = np.argmin(aic2[0])
            fam_best = families[best_idx]
            par_best = thetas_list[0][best_idx]
            cobj = cop_par_obj(fam_best, par_best)
            cop_list_level.append(cobj)

        vine.copulas.append(cop_list_level)
        flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(vine._internal_ind_vine, level, False, 1)
        vine.flip_flag.append(flip_flag1)
        vine.ind_edge_rel.append(ind_edge_rel1)

        for j, ind_edge in enumerate(ind_edge_rel1):
            if ind_edge >= len(cop_list_level):
                continue
            cobj = cop_list_level[ind_edge]
            uv = data_u[:, :, ind_edge]
            try:
                if flip_flag1[j]:
                    hval = copulaccdf(cobj, uv[:, [1, 0]]).clamp(1e-6, 1.0 - 1e-6)
                    hval = torch.where(torch.isfinite(hval), hval, uv[:, 0])
                    vine.theta_flip[:, level + 1, ind_edge] = hval
                else:
                    hval = copulaccdf(cobj, uv).clamp(1e-6, 1.0 - 1e-6)
                    hval = torch.where(torch.isfinite(hval), hval, uv[:, 1])
                    vine.theta[:, level + 1, ind_edge] = hval
            except Exception as exc:
                logger.error("Error partial transform (level=%d, edge=%d): %s", level, ind_edge, exc)
                if flip_flag1[j]:
                    vine.theta_flip[:, level + 1, ind_edge] = uv[:, 0]
                else:
                    vine.theta[:, level + 1, ind_edge] = uv[:, 1]

    vine.fitted = True
    return vine


def evaluate_vine(vine: vine_obj_bin, points: torch.Tensor):
    """Evaluate a fitted parametric vine density."""
    if not getattr(vine, "param", True):
        from .nonparametric_vine import evaluate_nonparametric_vine
        return evaluate_nonparametric_vine(vine, points)

    from .nonparametric_vine import _build_edge_input_pairs
    from .vine_tree import flip_check_all

    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    points = points.float()
    device = points.device
    n, d = points.shape

    edge_refs = getattr(vine, "_internal_ind_vine", None)
    if edge_refs is None:
        _prepare_parametric_internal_structure(vine, d)
        edge_refs = vine._internal_ind_vine

    log_marg, u_state = _evaluate_parametric_margin_state(vine, points)
    u_state_flip = torch.zeros_like(u_state)

    log_cop = torch.zeros(n, device=device)
    for level in range(d - 1):
        if not getattr(vine, "flip_flag", None) or level >= len(vine.flip_flag):
            flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(edge_refs, level, False, 1)
        else:
            flip_flag1 = vine.flip_flag[level]
            ind_edge_rel1 = vine.ind_edge_rel[level]

        point_u = _build_edge_input_pairs(
            state=u_state,
            state_flip=u_state_flip,
            edge_refs=edge_refs,
            level=level,
            device=device,
        )
        cop_now = vine.copulas[level] if level < len(vine.copulas) else []
        for j, ind_edge in enumerate(ind_edge_rel1):
            if ind_edge >= len(cop_now):
                continue
            cobj = cop_now[ind_edge]
            uv = point_u[:, :, ind_edge]
            pdf_val = copulapdf(cobj, uv).clamp_min(1e-30)
            log_cop = log_cop + torch.log(pdf_val)
            try:
                if flip_flag1[j]:
                    hval = copulaccdf(cobj, uv[:, [1, 0]]).clamp(1e-6, 1.0 - 1e-6)
                    u_state_flip[:, level + 1, ind_edge] = torch.where(torch.isfinite(hval), hval, uv[:, 0])
                else:
                    hval = copulaccdf(cobj, uv).clamp(1e-6, 1.0 - 1e-6)
                    u_state[:, level + 1, ind_edge] = torch.where(torch.isfinite(hval), hval, uv[:, 1])
            except Exception as exc:
                logger.warning("Partial transform failed at level=%d, edge=%d: %s", level, ind_edge, exc)
                if flip_flag1[j]:
                    u_state_flip[:, level + 1, ind_edge] = uv[:, 0]
                else:
                    u_state[:, level + 1, ind_edge] = uv[:, 1]

    log_p = log_marg + log_cop
    pdf_ = torch.exp(log_p)
    return pdf_, torch.exp(log_cop), log_marg


def sample_vine(vine: vine_obj_bin, nsamples: int, cfg: Optional[dict]=None):
    """
    Sample from a fitted parametric vine using the standard nested formula:
      1) sample U[:,0] from uniform => transform to X0 from margin
      2) for edges in level=0 => sample U_j from copulainvccdf( U_center, rand )
         => transform to X_j
      3) partial transform => store h(U_j|U_center) for next level
      4) move on to level=1 => center=1 => sample the leftover variables
      ...
    """
    if not getattr(vine, "param", True):
        from .sampling import vine_copula_sample
        try:
            sample1, _u, _sample_pdf, _sample_pds = vine_copula_sample(vine, nsamples)
            return sample1
        except Exception as exc:
            logger.warning("Nonparametric sampler fell back to bootstrap resampling: %s", exc)
            train = getattr(vine, "training_data", None)
            if train is None:
                raise
            idx = np.random.choice(train.shape[0], size=int(nsamples), replace=True)
            return np.asarray(train, dtype=np.float32)[idx]

    from .sampling import vine_cop_par_sample
    d = vine.n_cop

    def _parametric_fast_sampler_is_stable(sample_arr: np.ndarray, u_arr: np.ndarray) -> bool:
        sample_np = np.asarray(sample_arr, dtype=np.float32)
        u_np = np.asarray(u_arr, dtype=np.float32)
        if sample_np.shape != (nsamples, d) or u_np.shape != (nsamples, d):
            return False
        if not np.isfinite(sample_np).all() or not np.isfinite(u_np).all():
            return False
        if np.any(u_np <= 0.0) or np.any(u_np >= 1.0):
            return False

        # Uniform marginals should not collapse to the boundaries.
        boundary_rate = np.mean((u_np <= 1e-5) | (u_np >= 1.0 - 1e-5), axis=0)
        if np.any(boundary_rate > 0.05):
            return False

        u_std = np.std(u_np, axis=0)
        if np.any((u_std < 0.05) | (u_std > 0.40)):
            return False

        u_mean = np.mean(u_np, axis=0)
        if np.any(np.abs(u_mean - 0.5) > 0.20):
            return False

        return True

    try:
        sample1, _u, _sample_pdf, _sample_pds = vine_cop_par_sample(vine, nsamples)
        if _parametric_fast_sampler_is_stable(sample1, _u):
            return sample1
        logger.warning(
            "Parametric fast sampler produced unstable uniforms; using recursive/gaussian path."
        )
    except Exception as exc:
        logger.warning("Parametric vine sampler fell back to recursive/gaussian path: %s", exc)

    logger.info("Sampling from vine copula")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_x = torch.zeros((nsamples, d), dtype=torch.float32, device=device)

    def _get_margin_dist(var_idx: int):
        if (
            vine.margin
            and var_idx < len(vine.margin)
            and hasattr(vine.margin[var_idx], "dist")
        ):
            mar = vine.margin[var_idx]
            if mar.dist == "norm" and hasattr(mar, "theta") and mar.theta is not None:
                loc, scale = mar.theta
                return torch.distributions.Normal(float(loc), float(scale))
            if mar.dist == "uniform" and hasattr(mar, "theta") and mar.theta is not None:
                low, high = mar.theta
                return torch.distributions.Uniform(float(low), float(high))
        return torch.distributions.Normal(0.0, 1.0)

    def _sample_gaussian_copula_fallback() -> np.ndarray:
        """Robust fallback sampler using a Gaussian copula fit to stored margins."""
        data_cols = []
        for j in range(d):
            if vine.margin and j < len(vine.margin) and getattr(vine.margin[j], "ker", None) is not None:
                data_cols.append(np.asarray(vine.margin[j].ker).reshape(-1))
            else:
                data_cols.append(np.random.randn(max(nsamples, 128)))
        train_data = np.column_stack(data_cols)

        centered = train_data - np.mean(train_data, axis=0, keepdims=True)
        std = np.std(centered, axis=0, ddof=0)
        valid = std > 1e-12
        z = np.zeros_like(centered, dtype=np.float64)
        if np.any(valid):
            z[:, valid] = centered[:, valid] / std[valid]
        corr = (z.T @ z) / max(train_data.shape[0] - 1, 1)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        # Project to PSD-ish matrix with diagonal 1 for stable MVN sampling.
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-6, None)
        corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        dstd = np.sqrt(np.clip(np.diag(corr_psd), 1e-12, None))
        corr_psd = corr_psd / np.outer(dstd, dstd)
        np.fill_diagonal(corr_psd, 1.0)

        z = np.random.multivariate_normal(np.zeros(d), corr_psd, size=nsamples).astype(np.float32)
        z_t = torch.from_numpy(z).to(device=device, dtype=torch.float32)
        u_t = torch.distributions.Normal(0.0, 1.0).cdf(z_t).clamp(1e-6, 1 - 1e-6)

        x_t = torch.zeros_like(u_t)
        for j in range(d):
            distj = _get_margin_dist(j)
            x_t[:, j] = distj.icdf(u_t[:, j]).clamp(-1e3, 1e3)
        return x_t.cpu().numpy()

    # Use recursive inverse-h sampling for canonical C-vines.
    if vine.vine_family == "c-vine":
        U_ = torch.full((nsamples, d, d), torch.nan, dtype=torch.float32, device=device)

        # Infer C-vine ordering from structure (roots per level + remaining variable).
        root_order = []
        for level in range(max(d - 1, 0)):
            edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
            if edges_now:
                root_order.append(int(edges_now[0][0]))
        if len(root_order) != d - 1:
            root_order = list(range(d - 1))
        remaining = [v for v in range(d) if v not in root_order]
        variable_order = root_order + (remaining[:1] if remaining else [d - 1])

        # Draw base uniforms used for recursive inverse-h composition.
        w_all = torch.rand((nsamples, d), device=device).clamp(1e-6, 1 - 1e-6)

        # First/root variable.
        root_var = variable_order[0]
        U_[:, 0, root_var] = w_all[:, 0]
        dist_root = _get_margin_dist(root_var)
        out_x[:, root_var] = dist_root.icdf(U_[:, 0, root_var]).clamp(-1e3, 1e3)

        # Sample remaining variables in C-vine order using backward inverse-h recursion.
        for pos in range(1, d):
            target_var = variable_order[pos]
            z = w_all[:, pos]

            # Walk backward through tree levels: deepest conditional to level 0.
            for level in range(pos - 1, -1, -1):
                root = variable_order[level]
                edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
                cops_now = vine.copulas[level] if level < len(vine.copulas) else []

                edge_idx = None
                for e_idx, edge in enumerate(edges_now):
                    if int(edge[0]) == root and int(edge[1]) == target_var:
                        edge_idx = e_idx
                        break

                if edge_idx is None or edge_idx >= len(cops_now):
                    continue

                cobj = cops_now[edge_idx]
                u_cond = U_[:, level, root]
                uv = torch.stack([u_cond, z], dim=1)
                try:
                    z = copulainvccdf(cobj, uv).clamp(1e-6, 1 - 1e-6)
                except Exception as e:
                    logger.warning(
                        f"Inverse ccdf failed at level={level}, edge=({root},{target_var}): {e}"
                    )
                    z = z.clamp(1e-6, 1 - 1e-6)

            # Base-space uniform for target variable.
            U_[:, 0, target_var] = z
            dist_target = _get_margin_dist(target_var)
            out_x[:, target_var] = dist_target.icdf(z).clamp(-1e3, 1e3)

            # Forward propagate h-values for future conditioning levels.
            current = z
            for level in range(0, pos):
                root = variable_order[level]
                edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
                cops_now = vine.copulas[level] if level < len(vine.copulas) else []

                edge_idx = None
                for e_idx, edge in enumerate(edges_now):
                    if int(edge[0]) == root and int(edge[1]) == target_var:
                        edge_idx = e_idx
                        break

                if edge_idx is None or edge_idx >= len(cops_now):
                    U_[:, level + 1, target_var] = current
                    continue

                cobj = cops_now[edge_idx]
                u_cond = U_[:, level, root]
                uv = torch.stack([u_cond, current], dim=1)
                try:
                    current = copulaccdf(cobj, uv).clamp(1e-6, 1 - 1e-6)
                except Exception as e:
                    logger.warning(
                        f"Forward h failed at level={level}, edge=({root},{target_var}): {e}"
                    )
                    current = current.clamp(1e-6, 1 - 1e-6)
                U_[:, level + 1, target_var] = current

        # Guard against inverse-h numerical collapse: fallback to Gaussian copula.
        u_base = U_[:, 0, :]
        bad_uniforms = (
            torch.isnan(u_base).any()
            or torch.isinf(u_base).any()
            or ((u_base <= 1e-5) | (u_base >= 1 - 1e-5)).float().mean() > 0.05
            or (u_base.std(dim=0) < 0.02).any()
        )
        if bool(bad_uniforms):
            logger.warning(
                "C-vine inverse-h sampling became numerically unstable; using Gaussian-copula fallback."
            )
            return _sample_gaussian_copula_fallback()

        return out_x.cpu().numpy()

    # D-vine and R-vine sampling via generic inverse-h recursion.
    # The vine structure is encoded in vine.ind_vine[level] = [[i,j], ...].
    # We walk through the edges using the inverse Rosenblatt transform.
    if vine.vine_family in ("d-vine", "r-vine"):
        U_ = torch.full((nsamples, d, d), torch.nan, dtype=torch.float32, device=device)
        w_all = torch.rand((nsamples, d), device=device).clamp(1e-6, 1 - 1e-6)

        # For D-vine, the ordering is the path order from ind_vine.
        # For R-vine, use the structure as-is.
        # Determine variable order from tree-0 edges.
        if vine.vine_family == "d-vine" and len(vine.ind_vine) > 0:
            # D-vine: sequential path => extract ordering from level-0 edges
            edges_0 = vine.ind_vine[0]
            if edges_0:
                seen = []
                for e in edges_0:
                    for node in e:
                        if node not in seen:
                            seen.append(node)
                variable_order = seen
            else:
                variable_order = list(range(d))
        else:
            variable_order = list(range(d))

        # Pad if not all variables covered
        for v in range(d):
            if v not in variable_order:
                variable_order.append(v)

        # First variable: just a uniform draw
        first_var = variable_order[0]
        U_[:, 0, first_var] = w_all[:, 0]
        dist0 = _get_margin_dist(first_var)
        out_x[:, first_var] = dist0.icdf(U_[:, 0, first_var]).clamp(-1e3, 1e3)

        for pos in range(1, d):
            target_var = variable_order[pos]
            z = w_all[:, pos]

            # Walk backward through tree levels using inverse h-functions.
            for level in range(min(pos, len(vine.ind_vine)) - 1, -1, -1):
                edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
                cops_now = vine.copulas[level] if level < len(vine.copulas) else []

                # Find the edge at this level involving target_var
                edge_idx = None
                cond_var = None
                for e_idx, edge in enumerate(edges_now):
                    if int(edge[1]) == target_var:
                        edge_idx = e_idx
                        cond_var = int(edge[0])
                        break
                    if int(edge[0]) == target_var:
                        edge_idx = e_idx
                        cond_var = int(edge[1])
                        break

                if edge_idx is None or edge_idx >= len(cops_now):
                    continue

                cobj = cops_now[edge_idx]
                u_cond = U_[:, level, cond_var]
                u_cond = torch.where(torch.isfinite(u_cond), u_cond, torch.tensor(0.5, device=device))

                # Determine orientation: if target_var was edge[1], standard
                # if target_var was edge[0], we need to swap
                if int(edges_now[edge_idx][0]) == target_var:
                    # Swap: invert h_{target|cond} => h_{0|1}
                    uv = torch.stack([z, u_cond], dim=1)
                else:
                    uv = torch.stack([u_cond, z], dim=1)

                try:
                    z = copulainvccdf(cobj, uv).clamp(1e-6, 1 - 1e-6)
                except Exception as e:
                    logger.warning(
                        "Inverse ccdf failed at level=%d, target=%d: %s", level, target_var, e
                    )

            U_[:, 0, target_var] = z
            dist_t = _get_margin_dist(target_var)
            out_x[:, target_var] = dist_t.icdf(z).clamp(-1e3, 1e3)

            # Forward propagate h-values for future levels.
            current = z
            for level in range(min(pos, len(vine.ind_vine))):
                edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
                cops_now = vine.copulas[level] if level < len(vine.copulas) else []

                edge_idx = None
                cond_var = None
                for e_idx, edge in enumerate(edges_now):
                    if int(edge[1]) == target_var:
                        edge_idx = e_idx
                        cond_var = int(edge[0])
                        break
                    if int(edge[0]) == target_var:
                        edge_idx = e_idx
                        cond_var = int(edge[1])
                        break

                if edge_idx is None or edge_idx >= len(cops_now):
                    U_[:, level + 1, target_var] = current
                    continue

                cobj = cops_now[edge_idx]
                u_cond = U_[:, level, cond_var]
                u_cond = torch.where(torch.isfinite(u_cond), u_cond, torch.tensor(0.5, device=device))
                uv = torch.stack([u_cond, current], dim=1)
                try:
                    current = copulaccdf(cobj, uv).clamp(1e-6, 1 - 1e-6)
                except Exception as e:
                    logger.warning("Forward h failed at level=%d, target=%d: %s", level, target_var, e)
                U_[:, level + 1, target_var] = current

        # Numerical stability check
        u_base = U_[:, 0, :]
        bad_uniforms = (
            torch.isnan(u_base).any()
            or torch.isinf(u_base).any()
            or ((u_base <= 1e-5) | (u_base >= 1 - 1e-5)).float().mean() > 0.05
            or (u_base.std(dim=0) < 0.02).any()
        )
        if bool(bad_uniforms):
            logger.warning(
                "%s inverse-h sampling became numerically unstable; using Gaussian-copula fallback.",
                vine.vine_family,
            )
            return _sample_gaussian_copula_fallback()

        return out_x.cpu().numpy()

    # Final fallback for unrecognised vine types.
    logger.warning("Unknown vine family '%s'; using Gaussian-copula fallback.", vine.vine_family)
    return _sample_gaussian_copula_fallback()


def logpdf_vine(vine: vine_obj_bin, points: torch.Tensor):
    pdf_, cop_, logm_ = evaluate_vine(vine, points)
    pdf_clamped = pdf_.clamp_min(1e-30)
    logp = torch.log(pdf_clamped)
    return torch.where(torch.isfinite(logp), logp, torch.full_like(logp,-30.0))


def pdf_vine(vine: vine_obj_bin, points: torch.Tensor):
    pdf_,_,_ = evaluate_vine(vine, points)
    return pdf_


def cdf_vine(vine: vine_obj_bin, points: torch.Tensor, nsim=2000):
    """
    Monte Carlo approximation: sample from vine => P(X<=points).
    """
    device = points.device
    d = vine.n_cop
    samps = vine.sample(nsim)
    samps_t = torch.tensor(samps, device=device, dtype=points.dtype)
    out=[]
    for row in points:
        mask = (samps_t <= row).all(dim=1)
        out.append(mask.float().mean())
    return torch.stack(out, dim=0)
