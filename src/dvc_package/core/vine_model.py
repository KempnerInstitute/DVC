###############################################
# src/DVC/vine_model.py
###############################################
"""
A parametric C-vine implementation that:
1) Fits each "tree level" pair-copula in a partial correlation sense.
2) For sampling, applies the standard nested formula from Aas (2009).
3) Preserves correlations among all variables.

Families handled: 'ind','gaussian','clayton' (extend as needed).
No non-parametric code or binning is included here—param only.
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


def fit_vine(
    vine: vine_obj_bin,
    x: np.ndarray,
    gen_dict: dict,
    npc_dict: dict,
    par_dict: dict,
    bin_dict: dict,
    cfg: Optional[dict] = None
):
    """
    Fit a parametric C-vine (only) using partial correlation logic.
    
    Steps:
      (1) Initialize vine.theta[:,0,i] with ranks of X_i.
      (2) For level=0..(d-2):
           for each edge => (level, j) with j>level
               fit pair-copula( U_level, U_j )
               store best param => cop_par_obj
               transform => h(U_j| U_level) => vine.theta[:, level+1, j]
      (3) Done => vine.copulas[level] = list of copulas for edges (level, j).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)
    
    # We assume param==True => no non-param
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

    # Initialize vine.theta => shape [N, d, d]
    N = x_torch.shape[0]
    vine.theta = torch.zeros((N, d, d), dtype=torch.float32, device=device)

    # margins => rank transform => store in vine.theta[:,0,i]
    for i in range(d):
        values = x_torch[:, i].contiguous()
        sorted_col = torch.sort(values)[0]
        ranks = torch.searchsorted(sorted_col, values).float() + 1
        vine.theta[:, 0, i] = ranks / (N+1)
        # store in margin object
        if i < len(vine.margin):
            vine.margin[i].ker = x_torch[:, i].cpu().numpy()

    # Build a c-vine structure if not exist: level 0 => edges => (0,1),(0,2)...
    if not vine.ind_vine or len(vine.ind_vine)==0:
        vine.ind_vine = []
        for level in range(d-1):
            edges_level = []
            for j in range(level+1, d):
                edges_level.append([level,j])
            vine.ind_vine.append(edges_level)

    # We'll store all pair-copulas in vine.copulas => list-of-levels
    vine.copulas = []
    families = par_dict.get("param_families", ["ind","gaussian","clayton"])

    logger.info(f"Vine topology (family={vine.vine_family}, method={vine.method}, d={d})")
    for lv, eds in enumerate(vine.ind_vine):
        logger.info(f"  Level {lv}: {eds}")

    # Fit each level - generalized for C/D/R-vines
    for level in range(d-1):
        logger.info(f"Fitting level {level}/{d-1}...")
        vine_type = getattr(vine, 'vine_family', 'c-vine')
        logger.info(f"Using parametric fitting for {vine_type} level {level}")

        edges_now = vine.ind_vine[level] if level<len(vine.ind_vine) else []
        cop_list_level = []

        for edge_idx, edge in enumerate(edges_now):
            i, j = edge
            # No assumption about center node - use edges as given by vine structure
            # This works for C-vine (where center logic was applied during structure creation),
            # D-vine (path structure), and R-vine (arbitrary valid structure)

            ui = vine.theta[:, level, i]  # shape [N]
            uj = vine.theta[:, level, j]  # shape [N]
            # Stack => shape [N,2,1]
            data_pair = torch.stack([ui, uj], dim=1).unsqueeze(-1)
            
            # Fit param families => parametric_fit => returns (aic2, thetas, logp)
            data_np = data_pair.cpu().numpy()
            aic2, thetas_list, logp_list = parametric_fit(data_np, families, n_cop=1)

            # optional "independence penalty"
            # measure correlation
            xvals = data_np[:,0,0]
            yvals = data_np[:,1,0]
            if np.std(xvals)<1e-15 or np.std(yvals)<1e-15:
                emp_corr = 0.0
            else:
                c_ = np.corrcoef(xvals, yvals)[0,1]
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
            # build copula object
            cobj = cop_par_obj(fam_best, par_best)
            cop_list_level.append(cobj)

            # Now transform for next level => h(u_j|u_i)
            if level<(d-1):
                uv = torch.stack([ui, uj], dim=1)
                try:
                    # partial transform => hVal = F_j|i ( uj | ui )
                    hval = copulaccdf(cobj, uv).clamp(1e-6, 1-1e-6)
                    vine.theta[:, level+1, j] = hval
                except Exception as e:
                    logger.error(f"Error partial transform (level={level},edge=({i},{j})): {str(e)}")
                    vine.theta[:, level+1, j] = uj  # fallback => no transform

        vine.copulas.append(cop_list_level)

    vine.fitted = True
    return vine


def evaluate_vine(vine: vine_obj_bin, points: torch.Tensor):
    """
    Evaluate pdf for param c-vine. We'll do the same partial logic:
      1) Convert each col of points to U ~ margin.
      2) For level=0..(d-2):
          parse edges => (level,j)
          multiply logpdf by copula pdf( u_level, u_j ).
          partial transform => h( u_j| u_level ) => next level
    """
    device = points.device
    N, d = points.shape
    # margin => standard normal or user
    log_marg = torch.zeros(N, device=device)
    u_ = torch.zeros((N,d,d), dtype=torch.float32, device=device)
    for i in range(d):
        # margin i
        if i<len(vine.margin) and vine.margin[i].dist=='norm' and hasattr(vine.margin[i],'theta'):
            loc, scale = vine.margin[i].theta
            dist_i = torch.distributions.Normal(loc, scale)
        else:
            dist_i = torch.distributions.Normal(0.,1.)
        lpm = dist_i.log_prob(points[:, i])
        log_marg += lpm
        ui_ = dist_i.cdf(points[:, i]).clamp(1e-6, 1-1e-6)
        u_[:, 0, i] = ui_

    log_cop = torch.zeros(N, device=device)
    for level in range(d-1):
        edges_now = vine.ind_vine[level] if level<len(vine.ind_vine) else []
        cop_now = vine.copulas[level] if level<len(vine.copulas) else []
        for e_idx, edge in enumerate(edges_now):
            if e_idx>=len(cop_now):
                continue
            cobj = cop_now[e_idx]
            i, j = edge
            # Use edges as given by vine structure (no center node assumption)
            ui = u_[:, level, i]
            uj = u_[:, level, j]
            uv = torch.stack([ui, uj], dim=1)
            # compute pdf
            pdf_val = copulapdf(cobj, uv).clamp_min(1e-30)
            log_cop += torch.log(pdf_val)
            # partial transform => next level
            if level<(d-1):
                try:
                    hval = copulaccdf(cobj, uv).clamp(1e-6,1-1e-6)
                    u_[:, level+1, j] = hval
                except Exception as e:
                    logger.warning(f"Partial transform failed at level={level}, edge=({i},{j}): {e}")
                    u_[:, level+1, j] = uj

    log_p = log_marg + log_cop
    pdf_ = torch.exp(log_p)
    return pdf_, torch.exp(log_cop), log_marg


def sample_vine(vine: vine_obj_bin, nsamples: int, cfg: Optional[dict]=None):
    """
    Sample from param c-vine using the standard nested formula:
      1) sample U[:,0] from uniform => transform to X0 from margin
      2) for edges in level=0 => sample U_j from copulainvccdf( U_center, rand )
         => transform to X_j
      3) partial transform => store h(U_j|U_center) for next level
      4) move on to level=1 => center=1 => sample the leftover variables
      ...
    """
    logger.info("Sampling from vine copula")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = vine.n_cop
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

        corr = np.corrcoef(train_data.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
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

    # Fallback for non C-vines: use Gaussian copula approximation.
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
