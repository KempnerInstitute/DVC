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
        sorted_col = torch.sort(x_torch[:, i])[0]
        ranks = torch.searchsorted(sorted_col, x_torch[:, i]).float() + 1
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
        print(f"Fitting level {level}/{d-1}...")
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
                except:
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
    logger.info("Using simple C-vine parametric sampling")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = vine.n_cop
    out_x = torch.zeros((nsamples, d), dtype=torch.float32, device=device)
    # We'll maintain a big array U => shape [nsamples, d, d]
    U_ = torch.zeros((nsamples, d, d), dtype=torch.float32, device=device)

    # sample variable 0 from margin
    if d>0:
        if vine.margin and vine.margin[0].dist=='norm' and hasattr(vine.margin[0],'theta'):
            loc0, scale0 = vine.margin[0].theta
            dist0 = torch.distributions.Normal(loc0, scale0)
        else:
            dist0 = torch.distributions.Normal(0.,1.)
        x0 = dist0.sample((nsamples,))
        out_x[:,0] = x0
        u0 = dist0.cdf(x0).clamp(1e-6, 1-1e-6)
        U_[:,0,0] = u0

    # for each level in [0..d-2] - generalized for all vine types
    for level in range(d-1):
        edges_now = vine.ind_vine[level] if level<len(vine.ind_vine) else []
        cops_now = vine.copulas[level] if level<len(vine.copulas) else []
        
        for e_idx, edge in enumerate(edges_now):
            if e_idx>=len(cops_now):
                continue
            cobj = cops_now[e_idx]
            i, j = edge
            
            # Get the conditioning variable (first in edge) and target variable (second in edge)
            # This works for all vine types since the edge structure encodes the dependencies
            u_conditioning = U_[:, level, i]  # shape [nsamples]
            
            # Check if target variable j has already been sampled at this level
            # If not, we need to sample it
            if level == 0 or not torch.any(torch.isfinite(U_[:, level, j])):
                # step 1) random uniform => w
                w = torch.rand(nsamples, device=device).clamp(1e-6,1-1e-6)
                uv = torch.stack([u_conditioning, w], dim=1)
                
                # invert => u_j
                try:
                    u_j = copulainvccdf(cobj, uv)
                    u_j = torch.clamp(u_j, 1e-6,1-1e-6)
                except:
                    # fallback => just w
                    u_j = w
                U_[:, level, j] = u_j
                
                # convert to x_j via margin (only for level 0, i.e., original variables)
                if level == 0:
                    if vine.margin and len(vine.margin)>j and vine.margin[j].dist=='norm' and hasattr(vine.margin[j],'theta'):
                        locj, scalej = vine.margin[j].theta
                        distj = torch.distributions.Normal(locj, scalej)
                    else:
                        distj = torch.distributions.Normal(0.,1.)
                    xj = distj.icdf(u_j).clamp(-1e3,1e3)
                    out_x[:, j] = xj

            # partial transform => next level
            if level<(d-1):
                uv2 = torch.stack([u_conditioning, U_[:, level, j]], dim=1)
                try:
                    h_val = copulaccdf(cobj, uv2).clamp(1e-6,1-1e-6)
                    U_[:, level+1, j] = h_val
                except:
                    U_[:, level+1, j] = u_j
    
    return out_x.cpu().numpy()


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


def _attach_methods():
    vine_obj_bin.fit = fit_vine
    vine_obj_bin.evaluation = evaluate_vine
    vine_obj_bin.sample = sample_vine
    vine_obj_bin.logpdf = logpdf_vine
    vine_obj_bin.pdf = pdf_vine
    vine_obj_bin.cdf = cdf_vine

try:
    _attach_methods()
except:
    pass