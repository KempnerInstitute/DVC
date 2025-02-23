###############################################
# src/DVC/vine_model.py
###############################################

import torch
import numpy as np
import random
from scipy.stats import kendalltau

# Imports from your codebase
from .objects import vine_obj_bin, copula_obj, cop_par_obj
from .transformation import Transform
from .utils_bandwidth import bandwidth_rule_of_thumb
from .utils_locallik import loclik_batch_eval
from .dataset_ops import kfold
from .param_copula import parametric_fit, copulapdf, copulainvccdf
from .vine_tree import parent_var, flip_check_all
from .utils_prob import kernel_cdf, biv_norm
from .grid.grid_class import grid_obj   # if you have a grid approach
from .grid.grid_op import mk_grid       # etc. if needed for local-likelihood
from .evalu.vine_eval import evaluate_fit_bin, evaluate_fit


def fit_vine(vine: vine_obj_bin,
             x: np.ndarray,
             gen_dict: dict,
             npc_dict: dict,
             par_dict: dict,
             bin_dict: dict):
    """
    Fit the vine on data x, doing a multi-level approach for either:
      - param => store param copulas per edge
      - nonparam => store local-likelihood copulas per edge

    We'll do a "tree" approach for the chosen vine_family: 'c-vine','d-vine','r-vine'
    For dimension = d, we have (d-1) levels, each with (d-1)-level edges, etc.

    The final result is stored in vine.copulas => a list [level][edge] of copula_obj or cop_par_obj.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)

    vine.param = gen_dict['param']
    vine.binning = gen_dict['binning']
    vine.fitted = gen_dict['fitted']
    vine.n_bin = bin_dict['n_bin'] if vine.binning else 1
    d = x.shape[1]
    vine.n_cop = d  # or maybe store as well

    # We store per-level sets of bivariate copulas
    vine.copulas = []

    if vine.param:
        # Param approach => build multi-level structure with param copulas
        # For simplicity, we do a "chain" approach if c-vine => the "root" is col0 for level0, col1 for level1, etc.
        # Real c-vine logic: level i => root = i, edges with j in [i+1..d-1].
        families = par_dict['param_families']

        # We'll store data for each "level" in x_torch, but we do partial transformations or something more advanced.
        # For a fully correct c-vine, you'd do a conditional approach. We'll do a simpler approach for demonstration.

        # We'll define a function to get edges for c-vine
        # E.g. level=0 => edges = (0,1),(0,2)...(0,d-1)
        # level=1 => edges = (1,2),(1,3)...(1,d-1), etc.
        # Then parametric fit each pair => store in vine.copulas[level][edge].
        # This is still somewhat simplified, ignoring the truly correct conditional steps, but is multi-level.

        for level in range(d-1):
            # number of edges => d-1-level
            edges_cop = []
            root = level
            for col2 in range(level+1, d):
                # data for param fit => columns root,col2 of x
                # in a real c-vine, we have to condition on col [0..level-1], but let's do a simpler approach for demonstration
                pair_data = x_torch[:, [root, col2]]
                # shape => [N,2], convert to np => do param fit
                pair_np = pair_data.cpu().numpy()  # shape [N,2]
                # we expand shape => [N,2,1] to feed parametric_fit easily
                pair_np2 = pair_np[:,:,None]  # shape [N,2,1]
                aic2, theta_list, logp_list = parametric_fit(pair_np2, families, n_cop=1)
                # we have aic2 => shape [1, len(families]], pick best
                best_idx = np.argmin(aic2[0,:])
                fam_best = families[best_idx]
                param_best = theta_list[0][best_idx]
                # store as cop_par_obj
                cpo = cop_par_obj(fam_best, param_best)
                edges_cop.append(cpo)
            vine.copulas.append(edges_cop)

    else:
        # Nonparam => local-likelihood approach => multi-level
        # We'll do c-vine style. For each level, we have (d-1-level) bivariate edges.
        # We'll store them in vine.copulas[level][edge].
        # Then "flip" logic if needed. For a fully correct approach, we'd do conditionals, but let's do a simpler approach that
        # at least yields multi-level bivariate fits.

        # We'll define a function for c-vine edges: level => root=level, edges => (level, col2 in [level+1..d-1]).
        # Then do local-likelihood => store a copula_obj per bivariate.

        for level in range(d-1):
            edges_cop = []
            root = level
            for col2 in range(level+1, d):
                # Pair columns => shape [N,2], do local-likelihood => store bandwidth
                pair_data = x_torch[:, [root, col2]]
                # shape => [N,2], we do [N,2,1] to feed bandwidth_rule_of_thumb if needed
                pair_data3d = pair_data.unsqueeze(2)  # [N,2,1]
                # get bandwidth
                bw = bandwidth_rule_of_thumb(pair_data3d, deg=2, n_cop=1)  # shape [2,1]
                # store in a copula_obj
                cobj = copula_obj(opt_bw=bw)
                edges_cop.append(cobj)
            vine.copulas.append(edges_cop)

    vine.fitted = True
    return


def evaluate_vine(vine: vine_obj_bin, points: torch.Tensor):
    """
    Evaluate PDF at 'points'.  If param => sum of logpdf from each bivariate edge. 
    If nonparam => sum of local-likelihood. 
    This is still somewhat simplified for c-vine logic. 
    We'll do the "no real conditional" approach => just multiply all bivariate edges from the multi-level, ignoring conditionals. 
    But it uses more than one copula.

    For param => do \prod_{level=0..d-2} \prod_{edge in level} copula(x_{root}, x_{col2}).
    For nonparam => same approach, but local-likelihood with each stored bandwidth. 
    """
    device = points.device
    d = vine.n_cop
    # first margin => treat as standard normal
    normal_dist = torch.distributions.Normal(0.,1.)
    # sum of log marg
    log_marg = torch.zeros(points.shape[0], device=device)
    for i in range(d):
        zcol = points[:, i]
        log_marg += normal_dist.log_prob(zcol)

    if vine.param:
        # param approach => loop over vine.copulas[level]
        # points shape => [N,d]
        log_cop = torch.zeros(points.shape[0], device=device)
        for level in range(d-1):
            edges_cop = vine.copulas[level]
            root = level
            for e_idx, cpo in enumerate(edges_cop):
                col2 = root + 1 + e_idx
                # gather columns => shape [N,2]
                if col2 < d:
                    uv = points[:, [root, col2]]
                    pdf_val = copulapdf(cpo, uv)
                    log_cop += torch.log(torch.clamp(pdf_val, 1e-30, 1e30))
        logp = log_marg + log_cop
        p = torch.exp(logp)
        return p, torch.exp(log_cop), log_marg
    else:
        # nonparam => loop the same multi-level approach, each edge is a single local-likelihood "bandwidth"
        # We'll do naive multiplication
        log_cop = torch.zeros(points.shape[0], device=device)
        for level in range(d-1):
            edges_cop = vine.copulas[level]
            root = level
            for e_idx, cobj in enumerate(edges_cop):
                col2 = root + 1 + e_idx
                if col2 < d:
                    # local-likelihood => shape [N], do a single bivariate fit 
                    # we do a minimal approach => "evaluate" 
                    # For real logic, we might do a grid or direct formula. We'll do a naive kernel approach for demonstration.
                    uv = points[:, [root, col2]]  # shape [N,2]
                    # We can define a small function local_ll_pdf(uv, cobj)
                    # We'll do a naive approach => just do a single bandwidth Gaussian kernel (like a product kernel).
                    # opt_bw => shape [2,1]
                    bw = cobj.opt_bw  # shape [2,1]
                    if bw.ndim == 2 and bw.shape[1] == 1:
                        # kernel => product of Gaussian
                        # pdf( (u1,u2) ) ~ \prod exp(-0.5*( (u1-mean)/bw[0])^2 ) ignoring normalizing constants for demonstration
                        # We'll do a minimal approach for the sake of example. 
                        # In reality you'd do loclik_batch_eval or so, but let's show:
                        scale_x = bw[0,0].item()
                        scale_y = bw[1,0].item()
                        # do logpdf
                        # ignoring normalizing constant => partial 
                        # or if we do a simplified:
                        dx = uv[:,0] - uv[:,0].mean()
                        dy = uv[:,1] - uv[:,1].mean()
                        e = -0.5*((dx/scale_x)**2 + (dy/scale_y)**2)
                        # sum => logpdf 
                        logpdf_approx = e  # missing constants, but let's do partial
                        log_cop += logpdf_approx
        logp = log_marg + log_cop
        p = torch.exp(logp)
        return p, torch.exp(log_cop), log_marg


def sample_vine(vine: vine_obj_bin, nsamples: int):
    """
    Sample from the fitted vine. 
    If param => do multi-level approach to generate all variables. 
    If nonparam => we do a naive approach for demonstration (just random? or partial).
    For a real c-vine, we'd do col0 ~ Uniform(0,1), col1 ~ cond( col0 ), etc.

    We'll illustrate a partial approach for param. 
    For nonparam => we do i.i.d uniform as placeholder. Real logic would be advanced.
    """
    d = vine.n_cop
    if vine.param:
        # param approach => do c-vine style generation:
        # col0 ~ Uniform(0,1)
        # next => col1 = cond on col0 => we pick from the edge in level=0 => e_idx=0 => cpo => inverse cond
        # etc. This is a partial demonstration. 
        # for a real approach, you'd handle the multi-level condition. We'll do chain approach.
        samples = np.zeros((nsamples,d), dtype=np.float64)
        for n in range(nsamples):
            row = np.zeros(d, dtype=np.float64)
            # root=0 => uniform
            row[0] = random.random()
            # for col in [1..d-1], we find the relevant edge in vine.copulas[0][col-1], do inverse
            for col in range(1,d):
                cpo = vine.copulas[0][col-1]  # partial
                uv = np.array([[row[0], random.random()]], dtype=np.float32)
                uv_torch = torch.from_numpy(uv)
                row[col] = float(copulainvccdf(cpo, uv_torch).item())
            samples[n,:] = row
        return samples
    else:
        # nonparam => trivial placeholder => i.i.d. uniform
        return np.random.rand(nsamples, d)


# Attach to vine_obj_bin
vine_obj_bin.fit = fit_vine
vine_obj_bin.evaluation = evaluate_vine
vine_obj_bin.sample = sample_vine