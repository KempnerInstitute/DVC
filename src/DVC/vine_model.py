###############################################
# src/DVC/vine_model.py
###############################################

import torch
import numpy as np
import random
from scipy.stats import kendalltau

# Basic objects
from .objects import vine_obj_bin, copula_obj, cop_par_obj
from .utils_locallik import loclik_batch_eval
from .param_copula import parametric_fit, copulapdf, copulainvccdf
from .vine_tree import parent_var, flip_check_all
from .utils_prob import kernel_cdf
from .grid_ops import grid_obj, mk_grid
from .vine_eval import evaluate_fit_bin, evaluate_fit

############################################################
# 1) Internal Helpers for LL1 / LL2 bandwidth optimization
############################################################

def _ll_cost(a_bw: torch.Tensor,
             data_s: torch.Tensor,
             grid_x: torch.Tensor,
             n_cop: int,
             batch_size: int):
    """
    Negative average local-likelihood cost computed via loclik_batch_eval.
    We want to minimize: cost = - mean(loclik(...))
    """
    ker_grid = loclik_batch_eval(a_bw, data_s.unsqueeze(2), grid_x.unsqueeze(2), n_cop, batch_size)
    cost = - torch.mean(ker_grid)
    return cost

def _fit_ban_ll1(a_init: torch.Tensor,
                 data_s: torch.Tensor,
                 grid_x: torch.Tensor,
                 n_cop: int,
                 batch_size: int,
                 max_iter: int,
                 lr: float,
                 conv_tol: float,
                 verbose: bool = True):
    """
    LL1 style iterative optimization using a simple Adam optimizer.
    """
    device = data_s.device
    a = a_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([a], lr=lr)
    old_cost = 1e10
    for it in range(max_iter):
        optimizer.zero_grad()
        cost = _ll_cost(a, data_s, grid_x, n_cop, batch_size)
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            a.clamp_(0.001, 5.0)
        if verbose and (it+1) % 10 == 0:
            print(f"[LL1] Iteration {it+1}/{max_iter}, cost: {cost.item():.6f}")
        if abs(cost.item() - old_cost) < conv_tol:
            if verbose:
                print(f"[LL1] Converged at iteration {it+1} with cost: {cost.item():.6f}")
            break
        old_cost = cost.item()
    return a.detach()

def _fit_ban_ll2(a_init: torch.Tensor,
                 data_s: torch.Tensor,
                 grid_x: torch.Tensor,
                 n_cop: int,
                 batch_size: int,
                 max_iter: int,
                 lr: float,
                 conv_tol: float,
                 verbose: bool = True):
    """
    LL2 style iterative optimization (similar to LL1 but kept separate for potential differences).
    """
    device = data_s.device
    a = a_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([a], lr=lr)
    old_cost = 1e10
    for it in range(max_iter):
        optimizer.zero_grad()
        cost = _ll_cost(a, data_s, grid_x, n_cop, batch_size)
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            a.clamp_(0.001, 5.0)
        if verbose and (it+1) % 10 == 0:
            print(f"[LL2] Iteration {it+1}/{max_iter}, cost: {cost.item():.6f}")
        if abs(cost.item() - old_cost) < conv_tol:
            if verbose:
                print(f"[LL2] Converged at iteration {it+1} with cost: {cost.item():.6f}")
            break
        old_cost = cost.item()
    return a.detach()

def _optimization(grid_dict: dict, 
                 data_dict: dict, 
                 par_dict: dict):
    """
    Chooses between LL1 or LL2 approach and performs a two-phase iterative optimization.
    Returns the optimized bandwidth with shape [2,1].
    """
    data_s = data_dict['data_s']
    grid_x = grid_dict['grid_x']
    n_cop = par_dict.get('n_cop', 1)
    batch_size = par_dict.get('batch', 5)
    max_iter = par_dict.get('max_iter', [70, 100])
    lr = par_dict.get('lr', [0.1, 0.03])
    conv_tol = par_dict.get('conv_tol', [1e-5, 5e-5])
    opt_method = par_dict.get('opt_method', 'LL1')
    verbose = par_dict.get('verbose', True)

    device = data_s.device
    a_init = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)

    if opt_method == 'LL1':
        out = _fit_ban_ll1(a_init, data_s, grid_x, n_cop, batch_size,
                           max_iter[0], lr[0], conv_tol[0], verbose)
        out2 = _fit_ban_ll1(out, data_s, grid_x, n_cop, batch_size,
                            max_iter[1], lr[1], conv_tol[1], verbose)
    elif opt_method == 'LL2':
        out = _fit_ban_ll2(a_init, data_s, grid_x, n_cop, batch_size,
                           max_iter[0], lr[0], conv_tol[0], verbose)
        out2 = _fit_ban_ll2(out, data_s, grid_x, n_cop, batch_size,
                            max_iter[1], lr[1], conv_tol[1], verbose)
    else:
        out = _fit_ban_ll1(a_init, data_s, grid_x, n_cop, batch_size,
                           max_iter[0], lr[0], conv_tol[0], verbose)
        out2 = _fit_ban_ll1(out, data_s, grid_x, n_cop, batch_size,
                            max_iter[1], lr[1], conv_tol[1], verbose)
    return out2.view(2, 1)

############################################################
# 2) Fit Vine: Parametric and Nonparametric Branches
############################################################

def fit_vine(vine: vine_obj_bin,
             x: np.ndarray,
             gen_dict: dict,
             npc_dict: dict,
             par_dict: dict,
             bin_dict: dict):
    """
    Fit the vine on data x using a multi-level approach:
      - If vine.param==True, perform parametric copula fitting per edge.
      - Otherwise, use local-likelihood optimization (LL1/LL2) to optimize bandwidths.
    Implements a c-vine style where at level i, root=i and edges are (i, j) for j>i.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)

    vine.param = gen_dict['param']
    vine.binning = gen_dict['binning']
    vine.fitted = gen_dict['fitted']
    vine.n_bin = bin_dict['n_bin'] if vine.binning else 1

    d = x.shape[1]
    vine.n_cop = d
    vine.copulas = []

    if vine.param:
        families = par_dict['param_families']
        print(f"\n[PARAM] Fitting {vine.vine_family} vine dimension={d} with families={families}\n")
        for level in range(d-1):
            edges_cop = []
            root = level
            print(f"--- Level {level}, root={root}, #edges={d-1-level} ---")
            for e_idx, col2 in enumerate(range(level+1, d)):
                pair_data = x_torch[:, [root, col2]]
                N = pair_data.shape[0]
                print(f"   => (vars {root},{col2}), data shape=({N},2)")
                pair_np = pair_data.cpu().numpy()[:,:,None]
                aic2, theta_list, logp_list = parametric_fit(pair_np, families, n_cop=1)
                best_idx = np.argmin(aic2[0, :])
                fam_best = families[best_idx]
                param_best = theta_list[0][best_idx]
                cpo = cop_par_obj(fam_best, param_best)
                print(f"      => best family={fam_best}, param={param_best}")
                edges_cop.append(cpo)
            vine.copulas.append(edges_cop)
    else:
        opt_method = npc_dict.get('opt_method', 'LL1')
        print(f"\n[NON-PARAM] Fitting {vine.vine_family} vine dimension={d}, method={opt_method}\n")
        for level in range(d-1):
            edges_cop = []
            root = level
            print(f"--- Level {level}, root={root}, #edges={d-1-level} ---")
            for e_idx, col2 in enumerate(range(level+1, d)):
                pair_data = x_torch[:, [root, col2]]
                N = pair_data.shape[0]
                print(f"   => (vars {root},{col2}), data shape=({N},2)")
                knots = 50
                ex = mk_grid(knots=knots, dtype=torch.float32).to(device)
                grid_2d = grid_obj(ex)
                grid_dict = {
                    'grid_x': ex
                }
                data_dict = {
                    'data_s': pair_data
                }
                par_dict_ = {
                    'n_cop': 1,
                    'batch': 5,
                    'max_iter': [70, 100],
                    'lr': [0.1, 0.03],
                    'conv_tol': [1e-5, 5e-5],
                    'opt_method': opt_method,
                    'verbose': True
                }
                opt_bw = _optimization(grid_dict, data_dict, par_dict_)
                if opt_bw.ndim == 1:
                    opt_bw = opt_bw.unsqueeze(1)
                print(f"      => (vars {root},{col2}) => Fitted bandwidth=({opt_bw[0,0].item():.4f}, {opt_bw[1,0].item():.4f})")
                cobj = copula_obj(opt_bw=opt_bw)
                edges_cop.append(cobj)
            vine.copulas.append(edges_cop)

    vine.fitted = True
    print(f"\n==> Completed vine fitting. #levels={len(vine.copulas)}\n")
    for lvl, edge_list in enumerate(vine.copulas):
        print(f"  Level {lvl} has {len(edge_list)} bivariate copulas.")
    print()


############################################################
# 3) Evaluate Vine
############################################################

def evaluate_vine(vine: vine_obj_bin, points: torch.Tensor):
    """
    Evaluate the vine PDF at the given points.
    Computes the sum of marginal log-PDFs (using standard normal) plus
    the log-PDF contributions from each bivariate copula.
    """
    device = points.device
    d = vine.n_cop
    normal_dist = torch.distributions.Normal(0., 1.)

    log_marg = torch.zeros(points.shape[0], device=device)
    for i in range(d):
        log_marg += normal_dist.log_prob(points[:, i])

    log_cop = torch.zeros(points.shape[0], device=device)
    for level in range(d-1):
        edges_cop = vine.copulas[level]
        root = level
        for e_idx, cobj in enumerate(edges_cop):
            col2 = root + 1 + e_idx
            uv = points[:, [root, col2]]
            if vine.param:
                pdf_val = copulapdf(cobj, uv)
                log_cop += torch.log(torch.clamp(pdf_val, 1e-30, 1e30))
            else:
                bw = cobj.opt_bw
                scale_x = bw[0, 0].item()
                scale_y = bw[1, 0].item()
                dx = uv[:, 0] - uv[:, 0].mean()
                dy = uv[:, 1] - uv[:, 1].mean()
                e = -0.5 * ((dx/scale_x)**2 + (dy/scale_y)**2)
                log_cop += e
    logp = log_marg + log_cop
    p = torch.exp(logp)
    return p, torch.exp(log_cop), log_marg


############################################################
# 4) Sample Vine
############################################################

def _kerncopccdfinv(w: torch.Tensor, cdf_grid: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor):
    """
    Rough re-creation of the old kerncopccdfinv from your TF code:
    For each sample in w (shape [N,2]), find the index in u1 and u2 such that
    the cdf in the grid meets the criterion for w[:,1].
    """
    N = w.shape[0]
    out = torch.zeros(N, dtype=w.dtype, device=w.device)
    for i in range(N):
        x0 = w[i, 0]
        ycdf = w[i, 1]
        diff1 = torch.abs(u1 - x0)
        i1 = torch.argmin(diff1)
        rowcdf = cdf_grid[i1, :]
        bigger = (rowcdf >= ycdf)
        bigger_idxs = torch.where(bigger)[0]
        if len(bigger_idxs) == 0:
            j = rowcdf.shape[0] - 1
        else:
            j = bigger_idxs[0].item()
        out[i] = u2[j]
    return out

def sample_vine(vine: vine_obj_bin, nsamples: int):
    """
    Sample from the fitted vine.
    
    For the parametric branch, we use a simplified c-vine inversion:
      - Sample the first margin uniformly,
      - For each subsequent variable, use the fitted parametric copula (via copulainvccdf)
        to generate a conditional uniform sample.
      - Finally, transform the uniform vine-copula samples into x-space by applying the 
        inverse marginal CDF. For margins with dist=='norm', we use the standard normal 
        inverse CDF with the parameters provided in margin.theta; otherwise, we assume the 
        margin object holds the original data (in vine.margin[i].data) and use linear interpolation.
    
    For the nonparametric branch, a similar approach is used (here we also use the level 0 copula
    as a simplified demonstration).
    
    The final output is samples in x-space.
    """
    import random
    import numpy as np
    from scipy.stats import norm
    # d: dimension of the vine (number of margins)
    d = vine.n_cop

    # --- Step 1: Generate vine copula uniform samples (u-values) using a sequential (c-vine) approach ---
    # Here we use a simplified approach: for the first variable, sample U[0] ~ Uniform(0,1).
    # For each subsequent variable, use the fitted copula from level 0 (i.e. vine.copulas[0])
    # to obtain a conditional uniform value.
    samples_u = np.zeros((nsamples, d), dtype=np.float64)
    # Sample first margin uniformly:
    samples_u[:, 0] = np.random.rand(nsamples)
    
    if vine.param:
        # Parametric branch:
        for col in range(1, d):
            cpo = vine.copulas[0][col-1]  # using level 0 edge for simplicity
            for n in range(nsamples):
                u0 = samples_u[n, 0]
                # Generate an independent random value for the conditional component:
                u_rand = random.random()
                uv = np.array([[u0, u_rand]], dtype=np.float32)
                uv_torch = torch.from_numpy(uv)
                # copulainvccdf should compute the inverse conditional CDF given the copula parameters.
                # (Make sure that function is properly implemented.)
                samples_u[n, col] = float(copulainvccdf(cpo, uv_torch).item())
    else:
        # Nonparametric branch (using the local-likelihood fitted copula)
        for col in range(1, d):
            cobj = vine.copulas[0][col-1]  # using level 0 edge for simplicity
            for n in range(nsamples):
                u0 = samples_u[n, col-1]
                u_rand = random.random()
                uv = np.array([[u0, u_rand]], dtype=np.float32)
                uv_torch = torch.from_numpy(uv)
                samples_u[n, col] = float(copulainvccdf(cobj, uv_torch).item())
    
    # --- Step 2: Transform the uniform vine copula samples into x-space using the marginal inverse CDF ---
    # For each margin i, if the margin's distribution is 'norm', we use norm.ppf with parameters given in margin.theta.
    # Otherwise, we use a simple linear interpolation based on the stored original data in vine.margin[i].data.
    samples_x = np.zeros((nsamples, d), dtype=np.float64)
    for i in range(d):
        u_vals = samples_u[:, i]
        if vine.margin[i].dist == 'norm':
            # Use the inverse CDF of a normal distribution with parameters given by margin.theta.
            # Expect margin.theta to be a list/tuple [loc, scale]. Default to (0,1) if not provided.
            loc_param = vine.margin[i].theta[0] if isinstance(vine.margin[i].theta, (list, tuple, np.ndarray)) else 0.0
            scale_param = vine.margin[i].theta[1] if isinstance(vine.margin[i].theta, (list, tuple, np.ndarray)) else 1.0
            samples_x[:, i] = norm.ppf(u_vals, loc=loc_param, scale=scale_param)
        else:
            # For nonparametric margins, we assume that the original margin values are stored
            # in vine.margin[i].data (set during preparation). Then we use linear interpolation.
            if hasattr(vine.margin[i], 'data'):
                sorted_data = np.sort(vine.margin[i].data)
                num_pts = len(sorted_data)
                grid_u = np.linspace(0, 1, num_pts)
                samples_x[:, i] = np.interp(u_vals, grid_u, sorted_data)
            else:
                # Fallback: return the uniform values if no marginal data is available.
                samples_x[:, i] = u_vals
    return samples_x

############################################################
# Attach methods to vine_obj_bin
############################################################
vine_obj_bin.fit = fit_vine
vine_obj_bin.evaluation = evaluate_vine
vine_obj_bin.sample = sample_vine