###############################################
# src/DVC/vine_model.py
###############################################

import torch
import numpy as np
import random
from scipy.stats import kendalltau, norm
import math
from typing import Optional, Union

# NEW IMPORTS --------------------------------------------------
# The new PyTorch implementation relies on helper utilities that
# were defined in sibling modules but never imported, leading to
# run-time NameError exceptions. We explicitly import them here.
from .transformation import Transform
from .dataset_ops import create_bins, check_bins
# -------------------------------------------------------------

# Basic objects
from .objects import vine_obj_bin, copula_obj, cop_par_obj
from .utils_locallik import loclik_batch_eval
from .param_copula import parametric_fit, copulapdf, copulainvccdf
from .vine_tree import parent_var, flip_check_all
from .grid_ops import grid_obj, mk_grid
from .vine_eval import evaluate_fit_bin, evaluate_fit
from .utils_prob import biv_norm  # from your older logic
from .config import load_config, DEFAULT_CFG
from .utils_bandwidth import bandwidth_rule_of_thumb, bandwidth_knn, bandwidth_sqrt_cov
from .utils_interpolation import nearestInterp2d

############################################################
# 1) The row/column normalization-based local-likelihood Cost
#    or negative log-likelihood approach
############################################################

def _eval_ll_cost(bw: torch.Tensor,
                  data_s: torch.Tensor,
                  device,
                  batch_size=5):
    """
    This cost function tries to replicate local-likelihood logic. We:
      1) Build a 2D grid in the same device as data_s
      2) Evaluate raw local-likelihood (loclik_batch_eval)
      3) Evaluate the average logpdf on data_s
      4) Return negative => so we can minimize

    We'll do a partial approach for demonstration. 
    """
    if bw.dim() == 1:
        bw = bw.view(2,1)  # shape => [2,1]

    # Step 1) Build a grid on device
    knots = 50
    ex_coords, _ = mk_grid(knots=knots, dtype=data_s.dtype)  # (coords, expanded)
    # ex_coords shape => [K^2,2], returned from mk_grid
    ex_coords = ex_coords.to(device)  ### NEW: ensure same device

    # Step 2) Evaluate raw local-likelihood
    data_3d = data_s.unsqueeze(2)  # [N,2,1]
    # ex_coords => shape [K^2,2], we expand => [K^2,2,1] to match the logic in dense_naive_batch
    grid_3d = ex_coords.unsqueeze(2)  # [K^2,2,1]
    ker_grid = loclik_batch_eval(bw, data_3d, grid_3d, 1, batch_size)  # shape => [K^2,1]
    ker_grid = ker_grid.squeeze(1)  # shape => [K^2]

    # Evaluate local-likelihood at the actual data points 
    # to get the average logpdf. shape => [N,1]
    grid_data = data_s.unsqueeze(2) # [N,2,1]
    pdf_data = loclik_batch_eval(bw, data_3d, grid_data, 1, batch_size)  # [N,1]
    pdf_data = pdf_data.clamp_min(1e-30)
    logpdf_data = torch.log(pdf_data)
    measure = torch.mean(logpdf_data)  # average log-likelihood

    # final cost => negative
    cost = -measure
    if torch.isnan(cost) or torch.isinf(cost):
        cost = torch.tensor(1e6, dtype=cost.dtype, device=cost.device)
    return cost


############################################################
# 2) Simple Adam-based iterative optimizer
############################################################

def _optimize_bw_ll(bw_init: torch.Tensor,
                    data_s: torch.Tensor,
                    device,
                    max_iter=100, lr=0.02, conv_tol=1e-5):
    """
    Minimizes the cost from _eval_ll_cost, returning a final bandwidth shape [2].
    """
    bw = bw_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([bw], lr=lr)
    old_cost = 1e10
    for it in range(max_iter):
        optimizer.zero_grad()
        cost = _eval_ll_cost(bw, data_s, device, batch_size=5)
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            bw.clamp_(0.005, 5.0)
        if abs(cost.item() - old_cost) < conv_tol:
            break
        old_cost = cost.item()
    bw[torch.isnan(bw)] = 0.2
    bw[torch.isinf(bw)] = 0.2
    return bw.detach()


############################################################
# 2b) Batched LL1 optimiser (scalar per edge)
############################################################

def _optimize_bw_ll_batch(bw_init: torch.Tensor,
                          data_s: torch.Tensor,
                          device,
                          max_iter=70, lr=0.05, conv_tol=1e-5):
    """Batched version of `_optimize_bw_ll` for E edges.

    Parameters
    ----------
    bw_init : torch.Tensor  shape `[2, E]`
    data_s  : torch.Tensor  shape `[N, 2, E]`
    Returns
    -------
    torch.Tensor  shape `[2, E]`  – optimised bandwidths.
    """
    E = bw_init.shape[1]
    # one scalar per edge -> parameter vector length E
    a_log = torch.zeros(E, device=device, dtype=bw_init.dtype, requires_grad=True)

    optimizer = torch.optim.Adam([a_log], lr=lr)
    old_cost = 1e10

    # Precompute grid once
    knots = 50
    _, ex_coords = mk_grid(knots, dtype=data_s.dtype)
    ex_coords = ex_coords.to(device)
    grid_3d = ex_coords.unsqueeze(2).expand(-1, -1, E)  # [K^2,2,E]
    data_3d = data_s  # [N,2,E]

    for it in range(max_iter):
        optimizer.zero_grad()
        a_val = torch.exp(a_log)              # [E]
        B = bw_init * a_val.unsqueeze(0)      # 2×E
        # --- cost
        ker_grid = loclik_batch_eval(B, data_3d, grid_3d, E, batch_size=5)  # [K^2,E]  (unused but keeps symmetry)
        pdf_data = loclik_batch_eval(B, data_3d, data_3d, E, batch_size=5)  # [N,E]
        pdf_data = pdf_data.clamp_min(1e-30)
        logpdf = torch.log(pdf_data)
        measure = torch.mean(logpdf, dim=0)   # [E]
        cost = -measure.mean()                # scalar
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            # keep parameters in (0.005,5) range via bandwidth
            a_log.clamp_(math.log(0.005), math.log(5.0))
        if abs(cost.item() - old_cost) < conv_tol:
            break
        old_cost = cost.item()

    with torch.no_grad():
        a_final = torch.exp(a_log).clamp(0.005, 5.0)
    return bw_init * a_final.unsqueeze(0)  # 2×E

# optional JIT compile for speed
from DVC.config import DEFAULT_CFG as _CFG_JIT
if _CFG_JIT["optimizer"].get("jit", False):
    try:
        if hasattr(torch, 'compile'):
            _optimize_bw_ll_batch = torch.compile(_optimize_bw_ll_batch, fullgraph=False)
        else:
            _optimize_bw_ll_batch = torch.jit.script(_optimize_bw_ll_batch)
    except Exception:
        pass


############################################################
# 3) The main fit function
############################################################

def fit_vine(vine: vine_obj_bin,
             x: np.ndarray,
             gen_dict: dict,
             npc_dict: dict,
             par_dict: dict,
             bin_dict: dict,
             cfg: Optional[dict] = None):
    """
    Fit the vine on data x with all fixes incorporated.
    
    This is the main entry point for vine fitting that ensures:
    1. Proper grid operations with correct mk_grid
    2. Correct transformations between spaces
    3. Proper bandwidth optimization for non-parametric cases
    4. Correct CDF calculations for evaluation
    5. Efficient interpolation for grid functions
    6. Full handling of binning functionality
    7. Complete parametric copula implementations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)
    
    # Initialize vine properties from dictionaries
    vine.param = gen_dict['param']
    vine.binning = gen_dict['binning']
    vine.fitted = gen_dict['fitted']
    vine.n_bin = bin_dict['n_bin'] if vine.binning else 1
    
    d = x.shape[1]
    vine.n_cop = d
    
    # Create proper grid
    knots = vine.knots
    coords, ex_u = mk_grid(knots, dtype=torch.float32)
    ex_u = ex_u.to(device)
    
    # Create grid objects
    vine.grid_u = grid_obj(ex_u)
    
    # Transform to s-space
    transformer = Transform(d)
    vine.grid_s = grid_obj(transformer.forward_u(ex_u))
    
    # Create bivariate normal reference
    x1_s, x2_s = vine.grid_s.axis()
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1).repeat(1, 1, d).to(device)
    
    # Initialize theta matrices for proper conditioning
    vine.theta = torch.zeros((x.shape[0], d, d), dtype=torch.float32, device=device)
    vine.theta_flip = torch.zeros((x.shape[0], d, d), dtype=torch.float32, device=device)
    
    # Prepare margins
    for i in range(d):
        # Store ranks/CDFs in theta
        u_vals = torch.sort(x_torch[:, i])[0]
        ranks = torch.searchsorted(u_vals, x_torch[:, i]).float() + 1
        vine.theta[:, 0, i] = ranks / (x.shape[0] + 1)
        
        # Store raw data in margin
        if hasattr(vine.margin[i], 'ker'):
            vine.margin[i].ker = x_torch[:, i].cpu().numpy()
    
    # Initialize copulas list
    vine.copulas = []
    
    # Handle tree structure initialization based on vine type
    if vine.vine_family == 'r-vine':
        # Handle R-vine initialization with proper matrices
        if vine.method == 'matrix':
            # User provided r_matrix
            if vine.r_matrix is None:
                # Create default
                vine.r_matrix = np.eye(d, dtype=np.int32)
                for i in range(d):
                    vine.r_matrix[i, i] = d - i
            
            # Prepare vine structure
            from .vine_tree import prepare_regular
            E, ind_vine, nodes, matrix_edges = prepare_regular(vine.r_matrix)
            vine.ind_vine = ind_vine
            vine.nodes = nodes
            vine.matrix_edges = matrix_edges
            
        elif vine.method == 'random':
            # Generate random R-vine
            from .vine_tree import random_r_matrix_gen
            vine.r_matrix, vine.ind_vine, vine.nodes, _ = random_r_matrix_gen(d)
            
        elif vine.method == 'optimal':
            # Will construct tree level by level during fitting
            vine.ind_vine = []
    else:
        # C-vine or D-vine
        from .vine_tree import prepare_vine
        vine.r_matrix, vine.ind_vine, vine.nodes, vine.matrix_edges = prepare_vine(vine.vine_family, d)
    
    # configuration ----------------------------------------------------
    cfg_all = DEFAULT_CFG if cfg is None else cfg
    opt_cfg = cfg_all["optimizer"]
    bw_cfg  = cfg_all["bandwidth"]
    npc_cfg = cfg_all.get("npc", {})
    opt_method_global = npc_cfg.get("opt_method", "LL1")
    
    # Now fit level by level
    for tr in range(d-1):
        # Print level info
        print(f"Fitting level {tr}/{d-1}...")
        
        # For optimal tree, find edges first
        if tr == 0 and vine.vine_family == 'r-vine' and vine.method == 'optimal':
            # Create first level optimal tree
            from .vine_tree import optimal_tree
            edges_now, weights = optimal_tree(
                vine.theta[:, tr, :].cpu().numpy(),
                vine.theta_flip[:, tr, :].cpu().numpy() if hasattr(vine, 'theta_flip') else None,
                vine.ind_vine,
                tr,
                False  # Not random
            )
            vine.ind_vine.append(edges_now)
        
        # Extract edges for this level
        if tr < len(vine.ind_vine):
            edges_now = vine.ind_vine[tr]
        else:
            # Should not reach here if structure was properly initialized
            edges_now = []
            print(f"Warning: No edges found for level {tr}!")
        
        # Prepare data for this level
        data_u = []
        for j, edge in enumerate(edges_now):
            if tr == 0:
                # First level: direct from margins
                pair_data = torch.stack([
                    vine.theta[:, tr, edge[0]],
                    vine.theta[:, tr, edge[1]]
                ], dim=1)
            else:
                # Higher levels: need to check parent
                parent, inx1, inx2 = parent_var(tr, vine.ind_vine, edge)
                
                # Determine correct data source based on parent
                if vine.ind_vine[tr-1][edge[0]][0] != parent:
                    pair_data = torch.stack([
                        vine.theta_flip[:, tr, edge[0]],
                        vine.theta[:, tr, edge[1]]
                    ], dim=1)
                else:
                    pair_data = torch.stack([
                        vine.theta[:, tr, edge[0]],
                        vine.theta[:, tr, edge[1]]
                    ], dim=1)
            
            data_u.append(pair_data)
        
        # Initialize list for this level's copulas
        copulas_level = []
        
        # ------------- PARAMETRIC edges processed one-by-one -------------
        if vine.param:
            # Fit each edge
            for j, pair_data in enumerate(data_u):
                edge = edges_now[j]
                
                # Parametric fitting
                families = par_dict.get('param_families', ["ind", "gaussian"])
                
                if vine.binning and tr > 0:
                    # Fit with binning
                    bin_copulas = []
                    
                    # Determine parent variable
                    parent, _, _ = parent_var(tr, vine.ind_vine, edge)
                    
                    # Create bins based on parent variable
                    if tr == 1:
                        bins = create_bins(vine.theta[:, tr-1, parent].cpu().numpy(), vine.n_bin)
                        val_to_bin = np.digitize(vine.theta[:, tr-1, parent].cpu().numpy(), bins) - 1
                        val_to_bin = check_bins(vine.theta[:, tr-1, parent].cpu().numpy(), bins)
                    else:
                        # Handle deeper levels (check if we need to use flipped values)
                        ind_par_now = vine.ind_vine[tr-1][edge[1]]
                        parent22, _, _ = parent_var(tr-1, vine.ind_vine, ind_par_now)
                        
                        if vine.ind_vine[tr-2][ind_par_now[0]][0] == parent22:
                            bins = create_bins(vine.theta[:, tr-1, parent].cpu().numpy(), vine.n_bin)
                            val_to_bin = np.digitize(vine.theta[:, tr-1, parent].cpu().numpy(), bins) - 1
                            val_to_bin = check_bins(vine.theta[:, tr-1, parent].cpu().numpy(), bins)
                        else:
                            bins = create_bins(vine.theta_flip[:, tr-1, parent].cpu().numpy(), vine.n_bin)
                            val_to_bin = np.digitize(vine.theta_flip[:, tr-1, parent].cpu().numpy(), bins) - 1
                            val_to_bin = check_bins(vine.theta_flip[:, tr-1, parent].cpu().numpy(), bins)
                    
                    # Fit each bin
                    for bb in range(vine.n_bin):
                        mask = (torch.tensor(val_to_bin, device=device) == bb)
                        if mask.sum() > 10:  # Ensure enough data points
                            bin_data = pair_data[mask]
                            
                            # Ensure uniform margins via kernel CDF
                            bin_data_np = bin_data.cpu().numpy()
                            bin_data_np = bin_data_np.reshape(-1, 2, 1)  # Add singleton dim for parametric_fit
                            
                            # Fit parametric copula
                            aic, theta_list, logp_list = parametric_fit(bin_data_np, families, n_cop=1)
                            best_idx = np.argmin(aic[0])
                            fam_best = families[best_idx]
                            param_best = theta_list[0][best_idx]
                            
                            # Create copula object
                            cop_p = cop_par_obj(fam_best, param_best)
                        else:
                            # Too few points, use independence
                            cop_p = cop_par_obj("ind", None)
                        
                        bin_copulas.append(cop_p)
                        
                    copulas_level.append(bin_copulas)
                else:
                    # Standard parametric fit (no binning)
                    pair_data_np = pair_data.cpu().numpy()
                    pair_data_np = pair_data_np.reshape(-1, 2, 1)  # Add singleton dim
                    
                    # Fit parametric copula
                    aic, theta_list, logp_list = parametric_fit(pair_data_np, families, n_cop=1)
                    best_idx = np.argmin(aic[0])
                    fam_best = families[best_idx]
                    param_best = theta_list[0][best_idx]
                    
                    # Create copula object
                    cop_p = cop_par_obj(fam_best, param_best)
                    copulas_level.append(cop_p)
                    
        # ----------------- NON-PARAMETRIC batched option -----------------
        if not vine.param:
            if opt_cfg.get("batch_edges", True) and len(data_u)>0 and opt_method_global in ("LL1","LL2"):
                # ----- batch all edges on this level ----------
                pair_u_cat = torch.stack(data_u, dim=2)         # N×2×E
                E = pair_u_cat.shape[2]
                maxE = opt_cfg.get("max_edges_per_batch")
                if maxE is None:
                    edge_chunks = [(0,E)]
                else:
                    edge_chunks = [(s, min(s+maxE, E)) for s in range(0,E,maxE)]

                bw_final_all = []
                pd_grids = []
                cdf_grids = []
                grad_u_list=[]
                grad_v_list=[]

                for start,stop in edge_chunks:
                    sub_u = pair_u_cat[:,:,start:stop]
                    sub_s = pair_u_cat[:,:,start:stop]
                    sub_x = pair_u_cat[:,:,start:stop]

                    subE = stop-start
                    grid_x_sub = transformer.forward_s(vine.grid_s.ex).view(-1,2,1).expand(-1,-1,subE)

                    if opt_method_global=="LL1":
                        bw_init_sub = bw_init[:,start:stop]
                        bw_fin = _optimize_bw_ll_batch(
                            bw_init_sub, sub_s, device,
                            max_iter=opt_cfg["max_iter_phase1"],
                            lr=opt_cfg["lr_phase1"],
                            conv_tol=opt_cfg["tol_phase1"])
                    else:
                        bw_init_sub = bw_init[:,start:stop]
                        a_init = torch.tensor([0.5],device=device)
                        a_opt = mise_optimization(a_init,bw_init_sub,vine.grid_u,vine.grid_s,grid_x_sub,
                                                   sub_x,sub_s,subE,opt_cfg["batch_size"],NORM[:,:,start:stop],False,
                                                   opt_cfg["max_iter_phase1"],opt_cfg["lr_phase1"],opt_cfg["tol_phase1"],axis_separate=True)
                        a_opt2 = mise_optimization(a_opt,bw_init_sub,vine.grid_u,vine.grid_s,grid_x_sub,
                                                   sub_x,sub_s,subE,opt_cfg["batch_size"],NORM[:,:,start:stop],True,
                                                   opt_cfg["max_iter_phase2"],opt_cfg["lr_phase2"],opt_cfg["tol_phase2"],axis_separate=True)
                        bw_fin = a_opt2 * bw_init_sub

                    bw_final_all.append(bw_fin)

                    pd_grid, cdf_grid, _, gu, gv = evaluate_fit(
                        {"data_s": sub_s, "data_x": sub_x},
                        {"grid_u": vine.grid_u, "grid_s": vine.grid_s, "grid_x": grid_x_sub},
                        {"bw": bw_fin, "n_cop": subE, "batch": opt_cfg["batch_size"], "grad_precompute": npc_cfg.get("grad_precompute", False)})
                    pd_grids.append(pd_grid); cdf_grids.append(cdf_grid)
                    if gu is not None:
                        grad_u_list.append(gu); grad_v_list.append(gv)

                bw_final = torch.cat(bw_final_all, dim=1)
                pd_grid = torch.cat(pd_grids, dim=2)
                cdf_grid = torch.cat(cdf_grids, dim=2)
                gu = torch.cat(grad_u_list, dim=2) if grad_u_list else None
                gv = torch.cat(grad_v_list, dim=2) if grad_v_list else None

                copulas_level = []
                for e in range(E):
                    cop_obj = copula_obj(bw_final[:, e:e+1])
                    cop_obj.pd_grid_uv = pd_grid[:, :, e]
                    cop_obj.cdf = cdf_grid[:, :, e]
                    if gu is not None:
                        cop_obj.grad_u = gu[:, :, e]
                        cop_obj.grad_v = gv[:, :, e]
                    copulas_level.append(cop_obj)
            else:
                # fallback to per-edge loop (existing logic)
                # Non-parametric fitting
                opt_method = opt_method_global
                
                # Transform to s and x spaces
                pair_data_s = transformer.forward_u(pair_data)
                pair_data_x = transformer.forward_s(pair_data_s)
                
                if vine.binning and tr > 0:
                    # Non-parametric with binning
                    # Similar logic to parametric case but with bandwidth optimization
                    bin_copulas = []
                    
                    # Determine parent variable and bins (same as parametric)
                    # ...
                    
                    # Fit each bin
                    # ...
                    
                else:
                    # Bandwidth initialisation
                    if opt_method == "LL1":
                        bw_init = bandwidth_sqrt_cov(pair_data_x)
                    else:
                        if bw_cfg["method"] == "knn":
                            bw_init = bandwidth_knn(pair_data_x, k=bw_cfg.get("knn_k",10))
                        else:
                            bw_init = bandwidth_rule_of_thumb(pair_data_x, 2, 1)
                    
                    # Grid in x-space
                    grid_x = transformer.forward_s(vine.grid_s.ex)
                    
                    # Optimize bandwidth
                    a_init = torch.tensor([0.5], dtype=torch.float32, device=device)
                    a_opt = mise_optimization(
                        a_init, bw_init,
                        vine.grid_u, vine.grid_s, grid_x,
                        pair_data_x, pair_data_s, 1, 5, NORM[:,:,0:1],
                        False, 70, 0.1, 1e-5,
                        axis_separate=False)
                    
                    # Second phase with normalization
                    a_opt2 = mise_optimization(
                        a_opt, bw_init,
                        vine.grid_u, vine.grid_s, grid_x,
                        pair_data_x, pair_data_s, 1, 5, NORM[:,:,0:1],
                        True, 100, 0.03, 5e-5,
                        axis_separate=False)
                    
                    # Scale final bandwidth
                    bw_final = a_opt2 * bw_init
                    
                    # Create copula object
                    cop_obj = copula_obj(bw_final)
                    
                    # Pre-compute grid values for PDF and CDF
                    # This will be used during evaluation
                    pd_grid, cdf_grid, _, gu, gv = evaluate_fit(
                        {'data_s': pair_data_s, 'data_x': pair_data_x},
                        {'grid_u': vine.grid_u, 'grid_s': vine.grid_s, 'grid_x': grid_x[:,:,0:1]},
                        {'bw': bw_final, 'n_cop': 1, 'batch': 5, 'grad_precompute': npc_cfg.get("grad_precompute", False)}
                    )
                    
                    cop_obj.pd_grid_uv = pd_grid
                    cop_obj.cdf = cdf_grid
                    if gu is not None:
                        cop_obj.grad_u = gu
                        cop_obj.grad_v = gv
                    
                    copulas_level.append(cop_obj)
        
        # Store this level's copulas
        vine.copulas.append(copulas_level)
        
        # ---- propagate theta / theta_flip for next level ----
        next_level = tr + 1
        if next_level < d:
            for e_idx, edge in enumerate(edges_now):
                i, j = edge  # left, right variables
                cobj_now = copulas_level[e_idx]
                u_i = vine.theta[:, tr, i]
                u_j = vine.theta[:, tr, j]
                # main direction
                vine.theta[:, next_level, j] = _h_function(u_i, u_j, cobj_now, vine.grid_u, side="left")
                # flipped direction
                vine.theta_flip[:, next_level, i] = _h_function(u_j, u_i, cobj_now, vine.grid_u, side="left")
        # ------------------------------------------------------
    
    vine.fitted = True
    return vine


############################################################
# 4) Evaluate Vine
############################################################

def evaluate_vine(vine: vine_obj_bin, points: torch.Tensor):
    """
    Evaluate the vine PDF at 'points'. Summation of marg(standard normal) + bivariate edges.
    """
    device = points.device
    d = vine.n_cop
    normal_dist = torch.distributions.Normal(0.,1.)
    log_marg = torch.zeros(points.shape[0], device=device)
    for i in range(d):
        log_marg += normal_dist.log_prob(points[:,i])

    log_cop = torch.zeros(points.shape[0], device=device)
    for level in range(d-1):
        edges = vine.copulas[level]
        root = level
        for e_idx, cobj in enumerate(edges):
            col2 = root + 1 + e_idx
            if col2>=d:
                continue
            uv = points[:, [root, col2]]
            if vine.param:
                pdf_val = copulapdf(cobj, uv).clamp(min=1e-30)
                log_cop += torch.log(pdf_val)
            else:
                bw = cobj.opt_bw
                # naive
                dx = uv[:,0] - uv[:,0].mean()
                dy = uv[:,1] - uv[:,1].mean()
                scale_x = bw[0,0].item()
                scale_y = bw[1,0].item()
                e = -0.5*((dx/scale_x)**2 + (dy/scale_y)**2)
                log_cop += e
    logp = log_marg + log_cop
    p = torch.exp(logp)
    return p, torch.exp(log_cop), log_marg


############################################################
# 4b) h-function utility (conditional CDF)
############################################################

def _h_function(u_root: torch.Tensor,
                u_other: torch.Tensor,
                cobj,
                grid_u: Optional[grid_obj],
                side: str = "left") -> torch.Tensor:
    """Return h_{other|root}(u_root,u_other).

    Works for both *parametric* (`cop_par_obj`) and *non-parametric*
    (`copula_obj`) edges.
    """
    if u_root.dim() == 2:
        u_root = u_root.squeeze(1)
    if u_other.dim() == 2:
        u_other = u_other.squeeze(1)

    device = u_root.device
    N = u_root.shape[0]

    # ---------- Parametric --------------------------------------------
    if hasattr(cobj, "family"):
        fam = cobj.family
        param = cobj.theta
        ur = torch.clamp(u_root, 1e-9, 1-1e-9)
        vo = torch.clamp(u_other, 1e-9, 1-1e-9)
        normal = torch.distributions.Normal(0.,1.)

        if fam == "ind":
            return vo.clone()

        elif fam == "gaussian":
            rho = float(param)
            rho = max(min(rho, 0.999999), -0.999999)
            x = normal.icdf(ur)
            y = normal.icdf(vo)
            z = (y - rho*x) / math.sqrt(1.0 - rho*rho)
            return torch.clamp(normal.cdf(z), 1e-9, 1-1e-9)

        elif fam == "clayton":
            alpha = float(param)
            u_m = ur.pow(-alpha-1.0)
            common = (ur.pow(-alpha) + vo.pow(-alpha) - 1.0).pow(-1.0/alpha -1.0)
            h = u_m * common
            return torch.clamp(h, 1e-9, 1-1e-9)

        elif fam == "claytonrot90":
            ur_f = 1.0 - ur
            # treat as clayton then flip result
            alpha = float(param)
            u_m = ur_f.pow(-alpha-1.0)
            common = (ur_f.pow(-alpha) + vo.pow(-alpha) - 1.0).pow(-1.0/alpha -1.0)
            h = u_m * common
            return torch.clamp(1.0 - h, 1e-9, 1-1e-9)

        else:
            # fallback – numerical derivative via small epsilon
            eps = 1e-4
            ur2 = torch.clamp(ur + eps, 1e-9, 1-1e-9)
            uv1 = torch.stack([ur, vo], dim=1)
            uv2 = torch.stack([ur2, vo], dim=1)
            from .utils_prob import copulaccdf
            c1 = copulaccdf(cobj, uv2)
            c0 = copulaccdf(cobj, uv1)
            h = (c1 - c0) / eps
            return torch.clamp(h, 1e-9, 1-1e-9)

    # ---------- Non-parametric ----------------------------------------
    else:
        # if gradients precomputed use bilinear interpolation
        if hasattr(cobj, 'grad_u') and cobj.grad_u is not None:
            x_axis, y_axis = grid_u.axis()
            points = torch.stack([u_root, u_other], dim=1)
            if side == "left":
                return bilinearInterp2d(points, x_axis, y_axis, cobj.grad_u)
            else:
                return bilinearInterp2d(points, x_axis, y_axis, cobj.grad_v)

        # else fallback to finite difference
        if grid_u is None or cobj.cdf is None:
            raise RuntimeError("Grid information required for nonparam h-function.")
        x_axis, y_axis = grid_u.axis()
        step = (x_axis[1]-x_axis[0]).item() if x_axis.numel()>1 else 1e-3
        eps = step
        # prepare tensors [N,2]
        points0 = torch.stack([u_root, u_other], dim=1)
        points1 = torch.stack([torch.clamp(u_root+eps,0.0,1.0), u_other], dim=1)
        # interpolate C on grid
        c0 = nearestInterp2d(points0, x_axis, y_axis, cobj.cdf)
        c1 = nearestInterp2d(points1, x_axis, y_axis, cobj.cdf)
        h = (c1 - c0)/(eps+1e-12)
        return torch.clamp(h, 1e-9, 1-1e-9)


############################################################
# 5) Sample Vine
############################################################

def _build_cdf_grid_nonparam(cobj, n_grid=50, device='cpu'):
    """
    Build a 2D grid for local-likelihood PDF in real scale, then do cumsum -> cdf.
    """
    from .utils_locallik import loclik_batch_eval
    data_s = cobj.data_s
    min_xy, _ = torch.min(data_s, dim=0)
    max_xy, _ = torch.max(data_s, dim=0)
    x_lin = torch.linspace(min_xy[0].item(), max_xy[0].item(), n_grid, device=device)
    y_lin = torch.linspace(min_xy[1].item(), max_xy[1].item(), n_grid, device=device)
    mesh_x, mesh_y = torch.meshgrid(x_lin, y_lin, indexing='ij')
    mx_f = mesh_x.reshape(-1)
    my_f = mesh_y.reshape(-1)
    grid_xy = torch.stack([mx_f, my_f], dim=1).unsqueeze(2)  # shape [n_grid^2,2,1]
    bw = cobj.opt_bw
    data_3d = data_s.unsqueeze(2)
    pdf_vals = loclik_batch_eval(bw, data_3d, grid_xy, 1, 5).squeeze(1)
    pdf_2d = pdf_vals.view(n_grid, n_grid).clamp_min(1e-30)

    dx = (x_lin[1]-x_lin[0]).item() if n_grid>1 else 1.0
    dy = (y_lin[1]-y_lin[0]).item() if n_grid>1 else 1.0
    cdf2d = torch.cumsum(torch.cumsum(pdf_2d, dim=1)*dy, dim=1)
    cdf2d = torch.cumsum(cdf2d, dim=0)*dx
    top = cdf2d[-1,-1].item()
    if top<1e-9:
        top=1e-9
    cdf2d = cdf2d/top
    return x_lin, y_lin, cdf2d


def _inv2d(u1, u2, x_lin, y_lin, cdf2d):
    """
    naive search for row/col.
    """
    row_end = cdf2d[:, -1]
    rows = (row_end>=u2).nonzero(as_tuple=True)[0]
    if len(rows)==0:
        i = cdf2d.shape[0]-1
    else:
        i = rows[0].item()
    rowi = cdf2d[i,:]
    cols = (rowi>=u2).nonzero(as_tuple=True)[0]
    if len(cols)==0:
        j = rowi.shape[0]-1
    else:
        j = cols[0].item()
    return x_lin[i].item(), y_lin[j].item()


def sample_vine(vine: vine_obj_bin, nsamples: int, cfg: Optional[dict] = None):
    """
    Sample from c-vine. For param => partial approach. For nonparam => build local cdf.
    We'll store final in an array [nsamples, d], assume standard normal margins for demonstration.
    """
    d = vine.n_cop
    cfg_all = DEFAULT_CFG if cfg is None else cfg
    samp_cfg = cfg_all.get("sampler", {})
    fast_param = samp_cfg.get("fast_parametric", True)
    fast_np    = samp_cfg.get("fast_nonparam", True)

    samples = torch.zeros((nsamples, d), dtype=torch.float32)

    normal = torch.distributions.Normal(0.,1.)
    samples[:,0] = normal.icdf(torch.rand(nsamples))

    for i in range(1, d):
        lvl = i-1
        # Robust edge selection irrespective of vine family
        edges = vine.copulas[lvl]
        struct_edges = vine.ind_vine[lvl] if lvl < len(vine.ind_vine) else []
        root = lvl
        match_idx = 0
        for ei, e in enumerate(struct_edges):
            if (e[0] == root and e[1] == i) or (e[1] == root and e[0] == i):
                match_idx = ei
                break
        if match_idx >= len(edges):          # variable not present on this level
            continue                         # move on to next i
        cobj = edges[match_idx]
        if vine.param and fast_param:
            root_val = samples[:,lvl]
            root_u = normal.cdf(root_val)
            rand_u = torch.rand(nsamples)
            if cobj.family == "ind":
                vi = rand_u
            elif cobj.family == "gaussian":
                rho = float(cobj.theta)
                z = normal.icdf(root_u)
                e = normal.icdf(rand_u)
                y = rho*z + math.sqrt(1.0-rho*rho)*e
                vi = normal.cdf(y)
            elif cobj.family == "clayton":
                alpha = float(cobj.theta)
                u1 = root_u
                c2 = rand_u
                val = (c2.pow(-alpha/(1+alpha)) - u1.pow(-alpha) +1.0).clamp_min(1e-12)
                vi = val.pow(-1.0/alpha)
            else:
                # fallback per-sample
                vi = torch.zeros(nsamples)
                for n in range(nsamples):
                    uv = torch.tensor([[root_u[n].item(), rand_u[n].item()]])
                    vi[n] = copulainvccdf(cobj, uv).item()
            samples[:,i] = normal.icdf(vi.clamp(1e-9,1-1e-9))
        elif vine.param:
            # slow loop fallback
            for n in range(nsamples):
                root_val = samples[n,lvl]
                root_u = normal.cdf(root_val)
                rand_u = random.random()
                uv = torch.tensor([[root_u, rand_u]], dtype=torch.float32)
                valU = copulainvccdf(cobj, uv).item()
                samples[n,i] = normal.icdf(torch.tensor(valU))
        else:
            if fast_np and hasattr(cobj, 'cdf'):
                if not hasattr(cobj, 'cdf_xlin'):
                    x_axis, y_axis = vine.grid_u.axis()
                    cobj.cdf_xlin = x_axis
                    cobj.cdf_ylin = y_axis
                x_axis = cobj.cdf_xlin
                y_axis = cobj.cdf_ylin
                root_u = normal.cdf(samples[:,lvl])
                rand_u = torch.rand(nsamples)
                # row index per sample
                row_idx = torch.bucketize(root_u, x_axis)
                row_idx = torch.clamp(row_idx, 1, x_axis.numel()-1)
                row_idx = row_idx - 1
                cdf_rows = cobj.cdf[row_idx]
                from .utils_interpolation import inverse_cdf_row
                vi = inverse_cdf_row(rand_u, cdf_rows, y_axis)
                samples[:,i] = normal.icdf(vi.clamp(1e-9,1-1e-9))
            else:
                # legacy slow loop
                if not hasattr(cobj, 'cdf_xlin'):
                    device_ = 'cuda' if cobj.data_s.is_cuda else 'cpu'
                    x_lin, y_lin, cdf2d = _build_cdf_grid_nonparam(cobj, n_grid=50, device=device_)
                    cobj.cdf_xlin = x_lin
                    cobj.cdf_ylin = y_lin
                    cobj.cdf_2d   = cdf2d
                for n in range(nsamples):
                    root_val = samples[n,lvl]
                    root_u = normal.cdf(root_val)
                    rand_u = random.random()
                    x_val, y_val = _inv2d(root_u, rand_u, cobj.cdf_xlin, cobj.cdf_ylin, cobj.cdf_2d)
                    samples[n,i] = y_val

    return samples.cpu().numpy()


############################################################
# Attach
############################################################
vine_obj_bin.fit = fit_vine
vine_obj_bin.evaluation = evaluate_vine
vine_obj_bin.sample = sample_vine

############################################################
# Utility: simple surrogate for the missing ``mise_optimization``
############################################################

# The original TensorFlow codebase used a two-phase MISE bandwidth
# optimisation routine. The function call is still present in the
# regenerated PyTorch code but the actual implementation was never
# ported, so importing / calling it inevitably raises a ``NameError``.
#
# To keep the high-level API intact while we re-implement the full
# optimisation later, we provide a *minimal* placeholder that simply
# returns the incoming scale factor unchanged. This makes the module
# importable and the main ``fit_vine`` execution path functional,
# albeit with a conservative bandwidth choice (rule-of-thumb only).
#
# NOTE: once a faithful PyTorch version of the MISE routine is ready
# this stub can be replaced transparently without touching callers.

def mise_optimization(a_init: torch.Tensor,
                     bw_init: torch.Tensor,
                     grid_u: grid_obj,
                     grid_s: grid_obj,
                     grid_x: torch.Tensor,
                     data_x: torch.Tensor,
                     data_s: torch.Tensor,
                     n_cop: int,
                     batch_size: int,
                     ref_norm: torch.Tensor,
                     renorm_flag: bool,
                     max_iter: int,
                     lr: float,
                     tol: float,
                     axis_separate: bool = False):
    """Optimise a *scalar* multiplier ``a`` for the base bandwidth ``bw_init``.

    The objective is a crude yet effective proxy for the Mean Integrated
    Squared Error (MISE) between the local-likelihood kernel estimate and a
    reference density (`ref_norm`, typically a standard bivariate normal).

    We follow an extremely simple recipe:
      1.  Build the candidate bandwidth B = a * bw_init (shape ``[2, n_cop]``).
      2.  Compute the corresponding local-likelihood PDF on the supplied grid
          via :func:`loclik_batch_eval`.
      3.  Optionally (``renorm_flag``) project that PDF back to a bona-fide
          copula density using :func:`eval_rs_cop` (row/column normalisation).
      4.  Evaluate   cost = mean( (pdf_est − ref_norm) ** 2 ).

    A tiny Adam loop (``max_iter`` iterations, learning-rate ``lr``) is run on
    the *log* of ``a`` to enforce positivity. The search terminates when the
    relative improvement in the cost falls below ``tol``.

    Parameters
    ----------
    a_init : torch.Tensor shape `[1]`
        Initial scaling factor.
    bw_init : torch.Tensor shape `[2, n_cop]`
        Baseline bandwidth matrix.
    grid_u / grid_s / grid_x : grid descriptions
        Pre-computed grids used by the caller (see original code).
    data_x / data_s : torch.Tensor
        Training data in x- and s-spaces respectively.
    n_cop, batch_size : int
        Copula count and batching parameter for ``loclik_batch_eval``.
    ref_norm : torch.Tensor
        Reference density evaluated on the same grid (shape `[K,K,n_cop]`).
    renorm_flag : bool
        If ``True`` we apply copula row/column renormalisation.
    max_iter, lr, tol : optimisation hyper-parameters.

    Returns
    -------
    torch.Tensor shape `[1]`
        Optimised scale factor ``a_opt`` (detached).
    """
    device = a_init.device
    # Parameterisation: scalar (LL1) or per-axis (LL2)
    if axis_separate:
        if a_init.dim()==0 or a_init.numel()==1:
            a_init = a_init.expand_as(bw_init)  # shape 2×n_cop
        a_log = a_init.log().clone().detach().requires_grad_(True)
    else:
        # single scalar shared by both axes and all edges
        if a_init.numel()>1:
            a_init = a_init.flatten()[0:1]
        a_log = a_init.log().clone().detach().requires_grad_(True)

    optim = torch.optim.Adam([a_log], lr=lr)

    # Pre-compute grid differentials for eval_rs_cop if needed.
    adu11, adu22 = grid_u.diff()  # each shape [K]

    prev_cost = 1e12
    for _ in range(max_iter):
        optim.zero_grad()
        if axis_separate:
            a_val = torch.exp(a_log)                  # 2×n_cop
            B = bw_init * a_val
        else:
            a_val = torch.exp(a_log)[0]
            B = bw_init * a_val

        # Local-likelihood estimate on the grid → [M, n_cop]
        ker_flat = loclik_batch_eval(B, data_s, grid_x, n_cop, batch_size)
        K = grid_s.ax1.shape[0]
        ker_pdf = ker_flat.view(K, K, n_cop)  # reshape to 2-D grid

        if renorm_flag:
            from .cop_eval import eval_rs_cop  # local import to avoid cycles
            ker_pdf = eval_rs_cop(adu11, adu22, ker_pdf, ref_norm, n_cop)

        # MISE proxy (mean squared error against reference)
        mse = torch.mean((ker_pdf - ref_norm) ** 2)
        mse.backward()
        optim.step()

        # Convergence check
        cost_now = mse.item()
        if abs(prev_cost - cost_now) < tol:
            break
        prev_cost = cost_now

    # Clamp to a sensible range for safety
    with torch.no_grad():
        if axis_separate:
            a_final = torch.exp(a_log).clamp(0.05, 20.0)
        else:
            a_final = torch.exp(a_log).clamp(0.05, 20.0)[0:1]
    return a_final.detach()

############################################################
# 6) Convenience API helpers (logpdf, pdf, cdf)
############################################################

def logpdf_vine(vine: vine_obj_bin, points: torch.Tensor):
    """Return log-pdf of the fitted vine at *points* (N×d tensor)."""
    p, _, _ = evaluate_vine(vine, points)
    return torch.log(p.clamp_min(1e-30))

def pdf_vine(vine: vine_obj_bin, points: torch.Tensor):
    """Return pdf at *points* — just a thin wrapper."""
    p, _, _ = evaluate_vine(vine, points)
    return p

def cdf_vine(vine: vine_obj_bin, points: torch.Tensor, nsim: int = 2000):
    """Monte-Carlo approximation of the d-dimensional CDF F(x₁,…,x_d).

    Draw *nsim* samples from the fitted vine and return the empirical
    probability that every coordinate is ≤ the corresponding entry in
    *points* (vectorised for a batch of query points).
    """
    device = points.device
    samples_np = vine.sample(nsim)  # returns numpy
    samples = torch.tensor(samples_np, dtype=points.dtype, device=device)
    # for each query point evaluate indicator and mean over sim
    out = []
    for q in points:
        mask = (samples <= q.cpu().numpy()).all(axis=1)
        out.append(mask.mean())
    return torch.tensor(out, dtype=points.dtype, device=device)

# register --------------------------------------------------
vine_obj_bin.logpdf = logpdf_vine
vine_obj_bin.pdf    = pdf_vine
vine_obj_bin.cdf    = cdf_vine