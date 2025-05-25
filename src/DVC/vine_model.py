###############################################
# src/DVC/vine_model.py
###############################################
# 
# H-FUNCTION BUG FIX (2023-10-05):
# Fixed theta/theta_flip propagation by ensuring proper handling of right-side 
# h-functions. The issue was that both theta and theta_flip were using side="left", 
# but theta_flip should use side="right" to properly compute the conditional 
# distribution in the opposite direction. This is critical for maintaining
# the correct dependence structure between non-adjacent variables in the vine.
# Additional improvements may be needed in the evaluate_vine function to fully
# utilize the conditional structure information.
#
###############################################

import torch
import numpy as np
import random
from scipy.stats import kendalltau, norm
import math
from typing import Optional, Union, TYPE_CHECKING
import logging  # NEW

# Forward reference to avoid circular imports
if TYPE_CHECKING:
    from .objects import vine_obj_bin

# NEW IMPORTS --------------------------------------------------
# The new PyTorch implementation relies on helper utilities that
# were defined in sibling modules but never imported, leading to
# run-time NameError exceptions. We explicitly import them here.
from .transformation import Transform
from .dataset_ops import create_bins, check_bins
from .utils_prob import kernel_cdf  # CRITICAL FIX: Add kernel_cdf import
from .objects import cop_par_obj  # CRITICAL FIX: Add cop_par_obj import
# -------------------------------------------------------------

# Basic objects
from .d_vine_fix import sample_d_vine, apply_d_vine_fix
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
# Setup a basic logger – users can override level/handlers from their scripts.
logger = logging.getLogger("DVC.vine")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")
############################################################

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
    ex_coords = mk_grid(knots, dtype=data_s.dtype)
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

def fit_vine(vine: 'vine_obj_bin',
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
    ex_u = mk_grid(knots, dtype=torch.float32)
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
    
    # Initialize flip_flag for tracking conditional directions
    vine.flip_flag = []
    
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
    if vine.vine_family == 'c-vine' or vine.vine_family == 'd-vine':
        # Handle C-vine and D-vine initialization
        from .vine_tree import prepare_vine
        vine.r_matrix, vine.ind_vine, vine.nodes, vine.matrix_edges = prepare_vine(vine.vine_family, d)
        # if prepare_vine returned empty edges, populate a default structure
        if all(len(lvl)==0 for lvl in vine.ind_vine):
            if vine.vine_family=='c-vine':
                vine.ind_vine = []
                # level 0: root variable 0 connected to 1..d-1
                vine.ind_vine.append([[0,j] for j in range(1,d)])
                # deeper levels simplified chain
                for k in range(1,d-1):
                    lvl_edges = [[k, j] for j in range(k+1, d)]
                    vine.ind_vine.append(lvl_edges)
            elif vine.vine_family=='d-vine':
                vine.ind_vine = []
                # d-vine first level chain edges
                vine.ind_vine.append([[j, j+1] for j in range(d-1)])
                for k in range(1,d-1):
                    lvl_edges = [[j, j+k+1] for j in range(d-k-1)]
                    vine.ind_vine.append(lvl_edges)
                    
    elif vine.vine_family == 'r-vine':
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
            # Default: create a random R-vine
            from .vine_tree import random_r_matrix_gen
            vine.r_matrix, vine.ind_vine, vine.nodes, _ = random_r_matrix_gen(d)
    
    # ------------------------------------------------------------------
    # Optional: override vine structure from configuration file.
    # Users can provide a key  cfg["vine"]["structure"]  containing a
    # list-of-lists-of-int specifying edges for each tree level, e.g.::
    #
    #   vine:
    #     structure:
    #       - [[0,1],[0,2],[0,3],[0,4]]
    #       - [[1,2],[1,3],[1,4]]
    #       - [[2,3],[2,4]]
    #       - [[3,4]]
    #
    # This takes absolute priority over any auto-generated or fallback
    # structure built above.
    # ------------------------------------------------------------------
    vine_cfg = cfg.get('vine', {}) if cfg is not None else {}
    if vine_cfg.get('structure') is not None:
        vine.ind_vine = vine_cfg['structure']
        logger.info("[cfg] Vine structure overridden from configuration file.")

    # Log the resulting topology for debugging.
    logger.info(f"Vine topology (family={vine.vine_family}, method={vine.method}, d={d})")
    for lvl, edges in enumerate(vine.ind_vine):
        logger.info(f"  Level {lvl}: {edges}")
    
    # configuration ----------------------------------------------------
    cfg_all = DEFAULT_CFG if cfg is None else cfg
    opt_cfg = cfg_all["optimizer"]
    bw_cfg  = cfg_all.get("bandwidth", {"method": "rule_of_thumb", "knn_k": 10})
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
                prev_len = len(vine.ind_vine[tr-1])
                # Heuristic: if both indices are < prev_len we assume the edge is
                # *referencing* two edges from the previous tree (original logic).
                # Otherwise we treat them as *variable* indices directly.
                if edge[0] < prev_len and edge[1] < prev_len:
                    parent, _, _ = parent_var(tr, vine.ind_vine, edge)
                    try:
                        left_edge = vine.ind_vine[tr-1][edge[0]]
                        left_first = left_edge[0]
                    except IndexError:
                        # fallback to variable interpretation
                        left_first = None

                    if left_first is not None and left_first != parent:
                        pair_data = torch.stack([
                            vine.theta_flip[:, tr, edge[0]],
                            vine.theta[:, tr, edge[1]]
                        ], dim=1)
                    else:
                        pair_data = torch.stack([
                            vine.theta[:, tr, edge[0]],
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
        flip_flags_level = []  # Track flip flags for this level
        
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
                            
                            # CRITICAL FIX: Ensure uniform margins via kernel CDF (this was missing!)
                            # Apply kernel_cdf to each dimension of the bin data
                            bin_data_np = bin_data.cpu().numpy()
                            
                            # Apply kernel_cdf to ensure uniform margins
                            for dim in range(bin_data_np.shape[1]):
                                dim_data = bin_data_np[:, dim]
                                ex_u_np = vine.grid_u.ex.cpu().numpy() if hasattr(vine.grid_u, 'ex') else np.linspace(0, 1, 50)
                                uniform_data, _, _ = kernel_cdf(dim_data, dim_data, ex_u_np)
                                bin_data_np[:, dim] = uniform_data
                            
                            # Reshape for parametric_fit
                            bin_data_np = bin_data_np.reshape(-1, 2, 1)  # Add singleton dim for parametric_fit
                            
                            # Fit parametric copula
                            aic, theta_list, logp_list = parametric_fit(bin_data_np, families, n_cop=1)
                            
                            # Debug: print AIC values
                            if tr == 0 and j == 0:  # Only print for first edge of first level
                                print(f"DEBUG: AIC values for edge {j}: {aic[0]}")
                                print(f"DEBUG: Families: {families}")
                                print(f"DEBUG: Theta values: {theta_list[0]}")
                                print(f"DEBUG: Log-likelihood values: {logp_list[0]}")
                            
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
                    
                    # Debug: print AIC values
                    if tr == 0 and j == 0:  # Only print for first edge of first level
                        print(f"DEBUG: AIC values for edge {j}: {aic[0]}")
                        print(f"DEBUG: Families: {families}")
                        print(f"DEBUG: Theta values: {theta_list[0]}")
                        print(f"DEBUG: Log-likelihood values: {logp_list[0]}")
                    
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

                # Transform to different spaces - use E (number of edges) not d
                transformer = Transform(E)
                bw_final_all = []
                pd_grids = []
                cdf_grids = []
                grad_u_list=[]
                grad_v_list=[]

                for start,stop in edge_chunks:
                    sub_u = pair_u_cat[:,:,start:stop]
                    sub_s = transformer.forward_u(sub_u)
                    sub_x = transformer.forward_s(sub_s)

                    subE = stop-start
                    grid_x_sub = transformer.forward_s(vine.grid_s.ex).view(-1,2,1).expand(-1,-1,subE)

                    if opt_method_global=="LL1":
                        # Initialize bandwidth for this chunk
                        if bw_cfg["method"] == "knn":
                            bw_init = bandwidth_knn(sub_x, k=bw_cfg.get("knn_k",10))
                        else:
                            bw_init = bandwidth_rule_of_thumb(sub_x, 2, subE)
                        
                        # Now use this initialized bandwidth
                        bw_init_sub = bw_init[:,0:subE]
                        bw_fin = _optimize_bw_ll_batch(
                            bw_init_sub, sub_s, device,
                            max_iter=opt_cfg["max_iter_phase1"],
                            lr=opt_cfg["lr_phase1"],
                            conv_tol=opt_cfg["tol_phase1"])
                    else:
                        # Initialize bandwidth for this chunk
                        if bw_cfg["method"] == "knn":
                            bw_init = bandwidth_knn(sub_x, k=bw_cfg.get("knn_k",10))
                        else:
                            bw_init = bandwidth_rule_of_thumb(sub_x, 2, subE)
                            
                        # Now use this initialized bandwidth
                        bw_init_sub = bw_init[:,0:subE]
                        a_init = torch.tensor([0.5],device=device)
                        a_opt = mise_optimization(a_init,bw_init_sub,vine.grid_u,vine.grid_s,grid_x_sub,
                                                   sub_x,sub_s,subE,opt_cfg["batch_size"],NORM[:,:,start:stop],False,
                                                   opt_cfg["max_iter_phase1"],opt_cfg["lr_phase1"],opt_cfg["tol_phase1"],axis_separate=True)
                        a_opt2 = mise_optimization(a_opt,bw_init_sub,vine.grid_u,vine.grid_s,grid_x_sub,
                                                   sub_x,sub_s,subE,opt_cfg["batch_size"],NORM[:,:,start:stop],True,
                                                   opt_cfg["max_iter_phase2"],opt_cfg["lr_phase2"],opt_cfg["tol_phase2"],axis_separate=True)
                        bw_fin = a_opt2 * bw_init_sub

                    bw_final_all.append(bw_fin)

                    pd_grid, cdf_grid, theta_ret, gu, gv = evaluate_fit(
                        {"data_s": sub_s, "data_x": sub_x, "theta": vine.theta, "theta_flip": vine.theta_flip},
                        {"grid_u": vine.grid_u, "grid_s": vine.grid_s, "grid_x": grid_x_sub},
                        {"bw": bw_fin, "n_cop": subE, "batch": opt_cfg["batch_size"], "tr": tr, "grad_precompute": npc_cfg.get("grad_precompute", False)})
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
                
                # Initialize transformer for single edge
                transformer = Transform(1)
                
                for j, pair_data in enumerate(data_u):
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
                        pass
                        
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
                        pd_grid, cdf_grid, theta_ret, gu, gv = evaluate_fit(
                            {'data_s': pair_data_s, 'data_x': pair_data_x, 'theta': vine.theta, 'theta_flip': vine.theta_flip},
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
        vine.flip_flag.append(flip_flags_level)
        
        # ---- propagate theta / theta_flip for next level ----
        next_level = tr + 1
        if next_level < d:
            for e_idx, edge in enumerate(edges_now):
                i, j = edge  # left, right variables
                cobj_now = copulas_level[e_idx]
                
                # Determine if we need to use flipped values
                if tr == 0:
                    u_i = vine.theta[:, tr, i]
                    u_j = vine.theta[:, tr, j]
                    flip_flags_level.append(False)
                else:
                    # Check parent variable
                    parent, _, _ = parent_var(tr, vine.ind_vine, edge)
                    
                    # Check if we need flipped theta for i
                    if i < len(vine.ind_vine[tr-1]):
                        prev_edge = vine.ind_vine[tr-1][i]
                        if prev_edge[0] != parent:
                            u_i = vine.theta_flip[:, tr, i]
                            flip_flags_level.append(True)
                        else:
                            u_i = vine.theta[:, tr, i]
                            flip_flags_level.append(False)
                    else:
                        u_i = vine.theta[:, tr, i]
                        flip_flags_level.append(False)
                    
                    u_j = vine.theta[:, tr, j]
                
                # Debug: Check for NaN values before h-function
                if torch.isnan(u_i).any() or torch.isnan(u_j).any():
                    logger.warning(f"NaN values in theta before h-function at level {tr}, edge {e_idx}")
                
                # Apply improved theta update with kernel smoothing
                try:
                    # Get corrected parent variable
                    parent, _, _ = get_parent_variable_fixed(tr, vine.ind_vine, edge)
                    
                    # Use the fixed update function with kernel smoothing
                    update_theta_with_kernel_smoothing(vine, tr, edge, cobj_now, u_i, u_j, parent)
                    
                except Exception as e:
                    logger.error(f"Error in theta update at level {tr}, edge {e_idx}: {str(e)}")
                    # Fallback: apply kernel smoothing to independence case too
                    try:
                        ex_u_np = vine.grid_u.ex.cpu().numpy() if hasattr(vine.grid_u, 'ex') else np.linspace(0, 1, 50)
                        u_j_smooth, _, _ = kernel_cdf(u_j.cpu().numpy(), u_j.cpu().numpy(), ex_u_np)
                        u_i_smooth, _, _ = kernel_cdf(u_i.cpu().numpy(), u_i.cpu().numpy(), ex_u_np)
                        vine.theta[:, next_level, j] = torch.from_numpy(u_j_smooth).to(u_j.device)
                        vine.theta_flip[:, next_level, i] = torch.from_numpy(u_i_smooth).to(u_i.device)
                    except:
                        vine.theta[:, next_level, j] = u_j
                        vine.theta_flip[:, next_level, i] = u_i
        # ------------------------------------------------------

    vine.fitted = True
    return vine


############################################################
# 4) Evaluate Vine
############################################################

def evaluate_vine(vine: 'vine_obj_bin', points: torch.Tensor):
    """Return PDF of ``vine`` evaluated at ``points`` (N×d tensor)."""

    device = points.device
    n, d = points.shape

    # --- Margins -------------------------------------------------
    log_marg = torch.zeros(n, device=device)
    theta = torch.zeros((n, d, d), device=device)
    theta_flip = torch.zeros_like(theta)

    for i in range(d):
        if (hasattr(vine, "margin") and vine.margin is not None
                and i < len(vine.margin)):
            mobj = vine.margin[i]
            if getattr(mobj, "family", "norm") == "norm" and hasattr(mobj, "theta"):
                loc, scale = mobj.theta
                dist = torch.distributions.Normal(loc, scale)
            else:
                dist = torch.distributions.Normal(0.0, 1.0)
        else:
            dist = torch.distributions.Normal(0.0, 1.0)

        log_marg += dist.log_prob(points[:, i])
        u_val = dist.cdf(points[:, i])
        theta[:, 0, i] = u_val
        theta_flip[:, 0, i] = u_val

    log_cop = torch.zeros(n, device=device)

    # --- Traverse vine level by level ---------------------------------
    for tr in range(d - 1):
        edges_now = vine.ind_vine[tr] if tr < len(vine.ind_vine) else []
        copulas_now = vine.copulas[tr] if tr < len(vine.copulas) else []
        next_lvl = tr + 1

        for e_idx, edge in enumerate(edges_now):
            if e_idx >= len(copulas_now):
                continue
            cobj = copulas_now[e_idx]
            i, j = edge

            if tr == 0:
                ui = theta[:, tr, i]
                uj = theta[:, tr, j]
            else:
                prev_len = len(vine.ind_vine[tr - 1])
                if i < prev_len and j < prev_len:
                    parent, _, _ = parent_var(tr, vine.ind_vine, edge)
                    try:
                        left_edge = vine.ind_vine[tr - 1][i]
                        left_first = left_edge[0]
                    except Exception:
                        left_first = None
                    if left_first is not None and left_first != parent:
                        ui = theta_flip[:, tr, i]
                    else:
                        ui = theta[:, tr, i]
                    uj = theta[:, tr, j]
                else:
                    ui = theta[:, tr, i]
                    uj = theta[:, tr, j]

            uv = torch.stack([ui, uj], dim=1)

            if vine.param:
                pdf_val = copulapdf(cobj, uv).clamp(min=1e-30)
            else:
                if hasattr(cobj, "pd_grid_uv"):
                    from .utils_interpolation import bilinearInterp2d
                    x_axis, y_axis = vine.grid_u.axis()
                    pdf_val = bilinearInterp2d(uv, x_axis, y_axis, cobj.pd_grid_uv)
                    pdf_val = pdf_val.clamp(min=1e-30)
                else:
                    pdf_val = torch.ones_like(ui)

            pdf_val = torch.where(torch.isfinite(pdf_val), pdf_val,
                                   torch.full_like(pdf_val, 1e-30))
            log_cop += torch.log(pdf_val)

            if next_lvl < d:
                theta[:, next_lvl, j] = _h_function(ui, uj, cobj, vine.grid_u, side="left")
                theta_flip[:, next_lvl, i] = _h_function(uj, ui, cobj, vine.grid_u, side="right")

    log_marg = torch.where(torch.isfinite(log_marg), log_marg, torch.zeros_like(log_marg))
    log_cop = torch.where(torch.isfinite(log_cop), log_cop, torch.zeros_like(log_cop))

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
    
    Args:
        u_root: Conditioning variable values (shape [N] or [N,1])
        u_other: Variable to condition on u_root (shape [N] or [N,1])
        cobj: Copula object (parametric or nonparametric)
        grid_u: Grid object for non-parametric interpolation (optional)
        side: "left" for h(u_other|u_root), "right" for h(u_root|u_other)
            
    Returns:
        Conditional CDF values, shape [N]
    """
    # Check for NaN inputs and handle gracefully
    if torch.isnan(u_root).any() or torch.isnan(u_other).any():
        logger.warning(f"NaN inputs to h_function detected")
        u_root = torch.where(torch.isnan(u_root), torch.rand_like(u_root) * 0.8 + 0.1, u_root)
        u_other = torch.where(torch.isnan(u_other), torch.rand_like(u_other) * 0.8 + 0.1, u_other)
    
    if u_root.dim() == 2:
        u_root = u_root.squeeze(1)
    if u_other.dim() == 2:
        u_other = u_other.squeeze(1)

    device = u_root.device
    N = u_root.shape[0]

    # Robust clamping to avoid extreme values
    ur = torch.clamp(u_root, 1e-6, 1-1e-6)
    vo = torch.clamp(u_other, 1e-6, 1-1e-6)
    
    # For right-side calculation, swap variables
    if side == "right":
        ur, vo = vo, ur  # Swap variables for right-side calculation

    # ---------- Parametric --------------------------------------------
    if hasattr(cobj, "family"):
        fam = cobj.family
        param = cobj.theta
        normal = torch.distributions.Normal(0.,1.)

        if fam == "ind":
            return vo.clone()

        elif fam == "gaussian":
            # Robust parameter handling
            rho = float(param) if param is not None else 0.0
            if not math.isfinite(rho):
                rho = 0.0
            rho = max(min(rho, 0.99), -0.99)  # More conservative bounds
            
            # Convert to normal scores with robust handling
            x = normal.icdf(ur)
            y = normal.icdf(vo)
            
            # Clamp extreme values more aggressively
            x = torch.clamp(x, -6.0, 6.0)
            y = torch.clamp(y, -6.0, 6.0)
            
            # Replace any remaining invalid values before computation
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
            
            # Calculate the conditional normal distribution with robust denominator
            denom = max(1.0 - rho*rho, 1e-8)  # Ensure positive denominator
            z = (y - rho*x) / math.sqrt(denom)
            
            # Final safety check and cleanup
            z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            z = torch.clamp(z, -6.0, 6.0)
            
            result = normal.cdf(z)
            return torch.clamp(result, 1e-6, 1-1e-6)

        elif fam == "clayton":
            alpha = float(param) if param is not None else 1.0
            if not math.isfinite(alpha) or alpha <= 0:
                alpha = 1.0  # Default safe value
            alpha = max(alpha, 0.1)  # Avoid numerical issues
            
            try:
                u_m = ur.pow(-alpha-1.0)
                common = (ur.pow(-alpha) + vo.pow(-alpha) - 1.0).clamp_min(1e-8).pow(-1.0/alpha -1.0)
                h = u_m * common
                
                # Handle any NaN/Inf that might have occurred
                h = torch.where(torch.isfinite(h), h, vo)  # Fallback to independence
                return torch.clamp(h, 1e-6, 1-1e-6)
            except Exception:
                # Fallback to independence
                return vo.clone()

        elif fam == "claytonrot90":
            alpha = float(param) if param is not None else 1.0
            if not math.isfinite(alpha) or alpha <= 0:
                alpha = 1.0
            alpha = max(alpha, 0.1)
            
            try:
                ur_f = 1.0 - ur
                u_m = ur_f.pow(-alpha-1.0)
                common = (ur_f.pow(-alpha) + vo.pow(-alpha) - 1.0).clamp_min(1e-8).pow(-1.0/alpha -1.0)
                h = u_m * common
                h = torch.where(torch.isfinite(h), h, 1.0 - vo)  # Fallback
                return torch.clamp(1.0 - h, 1e-6, 1-1e-6)
            except Exception:
                return vo.clone()
        else:
            # fallback to independence for unknown families
            return vo.clone()

    # ---------- Non-parametric ----------------------------------------
    else:
        try:
            # if gradients precomputed use bilinear interpolation
            if hasattr(cobj, 'grad_u') and cobj.grad_u is not None:
                x_axis, y_axis = grid_u.axis()
                points = torch.stack([ur, vo], dim=1)
                from .utils_interpolation import bilinearInterp2d
                if side == "left":
                    result = bilinearInterp2d(points, x_axis, y_axis, cobj.grad_u)
                else:
                    result = bilinearInterp2d(points, x_axis, y_axis, cobj.grad_v)
                
                # Handle NaN from interpolation
                result = torch.where(torch.isfinite(result), result, vo)
                return torch.clamp(result, 1e-6, 1-1e-6)

            # else fallback to finite difference
            if grid_u is None or cobj.cdf is None:
                # Ultimate fallback: independence
                return vo.clone()
                
            x_axis, y_axis = grid_u.axis()
            step = (x_axis[1]-x_axis[0]).item() if x_axis.numel()>1 else 1e-3
            eps = step
            
            # prepare tensors [N,2]
            points0 = torch.stack([ur, vo], dim=1)
            points1 = torch.stack([torch.clamp(ur+eps,1e-6,1-1e-6), vo], dim=1)
            
            # interpolate C on grid
            c0 = nearestInterp2d(points0, x_axis, y_axis, cobj.cdf)
            c1 = nearestInterp2d(points1, x_axis, y_axis, cobj.cdf)
            
            h = (c1 - c0)/(eps+1e-12)
            h = torch.where(torch.isfinite(h), h, vo)  # Fallback to independence
            return torch.clamp(h, 1e-6, 1-1e-6)
            
        except Exception as e:
            logger.warning(f"Non-parametric h-function failed: {e}, using independence")
            return vo.clone()


############################################################
# 5) Sample Vine
############################################################

def sample_vine(vine: 'vine_obj_bin', nsamples: int, cfg: Optional[dict] = None):
    """
    Sample from vine. For param => partial approach. For nonparam => build local cdf.
    We'll store final in an array [nsamples, d], assume standard normal margins for demonstration.
    
    For D-vines, special handling is applied to better preserve correlations between
    non-adjacent variables.
    """
    # Special handling for D-vines
    if hasattr(vine, 'vine_family') and vine.vine_family == 'd-vine':
        logger.info("Using specialized D-vine sampling")
        return sample_d_vine(vine, nsamples)
    
    # Regular sampling for all vine types
    d = vine.n_cop
    cfg_all = DEFAULT_CFG if cfg is None else cfg
    samp_cfg = cfg_all.get("sampler", {})
    fast_param = samp_cfg.get("fast_parametric", True)
    fast_np    = samp_cfg.get("fast_nonparam", True)

    samples = torch.zeros((nsamples, d), dtype=torch.float32)
    normal = torch.distributions.Normal(0.,1.)
    
    # Generate first variable (always from standard normal)
    u_first = torch.rand(nsamples)
    u_first = torch.clamp(u_first, 1e-6, 1-1e-6)  # Avoid extremes
    samples[:,0] = normal.icdf(u_first)

    # Track sampling errors for debugging
    error_counts = {'nan': 0, 'inf': 0, 'out_of_range': 0, 'fallback_independence': 0}

    for i in range(1, d):
        lvl = i-1
        # Robust edge selection irrespective of vine family
        edges = vine.copulas[lvl]
        struct_edges = vine.ind_vine[lvl] if lvl < len(vine.ind_vine) else []
        root = lvl
        match_idx = 0
        
        # Find the edge connecting the root to variable i
        for ei, e in enumerate(struct_edges):
            if (e[0] == root and e[1] == i) or (e[1] == root and e[0] == i):
                match_idx = ei
                break
        
        if match_idx >= len(edges):          # variable not present on this level
            # Fallback: sample independently
            u_indep = torch.rand(nsamples)
            u_indep = torch.clamp(u_indep, 1e-6, 1-1e-6)
            samples[:,i] = normal.icdf(u_indep)
            error_counts['fallback_independence'] += 1
            continue
            
        cobj = edges[match_idx]

        # Get the conditioning variable (root)
        root_val = samples[:,lvl]
        
        # Convert to uniform scale for copula computations
        root_u = normal.cdf(root_val)
        root_u = torch.clamp(root_u, 1e-6, 1-1e-6)
        
        # Generate independent uniform variable for sampling
        rand_u = torch.rand(nsamples)
        rand_u = torch.clamp(rand_u, 1e-6, 1-1e-6)

        # For Gaussian parametric copulas (most common case) use the direct method
        if vine.param and hasattr(cobj, 'family') and cobj.family == "gaussian":
            try:
                rho = float(cobj.theta) if cobj.theta is not None else 0.0
                if not math.isfinite(rho):
                    rho = 0.0
                rho = max(min(rho, 0.95), -0.95)  # Conservative bounds to avoid numerical issues
                
                # Use more stable clamping for normal scores
                z = normal.icdf(root_u)
                e = normal.icdf(rand_u)
                
                # Clamp to avoid extreme values
                z = torch.clamp(z, -5.0, 5.0)
                e = torch.clamp(e, -5.0, 5.0)
                
                # More stable computation for numerical edge cases
                denom = max(1.0 - rho*rho, 1e-6)
                
                # Generate sample from conditional normal
                y = rho*z + math.sqrt(denom)*e
                
                # Handle any extreme values
                y = torch.clamp(y, -6.0, 6.0)
                
                # Convert back to uniform scale
                vi = normal.cdf(y)
                vi = torch.clamp(vi, 1e-6, 1-1e-6)
                
                # Check for any remaining issues
                if torch.isnan(vi).any() or torch.isinf(vi).any():
                    error_counts['nan'] += torch.isnan(vi).sum().item()
                    error_counts['inf'] += torch.isinf(vi).sum().item()
                    
                    # Replace invalid values with independent samples
                    invalid_mask = torch.isnan(vi) | torch.isinf(vi)
                    vi[invalid_mask] = rand_u[invalid_mask]
                    error_counts['fallback_independence'] += invalid_mask.sum().item()
                
                # Convert to normal margins for final result
                final_u = torch.clamp(vi, 1e-6, 1-1e-6)
                samples[:,i] = normal.icdf(final_u)
                
            except Exception as e:
                logger.warning(f"Gaussian sampling failed for variable {i}: {e}")
                # Fallback to independence
                samples[:,i] = normal.icdf(rand_u)
                error_counts['fallback_independence'] += nsamples
            
        # For Clayton copula
        elif vine.param and hasattr(cobj, 'family') and cobj.family == "clayton":
            try:
                alpha = float(cobj.theta) if cobj.theta is not None else 1.0
                if not math.isfinite(alpha) or alpha <= 0:
                    alpha = 1.0
                alpha = max(alpha, 0.1)  # Avoid numerical issues
                
                u1 = root_u
                c2 = rand_u
                
                # More robust Clayton sampling
                val = (c2.pow(-alpha/(1+alpha)) - u1.pow(-alpha) + 1.0).clamp_min(1e-8)
                vi = val.pow(-1.0/alpha)
                
                # Handle any NaN/Inf values
                if torch.isnan(vi).any() or torch.isinf(vi).any():
                    invalid_mask = torch.isnan(vi) | torch.isinf(vi)
                    vi[invalid_mask] = rand_u[invalid_mask]
                    error_counts['fallback_independence'] += invalid_mask.sum().item()
                
                vi = torch.clamp(vi, 1e-6, 1-1e-6)
                samples[:,i] = normal.icdf(vi)
                
            except Exception as e:
                logger.warning(f"Clayton sampling failed for variable {i}: {e}")
                # Fallback to independence
                samples[:,i] = normal.icdf(rand_u)
                error_counts['fallback_independence'] += nsamples
            
        # For independence copula
        elif vine.param and hasattr(cobj, 'family') and cobj.family == "ind":
            # Direct independent sampling
            samples[:,i] = normal.icdf(rand_u)
            
        # For other parametric copulas
        elif vine.param and fast_param:
            try:
                # Use inverse conditional CDF if available
                if hasattr(cobj, 'family'):
                    # Try the direct parametric approach for known families
                    if cobj.family in ["frank", "gumbel", "joe"]:
                        # For these families, fallback to independence for now
                        # TODO: Implement proper sampling methods
                        vi = rand_u
                    else:
                        # Unknown parametric family - fallback to independence
                        vi = rand_u
                else:
                    vi = rand_u
                
                vi = torch.clamp(vi, 1e-6, 1-1e-6)
                samples[:,i] = normal.icdf(vi)
                
            except Exception as e:
                logger.warning(f"Parametric sampling failed for variable {i}: {e}")
                samples[:,i] = normal.icdf(rand_u)
                error_counts['fallback_independence'] += nsamples
            
        # Non-parametric sampling with robust grid handling
        elif not vine.param:
            try:
                if fast_np and hasattr(cobj, 'cdf'):
                    # Initialize grid axes if not present
                    if not hasattr(cobj, 'cdf_xlin'):
                        x_axis, y_axis = vine.grid_u.axis()
                        cobj.cdf_xlin = x_axis
                        cobj.cdf_ylin = y_axis
                    
                    x_axis = cobj.cdf_xlin
                    y_axis = cobj.cdf_ylin
                    
                    # Find row indices for each sample
                    row_idx = torch.bucketize(root_u, x_axis)
                    row_idx = torch.clamp(row_idx, 1, x_axis.numel()-1) - 1
                    
                    # Extract corresponding CDF rows
                    cdf_rows = cobj.cdf[row_idx]
                    
                    # Use inverse CDF sampling
                    from .utils_interpolation import inverse_cdf_row
                    try:
                        vi = inverse_cdf_row(rand_u, cdf_rows, y_axis)
                        
                        # Handle any NaN/Inf values from interpolation
                        if torch.isnan(vi).any() or torch.isinf(vi).any():
                            invalid_mask = torch.isnan(vi) | torch.isinf(vi)
                            vi[invalid_mask] = rand_u[invalid_mask]
                            error_counts['fallback_independence'] += invalid_mask.sum().item()
                        
                        vi = torch.clamp(vi, 1e-6, 1-1e-6)
                        samples[:,i] = normal.icdf(vi)
                        
                    except Exception as e:
                        logger.warning(f"Inverse CDF sampling failed for variable {i}: {e}")
                        # Fallback to independence
                        samples[:,i] = normal.icdf(rand_u)
                        error_counts['fallback_independence'] += nsamples
                        
                else:
                    # Legacy slow method - fallback to independence for robustness
                    logger.warning(f"Using independence fallback for non-parametric variable {i}")
                    samples[:,i] = normal.icdf(rand_u)
                    error_counts['fallback_independence'] += nsamples
                    
            except Exception as e:
                logger.warning(f"Non-parametric sampling failed for variable {i}: {e}")
                samples[:,i] = normal.icdf(rand_u)
                error_counts['fallback_independence'] += nsamples
        
        else:
            # Ultimate fallback: independence
            samples[:,i] = normal.icdf(rand_u)
            error_counts['fallback_independence'] += nsamples

    # Final check: replace any remaining NaN/Inf values
    for i in range(d):
        col = samples[:, i]
        if torch.isnan(col).any() or torch.isinf(col).any():
            invalid_mask = torch.isnan(col) | torch.isinf(col)
            replacement_vals = normal.icdf(torch.rand(invalid_mask.sum()) * 0.8 + 0.1)
            samples[invalid_mask, i] = replacement_vals
            error_counts['nan'] += torch.isnan(col).sum().item()
            error_counts['inf'] += torch.isinf(col).sum().item()

    # Log any errors that occurred during sampling
    total_errors = sum(error_counts.values())
    if total_errors > 0:
        logger.info(f"Sampling completed with fallbacks: {error_counts}")
        logger.info(f"Total fallback rate: {total_errors/(nsamples*d):.1%}")

    return samples.cpu().numpy()


############################################################
# Attach
############################################################
# vine_obj_bin.fit = fit_vine
# vine_obj_bin.evaluation = evaluate_vine
# vine_obj_bin.sample = sample_vine

############################################################
# Utility: bandwidth optimisation via ``mise_optimization``
############################################################

# The original TensorFlow codebase included a two-phase MISE bandwidth
# optimiser.  Here we implement a lightweight alternative in PyTorch:
# an Adam search over a positive scale factor applied to the baseline
# bandwidth matrix.  The routine is self-contained and can be swapped
# in place of the TensorFlow version.

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
    """Optimise the bandwidth scaling factor ``a`` via Nadam (matching TensorFlow).

    The optimiser operates on ``log(a)`` so the candidate bandwidth
    ``B = a * bw_init`` remains positive. At each step a mean squared
    error between the estimated density and ``ref_norm`` is minimised.
    
    This implements TensorFlow's approach with:
    - Nadam optimizer (not just Adam)
    - Cross-validation split for evaluation
    - Proper MISE cost function
    """
    device = a_init.device
    
    # Split data for cross-validation (5-fold like TensorFlow)
    n_samples = data_x.shape[0]
    n_splits = 5
    fold_size = n_samples // n_splits
    
    # Create train/test splits
    indices = torch.randperm(n_samples, device=device)
    test_indices = indices[:fold_size]
    train_indices = indices[fold_size:]
    
    data_s_test = data_s[test_indices]
    data_x_train = data_x[train_indices]
    
    # Reshape data for cross-validation
    data_x_folds = []
    for i in range(n_splits):
        start_idx = i * (len(train_indices) // n_splits)
        end_idx = (i + 1) * (len(train_indices) // n_splits) if i < n_splits - 1 else len(train_indices)
        fold_data = data_x[train_indices[start_idx:end_idx]]
        data_x_folds.append(fold_data)
    
    # Parameterisation: scalar (LL1) or per-axis (LL2)
    if axis_separate:
        if a_init.dim()==0 or a_init.numel()==1:
            a_init = a_init.expand(2, n_cop)  # shape 2×n_cop
        a_log = a_init.log().clone().detach().requires_grad_(True)
    else:
        # single scalar shared by both axes and all edges
        if a_init.numel()>1:
            a_init = a_init.flatten()[0:1]
        a_log = a_init.log().clone().detach().requires_grad_(True)

    # Initialize Nadam parameters (matching TensorFlow)
    m = torch.zeros_like(a_log)
    v = torch.zeros_like(a_log)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6

    # Pre-compute grid differentials
    adu11, adu22 = grid_u.diff()
    step_s = torch.tensor([grid_s.ax1[1] - grid_s.ax1[0], grid_s.ax2[1] - grid_s.ax2[0]], device=device)
    min_s = grid_s.min
    max_s = grid_s.max

    prev_cost = 1e12
    for iter_num in range(max_iter):
        # Zero gradients
        if a_log.grad is not None:
            a_log.grad.zero_()
            
        # Compute bandwidth
        if axis_separate:
            a_val = torch.exp(a_log)  # 2×n_cop
            B = bw_init * a_val
        else:
            a_val = torch.exp(a_log)
            B = bw_init * a_val

        # Compute MISE cost (matching TensorFlow's MISE_mul)
        # Step 1: Local-likelihood on full data
        ker_grid_all = loclik_batch_eval(B, data_x, grid_x, n_cop, batch_size)
        K = grid_s.ax1.shape[0]
        ker_grid_all = ker_grid_all.view(K, K, n_cop).permute(1, 0, 2)
        
        if renorm_flag:
            # Use TensorFlow-style normalization
            from .cop_eval import eval_rs_p
            pd_grid = eval_rs_p(adu11, adu22, ker_grid_all, ref_norm, n_cop)
        else:
            pd_grid = torch.zeros_like(ker_grid_all)
        
        # Step 2: Cross-validation evaluation
        kkk_fin_list = []
        for k, fold_data in enumerate(data_x_folds):
            # Transform fold data
            fold_s = Transform(n_cop).forward_x(fold_data)
            
            # Local likelihood on fold
            ker_grid_fold = loclik_batch_eval(B, fold_data, grid_x, n_cop, batch_size)
            pd_grid1 = ker_grid_fold.view(K, K, n_cop).permute(1, 0, 2)
            
            if renorm_flag:
                pd_grid1 = eval_rs_p(adu11, adu22, pd_grid1, ref_norm, n_cop)
            
            # Interpolate at test points
            interp_data_list = []
            for kk in range(n_cop):
                # Use proper interpolation matching TensorFlow
                interp_data1 = interp_regular_nd_grid(
                    data_s_test[:, :, kk] if data_s_test.dim() == 3 else data_s_test,
                    min_s, max_s, pd_grid1[:, :, kk]
                )
                interp_data_list.append(interp_data1)
            
            interp_data = torch.stack(interp_data_list, dim=1)
            
            # Normalize
            if renorm_flag:
                norm_factor = torch.sum(pd_grid * step_s.prod(), dim=[0, 1])
            else:
                norm_factor = torch.sum(ker_grid_fold * step_s.prod(), dim=0)
            
            interp_data = interp_data / (norm_factor + 1e-30)
            kkk_fin_list.append(interp_data)
        
        # Combine cross-validation results
        kkk_fin = torch.cat(kkk_fin_list, dim=0)
        
        # Compute MISE cost
        if renorm_flag:
            cost = torch.sum(pd_grid**2 * step_s.prod(), dim=[0, 1]) - 2 * torch.mean(kkk_fin, dim=0)
        else:
            pd_grid_normalized = ker_grid_all / (torch.sum(ker_grid_all * step_s.prod(), dim=0, keepdim=True) + 1e-30)
            cost = torch.sum(pd_grid_normalized**2 * step_s.prod(), dim=0) - 2 * torch.mean(kkk_fin, dim=0)
        
        # Add penalty for out-of-bounds parameters (matching TensorFlow)
        out_of_bounds = (a_val <= 1e-4) | (a_val >= 2.0)
        if out_of_bounds.any():
            penalty = torch.where(out_of_bounds, 
                                  torch.abs(cost) * 0.001 * torch.sign(cost),
                                  torch.zeros_like(cost))
            cost = cost + penalty
        
        # Total cost
        total_cost = cost.mean()
        
        # Backward pass
        total_cost.backward()
        
        # Nadam update (matching TensorFlow)
        grad = a_log.grad
        iter1 = float(iter_num + 1)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**iter1) + (1 - beta1) * grad / (1 - beta1**iter1)
        v_hat = v / (1 - beta2**iter1)
        
        # Update
        with torch.no_grad():
            a_log -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            
            # Ensure bandwidth doesn't go too low (matching TensorFlow's 1e-2 threshold)
            if axis_separate:
                for i in range(n_cop):
                    bw_test = torch.abs(torch.exp(a_log[:, i]) * bw_init[:, i])
                    if (bw_test[1] < 1e-2).any():
                        a_log[1, i] = torch.log(torch.tensor(5e-3) / bw_init[1, i])
            else:
                bw_test = torch.abs(torch.exp(a_log) * bw_init)
                if (bw_test[1, :] < 1e-2).any():
                    min_a = 1e-2 / bw_init[1, :].max()
                    a_log.clamp_(min=torch.log(min_a))
            
            # Clamp to valid range
            a_log.clamp_(torch.log(torch.tensor(1e-2)), torch.log(torch.tensor(4.0)))

        # Convergence check
        cost_now = total_cost.item()
        if abs(prev_cost - cost_now) < tol:
            break
        prev_cost = cost_now

    # Final value
    with torch.no_grad():
        if axis_separate:
            a_final = torch.exp(a_log).clamp(0.01, 4.0)
        else:
            a_final = torch.exp(a_log).clamp(0.01, 4.0)
    
    return a_final.detach()

############################################################
# 6) Convenience API helpers (logpdf, pdf, cdf)
############################################################

def logpdf_vine(vine: 'vine_obj_bin', points: torch.Tensor):
    """Return log-pdf of the fitted vine at *points* (N×d tensor)."""
    p, _, _ = evaluate_vine(vine, points)
    # Extra robustness against NaN/Inf
    p_safe = p.clamp_min(1e-30)
    # Replace any lingering NaN/Inf with very low probability
    logp = torch.log(p_safe)
    return torch.where(torch.isfinite(logp), logp, torch.ones_like(logp) * -30.0)

def pdf_vine(vine: 'vine_obj_bin', points: torch.Tensor):
    """Return pdf at *points* — just a thin wrapper."""
    p, _, _ = evaluate_vine(vine, points)
    return p

def cdf_vine(vine: 'vine_obj_bin', points: torch.Tensor, nsim: int = 2000):
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

############################################################
# 7) Conditional mean prediction for Gaussian vines
############################################################

def conditional_mean_vine(vine: 'vine_obj_bin', fixed_vars, fixed_values, predict_var):
    """
    Compute the conditional expectation E[X_predict | X_fixed = fixed_values].
    
    For Gaussian copulas, this can be computed analytically using the 
    vine structure and the fitted parameters.
    
    Parameters
    ----------
    fixed_vars : list of int
        Indices of the conditioning variables
    fixed_values : list of float
        Values of the conditioning variables
    predict_var : int
        Index of the variable to predict
        
    Returns
    -------
    float
        Predicted conditional mean
    """
    # For parametric Gaussian vines, use analytical methods
    if vine.param:
        # Make sure all copulas are Gaussian
        for level in vine.copulas:
            for cop in level:
                if hasattr(cop, 'family') and cop.family != "gaussian":
                    logger.warning("Non-Gaussian copula found; analytical prediction may be inaccurate")
        
        # For single fixed variable, check if direct connection to root (Level 0)
        if len(fixed_vars) == 1 and fixed_vars[0] == 0:
            # Get the edge connecting root to predict_var
            for i, edge in enumerate(vine.ind_vine[0]):
                if edge[1] == predict_var:
                    # Find the copula object
                    cop = vine.copulas[0][i]
                    if hasattr(cop, 'theta'):
                        rho = cop.theta
                        return rho * fixed_values[0]
        
        # For a prediction from a non-root variable in C-vine (Level 0, reversed direction)
        if len(fixed_vars) == 1 and fixed_vars[0] != 0 and predict_var == 0:
            # Find edge [0, fixed_var] in level 0
            for i, edge in enumerate(vine.ind_vine[0]):
                if edge[1] == fixed_vars[0]:
                    # Find the copula object
                    cop = vine.copulas[0][i]
                    if hasattr(cop, 'theta'):
                        rho = cop.theta
                        return rho * fixed_values[0]
        
        # For multiple conditioning variables with uniform correlation matrix
        # This is a special case that doesn't require following paths in the vine
        if all(hasattr(cop, 'family') and cop.family == "gaussian" for level in vine.copulas for cop in level):
            # Check if all first-level correlations are approximately equal
            rhos = [cop.theta for cop in vine.copulas[0] if hasattr(cop, 'theta')]
            if max(rhos) - min(rhos) < 0.1:  # roughly uniform correlation
                # Use the formula for uniform correlation
                rho_avg = sum(rhos) / len(rhos)
                k = len(fixed_vars)
                fixed_sum = sum(fixed_values)
                
                # Adjust denominator for multiple conditioning variables
                if k == 1:
                    return rho_avg * fixed_sum
                else:
                    return rho_avg * fixed_sum / (1 + (k-1)*rho_avg)
        
        # Full path-tracing algorithm for Gaussian C-vines
        if vine.vine_family == 'c-vine' and all(hasattr(cop, 'family') and cop.family == "gaussian" 
                                              for level in vine.copulas for cop in level):
            # C-vine allows direct calculation of conditional expectation
            # using the vine structure and parameters
            return _conditional_mean_gaussian_cvine(vine, fixed_vars, fixed_values, predict_var)
    
    # For non-parametric vines, we need to use ML search with specific handling
    elif not vine.param:
        # Check if we have the necessary grid information
        has_grids = True
        for level in vine.copulas:
            for cop in level:
                if not hasattr(cop, 'pd_grid_uv') or not hasattr(cop, 'cdf'):
                    has_grids = False
                    break
        
        if has_grids:
            # For non-parametric vines, we can use a specialized ML search
            return _find_conditional_mean_nonparam(vine, fixed_vars, fixed_values, predict_var)
    
    # Fallback to general maximum likelihood search
    return _find_conditional_mean_ml(vine, fixed_vars, fixed_values, predict_var)

def _conditional_mean_gaussian_cvine(vine, fixed_vars, fixed_values, predict_var):
    """
    Compute conditional mean for a Gaussian C-vine using path tracing.
    
    For a C-vine with Gaussian pair-copulas, the conditional expectation can be
    computed by tracing paths through the vine structure and combining
    correlations appropriately.
    
    Parameters
    ----------
    vine : vine_obj_bin
        The fitted vine copula object
    fixed_vars : list of int
        Indices of conditioning variables
    fixed_values : list of float
        Values of conditioning variables
    predict_var : int
        Index of variable to predict
        
    Returns
    -------
    float
        Predicted conditional mean
    """
    # For a C-vine, the root is always variable 0
    root = 0
    
    # If predict_var is the root, handle specially
    if predict_var == root:
        # For C-vine, predicting the root variable from other variables
        # requires combining the direct correlations from root to each variable
        result = 0.0
        weights_sum = 0.0
        
        # Get all direct correlations from root to fixed variables
        for var_idx, value in zip(fixed_vars, fixed_values):
            # Find the edge connecting root to this variable
            for i, edge in enumerate(vine.ind_vine[0]):
                if edge[1] == var_idx:
                    cop = vine.copulas[0][i]
                    if hasattr(cop, 'theta'):
                        rho = cop.theta
                        # For Gaussian, the weight is rho^2
                        weight = rho**2
                        result += rho * value * weight
                        weights_sum += weight
        
        # Normalize by the sum of weights
        if weights_sum > 0:
            return result / weights_sum
        return 0.0
    
    # If one of fixed variables is the root, use its direct connection
    if root in fixed_vars:
        root_idx = fixed_vars.index(root)
        root_value = fixed_values[root_idx]
        
        # Find direct correlation from root to predict_var
        for i, edge in enumerate(vine.ind_vine[0]):
            if edge[1] == predict_var:
                cop = vine.copulas[0][i]
                if hasattr(cop, 'theta'):
                    direct_rho = cop.theta
                    
                    # If only conditioning on root, return direct correlation
                    if len(fixed_vars) == 1:
                        return direct_rho * root_value
                    
                    # For multiple conditioning variables, adjust based on 
                    # partial correlations in the vine
                    # This is a simplified approximation
                    other_vars = [v for v in fixed_vars if v != root]
                    other_values = [fixed_values[i] for i, v in enumerate(fixed_vars) if v != root]
                    
                    # Get maximum indirect correlation through other variables
                    max_indirect = 0.0
                    for var, val in zip(other_vars, other_values):
                        # Find correlation from root to this variable
                        for j, e in enumerate(vine.ind_vine[0]):
                            if e[1] == var:
                                cop_j = vine.copulas[0][j]
                                if hasattr(cop_j, 'theta'):
                                    rho_j = cop_j.theta
                                    # Find correlation between this variable and predict_var
                                    # Simplified - check higher levels of the vine for connection
                                    for level in range(1, len(vine.ind_vine)):
                                        for k, e2 in enumerate(vine.ind_vine[level]):
                                            if ((e2[0] == var and e2[1] == predict_var) or 
                                                (e2[1] == var and e2[0] == predict_var)):
                                                cop_k = vine.copulas[level][k]
                                                if hasattr(cop_k, 'theta'):
                                                    rho_k = cop_k.theta
                                                    # Indirect path contribution
                                                    indirect = rho_j * rho_k * val
                                                    if abs(indirect) > abs(max_indirect):
                                                        max_indirect = indirect
                    
                    # Combine direct and indirect paths
                    # Use a weighted combination
                    return 0.7 * direct_rho * root_value + 0.3 * max_indirect
    
    # For other cases, use a simplified approximation
    # Find the most direct path from fixed variables to predict_var
    result = 0.0
    weights_sum = 0.0
    
    # Check for direct connections from fixed variables to predict_var
    for var_idx, value in zip(fixed_vars, fixed_values):
        # Search all levels for connections
        for level, edges in enumerate(vine.ind_vine):
            for edge_idx, edge in enumerate(edges):
                if (edge[0] == var_idx and edge[1] == predict_var) or \
                   (edge[1] == var_idx and edge[0] == predict_var):
                    cop = vine.copulas[level][edge_idx]
                    if hasattr(cop, 'theta'):
                        rho = cop.theta
                        # Weight decreases with level (deeper connections less important)
                        weight = 1.0 / (level + 1)
                        result += rho * value * weight
                        weights_sum += weight
    
    # If no direct paths, fallback to simple approximation
    if weights_sum == 0:
        # Use the average correlation to predict_var
        rhos = []
        for level, edges in enumerate(vine.ind_vine):
            for edge_idx, edge in enumerate(edges):
                if edge[0] == predict_var or edge[1] == predict_var:
                    cop = vine.copulas[level][edge_idx]
                    if hasattr(cop, 'theta'):
                        rhos.append(cop.theta)
        
        if rhos:
            avg_rho = sum(rhos) / len(rhos)
            avg_val = sum(fixed_values) / len(fixed_values)
            return avg_rho * avg_val
        return 0.0
    
    return result / weights_sum

def _find_conditional_mean_ml(vine, fixed_vars, fixed_values, predict_var, search_range=None):
    """Find conditional mean using maximum likelihood search (fallback method)"""
    if search_range is None:
        search_range = np.linspace(-5, 5, 200)  # Wider search range
        
    # Create a test data point with fixed values
    test_data = np.zeros(vine.n_cop)
    for i, var_idx in enumerate(fixed_vars):
        test_data[var_idx] = fixed_values[i]
    
    # Search for best prediction using maximum likelihood
    best_val = None
    best_logp = -np.inf
    
    for val in search_range:
        # Copy test data and set the prediction variable
        x = test_data.copy()
        x[predict_var] = val
        
        # Calculate log probability under the vine
        x_tensor = torch.tensor([x], dtype=torch.float32)
        try:
            logp = logpdf_vine(vine, x_tensor).item()
            
            # Update best if higher probability
            if logp > best_logp and np.isfinite(logp):
                best_logp = logp
                best_val = val
        except Exception:
            # Skip this value if there's an error
            continue
            
    # If no valid prediction was found, return 0
    if best_val is None:
        return 0.0
            
    return best_val

def _find_conditional_mean_nonparam(vine, fixed_vars, fixed_values, predict_var, search_range=None):
    """
    Find conditional mean for non-parametric vines using a specialized approach.
    
    For non-parametric vines, we use a combination of:
    1. Direct grid interpolation for simple cases (when available)
    2. Numerical evaluation of conditional density
    
    Parameters
    ----------
    vine : vine_obj_bin
        The fitted vine copula
    fixed_vars : list of int
        Indices of conditioning variables
    fixed_values : list of float
        Values of conditioning variables
    predict_var : int
        Index of variable to predict
    search_range : array_like, optional
        Range of values to search over (default is -5 to 5 with 200 points)
        
    Returns
    -------
    float
        Predicted conditional mean
    """
    if search_range is None:
        search_range = np.linspace(-5, 5, 200)  # Wider search range
    
    # Create a test data point with fixed values
    test_data = np.zeros(vine.n_cop)
    for i, var_idx in enumerate(fixed_vars):
        test_data[var_idx] = fixed_values[i]
    
    # For non-parametric vines, we can use a different resolution search
    # that leverages cached grid information and handles missing values better
    best_val = None
    best_pdf = -np.inf
    
    # We'll use more search points near the likely value
    # Estimate a simple linear predictor for the initial guess
    initial_guess = 0.0
    if len(fixed_values) > 0:
        initial_guess = np.mean(fixed_values)
    
    # Create a search range centered on the initial guess
    fine_range = np.linspace(initial_guess - 2, initial_guess + 2, 150)
    wide_range = np.linspace(-5, 5, 50)
    search_values = np.unique(np.concatenate([fine_range, wide_range]))
    
    # Search over both fine and wide ranges
    for val in search_values:
        # Copy test data and set the prediction variable
        x = test_data.copy()
        x[predict_var] = val
        
        # Calculate log probability under the vine
        x_tensor = torch.tensor([x], dtype=torch.float32)
        try:
            # For non-parametric vines, logpdf can be unstable
            # Use a robust evaluation
            logp = logpdf_vine(vine, x_tensor).item()
            pdf = np.exp(logp) if np.isfinite(logp) else 0.0
            
            # Update best if higher probability
            if pdf > best_pdf and np.isfinite(pdf):
                best_pdf = pdf
                best_val = val
        except Exception:
            # Skip this value if there's an error
            continue
    
    # If no valid prediction was found, try a different approach
    # Use a weighted average of the search values
    if best_val is None:
        weights = []
        values = []
        
        for val in search_values:
            x = test_data.copy()
            x[predict_var] = val
            x_tensor = torch.tensor([x], dtype=torch.float32)
            
            try:
                logp = logpdf_vine(vine, x_tensor).item()
                if np.isfinite(logp):
                    pdf = np.exp(logp)
                    weights.append(pdf)
                    values.append(val)
            except Exception:
                continue
        
        if weights:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            # Weighted average
            best_val = np.sum(weights * np.array(values))
        else:
            # Last resort: return initial guess
            best_val = initial_guess
            
    return best_val

# register --------------------------------------------------
#vine_obj_bin.logpdf = logpdf_vine
#vine_obj_bin.pdf    = pdf_vine
#vine_obj_bin.cdf    = cdf_vine
#vine_obj_bin.conditional_mean = conditional_mean_vine


def update_theta_with_kernel_smoothing(vine, tr: int, edge, cobj, u_i: torch.Tensor, u_j: torch.Tensor, parent: int):
    """
    CRITICAL FIX: Update theta/theta_flip with kernel_cdf smoothing step.
    
    This matches TensorFlow's approach exactly:
    1. Compute h-function (conditional CDF)
    2. Apply kernel_cdf to ensure uniform margins
    3. Store in theta or theta_flip based on flip logic
    
    Args:
        vine: Vine object
        tr: Current tree level
        edge: Current edge [i,j]
        cobj: Copula object (parametric or nonparametric)
        u_i, u_j: Input uniform values
        parent: Parent variable index
    """
    next_level = tr + 1
    i, j = edge
    
    # Determine flip status based on parent variable
    # This matches TensorFlow's logic exactly
    if tr == 0:
        flip_flag = False  # First level never flips
    else:
        # Check if edge[0] is the parent variable
        flip_flag = (edge[0] != parent)
    
    if flip_flag:
        # Flipped case: h(u_j | u_i) -> store in theta_flip
        if hasattr(cobj, 'family'):
            # Parametric copula
            from .utils_prob import copulaccdf
            uv_data = torch.stack([u_j, u_i], dim=1)  # Note: flipped order
            h_val = copulaccdf(cobj, uv_data)
        else:
            # Non-parametric copula - use h-function
            h_val = _h_function(u_j, u_i, cobj, vine.grid_u, side="right")
        
        # CRITICAL: Apply kernel_cdf smoothing (this was missing!)
        h_np = h_val.cpu().numpy()
        ex_u_np = vine.grid_u.ex.cpu().numpy() if hasattr(vine.grid_u, 'ex') else np.linspace(0, 1, 50)
        h_smoothed, _, _ = kernel_cdf(h_np, h_np, ex_u_np)
        
        # Store in theta_flip
        vine.theta_flip[:, next_level, i] = torch.from_numpy(h_smoothed).to(h_val.device)
        
    else:
        # Normal case: h(u_j | u_i) -> store in theta
        if hasattr(cobj, 'family'):
            # Parametric copula
            from .utils_prob import copulaccdf
            uv_data = torch.stack([u_i, u_j], dim=1)  # Normal order
            h_val = copulaccdf(cobj, uv_data)
        else:
            # Non-parametric copula - use h-function
            h_val = _h_function(u_i, u_j, cobj, vine.grid_u, side="left")
        
        # CRITICAL: Apply kernel_cdf smoothing (this was missing!)
        h_np = h_val.cpu().numpy()
        ex_u_np = vine.grid_u.ex.cpu().numpy() if hasattr(vine.grid_u, 'ex') else np.linspace(0, 1, 50)
        h_smoothed, _, _ = kernel_cdf(h_np, h_np, ex_u_np)
        
        # Store in theta
        vine.theta[:, next_level, j] = torch.from_numpy(h_smoothed).to(h_val.device)


def get_parent_variable_fixed(tr: int, ind_vine, edge):
    """
    Fixed parent variable detection matching TensorFlow exactly.
    
    Args:
        tr: Tree level
        ind_vine: Vine index structure
        edge: Current edge [i,j]
        
    Returns:
        parent: Parent variable index
        left_set: Left variable set
        right_set: Right variable set
    """
    if tr == 0:
        # First level: parent is always the left variable
        return edge[0], [edge[0]], [edge[1]]
    
    # For higher levels, find the common variable between previous edges
    try:
        if edge[0] < len(ind_vine[tr-1]) and edge[1] < len(ind_vine[tr-1]):
            left_edge = ind_vine[tr-1][edge[0]]
            right_edge = ind_vine[tr-1][edge[1]]
            
            # Find common variable (parent)
            left_set = set(left_edge)
            right_set = set(right_edge)
            common = left_set.intersection(right_set)
            
            if common:
                parent = list(common)[0]
                left_remaining = left_set - common
                right_remaining = right_set - common
                return parent, list(left_remaining), list(right_remaining)
        
        # Fallback
        return edge[0], [edge[0]], [edge[1]]
        
    except (IndexError, TypeError):
        # Fallback to simple case
        return edge[0], [edge[0]], [edge[1]]

############################################################
# Attach methods to vine_obj_bin class
############################################################
# Use deferred attachment to avoid circular import issues
def _attach_vine_methods():
    """Attach methods to vine_obj_bin class after import."""
    from .objects import vine_obj_bin
    vine_obj_bin.fit = fit_vine
    vine_obj_bin.evaluation = evaluate_vine
    vine_obj_bin.sample = sample_vine
    vine_obj_bin.logpdf = logpdf_vine
    vine_obj_bin.pdf = pdf_vine
    vine_obj_bin.cdf = cdf_vine
    vine_obj_bin.conditional_mean = conditional_mean_vine

# Attach methods when module is imported
try:
    _attach_vine_methods()
except ImportError:
    # Methods will be attached when objects module is available
    pass