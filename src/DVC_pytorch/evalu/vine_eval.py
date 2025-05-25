import torch
import numpy as np
from utils.prob_op import biv_norm, kernel_cdf
from evalu.cop_eval import eval_rs_cop, cdf_grid_fun
from utils.interpolation import nearestInterp2d, interp1d_np
from optim.local_lik import loclik_batch_eval

# Note: loclik_batch and loclik_batch_eval functions need to be implemented
# after converting local_lik.py from the optim directory

################# EVALUATE PDF (UV-SPACE), CDF AND THETA ######################

def evaluate_fit(data_dict, grid_dict, par_dict):
    """
    Evaluate fitted copula: compute PDF, CDF and update theta values
    
    Args:
        data_dict: Dictionary containing data and theta arrays
        grid_dict: Dictionary containing grid objects
        par_dict: Dictionary containing parameters
        
    Returns:
        pd_grid_uv: PDF on UV grid
        cdf1: CDF values
        theta: Updated theta values
        theta_flip: Updated theta_flip values
    """
    device = data_dict['data_s'].device
    dtype = data_dict['data_s'].dtype
    
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11, adu22 = grid_u.diff()
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    theta = data_dict['theta']
    theta_flip = data_dict['theta_flip']
    
    copulas = par_dict['copulas']
    n_eval = par_dict['n_eval']
    batch_size = par_dict['batch']
    batch_size_cdf = par_dict['batch_cdf']
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    flip_flag = par_dict['flip_flag']
    
    # Get bandwidth parameters
    bw1 = torch.zeros((2, n_eval), dtype=dtype, device=device)
    for i in range(n_eval):
        ii = ind_edge_rel[i]
        bw1[:, i] = torch.tensor(copulas.opt_bw[:, ii], dtype=dtype, device=device)
    B = bw1.reshape(2, n_eval)
    
    # Bivariate normal
    x1_s, x2_s = grid_s.axis()
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1)
    NORM = NORM.repeat(1, 1, n_eval)
    
    # Compute local likelihood on grid
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_eval, batch_size)
    
    # Reshape kernel grid
    ker_grid_all = ker_grid_fin.reshape(adu11.shape[0], adu11.shape[0], n_eval).permute(1, 0, 2)
    
    # Add small value to avoid zero probability
    ker_grid_all = ker_grid_all + 1e-15 * NORM
    
    # Evaluate copula PDF
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_eval)
    
    # Normalize to get PDF in UV space
    pd_grid_uv = pdf1 / NORM
    
    # Compute CDF from PDF
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval)
    
    # Update theta values using h-functions (conditional CDFs)
    for i in range(n_eval):
        # Interpolate CDF at data points
        ccdf_data = _interp_regular_grid_2d(data_s[:, :, i], 
                                           grid_s.min_grid(), 
                                           grid_s.max_grid(), 
                                           cdf1[:, :, i])
        
        # Force to be uniform using kernel CDF
        interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccdf_data, ccdf_data, grid_u.ex)
        
        # Store in appropriate theta array
        if flip_flag[i] == False:
            theta[:, tr + 1, ind_edge_rel[i]] = interp_cdf
        else:
            theta_flip[:, tr + 1, ind_edge_rel[i]] = interp_cdf
    
    return pd_grid_uv, cdf1, theta, theta_flip

################# EVALUATE PDF AND CCDF ON THE POINTS ######################

def evaluate_points(points_s, batch_size, grid_s, cdf1, pd_grid_uv):
    """
    Evaluate PDF and CCDF on specific points
    
    Args:
        points_s: Points to evaluate at
        batch_size: Batch size for processing
        grid_s: Grid object
        cdf1: CDF values on grid
        pd_grid_uv: PDF values on grid
        
    Returns:
        pd_points: PDF values at points
        ccdf_points: CCDF values at points
    """
    device = points_s.device
    dtype = points_s.dtype
    
    pd_points_list = []
    ccdf_points_list = []
    batch_len = points_s.shape[0] // batch_size
    
    s_ax1 = grid_s.ax1
    s_ax2 = grid_s.ax2
    
    for j in range(batch_size):
        if j == batch_size - 1:
            points_batch = points_s[batch_len * j:]
        else:
            points_batch = points_s[batch_len * j:batch_len * (j + 1)]
        
        # Nearest neighbor interpolation for PDF
        pd_points1 = nearestInterp2d(points_batch, s_ax1, s_ax2, pd_grid_uv)
        
        # Regular grid interpolation for CDF
        # Note: This needs a PyTorch implementation of batch_interp_regular_nd_grid
        # For now, we'll use a simple bilinear interpolation approach
        ccdf_points1 = _interp_regular_grid_2d(points_batch, grid_s.min_grid(), 
                                               grid_s.max_grid(), cdf1)
        
        pd_points_list.append(pd_points1)
        ccdf_points_list.append(ccdf_points1)
    
    pd_points = torch.cat(pd_points_list)
    ccdf_points = torch.cat(ccdf_points_list)
    
    return pd_points, ccdf_points

def _interp_regular_grid_2d(points, grid_min, grid_max, values):
    """
    Simple 2D interpolation on regular grid
    
    Args:
        points: Points to interpolate at (N, 2)
        grid_min: Minimum grid values (2,)
        grid_max: Maximum grid values (2,)
        values: Grid values (H, W)
        
    Returns:
        Interpolated values at points
    """
    device = points.device
    dtype = points.dtype
    
    # Normalize points to [0, 1]
    normalized = (points - grid_min) / (grid_max - grid_min)
    
    # Get grid dimensions
    h, w = values.shape[:2]
    
    # Scale to grid indices
    scaled = normalized * torch.tensor([h - 1, w - 1], dtype=dtype, device=device)
    
    # Get integer indices and fractional parts
    indices = scaled.long()
    fracs = scaled - indices.float()
    
    # Clamp indices
    indices[:, 0] = torch.clamp(indices[:, 0], 0, h - 2)
    indices[:, 1] = torch.clamp(indices[:, 1], 0, w - 2)
    
    # Get corner values
    i0, j0 = indices[:, 0], indices[:, 1]
    i1, j1 = i0 + 1, j0 + 1
    
    # Bilinear interpolation
    v00 = values[i0, j0]
    v01 = values[i0, j1]
    v10 = values[i1, j0]
    v11 = values[i1, j1]
    
    fx, fy = fracs[:, 0], fracs[:, 1]
    
    interp = (v00 * (1 - fx) * (1 - fy) +
              v01 * (1 - fx) * fy +
              v10 * fx * (1 - fy) +
              v11 * fx * fy)
    
    return interp

#################### EVALUATE BINNING ###########################

def evaluate_fit_bin(data_dict, grid_dict, par_dict):
    """
    Evaluate fitted copula for binned data
    
    Args:
        data_dict: Dictionary containing data
        grid_dict: Dictionary containing grid objects
        par_dict: Dictionary containing parameters
        
    Returns:
        pd_grid_uv: PDF on UV grid
        cdf1: CDF values
    """
    device = data_dict['data_s'].device
    dtype = data_dict['data_s'].dtype
    
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11 = grid_u.diff1
    adu22 = grid_u.diff2
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    
    bw = par_dict['bw']
    n_cop1 = par_dict['n_cop']
    batch_size = par_dict['batch']
    tr = par_dict['tr']
    ind_edge_rel = par_dict['ind_edge_rel']
    
    # Bandwidth
    bw1 = torch.empty((2, n_cop1), dtype=dtype, device=device)
    for i in range(n_cop1):
        ii = ind_edge_rel[i]
        bw1[:, i] = bw[:, ii]
    
    B = bw1.reshape(2, n_cop1)
    
    # Bivariate normal
    x1_s, x2_s = grid_s.axis()
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1)
    NORM = NORM.repeat(1, 1, n_cop1)
    
    # Compute local likelihood
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_cop1, batch_size)
    
    # Reshape and process
    ker_grid_all = ker_grid_fin.reshape(adu11.shape[0], adu11.shape[0], n_cop1).permute(1, 0, 2)
    ker_grid_all = ker_grid_all + 1e-10 * NORM
    
    # Compute PDF and CDF
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_cop1)
    pd_grid_uv = pdf1 / NORM
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)
    
    return pd_grid_uv, cdf1 