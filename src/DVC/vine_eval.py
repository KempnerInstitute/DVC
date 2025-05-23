###############################################
# src/DVC/vine_eval.py
###############################################

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import interpolate

from .utils_tensor import check_bound3
from .cop_eval import eval_rs_cop, cdf_grid_fun
from .utils_interpolation import nearestInterp2d, interp_regular_nd_grid
from .utils_locallik import loclik_batch_eval
from .grid_ops import grid_obj
from .utils_prob import biv_norm, kernel_cdf, kernel_cdf_batch
from .dataset_ops import create_bins, check_bins
from .transformation import Transform


def evaluate_fit(data_dict: dict, grid_dict: dict, par_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Evaluate fitted copulas and update theta matrix.
    
    Args:
        data_dict: Contains data_s, data_x, theta, theta_flip
        grid_dict: Contains grid_u, grid_s, grid_x 
        par_dict: Contains copulas, n_eval, batch sizes, etc.

    Returns:
        pd_grid_uv: PDF on UV grid
        cdf1: CDF values
        theta: Updated theta matrix
        theta_flip: Updated theta_flip matrix
    """
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

    # Collect bandwidths
    bw1 = np.zeros([2, n_eval], data_s.dtype)
    for i in range(n_eval):
        ii = ind_edge_rel[i]
        bw1[:, i] = copulas.opt_bw[:, ii]
    B = torch.from_numpy(bw1).float()

    # Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1)
    NORM = NORM.repeat(1, 1, n_eval)

    # Local likelihood evaluation
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_eval, batch_size)
    
    ker_grid_all = ker_grid_fin.reshape(adu11.shape[0], adu11.shape[0], n_eval).permute(1, 0, 2)
    
    # Add small value to avoid log(0)
    ker_grid_all = ker_grid_all + 1e-15 * NORM
    
    # Evaluate copula PDF
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_eval)
    pd_grid_uv = pdf1 / NORM
    
    # Compute CDF
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_eval)

    # Update theta matrix
    device = data_s.device if torch.is_tensor(data_s) else torch.device('cpu')
    for i in range(n_eval):
        # Interpolate CDF at data points
        ccdf_data = interp_regular_nd_grid(
            torch.from_numpy(data_s[:, :, i]).to(device),
            grid_s.min.to(device),
            grid_s.max.to(device), 
            cdf1[:, :, i].to(device)
        )
        
        interp_cdf, _, _ = kernel_cdf(
            ccdf_data.cpu().numpy(),
            ccdf_data.cpu().numpy(),
            grid_u.ex
        )
        
        if flip_flag[i] == False:
            theta[:, tr+1, ind_edge_rel[i]] = interp_cdf
        else:
            theta_flip[:, tr+1, ind_edge_rel[i]] = interp_cdf
            
    return pd_grid_uv, cdf1, theta, theta_flip


def evaluate_points(points_s: torch.Tensor, batch_size: int, grid_s, cdf1: torch.Tensor, 
                   pd_grid_uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate PDF and CCDF on specific points.
    
    Args:
        points_s: Points in S-space
        batch_size: Batch size for processing
        grid_s: Grid object for S-space
        cdf1: CDF values on grid
        pd_grid_uv: PDF on UV grid
        
    Returns:
        pd_points: PDF at points
        ccdf_points: CCDF at points
    """
    n_points = points_s.shape[0]
    batch_len = n_points // batch_size
    
    pd_list = []
    ccdf_list = []
    
    s_ax1 = grid_s.ax1
    s_ax2 = grid_s.ax2
    
    for j in range(batch_size):
        start_idx = batch_len * j
        end_idx = batch_len * (j + 1) if j < batch_size - 1 else n_points
        
        points_batch = points_s[start_idx:end_idx, :]
        
        # Nearest neighbor interpolation for PDF
        pd_points1 = nearestInterp2d(points_batch, s_ax1, s_ax2, pd_grid_uv)
        
        # Regular grid interpolation for CCDF
        ccdf_points1 = interp_regular_nd_grid(
            points_batch,
            grid_s.min,
            grid_s.max,
            cdf1
        )
        
        pd_list.append(pd_points1)
        ccdf_list.append(ccdf_points1)
    
    pd_points = torch.cat(pd_list)
    ccdf_points = torch.cat(ccdf_list)

    return pd_points.flatten(), ccdf_points.flatten()


def evaluate_fit_bin(data_dict: dict, grid_dict: dict, par_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate fitted copulas for binned data.
    
    Args:
        data_dict: Contains data_s, data_x  
        grid_dict: Contains grid_u, grid_s, grid_x
        par_dict: Contains bandwidth, n_cop, batch size, etc.

    Returns:
        pd_grid_uv: PDF on UV grid
        cdf1: CDF values
    """
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
    ind_edge_rel = par_dict['ind_edge_rel']
    
    # Collect bandwidths
    bw1 = np.empty([2, n_cop1], data_s.dtype)
    for i in range(n_cop1):
        ii = ind_edge_rel[i]
        bw1[:, i] = bw[:, ii]
    
    B = torch.from_numpy(bw1).float()
    
    # Bivariate normal
    x1_s, x2_s = grid_s.ax1, grid_s.ax2
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1)
    NORM = NORM.repeat(1, 1, n_cop1)
    
    # Convert to tensors
    data_s = torch.from_numpy(data_s).float()
    data_x = torch.from_numpy(data_x).float()
    
    # Local likelihood evaluation
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_cop1, batch_size)
    
    ker_grid_all = ker_grid_fin.reshape(adu11.shape[0], adu11.shape[0], n_cop1).permute(1, 0, 2)
    
    # Add small value to avoid log(0)
    ker_grid_all = ker_grid_all + 1e-10 * NORM
    
    # Evaluate copula PDF
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_cop1)
    pd_grid_uv = pdf1 / NORM
    
    # Compute CDF
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop1)

    return pd_grid_uv, cdf1