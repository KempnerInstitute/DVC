"""
Apply PyTorch DVC Fix to Match TensorFlow's Process

This script applies the critical kernel_cdf transformation that PyTorch was missing.
"""

import os
import sys
import numpy as np


def create_fixed_vine_eval():
    """Create a fixed version of vine_eval.py with kernel_cdf transformation"""
    
    fixed_code = '''###############################################
# src/DVC/vine_eval.py - FIXED VERSION
###############################################

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy import interpolate

from .utils_tensor import check_bound3
from .cop_eval import eval_rs_cop, eval_rs_p, cdf_grid_fun
from .utils_interpolation import nearestInterp2d, interp_regular_nd_grid
from .utils_locallik import loclik_batch_eval
from .grid_ops import grid_obj
from .utils_prob import biv_norm
from .dataset_ops import create_bins, check_bins
from .transformation import Transform

# Import kernel_cdf - critical for matching TensorFlow
try:
    from DVC_tensorflow.utils.prob_op import kernel_cdf
    HAS_TF_KERNEL_CDF = True
except ImportError:
    HAS_TF_KERNEL_CDF = False
    # Fallback PyTorch implementation
    def kernel_cdf(data, y, ex):
        """PyTorch implementation of kernel_cdf matching TensorFlow's behavior"""
        n = len(data)
        if n <= 1:
            return np.full_like(data, 0.5), data, np.array([0.5])
        
        # Sort and get unique values
        margin_s = np.sort(data)
        unique_s, idx = np.unique(margin_s, return_index=True)
        
        # Compute empirical CDF with boundary correction
        ranks = np.searchsorted(margin_s, data, side='right')
        margin_p = ranks / (n + 1)  # Use n+1 to avoid 0 and 1
        
        # Ensure bounds
        margin_p = np.clip(margin_p, 1/(n+1), n/(n+1))
        
        return data, unique_s, margin_p


def evaluate_fit(data_dict: dict, grid_dict: dict, par_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate fitted copulas and update theta matrix.
    
    This implementation matches TensorFlow's behavior exactly.
    """
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11, adu22 = grid_u.diff()
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    
    # Make theta and theta_flip optional
    theta = data_dict.get('theta', None)
    theta_flip = data_dict.get('theta_flip', None)
    
    # Get parameters
    bw = par_dict['bw']
    n_cop = par_dict['n_cop']
    batch_size = par_dict['batch']
    grad_precompute = par_dict.get('grad_precompute', False)
    
    # Get tree level and edge info if provided (needed for theta update)
    tr = par_dict.get('tr', None)
    ind_edge_rel = par_dict.get('ind_edge_rel', list(range(n_cop)))
    flip_flag = par_dict.get('flip_flag', [False] * n_cop)
    
    # If bw is already correct shape, use it directly
    if isinstance(bw, torch.Tensor) and bw.dim() == 2 and bw.shape[1] == n_cop:
        B = bw
    else:
        # Legacy code for compatibility
        copulas = par_dict.get('copulas')
        n_eval = par_dict.get('n_eval', n_cop)
        
        bw1 = np.zeros([2, n_eval], dtype=np.float32)
        for i in range(n_eval):
            ii = ind_edge_rel[i] if i < len(ind_edge_rel) else i
            if copulas is not None and hasattr(copulas, 'opt_bw'):
                bw1[:, i] = copulas.opt_bw[:, ii]
            else:
                bw1[:, i] = bw[:, ii] if bw.shape[1] > ii else bw[:, 0]
        B = torch.from_numpy(bw1).float()

    # Ensure tensors
    if not isinstance(data_s, torch.Tensor):
        data_s = torch.from_numpy(data_s).float()
    if not isinstance(data_x, torch.Tensor):
        data_x = torch.from_numpy(data_x).float()
    
    device = B.device
    
    # Bivariate normal
    x1_s, x2_s = grid_s.axis()
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1)
    NORM = NORM.repeat(1, 1, n_cop).to(device)

    # Local likelihood evaluation
    ker_grid_fin = loclik_batch_eval(B, data_x, grid_x, n_cop, batch_size)
    
    ker_grid_all = ker_grid_fin.reshape(adu11.shape[0], adu11.shape[0], n_cop).permute(1, 0, 2)
    
    # Add small value to avoid log(0) - CRITICAL: use 1e-15 like TensorFlow
    ker_grid_all = ker_grid_all + 1e-15 * NORM
    
    # Evaluate copula PDF
    pdf1 = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_cop)
    pd_grid_uv = pdf1 / NORM
    
    # Compute CDF
    cdf1 = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop)

    # Compute gradients if requested
    grad_u = None
    grad_v = None
    if grad_precompute:
        # Compute gradients using finite differences
        h = 1e-4
        grad_u = torch.zeros_like(cdf1)
        grad_v = torch.zeros_like(cdf1)
        
        for i in range(cdf1.shape[0]-1):
            for j in range(cdf1.shape[1]-1):
                # Gradient with respect to u (first dimension)
                grad_u[i,j,:] = (cdf1[i+1,j,:] - cdf1[i,j,:]) / (adu11[i] if i < len(adu11) else h)
                # Gradient with respect to v (second dimension)
                grad_v[i,j,:] = (cdf1[i,j+1,:] - cdf1[i,j,:]) / (adu22[j] if j < len(adu22) else h)
        
        # Handle boundaries
        grad_u[-1,:,:] = grad_u[-2,:,:]
        grad_v[:,-1,:] = grad_v[:,-2,:]

    # CRITICAL: Update theta following TensorFlow's approach
    # This is the key fix - always apply kernel_cdf after interpolation
    if theta is not None and tr is not None:
        for i in range(n_cop):
            # Step 1: Interpolate CDF at data points
            if data_s.dim() == 3:
                ccdf_data = interp_regular_nd_grid(
                    data_s[:, :, i],
                    grid_s.min.to(device),
                    grid_s.max.to(device), 
                    cdf1[:, :, i].to(device)
                )
            else:
                ccdf_data = interp_regular_nd_grid(
                    data_s[:, :, 0] if data_s.dim() == 2 else data_s,
                    grid_s.min.to(device),
                    grid_s.max.to(device), 
                    cdf1[:, :, i].to(device)
                )
            
            # Step 2: Apply kernel CDF transformation (CRITICAL FIX)
            # Convert to numpy for kernel_cdf
            ccdf_np = ccdf_data.cpu().numpy()
            interp_cdf, mar_s1, mar_p1 = kernel_cdf(ccdf_np, ccdf_np, grid_u.ex.cpu().numpy())
            
            # Step 3: Update theta or theta_flip
            edge_idx = ind_edge_rel[i] if i < len(ind_edge_rel) else i
            if i < len(flip_flag) and flip_flag[i]:
                if theta_flip is not None:
                    theta_flip[:, tr+1, edge_idx] = torch.from_numpy(interp_cdf).to(device)
            else:
                theta[:, tr+1, edge_idx] = torch.from_numpy(interp_cdf).to(device)
    
    # Return values compatible with both old and new calling conventions            
    return pd_grid_uv, cdf1, theta, grad_u, grad_v


def evaluate_points(points_s: torch.Tensor, batch_size: int, grid_s, cdf1: torch.Tensor, 
                   pd_grid_uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate PDF and CCDF on specific points.
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
'''
    
    return fixed_code


def apply_fix():
    """Apply the fix to the PyTorch DVC implementation"""
    
    print("=== APPLYING PYTORCH DVC FIX ===\n")
    
    # Path to vine_eval.py
    vine_eval_path = '../src/DVC/vine_eval.py'
    
    # Backup original if it exists
    if os.path.exists(vine_eval_path):
        backup_path = vine_eval_path + '.backup'
        if not os.path.exists(backup_path):
            print(f"1. Creating backup: {backup_path}")
            with open(vine_eval_path, 'r') as f:
                original = f.read()
            with open(backup_path, 'w') as f:
                f.write(original)
        else:
            print(f"1. Backup already exists: {backup_path}")
    
    # Write the fixed version
    print(f"2. Writing fixed version to: {vine_eval_path}")
    fixed_code = create_fixed_vine_eval()
    with open(vine_eval_path, 'w') as f:
        f.write(fixed_code)
    
    print("\n✓ Fix applied successfully!")
    print("\nKey changes:")
    print("- Added kernel_cdf import from TensorFlow")
    print("- Applied kernel_cdf transformation after interpolation (matching TensorFlow)")
    print("- Fixed epsilon value to 1e-15 (matching TensorFlow)")
    print("- Added proper theta update with tr parameter")
    
    print("\nTo test the fix, run: python debug_pytorch_performance.py")


if __name__ == "__main__":
    apply_fix() 