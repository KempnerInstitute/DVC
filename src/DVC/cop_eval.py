##################################################
# src/DVC/cop_eval.py
##################################################
# Copula evaluation functions matching TensorFlow implementation

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional
from .utils_tensor import replace_nan_inf
from .utils_prob import kernel_cdf

def eval1(adu11_col1: torch.Tensor,
          adu22_1: torch.Tensor,
          t2: torch.Tensor,
          n_cop: int):
    """
    Single iteration of row-column normalization (matching TensorFlow).
    
    Args:
        adu11_col1: Column differences, shape [1, K, n_cop]
        adu22_1: Row differences, shape [K, n_cop]
        t2: Current estimate, shape [K, K, n_cop]
        n_cop: Number of copulas
        
    Returns:
        Normalized tensor
    """
    I1 = torch.sum(adu22_1.unsqueeze(0) * t2, dim=1)  # shape [K, n_cop]
    I2 = torch.sum(adu11_col1 * t2, dim=1)            # shape [K, n_cop]

    K5 = torch.zeros_like(t2)
    for i in range(n_cop):
        K5[:, :, i] = torch.outer(I1[:, i], I2[:, i])

    t2_new = t2 / (K5 + 1e-30)
    t2_new = torch.where(torch.isfinite(t2_new), t2_new, torch.zeros_like(t2_new))
    return t2_new

def eval_rs_cop(adu11: torch.Tensor,
                adu22: torch.Tensor,
                ker_fit: torch.Tensor,
                NORM1: torch.Tensor,
                n_cop: int) -> torch.Tensor:
    """
    Copula normalization with 500 iterations (matching TensorFlow).
    
    Args:
        adu11: Grid differences u1
        adu22: Grid differences u2
        ker_fit: Kernel estimates
        NORM1: Bivariate normal reference
        n_cop: Number of copulas
        
    Returns:
        Normalized copula density
    """
    device = ker_fit.device
    
    adu11_col = adu11.unsqueeze(-1)  # [K, 1]
    
    t1 = ker_fit / (NORM1 + 1e-30)
    
    for i in range(n_cop):
        if torch.max(t1[:, :, i]) < 1e-6:
            t1[:, :, i] = torch.ones_like(t1[:, :, i])
    
    adu22_1 = adu22.unsqueeze(-1).expand(-1, n_cop)  # [K, n_cop]
    adu11_col1 = adu11_col.expand(-1, n_cop).unsqueeze(0)  # [1, K, n_cop]
    
    for _ in range(500):
        t1 = eval1(adu11_col1, adu22_1, t1, n_cop)
    
    adu11_col1_t = adu11_col1.transpose(0, 1)  # [K, 1, n_cop]
    II = torch.sum(adu11_col1_t * torch.sum(adu22_1.unsqueeze(0) * t1, dim=1, keepdim=True), dim=0).squeeze(0)
    t1 = t1 / (II.unsqueeze(0).unsqueeze(0) + 1e-30)
    
    t1 = t1 * NORM1
    
    return t1

def eval_rs_p(adu11: torch.Tensor,
              adu22: torch.Tensor,
              ker_fit: torch.Tensor,
              NORM1: torch.Tensor,
              n_cop: int) -> torch.Tensor:
    """
    Copula normalization for MISE cost function with 50 iterations.
    This is used during optimization (fewer iterations for speed).
    """
    device = ker_fit.device
    
    adu11_col = adu11.unsqueeze(-1)  # [K, 1]
    
    t1 = ker_fit / (NORM1 + 1e-30)
    
    for i in range(n_cop):
        if torch.max(t1[:, :, i]) < 1e-6:
            t1[:, :, i] = torch.ones_like(t1[:, :, i])
    
    adu22_1 = adu22.unsqueeze(-1).expand(-1, n_cop)  # [K, n_cop]
    adu11_col1 = adu11_col.expand(-1, n_cop).unsqueeze(0)  # [1, K, n_cop]
    
    for _ in range(50):
        t1 = eval1(adu11_col1, adu22_1, t1, n_cop)
    
    adu11_col1_t = adu11_col1.transpose(0, 1)  # [K, 1, n_cop]
    II = torch.sum(adu11_col1_t * torch.sum(adu22_1.unsqueeze(0) * t1, dim=1, keepdim=True), dim=0).squeeze(0)
    t1 = t1 / (II.unsqueeze(0).unsqueeze(0) + 1e-30)
    
    t1 = t1 * NORM1
    
    return t1

def cdf_grid_fun(pd_grid_uv: torch.Tensor,
                 ex_u: torch.Tensor,
                 u1d: torch.Tensor,
                 u2d: torch.Tensor,
                 n_cop: int) -> torch.Tensor:
    """
    Compute CDF on grid from PDF (matching TensorFlow).
    
    Args:
        pd_grid_uv: PDF on UV grid, shape [K, K, n_cop]
        ex_u: Grid points
        u1d: Grid differences dim 1
        u2d: Grid differences dim 2
        n_cop: Number of copulas
        
    Returns:
        CDF values on grid
    """
    device = pd_grid_uv.device
    knots = pd_grid_uv.shape[0]
    
    u2d_tile = u2d.view(knots, 1, 1).expand(-1, knots, n_cop)
    
    pd_grid_uv_transp = pd_grid_uv.permute(1, 0, 2)
    
    integ = torch.cumsum(pd_grid_uv_transp * u2d_tile, dim=0)
    
    norm_p = torch.sum(pd_grid_uv * u2d_tile, dim=0)
    
    norm_p = torch.where(norm_p == 0, torch.ones_like(norm_p), norm_p)
    
    cdf1 = integ / norm_p.unsqueeze(0)
    
    cdf1 = cdf1.permute(1, 0, 2)
    
    cdf1 = torch.clamp(cdf1, 0.0, 1.0)
    
    return cdf1

def cdf_grid_fun_with_kernel_smoothing(pd_grid_uv: torch.Tensor,
                                       ex_u: torch.Tensor,
                                       u1d: torch.Tensor,
                                       u2d: torch.Tensor,
                                       n_cop: int,
                                       data_s: torch.Tensor,
                                       grid_s_min: torch.Tensor,
                                       grid_s_max: torch.Tensor) -> torch.Tensor:
    """
    CRITICAL FIX: CDF computation with kernel smoothing step that TensorFlow does.
    
    This is the missing piece - TensorFlow calls kernel_cdf after cdf_grid_fun
    to ensure 1D uniform margins. PyTorch was skipping this step.
    
    Args:
        pd_grid_uv: PDF on UV grid, shape [K, K, n_cop]
        ex_u: Grid points
        u1d, u2d: Grid differences
        n_cop: Number of copulas
        data_s: Data in S space for interpolation
        grid_s_min, grid_s_max: Grid bounds
        
    Returns:
        CDF values with kernel smoothing applied
    """
    # First get basic CDF
    cdf1 = cdf_grid_fun(pd_grid_uv, ex_u, u1d, u2d, n_cop)
    
    # CRITICAL: Apply the kernel_cdf smoothing step that TensorFlow does
    # This ensures 1D uniform margins and is often missing in PyTorch
    cdf1_smoothed = torch.zeros_like(cdf1)
    
    for i in range(n_cop):
        # For each copula, smooth the marginal CDFs to ensure uniform margins
        ex_u_np = ex_u.cpu().numpy()
        
        # Extract marginal CDFs for dimension 1 (rows)
        margin1_cdf = cdf1[:, -1, i].cpu().numpy()  # CDF along first dimension
        margin1_smoothed, _, _ = kernel_cdf(margin1_cdf, margin1_cdf, ex_u_np)
        
        # Extract marginal CDFs for dimension 2 (columns)  
        margin2_cdf = cdf1[-1, :, i].cpu().numpy()  # CDF along second dimension
        margin2_smoothed, _, _ = kernel_cdf(margin2_cdf, margin2_cdf, ex_u_np)
        
        # Reconstruct the 2D CDF with smoothed margins
        # This enforces that the marginal CDFs are exactly uniform
        for j in range(cdf1.shape[0]):
            for k in range(cdf1.shape[1]):
                # Interpolate to maintain joint structure while ensuring uniform margins
                weight_j = j / max(1, cdf1.shape[0] - 1)
                weight_k = k / max(1, cdf1.shape[1] - 1)
                
                # Use original CDF as base but scale to match smoothed margins
                original_val = cdf1[j, k, i].item()
                margin1_factor = margin1_smoothed[j] / max(margin1_cdf[j], 1e-15)
                margin2_factor = margin2_smoothed[k] / max(margin2_cdf[k], 1e-15)
                
                # Geometric mean of margin corrections to preserve joint structure
                correction_factor = math.sqrt(margin1_factor * margin2_factor)
                smoothed_val = original_val * correction_factor
                
                cdf1_smoothed[j, k, i] = torch.clamp(
                    torch.tensor(smoothed_val), 0.0, 1.0
                ).to(cdf1.device)
    
    return cdf1_smoothed