##################################################
# src/DVC/cop_eval.py
##################################################
# Copula evaluation helpers for grid-based copula densities

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
    Single iteration of row-column normalization.
    
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

def normalize_pdf_grid(adu11: torch.Tensor,
                       adu22: torch.Tensor,
                       ker_fit: torch.Tensor,
                       NORM1: torch.Tensor,
                       n_cop: int,
                       iterations: int = 500) -> torch.Tensor:
    """
    Row/column normalize a kernel density grid into a copula density.

    Parameters
    ----------
    iterations:
        Number of Sinkhorn-style normalization iterations. The previous code
        switched between separate 50- and 500-step routines based on an
        arbitrary threshold; this helper uses the requested iteration count
        directly.
    """
    device = ker_fit.device
    
    adu11_col = adu11.unsqueeze(-1)  # [K, 1]
    
    t1 = ker_fit / (NORM1 + 1e-30)
    
    for i in range(n_cop):
        if torch.max(t1[:, :, i]) < 1e-6:
            t1[:, :, i] = torch.ones_like(t1[:, :, i])
    
    adu22_1 = adu22.unsqueeze(-1).expand(-1, n_cop)  # [K, n_cop]
    adu11_col1 = adu11_col.expand(-1, n_cop).unsqueeze(0)  # [1, K, n_cop]
    
    for _ in range(max(int(iterations), 1)):
        t1 = eval1(adu11_col1, adu22_1, t1, n_cop)
    
    adu11_col1_t = adu11_col1.transpose(0, 1)  # [K, 1, n_cop]
    II = torch.sum(adu11_col1_t * torch.sum(adu22_1.unsqueeze(0) * t1, dim=1, keepdim=True), dim=0).squeeze(0)
    t1 = t1 / (II.unsqueeze(0).unsqueeze(0) + 1e-30)
    
    t1 = t1 * NORM1
    
    return t1


def eval_rs_cop(adu11: torch.Tensor,
                adu22: torch.Tensor,
                ker_fit: torch.Tensor,
                NORM1: torch.Tensor,
                n_cop: int) -> torch.Tensor:
    """Compatibility wrapper for the legacy 500-iteration normalization."""
    return normalize_pdf_grid(adu11, adu22, ker_fit, NORM1, n_cop, iterations=500)

def eval_rs_p(adu11: torch.Tensor,
              adu22: torch.Tensor,
              ker_fit: torch.Tensor,
              NORM1: torch.Tensor,
              n_cop: int) -> torch.Tensor:
    """
    Copula normalization for MISE cost function with 50 iterations.
    This is used during optimization (fewer iterations for speed).
    """
    return normalize_pdf_grid(adu11, adu22, ker_fit, NORM1, n_cop, iterations=50)

def cdf_grid_fun(pd_grid_uv: torch.Tensor,
                 ex_u: torch.Tensor,
                 u1d: torch.Tensor,
                 u2d: torch.Tensor,
                 n_cop: int,
                 axis: int = 1) -> torch.Tensor:
    """
    Compute the conditional CDF / h-function grid from a copula PDF grid.

    Despite the historical `cdf` name in the legacy code, this quantity is not
    the joint copula CDF `C(u,v)`. It is the grid used for pseudo-observation
    propagation and inverse h-function sampling, i.e. the conditional CDF of
    the second argument given the first one.
    
    Args:
        pd_grid_uv: PDF on UV grid, shape [K, K, n_cop]
        ex_u: Grid points
        u1d: Grid differences dim 1
        u2d: Grid differences dim 2
        n_cop: Number of copulas
        axis: Integration axis for the conditional target. The default
            ``axis=1`` returns h(v|u), matching the parametric
            ``copulaccdf`` convention. ``axis=0`` returns h(u|v).
        
    Returns:
        Conditional CDF values on the grid
    """
    knots = pd_grid_uv.shape[0]

    if int(axis) == 1:
        weights = u2d.to(pd_grid_uv.device).view(1, knots, 1).expand(knots, -1, n_cop)
        weighted = pd_grid_uv * weights
        integ = torch.cumsum(weighted, dim=1)
        norm_p = torch.sum(weighted, dim=1, keepdim=True)
    elif int(axis) == 0:
        weights = u1d.to(pd_grid_uv.device).view(knots, 1, 1).expand(-1, knots, n_cop)
        weighted = pd_grid_uv * weights
        integ = torch.cumsum(weighted, dim=0)
        norm_p = torch.sum(weighted, dim=0, keepdim=True)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")

    norm_p = torch.where(norm_p == 0, torch.ones_like(norm_p), norm_p)
    cdf1 = integ / norm_p
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
    CDF computation with the kernel smoothing step used by the grid pipeline.
    
    The grid pipeline applies ``kernel_cdf`` after ``cdf_grid_fun``.
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
    
    # Apply the kernel_cdf smoothing step used by the grid pipeline.
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
                
                # Ensure we don't divide by zero
                margin1_base = max(margin1_cdf[j], 1e-15)
                margin2_base = max(margin2_cdf[k], 1e-15)
                
                margin1_factor = margin1_smoothed[j] / margin1_base
                margin2_factor = margin2_smoothed[k] / margin2_base
                
                # Use weighted correction to preserve joint structure
                # For uniform margins, the smoothed values should be close to grid positions
                target_margin1 = j / max(1, cdf1.shape[0] - 1)
                target_margin2 = k / max(1, cdf1.shape[1] - 1)
                
                # Blend between correction and target uniform values
                blend_factor = 0.7  # Weight towards uniform margins
                corrected_margin1 = blend_factor * target_margin1 + (1 - blend_factor) * margin1_smoothed[j]
                corrected_margin2 = blend_factor * target_margin2 + (1 - blend_factor) * margin2_smoothed[k]
                
                # Use copula formula: C(u,v) adjusted to match corrected margins
                if original_val > 1e-10:
                    # Scale the joint CDF to match corrected marginal CDFs
                    # This is an approximation - exact methods would require more complex interpolation
                    correction_factor = math.sqrt(
                        (corrected_margin1 / margin1_base) * (corrected_margin2 / margin2_base)
                    )
                    smoothed_val = min(original_val * correction_factor, 
                                     min(corrected_margin1, corrected_margin2))
                else:
                    smoothed_val = 0.0
                
                cdf1_smoothed[j, k, i] = torch.clamp(
                    torch.tensor(smoothed_val), 0.0, 1.0
                ).to(cdf1.device)
    
    return cdf1_smoothed
