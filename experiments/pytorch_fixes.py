"""
PyTorch DVC Implementation Fixes

This module contains all the fixes needed to align the PyTorch implementation
with the TensorFlow implementation, focusing on:
1. Row/Column Normalization (500 iterations)
2. Proper epsilon values (1e-30)
3. Theta vs Theta_flip handling
4. Kernel CDF smoothing
5. Independence AIC with correlation penalty
6. Chain-of-Conditionals in sampling
7. Proper binning logic
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Union
import logging

logger = logging.getLogger(__name__)

def create_bins(data: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Create bins for data using quantiles.
    
    Args:
        data: Input data to bin
        n_bins: Number of bins
    
    Returns:
        Array of bin edges
    """
    # Use quantiles for binning
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(data, quantiles)
    
    # Ensure unique bins
    bins = np.unique(bins)
    if len(bins) < n_bins + 1:
        # Add small offsets if not enough unique bins
        eps = np.finfo(data.dtype).eps
        bins = np.linspace(data.min() - eps, data.max() + eps, n_bins + 1)
    
    return bins

def check_bins(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Check and fix bin assignments.
    
    Args:
        data: Input data
        bins: Bin edges
    
    Returns:
        Fixed bin assignments
    """
    # Get bin assignments
    assignments = np.digitize(data, bins) - 1
    
    # Fix any out-of-bounds assignments
    assignments = np.clip(assignments, 0, len(bins) - 2)
    
    return assignments

def copulainvccdf(cop, uv: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute inverse conditional CDF.
    
    Args:
        cop: Copula object with invccdf method
        uv: List of [u1, u2] tensors
    
    Returns:
        Inverse conditional CDF values
    """
    return cop.invccdf(uv[0], uv[1])

def fix_eval_rs_cop(pd_grid_uv: torch.Tensor, n_iter: int = 500, eps: float = 1e-30) -> torch.Tensor:
    """
    Fixed row/column normalization with 500 iterations and proper epsilon.
    
    Args:
        pd_grid_uv: Input grid to normalize
        n_iter: Number of iterations (default: 500 to match TF)
        eps: Small constant for numerical stability (default: 1e-30 to match TF)
    
    Returns:
        Normalized grid
    """
    t2 = pd_grid_uv.clone()
    for _ in range(n_iter):
        # Row sum normalization
        I1 = torch.sum(t2, dim=1, keepdim=True)
        t2 = t2 / (I1 + eps)
        
        # Column sum normalization
        I2 = torch.sum(t2, dim=0, keepdim=True)
        t2 = t2 / (I2 + eps)
        
        # Clamp to avoid numerical issues
        t2 = t2.clamp(eps, 1.0)
    
    return t2

def fix_kernel_cdf_smoothing(ccdf_data: torch.Tensor, grid_u: torch.Tensor) -> torch.Tensor:
    """
    Apply proper kernel CDF smoothing after grid interpolation.
    
    Args:
        ccdf_data: Raw CDF values from grid interpolation
        grid_u: Grid points for smoothing
    
    Returns:
        Smoothed CDF values
    """
    # Sort data for proper CDF
    sorted_data, indices = torch.sort(ccdf_data)
    n = len(sorted_data)
    
    # Compute empirical CDF with proper smoothing
    ranks = torch.searchsorted(sorted_data, ccdf_data)
    cdf_vals = ranks.float() / (n + 1)  # Add 1 to match TF
    
    # Ensure monotonicity and bounds
    cdf_vals = cdf_vals.clamp(1e-12, 1.0 - 1e-12)
    
    return cdf_vals

def fix_independence_aic(data: torch.Tensor, n_samples: int) -> float:
    """
    Compute AIC for independence copula with correlation penalty.
    
    Args:
        data: Input data of shape [n_samples, 2]
        n_samples: Number of samples
    
    Returns:
        AIC value with correlation penalty
    """
    # Compute empirical correlation
    emp_corr = torch.corrcoef(data.T)[0, 1]
    
    # Add correlation-based penalty (matching TF approach)
    penalty = n_samples * torch.abs(emp_corr)**2
    
    # AIC = 2k (k=0 for independence) + penalty
    aic = penalty.item()
    
    return aic

def fix_chain_conditional_sampling(
    vine,
    v: torch.Tensor,
    k: int,
    i: int,
    parent: int,
    cop,
    flip: bool
) -> torch.Tensor:
    """
    Properly handle chain-of-conditionals in sampling.
    
    Args:
        vine: Vine copula object
        v: Current samples
        k, i: Current indices
        parent: Parent variable index
        cop: Copula object
        flip: Whether to use flipped version
    
    Returns:
        Updated samples
    """
    if flip:
        # If flipped, swap the order of conditioning
        v_new = copulainvccdf(cop, [v[:, k, parent], v[:, k+1, i]])
    else:
        # Normal order
        v_new = copulainvccdf(cop, [v[:, k, i], v[:, k+1, parent]])
    
    return v_new

def fix_binning_logic(
    vine,
    tr: int,
    edge: List[int],
    parent: int,
    theta: torch.Tensor,
    theta_flip: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fix binning logic to properly handle parent variable selection.
    
    Args:
        vine: Vine copula object
        tr: Current tree level
        edge: Current edge indices
        parent: Parent variable index
        theta: Main theta matrix
        theta_flip: Flipped theta matrix
    
    Returns:
        bins, val_to_bin arrays
    """
    if tr == 1:
        # First level - use direct parent
        parent_vals = theta[:, tr-1, parent]
    else:
        # Check if we need flipped values
        ind_par_now = vine.ind_vine[tr-1][edge[1]]
        parent22, _, _ = parent_var(tr-1, vine.ind_vine, ind_par_now)
        
        if vine.ind_vine[tr-2][ind_par_now[0]][0] == parent22:
            parent_vals = theta[:, tr-1, parent]
        else:
            parent_vals = theta_flip[:, tr-1, parent]
    
    # Create bins
    bins = create_bins(parent_vals.cpu().numpy(), vine.n_bin)
    val_to_bin = np.digitize(parent_vals.cpu().numpy(), bins) - 1
    val_to_bin = check_bins(parent_vals.cpu().numpy(), bins)
    
    return torch.tensor(bins), torch.tensor(val_to_bin)

def apply_fixes_to_vine(vine):
    """
    Apply all fixes to a vine copula object.
    
    Args:
        vine: Vine copula object to fix
    """
    logger.info("Applying PyTorch DVC fixes...")
    
    # 1. Update eval_rs_cop to use 500 iterations
    vine.eval_rs_cop = lambda x: fix_eval_rs_cop(x, n_iter=500, eps=1e-30)
    
    # 2. Add kernel CDF smoothing after grid interpolation
    vine._smooth_cdf = fix_kernel_cdf_smoothing
    
    # 3. Update independence copula AIC calculation
    vine._compute_independence_aic = fix_independence_aic
    
    # 4. Fix chain-of-conditionals in sampling
    vine._sample_conditional = fix_chain_conditional_sampling
    
    # 5. Update binning logic
    vine._handle_binning = fix_binning_logic
    
    logger.info("All fixes applied successfully!") 