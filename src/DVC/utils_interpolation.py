###############################################
# src/DVC/utils_interpolation.py
###############################################

import torch
import torch.nn.functional as F
import numpy as np

def interp1d_linear_gpu(x: torch.Tensor,
                        xp: torch.Tensor,
                        fp: torch.Tensor) -> torch.Tensor:
    """
    A faster 1D linear interpolation approach using `torch.searchsorted`
    to avoid Python loops or NumPy calls. 
    This runs on GPU if x, xp, fp are on GPU.

    Steps (like np.interp):
      1) clamp 'x' to [xp[0], xp[-1]]
      2) i = searchsorted(xp, x_clamped)
      3) linear interpolation => y= y0 + w*(y1-y0), w= (x-x0)/(x1-x0).

    Args:
      x:  shape [N], query points
      xp: shape [M], sorted reference x-values
      fp: shape [M], reference y-values
    Returns:
      y:  shape [N], linearly interpolated output
    """
    # clamp x
    x_min, x_max = xp[0], xp[-1]
    x_clamped = torch.clamp(x, x_min, x_max)

    # searchsorted => i in [0..M-2]
    idx = torch.searchsorted(xp, x_clamped, right=False)
    idx = torch.clamp(idx, 0, xp.shape[0]-2)

    x0 = xp[idx]
    x1 = xp[idx+1]
    y0 = fp[idx]
    y1 = fp[idx+1]

    denom = x1 - x0
    denom = torch.where(denom == 0, torch.full_like(denom, 1e-12), denom)
    w = (x_clamped - x0) / denom
    y = y0 + w*(y1 - y0)
    return y


def batch_interp1d_linear(x: torch.Tensor,
                          xp: torch.Tensor,
                          fp: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper for multiple queries in 'x'. 
    Typically just calls interp1d_linear_gpu once,
    but you could add a loop if shape is 2D or bigger.

    Args:
      x:  shape [N], or possibly [N,*]
      xp: shape [M]
      fp: shape [M]
    Returns:
      y: shape [N], same shape or partial
    """
    return interp1d_linear_gpu(x, xp, fp)


def nearestInterp2d(sample_s: torch.Tensor,
                    pro_s1: torch.Tensor,
                    pro_s2: torch.Tensor,
                    pd_grid_uv: torch.Tensor) -> torch.Tensor:
    """
    Efficient nearest-neighbor 2D interpolation using pure PyTorch operations.
    
    Args:
        sample_s: shape [N, 2], query points
        pro_s1: shape [K], unique x-coordinates of the grid
        pro_s2: shape [K], unique y-coordinates of the grid
        pd_grid_uv: shape [K, K], values on the grid
    
    Returns:
        interp_values: shape [N], interpolated values at query points
    """
    device = sample_s.device
    dtype = sample_s.dtype
    
    # Extract dimensions
    N = sample_s.shape[0]
    K = pro_s1.shape[0]
    
    # For each query point, find the nearest grid point
    # This is done by finding the index in pro_s1 and pro_s2 that minimizes
    # the distance to the query point's x and y coordinates
    
    # Vectorized nearest neighbor finding
    # Create distance matrices
    # [K, N] matrices where each column is the distance from a query point
    # to all grid points along the specified axis
    x_dists = torch.abs(pro_s1.unsqueeze(1) - sample_s[:, 0].unsqueeze(0))  # shape [K, N]
    y_dists = torch.abs(pro_s2.unsqueeze(1) - sample_s[:, 1].unsqueeze(0))  # shape [K, N]
    
    # Find indices of nearest grid points
    x_indices = torch.argmin(x_dists, dim=0)  # shape [N]
    y_indices = torch.argmin(y_dists, dim=0)  # shape [N]
    
    # Use these indices to gather the interpolated values
    # This is equivalent to pd_grid_uv[x_indices, y_indices] but works for batched inputs
    interp_values = pd_grid_uv[x_indices, y_indices]
    
    return interp_values

def grid_sample_2d(pd_grid_uv: torch.Tensor,
                  pro_s1: torch.Tensor,
                  pro_s2: torch.Tensor,
                  sample_s: torch.Tensor,
                  mode: str = 'bilinear') -> torch.Tensor:
    """
    More advanced grid sampling using PyTorch's grid_sample function.
    This supports bilinear, bicubic, and nearest interpolation.
    
    Args:
        pd_grid_uv: shape [K, K], the 2D grid values
        pro_s1: shape [K], x coordinates
        pro_s2: shape [K], y coordinates
        sample_s: shape [N, 2], query points
        mode: 'bilinear', 'bicubic', or 'nearest'
    
    Returns:
        interp_values: shape [N], interpolated values
    """
    device = pd_grid_uv.device
    dtype = pd_grid_uv.dtype
    
    # We need to normalize sample_s to [-1, 1] range for grid_sample
    x_min, x_max = pro_s1.min(), pro_s1.max()
    y_min, y_max = pro_s2.min(), pro_s2.max()
    
    # Clamp query points to valid range
    sample_s_clamped = torch.zeros_like(sample_s)
    sample_s_clamped[:, 0] = torch.clamp(sample_s[:, 0], x_min, x_max)
    sample_s_clamped[:, 1] = torch.clamp(sample_s[:, 1], y_min, y_max)
    
    # Normalize to [-1, 1]
    sample_s_norm = torch.zeros_like(sample_s)
    sample_s_norm[:, 0] = 2.0 * (sample_s_clamped[:, 0] - x_min) / (x_max - x_min) - 1.0
    sample_s_norm[:, 1] = 2.0 * (sample_s_clamped[:, 1] - y_min) / (y_max - y_min) - 1.0
    
    # Reshape grid for grid_sample ([batch, channels, height, width])
    grid_reshaped = pd_grid_uv.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    
    # Format query points for grid_sample ([batch, height, width, 2])
    # Here we have a batch of 1, with N query points
    grid_coords = sample_s_norm.unsqueeze(0)  # [1, N, 2]
    
    # Use grid_sample
    # The output will be [1, 1, N, 1]
    output = torch.nn.functional.grid_sample(
        grid_reshaped, 
        grid_coords.unsqueeze(1),  # [1, 1, N, 2]
        mode=mode,
        align_corners=True
    )
    
    # Extract and reshape
    interp_values = output.squeeze()  # [N]
    
    return interp_values