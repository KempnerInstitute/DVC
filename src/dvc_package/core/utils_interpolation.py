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


def interp_regular_nd_grid(x: torch.Tensor,
                          x_grid_min: torch.Tensor,
                          x_grid_max: torch.Tensor,
                          y_ref: torch.Tensor,
                          axis: int = -2) -> torch.Tensor:
    """
    Interpolate on a regular N-D grid (similar to TensorFlow's 
    tfp.math.batch_interp_regular_nd_grid).
    
    Args:
        x: Query points, shape [..., N, D]
        x_grid_min: Minimum grid values for each dimension, shape [D]
        x_grid_max: Maximum grid values for each dimension, shape [D]
        y_ref: Values on the regular grid
        axis: Axis for batch interpolation
        
    Returns:
        Interpolated values at query points
    """
    device = x.device
    dtype = x.dtype
    
    # Get dimensions
    batch_shape = x.shape[:-1]
    n_dims = x.shape[-1]
    
    # Normalize query points to [0, 1]
    x_normalized = (x - x_grid_min) / (x_grid_max - x_grid_min + 1e-8)
    x_normalized = torch.clamp(x_normalized, 0.0, 1.0)
    
    # For 2D case (most common in copulas)
    if n_dims == 2 and y_ref.dim() == 2:
        # Convert to grid_sample format
        # grid_sample expects coordinates in [-1, 1]
        x_grid_sample = 2.0 * x_normalized - 1.0
        
        # Reshape for grid_sample
        # y_ref: [H, W] -> [1, 1, H, W]
        y_ref_4d = y_ref.unsqueeze(0).unsqueeze(0)
        
        # x_grid_sample: [..., 2] -> [1, ..., 1, 2]
        x_flat = x_grid_sample.reshape(-1, 2)
        grid = x_flat.unsqueeze(0).unsqueeze(1)  # [1, 1, N, 2]
        
        # Interpolate
        output = F.grid_sample(
            y_ref_4d,
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        
        # Reshape back
        output = output.squeeze(0).squeeze(0).squeeze(0)
        output = output.reshape(batch_shape)
        
        return output
    else:
        # For general N-D case, use a simpler nearest neighbor approach
        # This is a placeholder - for production, implement proper N-D interpolation
        x_indices = (x_normalized * (torch.tensor(y_ref.shape[:n_dims], device=device) - 1)).long()
        x_indices = torch.clamp(x_indices, 0, torch.tensor(y_ref.shape[:n_dims], device=device) - 1)
        
        # Convert indices to flat index
        flat_indices = x_indices[..., 0]
        for d in range(1, n_dims):
            flat_indices = flat_indices * y_ref.shape[d] + x_indices[..., d]
        
        # Gather values
        y_flat = y_ref.flatten()
        output = y_flat[flat_indices]
        
        return output


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

def bilinearInterp2d(points: torch.Tensor,
                      x_axis: torch.Tensor,
                      y_axis: torch.Tensor,
                      grid_vals: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of *grid_vals* at arbitrary *points*.

    • points : [N,2] in [0,1]×[0,1] (u,v)
    • x_axis : [K] grid coordinates (assumed uniform)
    • y_axis : [K]
    • grid_vals : [K,K,E]  (E arbitrary features)
    Returns
    --------
    out : [N,E]
    """
    K = x_axis.numel()
    step_x = (x_axis[1]-x_axis[0]).item() if K>1 else 1.0
    step_y = (y_axis[1]-y_axis[0]).item() if K>1 else 1.0
    xi = (points[:,0] - x_axis[0]) / step_x
    yi = (points[:,1] - y_axis[0]) / step_y
    x0 = torch.clamp(xi.floor().long(), 0, K-2)
    y0 = torch.clamp(yi.floor().long(), 0, K-2)
    x1 = x0+1
    y1 = y0+1
    wx = (xi - x0.float()).unsqueeze(1)
    wy = (yi - y0.float()).unsqueeze(1)
    # gather four corners
    g00 = grid_vals[x0, y0]
    g10 = grid_vals[x1, y0]
    g01 = grid_vals[x0, y1]
    g11 = grid_vals[x1, y1]
    interp = (1-wx)*(1-wy)*g00 + wx*(1-wy)*g10 + (1-wx)*wy*g01 + wx*wy*g11
    return interp

def inverse_cdf_row(rand_u: torch.Tensor,
                     cdf_rows: torch.Tensor,
                     y_axis: torch.Tensor) -> torch.Tensor:
    """Vectorised 1-D inversion on multiple CDF rows.

    Parameters
    ----------
    rand_u   : [N]  uniform(0,1) values.
    cdf_rows : [N,K]  cumulative values monotonically increasing in last dim.
    y_axis   : [K]   y grid (monotone asc).

    Returns
    -------
    torch.Tensor  shape [N]  - sampled y values.
    """
    K = y_axis.numel()
    idx = torch.searchsorted(cdf_rows, rand_u.unsqueeze(1))  # [N,1]
    idx = idx.squeeze(1)
    idx = torch.clamp(idx, 1, K-1)
    idx0 = idx - 1
    c0 = cdf_rows.gather(1, idx0.unsqueeze(1)).squeeze(1)
    c1 = cdf_rows.gather(1, idx.unsqueeze(1)).squeeze(1)
    y0 = y_axis[idx0]
    y1 = y_axis[idx]
    w = (rand_u - c0) / (c1 - c0 + 1e-12)
    return y0 + w*(y1 - y0)