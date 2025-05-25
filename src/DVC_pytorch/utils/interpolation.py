import torch
import numpy as np

###################### NEAREST INTERPOLATION #####################

def nearestInterp2d(sample_s, pro_s1, pro_s2, pd_grid_uv):
    """Nearest neighbor interpolation on the grid"""
    len_sample = sample_s.shape[0]
    len_grid = pro_s1.shape[0]
    device = sample_s.device
    
    # Tile the grid coordinates
    pro_s1_tile = pro_s1.repeat(len_sample).reshape(len_sample, len_grid).t()
    pro_s2_tile = pro_s2.repeat(len_sample).reshape(len_sample, len_grid).t()
    
    # Tile the sample coordinates
    sample_s1_tile = sample_s[:, 0].repeat(len_grid).reshape(len_grid, len_sample)
    sample_s2_tile = sample_s[:, 1].repeat(len_grid).reshape(len_grid, len_sample)
    
    # Find nearest neighbors
    xi = torch.argmin(torch.abs(pro_s1_tile - sample_s1_tile), dim=0)
    yi = torch.argmin(torch.abs(pro_s2_tile - sample_s2_tile), dim=0)
    
    # Gather values from grid
    ind_int = torch.stack([xi, yi], dim=1)
    inter = pd_grid_uv[ind_int[:, 0], ind_int[:, 1]]
    
    return inter

######################### LINEAR INTERPOLATION #################

def interp1d_np(x, xref, yref):
    """1-D linear interpolation using numpy"""
    # Convert to numpy, interpolate, and convert back
    x_np = x.cpu().numpy() if torch.is_tensor(x) else x
    xref_np = xref.cpu().numpy() if torch.is_tensor(xref) else xref
    yref_np = yref.cpu().numpy() if torch.is_tensor(yref) else yref
    
    y_np = np.interp(x_np, xref_np, yref_np)
    
    # Convert back to tensor on the same device as x
    if torch.is_tensor(x):
        y = torch.from_numpy(y_np).to(x.device).to(x.dtype)
    else:
        y = y_np
    
    return y

def interp1d_torch(x, xp, fp):
    """
    1D linear interpolation in PyTorch (similar to numpy.interp)
    
    Args:
        x: Points to interpolate at
        xp: x-coordinates of data points (must be increasing)
        fp: y-coordinates of data points
        
    Returns:
        Interpolated values at x
    """
    device = x.device if torch.is_tensor(x) else torch.device('cpu')
    dtype = x.dtype if torch.is_tensor(x) else torch.float32
    
    # Convert to tensors if needed
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=dtype, device=device)
    if not torch.is_tensor(xp):
        xp = torch.tensor(xp, dtype=dtype, device=device)
    if not torch.is_tensor(fp):
        fp = torch.tensor(fp, dtype=dtype, device=device)
    
    # Ensure all on same device
    xp = xp.to(device)
    fp = fp.to(device)
    
    # Handle scalar input
    x_shape = x.shape
    x = x.flatten()
    
    # Find indices for interpolation
    indices = torch.searchsorted(xp, x)
    indices = torch.clamp(indices, 1, len(xp) - 1)
    
    # Get surrounding points
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]
    
    # Linear interpolation
    slope = (y1 - y0) / (x1 - x0 + 1e-10)
    result = y0 + slope * (x - x0)
    
    # Handle extrapolation
    result = torch.where(x < xp[0], fp[0], result)
    result = torch.where(x > xp[-1], fp[-1], result)
    
    return result.reshape(x_shape)

def interp_regular_nd_grid(points, grid_min, grid_max, values):
    """
    Interpolate on a regular N-D grid
    
    Args:
        points: Points to interpolate at (n_points, n_dims)
        grid_min: Minimum values for each dimension
        grid_max: Maximum values for each dimension  
        values: Grid values
        
    Returns:
        Interpolated values at points
    """
    device = points.device
    dtype = points.dtype
    
    # Ensure inputs are tensors
    if not torch.is_tensor(grid_min):
        grid_min = torch.tensor(grid_min, dtype=dtype, device=device)
    if not torch.is_tensor(grid_max):
        grid_max = torch.tensor(grid_max, dtype=dtype, device=device)
    
    # Get grid shape
    grid_shape = values.shape
    n_dims = len(grid_shape)
    
    # Normalize points to [0, 1]
    normalized = (points - grid_min) / (grid_max - grid_min + 1e-10)
    
    # Scale to grid indices
    scaled = normalized * (torch.tensor(grid_shape, dtype=dtype, device=device) - 1)
    
    # Get integer indices and fractions
    indices_low = torch.floor(scaled).long()
    indices_high = indices_low + 1
    fractions = scaled - indices_low.float()
    
    # Clamp indices
    for dim in range(n_dims):
        indices_low[:, dim] = torch.clamp(indices_low[:, dim], 0, grid_shape[dim] - 1)
        indices_high[:, dim] = torch.clamp(indices_high[:, dim], 0, grid_shape[dim] - 1)
    
    # For 2D case (most common for copulas)
    if n_dims == 2:
        # Get corner values
        v00 = values[indices_low[:, 0], indices_low[:, 1]]
        v01 = values[indices_low[:, 0], indices_high[:, 1]]
        v10 = values[indices_high[:, 0], indices_low[:, 1]]
        v11 = values[indices_high[:, 0], indices_high[:, 1]]
        
        # Bilinear interpolation
        fx = fractions[:, 0]
        fy = fractions[:, 1]
        
        v0 = v00 * (1 - fy) + v01 * fy
        v1 = v10 * (1 - fy) + v11 * fy
        result = v0 * (1 - fx) + v1 * fx
    else:
        # For higher dimensions, use nearest neighbor
        # (Full N-D interpolation would be more complex)
        nearest_indices = torch.round(scaled).long()
        for dim in range(n_dims):
            nearest_indices[:, dim] = torch.clamp(nearest_indices[:, dim], 0, grid_shape[dim] - 1)
        
        # Create tuple of indices for gathering
        idx_tuple = tuple(nearest_indices[:, i] for i in range(n_dims))
        result = values[idx_tuple]
    
    return result 