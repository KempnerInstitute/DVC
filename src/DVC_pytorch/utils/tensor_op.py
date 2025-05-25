import torch
import numpy as np

#################### CHECK BOUNDARIES OF TENSOR ########################

def check_bound(data, mesh):
    """Clips tensor value to its minimum and maximum"""
    mesh = mesh.to(data.dtype)
    
    max_m = torch.max(mesh)
    min_m = torch.min(mesh)
    
    # Clamp data to the bounds
    data = torch.clamp(data, min=min_m, max=max_m)
    return data

def check_bound3(data, maxx, minn):
    """Replace values outside bounds"""
    eps = 1e-10
    data = torch.where(data >= maxx, maxx - eps, data)
    data = torch.where(data <= minn, minn + eps, data)
    return data

def constraints_bound(data, mesh):
    """Clips tensor value to its minimum and maximum with small random perturbation"""
    max_m = torch.max(mesh).to(data.dtype)
    min_m = torch.min(mesh).to(data.dtype)
    
    # Find indices where data exceeds bounds
    ind_max = data > max_m
    ind_min = data < min_m
    
    # Create random perturbations
    if ind_max.sum() > 0:
        rand_max = torch.randn(ind_max.sum(), dtype=data.dtype, device=data.device)
        data[ind_max] = max_m * (1 - 1e-10 * rand_max)
    
    if ind_min.sum() > 0:
        rand_min = torch.randn(ind_min.sum(), dtype=data.dtype, device=data.device)
        data[ind_min] = min_m * (1 + 1e-10 * rand_min)
    
    return data

def check_bound_and_nan(data, maxx, minn):
    """Replace nan, inf and out of bound values"""
    # Handle max bounds
    ind_max = data >= maxx
    if ind_max.sum() > 0:
        rand_max = torch.randn(ind_max.sum(), dtype=data.dtype, device=data.device)
        data[ind_max] = maxx * (1 - 1e-10 * rand_max)
    
    # Handle min bounds
    ind_min = data <= minn
    if ind_min.sum() > 0:
        rand_min = torch.randn(ind_min.sum(), dtype=data.dtype, device=data.device)
        data[ind_min] = minn * (1 + 1e-10 * rand_min)
    
    # Handle NaN values
    ind_nan = torch.isnan(data)
    if ind_nan.sum() > 0:
        rand_nan = torch.randn(ind_nan.sum(), dtype=data.dtype, device=data.device)
        data[ind_nan] = minn * (1 + 1e-10 * rand_nan)
    
    return data

##################### UNIQUE TENSORS #################################

def uniquetol(data, tol):
    """Return unique values that are all above a given tolerance"""
    y = torch.unique(data, sorted=True)
    if len(y) <= 1:
        return y
    
    # Calculate differences between consecutive unique values
    d = torch.abs(y[1:] - y[:-1])
    check = d > tol
    
    # Include first element and those that pass tolerance check
    isTol = torch.cat([torch.tensor([True], dtype=torch.bool, device=data.device), check])
    z = y[isTol]
    return z

#################### UPDATE TENSORS 2D ################################

def update_tensor2D(tensor, i, newval):
    """Update a column of a 2D tensor"""
    tensor = tensor.clone()
    tensor[:, i] = newval
    return tensor

################### UPDATE TENSOR 3D ################################

def update_tensor(tensor, newval, i, j):
    """Update a specific slice of a 3D tensor"""
    tensor = tensor.clone()
    tensor[:, i, j] = newval
    return tensor

################ REPLACE NEGATIVE/INF OR NAN ###########################

def replace_inf(data, newval):
    """Replace negative infinity values"""
    data = data.clone()
    mask = torch.logical_and(data < 0, torch.isinf(data))
    data[mask] = newval
    return data

def replace_negative(data, newval):
    """Replace negative values"""
    data = data.clone()
    data[data < 0] = newval
    return data

def replace_nan_inf(data):
    """Replace nan and inf values"""
    data = data.clone()
    data[torch.isnan(data)] = 0
    data[torch.isinf(data)] = torch.finfo(data.dtype).max
    return data

def replace_nan_with(data, newval):
    """Replace nan values with a specific value"""
    data = data.clone()
    data[torch.isnan(data)] = newval
    return data

def replace_inf_with(data, newval):
    """Replace inf values with a specific value"""
    data = data.clone()
    data[torch.isinf(data)] = newval
    return data

############################ MOVING AVERAGE ###############################

def moving_average(a, n):
    """Calculate moving average"""
    if n >= len(a):
        return a
    
    ret = torch.cumsum(a, dim=0)
    ret[n:] = ret[n:] - ret[:-n]
    
    # Create the smoothed tensor
    smoothed = a.clone()
    smoothed[:n] = ret[:n]
    smoothed[n:] = ret[n:]
    
    # Normalize by window size
    smoothed[n-1:] = smoothed[n-1:] / float(n)
    
    # Keep original values for the first n-1 elements
    result = torch.cat([a[:n-1], smoothed[n-1:]])
    
    return result

########################### Extend tensor from one dimension #################

def create_points(x, dim, exp_dim):
    """
    Create expansion points for prediction
    
    Args:
        x: Input data (n_samples, n_dims)
        dim: Dimension to expand
        exp_dim: Number of expansion points
        
    Returns:
        points: Expanded points tensor
    """
    device = x.device
    dtype = x.dtype
    n_samples, n_dims = x.shape
    
    # Create expansion range
    min_val = torch.min(x[:, dim])
    max_val = torch.max(x[:, dim])
    exp_range = torch.linspace(min_val - 2e-16 + 1e-5, max_val + 2e-16, 
                              exp_dim, dtype=dtype, device=device)
    
    # Create expanded points
    points = torch.zeros(n_samples * exp_dim, n_dims, dtype=dtype, device=device)
    
    for i in range(n_samples):
        start_idx = i * exp_dim
        end_idx = (i + 1) * exp_dim
        
        # Copy the original data
        points[start_idx:end_idx, :] = x[i, :].unsqueeze(0).expand(exp_dim, -1)
        
        # Replace the expanded dimension
        points[start_idx:end_idx, dim] = exp_range
    
    return points 