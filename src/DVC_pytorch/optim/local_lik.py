import torch
import math as m
from utils.tensor_op import replace_nan_inf

########################## COMPUTE LOCAL LIKELIHOOD ################################

def loclik_batch(B, data, grid_points, n_cop, batch_size):
    """
    Compute local likelihood in batches
    
    Args:
        B: Bandwidth tensor of shape (2, n_cop)
        data: Data tensor of shape (n_samples, 2, n_cop)
        grid_points: Grid points tensor
        n_cop: Number of copulas
        batch_size: Batch size for processing
        
    Returns:
        ker_grid_fin: Kernel density estimates on grid
    """
    device = data.device
    dtype = data.dtype
    
    batch_len = grid_points.shape[0] // batch_size
    
    ker_grid1_list = []
    ker_grid2_list = []
    ker_grid3_list = []
    ker_grid4_list = []
    ker_grid5_list = []
    
    for i in range(batch_size):
        start_idx = batch_len * i
        end_idx = batch_len * (i + 1) if i < batch_size - 1 else grid_points.shape[0]
        
        pp1, pp2, pp3, pp4, pp5 = dense_naive_batch(B, data, grid_points[start_idx:end_idx])
        
        ker_grid1_list.append(pp1)
        ker_grid2_list.append(pp2)
        ker_grid3_list.append(pp3)
        ker_grid4_list.append(pp4)
        ker_grid5_list.append(pp5)
    
    # Concatenate results
    ker_grid1 = torch.cat(ker_grid1_list, dim=0)
    ker_grid2 = torch.cat(ker_grid2_list, dim=0)
    ker_grid3 = torch.cat(ker_grid3_list, dim=0)
    ker_grid4 = torch.cat(ker_grid4_list, dim=0)
    ker_grid5 = torch.cat(ker_grid5_list, dim=0)
    
    # Compute final kernel estimates
    ker_grid_fin = kern_LL(B, ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5)
    
    return ker_grid_fin

def loclik_batch_eval(B, data, grid_points, n_cop, batch_size):
    """
    Compute local likelihood in batches for evaluation (with higher precision)
    
    Args:
        B: Bandwidth tensor
        data: Data tensor
        grid_points: Grid points tensor
        n_cop: Number of copulas
        batch_size: Batch size
        
    Returns:
        ker_grid_fin: Kernel density estimates
    """
    # Convert to float64 for higher precision
    original_dtype = B.dtype
    B = B.double()
    data = data.double()
    grid_points = grid_points.double()
    
    # Call the batch function
    ker_grid_fin = loclik_batch(B, data, grid_points, n_cop, batch_size)
    
    # Convert back to original dtype
    ker_grid_fin = ker_grid_fin.to(original_dtype)
    
    return ker_grid_fin

def kern_LL(B, ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5):
    """
    Compute kernel local likelihood from grid statistics
    
    Args:
        B: Bandwidth tensor of shape (2, n_cop)
        ker_grid1-5: Grid statistics tensors
        
    Returns:
        ker_grid_fin: Final kernel estimates
    """
    # Compute error terms
    e1 = B[0, :] * torch.sqrt(torch.abs((ker_grid4 / ker_grid1) - (ker_grid2 / ker_grid1)**2))
    e2 = B[1, :] * torch.sqrt(torch.abs((ker_grid5 / ker_grid1) - (ker_grid3 / ker_grid1)**2))
    
    e1 = replace_nan_inf(e1)
    e2 = replace_nan_inf(e2)
    
    # Compute C term
    C = (-e1**2 * ((ker_grid2 / ker_grid1)**2 / (2 * B[0, :]**2)) - 
         e2**2 * ((ker_grid3 / ker_grid1)**2 / (2 * B[1, :]**2)))
    
    C = replace_nan_inf(C)
    
    # Final kernel estimate
    ker_grid_fin = ker_grid1 * e1 * e2 * torch.exp(C)
    
    return ker_grid_fin

def dense_naive_batch(B, data_p, grid_point):
    """
    Compute kernel density estimates using naive method
    
    Args:
        B: Bandwidth tensor of shape (2, n_cop)
        data_p: Data points tensor of shape (n_samples, 2, n_cop)
        grid_point: Grid points tensor of shape (n_grid, 2, n_cop)
        
    Returns:
        ker_grid1-5: Kernel statistics
    """
    device = data_p.device
    dtype = data_p.dtype
    
    d = data_p.shape[2]  # n_cop
    d_n = data_p.shape[0]  # n_samples
    
    # Reshape for broadcasting
    # grid_point: (n_grid, 2, n_cop) -> (1, n_grid, 2, n_cop)
    gr1_tile = grid_point.unsqueeze(0)
    # data_p: (n_samples, 2, n_cop) -> (n_samples, 1, 2, n_cop)
    d1_tile = data_p.unsqueeze(1)
    
    # Compute differences
    c = gr1_tile - d1_tile  # (n_samples, n_grid, 2, n_cop)
    
    # Constants
    d_n = torch.tensor(d_n, dtype=dtype, device=device)
    pi = torch.tensor(m.pi, dtype=dtype, device=device)
    
    # Compute kernel values
    # Note: c[:, :, 0, :] is x-differences, c[:, :, 1, :] is y-differences
    a = (torch.exp(-(c[:, :, 0, :]**2) / (2 * B[0, :]**2)) * 
         torch.exp(-(c[:, :, 1, :]**2) / (2 * B[1, :]**2)) / 
         (2 * pi * B[0, :] * B[1, :] * d_n))
    
    # Compute statistics
    ker_grid1 = torch.sum(a, dim=0)
    ker_grid2 = torch.sum(a * c[:, :, 0, :], dim=0)
    ker_grid3 = torch.sum(a * c[:, :, 1, :], dim=0)
    ker_grid4 = torch.sum(a * c[:, :, 0, :]**2, dim=0)
    ker_grid5 = torch.sum(a * c[:, :, 1, :]**2, dim=0)
    
    return ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5 