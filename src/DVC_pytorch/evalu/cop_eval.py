import torch
from utils.tensor_op import *

############################## COPULA PDF #####################################

def eval1(adu11_col1, adu22_1, t2, n_cop):
    """Compute normalization for copula PDF"""
    # Compute normalization
    I1 = torch.sum(adu22_1 * t2, dim=1)
    I2 = torch.sum(adu11_col1 * t2, dim=0)
    
    K5_list = []
    for i in range(n_cop):
        K1 = torch.outer(I1[:, i], I2[:, i])
        K5_list.append(K1)
    
    K5 = torch.stack(K5_list, dim=0)
    K5 = K5.permute(1, 2, 0)
    
    # Divide t2 by K5 (element-wise)
    t2 = t2 * torch.reciprocal(K5)
    
    # Handle NaN and Inf values
    if torch.any(torch.isnan(t2) | torch.isinf(t2)):
        t2_flat = t2.reshape(-1)
        t2_flat = replace_nan_inf(t2_flat)
        t2 = t2_flat.reshape(K5.shape[0], K5.shape[1], n_cop)
    
    return t2

def eval_rs_p(adu11, adu22, ker_fit, NORM1, n_cop):
    """Copula normalization for MISE cost function with 50 iterations"""
    device = ker_fit.device
    dtype = ker_fit.dtype
    
    # Make adu11 a column vector
    adu11_col = adu11.unsqueeze(-1)
    
    # Project on the u-v space
    t1 = ker_fit * torch.reciprocal(NORM1)
    
    # Check if any copula has very small values
    t1_reshaped = t1.reshape(t1.shape[0], t1.shape[1], -1)
    # Compute max along first two dimensions sequentially
    max_vals = torch.max(torch.max(t1_reshaped, dim=0)[0], dim=0)[0]
    if torch.any(max_vals < 1e-6):
        t2_list = []
        for i in range(t1.shape[2]):
            if torch.max(t1[:, :, i]) < 1e-6:
                # Set all copula values to one
                upd = torch.ones_like(t1[:, :, i])
                t2_list.append(upd)
            else:
                t2_list.append(t1[:, :, i])
        t1 = torch.stack(t2_list, dim=2)
    
    # Prepare adu22_1 and adu11_col1
    adu22_1 = adu22.unsqueeze(-1)
    adu22_1 = adu22_1.repeat(1, n_cop)
    
    adu11_col1 = adu11_col.repeat(1, n_cop)
    adu11_col1 = adu11_col1.reshape(adu11.shape[0], 1, n_cop)
    
    # Iterate 50 times
    for i in range(50):
        t1 = eval1(adu11_col1, adu22_1, t1, n_cop).reshape(t1.shape)
    
    # Final normalization
    adu11_col1 = adu11_col1.permute(1, 0, 2)
    II = torch.sum(adu11_col1 * torch.sum(adu22_1 * t1, dim=1), dim=1)
    t1 = t1 / II
    t1 = t1 * NORM1  # Project back on the r-s space
    
    return t1

def eval_rs_cop(adu11, adu22, ker_fit, NORM1, n_cop):
    """Copula normalization for MISE cost function with 500 iterations"""
    device = ker_fit.device
    dtype = ker_fit.dtype
    
    # Make adu11 a column vector
    adu11_col = adu11.unsqueeze(-1)
    
    # Project on the u-v space
    t1 = ker_fit * torch.reciprocal(NORM1)
    
    # Check if any copula has very small values
    t1_reshaped = t1.reshape(t1.shape[0], t1.shape[1], -1)
    # Compute max along first two dimensions sequentially
    max_vals = torch.max(torch.max(t1_reshaped, dim=0)[0], dim=0)[0]
    if torch.any(max_vals < 1e-6):
        t2_list = []
        for i in range(t1.shape[2]):
            if torch.max(t1[:, :, i]) < 1e-6:
                # Set all copula values to one
                upd = torch.ones_like(t1[:, :, i])
                t2_list.append(upd)
            else:
                t2_list.append(t1[:, :, i])
        t1 = torch.stack(t2_list, dim=2)
    
    # Prepare adu22_1 and adu11_col1
    adu22_1 = adu22.unsqueeze(-1)
    adu22_1 = adu22_1.repeat(1, n_cop)
    
    adu11_col1 = adu11_col.repeat(1, n_cop)
    adu11_col1 = adu11_col1.reshape(adu11.shape[0], 1, n_cop)
    
    # Iterate 500 times
    for i in range(500):
        t1 = eval1(adu11_col1, adu22_1, t1, n_cop).reshape(t1.shape)
    
    # Final normalization
    adu11_col1 = adu11_col1.permute(1, 0, 2)
    II = torch.sum(adu11_col1 * torch.sum(adu22_1 * t1, dim=1), dim=1)
    t1 = t1 / II
    t1 = t1 * NORM1  # Project back on the r-s space
    
    return t1

######################### COPULA CDF #####################################

def cdf_grid_fun(pd_grid_uv, ex_u, u1d, u2d, n_cop):
    """Compute the CDF on the grid"""
    knots = pd_grid_uv.shape[0]
    device = pd_grid_uv.device
    dtype = pd_grid_uv.dtype
    
    u2d = u2d.reshape(knots, 1, 1)
    u2d_tile = u2d.repeat(1, knots, n_cop)
    
    pd_grid_uv_transp = pd_grid_uv.reshape(knots, knots, n_cop).permute(1, 0, 2)
    integ = torch.cumsum(pd_grid_uv_transp * u2d_tile, dim=0)
    norm_p = torch.sum(pd_grid_uv * u2d_tile, dim=0)
    
    # Replace zeros
    ind_zeros = norm_p == 0
    norm_p[ind_zeros] = 1.0
    
    cdf1 = (integ / norm_p).permute(1, 0, 2)
    cdf1 = cdf1.reshape(-1)
    cdf1 = check_bound(cdf1, ex_u)
    cdf1 = cdf1.reshape(u1d.shape[0], u2d.shape[0], n_cop)
    
    return cdf1 