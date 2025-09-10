##################################################
# src/DVC/sampling.py
##################################################

import torch
import numpy as np
from typing import Tuple, Optional
from scipy.stats import kendalltau

from .utils_prob import kernel_cdf, copulainvccdf, copulaccdf
from .dataset_ops import create_bins, check_bins
from .vine_tree import parent_var
from .transformation import Transform
from .vine_eval import evaluate_points
from .preparation import prep_copula
from .utils_interpolation import interp1d_linear_gpu


def kerncopccdfinv(w: torch.Tensor, cdf_grid: torch.Tensor, 
                   u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
    """
    Inverse CDF for kernel copulas - sample from the copula.
    
    Args:
        w: Random uniform values, shape [N, 2]
        cdf_grid: CDF grid values
        u1, u2: Grid axes
        
    Returns:
        U2 values sampled from the copula
    """
    device = w.device
    len_w = w.shape[0]
    len_ax = u1.shape[0]
    
    # Tile for broadcasting
    u1_tile = u1.repeat(len_w, 1).t()  # [len_ax, len_w]
    w0_tile = w[:, 0].repeat(len_ax, 1)  # [len_ax, len_w]
    
    # Find nearest indices
    m1 = torch.argmin(torch.abs(u1_tile - w0_tile), dim=0)
    
    # Gather CDF values
    g = cdf_grid[m1, :].t()  # [len_ax, len_w]
    
    # Find where CDF exceeds w[:,1]
    w1_tile = w[:, 1].repeat(len_ax, 1)  # [len_ax, len_w]
    propro = g - w1_tile
    
    # Find first index where propro > 0
    mask1 = (propro > 0).int()
    ind = torch.argmax(mask1, dim=0)
    
    # Return corresponding u2 values
    U2 = u2[ind]
    return U2


def vine_copula_sample(vine, cases: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample from a non-parametric vine copula.
    
    Args:
        vine: Vine copula object
        cases: Number of samples to generate
        
    Returns:
        sample1: Samples in original space
        u: Samples in uniform space
        sample_pdf: PDF grid values
        sample_pds: PDS grid values
    """
    d = len(vine.r_matrix) if hasattr(vine, 'r_matrix') and vine.r_matrix is not None else vine.n_cop
    n = d - 1
    depth = vine.vine_depth if hasattr(vine, 'vine_depth') else d - 1
    
    # Generate uniform random values
    w = np.random.uniform(0, 1, (cases, d))
    
    # Get grid bounds
    if hasattr(vine, 'grid_u') and vine.grid_u is not None:
        mag = np.max(vine.grid_u.ex.cpu().numpy()) if torch.is_tensor(vine.grid_u.ex) else np.max(vine.grid_u.ex)
        mig = np.min(vine.grid_u.ex.cpu().numpy()) if torch.is_tensor(vine.grid_u.ex) else np.min(vine.grid_u.ex)
    else:
        mag, mig = 1.0, 0.0
    
    # Scale w to grid bounds
    w = (mag - mig) * (w - np.min(w)) / (np.max(w) - np.min(w)) + mig
    w = w.astype(np.float32)
    
    # Initialize arrays
    v = np.zeros([cases, d, d], w.dtype)
    v_flip = np.zeros([cases, d, d], w.dtype)
    v[:, 0, 0] = w[:, 0]
    
    # Get grid axes
    if hasattr(vine, 'grid_u'):
        u1 = vine.grid_u.ax1
        u2 = vine.grid_u.ax2
        device = u1.device if torch.is_tensor(u1) else torch.device('cpu')
    else:
        device = torch.device('cpu')
        u1 = torch.linspace(0, 1, 50, device=device)
        u2 = torch.linspace(0, 1, 50, device=device)
    
    # Initialize flip flags
    flip_flag1 = []
    for ii in range(1, d-1):
        flip_flag2 = []
        for j in range(ii):
            flip_flag2.append([])
        flip_flag1.append(flip_flag2)
    
    # Sample each column
    for i in range(1, d):
        v[:, i, i] = w[:, i]
        
        c = 0
        for k in range(i-1, -1, -1):
            tr = k
            col = i - k - 1
            
            if hasattr(vine, 'ind_vine') and k < len(vine.ind_vine):
                ind_now = vine.ind_vine[k][c] if c < len(vine.ind_vine[k]) else [0, 1]
            else:
                ind_now = [0, 1]
            
            if hasattr(vine, 'ind_edge_rel') and tr < len(vine.ind_edge_rel):
                ind_array = np.array(vine.ind_edge_rel[tr])
                ind_col = np.where(ind_array == col)[0]
                col = ind_col[0] if len(ind_col) > 0 else col
            
            if k == 0:
                # First tree level
                if hasattr(vine, 'r_matrix') and hasattr(vine, 'nodes'):
                    tr1 = n - k
                    col1 = n - i
                    ind1 = vine.r_matrix[tr1, col1] if tr1 < vine.r_matrix.shape[0] and col1 < vine.r_matrix.shape[1] else 1
                    ind1 = np.where(vine.nodes == ind1)[0]
                    ind1 = ind1[0] if len(ind1) > 0 else 0
                else:
                    ind1 = k
                
                v2 = v[:, k, ind1][:, np.newaxis]
                v1 = v[:, k+1, i][:, np.newaxis]
                vv = torch.from_numpy(np.concatenate((v2, v1), 1)).to(device)
                
                # Apply inverse CDF
                if tr <= depth:
                    if hasattr(vine, 'copulas') and tr < len(vine.copulas) and col < len(vine.copulas[tr]):
                        cop = vine.copulas[tr][col] if not isinstance(vine.copulas[tr], list) else vine.copulas[tr][col]
                        if hasattr(cop, 'cdf') and cop.cdf is not None:
                            cdf_grid = cop.cdf[:, :, col] if cop.cdf.dim() > 2 else cop.cdf
                            v[:, k, i] = kerncopccdfinv(vv, cdf_grid, u1, u2).cpu().numpy()
                        else:
                            # Fallback to parametric
                            v[:, k, i] = copulainvccdf(cop, vv).cpu().numpy()
                    else:
                        v[:, k, i] = vv[:, 1].cpu().numpy()
                else:
                    # Independent copula
                    v[:, k, i] = vv[:, 1].cpu().numpy()
                    
            else:
                # Higher tree levels with conditioning
                if hasattr(vine, 'ind_vine'):
                    parent, inx1, inx2 = parent_var(k, vine.ind_vine, ind_now)
                else:
                    parent = 0
                    
                # Determine which v to use based on flipping
                if hasattr(vine, 'ind_vine') and k > 0 and ind_now[0] < len(vine.ind_vine[k-1]):
                    if vine.ind_vine[k-1][ind_now[0]][0] != parent:
                        v2 = v_flip[:, k, k+ind_now[0]][:, np.newaxis]
                    else:
                        v2 = v[:, k, k+ind_now[0]][:, np.newaxis]
                else:
                    v2 = v[:, k, k][:, np.newaxis]
                    
                v1 = v[:, k+1, i][:, np.newaxis]
                vv = np.concatenate((v2, v1), 1)
                
                # Apply appropriate copula
                if tr > depth:
                    # Parametric copula
                    if vine.binning == False:
                        if hasattr(vine, 'copulas') and tr < len(vine.copulas) and col < len(vine.copulas[tr]):
                            cop = vine.copulas[tr][col]
                            v[:, k, i] = copulainvccdf(cop, torch.from_numpy(vv).to(device)).cpu().numpy()
                        else:
                            v[:, k, i] = vv[:, 1]
                    else:
                        # Binning case - not fully implemented
                        v[:, k, i] = vv[:, 1]
                else:
                    # Non-parametric copula
                    if vine.binning == False:
                        if hasattr(vine, 'copulas') and tr < len(vine.copulas):
                            cop = vine.copulas[tr]
                            if hasattr(cop, 'cdf') and cop.cdf is not None:
                                cdf_grid = cop.cdf[:, :, col] if cop.cdf.dim() > 2 else cop.cdf
                                vv_tensor = torch.from_numpy(vv).to(device)
                                v[:, k, i] = kerncopccdfinv(vv_tensor, cdf_grid, u1, u2).cpu().numpy()
                            else:
                                v[:, k, i] = vv[:, 1]
                        else:
                            v[:, k, i] = vv[:, 1]
                    else:
                        # Binning case - simplified implementation
                        v[:, k, i] = vv[:, 1]
            
            c += 1
        
        # Update v_flip for next iterations
        if i < d - 1:
            # Simplified flipping logic
            for ii in range(1, i+1):
                for j in range(ii):
                    # Placeholder for full flipping logic
                    pass
    
    # Extract final samples
    u = np.reshape(v[:, 0, :], w.shape)
    u1 = np.zeros_like(u)
    
    # Reorder based on r_matrix if available
    if hasattr(vine, 'r_matrix'):
        for i in range(d-1, -1, -1):
            ind = vine.r_matrix[i, i] - 1 if i < vine.r_matrix.shape[0] else i
            u1[:, ind] = u[:, d-1-i]
    else:
        u1 = u
    
    # Add small noise to avoid exact grid values
    if hasattr(vine, 'grid_u'):
        u_ax = vine.grid_u.ax1.cpu().numpy() if torch.is_tensor(vine.grid_u.ax1) else vine.grid_u.ax1
        gr_diff = vine.grid_u.diff1.cpu().numpy() if torch.is_tensor(vine.grid_u.diff1) else np.diff(u_ax)
        
        for i in range(d):
            u_p1 = u1[:, i][:, np.newaxis]
            u_ax1 = np.tile(u_ax[:, np.newaxis], [1, u_p1.shape[0]]).T
            u_upd = np.tile(u_p1, [1, u_ax.shape[0]])
            u_diff = np.abs(u_ax1 - u_upd)
            ind1 = np.argmin(u_diff, axis=1)
            diff_val = gr_diff[np.minimum(ind1, len(gr_diff)-1)]
            u1[:, i] = u1[:, i] + diff_val * np.random.uniform(0., 1., diff_val.shape[0])
    
    u = u1
    
    # Transform to original space using margins
    sample1 = np.zeros([cases, d], w.dtype)
    sample_pdf = np.zeros([cases, d], w.dtype)
    sample_pds = np.zeros([cases, d], w.dtype)
    
    for i in range(d):
        if hasattr(vine, 'Mar_G') and vine.Mar_G is not None and i < len(vine.Mar_G):
            mar_s1, mar_p1 = vine.Mar_G[i]
            mar_s1 = mar_s1 if isinstance(mar_s1, np.ndarray) else mar_s1.cpu().numpy()
            mar_p1 = mar_p1 if isinstance(mar_p1, np.ndarray) else mar_p1.cpu().numpy()
            
            # Interpolate to get samples
            sample1_pro = np.interp(u[:, i], mar_p1, mar_s1)
            sample1[:, i] = prep_copula(sample1_pro, 0)
            sample_pdf[:, i] = mar_p1
            sample_pds[:, i] = mar_s1
        else:
            # Use margin distribution if available
            if hasattr(vine, 'margin') and i < len(vine.margin):
                margin = vine.margin[i]
                if margin.dist == 'norm':
                    loc, scale = margin.theta
                    from scipy.stats import norm
                    sample1[:, i] = norm.ppf(u[:, i], loc=loc, scale=scale)
                else:
                    sample1[:, i] = u[:, i]
            else:
                sample1[:, i] = u[:, i]
    
    return sample1, u, sample_pdf, sample_pds


def vine_cop_par_sample(vine, cases: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample from a parametric vine copula.
    
    Args:
        vine: Vine copula object
        cases: Number of samples to generate
        
    Returns:
        sample1: Samples in original space  
        u: Samples in uniform space
        sample_pdf: PDF grid values
        sample_pds: PDS grid values
    """
    d = len(vine.r_matrix) if hasattr(vine, 'r_matrix') and vine.r_matrix is not None else vine.n_cop
    n = d - 1
    
    # Generate uniform random values
    w = np.random.uniform(0, 1, (cases, d))
    
    # Get grid bounds
    if hasattr(vine, 'grid_u') and vine.grid_u is not None:
        mag = np.max(vine.grid_u.ex.cpu().numpy()) if torch.is_tensor(vine.grid_u.ex) else np.max(vine.grid_u.ex)
        mig = np.min(vine.grid_u.ex.cpu().numpy()) if torch.is_tensor(vine.grid_u.ex) else np.min(vine.grid_u.ex)
    else:
        mag = 0.999
        mig = 0.001
    
    # Scale w to avoid boundaries
    w = (mag - mig) * (w - np.min(w)) / (np.max(w) - np.min(w)) + mig
    w = w.astype(np.float32)
    
    # For parametric, we can use a simpler approach
    # This is a placeholder - full implementation would follow the vine structure
    u = w.copy()
    
    # Transform to original space
    sample1 = np.zeros_like(u)
    for i in range(d):
        if hasattr(vine, 'margin') and i < len(vine.margin):
            margin = vine.margin[i]
            if margin.dist == 'norm':
                loc, scale = margin.theta
                from scipy.stats import norm
                sample1[:, i] = norm.ppf(u[:, i], loc=loc, scale=scale)
            else:
                sample1[:, i] = u[:, i]
        else:
            sample1[:, i] = u[:, i]
    
    # Return empty arrays for pdf/pds in parametric case
    sample_pdf = np.zeros_like(u)
    sample_pds = np.zeros_like(u)
    
    return sample1, u, sample_pdf, sample_pds 