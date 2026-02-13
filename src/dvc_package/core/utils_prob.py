###############################################
# src/DVC/utils_prob.py
###############################################

import torch
import math
import logging
import numpy as np
from torch.distributions import Normal
from scipy import stats
from scipy.stats import norm
from .utils_tensor import replace_nan_inf

logger = logging.getLogger(__name__)

################################################
# Nonparam Bivariate Normal Reference
################################################

def biv_norm(x1_s: torch.Tensor, x2_s: torch.Tensor) -> torch.Tensor:
    """
    Create a bivariate standard Normal PDF grid = outer product of 1D pdfs.
    x1_s, x2_s: shape [K].
    Return: shape [K,K], a 2D grid of values ~ N(0,1) x N(0,1).
    """
    normal = Normal(loc=0.0, scale=1.0)
    p1 = normal.log_prob(x1_s).exp()  # [K]
    p2 = normal.log_prob(x2_s).exp()  # [K]
    # outer product => shape [K,K]
    grid = torch.ger(p1, p2)
    return grid


################################################
# Simple 1D Kernel CDF approach
################################################

def kernel_cdf(data: np.ndarray,
               query_y: np.ndarray,
               ex: np.ndarray):
    """
    Simple empirical cdf for 1D data (NumPy). Then we
    interpolate on 'query_y'.

    Steps:
      1) sort 'data'
      2) cdf_vals = (1..n)/(n+1)
      3) cdf_query = np.interp(query_y, sorted_data, cdf_vals)
      4) clamp to [1e-15, 1-1e-15]
    Returns:
      (cdf_query, sorted_data, cdf_vals)
    """
    sorted_data = np.sort(data)
    n = len(data)
    cdf_vals = np.arange(1, n+1, dtype=np.float64)/(n+1)
    cdf_query = np.interp(query_y, sorted_data, cdf_vals)
    cdf_query = np.clip(cdf_query, 1e-15, 1-1e-15)
    return cdf_query, sorted_data, cdf_vals


def kernel_cdf_batch(data: torch.Tensor,
                     query_y: torch.Tensor,
                     ex: torch.Tensor,
                     batch_size: int = 1000) -> torch.Tensor:
    """
    Batch version of kernel CDF estimation using PyTorch.
    
    Args:
        data: Data points, shape [N, d]
        query_y: Query points, shape [M, d]
        ex: Grid extent
        batch_size: Batch size for processing
        
    Returns:
        CDF values at query points, shape [M, d]
    """
    device = data.device
    N, d = data.shape
    M = query_y.shape[0]
    
    cdf_result = torch.zeros_like(query_y)
    
    # Process each dimension
    for dim in range(d):
        data_dim = data[:, dim]
        query_dim = query_y[:, dim]
        
        # Sort data for this dimension
        sorted_data, _ = torch.sort(data_dim)
        cdf_vals = torch.arange(1, N+1, dtype=data.dtype, device=device) / (N + 1)
        
        # Process queries in batches
        for i in range(0, M, batch_size):
            end_idx = min(i + batch_size, M)
            batch_query = query_dim[i:end_idx]
            
            # Find insertion points
            indices = torch.searchsorted(sorted_data, batch_query)
            indices = torch.clamp(indices, 0, N-1)
            
            # Linear interpolation
            idx_low = torch.clamp(indices - 1, 0, N-1)
            idx_high = torch.clamp(indices, 0, N-1)
            
            # Get values at boundaries
            low_vals = sorted_data[idx_low]
            high_vals = sorted_data[idx_high]
            cdf_low = cdf_vals[idx_low]
            cdf_high = cdf_vals[idx_high]
            
            # Interpolate
            weights = torch.where(
                high_vals > low_vals,
                (batch_query - low_vals) / (high_vals - low_vals + 1e-10),
                torch.zeros_like(batch_query)
            )
            
            cdf_interp = cdf_low + weights * (cdf_high - cdf_low)
            cdf_result[i:end_idx, dim] = torch.clamp(cdf_interp, 1e-15, 1-1e-15)
    
    return cdf_result


def kernel_pdf2(data: torch.Tensor,
                query_points: torch.Tensor,
                bandwidth: torch.Tensor,
                kernel_type: str = 'gaussian') -> torch.Tensor:
    """
    2D kernel density estimation.
    
    Args:
        data: Data points, shape [N, 2]
        query_points: Query points, shape [M, 2]
        bandwidth: Bandwidth for each dimension, shape [2]
        kernel_type: Type of kernel ('gaussian', 'epanechnikov')
        
    Returns:
        PDF values at query points, shape [M]
    """
    device = data.device
    N = data.shape[0]
    M = query_points.shape[0]
    
    # Expand dimensions for broadcasting
    data_expanded = data.unsqueeze(0)  # [1, N, 2]
    query_expanded = query_points.unsqueeze(1)  # [M, 1, 2]
    
    # Compute scaled distances
    diff = (query_expanded - data_expanded) / bandwidth  # [M, N, 2]
    
    if kernel_type == 'gaussian':
        # Gaussian kernel
        kernel_vals = torch.exp(-0.5 * (diff ** 2))  # [M, N, 2]
        kernel_prod = kernel_vals.prod(dim=2)  # [M, N]
        normalizer = 1.0 / (2.0 * math.pi * bandwidth.prod())
    elif kernel_type == 'epanechnikov':
        # Epanechnikov kernel
        u = torch.norm(diff, dim=2)  # [M, N]
        kernel_prod = torch.where(u < 1, 0.75 * (1 - u**2), torch.zeros_like(u))
        normalizer = 1.0 / bandwidth.prod()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Average over data points
    pdf_values = normalizer * kernel_prod.mean(dim=1)
    
    return pdf_values


def kernel_pdf1d(data: torch.Tensor, npts: int = 128):
    """
    A minimal 1D kernel/pdf approach. 
    Using histogram-based approximation.

    Returns:
      (density, mesh) as Tensors on CPU.
    """
    data_np = data.detach().cpu().numpy()
    mi, ma = data_np.min(), data_np.max()
    if mi == ma:
        # degenerate => small range
        mesh = np.linspace(mi - 1e-6, mi + 1e-6, npts)
        den = np.ones_like(mesh)
        den /= den.sum()
        return torch.from_numpy(den), torch.from_numpy(mesh)
    hist, bin_edges = np.histogram(data_np, bins=npts, density=True)
    midpoints = 0.5*(bin_edges[:-1] + bin_edges[1:])
    den_t = torch.from_numpy(hist)
    mesh_t = torch.from_numpy(midpoints)
    return den_t, mesh_t


################################################
# Parametric copula functions (PDF, CCDF, Inverse CCDF)
# Canonical implementations live in param_copula.py;
# re-exported here for backward compatibility.
################################################

from .param_copula import copulapdf, copulaccdf, copulainvccdf