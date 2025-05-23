###############################################
# src/DVC/utils_prob.py
###############################################

import torch
import math
import numpy as np
from torch.distributions import Normal
from scipy import stats
from scipy.stats import norm
from .utils_tensor import replace_nan_inf

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
# Param Copula placeholders => Completed
# For "ind", "gaussian", "clayton", "claytonrot90", "student" partial
################################################

def copulapdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Evaluate PDF of the param copula 'cop_p' at points 'uv' shape [N,2].
    The logic mirrors the param_copula.py approach:
       family => 'ind','gaussian','clayton','claytonrot90','student'
    """
    fam = getattr(cop_p, 'family', 'ind')
    param = getattr(cop_p, 'theta', None)
    uv_clamped = torch.clamp(uv, 1e-9, 1 - 1e-9)

    # 'ind'
    if fam=='ind':
        # pdf=1
        return torch.ones(uv.shape[0], dtype=uv.dtype, device=uv.device)

    elif fam=='gaussian':
        rho = float(param)
        r = max(min(rho,0.999999), -0.999999)
        one_m_r2 = 1.0 - r*r
        if one_m_r2<1e-12:
            one_m_r2=1e-12
        normal_dist = torch.distributions.Normal(0.,1.)
        z = normal_dist.icdf(uv_clamped)  # shape [N,2]
        z1 = z[:,0]
        z2 = z[:,1]
        logC = -0.5*math.log(one_m_r2)
        num = z1*z1 - 2*r*z1*z2 + z2*z2
        den = 2*one_m_r2
        logpdf_part = -0.5*(num/den)
        logpdf = logC + logpdf_part
        return torch.exp(logpdf)

    elif fam=='student':
        # not fully implemented => raise or approximate
        raise NotImplementedError("Student PDF not fully implemented in 'utils_prob'.")

    elif fam=='clayton':
        alpha = float(param)
        u_ = uv_clamped[:,0]
        v_ = uv_clamped[:,1]
        u_m_alpha = torch.pow(u_, -alpha)
        v_m_alpha = torch.pow(v_, -alpha)
        sum_ = u_m_alpha + v_m_alpha - 1.0
        sum_ = torch.clamp(sum_, min=1e-14)
        # c_ = (alpha+1) * sum_^(-2 -1/alpha)* u_^(-alpha-1)*v_^(-alpha-1)
        c_ = (alpha+1.0) * sum_.pow(- (2.0 + 1.0/alpha)) \
              * (u_.pow(- (alpha+1.0))) * (v_.pow(- (alpha+1.0)))
        return replace_nan_inf(c_)

    elif fam=='claytonrot90':
        # flip => call 'clayton'
        alpha = float(param)
        uv_flip = uv_clamped.clone()
        uv_flip[:,0] = 1.0 - uv_clamped[:,0]
        # define a temp cop => same param but family='clayton'
        class TempCop:
            pass
        tmp = TempCop()
        tmp.family='clayton'
        tmp.theta=alpha
        return copulapdf(tmp, uv_flip)

    else:
        # default => 0
        return torch.zeros(uv.shape[0], dtype=uv.dtype, device=uv.device)


def copulaccdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the conditional CDF of a param copula: F(v|u).
    The logic is the same as param_copula's approach.
    """
    fam = getattr(cop_p, 'family', 'ind')
    param = getattr(cop_p, 'theta', None)
    uv_clamped = torch.clamp(uv, 1e-9, 1 - 1e-9)

    if fam=='ind':
        # For independence copula, F(v|u) = v
        return uv_clamped[:,1]

    elif fam=='gaussian':
        rho = float(param)
        r = max(min(rho, 0.999999), -0.999999)
        normal_dist = torch.distributions.Normal(0., 1.)
        
        # Transform to normal space
        z1 = normal_dist.icdf(uv_clamped[:, 0])
        z2 = normal_dist.icdf(uv_clamped[:, 1])
        
        # Conditional distribution: Z2|Z1 ~ N(rho*Z1, sqrt(1-rho^2))
        mean_cond = r * z1
        std_cond = math.sqrt(1 - r**2)
        
        # Standardize
        z_std = (z2 - mean_cond) / std_cond
        
        # Return CDF
        return normal_dist.cdf(z_std)

    elif fam=='student':
        raise NotImplementedError("Student conditional CDF not implemented in 'utils_prob'.")

    elif fam=='clayton':
        alpha = float(param)
        u = uv_clamped[:, 0]
        v = uv_clamped[:, 1]
        
        # h(v|u) = u^(-alpha-1) * (u^(-alpha) + v^(-alpha) - 1)^(-1/alpha - 1)
        u_neg_alpha = torch.pow(u, -alpha)
        v_neg_alpha = torch.pow(v, -alpha)
        sum_term = u_neg_alpha + v_neg_alpha - 1.0
        sum_term = torch.clamp(sum_term, min=1e-14)
        
        h_vu = torch.pow(u, -alpha - 1) * torch.pow(sum_term, -1.0/alpha - 1.0)
        return torch.clamp(h_vu, 0.0, 1.0)

    elif fam=='claytonrot90':
        alpha = float(param)
        uv_flip = uv_clamped.clone()
        uv_flip[:,0] = 1.0- uv_clamped[:,0]
        # temporary
        class TempCop:
            pass
        tmp = TempCop()
        tmp.family='clayton'
        tmp.theta=alpha
        h_flip = copulaccdf(tmp, uv_flip)
        # For 90 degree rotation, we need to adjust
        return 1.0 - h_flip

    else:
        return uv_clamped[:,1]


def copulainvccdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Inverse conditional cdf for param copula => sample from copula.
    For 2D: given U1=u and W=w (uniform), find U2 such that F(U2|U1=u) = w.
    """
    fam = getattr(cop_p, 'family', 'ind')
    param = getattr(cop_p, 'theta', None)
    uv_clamped = torch.clamp(uv, 1e-9, 1-1e-9)

    if fam=='ind':
        # For independence, F^{-1}(w|u) = w
        return uv_clamped[:,1]

    elif fam=='gaussian':
        # Y|X=x => Normal(r*x, sqrt(1-r^2)), then transform => cdf
        normal_dist = torch.distributions.Normal(0.,1.)
        rho = float(param)
        r = max(min(rho,0.999999), -0.999999)
        x = normal_dist.icdf(uv_clamped[:,0])
        e = normal_dist.icdf(uv_clamped[:,1])
        y = r*x + math.sqrt(1.0-r*r)* e
        v2 = normal_dist.cdf(y)
        return torch.clamp(v2, 0.0, 1.0)

    elif fam=='student':
        raise NotImplementedError("Student inverse conditional CDF not implemented in 'utils_prob'.")

    elif fam=='clayton':
        alpha = float(param)
        u1 = uv_clamped[:,0]
        w = uv_clamped[:,1]
        
        # Inverse of h-function for Clayton copula
        # v = [(1 - u^alpha + u^alpha * w^(-alpha/(alpha+1))]^(-1/alpha)
        u_alpha = torch.pow(u1, alpha)
        w_term = torch.pow(w, -alpha/(alpha + 1.0))
        inner = 1.0 - u_alpha + u_alpha * w_term
        inner = torch.clamp(inner, min=1e-14)
        v = torch.pow(inner, -1.0/alpha)
        return torch.clamp(v, 0.0, 1.0)

    elif fam=='claytonrot90':
        alpha = float(param)
        uv_flip = uv_clamped.clone()
        uv_flip[:,0] = 1.0 - uv_clamped[:,0]
        # define a temp cop
        class TempCop:
            pass
        tmp = TempCop()
        tmp.family='clayton'
        tmp.theta=alpha
        res = copulainvccdf(tmp, uv_flip)
        # for 90 deg => might do partial => 1.0 - res
        return 1.0 - res

    else:
        # unknown => just second
        return uv_clamped[:,1]