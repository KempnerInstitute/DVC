###############################################
# src/DVC/utils_prob.py
###############################################

import torch
import math
import numpy as np
from torch.distributions import Normal
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
    Evaluate the CDF of a param copula. 
    The logic is the same as param_copula's approach.
    """
    fam = getattr(cop_p, 'family', 'ind')
    param = getattr(cop_p, 'theta', None)
    uv_clamped = torch.clamp(uv, 1e-9, 1 - 1e-9)

    if fam=='ind':
        return uv_clamped[:,0]*uv_clamped[:,1]

    elif fam=='gaussian':
        import math
        from scipy.stats import multivariate_normal
        rho = float(param)
        # do bivariate normal cdf
        out_list = []
        for i in range(uv_clamped.shape[0]):
            uval = uv_clamped[i,0].item()
            vval = uv_clamped[i,1].item()
            x = norm.ppf(uval)
            y = norm.ppf(vval)
            mean_ = [0.0,0.0]
            cov_ = [[1.0, rho],[rho,1.0]]
            cdf_val = multivariate_normal.cdf([x,y], mean=mean_, cov=cov_)
            out_list.append(cdf_val)
        return torch.tensor(out_list, dtype=uv.dtype, device=uv.device)

    elif fam=='student':
        raise NotImplementedError("Student CDF not implemented in 'utils_prob'.")

    elif fam=='clayton':
        alpha = float(param)
        u_ = uv_clamped[:,0]
        v_ = uv_clamped[:,1]
        sum_ = u_.pow(-alpha)+ v_.pow(-alpha)-1.0
        sum_ = torch.clamp(sum_, min=0.0)
        cdf_ = sum_.pow(-1.0/ alpha)
        cdf_ = torch.clamp(cdf_, 0.0, 1.0)
        return cdf_

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
        cdf_flip = copulaccdf(tmp, uv_flip)
        return cdf_flip

    else:
        return torch.zeros(uv.shape[0], dtype=uv.dtype, device=uv.device)


def copulainvccdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Inverse conditional cdf for param copula => sample from copula.
    For 2D: given U1=..., find U2 = F^-1( c2 | U1 ).
    """
    fam = getattr(cop_p, 'family', 'ind')
    param = getattr(cop_p, 'theta', None)
    uv_clamped = torch.clamp(uv, 1e-9, 1-1e-9)

    if fam=='ind':
        # trivially => second = uv[:,1]
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
        raise NotImplementedError("Student inverse CCDF not implemented in 'utils_prob'.")

    elif fam=='clayton':
        alpha = float(param)
        u1 = uv_clamped[:,0]
        c2 = uv_clamped[:,1]
        # formula => ( c2^( -alpha/(1+ alpha)) - u1^-alpha +1 )^(-1/ alpha)
        u1_m_alpha = torch.pow(u1, -alpha)
        c2_pow = torch.pow(c2, -alpha/(1.0+ alpha))
        val = c2_pow - u1_m_alpha + 1.0
        val = torch.clamp(val, min=1e-14)
        u2 = torch.pow(val, -1.0/ alpha)
        return torch.clamp(u2, 0.0, 1.0)

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