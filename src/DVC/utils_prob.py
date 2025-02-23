###############################################
# src/torch_vine/utils_prob.py
###############################################

import torch
import math
import numpy as np
from torch.distributions import Normal
from .utils_tensor import replace_nan_inf


def biv_norm(x1_s: torch.Tensor, x2_s: torch.Tensor) -> torch.Tensor:
    """
    Create a bivariate standard Normal PDF grid = outer product of 1D pdfs.
    x1_s, x2_s: shape [K].
    Return shape [K, K].
    """
    normal = Normal(loc=0.0, scale=1.0)
    p1 = normal.log_prob(x1_s).exp()  # [K]
    p2 = normal.log_prob(x2_s).exp()  # [K]
    grid = torch.ger(p1, p2)  # outer product => [K,K]
    return grid


def kernel_cdf(data: np.ndarray,
               query_y: np.ndarray,
               ex: np.ndarray):
    """
    Simple empirical cdf for 1D data (NumPy). Then interp on query_y.
    Return: (cdf_query, sorted_data, cdf_vals)
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
    Using histogram. 
    Return (density, mesh)
    """
    data_np = data.detach().cpu().numpy()
    mi, ma = data_np.min(), data_np.max()
    if mi == ma:
        mesh = np.linspace(mi - 1e-6, mi + 1e-6, npts)
        den = np.ones_like(mesh)
        den /= den.sum()
        return torch.from_numpy(den), torch.from_numpy(mesh)
    hist, bin_edges = np.histogram(data_np, bins=npts, density=True)
    midpoints = 0.5*(bin_edges[:-1] + bin_edges[1:])
    den_t = torch.from_numpy(hist)
    mesh_t = torch.from_numpy(midpoints)
    return den_t, mesh_t


def copulaccdf(cop_p, uv):
    """
    Evaluate the CDF of a param copula object cop_p at points uv [N,2].
    Placeholder.
    """
    pass


def copulapdf(cop_p, uv):
    """
    Evaluate the PDF of a param copula. 
    """
    pass


def copulainvccdf(cop_p, uv):
    """
    Inverse conditional CDF for sampling. 
    """
    pass