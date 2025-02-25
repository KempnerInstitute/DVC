###############################################
# src/DVC/utils_bandwidth.py
###############################################

import torch
import math
from .utils_tensor import check_bound3

def bandwidth_rule_of_thumb(data: torch.Tensor,
                            deg: int,
                            n_cop: int) -> torch.Tensor:
    """
    Compute a 'rule of thumb' bandwidth for each bivariate edge in 'data'.

    The typical formula: 
       factor = 5 * n^(-1/(4*deg + 2))
    and then multiply by stdev along each dimension (x,y).
    Then we scale by 1/10 for additional shrinking (per your original code).

    Args:
      data: shape [N, 2, n_cop], the data for each of n_cop edges
      deg:  typically 2 for bivariate
      n_cop: number of edges

    Returns:
      bw_matrix: shape [2, n_cop], where each column is the (bw_x, bw_y) 
                 for one copula edge.
    """
    # data has shape [N,2,n_cop]
    # for each i in [0..n_cop), we compute stdev in x, stdev in y
    N = data.shape[0]
    dtype_ = data.dtype
    device_ = data.device

    # factor
    factor = 5.0 * (N ** (-1.0 / (4.0 * deg + 2.0)))
    bw_matrix = torch.zeros((2, n_cop), dtype=dtype_, device=device_)

    for i in range(n_cop):
        dat_i = data[:, :, i]  # shape [N,2]
        # stdev along x,y => shape [2]
        stdevs = dat_i.std(dim=0)
        bw_matrix[:, i] = factor * stdevs

    # per your original code, scale down further by factor of 10
    bw_matrix = bw_matrix / 10.0
    return bw_matrix


def check_bound_bw(bw: torch.Tensor,
                   upper: float = 2.0,
                   lower: float = 1e-2) -> torch.Tensor:
    """
    Clamp the bandwidth values 'bw' to the range [lower+1e-10, upper-1e-10].

    The original code uses a small offset to avoid zero or extremely large values.
    Typically we do something like:
       bw = torch.clamp(bw, lower+1e-10, upper-1e-10)

    Args:
      bw: shape [2, n_cop] or [2, n_cop, n_bin], the bandwidth values
      upper: default 2.0
      lower: default 1e-2

    Returns:
      out: shape same as bw, with each value clamped to (lower+1e-10, upper-1e-10).
    """
    out = bw.clone()
    out = torch.clamp(out, lower + 1e-10, upper - 1e-10)
    return out