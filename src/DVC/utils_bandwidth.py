###############################################
# src/DVC/utils_bandwidth.py
###############################################

import torch
import math
from .utils_tensor import check_bound3

def bandwidth_rule_of_thumb(data: torch.Tensor, deg: int, n_cop: int) -> torch.Tensor:
    """
    'Rule of thumb' bandwidth. 
    data: shape [N, 2, n_cop]
    We'll do a quick approach:
       factor = 5 * n^(-1/(4*deg + 2)) 
       then multiply by stdev along each dimension.
    """
    N = data.shape[0]
    dtype_ = data.dtype
    device_ = data.device

    factor = 5.0 * (N ** (-1.0 / (4.0*deg + 2.0)))
    bw_matrix = torch.zeros((2, n_cop), dtype=dtype_, device=device_)

    for i in range(n_cop):
        dat_i = data[:, :, i]  # shape [N,2]
        stdevs = dat_i.std(dim=0)
        bw_matrix[:, i] = factor * stdevs
    # scale down by 10
    bw_matrix = bw_matrix / 10.0
    return bw_matrix


def check_bound_bw(bw: torch.Tensor, upper=2.0, lower=1e-2) -> torch.Tensor:
    """
    Bound the bandwidth. 
    """
    out = torch.clone(bw)
    out = torch.clamp(out, lower + 1e-10, upper - 1e-10)
    return out