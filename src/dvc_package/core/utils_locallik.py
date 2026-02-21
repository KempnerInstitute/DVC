###############################################
# src/DVC/utils_locallik.py
###############################################

import torch
import math
import numpy as np
import logging
from .utils_tensor import replace_nan_inf

logger = logging.getLogger("DVC.locallik")


def dense_naive_batch(B: torch.Tensor,
                      data_p: torch.Tensor,
                      grid_points: torch.Tensor):
    """
    Prepare partial sums for the local-likelihood (the "naive" kernel approach).

    For each grid point 'g' and data point 'x':
      delta_x = g_x - x_x
      delta_y = g_y - x_y
      exponent = exp( - [ delta_x^2/(2*bw_x^2) + delta_y^2/(2*bw_y^2) ] )
      normal_factor = 1 / (2 * pi * bw_x * bw_y * N)
      a = normal_factor * exponent

    Returns partial sums over data points:
      ker_grid1(g) = sum_n a
      ker_grid2(g) = sum_n a * delta_x
      ker_grid3(g) = sum_n a * delta_y
      ker_grid4(g) = sum_n a * delta_x^2
      ker_grid5(g) = sum_n a * delta_y^2

    Args:
      B: shape [2, n_cop], the bandwidth parameters.
         B[0,i] => bw_x for copula i, B[1,i] => bw_y for copula i
      data_p: shape [N, 2, n_cop], data points
      grid_points: shape [M, 2, n_cop], grid points for evaluation

    Returns:
      ker_grid1..5: each shape [M, n_cop], partial sums used for local-likelihood
    """
    N = data_p.shape[0]
    M = grid_points.shape[0]
    n_cop = data_p.shape[2]

    # Expand data and grid for broadcasting:
    # data_exp => [N,1,2,n_cop], grid_exp => [1,M,2,n_cop]
    data_exp = data_p.unsqueeze(1).expand(-1, M, -1, -1)
    grid_exp = grid_points.unsqueeze(0).expand(N, -1, -1, -1)

    # c => difference array shape [N, M, 2, n_cop]
    c = grid_exp - data_exp

    # Bandwidth shapes => [2,n_cop], broadcast to [1,1,n_cop] to match
    # the (N,M,n_cop) layout produced by slicing `c[:, :, k, :]`.
    #
    # Note: using a 4D bandwidth tensor here would trigger an extra leading
    # singleton dimension via broadcasting (PyTorch aligns shapes from the right),
    # which breaks the intended reductions over the data dimension.
    if B.dim() == 1:
        B = B.view(2, 1)
    b0 = B[0, :].view(1, 1, n_cop)  # bw_x
    b1 = B[1, :].view(1, 1, n_cop)  # bw_y

    # exponent = exp( - ( delta_x^2 / (2*b0^2) + delta_y^2 / (2*b1^2) ) )
    val_x = (c[:, :, 0, :] ** 2) / (2.0 * b0 ** 2)  # shape [N,M,n_cop]
    val_y = (c[:, :, 1, :] ** 2) / (2.0 * b1 ** 2)  # shape [N,M,n_cop]
    val_exp = torch.exp(-(val_x + val_y))

    # factor => 1/(2*pi*bw_x*bw_y*N)
    const = 1.0 / (2.0 * math.pi) * (1.0 / (b0 * b1)) * (1.0 / float(N))  # [1,1,n_cop]
    # broadcast => a => shape [N,M,n_cop]
    a = val_exp * const

    # partial sums along data dimension (axis=0)
    ker_grid1 = a.sum(dim=0)                             # shape [M,n_cop]
    ker_grid2 = (a * c[:, :, 0, :]).sum(dim=0)
    ker_grid3 = (a * c[:, :, 1, :]).sum(dim=0)
    ker_grid4 = (a * (c[:, :, 0, :] ** 2)).sum(dim=0)
    ker_grid5 = (a * (c[:, :, 1, :] ** 2)).sum(dim=0)

    return ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5


def kern_LL(B: torch.Tensor,
            ker_grid1: torch.Tensor,
            ker_grid2: torch.Tensor,
            ker_grid3: torch.Tensor,
            ker_grid4: torch.Tensor,
            ker_grid5: torch.Tensor):
    """
    Final local-likelihood correction step.

    Computes adjusted kernel density using local polynomial correction:
      e1 = B[0] * sqrt(|var_x - mean_x^2|)
      e2 = B[1] * sqrt(|var_y - mean_y^2|)
      C  = -(e1^2 * mean_x^2 / (2*b0^2)) - (e2^2 * mean_y^2 / (2*b1^2))
      ker_grid_fin = ker_grid1 * e1 * e2 * exp(C)

    Args:
      B: shape [2, n_cop]
      ker_grid1..5: shape [M, n_cop]

    Returns:
      ker_grid_fin: shape [M, n_cop]
    """
    if B.dim() == 1:
        B = B.unsqueeze(1)
    b0 = B[0, :].unsqueeze(0)  # [1,n_cop]
    b1 = B[1, :].unsqueeze(0)  # [1,n_cop]

    # Avoid division by zero
    ker_grid1_safe = ker_grid1.clamp(min=1e-30)

    ratio1 = ker_grid2 / ker_grid1_safe
    ratio2 = ker_grid3 / ker_grid1_safe
    ratio4 = ker_grid4 / ker_grid1_safe
    ratio5 = ker_grid5 / ker_grid1_safe

    val_e1 = b0 * torch.sqrt(torch.abs(ratio4 - ratio1 ** 2))
    val_e2 = b1 * torch.sqrt(torch.abs(ratio5 - ratio2 ** 2))

    small_val = 1e-12
    c_part1 = -(val_e1 ** 2 * (ratio1 ** 2 / (2.0 * (b0 ** 2 + small_val))))
    c_part2 = -(val_e2 ** 2 * (ratio2 ** 2 / (2.0 * (b1 ** 2 + small_val))))
    C = c_part1 + c_part2

    ker_grid_fin = ker_grid1 * val_e1 * val_e2 * torch.exp(C)
    ker_grid_fin = replace_nan_inf(ker_grid_fin)
    return ker_grid_fin


def loclik_batch_eval(B: torch.Tensor,
                      data: torch.Tensor,
                      grid_x: torch.Tensor,
                      n_cop: int,
                      batch_size: int):
    """
    Evaluate local-likelihood on a grid, chunking for memory efficiency.

    Args:
      B: shape [2, n_cop], bandwidth
      data: shape [N, 2, n_cop], the data
      grid_x: shape [M, 2, n_cop], the grid points
      n_cop: int
      batch_size: int (number of chunks)

    Returns:
      ker_grid_fin: shape [M, n_cop], local-likelihood on each grid point
    """
    M = grid_x.shape[0]

    ker1_list = []
    ker2_list = []
    ker3_list = []
    ker4_list = []
    ker5_list = []

    batch_len = M // batch_size
    remainder = M % batch_size
    start_idx = 0

    for i in range(batch_size):
        end_idx = start_idx + batch_len
        if i == batch_size - 1:
            end_idx += remainder
        grid_chunk = grid_x[start_idx:end_idx]
        ker1, ker2, ker3, ker4, ker5 = dense_naive_batch(B, data, grid_chunk)
        ker1_list.append(ker1)
        ker2_list.append(ker2)
        ker3_list.append(ker3)
        ker4_list.append(ker4)
        ker5_list.append(ker5)
        start_idx = end_idx

    ker_grid1 = torch.cat(ker1_list, dim=0)
    ker_grid2 = torch.cat(ker2_list, dim=0)
    ker_grid3 = torch.cat(ker3_list, dim=0)
    ker_grid4 = torch.cat(ker4_list, dim=0)
    ker_grid5 = torch.cat(ker5_list, dim=0)

    ker_grid_fin = kern_LL(B, ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5)
    return ker_grid_fin
