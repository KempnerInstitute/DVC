###############################################
# src/torch_vine/utils_locallik.py
###############################################

import torch
import math
import numpy as np
from .utils_tensor import replace_nan_inf

def dense_naive_batch(B: torch.Tensor,
                      data_p: torch.Tensor,
                      grid_points: torch.Tensor):
    """
    Prepare partial sums for local-likelihood (the "naive" kernel approach),
    as in the original TF code.

    Args:
      B: shape [2, n_cop], the bandwidth parameters (B[0] for x-dim, B[1] for y-dim)
      data_p: shape [N, 2, n_cop], the data for N samples, 2 coords, for each of n_cop edges
      grid_points: shape [M, 2, n_cop], the M grid points to evaluate on, for each edge

    Returns:
      ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5: each shape [M, n_cop],
      representing partial sums used later in local-likelihood (kern_LL).
      Specifically:
        ker_grid1 = sum over N of a
        ker_grid2 = sum over N of a * (delta_x)
        ker_grid3 = sum over N of a * (delta_y)
        ker_grid4 = sum over N of a * (delta_x^2)
        ker_grid5 = sum over N of a * (delta_y^2)
      where "a" is basically the kernel's exponent factor, normalized by (2*pi*bw_x*bw_y*N).
    """
    device = data_p.device
    N = data_p.shape[0]
    M = grid_points.shape[0]
    n_cop = data_p.shape[2]

    # Expand grid_points and data_p so we can do (grid - data) in a big broadcast
    # shape => [N, M, 2, n_cop]
    # data_exp: [N,1,2,n_cop] => repeated along dim=1 => [N,M,2,n_cop]
    data_exp = data_p.unsqueeze(1).expand(-1, M, -1, -1)
    # grid_exp: [1,M,2,n_cop] => repeated along dim=0 => [N,M,2,n_cop]
    grid_exp = grid_points.unsqueeze(0).expand(N, -1, -1, -1)

    # c => difference array: shape [N, M, 2, n_cop]
    c = grid_exp - data_exp

    # B shape [2,n_cop], we separate bandwidth for x-dim and y-dim
    # For each copula i, b0 = B[0,i], b1 = B[1,i]
    # We'll broadcast them to shape [1,1,1,n_cop]
    b0 = B[0,:].view(1,1,1,n_cop)
    b1 = B[1,:].view(1,1,1,n_cop)

    # compute exponent: c[:,:,0,:]^2/(2*b0^2) + c[:,:,1,:]^2/(2*b1^2)
    val_x = (c[:,:,0,:]**2) / (2.0 * b0**2)
    val_y = (c[:,:,1,:]**2) / (2.0 * b1**2)
    val_exp = torch.exp(-(val_x + val_y))

    # Normalization factor => 1 / (2*pi * b0 * b1 * N)
    # shape => [1,1,n_cop]
    pi_val = math.pi
    const = 1.0/(2.0 * pi_val) * 1.0/(b0*b1) * (1.0/float(N))
    # shape => [1,1,1,n_cop], so let's squeeze if needed
    # val_exp shape [N,M,n_cop], let's unify shapes carefully
    # val_exp => [N,M,n_cop], after we remove the dimension for 2
    # Actually val_exp is [N,M], well let's see:
    #   c is [N,M,2,n_cop], so val_x, val_y => [N,M,n_cop].
    # We'll do a:
    a = val_exp * const.squeeze(2)  # shape => [N,M,n_cop]

    # Now compute partial sums:
    # sum over N => dimension 0
    ker_grid1 = a.sum(dim=0)  # shape [M,n_cop]
    ker_grid2 = (a * c[:,:,0,:]).sum(dim=0)  # shape [M,n_cop]
    ker_grid3 = (a * c[:,:,1,:]).sum(dim=0)
    ker_grid4 = (a * (c[:,:,0,:]**2)).sum(dim=0)
    ker_grid5 = (a * (c[:,:,1,:]**2)).sum(dim=0)

    return ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5


def kern_LL(B: torch.Tensor,
            ker_grid1: torch.Tensor,
            ker_grid2: torch.Tensor,
            ker_grid3: torch.Tensor,
            ker_grid4: torch.Tensor,
            ker_grid5: torch.Tensor):
    """
    Replicates the "kern_LL" logic from original code:
    - we compute e1, e2 from the partial sums
    - we build the exponent "C"
    - final 'ker_grid_fin'

    Args:
      B: shape [2, n_cop]
      ker_grid1..5: shape [M,n_cop]

    Returns:
      ker_grid_fin: shape [M,n_cop], the final kernel-likelihood values
    """
    # e1 = B[0]* sqrt( abs( (ker_grid4/ker_grid1) - (ker_grid2/ker_grid1)^2 ) )
    # e2 = B[1]* sqrt( abs( (ker_grid5/ker_grid1) - (ker_grid3/ker_grid1)^2 ) )
    # then C = - e1^2 * ((ker_grid2/ker_grid1)^2 / (2*B[0]^2)) - e2^2 * ...
    # final => ker_grid1 * e1 * e2 * exp(C)
    # we'll adapt from original.

    device = ker_grid1.device
    b0 = B[0,:].unsqueeze(0)  # shape [1,n_cop]
    b1 = B[1,:].unsqueeze(0)  # shape [1,n_cop]

    # shape => [M,n_cop]
    ratio1 = ker_grid2 / ker_grid1
    ratio2 = ker_grid3 / ker_grid1
    ratio4 = ker_grid4 / ker_grid1
    ratio5 = ker_grid5 / ker_grid1

    val_e1 = b0 * torch.sqrt( torch.abs( ratio4 - ratio1**2 ) )
    val_e2 = b1 * torch.sqrt( torch.abs( ratio5 - ratio2**2 ) )

    # Now compute exponent C = - e1^2 *(...) - e2^2*(...)
    # from original code:
    #   C = - e1^2 * ((ker_grid2/ker_grid1)^2 / (2*b0^2)) - ...
    # We'll do:
    #   C = - e1^2*( (ratio1^2)/(2*b0^2)) - e2^2*( (ratio2^2)/(2*b1^2))
    # shape => [M,n_cop]
    small_val = 1e-12
    c_part1 = - ( val_e1**2 * ( ratio1**2 / (2.0*(b0**2 + small_val)) ) )
    c_part2 = - ( val_e2**2 * ( ratio2**2 / (2.0*(b1**2 + small_val)) ) )
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
    Evaluate local-likelihood on the given grid by dividing 'grid_x'
    into 'batch_size' chunks (for memory reasons, as in original code).
    
    Steps:
      1) We'll gather partial sums (ker_grid1..5) for each chunk 
         from dense_naive_batch
      2) We'll do 'kern_LL' at the end or chunk by chunk
      3) Return final kernel-likelihood on the entire grid

    Args:
      B: shape [2,n_cop], bandwidth
      data: shape [N,2,n_cop], the data
      grid_x: shape [M,2,n_cop], the grid points
      n_cop: int
      batch_size: int (number of chunks)

    Returns:
      ker_grid_fin: shape [M, n_cop], final local-likelihood on each grid point for each copula
    """
    device = data.device
    M = grid_x.shape[0]

    # We'll accumulate partial sums for each chunk, then combine
    ker1_list = []
    ker2_list = []
    ker3_list = []
    ker4_list = []
    ker5_list = []

    # chunk size
    batch_len = M // batch_size
    remainder = M % batch_size
    start_idx = 0

    for i in range(batch_size):
        end_idx = start_idx + batch_len
        if i == batch_size -1:
            end_idx += remainder
        # sub-grid
        grid_chunk = grid_x[start_idx:end_idx,...]  # shape [?,2,n_cop]
        # call dense_naive_batch
        ker1, ker2, ker3, ker4, ker5 = dense_naive_batch(B, data, grid_chunk)
        ker1_list.append(ker1)
        ker2_list.append(ker2)
        ker3_list.append(ker3)
        ker4_list.append(ker4)
        ker5_list.append(ker5)
        start_idx = end_idx

    # now combine chunk results
    ker_grid1 = torch.cat(ker1_list, dim=0)  # shape [M,n_cop]
    ker_grid2 = torch.cat(ker2_list, dim=0)
    ker_grid3 = torch.cat(ker3_list, dim=0)
    ker_grid4 = torch.cat(ker4_list, dim=0)
    ker_grid5 = torch.cat(ker5_list, dim=0)

    # final local-likelihood aggregator:
    ker_grid_fin = kern_LL(B, ker_grid1, ker_grid2, ker_grid3, ker_grid4, ker_grid5)
    return ker_grid_fin