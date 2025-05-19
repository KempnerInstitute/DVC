###############################################
# src/DVC/vine_eval.py
###############################################

import torch
import numpy as np

from .utils_tensor import check_bound3
from .cop_eval import eval_rs_cop, cdf_grid_fun
from .utils_interpolation import nearestInterp2d
from .utils_locallik import loclik_batch_eval
from .grid_ops import grid_obj
from .utils_prob import biv_norm, kernel_cdf
from .dataset_ops import create_bins, check_bins
from .transformation import Transform


def evaluate_fit(data_dict, grid_dict, par_dict):
    """
    Evaluate a nonparametric local-likelihood copula fit on the provided grid.

    Steps (mirroring original TF logic):
      1) local-likelihood kernel estimates (loclik_batch_eval) => unnormalized pdf on the grid
      2) reshape to [knots, knots, n_cop]
      3) multiply or re-normalize w.r.t. a bivariate normal reference (eval_rs_cop)
      4) compute a 2D cdf on the grid (cdf_grid_fun)
      5) optionally update 'theta' in data_dict by interpolation (skipped by default).

    Returns:
      pd_grid_uv: shape [knots, knots, n_cop] => final PDF on the grid
      cdf_grid:   shape [knots, knots, n_cop] => final CDF on the grid
      updated_theta (or None if we do not update).
    """
    # Unpack required info from dicts
    grid_u = grid_dict['grid_u']  # e.g. an instance of grid_obj
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']   # shape [knots^2, 2, n_cop]
    data_s = data_dict['data_s']   # shape [N,2,n_cop]
    data_x = data_dict.get('data_x', None)
    B = par_dict['bw']             # shape [2, n_cop]
    n_cop = par_dict['n_cop']
    batch_size = par_dict['batch']

    device = grid_x.device

    # 1) local-likelihood estimate => shape [M, n_cop], M=knots^2
    ker_grid_fin = loclik_batch_eval(B, data_s, grid_x, n_cop, batch_size)
    # shape => [M, n_cop]

    # 2) reshape to [knots, knots, n_cop]
    knots = grid_s.ax1.shape[0]
    M = ker_grid_fin.shape[0]
    if M != knots * knots:
        raise ValueError(f"Grid size mismatch: {M} != {knots}^2 = {knots*knots}.")

    ker_grid_3d = ker_grid_fin.view(knots, knots, n_cop)

    # 3) build a bivariate normal reference => shape [knots, knots, n_cop]
    x1_s, x2_s = grid_s.axis()  # each shape [knots]
    normal_ref_2d = biv_norm(x1_s, x2_s)    # shape [knots, knots]
    normal_ref_3d = normal_ref_2d.unsqueeze(-1).repeat(1, 1, n_cop).to(device)

    # Now do final "eval_rs_cop" => ensures the integrated PDF is a proper copula
    adu11, adu22 = grid_u.diff()  # each shape [knots]
    pd_grid_norm = eval_rs_cop(adu11, adu22, ker_grid_3d, normal_ref_3d, n_cop)

    # 4) compute cdf => cdf_grid_fun => shape [knots, knots, n_cop]
    cdf_grid = cdf_grid_fun(pd_grid_norm, grid_u.ex, adu11, adu22, n_cop)

    # optional gradient grids
    grad_u = grad_v = None
    if par_dict.get('grad_precompute', False):
        step_u = adu11[0].item()
        step_v = adu22[0].item()
        # central differences
        grad_u = torch.zeros_like(cdf_grid)
        grad_v = torch.zeros_like(cdf_grid)
        grad_u[1:-1,:,:] = (cdf_grid[2:,:,:]-cdf_grid[:-2,:,:])/(2*step_u)
        grad_u[0,:,:] = grad_u[1,:,:]
        grad_u[-1,:,:] = grad_u[-2,:,:]
        grad_v[:,1:-1,:] = (cdf_grid[:,2:,:]-cdf_grid[:,:-2,:])/(2*step_v)
        grad_v[:,0,:] = grad_v[:,1,:]
        grad_v[:,-1,:] = grad_v[:,-2,:]

    # 5) optionally update data_dict['theta']
    updated_theta = None
    if 'theta' in data_dict:
        # If we want to do partial updates by interpolation, we do it here. 
        updated_theta = None

    return pd_grid_norm, cdf_grid, updated_theta, grad_u, grad_v


def evaluate_points(points_s, batch_size, grid_s, cdf1, pd_grid_uv):
    """
    Evaluate PDF and CDF at arbitrary 'points_s'.

    - 'pd_grid_uv' shape [knots, knots, n_cop]
    - 'cdf1'       shape [knots, knots, n_cop]
    - 'points_s'   shape [N,2,n_cop]? or [N,2] if single cop?

    The function:
      1) flatten or pick the relevant copula dimension
      2) do a 2D interpolation for each dimension
      3) Return (pdf_points, cdf_points) shape [N]

    For demonstration, we do a naive "nearestInterp2d" approach 
    (one could do advanced methods or a loop for multi-cop).
    """
    device = points_s.device
    n_pts = points_s.shape[0]

    # parse grid_s => x1_s, x2_s
    x1_s, x2_s = grid_s.axis()  # each shape [knots]
    knots = x1_s.shape[0]

    # if pd_grid_uv.dim()==3 => we have [knots, knots, n_cop]
    # else => expand
    if pd_grid_uv.dim() == 3:
        n_cop = pd_grid_uv.shape[2]
    else:
        n_cop = 1
        pd_grid_uv = pd_grid_uv.unsqueeze(-1)
        cdf1 = cdf1.unsqueeze(-1)

    # For demonstration, we handle only the first copula => 0
    # or if you want multi-cop approach, you'd loop 
    pdf_points = nearestInterp2d(points_s, x1_s, x2_s, pd_grid_uv[:,:,0])
    cdf_points = nearestInterp2d(points_s, x1_s, x2_s, cdf1[:,:,0])

    return pdf_points, cdf_points


def evaluate_fit_bin(data_dict, grid_dict, par_dict):
    """
    If binning is used, we do the same approach for each bin and combine.

    Original logic:
      1) For each bin b=0..n_bin-1, we slice data_s for that bin
      2) call evaluate_fit => get (pdf, cdf) => store
      3) at the end => cat in last dimension => shape [..., n_bin]

    We replicate a simpler approach, ignoring parent-based splits.

    Returns:
      pdf_out: shape [knots, knots, n_cop, n_bin]
      cdf_out: shape [knots, knots, n_cop, n_bin]
    """
    n_bin = par_dict['n_bin']
    data_s = data_dict['data_s']   # shape [N,2,n_cop]
    data_x = data_dict.get('data_x', None)
    # parse other 
    bw = par_dict['bw']     # shape [2,n_cop] or [2,n_cop,n_bin]
    n_cop = par_dict['n_cop']
    batch_size = par_dict['batch']

    # We'll store the results from each bin
    pdf_stacked = []
    cdf_stacked = []

    N = data_s.shape[0]
    chunk_size = N // n_bin
    for b in range(n_bin):
        start_idx = b*chunk_size
        end_idx = N if (b == n_bin-1) else (b+1)*chunk_size

        data_s_bin = data_s[start_idx:end_idx, :, :]
        data_x_bin = None
        if data_x is not None:
            data_x_bin = data_x[start_idx:end_idx, :, :]

        # sub_data_dict
        sub_data_dict = {
            'data_s': data_s_bin,
            'data_x': data_x_bin
        }
        sub_grid_dict = grid_dict
        # if bw.dim()==3 => each bin has separate bandwidth in the 3rd dim
        if bw.dim() == 3:
            bw_bin = bw[:,:,b]
        else:
            bw_bin = bw  # same for all bins
        sub_par_dict = {
            'bw': bw_bin,
            'n_cop': n_cop,
            'batch': batch_size
        }
        pd_grid_uv_bin, cdf_bin, _ = evaluate_fit(sub_data_dict, sub_grid_dict, sub_par_dict)
        # shape => [knots,knots,n_cop]
        pdf_stacked.append(pd_grid_uv_bin.unsqueeze(-1))  # => [knots,knots,n_cop,1]
        if cdf_bin is not None:
            cdf_stacked.append(cdf_bin.unsqueeze(-1))
        else:
            cdf_fake = torch.zeros_like(pd_grid_uv_bin)
            cdf_stacked.append(cdf_fake.unsqueeze(-1))

    # now cat along last dim => shape [knots, knots, n_cop, n_bin]
    pdf_out = torch.cat(pdf_stacked, dim=-1)
    cdf_out = torch.cat(cdf_stacked, dim=-1)

    return pdf_out, cdf_out