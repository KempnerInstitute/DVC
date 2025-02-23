###############################################
# src/torch_vine/vine_eval.py
###############################################

import torch
from .utils_tensor import check_bound3
from .cop_eval import eval_rs_cop
from .utils_interpolation import nearestInterp2d


def evaluate_fit(data_dict, grid_dict, par_dict):
    """
    Evaluate the fitted non-param copulas on a grid => produce pdf, cdf.
    data_dict, grid_dict, par_dict as in the original code.
    Skeleton only.
    """
    pass


def evaluate_points(points_s, batch_size, grid_s, cdf1, pd_grid_uv):
    """
    Evaluate pdf, cdf at 'points_s' in [s1, s2].
    We'll do a naive approach with nearestInterp2d for PDF,
    Then cdf could be partial. 
    """
    pd_points = nearestInterp2d(points_s, grid_s.ax1, grid_s.ax2, pd_grid_uv)
    # cdf_points => partial
    ccdf_points = pd_points.cumsum(dim=0)
    return pd_points, ccdf_points


def evaluate_fit_bin(data_dict, grid_dict, par_dict):
    """
    If binning is used, do the same approach in each bin.
    """
    pass