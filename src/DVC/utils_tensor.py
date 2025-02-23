###############################################
# src/DVC/utils_tensor.py
###############################################

import torch
import math

def check_bound(data: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
    """
    Clips data to [min(mesh), max(mesh)].
    """
    max_m = mesh.max()
    min_m = mesh.min()
    return torch.clamp(data, min_m, max_m)


def check_bound3(data: torch.Tensor, maxx: float, minn: float) -> torch.Tensor:
    """
    Clamp data to the range (minn, maxx) with small epsilon offset for safety.
    """
    return torch.clamp(data, minn + 1e-10, maxx - 1e-10)


def replace_nan_inf(data: torch.Tensor) -> torch.Tensor:
    """
    Replace any NaN or Inf with finite defaults:
      NaN -> 0
      +Inf/-Inf -> 0
    """
    data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
    data = torch.where(torch.isinf(data), torch.zeros_like(data), data)
    return data


def replace_negative(data: torch.Tensor, newval: float) -> torch.Tensor:
    """
    Replace negative values in 'data' with 'newval'.
    """
    return torch.where(data < 0.0, torch.full_like(data, newval), data)


def create_points(x: torch.Tensor, dim: int, exp_dim: int) -> torch.Tensor:
    """
    Expands the 'dim' dimension in x to exp_dim points from min..max.
    Repeats other dims' values. 
    Similar to the original TF code that created a grid for evaluation.
    """
    min_val = x[:, dim].min()
    max_val = x[:, dim].max()
    y_vec = torch.linspace(min_val, max_val, exp_dim, device=x.device)

    out_list = []
    N = x.shape[0]
    D = x.shape[1]
    for i in range(D):
        if i == dim:
            tile_ = y_vec.unsqueeze(0).expand(N, exp_dim)
            col_i = tile_.reshape(-1)
        else:
            col_ = x[:, i].unsqueeze(1).expand(N, exp_dim)
            col_i = col_.reshape(-1)
        out_list.append(col_i.unsqueeze(1))

    out_pts = torch.cat(out_list, dim=1)
    return out_pts


def moving_average(a: torch.Tensor, window_len: int) -> torch.Tensor:
    """
    Computes a 1D moving average of 'a' using window_len.
    """
    if window_len < 2:
        return a
    csum = torch.cumsum(a, dim=0)
    result = torch.clone(a)
    # For the first window_len
    result[:window_len] = csum[:window_len] / float(window_len)
    # For the rest
    for i in range(window_len, len(a)):
        result[i] = (csum[i] - csum[i - window_len]) / float(window_len)
    return result


def update_tensor_2d(tensor: torch.Tensor, index_col: int, new_val: torch.Tensor):
    """
    Replace the entire column 'index_col' of 'tensor' with new_val.
    new_val shape must match the row dimension.
    """
    out = tensor.clone()
    out[:, index_col] = new_val
    return out