###############################################
# src/DVC/utils_tensor.py
###############################################

import torch
import math

def check_bound(data: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
    """
    Clip 'data' to [mesh.min(), mesh.max()].
    Typically used to ensure we don't step outside a reference grid.

    Args:
      data: shape [...], any
      mesh: shape [M], or [some], from which we extract min/max
    Returns:
      out => same shape as data
    """
    max_m = mesh.max()
    min_m = mesh.min()
    return torch.clamp(data, min_m, max_m)


def check_bound3(data: torch.Tensor, maxx: float, minn: float) -> torch.Tensor:
    """
    Clamp 'data' to (minn + 1e-10, maxx - 1e-10).
    Helps avoid exact 0 or 1 in copula transformations.

    Args:
      data: shape [...]
      maxx, minn: float
    Returns:
      out => same shape as data
    """
    return torch.clamp(data, minn + 1e-10, maxx - 1e-10)


def replace_nan_inf(data: torch.Tensor) -> torch.Tensor:
    """
    Replace any NaN or +/- Inf in 'data' with 0. 
    This helps avoid errors in log-likelihood or further computations.

    Args:
      data: shape [...]
    Returns:
      out => same shape as data, with no NaN/Inf
    """
    data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
    data = torch.where(torch.isinf(data), torch.zeros_like(data), data)
    return data


def replace_negative(data: torch.Tensor, newval: float) -> torch.Tensor:
    """
    Replace negative values in 'data' with 'newval'.
    Useful if we want strictly non-negative arrays for certain ops.

    Args:
      data: shape [...]
      newval: float
    Returns:
      out => same shape as data
    """
    return torch.where(data < 0.0, torch.full_like(data, newval), data)


def create_points(x: torch.Tensor, dim: int, exp_dim: int) -> torch.Tensor:
    """
    Expands a single dimension 'dim' of x into 'exp_dim' grid points 
    from min..max of x[:,dim]. Repeats other dimensions, 
    effectively producing an Nxexp_dim set of points 
    (flattened) for "evaluation" usage.

    Example:
      x shape [N,D], dim=0, exp_dim=100 => we produce [N*100, D] with 
      each row repeated in all columns except 'dim' is replaced by a linspace.

    Args:
      x: shape [N,D]
      dim: which dimension to expand
      exp_dim: number of grid points
    Returns:
      out_pts: shape [N*exp_dim, D]
    """
    # find min..max along that 'dim'
    min_val = x[:, dim].min()
    max_val = x[:, dim].max()

    y_vec = torch.linspace(min_val, max_val, exp_dim, device=x.device)
    out_list = []
    N = x.shape[0]
    D = x.shape[1]
    for i in range(D):
        if i == dim:
            # we tile y_vec for each of N rows => shape [N,exp_dim], flatten => [N*exp_dim]
            tile_ = y_vec.unsqueeze(0).expand(N, exp_dim)
            col_i = tile_.reshape(-1)
        else:
            # repeat x[:,i] for exp_dim
            col_ = x[:, i].unsqueeze(1).expand(N, exp_dim)
            col_i = col_.reshape(-1)
        out_list.append(col_i.unsqueeze(1))
    out_pts = torch.cat(out_list, dim=1)  # shape [N*exp_dim, D]
    return out_pts


def moving_average(a: torch.Tensor, window_len: int) -> torch.Tensor:
    """
    Compute a 1D moving average of 'a' over 'window_len'.

    Implementation:
      csum = cumsum(a)
      result[i] = ( csum[i] - csum[i-window_len] )/ window_len for i>=window_len
      with the initial window handled specially.

    Args:
      a: shape [N]
      window_len: int
    Returns:
      shape [N]
    """
    if window_len < 2:
        return a
    csum = torch.cumsum(a, dim=0)
    result = a.clone()

    # For the first window_len
    result[:window_len] = csum[:window_len] / float(window_len)
    # For the rest
    for i in range(window_len, a.shape[0]):
        result[i] = (csum[i] - csum[i - window_len]) / float(window_len)
    return result


def update_tensor_2d(tensor: torch.Tensor, index_col: int, new_val: torch.Tensor):
    """
    Replace the entire column 'index_col' of 'tensor' with 'new_val'.

    Args:
      tensor: shape [N,D]
      index_col: which column in [0..D-1]
      new_val: shape [N], the new values

    Returns:
      out => shape [N,D], same as tensor
    """
    out = tensor.clone()
    out[:, index_col] = new_val
    return out