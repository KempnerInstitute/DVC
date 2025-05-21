# src/utils/tensor_op.py
import torch

def check_bound_1d(data: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp values to [min_val, max_val]."""
    return torch.clamp(data, min=min_val, max=max_val)

def replace_nan_inf(data: torch.Tensor, nan_repl=0.0, pos_inf_repl=1e15, neg_inf_repl=-1e15) -> torch.Tensor:
    """
    Replace NaN with nan_repl and ±inf with specified values.
    """
    out = data.clone()
    out[torch.isnan(out)] = nan_repl
    out[out == float('inf')] = pos_inf_repl
    out[out == float('-inf')] = neg_inf_repl
    return out


def create_points(x: torch.Tensor, dim: int, exp_dim: int) -> torch.Tensor:
    """
    For each row in x (shape [N, D]), create exp_dim evaluation points for column `dim`
    by replacing that column with a linspace from min to max. Returns tensor of shape [N*exp_dim, D].
    """
    device = x.device
    N, D = x.shape
    col_vals = x[:, dim]
    mn = torch.min(col_vals)
    mx = torch.max(col_vals)
    y_vec = torch.linspace(mn, mx, exp_dim, device=device, dtype=x.dtype)
    out_rows = []
    for i in range(N):
        # Use clone() after expand to ensure unique memory allocation
        row = x[i].clone().unsqueeze(0).expand(exp_dim, -1).clone()
        row[:, dim] = y_vec
        out_rows.append(row)
    return torch.cat(out_rows, dim=0)


def smooth_moving_average(arr: torch.Tensor, window_len: int = 4) -> torch.Tensor:
    """Compute simple moving average smoothing."""
    if window_len < 2:
        return arr
    cumsum = torch.cumsum(arr, dim=0)
    out = arr.clone()
    for i in range(arr.shape[0]):
        left = max(0, i - window_len + 1)
        # If left > 0, subtract previous cumsum; otherwise use current value.
        out[i] = (cumsum[i] - (cumsum[left-1] if left > 0 else 0)) / (i - left + 1)
    return out

def create_bins(data: torch.Tensor, n_bin: int):
    """
    Create bin edges for the 1D data.
    Returns a list of bin edge values.
    """
    sorted_vals, _ = torch.sort(data)
    length = sorted_vals.shape[0]
    step = length // n_bin
    bins = [sorted_vals[0].item() - 1e-12]
    for i in range(1, n_bin):
        bins.append(sorted_vals[step * i].item())
    bins.append(sorted_vals[-1].item() + 1e-12)
    return bins

def check_bins(data: torch.Tensor, bins: list) -> torch.Tensor:
    """
    Assigns each value in data to a bin index using torch.bucketize.
    Returns a LongTensor of indices.
    """
    bin_edges = torch.tensor(bins, dtype=data.dtype, device=data.device)
    idx = torch.bucketize(data, bin_edges, right=False) - 1
    # Clamp to valid range.
    return torch.clamp(idx, 0, len(bins)-2).long()