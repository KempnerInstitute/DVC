# src/pred/prediction.py
import torch
from utils.tensor_op import create_points, smooth_moving_average, replace_nan_inf

def predict_vine(x: torch.Tensor, vine, dim: int, exp_dim: int):
    """
    Create evaluation points using create_points, then use vine.evaluation to get density.
    Compute maximum-likelihood (y_ml) and expectation (y_em) estimates.
    """
    points = create_points(x, dim, exp_dim)
    p, p_cop, logf = vine.evaluation(points)
    N = x.shape[0]
    p_reshaped = p.view(N, exp_dim)
    idx_ml = p_reshaped.argmax(dim=1)
    y_ml = torch.zeros(N, dtype=x.dtype, device=x.device)
    y_em = torch.zeros(N, dtype=x.dtype, device=x.device)
    col_vals = x[:, dim]
    mn = torch.min(col_vals)
    mx = torch.max(col_vals)
    y_grid = torch.linspace(mn, mx, exp_dim, device=x.device, dtype=x.dtype)
    for i in range(N):
        y_ml[i] = y_grid[idx_ml[i]]
        weights = p_reshaped[i]
        weights = weights / (weights.sum() + 1e-16)
        y_em[i] = torch.sum(weights * y_grid)
    y_em = replace_nan_inf(y_em)
    return p, y_ml, y_em