##################################################
# DVC/cop_eval.py
##################################################
import torch
from .utils_tensor import replace_nan_inf

def eval1(adu11_col1: torch.Tensor,
          adu22_1: torch.Tensor,
          t2: torch.Tensor,
          n_cop: int):
    """
    One iteration of row/col normalization in the local-likelihood PDF.
    """
    I1 = torch.sum(adu22_1.unsqueeze(-1) * t2, dim=0)  # shape [X, n_cop]
    I2 = torch.sum(adu11_col1 * t2, dim=1)            # shape [X, n_cop]

    X = t2.shape[0]
    tlist = []
    for i in range(n_cop):
        outer_xy = torch.ger(I1[:, i], I2[:, i])  # shape [X, X]
        tlist.append(outer_xy.unsqueeze(-1))
    K5 = torch.cat(tlist, dim=2)  # shape [X, X, n_cop]

    out = t2 * torch.reciprocal(K5)
    out = replace_nan_inf(out)
    return out

def eval_rs_cop(adu11: torch.Tensor,
                adu22: torch.Tensor,
                ker_fit: torch.Tensor,
                NORM1: torch.Tensor,
                n_cop: int) -> torch.Tensor:
    """
    Similar to TF 'eval_rs_cop', normalizing 2D PDF on a grid.
    """
    small_val = 1e-12
    denom = torch.where(NORM1==0., torch.full_like(NORM1, small_val), NORM1)
    t1 = ker_fit / denom
    t1 = torch.where(t1 < 1e-7, torch.ones_like(t1), t1)
    t1 = replace_nan_inf(t1)

    for _ in range(50):
        t1 = eval1(adu11.view(1, -1, 1), adu22, t1, n_cop)

    sum_y = torch.sum(adu22.unsqueeze(-1) * t1, dim=0)
    sum_x = torch.sum(adu11.view(-1,1)*sum_y, dim=0)

    t1 = t1 / sum_x.view(1,1,n_cop)
    out = t1 * NORM1
    out = replace_nan_inf(out)
    return out

def cdf_grid_fun(pd_grid_uv: torch.Tensor,
                 ex_u: torch.Tensor,
                 u1d: torch.Tensor,
                 u2d: torch.Tensor,
                 n_cop: int) -> torch.Tensor:
    """
    Compute 2D CDF from PDF, analogous to your TF code.
    """
    knots = pd_grid_uv.shape[0]
    device = pd_grid_uv.device

    u2d_expanded = u2d.view(knots, 1, 1).expand(-1, knots, n_cop)
    pd_transposed = pd_grid_uv.permute(1, 0, 2)

    weighted_pd = pd_transposed * u2d_expanded
    integ = torch.cumsum(weighted_pd, dim=0)

    norm_p = weighted_pd.sum(dim=0)
    zero_mask = (norm_p == 0)
    if zero_mask.any():
        norm_p = torch.where(zero_mask, torch.ones_like(norm_p), norm_p)

    cdf = integ / norm_p.unsqueeze(0)
    cdf = cdf.permute(1, 0, 2)

    cdf_flat = cdf.reshape(-1)
    cdf_flat = torch.clamp(cdf_flat, 0.0, 1.0)
    cdf = cdf_flat.reshape(knots, knots, n_cop)
    return cdf