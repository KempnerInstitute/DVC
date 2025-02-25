###############################################
# src/DVC/cop_eval.py
###############################################

import torch
from .utils_tensor import replace_nan_inf


def eval1(adu11_col1: torch.Tensor,
          adu22_1: torch.Tensor,
          t2: torch.Tensor,
          n_cop: int):
    """
    Performs one iteration of the row/column normalization step in copula PDF projection.

    Args:
      adu11_col1: shape [X, 1, n_cop], typically the diff vector along x (column form).
      adu22_1:    shape [X, n_cop],       diff vector along y.
      t2:         shape [X, X, n_cop],    current PDF guess.
      n_cop:      number of copulas.

    Returns:
      out:        shape [X, X, n_cop], updated PDF after row/col normalization pass.
    """
    # I1 = sum over x of ( adu22_1[x,:] * t2[x,:,:] ), i.e. shape [X, n_cop].
    # I2 = sum over y of ( adu11_col1[y,:] * t2[:,y,:] ), i.e. shape [X, n_cop].
    # Then for each x,y: K1=I1[y]*I2[x], and we do t2[x,y] *= 1.0 / K1.
    #
    # This matches the logic from your original TF code: 
    #   I1 = tf.math.reduce_sum(adu22_1 * t2, axis=1)
    #   I2 = tf.math.reduce_sum(adu11_col1 * t2, axis=0)
    #   K5 = outer(I1, I2) => shape [X,X,n_cop]
    #   out = t2 / K5.

    # Sum across axis=0 => "y" dimension
    I1 = torch.sum(adu22_1.unsqueeze(-1) * t2, dim=0)  # shape [X, n_cop]
    # Sum across axis=1 => "x" dimension
    I2 = torch.sum(adu11_col1 * t2, dim=1)            # shape [X, n_cop]

    # Build K5[x,y,i] = I1[y,i] * I2[x,i]
    X = t2.shape[0]
    tlist = []
    for i in range(n_cop):
        # shape => [X], [X] => outer => [X,X]
        outer_xy = torch.ger(I1[:, i], I2[:, i])  # ger => outer product
        tlist.append(outer_xy.unsqueeze(-1))
    K5 = torch.cat(tlist, dim=2)  # shape [X, X, n_cop]

    # multiply t2 by reciprocal of K5
    out = t2 * torch.reciprocal(K5)
    out = replace_nan_inf(out)
    return out


def eval_rs_cop(adu11: torch.Tensor,
                adu22: torch.Tensor,
                ker_fit: torch.Tensor,
                NORM1: torch.Tensor,
                n_cop: int) -> torch.Tensor:
    """
    Copula normalization (2D) for MISE-based local-likelihood function,
    using iterative row/column scaling as in your original TF 'eval_rs_cop'.

    Steps (in the original code logic):
      1) Project the kernel estimate onto the 'u-v' space by dividing by NORM1 => t1
      2) Perform ~50 row/col normalization passes (eval1) => ensuring integrals match
      3) Compute the final integral => sum_x,y of ( adu11[x]* adu22[y]* t1[x,y] ) => scale
      4) Multiply by NORM1 to project back => final PDF

    Args:
      adu11: shape [X], diff along x-axis
      adu22: shape [X], diff along y-axis
      ker_fit: shape [X, X, n_cop], the raw local-likelihood kernel estimate on the grid
      NORM1:   shape [X, X, n_cop], e.g. the bivariate normal reference or “r-s” factor
      n_cop:   number of copulas

    Returns:
      out: shape [X, X, n_cop], the final normalized PDF that integrates to 1 along x,y.
    """
    # Step 1: t1 = ker_fit / NORM1
    # avoid /0 by replacing with a small constant
    small_val = 1e-12
    denom = torch.where(NORM1==0., torch.full_like(NORM1, small_val), NORM1)
    t1 = ker_fit / denom

    # If t1 is extremely small or infinite => clamp
    t1 = torch.where(t1 < 1e-7, torch.ones_like(t1), t1)
    t1 = replace_nan_inf(t1)

    # Step 2: Do ~50 iterative row/column passes
    X = adu11.shape[0]
    for _ in range(50):
        # shape => eval1( [1,X,1], [X], [X,X,n_cop], n_cop )
        t1 = eval1(adu11.view(1, -1, 1), adu22, t1, n_cop)

    # Step 3: final integral => sum_x sum_y of ( adu11[x]* adu22[y]* t1[x,y,:] )
    # shape => sum_y => [X,n_cop], sum_x => [n_cop]
    sum_y = torch.sum(adu22.unsqueeze(-1) * t1, dim=0)  # shape [X,n_cop]
    sum_x = torch.sum(adu11.view(-1,1)*sum_y, dim=0)   # shape [n_cop]

    t1 = t1 / sum_x.view(1,1,n_cop)  # scale so total integral=1

    # Step 4: Multiply by NORM1 => final PDF
    out = t1 * NORM1
    out = replace_nan_inf(out)
    return out


def cdf_grid_fun(pd_grid_uv: torch.Tensor,
                 ex_u: torch.Tensor,
                 u1d: torch.Tensor,
                 u2d: torch.Tensor,
                 n_cop: int) -> torch.Tensor:
    """
    Compute 2D CDF on the grid from PDF values.
    
    Args:
        pd_grid_uv: shape [knots, knots, n_cop], the PDF on grid
        ex_u: shape [knots*knots, 2], expanded grid
        u1d: shape [knots], differential along x
        u2d: shape [knots], differential along y
        n_cop: number of copulas
    
    Returns:
        cdf: shape [knots, knots, n_cop], the 2D CDF on grid
    """
    device = pd_grid_uv.device
    dtype = pd_grid_uv.dtype
    
    # Get knots size
    knots = pd_grid_uv.shape[0]
    
    # Reshape u2d for broadcasting
    u2d_expanded = u2d.view(knots, 1, 1).expand(-1, knots, n_cop)
    
    # Transpose the PDF for cumulative sum along rows
    pd_transposed = pd_grid_uv.permute(1, 0, 2)  # [knots, knots, n_cop] -> [knots, knots, n_cop]
    
    # Multiply by differentials and cumsum
    weighted_pd = pd_transposed * u2d_expanded
    integ = torch.cumsum(weighted_pd, dim=0)  # cumulative sum along y
    
    # Total sum for normalization
    norm_p = weighted_pd.sum(dim=0)  # shape [knots, n_cop]
    
    # Handle zero entries in norm_p
    zero_mask = (norm_p == 0)
    if zero_mask.any():
        norm_p = torch.where(zero_mask, torch.ones_like(norm_p), norm_p)
    
    # Normalize and transpose back
    cdf = integ / norm_p.unsqueeze(0)  # shape [knots, knots, n_cop]
    cdf = cdf.permute(1, 0, 2)  # back to [knots, knots, n_cop]
    
    # Flatten, bound check, and reshape
    cdf_flat = cdf.reshape(-1)
    cdf_flat = torch.clamp(cdf_flat, 0.0, 1.0)
    cdf = cdf_flat.reshape(knots, knots, n_cop)
    
    return cdf