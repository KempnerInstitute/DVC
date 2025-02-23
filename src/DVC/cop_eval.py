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
    Compute a 2D CDF on the grid from 'pd_grid_uv' by cumulative sums,
    mirroring the original TF code logic:

    Steps:
      - we have pd_grid_uv shape [X, X, n_cop], representing f(x_i, x_j) over a 2D grid
      - multiply each row by u2d (the differential in the 'y' axis), do cumsum => partial integral
      - divide by total integrals => ensures it's in [0,1]
      - reorder the axes if needed, produce cdf array shape [X,X,n_cop]

    Args:
      pd_grid_uv: shape [X, X, n_cop], final PDF on a 2D grid
      ex_u:       shape [X*X, 2], not used here except for referencing min/max if needed
      u1d:        shape [X], differential along x
      u2d:        shape [X], differential along y
      n_cop:      number of copulas

    Returns:
      cdf_grid: shape [X, X, n_cop], the 2D CDF on the grid
    """
    device = pd_grid_uv.device
    X = pd_grid_uv.shape[0]
    # we interpret "multiply each column by u2d => cumsum along axis=0"
    # The original code does something like:
    #   pd_transp = tf.transpose(pd_grid_uv, perm=[1,0,2])
    #   integ = cumsum( pd_transp*u2d_tile, axis=0)
    #   norm_p= reduce_sum(pd_grid_uv*u2d_tile, axis=0)
    #   cdf1= tf.transpose( integ/norm_p, perm=[1,0,2])
    #
    # We'll replicate that in PyTorch.

    # 1) expand u2d => shape [X,1,1], tile to [X,X,n_cop], multiply
    u2d_tile = u2d.view(X, 1, 1).expand(-1, X, n_cop)
    pd_grid_uv_t = pd_grid_uv.transpose(0,1)  # shape [X,X,n_cop] => [X,X,n_cop], swapped
    # multiply
    mul_ = pd_grid_uv_t * u2d_tile

    # cumsum along axis=0 => shape => [X,X,n_cop]
    integ = torch.cumsum(mul_, dim=0)  # integrate along 'y'
    # norm_p => sum( pd_grid_uv * u2d, axis=0 ) => shape [X,n_cop]
    norm_p = torch.sum(mul_, dim=0)  # shape [X,n_cop]
    # avoid zero => clamp
    norm_p = torch.where(norm_p==0., torch.ones_like(norm_p)*1e-12, norm_p)

    cdf1 = integ / norm_p.unsqueeze(0)  # shape => [X,X,n_cop]
    # transpose back => shape [X,X,n_cop]
    cdf1 = cdf1.transpose(0,1)

    # We do a final clamp to [0,1]
    cdf1 = torch.clamp(cdf1, 0.0, 1.0)
    return cdf1