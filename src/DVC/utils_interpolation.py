###############################################
# src/DVC/utils_interpolation.py
###############################################

import torch
import torch.nn.functional as F
import numpy as np


def interp1d_linear_gpu(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    A faster 1D linear interpolation approach using `torch.searchsorted` 
    to avoid Python loops or NumPy calls. 
    This runs on GPU if x,xp,fp are on GPU.

    Args:
      x: shape [N], query points
      xp: shape [M], reference x-values, must be sorted ascending
      fp: shape [M], reference y-values
    Returns:
      y: shape [N], linearly interpolated output
    """
    # clamp x to xp's domain
    x_min, x_max = xp[0], xp[-1]
    x_clamped = torch.clamp(x, x_min, x_max)

    # searchsorted => get index i s.t. xp[i] <= x < xp[i+1]
    idx = torch.searchsorted(xp, x_clamped, right=False)
    # clip to valid range for indexing i,i+1
    idx = torch.clamp(idx, 0, xp.shape[0]-2)

    x0 = xp[idx]
    x1 = xp[idx+1]
    y0 = fp[idx]
    y1 = fp[idx+1]

    denom = x1 - x0
    denom = torch.where(denom==0, torch.ones_like(denom)*1e-12, denom)
    w = (x_clamped - x0) / denom
    y = y0 + w*(y1 - y0)
    return y


def batch_interp1d_linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    If you want to do many queries in 'x' in parallel on GPU, 
    just call interp1d_linear_gpu. This is a convenience wrapper 
    if you had 2D shape or something. 
    """
    return interp1d_linear_gpu(x, xp, fp)


def nearestInterp2d(sample_s: torch.Tensor,
                    pro_s1: torch.Tensor,
                    pro_s2: torch.Tensor,
                    pd_grid_uv: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor interpolation in 2D (the old approach).
    For advanced methods, see 'grid_sample_2d' below.
    """
    sample_cpu = sample_s.detach().cpu().numpy()
    s1_cpu = pro_s1.detach().cpu().numpy()
    s2_cpu = pro_s2.detach().cpu().numpy()
    grid_cpu = pd_grid_uv.detach().cpu().numpy()

    out_list = []
    for i in range(sample_cpu.shape[0]):
        val1 = sample_cpu[i,0]
        val2 = sample_cpu[i,1]
        ix1 = np.argmin(np.abs(s1_cpu - val1))
        ix2 = np.argmin(np.abs(s2_cpu - val2))
        out_list.append(grid_cpu[ix1, ix2])

    out = torch.tensor(out_list, dtype=sample_s.dtype, device=sample_s.device)
    return out


def grid_sample_2d(data_2d: torch.Tensor, 
                   x_coords: torch.Tensor, 
                   y_coords: torch.Tensor, 
                   queries: torch.Tensor,
                   mode='bilinear'):
    """
    A more advanced 2D interpolation using PyTorch's `F.grid_sample`.
    data_2d shape: [1,1,H,W], a single "image" 
       H = size of y dimension, W = size of x dimension
    x_coords, y_coords: sorted 1D Tensors specifying the pixel centers 
    queries: shape [N,2], each is (qx, qy) in the same domain as x_coords, y_coords
    mode: 'bilinear' or 'bicubic' or 'nearest'

    Steps:
      1) convert (qx,qy) into normalized [-1,1]
      2) build a grid shape [1, N, 1, 2]
      3) call F.grid_sample

    Returns:
      Tensor shape [N] with the interpolated values.
    """
    # 1) we map x in [x_coords.min(), x_coords.max()] -> [-1,1]
    # similarly for y.
    device_ = data_2d.device
    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]

    # queries shape [N,2], let queries[:,0]= qx, queries[:,1]= qy
    qx = queries[:,0]
    qy = queries[:,1]

    # clamp
    qx = torch.clamp(qx, x_min, x_max)
    qy = torch.clamp(qy, y_min, y_max)

    # scale to [-1,1]
    # Nx, Ny = W, H => x -> col => index
    # normalized_x = 2*(qx - x_min)/(x_max - x_min) -1
    # normalized_y = 2*(qy - y_min)/(y_max - y_min) -1
    # note that in PyTorch, the vertical dimension is reversed if needed. We'll keep it simple.
    nx = 2*(qx - x_min)/max((x_max - x_min),1e-12) -1
    ny = 2*(qy - y_min)/max((y_max - y_min),1e-12) -1

    # build grid => [1, N, 1, 2]
    N = queries.shape[0]
    grid = torch.stack([nx, ny], dim=1).view(1, N, 1, 2)

    # do sampling
    # data_2d shape: [1,1,H,W]
    # grid_sample => output shape [1,1,N,1]
    out = F.grid_sample(data_2d, grid, mode=mode, align_corners=True)
    # shape => [1,1,N,1]
    out_val = out.view(-1)
    return out_val