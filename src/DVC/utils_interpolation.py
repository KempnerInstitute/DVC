###############################################
# src/DVC/utils_interpolation.py
###############################################

import torch
import torch.nn.functional as F
import numpy as np

def interp1d_linear_gpu(x: torch.Tensor,
                        xp: torch.Tensor,
                        fp: torch.Tensor) -> torch.Tensor:
    """
    A faster 1D linear interpolation approach using `torch.searchsorted`
    to avoid Python loops or NumPy calls. 
    This runs on GPU if x, xp, fp are on GPU.

    Steps (like np.interp):
      1) clamp 'x' to [xp[0], xp[-1]]
      2) i = searchsorted(xp, x_clamped)
      3) linear interpolation => y= y0 + w*(y1-y0), w= (x-x0)/(x1-x0).

    Args:
      x:  shape [N], query points
      xp: shape [M], sorted reference x-values
      fp: shape [M], reference y-values
    Returns:
      y:  shape [N], linearly interpolated output
    """
    # clamp x
    x_min, x_max = xp[0], xp[-1]
    x_clamped = torch.clamp(x, x_min, x_max)

    # searchsorted => i in [0..M-2]
    idx = torch.searchsorted(xp, x_clamped, right=False)
    idx = torch.clamp(idx, 0, xp.shape[0]-2)

    x0 = xp[idx]
    x1 = xp[idx+1]
    y0 = fp[idx]
    y1 = fp[idx+1]

    denom = x1 - x0
    denom = torch.where(denom == 0, torch.full_like(denom, 1e-12), denom)
    w = (x_clamped - x0) / denom
    y = y0 + w*(y1 - y0)
    return y


def batch_interp1d_linear(x: torch.Tensor,
                          xp: torch.Tensor,
                          fp: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper for multiple queries in 'x'. 
    Typically just calls interp1d_linear_gpu once,
    but you could add a loop if shape is 2D or bigger.

    Args:
      x:  shape [N], or possibly [N,*]
      xp: shape [M]
      fp: shape [M]
    Returns:
      y: shape [N], same shape or partial
    """
    return interp1d_linear_gpu(x, xp, fp)


def nearestInterp2d(sample_s: torch.Tensor,
                    pro_s1: torch.Tensor,
                    pro_s2: torch.Tensor,
                    pd_grid_uv: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor 2D interpolation, matching original code's approach.
    For each sample_s[i], we find the closest index in pro_s1 and in pro_s2, 
    then take that cell of pd_grid_uv.

    This is a naive O(N*M) python loop approach if sample_s is large.
    For advanced usage, see grid_sample_2d or other approach.

    Args:
      sample_s: shape [N,2], queries
      pro_s1:   shape [K], sorted x coords
      pro_s2:   shape [K], sorted y coords
      pd_grid_uv: shape [K,K], the 2D array to interpolate from

    Returns:
      out: shape [N], the interpolated values.
    """
    # move everything to CPU numpy to do the old naive approach
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
    An advanced 2D interpolation using PyTorch's F.grid_sample. 
    Typically:
      - data_2d shape: [1,1,H,W] like an image
      - x_coords => sorted x-locs => width dimension
      - y_coords => sorted y-locs => height dimension
      - queries => shape [N,2], each is (qx, qy)
      - mode => 'bilinear','bicubic','nearest'

    Steps:
      1) convert (qx,qy) to normalized [-1,+1]
      2) build grid shape [1,N,1,2]
      3) call F.grid_sample
      4) reshape => [N]

    Returns:
      out_val => shape [N], interpolated result
    """
    device_ = data_2d.device
    # check data_2d shape => [1,1,H,W]
    # parse domain
    x_min, x_max = x_coords[0], x_coords[-1]
    y_min, y_max = y_coords[0], y_coords[-1]

    # queries => shape [N,2]
    qx = queries[:,0].clamp(x_min, x_max)
    qy = queries[:,1].clamp(y_min, y_max)

    # normalize => nx in [-1,+1], ny in [-1,+1]
    nx = 2.0*(qx - x_min)/max((x_max - x_min),1e-12) -1.0
    ny = 2.0*(qy - y_min)/max((y_max - y_min),1e-12) -1.0

    # build grid => shape [1,N,1,2]
    N = queries.shape[0]
    grid = torch.stack([nx, ny], dim=1).view(1, N, 1, 2)

    # call F.grid_sample => returns [1,1,N,1]
    out = F.grid_sample(data_2d, grid, mode=mode, align_corners=True)
    out_val = out.view(-1)
    return out_val