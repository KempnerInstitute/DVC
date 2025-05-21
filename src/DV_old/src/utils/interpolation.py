# src/utils/interpolation.py
import torch
import numpy as np
from scipy import interpolate

def interp1d_np(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    1D linear interpolation via NumPy.
    x, xp, fp should be 1D tensors.
    """
    x_np = x.cpu().numpy()
    xp_np = xp.cpu().numpy()
    fp_np = fp.cpu().numpy()
    y_np = np.interp(x_np, xp_np, fp_np)
    return torch.from_numpy(y_np).to(x.device).to(x.dtype)

def nearest_interp2d(samples: torch.Tensor, ax1: torch.Tensor, ax2: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    For each sample (with 2 coordinates), find the nearest indices in ax1 and ax2 and return the corresponding value from grid.
    grid is assumed to be a 2D tensor of shape [K, K].
    """
    N = samples.shape[0]
    out = torch.zeros(N, dtype=samples.dtype, device=samples.device)
    for i in range(N):
        s1, s2 = samples[i, 0].item(), samples[i, 1].item()
        idx1 = torch.abs(ax1 - s1).argmin().item()
        idx2 = torch.abs(ax2 - s2).argmin().item()
        out[i] = grid[idx1, idx2]
    return out