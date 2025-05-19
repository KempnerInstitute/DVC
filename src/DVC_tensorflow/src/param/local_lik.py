# src/param/local_lik.py
import torch
import math
from utils.bandwidth import bandwidth_rule_of_thumb

def dense_naive_batch(B: torch.Tensor, data: torch.Tensor, grid: torch.Tensor) -> tuple:
    """
    Compute kernel density estimates on a grid using a naive double loop.
    B: [2, n_cop] bandwidth parameters.
    data: [N, 2, n_cop] sample for a given edge.
    grid: [K, 2, n_cop] evaluation grid.
    Returns: (ker_sum, ker_first, ker_second) each of shape [K, n_cop].
    """
    N, _, n_cop = data.shape
    K = grid.shape[0]
    ker_sum = torch.zeros(K, n_cop, device=data.device, dtype=data.dtype)
    ker_first = torch.zeros(K, n_cop, device=data.device, dtype=data.dtype)
    ker_second = torch.zeros(K, n_cop, device=data.device, dtype=data.dtype)
    for j in range(n_cop):
        h1 = B[0, j]
        h2 = B[1, j]
        for i in range(K):
            gx = grid[i, 0, j]
            gy = grid[i, 1, j]
            dx = (data[:, 0, j] - gx) / h1
            dy = (data[:, 1, j] - gy) / h2
            exponent = -0.5 * (dx**2 + dy**2)
            ker = torch.exp(exponent) / (2 * math.pi * h1 * h2 * N)
            ker_sum[i, j] = torch.sum(ker)
            ker_first[i, j] = torch.sum(ker * data[:, 0, j])
            ker_second[i, j] = torch.sum(ker * data[:, 0, j]**2)
    return ker_sum, ker_first, ker_second

def loclik_batch_eval(B: torch.Tensor, data: torch.Tensor, grid: torch.Tensor, n_cop: int, batch_size: int):
    """
    Evaluate the kernel density on grid in batches.
    """
    K = grid.shape[0]
    outputs = []
    batch_len = K // batch_size
    for i in range(batch_size):
        grid_batch = grid[i*batch_len:(i+1)*batch_len]
        ker_sum, _, _ = dense_naive_batch(B, data, grid_batch)
        outputs.append(ker_sum)
    return torch.cat(outputs, dim=0)

def local_likelihood_fit(data_x: torch.Tensor, n_cop: int):
    """
    Compute the bandwidth for each edge using the rule-of-thumb.
    """
    return bandwidth_rule_of_thumb(data_x, deg=2, n_cop=n_cop)