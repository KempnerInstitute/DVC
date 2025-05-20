# src/utils/bandwidth.py
import torch
import math

def bandwidth_rule_of_thumb(data: torch.Tensor, deg: int, n_cop: int) -> torch.Tensor:
    """
    Compute the bandwidth for each of n_cop edges using a rule-of-thumb.
    'data' is expected to have shape [N, 2, n_cop].
    A small ridge (epsilon*I) is added to the covariance for numerical stability.
    """
    N = data.shape[0]
    bw = torch.zeros(2, n_cop, dtype=data.dtype, device=data.device)
    eps = 1e-6  # small ridge factor
    for j in range(n_cop):
        sample = data[:, :, j]  # shape [N, 2]
        mu = torch.mean(sample, dim=0)
        xc = sample - mu
        cov = (xc.t() @ xc) / (N - 1)
        cov = cov + eps * torch.eye(2, dtype=cov.dtype, device=cov.device)
        chol = torch.linalg.cholesky(cov)
        scale = 5 * (N ** (-1 / (4 * deg + 2)))
        bw[0, j] = scale * chol[0, 0]
        bw[1, j] = scale * chol[1, 1]
    return bw / 10.0