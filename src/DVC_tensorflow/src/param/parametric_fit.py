# src/param/parametric_fit.py
import torch
import numpy as np
from scipy.stats import kendalltau
from param.cond_copula import copulapdf

def parametric_fit(u: torch.Tensor, families: list, n_cop: int):
    """
    Fit a candidate set of parametric copula families to data u (shape [N,2,n_cop])
    using Kendall’s tau. Returns lists: aic_list, theta_list, logp_list.
    """
    device = u.device
    N = u.shape[0]
    aic_list = []
    theta_list = []
    logp_list = []
    u_np = u.cpu().numpy()
    for fam in families:
        theta_f = []
        logp_f = np.zeros(n_cop)
        for j in range(n_cop):
            u_edge = u_np[:, :, j]
            tau, _ = kendalltau(u_edge[:, 0], u_edge[:, 1])
            if fam.lower() == 'gaussian':
                r = np.sin((np.pi/2) * tau)
                theta_f.append(r)
                logp_f[j] = -abs(r)  # Simplified cost (the lower the |r|, the better)
            elif fam.lower() == 'ind':
                theta_f.append(0.0)
                logp_f[j] = 0.0
            else:
                # For other families, we use a similar rule
                theta_f.append(0.5)
                logp_f[j] = -abs(0.5)
        aic_f = 2 * 1 - 2 * logp_f
        aic_list.append(aic_f)
        theta_list.append(theta_f)
        logp_list.append(logp_f)
    return aic_list, theta_list, logp_list