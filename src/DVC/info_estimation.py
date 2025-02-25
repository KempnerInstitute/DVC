###############################################
# src/DVC/info_estimation.py
###############################################

import torch
import numpy as np

def vine_entropy(vine, info_dict: dict):
    """
    Approximate the vine's entropy by Monte Carlo sampling.

    We do repeated sampling from 'vine' (cases each time),
    estimate the mean of log(p(x)) in an incremental / running fashion,
    and track an approximate variance so that a standard error
    or confidence interval can be computed.

    Args:
      vine: a fitted vine object with a .sample() and .evaluation() method
      info_dict: dictionary with possible keys:
        'alpha': float, e.g. 0.05 => confidence level
        'cases': number of samples each iteration
        'iterations': max number of Monte Carlo iterations

    Returns:
      H_est: the estimated mean of log p(X), i.e. an approximate "entropy" if you interpret -E[log p(X)].
             If you want the Shannon entropy in nats, you might do -H_est.
    """
    alpha = info_dict.get('alpha', 0.05)
    cases = info_dict.get('cases', 1000)
    max_iter = info_dict.get('iterations', 10)

    import math
    from torch.distributions import Normal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 'conf' is the z-value for alpha (confidence intervals)
    norm = Normal(0.,1.)
    conf = norm.icdf(torch.tensor([1 - alpha], device=device)).item()

    # Running stats
    mo = 0                 # iteration count
    varsum = 0.0           # sum of squared deviations for incremental variance
    H_est = 0.0            # running average of log p

    # We'll do a simple repeated approach:
    for i in range(max_iter):
        mo += 1
        # 1) sample from vine
        sample_np = vine.sample(cases)  # shape [cases, dimension]
        sample_t = torch.tensor(sample_np, device=device, dtype=torch.float32)

        # 2) evaluate pdf => p, p_cop, etc.
        p, p_cop, logmarg = vine.evaluation(sample_t)
        # p => shape [cases], p(x)
        # 3) compute log p
        log_p = torch.log(torch.clamp(p, 1e-30, 1e30))

        # 4) average log p for this batch
        mean_lp = log_p.mean().item()

        # 5) incremental update H_est
        old_est = H_est
        H_est += (mean_lp - H_est) / mo

        # 6) track sum of squared deviations for variance estimate
        varsum += (mean_lp - H_est)*(mean_lp - old_est)*cases

    # If desired, we can compute a standard error:
    #    stderr = conf * sqrt(varsum / (mo*cases*( mo*cases -1 ))) 
    # or we can simply return H_est.

    return H_est