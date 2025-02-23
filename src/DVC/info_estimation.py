###############################################
# src/torch_vine/info_estimation.py
###############################################

import torch
import numpy as np

def vine_entropy(vine, info_dict: dict):
    alpha = info_dict.get('alpha', 0.05)
    cases = info_dict.get('cases', 1000)
    max_iter = info_dict.get('iterations', 10)
    import math
    from torch.distributions import Normal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm = Normal(0.,1.)
    conf = norm.icdf(torch.tensor([1-alpha], device=device)).item()
    mo = 0
    varsum = 0.0
    H_est = 0.0

    for i in range(max_iter):
        mo+=1
        sample_np = vine.sample(cases)
        sample_t = torch.tensor(sample_np, device=device, dtype=torch.float32)
        p, p_cop, logmarg = vine.evaluation(sample_t)
        log_p = torch.log(torch.clamp(p, 1e-30, 1e30))
        m_ = log_p.mean().item()
        old_est = H_est
        H_est += (m_ - H_est)/mo
        varsum += (m_ - H_est)*(m_ - old_est)*cases

    return H_est