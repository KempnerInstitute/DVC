# src/utils/nadam.py
import torch

def nadam_1d(obj_fn, a_init: torch.Tensor, lr=0.1, max_iter=100, tol=1e-5):
    """
    Optimize a scalar parameter using the NAdam optimizer.
    """
    a = a_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.NAdam([a], lr=lr)
    prev_val = obj_fn(a).item()
    for it in range(max_iter):
        optimizer.zero_grad()
        val = obj_fn(a)
        val.backward()
        optimizer.step()
        cur_val = val.item()
        if abs(cur_val - prev_val) < tol:
            break
        prev_val = cur_val
    return a.detach(), prev_val, it