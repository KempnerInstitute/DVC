# src/pre_proc/preparation.py
import torch

def prep_cop(x: torch.Tensor, vine, sort_n: str):
    """
    Preprocess x for vine fitting. If sort_n == 'sort', sort by first column;
    if 'rand', shuffle randomly.
    """
    if sort_n == 'sort':
        idx = torch.argsort(x[:, 0])
        return x[idx]
    elif sort_n == 'rand':
        perm = torch.randperm(x.shape[0], device=x.device)
        return x[perm]
    else:
        return x