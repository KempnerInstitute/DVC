###############################################
# src/torch_vine/transformation.py
###############################################

import torch
import math
from torch.distributions import Normal

class Transform:
    """
    Transform class for mapping uniform u-> normal s, or possibly PCA in s->x space, etc.
    """

    def __init__(self, n_cop: int):
        self.n_cop = n_cop
        self.coeff = None
        self.mu = None

    def forward_u(self, obj_u: torch.Tensor):
        eps = 1e-7
        clipped = torch.clamp(obj_u, eps, 1.0-eps)
        s = Normal(0.,1.).icdf(clipped)
        s = torch.clamp(s, -3.2, 3.2)
        return s

    def forward_s(self, obj_s: torch.Tensor):
        return obj_s