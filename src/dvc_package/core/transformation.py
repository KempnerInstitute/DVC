###############################################
# src/DVC/transformation.py
###############################################

import torch
import math
from torch.distributions import Normal

class Transform:
    """
    Transform class for mapping between copula spaces:
    - uniform u-space [0,1]^2
    - normal s-space [-3.2,3.2]^2
    - rotated x-space via SVD/PCA
    """
    def __init__(self, n_cop: int):
        """
        Args:
            n_cop: number of copulas
        """
        self.n_cop = n_cop
        self.mu = None
        self.coeff = None
    
    def forward_u(self, obj_u: torch.Tensor) -> torch.Tensor:
        """
        Transform from uniform [0,1] space to normal [-3.2,3.2] space
        using inverse normal CDF.
        
        Args:
            obj_u: tensor in [0,1] space, shape [..., 2, n_cop] or [..., 2]
        Returns:
            obj_s: tensor in normal space, same shape
        """
        device = obj_u.device
        dtype = obj_u.dtype
        
        # Handle extreme values
        obj_u_safe = torch.clamp(obj_u, 1e-9, 1-1e-9)
        
        # Transform via inverse normal CDF
        normal_dist = torch.distributions.Normal(0., 1.)
        obj_s = normal_dist.icdf(obj_u_safe)
        
        # Clamp to [-3.2, 3.2] for numerical stability
        obj_s = torch.clamp(obj_s, -3.2, 3.2)
        
        return obj_s
    
    def forward_s(self, obj_s: torch.Tensor) -> torch.Tensor:
        """
        Transform from s-space to rotated x-space using SVD/PCA.
        
        Args:
            obj_s: tensor in normal space, shape [N, 2, n_cop] or [N, 2]
        Returns:
            obj_x: tensor in rotated space, same shape
        """
        device = obj_s.device
        dtype = obj_s.dtype
        
        # Handle 2D input
        if obj_s.dim() == 2:
            obj_s = obj_s.unsqueeze(-1)
            expand_needed = True
        else:
            expand_needed = False
        
        # Make compatible with n_cop if needed
        if obj_s.size(2) != self.n_cop:
            obj_s = obj_s.expand(-1, -1, self.n_cop)
        
        # Compute PCA coefficients if not already done
        if self.coeff is None:
            self.coeff = torch.zeros(2, 2, self.n_cop, dtype=dtype, device=device)
            for i in range(self.n_cop):
                # Compute SVD for each copula
                u, s, v = torch.linalg.svd(obj_s[:,:,i])
                coeff = v  # Use V as the rotation matrix (shape [2,2])
                
                # Ensure consistent sign (like TF implementation)
                # Find index of max abs value in first row
                max_idx = torch.argmax(torch.abs(coeff[0]))
                max_val = coeff[0, max_idx]
                sign_val = torch.sign(max_val)
                
                # Multiply by sign to ensure consistent direction
                coeff = coeff * sign_val
                
                # Store
                self.coeff[:,:,i] = coeff
        
        # Compute mean if not already done
        if self.mu is None:
            self.mu = obj_s.mean(dim=0)  # shape [2, n_cop]
        
        # Apply transformation: (x - mu) @ coeff
        obj_x = torch.zeros_like(obj_s)
        for i in range(self.n_cop):
            centered = obj_s[:,:,i] - self.mu[:,i]
            obj_x[:,:,i] = centered @ self.coeff[:,:,i]
        
        # Return in original format
        if expand_needed:
            obj_x = obj_x.squeeze(-1)
            
        return obj_x