"""
Normalizing flow utilities for time-dependent vine copulas.

This module provides bijectors, transformations, and utility functions
for normalizing flows, similar to TensorFlow Probability's bijectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Callable
import math


class Bijector(nn.Module):
    """
    Base class for bijective transformations.
    
    A bijector implements a bijective transformation with efficient
    computation of the forward transformation, inverse, and log-Jacobian.
    """
    
    def __init__(self, name: Optional[str] = None):
        super(Bijector, self).__init__()
        self.name = name or self.__class__.__name__
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transformation: x -> y."""
        raise NotImplementedError
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transformation: y -> x."""
        raise NotImplementedError
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log absolute determinant of Jacobian."""
        raise NotImplementedError


class Softplus(Bijector):
    """
    Softplus bijector for ensuring positivity.
    
    Forward: y = softplus(x) = log(1 + exp(x))
    Inverse: x = log(exp(y) - 1)
    """
    
    def __init__(self, low: float = 1e-6):
        super(Softplus, self).__init__()
        self.low = low
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x -> softplus(x) + low."""
        return F.softplus(x) + self.low
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse: y -> log(exp(y - low) - 1)."""
        y_shifted = y - self.low
        return torch.log(torch.expm1(y_shifted))
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log|det(dy/dx)| = log(sigmoid(x))."""
        return F.logsigmoid(x).sum(dim=-1)


class Exp(Bijector):
    """
    Exponential bijector.
    
    Forward: y = exp(x)
    Inverse: x = log(y)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x -> exp(x)."""
        return torch.exp(x)
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse: y -> log(y)."""
        return torch.log(y)
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log|det(dy/dx)| = x."""
        return x.sum(dim=-1)


class Sigmoid(Bijector):
    """
    Sigmoid bijector for mapping to (0,1).
    
    Forward: y = sigmoid(x)
    Inverse: x = logit(y)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x -> sigmoid(x)."""
        return torch.sigmoid(x)
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse: y -> logit(y)."""
        y_clamped = torch.clamp(y, 1e-7, 1-1e-7)
        return torch.log(y_clamped / (1 - y_clamped))
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log|det(dy/dx)| = log(sigmoid(x) * (1 - sigmoid(x)))."""
        return (F.logsigmoid(x) + F.logsigmoid(-x)).sum(dim=-1)


class Chain(Bijector):
    """
    Chain of bijectors: applies bijectors in sequence.
    """
    
    def __init__(self, bijectors: List[Bijector]):
        super(Chain, self).__init__()
        self.bijectors = nn.ModuleList(bijectors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bijectors in forward order."""
        y = x
        for bijector in self.bijectors:
            y = bijector.forward(y)
        return y
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply bijectors in reverse order."""
        x = y
        for bijector in reversed(self.bijectors):
            x = bijector.inverse(x)
        return x
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum of log-Jacobians."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        
        # Forward pass to compute intermediate values
        current = x
        for bijector in self.bijectors:
            next_val = bijector.forward(current)
            log_det += bijector.log_abs_det_jacobian(current, next_val)
            current = next_val
        
        return log_det


class ConditionalBijector(nn.Module):
    """
    Base class for conditional bijectors that depend on context.
    """
    
    def __init__(self, name: Optional[str] = None):
        super(ConditionalBijector, self).__init__()
        self.name = name or self.__class__.__name__
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward transformation conditioned on context."""
        raise NotImplementedError
    
    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inverse transformation conditioned on context."""
        raise NotImplementedError
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Log absolute determinant of Jacobian conditioned on context."""
        raise NotImplementedError


class ConditionalAffine(ConditionalBijector):
    """
    Conditional affine transformation: y = scale * x + shift.
    
    Both scale and shift are predicted from context using neural networks.
    """
    
    def __init__(self, 
                 input_dim: int,
                 context_dim: int,
                 hidden_dim: int = 64):
        super(ConditionalAffine, self).__init__()
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Networks for predicting scale and shift
        self.scale_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.shift_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def _get_params(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get scale and shift parameters from context."""
        scale = F.softplus(self.scale_net(context)) + 1e-6  # Ensure positive scale
        shift = self.shift_net(context)
        return scale, shift
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward: y = scale * x + shift."""
        scale, shift = self._get_params(context)
        return scale * x + shift
    
    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Inverse: x = (y - shift) / scale."""
        scale, shift = self._get_params(context)
        return (y - shift) / scale
    
    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Log|det(dy/dx)| = log|scale|."""
        scale, _ = self._get_params(context)
        return torch.log(torch.abs(scale)).sum(dim=-1)


# Utility functions for normalizing flows

def compute_kl_divergence(mean: torch.Tensor, 
                         logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between N(mean, var) and N(0, 1).
    
    Args:
        mean: Mean tensor
        logvar: Log variance tensor
        
    Returns:
        KL divergence
    """
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)


def sample_gaussian(mean: torch.Tensor, 
                   logvar: torch.Tensor, 
                   n_samples: int = 1) -> torch.Tensor:
    """
    Sample from Gaussian distribution using reparameterization trick.
    
    Args:
        mean: Mean tensor
        logvar: Log variance tensor
        n_samples: Number of samples
        
    Returns:
        Samples from the distribution
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(n_samples, *mean.shape, device=mean.device)
    return mean + std * eps


def log_normal_pdf(x: torch.Tensor, 
                  mean: torch.Tensor, 
                  logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute log PDF of multivariate normal distribution.
    
    Args:
        x: Input tensor
        mean: Mean tensor
        logvar: Log variance tensor
        
    Returns:
        Log PDF values
    """
    var = torch.exp(logvar)
    log_pdf = -0.5 * (torch.log(2 * math.pi * var) + (x - mean).pow(2) / var)
    return log_pdf.sum(dim=-1)


def create_flow_for_bandwidth(input_dim: int = 1,
                             output_dim: int = 2,
                             hidden_dim: int = 64,
                             n_layers: int = 3) -> Chain:
    """
    Create a normalizing flow for bandwidth parameters.
    
    Args:
        input_dim: Input dimension (time)
        output_dim: Output dimension (bandwidth dimensions)
        hidden_dim: Hidden dimension for networks
        n_layers: Number of layers
        
    Returns:
        Chain of bijectors
    """
    from .flows import TimeBandwidthFlow
    
    # Create the main transformation network
    transform_net = TimeBandwidthFlow(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers
    )
    
    # Wrap in a bijector that uses the network
    class NetworkBijector(Bijector):
        def __init__(self, network):
            super().__init__()
            self.network = network
        
        def forward(self, x):
            return self.network(x)
        
        def inverse(self, y):
            # For bandwidth flows, inverse is not typically needed
            # This is a placeholder - in practice, you'd need to solve numerically
            raise NotImplementedError("Inverse not implemented for NetworkBijector")
        
        def log_abs_det_jacobian(self, x, y):
            # Approximate log-Jacobian (would need exact computation in practice)
            return torch.zeros(x.shape[0], device=x.device)
    
    # Create chain with softplus for positivity
    bijectors = [
        NetworkBijector(transform_net),
        Softplus(low=1e-4)
    ]
    
    return Chain(bijectors)


__all__ = [
    'Bijector',
    'Softplus', 
    'Exp',
    'Sigmoid',
    'Chain',
    'ConditionalBijector',
    'ConditionalAffine',
    'compute_kl_divergence',
    'sample_gaussian',
    'log_normal_pdf',
    'create_flow_for_bandwidth'
]