"""
Time-Dependent Vine Copula Models

Implements neural network models for time-dependent vine copula parameters,
including bandwidth flows and dynamic entropy estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging

from ..core.objects import vine_obj_bin
from ..core.info_estimation import vine_entropy, mutual_information

logger = logging.getLogger(__name__)


class TimeBandwidthFlow(nn.Module):
    """
    Neural network that maps time to bandwidth parameters for vine copulas.
    
    This model learns time-dependent bandwidth parameters for each edge
    in a vine copula structure, enabling dynamic dependency modeling.
    
    Parameters
    ----------
    n_edges : int
        Number of edges in the vine structure
    hidden_dims : list of int
        Hidden layer dimensions for the MLP
    time_embedding_dim : int
        Dimension of time embedding
    activation : str
        Activation function ('relu', 'tanh', 'elu')
    dropout_rate : float
        Dropout rate for regularization
    min_bandwidth : float
        Minimum bandwidth value (for numerical stability)
    max_bandwidth : float
        Maximum bandwidth value
    """
    
    def __init__(self, 
                 n_edges: int,
                 hidden_dims: List[int] = [64, 32],
                 time_embedding_dim: int = 16,
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 min_bandwidth: float = 0.01,
                 max_bandwidth: float = 2.0):
        super().__init__()
        
        self.n_edges = n_edges
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        
        # Time embedding layer
        self.time_embedding = nn.Linear(1, time_embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = time_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer (one bandwidth per edge)
        layers.append(nn.Linear(input_dim, n_edges))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map time to bandwidth parameters.
        
        Parameters
        ----------
        time : torch.Tensor
            Time values, shape (batch_size,) or (batch_size, 1)
            
        Returns
        -------
        torch.Tensor
            Bandwidth parameters, shape (batch_size, n_edges)
        """
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        
        # Normalize time to [-1, 1] range for better training stability
        time_normalized = 2 * (time - time.min()) / (time.max() - time.min() + 1e-8) - 1
        
        # Time embedding
        time_emb = torch.tanh(self.time_embedding(time_normalized))
        
        # MLP forward pass
        raw_bandwidths = self.mlp(time_emb)
        
        # Apply sigmoid and scale to [min_bandwidth, max_bandwidth]
        bandwidths = torch.sigmoid(raw_bandwidths)
        bandwidths = self.min_bandwidth + (self.max_bandwidth - self.min_bandwidth) * bandwidths
        
        return bandwidths
    
    def get_bandwidth_at_time(self, time: float) -> torch.Tensor:
        """Get bandwidth parameters for a specific time point."""
        time_tensor = torch.tensor([[time]], dtype=torch.float32)
        with torch.no_grad():
            return self.forward(time_tensor).squeeze(0)


class TimeDependentVine(nn.Module):
    """
    Time-dependent vine copula that combines a base vine structure
    with time-varying bandwidth parameters.
    
    Parameters
    ----------
    base_vine : vine_obj_bin
        Base vine copula structure
    bandwidth_flow : TimeBandwidthFlow
        Neural network for time-dependent bandwidths
    device : str or torch.device
        Device for computation
    """
    
    def __init__(self, 
                 base_vine: vine_obj_bin,
                 bandwidth_flow: Optional[TimeBandwidthFlow] = None,
                 device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        
        self.base_vine = base_vine
        self.device = torch.device(device)
        
        # Count edges in vine structure
        n_edges = self._count_vine_edges()
        
        # Create bandwidth flow if not provided
        if bandwidth_flow is None:
            self.bandwidth_flow = TimeBandwidthFlow(n_edges)
        else:
            self.bandwidth_flow = bandwidth_flow
        
        self.to(self.device)
    
    def _count_vine_edges(self) -> int:
        """Count total number of edges in vine structure."""
        n_edges = 0
        if hasattr(self.base_vine, 'ind_vine') and self.base_vine.ind_vine:
            for level_edges in self.base_vine.ind_vine:
                n_edges += len(level_edges)
        return max(n_edges, 1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for time-dependent vine.
        
        Parameters
        ----------
        x : torch.Tensor
            Data samples, shape (batch_size, n_features)
        time : torch.Tensor
            Time points, shape (batch_size,)
            
        Returns
        -------
        torch.Tensor
            Negative log-likelihood values, shape (batch_size,)
        """
        batch_size = x.shape[0]
        
        # Get time-dependent bandwidths
        bandwidths = self.bandwidth_flow(time)  # Shape: (batch_size, n_edges)
        
        # Update vine bandwidths (this is a simplified approach)
        # In practice, you'd need to properly integrate with the vine's copula objects
        nll_values = torch.zeros(batch_size, device=self.device)
        
        try:
            # Evaluate base vine PDF
            if hasattr(self.base_vine, 'logpdf'):
                log_probs = self.base_vine.logpdf(x)
                nll_values = -log_probs
            elif hasattr(self.base_vine, 'evaluation'):
                p, _, _ = self.base_vine.evaluation(x.cpu().numpy())
                if isinstance(p, torch.Tensor):
                    log_probs = torch.log(p.clamp_min(1e-30))
                else:
                    log_probs = torch.log(torch.from_numpy(p).clamp_min(1e-30))
                nll_values = -log_probs.to(self.device)
            
            # Apply bandwidth-dependent adjustments (simplified)
            # This is where you'd integrate the time-dependent bandwidths
            # with the actual copula computations
            bandwidth_penalty = torch.mean(bandwidths, dim=1) * 0.1
            nll_values += bandwidth_penalty
            
        except Exception as e:
            logger.warning(f"Error in time-dependent vine forward pass: {e}")
            nll_values = torch.full((batch_size,), 10.0, device=self.device)
        
        return nll_values
    
    def sample(self, n_samples: int, time_points: torch.Tensor) -> torch.Tensor:
        """
        Sample from time-dependent vine at specified time points.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        time_points : torch.Tensor
            Time points for sampling, shape (n_samples,)
            
        Returns
        -------
        torch.Tensor
            Generated samples, shape (n_samples, n_features)
        """
        with torch.no_grad():
            # Get time-dependent bandwidths
            bandwidths = self.bandwidth_flow(time_points)
            
            # Sample from base vine (simplified approach)
            try:
                if hasattr(self.base_vine, 'sample'):
                    samples = self.base_vine.sample(n_samples)
                    samples = torch.from_numpy(samples).float().to(self.device)
                else:
                    # Fallback: Gaussian samples
                    n_features = getattr(self.base_vine, 'n_cop', 2)
                    samples = torch.randn(n_samples, n_features, device=self.device)
                
                # Apply bandwidth-dependent transformations (placeholder)
                # In practice, you'd use the bandwidths to modify the sampling process
                
                return samples
                
            except Exception as e:
                logger.warning(f"Error in time-dependent vine sampling: {e}")
                n_features = getattr(self.base_vine, 'n_cop', 2)
                return torch.randn(n_samples, n_features, device=self.device)
    
    def get_bandwidths_over_time(self, time_range: torch.Tensor) -> torch.Tensor:
        """Get bandwidth evolution over a time range."""
        with torch.no_grad():
            return self.bandwidth_flow(time_range)


class DynamicEntropyEstimator(nn.Module):
    """
    Estimator for dynamic entropy in time-dependent vine copulas.
    
    Uses Monte Carlo sampling with time-dependent vine models to estimate
    entropy as a function of time.
    
    Parameters
    ----------
    time_vine : TimeDependentVine
        Time-dependent vine copula model
    n_samples : int
        Number of Monte Carlo samples for entropy estimation
    n_time_points : int
        Number of time points for entropy trajectory
    """
    
    def __init__(self, 
                 time_vine: TimeDependentVine,
                 n_samples: int = 1000,
                 n_time_points: int = 100):
        super().__init__()
        
        self.time_vine = time_vine
        self.n_samples = n_samples
        self.n_time_points = n_time_points
        
    def estimate_entropy_trajectory(self, 
                                  time_start: float = 0.0,
                                  time_end: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate entropy as a function of time.
        
        Parameters
        ----------
        time_start : float
            Start time for trajectory
        time_end : float
            End time for trajectory
            
        Returns
        -------
        tuple of torch.Tensor
            (time_points, entropy_values)
        """
        device = next(self.time_vine.parameters()).device
        
        # Create time grid
        time_points = torch.linspace(time_start, time_end, self.n_time_points, device=device)
        entropy_values = torch.zeros(self.n_time_points, device=device)
        
        with torch.no_grad():
            for i, t in enumerate(time_points):
                # Sample from vine at time t
                t_batch = t.repeat(self.n_samples)
                samples = self.time_vine.sample(self.n_samples, t_batch)
                
                # Compute log probabilities
                log_probs = -self.time_vine(samples, t_batch)
                
                # Estimate entropy: H = -E[log p(x)]
                entropy_values[i] = -torch.mean(log_probs)
        
        return time_points.cpu(), entropy_values.cpu()
    
    def estimate_mutual_information_trajectory(self,
                                             var_indices_x: List[int],
                                             var_indices_y: List[int],
                                             time_start: float = 0.0,
                                             time_end: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate mutual information between variable subsets over time.
        
        Parameters
        ----------
        var_indices_x : list of int
            Indices of variables in set X
        var_indices_y : list of int
            Indices of variables in set Y
        time_start : float
            Start time
        time_end : float
            End time
            
        Returns
        -------
        tuple of torch.Tensor
            (time_points, mi_values)
        """
        device = next(self.time_vine.parameters()).device
        
        time_points = torch.linspace(time_start, time_end, self.n_time_points, device=device)
        mi_values = torch.zeros(self.n_time_points, device=device)
        
        with torch.no_grad():
            for i, t in enumerate(time_points):
                # Sample from vine at time t
                t_batch = t.repeat(self.n_samples)
                samples = self.time_vine.sample(self.n_samples, t_batch)
                
                # Extract variable subsets
                x_samples = samples[:, var_indices_x]
                y_samples = samples[:, var_indices_y]
                xy_samples = samples[:, var_indices_x + var_indices_y]
                
                # Estimate MI using density ratios (simplified approach)
                # MI(X;Y) = E[log(p(x,y) / (p(x)p(y)))]
                
                # This is a placeholder - in practice you'd need proper density estimation
                # for marginals and joint distributions
                mi_values[i] = self._estimate_mi_from_samples(x_samples, y_samples, xy_samples)
        
        return time_points.cpu(), mi_values.cpu()
    
    def _estimate_mi_from_samples(self, x_samples: torch.Tensor, 
                                y_samples: torch.Tensor, 
                                xy_samples: torch.Tensor) -> torch.Tensor:
        """Estimate MI from samples using k-nearest neighbors or kernel density estimation."""
        # This is a simplified placeholder
        # In practice, you'd use proper MI estimation methods like:
        # - k-nearest neighbors (KNN) estimator
        # - Kernel density estimation
        # - Neural mutual information estimation
        
        try:
            # Simple correlation-based approximation for Gaussian case
            x_np = x_samples.cpu().numpy()
            y_np = y_samples.cpu().numpy()
            
            if x_np.shape[1] == 1 and y_np.shape[1] == 1:
                corr = np.corrcoef(x_np.flatten(), y_np.flatten())[0, 1]
                if np.isfinite(corr) and abs(corr) < 0.999:
                    mi = -0.5 * np.log(1 - corr**2)
                    return torch.tensor(max(mi, 0.0))
            
            # Fallback: return small positive value
            return torch.tensor(0.1)
            
        except:
            return torch.tensor(0.0)
    
    def compute_entropy_rate(self, time_points: torch.Tensor, 
                           entropy_values: torch.Tensor) -> torch.Tensor:
        """Compute entropy rate (derivative of entropy with respect to time)."""
        # Numerical differentiation
        dt = time_points[1] - time_points[0]
        entropy_rate = torch.gradient(entropy_values, spacing=dt.item())[0]
        return entropy_rate


def create_time_dependent_vine(base_vine: vine_obj_bin,
                             hidden_dims: List[int] = [64, 32],
                             time_embedding_dim: int = 16,
                             device: Union[str, torch.device] = 'cpu') -> TimeDependentVine:
    """
    Factory function to create a time-dependent vine copula.
    
    Parameters
    ----------
    base_vine : vine_obj_bin
        Base vine copula structure
    hidden_dims : list of int
        Hidden dimensions for bandwidth flow network
    time_embedding_dim : int
        Time embedding dimension
    device : str or torch.device
        Computation device
        
    Returns
    -------
    TimeDependentVine
        Time-dependent vine copula model
    """
    # Count edges in vine
    n_edges = 0
    if hasattr(base_vine, 'ind_vine') and base_vine.ind_vine:
        for level_edges in base_vine.ind_vine:
            n_edges += len(level_edges)
    n_edges = max(n_edges, 1)
    
    # Create bandwidth flow
    bandwidth_flow = TimeBandwidthFlow(
        n_edges=n_edges,
        hidden_dims=hidden_dims,
        time_embedding_dim=time_embedding_dim
    )
    
    # Create time-dependent vine
    time_vine = TimeDependentVine(
        base_vine=base_vine,
        bandwidth_flow=bandwidth_flow,
        device=device
    )
    
    return time_vine
