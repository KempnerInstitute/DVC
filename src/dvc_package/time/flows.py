"""
Time-dependent bandwidth flows for vine copulas.

This module exposes the canonical time -> bandwidth network used across the
time-dependent API. It supports both the legacy ``output_dim`` interface and
the edge-aware ``n_edges`` interface used by ``TimeDependentVine``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TimeBandwidthFlow(nn.Module):
    """
    Unified time -> bandwidth network.

    Two constructor modes are supported:
    - Legacy mode: ``hidden_dim``/``output_dim``/``n_layers`` with softplus outputs.
    - Edge-aware mode: ``n_edges``/``hidden_dims`` with bounded outputs in
      ``[min_bandwidth, max_bandwidth]`` for local-likelihood evaluation.
    """

    def __init__(
        self,
        n_edges: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        time_embedding_dim: int = 16,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        min_bandwidth: float = 0.01,
        max_bandwidth: float = 2.0,
        hidden_dim: int = 64,
        output_dim: int = 2,
        n_layers: int = 3,
        use_batch_norm: Optional[bool] = None,
        name: Optional[str] = None,
    ):
        super().__init__()

        self.name = name
        self.mode = "edge" if n_edges is not None else "legacy"
        self.dropout_rate = float(dropout_rate)
        self.min_bandwidth = float(min_bandwidth)
        self.max_bandwidth = float(max_bandwidth)
        self.register_buffer("_time_min", torch.tensor(0.0), persistent=False)
        self.register_buffer("_time_max", torch.tensor(1.0), persistent=False)

        if self.mode == "edge":
            self.n_edges = int(max(n_edges, 1))
            self.output_dim = self.n_edges
            self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [64, 32]
            self.time_embedding_dim = int(time_embedding_dim)
            self.use_batch_norm = False if use_batch_norm is None else bool(use_batch_norm)
            self.time_embedding = nn.Linear(1, self.time_embedding_dim)

            layers = []
            input_dim = self.time_embedding_dim
            for width in self.hidden_dims:
                layers.append(nn.Linear(input_dim, int(width)))
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(int(width)))
                layers.append(self._get_activation(activation))
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
                input_dim = int(width)
            layers.append(nn.Linear(input_dim, self.n_edges))
            self.network = nn.Sequential(*layers)
        else:
            self.hidden_dim = int(hidden_dim)
            self.output_dim = int(output_dim)
            self.n_edges = self.output_dim
            self.n_layers = int(n_layers)
            self.use_batch_norm = True if use_batch_norm is None else bool(use_batch_norm)
            self.time_embedding = None

            layers = [nn.Linear(1, self.hidden_dim)]
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self._get_activation(activation))
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

            for _ in range(self.n_layers - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(self.hidden_dim))
                layers.append(self._get_activation(activation))
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))

            layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _get_activation(self, activation: str) -> nn.Module:
        """Return a fresh activation module by name."""
        if activation == "relu":
            return nn.ReLU()
        if activation == "elu":
            return nn.ELU()
        if activation == "tanh":
            return nn.Tanh()
        if activation == "leaky_relu":
            return nn.LeakyReLU(0.2)
        return nn.ReLU()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """Map absolute time values to [0, 1] for stable network inputs."""
        denom = (self._time_max - self._time_min).clamp_min(1e-8)
        return ((t - self._time_min) / denom).clamp(0.0, 1.0)

    def forward(self, t: torch.Tensor, training: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass: time -> bandwidth.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t = t.to(dtype=torch.float32)
        if training is not None:
            self.train(training)
        t01 = self._normalize_time(t)

        if self.mode == "edge":
            t_embed = torch.tanh(self.time_embedding(2.0 * t01 - 1.0))
            raw = self.network(t_embed)
            bw = torch.sigmoid(raw)
            return self.min_bandwidth + (self.max_bandwidth - self.min_bandwidth) * bw

        raw = self.network(t01)
        return F.softplus(raw) + 1e-4

    def set_time_range(self, t_min: float, t_max: float) -> None:
        """Store the absolute time span used by forward()."""
        t_lo = float(min(t_min, t_max))
        t_hi = float(max(t_min, t_max))
        if abs(t_hi - t_lo) < 1e-8:
            t_hi = t_lo + 1.0
        self._time_min = torch.tensor(t_lo, dtype=torch.float32, device=self._time_min.device)
        self._time_max = torch.tensor(t_hi, dtype=torch.float32, device=self._time_max.device)

    def get_bandwidth_at_time(self, time: float) -> torch.Tensor:
        """Evaluate the network at a single scalar time."""
        time_tensor = torch.tensor([[time]], dtype=torch.float32, device=self._time_min.device)
        with torch.no_grad():
            return self.forward(time_tensor).squeeze(0)


class MLPEdgeFlow(nn.Module):
    """
    MLP-based normalizing flow for individual vine edges.
    
    This is a simpler version focused on individual edge transformations,
    following the architecture in DVC_NF's MLPEdgeFlow.
    """
    
    def __init__(self,
                 hidden_dim: int = 64,
                 out_dim: int = 2,
                 dropout: float = 0.1):
        """
        Initialize MLPEdgeFlow.
        
        Args:
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            dropout: Dropout rate
        """
        super(MLPEdgeFlow, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # MLP layers
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, out_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: Time indices, shape (batch_size, 1) or (batch_size,)
            
        Returns:
            Output tensor, shape (batch_size, out_dim)
        """
        # Ensure proper shape
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        
        # Forward pass
        x = self.fc1(t)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc_out(x)
        
        # Ensure positivity for bandwidth parameters
        x = F.softplus(x) + 1e-4
        
        return x


__all__ = [
    'TimeBandwidthFlow',
    'MLPEdgeFlow'
]
