"""
Time-dependent bandwidth flows for vine copulas.

This module implements normalizing flows for modeling time-varying bandwidth
parameters in local likelihood estimation, enabling time-dependent vine copulas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class TimeBandwidthFlow(nn.Module):
    """
    Normalizing Flow for Time-Dependent Bandwidth Parameters.
    
    This neural network maps time indices to bandwidth parameters for 
    local likelihood estimation in vine copulas. Based on the TensorFlow
    implementation in DVC_NF.
    
    Architecture:
    - Input: time index t (normalized to [0,1])
    - Output: bandwidth parameter b(t) > 0 for each dimension
    - Constraint: b(t) must be positive for valid kernel estimation
    """
    
    def __init__(self, 
                 hidden_dim: int = 64,
                 output_dim: int = 2,
                 n_layers: int = 3,
                 activation: str = 'relu',
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True,
                 name: Optional[str] = None):
        """
        Initialize TimeBandwidthFlow.
        
        Args:
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (typically 2 for bivariate copulas)
            n_layers: Number of hidden layers
            activation: Activation function ('relu', 'elu', 'leaky_relu')
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            name: Optional name for the module
        """
        super(TimeBandwidthFlow, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(1, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor, training: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass: time -> bandwidth.
        
        Args:
            t: Time indices, shape (batch_size, 1) or (batch_size,)
                Should be normalized to [0, 1] range
            training: Whether in training mode (for dropout/batch_norm)
                
        Returns:
            bandwidth: Positive bandwidth values, shape (batch_size, output_dim)
        """
        # Ensure proper shape
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        
        # Set training mode if specified
        if training is not None:
            self.train(training)
        
        # Forward pass through network
        x = self.network(t)
        
        # Ensure positivity using softplus + small constant
        # This matches the TensorFlow implementation approach
        bandwidth = F.softplus(x) + 1e-4
        
        return bandwidth


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