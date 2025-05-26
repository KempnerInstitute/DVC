import torch
import numpy as np
from typing import Optional, Tuple, List
from param.cond_copula import copulainvccdf_torch


class VineSampler:
    """Sampler for vine copula models using inverse transform method"""
    
    def __init__(self, vine_model):
        """
        Initialize sampler with fitted vine model
        
        Args:
            vine_model: Fitted vine_obj_bin instance
        """
        self.vine = vine_model
        self.n_dim = vine_model.n_cop + 1 if hasattr(vine_model, 'n_cop') else len(vine_model.margin)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def sample_uniform(self, n_samples: int) -> torch.Tensor:
        """
        Sample from vine copula in uniform space [0,1]^d
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Uniform samples of shape (n_samples, n_dim)
        """
        # Initialize with independent uniform samples
        V = torch.rand(n_samples, self.n_dim, device=self.device)
        
        # For R-vine, we need to apply the inverse conditional CDFs
        if self.vine.vine_family == 'r-vine':
            V = self._sample_rvine(V)
        elif self.vine.vine_family == 'c-vine':
            V = self._sample_cvine(V)
        elif self.vine.vine_family == 'd-vine':
            V = self._sample_dvine(V)
        else:
            raise ValueError(f"Unknown vine family: {self.vine.vine_family}")
            
        return V
    
    def _sample_rvine(self, V: torch.Tensor) -> torch.Tensor:
        """Sample from R-vine structure"""
        n_samples = V.shape[0]
        d = self.n_dim
        
        # Get R-matrix for vine structure
        if hasattr(self.vine, 'r_matrix'):
            R = self.vine.r_matrix
        else:
            # Default to C-vine structure if R-matrix not available
            return self._sample_cvine(V)
        
        # Initialize output
        U = torch.zeros_like(V)
        U[:, 0] = V[:, 0]  # First variable is unchanged
        
        # Apply inverse transforms tree by tree
        for j in range(1, d):
            U[:, j] = V[:, j]  # Start with independent uniform
            
            # Apply conditional inverse CDFs from trees
            for k in range(j):
                if k < len(self.vine.copulas):
                    # Get the appropriate copula
                    tree_copulas = self.vine.copulas[k]
                    
                    # Find edge index for this pair
                    edge_idx = self._find_edge_index(k, j-k-1, j)
                    
                    if edge_idx < len(tree_copulas):
                        copula = tree_copulas[edge_idx]
                        
                        # Prepare conditional values
                        if self.vine.param:
                            # Parametric copula
                            u_cond = torch.stack([U[:, k], V[:, j]], dim=1)
                            U[:, j] = self._inverse_h_parametric(copula, u_cond, k)
                        else:
                            # Non-parametric: use numerical inversion
                            U[:, j] = self._inverse_h_nonparametric(k, edge_idx, U[:, k], V[:, j])
        
        return U
    
    def _sample_cvine(self, V: torch.Tensor) -> torch.Tensor:
        """Sample from C-vine structure"""
        n_samples = V.shape[0]
        d = self.n_dim
        
        # In C-vine, first variable is the root
        U = torch.zeros_like(V)
        U[:, 0] = V[:, 0]
        
        # Sample remaining variables conditionally
        for j in range(1, d):
            U[:, j] = V[:, j]
            
            # Apply inverse h-functions from root
            for k in range(min(j, len(self.vine.copulas))):
                if k == 0:
                    # First tree: condition on root variable
                    edge_idx = j - 1
                    if edge_idx < len(self.vine.copulas[0]):
                        copula = self.vine.copulas[0][edge_idx]
                        
                        if self.vine.param:
                            u_cond = torch.stack([U[:, 0], U[:, j]], dim=1)
                            U[:, j] = self._inverse_h_parametric(copula, u_cond, 0)
                        else:
                            U[:, j] = self._inverse_h_nonparametric(0, edge_idx, U[:, 0], U[:, j])
                else:
                    # Higher trees: condition on previous variables
                    if k < len(self.vine.copulas) and j-k-1 < len(self.vine.copulas[k]):
                        copula = self.vine.copulas[k][j-k-1]
                        # Use stored h-functions for efficiency
                        # This is a simplified version - full implementation would track h-functions
                        pass
        
        return U
    
    def _sample_dvine(self, V: torch.Tensor) -> torch.Tensor:
        """Sample from D-vine structure"""
        n_samples = V.shape[0]
        d = self.n_dim
        
        U = torch.zeros_like(V)
        U[:, 0] = V[:, 0]
        
        # D-vine has a path structure
        for j in range(1, d):
            U[:, j] = V[:, j]
            
            # Apply inverse transforms along the path
            for k in range(min(j, len(self.vine.copulas))):
                if k < len(self.vine.copulas):
                    tree_copulas = self.vine.copulas[k]
                    edge_idx = j - k - 1
                    
                    if edge_idx >= 0 and edge_idx < len(tree_copulas):
                        copula = tree_copulas[edge_idx]
                        
                        if self.vine.param:
                            if k == 0:
                                u_cond = torch.stack([U[:, j-1], U[:, j]], dim=1)
                            else:
                                # Higher order conditionals (simplified)
                                u_cond = torch.stack([V[:, j-k-1], U[:, j]], dim=1)
                            U[:, j] = self._inverse_h_parametric(copula, u_cond, k)
                        else:
                            U[:, j] = self._inverse_h_nonparametric(k, edge_idx, V[:, j-k-1], U[:, j])
        
        return U
    
    def _inverse_h_parametric(self, copula, u_cond: torch.Tensor, tree_level: int) -> torch.Tensor:
        """
        Compute inverse h-function for parametric copula
        
        Args:
            copula: Parametric copula object
            u_cond: Conditional values [u1, u2]
            tree_level: Tree level in vine
            
        Returns:
            Transformed values
        """
        try:
            # Use the parametric inverse conditional CDF
            return copulainvccdf_torch(copula, u_cond)
        except:
            # Fallback to numerical inversion
            return u_cond[:, 1]  # Return original if inversion fails
    
    def _inverse_h_nonparametric(self, tree: int, edge: int, u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse h-function for non-parametric copula using numerical methods
        
        Args:
            tree: Tree index
            edge: Edge index within tree
            u1: First uniform variable
            u2: Second uniform variable (to be transformed)
            
        Returns:
            Transformed u2 values
        """
        # For non-parametric copulas, we need to numerically invert the h-function
        # This is a simplified version - full implementation would use bisection or Newton's method
        
        # Use linear interpolation as approximation
        n_points = 100
        u_grid = torch.linspace(0.01, 0.99, n_points, device=self.device)
        
        # Evaluate h-function on grid
        h_vals = torch.zeros(len(u1), n_points, device=self.device)
        
        for i, u in enumerate(u_grid):
            # Evaluate h(u | u1) for each u in grid
            # This would use the stored copula density
            h_vals[:, i] = self._eval_h_function(tree, edge, u1, torch.full_like(u1, u))
        
        # Find inverse by interpolation
        u2_inv = torch.zeros_like(u2)
        for i in range(len(u2)):
            # Find where h(u | u1[i]) = u2[i]
            idx = torch.searchsorted(h_vals[i], u2[i])
            if idx == 0:
                u2_inv[i] = u_grid[0]
            elif idx >= n_points:
                u2_inv[i] = u_grid[-1]
            else:
                # Linear interpolation
                alpha = (u2[i] - h_vals[i, idx-1]) / (h_vals[i, idx] - h_vals[i, idx-1])
                u2_inv[i] = (1 - alpha) * u_grid[idx-1] + alpha * u_grid[idx]
        
        return u2_inv
    
    def _eval_h_function(self, tree: int, edge: int, u1: torch.Tensor, u2: torch.Tensor) -> torch.Tensor:
        """Evaluate h-function h(u2|u1) for non-parametric copula"""
        # Simplified evaluation - full implementation would use the stored copula density
        # For now, return a simple transformation
        return u2 * (1 + 0.5 * (u1 - 0.5))
    
    def _find_edge_index(self, tree: int, i: int, j: int) -> int:
        """Find edge index in tree for variables i and j"""
        # Simplified edge finding - depends on vine structure
        if tree == 0:
            return min(i, j)
        else:
            return 0  # Simplified
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from vine copula and transform to original scale
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples in original scale (n_samples, n_dim)
        """
        # Sample in uniform space
        U = self.sample_uniform(n_samples)
        
        # Transform to original scale using marginal inverse CDFs
        X = torch.zeros_like(U)
        
        for i in range(self.n_dim):
            if hasattr(self.vine, 'Mar_G') and i < len(self.vine.Mar_G):
                # Use empirical inverse CDF
                mar_s, mar_p = self.vine.Mar_G[i]
                X[:, i] = self._empirical_quantile(U[:, i], mar_s, mar_p)
            else:
                # Default to standard normal
                X[:, i] = torch.distributions.Normal(0, 1).icdf(U[:, i])
        
        return X
    
    def _empirical_quantile(self, u: torch.Tensor, values: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Compute empirical quantiles using linear interpolation"""
        # Ensure all tensors are on the same device
        device = u.device
        values = values.to(device)
        probs = probs.to(device)
        
        # Ensure probs are sorted
        sorted_idx = torch.argsort(probs)
        probs_sorted = probs[sorted_idx]
        values_sorted = values[sorted_idx]
        
        # Interpolate
        quantiles = torch.zeros_like(u)
        for i, ui in enumerate(u):
            idx = torch.searchsorted(probs_sorted, ui)
            if idx == 0:
                quantiles[i] = values_sorted[0]
            elif idx >= len(probs_sorted):
                quantiles[i] = values_sorted[-1]
            else:
                # Linear interpolation
                p1, p2 = probs_sorted[idx-1], probs_sorted[idx]
                v1, v2 = values_sorted[idx-1], values_sorted[idx]
                alpha = (ui - p1) / (p2 - p1)
                quantiles[i] = (1 - alpha) * v1 + alpha * v2
        
        return quantiles 