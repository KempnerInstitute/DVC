import torch
import numpy as np
from typing import Optional, Union
from param.margin_pdf import gaussian_pdf, student_pdf, clayton_pdf


class VineEntropyCalculator:
    """Calculate entropy and mutual information for vine copula models"""
    
    def __init__(self, vine_model):
        """
        Initialize entropy calculator
        
        Args:
            vine_model: Fitted vine_obj_bin instance
        """
        self.vine = vine_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def total_entropy(self, n_samples: int = 10000) -> float:
        """
        Calculate total entropy H(X) of the vine copula model
        
        Uses Monte Carlo integration:
        H(X) = -E[log p(X)] ≈ -1/n Σ log p(X_i)
        
        Args:
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Total entropy in nats
        """
        # Sample from the vine
        from sampling.vine_sampling import VineSampler
        sampler = VineSampler(self.vine)
        
        # Sample in uniform space
        U = sampler.sample_uniform(n_samples)
        
        # Calculate log density at each sample
        log_densities = self.log_density(U, uniform_space=True)
        
        # Monte Carlo estimate of entropy
        entropy = -torch.mean(log_densities).item()
        
        # Add marginal entropies if using empirical marginals
        marginal_entropy = self._marginal_entropy()
        
        return entropy + marginal_entropy
    
    def _marginal_entropy(self) -> float:
        """Calculate sum of marginal entropies"""
        total = 0.0
        
        for i, margin in enumerate(self.vine.margin):
            if margin.dist == 'kernel':
                # For kernel marginals, use differential entropy approximation
                # H(X) ≈ log(n) + γ + log(h) where h is bandwidth
                if hasattr(self.vine, 'Mar_G') and i < len(self.vine.Mar_G):
                    mar_s, mar_p = self.vine.Mar_G[i]
                    n = len(mar_s)
                    # Approximate using sample spacing
                    h = torch.mean(mar_s[1:] - mar_s[:-1]).item()
                    entropy_i = np.log(n) + 0.5772 + np.log(h)  # Euler's constant
                else:
                    entropy_i = 0.0
            elif margin.dist == 'gaussian':
                # H(X) = 0.5 * log(2πeσ²)
                if margin.theta is not None:
                    sigma = margin.theta[1] if len(margin.theta) > 1 else 1.0
                    entropy_i = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
                else:
                    entropy_i = 0.5 * np.log(2 * np.pi * np.e)
            else:
                # Default to 0 for unknown distributions
                entropy_i = 0.0
                
            total += entropy_i
            
        return total
    
    def copula_entropy(self, n_samples: int = 10000) -> float:
        """
        Calculate copula entropy (mutual information between variables)
        
        I(X1, ..., Xd) = Σ H(Xi) - H(X)
        
        Args:
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Copula entropy (mutual information) in nats
        """
        # Total entropy
        total_entropy = self.total_entropy(n_samples)
        
        # Sum of marginal entropies
        marginal_sum = self._marginal_entropy()
        
        # Mutual information
        return marginal_sum - total_entropy
    
    def conditional_entropy(self, target_dims: list, given_dims: list, 
                          n_samples: int = 10000) -> float:
        """
        Calculate conditional entropy H(X_target | X_given)
        
        Args:
            target_dims: Indices of target variables
            given_dims: Indices of conditioning variables
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Conditional entropy in nats
        """
        # H(Y|X) = H(X,Y) - H(X)
        
        # Get joint entropy of all variables
        all_dims = target_dims + given_dims
        joint_entropy = self._subset_entropy(all_dims, n_samples)
        
        # Get entropy of conditioning variables
        given_entropy = self._subset_entropy(given_dims, n_samples)
        
        return joint_entropy - given_entropy
    
    def _subset_entropy(self, dims: list, n_samples: int) -> float:
        """Calculate entropy of a subset of variables"""
        if len(dims) == 0:
            return 0.0
            
        # For subset entropy, we need to marginalize the vine
        # This is approximate - exact computation would require refitting
        
        # Sample from full vine
        from sampling.vine_sampling import VineSampler
        sampler = VineSampler(self.vine)
        U = sampler.sample_uniform(n_samples)
        
        # Keep only specified dimensions
        U_subset = U[:, dims]
        
        # Approximate entropy using k-NN or kernel density estimation
        return self._knn_entropy(U_subset)
    
    def _knn_entropy(self, X: torch.Tensor, k: int = 3) -> float:
        """
        Estimate entropy using k-nearest neighbors
        
        Kozachenko-Leonenko estimator:
        H(X) = ψ(n) - ψ(k) + log(cd) + d/n Σ log(2*ε_i)
        
        where ε_i is distance to k-th nearest neighbor
        """
        n, d = X.shape
        
        # Constants
        if d == 1:
            cd = 2
        elif d == 2:
            cd = np.pi
        else:
            cd = np.pi**(d/2) / np.math.gamma(d/2 + 1)
        
        # Compute pairwise distances
        distances = torch.cdist(X, X)
        
        # Get k-th nearest neighbor distances (excluding self)
        knn_distances, _ = torch.topk(distances, k+1, dim=1, largest=False)
        eps = knn_distances[:, k]  # k-th neighbor distance
        
        # Kozachenko-Leonenko estimator
        from scipy.special import digamma
        entropy = digamma(n) - digamma(k) + np.log(cd) + d * torch.mean(torch.log(2 * eps)).item()
        
        return entropy
    
    def log_density(self, U: torch.Tensor, uniform_space: bool = True) -> torch.Tensor:
        """
        Calculate log density at given points
        
        Args:
            U: Points to evaluate (n_samples, n_dim)
            uniform_space: Whether U is in uniform [0,1] space
            
        Returns:
            Log densities at each point
        """
        n_samples, n_dim = U.shape
        log_density = torch.zeros(n_samples, device=self.device)
        
        # If parametric vine
        if self.vine.param:
            # Add log densities from each copula in the vine
            for tree_idx, tree_copulas in enumerate(self.vine.copulas):
                for edge_idx, copula in enumerate(tree_copulas):
                    # Get the pair of variables for this copula
                    # This depends on vine structure - simplified here
                    if tree_idx == 0:
                        # First tree: original pairs
                        u1_idx, u2_idx = self._get_edge_indices(tree_idx, edge_idx)
                        if u1_idx < n_dim and u2_idx < n_dim:
                            u_pair = U[:, [u1_idx, u2_idx]]
                            log_density += self._parametric_log_pdf(copula, u_pair)
                    else:
                        # Higher trees: use h-functions (simplified)
                        pass
        else:
            # Non-parametric: use kernel density estimates
            # This is more complex and requires stored densities
            log_density = self._nonparametric_log_density(U)
            
        return log_density
    
    def _parametric_log_pdf(self, copula, u_pair: torch.Tensor) -> torch.Tensor:
        """Calculate log PDF for parametric copula"""
        eps = 1e-10
        u_pair = torch.clamp(u_pair, eps, 1-eps)
        
        if copula.family == 'gaussian':
            # Gaussian copula log density
            # c(u,v) = 1/sqrt(1-ρ²) * exp(-(ρ²(Φ⁻¹(u)² + Φ⁻¹(v)²) - 2ρΦ⁻¹(u)Φ⁻¹(v))/(2(1-ρ²)))
            rho = torch.tensor(copula.theta, device=self.device, dtype=torch.float32)
            
            # Inverse normal CDF
            normal = torch.distributions.Normal(0, 1)
            z1 = normal.icdf(u_pair[:, 0])
            z2 = normal.icdf(u_pair[:, 1])
            
            # Log density
            rho2 = rho * rho
            log_c = -0.5 * torch.log(1 - rho2).expand_as(z1)
            log_c += -(rho2 * (z1**2 + z2**2) - 2*rho*z1*z2) / (2*(1-rho2))
            log_c += (z1**2 + z2**2) / 2  # Remove marginal densities
            
            return log_c
            
        elif copula.family == 'student':
            # Student-t copula
            rho = copula.theta[0]
            nu = copula.theta[1]
            
            # Similar to Gaussian but with Student-t marginals
            t_dist = torch.distributions.StudentT(nu)
            t1 = t_dist.icdf(u_pair[:, 0])
            t2 = t_dist.icdf(u_pair[:, 1])
            
            # Log density (simplified)
            return self._student_copula_log_pdf(t1, t2, rho, nu)
            
        elif copula.family == 'clayton':
            # Clayton copula: C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
            theta = copula.theta
            
            if theta > 0:
                log_c = torch.log(1 + theta)
                log_c += -(theta + 1) * (torch.log(u_pair[:, 0]) + torch.log(u_pair[:, 1]))
                log_c += -(2 + 1/theta) * torch.log(
                    u_pair[:, 0]**(-theta) + u_pair[:, 1]**(-theta) - 1
                )
                return log_c
            else:
                return torch.zeros(len(u_pair), device=self.device)
                
        elif copula.family == 'ind':
            # Independence copula: density = 1
            return torch.zeros(len(u_pair), device=self.device)
            
        else:
            # Unknown family
            return torch.zeros(len(u_pair), device=self.device)
    
    def _student_copula_log_pdf(self, t1: torch.Tensor, t2: torch.Tensor, 
                                rho: float, nu: float) -> torch.Tensor:
        """Student-t copula log density"""
        # Simplified implementation
        rho2 = rho * rho
        factor = (t1**2 + t2**2 - 2*rho*t1*t2) / (nu * (1 - rho2))
        
        log_c = -0.5 * torch.log(1 - rho2)
        log_c += torch.lgamma((nu + 2)/2) - torch.lgamma(nu/2)
        log_c += -0.5 * torch.log(torch.tensor(nu * np.pi))
        log_c += -(nu + 2)/2 * torch.log(1 + factor)
        
        # Remove marginal contributions
        log_c += (nu + 1)/2 * (torch.log(1 + t1**2/nu) + torch.log(1 + t2**2/nu))
        
        return log_c
    
    def _nonparametric_log_density(self, U: torch.Tensor) -> torch.Tensor:
        """Estimate log density for non-parametric vine"""
        # This would require evaluating the stored kernel densities
        # Simplified implementation using KDE
        n_samples = len(U)
        log_density = torch.zeros(n_samples, device=self.device)
        
        # Add small noise to avoid exact zeros
        eps = 1e-10
        U = U + eps * torch.randn_like(U)
        
        # Simple product of marginals as approximation
        for i in range(U.shape[1]):
            # Use bounded KDE if available
            from utils.kde_wrapper import kde_wrapper
            density, mesh = kde_wrapper(U[:, i], method='fft', bounded=True)
            
            # Interpolate density at data points
            log_density += torch.log(density + eps).mean()
            
        return log_density
    
    def _get_edge_indices(self, tree: int, edge: int) -> tuple:
        """Get variable indices for given edge in tree"""
        # This depends on vine structure - simplified for C-vine
        if self.vine.vine_family == 'c-vine':
            if tree == 0:
                return (0, edge + 1)
            else:
                return (edge, edge + tree + 1)
        elif self.vine.vine_family == 'd-vine':
            return (edge, edge + tree + 1)
        else:
            # R-vine: would need to consult R-matrix
            return (0, 1)  # Simplified
    
    def mutual_information(self, dims1: list, dims2: list, n_samples: int = 10000) -> float:
        """
        Calculate mutual information between two groups of variables
        
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            dims1: Indices of first group
            dims2: Indices of second group
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Mutual information in nats
        """
        # Marginal entropies
        H_X = self._subset_entropy(dims1, n_samples)
        H_Y = self._subset_entropy(dims2, n_samples)
        
        # Joint entropy
        H_XY = self._subset_entropy(dims1 + dims2, n_samples)
        
        return H_X + H_Y - H_XY 