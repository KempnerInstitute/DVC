import torch
import numpy as np
from typing import Optional, Tuple, List, Union
from utils.prob_op import kernel_cdf, kde_wrapper


class VineConditionalPredictor:
    """Predict conditional distributions using vine copula models"""
    
    def __init__(self, vine_model):
        """
        Initialize conditional predictor
        
        Args:
            vine_model: Fitted vine_obj_bin instance
        """
        self.vine = vine_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._cache_h_functions()
        
    def _cache_h_functions(self):
        """Pre-compute and cache h-functions for efficiency"""
        self.h_functions = {}
        self.h_inverse_functions = {}
        
        # Cache h-functions for each edge in the vine
        for tree_idx, tree_copulas in enumerate(self.vine.copulas):
            self.h_functions[tree_idx] = {}
            self.h_inverse_functions[tree_idx] = {}
            
            for edge_idx, copula in enumerate(tree_copulas):
                # Store forward and inverse h-functions
                self.h_functions[tree_idx][edge_idx] = lambda u, v, cop=copula: self._h_function(u, v, cop)
                self.h_inverse_functions[tree_idx][edge_idx] = lambda u, v, cop=copula: self._h_inverse(u, v, cop)
    
    def predict_conditional(self, target_dims: List[int], given_dims: List[int], 
                          given_values: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """
        Predict conditional distribution P(X_target | X_given = given_values)
        
        Args:
            target_dims: Indices of variables to predict
            given_dims: Indices of conditioning variables
            given_values: Values of conditioning variables (n_obs, len(given_dims))
            n_samples: Number of samples to generate for each observation
            
        Returns:
            Conditional samples (n_obs, n_samples, len(target_dims))
        """
        n_obs = given_values.shape[0]
        n_target = len(target_dims)
        
        # Initialize output
        conditional_samples = torch.zeros(n_obs, n_samples, n_target, device=self.device)
        
        # For each observation
        for i in range(n_obs):
            # Transform given values to uniform scale
            u_given = self._transform_to_uniform(given_values[i], given_dims)
            
            # Sample from conditional distribution
            samples = self._sample_conditional(target_dims, given_dims, u_given, n_samples)
            
            # Transform back to original scale
            conditional_samples[i] = self._transform_from_uniform(samples, target_dims)
        
        return conditional_samples
    
    def conditional_mean(self, target_dims: List[int], given_dims: List[int],
                        given_values: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """
        Estimate conditional mean E[X_target | X_given = given_values]
        
        Args:
            target_dims: Indices of variables to predict
            given_dims: Indices of conditioning variables
            given_values: Values of conditioning variables (n_obs, len(given_dims))
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            Conditional means (n_obs, len(target_dims))
        """
        # Generate conditional samples
        samples = self.predict_conditional(target_dims, given_dims, given_values, n_samples)
        
        # Compute mean
        return torch.mean(samples, dim=1)
    
    def conditional_quantiles(self, target_dims: List[int], given_dims: List[int],
                            given_values: torch.Tensor, quantiles: List[float],
                            n_samples: int = 5000) -> torch.Tensor:
        """
        Estimate conditional quantiles
        
        Args:
            target_dims: Indices of variables to predict
            given_dims: Indices of conditioning variables  
            given_values: Values of conditioning variables (n_obs, len(given_dims))
            quantiles: List of quantiles to compute (e.g., [0.25, 0.5, 0.75])
            n_samples: Number of samples for estimation
            
        Returns:
            Conditional quantiles (n_obs, len(target_dims), len(quantiles))
        """
        # Generate conditional samples
        samples = self.predict_conditional(target_dims, given_dims, given_values, n_samples)
        
        # Compute quantiles
        n_obs, _, n_target = samples.shape
        n_quantiles = len(quantiles)
        
        result = torch.zeros(n_obs, n_target, n_quantiles, device=self.device)
        
        for i in range(n_obs):
            for j in range(n_target):
                sorted_samples, _ = torch.sort(samples[i, :, j])
                for k, q in enumerate(quantiles):
                    idx = int(q * n_samples)
                    result[i, j, k] = sorted_samples[min(idx, n_samples-1)]
        
        return result
    
    def conditional_density(self, target_dim: int, given_dims: List[int],
                          given_values: torch.Tensor, eval_points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate conditional density p(x_target | x_given = given_values)
        
        Args:
            target_dim: Index of target variable (single dimension)
            given_dims: Indices of conditioning variables
            given_values: Values of conditioning variables (n_obs, len(given_dims))
            eval_points: Points to evaluate density at (n_points,)
            
        Returns:
            Conditional densities (n_obs, n_points)
        """
        n_obs = given_values.shape[0]
        n_points = len(eval_points)
        
        densities = torch.zeros(n_obs, n_points, device=self.device)
        
        for i in range(n_obs):
            # Transform to uniform scale
            u_given = self._transform_to_uniform(given_values[i], given_dims)
            u_eval = self._transform_to_uniform(eval_points, [target_dim])
            
            # Evaluate conditional copula density
            for j in range(n_points):
                densities[i, j] = self._eval_conditional_density(
                    target_dim, given_dims, u_eval[j], u_given
                )
        
        # Multiply by marginal density of target
        marginal_density = self._eval_marginal_density(eval_points, target_dim)
        densities *= marginal_density.unsqueeze(0)
        
        return densities
    
    def _sample_conditional(self, target_dims: List[int], given_dims: List[int],
                          u_given: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Sample from conditional distribution in uniform space
        
        Uses the vine structure to sample conditionally
        """
        d = len(self.vine.margin)
        all_dims = list(range(d))
        
        # Initialize samples
        u_samples = torch.zeros(n_samples, d, device=self.device)
        
        # Set given values
        for i, dim in enumerate(given_dims):
            u_samples[:, dim] = u_given[i]
        
        # Sample remaining dimensions using vine structure
        remaining_dims = [dim for dim in all_dims if dim not in given_dims]
        
        # Sort dimensions according to vine ordering
        if self.vine.vine_family == 'c-vine':
            # In C-vine, sample from root first if not given
            ordered_dims = self._order_for_cvine(remaining_dims, given_dims)
        elif self.vine.vine_family == 'd-vine':
            # In D-vine, sample along the path
            ordered_dims = self._order_for_dvine(remaining_dims, given_dims)
        else:
            # R-vine: use natural ordering
            ordered_dims = sorted(remaining_dims)
        
        # Sample each dimension conditionally
        for dim in ordered_dims:
            if dim in target_dims:
                # Sample using conditional inverse transform
                u_samples[:, dim] = self._sample_dimension_conditional(
                    dim, u_samples, given_dims + [d for d in ordered_dims if d < dim]
                )
        
        # Return only target dimensions
        return u_samples[:, target_dims]
    
    def _sample_dimension_conditional(self, dim: int, u_current: torch.Tensor,
                                    conditioning_dims: List[int]) -> torch.Tensor:
        """Sample a single dimension conditionally on others"""
        n_samples = u_current.shape[0]
        
        # Generate uniform samples
        u_indep = torch.rand(n_samples, device=self.device)
        
        # Apply inverse h-transforms based on vine structure
        u_transformed = u_indep
        
        # Find path in vine to this dimension
        if self.vine.vine_family == 'c-vine':
            # Apply h-transforms from root
            if 0 in conditioning_dims and dim > 0:
                # First tree transformation
                edge_idx = dim - 1
                if edge_idx < len(self.vine.copulas[0]):
                    h_inv = self.h_inverse_functions[0][edge_idx]
                    u_transformed = h_inv(u_current[:, 0], u_transformed)
        
        elif self.vine.vine_family == 'd-vine':
            # Apply h-transforms along path
            for cond_dim in conditioning_dims:
                if abs(cond_dim - dim) == 1:
                    # Adjacent in D-vine
                    tree = 0
                    edge = min(cond_dim, dim)
                    if edge < len(self.vine.copulas[0]):
                        h_inv = self.h_inverse_functions[tree][edge]
                        u_transformed = h_inv(u_current[:, cond_dim], u_transformed)
        
        return u_transformed
    
    def _h_function(self, u: torch.Tensor, v: torch.Tensor, copula) -> torch.Tensor:
        """
        Compute h-function h(v|u) = ∂C(u,v)/∂u
        
        For parametric copulas, use analytical derivatives
        For non-parametric, use numerical differentiation
        """
        if self.vine.param and hasattr(copula, 'family'):
            if copula.family == 'gaussian':
                return self._h_gaussian(u, v, copula.theta)
            elif copula.family == 'student':
                return self._h_student(u, v, copula.theta[0], copula.theta[1])
            elif copula.family == 'clayton':
                return self._h_clayton(u, v, copula.theta)
            elif copula.family == 'ind':
                return v  # Independence copula
        
        # Non-parametric: numerical differentiation
        return self._h_numerical(u, v, copula)
    
    def _h_gaussian(self, u: torch.Tensor, v: torch.Tensor, rho: float) -> torch.Tensor:
        """h-function for Gaussian copula"""
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(u)
        z2 = normal.icdf(v)
        
        # h(v|u) = Φ((z2 - ρz1) / sqrt(1 - ρ²))
        h = normal.cdf((z2 - rho * z1) / torch.sqrt(1 - rho**2))
        return torch.clamp(h, 1e-10, 1 - 1e-10)
    
    def _h_student(self, u: torch.Tensor, v: torch.Tensor, rho: float, nu: float) -> torch.Tensor:
        """h-function for Student-t copula"""
        t_dist = torch.distributions.StudentT(nu)
        t1 = t_dist.icdf(u)
        t2 = t_dist.icdf(v)
        
        # h(v|u) = T_{ν+1}((t2 - ρt1) / sqrt((ν + t1²)(1 - ρ²) / (ν + 1)))
        numerator = t2 - rho * t1
        denominator = torch.sqrt((nu + t1**2) * (1 - rho**2) / (nu + 1))
        
        t_dist_cond = torch.distributions.StudentT(nu + 1)
        h = t_dist_cond.cdf(numerator / denominator)
        return torch.clamp(h, 1e-10, 1 - 1e-10)
    
    def _h_clayton(self, u: torch.Tensor, v: torch.Tensor, theta: float) -> torch.Tensor:
        """h-function for Clayton copula"""
        if theta > 0:
            # h(v|u) = u^(-θ-1) * (u^(-θ) + v^(-θ) - 1)^(-1/θ - 1)
            term1 = u**(-theta - 1)
            term2 = (u**(-theta) + v**(-theta) - 1)**(-1/theta - 1)
            h = term1 * term2
            return torch.clamp(h, 1e-10, 1 - 1e-10)
        else:
            return v  # Independence
    
    def _h_numerical(self, u: torch.Tensor, v: torch.Tensor, copula) -> torch.Tensor:
        """Numerical h-function using finite differences"""
        eps = 1e-6
        
        # Evaluate copula CDF at (u+eps, v) and (u-eps, v)
        u_plus = torch.clamp(u + eps, 0, 1)
        u_minus = torch.clamp(u - eps, 0, 1)
        
        # This would require access to copula CDF evaluation
        # Simplified: return v for now
        return v
    
    def _h_inverse(self, u: torch.Tensor, h: torch.Tensor, copula) -> torch.Tensor:
        """
        Compute inverse h-function: find v such that h(v|u) = h
        
        Uses bisection method for numerical inversion
        """
        if self.vine.param and hasattr(copula, 'family'):
            if copula.family == 'gaussian':
                return self._h_inverse_gaussian(u, h, copula.theta)
            elif copula.family == 'ind':
                return h  # Independence copula
        
        # Numerical inversion using bisection
        return self._h_inverse_bisection(u, h, copula)
    
    def _h_inverse_gaussian(self, u: torch.Tensor, h: torch.Tensor, rho: float) -> torch.Tensor:
        """Inverse h-function for Gaussian copula"""
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(u)
        
        # v = Φ(ρz1 + sqrt(1-ρ²) * Φ^(-1)(h))
        z_h = normal.icdf(h)
        z2 = rho * z1 + torch.sqrt(1 - rho**2) * z_h
        v = normal.cdf(z2)
        
        return torch.clamp(v, 1e-10, 1 - 1e-10)
    
    def _h_inverse_bisection(self, u: torch.Tensor, h_target: torch.Tensor, 
                           copula, tol: float = 1e-6, max_iter: int = 50) -> torch.Tensor:
        """Numerical inverse using bisection"""
        v_low = torch.zeros_like(u)
        v_high = torch.ones_like(u)
        
        for _ in range(max_iter):
            v_mid = (v_low + v_high) / 2
            h_mid = self._h_function(u, v_mid, copula)
            
            # Update bounds
            mask_low = h_mid < h_target
            v_low = torch.where(mask_low, v_mid, v_low)
            v_high = torch.where(~mask_low, v_mid, v_high)
            
            # Check convergence
            if torch.max(v_high - v_low) < tol:
                break
        
        return (v_low + v_high) / 2
    
    def _transform_to_uniform(self, values: torch.Tensor, dims: List[int]) -> torch.Tensor:
        """Transform values to uniform scale using marginal CDFs"""
        uniform_values = torch.zeros_like(values)
        
        for i, dim in enumerate(dims):
            if dim < len(self.vine.margin):
                margin = self.vine.margin[dim]
                
                if margin.dist == 'kernel' and hasattr(self.vine, 'Mar_G'):
                    # Use empirical CDF
                    mar_s, mar_p = self.vine.Mar_G[dim]
                    cdf_result = kernel_cdf(values[i:i+1], mar_s, mar_p)
                    uniform_values[i] = cdf_result[0]  # First element is the CDF value
                elif margin.dist == 'gaussian':
                    # Gaussian marginal
                    if margin.theta is not None:
                        mu, sigma = margin.theta[0], margin.theta[1]
                        normal = torch.distributions.Normal(mu, sigma)
                        uniform_values[i] = normal.cdf(values[i])
                    else:
                        normal = torch.distributions.Normal(0, 1)
                        uniform_values[i] = normal.cdf(values[i])
                else:
                    # Default: no transformation
                    uniform_values[i] = values[i]
        
        return uniform_values
    
    def _transform_from_uniform(self, u_values: torch.Tensor, dims: List[int]) -> torch.Tensor:
        """Transform from uniform scale to original scale using inverse marginal CDFs"""
        n_samples, n_dims = u_values.shape
        original_values = torch.zeros_like(u_values)
        
        for j, dim in enumerate(dims):
            if dim < len(self.vine.margin):
                margin = self.vine.margin[dim]
                
                if margin.dist == 'kernel' and hasattr(self.vine, 'Mar_G'):
                    # Use empirical quantile function
                    mar_s, mar_p = self.vine.Mar_G[dim]
                    for i in range(n_samples):
                        original_values[i, j] = self._empirical_quantile(
                            u_values[i, j:j+1], mar_s, mar_p
                        )
                elif margin.dist == 'gaussian':
                    # Gaussian marginal
                    if margin.theta is not None:
                        mu, sigma = margin.theta[0], margin.theta[1]
                        normal = torch.distributions.Normal(mu, sigma)
                        original_values[:, j] = normal.icdf(u_values[:, j])
                    else:
                        normal = torch.distributions.Normal(0, 1)
                        original_values[:, j] = normal.icdf(u_values[:, j])
                else:
                    # Default: no transformation
                    original_values[:, j] = u_values[:, j]
        
        return original_values
    
    def _empirical_quantile(self, u: torch.Tensor, values: torch.Tensor, 
                          probs: torch.Tensor) -> torch.Tensor:
        """Compute empirical quantile"""
        idx = torch.searchsorted(probs, u)
        idx = torch.clamp(idx, 0, len(values) - 1)
        return values[idx]
    
    def _eval_marginal_density(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Evaluate marginal density"""
        margin = self.vine.margin[dim]
        
        if margin.dist == 'kernel':
            # Use KDE
            density, _ = kde_wrapper(x, method='fft', bounded=False)
            return density
        elif margin.dist == 'gaussian':
            if margin.theta is not None:
                mu, sigma = margin.theta[0], margin.theta[1]
                normal = torch.distributions.Normal(mu, sigma)
                return torch.exp(normal.log_prob(x))
            else:
                normal = torch.distributions.Normal(0, 1)
                return torch.exp(normal.log_prob(x))
        else:
            # Default: uniform density
            return torch.ones_like(x)
    
    def _eval_conditional_density(self, target_dim: int, given_dims: List[int],
                                u_target: torch.Tensor, u_given: torch.Tensor) -> torch.Tensor:
        """Evaluate conditional copula density"""
        # This is a simplified implementation
        # Full implementation would trace through vine structure
        
        # For now, return product of relevant pair copula densities
        density = torch.tensor(1.0, device=self.device)
        
        # Add contributions from relevant copulas
        # This depends on vine structure
        
        return density
    
    def _order_for_cvine(self, remaining_dims: List[int], given_dims: List[int]) -> List[int]:
        """Order dimensions for C-vine sampling"""
        # In C-vine, dimension 0 is root
        if 0 in remaining_dims:
            # Start with root
            ordered = [0]
            ordered.extend([d for d in remaining_dims if d != 0])
        else:
            # Root is given, use natural order
            ordered = sorted(remaining_dims)
        return ordered
    
    def _order_for_dvine(self, remaining_dims: List[int], given_dims: List[int]) -> List[int]:
        """Order dimensions for D-vine sampling"""
        # In D-vine, sample along the path
        # Find connected components
        ordered = []
        remaining = set(remaining_dims)
        
        while remaining:
            # Find a starting point
            start = min(remaining)
            ordered.append(start)
            remaining.remove(start)
            
            # Extend path in both directions
            current = start
            while True:
                # Find neighbor
                found = False
                for dim in remaining:
                    if abs(dim - current) == 1:
                        ordered.append(dim)
                        remaining.remove(dim)
                        current = dim
                        found = True
                        break
                if not found:
                    break
        
        return ordered 