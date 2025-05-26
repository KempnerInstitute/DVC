import torch
import numpy as np
from typing import Optional, Tuple, Union
from scipy.optimize import minimize_scalar, differential_evolution
from utils.prob_op import kde_wrapper
from utils.kde_simple import kde_1d_cdist_chunked


class BandwidthSelector:
    """Advanced bandwidth selection methods for kernel density estimation"""
    
    def __init__(self, method: str = 'cv', kernel: str = 'gaussian'):
        """
        Initialize bandwidth selector
        
        Args:
            method: Selection method ('cv', 'ml', 'plugin', 'adaptive')
            kernel: Kernel type ('gaussian', 'epanechnikov')
        """
        self.method = method
        self.kernel = kernel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def select_bandwidth(self, data: torch.Tensor, 
                        bounds: Optional[Tuple[float, float]] = None,
                        init_h: Optional[float] = None) -> float:
        """
        Select optimal bandwidth for given data
        
        Args:
            data: 1D or 2D data tensor
            bounds: Bandwidth search bounds
            init_h: Initial bandwidth guess
            
        Returns:
            Optimal bandwidth
        """
        if data.dim() == 1:
            return self._select_1d(data, bounds, init_h)
        else:
            return self._select_2d(data, bounds, init_h)
            
    def _select_1d(self, data: torch.Tensor, bounds: Optional[Tuple[float, float]] = None,
                   init_h: Optional[float] = None) -> float:
        """Select bandwidth for 1D data"""
        n = len(data)
        
        # Default bounds based on data
        if bounds is None:
            sigma = torch.std(data).item()
            iqr = (torch.quantile(data, 0.75) - torch.quantile(data, 0.25)).item()
            h_min = 0.1 * min(sigma, iqr/1.349) * n**(-1/5)
            h_max = 2.0 * max(sigma, iqr/1.349) * n**(-1/5)
            bounds = (max(1e-6, h_min), h_max)
            
        # Initial guess
        if init_h is None:
            init_h = self._silverman_rule_1d(data)
            
        if self.method == 'cv':
            return self._cv_bandwidth_1d(data, bounds)
        elif self.method == 'ml':
            return self._ml_bandwidth_1d(data, bounds)
        elif self.method == 'plugin':
            return self._plugin_bandwidth_1d(data)
        elif self.method == 'adaptive':
            return self._adaptive_bandwidth_1d(data, init_h)
        else:
            # Default to Silverman's rule
            return self._silverman_rule_1d(data)
            
    def _silverman_rule_1d(self, data: torch.Tensor) -> float:
        """Silverman's rule of thumb for 1D data"""
        n = len(data)
        sigma = torch.std(data).item()
        iqr = (torch.quantile(data, 0.75) - torch.quantile(data, 0.25)).item()
        
        # Robust estimate of scale
        scale = min(sigma, iqr/1.349)
        
        # Silverman's rule
        return 0.9 * scale * n**(-1/5)
        
    def _scott_rule_1d(self, data: torch.Tensor) -> float:
        """Scott's rule for 1D data"""
        n = len(data)
        sigma = torch.std(data).item()
        return 1.06 * sigma * n**(-1/5)
        
    def _cv_bandwidth_1d(self, data: torch.Tensor, bounds: Tuple[float, float]) -> float:
        """Cross-validation bandwidth selection"""
        n = len(data)
        
        def cv_score(h):
            """Leave-one-out cross-validation score"""
            if h <= 0:
                return np.inf
                
            # Compute pairwise distances
            dists = torch.cdist(data.unsqueeze(1), data.unsqueeze(1)).squeeze()
            
            # Gaussian kernel: K(u) = 1/sqrt(2π) * exp(-u²/2)
            # For leave-one-out, exclude diagonal
            mask = ~torch.eye(n, dtype=torch.bool, device=data.device)
            
            # Kernel evaluations
            K = torch.exp(-0.5 * (dists/h)**2) / (h * np.sqrt(2*np.pi))
            K_loo = K * mask  # Zero out diagonal
            
            # Leave-one-out density estimates
            f_loo = torch.sum(K_loo, dim=1) / (n - 1)
            
            # Score: -2 * mean(log(f_loo)) + R(K)
            # R(K) is the integrated squared kernel (constant for Gaussian)
            R_K = 1 / (2 * np.sqrt(np.pi))
            
            # Avoid log(0)
            f_loo = torch.clamp(f_loo, min=1e-10)
            score = -2 * torch.mean(torch.log(f_loo)) + R_K / (n * h)
            
            return score.item()
            
        # Optimize
        result = minimize_scalar(cv_score, bounds=bounds, method='bounded')
        return result.x
        
    def _ml_bandwidth_1d(self, data: torch.Tensor, bounds: Tuple[float, float]) -> float:
        """Maximum likelihood cross-validation"""
        n = len(data)
        
        def ml_score(h):
            """Maximum likelihood score"""
            if h <= 0:
                return -np.inf
                
            # Similar to CV but maximize likelihood
            dists = torch.cdist(data.unsqueeze(1), data.unsqueeze(1)).squeeze()
            mask = ~torch.eye(n, dtype=torch.bool, device=data.device)
            
            K = torch.exp(-0.5 * (dists/h)**2) / (h * np.sqrt(2*np.pi))
            K_loo = K * mask
            
            f_loo = torch.sum(K_loo, dim=1) / (n - 1)
            f_loo = torch.clamp(f_loo, min=1e-10)
            
            # Maximize log-likelihood
            return torch.sum(torch.log(f_loo)).item()
            
        # Optimize (note: we minimize negative log-likelihood)
        result = minimize_scalar(lambda h: -ml_score(h), bounds=bounds, method='bounded')
        return result.x
        
    def _plugin_bandwidth_1d(self, data: torch.Tensor) -> float:
        """Plug-in bandwidth selection (Sheather-Jones)"""
        n = len(data)
        sigma = torch.std(data).item()
        
        # Estimate integrated squared second derivative
        # For Gaussian reference: ψ₄ = 3/(8√π σ⁵)
        psi4_ref = 3 / (8 * np.sqrt(np.pi) * sigma**5)
        
        # Pilot bandwidth for estimating ψ₄
        g1 = 0.9 * sigma * n**(-1/7)
        
        # Estimate ψ₄ using pilot bandwidth
        # Simplified: use reference value scaled by kurtosis
        kurt = self._kurtosis(data)
        # Ensure positive value to avoid NaN
        kurt_factor = max(abs(kurt) / 3, 0.1)
        psi4 = psi4_ref * np.sqrt(kurt_factor)  # Adjust for non-normality
        
        # Optimal bandwidth: h = (1/(4π ψ₄ n))^(1/5)
        h_opt = (1 / (4 * np.pi * psi4 * n))**(1/5)
        
        return h_opt
        
    def _adaptive_bandwidth_1d(self, data: torch.Tensor, init_h: float) -> float:
        """Adaptive bandwidth using local density estimates"""
        n = len(data)
        
        # Step 1: Get pilot density estimate
        pilot_density, _ = kde_wrapper(data, method='fft')
        
        # Interpolate pilot density at data points
        pilot_f = self._interpolate_density(data, pilot_density)
        pilot_f = torch.clamp(pilot_f, min=1e-6)
        
        # Step 2: Compute local bandwidth factors
        # λᵢ = (f̃/f̃(Xᵢ))^α where α = 0.5 (sensitivity parameter)
        alpha = 0.5
        geom_mean = torch.exp(torch.mean(torch.log(pilot_f)))
        lambda_i = (geom_mean / pilot_f)**alpha
        
        # Step 3: Compute adaptive bandwidth
        # h_adapt = h * (∏λᵢ)^(1/n)
        h_adapt = init_h * torch.exp(torch.mean(torch.log(lambda_i))).item()
        
        return h_adapt
        
    def _kurtosis(self, data: torch.Tensor) -> float:
        """Compute excess kurtosis"""
        mean = torch.mean(data)
        std = torch.std(data)
        z = (data - mean) / std
        return torch.mean(z**4).item() - 3
        
    def _interpolate_density(self, x: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """Interpolate density values at data points"""
        # Simple linear interpolation
        # In practice, would use more sophisticated interpolation
        n_grid = len(density)
        x_min, x_max = torch.min(x), torch.max(x)
        
        # Map data points to grid indices
        indices = (x - x_min) / (x_max - x_min) * (n_grid - 1)
        indices = torch.clamp(indices, 0, n_grid - 1)
        
        # Linear interpolation
        idx_low = torch.floor(indices).long()
        idx_high = torch.ceil(indices).long()
        alpha = indices - idx_low.float()
        
        f_interp = (1 - alpha) * density[idx_low] + alpha * density[idx_high]
        
        return f_interp
        
    def _select_2d(self, data: torch.Tensor, bounds: Optional[Tuple[float, float]] = None,
                   init_h: Optional[float] = None) -> Union[float, torch.Tensor]:
        """Select bandwidth for 2D data (copula density)"""
        n, d = data.shape
        
        if d != 2:
            raise ValueError("2D bandwidth selection requires 2-dimensional data")
            
        # For 2D copula data, use specialized methods
        if self.method == 'copula_ml':
            return self._copula_ml_bandwidth(data, bounds)
        elif self.method == 'copula_transform':
            return self._copula_transform_bandwidth(data)
        else:
            # Default: product of marginal bandwidths
            h1 = self._select_1d(data[:, 0], bounds, init_h)
            h2 = self._select_1d(data[:, 1], bounds, init_h)
            return torch.tensor([h1, h2])
            
    def _copula_ml_bandwidth(self, data: torch.Tensor, bounds: Optional[Tuple[float, float]]) -> float:
        """Maximum likelihood bandwidth for copula density"""
        n = len(data)
        
        # Transform to copula scale (if not already)
        u = self._to_uniform(data[:, 0])
        v = self._to_uniform(data[:, 1])
        
        if bounds is None:
            bounds = (0.01, 0.5)  # Reasonable bounds for copula data
            
        def ml_score(h):
            """Pseudo-likelihood score for copula"""
            if h <= 0:
                return -np.inf
                
            # Beta kernel for bounded support
            # K(u) = u^(a-1) * (1-u)^(b-1) / B(a,b)
            # For simplicity, use transformed Gaussian
            
            # Transform to unbounded
            z1 = torch.erfinv(2*u - 1) * np.sqrt(2)
            z2 = torch.erfinv(2*v - 1) * np.sqrt(2)
            
            # Apply Gaussian kernel in transformed space
            z = torch.stack([z1, z2], dim=1)
            dists = torch.cdist(z, z)
            
            # Leave-one-out
            mask = ~torch.eye(n, dtype=torch.bool, device=data.device)
            K = torch.exp(-0.5 * (dists/h)**2) / (2*np.pi*h**2)
            K_loo = K * mask
            
            # Density in transformed space
            f_z = torch.sum(K_loo, dim=1) / (n - 1)
            
            # Transform back (Jacobian correction)
            # f_u,v = f_z * |J| where J is Jacobian
            phi_inv_u = torch.exp(-0.5 * z1**2) / np.sqrt(2*np.pi)
            phi_inv_v = torch.exp(-0.5 * z2**2) / np.sqrt(2*np.pi)
            
            f_uv = f_z / (phi_inv_u * phi_inv_v + 1e-10)
            f_uv = torch.clamp(f_uv, min=1e-10)
            
            return torch.sum(torch.log(f_uv)).item()
            
        result = minimize_scalar(lambda h: -ml_score(h), bounds=bounds, method='bounded')
        return result.x
        
    def _copula_transform_bandwidth(self, data: torch.Tensor) -> float:
        """Bandwidth selection using copula transformation"""
        # Transform margins to uniform
        u = self._to_uniform(data[:, 0])
        v = self._to_uniform(data[:, 1])
        
        # Transform to normal
        normal = torch.distributions.Normal(0, 1)
        z1 = normal.icdf(torch.clamp(u, 1e-6, 1-1e-6))
        z2 = normal.icdf(torch.clamp(v, 1e-6, 1-1e-6))
        
        # Select bandwidth in normal space
        h_z1 = self._select_1d(z1)
        h_z2 = self._select_1d(z2)
        
        # Average and scale back
        h_avg = (h_z1 + h_z2) / 2
        
        # Adjust for transformation
        # Rough approximation: bandwidth in copula space is smaller
        return h_avg * 0.5
        
    def _to_uniform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to uniform using empirical CDF"""
        n = len(x)
        ranks = torch.argsort(torch.argsort(x)).float()
        return (ranks + 0.5) / n


class BandwidthOptimizer:
    """Optimize bandwidth selection for vine copula estimation"""
    
    def __init__(self, vine_structure: str = 'c-vine'):
        self.vine_structure = vine_structure
        self.selector = BandwidthSelector(method='cv')
        
    def optimize_vine_bandwidths(self, data: torch.Tensor, 
                               tree_level: int = 0) -> dict:
        """
        Optimize bandwidths for all pairs in a vine tree level
        
        Args:
            data: Full data matrix (n_samples, n_dim)
            tree_level: Tree level in vine (0 = first tree)
            
        Returns:
            Dictionary of optimal bandwidths for each edge
        """
        n, d = data.shape
        bandwidths = {}
        
        if self.vine_structure == 'c-vine':
            # C-vine: first tree connects variable 0 to all others
            if tree_level == 0:
                for j in range(1, d):
                    u = self._to_copula_scale(data[:, 0])
                    v = self._to_copula_scale(data[:, j])
                    h_opt = self.selector._copula_ml_bandwidth(
                        torch.stack([u, v], dim=1), bounds=(0.01, 0.5)
                    )
                    bandwidths[(0, j)] = h_opt
            else:
                # Higher trees: use conditional copulas
                # Simplified: use same bandwidth
                h_default = 0.1
                for j in range(tree_level + 1, d):
                    bandwidths[(tree_level, j)] = h_default
                    
        elif self.vine_structure == 'd-vine':
            # D-vine: path structure
            if tree_level == 0:
                for j in range(d - 1):
                    u = self._to_copula_scale(data[:, j])
                    v = self._to_copula_scale(data[:, j + 1])
                    h_opt = self.selector._copula_ml_bandwidth(
                        torch.stack([u, v], dim=1), bounds=(0.01, 0.5)
                    )
                    bandwidths[(j, j + 1)] = h_opt
            else:
                # Higher trees
                for j in range(d - tree_level - 1):
                    bandwidths[(j, j + tree_level + 1)] = 0.1
                    
        return bandwidths
        
    def _to_copula_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to copula scale [0,1]"""
        # Use empirical CDF
        n = len(x)
        ranks = torch.argsort(torch.argsort(x)).float()
        return (ranks + 0.5) / n
        
    def adaptive_bandwidth_matrix(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive bandwidth matrix for multivariate data
        
        Returns bandwidth matrix H such that kernel is K(H^(-1/2)(x-Xi))
        """
        n, d = data.shape
        
        # Step 1: Compute pilot bandwidth matrix (Scott's rule)
        cov = torch.cov(data.T)
        H_pilot = cov * (4/(d+2))**(2/(d+4)) * n**(-2/(d+4))
        
        # Step 2: Compute local density with pilot bandwidth
        # Simplified: use diagonal bandwidth
        h_pilot = torch.sqrt(torch.diag(H_pilot))
        
        # Compute pilot densities
        pilot_densities = torch.zeros(n)
        for i in range(n):
            # Leave-one-out density
            dists = torch.norm((data - data[i]) / h_pilot, dim=1)
            mask = torch.arange(n) != i
            K = torch.exp(-0.5 * dists**2) * mask.float()
            pilot_densities[i] = torch.sum(K) / ((n-1) * torch.prod(h_pilot) * (2*np.pi)**(d/2))
            
        pilot_densities = torch.clamp(pilot_densities, min=1e-6)
        
        # Step 3: Compute adaptive factors
        geom_mean = torch.exp(torch.mean(torch.log(pilot_densities)))
        lambda_i = (geom_mean / pilot_densities)**0.5
        
        # Step 4: Adaptive bandwidth matrix for each point
        # For simplicity, scale pilot matrix
        H_adaptive = H_pilot * torch.mean(lambda_i)
        
        return H_adaptive 