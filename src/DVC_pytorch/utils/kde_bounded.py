import torch
import numpy as np
from typing import Optional, Tuple, Union


def bounded_kde_gaussian(data: torch.Tensor, 
                         n: int = 128,
                         bandwidth: Optional[float] = None,
                         method: str = 'fft',
                         bounds: Optional[Tuple[float, float]] = None,
                         boundary_correction: str = 'truncate',
                         eps: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bounded Gaussian KDE that enforces zero probability outside data range.
    
    Args:
        data: 1D tensor of data points
        n: Number of evaluation points
        bandwidth: Kernel bandwidth (if None, uses Silverman's rule)
        method: KDE method ('fft', 'cdist', 'cdist_chunked')
        bounds: Explicit bounds (min, max). If None, uses data range
        boundary_correction: Method for boundary correction
            - 'truncate': Set density to 0 outside bounds
            - 'reflect': Reflect kernel at boundaries
            - 'renormalize': Truncate and renormalize
        eps: Small value to extend bounds slightly for numerical stability
    
    Returns:
        density: Probability density values
        mesh: Evaluation points
    """
    from .kde_simple import kde_gaussian, silverman_bandwidth
    
    # Determine bounds
    if bounds is None:
        data_min = data.min().item()
        data_max = data.max().item()
        # Add small epsilon to avoid numerical issues at exact boundaries
        bounds = (data_min - eps, data_max + eps)
    else:
        bounds = (float(bounds[0]), float(bounds[1]))
    
    # Get base KDE
    density, mesh = kde_gaussian(data, n=n, bandwidth=bandwidth, method=method)
    
    if boundary_correction == 'truncate':
        # Simple truncation: set density to 0 outside bounds
        mask = (mesh >= bounds[0]) & (mesh <= bounds[1])
        density = density * mask.float()
        
    elif boundary_correction == 'reflect':
        # Reflection method: add reflected kernels at boundaries
        density_reflected = reflect_kde_at_boundaries(
            data, mesh, bandwidth, bounds, method='gaussian'
        )
        density = density_reflected
        
    elif boundary_correction == 'renormalize':
        # Truncate and renormalize to ensure integral = 1
        mask = (mesh >= bounds[0]) & (mesh <= bounds[1])
        density = density * mask.float()
        
        # Renormalize
        dx = (mesh[-1] - mesh[0]) / (len(mesh) - 1)
        integral = torch.sum(density) * dx
        if integral > 0:
            density = density / integral
    
    return density, mesh


def reflect_kde_at_boundaries(data: torch.Tensor,
                             mesh: torch.Tensor,
                             bandwidth: float,
                             bounds: Tuple[float, float],
                             method: str = 'gaussian') -> torch.Tensor:
    """
    Implement boundary reflection method for KDE.
    
    This method adds reflected copies of data points at boundaries
    to ensure smooth behavior near edges.
    """
    if bandwidth is None:
        from .kde_simple import silverman_bandwidth
        bandwidth = silverman_bandwidth(data)
    
    lower_bound, upper_bound = bounds
    
    # Original density contribution
    density = torch.zeros_like(mesh)
    
    # Compute distances for all combinations
    data_expanded = data.unsqueeze(1)  # [N, 1]
    mesh_expanded = mesh.unsqueeze(0)  # [1, M]
    
    # Original kernels
    if method == 'gaussian':
        # Standard Gaussian kernel
        distances = (mesh_expanded - data_expanded) / bandwidth
        kernels = torch.exp(-0.5 * distances**2) / (bandwidth * np.sqrt(2 * np.pi))
        density += torch.sum(kernels, dim=0)
        
        # Reflected kernels at lower boundary
        reflected_lower = 2 * lower_bound - data
        distances_lower = (mesh_expanded - reflected_lower.unsqueeze(1)) / bandwidth
        kernels_lower = torch.exp(-0.5 * distances_lower**2) / (bandwidth * np.sqrt(2 * np.pi))
        # Only add reflection contribution within bounds
        mask_lower = mesh >= lower_bound
        density += torch.sum(kernels_lower, dim=0) * mask_lower.float()
        
        # Reflected kernels at upper boundary
        reflected_upper = 2 * upper_bound - data
        distances_upper = (mesh_expanded - reflected_upper.unsqueeze(1)) / bandwidth
        kernels_upper = torch.exp(-0.5 * distances_upper**2) / (bandwidth * np.sqrt(2 * np.pi))
        # Only add reflection contribution within bounds
        mask_upper = mesh <= upper_bound
        density += torch.sum(kernels_upper, dim=0) * mask_upper.float()
    
    # Normalize by number of data points
    density = density / len(data)
    
    # Final truncation to ensure exactly 0 outside bounds
    final_mask = (mesh >= lower_bound) & (mesh <= upper_bound)
    density = density * final_mask.float()
    
    return density


def bounded_kde_wrapper(data: torch.Tensor,
                       n: int = 128,
                       method: str = 'fft_bounded',
                       enforce_bounds: bool = True,
                       bounds: Optional[Tuple[float, float]] = None,
                       extend_factor: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for easy switching between bounded and unbounded KDE.
    
    Args:
        data: Input data
        n: Number of evaluation points
        method: KDE method name (append '_bounded' for bounded versions)
        enforce_bounds: Whether to enforce zero probability outside data range
        bounds: Explicit bounds. If None and enforce_bounds=True, uses data range
        extend_factor: Factor to extend bounds by (as fraction of range)
    
    Returns:
        density: Probability density
        mesh: Evaluation points
    """
    if not enforce_bounds or '_bounded' not in method:
        # Use standard KDE
        from .kde_simple import kde_gaussian
        base_method = method.replace('_bounded', '')
        return kde_gaussian(data, n=n, method=base_method)
    
    # Use bounded KDE
    if bounds is None and enforce_bounds:
        data_range = data.max() - data.min()
        extension = data_range * extend_factor
        bounds = (data.min().item() - extension, 
                 data.max().item() + extension)
    
    # Determine boundary correction method
    if 'reflect' in method:
        boundary_correction = 'reflect'
    elif 'renorm' in method:
        boundary_correction = 'renormalize'
    else:
        boundary_correction = 'truncate'
    
    base_method = method.split('_')[0]  # Extract base method (fft, cdist, etc)
    
    return bounded_kde_gaussian(
        data, n=n, method=base_method,
        bounds=bounds, boundary_correction=boundary_correction
    )


def adaptive_bounded_kde(data: torch.Tensor,
                        n: int = 128,
                        alpha: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive bounded KDE that automatically determines bounds based on
    percentiles to handle outliers gracefully.
    
    Args:
        data: Input data
        n: Number of evaluation points  
        alpha: Tail probability to exclude (e.g., 0.05 excludes 5% tails)
    
    Returns:
        density: Probability density
        mesh: Evaluation points
    """
    # Compute percentile-based bounds
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = torch.quantile(data, lower_percentile / 100).item()
    upper_bound = torch.quantile(data, upper_percentile / 100).item()
    
    # Add small buffer
    data_range = upper_bound - lower_bound
    buffer = data_range * 0.05
    bounds = (lower_bound - buffer, upper_bound + buffer)
    
    return bounded_kde_gaussian(
        data, n=n, bounds=bounds, 
        boundary_correction='renormalize'
    )


def transform_bounded_kde(data: torch.Tensor,
                         n: int = 128,
                         transform: str = 'logit',
                         bounds: Optional[Tuple[float, float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KDE with transformation to handle bounded domains.
    
    Transforms data to unbounded domain, performs KDE, then transforms back.
    
    Args:
        data: Input data
        n: Number of evaluation points
        transform: Transformation type ('logit', 'log', 'probit')
        bounds: Domain bounds for transformation
    
    Returns:
        density: Probability density in original domain
        mesh: Evaluation points in original domain
    """
    if bounds is None:
        bounds = (data.min().item(), data.max().item())
        
    eps = 1e-6
    lower, upper = bounds
    
    # Normalize data to [0, 1]
    data_norm = (data - lower) / (upper - lower)
    data_norm = torch.clamp(data_norm, eps, 1 - eps)
    
    if transform == 'logit':
        # Logit transform: log(x / (1-x))
        data_transformed = torch.log(data_norm / (1 - data_norm))
        
        # Perform KDE in transformed space
        from .kde_simple import kde_gaussian
        density_trans, mesh_trans = kde_gaussian(data_transformed, n=n)
        
        # Transform mesh back
        mesh_norm = torch.sigmoid(mesh_trans)
        mesh = mesh_norm * (upper - lower) + lower
        
        # Adjust density for Jacobian
        jacobian = mesh_norm * (1 - mesh_norm) * (upper - lower)
        density = density_trans / (jacobian + eps)
        
    elif transform == 'log':
        # Log transform (for positive data)
        if lower <= 0:
            raise ValueError("Log transform requires positive data")
            
        data_transformed = torch.log(data)
        
        from .kde_simple import kde_gaussian
        density_trans, mesh_trans = kde_gaussian(data_transformed, n=n)
        
        # Transform back
        mesh = torch.exp(mesh_trans)
        
        # Adjust for Jacobian
        density = density_trans / (mesh + eps)
        
    elif transform == 'probit':
        # Probit transform: inverse normal CDF
        from scipy.stats import norm
        data_np = data_norm.numpy()
        data_transformed = torch.tensor(norm.ppf(data_np))
        
        from .kde_simple import kde_gaussian  
        density_trans, mesh_trans = kde_gaussian(data_transformed, n=n)
        
        # Transform back
        mesh_norm = torch.tensor(norm.cdf(mesh_trans.numpy()))
        mesh = mesh_norm * (upper - lower) + lower
        
        # Adjust for Jacobian
        jacobian_trans = torch.tensor(norm.pdf(mesh_trans.numpy()))
        jacobian = jacobian_trans * (upper - lower)
        density = density_trans / (jacobian + eps)
    
    # Ensure density is 0 outside bounds
    mask = (mesh >= lower) & (mesh <= upper)
    density = density * mask.float()
    
    # Renormalize
    dx = (mesh[-1] - mesh[0]) / (len(mesh) - 1)
    integral = torch.sum(density) * dx
    if integral > 0:
        density = density / integral
        
    return density, mesh 