"""
Tensor utilities for robustness and numerical stability.

This module provides utility functions for handling NaN/Inf values,
numerical stability, and tensor operations, similar to TensorFlow's
replace_nan_inf functionality.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger("DVC.utils.tensor")


def replace_nan_inf(tensor: torch.Tensor, 
                   nan_value: float = 0.0,
                   inf_value: float = 1e6,
                   neginf_value: float = -1e6) -> torch.Tensor:
    """
    Replace NaN and Inf values in tensor (matching TensorFlow's replace_nan_inf).
    
    Args:
        tensor: Input tensor
        nan_value: Value to replace NaN with
        inf_value: Value to replace +Inf with
        neginf_value: Value to replace -Inf with
        
    Returns:
        Tensor with NaN/Inf values replaced
    """
    result = tensor.clone()
    
    # Replace NaN values
    nan_mask = torch.isnan(result)
    if nan_mask.any():
        result[nan_mask] = nan_value
        logger.debug(f"Replaced {nan_mask.sum().item()} NaN values with {nan_value}")
    
    # Replace +Inf values
    posinf_mask = torch.isposinf(result)
    if posinf_mask.any():
        result[posinf_mask] = inf_value
        logger.debug(f"Replaced {posinf_mask.sum().item()} +Inf values with {inf_value}")
    
    # Replace -Inf values
    neginf_mask = torch.isneginf(result)
    if neginf_mask.any():
        result[neginf_mask] = neginf_value
        logger.debug(f"Replaced {neginf_mask.sum().item()} -Inf values with {neginf_value}")
    
    return result


def check_finite(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """
    Check if tensor contains only finite values and raise error if not.
    
    Args:
        tensor: Input tensor
        name: Name for error message
        
    Returns:
        Input tensor (unchanged)
        
    Raises:
        ValueError: If tensor contains NaN or Inf values
    """
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(f"{name} contains non-finite values: {nan_count} NaN, {inf_count} Inf")
    
    return tensor


def safe_log(tensor: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """
    Compute log with numerical stability.
    
    Args:
        tensor: Input tensor
        eps: Small value to add for numerical stability
        
    Returns:
        log(tensor + eps)
    """
    return torch.log(torch.clamp(tensor, min=eps))


def safe_exp(tensor: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
    """
    Compute exp with numerical stability.
    
    Args:
        tensor: Input tensor
        max_val: Maximum value before clamping
        
    Returns:
        exp(clamp(tensor, max=-max_val, min=max_val))
    """
    return torch.exp(torch.clamp(tensor, min=-max_val, max=max_val))


def safe_div(numerator: torch.Tensor, 
            denominator: torch.Tensor, 
            eps: float = 1e-15) -> torch.Tensor:
    """
    Safe division with numerical stability.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor  
        eps: Small value to add to denominator
        
    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def clamp_preserve_gradients(tensor: torch.Tensor, 
                           min_val: Optional[float] = None,
                           max_val: Optional[float] = None) -> torch.Tensor:
    """
    Clamp tensor while preserving gradients outside the clamping range.
    
    This is useful when you want to clamp values for numerical stability
    but still allow gradients to flow through.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped tensor with preserved gradients
    """
    if min_val is not None:
        tensor = torch.where(tensor < min_val, 
                           tensor - tensor.detach() + min_val, 
                           tensor)
    
    if max_val is not None:
        tensor = torch.where(tensor > max_val,
                           tensor - tensor.detach() + max_val,
                           tensor)
    
    return tensor


def handle_small_sample_size(data: torch.Tensor, 
                            min_samples: int = 50) -> Tuple[torch.Tensor, bool]:
    """
    Handle small sample sizes by adding regularization or fallback behavior.
    
    Args:
        data: Input data tensor
        min_samples: Minimum number of samples required
        
    Returns:
        (processed_data, is_small_sample) tuple
    """
    n_samples = data.shape[0]
    is_small_sample = n_samples < min_samples
    
    if is_small_sample:
        logger.warning(f"Small sample size detected: {n_samples} < {min_samples}")
        
        # Add noise regularization for small samples
        noise_scale = 0.01 * torch.std(data)
        noise = torch.randn_like(data) * noise_scale
        processed_data = data + noise
        
        logger.info(f"Added noise regularization with scale {noise_scale:.6f}")
    else:
        processed_data = data
    
    return processed_data, is_small_sample


def robust_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute robust correlation coefficient with numerical stability.
    
    Args:
        x: First variable tensor
        y: Second variable tensor
        
    Returns:
        Correlation coefficient (scalar tensor)
    """
    # Center the variables
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)
    
    # Compute standard deviations with numerical stability
    x_std = torch.std(x_centered) + 1e-8
    y_std = torch.std(y_centered) + 1e-8
    
    # Compute correlation
    covariance = torch.mean(x_centered * y_centered)
    correlation = covariance / (x_std * y_std)
    
    # Clamp to valid correlation range
    correlation = torch.clamp(correlation, -0.9999, 0.9999)
    
    return correlation


def check_positive_definite(matrix: torch.Tensor, 
                          regularization: float = 1e-6) -> torch.Tensor:
    """
    Check if matrix is positive definite and regularize if needed.
    
    Args:
        matrix: Input matrix
        regularization: Regularization parameter
        
    Returns:
        Positive definite matrix
    """
    # Check if matrix is symmetric
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        logger.warning("Matrix is not symmetric, symmetrizing...")
        matrix = (matrix + matrix.T) / 2
    
    # Compute eigenvalues
    try:
        eigenvals = torch.linalg.eigvals(matrix)
        min_eigenval = torch.min(eigenvals.real)
        
        if min_eigenval <= 0:
            logger.warning(f"Matrix is not positive definite (min eigenval: {min_eigenval:.6f})")
            
            # Add regularization to diagonal
            regularized_matrix = matrix + regularization * torch.eye(matrix.shape[0], 
                                                                   device=matrix.device,
                                                                   dtype=matrix.dtype)
            
            # Verify it's now positive definite
            new_eigenvals = torch.linalg.eigvals(regularized_matrix)
            new_min_eigenval = torch.min(new_eigenvals.real)
            
            if new_min_eigenval > 0:
                logger.info(f"Regularization successful (new min eigenval: {new_min_eigenval:.6f})")
                return regularized_matrix
            else:
                logger.error("Regularization failed, returning identity matrix")
                return torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        
    except Exception as e:
        logger.error(f"Error checking positive definiteness: {e}")
        return torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    
    return matrix


def adaptive_bandwidth_selection(data: torch.Tensor, 
                                method: str = 'silverman') -> torch.Tensor:
    """
    Adaptive bandwidth selection for kernel density estimation.
    
    Args:
        data: Input data tensor, shape (n_samples, n_dims)
        method: Bandwidth selection method ('silverman', 'scott')
        
    Returns:
        Bandwidth tensor, shape (n_dims,)
    """
    n_samples, n_dims = data.shape
    
    if method == 'silverman':
        # Silverman's rule of thumb
        std_devs = torch.std(data, dim=0)
        iqr = torch.quantile(data, 0.75, dim=0) - torch.quantile(data, 0.25, dim=0)
        
        # Use minimum of std and IQR/1.34
        scale = torch.minimum(std_devs, iqr / 1.34)
        bandwidth = 0.9 * scale * (n_samples ** (-1/5))
        
    elif method == 'scott':
        # Scott's rule
        std_devs = torch.std(data, dim=0)
        bandwidth = std_devs * (n_samples ** (-1/(n_dims + 4)))
        
    else:
        raise ValueError(f"Unknown bandwidth selection method: {method}")
    
    # Ensure minimum bandwidth for numerical stability
    min_bandwidth = 1e-4
    bandwidth = torch.clamp(bandwidth, min=min_bandwidth)
    
    return bandwidth


def robust_matrix_inverse(matrix: torch.Tensor, 
                         regularization: float = 1e-6) -> torch.Tensor:
    """
    Compute robust matrix inverse with regularization.
    
    Args:
        matrix: Input matrix
        regularization: Regularization parameter
        
    Returns:
        Inverse matrix
    """
    try:
        # Try direct inversion first
        inverse = torch.linalg.inv(matrix)
        
        # Check if result is reasonable
        if torch.isfinite(inverse).all():
            return inverse
        else:
            logger.warning("Direct inversion produced non-finite values")
            
    except Exception as e:
        logger.warning(f"Direct inversion failed: {e}")
    
    # Fallback: regularized inversion
    try:
        regularized_matrix = matrix + regularization * torch.eye(matrix.shape[0],
                                                               device=matrix.device,
                                                               dtype=matrix.dtype)
        inverse = torch.linalg.inv(regularized_matrix)
        
        if torch.isfinite(inverse).all():
            logger.info(f"Regularized inversion successful (reg={regularization})")
            return inverse
        else:
            logger.error("Regularized inversion also produced non-finite values")
            
    except Exception as e:
        logger.error(f"Regularized inversion failed: {e}")
    
    # Last resort: return identity matrix
    logger.error("All inversion methods failed, returning identity matrix")
    return torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)


__all__ = [
    'replace_nan_inf',
    'check_finite', 
    'safe_log',
    'safe_exp',
    'safe_div',
    'clamp_preserve_gradients',
    'handle_small_sample_size',
    'robust_correlation',
    'check_positive_definite',
    'adaptive_bandwidth_selection',
    'robust_matrix_inverse'
]
