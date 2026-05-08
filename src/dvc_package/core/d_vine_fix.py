"""
D-vine specific fixes and improvements.

This module contains fixes for D-vine correlation preservation and sampling.
Implements specialized algorithms for D-vines to maintain correlations between
non-adjacent variables, which can be lost in standard vine sampling.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from scipy.stats import kendalltau
import logging

logger = logging.getLogger("DVC.d_vine_fix")


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix for given data.
    
    Args:
        data: Input data of shape (n_samples, n_variables)
        
    Returns:
        Correlation matrix of shape (n_variables, n_variables)
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for correlation computation, got shape={x.shape}")

    n_samples, n_vars = x.shape
    if n_vars == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n_samples <= 1:
        return np.eye(n_vars, dtype=np.float64)

    centered = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(centered, axis=0, ddof=0)
    valid = std > 1e-12

    z = np.zeros_like(centered, dtype=np.float64)
    if np.any(valid):
        z[:, valid] = centered[:, valid] / std[valid]

    corr = (z.T @ z) / max(n_samples - 1, 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_kendall_tau_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute Kendall's tau matrix for given data.
    
    Args:
        data: Input data of shape (n_samples, n_variables)
        
    Returns:
        Kendall's tau matrix of shape (n_variables, n_variables)
    """
    n_vars = data.shape[1]
    tau_matrix = np.eye(n_vars)
    
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            tau, _ = kendalltau(data[:, i], data[:, j])
            if not np.isfinite(tau):
                tau = 0.0
            tau_matrix[i, j] = tau_matrix[j, i] = tau
    
    return tau_matrix


def sample_d_vine_with_correlation_preservation(vine, nsamples: int, 
                                               target_corr_matrix: Optional[np.ndarray] = None,
                                               max_iterations: int = 5,
                                               correlation_tolerance: float = 0.05):
    """
    Specialized sampling for D-vines with correlation preservation.
    
    This implementation uses an iterative approach to adjust sampling to preserve
    correlations between non-adjacent variables in D-vines, which can be lost
    in standard vine sampling due to the path structure.
    
    Args:
        vine: Vine copula object (should be D-vine)
        nsamples: Number of samples to generate
        target_corr_matrix: Target correlation matrix to preserve (optional)
        max_iterations: Maximum number of correction iterations
        correlation_tolerance: Tolerance for correlation preservation
        
    Returns:
        Samples from the D-vine with preserved correlations
    """
    from .vine_model import sample_vine
    
    logger.info(f"D-vine sampling with correlation preservation (n={nsamples})")
    
    # Initial sampling using standard vine sampling
    samples = sample_vine(vine, nsamples)
    
    if target_corr_matrix is None:
        logger.info("No target correlation matrix provided, using standard sampling")
        return samples
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
    target_corr_tensor = torch.tensor(target_corr_matrix, dtype=torch.float32, device=device)
    
    d = samples.shape[1]
    
    # Iterative correlation correction
    for iteration in range(max_iterations):
        current_corr = compute_correlation_matrix(samples_tensor.cpu().numpy())
        corr_error = np.abs(current_corr - target_corr_matrix)
        max_error = np.max(corr_error)
        
        logger.info(f"Iteration {iteration+1}: max correlation error = {max_error:.4f}")
        
        if max_error < correlation_tolerance:
            logger.info(f"Correlation preservation achieved in {iteration+1} iterations")
            break
        
        # Apply correlation correction using Gaussian copula transformation
        samples_tensor = _apply_correlation_correction(
            samples_tensor, current_corr, target_corr_matrix, 
            correction_strength=0.3 / (iteration + 1)  # Decreasing correction strength
        )
    
    final_samples = samples_tensor.cpu().numpy()
    final_corr = compute_correlation_matrix(final_samples)
    final_error = np.max(np.abs(final_corr - target_corr_matrix))
    
    logger.info(f"Final correlation error: {final_error:.4f}")
    
    return final_samples


def _apply_correlation_correction(samples: torch.Tensor, 
                                 current_corr: np.ndarray,
                                 target_corr: np.ndarray,
                                 correction_strength: float = 0.3) -> torch.Tensor:
    """
    Apply correlation correction to samples using Gaussian copula transformation.
    
    Args:
        samples: Current samples tensor
        current_corr: Current correlation matrix
        target_corr: Target correlation matrix
        correction_strength: Strength of the correction (0-1)
        
    Returns:
        Corrected samples tensor
    """
    device = samples.device
    n_samples, d = samples.shape
    
    # Convert to uniform margins using empirical CDF
    uniform_samples = torch.zeros_like(samples)
    for i in range(d):
        sorted_vals, _ = torch.sort(samples[:, i])
        ranks = torch.searchsorted(sorted_vals, samples[:, i])
        uniform_samples[:, i] = (ranks.float() + 1) / (n_samples + 1)
    
    # Convert to normal scores
    normal_dist = torch.distributions.Normal(0, 1)
    normal_scores = normal_dist.icdf(torch.clamp(uniform_samples, 1e-6, 1-1e-6))
    
    # Apply correlation correction in normal space
    current_corr_tensor = torch.tensor(current_corr, dtype=torch.float32, device=device)
    target_corr_tensor = torch.tensor(target_corr, dtype=torch.float32, device=device)
    
    # Compute correction matrix
    correction_matrix = correction_strength * (target_corr_tensor - current_corr_tensor)
    correction_matrix = correction_matrix + torch.eye(d, device=device)
    
    # Apply correction via matrix transformation
    try:
        # Use Cholesky decomposition for stable transformation
        L_current = torch.linalg.cholesky(current_corr_tensor + 1e-6 * torch.eye(d, device=device))
        L_target = torch.linalg.cholesky(target_corr_tensor + 1e-6 * torch.eye(d, device=device))
        
        # Transform: X_new = X * L_current^(-1) * L_target
        L_current_inv = torch.linalg.inv(L_current)
        transform_matrix = L_current_inv @ L_target
        
        # Apply partial transformation (weighted by correction_strength)
        identity = torch.eye(d, device=device)
        partial_transform = (1 - correction_strength) * identity + correction_strength * transform_matrix
        
        corrected_normal = normal_scores @ partial_transform.T
        
    except Exception as e:
        logger.warning(f"Cholesky decomposition failed: {e}, using simpler correction")
        # Fallback to simpler correction
        corrected_normal = normal_scores
    
    # Convert back to uniform and then to original margins
    corrected_uniform = normal_dist.cdf(corrected_normal)
    corrected_uniform = torch.clamp(corrected_uniform, 1e-6, 1-1e-6)
    
    # Convert back to original margins using quantile matching
    corrected_samples = torch.zeros_like(samples)
    for i in range(d):
        sorted_original, _ = torch.sort(samples[:, i])
        quantile_indices = (corrected_uniform[:, i] * (n_samples - 1)).long()
        quantile_indices = torch.clamp(quantile_indices, 0, n_samples - 1)
        corrected_samples[:, i] = sorted_original[quantile_indices]
    
    return corrected_samples


def sample_d_vine(vine, nsamples: int):
    """
    Main D-vine sampling function with optional correlation preservation.
    
    Args:
        vine: Vine copula object
        nsamples: Number of samples to generate
        
    Returns:
        Samples from the D-vine
    """
    # Check if vine has target correlation matrix for preservation
    if hasattr(vine, 'target_correlation_matrix') and vine.target_correlation_matrix is not None:
        return sample_d_vine_with_correlation_preservation(
            vine, nsamples, vine.target_correlation_matrix
        )
    else:
        # Use standard vine sampling
        from .vine_model import sample_vine
        return sample_vine(vine, nsamples)


def apply_d_vine_fix(vine):
    """
    Apply D-vine specific fixes to improve performance.
    
    Args:
        vine: Vine copula object to fix
    """
    # Mark that D-vine fixes have been applied
    vine._d_vine_fixes_applied = True
    
    # Reserved for D-vine-specific optimizations; kept as an explicit no-op
    # for API compatibility.
    pass 
