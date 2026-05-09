"""
Time-series data utilities for time-dependent vine copulas.

This module provides data loading, preprocessing, and generation utilities
for time-dependent vine copula modeling.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger("DVC.time.data")


class TimeSeriesVineDataset(Dataset):
    """
    Dataset for time-dependent vine copula training.
    
    Handles time-series data with associated time indices for
    time-dependent bandwidth modeling.
    """
    
    def __init__(self, 
                 data: np.ndarray,
                 time_indices: np.ndarray,
                 window_size: Optional[int] = None,
                 normalize_time: bool = True):
        """
        Initialize TimeSeriesVineDataset.
        
        Args:
            data: Time series data, shape (n_time_steps, n_samples, n_variables)
            time_indices: Time indices, shape (n_time_steps,)
            window_size: Optional window size for sliding window approach
            normalize_time: Whether to normalize time indices to [0,1]
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.time_indices = torch.tensor(time_indices, dtype=torch.float32)
        self.window_size = window_size
        self.normalize_time = normalize_time
        
        if self.normalize_time:
            t_min = self.time_indices.min()
            t_max = self.time_indices.max()
            self.time_indices = (self.time_indices - t_min) / (t_max - t_min + 1e-8)
        
        # Validate shapes
        assert self.data.shape[0] == self.time_indices.shape[0], \
            "Data and time indices must have same length"
        
        logger.info(f"Created TimeSeriesVineDataset with {len(self)} samples")
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Time range: [{self.time_indices.min():.3f}, {self.time_indices.max():.3f}]")
    
    def __len__(self) -> int:
        """Return dataset length."""
        if self.window_size is not None:
            return max(0, self.data.shape[0] - self.window_size + 1)
        else:
            return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            (data, time) tuple
        """
        if self.window_size is not None:
            # Return sliding window
            data_window = self.data[idx:idx + self.window_size]
            time_window = self.time_indices[idx:idx + self.window_size]
            return data_window, time_window
        else:
            # Return single time step
            return self.data[idx], self.time_indices[idx]


def generate_synthetic_time_series(n_time_steps: int,
                                 n_samples: int,
                                 n_variables: int,
                                 correlation_evolution: str = 'sinusoidal',
                                 noise_level: float = 0.1,
                                 seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series data with time-varying correlations.
    
    Args:
        n_time_steps: Number of time steps
        n_samples: Number of samples per time step
        n_variables: Number of variables
        correlation_evolution: Type of correlation evolution ('sinusoidal', 'linear', 'piecewise')
        noise_level: Noise level for correlation evolution
        seed: Random seed
        
    Returns:
        (data, time_indices) tuple
    """
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(f"Generating synthetic time series: {n_time_steps} x {n_samples} x {n_variables}")
    logger.info(f"Correlation evolution: {correlation_evolution}")
    
    data = np.zeros((n_time_steps, n_samples, n_variables))
    time_indices = np.arange(n_time_steps, dtype=np.float32)
    
    for t in range(n_time_steps):
        # Time-dependent correlation structure
        time_norm = t / max(1, n_time_steps - 1)  # Normalize to [0,1]
        
        if correlation_evolution == 'sinusoidal':
            # Sinusoidal correlation evolution
            base_corr = 0.3 + 0.5 * np.sin(2 * np.pi * time_norm)
        elif correlation_evolution == 'linear':
            # Linear correlation evolution
            base_corr = 0.1 + 0.6 * time_norm
        elif correlation_evolution == 'piecewise':
            # Piecewise constant correlation
            if time_norm < 0.33:
                base_corr = 0.2
            elif time_norm < 0.67:
                base_corr = 0.7
            else:
                base_corr = 0.4
        else:
            # Constant correlation
            base_corr = 0.5
        
        # Add noise to correlation
        base_corr += noise_level * np.random.randn()
        base_corr = np.clip(base_corr, -0.95, 0.95)
        
        # Create correlation matrix
        corr_matrix = np.eye(n_variables)
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                # Decay correlation with distance
                distance_factor = 1.0 / (1.0 + 0.5 * abs(i - j))
                corr_matrix[i, j] = corr_matrix[j, i] = base_corr * distance_factor
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive
        corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Generate multivariate normal data
        try:
            data[t] = np.random.multivariate_normal(
                mean=np.zeros(n_variables),
                cov=corr_matrix,
                size=n_samples
            )
        except np.linalg.LinAlgError:
            # Fallback to independent variables if correlation matrix is problematic
            logger.warning(f"Correlation matrix issue at time {t}, using independent variables")
            data[t] = np.random.randn(n_samples, n_variables)
    
    logger.info(f"Generated synthetic data with shape: {data.shape}")
    
    return data, time_indices


def create_data_loader(data: np.ndarray,
                      time_indices: np.ndarray,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      window_size: Optional[int] = None,
                      normalize_time: bool = True) -> DataLoader:
    """
    Create DataLoader for time-dependent vine copula training.
    
    Args:
        data: Time series data
        time_indices: Time indices
        batch_size: Batch size
        shuffle: Whether to shuffle data
        window_size: Optional window size
        normalize_time: Whether to normalize time
        
    Returns:
        DataLoader instance
    """
    dataset = TimeSeriesVineDataset(
        data=data,
        time_indices=time_indices,
        window_size=window_size,
        normalize_time=normalize_time
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Keep simple for now
        pin_memory=torch.cuda.is_available()
    )


def preprocess_real_data(data: np.ndarray,
                        time_indices: Optional[np.ndarray] = None,
                        standardize: bool = True,
                        remove_outliers: bool = True,
                        outlier_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess real time series data for vine copula modeling.
    
    Args:
        data: Raw time series data, shape (n_time_steps, n_samples, n_variables) or (n_samples, n_variables)
        time_indices: Optional time indices
        standardize: Whether to standardize variables
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier removal
        
    Returns:
        (processed_data, time_indices) tuple
    """
    logger.info(f"Preprocessing data with shape: {data.shape}")
    
    # Handle 2D data (assume single time step)
    if len(data.shape) == 2:
        data = data.reshape(1, *data.shape)
        logger.info(f"Reshaped 2D data to: {data.shape}")
    
    n_time_steps, n_samples, n_variables = data.shape
    
    # Create time indices if not provided
    if time_indices is None:
        time_indices = np.arange(n_time_steps, dtype=np.float32)
        logger.info("Created default time indices")
    
    processed_data = data.copy()
    
    # Standardize variables
    if standardize:
        logger.info("Standardizing variables...")
        for t in range(n_time_steps):
            for v in range(n_variables):
                var_data = processed_data[t, :, v]
                mean_val = np.mean(var_data)
                std_val = np.std(var_data)
                if std_val > 1e-8:
                    processed_data[t, :, v] = (var_data - mean_val) / std_val
    
    # Remove outliers
    if remove_outliers:
        logger.info(f"Removing outliers with threshold {outlier_threshold}...")
        outlier_count = 0
        
        for t in range(n_time_steps):
            for v in range(n_variables):
                var_data = processed_data[t, :, v]
                z_scores = np.abs((var_data - np.mean(var_data)) / (np.std(var_data) + 1e-8))
                outlier_mask = z_scores > outlier_threshold
                
                if np.any(outlier_mask):
                    # Replace outliers with median
                    median_val = np.median(var_data[~outlier_mask])
                    processed_data[t, outlier_mask, v] = median_val
                    outlier_count += np.sum(outlier_mask)
        
        if outlier_count > 0:
            logger.info(f"Replaced {outlier_count} outliers")
    
    logger.info(f"Preprocessing complete. Final shape: {processed_data.shape}")
    
    return processed_data, time_indices


def compute_time_varying_correlations(data: np.ndarray) -> np.ndarray:
    """
    Compute time-varying correlation matrices for time series data.
    
    Args:
        data: Time series data, shape (n_time_steps, n_samples, n_variables)
        
    Returns:
        Correlation matrices, shape (n_time_steps, n_variables, n_variables)
    """
    n_time_steps, n_samples, n_variables = data.shape
    correlations = np.zeros((n_time_steps, n_variables, n_variables))
    
    logger.info("Computing time-varying correlations...")
    
    for t in range(n_time_steps):
        try:
            correlations[t] = np.corrcoef(data[t].T)
            
            # Handle NaN values
            if np.any(np.isnan(correlations[t])):
                logger.warning(f"NaN correlations at time {t}, using identity matrix")
                correlations[t] = np.eye(n_variables)
                
        except Exception as e:
            logger.warning(f"Error computing correlations at time {t}: {e}")
            correlations[t] = np.eye(n_variables)
    
    logger.info("Correlation computation complete")
    
    return correlations


__all__ = [
    'TimeSeriesVineDataset',
    'generate_synthetic_time_series',
    'create_data_loader',
    'preprocess_real_data',
    'compute_time_varying_correlations'
]