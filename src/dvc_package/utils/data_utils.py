"""
Data utilities for DVC package.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)


def load_data(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load data from various file formats.
    
    Parameters
    ----------
    file_path : str or Path
        Path to data file
        
    Returns
    -------
    np.ndarray
        Loaded data
    """
    path = Path(file_path)
    
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
        return df.values
    elif path.suffix.lower() == '.npy':
        return np.load(path)
    elif path.suffix.lower() in ['.pkl', '.pickle']:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_data(data: np.ndarray, file_path: Union[str, Path], 
              format: str = 'auto') -> None:
    """
    Save data to file.
    
    Parameters
    ----------
    data : np.ndarray
        Data to save
    file_path : str or Path
        Output file path
    format : str
        Output format ('auto', 'csv', 'npy', 'pickle')
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'auto':
        format = path.suffix.lower().lstrip('.')
    
    if format == 'csv':
        pd.DataFrame(data).to_csv(path, index=False)
    elif format == 'npy':
        np.save(path, data)
    elif format in ['pkl', 'pickle']:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_data(data: np.ndarray, min_samples: int = 10, 
                  max_features: int = 100) -> Tuple[bool, str]:
    """
    Validate input data for vine copula modeling.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    min_samples : int
        Minimum number of samples required
    max_features : int
        Maximum number of features allowed
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if not isinstance(data, np.ndarray):
        return False, "Data must be a numpy array"
    
    if data.ndim != 2:
        return False, "Data must be 2-dimensional"
    
    n_samples, n_features = data.shape
    
    if n_samples < min_samples:
        return False, f"Need at least {min_samples} samples, got {n_samples}"
    
    if n_features > max_features:
        return False, f"Too many features: {n_features} > {max_features}"
    
    if np.any(np.isnan(data)):
        return False, "Data contains NaN values"
    
    if np.any(np.isinf(data)):
        return False, "Data contains infinite values"
    
    return True, "Data is valid"


def preprocess_data(data: np.ndarray, method: str = 'standardize') -> np.ndarray:
    """
    Preprocess data for vine copula modeling.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str
        Preprocessing method ('standardize', 'normalize', 'rank_transform')
        
    Returns
    -------
    np.ndarray
        Preprocessed data
    """
    if method == 'standardize':
        # Z-score standardization
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    elif method == 'normalize':
        # Min-max normalization to [0, 1]
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        return (data - data_min) / (data_max - data_min)
    
    elif method == 'rank_transform':
        # Transform to uniform margins using ranks
        n_samples = data.shape[0]
        uniform_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):
            ranks = data[:, i].argsort().argsort() + 1
            uniform_data[:, i] = ranks / (n_samples + 1)
        
        return uniform_data
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
