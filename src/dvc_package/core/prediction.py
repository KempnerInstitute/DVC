##################################################
# src/DVC/prediction.py
##################################################

import torch
import numpy as np
from typing import Tuple


def create_points(x: torch.Tensor, dim: int, exp_dim: int) -> torch.Tensor:
    """
    Create a grid of points for evaluation.
    
    Args:
        x: Input data, shape [N, d]
        dim: Dimension to vary
        exp_dim: Number of expansion points
        
    Returns:
        points: Expanded points, shape [N * exp_dim, d]
    """
    N, d = x.shape
    device = x.device
    
    # Create range for dimension 'dim'
    min_val = x[:, dim].min()
    max_val = x[:, dim].max()
    y_vec = torch.linspace(min_val - 2e-16 + 1e-5, max_val + 2e-16, exp_dim, device=device)
    
    # Expand x to create points
    points = x.unsqueeze(1).repeat(1, exp_dim, 1).reshape(N * exp_dim, d)
    
    # Replace dimension 'dim' with grid values
    y_expanded = y_vec.unsqueeze(0).repeat(N, 1).reshape(-1)
    points[:, dim] = y_expanded
    
    return points


def smooth(x: np.ndarray, window_len: int = 11, window: str = 'hanning') -> np.ndarray:
    """
    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    Args:
        x: The input signal 
        window_len: The dimension of the smoothing window; should be an odd integer
        window: The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns:
        The smoothed signal
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval(f'np.{window}({window_len})')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def replace_nan_inf(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN and Inf values in tensor with zeros."""
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def predict_vine(x: torch.Tensor, vine, dim: int, exp_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict conditional distribution using vine copula.
    
    Args:
        x: Input data, shape [N, d]
        vine: Fitted vine copula object
        dim: Dimension to predict
        exp_dim: Number of expansion points
        
    Returns:
        p: Probability values, shape [N * exp_dim]
        y_ml: Maximum likelihood predictions, shape [N]
        y_em: Expectation maximization predictions, shape [N]
    """
    device = x.device if torch.is_tensor(x) else torch.device('cpu')
    x = torch.as_tensor(x, device=device, dtype=torch.float32)
    
    # Create evaluation points
    points = create_points(x, dim, exp_dim)
    
    # Evaluate vine at points
    p, p_cop, logp = vine.evaluation(points)
    
    # Reshape to [N, exp_dim]
    p1 = p.reshape(x.shape[0], exp_dim)
    p1 = replace_nan_inf(p1)
    
    # Create y vector
    min_dim = x[:, dim].min()
    max_dim = x[:, dim].max()
    y_vec = torch.linspace(min_dim - 2e-16 + 1e-5, max_dim + 2e-16, exp_dim, device=device)
    
    # Smooth probabilities
    mov_p = torch.zeros_like(p1)
    for i in range(p1.shape[0]):
        if p1.shape[1] > 4:
            smoothed = smooth(p1[i, :].cpu().numpy(), 4, 'flat')
            mov_p[i, 3:] = torch.from_numpy(smoothed[3:]).to(device)
        else:
            mov_p[i, :] = p1[i, :]
    
    # Maximum likelihood prediction
    ind_max = torch.argmax(mov_p, dim=1)
    y_ml = y_vec[ind_max]
    
    # Expectation maximization prediction
    y_diff = torch.diff(y_vec, prepend=y_vec[0:1])
    
    # Normalize probabilities
    q1 = torch.sum(mov_p * y_diff, dim=1, keepdim=True)
    q = mov_p / (q1 + 1e-10)
    
    # Compute expectation
    y_em = torch.sum(q * y_vec * y_diff, dim=1)
    
    return p, y_ml, y_em


def predict_response(p1: torch.Tensor, y_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict response variables from probability distribution.
    
    Args:
        p1: Probability values, shape [N, exp_dim]
        y_vec: Y values corresponding to probabilities
        
    Returns:
        y_ml: Maximum likelihood predictions, shape [N]
        y_em: Expectation maximization predictions, shape [N]
    """
    device = p1.device
    
    # Smooth probabilities
    mov_p = torch.zeros_like(p1)
    for i in range(p1.shape[0]):
        if p1.shape[1] > 4:
            smoothed = smooth(p1[i, :].cpu().numpy(), 4, 'flat')
            mov_p[i, 3:] = torch.from_numpy(smoothed[3:]).to(device)
        else:
            mov_p[i, :] = p1[i, :]
    
    # Maximum likelihood prediction
    ind_max = torch.argmax(mov_p, dim=1)
    y_ml = y_vec[ind_max]
    
    # Expectation maximization prediction
    y_diff = torch.diff(y_vec, prepend=y_vec[0:1])
    
    # Normalize probabilities
    q1 = torch.sum(mov_p * y_diff, dim=1, keepdim=True)
    q = mov_p / (q1 + 1e-10)
    
    # Compute expectation
    y_em = torch.sum(q * y_vec * y_diff, dim=1)
    
    return y_ml, y_em


def predict_conditional(vine, observed_data: np.ndarray, observed_indices: list, 
                       target_indices: list, n_samples: int = 1000) -> np.ndarray:
    """
    Predict conditional distribution of target variables given observed variables.
    
    Args:
        vine: Fitted vine copula object
        observed_data: Values of observed variables, shape [n_obs, len(observed_indices)]
        observed_indices: Indices of observed variables
        target_indices: Indices of variables to predict
        n_samples: Number of samples to use for Monte Carlo estimation
        
    Returns:
        predictions: Predicted values for target variables, shape [n_obs, len(target_indices)]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_obs = observed_data.shape[0]
    n_target = len(target_indices)
    d = vine.n_cop
    
    # Convert to tensors
    observed_data = torch.tensor(observed_data, dtype=torch.float32, device=device)
    
    predictions = np.zeros((n_obs, n_target))
    
    for i in range(n_obs):
        # Current observation
        obs_values = observed_data[i]
        
        # Generate many samples from the vine
        samples = vine.sample(n_samples)
        samples_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
        
        # Compute distances from samples to observed values
        # Only consider observed dimensions
        sample_obs = samples_tensor[:, observed_indices]
        obs_expanded = obs_values.unsqueeze(0).expand(n_samples, -1)
        
        # Use kernel density estimation for weighting
        # Compute squared distances
        sq_distances = torch.sum((sample_obs - obs_expanded) ** 2, dim=1)
        
        # Apply Gaussian kernel
        bandwidth = 0.5  # Can be optimized
        weights = torch.exp(-sq_distances / (2 * bandwidth ** 2))
        weights = weights / torch.sum(weights)
        
        # Extract target variables from samples
        target_samples = samples_tensor[:, target_indices]
        
        # Compute weighted average (expectation)
        weighted_targets = target_samples * weights.unsqueeze(1)
        predictions[i, :] = torch.sum(weighted_targets, dim=0).cpu().numpy()
    
    return predictions


def predict_conditional_quantiles(vine, observed_data: np.ndarray, observed_indices: list,
                                target_indices: list, quantiles: list = [0.25, 0.5, 0.75],
                                n_samples: int = 5000) -> dict:
    """
    Predict conditional quantiles of target variables given observed variables.
    
    Args:
        vine: Fitted vine copula object
        observed_data: Values of observed variables, shape [n_obs, len(observed_indices)]
        observed_indices: Indices of observed variables
        target_indices: Indices of variables to predict
        quantiles: List of quantiles to compute
        n_samples: Number of samples for Monte Carlo estimation
        
    Returns:
        Dictionary with quantiles for each target variable
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_obs = observed_data.shape[0]
    n_target = len(target_indices)
    
    # Convert to tensors
    observed_data = torch.tensor(observed_data, dtype=torch.float32, device=device)
    
    results = {q: np.zeros((n_obs, n_target)) for q in quantiles}
    
    for i in range(n_obs):
        # Current observation
        obs_values = observed_data[i]
        
        # Generate samples from vine
        samples = vine.sample(n_samples)
        samples_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
        
        # Weight samples based on proximity to observed values
        sample_obs = samples_tensor[:, observed_indices]
        obs_expanded = obs_values.unsqueeze(0).expand(n_samples, -1)
        
        # Compute weights using kernel
        sq_distances = torch.sum((sample_obs - obs_expanded) ** 2, dim=1)
        bandwidth = 0.5
        weights = torch.exp(-sq_distances / (2 * bandwidth ** 2))
        
        # Normalize weights
        weights = weights / torch.sum(weights)
        
        # Extract target samples
        target_samples = samples_tensor[:, target_indices].cpu().numpy()
        weights_np = weights.cpu().numpy()
        
        # Compute weighted quantiles for each target variable
        for j in range(n_target):
            # Sort by target values
            sorted_indices = np.argsort(target_samples[:, j])
            sorted_values = target_samples[sorted_indices, j]
            sorted_weights = weights_np[sorted_indices]
            
            # Compute cumulative weights
            cum_weights = np.cumsum(sorted_weights)
            
            # Find quantile values
            for q in quantiles:
                idx = np.searchsorted(cum_weights, q)
                if idx >= len(sorted_values):
                    idx = len(sorted_values) - 1
                results[q][i, j] = sorted_values[idx]
    
    return results 