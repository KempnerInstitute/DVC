"""
Optimization Criteria for Vine Copula Structure Selection

Implements various criteria for evaluating and comparing vine copula structures:
- Information-theoretic criteria (AIC, BIC, entropy)
- Dependence-based criteria (Kendall's tau, Spearman's rho)
- Likelihood-based criteria
- Custom domain-specific criteria
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from scipy.stats import kendalltau, spearmanr
import logging

from ..core.objects import vine_obj_bin
from ..core.info_estimation import vine_entropy, mutual_information

logger = logging.getLogger(__name__)


def aic_criterion(vine: vine_obj_bin, data: np.ndarray, 
                  info_dict: Optional[Dict[str, Any]] = None) -> float:
    """
    Akaike Information Criterion for vine copula model selection.
    
    AIC = -2 * log-likelihood + 2 * k
    where k is the number of parameters.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Fitted vine copula model
    data : np.ndarray
        Data used for fitting, shape (n_samples, n_features)
    info_dict : dict, optional
        Additional parameters for likelihood computation
        
    Returns
    -------
    float
        AIC value (lower is better)
    """
    try:
        # Compute log-likelihood
        log_likelihood = _compute_log_likelihood(vine, data, info_dict)
        
        # Estimate number of parameters
        n_params = _count_vine_parameters(vine)
        
        # AIC = -2 * log-likelihood + 2 * k
        aic = -2 * log_likelihood + 2 * n_params
        
        return aic
        
    except Exception as e:
        logger.warning(f"Failed to compute AIC: {e}")
        return float('inf')  # Return worst possible score


def bic_criterion(vine: vine_obj_bin, data: np.ndarray,
                  info_dict: Optional[Dict[str, Any]] = None) -> float:
    """
    Bayesian Information Criterion for vine copula model selection.
    
    BIC = -2 * log-likelihood + k * log(n)
    where k is the number of parameters and n is the sample size.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Fitted vine copula model
    data : np.ndarray
        Data used for fitting, shape (n_samples, n_features)
    info_dict : dict, optional
        Additional parameters for likelihood computation
        
    Returns
    -------
    float
        BIC value (lower is better)
    """
    try:
        # Compute log-likelihood
        log_likelihood = _compute_log_likelihood(vine, data, info_dict)
        
        # Estimate number of parameters
        n_params = _count_vine_parameters(vine)
        n_samples = data.shape[0]
        
        # BIC = -2 * log-likelihood + k * log(n)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return bic
        
    except Exception as e:
        logger.warning(f"Failed to compute BIC: {e}")
        return float('inf')


def entropy_criterion(vine: vine_obj_bin, data: np.ndarray,
                     info_dict: Optional[Dict[str, Any]] = None) -> float:
    """
    Entropy-based criterion for vine copula evaluation.
    
    Uses the vine to estimate the entropy of the data distribution.
    Higher entropy indicates better model flexibility.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Fitted vine copula model
    data : np.ndarray
        Data for entropy estimation
    info_dict : dict, optional
        Parameters for entropy estimation
        
    Returns
    -------
    float
        Estimated entropy (higher is better for this criterion)
    """
    if info_dict is None:
        info_dict = {
            'alpha': 0.05,
            'cases': 1000,
            'iterations': 10
        }
    
    try:
        # Compute vine entropy
        entropy = vine_entropy(vine, info_dict)
        return entropy
        
    except Exception as e:
        logger.warning(f"Failed to compute entropy criterion: {e}")
        return float('-inf')  # Return worst possible score


def kendall_tau_criterion(vine: vine_obj_bin, data: np.ndarray) -> float:
    """
    Kendall's tau based criterion for vine structure evaluation.
    
    Computes the sum of absolute Kendall's tau values for all edges
    in the vine structure. Higher values indicate stronger dependencies
    captured by the vine.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Vine copula model with structure
    data : np.ndarray
        Data for dependence computation
        
    Returns
    -------
    float
        Sum of absolute Kendall's tau values
    """
    try:
        d = data.shape[1]
        n_samples = data.shape[0]
        
        # Convert to uniform margins using ranks
        data_u = np.zeros_like(data)
        for i in range(d):
            ranks = data[:, i].argsort().argsort() + 1
            data_u[:, i] = ranks / (n_samples + 1)
        
        total_tau = 0.0
        edge_count = 0
        
        # Sum Kendall's tau for all edges in vine structure
        if hasattr(vine, 'ind_vine') and vine.ind_vine:
            for level, edges in enumerate(vine.ind_vine):
                for edge in edges:
                    i, j = edge
                    if 0 <= i < d and 0 <= j < d:
                        tau, _ = kendalltau(data_u[:, i], data_u[:, j])
                        total_tau += abs(tau)
                        edge_count += 1
        
        # Return average absolute tau if edges exist
        return total_tau / max(edge_count, 1)
        
    except Exception as e:
        logger.warning(f"Failed to compute Kendall's tau criterion: {e}")
        return 0.0


def spearman_rho_criterion(vine: vine_obj_bin, data: np.ndarray) -> float:
    """
    Spearman's rho based criterion for vine structure evaluation.
    
    Similar to Kendall's tau criterion but uses Spearman's rank correlation.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Vine copula model with structure
    data : np.ndarray
        Data for dependence computation
        
    Returns
    -------
    float
        Sum of absolute Spearman's rho values
    """
    try:
        d = data.shape[1]
        total_rho = 0.0
        edge_count = 0
        
        # Sum Spearman's rho for all edges in vine structure
        if hasattr(vine, 'ind_vine') and vine.ind_vine:
            for level, edges in enumerate(vine.ind_vine):
                for edge in edges:
                    i, j = edge
                    if 0 <= i < d and 0 <= j < d:
                        rho, _ = spearmanr(data[:, i], data[:, j])
                        total_rho += abs(rho)
                        edge_count += 1
        
        return total_rho / max(edge_count, 1)
        
    except Exception as e:
        logger.warning(f"Failed to compute Spearman's rho criterion: {e}")
        return 0.0


def mutual_information_criterion(vine: vine_obj_bin, data: np.ndarray,
                                info_dict: Optional[Dict[str, Any]] = None) -> float:
    """
    Mutual information based criterion for vine evaluation.
    
    Computes total mutual information captured by the vine structure.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Vine copula model
    data : np.ndarray
        Data for MI computation
    info_dict : dict, optional
        Parameters for MI estimation
        
    Returns
    -------
    float
        Total mutual information
    """
    if info_dict is None:
        info_dict = {
            'alpha': 0.05,
            'cases': 1000,
            'iterations': 5
        }
    
    try:
        d = data.shape[1]
        total_mi = 0.0
        edge_count = 0
        
        # Compute MI for each edge in vine structure
        if hasattr(vine, 'ind_vine') and vine.ind_vine:
            for level, edges in enumerate(vine.ind_vine):
                for edge in edges:
                    i, j = edge
                    if 0 <= i < d and 0 <= j < d:
                        # Compute pairwise MI
                        mi = _pairwise_mutual_information(
                            data[:, i], data[:, j], info_dict
                        )
                        total_mi += mi
                        edge_count += 1
        
        return total_mi / max(edge_count, 1)
        
    except Exception as e:
        logger.warning(f"Failed to compute MI criterion: {e}")
        return 0.0


def log_likelihood_criterion(vine: vine_obj_bin, data: np.ndarray,
                           info_dict: Optional[Dict[str, Any]] = None) -> float:
    """
    Log-likelihood criterion for vine model evaluation.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Fitted vine copula model
    data : np.ndarray
        Data for likelihood computation
    info_dict : dict, optional
        Additional parameters
        
    Returns
    -------
    float
        Log-likelihood value (higher is better)
    """
    try:
        return _compute_log_likelihood(vine, data, info_dict)
    except Exception as e:
        logger.warning(f"Failed to compute log-likelihood: {e}")
        return float('-inf')


def cross_validation_criterion(vine_factory_fn, data: np.ndarray, 
                             k_folds: int = 5, 
                             info_dict: Optional[Dict[str, Any]] = None) -> float:
    """
    K-fold cross-validation criterion for vine model evaluation.
    
    Parameters
    ----------
    vine_factory_fn : callable
        Function that creates and fits a vine model given data
    data : np.ndarray
        Data for cross-validation
    k_folds : int
        Number of cross-validation folds
    info_dict : dict, optional
        Additional parameters
        
    Returns
    -------
    float
        Average cross-validation log-likelihood
    """
    try:
        n_samples = data.shape[0]
        fold_size = n_samples // k_folds
        cv_scores = []
        
        for fold in range(k_folds):
            # Create train/test split
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else n_samples
            
            test_indices = list(range(start_idx, end_idx))
            train_indices = [i for i in range(n_samples) if i not in test_indices]
            
            train_data = data[train_indices]
            test_data = data[test_indices]
            
            # Fit vine on training data
            vine = vine_factory_fn(train_data)
            
            # Evaluate on test data
            test_score = _compute_log_likelihood(vine, test_data, info_dict)
            cv_scores.append(test_score)
        
        return np.mean(cv_scores)
        
    except Exception as e:
        logger.warning(f"Failed to compute cross-validation criterion: {e}")
        return float('-inf')


def composite_criterion(vine: vine_obj_bin, data: np.ndarray,
                       weights: Optional[Dict[str, float]] = None,
                       info_dict: Optional[Dict[str, Any]] = None) -> float:
    """
    Composite criterion combining multiple evaluation metrics.
    
    Parameters
    ----------
    vine : vine_obj_bin
        Vine copula model
    data : np.ndarray
        Data for evaluation
    weights : dict, optional
        Weights for different criteria components
    info_dict : dict, optional
        Additional parameters
        
    Returns
    -------
    float
        Weighted composite score
    """
    if weights is None:
        weights = {
            'aic': -0.3,        # Negative because lower AIC is better
            'entropy': 0.4,     # Positive because higher entropy is better
            'kendall_tau': 0.3  # Positive because higher dependence is better
        }
    
    try:
        composite_score = 0.0
        
        # Compute individual criteria
        criteria_values = {}
        
        if 'aic' in weights:
            criteria_values['aic'] = aic_criterion(vine, data, info_dict)
            
        if 'bic' in weights:
            criteria_values['bic'] = bic_criterion(vine, data, info_dict)
            
        if 'entropy' in weights:
            criteria_values['entropy'] = entropy_criterion(vine, data, info_dict)
            
        if 'kendall_tau' in weights:
            criteria_values['kendall_tau'] = kendall_tau_criterion(vine, data)
            
        if 'spearman_rho' in weights:
            criteria_values['spearman_rho'] = spearman_rho_criterion(vine, data)
            
        if 'mutual_information' in weights:
            criteria_values['mutual_information'] = mutual_information_criterion(
                vine, data, info_dict
            )
        
        # Normalize criteria values to [0, 1] range (simple min-max scaling)
        # This is a simplified approach - in practice you might want more sophisticated normalization
        
        # Compute weighted sum
        for criterion, value in criteria_values.items():
            if criterion in weights and np.isfinite(value):
                composite_score += weights[criterion] * value
        
        return composite_score
        
    except Exception as e:
        logger.warning(f"Failed to compute composite criterion: {e}")
        return float('-inf')


# Helper functions

def _compute_log_likelihood(vine: vine_obj_bin, data: np.ndarray,
                           info_dict: Optional[Dict[str, Any]] = None) -> float:
    """Compute log-likelihood of data under vine model."""
    try:
        if hasattr(vine, 'logpdf'):
            # Use vine's built-in log-pdf method
            data_tensor = torch.from_numpy(data).float()
            log_probs = vine.logpdf(data_tensor)
            return torch.sum(log_probs).item()
            
        elif hasattr(vine, 'evaluation'):
            # Use evaluation method
            p, p_copula, log_marg = vine.evaluation(data)
            
            if isinstance(p, torch.Tensor):
                p = p.cpu().numpy()
            
            # Compute log-likelihood
            log_p = np.log(np.maximum(p, 1e-30))  # Avoid log(0)
            return np.sum(log_p)
            
        else:
            logger.warning("Vine does not have logpdf or evaluation method")
            return float('-inf')
            
    except Exception as e:
        logger.warning(f"Error computing log-likelihood: {e}")
        return float('-inf')


def _count_vine_parameters(vine: vine_obj_bin) -> int:
    """Estimate the number of parameters in a vine copula model."""
    try:
        n_params = 0
        
        # Count parameters in copulas
        if hasattr(vine, 'copulas') and vine.copulas:
            for level_copulas in vine.copulas:
                for copula in level_copulas:
                    if hasattr(copula, 'family'):
                        # Parametric copula
                        if copula.family == 'independence':
                            n_params += 0
                        elif copula.family in ['gaussian', 'frank']:
                            n_params += 1  # One parameter
                        elif copula.family in ['clayton', 'gumbel']:
                            n_params += 1  # One parameter
                        elif copula.family == 'student':
                            n_params += 2  # Correlation + degrees of freedom
                        else:
                            n_params += 1  # Default: one parameter
                    else:
                        # Non-parametric copula - estimate based on bandwidth/grid
                        if hasattr(copula, 'opt_bw'):
                            n_params += 1  # Bandwidth parameter
                        else:
                            n_params += 2  # Default estimate
        
        # Count parameters in margins
        if hasattr(vine, 'margin') and vine.margin:
            for margin in vine.margin:
                if hasattr(margin, 'dist'):
                    if margin.dist == 'norm':
                        n_params += 2  # Mean and variance
                    elif margin.dist == 'uniform':
                        n_params += 2  # Lower and upper bounds
                    else:
                        n_params += 2  # Default: two parameters
        
        return max(n_params, 1)  # At least one parameter
        
    except Exception as e:
        logger.warning(f"Error counting vine parameters: {e}")
        return 1


def _pairwise_mutual_information(x: np.ndarray, y: np.ndarray, 
                               info_dict: Optional[Dict[str, Any]] = None) -> float:
    """Estimate mutual information between two variables."""
    try:
        # Use sklearn's mutual information estimator
        from sklearn.feature_selection import mutual_info_regression
        
        # Reshape for sklearn
        x_reshaped = x.reshape(-1, 1)
        
        # Estimate MI
        mi = mutual_info_regression(x_reshaped, y, random_state=42)[0]
        
        return max(mi, 0.0)  # MI is non-negative
        
    except ImportError:
        # Fallback: use correlation-based approximation
        corr = np.corrcoef(x, y)[0, 1]
        if np.isfinite(corr) and abs(corr) < 0.999:
            # MI approximation for Gaussian: -0.5 * log(1 - rho^2)
            return -0.5 * np.log(1 - corr**2)
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Error computing pairwise MI: {e}")
        return 0.0


def create_custom_criterion(criterion_fn: callable, 
                          name: str = "custom",
                          higher_is_better: bool = True) -> callable:
    """
    Create a custom criterion function.
    
    Parameters
    ----------
    criterion_fn : callable
        Function that takes (vine, data) and returns a score
    name : str
        Name for the criterion
    higher_is_better : bool
        Whether higher scores are better
        
    Returns
    -------
    callable
        Wrapped criterion function with error handling
    """
    def wrapped_criterion(vine: vine_obj_bin, data: np.ndarray, 
                         info_dict: Optional[Dict[str, Any]] = None) -> float:
        try:
            score = criterion_fn(vine, data)
            return score if np.isfinite(score) else (float('-inf') if higher_is_better else float('inf'))
        except Exception as e:
            logger.warning(f"Error in custom criterion '{name}': {e}")
            return float('-inf') if higher_is_better else float('inf')
    
    wrapped_criterion.__name__ = f"{name}_criterion"
    return wrapped_criterion
