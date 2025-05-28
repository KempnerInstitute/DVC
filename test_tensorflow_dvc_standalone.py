"""
Standalone TensorFlow DVC Correlation Prediction Test
====================================================

This test evaluates how well the TensorFlow implementation of Deep Vine Copula (DVC)
can fit to multivariate data, generate samples, and predict pairwise correlations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Explicitly add TensorFlow DVC to path and remove conflicting paths
current_dir = os.path.dirname(os.path.abspath(__file__))
tensorflow_path = os.path.join(current_dir, 'src', 'DVC_tensorflow')

# Remove any existing DVC paths to avoid conflicts
paths_to_remove = []
for path in sys.path:
    if 'DVC' in path and 'tensorflow' not in path.lower():
        paths_to_remove.append(path)

for path in paths_to_remove:
    sys.path.remove(path)

# Add TensorFlow path at the beginning
sys.path.insert(0, tensorflow_path)

print(f"TensorFlow DVC path: {tensorflow_path}")
print(f"Path exists: {os.path.exists(tensorflow_path)}")

try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    from classes.objects import vine_obj_bin, margin_obj, cop_par_obj
    from sampling.vine_sample import vine_copula_sample, vine_cop_par_sample
    from pre_proc.preparation import prep_cop
    # Use scipy's kendalltau instead of TensorFlow's
    # from utils.prob_op import kendalltau
    
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow DVC modules loaded successfully")
    
except ImportError as e:
    print(f"Failed to import TensorFlow DVC modules: {e}")
    TENSORFLOW_AVAILABLE = False


def generate_multivariate_data(n_samples: int, dim: int, correlation_type: str = 'ar1') -> Tuple[np.ndarray, np.ndarray]:
    """Generate multivariate data with known correlation structure"""
    np.random.seed(42)  # For reproducibility
    
    # Create correlation matrix based on type
    if correlation_type == 'ar1':
        rho = 0.7
        corr = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                corr[i, j] = rho ** abs(i - j)
                
    elif correlation_type == 'block':
        corr = np.eye(dim)
        block_size = dim // 2
        # First block
        for i in range(block_size):
            for j in range(block_size):
                if i != j:
                    corr[i, j] = 0.8
        # Second block
        for i in range(block_size, dim):
            for j in range(block_size, dim):
                if i != j:
                    corr[i, j] = 0.6
    else:
        corr = np.eye(dim)
    
    # Generate multivariate normal data
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, corr, n_samples)
    
    # Transform some margins to make it more interesting
    if dim > 1:
        # Transform second variable to exponential
        data[:, 1] = stats.expon.ppf(stats.norm.cdf(data[:, 1]), scale=2)
    if dim > 2:
        # Transform third variable to uniform
        data[:, 2] = stats.uniform.ppf(stats.norm.cdf(data[:, 2]), loc=-1, scale=2)
    
    return data.astype(np.float32), corr


def compute_correlation_matrix(data: np.ndarray, method: str = 'kendall') -> np.ndarray:
    """Compute correlation matrix using specified method"""
    n, d = data.shape
    corr = np.eye(d)
    
    for i in range(d):
        for j in range(i+1, d):
            if method == 'kendall':
                tau, _ = stats.kendalltau(data[:, i], data[:, j])
                corr[i, j] = tau
                corr[j, i] = tau
            elif method == 'spearman':
                rho, _ = stats.spearmanr(data[:, i], data[:, j])
                corr[i, j] = rho
                corr[j, i] = rho
            else:  # pearson
                r = np.corrcoef(data[:, i], data[:, j])[0, 1]
                corr[i, j] = r
                corr[j, i] = r
    
    return corr


def create_vine_margins(dim: int) -> List:
    """Create margin objects for vine copula"""
    margin_vine = []
    for i in range(dim):
        # Use normal margins for simplicity
        mar_p = margin_obj('norm', [0, 1], True)
        margin_vine.append(mar_p)
    return margin_vine


def fit_tensorflow_vine(data: np.ndarray, vine_type: str = 'c-vine', 
                       parametric: bool = True, vine_depth: Optional[int] = None) -> object:
    """Fit TensorFlow DVC vine copula to data"""
    n_samples, dim = data.shape
    
    if vine_depth is None:
        vine_depth = dim
    
    # Create margins
    margin_vine = create_vine_margins(dim)
    
    # Create vine object
    families = "kercop" if not parametric else "param"
    knots = 50
    method = 'matrix'
    
    vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method)
    
    # Prepare data
    sort_n = 'rand'
    prep_cop(data, vine, sort_n)
    
    # Set fitting parameters
    gen_dict = {
        'parallel': True, 
        'binning': False, 
        'param': parametric, 
        'vine_depth': vine_depth - 1,  # TensorFlow uses depth-1 
        'fitted': False
    }
    
    if parametric:
        par_dict = {'param_families': ["ind", "gaussian", "student", "clayton"]}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
    else:
        par_dict = {'param_families': ["ind", "gaussian"]}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
    
    bin_dict = {'n_bin': 3}
    
    # Fit the vine
    print(f"Fitting {vine_type} ({'parametric' if parametric else 'non-parametric'})...")
    start_time = time.time()
    
    try:
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        print(f"Fitting completed in {fit_time:.2f} seconds")
        return vine, fit_time
        
    except Exception as e:
        print(f"Fitting failed: {str(e)}")
        return None, None


def sample_from_vine(vine: object, n_samples: int, parametric: bool = True) -> np.ndarray:
    """Generate samples from fitted vine copula"""
    try:
        if parametric:
            samples = vine_cop_par_sample(vine, n_samples)
        else:
            samples, _, _, _ = vine_copula_sample(vine, n_samples)
        return samples
    except Exception as e:
        print(f"Sampling failed: {str(e)}")
        return None


def run_simple_test():
    """Run a simple TensorFlow DVC test"""
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow DVC not available. Skipping test.")
        return
    
    print("TensorFlow DVC Simple Correlation Test")
    print("=" * 50)
    
    # Generate simple test data
    dim = 3
    n_samples = 300
    data, true_corr = generate_multivariate_data(n_samples, dim, 'ar1')
    data_corr = compute_correlation_matrix(data, method='kendall')
    
    print(f"Data shape: {data.shape}")
    print(f"True correlation matrix:")
    print(true_corr)
    print(f"Data correlation matrix:")
    print(data_corr)
    
    # Test parametric fitting
    print("\nTesting parametric c-vine...")
    try:
        vine, fit_time = fit_tensorflow_vine(data, vine_type='c-vine', parametric=True, vine_depth=dim)
        
        if vine is not None:
            print(f"Fitting successful in {fit_time:.2f} seconds")
            
            # Generate samples
            print("Generating samples...")
            samples = sample_from_vine(vine, 500, parametric=True)
            
            if samples is not None:
                sample_corr = compute_correlation_matrix(samples, method='kendall')
                print(f"Sample correlation matrix:")
                print(sample_corr)
                
                # Compute error
                error = np.mean(np.abs(sample_corr - true_corr))
                print(f"Mean absolute error: {error:.4f}")
                
                # Compute correlation between true and recovered
                true_flat = true_corr[np.triu_indices_from(true_corr, k=1)]
                sample_flat = sample_corr[np.triu_indices_from(sample_corr, k=1)]
                
                if len(true_flat) > 1:
                    recovery_corr, _ = stats.pearsonr(true_flat, sample_flat)
                    print(f"Recovery correlation: {recovery_corr:.4f}")
                
                print("Test completed successfully!")
                
            else:
                print("Sampling failed")
        else:
            print("Fitting failed")
            
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_simple_test() 