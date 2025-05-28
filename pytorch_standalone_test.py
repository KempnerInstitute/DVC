"""
Standalone PyTorch DVC Test
===========================

This test evaluates the PyTorch DVC implementation in isolation to avoid
conflicts with TensorFlow imports.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup PyTorch path only
current_dir = os.path.dirname(os.path.abspath(__file__))
pytorch_path = os.path.join(current_dir, 'src', 'DVC_pytorch')

# Add only PyTorch path
if pytorch_path not in sys.path:
    sys.path.insert(0, pytorch_path)

# Import PyTorch DVC modules
import torch
from classes.objects import vine_obj_bin, margin_obj
from pre_proc.preparation import prep_cop
from info.info_estimation import vine_entropy

def generate_multivariate_data(n_samples: int, dim: int, correlation_type: str = 'ar1', 
                             seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate multivariate data with known correlation structure"""
    np.random.seed(seed)
    
    # Create correlation matrix
    if correlation_type == 'ar1':
        rho = 0.7
        corr = np.array([[rho ** abs(i - j) for j in range(dim)] for i in range(dim)])
    elif correlation_type == 'toeplitz':
        rho = 0.6
        corr = np.array([[rho ** abs(i - j) for j in range(dim)] for i in range(dim)])
    elif correlation_type == 'block':
        corr = np.eye(dim)
        block_size = dim // 2
        for i in range(block_size):
            for j in range(block_size):
                if i != j:
                    corr[i, j] = 0.8
        for i in range(block_size, dim):
            for j in range(block_size, dim):
                if i != j:
                    corr[i, j] = 0.6
    else:
        corr = np.eye(dim)
    
    # Generate multivariate normal data
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, corr, n_samples)
    
    # Transform margins for complexity
    if dim > 1:
        data[:, 1] = stats.expon.ppf(stats.norm.cdf(data[:, 1]), scale=2)
    if dim > 2:
        data[:, 2] = stats.uniform.ppf(stats.norm.cdf(data[:, 2]), loc=-1, scale=2)
    if dim > 3:
        data[:, 3] = stats.gamma.ppf(stats.norm.cdf(data[:, 3]), a=2, scale=1)
    
    # Calculate ground truth entropy
    det_corr = np.linalg.det(corr)
    ground_truth_entropy = 0.5 * dim * (1 + np.log(2 * np.pi)) + 0.5 * np.log(det_corr)
    
    return data.astype(np.float32), corr, ground_truth_entropy

def compute_correlation_matrix(data: np.ndarray, method: str = 'kendall') -> np.ndarray:
    """Compute correlation matrix"""
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

def fit_pytorch_vine(data: np.ndarray, vine_type: str, parametric: bool, 
                    vine_depth: int) -> Tuple[Any, float, bool]:
    """Fit PyTorch vine copula"""
    try:
        n_samples, dim = data.shape
        
        # Create margin objects
        margins = []
        for i in range(dim):
            margins.append(margin_obj('norm', [0, 1], True))
        
        # Create vine object
        if vine_type == 'r-vine':
            vine = vine_obj_bin('r-vine', ['gaussian'], dim, margins, 11, 'random')
        else:
            vine = vine_obj_bin(vine_type, ['gaussian'], dim, margins, 11, 'matrix')
        
        # Prepare data - use numpy directly to avoid tensor conversion issues
        data_prep = prep_cop(data, vine, 'no_sort')
        
        # Convert to torch tensor
        device = torch.device('cpu')
        if isinstance(data_prep, np.ndarray):
            data_prep = torch.from_numpy(data_prep.astype(np.float32)).to(device)
        else:
            data_prep = torch.tensor(data_prep, dtype=torch.float32, device=device)
        
        # Configure fitting parameters
        if parametric:
            gen_dict = {
                'param': True,
                'binning': False,
                'fitted': False,
                'parallel': True,
                'vine_depth': min(dim - 1, 4)
            }
            
            par_dict = {
                'param_families': ['gaussian', 'student', 'clayton']
            }
        else:
            gen_dict = {
                'param': False,
                'binning': True,
                'fitted': False,
                'parallel': True,
                'vine_depth': min(dim - 1, 3)
            }
            
            par_dict = {
                'param_families': ['gaussian']
            }
        
        npc_dict = {
            'opt_method': 'trust-exact',
            'batch_paral': 10
        }
        
        bin_dict = {
            'n_bin': 10 if not parametric else 1
        }
        
        start_time = time.time()
        vine.fit(data_prep, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        return vine, fit_time, True
        
    except Exception as e:
        print(f"PyTorch fitting failed: {str(e)}")
        return None, 0.0, False

def sample_pytorch_vine(vine: Any, n_samples: int) -> Tuple[np.ndarray, bool]:
    """Sample from PyTorch vine"""
    try:
        samples = vine.sample(n_samples)
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        return samples, True
    except Exception as e:
        print(f"PyTorch sampling failed: {str(e)}")
        return None, False

def compute_pytorch_entropy(vine: Any, data: np.ndarray) -> Tuple[float, bool]:
    """Compute entropy using PyTorch implementation"""
    try:
        info_dict = {'alpha': 0.05, 'cases': 1000, 'iterations': 10}
        entropy = vine_entropy(vine, info_dict)
        return float(entropy), True
    except Exception as e:
        print(f"PyTorch entropy computation failed: {str(e)}")
        return 0.0, False

def analyze_results(true_corr: np.ndarray, data_corr: np.ndarray, 
                   sample_corr: np.ndarray, method_name: str) -> Dict:
    """Analyze correlation recovery results"""
    
    data_error = np.mean(np.abs(data_corr - true_corr))
    sample_error = np.mean(np.abs(sample_corr - true_corr))
    
    true_flat = true_corr[np.triu_indices_from(true_corr, k=1)]
    sample_flat = sample_corr[np.triu_indices_from(sample_corr, k=1)]
    
    if len(true_flat) > 1:
        recovery_corr, _ = stats.pearsonr(true_flat, sample_flat)
    else:
        recovery_corr = 1.0 if len(true_flat) == 1 and np.abs(true_flat[0] - sample_flat[0]) < 0.1 else 0.0
    
    max_error = np.max(np.abs(sample_corr - true_corr))
    
    return {
        'method': method_name,
        'data_error': data_error,
        'sample_error': sample_error,
        'recovery_correlation': recovery_corr,
        'max_error': max_error,
        'mean_true_corr': np.mean(np.abs(true_flat)),
        'mean_sample_corr': np.mean(np.abs(sample_flat))
    }

def run_pytorch_test():
    """Run PyTorch DVC test"""
    
    print("PyTorch DVC Standalone Test")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {'dim': 3, 'n_samples': 300, 'corr_type': 'ar1'},
        {'dim': 3, 'n_samples': 300, 'corr_type': 'block'},
        {'dim': 4, 'n_samples': 400, 'corr_type': 'toeplitz'},
    ]
    
    vine_types = ['c-vine', 'd-vine']
    methods = [
        {'name': 'Parametric', 'parametric': True},
        {'name': 'Non-parametric', 'parametric': False}
    ]
    
    all_results = []
    
    for config in test_configs:
        dim = config['dim']
        n_samples = config['n_samples']
        corr_type = config['corr_type']
        
        print(f"\nTesting: dim={dim}, n_samples={n_samples}, correlation={corr_type}")
        print("-" * 50)
        
        # Generate test data
        data, true_corr, ground_truth_entropy = generate_multivariate_data(n_samples, dim, corr_type)
        data_corr = compute_correlation_matrix(data, method='kendall')
        
        print(f"Data shape: {data.shape}")
        print(f"True correlation range: [{np.min(true_corr):.3f}, {np.max(true_corr):.3f}]")
        print(f"Ground truth entropy: {ground_truth_entropy:.4f}")
        
        for vine_type in vine_types:
            print(f"\n  Testing {vine_type}:")
            
            for method in methods:
                method_name = method['name']
                parametric = method['parametric']
                
                print(f"    {method_name} method...", end=' ')
                
                try:
                    # Fit vine
                    vine, fit_time, fit_success = fit_pytorch_vine(
                        data, vine_type, parametric, dim
                    )
                    
                    if not fit_success:
                        print("Fit failed")
                        continue
                    
                    # Sample from vine
                    samples, sample_success = sample_pytorch_vine(vine, 500)
                    
                    if not sample_success:
                        print("Sampling failed")
                        continue
                    
                    # Compute sample correlations
                    sample_corr = compute_correlation_matrix(samples, method='kendall')
                    
                    # Compute entropy
                    estimated_entropy, entropy_success = compute_pytorch_entropy(vine, data)
                    
                    if not entropy_success:
                        estimated_entropy = np.nan
                    
                    # Analyze results
                    analysis = analyze_results(true_corr, data_corr, sample_corr, 
                                             f"PyTorch {method_name} {vine_type}")
                    
                    # Add additional metrics
                    analysis.update({
                        'framework': 'pytorch',
                        'fit_time': fit_time,
                        'vine_type': vine_type,
                        'dimension': dim,
                        'n_samples': n_samples,
                        'correlation_type': corr_type,
                        'ground_truth_entropy': ground_truth_entropy,
                        'estimated_entropy': estimated_entropy,
                        'entropy_error': abs(estimated_entropy - ground_truth_entropy) if not np.isnan(estimated_entropy) else np.nan,
                        'sample_correlation': sample_corr
                    })
                    
                    all_results.append(analysis)
                    
                    entropy_str = f"Entropy Error: {analysis['entropy_error']:.4f}, " if not np.isnan(estimated_entropy) else ""
                    print(f"Success! Corr Error: {analysis['sample_error']:.4f}, "
                          f"Recovery: {analysis['recovery_correlation']:.4f}, "
                          f"{entropy_str}"
                          f"Time: {fit_time:.2f}s")
                    
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue
    
    # Create analysis
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 50)
        print("PYTORCH DVC TEST SUMMARY")
        print("=" * 50)
        
        print(f"\nTotal successful tests: {len(df)}")
        print(f"Average correlation error: {df['sample_error'].mean():.4f} ± {df['sample_error'].std():.4f}")
        print(f"Average recovery correlation: {df['recovery_correlation'].mean():.4f} ± {df['recovery_correlation'].std():.4f}")
        print(f"Average fit time: {df['fit_time'].mean():.2f} ± {df['fit_time'].std():.2f} seconds")
        
        # Entropy analysis
        entropy_errors = df['entropy_error'].dropna()
        if len(entropy_errors) > 0:
            print(f"Average entropy error: {entropy_errors.mean():.4f} ± {entropy_errors.std():.4f}")
        
        # Method comparison
        print(f"\nMethod Comparison:")
        for method_type in ['Parametric', 'Non-parametric']:
            method_df = df[df['method'].str.contains(method_type)]
            if len(method_df) > 0:
                print(f"  {method_type}:")
                print(f"    Average correlation error: {method_df['sample_error'].mean():.4f} ± {method_df['sample_error'].std():.4f}")
                print(f"    Average recovery correlation: {method_df['recovery_correlation'].mean():.4f} ± {method_df['recovery_correlation'].std():.4f}")
                print(f"    Average fit time: {method_df['fit_time'].mean():.2f} ± {method_df['fit_time'].std():.2f} seconds")
        
        # Save results
        df.to_csv('pytorch_dvc_test_results.csv', index=False)
        print(f"\nResults saved to: pytorch_dvc_test_results.csv")
        
        # Best result
        best_idx = df['sample_error'].idxmin()
        best_method = df.loc[best_idx, 'method']
        best_error = df.loc[best_idx, 'sample_error']
        print(f"Best result: {best_method} (error: {best_error:.4f})")
        
        return df
    else:
        print("No successful tests completed")
        return None

if __name__ == "__main__":
    run_pytorch_test() 