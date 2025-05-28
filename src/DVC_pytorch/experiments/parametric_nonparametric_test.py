"""
Parametric vs Non-parametric Vine Copula Comparison Test
======================================================

This experiment focuses specifically on comparing parametric and non-parametric
vine copula approaches within the PyTorch implementation:

1. Generates data from known correlation structures
2. Fits both parametric and non-parametric vine models
3. Compares their performance in terms of:
   - Correlation accuracy
   - Entropy estimation
   - Computational time
   - Sample quality
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DVC PyTorch modules
from classes.objects import vine_obj_bin, margin_obj
from pre_proc.preparation import prep_cop
from info.info_estimation import vine_entropy
from utils.prob_op import kendalltau
from plot.plot_vine import plot_vine


def generate_test_data(n_samples: int, dim: int, correlation_type: str = 'ar1') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data with known correlation structure
    
    Args:
        n_samples: Number of samples
        dim: Dimension
        correlation_type: Type of correlation structure
        
    Returns:
        Data array and true correlation matrix
    """
    # Create correlation matrix
    if correlation_type == 'ar1':
        rho = 0.7
        corr = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                corr[i, j] = rho ** abs(i - j)
    elif correlation_type == 'toeplitz':
        rho = 0.6
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
        data[:, 1] = stats.expon.ppf(stats.norm.cdf(data[:, 1]), scale=2)
    if dim > 2:
        data[:, 2] = stats.uniform.ppf(stats.norm.cdf(data[:, 2]), loc=-1, scale=2)
    
    return data, corr


def fit_vine_comparison(data: torch.Tensor, vine_type: str, dim: int) -> Dict:
    """
    Fit both parametric and non-parametric vines and compare results
    
    Args:
        data: Data tensor
        vine_type: Type of vine ('c-vine', 'd-vine', 'r-vine')
        dim: Dimension
        
    Returns:
        Dictionary with comparison results
    """
    device = data.device
    dtype = data.dtype
    results = {}
    
    for param_type in ['parametric', 'nonparametric']:
        parametric = (param_type == 'parametric')
        
        try:
            print(f"    Fitting {param_type} {vine_type}...")
            
            # Create margin objects
            margins = [margin_obj('norm', [0, 1], True) for _ in range(dim)]
            
            # Create vine object
            if vine_type == 'r-vine':
                vine = vine_obj_bin('r-vine', ['gaussian'], dim, margins, 11, 'random')
            else:
                vine = vine_obj_bin(vine_type, ['gaussian'], dim, margins, 11, 'matrix')
            
            # Prepare data
            data_prep = prep_cop(data, vine, 'no_sort')
            data_prep = torch.tensor(data_prep, dtype=dtype, device=device)
            
            # Configure fitting parameters
            if parametric:
                gen_dict = {
                    'param': True,
                    'binning': False,
                    'fitted': False,
                    'parallel': True,
                    'vine_depth': min(dim - 1, 3)
                }
                par_dict = {
                    'param_families': ['gaussian', 'student', 'clayton']
                }
                bin_dict = {'n_bin': 1}
            else:
                gen_dict = {
                    'param': False,
                    'binning': True,
                    'fitted': False,
                    'parallel': True,
                    'vine_depth': min(dim - 1, 2)  # Reduced depth for non-parametric
                }
                par_dict = {
                    'param_families': ['gaussian']  # Fallback
                }
                bin_dict = {'n_bin': 20}  # More bins for non-parametric
            
            npc_dict = {
                'opt_method': 'trust-exact',
                'batch_paral': 5
            }
            
            # Fit the vine
            start_time = time.time()
            vine.fit(data_prep, gen_dict, npc_dict, par_dict, bin_dict)
            fit_time = time.time() - start_time
            
            # Generate samples
            n_samples = 1000
            samples = vine.sample(n_samples)
            samples_np = samples.cpu().numpy()
            
            # Compute sample statistics
            sample_corr = compute_correlation_matrix(samples_np, method='kendall')
            
            # Estimate entropy
            info_dict = {'alpha': 0.05, 'cases': 500, 'iterations': 5}
            entropy = vine_entropy(vine, info_dict)
            
            # Store results
            results[param_type] = {
                'vine': vine,
                'fit_time': fit_time,
                'samples': samples_np,
                'correlation': sample_corr,
                'entropy': entropy,
                'success': True
            }
            
            print(f"      Success! Fit time: {fit_time:.2f}s, Entropy: {entropy:.4f}")
            
        except Exception as e:
            print(f"      Failed: {str(e)}")
            results[param_type] = {
                'success': False,
                'error': str(e)
            }
    
    return results


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
            else:
                r = np.corrcoef(data[:, i], data[:, j])[0, 1]
                corr[i, j] = r
                corr[j, i] = r
    
    return corr


def plot_comparison_results(true_corr: np.ndarray, data_corr: np.ndarray,
                          parametric_corr: np.ndarray, nonparametric_corr: np.ndarray,
                          title: str):
    """Plot correlation matrix comparison"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    matrices = [true_corr, data_corr, parametric_corr, nonparametric_corr]
    titles = ['True Correlation', 'Data Correlation', 'Parametric Samples', 'Non-parametric Samples']
    
    for i, (matrix, subtitle) in enumerate(zip(matrices, titles)):
        sns.heatmap(matrix, ax=axes[i], cmap='coolwarm', center=0,
                    vmin=-1, vmax=1, square=True, cbar=True,
                    annot=True, fmt='.2f', annot_kws={'size': 8})
        axes[i].set_title(subtitle)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def run_parametric_nonparametric_test():
    """
    Run comprehensive parametric vs non-parametric comparison
    """
    print("Starting Parametric vs Non-parametric Vine Copula Comparison")
    print("=" * 70)
    
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test configuration
    test_configs = [
        {'dim': 3, 'n_samples': 500, 'corr_type': 'ar1'},
        {'dim': 3, 'n_samples': 500, 'corr_type': 'toeplitz'},
        {'dim': 4, 'n_samples': 800, 'corr_type': 'block'},
    ]
    
    vine_types = ['c-vine', 'd-vine']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Results storage
    all_results = []
    
    for config in test_configs:
        dim = config['dim']
        n_samples = config['n_samples']
        corr_type = config['corr_type']
        
        print(f"\nTesting: dim={dim}, n_samples={n_samples}, correlation={corr_type}")
        print("-" * 50)
        
        # Generate test data
        data, true_corr = generate_test_data(n_samples, dim, corr_type)
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        
        # Compute data correlation
        data_corr = compute_correlation_matrix(data, method='kendall')
        
        for vine_type in vine_types:
            print(f"  Testing {vine_type}:")
            
            # Fit and compare
            comparison = fit_vine_comparison(data_tensor, vine_type, dim)
            
            # Analyze results
            if (comparison['parametric']['success'] and 
                comparison['nonparametric']['success']):
                
                param_result = comparison['parametric']
                nonparam_result = comparison['nonparametric']
                
                # Compute errors
                param_corr_error = np.mean(np.abs(param_result['correlation'] - true_corr))
                nonparam_corr_error = np.mean(np.abs(nonparam_result['correlation'] - true_corr))
                
                # Store results
                result_entry = {
                    'dimension': dim,
                    'n_samples': n_samples,
                    'correlation_type': corr_type,
                    'vine_type': vine_type,
                    'parametric_fit_time': param_result['fit_time'],
                    'nonparametric_fit_time': nonparam_result['fit_time'],
                    'parametric_entropy': param_result['entropy'],
                    'nonparametric_entropy': nonparam_result['entropy'],
                    'parametric_corr_error': param_corr_error,
                    'nonparametric_corr_error': nonparam_corr_error,
                    'time_ratio': nonparam_result['fit_time'] / param_result['fit_time'],
                    'entropy_diff': abs(param_result['entropy'] - nonparam_result['entropy']),
                    'corr_error_diff': abs(param_corr_error - nonparam_corr_error)
                }
                all_results.append(result_entry)
                
                # Print comparison
                print(f"    Parametric:     Error={param_corr_error:.4f}, "
                      f"Time={param_result['fit_time']:.2f}s, "
                      f"Entropy={param_result['entropy']:.4f}")
                print(f"    Non-parametric: Error={nonparam_corr_error:.4f}, "
                      f"Time={nonparam_result['fit_time']:.2f}s, "
                      f"Entropy={nonparam_result['entropy']:.4f}")
                
                # Create visualization
                fig = plot_comparison_results(
                    true_corr, data_corr,
                    param_result['correlation'],
                    nonparam_result['correlation'],
                    f'Parametric vs Non-parametric: {vine_type}, d={dim}, {corr_type}'
                )
                plt.savefig(f'param_vs_nonparam_{vine_type}_d{dim}_{corr_type}.png',
                           dpi=150, bbox_inches='tight')
                plt.close()
                
            else:
                print(f"    One or both methods failed for {vine_type}")
                if not comparison['parametric']['success']:
                    print(f"      Parametric error: {comparison['parametric']['error']}")
                if not comparison['nonparametric']['success']:
                    print(f"      Non-parametric error: {comparison['nonparametric']['error']}")
    
    # Create summary analysis
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 70)
        print("PARAMETRIC vs NON-PARAMETRIC SUMMARY")
        print("=" * 70)
        
        # Overall statistics
        print(f"\nTotal successful comparisons: {len(df)}")
        
        print("\nAverage performance:")
        print(f"  Parametric correlation error:     {df['parametric_corr_error'].mean():.4f} ± {df['parametric_corr_error'].std():.4f}")
        print(f"  Non-parametric correlation error: {df['nonparametric_corr_error'].mean():.4f} ± {df['nonparametric_corr_error'].std():.4f}")
        print(f"  Parametric fit time:              {df['parametric_fit_time'].mean():.2f} ± {df['parametric_fit_time'].std():.2f} seconds")
        print(f"  Non-parametric fit time:          {df['nonparametric_fit_time'].mean():.2f} ± {df['nonparametric_fit_time'].std():.2f} seconds")
        print(f"  Time ratio (nonparam/param):      {df['time_ratio'].mean():.2f} ± {df['time_ratio'].std():.2f}")
        
        # By vine type
        print("\nBy vine type:")
        vine_summary = df.groupby('vine_type').agg({
            'parametric_corr_error': 'mean',
            'nonparametric_corr_error': 'mean',
            'parametric_fit_time': 'mean',
            'nonparametric_fit_time': 'mean',
            'time_ratio': 'mean'
        }).round(4)
        print(vine_summary)
        
        # Save results
        df.to_csv('parametric_vs_nonparametric_results.csv', index=False)
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Correlation error comparison
        methods = ['Parametric', 'Non-parametric']
        errors = [df['parametric_corr_error'].mean(), df['nonparametric_corr_error'].mean()]
        error_stds = [df['parametric_corr_error'].std(), df['nonparametric_corr_error'].std()]
        
        axes[0, 0].bar(methods, errors, yerr=error_stds, capsize=10, alpha=0.7, 
                       color=['lightblue', 'lightcoral'])
        axes[0, 0].set_ylabel('Mean Correlation Error')
        axes[0, 0].set_title('Correlation Accuracy Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Fit time comparison
        times = [df['parametric_fit_time'].mean(), df['nonparametric_fit_time'].mean()]
        time_stds = [df['parametric_fit_time'].std(), df['nonparametric_fit_time'].std()]
        
        axes[0, 1].bar(methods, times, yerr=time_stds, capsize=10, alpha=0.7,
                       color=['lightgreen', 'orange'])
        axes[0, 1].set_ylabel('Mean Fit Time (seconds)')
        axes[0, 1].set_title('Computational Time Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot of errors
        axes[1, 0].scatter(df['parametric_corr_error'], df['nonparametric_corr_error'],
                          alpha=0.7, s=80)
        max_error = max(df['parametric_corr_error'].max(), df['nonparametric_corr_error'].max())
        axes[1, 0].plot([0, max_error], [0, max_error], 'r--', alpha=0.7, label='y=x')
        axes[1, 0].set_xlabel('Parametric Correlation Error')
        axes[1, 0].set_ylabel('Non-parametric Correlation Error')
        axes[1, 0].set_title('Error Comparison Scatter Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Time ratio distribution
        axes[1, 1].hist(df['time_ratio'], bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Equal time')
        axes[1, 1].set_xlabel('Time Ratio (Non-parametric / Parametric)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Computational Time Ratio Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Parametric vs Non-parametric Vine Copula Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig('parametric_vs_nonparametric_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to:")
        print(f"  - parametric_vs_nonparametric_results.csv")
        print(f"  - parametric_vs_nonparametric_summary.png")
        print(f"  - param_vs_nonparam_*.png (individual comparisons)")


if __name__ == "__main__":
    run_parametric_nonparametric_test() 