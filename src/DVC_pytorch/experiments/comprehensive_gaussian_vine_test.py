"""
Comprehensive Gaussian Vine Copula Test
=======================================

This experiment:
1. Generates data from multivariate Gaussian copula with known correlation structure
2. Tests different marginal distributions (normal, exponential, uniform, student-t)
3. Fits C-vine, D-vine, and R-vine models
4. Generates samples from fitted models
5. Compares pairwise correlations and entropies
6. Plots correlation matrices for visual comparison
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple
import time

# Import DVC modules
from classes.objects import vine_obj_bin, margin_obj
from pre_proc.preparation import prep_cop
from info.info_estimation import entropy_h_estimation
from utils.prob_op import kendalltau
from plot.plot_vine import plot_vine


def generate_gaussian_copula_data(n_samples: int, dim: int, correlation_matrix: np.ndarray,
                                marginals: List[Dict]) -> np.ndarray:
    """
    Generate data from Gaussian copula with specified marginals
    
    Args:
        n_samples: Number of samples
        dim: Dimension
        correlation_matrix: Target correlation matrix
        marginals: List of marginal distribution specifications
        
    Returns:
        Generated data
    """
    # Generate multivariate normal with specified correlation
    mean = np.zeros(dim)
    mvn = np.random.multivariate_normal(mean, correlation_matrix, n_samples)
    
    # Transform to uniform using normal CDF
    uniform_data = stats.norm.cdf(mvn)
    
    # Transform to specified marginals
    data = np.zeros_like(uniform_data)
    for i in range(dim):
        marg = marginals[i]
        if marg['type'] == 'normal':
            data[:, i] = stats.norm.ppf(uniform_data[:, i], 
                                       loc=marg['loc'], scale=marg['scale'])
        elif marg['type'] == 'exponential':
            data[:, i] = stats.expon.ppf(uniform_data[:, i], 
                                        scale=marg['scale'])
        elif marg['type'] == 'uniform':
            data[:, i] = stats.uniform.ppf(uniform_data[:, i], 
                                          loc=marg['loc'], scale=marg['scale'])
        elif marg['type'] == 'student':
            data[:, i] = stats.t.ppf(uniform_data[:, i], 
                                    df=marg['df'], loc=marg['loc'], scale=marg['scale'])
        elif marg['type'] == 'gamma':
            data[:, i] = stats.gamma.ppf(uniform_data[:, i], 
                                        a=marg['a'], scale=marg['scale'])
    
    return data


def create_correlation_matrix(dim: int, structure: str = 'toeplitz', rho: float = 0.5) -> np.ndarray:
    """
    Create correlation matrix with specified structure
    
    Args:
        dim: Dimension
        structure: 'toeplitz', 'ar1', 'block', or 'random'
        rho: Base correlation parameter
        
    Returns:
        Correlation matrix
    """
    if structure == 'toeplitz':
        # Toeplitz structure: corr[i,j] = rho^|i-j|
        corr = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                corr[i, j] = rho ** abs(i - j)
    
    elif structure == 'ar1':
        # AR(1) structure
        corr = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                corr[i, j] = rho ** abs(i - j)
    
    elif structure == 'block':
        # Block diagonal structure
        block_size = dim // 2
        corr = np.eye(dim)
        # First block
        for i in range(block_size):
            for j in range(block_size):
                if i != j:
                    corr[i, j] = rho
        # Second block
        for i in range(block_size, dim):
            for j in range(block_size, dim):
                if i != j:
                    corr[i, j] = rho * 0.7
    
    elif structure == 'random':
        # Random positive definite matrix
        A = np.random.randn(dim, dim) * 0.3
        corr = np.dot(A, A.T)
        # Normalize to correlation matrix
        D = np.diag(1.0 / np.sqrt(np.diag(corr)))
        corr = np.dot(D, np.dot(corr, D))
    
    else:
        raise ValueError(f"Unknown structure: {structure}")
    
    return corr


def compute_correlation_matrix(data: np.ndarray, method: str = 'kendall') -> np.ndarray:
    """
    Compute correlation matrix from data
    
    Args:
        data: Data array
        method: 'kendall' or 'pearson'
        
    Returns:
        Correlation matrix
    """
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


def fit_vine_model(data: torch.Tensor, vine_type: str, dim: int) -> vine_obj_bin:
    """
    Fit vine copula model to data
    
    Args:
        data: Data tensor
        vine_type: 'c-vine', 'd-vine', or 'r-vine'
        dim: Dimension
        
    Returns:
        Fitted vine object
    """
    device = data.device
    dtype = data.dtype
    
    # Create margin objects
    margins = []
    for i in range(dim):
        margins.append(margin_obj('norm', [0, 1], True))
    
    # Create vine object
    if vine_type == 'r-vine':
        vine = vine_obj_bin('r-vine', ['gaussian'], dim, margins, 11, 'random')
    else:
        vine = vine_obj_bin(vine_type, ['gaussian'], dim, margins, 11, 'matrix')
    
    # Prepare data
    data_prep = prep_cop(data, vine, 'no_sort')
    data_prep = torch.tensor(data_prep, dtype=dtype, device=device)
    
    # Fit parameters
    gen_dict = {
        'param': True,
        'binning': False,
        'fitted': False,
        'parallel': True,
        'vine_depth': dim
    }
    
    par_dict = {
        'param_families': ['gaussian', 'student', 'clayton', 'frank']
    }
    
    npc_dict = {
        'opt_method': 'trust-exact',
        'batch_paral': 10
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit the vine
    start_time = time.time()
    vine.fit(data_prep, gen_dict, npc_dict, par_dict, bin_dict)
    fit_time = time.time() - start_time
    
    return vine, fit_time


def plot_correlation_comparison(true_corr: np.ndarray, 
                              data_corr: np.ndarray,
                              sample_corrs: Dict[str, np.ndarray],
                              title: str = "Correlation Matrix Comparison"):
    """
    Plot correlation matrices for comparison
    """
    n_vines = len(sample_corrs) + 2
    fig, axes = plt.subplots(1, n_vines, figsize=(5*n_vines, 4))
    
    # Plot true correlation
    sns.heatmap(true_corr, ax=axes[0], cmap='coolwarm', center=0, 
                vmin=-1, vmax=1, square=True, cbar=True,
                annot=True, fmt='.2f', annot_kws={'size': 8})
    axes[0].set_title('True Correlation')
    
    # Plot data correlation
    sns.heatmap(data_corr, ax=axes[1], cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, cbar=True,
                annot=True, fmt='.2f', annot_kws={'size': 8})
    axes[1].set_title('Data Correlation')
    
    # Plot sample correlations
    for idx, (vine_type, corr) in enumerate(sample_corrs.items()):
        sns.heatmap(corr, ax=axes[idx+2], cmap='coolwarm', center=0,
                    vmin=-1, vmax=1, square=True, cbar=True,
                    annot=True, fmt='.2f', annot_kws={'size': 8})
        axes[idx+2].set_title(f'{vine_type} Samples')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def run_comprehensive_test():
    """
    Run comprehensive test with different configurations
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    n_samples = 2000
    n_test_samples = 2000
    dimensions = [3, 5]
    correlation_structures = ['toeplitz', 'block']
    vine_types = ['c-vine', 'd-vine', 'r-vine']
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Marginal configurations
    marginal_configs = {
        'mixed': [
            {'type': 'normal', 'loc': 0, 'scale': 1},
            {'type': 'exponential', 'scale': 2},
            {'type': 'uniform', 'loc': -1, 'scale': 2},
            {'type': 'student', 'df': 5, 'loc': 0, 'scale': 1},
            {'type': 'gamma', 'a': 2, 'scale': 1}
        ],
        'all_normal': [
            {'type': 'normal', 'loc': i, 'scale': 1} for i in range(5)
        ],
        'all_student': [
            {'type': 'student', 'df': 5, 'loc': 0, 'scale': 1} for _ in range(5)
        ]
    }
    
    # Results storage
    results = []
    
    for dim in dimensions:
        for corr_structure in correlation_structures:
            for marginal_type, marginals in marginal_configs.items():
                # Use only first 'dim' marginals
                marginals_subset = marginals[:dim]
                
                print(f"\n{'='*60}")
                print(f"Testing: dim={dim}, structure={corr_structure}, marginals={marginal_type}")
                print(f"{'='*60}")
                
                # Generate true correlation matrix
                true_corr = create_correlation_matrix(dim, corr_structure, rho=0.6)
                
                # Generate data
                data = generate_gaussian_copula_data(n_samples, dim, true_corr, marginals_subset)
                data_tensor = torch.tensor(data, dtype=dtype, device=device)
                
                # Compute data correlation
                data_corr = compute_correlation_matrix(data, method='kendall')
                
                # Store sample correlations
                sample_corrs = {}
                entropies = {}
                fit_times = {}
                
                # Fit different vine types
                for vine_type in vine_types:
                    print(f"\nFitting {vine_type}...")
                    
                    try:
                        # Fit vine
                        vine, fit_time = fit_vine_model(data_tensor, vine_type, dim)
                        fit_times[vine_type] = fit_time
                        
                        # Generate samples
                        samples = vine.sample(n_test_samples)
                        samples_np = samples.cpu().numpy()
                        
                        # Compute sample correlation
                        sample_corr = compute_correlation_matrix(samples_np, method='kendall')
                        sample_corrs[vine_type] = sample_corr
                        
                        # Estimate entropy
                        entropy = entropy_h_estimation(vine, 'copula', cases=5000)
                        entropies[vine_type] = entropy.item()
                        
                        # Compute correlation error
                        corr_error = np.mean(np.abs(sample_corr - true_corr))
                        
                        print(f"  Fit time: {fit_time:.2f}s")
                        print(f"  Entropy: {entropy.item():.4f}")
                        print(f"  Mean correlation error: {corr_error:.4f}")
                        
                        # Store results
                        results.append({
                            'dimension': dim,
                            'corr_structure': corr_structure,
                            'marginal_type': marginal_type,
                            'vine_type': vine_type,
                            'fit_time': fit_time,
                            'entropy': entropy.item(),
                            'corr_error': corr_error
                        })
                        
                    except Exception as e:
                        print(f"  Error fitting {vine_type}: {str(e)}")
                        continue
                
                # Plot correlation comparison
                if len(sample_corrs) > 0:
                    fig = plot_correlation_comparison(
                        true_corr, data_corr, sample_corrs,
                        title=f"Correlation Comparison (d={dim}, {corr_structure}, {marginal_type})"
                    )
                    plt.savefig(f'correlation_comparison_d{dim}_{corr_structure}_{marginal_type}.png', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
    
    # Create results summary
    if results:
        df = pd.DataFrame(results)
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        summary = df.groupby(['dimension', 'vine_type']).agg({
            'corr_error': ['mean', 'std'],
            'entropy': ['mean', 'std'],
            'fit_time': ['mean', 'std']
        }).round(4)
        
        print(summary)
        
        # Save results
        df.to_csv('comprehensive_vine_test_results.csv', index=False)
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Correlation error by vine type and dimension
        for dim in dimensions:
            dim_data = df[df['dimension'] == dim]
            x = np.arange(len(vine_types))
            means = [dim_data[dim_data['vine_type'] == vt]['corr_error'].mean() 
                    for vt in vine_types]
            stds = [dim_data[dim_data['vine_type'] == vt]['corr_error'].std() 
                   for vt in vine_types]
            axes[0, 0].bar(x + dim*0.3, means, 0.25, yerr=stds, 
                          label=f'd={dim}', alpha=0.7)
        axes[0, 0].set_xticks(x + 0.15)
        axes[0, 0].set_xticklabels(vine_types)
        axes[0, 0].set_ylabel('Mean Correlation Error')
        axes[0, 0].set_title('Correlation Error by Vine Type')
        axes[0, 0].legend()
        
        # Plot 2: Entropy by marginal type
        marginal_types = df['marginal_type'].unique()
        for i, mt in enumerate(marginal_types):
            mt_data = df[df['marginal_type'] == mt]
            x = np.arange(len(vine_types))
            means = [mt_data[mt_data['vine_type'] == vt]['entropy'].mean() 
                    for vt in vine_types]
            axes[0, 1].plot(x, means, 'o-', label=mt, linewidth=2, markersize=8)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(vine_types)
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].set_title('Entropy by Marginal Type')
        axes[0, 1].legend()
        
        # Plot 3: Fit time comparison
        for dim in dimensions:
            dim_data = df[df['dimension'] == dim]
            x = np.arange(len(vine_types))
            means = [dim_data[dim_data['vine_type'] == vt]['fit_time'].mean() 
                    for vt in vine_types]
            axes[1, 0].plot(x, means, 'o-', label=f'd={dim}', linewidth=2, markersize=8)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(vine_types)
        axes[1, 0].set_ylabel('Fit Time (seconds)')
        axes[1, 0].set_title('Computational Time by Vine Type')
        axes[1, 0].legend()
        
        # Plot 4: Error by correlation structure
        for cs in correlation_structures:
            cs_data = df[df['corr_structure'] == cs]
            x = np.arange(len(vine_types))
            means = [cs_data[cs_data['vine_type'] == vt]['corr_error'].mean() 
                    for vt in vine_types]
            axes[1, 1].plot(x, means, 'o-', label=cs, linewidth=2, markersize=8)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(vine_types)
        axes[1, 1].set_ylabel('Mean Correlation Error')
        axes[1, 1].set_title('Error by Correlation Structure')
        axes[1, 1].legend()
        
        plt.suptitle('Comprehensive Vine Copula Test Results', fontsize=16)
        plt.tight_layout()
        plt.savefig('comprehensive_test_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\nResults saved to:")
        print("  - comprehensive_vine_test_results.csv")
        print("  - comprehensive_test_summary.png")
        print("  - correlation_comparison_*.png")


if __name__ == "__main__":
    run_comprehensive_test() 