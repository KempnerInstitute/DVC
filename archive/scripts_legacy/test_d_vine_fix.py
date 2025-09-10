#!/usr/bin/env python
"""
Test script for D-vine correlation fix

This script tests the enhanced D-vine sampling with correlation preservation
and compares it to the original implementation.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

# Ensure the DVC package is in the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src'))

from DVC_pyolder.objects import vine_obj_bin, margin_obj
from DVC_pyolder.vine_model import fit_vine
from DVC_pyolder.d_vine_fix import apply_d_vine_correlation_fix, validate_d_vine_correlations

def test_correlation_preservation(dimension=4, rho=0.6, n_samples=5000, seed=42):
    """
    Test correlation preservation for different vine structures
    
    Parameters
    ----------
    dimension : int, optional
        Dimension of the test data, by default 4
    rho : float, optional
        Target correlation coefficient, by default 0.6
    n_samples : int, optional
        Number of samples to generate, by default 5000
    seed : int, optional
        Random seed for reproducibility, by default 42
    """
    print(f"Testing {dimension}D vine copulas with rho={rho}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Generate correlated normal data as ground truth
    cov = np.full((dimension, dimension), rho)
    np.fill_diagonal(cov, 1.0)
    data = np.random.multivariate_normal(np.zeros(dimension), cov, size=n_samples)
    true_corr = np.corrcoef(data, rowvar=False)
    
    # 2. Define margins
    margins = []
    for i in range(dimension):
        loc, scale = norm.fit(data[:, i])
        margins.append(margin_obj('norm', [loc, scale], True))
    
    # 3. Setup vine configurations
    configs = {
        "C-Vine": {
            'vine_family': 'c-vine',
            'method': 'optimal'
        },
        "D-Vine": {
            'vine_family': 'd-vine',
            'method': 'optimal'
        },
        "D-Vine (Enhanced)": {
            'vine_family': 'd-vine',
            'method': 'optimal',
            'apply_fix': True
        }
    }
    
    # 4. Create and fit vine models
    results = {}
    
    for name, config in configs.items():
        print(f"\nFitting {name}...")
        
        # Create vine object
        vine = vine_obj_bin(
            config['vine_family'],
            ['gaussian'],
            dimension,
            margins,
            knots=40,
            method=config['method']
        )
        
        # Prepare dicts for fitting
        gen_dict = {'param': True, 'binning': False, 'fitted': False, 'parallel': True, 'vine_depth': dimension}
        npc_dict = {}
        par_dict = {'param_families': ['gaussian']}
        bin_dict = {'n_bin': 1}
        
        # Configure
        cfg = {
            'vine': {
                'knots': 40,
                'family': config['vine_family'],
                'method': config['method']
            },
            'general': {
                'param': True, 
                'binning': False,
                'fitted': False
            },
            'optimizer': {
                'jit': True,
                'batch_edges': True,
                'batch_size': 5,
                'max_iter_phase1': 70,
                'lr_phase1': 0.10,
                'tol_phase1': 1e-5,
                'max_iter_phase2': 100,
                'lr_phase2': 0.03,
                'tol_phase2': 5e-5
            },
            'bandwidth': {'method': 'rule_of_thumb', 'knn_k': 10},
            'npc': {'opt_method': 'LL1', 'grad_precompute': True},
            'sampler': {'fast_parametric': True, 'fast_nonparam': True}
        }
        
        # Fit the vine
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
        
        # Apply D-vine fix if requested
        if config.get('apply_fix', False):
            apply_d_vine_correlation_fix(vine)
        
        # Generate samples
        samples = vine.sample(n_samples)
        
        # Calculate correlation matrix
        sample_corr = np.corrcoef(samples, rowvar=False)
        corr_error = np.mean(np.abs(sample_corr - true_corr))
        
        print(f"  Correlation error: {corr_error:.4f}")
        
        # Store results
        results[name] = {
            'true_corr': true_corr,
            'sample_corr': sample_corr,
            'corr_error': corr_error,
            'samples': samples
        }
    
    # 5. Visualize results
    plot_correlation_comparison(results, f"{dimension}D_comparison")
    
    return results

def plot_correlation_comparison(results, filename_prefix):
    """
    Plot comparison of correlation matrices
    
    Parameters
    ----------
    results : dict
        Dictionary of results
    filename_prefix : str
        Prefix for output filename
    """
    # Get dimension from the first result
    first_model = list(results.keys())[0]
    dimension = results[first_model]['true_corr'].shape[0]
    
    # Create figure
    fig, axes = plt.subplots(2, len(results) + 1, figsize=(4 * (len(results) + 1), 8))
    
    # Plot true correlation matrix
    im0 = axes[0, 0].imshow(results[first_model]['true_corr'], cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title("True Correlation")
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Add text annotations
    for i in range(dimension):
        for j in range(dimension):
            axes[0, 0].text(j, i, f'{results[first_model]["true_corr"][i, j]:.2f}', 
                    ha='center', va='center', color='black')
    
    # Plot each model's correlation matrix and error
    for idx, (name, result) in enumerate(results.items()):
        # Correlation matrix
        im1 = axes[0, idx + 1].imshow(result['sample_corr'], cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, idx + 1].set_title(f"{name}\nError: {result['corr_error']:.4f}")
        plt.colorbar(im1, ax=axes[0, idx + 1])
        
        # Add text annotations
        for i in range(dimension):
            for j in range(dimension):
                axes[0, idx + 1].text(j, i, f'{result["sample_corr"][i, j]:.2f}', 
                        ha='center', va='center', color='black')
        
        # Error matrix
        error_matrix = np.abs(result['sample_corr'] - result['true_corr'])
        im2 = axes[1, idx + 1].imshow(error_matrix, cmap='Reds', vmin=0, vmax=0.6)
        axes[1, idx + 1].set_title(f"Absolute Error")
        plt.colorbar(im2, ax=axes[1, idx + 1])
        
        # Add text annotations
        for i in range(dimension):
            for j in range(dimension):
                axes[1, idx + 1].text(j, i, f'{error_matrix[i, j]:.3f}', 
                        ha='center', va='center', color='black')
    
    # Barplot of error by distance in chain
    error_by_distance = {name: [] for name in results.keys()}
    
    for dist in range(1, dimension):
        for name, result in results.items():
            # Extract errors for pairs at this distance
            errors = []
            for i in range(dimension - dist):
                j = i + dist
                error = abs(result['sample_corr'][i, j] - result['true_corr'][i, j])
                errors.append(error)
            error_by_distance[name].append(np.mean(errors))
    
    # Create barplot
    x = np.arange(dimension - 1)
    width = 0.8 / len(results)
    
    for idx, (name, errors) in enumerate(error_by_distance.items()):
        offset = idx * width - 0.4 + width/2
        axes[1, 0].bar(x + offset, errors, width, label=name)
    
    axes[1, 0].set_title("Error by Chain Distance")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"Dist {d+1}" for d in x])
    axes[1, 0].set_ylabel("Mean Absolute Error")
    axes[1, 0].legend()
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_correlation_comparison.png", dpi=150)
    plt.close()
    
    # Plot scatter comparison for first pair and longest-range pair
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # First pair (i=0, j=1)
    i, j = 0, 1
    for name, result in results.items():
        axes[0].scatter(result['samples'][:, i], result['samples'][:, j], alpha=0.3, 
                      label=f"{name}: r={result['sample_corr'][i, j]:.2f}")
    
    axes[0].set_title(f"Variables {i} vs {j} (Adjacent)")
    axes[0].set_xlabel(f"Variable {i}")
    axes[0].set_ylabel(f"Variable {j}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Longest-range pair (0 and dimension-1)
    i, j = 0, dimension - 1
    for name, result in results.items():
        axes[1].scatter(result['samples'][:, i], result['samples'][:, j], alpha=0.3, 
                      label=f"{name}: r={result['sample_corr'][i, j]:.2f}")
    
    axes[1].set_title(f"Variables {i} vs {j} (Distance {j-i})")
    axes[1].set_xlabel(f"Variable {i}")
    axes[1].set_ylabel(f"Variable {j}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_pair_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    # Test with different dimensions
    for dim in [3, 4, 6]:
        test_correlation_preservation(dimension=dim, rho=0.6, n_samples=5000, seed=42)
    
    print("\nAll tests completed.") 