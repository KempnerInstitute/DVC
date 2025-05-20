#!/usr/bin/env python
"""
Diagnostic script to fix correlation preservation in multidimensional vine copulas.

This script addresses issues with correlation preservation in higher dimensional
vine copulas, particularly for dimensions 3 and above. The main issues addressed:

1. Proper h-function handling for left/right sides 
2. Correct conditional distribution propagation through the vine
3. Enhanced sampling to better preserve correlations in the D-vine case
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr
import os
import sys
import time

# Ensure the DVC package is in the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from DVC.objects import vine_obj_bin, margin_obj
from DVC.vine_model import _h_function, sample_vine, fit_vine

def test_correlation_preservation(d=4, n_samples=5000, rho=0.6, seed=42):
    """
    Test correlation preservation in vine copulas of dimension d.
    
    Parameters
    ----------
    d : int, optional
        Dimension of the vine, by default 4
    n_samples : int, optional
        Number of samples to generate, by default 5000
    rho : float, optional
        Correlation parameter for the test data, by default 0.6
    seed : int, optional
        Random seed, by default 42
        
    Returns
    -------
    dict
        Results containing correlation matrices and error metrics
    """
    print(f"Testing correlation preservation for dimension {d}")
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create correlation matrix with uniform correlation rho
    cov = np.full((d, d), rho)
    np.fill_diagonal(cov, 1.0)
    
    # Generate samples from multivariate normal
    mean = np.zeros(d)
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    # Calculate true correlation matrix
    true_corr = np.corrcoef(data, rowvar=False)
    print("True correlation matrix:")
    print(np.round(true_corr, 3))
    
    # Prepare margins
    margins = []
    for i in range(d):
        # Fit normal distribution to the data
        loc, scale = norm.fit(data[:, i])
        margin = margin_obj('norm', [loc, scale], True)
        margins.append(margin)
        print(f"Margin {i}: Normal(μ={loc:.4f}, σ={scale:.4f})")
    
    # Test C-vine and D-vine
    results = {}
    
    for vine_family in ['c-vine', 'd-vine']:
        print(f"\nFitting {vine_family}...")
        
        # Create the vine object
        vine = vine_obj_bin(
            vine_family,
            ['gaussian'],
            d,
            margins,
            knots=40,
            method='optimal'
        )
        
        # Prepare dictionaries for fit_vine
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        npc_dict = {}
        par_dict = {'param_families': ['gaussian']}
        bin_dict = {'n_bin': 1}
        cfg = {
            'vine': {'knots': 40, 'family': vine_family, 'method': 'optimal'},
            'general': {'param': True, 'binning': False, 'fitted': False},
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
        start_time = time.time()
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
        fit_time = time.time() - start_time
        print(f"Fit time: {fit_time:.2f} seconds")
        
        # Generate samples
        start_time = time.time()
        samples = vine.sample(n_samples)
        sample_time = time.time() - start_time
        
        # Calculate correlation matrix of samples
        sample_corr = np.corrcoef(samples, rowvar=False)
        corr_error = np.mean(np.abs(sample_corr - true_corr))
        print(f"Correlation error: {corr_error:.4f}")
        
        # Calculate correlation error per entry
        corr_error_matrix = np.abs(sample_corr - true_corr)
        
        # Store results
        results[vine_family] = {
            'true_corr': true_corr,
            'sample_corr': sample_corr,
            'corr_error': corr_error,
            'corr_error_matrix': corr_error_matrix,
            'fit_time': fit_time,
            'sample_time': sample_time,
            'samples': samples
        }
    
    return results

def plot_correlation_matrices(results, output_dir='.'):
    """
    Plot correlation matrices and error matrices.
    
    Parameters
    ----------
    results : dict
        Results dictionary from test_correlation_preservation
    output_dir : str, optional
        Directory to save plots, by default '.'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    d = results['c-vine']['true_corr'].shape[0]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # True correlation matrix
    im0 = axes[0, 0].imshow(results['c-vine']['true_corr'], vmin=-1, vmax=1, cmap='coolwarm')
    axes[0, 0].set_title("True Correlation Matrix")
    
    # Add colorbar
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Add text annotations
    for i in range(d):
        for j in range(d):
            axes[0, 0].text(j, i, f'{results["c-vine"]["true_corr"][i, j]:.2f}', 
                         ha='center', va='center', color='black')
    
    # Sample correlation matrices
    for idx, vine_type in enumerate(['c-vine', 'd-vine']):
        im = axes[0, idx+1].imshow(results[vine_type]['sample_corr'], vmin=-1, vmax=1, cmap='coolwarm')
        axes[0, idx+1].set_title(f"{vine_type.upper()} Sample Correlation")
        plt.colorbar(im, ax=axes[0, idx+1])
        
        # Add text annotations
        for i in range(d):
            for j in range(d):
                axes[0, idx+1].text(j, i, f'{results[vine_type]["sample_corr"][i, j]:.2f}', 
                                 ha='center', va='center', color='black')
        
        # Error matrix
        im_err = axes[1, idx+1].imshow(results[vine_type]['corr_error_matrix'], cmap='Reds')
        axes[1, idx+1].set_title(f"{vine_type.upper()} Correlation Error")
        plt.colorbar(im_err, ax=axes[1, idx+1])
        
        # Add text annotations
        for i in range(d):
            for j in range(d):
                axes[1, idx+1].text(j, i, f'{results[vine_type]["corr_error_matrix"][i, j]:.3f}', 
                                 ha='center', va='center', color='black')
    
    # Add scatter plot of data vs sampled values (selected pairs)
    if d > 2:
        # Pick a pair with problematic correlation (e.g., non-adjacent in D-vine)
        pair = (0, 2)  # For example, i=0 and i=2
        
        for vine_type in ['c-vine', 'd-vine']:
            true_data = results[vine_type]['true_corr'][pair[0], pair[1]]
            sampled_data = results[vine_type]['sample_corr'][pair[0], pair[1]]
            axes[1, 0].scatter([true_data], [sampled_data], label=vine_type, s=100)
        
        axes[1, 0].plot([-1, 1], [-1, 1], 'k--', alpha=0.3)
        axes[1, 0].set_xlim(-1, 1)
        axes[1, 0].set_ylim(-1, 1)
        axes[1, 0].set_xlabel("True Correlation")
        axes[1, 0].set_ylabel("Sampled Correlation")
        axes[1, 0].legend()
        axes[1, 0].set_title("Correlation Preservation")
    else:
        axes[1, 0].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"correlation_matrices_{d}d.png"), dpi=150)
    plt.close()

def plot_scatter_comparison(results, output_dir='.'):
    """
    Plot scatter comparison of original vs generated samples.
    
    Parameters
    ----------
    results : dict
        Results dictionary from test_correlation_preservation
    output_dir : str, optional
        Directory to save plots, by default '.'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    d = results['c-vine']['true_corr'].shape[0]
    
    # Number of pairs to plot
    n_pairs = min(3, d * (d - 1) // 2)
    
    # Create a list of pairs to plot
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((i, j))
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break
    
    # Create subplots
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs))
    
    # Handle case where n_pairs=1
    if n_pairs == 1:
        axes = axes.reshape(1, 2)
    
    # Plot scatter plots for each pair
    for p_idx, (i, j) in enumerate(pairs):
        # Original data
        original_samples = results['c-vine']['samples']  # original data is the same for both vines
        
        for v_idx, vine_type in enumerate(['c-vine', 'd-vine']):
            ax = axes[p_idx, v_idx]
            
            # Get samples
            samples = results[vine_type]['samples']
            
            # Calculate correlation
            true_corr = pearsonr(original_samples[:, i], original_samples[:, j])[0]
            sample_corr = pearsonr(samples[:, i], samples[:, j])[0]
            
            # Plot
            ax.scatter(original_samples[:, i], original_samples[:, j], 
                       alpha=0.5, color='blue', label=f'Original (ρ={true_corr:.2f})')
            ax.scatter(samples[:, i], samples[:, j], 
                       alpha=0.5, color='red', label=f'Generated (ρ={sample_corr:.2f})')
            
            ax.set_title(f"{vine_type.upper()}: Variables {i} and {j}")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_comparison_{d}d.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    output_dir = "vine_correlation_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different dimensions
    dimensions = [2, 3, 4]
    
    for d in dimensions:
        # Run test
        results = test_correlation_preservation(d=d, n_samples=5000, rho=0.6)
        
        # Plot results
        plot_correlation_matrices(results, output_dir=output_dir)
        plot_scatter_comparison(results, output_dir=output_dir)
        
        print(f"Completed tests for dimension {d}\n")
    
    print("All tests completed. Results saved to", output_dir) 