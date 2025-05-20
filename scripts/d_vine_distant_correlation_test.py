#!/usr/bin/env python
"""
Specialized test script focusing on distant variable correlations in D-vines

This script directly tests the improvement in correlation preservation for
non-adjacent variables in D-vine structures, which is a key issue our fix addresses.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr

# Ensure the DVC package is in the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src'))

from DVC.objects import vine_obj_bin, margin_obj
from DVC.vine_model import fit_vine, sample_vine
from DVC.d_vine_fix import enhanced_d_vine_sample, validate_d_vine_correlations

# This ensures different seeds produce different results
import random
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def generate_test_data(dim=5, n_samples=5000, rho=0.6):
    """
    Generate test data with uniform correlation structure
    
    Parameters
    ----------
    dim : int, optional
        Dimension of the data, by default 5
    n_samples : int, optional
        Number of samples, by default 5000
    rho : float, optional
        Target correlation, by default 0.6
        
    Returns
    -------
    tuple
        (data, true_corr, margins)
    """
    # Create correlation matrix with uniform correlation
    cov = np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)
    
    # Generate multivariate normal data
    data = np.random.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    true_corr = np.corrcoef(data, rowvar=False)
    
    # Create margin objects
    margins = []
    for i in range(dim):
        loc, scale = norm.fit(data[:, i])
        margins.append(margin_obj('norm', [loc, scale], True))
    
    return data, true_corr, margins

def fit_d_vine(data, margins, dim):
    """
    Fit a D-vine to the data
    
    Parameters
    ----------
    data : np.ndarray
        Training data
    margins : list
        List of margin objects
    dim : int
        Dimension of the data
        
    Returns
    -------
    vine_obj_bin
        Fitted vine object
    """
    # Create vine object
    vine = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian'],
        vine_depth=dim,
        margin=margins,
        knots=40,
        method='optimal'
    )
    
    # Prepare dictionaries for fitting
    gen_dict = {'param': True, 'binning': False, 'fitted': False, 'parallel': True, 'vine_depth': dim}
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    
    # Configure
    cfg = {
        'vine': {
            'knots': 40,
            'family': 'd-vine',
            'method': 'optimal'
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
    
    return vine

def test_d_vine_distant_correlation(dim=6, n_samples=10000, rho=0.6, output_dir="."):
    """
    Test distant variable correlation preservation in D-vines
    
    Parameters
    ----------
    dim : int, optional
        Dimension of the data, by default 6
    n_samples : int, optional
        Number of samples, by default 10000
    rho : float, optional
        Target correlation, by default 0.6
    output_dir : str, optional
        Directory to save outputs, by default "."
        
    Returns
    -------
    dict
        Dictionary of results
    """
    print(f"\nTesting {dim}D D-vine with rho={rho}")
    
    # Generate test data
    data, true_corr, margins = generate_test_data(dim, n_samples, rho)
    
    print(f"True correlation matrix:")
    print(np.round(true_corr, 3))
    
    # Fit D-vine
    vine = fit_d_vine(data, margins, dim)
    
    # Generate samples with original method
    samples_orig = sample_vine(vine, n_samples)
    orig_corr = np.corrcoef(samples_orig, rowvar=False)
    orig_error = np.mean(np.abs(orig_corr - true_corr))
    print(f"Original method correlation error: {orig_error:.4f}")
    
    # Generate samples with enhanced method
    samples_enh = enhanced_d_vine_sample(vine, n_samples)
    enh_corr = np.corrcoef(samples_enh, rowvar=False)
    enh_error = np.mean(np.abs(enh_corr - true_corr))
    print(f"Enhanced method correlation error: {enh_error:.4f}")
    
    # Calculate error by distance in the chain
    print("\nError by distance in D-vine chain:")
    print("Distance | Original | Enhanced | Improvement")
    print("-" * 50)
    
    distance_errors = {}
    for dist in range(1, dim):
        # Calculate errors for pairs at each distance
        orig_errors = []
        enh_errors = []
        
        for i in range(dim - dist):
            j = i + dist
            orig_error = abs(orig_corr[i, j] - true_corr[i, j])
            enh_error = abs(enh_corr[i, j] - true_corr[i, j])
            orig_errors.append(orig_error)
            enh_errors.append(enh_error)
        
        # Calculate average error for this distance
        orig_mean = np.mean(orig_errors)
        enh_mean = np.mean(enh_errors)
        improvement = (orig_mean - enh_mean) / orig_mean * 100 if orig_mean > 0 else 0.0
        
        print(f"{dist:8d} | {orig_mean:.4f}   | {enh_mean:.4f}   | {improvement:+.1f}%")
        
        distance_errors[dist] = {
            'original': orig_mean,
            'enhanced': enh_mean,
            'improvement': improvement
        }
    
    # Visualize results
    plot_correlation_matrices(true_corr, orig_corr, enh_corr, output_dir, dim)
    plot_error_by_distance(distance_errors, output_dir, dim)
    plot_scatter_comparison(samples_orig, samples_enh, data, output_dir, dim)
    
    return {
        'true_corr': true_corr,
        'orig_corr': orig_corr,
        'enh_corr': enh_corr,
        'orig_error': orig_error,
        'enh_error': enh_error,
        'distance_errors': distance_errors
    }

def plot_correlation_matrices(true_corr, orig_corr, enh_corr, output_dir, dim):
    """
    Plot correlation matrices
    
    Parameters
    ----------
    true_corr : np.ndarray
        True correlation matrix
    orig_corr : np.ndarray
        Original method correlation matrix
    enh_corr : np.ndarray
        Enhanced method correlation matrix
    output_dir : str
        Output directory
    dim : int
        Dimension
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot correlation matrices
    im0 = axes[0, 0].imshow(true_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title("True Correlation")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(orig_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 1].set_title(f"Original Method\nMean Error: {np.mean(np.abs(orig_corr - true_corr)):.4f}")
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(enh_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 2].set_title(f"Enhanced Method\nMean Error: {np.mean(np.abs(enh_corr - true_corr)):.4f}")
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Plot error matrices
    orig_error = np.abs(orig_corr - true_corr)
    enh_error = np.abs(enh_corr - true_corr)
    
    im3 = axes[1, 1].imshow(orig_error, cmap='Reds', vmin=0, vmax=0.6)
    axes[1, 1].set_title("Original Method Error")
    plt.colorbar(im3, ax=axes[1, 1])
    
    im4 = axes[1, 2].imshow(enh_error, cmap='Reds', vmin=0, vmax=0.6)
    axes[1, 2].set_title("Enhanced Method Error")
    plt.colorbar(im4, ax=axes[1, 2])
    
    # Plot improvement matrix
    improvement = (orig_error - enh_error) / np.maximum(orig_error, 1e-10) * 100
    im5 = axes[1, 0].imshow(improvement, cmap='PiYG', vmin=-20, vmax=40)
    axes[1, 0].set_title("Improvement (%)")
    plt.colorbar(im5, ax=axes[1, 0])
    
    # Add text annotations for all matrices
    for i in range(dim):
        for j in range(dim):
            if i != j:  # Skip diagonal elements
                # True correlation values
                axes[0, 0].text(j, i, f'{true_corr[i, j]:.2f}', 
                              ha='center', va='center', color='black')
                
                # Original correlation values
                axes[0, 1].text(j, i, f'{orig_corr[i, j]:.2f}', 
                              ha='center', va='center', color='black')
                
                # Enhanced correlation values
                axes[0, 2].text(j, i, f'{enh_corr[i, j]:.2f}', 
                              ha='center', va='center', color='black')
                
                # Original error values
                axes[1, 1].text(j, i, f'{orig_error[i, j]:.3f}', 
                              ha='center', va='center', color='black')
                
                # Enhanced error values
                axes[1, 2].text(j, i, f'{enh_error[i, j]:.3f}', 
                              ha='center', va='center', color='black')
                
                # Improvement values
                axes[1, 0].text(j, i, f'{improvement[i, j]:.1f}%', 
                              ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d_vine_{dim}d_correlation_matrices.png"), dpi=150)
    plt.close()

def plot_error_by_distance(distance_errors, output_dir, dim):
    """
    Plot error by distance in the D-vine chain
    
    Parameters
    ----------
    distance_errors : dict
        Dictionary of errors by distance
    output_dir : str
        Output directory
    dim : int
        Dimension
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    distances = sorted(distance_errors.keys())
    x = np.arange(len(distances))
    width = 0.35
    
    orig_errors = [distance_errors[d]['original'] for d in distances]
    enh_errors = [distance_errors[d]['enhanced'] for d in distances]
    
    plt.bar(x - width/2, orig_errors, width, label='Original Method')
    plt.bar(x + width/2, enh_errors, width, label='Enhanced Method')
    
    plt.xlabel('Distance between variables in D-vine chain')
    plt.ylabel('Mean Absolute Error')
    plt.title('Correlation Preservation Error by Distance in D-vine')
    plt.xticks(x, [str(d) for d in distances])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add improvement percentage text
    for i, d in enumerate(distances):
        improvement = distance_errors[d]['improvement']
        plt.text(i, max(orig_errors[i], enh_errors[i]) + 0.01, 
                f"{improvement:+.1f}%", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d_vine_{dim}d_error_by_distance.png"), dpi=150)
    plt.close()

def plot_scatter_comparison(samples_orig, samples_enh, data_orig, output_dir, dim):
    """
    Plot scatter comparison for distant variables
    
    Parameters
    ----------
    samples_orig : np.ndarray
        Samples from original method
    samples_enh : np.ndarray
        Samples from enhanced method
    data_orig : np.ndarray
        Original data
    output_dir : str
        Output directory
    dim : int
        Dimension
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # For a D-vine, the most problematic correlation is typically between variables 
    # at maximum distance from each other (e.g., X_0 and X_{d-1})
    i, j = 0, dim - 1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original data
    axes[0].scatter(data_orig[:, i], data_orig[:, j], alpha=0.3, color='blue')
    r_true = pearsonr(data_orig[:, i], data_orig[:, j])[0]
    axes[0].set_title(f"Original Data\nCorrelation: {r_true:.4f}")
    axes[0].set_xlabel(f"Variable {i}")
    axes[0].set_ylabel(f"Variable {j}")
    axes[0].grid(True, alpha=0.3)
    
    # Original method
    axes[1].scatter(samples_orig[:, i], samples_orig[:, j], alpha=0.3, color='red')
    r_orig = pearsonr(samples_orig[:, i], samples_orig[:, j])[0]
    err_orig = abs(r_orig - r_true)
    axes[1].set_title(f"Original Method\nCorrelation: {r_orig:.4f} (Error: {err_orig:.4f})")
    axes[1].set_xlabel(f"Variable {i}")
    axes[1].grid(True, alpha=0.3)
    
    # Enhanced method
    axes[2].scatter(samples_enh[:, i], samples_enh[:, j], alpha=0.3, color='green')
    r_enh = pearsonr(samples_enh[:, i], samples_enh[:, j])[0]
    err_enh = abs(r_enh - r_true)
    improvement = (err_orig - err_enh) / err_orig * 100 if err_orig > 0 else 0.0
    axes[2].set_title(f"Enhanced Method\nCorrelation: {r_enh:.4f} (Error: {err_enh:.4f}, Improved: {improvement:.1f}%)")
    axes[2].set_xlabel(f"Variable {i}")
    axes[2].grid(True, alpha=0.3)
    
    # Match axis limits
    for ax in axes:
        ax.set_xlim(np.min(data_orig[:, i]) - 0.5, np.max(data_orig[:, i]) + 0.5)
        ax.set_ylim(np.min(data_orig[:, j]) - 0.5, np.max(data_orig[:, j]) + 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"d_vine_{dim}d_scatter_comparison.png"), dpi=150)
    plt.close()
    
    # Additional plot comparing all pairs for the maximum distance
    if dim > 3:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Find the most distant pairs (up to 6)
        distant_pairs = []
        for dist in range(dim-1, 0, -1):
            for i in range(dim - dist):
                j = i + dist
                distant_pairs.append((i, j))
                if len(distant_pairs) >= 6:
                    break
            if len(distant_pairs) >= 6:
                break
        
        # Plot the pairs
        for idx, (i, j) in enumerate(distant_pairs[:6]):
            if idx < len(axes):
                distance = j - i
                r_true = pearsonr(data_orig[:, i], data_orig[:, j])[0]
                r_orig = pearsonr(samples_orig[:, i], samples_orig[:, j])[0]
                r_enh = pearsonr(samples_enh[:, i], samples_enh[:, j])[0]
                
                err_orig = abs(r_orig - r_true)
                err_enh = abs(r_enh - r_true)
                
                # Scatter plot with regression lines
                axes[idx].scatter(samples_orig[:, i], samples_orig[:, j], alpha=0.2, color='red', label=f'Original (r={r_orig:.2f})')
                axes[idx].scatter(samples_enh[:, i], samples_enh[:, j], alpha=0.2, color='green', label=f'Enhanced (r={r_enh:.2f})')
                
                # Calculate and plot regression lines
                x_line = np.linspace(np.min(samples_orig[:, i]), np.max(samples_orig[:, i]), 100)
                axes[idx].plot(x_line, r_true * x_line, 'b--', linewidth=2, label=f'True (r={r_true:.2f})')
                axes[idx].plot(x_line, r_orig * x_line, 'r-', linewidth=1, alpha=0.7)
                axes[idx].plot(x_line, r_enh * x_line, 'g-', linewidth=1, alpha=0.7)
                
                axes[idx].set_title(f"Variables {i} vs {j} (Dist={distance})")
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        # Hide any unused subplots
        for idx in range(len(distant_pairs), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"d_vine_{dim}d_distant_pairs.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    output_dir = "d_vine_correlation_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test with different dimensions
    for dim in [3, 4, 6, 8]:
        test_d_vine_distant_correlation(dim=dim, n_samples=5000, rho=0.6, output_dir=output_dir)
    
    print("\nAll tests completed.") 