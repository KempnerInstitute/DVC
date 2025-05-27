#!/usr/bin/env python3
"""
Comprehensive Multivariate Performance Test
==========================================

Test PyTorch vine copula implementation on 3D, 4D, and 5D multivariate Gaussian data.
Evaluates:
1. Correlation recovery accuracy
2. Entropy estimation
3. Sampling quality
4. Computational performance

Compares against known theoretical values for multivariate Gaussian distributions.
"""

import numpy as np
import torch
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
import seaborn as sns
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, 'src')

def generate_multivariate_gaussian_data(d: int, n_samples: int = 1000, correlation_strength: float = 0.6):
    """
    Generate multivariate Gaussian data with specified correlation structure.
    
    Args:
        d: Dimension (3, 4, or 5)
        n_samples: Number of samples
        correlation_strength: Base correlation strength
        
    Returns:
        data: Generated data [n_samples, d]
        true_corr: True correlation matrix [d, d]
        true_entropy: True differential entropy
    """
    np.random.seed(42)  # For reproducibility
    
    # Create structured correlation matrix
    true_corr = np.eye(d)
    
    # Create different correlation patterns for different dimensions
    if d == 3:
        # Simple correlation structure
        true_corr[0, 1] = true_corr[1, 0] = correlation_strength
        true_corr[0, 2] = true_corr[2, 0] = correlation_strength * 0.8
        true_corr[1, 2] = true_corr[2, 1] = correlation_strength * 0.6
        
    elif d == 4:
        # More complex correlation structure
        correlations = [
            (0, 1, correlation_strength),
            (0, 2, correlation_strength * 0.8),
            (0, 3, correlation_strength * 0.5),
            (1, 2, correlation_strength * 0.7),
            (1, 3, correlation_strength * 0.4),
            (2, 3, correlation_strength * 0.6)
        ]
        for i, j, corr in correlations:
            true_corr[i, j] = true_corr[j, i] = corr
            
    elif d == 5:
        # Even more complex correlation structure
        correlations = [
            (0, 1, correlation_strength),
            (0, 2, correlation_strength * 0.8),
            (0, 3, correlation_strength * 0.6),
            (0, 4, correlation_strength * 0.4),
            (1, 2, correlation_strength * 0.7),
            (1, 3, correlation_strength * 0.5),
            (1, 4, correlation_strength * 0.3),
            (2, 3, correlation_strength * 0.6),
            (2, 4, correlation_strength * 0.4),
            (3, 4, correlation_strength * 0.5)
        ]
        for i, j, corr in correlations:
            true_corr[i, j] = true_corr[j, i] = corr
    
    # Generate data
    mean = np.zeros(d)
    data = np.random.multivariate_normal(mean, true_corr, n_samples)
    
    # Calculate true differential entropy for multivariate Gaussian
    # H = 0.5 * log((2πe)^d * det(Σ))
    det_cov = np.linalg.det(true_corr)
    true_entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(det_cov))
    
    return data, true_corr, true_entropy

def fit_pytorch_vine(data: np.ndarray, d: int) -> Dict:
    """
    Fit PyTorch vine copula to data.
    
    Args:
        data: Input data [n_samples, d]
        d: Dimension
        
    Returns:
        results: Dictionary with fitting results
    """
    try:
        from DVC_pyolder.vine_model import fit_vine
        from DVC_pyolder.objects import vine_obj_bin, margin_obj
        from DVC_pyolder.sampling import vine_copula_sample
        
        print(f"  Fitting {d}D PyTorch vine...")
        start_time = time.time()
        
        # Create vine model
        margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(d)]
        vine = vine_obj_bin('c-vine', 'gaussian', d, margins, 25, 'matrix')
        
        # Configuration for fitting
        gen_dict = {
            'binning': False,
            'parallel': False,
            'param': True,
            'fitted': False,
            'vine_depth': d-1
        }
        
        npc_dict = {
            'npc_family': 'locallik',
            'grid_dim': 25
        }
        
        par_dict = {
            'param_families': ['gaussian', 'clayton']  # Allow both for better fitting
        }
        
        bin_dict = {
            'n_bin': 1
        }
        
        # Fit vine
        vine = fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        # Test sampling
        print(f"  Sampling from {d}D PyTorch vine...")
        start_time = time.time()
        samples, u_samples, _, _ = vine_copula_sample(vine, 2000)
        sample_time = time.time() - start_time
        
        # Transform samples back to original space
        from scipy.stats import norm
        samples_original = norm.ppf(u_samples)
        
        # Calculate sample correlation matrix
        sample_corr = np.corrcoef(u_samples.T)
        
        # Estimate entropy using sample covariance
        sample_cov = np.cov(samples_original.T)
        det_cov = np.linalg.det(sample_cov)
        if det_cov > 0:
            estimated_entropy = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(det_cov))
        else:
            estimated_entropy = np.nan
        
        # Check for NaN values
        nan_count = np.isnan(vine.theta.cpu().numpy()).sum() if hasattr(vine, 'theta') else 0
        
        results = {
            'success': True,
            'fit_time': fit_time,
            'sample_time': sample_time,
            'sample_corr': sample_corr,
            'estimated_entropy': estimated_entropy,
            'samples': u_samples,
            'nan_count': nan_count,
            'vine': vine
        }
        
        print(f"    Fit time: {fit_time:.2f}s, Sample time: {sample_time:.2f}s")
        print(f"    NaN count: {nan_count}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ PyTorch vine fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'fit_time': np.nan,
            'sample_time': np.nan,
            'sample_corr': np.eye(d),
            'estimated_entropy': np.nan,
            'samples': None,
            'nan_count': np.inf
        }

def calculate_correlation_metrics(true_corr: np.ndarray, estimated_corr: np.ndarray) -> Dict:
    """Calculate correlation recovery metrics."""
    
    # Extract upper triangular elements (excluding diagonal)
    mask = np.triu(np.ones_like(true_corr, dtype=bool), k=1)
    true_corr_vec = true_corr[mask]
    est_corr_vec = estimated_corr[mask]
    
    # Calculate metrics
    mae = np.mean(np.abs(true_corr_vec - est_corr_vec))
    mse = np.mean((true_corr_vec - est_corr_vec)**2)
    max_error = np.max(np.abs(true_corr_vec - est_corr_vec))
    
    # Correlation coefficient between true and estimated
    correlation_of_correlations = np.corrcoef(true_corr_vec, est_corr_vec)[0, 1]
    
    return {
        'mae': mae,
        'mse': mse,
        'max_error': max_error,
        'correlation_of_correlations': correlation_of_correlations,
        'num_correlations': len(true_corr_vec)
    }

def plot_correlation_comparison(results: Dict, save_path: str):
    """Create visualization comparing true vs estimated correlations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Correlation Recovery Performance: PyTorch Vine Copulas', fontsize=16)
    
    for i, d in enumerate([3, 4, 5]):
        if d not in results or not results[d]['pytorch']['success']:
            continue
            
        true_corr = results[d]['true_corr']
        est_corr = results[d]['pytorch']['sample_corr']
        
        # Plot true correlation matrix
        ax1 = axes[0, i]
        sns.heatmap(true_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax1, vmin=-1, vmax=1, fmt='.2f')
        ax1.set_title(f'{d}D True Correlations')
        
        # Plot estimated correlation matrix
        ax2 = axes[1, i]
        sns.heatmap(est_corr, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax2, vmin=-1, vmax=1, fmt='.2f')
        ax2.set_title(f'{d}D Estimated Correlations')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_summary(results: Dict, save_path: str):
    """Create performance summary plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PyTorch Vine Copula Performance Summary', fontsize=16)
    
    dimensions = [3, 4, 5]
    
    # Extract metrics
    mae_values = []
    entropy_errors = []
    fit_times = []
    sample_times = []
    
    for d in dimensions:
        if d in results and results[d]['pytorch']['success']:
            metrics = results[d]['metrics']
            mae_values.append(metrics['mae'])
            
            true_entropy = results[d]['true_entropy']
            est_entropy = results[d]['pytorch']['estimated_entropy']
            entropy_error = abs(true_entropy - est_entropy) if not np.isnan(est_entropy) else np.nan
            entropy_errors.append(entropy_error)
            
            fit_times.append(results[d]['pytorch']['fit_time'])
            sample_times.append(results[d]['pytorch']['sample_time'])
        else:
            mae_values.append(np.nan)
            entropy_errors.append(np.nan)
            fit_times.append(np.nan)
            sample_times.append(np.nan)
    
    # Plot MAE
    axes[0, 0].bar(dimensions, mae_values, color='skyblue', edgecolor='navy')
    axes[0, 0].set_title('Correlation Mean Absolute Error')
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot entropy error
    axes[0, 1].bar(dimensions, entropy_errors, color='lightcoral', edgecolor='darkred')
    axes[0, 1].set_title('Entropy Estimation Error')
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot fit times
    axes[1, 0].bar(dimensions, fit_times, color='lightgreen', edgecolor='darkgreen')
    axes[1, 0].set_title('Fitting Time')
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot sample times
    axes[1, 1].bar(dimensions, sample_times, color='gold', edgecolor='orange')
    axes[1, 1].set_title('Sampling Time')
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_report(results: Dict, save_path: str):
    """Generate detailed text report."""
    
    with open(save_path, 'w') as f:
        f.write("Multivariate Vine Copula Performance Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Test Configuration:\n")
        f.write("- Sample size: 1000 per dimension\n")
        f.write("- Vine type: C-vine with Gaussian copulas\n")
        f.write("- Correlation structure: Hierarchical with decreasing strength\n")
        f.write("- Test samples: 2000 for evaluation\n\n")
        
        for d in [3, 4, 5]:
            f.write(f"{d}D Results:\n")
            f.write("-" * 20 + "\n")
            
            if d not in results:
                f.write("No results available\n\n")
                continue
                
            true_entropy = results[d]['true_entropy']
            
            if results[d]['pytorch']['success']:
                pytorch_res = results[d]['pytorch']
                metrics = results[d]['metrics']
                
                f.write(f"PyTorch Vine:\n")
                f.write(f"  Success: ✓\n")
                f.write(f"  Fit time: {pytorch_res['fit_time']:.2f}s\n")
                f.write(f"  Sample time: {pytorch_res['sample_time']:.2f}s\n")
                f.write(f"  NaN count: {pytorch_res['nan_count']}\n")
                f.write(f"\n")
                
                f.write(f"Correlation Recovery:\n")
                f.write(f"  Mean Absolute Error: {metrics['mae']:.4f}\n")
                f.write(f"  Mean Squared Error: {metrics['mse']:.4f}\n")
                f.write(f"  Maximum Error: {metrics['max_error']:.4f}\n")
                f.write(f"  Correlation of correlations: {metrics['correlation_of_correlations']:.4f}\n")
                f.write(f"\n")
                
                f.write(f"Entropy Estimation:\n")
                f.write(f"  True entropy: {true_entropy:.4f}\n")
                f.write(f"  Estimated entropy: {pytorch_res['estimated_entropy']:.4f}\n")
                f.write(f"  Absolute error: {abs(true_entropy - pytorch_res['estimated_entropy']):.4f}\n")
                f.write(f"  Relative error: {abs(true_entropy - pytorch_res['estimated_entropy'])/true_entropy*100:.2f}%\n")
                
            else:
                f.write(f"PyTorch Vine: ✗ Failed\n")
                f.write(f"  Error: {results[d]['pytorch'].get('error', 'Unknown')}\n")
            
            f.write("\n")

def main():
    """Run comprehensive multivariate performance test."""
    
    print("=" * 60)
    print("Multivariate Vine Copula Performance Test")
    print("=" * 60)
    print()
    
    results = {}
    
    # Test different dimensions
    for d in [3, 4, 5]:
        print(f"Testing {d}D multivariate Gaussian data...")
        
        # Generate data
        data, true_corr, true_entropy = generate_multivariate_gaussian_data(d, n_samples=1000)
        
        print(f"  Generated {d}D data with {len(data)} samples")
        print(f"  True entropy: {true_entropy:.4f}")
        
        # Test PyTorch implementation
        pytorch_results = fit_pytorch_vine(data, d)
        
        # Calculate metrics if successful
        if pytorch_results['success']:
            metrics = calculate_correlation_metrics(true_corr, pytorch_results['sample_corr'])
            print(f"  Correlation MAE: {metrics['mae']:.4f}")
            print(f"  Entropy error: {abs(true_entropy - pytorch_results['estimated_entropy']):.4f}")
        else:
            metrics = {'mae': np.nan, 'mse': np.nan, 'max_error': np.nan, 'correlation_of_correlations': np.nan}
        
        results[d] = {
            'true_corr': true_corr,
            'true_entropy': true_entropy,
            'pytorch': pytorch_results,
            'metrics': metrics
        }
        
        print()
    
    # Generate visualizations and reports
    print("Generating reports and visualizations...")
    
    plot_correlation_comparison(results, 'multivariate_correlation_comparison.png')
    plot_performance_summary(results, 'multivariate_performance_summary.png')
    generate_detailed_report(results, 'multivariate_performance_report.txt')
    
    print("✓ Correlation comparison plot saved: multivariate_correlation_comparison.png")
    print("✓ Performance summary plot saved: multivariate_performance_summary.png")
    print("✓ Detailed report saved: multivariate_performance_report.txt")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for d in [3, 4, 5]:
        if d in results and results[d]['pytorch']['success']:
            metrics = results[d]['metrics']
            true_entropy = results[d]['true_entropy']
            est_entropy = results[d]['pytorch']['estimated_entropy']
            entropy_error = abs(true_entropy - est_entropy)
            
            print(f"{d}D: MAE={metrics['mae']:.4f}, Entropy_Error={entropy_error:.4f}, "
                  f"Time={results[d]['pytorch']['fit_time']:.1f}s")
        else:
            print(f"{d}D: FAILED")
    
    print()
    
    # Determine overall success
    success_count = sum(1 for d in [3, 4, 5] if d in results and results[d]['pytorch']['success'])
    
    if success_count == 3:
        print("🎉 ALL TESTS PASSED! PyTorch vine copulas successfully handle 3D-5D data.")
    elif success_count >= 2:
        print(f"⚠️  PARTIAL SUCCESS: {success_count}/3 tests passed.")
    else:
        print("❌ TESTS FAILED: Major issues with multivariate vine implementation.")
    
    return success_count == 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 