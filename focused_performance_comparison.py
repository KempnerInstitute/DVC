"""
Focused Performance Comparison: PyTorch vs TensorFlow DVC

This script provides a focused comparison of the key performance metrics
between PyTorch and TensorFlow DVC implementations.
"""

import numpy as np
import torch
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

# PyTorch imports
from DVC import vine_obj_bin, margin_obj
from DVC.vine_model import fit_vine, evaluate_vine
from DVC.sampling import vine_copula_sample

# TensorFlow imports
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj


def generate_test_data(n=1000, d=4, rho=0.6, seed=42):
    """Generate test data with known correlation structure"""
    np.random.seed(seed)
    
    # Create correlation matrix
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(d), corr, n)
    
    return data.astype(np.float32), corr


def fit_and_evaluate_pytorch(data, vine_type='d-vine'):
    """Fit PyTorch vine and evaluate performance"""
    n, d = data.shape
    
    print("\n--- PYTORCH PERFORMANCE ---")
    
    # Create and fit vine
    vine = vine_obj_bin(
        vine_family=vine_type,
        families=['gaussian', 'ind'],
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {}
    bin_dict = {"n_bin": 1}
    
    # Time fitting
    start_time = time.time()
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    fit_time = time.time() - start_time
    print(f"Fit time: {fit_time:.3f}s")
    
    # Print copula parameters
    print("\nFitted copula parameters:")
    for level, copulas in enumerate(vine.copulas):
        for i, cop in enumerate(copulas):
            if hasattr(cop, 'family') and hasattr(cop, 'theta'):
                if cop.theta is not None:
                    print(f"  Level {level}, Edge {i}: {cop.family}, theta={cop.theta:.4f}")
                else:
                    print(f"  Level {level}, Edge {i}: {cop.family}, theta=None")
    
    # Generate samples
    print("\nGenerating samples...")
    start_time = time.time()
    try:
        samples = vine.sample(500)
        sample_time = time.time() - start_time
        print(f"Sample time: {sample_time:.3f}s")
        
        # Calculate correlation
        sample_corr = np.corrcoef(samples.T)
        
        # Test theta uniformity
        print("\nTesting theta uniformity (p-values from KS test):")
        theta_np = vine.theta.cpu().numpy()
        for level in range(min(d, theta_np.shape[1])):
            pvals = []
            for var in range(d):
                vals = theta_np[:, level, var]
                if np.any(vals != 0):
                    _, pval = stats.kstest(vals, 'uniform')
                    pvals.append(pval)
                else:
                    pvals.append(np.nan)
            print(f"  Level {level}: {[f'{p:.3f}' if not np.isnan(p) else 'N/A' for p in pvals]}")
        
        return {
            'fit_time': fit_time,
            'sample_time': sample_time,
            'sample_corr': sample_corr,
            'samples': samples,
            'vine': vine,
            'theta': theta_np
        }
        
    except Exception as e:
        print(f"Sampling failed: {e}")
        return {
            'fit_time': fit_time,
            'error': str(e),
            'vine': vine
        }


def fit_and_evaluate_tensorflow(data, vine_type='d-vine'):
    """Fit TensorFlow vine and evaluate performance"""
    n, d = data.shape
    
    print("\n--- TENSORFLOW PERFORMANCE ---")
    
    # Create margins
    margins_tf = []
    for i in range(d):
        margin = tf_margin_obj('norm', [0.0, 1.0], True)
        margin.ker = data[:, i]
        margins_tf.append(margin)
    
    # Create and fit vine
    vine = tf_vine_obj_bin(
        vine_family=vine_type,
        families=['gaussian', 'ind'],
        vine_depth=d,
        margin=margins_tf,
        knots=50,
        method='matrix'
    )
    
    gen_dict_tf = {"parallel": False, "param": True, "binning": False, 
                   "fitted": False, "vine_depth": d}
    par_dict_tf = {"param_families": ["gaussian", "ind"]}
    npc_dict_tf = {"opt_method": "local", "batch_paral": False}
    bin_dict_tf = {"n_bin": 1}
    
    # Time fitting
    start_time = time.time()
    vine.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
    fit_time = time.time() - start_time
    print(f"Fit time: {fit_time:.3f}s")
    
    # Print copula parameters
    print("\nFitted copula parameters:")
    if hasattr(vine, 'copulas'):
        for level, copulas in enumerate(vine.copulas):
            for i, cop in enumerate(copulas):
                if hasattr(cop, 'family'):
                    param = cop.param if hasattr(cop, 'param') else None
                    print(f"  Level {level}, Edge {i}: {cop.family}, param={param}")
    
    # Try to get theta values
    if hasattr(vine, 'theta'):
        theta_tf = vine.theta.numpy() if hasattr(vine.theta, 'numpy') else vine.theta
        print("\nTensorFlow theta shape:", theta_tf.shape)
    
    return {
        'fit_time': fit_time,
        'vine': vine
    }


def create_comparison_visualization(results_pt, results_tf, true_corr):
    """Create visualization comparing PyTorch and TensorFlow results"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Correlation matrices
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(true_corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Ground Truth Correlation', fontsize=14)
    
    if 'sample_corr' in results_pt:
        ax2 = plt.subplot(2, 3, 2)
        sns.heatmap(results_pt['sample_corr'], annot=True, fmt='.2f', cmap='coolwarm',
                    vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Correlation'})
        ax2.set_title('PyTorch Sample Correlation', fontsize=14)
        
        # Correlation error
        ax3 = plt.subplot(2, 3, 3)
        error = results_pt['sample_corr'] - true_corr
        sns.heatmap(error, annot=True, fmt='.3f', cmap='RdBu_r',
                    vmin=-0.5, vmax=0.5, ax=ax3, cbar_kws={'label': 'Error'})
        ax3.set_title('PyTorch Correlation Error', fontsize=14)
        
        # Calculate MAE
        mae = np.mean(np.abs(error))
        ax3.text(0.5, -0.1, f'MAE: {mae:.4f}', transform=ax3.transAxes,
                ha='center', fontsize=12)
    
    # 2. Sample distributions
    if 'samples' in results_pt:
        samples = results_pt['samples']
        d = samples.shape[1]
        
        for i in range(min(3, d)):
            ax = plt.subplot(2, 3, 4 + i)
            
            # Plot samples
            ax.hist(samples[:, i], bins=30, density=True, alpha=0.7,
                   label='PyTorch samples')
            
            # Overlay normal distribution
            x = np.linspace(-4, 4, 100)
            ax.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard normal')
            
            ax.set_xlabel(f'Variable {i+1}')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('PyTorch vs TensorFlow DVC Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('focused_performance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to focused_performance_comparison.png")


def analyze_theta_propagation(vine_pt):
    """Analyze theta propagation in detail"""
    print("\n" + "="*80)
    print("THETA PROPAGATION ANALYSIS")
    print("="*80)
    
    if not hasattr(vine_pt, 'theta'):
        print("No theta values available")
        return
    
    theta = vine_pt.theta.cpu().numpy()
    n, levels, d = theta.shape
    
    print(f"\nTheta shape: {theta.shape}")
    print("Sample of theta values (first 5 samples):")
    
    for sample in range(min(5, n)):
        print(f"\nSample {sample}:")
        for level in range(levels):
            non_zero = theta[sample, level] != 0
            if non_zero.any():
                print(f"  Level {level}: {theta[sample, level]}")


def main():
    """Run focused performance comparison"""
    print("="*80)
    print("FOCUSED DVC PERFORMANCE COMPARISON")
    print("="*80)
    
    # Test parameters
    dimensions = [3, 4, 5]
    rho_values = [0.3, 0.5, 0.7]
    
    for d in dimensions:
        for rho in rho_values:
            print(f"\n{'='*80}")
            print(f"TESTING: {d} dimensions, rho = {rho}")
            print(f"{'='*80}")
            
            # Generate data
            data, true_corr = generate_test_data(n=800, d=d, rho=rho)
            
            print(f"\nGenerated {data.shape[0]} samples with {d} dimensions")
            print("True correlation matrix:")
            print(true_corr)
            
            # Fit PyTorch
            results_pt = fit_and_evaluate_pytorch(data, 'd-vine')
            
            # Fit TensorFlow
            results_tf = fit_and_evaluate_tensorflow(data, 'd-vine')
            
            # Create visualization for first configuration
            if d == 4 and rho == 0.5:
                create_comparison_visualization(results_pt, results_tf, true_corr)
                
                # Detailed theta analysis
                if 'vine' in results_pt:
                    analyze_theta_propagation(results_pt['vine'])
            
            # Summary
            print(f"\n--- SUMMARY ---")
            print(f"PyTorch fit time: {results_pt.get('fit_time', 'N/A'):.3f}s")
            print(f"TensorFlow fit time: {results_tf.get('fit_time', 'N/A'):.3f}s")
            
            if 'sample_corr' in results_pt:
                mae = np.mean(np.abs(results_pt['sample_corr'] - true_corr))
                print(f"PyTorch correlation MAE: {mae:.4f}")


if __name__ == "__main__":
    main() 