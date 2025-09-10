#!/usr/bin/env python3
"""
Lightweight R-vine Optimization Comparison

This script demonstrates the R-vine optimization algorithm in your DVC_tensorflow codebase
by comparing optimal vs random R-vine structures on a manageable scale.

Key Focus:
- Shows the value of optimal R-vine selection using Prim's MST algorithm
- Compares performance metrics between different vine structures
- Lightweight enough to run on HPC systems with resource limits

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add DVC_tensorflow to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dvc_tensorflow_dir = os.path.join(project_root, 'src', 'DVC_tensorflow')
sys.path.append(dvc_tensorflow_dir)

from classes.objects import *
from vine_tree.tree_op import *
from param.generate_rvine import *
from pre_proc.preparation import prep_cop
from sampling.vine_sample import *
from scipy.stats import multivariate_normal

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Results directory
results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)

def generate_test_data(dim=4, n_samples=1200):
    """Generate simple multivariate Gaussian test data"""
    print(f"Generating {dim}D test data with {n_samples} samples...")
    
    # Create a simple but interesting correlation structure
    corr_matrix = np.eye(dim)
    
    # Add some strong correlations
    for i in range(dim-1):
        corr_matrix[i, i+1] = 0.7  # Sequential correlations
        corr_matrix[i+1, i] = 0.7
    
    # Add one cross correlation
    if dim >= 4:
        corr_matrix[0, 3] = -0.5
        corr_matrix[3, 0] = -0.5
    
    # Generate data
    mean = np.zeros(dim)
    data = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples)
    
    print(f"Data shape: {data.shape}")
    print(f"Correlation matrix has {np.sum(np.abs(corr_matrix) > 0.3) - dim} significant correlations")
    
    return data, corr_matrix

def fit_and_analyze_vine(data, vine_type, method, method_name):
    """Fit vine and calculate basic metrics"""
    print(f"\n--- Fitting {method_name} ---")
    dim = data.shape[1]
    
    # Setup margins
    margin_vine = []
    for i in range(dim):
        mar_p = margin_obj('norm', [0, 1], True)
        margin_vine.append(mar_p)
    
    # Create vine object
    if vine_type == 'r-vine':
        r_matrix = None  # Will be generated
    else:  # c-vine
        r_matrix, _, _, _ = prepare_vine(vine_type, dim)
    
    vine = vine_obj_bin(vine_type, "kercop", dim, margin_vine, 30, method, r_matrix)
    
    # Prepare data
    x = data.astype(np.float32)
    exc = tf.math.floormod(tf.shape(x)[0], 5)
    x = x[:tf.shape(x)[0]-exc, :]
    
    # Transform and fit
    e = prep_cop(x, vine, 'rand')
    
    gen_dict = {'parallel': True, 'binning': False, 'param': False, 'vine_depth': dim, 'fitted': False}
    par_dict = {'param_families': ["ind", "gaussian"]}
    npc_dict = {'opt_method': 'LL1', 'batch_paral': 2}
    bin_dict = {'n_bin': 3}
    
    start_time = time.time()
    vine.fit(x, gen_dict, npc_dict, par_dict, bin_dict)
    fit_time = time.time() - start_time
    
    # Generate samples and evaluate
    try:
        vine_samples, _, _, _ = vine_copula_sample(vine, 800)
        vine_corr = np.corrcoef(vine_samples.T)
        empirical_corr = np.corrcoef(data.T)
        
        corr_error = np.mean(np.abs(vine_corr - empirical_corr))
        
        print(f"✓ {method_name} completed in {fit_time:.1f}s")
        print(f"  Correlation MAE: {corr_error:.4f}")
        
        return {
            'vine': vine,
            'fit_time': fit_time,
            'vine_samples': vine_samples,
            'vine_correlation': vine_corr,
            'correlation_error': corr_error,
            'r_matrix': vine.r_matrix if hasattr(vine, 'r_matrix') else None
        }
        
    except Exception as e:
        print(f"✗ Error in {method_name}: {e}")
        return {'error': str(e), 'fit_time': fit_time}

def create_comparison_plot(results):
    """Create simple comparison visualization"""
    print("Creating comparison plots...")
    
    # Extract valid results
    methods = []
    fit_times = []
    corr_errors = []
    
    for method_name, result in results.items():
        if 'error' not in result:
            methods.append(method_name)
            fit_times.append(result['fit_time'])
            corr_errors.append(result['correlation_error'])
    
    if len(methods) < 2:
        print("Not enough valid results for comparison")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Fitting time comparison
    bars1 = ax1.bar(methods, fit_times, color=['blue', 'orange', 'green'][:len(methods)])
    ax1.set_title('Fitting Time Comparison', fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, fit_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Correlation error comparison
    bars2 = ax2.bar(methods, corr_errors, color=['blue', 'orange', 'green'][:len(methods)])
    ax2.set_title('Correlation Preservation Error\n(Lower = Better)', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, error_val in zip(bars2, corr_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{error_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lightweight_rvine_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Comparison plot saved")

def print_summary(results, data, true_corr):
    """Print analysis summary"""
    print("\n" + "="*60)
    print("R-VINE OPTIMIZATION COMPARISON SUMMARY")
    print("="*60)
    
    # Data info
    print(f"Dataset: {data.shape[0]} samples, {data.shape[1]} variables")
    
    # Results table
    print(f"\n{'Method':<15} {'Fit Time':<10} {'Corr Error':<12} {'Status'}")
    print("-" * 50)
    
    for method_name, result in results.items():
        if 'error' in result:
            print(f"{method_name:<15} {result['fit_time']:<10.1f} {'Failed':<12} Error")
        else:
            fit_time = result['fit_time']
            corr_err = result['correlation_error']
            status = "Good" if corr_err < 0.1 else "Fair"
            print(f"{method_name:<15} {fit_time:<10.1f} {corr_err:<12.4f} {status}")
    
    # Performance comparison
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if len(valid_results) >= 2:
        print(f"\nKEY FINDINGS:")
        
        if 'Optimal R-vine' in valid_results and 'Random R-vine' in valid_results:
            opt_error = valid_results['Optimal R-vine']['correlation_error']
            rand_error = valid_results['Random R-vine']['correlation_error']
            improvement = ((rand_error - opt_error) / rand_error) * 100
            print(f"• Optimal R-vine improves correlation preservation by {improvement:.1f}%")
        
        if 'Optimal R-vine' in valid_results and 'C-vine' in valid_results:
            opt_error = valid_results['Optimal R-vine']['correlation_error']
            c_error = valid_results['C-vine']['correlation_error']
            improvement = ((c_error - opt_error) / c_error) * 100
            print(f"• Optimal R-vine improves over C-vine by {improvement:.1f}%")
        
        best_method = min(valid_results.keys(), key=lambda k: valid_results[k]['correlation_error'])
        print(f"• Best performing method: {best_method}")
    
    print("\nR-vine Optimization Algorithm Details:")
    print("• Implementation: vine_tree/tree_op.py -> optimal_tree()")
    print("• Method: Prim's minimum spanning tree algorithm")
    print("• Criterion: Maximize |Kendall's tau| correlations")
    print("• Advantage: Data-driven structure selection")
    print("="*60)

def main():
    """Main comparison function"""
    print("="*60)
    print("LIGHTWEIGHT R-VINE OPTIMIZATION COMPARISON")
    print("="*60)
    print("This script demonstrates the R-vine optimization algorithm")
    print("already implemented in your DVC_tensorflow codebase.")
    print()
    print("Comparing:")
    print("1. C-vine (baseline canonical structure)")
    print("2. Random R-vine (random valid structure)")
    print("3. Optimal R-vine (data-driven optimization)")
    print("="*60)
    
    # Generate test data (smaller scale)
    dim = 4           # Reduced dimension
    n_samples = 1200  # Reduced sample size
    
    data, true_corr = generate_test_data(dim, n_samples)
    
    # Test different vine methods
    results = {}
    
    # Test in order of complexity
    test_methods = [
        ('c-vine', 'matrix', 'C-vine'),
        ('r-vine', 'random', 'Random R-vine'),
        ('r-vine', 'optimal', 'Optimal R-vine')
    ]
    
    for vine_type, method, method_name in test_methods:
        try:
            result = fit_and_analyze_vine(data, vine_type, method, method_name)
            results[method_name] = result
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results[method_name] = {'error': str(e), 'fit_time': 0}
    
    # Create visualizations and summary
    create_comparison_plot(results)
    print_summary(results, data, true_corr)
    
    # Save detailed results
    import json
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': data.shape,
        'methods': {}
    }
    
    for method_name, result in results.items():
        if 'error' not in result:
            json_results['methods'][method_name] = {
                'fit_time': float(result['fit_time']),
                'correlation_error': float(result['correlation_error'])
            }
    
    with open(os.path.join(results_dir, 'lightweight_rvine_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Analysis completed! Results saved in {results_dir}")
    print("Files created:")
    print("- lightweight_rvine_comparison.png")
    print("- lightweight_rvine_results.json")

if __name__ == "__main__":
    main() 