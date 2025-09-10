"""
Comprehensive Performance Test: PyTorch vs TensorFlow DVC

This script tests the improved PyTorch implementation against TensorFlow
across different vine types (C-vine, D-vine, R-vine) and fitting approaches
(parametric and non-parametric).
"""

import numpy as np
import torch
import sys
import time
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

from scipy import stats
import matplotlib.pyplot as plt

# PyTorch imports
from DVC_pyolder import vine_obj_bin, margin_obj
from DVC_pyolder.vine_model import fit_vine

# TensorFlow imports
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj
import tensorflow as tf


def generate_test_data(n=1000, d=5, correlation_type='decreasing'):
    """Generate test data with different correlation structures"""
    np.random.seed(42)
    
    if correlation_type == 'decreasing':
        # Decreasing correlation with distance
        rho = 0.7
        corr = np.eye(d)
        for i in range(d):
            for j in range(i+1, d):
                corr[i, j] = corr[j, i] = rho ** abs(i-j)
                
    elif correlation_type == 'block':
        # Block correlation structure
        corr = np.eye(d)
        # First block
        for i in range(min(3, d)):
            for j in range(i+1, min(3, d)):
                corr[i, j] = corr[j, i] = 0.8
        # Second block
        for i in range(3, d):
            for j in range(i+1, d):
                corr[i, j] = corr[j, i] = 0.6
                
    elif correlation_type == 'hub':
        # Hub structure - first variable correlated with all others
        corr = np.eye(d)
        for i in range(1, d):
            corr[0, i] = corr[i, 0] = 0.7
            
    else:  # uniform
        # Uniform correlation
        corr = np.full((d, d), 0.5)
        np.fill_diagonal(corr, 1.0)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    return data, corr


def test_vine_performance(vine_type, data, true_corr, param=True, verbose=True):
    """Test performance of a specific vine configuration"""
    n, d = data.shape
    results = {}
    
    # Test PyTorch
    if verbose:
        print(f"\n--- PyTorch {vine_type} ({'Parametric' if param else 'Non-parametric'}) ---")
    
    try:
        # Create vine
        vine_pt = vine_obj_bin(
            vine_family=vine_type,
            families=['gaussian', 'clayton', 'frank'] if param else [],
            vine_depth=d,
            margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
            knots=50,
            method='matrix' if vine_type == 'r-vine' else None
        )
        
        # Configuration
        gen_dict = {"parallel": False, "param": param, "binning": False, "fitted": False}
        par_dict = {"param_families": ["gaussian", "clayton", "frank"]} if param else {}
        npc_dict = {"method": "local", "n_iter": 50 if param else 100}
        bin_dict = {"n_bin": 1}
        
        # Fit vine
        start_time = time.time()
        fit_vine(vine_pt, data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        # Sample from vine
        start_time = time.time()
        samples = vine_pt.sample(500)
        sample_time = time.time() - start_time
        
        # Calculate correlations
        sample_corr = np.corrcoef(samples.T)
        corr_mae = np.mean(np.abs(sample_corr - true_corr))
        
        results['pytorch'] = {
            'fit_time': fit_time,
            'sample_time': sample_time,
            'corr_mae': corr_mae,
            'sample_corr': sample_corr,
            'success': True
        }
        
        if verbose:
            print(f"  Fit time: {fit_time:.3f}s")
            print(f"  Sample time: {sample_time:.3f}s")
            print(f"  Correlation MAE: {corr_mae:.4f}")
            
    except Exception as e:
        results['pytorch'] = {
            'success': False,
            'error': str(e)
        }
        if verbose:
            print(f"  Failed: {str(e)}")
    
    # Test TensorFlow
    if verbose:
        print(f"\n--- TensorFlow {vine_type} ({'Parametric' if param else 'Non-parametric'}) ---")
    
    try:
        # Create margins
        margins_tf = []
        for i in range(d):
            margin = tf_margin_obj('norm', [0.0, 1.0], True)
            margin.ker = data[:, i]
            margins_tf.append(margin)
        
        # Create vine
        vine_tf = tf_vine_obj_bin(
            vine_family=vine_type,
            families=['gaussian', 'clayton', 'frank'] if param else [],
            vine_depth=d,
            margin=margins_tf,
            knots=50,
            method='matrix'
        )
        
        # Configuration
        gen_dict_tf = {"parallel": False, "param": param, "binning": False, 
                       "fitted": False, "vine_depth": d}
        par_dict_tf = {"param_families": ["gaussian", "clayton", "frank"]} if param else {}
        npc_dict_tf = {"opt_method": "local", "batch_paral": False}
        bin_dict_tf = {"n_bin": 1}
        
        # Fit vine
        start_time = time.time()
        vine_tf.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
        fit_time = time.time() - start_time
        
        # Sample from vine
        start_time = time.time()
        samples = vine_tf.sample(500)
        sample_time = time.time() - start_time
        
        # Calculate correlations
        sample_corr = np.corrcoef(samples.T)
        corr_mae = np.mean(np.abs(sample_corr - true_corr))
        
        results['tensorflow'] = {
            'fit_time': fit_time,
            'sample_time': sample_time,
            'corr_mae': corr_mae,
            'sample_corr': sample_corr,
            'success': True
        }
        
        if verbose:
            print(f"  Fit time: {fit_time:.3f}s")
            print(f"  Sample time: {sample_time:.3f}s")
            print(f"  Correlation MAE: {corr_mae:.4f}")
            
    except Exception as e:
        results['tensorflow'] = {
            'success': False,
            'error': str(e)
        }
        if verbose:
            print(f"  Failed: {str(e)}")
    
    return results


def visualize_results(all_results):
    """Create visualization of performance comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PyTorch vs TensorFlow DVC Performance Comparison', fontsize=16)
    
    # Prepare data for plotting
    vine_types = []
    pt_mae_param = []
    tf_mae_param = []
    pt_mae_nonparam = []
    tf_mae_nonparam = []
    
    for config, results in all_results.items():
        vine_type = config.split('_')[0]
        param = 'param' in config
        
        if vine_type not in vine_types:
            vine_types.append(vine_type)
        
        if results['pytorch']['success']:
            mae = results['pytorch']['corr_mae']
            if param:
                pt_mae_param.append(mae)
            else:
                pt_mae_nonparam.append(mae)
        else:
            if param:
                pt_mae_param.append(np.nan)
            else:
                pt_mae_nonparam.append(np.nan)
                
        if results['tensorflow']['success']:
            mae = results['tensorflow']['corr_mae']
            if param:
                tf_mae_param.append(mae)
            else:
                tf_mae_nonparam.append(mae)
        else:
            if param:
                tf_mae_param.append(np.nan)
            else:
                tf_mae_nonparam.append(np.nan)
    
    # Plot 1: Correlation MAE comparison (parametric)
    ax = axes[0, 0]
    x = np.arange(len(vine_types))
    width = 0.35
    ax.bar(x - width/2, pt_mae_param[:len(vine_types)], width, label='PyTorch', alpha=0.8)
    ax.bar(x + width/2, tf_mae_param[:len(vine_types)], width, label='TensorFlow', alpha=0.8)
    ax.set_xlabel('Vine Type')
    ax.set_ylabel('Correlation MAE')
    ax.set_title('Parametric Fitting - Correlation Error')
    ax.set_xticks(x)
    ax.set_xticklabels(vine_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Correlation MAE comparison (non-parametric)
    ax = axes[0, 1]
    ax.bar(x - width/2, pt_mae_nonparam[:len(vine_types)], width, label='PyTorch', alpha=0.8)
    ax.bar(x + width/2, tf_mae_nonparam[:len(vine_types)], width, label='TensorFlow', alpha=0.8)
    ax.set_xlabel('Vine Type')
    ax.set_ylabel('Correlation MAE')
    ax.set_title('Non-parametric Fitting - Correlation Error')
    ax.set_xticks(x)
    ax.set_xticklabels(vine_types)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fitting time comparison
    ax = axes[1, 0]
    pt_times = []
    tf_times = []
    labels = []
    
    for config, results in all_results.items():
        if results['pytorch']['success']:
            pt_times.append(results['pytorch']['fit_time'])
        else:
            pt_times.append(0)
            
        if results['tensorflow']['success']:
            tf_times.append(results['tensorflow']['fit_time'])
        else:
            tf_times.append(0)
            
        labels.append(config.replace('_', '\n'))
    
    x = np.arange(len(labels))
    ax.bar(x - width/2, pt_times, width, label='PyTorch', alpha=0.8)
    ax.bar(x + width/2, tf_times, width, label='TensorFlow', alpha=0.8)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Fitting Time (s)')
    ax.set_title('Fitting Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Success rate
    ax = axes[1, 1]
    methods = ['PyTorch', 'TensorFlow']
    param_success = [0, 0]
    nonparam_success = [0, 0]
    total = 0
    
    for config, results in all_results.items():
        total += 0.5  # Each config counts as 0.5 to normalize
        if 'param' in config:
            if results['pytorch']['success']:
                param_success[0] += 1
            if results['tensorflow']['success']:
                param_success[1] += 1
        else:
            if results['pytorch']['success']:
                nonparam_success[0] += 1
            if results['tensorflow']['success']:
                nonparam_success[1] += 1
    
    x = np.arange(len(methods))
    ax.bar(x - width/2, param_success, width, label='Parametric', alpha=0.8)
    ax.bar(x + width/2, nonparam_success, width, label='Non-parametric', alpha=0.8)
    ax.set_xlabel('Method')
    ax.set_ylabel('Successful Fits')
    ax.set_title('Success Rate by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_performance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to comprehensive_performance_comparison.png")


def main():
    """Run comprehensive performance tests"""
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE TEST: PyTorch vs TensorFlow DVC")
    print("="*80)
    
    # Test configurations
    vine_types = ['c-vine', 'd-vine', 'r-vine']
    data_configs = [
        {'n': 800, 'd': 5, 'correlation_type': 'decreasing'},
        {'n': 800, 'd': 5, 'correlation_type': 'block'},
        {'n': 800, 'd': 5, 'correlation_type': 'hub'}
    ]
    
    all_results = {}
    
    # Test each vine type with parametric fitting
    print("\n" + "="*80)
    print("PARAMETRIC FITTING TESTS")
    print("="*80)
    
    for vine_type in vine_types:
        # Use decreasing correlation for main test
        data, true_corr = generate_test_data(**data_configs[0])
        
        print(f"\n{'='*50}")
        print(f"Testing {vine_type.upper()}")
        print(f"{'='*50}")
        
        results = test_vine_performance(vine_type, data, true_corr, param=True)
        all_results[f'{vine_type}_param'] = results
    
    # Test non-parametric fitting (only for smaller dimensions due to computational cost)
    print("\n" + "="*80)
    print("NON-PARAMETRIC FITTING TESTS")
    print("="*80)
    
    # Use smaller data for non-parametric
    data_small, true_corr_small = generate_test_data(n=300, d=3, correlation_type='decreasing')
    
    for vine_type in ['c-vine', 'd-vine']:  # Skip r-vine for non-parametric due to complexity
        print(f"\n{'='*50}")
        print(f"Testing {vine_type.upper()} (Non-parametric)")
        print(f"{'='*50}")
        
        results = test_vine_performance(vine_type, data_small, true_corr_small, param=False)
        all_results[f'{vine_type}_nonparam'] = results
    
    # Test different correlation structures with D-vine
    print("\n" + "="*80)
    print("TESTING DIFFERENT CORRELATION STRUCTURES (D-VINE)")
    print("="*80)
    
    for i, corr_type in enumerate(['block', 'hub']):
        data, true_corr = generate_test_data(
            n=data_configs[i+1]['n'], 
            d=data_configs[i+1]['d'], 
            correlation_type=corr_type
        )
        
        print(f"\n{'='*50}")
        print(f"Testing D-vine with {corr_type} correlation")
        print(f"{'='*50}")
        
        results = test_vine_performance('d-vine', data, true_corr, param=True)
        all_results[f'd-vine_{corr_type}'] = results
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nSuccessful fits:")
    for config, results in all_results.items():
        pt_success = "✓" if results['pytorch']['success'] else "✗"
        tf_success = "✓" if results['tensorflow']['success'] else "✗"
        print(f"  {config:<20} PyTorch: {pt_success}  TensorFlow: {tf_success}")
    
    print("\nCorrelation MAE (lower is better):")
    for config, results in all_results.items():
        if results['pytorch']['success'] and results['tensorflow']['success']:
            pt_mae = results['pytorch']['corr_mae']
            tf_mae = results['tensorflow']['corr_mae']
            improvement = (tf_mae - pt_mae) / tf_mae * 100 if tf_mae > 0 else 0
            print(f"  {config:<20} PyTorch: {pt_mae:.4f}  TensorFlow: {tf_mae:.4f}  "
                  f"(PyTorch {'better' if improvement > 0 else 'worse'} by {abs(improvement):.1f}%)")
    
    # Create visualization
    visualize_results(all_results)
    
    return all_results


if __name__ == "__main__":
    results = main() 