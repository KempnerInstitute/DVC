"""
Simple Performance Comparison: PyTorch vs TensorFlow DVC

Tests the improved PyTorch implementation against TensorFlow
with focus on working configurations.
"""

import numpy as np
import torch
import sys
import time
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

import matplotlib.pyplot as plt

# PyTorch imports
from DVC_pyolder import vine_obj_bin, margin_obj
from DVC_pyolder.vine_model import fit_vine

# TensorFlow imports
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj


def generate_test_data(n=800, d=4, rho=0.6):
    """Generate test data with decreasing correlation structure"""
    np.random.seed(42)
    
    # Create correlation matrix
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    return data, corr


def test_parametric_performance():
    """Test parametric fitting performance"""
    print("="*80)
    print("PARAMETRIC PERFORMANCE COMPARISON")
    print("="*80)
    
    # Test different dimensions
    dimensions = [3, 4, 5]
    results = {'pytorch': [], 'tensorflow': []}
    
    for d in dimensions:
        print(f"\n--- Testing {d} dimensions ---")
        data, true_corr = generate_test_data(n=800, d=d)
        
        # Test PyTorch
        print("PyTorch D-vine (Parametric)...")
        try:
            vine_pt = vine_obj_bin(
                vine_family='d-vine',
                families=['gaussian', 'ind'],
                vine_depth=d,
                margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
                knots=50
            )
            
            gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
            par_dict = {"param_families": ["gaussian", "ind"]}
            npc_dict = {}
            bin_dict = {"n_bin": 1}
            
            start_time = time.time()
            fit_vine(vine_pt, data, gen_dict, npc_dict, par_dict, bin_dict)
            fit_time = time.time() - start_time
            
            # Sample and compute correlation
            samples = vine_pt.sample(500)
            sample_corr = np.corrcoef(samples.T)
            corr_mae = np.mean(np.abs(sample_corr - true_corr))
            
            results['pytorch'].append({
                'd': d,
                'fit_time': fit_time,
                'corr_mae': corr_mae,
                'success': True
            })
            
            print(f"  Success! Fit time: {fit_time:.3f}s, Correlation MAE: {corr_mae:.4f}")
            
        except Exception as e:
            results['pytorch'].append({
                'd': d,
                'success': False,
                'error': str(e)
            })
            print(f"  Failed: {str(e)}")
        
        # Test TensorFlow
        print("TensorFlow D-vine (Parametric)...")
        try:
            margins_tf = []
            for i in range(d):
                margin = tf_margin_obj('norm', [0.0, 1.0], True)
                margin.ker = data[:, i]
                margins_tf.append(margin)
            
            vine_tf = tf_vine_obj_bin(
                vine_family='d-vine',
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
            
            start_time = time.time()
            vine_tf.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
            fit_time = time.time() - start_time
            
            # Sample and compute correlation
            samples = vine_tf.sample(500)
            sample_corr = np.corrcoef(samples.T)
            corr_mae = np.mean(np.abs(sample_corr - true_corr))
            
            results['tensorflow'].append({
                'd': d,
                'fit_time': fit_time,
                'corr_mae': corr_mae,
                'success': True
            })
            
            print(f"  Success! Fit time: {fit_time:.3f}s, Correlation MAE: {corr_mae:.4f}")
            
        except Exception as e:
            results['tensorflow'].append({
                'd': d,
                'success': False,
                'error': str(e)
            })
            print(f"  Failed: {str(e)}")
    
    return results


def test_different_correlations():
    """Test with different correlation strengths"""
    print("\n" + "="*80)
    print("TESTING DIFFERENT CORRELATION STRENGTHS")
    print("="*80)
    
    rho_values = [0.3, 0.5, 0.7, 0.9]
    d = 4
    results = {'pytorch': [], 'tensorflow': []}
    
    for rho in rho_values:
        print(f"\n--- Testing rho = {rho} ---")
        data, true_corr = generate_test_data(n=800, d=d, rho=rho)
        
        # Test PyTorch
        try:
            vine_pt = vine_obj_bin(
                vine_family='d-vine',
                families=['gaussian'],
                vine_depth=d,
                margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
                knots=50
            )
            
            gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
            par_dict = {"param_families": ["gaussian"]}
            npc_dict = {}
            bin_dict = {"n_bin": 1}
            
            fit_vine(vine_pt, data, gen_dict, npc_dict, par_dict, bin_dict)
            
            samples = vine_pt.sample(500)
            sample_corr = np.corrcoef(samples.T)
            corr_mae = np.mean(np.abs(sample_corr - true_corr))
            
            results['pytorch'].append({
                'rho': rho,
                'corr_mae': corr_mae,
                'success': True
            })
            
            print(f"  PyTorch - Correlation MAE: {corr_mae:.4f}")
            
        except Exception as e:
            results['pytorch'].append({
                'rho': rho,
                'success': False
            })
            print(f"  PyTorch failed: {str(e)}")
        
        # Test TensorFlow
        try:
            margins_tf = []
            for i in range(d):
                margin = tf_margin_obj('norm', [0.0, 1.0], True)
                margin.ker = data[:, i]
                margins_tf.append(margin)
            
            vine_tf = tf_vine_obj_bin(
                vine_family='d-vine',
                families=['gaussian'],
                vine_depth=d,
                margin=margins_tf,
                knots=50,
                method='matrix'
            )
            
            gen_dict_tf = {"parallel": False, "param": True, "binning": False, 
                           "fitted": False, "vine_depth": d}
            par_dict_tf = {"param_families": ["gaussian"]}
            npc_dict_tf = {"opt_method": "local", "batch_paral": False}
            bin_dict_tf = {"n_bin": 1}
            
            vine_tf.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
            
            samples = vine_tf.sample(500)
            sample_corr = np.corrcoef(samples.T)
            corr_mae = np.mean(np.abs(sample_corr - true_corr))
            
            results['tensorflow'].append({
                'rho': rho,
                'corr_mae': corr_mae,
                'success': True
            })
            
            print(f"  TensorFlow - Correlation MAE: {corr_mae:.4f}")
            
        except Exception as e:
            results['tensorflow'].append({
                'rho': rho,
                'success': False
            })
            print(f"  TensorFlow failed: {str(e)}")
    
    return results


def visualize_results(param_results, corr_results):
    """Create visualization of results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Correlation MAE vs Dimensions
    ax1.set_title('Correlation Recovery vs Dimensions', fontsize=14)
    
    # Extract successful results
    pt_dims = [r['d'] for r in param_results['pytorch'] if r['success']]
    pt_mae = [r['corr_mae'] for r in param_results['pytorch'] if r['success']]
    
    tf_dims = [r['d'] for r in param_results['tensorflow'] if r['success']]
    tf_mae = [r['corr_mae'] for r in param_results['tensorflow'] if r['success']]
    
    if pt_dims:
        ax1.plot(pt_dims, pt_mae, 'bo-', label='PyTorch', linewidth=2, markersize=8)
    if tf_dims:
        ax1.plot(tf_dims, tf_mae, 'rs-', label='TensorFlow', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Dimensions', fontsize=12)
    ax1.set_ylabel('Correlation MAE', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation MAE vs Rho
    ax2.set_title('Correlation Recovery vs True Correlation', fontsize=14)
    
    pt_rhos = [r['rho'] for r in corr_results['pytorch'] if r['success']]
    pt_mae = [r['corr_mae'] for r in corr_results['pytorch'] if r['success']]
    
    tf_rhos = [r['rho'] for r in corr_results['tensorflow'] if r['success']]
    tf_mae = [r['corr_mae'] for r in corr_results['tensorflow'] if r['success']]
    
    if pt_rhos:
        ax2.plot(pt_rhos, pt_mae, 'bo-', label='PyTorch', linewidth=2, markersize=8)
    if tf_rhos:
        ax2.plot(tf_rhos, tf_mae, 'rs-', label='TensorFlow', linewidth=2, markersize=8)
    
    ax2.set_xlabel('True Correlation (ρ)', fontsize=12)
    ax2.set_ylabel('Correlation MAE', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_performance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to simple_performance_comparison.png")


def main():
    """Run simple performance comparison"""
    print("="*80)
    print("SIMPLE PERFORMANCE COMPARISON: PyTorch vs TensorFlow DVC")
    print("="*80)
    
    # Run tests
    param_results = test_parametric_performance()
    corr_results = test_different_correlations()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Calculate average performance
    pt_mae_vals = []
    tf_mae_vals = []
    
    for results in [param_results, corr_results]:
        pt_mae_vals.extend([r['corr_mae'] for r in results['pytorch'] if r['success']])
        tf_mae_vals.extend([r['corr_mae'] for r in results['tensorflow'] if r['success']])
    
    if pt_mae_vals:
        print(f"\nPyTorch average correlation MAE: {np.mean(pt_mae_vals):.4f}")
    else:
        print("\nPyTorch: No successful fits")
        
    if tf_mae_vals:
        print(f"TensorFlow average correlation MAE: {np.mean(tf_mae_vals):.4f}")
    else:
        print("TensorFlow: No successful fits")
    
    if pt_mae_vals and tf_mae_vals:
        improvement = (np.mean(tf_mae_vals) - np.mean(pt_mae_vals)) / np.mean(tf_mae_vals) * 100
        print(f"\nPyTorch is {'better' if improvement > 0 else 'worse'} by {abs(improvement):.1f}%")
    
    # Create visualization
    visualize_results(param_results, corr_results)
    
    return param_results, corr_results


if __name__ == "__main__":
    param_results, corr_results = main() 