"""
Test Theta Uniformity After kernel_cdf Fix

This script tests whether theta values maintain uniform distribution
across vine levels after applying the kernel_cdf transformation.
"""

import numpy as np
import torch
import sys
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


def test_theta_uniformity():
    """Test if theta values are uniformly distributed at all levels"""
    print("="*80)
    print("TESTING THETA UNIFORMITY")
    print("="*80)
    
    # Generate test data
    np.random.seed(42)
    n = 1000
    d = 4
    
    # Create correlated data
    rho = 0.6
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    print(f"\nTest data: {n} samples, {d} dimensions")
    print("True correlation matrix:")
    print(corr)
    
    # Test PyTorch
    print("\n--- PYTORCH PARAMETRIC ---")
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
    
    # Test uniformity at each level
    theta_pt = vine_pt.theta.cpu().numpy()
    
    print("\nKolmogorov-Smirnov test for uniformity (p-values):")
    print("Level | Var 0   | Var 1   | Var 2   | Var 3   |")
    print("-" * 50)
    
    uniform_pvalues = []
    
    for level in range(d):
        pvals = []
        for var in range(d):
            # Get non-zero values
            vals = theta_pt[:, level, var]
            if np.any(vals != 0):
                # Test against uniform distribution
                _, pval = stats.kstest(vals, 'uniform')
                pvals.append(pval)
            else:
                pvals.append(np.nan)
        
        uniform_pvalues.append(pvals)
        print(f"  {level}   | {pvals[0]:7.4f} | {pvals[1]:7.4f} | {pvals[2]:7.4f} | {pvals[3]:7.4f} |")
    
    # Create visualization
    fig, axes = plt.subplots(d, d, figsize=(15, 15))
    fig.suptitle('Theta Distribution at Each Level/Variable (PyTorch)', fontsize=16)
    
    for level in range(d):
        for var in range(d):
            ax = axes[level, var]
            vals = theta_pt[:, level, var]
            
            if np.any(vals != 0):
                # Plot histogram
                ax.hist(vals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
                ax.axhline(y=1.0, color='red', linestyle='--', label='Uniform')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 2])
                ax.set_title(f'Level {level}, Var {var}')
                
                # Add KS test p-value
                _, pval = stats.kstest(vals, 'uniform')
                ax.text(0.05, 1.8, f'p={pval:.3f}', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('theta_uniformity_pytorch.png', dpi=150)
    print("\nSaved visualization to theta_uniformity_pytorch.png")
    
    # Test TensorFlow
    print("\n--- TENSORFLOW PARAMETRIC ---")
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
    
    gen_dict_tf = {"parallel": False, "param": True, "binning": False, "fitted": False, "vine_depth": d}
    par_dict_tf = {"param_families": ["gaussian"]}
    npc_dict_tf = {"opt_method": "local", "batch_paral": False}
    bin_dict_tf = {"n_bin": 1}
    
    vine_tf.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
    
    # Test uniformity for TensorFlow
    theta_tf = vine_tf.theta.numpy() if hasattr(vine_tf.theta, 'numpy') else vine_tf.theta
    
    print("\nKolmogorov-Smirnov test for uniformity (p-values):")
    print("Level | Var 0   | Var 1   | Var 2   | Var 3   |")
    print("-" * 50)
    
    for level in range(d):
        pvals = []
        for var in range(d):
            vals = theta_tf[:, level, var]
            if np.any(vals != 0):
                _, pval = stats.kstest(vals, 'uniform')
                pvals.append(pval)
            else:
                pvals.append(np.nan)
        
        print(f"  {level}   | {pvals[0]:7.4f} | {pvals[1]:7.4f} | {pvals[2]:7.4f} | {pvals[3]:7.4f} |")
    
    # Compare distributions
    print("\n--- DISTRIBUTION COMPARISON ---")
    print("\nTheta mean values (should be ~0.5 for uniform):")
    print("PyTorch:")
    for level in range(d):
        means = []
        for var in range(d):
            vals = theta_pt[:, level, var]
            if np.any(vals != 0):
                means.append(np.mean(vals))
            else:
                means.append(np.nan)
        print(f"  Level {level}: {means}")
    
    print("\nTensorFlow:")
    for level in range(d):
        means = []
        for var in range(d):
            vals = theta_tf[:, level, var]
            if np.any(vals != 0):
                means.append(np.mean(vals))
            else:
                means.append(np.nan)
        print(f"  Level {level}: {means}")
    
    # Return summary statistics
    return {
        'pytorch_pvalues': uniform_pvalues,
        'pytorch_theta': theta_pt,
        'tensorflow_theta': theta_tf
    }


if __name__ == "__main__":
    results = test_theta_uniformity() 