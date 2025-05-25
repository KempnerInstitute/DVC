"""
Test theta propagation in PyTorch vs TensorFlow

This script checks if theta values are being propagated correctly
through the vine levels.
"""

import numpy as np
import torch
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

# PyTorch imports
from DVC import vine_obj_bin, margin_obj
from DVC.vine_model import fit_vine

# TensorFlow imports
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj


def test_theta_propagation():
    """Test theta propagation through vine levels"""
    
    # Generate simple test data
    np.random.seed(42)
    n = 500
    d = 4
    
    # Create correlated data
    rho = 0.6
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    print("Test data correlation matrix:")
    print(np.corrcoef(data.T))
    
    # Test PyTorch
    print("\n=== PYTORCH THETA PROPAGATION ===")
    
    vine_pt = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian', 'ind'],
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian"]}  # Only Gaussian for simplicity
    npc_dict = {}
    bin_dict = {"n_bin": 1}
    
    fit_vine(vine_pt, data, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Check theta values at each level
    print("\nPyTorch theta values (first 5 samples):")
    for level in range(d):
        print(f"\nLevel {level}:")
        for var in range(d):
            vals = vine_pt.theta[:5, level, var].cpu().numpy()
            print(f"  Var {var}: {vals}")
    
    # Check copula parameters
    print("\nPyTorch copula parameters:")
    for level, copulas in enumerate(vine_pt.copulas):
        print(f"Level {level}:")
        for i, cop in enumerate(copulas):
            print(f"  Edge {i}: family={cop.family}, theta={cop.theta}")
    
    # Test TensorFlow
    print("\n\n=== TENSORFLOW THETA PROPAGATION ===")
    
    # Create margins
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
    
    gen_dict_tf = {"parallel": False, "param": True, "binning": False, "fitted": False, "vine_depth": d}
    par_dict_tf = {"param_families": ["gaussian"]}
    npc_dict_tf = {"opt_method": "local", "batch_paral": False}
    bin_dict_tf = {"n_bin": 1}
    
    vine_tf.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
    
    # Check theta values at each level
    print("\nTensorFlow theta values (first 5 samples):")
    theta_tf = vine_tf.theta.numpy() if hasattr(vine_tf.theta, 'numpy') else vine_tf.theta
    for level in range(d):
        print(f"\nLevel {level}:")
        for var in range(d):
            vals = theta_tf[:5, level, var]
            print(f"  Var {var}: {vals}")
    
    # Check copula parameters
    print("\nTensorFlow copula parameters:")
    for level, copulas in enumerate(vine_tf.copulas):
        print(f"Level {level}:")
        for i, cop in enumerate(copulas):
            if hasattr(cop, 'family'):
                print(f"  Edge {i}: family={cop.family}, theta={cop.param}")
            else:
                print(f"  Edge {i}: {cop}")
    
    # Compare theta differences
    print("\n\n=== THETA DIFFERENCES ===")
    print("Average absolute difference in theta values by level:")
    for level in range(min(vine_pt.theta.shape[1], theta_tf.shape[1])):
        theta_pt_level = vine_pt.theta[:, level, :].cpu().numpy()
        theta_tf_level = theta_tf[:, level, :]
        
        # Only compare valid entries
        mask = ~(np.isnan(theta_pt_level) | np.isnan(theta_tf_level))
        if mask.any():
            diff = np.abs(theta_pt_level[mask] - theta_tf_level[mask]).mean()
            print(f"Level {level}: {diff:.6f}")
    
    # Test h-function behavior
    print("\n\n=== H-FUNCTION TEST ===")
    print("Testing h-function with known values...")
    
    # Test with uniform values
    u1 = torch.tensor([0.3, 0.5, 0.7])
    u2 = torch.tensor([0.4, 0.6, 0.8])
    
    # Test Gaussian copula with rho=0.5
    from DVC.objects import cop_par_obj
    cop = cop_par_obj('gaussian', 0.5)
    
    from DVC.vine_model import _h_function
    h_left = _h_function(u1, u2, cop, vine_pt.grid_u, side="left")
    h_right = _h_function(u2, u1, cop, vine_pt.grid_u, side="right")
    
    print(f"u1: {u1.numpy()}")
    print(f"u2: {u2.numpy()}")
    print(f"h(u2|u1): {h_left.numpy()}")
    print(f"h(u1|u2): {h_right.numpy()}")
    
    # The h-functions should be different for non-independent copulas
    print(f"\nDifference between left and right h-functions: {(h_left - h_right).abs().mean():.6f}")


if __name__ == "__main__":
    test_theta_propagation() 