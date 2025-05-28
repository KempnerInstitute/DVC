#!/usr/bin/env python3
"""
Comprehensive Test Script for TensorFlow-PyTorch Alignment Fixes
===============================================================

This script tests all the fixes applied to ensure they work correctly.
"""

import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from DVC_pyolder.vine_model import fit_vine
from DVC_pyolder.objects import vine_obj_bin, margin_obj
from DVC_pyolder.grid_ops import grid_obj
from DVC_pyolder.sampling import vine_copula_sample

def test_correlation_recovery():
    """Test if the fixes properly recover correlations"""
    print("Testing correlation recovery...")
    
    # Generate correlated data
    np.random.seed(42)
    n_samples = 500
    correlation = 0.7
    
    # Create correlated normal data
    mean = [0, 0, 0]
    cov = [[1, correlation, correlation*0.5],
           [correlation, 1, correlation*0.8], 
           [correlation*0.5, correlation*0.8, 1]]
    
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Transform to uniform margins
    from scipy.stats import norm
    data_uniform = norm.cdf(data)
    
    print(f"Original correlations:")
    orig_corr = np.corrcoef(data_uniform.T)
    print(orig_corr)
    
    # Create vine model
    margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(3)]
    vine = vine_obj_bin('c-vine', 'gaussian', 3, margins, 30, 'matrix')
    
    # Configuration for fitting
    gen_dict = {
        'binning': False,
        'parallel': False,
        'param': True,
        'fitted': False,
        'vine_depth': 2
    }
    
    npc_dict = {
        'npc_family': 'locallik',
        'grid_dim': 30
    }
    
    par_dict = {
        'param_families': ['ind', 'gaussian', 'clayton']
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit vine
    try:
        vine = fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        print("✓ Vine fitting successful")
        
        # Check theta matrices for NaN
        if hasattr(vine, 'theta'):
            nan_count = torch.isnan(vine.theta).sum().item()
            print(f"Theta NaN count: {nan_count}")
            if nan_count == 0:
                print("✓ No NaN values in theta matrix")
            else:
                print("✗ Found NaN values in theta matrix")
        
        # Test sampling
        samples, u_samples, _, _ = vine_copula_sample(vine, 1000)
        
        # Check sample correlations
        sample_corr = np.corrcoef(u_samples.T)
        print(f"Sample correlations:")
        print(sample_corr)
        
        # Compare correlations
        correlation_diff = np.abs(orig_corr - sample_corr)
        max_diff = np.max(correlation_diff[np.triu_indices_from(correlation_diff, k=1)])
        print(f"Maximum correlation difference: {max_diff:.4f}")
        
        if max_diff < 0.2:
            print("✓ Correlation recovery successful")
            return True
        else:
            print("✗ Correlation recovery failed")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernel_cdf_smoothing():
    """Test the kernel_cdf smoothing functionality"""
    print("\nTesting kernel_cdf smoothing...")
    
    try:
        from DVC_pyolder.utils_prob import kernel_cdf
        
        # Test data
        data = np.random.uniform(0, 1, 100)
        eval_points = np.linspace(0, 1, 50)
        grid_points = np.linspace(0, 1, 50)
        
        cdf_vals, smooth_cdf, pdf_vals = kernel_cdf(data, eval_points, grid_points)
        
        # Check properties
        if np.all(np.diff(smooth_cdf) >= -1e-10):  # Monotonic (allowing small numerical errors)
            print("✓ CDF is monotonic")
        else:
            print("✗ CDF is not monotonic")
            return False
            
        if np.all((smooth_cdf >= 0) & (smooth_cdf <= 1)):
            print("✓ CDF values in [0,1]")
        else:
            print("✗ CDF values outside [0,1]")
            return False
            
        print("✓ kernel_cdf smoothing test passed")
        return True
        
    except Exception as e:
        print(f"✗ kernel_cdf test failed: {e}")
        return False

def test_parametric_copulas():
    """Test parametric copula fitting and AIC calculation"""
    print("\nTesting parametric copulas...")
    
    try:
        from DVC_pyolder.param_copula import parametric_fit
        
        # Generate data with known correlation
        np.random.seed(42)
        n = 300
        rho = 0.6
        
        # Gaussian copula data
        z1 = np.random.normal(0, 1, n)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)
        
        # Transform to uniform
        from scipy.stats import norm
        u1 = norm.cdf(z1)
        u2 = norm.cdf(z2)
        
        data = np.stack([u1, u2], axis=1)
        data = np.expand_dims(data, axis=2)  # Add copula dimension
        
        # Fit parametric copulas
        families = ['ind', 'gaussian', 'clayton']
        aic_matrix, theta_list, logp_list = parametric_fit(data, families, 1)
        
        print(f"AIC values: {aic_matrix[0]}")
        print(f"Best family: {families[np.argmin(aic_matrix[0])]}")
        
        # Gaussian should win for this data
        best_idx = np.argmin(aic_matrix[0])
        if families[best_idx] == 'gaussian':
            print("✓ Gaussian copula correctly selected")
            
            # Check parameter value
            estimated_rho = theta_list[0][best_idx]
            if abs(estimated_rho - rho) < 0.1:
                print(f"✓ Parameter estimation good: {estimated_rho:.3f} vs {rho:.3f}")
                return True
            else:
                print(f"✗ Parameter estimation poor: {estimated_rho:.3f} vs {rho:.3f}")
                return False
        else:
            print(f"✗ Wrong copula selected: {families[best_idx]}")
            return False
            
    except Exception as e:
        print(f"✗ Parametric copula test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running comprehensive TensorFlow-PyTorch alignment tests...")
    
    tests = [
        test_kernel_cdf_smoothing,
        test_parametric_copulas,
        test_correlation_recovery
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! TensorFlow-PyTorch alignment successful.")
        return True
    else:
        print("✗ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
