#!/usr/bin/env python3
"""
Debug script to identify and fix NaN issues in PyTorch vine implementation.
"""

import sys, os, numpy as np, torch
from scipy.stats import multivariate_normal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_parametric():
    """Test basic parametric fitting to identify NaN sources."""
    print("=== DEBUGGING PYTORCH PARAMETRIC FITTING ===")
    
    # Simple 3D test case
    np.random.seed(42)
    true_corr = np.array([
        [1.00, 0.70, 0.50],
        [0.70, 1.00, 0.60],
        [0.50, 0.60, 1.00]
    ])
    data = multivariate_normal.rvs(mean=np.zeros(3), cov=true_corr, size=200)
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Data contains NaN: {np.isnan(data).any()}")
    
    # Test PyTorch fitting
    try:
        from DVC_pyolder.objects import vine_obj_bin, margin_obj
        
        # Create margins
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(3)]
        vine = vine_obj_bin('c-vine', ['gaussian'], 3, margins, 25)
        
        # Fit with debug info
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        par_dict = {'param_families': ['gaussian']}
        npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
        bin_dict = {'n_bin': 5}
        
        print("Starting fit...")
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        
        print("Fit completed successfully!")
        
        # Check theta matrices
        print(f"Theta shape: {vine.theta.shape}")
        print(f"Theta contains NaN: {torch.isnan(vine.theta).any()}")
        print(f"Theta_flip contains NaN: {torch.isnan(vine.theta_flip).any()}")
        
        # Test sampling
        print("Testing sampling...")
        samples = vine.sample(100)
        print(f"Samples shape: {samples.shape}")
        print(f"Samples contains NaN: {np.isnan(samples).any()}")
        
        if not np.isnan(samples).any():
            pred_corr = np.corrcoef(samples, rowvar=False)
            print(f"Predicted correlation matrix:\n{pred_corr}")
            
            # Calculate metrics
            mask = np.triu(np.ones_like(true_corr, dtype=bool), k=1)
            true_vals = true_corr[mask]
            pred_vals = pred_corr[mask]
            mae = np.mean(np.abs(true_vals - pred_vals))
            recovery = np.corrcoef(true_vals, pred_vals)[0, 1]
            
            print(f"MAE: {mae:.3f}")
            print(f"Recovery: {recovery:.3f}")
            print("SUCCESS: 3D parametric test passed!")
            return True
        else:
            print("FAILED: Samples contain NaN")
            return False
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_theta_flip_logic():
    """Test theta/theta_flip logic in isolation."""
    print("\n=== DEBUGGING THETA/THETA_FLIP LOGIC ===")
    
    # Create test data
    n_samples = 100
    u1 = torch.rand(n_samples).clamp(1e-6, 1-1e-6)
    u2 = torch.rand(n_samples).clamp(1e-6, 1-1e-6)
    
    print(f"u1 range: [{u1.min():.6f}, {u1.max():.6f}]")
    print(f"u2 range: [{u2.min():.6f}, {u2.max():.6f}]")
    print(f"u1 contains NaN: {torch.isnan(u1).any()}")
    print(f"u2 contains NaN: {torch.isnan(u2).any()}")
    
    # Test parametric copula h-function
    try:
        from DVC_pyolder.objects import cop_par_obj
        from DVC_pyolder.param_copula import copulaccdf
        
        # Test Gaussian copula
        cop = cop_par_obj("gaussian", 0.5)
        uv_data = torch.stack([u1, u2], dim=1)
        print(f"uv_data shape: {uv_data.shape}")
        print(f"uv_data range: [{uv_data.min():.6f}, {uv_data.max():.6f}]")
        
        h_val = copulaccdf(cop, uv_data)
        print(f"h_val shape: {h_val.shape}")
        print(f"h_val range: [{h_val.min():.6f}, {h_val.max():.6f}]")
        print(f"h_val contains NaN: {torch.isnan(h_val).any()}")
        print(f"h_val contains Inf: {torch.isinf(h_val).any()}")
        
        if torch.isnan(h_val).any():
            print("FAILED: h-function produces NaN")
            nan_mask = torch.isnan(h_val)
            print(f"NaN count: {nan_mask.sum()}")
            print(f"First few NaN indices: {torch.where(nan_mask)[0][:5]}")
            return False
        else:
            print("SUCCESS: h-function produces valid values")
            
            # Test kernel smoothing
            try:
                from DVC_pyolder.utils_prob import kernel_cdf
                h_np = h_val.cpu().numpy()
                ex_u_np = np.linspace(0, 1, 50)
                h_smoothed, _, _ = kernel_cdf(h_np, h_np, ex_u_np)
                print(f"h_smoothed shape: {h_smoothed.shape}")
                print(f"h_smoothed contains NaN: {np.isnan(h_smoothed).any()}")
                
                if np.isnan(h_smoothed).any():
                    print("FAILED: kernel_cdf produces NaN")
                    return False
                else:
                    print("SUCCESS: kernel_cdf produces valid values")
                    return True
                    
            except Exception as e:
                print(f"FAILED: kernel_cdf error: {e}")
                return False
                
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_margin_cdf():
    """Test margin CDF computation."""
    print("\n=== DEBUGGING MARGIN CDF ===")
    
    # Test data
    np.random.seed(42)
    data = np.random.normal(0, 1, 200)
    
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    # Convert to PyTorch
    x_torch = torch.tensor(data, dtype=torch.float32)
    
    # Test ranking to uniform
    u_vals = torch.sort(x_torch)[0]
    ranks = torch.searchsorted(u_vals, x_torch).float() + 1
    u_margin = ranks / (len(data) + 1)
    
    print(f"u_margin range: [{u_margin.min():.6f}, {u_margin.max():.6f}]")
    print(f"u_margin contains NaN: {torch.isnan(u_margin).any()}")
    
    if torch.isnan(u_margin).any():
        print("FAILED: Margin transformation produces NaN")
        return False
    else:
        print("SUCCESS: Margin transformation produces valid values")
        return True

if __name__ == "__main__":
    print("PyTorch Vine Copula Debug Tests")
    print("=" * 50)
    
    # Run tests
    test1 = test_margin_cdf()
    test2 = test_theta_flip_logic()
    test3 = test_basic_parametric()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Margin CDF test: {'PASS' if test1 else 'FAIL'}")
    print(f"Theta/theta_flip test: {'PASS' if test2 else 'FAIL'}")
    print(f"Basic parametric test: {'PASS' if test3 else 'FAIL'}")
    
    if all([test1, test2, test3]):
        print("All tests PASSED! Basic functionality is working.")
    else:
        print("Some tests FAILED. Issues need to be addressed.") 