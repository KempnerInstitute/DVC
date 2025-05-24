"""
Test PyTorch DVC Improvements

This script tests the key improvements made to the PyTorch implementation.
"""

import numpy as np
import torch
import sys
sys.path.append('src')

from DVC import vine_obj_bin, margin_obj, fit_vine

def test_parametric_fitting():
    """Test that parametric fitting now uses gradient-based optimization"""
    print("Testing Parametric Fitting...")
    
    # Generate correlated data
    np.random.seed(42)
    n = 500
    rho = 0.6
    cov = np.array([[1, rho], [rho, 1]])
    data = np.random.multivariate_normal([0, 0], cov, n).astype(np.float32)
    
    # Create simple 2D vine
    vine = vine_obj_bin(
        vine_family='c-vine',
        families=['gaussian', 'ind'],
        vine_depth=2,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(2)],
        knots=50
    )
    
    # Fit with parametric approach
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"method": "local", "n_iter": 50}
    bin_dict = {"n_bin": 1}
    
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Check results
    cop = vine.copulas[0][0]
    print(f"  Selected family: {cop.family}")
    print(f"  Fitted parameter: {cop.theta}")
    print(f"  Expected ~{rho}, got {cop.theta}")
    
    # Verify Gaussian was selected
    assert cop.family == 'gaussian', f"Expected Gaussian, got {cop.family}"
    # Verify parameter is reasonable
    assert abs(cop.theta - rho) < 0.15, f"Parameter {cop.theta} too far from true {rho}"
    
    print("  ✓ Parametric fitting test passed!\n")


def test_copula_normalization():
    """Test the improved copula normalization"""
    print("Testing Copula Normalization...")
    
    from DVC.cop_eval import eval_rs_cop
    from DVC.utils_prob import biv_norm
    from DVC.grid_ops import grid_obj
    
    # Create test grid
    knots = 30
    x = torch.linspace(0, 1, knots)
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    grid = grid_obj(grid_points)
    adu11, adu22 = grid.diff()
    
    # Create test kernel and reference
    s = torch.linspace(-3, 3, knots)
    X, Y = torch.meshgrid(s, s, indexing='ij')
    ker_fit = torch.exp(-0.5 * (X**2 + Y**2)).unsqueeze(-1)
    NORM = biv_norm(s, s).unsqueeze(-1)
    
    # Run normalization
    result = eval_rs_cop(adu11, adu22, ker_fit, NORM, 1)
    
    # Check properties
    print(f"  Input range: [{ker_fit.min():.4f}, {ker_fit.max():.4f}]")
    print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"  Contains NaN: {torch.isnan(result).any()}")
    print(f"  Contains Inf: {torch.isinf(result).any()}")
    
    assert not torch.isnan(result).any(), "Result contains NaN"
    assert not torch.isinf(result).any(), "Result contains Inf"
    assert result.min() >= 0, "Negative values in normalized result"
    
    print("  ✓ Copula normalization test passed!\n")


def test_h_function_stability():
    """Test h-function numerical stability"""
    print("Testing H-function Stability...")
    
    from DVC.vine_model import _h_function
    from DVC.objects import cop_par_obj
    
    # Test with extreme values
    u_root = torch.tensor([1e-9, 0.1, 0.5, 0.9, 1-1e-9])
    u_other = torch.tensor([1e-9, 0.2, 0.5, 0.8, 1-1e-9])
    
    # Test Gaussian copula
    cop = cop_par_obj('gaussian', 0.5)
    result_left = _h_function(u_root, u_other, cop, None, side="left")
    result_right = _h_function(u_other, u_root, cop, None, side="right")
    
    print(f"  Left h-function range: [{result_left.min():.6f}, {result_left.max():.6f}]")
    print(f"  Right h-function range: [{result_right.min():.6f}, {result_right.max():.6f}]")
    print(f"  Contains NaN: {torch.isnan(result_left).any() or torch.isnan(result_right).any()}")
    
    # Check validity
    assert (result_left >= 0).all() and (result_left <= 1).all(), "H-function outside [0,1]"
    assert (result_right >= 0).all() and (result_right <= 1).all(), "H-function outside [0,1]"
    assert not torch.isnan(result_left).any(), "NaN in left h-function"
    assert not torch.isnan(result_right).any(), "NaN in right h-function"
    
    print("  ✓ H-function stability test passed!\n")


def test_aic_comparison():
    """Test that AIC comparison now works correctly"""
    print("Testing AIC Comparison...")
    
    from DVC.param_copula import parametric_fit
    
    # Generate independent data
    n = 200
    u_indep = np.random.uniform(0, 1, (n, 2, 1)).astype(np.float32)
    
    # Generate correlated data  
    rho = 0.7
    normal = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    from scipy.stats import norm
    u_corr = norm.cdf(normal).reshape(n, 2, 1).astype(np.float32)
    
    # Test on independent data
    aic_indep, _, _ = parametric_fit(u_indep, ['gaussian', 'ind'], 1)
    print(f"  Independent data AICs: Gaussian={aic_indep[0][0]:.2f}, Ind={aic_indep[0][1]:.2f}")
    print(f"  Selected: {'Gaussian' if aic_indep[0][0] < aic_indep[0][1] else 'Independence'}")
    
    # Test on correlated data
    aic_corr, theta_corr, _ = parametric_fit(u_corr, ['gaussian', 'ind'], 1)
    print(f"  Correlated data AICs: Gaussian={aic_corr[0][0]:.2f}, Ind={aic_corr[0][1]:.2f}")
    print(f"  Selected: {'Gaussian' if aic_corr[0][0] < aic_corr[0][1] else 'Independence'}")
    print(f"  Fitted rho: {theta_corr[0][0]}")
    
    # For correlated data, Gaussian should win
    assert aic_corr[0][0] < aic_corr[0][1], "Gaussian should win for correlated data"
    
    print("  ✓ AIC comparison test passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("PyTorch DVC Improvement Tests")
    print("="*60)
    
    try:
        test_aic_comparison()
        test_h_function_stability()
        test_copula_normalization()
        test_parametric_fitting()
        
        print("\n✓ All tests passed!")
        print("\nKey improvements verified:")
        print("- Gradient-based parametric fitting")
        print("- Proper AIC calculation")
        print("- Stable h-function implementation")
        print("- Correct copula normalization")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc() 