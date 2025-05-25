"""
Test kernel_cdf Fix with Monkey Patch

This script tests the kernel_cdf fix by applying the monkey patch
and comparing results with and without the fix.
"""

import numpy as np
import torch
import sys
sys.path.append('src')

from scipy.stats import norm


def test_without_fix():
    """Test without the kernel_cdf fix"""
    print("\n=== TESTING WITHOUT KERNEL_CDF FIX ===")
    
    from DVC import vine_obj_bin, margin_obj
    from DVC.vine_model import fit_vine
    
    # Generate test data
    np.random.seed(42)
    n = 500
    d = 4
    rho = 0.6
    
    # Create correlation matrix
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    print("\nTrue correlation matrix:")
    print(corr)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    # Fit parametric vine
    vine = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian', 'ind'],
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"method": "local", "n_iter": 50}
    bin_dict = {"n_bin": 1}
    
    try:
        fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        
        # Sample and check correlation recovery
        samples = vine.sample(5000)
        corr_recovered = np.corrcoef(samples.T)
        
        mae = np.mean(np.abs(corr_recovered - corr))
        print(f"\nWithout fix - MAE: {mae:.6f}")
        
        # Count None parameters
        none_count = 0
        for level, copulas in enumerate(vine.copulas):
            for cop in copulas:
                if hasattr(cop, 'theta') and cop.theta is None:
                    none_count += 1
        print(f"None parameters: {none_count}")
        
        return mae, corr_recovered
    except Exception as e:
        print(f"Error without fix: {e}")
        return None, None


def test_with_fix():
    """Test with the kernel_cdf fix"""
    print("\n\n=== TESTING WITH KERNEL_CDF FIX ===")
    
    # Apply the monkey patch
    import kernel_cdf_patch
    
    # Now test with the same data
    from DVC import vine_obj_bin, margin_obj
    from DVC.vine_model import fit_vine
    
    # Generate test data (same seed for fair comparison)
    np.random.seed(42)
    n = 500
    d = 4
    rho = 0.6
    
    # Create correlation matrix
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    # Fit parametric vine
    vine = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian', 'ind'],
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"method": "local", "n_iter": 50}
    bin_dict = {"n_bin": 1}
    
    try:
        fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        
        # Sample and check correlation recovery
        samples = vine.sample(5000)
        corr_recovered = np.corrcoef(samples.T)
        
        mae = np.mean(np.abs(corr_recovered - corr))
        print(f"\nWith fix - MAE: {mae:.6f}")
        
        # Count None parameters
        none_count = 0
        for level, copulas in enumerate(vine.copulas):
            for cop in copulas:
                if hasattr(cop, 'theta') and cop.theta is None:
                    none_count += 1
        print(f"None parameters: {none_count}")
        
        # Print recovered correlation
        print("\nRecovered correlation:")
        print(corr_recovered)
        
        return mae, corr_recovered
    except Exception as e:
        print(f"Error with fix: {e}")
        return None, None


def test_non_parametric():
    """Test non-parametric vine with the fix"""
    print("\n\n=== TESTING NON-PARAMETRIC VINE WITH FIX ===")
    
    # Apply the monkey patch if not already applied
    try:
        import kernel_cdf_patch
    except:
        pass
    
    from DVC import vine_obj_bin, margin_obj
    from DVC.vine_model import fit_vine
    
    # Generate test data
    np.random.seed(42)
    n = 500
    d = 4
    rho = 0.6
    
    # Create correlation matrix
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    # Fit non-parametric vine
    print("\nFitting non-parametric vine...")
    vine = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian', 'ind'],  # Still need families for selection
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    # Non-parametric settings
    gen_dict = {"parallel": False, "param": False, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"method": "local", "n_iter": 500}  # More iterations for non-parametric
    bin_dict = {"n_bin": 1}
    
    try:
        fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        
        # Sample and check correlation recovery
        print("\nSampling from non-parametric vine...")
        samples = vine.sample(5000)
        corr_recovered = np.corrcoef(samples.T)
        
        mae = np.mean(np.abs(corr_recovered - corr))
        print(f"\nNon-parametric MAE: {mae:.6f}")
        
        print("\nRecovered correlation:")
        print(corr_recovered)
        
        return mae, corr_recovered
    except Exception as e:
        print(f"Error in non-parametric: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_with_tensorflow():
    """Compare results with TensorFlow implementation"""
    print("\n\n=== COMPARING WITH TENSORFLOW ===")
    
    try:
        sys.path.append('src/DVC_tensorflow')
        from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
        from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj
        
        # Generate same test data
        np.random.seed(42)
        n = 500
        d = 4
        rho = 0.6
        
        corr = np.eye(d)
        for i in range(d):
            for j in range(i+1, d):
                corr[i, j] = corr[j, i] = rho ** abs(i-j)
        
        data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
        
        # Fit TensorFlow vine
        print("\nFitting TensorFlow vine...")
        vine_tf = tf_vine_obj_bin(
            vine_family='d-vine',
            families=['gaussian', 'ind'],
            vine_depth=d,
            margin=[],
            knots=50,
            method='matrix'
        )
        
        # Set margins
        for i in range(d):
            margin = tf_margin_obj('norm', [0, 1], True)
            margin.ker = data[:, i]
            vine_tf.margin.append(margin)
        
        # TensorFlow dictionaries
        gen_dict_tf = {"parallel": False, "param": True, "binning": False, "fitted": False}
        npc_dict_tf = {"method": "local", "n_iter": 50}
        par_dict_tf = {"param_families": ["gaussian", "ind"]}
        bin_dict_tf = {"n_bin": 1}
        
        # Fit
        vine_tf.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
        
        # Sample
        samples_tf = vine_tf.sample(5000)
        corr_tf = np.corrcoef(samples_tf.T)
        
        mae_tf = np.mean(np.abs(corr_tf - corr))
        print(f"\nTensorFlow MAE: {mae_tf:.6f}")
        
        return mae_tf, corr_tf
        
    except Exception as e:
        print(f"Could not run TensorFlow comparison: {e}")
        return None, None


def main():
    """Run all tests"""
    print("="*70)
    print("TESTING KERNEL_CDF FIX")
    print("="*70)
    
    # Test without fix
    mae_without, corr_without = test_without_fix()
    
    # Test with fix
    mae_with, corr_with = test_with_fix()
    
    # Test non-parametric
    mae_nonparam, corr_nonparam = test_non_parametric()
    
    # Compare with TensorFlow
    mae_tf, corr_tf = compare_with_tensorflow()
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    
    if mae_without is not None:
        print(f"\nParametric without fix: MAE = {mae_without:.6f}")
    
    if mae_with is not None:
        print(f"Parametric with fix:    MAE = {mae_with:.6f}")
        if mae_without is not None:
            improvement = (mae_without - mae_with) / mae_without * 100
            print(f"Improvement: {improvement:.1f}%")
    
    if mae_nonparam is not None:
        print(f"\nNon-parametric with fix: MAE = {mae_nonparam:.6f}")
    
    if mae_tf is not None:
        print(f"\nTensorFlow reference: MAE = {mae_tf:.6f}")
    
    print("\nExpected results after full fix:")
    print("- Parametric MAE should drop to ~0.05")
    print("- Non-parametric MAE should be similar or better")
    print("- Results should match TensorFlow")


if __name__ == "__main__":
    main() 