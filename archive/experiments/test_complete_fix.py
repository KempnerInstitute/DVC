"""
Test Complete kernel_cdf Fix

This script tests the complete fix including:
1. Fixed vine_eval.py with kernel_cdf transformation
2. Fixed vine_model.py passing tr parameter
"""

import numpy as np
import torch
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

from scipy.stats import norm
from DVC_pyolder import vine_obj_bin, margin_obj
from DVC_pyolder.vine_model import fit_vine


def test_parametric_vine():
    """Test parametric vine with the complete fix"""
    print("\n=== TESTING PARAMETRIC VINE WITH COMPLETE FIX ===")
    
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
        
        # Check fitted parameters
        print("\nFitted parameters:")
        for level, copulas in enumerate(vine.copulas):
            print(f"   Level {level}:")
            for i, cop in enumerate(copulas):
                if hasattr(cop, 'family') and hasattr(cop, 'theta'):
                    print(f"     Edge {i}: {cop.family}, theta={cop.theta:.6f}")
        
        # Sample and check correlation recovery
        print("\nSampling from vine...")
        samples = vine.sample(5000)
        corr_recovered = np.corrcoef(samples.T)
        
        print("\nRecovered correlation matrix:")
        print(corr_recovered)
        
        mae = np.mean(np.abs(corr_recovered - corr))
        print(f"\nParametric MAE: {mae:.6f}")
        
        # Check specific correlations
        print("\nSpecific correlation errors:")
        print(f"  1-2: True={corr[0,1]:.3f}, Recovered={corr_recovered[0,1]:.3f}, Error={abs(corr[0,1]-corr_recovered[0,1]):.3f}")
        print(f"  1-3: True={corr[0,2]:.3f}, Recovered={corr_recovered[0,2]:.3f}, Error={abs(corr[0,2]-corr_recovered[0,2]):.3f}")
        print(f"  1-4: True={corr[0,3]:.3f}, Recovered={corr_recovered[0,3]:.3f}, Error={abs(corr[0,3]-corr_recovered[0,3]):.3f}")
        print(f"  2-3: True={corr[1,2]:.3f}, Recovered={corr_recovered[1,2]:.3f}, Error={abs(corr[1,2]-corr_recovered[1,2]):.3f}")
        print(f"  2-4: True={corr[1,3]:.3f}, Recovered={corr_recovered[1,3]:.3f}, Error={abs(corr[1,3]-corr_recovered[1,3]):.3f}")
        print(f"  3-4: True={corr[2,3]:.3f}, Recovered={corr_recovered[2,3]:.3f}, Error={abs(corr[2,3]-corr_recovered[2,3]):.3f}")
        
        return mae, corr_recovered
        
    except Exception as e:
        print(f"Error in parametric test: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_nonparametric_vine():
    """Test non-parametric vine with the complete fix"""
    print("\n\n=== TESTING NON-PARAMETRIC VINE WITH COMPLETE FIX ===")
    
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
    vine = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian', 'ind'],
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    gen_dict = {"parallel": False, "param": False, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"method": "local", "n_iter": 200}
    bin_dict = {"n_bin": 1}
    
    try:
        fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        
        print("\nNon-parametric vine fitted successfully")
        
        # Sample and check correlation recovery
        print("\nSampling from vine...")
        samples = vine.sample(5000)
        corr_recovered = np.corrcoef(samples.T)
        
        print("\nRecovered correlation matrix:")
        print(corr_recovered)
        
        mae = np.mean(np.abs(corr_recovered - corr))
        print(f"\nNon-parametric MAE: {mae:.6f}")
        
        return mae, corr_recovered
        
    except Exception as e:
        print(f"Error in non-parametric test: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_with_tensorflow():
    """Compare results with TensorFlow implementation"""
    print("\n\n=== COMPARING WITH TENSORFLOW ===")
    
    try:
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
        
        print("\nTensorFlow recovered correlation:")
        print(corr_tf)
        
        return mae_tf, corr_tf
        
    except Exception as e:
        print(f"Could not run TensorFlow comparison: {e}")
        return None, None


def main():
    """Run all tests and compare results"""
    print("="*70)
    print("TESTING COMPLETE KERNEL_CDF FIX")
    print("="*70)
    
    # Test parametric
    mae_param, corr_param = test_parametric_vine()
    
    # Test non-parametric
    mae_nonparam, corr_nonparam = test_nonparametric_vine()
    
    # Compare with TensorFlow
    mae_tf, corr_tf = compare_with_tensorflow()
    
    # Summary
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if mae_param is not None:
        print(f"\nParametric PyTorch MAE: {mae_param:.6f}")
        if mae_param < 0.1:
            print("✓ Parametric model is working well!")
        else:
            print("✗ Parametric model still needs improvement")
    
    if mae_nonparam is not None:
        print(f"\nNon-parametric PyTorch MAE: {mae_nonparam:.6f}")
        if mae_nonparam < 0.1:
            print("✓ Non-parametric model is working well!")
        else:
            print("✗ Non-parametric model still needs improvement")
    
    if mae_tf is not None:
        print(f"\nTensorFlow reference MAE: {mae_tf:.6f}")
    
    print("\nExpected results after fix:")
    print("- PyTorch parametric MAE: ~0.05")
    print("- PyTorch non-parametric MAE: ~0.04-0.05")
    print("- Should match TensorFlow MAE")
    
    if mae_param is not None and mae_param > 0.1:
        print("\n⚠️ The fix may not be fully applied. Check that:")
        print("1. vine_eval.py has been replaced with the fixed version")
        print("2. vine_model.py has been updated to pass 'tr' parameter")
        print("3. The kernel_cdf transformation is being applied")


if __name__ == "__main__":
    main() 