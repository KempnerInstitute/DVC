"""
Debug PyTorch vs TensorFlow Performance Issues

This script performs focused investigation to understand why PyTorch 
vine.copulass are performing worse than TensorFlow implementation.
"""

import numpy as np
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time

# Import PyTorch implementation
from DVC_pyolder import vine_obj_bin, margin_obj, fit_vine

# Import TensorFlow implementation
import tensorflow as tf
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj
from DVC_tensorflow.sampling.vine_sample import vine_cop_par_sample

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_test_data(dim=4, n_samples=1000, correlation_strength=0.6):
    """Generate test data with known correlation structure"""
    
    # Create structured correlation matrix
    cov_matrix = np.eye(dim)
    
    # Add correlations
    for i in range(dim):
        for j in range(i+1, dim):
            if j == i + 1:  # Adjacent variables
                cov_matrix[i, j] = cov_matrix[j, i] = correlation_strength
            elif j == i + 2:  # Skip-one variables
                cov_matrix[i, j] = cov_matrix[j, i] = correlation_strength * 0.6
            else:  # Distant variables
                cov_matrix[i, j] = cov_matrix[j, i] = correlation_strength * 0.3
                
    # Ensure positive definite
    cov_matrix = cov_matrix + 0.1 * np.eye(dim)
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(dim), cov_matrix, n_samples).astype(np.float32)
    
    return data, cov_matrix

def debug_pytorch_fitting(data, vine_type='d-vine', approach='parametric'):
    """Debug PyTorch vine fitting process step by step"""
    
    print(f"\n=== DEBUGGING PYTORCH {approach.upper()} {vine_type.upper()} ===")
    dim = data.shape[1]
    
    # Create vine
    print("1. Creating vine object...")
    vine = vine_obj_bin(
        vine_family=vine_type,
        families=['gaussian', 'ind'],
        vine_depth=dim,
        margin=[],
        knots=50
    )
    print(f"   Created vine: {vine_type}, depth: {dim}")
    
    # Set margins
    print("2. Setting margins...")
    for i in range(dim):
        margin = margin_obj('norm', [0, 1], True)
        vine.margin.append(margin)
        print(f"   Margin {i}: {margin.dist}")
    
    # Configuration
    is_parametric = (approach == 'parametric')
    gen_dict = {"parallel": False, "param": is_parametric, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"method": "local", "n_iter": 50 if is_parametric else 100}
    bin_dict = {"n_bin": 1}
    
    print(f"3. Configuration:")
    print(f"   gen_dict: {gen_dict}")
    print(f"   par_dict: {par_dict}")
    print(f"   npc_dict: {npc_dict}")
    print(f"   bin_dict: {bin_dict}")
    
    # Fit vine with detailed timing
    print("4. Fitting vine...")
    start_time = time.time()
    
    try:
        fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        print("   ✓ Fit successful in {:.3f}s".format(fit_time))
        
        # Debug: Print fitted parameters
        print("7. Fitted copula parameters:")
        for level_idx, level_copulas in enumerate(vine.copulas):
            print(f"   Level {level_idx}:")
            for edge_idx, cop in enumerate(level_copulas):
                if hasattr(cop, 'family'):
                    print(f"      Edge {edge_idx}: family={cop.family}, theta={cop.theta}")
                    
        print("8. Checking fitted vine structure...")
        print(f"   Vine family: {vine.vine_family}")
        print(f"   Vine depth: {vine.n_cop}")
        print(f"   Number of margins: {len(vine.margin)}")
        
        # Check copula parameters
        if hasattr(vine, 'copula') and vine.copulas:
            print(f"   Number of copulas: {len(vine.copulas)}")
            for level, cop_level in enumerate(vine.copulas):
                print(f"   Level {level}: {len(cop_level)} copulas")
                for i, cop in enumerate(cop_level):
                    if hasattr(cop, 'theta') and cop.theta is not None:
                        if isinstance(cop.theta, list):
                            theta_val = cop.theta[0] if len(cop.theta) > 0 else "empty"
                        else:
                            theta_val = cop.theta
                        print(f"     Copula {i}: family={cop.family}, theta={theta_val}")
                    else:
                        print(f"     Copula {i}: family={cop.family}, theta=None")
        
        # Test sampling
        print("6. Testing sampling...")
        start_time = time.time()
        samples = vine.sample(500)
        sample_time = time.time() - start_time
        print(f"   ✓ Sampling successful: {samples.shape}, time: {sample_time:.3f}s")
        
        # Check sample quality
        sample_corr = np.corrcoef(samples.T)
        print(f"   Sample correlation range: [{np.min(sample_corr):.3f}, {np.max(sample_corr):.3f}]")
        
        return vine, {'fit_time': fit_time, 'sample_time': sample_time, 'samples': samples}
        
    except Exception as e:
        print(f"   ✗ Fit failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def debug_tensorflow_fitting(data, vine_type='d-vine', approach='parametric'):
    """Debug TensorFlow vine fitting process step by step"""
    
    print(f"\n=== DEBUGGING TENSORFLOW {approach.upper()} {vine_type.upper()} ===")
    dim = data.shape[1]
    
    # Create vine
    print("1. Creating vine object...")
    vine = tf_vine_obj_bin(
        vine_family=vine_type,
        families=['gaussian', 'ind'],
        vine_depth=dim,
        margin=[],
        knots=50,
        method='matrix'
    )
    print(f"   Created vine: {vine_type}, depth: {dim}")
    
    # Set margins
    print("2. Setting margins...")
    for i in range(dim):
        margin = tf_margin_obj('norm', [0, 1], True)
        margin.ker = data[:, i]
        vine.margin.append(margin)
        print(f"   Margin {i}: {margin.dist}")
    
    # Configuration
    is_parametric = (approach == 'parametric')
    gen_dict = {"parallel": False, "param": is_parametric, "binning": False, "fitted": False, "vine_depth": dim}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"opt_method": "local", "batch_paral": False}
    bin_dict = {"n_bin": 1}
    
    print(f"3. Configuration:")
    print(f"   gen_dict: {gen_dict}")
    print(f"   par_dict: {par_dict}")
    print(f"   npc_dict: {npc_dict}")
    print(f"   bin_dict: {bin_dict}")
    
    # Fit vine with detailed timing
    print("4. Fitting vine...")
    start_time = time.time()
    
    try:
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        print(f"   ✓ Fit successful in {fit_time:.3f}s")
        
        # Debug: Print fitted parameters
        print("7. Fitted copula parameters:")
        for level_idx, level_copulas in enumerate(vine.copulas):
            print(f"   Level {level_idx}:")
            for edge_idx, cop in enumerate(level_copulas):
                if hasattr(cop, 'family'):
                    print(f"      Edge {edge_idx}: family={cop.family}, theta={cop.theta}")
        
        print("8. Checking fitted vine structure...")
        print(f"   Vine family: {vine.vine_family}")
        print(f"   Vine depth: {vine.n_cop}")
        print(f"   Number of margins: {len(vine.margin)}")
        
        # Check copula parameters
        if hasattr(vine, 'copula') and vine.copulas:
            print(f"   Number of copulas: {len(vine.copulas)}")
            for level, cop_level in enumerate(vine.copulas):
                print(f"   Level {level}: {len(cop_level)} copulas")
                for i, cop in enumerate(cop_level):
                    if hasattr(cop, 'theta') and cop.theta is not None:
                        if isinstance(cop.theta, list):
                            theta_val = cop.theta[0] if len(cop.theta) > 0 else "empty"
                        else:
                            theta_val = cop.theta
                        print(f"     Copula {i}: family={cop.family}, theta={theta_val}")
                    else:
                        print(f"     Copula {i}: family={cop.family}, theta=None")
        
        # Test sampling
        print("6. Testing sampling...")
        start_time = time.time()
        if is_parametric:
            samples = vine_cop_par_sample(vine, 500)
        else:
            from DVC_tensorflow.sampling.vine_sample import vine_copula_sample
            samples = vine_copula_sample(vine, 500)[0]
        sample_time = time.time() - start_time
        print(f"   ✓ Sampling successful: {samples.shape}, time: {sample_time:.3f}s")
        
        # Check sample quality
        sample_corr = np.corrcoef(samples.T)
        print(f"   Sample correlation range: [{np.min(sample_corr):.3f}, {np.max(sample_corr):.3f}]")
        
        return vine, {'fit_time': fit_time, 'sample_time': sample_time, 'samples': samples}
        
    except Exception as e:
        print(f"   ✗ Fit failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_correlation_recovery(true_cov, pytorch_samples, tf_samples):
    """Compare correlation recovery between implementations"""
    
    print(f"\n=== CORRELATION RECOVERY COMPARISON ===")
    
    # True correlations
    true_corr = true_cov / np.sqrt(np.outer(np.diag(true_cov), np.diag(true_cov)))
    pytorch_corr = np.corrcoef(pytorch_samples.T)
    tf_corr = np.corrcoef(tf_samples.T)
    
    # Compute errors
    pytorch_mae = np.mean(np.abs(pytorch_corr - true_corr))
    tf_mae = np.mean(np.abs(tf_corr - true_corr))
    
    print(f"True correlation matrix:")
    print(true_corr)
    print(f"\nPyTorch correlation matrix:")
    print(pytorch_corr)
    print(f"PyTorch MAE: {pytorch_mae:.4f}")
    print(f"\nTensorFlow correlation matrix:")
    print(tf_corr)
    print(f"TensorFlow MAE: {tf_mae:.4f}")
    
    # Compare individual correlations
    print(f"\nDetailed comparison:")
    print(f"{'i-j':<6} {'True':<8} {'PyTorch':<10} {'TF':<10} {'PyT_Err':<10} {'TF_Err':<10}")
    print("-" * 60)
    
    for i in range(len(true_corr)):
        for j in range(i+1, len(true_corr)):
            true_val = true_corr[i, j]
            pytorch_val = pytorch_corr[i, j]
            tf_val = tf_corr[i, j]
            pytorch_err = abs(pytorch_val - true_val)
            tf_err = abs(tf_val - true_val)
            print(f"{i}-{j:<4} {true_val:<8.3f} {pytorch_val:<10.3f} {tf_val:<10.3f} {pytorch_err:<10.3f} {tf_err:<10.3f}")
    
    return pytorch_mae, tf_mae

def main():
    """Run focused debugging comparison"""
    
    print("="*80)
    print("PYTORCH vs TENSORFLOW PERFORMANCE DEBUGGING")
    print("="*80)
    
    # Generate test data
    data, true_cov = generate_test_data(dim=4, n_samples=800, correlation_strength=0.6)
    print(f"Generated test data: {data.shape}")
    print(f"True correlation matrix:")
    true_corr = true_cov / np.sqrt(np.outer(np.diag(true_cov), np.diag(true_cov)))
    print(true_corr)
    
    # Test parametric approach
    print(f"\n{'='*20} PARAMETRIC APPROACH {'='*20}")
    
    pytorch_vine_param, pytorch_results_param = debug_pytorch_fitting(data, 'd-vine', 'parametric')
    tf_vine_param, tf_results_param = debug_tensorflow_fitting(data, 'd-vine', 'parametric')
    
    if pytorch_results_param and tf_results_param:
        pytorch_mae_param, tf_mae_param = compare_correlation_recovery(
            true_cov, pytorch_results_param['samples'], tf_results_param['samples']
        )
        print(f"\nParametric Results Summary:")
        print(f"PyTorch MAE: {pytorch_mae_param:.4f} (Fit: {pytorch_results_param['fit_time']:.2f}s)")
        print(f"TensorFlow MAE: {tf_mae_param:.4f} (Fit: {tf_results_param['fit_time']:.2f}s)")
    
    # Test non-parametric approach
    print(f"\n{'='*20} NON-PARAMETRIC APPROACH {'='*20}")
    
    pytorch_vine_nonparam, pytorch_results_nonparam = debug_pytorch_fitting(data, 'd-vine', 'non-parametric')
    tf_vine_nonparam, tf_results_nonparam = debug_tensorflow_fitting(data, 'd-vine', 'non-parametric')
    
    if pytorch_results_nonparam and tf_results_nonparam:
        pytorch_mae_nonparam, tf_mae_nonparam = compare_correlation_recovery(
            true_cov, pytorch_results_nonparam['samples'], tf_results_nonparam['samples']
        )
        print(f"\nNon-parametric Results Summary:")
        print(f"PyTorch MAE: {pytorch_mae_nonparam:.4f} (Fit: {pytorch_results_nonparam['fit_time']:.2f}s)")
        print(f"TensorFlow MAE: {tf_mae_nonparam:.4f} (Fit: {tf_results_nonparam['fit_time']:.2f}s)")
    
    # Summary and recommendations
    print(f"\n{'='*20} DIAGNOSIS & RECOMMENDATIONS {'='*20}")
    
    if pytorch_results_param and tf_results_param:
        if pytorch_mae_param > tf_mae_param:
            print(f"• PyTorch parametric is less accurate (MAE diff: {pytorch_mae_param - tf_mae_param:.4f})")
            print("  Potential issues:")
            print("    - Parameter initialization differences")
            print("    - Optimization procedure differences")
            print("    - Theta parameter handling (list vs scalar)")
        else:
            print(f"• PyTorch parametric is more accurate!")
    
    if pytorch_results_nonparam and tf_results_nonparam:
        if pytorch_mae_nonparam > tf_mae_nonparam:
            print(f"• PyTorch non-parametric is less accurate (MAE diff: {pytorch_mae_nonparam - tf_mae_nonparam:.4f})")
            print("  Potential issues:")
            print("    - Kernel density estimation differences")
            print("    - Grid initialization issues")
            print("    - H-function implementation differences")
        else:
            print(f"• PyTorch non-parametric is more accurate!")
    
    print("\nNext steps for improvement:")
    print("1. Check parameter initialization consistency")
    print("2. Verify h-function implementations")
    print("3. Compare optimization procedures")
    print("4. Test with different family selections")
    print("5. Debug theta parameter handling")

if __name__ == "__main__":
    main() 