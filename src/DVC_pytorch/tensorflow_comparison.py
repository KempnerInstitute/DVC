"""
Comparison between PyTorch and TensorFlow DVC implementations
"""

import numpy as np
import torch
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt
from time import perf_counter
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
tensorflow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DVC_tensorflow')
sys.path.append(tensorflow_path)

print(f"TensorFlow path: {tensorflow_path}")
print(f"Path exists: {os.path.exists(tensorflow_path)}")

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch device: {device}")
print(f"TensorFlow GPUs: {len(tf.config.list_physical_devices('GPU'))}")
print("="*80)

def generate_test_data(n_samples=500, n_dims=3):
    """Generate simple test data"""
    # Create correlation matrix
    corr_matrix = np.eye(n_dims)
    if n_dims > 1:
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.7
    if n_dims > 2:
        corr_matrix[1, 2] = corr_matrix[2, 1] = 0.5
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(n_dims), corr_matrix, n_samples)
    
    # Convert to uniform margins
    data_uniform = np.zeros_like(data)
    for i in range(n_dims):
        data_uniform[:, i] = stats.rankdata(data[:, i]) / (n_samples + 1)
    
    return data, data_uniform

def test_pytorch_implementation(data_uniform):
    """Test PyTorch implementation"""
    print("\nPyTorch Implementation Test:")
    print("-" * 40)
    
    try:
        # Import PyTorch version
        from classes.objects import vine_obj_bin, margin_obj
        from grid.grid_op import create_grids
        
        # Convert to PyTorch
        data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=device)
        n_dims = data_uniform.shape[1]
        
        # Create margins
        margins = [margin_obj('empirical', None, True) for _ in range(n_dims)]
        
        # Create vine
        vine = vine_obj_bin(
            vine_family='c-vine',
            families=['gaussian'],
            vine_depth=n_dims - 1,
            margin=margins,
            knots=32,
            method=None
        )
        
        # Create grids
        vine.grid_u, vine.grid_s, vine.grid_x = create_grids(vine.knots, device=device)
        
        # Set parameters
        gen_dict = {
            'binning': False,
            'parallel': False,
            'param': True,
            'vine_depth': n_dims - 1
        }
        par_dict = {
            'param_families': ['gaussian', 'clayton', 'ind']
        }
        bin_dict = {'n_bin': 1}
        
        # Fit
        start_time = perf_counter()
        vine.fit(data_torch, gen_dict, {}, par_dict, bin_dict)
        fit_time = perf_counter() - start_time
        
        print(f"✓ Fitting successful in {fit_time:.3f}s")
        
        # Get correlations
        if hasattr(vine, 'correlations') and len(vine.correlations) > 0:
            print(f"Fitted correlations: {vine.correlations[0]}")
        
        # Evaluate
        test_data = data_torch[:100]
        p, p_cop, log_p = vine.evaluation(test_data)
        mean_loglik = log_p.mean().item()
        print(f"Mean log-likelihood: {mean_loglik:.3f}")
        
        return {
            'success': True,
            'fit_time': fit_time,
            'loglik': mean_loglik,
            'correlations': vine.correlations[0] if hasattr(vine, 'correlations') else None
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def test_tensorflow_implementation(data_uniform):
    """Test TensorFlow implementation"""
    print("\nTensorFlow Implementation Test:")
    print("-" * 40)
    
    try:
        # Need to clear any PyTorch imports first
        import importlib
        
        # Import TensorFlow version with explicit path
        spec = importlib.util.spec_from_file_location(
            "tf_objects", 
            os.path.join(tensorflow_path, "classes", "objects.py")
        )
        tf_objects = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tf_objects)
        
        spec = importlib.util.spec_from_file_location(
            "tf_grid", 
            os.path.join(tensorflow_path, "grid", "grid_op.py")
        )
        tf_grid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tf_grid)
        
        # Convert to TensorFlow
        data_tf = tf.constant(data_uniform, dtype=tf.float32)
        n_dims = data_uniform.shape[1]
        
        # Create margins
        margins = [tf_objects.margin_obj('empirical', None, True) for _ in range(n_dims)]
        
        # Create vine
        vine = tf_objects.vine_obj_bin(
            vine_family='c-vine',
            families=['gaussian'],
            vine_depth=n_dims - 1,
            margin=margins,
            knots=32,
            method=None
        )
        
        # Create grids
        from grid.grid_class import grid_obj
        vine.grid_u = tf_grid.mk_grid(vine.knots, tf.float32)
        vine.grid_s = tf_grid.mk_grid(vine.knots, tf.float32)
        vine.grid_x = tf_grid.mk_grid(vine.knots, tf.float32)
        
        # Set parameters
        gen_dict = {
            'binning': False,
            'parallel': False,
            'param': True,
            'vine_depth': n_dims - 1
        }
        par_dict = {
            'param_families': ['gaussian', 'clayton', 'ind']
        }
        bin_dict = {'n_bin': 1}
        
        # Fit
        start_time = perf_counter()
        vine.fit(data_tf, gen_dict, {}, par_dict, bin_dict)
        fit_time = perf_counter() - start_time
        
        print(f"✓ Fitting successful in {fit_time:.3f}s")
        
        # Get correlations
        if hasattr(vine, 'correlations') and len(vine.correlations) > 0:
            print(f"Fitted correlations: {vine.correlations[0]}")
        
        # Evaluate
        test_data = data_tf[:100]
        p, p_cop, log_p = vine.evaluation(test_data)
        mean_loglik = tf.reduce_mean(log_p).numpy()
        print(f"Mean log-likelihood: {mean_loglik:.3f}")
        
        return {
            'success': True,
            'fit_time': fit_time,
            'loglik': mean_loglik,
            'correlations': vine.correlations[0] if hasattr(vine, 'correlations') else None
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def compare_results(pt_result, tf_result, true_tau):
    """Compare results from both implementations"""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if pt_result['success'] and tf_result['success']:
        # Timing comparison
        speedup = tf_result['fit_time'] / pt_result['fit_time']
        print(f"\nTiming:")
        print(f"  PyTorch: {pt_result['fit_time']:.3f}s")
        print(f"  TensorFlow: {tf_result['fit_time']:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Log-likelihood comparison
        print(f"\nLog-likelihood:")
        print(f"  PyTorch: {pt_result['loglik']:.3f}")
        print(f"  TensorFlow: {tf_result['loglik']:.3f}")
        print(f"  Difference: {abs(pt_result['loglik'] - tf_result['loglik']):.3f}")
        
        # Correlation comparison
        if pt_result['correlations'] and tf_result['correlations']:
            print(f"\nCorrelation Estimation:")
            print(f"  True Kendall's tau: {true_tau}")
            print(f"  PyTorch estimates: {pt_result['correlations']}")
            print(f"  TensorFlow estimates: {tf_result['correlations']}")
            
            # Calculate errors
            pt_errors = [abs(pt_result['correlations'][i] - true_tau[i]) 
                        for i in range(len(true_tau))]
            tf_errors = [abs(tf_result['correlations'][i] - true_tau[i]) 
                        for i in range(len(true_tau))]
            
            print(f"  PyTorch mean error: {np.mean(pt_errors):.4f}")
            print(f"  TensorFlow mean error: {np.mean(tf_errors):.4f}")
    else:
        print("One or both implementations failed to run successfully.")

def main():
    """Main comparison function"""
    print("DVC Implementation Comparison: PyTorch vs TensorFlow")
    print("="*80)
    
    # Generate test data
    print("\nGenerating test data...")
    data, data_uniform = generate_test_data(n_samples=500, n_dims=3)
    print(f"Data shape: {data.shape}")
    
    # Compute true correlations
    true_tau = []
    for i in range(data.shape[1] - 1):
        tau, _ = stats.kendalltau(data[:, i], data[:, i+1])
        true_tau.append(tau)
    print(f"True Kendall's tau: {true_tau}")
    
    # Test PyTorch
    pt_result = test_pytorch_implementation(data_uniform)
    
    # Test TensorFlow
    tf_result = test_tensorflow_implementation(data_uniform)
    
    # Compare results
    compare_results(pt_result, tf_result, true_tau)

if __name__ == "__main__":
    main() 