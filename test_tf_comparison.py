"""
Compare PyTorch and TensorFlow implementations of D-vine copulas
"""

import numpy as np
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')  # Add this line to fix TensorFlow imports

from DVC import vine_obj_bin, margin_obj, fit_vine
import tensorflow as tf
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_test_data(dim=5, n_samples=1000):
    """Generate correlated data for testing"""
    A = np.random.randn(dim, dim)
    cov_matrix = np.dot(A.T, A)  # Ensure positive definite
    # Normalize to correlation matrix with unit variance
    D = np.sqrt(np.diag(cov_matrix))
    cov_matrix = cov_matrix / np.outer(D, D)
    
    data = np.random.multivariate_normal(np.zeros(dim), cov_matrix, n_samples).astype(np.float32)
    return data, cov_matrix

def test_pytorch_implementation(data, dim):
    """Test PyTorch implementation"""
    print("\nTesting PyTorch implementation...")
    
    # Create a D-vine
    vine = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian', 'ind'],
        vine_depth=dim,
        margin=[],
        knots=50
    )

    # Set margins
    for i in range(dim):
        vine.margin.append(margin_obj('norm', [0, 1], True))

    # Configuration
    gen_dict = {"parallel": False, "param": True, "binning": False}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"method": "local", "n_iter": 100}
    bin_dict = {"n_bin": 1}

    # Fit the vine
    print("Fitting vine...")
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    print("Vine fitted!")

    # Generate samples
    print("Generating samples...")
    samples = vine.sample(1000)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(samples.T)
    return corr_matrix

def test_tensorflow_implementation(data, dim):
    """Test TensorFlow implementation"""
    print("\nTesting TensorFlow implementation...")
    
    # Create and fit D-vine
    print("Fitting vine...")
    # TensorFlow vine constructor: vine_family, families, vine_depth, margin, knots, method, *args
    dvine = tf_vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian', 'ind'],
        vine_depth=dim,
        margin=[],
        knots=50,
        method='matrix'
    )
    
    # Set margins
    from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj
    for i in range(dim):
        margin = tf_margin_obj('norm', [0, 1], True)
        # Set the kernel for the margin - use the raw data column
        margin.ker = data[:, i]
        dvine.margin.append(margin)
    
    # Configuration dictionaries for TensorFlow
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False, "vine_depth": dim}
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"opt_method": "local", "batch_paral": False}
    bin_dict = {"n_bin": 1}
    
    dvine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
    print("Vine fitted!")

    # Generate samples using proper TensorFlow sampling
    print("Generating samples...")
    from DVC_tensorflow.sampling.vine_sample import vine_cop_par_sample
    samples = vine_cop_par_sample(dvine, 1000)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(samples.T)
    return corr_matrix

def main():
    # Test parameters
    dim = 5
    n_samples = 1000
    
    # Generate test data
    print(f"Generating {dim}D test data with {n_samples} samples...")
    data, true_cov = generate_test_data(dim, n_samples)
    true_corr = true_cov / np.sqrt(np.outer(np.diag(true_cov), np.diag(true_cov)))
    
    print("\nTrue correlation matrix:")
    print(np.round(true_corr, 3))
    
    # Test both implementations
    pytorch_corr = test_pytorch_implementation(data, dim)
    tf_corr = test_tensorflow_implementation(data, dim)
    
    # Compare results
    print("\nResults comparison:")
    print("\nPyTorch correlation matrix:")
    print(np.round(pytorch_corr, 3))
    print("\nTensorFlow correlation matrix:")
    print(np.round(tf_corr, 3))
    
    # Compute errors
    pytorch_error = np.mean(np.abs(pytorch_corr - true_corr))
    tf_error = np.mean(np.abs(tf_corr - true_corr))
    
    print("\nMean absolute error from true correlation:")
    print(f"PyTorch: {pytorch_error:.4f}")
    print(f"TensorFlow: {tf_error:.4f}")

if __name__ == "__main__":
    main() 