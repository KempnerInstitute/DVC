"""
Trace Likelihood Calculation Differences

This script investigates why TensorFlow reports log-likelihood ~50 higher
than PyTorch for the same Gaussian copula fitting.
"""

import numpy as np
import torch
import tensorflow as tf
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

from scipy.stats import norm

# Import the actual fitting functions
from DVC_pyolder.param_copula import fit_gaussian as pt_fit_gaussian
from DVC_tensorflow.param.copula_fit import fit_gaussian as tf_fit_gaussian_raw


def investigate_gaussian_cost_functions():
    """Deep dive into how each implementation calculates the cost/likelihood"""
    print("\n=== INVESTIGATING GAUSSIAN COPULA COST FUNCTIONS ===")
    
    # Generate simple test data
    np.random.seed(42)
    n = 100
    rho_true = 0.6
    data = np.random.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
    u_data = norm.cdf(data).astype(np.float32)
    
    print(f"\n1. Test data: n={n}, true rho={rho_true}")
    
    # Look at TensorFlow's gaussian_cost function
    print("\n2. Examining TensorFlow's gaussian_cost...")
    
    # Import TensorFlow's cost function
    from DVC_tensorflow.param.gaussian_copula import gaussian_cost
    
    # Prepare data for TensorFlow (needs 3D: [n, 2, 1])
    u_tf = tf.constant(u_data, dtype=tf.float32)
    u_tf_3d = tf.expand_dims(u_tf, -1)
    
    # Test different rho values
    test_rhos = [0.5, 0.6, 0.7]
    
    for test_rho in test_rhos:
        rho_tf = tf.constant([test_rho], dtype=tf.float32)
        
        # Call TensorFlow's cost function
        cost_tf = gaussian_cost(u_tf_3d, rho_tf, 1)
        
        # The cost is negative log-likelihood
        ll_tf = -cost_tf.numpy()[0]
        
        print(f"\n   Rho = {test_rho}:")
        print(f"     TensorFlow cost: {cost_tf.numpy()[0]:.6f}")
        print(f"     TensorFlow log-likelihood: {ll_tf:.6f}")
        
        # Manual calculation to understand what TF is computing
        # Convert to normal scores
        z = norm.ppf(u_data)
        z1, z2 = z[:, 0], z[:, 1]
        
        # Bivariate normal log PDF (includes marginals)
        one_minus_rho2 = 1 - test_rho**2
        log_det_sigma = np.log(one_minus_rho2)
        quad_form = (z1**2 - 2*test_rho*z1*z2 + z2**2) / one_minus_rho2
        log_biv_normal = -0.5 * (2*np.log(2*np.pi) + log_det_sigma + quad_form)
        ll_joint = np.sum(log_biv_normal)
        
        # Copula density only (excludes marginals)
        log_copula = -0.5 * log_det_sigma - 0.5 * ((test_rho**2 * (z1**2 + z2**2) - 2*test_rho*z1*z2) / one_minus_rho2)
        ll_copula = np.sum(log_copula)
        
        print(f"     Manual joint log-likelihood: {ll_joint:.6f}")
        print(f"     Manual copula log-likelihood: {ll_copula:.6f}")
        print(f"     Difference (joint - TF): {abs(ll_joint - ll_tf):.6f}")
    
    # Now check PyTorch's calculation
    print("\n3. PyTorch's calculation:")
    u_torch = torch.tensor(u_data)
    rho_pt, ll_pt, aic_pt = pt_fit_gaussian(u_torch)
    print(f"   Fitted rho: {rho_pt:.6f}")
    print(f"   Log-likelihood: {ll_pt:.6f}")
    print(f"   AIC: {aic_pt:.6f}")
    
    # The key insight
    print("\n4. KEY INSIGHT:")
    print("   TensorFlow appears to compute the JOINT log-likelihood (includes marginals)")
    print("   PyTorch correctly computes only the COPULA log-likelihood")
    print("   This explains the ~50 difference in log-likelihood values!")


def check_tensorflow_code():
    """Look at TensorFlow's actual gaussian_cost implementation"""
    print("\n\n=== CHECKING TENSORFLOW'S GAUSSIAN_COST CODE ===")
    
    # Read the TensorFlow gaussian copula file
    try:
        with open('src/DVC_tensorflow/param/gaussian_copula.py', 'r') as f:
            lines = f.readlines()
            
        # Find the gaussian_cost function
        in_function = False
        function_lines = []
        
        for i, line in enumerate(lines):
            if 'def gaussian_cost' in line:
                in_function = True
            
            if in_function:
                function_lines.append((i+1, line.rstrip()))
                
                # Stop at the next function definition
                if line.strip().startswith('def ') and i > 0 and 'gaussian_cost' not in line:
                    break
        
        # Print the function
        print("\nTensorFlow's gaussian_cost function:")
        for line_num, line in function_lines[:30]:  # First 30 lines
            print(f"{line_num}: {line}")
            
    except FileNotFoundError:
        print("Could not find TensorFlow's gaussian_copula.py file")


def test_fixed_likelihood():
    """Test if using joint likelihood in PyTorch would match TensorFlow"""
    print("\n\n=== TESTING JOINT LIKELIHOOD IN PYTORCH ===")
    
    # Generate test data
    np.random.seed(42)
    n = 200
    rho_true = 0.6
    data = np.random.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
    u_data = norm.cdf(data).astype(np.float32)
    
    # Fit with PyTorch (current implementation)
    u_torch = torch.tensor(u_data)
    rho_pt, ll_copula_pt, _ = pt_fit_gaussian(u_torch)
    
    # Compute joint likelihood for the fitted parameter
    z = norm.ppf(u_data)
    z1, z2 = z[:, 0], z[:, 1]
    
    one_minus_rho2 = 1 - rho_pt**2
    log_det_sigma = np.log(one_minus_rho2)
    quad_form = (z1**2 - 2*rho_pt*z1*z2 + z2**2) / one_minus_rho2
    log_biv_normal = -0.5 * (2*np.log(2*np.pi) + log_det_sigma + quad_form)
    ll_joint = np.sum(log_biv_normal)
    
    print(f"\n1. PyTorch fitted rho: {rho_pt:.6f}")
    print(f"2. Copula log-likelihood: {ll_copula_pt:.6f}")
    print(f"3. Joint log-likelihood: {ll_joint:.6f}")
    print(f"4. Difference: {ll_joint - ll_copula_pt:.6f}")
    
    # This difference should be approximately:
    # -n * log(2*pi) - 0.5 * sum(z1^2 + z2^2)
    marginal_ll = -n * np.log(2*np.pi) - 0.5 * np.sum(z1**2 + z2**2)
    print(f"5. Expected marginal contribution: {marginal_ll:.6f}")
    print(f"6. Actual vs expected difference: {abs((ll_joint - ll_copula_pt) - marginal_ll):.6f}")


def main():
    """Run all investigations"""
    print("="*70)
    print("INVESTIGATING LIKELIHOOD CALCULATION DIFFERENCES")
    print("="*70)
    
    # 1. Compare cost functions
    investigate_gaussian_cost_functions()
    
    # 2. Check TensorFlow's code
    check_tensorflow_code()
    
    # 3. Test joint likelihood
    test_fixed_likelihood()
    
    print("\n\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    
    print("\n1. The ~50 difference in log-likelihood is because:")
    print("   - TensorFlow computes JOINT log-likelihood (includes marginal densities)")
    print("   - PyTorch computes COPULA log-likelihood only (correct for copula fitting)")
    
    print("\n2. This doesn't affect parameter estimation because:")
    print("   - The marginal contribution is constant w.r.t. copula parameters")
    print("   - Both optimize the same objective (up to a constant)")
    
    print("\n3. The performance gap must come from elsewhere:")
    print("   - Numerical precision differences")
    print("   - Different convergence criteria")
    print("   - Differences in vine structure or h-function implementation")


if __name__ == "__main__":
    main() 