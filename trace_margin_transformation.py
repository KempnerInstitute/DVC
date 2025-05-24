"""
Trace Margin Transformation Differences

This script focuses on the initial margin transformation step which appears
to be the root cause of performance differences.
"""

import numpy as np
import torch
import tensorflow as tf
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

from scipy.stats import norm

# PyTorch imports
from DVC.vine_model import fit_vine
from DVC import vine_obj_bin, margin_obj

# TensorFlow imports
from DVC_tensorflow.utils.prob_op import kernel_cdf
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj


def compare_margin_transformations():
    """Compare how each implementation transforms raw data to uniform margins"""
    print("\n=== COMPARING MARGIN TRANSFORMATIONS ===")
    
    # Generate test data
    np.random.seed(42)
    n = 200
    raw_data = np.random.normal(0, 1, n).astype(np.float32)
    raw_data = np.sort(raw_data)  # Sort for easier comparison
    
    print(f"\n1. Raw data statistics:")
    print(f"   Shape: {raw_data.shape}")
    print(f"   Range: [{raw_data.min():.4f}, {raw_data.max():.4f}]")
    print(f"   First 10 values: {raw_data[:10]}")
    
    # PyTorch transformation (empirical ranks)
    print("\n2. PyTorch margin transformation (empirical ranks):")
    sorted_vals = torch.sort(torch.tensor(raw_data))[0]
    ranks = torch.searchsorted(sorted_vals, torch.tensor(raw_data)).float() + 1
    u_pytorch = ranks / (n + 1)
    
    print(f"   Range: [{u_pytorch.min():.6f}, {u_pytorch.max():.6f}]")
    print(f"   First 10 values: {u_pytorch[:10].numpy()}")
    print(f"   Last 10 values: {u_pytorch[-10:].numpy()}")
    
    # TensorFlow transformation (kernel CDF)
    print("\n3. TensorFlow margin transformation (kernel CDF):")
    grid = np.linspace(0, 1, 50)
    interp_cdf, _, _ = kernel_cdf(raw_data, raw_data, grid)
    u_tensorflow = interp_cdf.numpy() if hasattr(interp_cdf, 'numpy') else interp_cdf
    
    print(f"   Range: [{u_tensorflow.min():.6f}, {u_tensorflow.max():.6f}]")
    print(f"   First 10 values: {u_tensorflow[:10]}")
    print(f"   Last 10 values: {u_tensorflow[-10:]}")
    
    # Compare
    print("\n4. Differences:")
    diff = np.abs(u_pytorch.numpy() - u_tensorflow)
    print(f"   Max difference: {diff.max():.6f}")
    print(f"   Mean difference: {diff.mean():.6f}")
    print(f"   Std difference: {diff.std():.6f}")
    
    # Check distribution properties
    print("\n5. Distribution properties:")
    from scipy import stats
    
    # KS test for uniformity
    ks_pt, p_pt = stats.kstest(u_pytorch.numpy(), 'uniform')
    ks_tf, p_tf = stats.kstest(u_tensorflow, 'uniform')
    
    print(f"   PyTorch KS test: stat={ks_pt:.4f}, p-value={p_pt:.4f}")
    print(f"   TensorFlow KS test: stat={ks_tf:.4f}, p-value={p_tf:.4f}")
    
    return u_pytorch.numpy(), u_tensorflow, raw_data


def trace_full_vine_fitting():
    """Trace a complete vine fitting to see how margin differences propagate"""
    print("\n\n=== TRACING FULL VINE FITTING ===")
    
    # Generate 4D correlated data
    np.random.seed(42)
    n = 500
    d = 4
    rho = 0.6
    
    # Create correlation matrix
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    print("\n1. True correlation matrix:")
    print(corr)
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    # Create and fit PyTorch vine
    print("\n2. Fitting PyTorch vine...")
    vine_pt = vine_obj_bin(
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
    
    fit_vine(vine_pt, data, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Extract PyTorch parameters
    print("\n3. PyTorch fitted parameters:")
    for level, copulas in enumerate(vine_pt.copulas):
        print(f"   Level {level}:")
        for i, cop in enumerate(copulas):
            if hasattr(cop, 'family') and hasattr(cop, 'theta'):
                print(f"     Edge {i}: {cop.family}, theta={cop.theta:.6f}")
    
    # Create and fit TensorFlow vine
    print("\n4. Fitting TensorFlow vine...")
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
    
    # Extract TensorFlow parameters
    print("\n5. TensorFlow fitted parameters:")
    for level, copulas in enumerate(vine_tf.copulas):
        print(f"   Level {level}:")
        for i, cop in enumerate(copulas):
            if hasattr(cop, 'family') and hasattr(cop, 'theta'):
                print(f"     Edge {i}: {cop.family}, theta={cop.theta:.6f}")
    
    # Test correlation recovery
    print("\n6. Testing correlation recovery...")
    
    # Sample from both vines
    samples_pt = vine_pt.sample(5000)
    samples_tf = vine_tf.sample(5000)
    
    # Compute correlations
    corr_pt = np.corrcoef(samples_pt.T)
    corr_tf = np.corrcoef(samples_tf.T)
    
    print("\n   PyTorch recovered correlation:")
    print(corr_pt)
    
    print("\n   TensorFlow recovered correlation:")
    print(corr_tf)
    
    # Compute MAE
    mae_pt = np.mean(np.abs(corr_pt - corr))
    mae_tf = np.mean(np.abs(corr_tf - corr))
    
    print(f"\n   PyTorch MAE: {mae_pt:.6f}")
    print(f"   TensorFlow MAE: {mae_tf:.6f}")
    
    return vine_pt, vine_tf


def investigate_likelihood_calculation():
    """Deep dive into the log-likelihood calculation differences"""
    print("\n\n=== INVESTIGATING LOG-LIKELIHOOD CALCULATION ===")
    
    # Generate simple 2D data
    np.random.seed(42)
    n = 100
    rho = 0.6
    data = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    u_data = norm.cdf(data).astype(np.float32)
    
    print("\n1. Test data:")
    print(f"   Shape: {u_data.shape}")
    print(f"   True rho: {rho}")
    
    # Manual log-likelihood calculation
    print("\n2. Manual Gaussian copula log-likelihood calculation:")
    
    # Convert to normal scores
    z = norm.ppf(u_data)
    z1, z2 = z[:, 0], z[:, 1]
    
    # For different rho values
    test_rhos = [0.5, 0.6, 0.7]
    
    for test_rho in test_rhos:
        # Copula density (not joint PDF)
        one_minus_rho2 = 1 - test_rho**2
        
        # Log copula density
        log_copula_pdf = -0.5 * np.log(one_minus_rho2) - \
                         ((test_rho**2 * (z1**2 + z2**2) - 2*test_rho*z1*z2) / (2*one_minus_rho2))
        
        # Total log-likelihood
        ll_copula = np.sum(log_copula_pdf)
        
        # Also compute with marginal densities (joint PDF)
        log_joint_pdf = log_copula_pdf - 0.5 * (z1**2 + z2**2) - np.log(2*np.pi)
        ll_joint = np.sum(log_joint_pdf)
        
        print(f"\n   Rho = {test_rho}:")
        print(f"     Log-likelihood (copula only): {ll_copula:.6f}")
        print(f"     Log-likelihood (joint PDF): {ll_joint:.6f}")
        print(f"     AIC (copula): {2 - 2*ll_copula:.6f}")
    
    # Check what PyTorch is computing
    print("\n3. PyTorch computation:")
    from DVC.param_copula import fit_gaussian
    u_torch = torch.tensor(u_data)
    rho_fit, ll_fit, aic_fit = fit_gaussian(u_torch)
    print(f"   Fitted rho: {rho_fit:.6f}")
    print(f"   Log-likelihood: {ll_fit:.6f}")
    print(f"   AIC: {aic_fit:.6f}")
    
    # The issue might be that TensorFlow includes marginal densities
    # while PyTorch correctly computes only copula density


def main():
    """Run all investigations"""
    print("="*70)
    print("DEEP DIVE: MARGIN TRANSFORMATION AND LIKELIHOOD DIFFERENCES")
    print("="*70)
    
    # 1. Compare margin transformations
    u_pt, u_tf, raw_data = compare_margin_transformations()
    
    # 2. Trace full vine fitting
    vine_pt, vine_tf = trace_full_vine_fitting()
    
    # 3. Investigate likelihood calculation
    investigate_likelihood_calculation()
    
    print("\n\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    print("\n1. MARGIN TRANSFORMATION:")
    print("   - PyTorch uses empirical ranks: i/(n+1)")
    print("   - TensorFlow uses kernel CDF estimation")
    print("   - This creates small but systematic differences")
    
    print("\n2. LOG-LIKELIHOOD CALCULATION:")
    print("   - The 50+ difference suggests different formulations")
    print("   - PyTorch: Copula density only (correct)")
    print("   - TensorFlow: Might include marginal densities")
    
    print("\n3. IMPACT ON VINE:")
    print("   - Small margin differences compound through vine levels")
    print("   - Different likelihood affects parameter estimation")
    print("   - Results in poor correlation recovery for non-adjacent variables")
    
    print("\nRECOMMENDATIONS:")
    print("1. Implement TensorFlow's kernel CDF margin transformation in PyTorch")
    print("2. Verify likelihood calculation matches between implementations")
    print("3. Ensure identical numerical precision in all operations")


if __name__ == "__main__":
    main() 