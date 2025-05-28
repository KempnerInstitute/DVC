"""
Deep Analysis of Vine Tree Mechanics

This script provides a detailed analysis of how vine trees work,
how theta values are computed and propagated, and how the h-function
transforms conditional distributions through the vine levels.
"""

import numpy as np
import torch
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

# PyTorch imports
from DVC_pyolder import vine_obj_bin, margin_obj
from DVC_pyolder.vine_model import fit_vine, _h_function
from DVC_pyolder.objects import cop_par_obj

# TensorFlow imports
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj
import tensorflow as tf


def explain_vine_structure():
    """Explain the D-vine structure conceptually"""
    print("="*80)
    print("UNDERSTANDING D-VINE STRUCTURE")
    print("="*80)
    
    print("\nFor a 4-dimensional D-vine:")
    print("Variables: X1, X2, X3, X4")
    
    print("\nLevel 0 (Tree 1): Direct dependencies")
    print("  Edges: (1,2), (2,3), (3,4)")
    print("  These model direct correlations between adjacent variables")
    
    print("\nLevel 1 (Tree 2): Conditional dependencies given 1 variable")
    print("  Edges: (1,3|2), (2,4|3)")
    print("  These model correlations between variables separated by 1, given the intermediate")
    
    print("\nLevel 2 (Tree 3): Conditional dependencies given 2 variables")
    print("  Edges: (1,4|2,3)")
    print("  These model correlations between the extremes, given the middle variables")
    
    print("\n" + "="*80)


def analyze_theta_storage():
    """Analyze how theta values are stored in the matrix"""
    print("\nTHETA MATRIX STRUCTURE")
    print("="*80)
    
    print("\nTheta matrix has shape [N, d, d] where:")
    print("  N = number of samples")
    print("  d = number of variables")
    
    print("\nStorage pattern:")
    print("  theta[i, level, var] = transformed value for sample i at given level and variable")
    
    print("\nLevel 0: Original uniform margins (U1, U2, U3, U4)")
    print("Level 1: After first h-function transformation")
    print("Level 2: After second h-function transformation")
    print("Level 3: After third h-function transformation")


def trace_single_sample():
    """Trace how a single sample flows through the vine"""
    print("\n" + "="*80)
    print("TRACING A SINGLE SAMPLE THROUGH D-VINE")
    print("="*80)
    
    # Create a simple 4D sample
    np.random.seed(42)
    d = 4
    
    # Create a sample with known correlations
    rho = 0.7
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    # Generate one sample
    sample = np.random.multivariate_normal(np.zeros(d), corr, 1).flatten()
    print(f"\nOriginal sample (normal scale): {sample}")
    
    # Convert to uniform margins
    from scipy.stats import norm
    u_sample = norm.cdf(sample)
    print(f"Uniform margins: {u_sample}")
    
    # Simulate D-vine transformations
    print("\n--- LEVEL 0 (Original margins) ---")
    theta = np.zeros((d, d))
    theta[0, :] = u_sample
    print(f"theta[0,:] = {theta[0,:]}")
    
    # Create Gaussian copulas for demonstration
    rho_12 = 0.7
    rho_23 = 0.7
    rho_34 = 0.7
    
    print(f"\nCopula parameters:")
    print(f"  C(1,2): rho = {rho_12}")
    print(f"  C(2,3): rho = {rho_23}")
    print(f"  C(3,4): rho = {rho_34}")
    
    # Level 1 transformations
    print("\n--- LEVEL 1 (After first h-functions) ---")
    
    # For edge (1,2): compute h(u1|u2) and h(u2|u1)
    u1, u2 = u_sample[0], u_sample[1]
    h_1_2 = h_function_gaussian(u1, u2, rho_12, direction='right')  # h(u1|u2)
    h_2_1 = h_function_gaussian(u2, u1, rho_12, direction='left')   # h(u2|u1)
    
    # For edge (2,3): compute h(u2|u3) and h(u3|u2)
    u3 = u_sample[2]
    h_2_3 = h_function_gaussian(u2, u3, rho_23, direction='right')  # h(u2|u3)
    h_3_2 = h_function_gaussian(u3, u2, rho_23, direction='left')   # h(u3|u2)
    
    # For edge (3,4): compute h(u3|u4) and h(u4|u3)
    u4 = u_sample[3]
    h_3_4 = h_function_gaussian(u3, u4, rho_34, direction='right')  # h(u3|u4)
    h_4_3 = h_function_gaussian(u4, u3, rho_34, direction='left')   # h(u4|u3)
    
    print(f"h(u1|u2) = {h_1_2:.4f}")
    print(f"h(u2|u1) = {h_2_1:.4f}")
    print(f"h(u2|u3) = {h_2_3:.4f}")
    print(f"h(u3|u2) = {h_3_2:.4f}")
    print(f"h(u3|u4) = {h_3_4:.4f}")
    print(f"h(u4|u3) = {h_4_3:.4f}")
    
    # Now the key question: how are these stored in theta?
    print("\nTHETA STORAGE PATTERNS:")
    print("PyTorch pattern vs TensorFlow pattern (hypothetical)")


def h_function_gaussian(u, v, rho, direction='left'):
    """
    Compute h-function for Gaussian copula
    
    direction='left': h(v|u) = P(V <= v | U = u)
    direction='right': h(u|v) = P(U <= u | V = v)
    """
    from scipy.stats import norm
    
    # Convert to normal scale
    x = norm.ppf(u)
    y = norm.ppf(v)
    
    if direction == 'left':
        # h(v|u) = Phi((y - rho*x) / sqrt(1-rho^2))
        z = (y - rho * x) / np.sqrt(1 - rho**2)
    else:
        # h(u|v) = Phi((x - rho*y) / sqrt(1-rho^2))
        z = (x - rho * y) / np.sqrt(1 - rho**2)
    
    return norm.cdf(z)


def compare_vine_structures():
    """Compare how PyTorch and TensorFlow structure their vines"""
    print("\n" + "="*80)
    print("COMPARING VINE STRUCTURES")
    print("="*80)
    
    # Generate test data
    np.random.seed(42)
    n = 100
    d = 4
    
    # Create correlated data
    rho = 0.6
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    # Fit PyTorch vine
    print("\n--- PYTORCH VINE STRUCTURE ---")
    vine_pt = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian'],
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
    par_dict = {"param_families": ["gaussian"]}
    npc_dict = {}
    bin_dict = {"n_bin": 1}
    
    fit_vine(vine_pt, data, gen_dict, npc_dict, par_dict, bin_dict)
    
    print("PyTorch vine edges by level:")
    for level, edges in enumerate(vine_pt.ind_vine):
        print(f"  Level {level}: {edges}")
    
    # Fit TensorFlow vine
    print("\n--- TENSORFLOW VINE STRUCTURE ---")
    margins_tf = []
    for i in range(d):
        margin = tf_margin_obj('norm', [0.0, 1.0], True)
        margin.ker = data[:, i]
        margins_tf.append(margin)
    
    vine_tf = tf_vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian'],
        vine_depth=d,
        margin=margins_tf,
        knots=50,
        method='matrix'
    )
    
    gen_dict_tf = {"parallel": False, "param": True, "binning": False, "fitted": False, "vine_depth": d}
    par_dict_tf = {"param_families": ["gaussian"]}
    npc_dict_tf = {"opt_method": "local", "batch_paral": False}
    bin_dict_tf = {"n_bin": 1}
    
    vine_tf.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
    
    print("TensorFlow vine edges:")
    print(f"  Level 0: {vine_tf.E[0]}")
    print(f"  Level 1: {vine_tf.E[1]}")
    print(f"  Level 2: {vine_tf.E[2]}")
    
    # Analyze theta propagation for first sample
    print("\n--- THETA PROPAGATION ANALYSIS ---")
    print("First sample theta values:\n")
    
    theta_pt = vine_pt.theta[0].cpu().numpy()
    theta_tf = vine_tf.theta.numpy()[0] if hasattr(vine_tf.theta, 'numpy') else vine_tf.theta[0]
    
    print("PyTorch theta[0]:")
    for level in range(d):
        print(f"  Level {level}: {theta_pt[level]}")
    
    print("\nTensorFlow theta[0]:")
    for level in range(d):
        print(f"  Level {level}: {theta_tf[level]}")
    
    # Track which variables get updated at each level
    print("\n--- VARIABLE UPDATE PATTERN ---")
    print("\nPyTorch updates:")
    for level in range(1, d):
        updated = np.where(theta_pt[level] != theta_pt[level-1])[0]
        print(f"  Level {level}: Variables {updated} were updated")
    
    print("\nTensorFlow updates:")
    for level in range(1, d):
        updated = np.where(theta_tf[level] != 0)[0]
        print(f"  Level {level}: Variables {updated} have non-zero values")


def test_h_function_symmetry():
    """Test h-function properties and symmetry"""
    print("\n" + "="*80)
    print("H-FUNCTION PROPERTIES TEST")
    print("="*80)
    
    # Test values
    u1, u2 = 0.3, 0.7
    rho = 0.5
    
    # Compute both directions
    h_left = h_function_gaussian(u1, u2, rho, direction='left')   # h(u2|u1)
    h_right = h_function_gaussian(u1, u2, rho, direction='right')  # h(u1|u2)
    
    print(f"\nFor u1={u1}, u2={u2}, rho={rho}:")
    print(f"  h(u2|u1) = {h_left:.6f}")
    print(f"  h(u1|u2) = {h_right:.6f}")
    
    # Test independence case
    h_left_ind = h_function_gaussian(u1, u2, 0.0, direction='left')
    h_right_ind = h_function_gaussian(u1, u2, 0.0, direction='right')
    
    print(f"\nFor independence (rho=0):")
    print(f"  h(u2|u1) = {h_left_ind:.6f} (should be close to u2={u2})")
    print(f"  h(u1|u2) = {h_right_ind:.6f} (should be close to u1={u1})")
    
    # Test perfect correlation
    h_left_perf = h_function_gaussian(u1, u2, 0.999, direction='left')
    h_right_perf = h_function_gaussian(u1, u2, 0.999, direction='right')
    
    print(f"\nFor perfect correlation (rho=0.999):")
    print(f"  h(u2|u1) = {h_left_perf:.6f}")
    print(f"  h(u1|u2) = {h_right_perf:.6f}")


def analyze_d_vine_sampling():
    """Analyze how D-vine sampling should work"""
    print("\n" + "="*80)
    print("D-VINE SAMPLING PROCESS")
    print("="*80)
    
    print("\nFor a 4D D-vine, sampling works as follows:")
    print("\n1. Sample V1 ~ U(0,1)")
    print("2. Sample V2 from C(1,2) given V1")
    print("3. Sample V3 from vine structure:")
    print("   - First get h(V2|V1) using copula C(1,2)")
    print("   - Then sample from C(2,3|1) using h(V2|V1) and transform back")
    print("4. Sample V4 from vine structure:")
    print("   - Get h(V3|V2) using copula C(2,3)")
    print("   - Get h(h(V2|V1)|h(V3|V2)) using copula C(1,3|2)")
    print("   - Sample from C(3,4|1,2) and transform back")
    
    print("\nThe key insight: Each variable's sampling depends on")
    print("the h-transformations of previous variables!")


if __name__ == "__main__":
    # Run all analyses
    explain_vine_structure()
    analyze_theta_storage()
    trace_single_sample()
    test_h_function_symmetry()
    compare_vine_structures()
    analyze_d_vine_sampling() 