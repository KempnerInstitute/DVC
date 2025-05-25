"""
Deep Dive into D-Vine Theta Propagation

This script analyzes exactly how theta values flow through a D-vine
and identifies the key differences between PyTorch and TensorFlow implementations.
"""

import numpy as np
import torch
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

# PyTorch imports
from DVC import vine_obj_bin, margin_obj
from DVC.vine_model import fit_vine, _h_function
from DVC.objects import cop_par_obj


def manual_d_vine_propagation():
    """Manually trace theta propagation in a D-vine"""
    print("="*80)
    print("MANUAL D-VINE THETA PROPAGATION")
    print("="*80)
    
    # Create a simple 4D example
    np.random.seed(42)
    d = 4
    
    # Create correlated data
    rho = 0.6
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    # Generate one sample
    sample = np.random.multivariate_normal(np.zeros(d), corr, 1).flatten()
    print(f"\nOriginal sample: {sample}")
    
    # Convert to uniform margins
    from scipy.stats import norm
    u = norm.cdf(sample)
    print(f"Uniform margins: {u}")
    
    # Initialize theta matrix
    theta = np.zeros((d, d))
    theta_flip = np.zeros((d, d))
    
    # Level 0: Original margins
    theta[0, :] = u
    print(f"\nLevel 0 theta: {theta[0, :]}")
    
    # D-vine structure for 4 variables:
    # Level 0: (0,1), (1,2), (2,3)
    # Level 1: (0,2|1), (1,3|2)
    # Level 2: (0,3|1,2)
    
    # Assume Gaussian copulas with parameter 0.5
    rho_cop = 0.5
    
    print(f"\n--- LEVEL 1 PROPAGATION ---")
    print(f"Edges: (0,1), (1,2), (2,3)")
    
    # Edge (0,1): Compute h-functions
    u0, u1 = theta[0, 0], theta[0, 1]
    h_1_0 = h_function_gaussian(u1, u0, rho_cop, 'left')   # h(u1|u0)
    h_0_1 = h_function_gaussian(u0, u1, rho_cop, 'right')  # h(u0|u1)
    
    print(f"\nEdge (0,1): u0={u0:.4f}, u1={u1:.4f}")
    print(f"  h(u1|u0) = {h_1_0:.4f} -> stores in theta[1, 1]")
    print(f"  h(u0|u1) = {h_0_1:.4f} -> stores in theta_flip[1, 0]")
    
    theta[1, 1] = h_1_0
    theta_flip[1, 0] = h_0_1
    
    # Edge (1,2): Compute h-functions
    u1, u2 = theta[0, 1], theta[0, 2]
    h_2_1 = h_function_gaussian(u2, u1, rho_cop, 'left')   # h(u2|u1)
    h_1_2 = h_function_gaussian(u1, u2, rho_cop, 'right')  # h(u1|u2)
    
    print(f"\nEdge (1,2): u1={u1:.4f}, u2={u2:.4f}")
    print(f"  h(u2|u1) = {h_2_1:.4f} -> stores in theta[1, 2]")
    print(f"  h(u1|u2) = {h_1_2:.4f} -> stores in theta_flip[1, 1]")
    
    theta[1, 2] = h_2_1
    theta_flip[1, 1] = h_1_2
    
    # Edge (2,3): Compute h-functions
    u2, u3 = theta[0, 2], theta[0, 3]
    h_3_2 = h_function_gaussian(u3, u2, rho_cop, 'left')   # h(u3|u2)
    h_2_3 = h_function_gaussian(u2, u3, rho_cop, 'right')  # h(u2|u3)
    
    print(f"\nEdge (2,3): u2={u2:.4f}, u3={u3:.4f}")
    print(f"  h(u3|u2) = {h_3_2:.4f} -> stores in theta[1, 3]")
    print(f"  h(u2|u3) = {h_2_3:.4f} -> stores in theta_flip[1, 2]")
    
    theta[1, 3] = h_3_2
    theta_flip[1, 2] = h_2_3
    
    print(f"\nLevel 1 theta:      {theta[1, :]}")
    print(f"Level 1 theta_flip: {theta_flip[1, :]}")
    
    print(f"\n--- LEVEL 2 PROPAGATION ---")
    print(f"Edges: (0,2|1), (1,3|2)")
    
    # Edge (0,2|1): Need h(u0|u1) and h(u2|u1)
    # We have theta_flip[1, 0] = h(u0|u1) and theta[1, 2] = h(u2|u1)
    v0 = theta_flip[1, 0]  # h(u0|u1)
    v2 = theta[1, 2]       # h(u2|u1)
    
    h_2_0 = h_function_gaussian(v2, v0, rho_cop, 'left')   # h(v2|v0)
    h_0_2 = h_function_gaussian(v0, v2, rho_cop, 'right')  # h(v0|v2)
    
    print(f"\nEdge (0,2|1): v0={v0:.4f}, v2={v2:.4f}")
    print(f"  h(v2|v0) = {h_2_0:.4f} -> stores in theta[2, 2]")
    print(f"  h(v0|v2) = {h_0_2:.4f} -> stores in theta_flip[2, 0]")
    
    theta[2, 2] = h_2_0
    theta_flip[2, 0] = h_0_2
    
    # Edge (1,3|2): Need h(u1|u2) and h(u3|u2)
    # We have theta_flip[1, 1] = h(u1|u2) and theta[1, 3] = h(u3|u2)
    v1 = theta_flip[1, 1]  # h(u1|u2)
    v3 = theta[1, 3]       # h(u3|u2)
    
    h_3_1 = h_function_gaussian(v3, v1, rho_cop, 'left')   # h(v3|v1)
    h_1_3 = h_function_gaussian(v1, v3, rho_cop, 'right')  # h(v1|v3)
    
    print(f"\nEdge (1,3|2): v1={v1:.4f}, v3={v3:.4f}")
    print(f"  h(v3|v1) = {h_3_1:.4f} -> stores in theta[2, 3]")
    print(f"  h(v1|v3) = {h_1_3:.4f} -> stores in theta_flip[2, 1]")
    
    theta[2, 3] = h_3_1
    theta_flip[2, 1] = h_1_3
    
    print(f"\nLevel 2 theta:      {theta[2, :]}")
    print(f"Level 2 theta_flip: {theta_flip[2, :]}")
    
    return theta, theta_flip


def h_function_gaussian(u, v, rho, direction='left'):
    """
    Compute h-function for Gaussian copula
    
    direction='left': h(v|u) = P(V <= v | U = u)
    direction='right': h(u|v) = P(U <= u | V = v)
    """
    from scipy.stats import norm
    
    # Clamp to avoid numerical issues
    u = np.clip(u, 1e-9, 1-1e-9)
    v = np.clip(v, 1e-9, 1-1e-9)
    
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


def analyze_pytorch_theta_update():
    """Analyze how PyTorch updates theta in vine_model.py"""
    print("\n" + "="*80)
    print("PYTORCH THETA UPDATE MECHANISM")
    print("="*80)
    
    print("\nFrom vine_model.py lines 683-695:")
    print("```python")
    print("# main direction - conditional CDF of u_j given u_i")
    print("vine.theta[:, next_level, j] = _h_function(u_i, u_j, cobj_now, vine.grid_u, side='left')")
    print("# flipped direction - conditional CDF of u_i given u_j")
    print("vine.theta_flip[:, next_level, i] = _h_function(u_j, u_i, cobj_now, vine.grid_u, side='right')")
    print("```")
    
    print("\nKey observation:")
    print("- For edge (i,j), PyTorch stores h(u_j|u_i) in theta[next_level, j]")
    print("- And h(u_i|u_j) in theta_flip[next_level, i]")
    print("- This means the SECOND variable index determines storage location")
    
    print("\nExample for edge (0,1):")
    print("- h(u1|u0) -> theta[1, 1]")
    print("- h(u0|u1) -> theta_flip[1, 0]")


def test_pytorch_d_vine():
    """Test PyTorch D-vine with known data"""
    print("\n" + "="*80)
    print("TESTING PYTORCH D-VINE")
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
    
    # Analyze first sample
    print("\nFirst sample analysis:")
    sample_idx = 0
    
    print(f"Original data: {data[sample_idx]}")
    
    # Check theta values
    theta_pt = vine_pt.theta[sample_idx].cpu().numpy()
    theta_flip_pt = vine_pt.theta_flip[sample_idx].cpu().numpy()
    
    print("\nTheta matrix:")
    for level in range(d):
        print(f"Level {level}: {theta_pt[level]}")
    
    print("\nTheta_flip matrix:")
    for level in range(d):
        print(f"Level {level}: {theta_flip_pt[level]}")
    
    # Check which positions get updated
    print("\nNon-zero positions in theta:")
    for level in range(d):
        non_zero = np.where(theta_pt[level] != 0)[0]
        print(f"Level {level}: positions {non_zero}")
    
    print("\nNon-zero positions in theta_flip:")
    for level in range(d):
        non_zero = np.where(theta_flip_pt[level] != 0)[0]
        print(f"Level {level}: positions {non_zero}")
    
    # Manually verify some h-functions
    print("\n--- MANUAL VERIFICATION ---")
    
    # Get copula parameters
    print("\nCopula parameters:")
    for level, copulas in enumerate(vine_pt.copulas):
        print(f"Level {level}:")
        for i, cop in enumerate(copulas):
            print(f"  Edge {vine_pt.ind_vine[level][i]}: rho = {cop.theta:.4f}")
    
    # Verify edge (0,1)
    u0 = theta_pt[0, 0]
    u1 = theta_pt[0, 1]
    rho_01 = vine_pt.copulas[0][0].theta
    
    h_1_0_manual = h_function_gaussian(u1, u0, rho_01, 'left')
    h_0_1_manual = h_function_gaussian(u0, u1, rho_01, 'right')
    
    print(f"\nEdge (0,1) verification:")
    print(f"  PyTorch theta[1,1] = {theta_pt[1,1]:.6f}")
    print(f"  Manual h(u1|u0) = {h_1_0_manual:.6f}")
    print(f"  Difference: {abs(theta_pt[1,1] - h_1_0_manual):.6f}")
    
    print(f"\n  PyTorch theta_flip[1,0] = {theta_flip_pt[1,0]:.6f}")
    print(f"  Manual h(u0|u1) = {h_0_1_manual:.6f}")
    print(f"  Difference: {abs(theta_flip_pt[1,0] - h_0_1_manual):.6f}")


def explain_d_vine_indexing():
    """Explain the critical indexing issue in D-vines"""
    print("\n" + "="*80)
    print("D-VINE INDEXING EXPLANATION")
    print("="*80)
    
    print("\nThe key issue: How do we interpret edges at higher levels?")
    
    print("\nFor a 4D D-vine:")
    print("Level 0: edges = [[0,1], [1,2], [2,3]]")
    print("Level 1: edges = [[0,2], [1,3]]")
    print("Level 2: edges = [[0,3]]")
    
    print("\nAt Level 1, edge [0,2] could mean:")
    print("1. Variables 0 and 2 conditioned on their common neighbor (1)")
    print("2. Reference to edges 0 and 2 from Level 0")
    
    print("\nThe PyTorch code (lines 409-420) has this logic:")
    print("```python")
    print("if edge[0] < prev_len and edge[1] < prev_len:")
    print("    # Treat as edge references")
    print("else:")
    print("    # Treat as variable indices")
    print("```")
    
    print("\nThis is the source of confusion!")


if __name__ == "__main__":
    # Run analyses
    manual_theta, manual_theta_flip = manual_d_vine_propagation()
    analyze_pytorch_theta_update()
    test_pytorch_d_vine()
    explain_d_vine_indexing() 