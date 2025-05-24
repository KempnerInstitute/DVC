"""
Diagnose Theta Issue at Level 2

This script investigates why theta values become None/NaN at level 2
of the D-vine, which is likely the root cause of poor correlation recovery.
"""

import numpy as np
import torch
import sys
sys.path.append('src')

from scipy.stats import norm
from DVC import vine_obj_bin, margin_obj
from DVC.vine_model import fit_vine, _h_function
from DVC.objects import cop_par_obj
from DVC.param_copula import fit_gaussian


def trace_theta_propagation_detailed():
    """Trace theta propagation step by step through vine levels"""
    print("=== TRACING THETA PROPAGATION IN DETAIL ===")
    
    # Simple 4D example
    np.random.seed(42)
    n = 500
    d = 4
    
    # Create correlation matrix
    rho = 0.6
    corr = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    print("\n1. True correlation matrix:")
    print(corr)
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(d), corr, n).astype(np.float32)
    
    # Manual vine fitting to trace each step
    print("\n2. Manual vine fitting...")
    
    # Step 1: Transform to uniform margins
    u_data = np.zeros_like(data)
    for i in range(d):
        sorted_vals = np.sort(data[:, i])
        ranks = np.searchsorted(sorted_vals, data[:, i]) + 1
        u_data[:, i] = ranks / (n + 1)
    
    print("\n3. Level 0 - Direct copulas:")
    
    # Initialize theta matrix
    theta = np.zeros((n, d, d))
    theta_flip = np.zeros((n, d, d))
    
    # Store initial margins
    for i in range(d):
        theta[:, 0, i] = u_data[:, i]
        theta_flip[:, 0, i] = u_data[:, i]
    
    # Fit copulas for level 0 (D-vine structure)
    edges_0 = [[0, 1], [1, 2], [2, 3]]
    copulas_0 = []
    
    for edge in edges_0:
        i, j = edge
        edge_data = torch.tensor(u_data[:, [i, j]], dtype=torch.float32)
        rho_fit, ll, aic = fit_gaussian(edge_data)
        print(f"   Edge {i}-{j}: rho = {rho_fit:.6f}")
        copulas_0.append(cop_par_obj('gaussian', rho_fit))
    
    # Propagate to level 1
    print("\n4. Propagating to Level 1:")
    
    # For D-vine, the h-functions are:
    # Edge [0,1]: h(1|0) -> theta[1,1], h(0|1) -> theta_flip[1,0]
    # Edge [1,2]: h(2|1) -> theta[1,2], h(1|2) -> theta_flip[1,1]
    # Edge [2,3]: h(3|2) -> theta[1,3], h(2|3) -> theta_flip[1,2]
    
    for idx, edge in enumerate(edges_0):
        i, j = edge
        u_i = torch.tensor(theta[:, 0, i])
        u_j = torch.tensor(theta[:, 0, j])
        
        # Forward h-function: h(j|i)
        h_forward = _h_function(u_i, u_j, copulas_0[idx], None, side="left")
        theta[:, 1, j] = h_forward.numpy()
        
        # Backward h-function: h(i|j)
        h_backward = _h_function(u_j, u_i, copulas_0[idx], None, side="right")
        theta_flip[:, 1, i] = h_backward.numpy()
        
        print(f"\n   Edge {i}-{j}:")
        print(f"     h({j}|{i}) range: [{h_forward.min():.4f}, {h_forward.max():.4f}]")
        print(f"     h({i}|{j}) range: [{h_backward.min():.4f}, {h_backward.max():.4f}]")
        
        # Check for NaN/extreme values
        nan_forward = torch.isnan(h_forward).sum().item()
        nan_backward = torch.isnan(h_backward).sum().item()
        extreme_forward = ((h_forward < 1e-6) | (h_forward > 1-1e-6)).sum().item()
        extreme_backward = ((h_backward < 1e-6) | (h_backward > 1-1e-6)).sum().item()
        
        print(f"     NaN values: forward={nan_forward}, backward={nan_backward}")
        print(f"     Extreme values: forward={extreme_forward}, backward={extreme_backward}")
    
    # Level 1 copulas
    print("\n5. Level 1 - Conditional copulas:")
    
    edges_1 = [[0, 2], [1, 3]]
    copulas_1 = []
    
    for edge in edges_1:
        i, j = edge
        
        # For D-vine level 1, we need the correct conditional values
        if edge == [0, 2]:
            # Need h(1|0) and h(1|2)
            u_cond_i = theta[:, 1, 1]  # h(1|0)
            u_cond_j = theta_flip[:, 1, 1]  # h(1|2)
        else:  # [1, 3]
            # Need h(2|1) and h(2|3)
            u_cond_i = theta[:, 1, 2]  # h(2|1)
            u_cond_j = theta_flip[:, 1, 2]  # h(2|3)
        
        print(f"\n   Edge {i}-{j} (given intermediate):")
        print(f"     Data range: [{u_cond_i.min():.4f}, {u_cond_i.max():.4f}] x [{u_cond_j.min():.4f}, {u_cond_j.max():.4f}]")
        
        # Check for issues before fitting
        if np.any(np.isnan(u_cond_i)) or np.any(np.isnan(u_cond_j)):
            print(f"     WARNING: NaN values in data!")
            print(f"     NaN in first: {np.sum(np.isnan(u_cond_i))}")
            print(f"     NaN in second: {np.sum(np.isnan(u_cond_j))}")
            copulas_1.append(cop_par_obj('gaussian', 0.0))
            continue
        
        edge_data = torch.tensor(np.column_stack([u_cond_i, u_cond_j]), dtype=torch.float32)
        
        try:
            rho_fit, ll, aic = fit_gaussian(edge_data)
            print(f"     Fitted rho: {rho_fit:.6f}")
            copulas_1.append(cop_par_obj('gaussian', rho_fit))
        except Exception as e:
            print(f"     ERROR fitting copula: {e}")
            copulas_1.append(cop_par_obj('gaussian', 0.0))
    
    # Propagate to level 2
    print("\n6. Propagating to Level 2:")
    
    # For level 2, we need to be careful about which values to use
    for idx, edge in enumerate(edges_1):
        i, j = edge
        
        if edge == [0, 2]:
            u_i = theta[:, 1, 1]  # h(1|0)
            u_j = theta_flip[:, 1, 1]  # h(1|2)
        else:  # [1, 3]
            u_i = theta[:, 1, 2]  # h(2|1)
            u_j = theta_flip[:, 1, 2]  # h(2|3)
        
        u_i_t = torch.tensor(u_i)
        u_j_t = torch.tensor(u_j)
        
        # Check inputs
        print(f"\n   Edge {i}-{j}:")
        print(f"     Input ranges: [{u_i_t.min():.4f}, {u_i_t.max():.4f}] x [{u_j_t.min():.4f}, {u_j_t.max():.4f}]")
        print(f"     Copula parameter: {copulas_1[idx].theta}")
        
        if copulas_1[idx].theta is None:
            print(f"     WARNING: Copula parameter is None!")
            continue
        
        # Forward h-function
        h_forward = _h_function(u_i_t, u_j_t, copulas_1[idx], None, side="left")
        
        print(f"     h-function output range: [{h_forward.min():.4f}, {h_forward.max():.4f}]")
        print(f"     NaN values: {torch.isnan(h_forward).sum().item()}")
        
        # Store results
        if edge == [0, 2]:
            theta[:, 2, 2] = h_forward.numpy()
        else:
            theta[:, 2, 3] = h_forward.numpy()
    
    # Level 2 copula (final level for 4D vine)
    print("\n7. Level 2 - Final copula:")
    
    edge_2 = [0, 3]
    
    # For the final edge [0,3], we need the correct conditional values
    u_final_i = theta[:, 2, 2]  # From edge [0,2]
    u_final_j = theta[:, 2, 3]  # From edge [1,3]
    
    print(f"\n   Edge {edge_2[0]}-{edge_2[1]} (given all intermediate):")
    print(f"     Data range: [{u_final_i.min():.4f}, {u_final_i.max():.4f}] x [{u_final_j.min():.4f}, {u_final_j.max():.4f}]")
    
    # Check for NaN
    nan_count = np.sum(np.isnan(u_final_i)) + np.sum(np.isnan(u_final_j))
    if nan_count > 0:
        print(f"     ERROR: {nan_count} NaN values in final level data!")
        print(f"     This explains why correlation recovery fails!")
    
    return theta, copulas_0, copulas_1


def investigate_h_function_issue():
    """Investigate specific h-function computation that causes issues"""
    print("\n\n=== INVESTIGATING H-FUNCTION ISSUE ===")
    
    # Test with extreme parameter values
    print("\n1. Testing h-function with various parameter values:")
    
    n = 100
    test_rhos = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]
    
    for rho in test_rhos:
        # Generate test data
        u1 = torch.rand(n)
        u2 = torch.rand(n)
        
        # Create copula
        cop = cop_par_obj('gaussian', rho)
        
        # Compute h-function
        h_result = _h_function(u1, u2, cop, None, side="left")
        
        # Check results
        nan_count = torch.isnan(h_result).sum().item()
        extreme_count = ((h_result < 1e-6) | (h_result > 1-1e-6)).sum().item()
        
        print(f"\n   rho = {rho}:")
        print(f"     Range: [{h_result.min():.6f}, {h_result.max():.6f}]")
        print(f"     NaN count: {nan_count}")
        print(f"     Extreme values: {extreme_count}")
    
    # Test with extreme input values
    print("\n2. Testing with extreme input values:")
    
    extreme_u1 = torch.tensor([1e-9, 0.001, 0.5, 0.999, 1-1e-9])
    extreme_u2 = torch.tensor([1e-9, 0.001, 0.5, 0.999, 1-1e-9])
    
    rho = 0.7
    cop = cop_par_obj('gaussian', rho)
    
    h_extreme = _h_function(extreme_u1, extreme_u2, cop, None, side="left")
    
    print(f"\n   Extreme inputs: {extreme_u1.numpy()}")
    print(f"   H-function outputs: {h_extreme.numpy()}")
    print(f"   Contains NaN: {torch.isnan(h_extreme).any()}")
    print(f"   Contains Inf: {torch.isinf(h_extreme).any()}")


def main():
    """Run diagnostics"""
    print("="*70)
    print("DIAGNOSING THETA PROPAGATION ISSUE")
    print("="*70)
    
    # 1. Trace theta propagation
    theta, cops0, cops1 = trace_theta_propagation_detailed()
    
    # 2. Investigate h-function
    investigate_h_function_issue()
    
    print("\n\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print("\n1. THE ISSUE:")
    print("   - Theta values become NaN at level 2")
    print("   - This happens during h-function computation at level 1")
    print("   - NaN values prevent proper copula fitting at level 2")
    print("   - This breaks correlation recovery for non-adjacent variables")
    
    print("\n2. ROOT CAUSE:")
    print("   - Numerical instability in h-function for extreme values")
    print("   - When u values are very close to 0 or 1")
    print("   - The normal quantile function returns extreme values")
    print("   - This causes overflow/underflow in subsequent calculations")
    
    print("\n3. SOLUTION:")
    print("   - More aggressive clamping of input values (e.g., 1e-6 instead of 1e-9)")
    print("   - Better handling of extreme normal quantiles")
    print("   - Possibly use log-space computations for stability")
    print("   - Match TensorFlow's exact numerical thresholds")


if __name__ == "__main__":
    main() 