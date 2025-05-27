"""
Debug script to trace the sampling issue in DVC PyTorch
"""

import numpy as np
import torch
from classes.objects import vine_obj_bin, margin_obj
from sampling import VineSampler


def debug_sampling():
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create simple 3D correlated data
    n_samples = 1000
    d = 3
    
    # Strong correlations to make the issue obvious
    rho = 0.8
    corr_matrix = np.array([[1.0, rho, rho*0.7],
                           [rho, 1.0, rho*0.9],
                           [rho*0.7, rho*0.9, 1.0]])
    
    print("Target correlation matrix:")
    print(corr_matrix)
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(d), corr_matrix, n_samples)
    data = torch.tensor(data, dtype=torch.float32)
    
    print("\nOriginal data statistics:")
    print(f"Means: {data.mean(axis=0)}")
    print(f"Stds: {data.std(axis=0)}")
    
    # Create and fit vine
    margins = [margin_obj('norm', [0, 1], True) for _ in range(d)]
    
    vine = vine_obj_bin(
        vine_family='c-vine',
        families=['gaussian', 'ind'],
        vine_depth=d-1,
        margin=margins,
        knots=50,
        method='matrix'
    )
    
    gen_dict = {
        "parallel": False,
        "param": True,
        "binning": False,
        "fitted": False,
        "vine_depth": d-1
    }
    par_dict = {"param_families": ["gaussian", "ind"]}
    npc_dict = {"opt_method": "LL1", "batch_paral": False}
    bin_dict = {"n_bin": 1}
    
    print("\nFitting vine...")
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Check fitted copulas
    print("\nFitted copulas:")
    for tr in range(len(vine.copulas)):
        print(f"Tree {tr}:")
        for j, cop in enumerate(vine.copulas[tr]):
            print(f"  Copula {j}: family={cop.family}, theta={cop.theta}")
    
    # Sample and debug
    print("\n=== DEBUGGING SAMPLING ===")
    sampler = VineSampler(vine)
    
    # Small sample for detailed tracing
    n_debug = 5
    print(f"\nSampling {n_debug} points for detailed analysis...")
    
    # Get access to internal sampling method
    w = torch.rand(n_debug, d, device=sampler.device)
    print(f"\nInitial uniform samples w:")
    print(w)
    
    # Initialize v matrix
    v = torch.zeros(n_debug, d, d, device=sampler.device, dtype=w.dtype)
    v_flip = torch.zeros_like(v)
    v[:, 0, 0] = w[:, 0]
    
    # Trace first few steps manually
    print(f"\nv[:, 0, 0] = {v[:, 0, 0]}")
    
    # Sample variable 1
    v[:, 1, 1] = w[:, 1]
    print(f"v[:, 1, 1] = {v[:, 1, 1]}")
    
    # Show what happens in sampling
    print("\n--- Sampling v[:,0,1] ---")
    # This should use copula to generate dependent sample
    
    # Now do full sampling
    print("\n--- Full sampling test ---")
    samples, u_samples = sampler.sample(1000)
    
    print(f"\nSampled data statistics:")
    print(f"Means: {samples.mean(axis=0)}")
    print(f"Stds: {samples.std(axis=0)}")
    
    sample_corr = np.corrcoef(samples.T)
    print(f"\nSample correlations:")
    print(sample_corr)
    
    print(f"\nCorrelation errors:")
    print(sample_corr - corr_matrix)
    
    # Check the uniform samples
    print(f"\n--- Checking uniform samples ---")
    print(f"u_samples shape: {u_samples.shape}")
    print(f"u_samples min: {u_samples.min(axis=0).values}")
    print(f"u_samples max: {u_samples.max(axis=0).values}")
    print(f"u_samples mean: {u_samples.mean(axis=0)}")
    
    # Check marginal transformations
    print(f"\n--- Checking marginal transformations ---")
    if hasattr(vine, 'Mar_G'):
        for i in range(d):
            if i < len(vine.Mar_G):
                mar_s, mar_p = vine.Mar_G[i]
                print(f"\nVariable {i}:")
                print(f"  mar_s shape: {mar_s.shape}")
                print(f"  mar_s range: [{mar_s.min():.4f}, {mar_s.max():.4f}]")
                print(f"  mar_p range: [{mar_p.min():.4f}, {mar_p.max():.4f}]")


if __name__ == "__main__":
    debug_sampling() 