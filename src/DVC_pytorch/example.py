"""
Simple example of using DVC PyTorch implementation
"""

import numpy as np
import torch
from classes.objects import vine_obj_bin, margin_obj
from sampling import VineSampler


def main():
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate sample data - 3D correlated Gaussian
    n_samples = 1000
    d = 3
    
    # Create correlation matrix
    rho = 0.7
    corr_matrix = np.array([[1.0, rho, rho*0.5],
                           [rho, 1.0, rho*0.8],
                           [rho*0.5, rho*0.8, 1.0]])
    
    # Generate data
    data = np.random.multivariate_normal(np.zeros(d), corr_matrix, n_samples)
    data = torch.tensor(data, dtype=torch.float32)
    
    print("Original data shape:", data.shape)
    print("Original correlations:")
    print(np.corrcoef(data.T))
    
    # Create margin objects (assuming normal margins)
    margins = [margin_obj('norm', [0, 1], True) for _ in range(d)]
    
    # Create vine object
    vine = vine_obj_bin(
        vine_family='c-vine',  # Can also use 'd-vine'
        families=['gaussian', 'clayton'],  # Copula families to consider
        vine_depth=d-1,  # Full vine
        margin=margins,
        knots=50,
        method='matrix'
    )
    
    # Set up fitting parameters
    gen_dict = {
        "parallel": False,
        "param": True,  # Use parametric copulas
        "binning": False,
        "fitted": False,
        "vine_depth": d-1
    }
    
    par_dict = {
        "param_families": ["gaussian", "clayton", "student"]
    }
    
    npc_dict = {
        "opt_method": "LL1",
        "batch_paral": False
    }
    
    bin_dict = {
        "n_bin": 1
    }
    
    # Fit the vine
    print("\nFitting vine copula...")
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Print fitted copulas
    print("\nFitted copulas:")
    for tr in range(len(vine.copulas)):
        print(f"Tree {tr}:")
        for j, cop in enumerate(vine.copulas[tr]):
            print(f"  Edge {j}: family={cop.family}, theta={cop.theta}")
    
    # Sample from the fitted vine
    print("\nSampling from fitted vine...")
    sampler = VineSampler(vine)
    samples, u_samples = sampler.sample(1000)
    
    # Check sample statistics
    print("\nSample statistics:")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {samples.mean(axis=0)}")
    print(f"Sample std: {samples.std(axis=0)}")
    
    sample_corr = np.corrcoef(samples.T)
    print("\nSample correlations:")
    print(sample_corr)
    
    print("\nDone!")


if __name__ == "__main__":
    main() 