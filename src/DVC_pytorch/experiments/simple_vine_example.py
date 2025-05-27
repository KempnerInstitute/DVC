"""
Simple Vine Copula Example
==========================

This script demonstrates basic usage of the DVC PyTorch implementation:
1. Generate synthetic data
2. Fit a vine copula model
3. Generate samples
4. Compare correlations
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# Import DVC modules
from classes.objects import vine_obj_bin, margin_obj
from pre_proc.preparation import prep_cop
from info.info_estimation import entropy_h_estimation
from plot.plot_vine import plot_vine


def main():
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration
    n_samples = 1000
    dim = 4
    
    # Generate correlated data
    print("Generating synthetic data...")
    correlation_matrix = np.array([
        [1.0, 0.7, 0.5, 0.3],
        [0.7, 1.0, 0.6, 0.4],
        [0.5, 0.6, 1.0, 0.5],
        [0.3, 0.4, 0.5, 1.0]
    ])
    
    # Generate multivariate normal data
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, correlation_matrix, n_samples)
    
    # Transform to different marginals
    # Column 0: Normal(0, 1)
    # Column 1: Exponential(scale=2)
    data[:, 1] = stats.expon.ppf(stats.norm.cdf(data[:, 1]), scale=2)
    # Column 2: Uniform(-1, 1)
    data[:, 2] = stats.uniform.ppf(stats.norm.cdf(data[:, 2]), loc=-1, scale=2)
    # Column 3: Student-t(df=5)
    data[:, 3] = stats.t.ppf(stats.norm.cdf(data[:, 3]), df=5)
    
    # Convert to torch tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    
    print(f"Data shape: {data.shape}")
    print(f"Using device: {device}")
    
    # Create margin objects
    margins = []
    for i in range(dim):
        margins.append(margin_obj('norm', [0, 1], True))
    
    # Create and fit C-vine
    print("\nFitting C-vine copula...")
    vine = vine_obj_bin('c-vine', ['gaussian'], dim, margins, 11, 'matrix')
    
    # Prepare data
    data_prep = prep_cop(data_tensor, vine, 'no_sort')
    data_prep = torch.tensor(data_prep, dtype=torch.float32, device=device)
    
    # Fit parameters
    gen_dict = {
        'param': True,
        'binning': False,
        'fitted': False,
        'parallel': True,
        'vine_depth': dim
    }
    
    par_dict = {
        'param_families': ['gaussian', 'student', 'clayton']
    }
    
    npc_dict = {
        'opt_method': 'trust-exact',
        'batch_paral': 10
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit the vine
    vine.fit(data_prep, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Print fitted copula families
    print("\nFitted copula families:")
    for tr in range(len(vine.copulas)):
        print(f"  Tree {tr}:")
        for j, cop in enumerate(vine.copulas[tr]):
            print(f"    Copula {j}: {cop.family} (theta={cop.theta})")
    
    # Generate samples
    print("\nGenerating samples from fitted vine...")
    n_test = 1000
    samples = vine.sample(n_test)
    samples_np = samples.cpu().numpy()
    
    # Compute correlations
    print("\nCorrelation comparison:")
    print("Original data correlations (Kendall's tau):")
    for i in range(dim):
        for j in range(i+1, dim):
            tau, _ = stats.kendalltau(data[:, i], data[:, j])
            print(f"  τ({i},{j}) = {tau:.3f}")
    
    print("\nSampled data correlations (Kendall's tau):")
    for i in range(dim):
        for j in range(i+1, dim):
            tau, _ = stats.kendalltau(samples_np[:, i], samples_np[:, j])
            print(f"  τ({i},{j}) = {tau:.3f}")
    
    # Estimate entropy
    print("\nEstimating copula entropy...")
    entropy = entropy_h_estimation(vine, 'copula', cases=5000)
    print(f"Copula entropy: {entropy.item():.4f}")
    
    # Plot vine structure
    print("\nPlotting vine structure...")
    fig = plot_vine('structure', vine)
    plt.savefig('vine_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original data scatter plots
    axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    axes[0, 0].set_title('Original Data: X0 vs X1')
    axes[0, 0].set_xlabel('X0 (Normal)')
    axes[0, 0].set_ylabel('X1 (Exponential)')
    
    axes[0, 1].scatter(data[:, 2], data[:, 3], alpha=0.5, s=10)
    axes[0, 1].set_title('Original Data: X2 vs X3')
    axes[0, 1].set_xlabel('X2 (Uniform)')
    axes[0, 1].set_ylabel('X3 (Student-t)')
    
    # Sampled data scatter plots
    axes[1, 0].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10, color='orange')
    axes[1, 0].set_title('Sampled Data: X0 vs X1')
    axes[1, 0].set_xlabel('X0')
    axes[1, 0].set_ylabel('X1')
    
    axes[1, 1].scatter(samples_np[:, 2], samples_np[:, 3], alpha=0.5, s=10, color='orange')
    axes[1, 1].set_title('Sampled Data: X2 vs X3')
    axes[1, 1].set_xlabel('X2')
    axes[1, 1].set_ylabel('X3')
    
    plt.suptitle('Original vs Sampled Data Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('data_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nPlots saved:")
    print("  - vine_structure.png")
    print("  - data_comparison.png")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 