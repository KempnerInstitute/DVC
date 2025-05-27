import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats

# Add DVC to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from DVC_pyolder.vine_model import _h_function, fit_vine, evaluate_vine
from DVC_pyolder.objects import cop_par_obj, vine_obj_bin, margin_obj
from DVC_pyolder.grid_ops import grid_obj

def test_theta_propagation():
    """
    Test if the theta and theta_flip variables are properly populated during vine fitting.
    This is crucial for the correct behavior of the vine model.
    """
    # Generate synthetic data from a Gaussian vine
    N = 1000
    dimension = 3
    rho = 0.7
    
    # Create correlated normal data (using a simple Markov structure)
    normal = torch.distributions.Normal(0, 1)
    z1 = normal.sample((N,))
    z2 = rho * z1 + torch.sqrt(torch.tensor(1 - rho**2)) * normal.sample((N,))
    z3 = rho * z2 + torch.sqrt(torch.tensor(1 - rho**2)) * normal.sample((N,))
    
    # Combine into a dataset
    X = torch.stack([z1, z2, z3], dim=1).numpy()
    
    # Create margin objects for each variable (standard normal for all)
    margins = []
    for i in range(dimension):
        margins.append(margin_obj('norm', [0.0, 1.0], True))
    
    # Configure vine model (C-vine)
    vine = vine_obj_bin('c-vine', 'gaussian', 3, margins, 30, 'matrix')
    
    # Configuration dictionaries for fit_vine
    gen_dict = {
        'binning': False,
        'parallel': False,
        'param': True,
        'fitted': False,
        'vine_depth': 3
    }
    
    npc_dict = {
        'npc_family': 'locallik',
        'grid_dim': 30
    }
    
    par_dict = {
        'param_families': ['gaussian', 'clayton']
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit the vine to our data
    vine = fit_vine(vine, X, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Now examine theta and theta_flip for uniformity and correlation structure
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # First check: histograms of theta values - should be uniform on 0,1
    for i in range(dimension):
        for j in range(dimension):
            if i == 0 and j < dimension:  # Original margins should be uniformly distributed
                axes[i, j].hist(vine.theta[:, i, j].cpu().numpy(), bins=30, alpha=0.7, density=True)
                axes[i, j].set_title(f'Theta[{i},{j}]')
                axes[i, j].axhline(y=1.0, color='r', linestyle='--', label='Uniform density')
                axes[i, j].set_xlim([0, 1])
                axes[i, j].legend()
            elif j > i and i > 0:  # Values after applying h-function, should also be uniform
                axes[i, j].hist(vine.theta[:, i, j].cpu().numpy(), bins=30, alpha=0.7, density=True)
                axes[i, j].set_title(f'Theta[{i},{j}]')
                axes[i, j].axhline(y=1.0, color='r', linestyle='--', label='Uniform density')
                axes[i, j].set_xlim([0, 1])
                axes[i, j].legend()
    
    plt.savefig('vine_theta_distribution_test.png')
    print("Saved theta distribution test to 'vine_theta_distribution_test.png'")
    
    # Second check: correlation between theta values (scatter plots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Tree 0: Original variables vs first-level h-functions
    axes[0].scatter(vine.theta[:, 0, 0].cpu().numpy(), vine.theta[:, 0, 1].cpu().numpy(), alpha=0.3, label='Original')
    axes[0].scatter(vine.theta[:, 1, 1].cpu().numpy(), vine.theta[:, 1, 2].cpu().numpy(), alpha=0.3, label='h-functions')
    axes[0].set_title('Tree 0: Original vs h-functions')
    axes[0].set_xlabel('u[0] / h(u[1]|u[0])')
    axes[0].set_ylabel('u[1] / h(u[2]|u[1])')
    axes[0].grid(True)
    axes[0].legend()
    
    # Tree 1: Level 1 transformed variables
    axes[1].scatter(vine.theta[:, 1, 1].cpu().numpy(), vine.theta[:, 1, 2].cpu().numpy(), alpha=0.3)
    axes[1].set_title('Tree 1: h-functions')
    axes[1].set_xlabel('h(u[1]|u[0])')
    axes[1].set_ylabel('h(u[2]|u[1])')
    axes[1].grid(True)
    
    # Tree 1: Correlation with original variables
    axes[2].scatter(vine.theta[:, 0, 0].cpu().numpy(), vine.theta[:, 1, 2].cpu().numpy(), alpha=0.3)
    axes[2].set_title('Correlation between original and transformed')
    axes[2].set_xlabel('u[0]')
    axes[2].set_ylabel('h(u[2]|u[1])')
    axes[2].grid(True)
    
    plt.savefig('vine_theta_correlation_test.png')
    print("Saved theta correlation test to 'vine_theta_correlation_test.png'")
    
    # Third check: KS-test to verify uniformity
    ks_stats = {}
    for i in range(dimension):
        for j in range(dimension):
            if j > i or (i == 0 and j < dimension):
                values = vine.theta[:, i, j].cpu().numpy()
                ks_stat = stats.kstest(values, 'uniform', args=(0, 1)).statistic
                ks_stats[(i, j)] = ks_stat
    
    # Fourth check: manually compute h-functions and compare with vine.theta values
    if hasattr(vine, 'copulas') and len(vine.copulas) > 0:
        # Extract the fitted copulas
        first_level_copulas = vine.copulas[0]
        
        # First edge (variables 0 and 1)
        cop01 = first_level_copulas[0]
        u0 = vine.theta[:, 0, 0]
        u1 = vine.theta[:, 0, 1]
        manual_h01 = _h_function(u0, u1, cop01, vine.grid_u if hasattr(vine, 'grid_u') else None, side="left")
        
        # Second edge (variables 0 and 2, or 1 and 2 in C-vine)
        if len(first_level_copulas) > 1:
            cop12 = first_level_copulas[1]
            u1 = vine.theta[:, 0, 1]
            u2 = vine.theta[:, 0, 2]
            manual_h12 = _h_function(u1, u2, cop12, vine.grid_u if hasattr(vine, 'grid_u') else None, side="left")
            
            # Compare with vine.theta values
            h01_diff = torch.abs(manual_h01 - vine.theta[:, 1, 1]).mean().item()
            h12_diff = torch.abs(manual_h12 - vine.theta[:, 1, 2]).mean().item()
            
            # If the differences are significant, plot for visualization
            if h01_diff > 0.001 or h12_diff > 0.001:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                axes[0].scatter(manual_h01.cpu().numpy(), vine.theta[:, 1, 1].cpu().numpy(), alpha=0.3)
                axes[0].plot([0, 1], [0, 1], 'r--')
                axes[0].set_title(f'Manual h(u[1]|u[0]) vs vine.theta [1,1]\nMean diff: {h01_diff:.6f}')
                axes[0].set_xlabel('Manual h(u[1]|u[0])')
                axes[0].set_ylabel('vine.theta[:, 1, 1]')
                axes[0].grid(True)
                
                axes[1].scatter(manual_h12.cpu().numpy(), vine.theta[:, 1, 2].cpu().numpy(), alpha=0.3)
                axes[1].plot([0, 1], [0, 1], 'r--')
                axes[1].set_title(f'Manual h(u[2]|u[1]) vs vine.theta [1,2]\nMean diff: {h12_diff:.6f}')
                axes[1].set_xlabel('Manual h(u[2]|u[1])')
                axes[1].set_ylabel('vine.theta[:, 1, 2]')
                axes[1].grid(True)
                
                plt.savefig('vine_h_function_comparison.png')
                print("Saved h-function comparison to 'vine_h_function_comparison.png'")
                
                # Return CPU values for comparison
                return {
                    'ks_stats': ks_stats,
                    'h01_diff': h01_diff,
                    'h12_diff': h12_diff,
                    'h01_manual': manual_h01.cpu().numpy(),
                    'h12_manual': manual_h12.cpu().numpy(),
                    'h01_vine': vine.theta[:, 1, 1].cpu().numpy(),
                    'h12_vine': vine.theta[:, 1, 2].cpu().numpy(),
                }
    
    return {
        'ks_stats': ks_stats
    }

def test_vine_density():
    """
    Test if the vine density evaluation properly uses the h-functions.
    """
    # Generate synthetic data from a Gaussian vine
    N = 1000
    dimension = 3
    rho = 0.7
    
    # Create correlated normal data (using a simple Markov structure)
    normal = torch.distributions.Normal(0, 1)
    z1 = normal.sample((N,))
    z2 = rho * z1 + torch.sqrt(torch.tensor(1 - rho**2)) * normal.sample((N,))
    z3 = rho * z2 + torch.sqrt(torch.tensor(1 - rho**2)) * normal.sample((N,))
    
    # Combine into a dataset
    X = torch.stack([z1, z2, z3], dim=1).numpy()
    
    # Create margin objects for each variable (standard normal for all)
    margins = []
    for i in range(dimension):
        margins.append(margin_obj('norm', [0.0, 1.0], True))
    
    # Configure vine model (C-vine)
    vine = vine_obj_bin('c-vine', 'gaussian', 3, margins, 30, 'matrix')
    
    # Configuration dictionaries for fit_vine
    gen_dict = {
        'binning': False,
        'parallel': False,
        'param': True,
        'fitted': False,
        'vine_depth': 3
    }
    
    npc_dict = {
        'npc_family': 'locallik',
        'grid_dim': 30
    }
    
    par_dict = {
        'param_families': ['gaussian', 'clayton']
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit the vine to our data
    vine = fit_vine(vine, X, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Evaluate the vine density on the original data
    X_torch = torch.tensor(X, dtype=torch.float32)
    p, p_cop, logp_marg = evaluate_vine(vine, X_torch)
    
    # Calculate the correlation matrices for original data and for the transformed variables
    corr_orig = np.corrcoef(X.T)
    
    # For the transformed copula variables (uniform margins), convert from theta
    u_values = np.zeros((N, dimension))
    for i in range(dimension):
        u_values[:, i] = vine.theta[:, 0, i].cpu().numpy()
    corr_uniform = np.corrcoef(u_values.T)
    
    # For the h-functions in the next level
    h_values = np.zeros((N, dimension-1))
    for i in range(dimension-1):
        h_values[:, i] = vine.theta[:, 1, i+1].cpu().numpy()
    corr_h = np.corrcoef(h_values.T)
    
    # Plot the correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(corr_orig, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Correlation: Original Data')
    axes[0].set_xticks(np.arange(dimension))
    axes[0].set_yticks(np.arange(dimension))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(corr_uniform, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Correlation: Uniform Margins')
    axes[1].set_xticks(np.arange(dimension))
    axes[1].set_yticks(np.arange(dimension))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(corr_h, cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title('Correlation: h-functions')
    axes[2].set_xticks(np.arange(dimension-1))
    axes[2].set_yticks(np.arange(dimension-1))
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.savefig('vine_correlation_matrices.png')
    print("Saved correlation matrices to 'vine_correlation_matrices.png'")
    
    # Also check if the data points with original high correlation show appropriate 
    # patterns in the h-function space (for higher trees)
    
    # Sort the data by z1 value
    z1_cpu = z1.cpu().numpy()
    sorted_indices = np.argsort(z1_cpu)
    z1_sorted = z1_cpu[sorted_indices]
    z3_sorted = z3.cpu().numpy()[sorted_indices]
    h12_sorted = vine.theta[:, 1, 1].cpu().numpy()[sorted_indices]
    h23_sorted = vine.theta[:, 1, 2].cpu().numpy()[sorted_indices]
    
    # Plot the relationship between original variables and h-functions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original z1 vs z3
    axes[0].scatter(z1_sorted, z3_sorted, alpha=0.3, c=sorted_indices, cmap='viridis')
    axes[0].set_title('Original: z1 vs z3')
    axes[0].set_xlabel('z1')
    axes[0].set_ylabel('z3')
    axes[0].grid(True)
    
    # h-functions from tree 1
    axes[1].scatter(h12_sorted, h23_sorted, alpha=0.3, c=sorted_indices, cmap='viridis')
    axes[1].set_title('h-functions: h(u2|u1) vs h(u3|u2)')
    axes[1].set_xlabel('h(u2|u1)')
    axes[1].set_ylabel('h(u3|u2)')
    axes[1].grid(True)
    
    plt.savefig('vine_h_function_relationships.png')
    print("Saved h-function relationships to 'vine_h_function_relationships.png'")
    
    return {
        'corr_orig': corr_orig,
        'corr_uniform': corr_uniform,
        'corr_h': corr_h,
    }

if __name__ == "__main__":
    print("Testing theta propagation...")
    theta_results = test_theta_propagation()
    
    print("\nKS statistics for theta uniformity (should be close to 0):")
    for (i, j), ks in theta_results['ks_stats'].items():
        print(f"  theta[{i},{j}]: KS statistic = {ks:.4f}")
    
    if 'h01_diff' in theta_results:
        print(f"\nDifference between manual and vine h-functions:")
        print(f"  h(u[1]|u[0]): Mean diff = {theta_results['h01_diff']:.6f}")
        print(f"  h(u[2]|u[1]): Mean diff = {theta_results['h12_diff']:.6f}")
    
    print("\nTesting vine density evaluation...")
    density_results = test_vine_density()
    
    print("\nCorrelation matrices:")
    print("Original data:")
    print(density_results['corr_orig'])
    print("\nUniform margins:")
    print(density_results['corr_uniform'])
    print("\nh-functions:")
    print(density_results['corr_h']) 