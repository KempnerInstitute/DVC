import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add DVC to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from DVC.vine_model import _h_function
from DVC.objects import cop_par_obj

def test_h_function_uniform_property():
    """
    Test if h-function maintains uniform distribution property.
    For any pair of uniform random variables u and v, the conditional CDF h(v|u)
    should also be uniform over [0,1] regardless of the value of u.
    """
    # Create a grid of values to test
    n_grid = 100
    u_range = torch.linspace(0.05, 0.95, n_grid)
    v_range = torch.linspace(0.0001, 0.9999, n_grid)
    
    # Test different copula families and parameters
    families = ['ind', 'gaussian', 'clayton']
    params = [None, 0.7, 1.5]  # Parameters for each family
    
    # Create figure
    fig, axes = plt.subplots(2, len(families), figsize=(15, 8))
    plt.subplots_adjust(hspace=0.4)
    
    for i, (family, param) in enumerate(zip(families, params)):
        # Create copula object
        cop = cop_par_obj(family, param)
        
        # Test at different fixed u values
        u_fixed_values = [0.2, 0.5, 0.8]
        
        # Plot conditional distributions
        ax1 = axes[0, i]
        for u_val in u_fixed_values:
            u_fixed = torch.ones(n_grid) * u_val
            h_values = _h_function(u_fixed, v_range, cop, None, side="left")
            ax1.plot(v_range.numpy(), h_values.numpy(), label=f'u={u_val:.1f}')
        
        ax1.set_title(f'{family.capitalize()} Copula\nθ={param}')
        ax1.set_xlabel('v')
        ax1.set_ylabel('h(v|u)')
        ax1.legend()
        ax1.grid(True)
        
        # Now sample many points and check uniformity of h-function output
        n_samples = 10000
        u_samples = torch.rand(n_samples)
        v_samples = torch.rand(n_samples)
        
        # Apply h-function
        h_samples = _h_function(u_samples, v_samples, cop, None, side="left")
        
        # Plot histogram to check uniformity
        ax2 = axes[1, i]
        ax2.hist(h_samples.numpy(), bins=30, alpha=0.6, density=True)
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Uniform density')
        ax2.set_xlabel('h(v|u) values')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Histogram of h-function values\nShould be uniform if correctly implemented')
        ax2.legend()
        ax2.grid(True)
    
    plt.savefig('h_function_test_uniform.png')
    print("Saved uniformity test plot to 'h_function_test_uniform.png'")
    
    # Return statistics for checking
    return {
        'h_samples_mean': h_samples.mean().item(),
        'h_samples_std': h_samples.std().item(),
        'uniform_samples_from_h': h_samples,
    }

def test_h_function_tree_propagation():
    """
    Test how h-function values propagate through vine tree levels.
    """
    # Create a synthetic dataset with known correlation
    N = 1000
    rho = 0.7
    
    # Create correlated normal data
    normal = torch.distributions.Normal(0, 1)
    z1 = normal.sample((N,))
    z2 = rho * z1 + torch.sqrt(torch.tensor(1 - rho**2)) * normal.sample((N,))
    z3 = rho * z2 + torch.sqrt(torch.tensor(1 - rho**2)) * normal.sample((N,))
    
    # Transform to uniform margins
    u1 = normal.cdf(z1)
    u2 = normal.cdf(z2)
    u3 = normal.cdf(z3)
    
    # Level 0 copulas
    cop12 = cop_par_obj('gaussian', rho)
    cop23 = cop_par_obj('gaussian', rho)
    
    # Compute h-functions for next level
    h12 = _h_function(u1, u2, cop12, None, side="left")
    h23 = _h_function(u2, u3, cop23, None, side="left")
    
    # Direct copula between u1 and u3
    direct_cop13 = cop_par_obj('gaussian', rho**2)  # In gaussian case should be rho^2
    
    # Create figure for the tree propagation test
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original data
    axes[0].scatter(u1.numpy(), u3.numpy(), alpha=0.2)
    axes[0].set_title('Direct Relationship: u1 vs u3')
    axes[0].set_xlabel('u1')
    axes[0].set_ylabel('u3')
    axes[0].grid(True)
    
    # Plot h-function transformed values
    axes[1].scatter(h12.numpy(), h23.numpy(), alpha=0.2)
    axes[1].set_title('h-function Transformed: h(u2|u1) vs h(u3|u2)')
    axes[1].set_xlabel('h(u2|u1)')
    axes[1].set_ylabel('h(u3|u2)')
    axes[1].grid(True)
    
    # Test if h-functions maintain correct correlation
    # Compute theoretical and empirical conditional distributions
    u1_grid = torch.linspace(0.1, 0.9, 9)
    results = []
    
    for u1_val in u1_grid:
        # Theoretical approach via copula
        u1_fixed = torch.ones(100) * u1_val
        v_range = torch.linspace(0.01, 0.99, 100)
        
        # Direct approach - what u3|u1 should be theoretically
        direct_h = _h_function(u1_fixed, v_range, direct_cop13, None, side="left")
        
        # Two-step approach through h-functions
        # First get u2|u1
        u2_given_u1 = _h_function(u1_fixed, v_range, cop12, None, side="left")
        # Then get u3|u2
        u3_given_u2 = _h_function(u2_given_u1, v_range, cop23, None, side="left")
        
        # Store a few points to plot
        results.append((u1_val.item(), direct_h.numpy(), u3_given_u2.numpy()))
    
    # Plot comparison of direct vs propagated conditional CDF
    for u1_val, direct, propagated in results[::2]:  # Plot every other result
        axes[2].plot(v_range.numpy(), direct, '-', label=f'Direct u1={u1_val:.1f}')
        axes[2].plot(v_range.numpy(), propagated, '--', label=f'Propagated u1={u1_val:.1f}')
    
    axes[2].set_title('Direct vs Propagated Conditional CDF')
    axes[2].set_xlabel('v')
    axes[2].set_ylabel('CDF')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('h_function_tree_propagation_test.png')
    print("Saved tree propagation test to 'h_function_tree_propagation_test.png'")
    
    # Also check KS statistic between direct and propagated
    from scipy import stats
    ks_stats = []
    for u1_val, direct, propagated in results:
        ks_stat = stats.kstest(direct, propagated).statistic
        ks_stats.append((u1_val, ks_stat))
    
    return {
        'ks_stats': ks_stats,
        'results': results
    }

if __name__ == "__main__":
    uniform_test_results = test_h_function_uniform_property()
    print(f"Uniformity test results:")
    print(f"  Mean of h-function values: {uniform_test_results['h_samples_mean']:.4f} (should be ~0.5)")
    print(f"  Std of h-function values: {uniform_test_results['h_samples_std']:.4f} (should be ~0.289)")
    
    tree_results = test_h_function_tree_propagation()
    print(f"\nTree propagation KS statistics (should be close to 0):")
    for u1_val, ks_stat in tree_results['ks_stats']:
        print(f"  u1={u1_val:.1f}: KS statistic = {ks_stat:.4f}") 