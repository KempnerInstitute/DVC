import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.kde_bounded import (
    bounded_kde_gaussian, bounded_kde_wrapper, 
    adaptive_bounded_kde, transform_bounded_kde
)
from utils.kde_simple import kde_gaussian

def test_bounded_kde_methods():
    """Test and visualize different bounded KDE methods"""
    
    # Generate test data with clear bounds
    torch.manual_seed(42)
    
    # Example 1: Uniform distribution with some noise
    uniform_data = torch.rand(500) * 4 - 1  # [-1, 3]
    
    # Example 2: Beta distribution (naturally bounded [0,1])
    beta_data = torch.distributions.Beta(2.0, 5.0).sample((500,))
    
    # Example 3: Truncated normal
    normal_data = torch.randn(1000)
    truncated_normal = normal_data[(normal_data > -2) & (normal_data < 2)]
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    datasets = [
        ('Uniform [-1, 3]', uniform_data, (-1, 3)),
        ('Beta(2, 5)', beta_data, (0, 1)),
        ('Truncated Normal', truncated_normal, (-2, 2))
    ]
    
    for row, (name, data, true_bounds) in enumerate(datasets):
        # Standard KDE (unbounded)
        ax = axes[row, 0]
        density_std, mesh_std = kde_gaussian(data, n=200, method='fft')
        ax.plot(mesh_std.numpy(), density_std.numpy(), 'b-', label='Standard KDE')
        ax.hist(data.numpy(), bins=30, density=True, alpha=0.3, color='gray')
        ax.axvline(true_bounds[0], color='r', linestyle='--', alpha=0.5)
        ax.axvline(true_bounds[1], color='r', linestyle='--', alpha=0.5)
        ax.set_title(f'{name} - Standard KDE')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Bounded KDE with truncation
        ax = axes[row, 1]
        density_trunc, mesh_trunc = bounded_kde_gaussian(
            data, n=200, bounds=true_bounds, boundary_correction='truncate'
        )
        ax.plot(mesh_trunc.numpy(), density_trunc.numpy(), 'g-', label='Truncated')
        ax.hist(data.numpy(), bins=30, density=True, alpha=0.3, color='gray')
        ax.axvline(true_bounds[0], color='r', linestyle='--', alpha=0.5)
        ax.axvline(true_bounds[1], color='r', linestyle='--', alpha=0.5)
        ax.set_title('Bounded KDE (Truncate)')
        ax.grid(True, alpha=0.3)
        
        # Bounded KDE with reflection
        ax = axes[row, 2]
        density_reflect, mesh_reflect = bounded_kde_gaussian(
            data, n=200, bounds=true_bounds, boundary_correction='reflect'
        )
        ax.plot(mesh_reflect.numpy(), density_reflect.numpy(), 'orange', label='Reflected')
        ax.hist(data.numpy(), bins=30, density=True, alpha=0.3, color='gray')
        ax.axvline(true_bounds[0], color='r', linestyle='--', alpha=0.5)
        ax.axvline(true_bounds[1], color='r', linestyle='--', alpha=0.5)
        ax.set_title('Bounded KDE (Reflect)')
        ax.grid(True, alpha=0.3)
        
        if row == 2:
            axes[row, 0].set_xlabel('Value')
            axes[row, 1].set_xlabel('Value')
            axes[row, 2].set_xlabel('Value')
    
    plt.tight_layout()
    plt.savefig('bounded_kde_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved bounded KDE comparison plot")
    
    # Test transform methods
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    
    # Beta data is good for testing transforms
    test_data = beta_data
    bounds = (0, 1)
    
    methods = [
        ('Standard KDE', lambda d: kde_gaussian(d, n=200)),
        ('Bounded (Renormalized)', lambda d: bounded_kde_gaussian(
            d, n=200, bounds=bounds, boundary_correction='renormalize')),
        ('Logit Transform', lambda d: transform_bounded_kde(
            d, n=200, transform='logit', bounds=bounds)),
        ('Probit Transform', lambda d: transform_bounded_kde(
            d, n=200, transform='probit', bounds=bounds))
    ]
    
    for idx, (method_name, method_func) in enumerate(methods):
        ax = axes2[idx // 2, idx % 2]
        
        density, mesh = method_func(test_data)
        ax.plot(mesh.numpy(), density.numpy(), linewidth=2, label=method_name)
        ax.hist(test_data.numpy(), bins=30, density=True, alpha=0.3, color='gray')
        ax.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(1, color='r', linestyle='--', alpha=0.5)
        ax.set_title(method_name)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 1.2)
        
        # Calculate integral
        dx = (mesh[-1] - mesh[0]) / (len(mesh) - 1)
        integral = torch.sum(density) * dx
        ax.text(0.02, 0.95, f'∫ = {integral:.4f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('bounded_kde_transforms.png', dpi=300, bbox_inches='tight')
    print("Saved bounded KDE transform comparison plot")

def test_adaptive_bounds():
    """Test adaptive boundary selection"""
    
    # Generate data with outliers
    torch.manual_seed(42)
    main_data = torch.randn(980) * 0.5
    outliers = torch.tensor([-5, -4.5, 4.5, 5])
    data = torch.cat([main_data, outliers])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Standard KDE (affected by outliers)
    ax = axes[0]
    density_std, mesh_std = kde_gaussian(data, n=200)
    ax.plot(mesh_std.numpy(), density_std.numpy(), 'b-', linewidth=2)
    ax.scatter(data.numpy(), torch.zeros_like(data).numpy(), alpha=0.3, s=20)
    ax.set_title('Standard KDE (with outliers)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # Adaptive bounded KDE (95% coverage)
    ax = axes[1]
    density_adapt, mesh_adapt = adaptive_bounded_kde(data, n=200, alpha=0.05)
    ax.plot(mesh_adapt.numpy(), density_adapt.numpy(), 'g-', linewidth=2)
    ax.scatter(data.numpy(), torch.zeros_like(data).numpy(), alpha=0.3, s=20)
    ax.set_title('Adaptive Bounded KDE (95% coverage)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # Comparison with different alpha values
    ax = axes[2]
    alphas = [0.01, 0.05, 0.10]
    colors = ['red', 'green', 'blue']
    
    for alpha, color in zip(alphas, colors):
        density, mesh = adaptive_bounded_kde(data, n=200, alpha=alpha)
        ax.plot(mesh.numpy(), density.numpy(), color=color, 
                linewidth=2, label=f'α = {alpha}')
    
    ax.scatter(data.numpy(), torch.zeros_like(data).numpy(), alpha=0.3, s=20, color='black')
    ax.set_title('Effect of α on Adaptive Bounds')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_bounded_kde.png', dpi=300, bbox_inches='tight')
    print("Saved adaptive bounded KDE plot")

def test_marginal_with_bounds():
    """Test marginal PDF estimation with bounded KDE"""
    
    # Generate some example data
    torch.manual_seed(42)
    
    # Create data that should be bounded
    # Example: correlation coefficient (bounded [-1, 1])
    n_samples = 1000
    correlation_data = torch.tanh(torch.randn(n_samples) * 0.5)
    
    # Example: probability values (bounded [0, 1])
    prob_data = torch.sigmoid(torch.randn(n_samples))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    datasets = [
        ('Correlation Values', correlation_data, (-1, 1)),
        ('Probability Values', prob_data, (0, 1))
    ]
    
    for row, (name, data, bounds) in enumerate(datasets):
        # Unbounded vs Bounded comparison
        ax1 = axes[row, 0]
        
        # Unbounded KDE
        density_unb, mesh_unb = kde_gaussian(data, n=200)
        ax1.plot(mesh_unb.numpy(), density_unb.numpy(), 'b-', 
                linewidth=2, label='Unbounded KDE')
        
        # Bounded KDE
        density_b, mesh_b = bounded_kde_gaussian(
            data, n=200, bounds=bounds, boundary_correction='renormalize'
        )
        ax1.plot(mesh_b.numpy(), density_b.numpy(), 'r-', 
                linewidth=2, label='Bounded KDE')
        
        ax1.hist(data.numpy(), bins=30, density=True, alpha=0.3, color='gray')
        ax1.axvline(bounds[0], color='black', linestyle='--', alpha=0.5)
        ax1.axvline(bounds[1], color='black', linestyle='--', alpha=0.5)
        ax1.set_title(f'{name} - KDE Comparison')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Zoom in on boundary behavior
        ax2 = axes[row, 1]
        
        # Focus on upper boundary
        upper_region = (mesh_b > bounds[1] - 0.3) & (mesh_b < bounds[1] + 0.1)
        
        ax2.plot(mesh_unb[upper_region].numpy(), density_unb[upper_region].numpy(), 
                'b-', linewidth=2, label='Unbounded')
        ax2.plot(mesh_b.numpy(), density_b.numpy(), 'r-', 
                linewidth=2, label='Bounded')
        ax2.axvline(bounds[1], color='black', linestyle='--', alpha=0.5, 
                   label=f'Boundary ({bounds[1]})')
        ax2.set_title('Boundary Behavior (zoomed)')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(bounds[1] - 0.3, bounds[1] + 0.1)
    
    plt.tight_layout()
    plt.savefig('marginal_bounded_kde.png', dpi=300, bbox_inches='tight')
    print("Saved marginal bounded KDE comparison")

def main():
    """Run all bounded KDE tests"""
    print("Testing Bounded KDE implementations...\n")
    
    print("1. Testing different boundary correction methods...")
    test_bounded_kde_methods()
    
    print("\n2. Testing adaptive boundary selection...")
    test_adaptive_bounds()
    
    print("\n3. Testing marginal PDF estimation with bounds...")
    test_marginal_with_bounds()
    
    print("\n=== All tests completed ===")
    print("Generated plots:")
    print("  - bounded_kde_comparison.png")
    print("  - bounded_kde_transforms.png") 
    print("  - adaptive_bounded_kde.png")
    print("  - marginal_bounded_kde.png")

if __name__ == "__main__":
    main() 