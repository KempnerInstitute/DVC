"""
Test and fix marginal density estimation
"""

import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from utils.prob_op import kernel_pdf2

# Import functions
from utils.prob_op import kernel_cdf
from utils.interpolation import interp1d_np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def test_marginal_pdf():
    """Test marginal PDF estimation in the context of DVC"""
    print("=== Testing Marginal PDF Estimation ===")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated data
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    data = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Convert to torch tensors
    x1 = torch.tensor(data[:, 0], dtype=torch.float32)
    x2 = torch.tensor(data[:, 1], dtype=torch.float32)
    
    # Test kernel_pdf2 directly
    print("\n1. Testing kernel_pdf2 directly:")
    density1, mesh1 = kernel_pdf2(x1)
    print(f"   Density shape: {density1.shape}")
    print(f"   Mesh shape: {mesh1.shape}")
    print(f"   Density variance: {density1.var():.6f}")
    print(f"   Max density: {density1.max():.6f}")
    print(f"   Min density: {density1.min():.6f}")
    
    # Test with different data types
    print("\n2. Testing different data types:")
    test_data_types = {
        'Normal': torch.randn(500),
        'Uniform': torch.rand(500) * 4 - 2,
        'Integer': torch.randint(0, 10, (500,)).float(),
        'Bimodal': torch.cat([torch.randn(250) - 2, torch.randn(250) + 2]),
        'Repeated': torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0] * 100)
    }
    
    results = []
    for name, test_data in test_data_types.items():
        density, mesh = kernel_pdf2(test_data)
        var = density.var().item()
        max_val = density.max().item()
        results.append((name, var, max_val))
        print(f"   {name:10s}: var={var:.2e}, max={max_val:.4f}, shape={density.shape}")
        if var < 1e-6:
            print(f"      WARNING: Flat density detected!")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot original data histogram and KDE
    axes[0].hist(x1.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
    axes[0].plot(mesh1.numpy(), density1.numpy(), 'r-', linewidth=2, label='KDE')
    axes[0].set_title('Original Data (X1)')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # Plot different data types
    for i, (name, test_data) in enumerate(test_data_types.items(), 1):
        density, mesh = kernel_pdf2(test_data)
        axes[i].hist(test_data.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
        axes[i].plot(mesh.numpy(), density.numpy(), 'r-', linewidth=2, label='KDE')
        axes[i].set_title(f'{name} Distribution')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].text(0.05, 0.95, f'Var: {density.var():.2e}', 
                     transform=axes[i].transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('marginal_pdf_test.png', dpi=150)
    print("\nPlot saved as 'marginal_pdf_test.png'")
    
    # Check if all tests passed
    all_passed = all(var > 1e-6 for _, var, _ in results)
    
    return all_passed

if __name__ == "__main__":
    success = test_marginal_pdf()
    if success:
        print("\n✓ All marginal PDF estimations have proper variance!")
    else:
        print("\n✗ Some marginal PDF estimations are still flat")

# Test interpolation
print("\nTesting interpolation...")
test_points = torch.linspace(-3, 3, 10, device=device)
try:
    # Make sure we have valid density and mesh
    if 'density' in locals() and 'mesh' in locals():
        interp_values = interp1d_np(test_points, mesh, density)
        print(f"Interpolated values: {interp_values}")
        print(f"Has NaN: {torch.any(torch.isnan(interp_values))}")
        
        # Compare with scipy normal
        true_density = stats.norm.pdf(test_points.cpu().numpy())
        print(f"True normal density: {true_density}")
        
        if not torch.any(torch.isnan(interp_values)):
            error = np.mean(np.abs(interp_values.cpu().numpy() - true_density))
            print(f"Mean absolute error: {error:.4f}")
except Exception as e:
    print(f"Error in interpolation: {e}")
    import traceback
    traceback.print_exc()

# Test kernel_cdf
print("\nTesting kernel_cdf...")
grid = torch.linspace(0, 1, 100, device=device)
try:
    cdf_values, margin_s, margin_p = kernel_cdf(x1, x2, grid)
    print(f"CDF values shape: {cdf_values.shape}")
    print(f"CDF range: [{cdf_values.min():.3f}, {cdf_values.max():.3f}]")
    print(f"Has NaN: {torch.any(torch.isnan(cdf_values))}")
    
    # Plot CDF
    if not torch.any(torch.isnan(cdf_values)):
        plt.subplot(1, 2, 2)
        plt.plot(margin_s.cpu().numpy(), margin_p.cpu().numpy(), 'b-', label='Empirical CDF')
        plt.xlabel('Value')
        plt.ylabel('CDF')
        plt.title('Empirical CDF')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('marginal_density_test.png')
        print("Saved plot to marginal_density_test.png")
except Exception as e:
    print(f"Error in kernel_cdf: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!") 