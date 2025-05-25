"""
Test and fix marginal density estimation
"""

import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Import functions
from utils.prob_op import kernel_pdf2, kernel_cdf
from utils.interpolation import interp1d_np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate simple test data
np.random.seed(42)
n_samples = 1000
data = torch.randn(n_samples, device=device)
print(f"Data stats: min={data.min():.3f}, max={data.max():.3f}, mean={data.mean():.3f}")

# Test kernel_pdf2
print("\nTesting kernel_pdf2...")
try:
    density, mesh = kernel_pdf2(data)
    print(f"Density shape: {density.shape}")
    print(f"Mesh shape: {mesh.shape}")
    print(f"Density stats: min={density.min():.3f}, max={density.max():.3f}")
    print(f"Has NaN: {torch.any(torch.isnan(density))}")
    print(f"Has negative: {torch.any(density < 0)}")
    
    # Plot
    if not torch.any(torch.isnan(density)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(data.cpu().numpy(), bins=50, density=True, alpha=0.5, label='Data histogram')
        plt.plot(mesh.cpu().numpy(), density.cpu().numpy(), 'r-', label='KDE')
        plt.legend()
        plt.title('Kernel Density Estimation')
        
except Exception as e:
    print(f"Error in kernel_pdf2: {e}")
    import traceback
    traceback.print_exc()

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
    cdf_values, margin_s, margin_p = kernel_cdf(data, data, grid)
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