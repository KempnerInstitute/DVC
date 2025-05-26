import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.prob_op import kde, kernel_pdf2

def test_kde_normalization():
    """Test if KDE integrates to 1"""
    print("=== Testing KDE Normalization ===")
    
    # Create test data
    torch.manual_seed(42)
    data = torch.randn(1000)
    
    print(f"Data shape: {data.shape}")
    print(f"Data mean: {data.mean():.4f}, std: {data.std():.4f}")
    
    # Test kde function
    print("\n1. Testing kde function:")
    density, mesh = kde(data, n=128)
    
    # Calculate integral using trapezoidal rule
    dx = mesh[1:] - mesh[:-1]
    dx = torch.cat([dx, dx[-1:]])  # Extend for last point
    integral = torch.sum(density * dx)
    
    print(f"   Density shape: {density.shape}")
    print(f"   Mesh range: [{mesh.min():.4f}, {mesh.max():.4f}]")
    print(f"   Integral of density: {integral:.6f}")
    print(f"   Error from 1.0: {abs(integral - 1.0):.6f}")
    
    # Test kernel_pdf2
    print("\n2. Testing kernel_pdf2:")
    density2, mesh2 = kernel_pdf2(data)
    
    # Calculate integral
    dx2 = mesh2[1:] - mesh2[:-1]
    dx2 = torch.cat([dx2, dx2[-1:]])
    integral2 = torch.sum(density2 * dx2)
    
    print(f"   Density shape: {density2.shape}")
    print(f"   Mesh range: [{mesh2.min():.4f}, {mesh2.max():.4f}]")
    print(f"   Integral of density: {integral2:.6f}")
    print(f"   Error from 1.0: {abs(integral2 - 1.0):.6f}")
    
    # Plot to visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(data.numpy(), bins=50, density=True, alpha=0.5, label='Histogram')
    plt.plot(mesh.numpy(), density.numpy(), 'r-', linewidth=2, label=f'KDE (integral={integral:.3f})')
    plt.title('kde function')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(data.numpy(), bins=50, density=True, alpha=0.5, label='Histogram')
    if density2.shape[0] == mesh2.shape[0]:
        plt.plot(mesh2.numpy(), density2.numpy(), 'g-', linewidth=2, label=f'kernel_pdf2 (integral={integral2:.3f})')
    plt.title('kernel_pdf2 function')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kde_normalization_test.png', dpi=150)
    print("\nPlot saved as 'kde_normalization_test.png'")
    
    # Test with different data types
    print("\n3. Testing different data types:")
    test_cases = {
        'Uniform': torch.rand(1000) * 4 - 2,
        'Exponential': torch.distributions.Exponential(1.0).sample((1000,)),
        'Integer': torch.randint(0, 10, (1000,)).float()
    }
    
    for name, test_data in test_cases.items():
        density, mesh = kde(test_data, n=128)
        dx = mesh[1:] - mesh[:-1]
        dx = torch.cat([dx, dx[-1:]])
        integral = torch.sum(density * dx)
        print(f"   {name}: integral = {integral:.6f}, error = {abs(integral - 1.0):.6f}")

if __name__ == "__main__":
    test_kde_normalization() 