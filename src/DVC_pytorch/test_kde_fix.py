import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.prob_op import kde

# Test the KDE fix with different types of data
def test_kde_fix():
    """Test that KDE now works correctly with the fixed N calculation"""
    
    # Test 1: Data with many repeated values (worst case for the bug)
    print("Test 1: Data with many repeated values")
    data1 = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0] * 100, dtype=torch.float32)
    print(f"Data shape: {data1.shape}")
    print(f"Unique values: {len(torch.unique(data1))}")
    print(f"Sample size: {data1.shape[0]}")
    
    density1, xmesh1 = kde(data1, n=128)
    print(f"Max density: {density1.max():.6f}")
    print(f"Min density: {density1.min():.6f}")
    print(f"Density variance: {density1.var():.6f}")
    
    # Test 2: Normal distribution data
    print("\nTest 2: Normal distribution")
    torch.manual_seed(42)
    data2 = torch.randn(1000)
    print(f"Data shape: {data2.shape}")
    print(f"Unique values: {len(torch.unique(data2))}")
    
    density2, xmesh2 = kde(data2, n=128)
    print(f"Max density: {density2.max():.6f}")
    print(f"Min density: {density2.min():.6f}")
    print(f"Density variance: {density2.var():.6f}")
    
    # Test 3: Integer data (common case that triggers the bug)
    print("\nTest 3: Integer data")
    data3 = torch.randint(0, 10, (500,)).float()
    print(f"Data shape: {data3.shape}")
    print(f"Unique values: {len(torch.unique(data3))}")
    
    density3, xmesh3 = kde(data3, n=128)
    print(f"Max density: {density3.max():.6f}")
    print(f"Min density: {density3.min():.6f}")
    print(f"Density variance: {density3.var():.6f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Repeated values
    axes[0].hist(data1.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
    axes[0].plot(xmesh1.numpy(), density1.numpy(), 'r-', linewidth=2, label='KDE')
    axes[0].set_title('Test 1: Repeated Values')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # Plot 2: Normal distribution
    axes[1].hist(data2.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
    axes[1].plot(xmesh2.numpy(), density2.numpy(), 'r-', linewidth=2, label='KDE')
    axes[1].set_title('Test 2: Normal Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    
    # Plot 3: Integer data
    axes[2].hist(data3.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
    axes[2].plot(xmesh3.numpy(), density3.numpy(), 'r-', linewidth=2, label='KDE')
    axes[2].set_title('Test 3: Integer Data')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('kde_fix_test.png', dpi=150)
    print("\nPlot saved as 'kde_fix_test.png'")
    
    # Verify that densities are not flat
    for i, (density, name) in enumerate([(density1, "Repeated values"), 
                                          (density2, "Normal"), 
                                          (density3, "Integer")]):
        if density.var() < 1e-6:
            print(f"\nWARNING: {name} density is nearly flat! Variance: {density.var():.2e}")
        else:
            print(f"\n{name} density looks good. Variance: {density.var():.6f}")

if __name__ == "__main__":
    test_kde_fix() 