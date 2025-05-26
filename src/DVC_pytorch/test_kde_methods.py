import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.prob_op import kde, kernel_pdf2
from utils.kde_simple import kde_gaussian, silverman_bandwidth, scott_bandwidth

def test_all_kde_methods():
    """Test and compare all KDE methods"""
    print("=== Testing All KDE Methods ===")
    
    # Create test data
    torch.manual_seed(42)
    data = torch.randn(1000)
    
    print(f"Data shape: {data.shape}")
    print(f"Data mean: {data.mean():.4f}, std: {data.std():.4f}")
    print(f"Silverman bandwidth: {silverman_bandwidth(data):.4f}")
    print(f"Scott bandwidth: {scott_bandwidth(data):.4f}")
    
    # Methods to test
    methods = {
        'Original KDE (DCT)': lambda d: kde(d, n=128),
        'kernel_pdf2': lambda d: kernel_pdf2(d),
        'Simple FFT': lambda d: kde_gaussian(d, n=128, method='fft'),
        'Simple cdist': lambda d: kde_gaussian(d, n=128, method='cdist'),
        'Simple cdist (chunked)': lambda d: kde_gaussian(d, n=128, method='cdist_chunked')
    }
    
    results = {}
    times = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, method) in enumerate(methods.items()):
        print(f"\n{name}:")
        
        # Time the method
        start = time.time()
        density, mesh = method(data)
        elapsed = time.time() - start
        times[name] = elapsed
        
        # Calculate integral
        dx = mesh[1:] - mesh[:-1]
        dx = torch.cat([dx, dx[-1:]])
        integral = torch.sum(density * dx)
        
        print(f"  Time: {elapsed:.4f}s")
        print(f"  Density shape: {density.shape}")
        print(f"  Mesh range: [{mesh.min():.4f}, {mesh.max():.4f}]")
        print(f"  Integral: {integral:.6f}")
        print(f"  Error from 1.0: {abs(integral - 1.0):.6f}")
        
        results[name] = {
            'density': density,
            'mesh': mesh,
            'integral': integral,
            'time': elapsed
        }
        
        # Plot
        if idx < len(axes):
            axes[idx].hist(data.numpy(), bins=50, density=True, alpha=0.5, label='Histogram')
            axes[idx].plot(mesh.numpy(), density.numpy(), 'r-', linewidth=2, 
                          label=f'KDE (∫={integral:.3f})')
            axes[idx].set_title(f'{name}\nTime: {elapsed:.4f}s')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('kde_methods_comparison.png', dpi=150)
    print("\nPlot saved as 'kde_methods_comparison.png'")
    
    # Test with larger dataset
    print("\n=== Testing with Large Dataset (10k points) ===")
    large_data = torch.randn(10000)
    
    for name, method in methods.items():
        print(f"\n{name}:")
        start = time.time()
        try:
            density, mesh = method(large_data)
            elapsed = time.time() - start
            
            dx = mesh[1:] - mesh[:-1]
            dx = torch.cat([dx, dx[-1:]])
            integral = torch.sum(density * dx)
            
            print(f"  Time: {elapsed:.4f}s")
            print(f"  Integral: {integral:.6f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test with different data types
    print("\n=== Testing Different Data Types ===")
    test_cases = {
        'Bimodal': torch.cat([torch.randn(500) - 2, torch.randn(500) + 2]),
        'Uniform': torch.rand(1000) * 4 - 2,
        'Exponential': torch.distributions.Exponential(1.0).sample((1000,))
    }
    
    for data_name, test_data in test_cases.items():
        print(f"\n{data_name} distribution:")
        
        # Test simple FFT method
        density, mesh = kde_gaussian(test_data, n=128, method='fft')
        dx = mesh[1:] - mesh[:-1]
        dx = torch.cat([dx, dx[-1:]])
        integral = torch.sum(density * dx)
        print(f"  Simple FFT: integral = {integral:.6f}")
        
        # Test original kde
        density2, mesh2 = kde(test_data, n=128)
        dx2 = mesh2[1:] - mesh2[:-1]
        dx2 = torch.cat([dx2, dx2[-1:]])
        integral2 = torch.sum(density2 * dx2)
        print(f"  Original KDE: integral = {integral2:.6f}")

if __name__ == "__main__":
    test_all_kde_methods() 