import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.kde_simple import kde_fft_1d, silverman_bandwidth

def debug_fft_kde():
    """Debug the FFT KDE normalization issue"""
    print("=== Debugging FFT KDE ===")
    
    # Simple test data
    torch.manual_seed(42)
    data = torch.randn(100)
    
    # Get bandwidth
    bandwidth = silverman_bandwidth(data)
    print(f"Data shape: {data.shape}")
    print(f"Bandwidth: {bandwidth:.4f}")
    
    # Run FFT KDE with explicit parameters
    x_min = data.min().item() - 3 * bandwidth
    x_max = data.max().item() + 3 * bandwidth
    num_bins = 128
    
    print(f"\nParameters:")
    print(f"x_min: {x_min:.4f}")
    print(f"x_max: {x_max:.4f}")
    print(f"num_bins: {num_bins}")
    
    # Step by step debugging
    N = data.shape[0]
    
    # Create histogram
    hist = torch.histc(data, bins=num_bins, min=x_min, max=x_max).float()
    dx = (x_max - x_min) / num_bins
    
    print(f"\nHistogram:")
    print(f"hist sum: {hist.sum()}")
    print(f"dx: {dx:.4f}")
    print(f"hist sum * dx: {hist.sum() * dx:.4f}")
    
    # Create Gaussian kernel
    x_kernel = torch.arange(num_bins, dtype=torch.float32) - num_bins // 2
    x_kernel = x_kernel * dx
    
    # Gaussian kernel
    gauss = torch.exp(-0.5 * (x_kernel ** 2) / (bandwidth ** 2))
    gauss = gauss / (np.sqrt(2 * np.pi) * bandwidth)
    
    print(f"\nKernel before normalization:")
    print(f"gauss sum: {gauss.sum()}")
    print(f"gauss sum * dx: {gauss.sum() * dx:.4f}")
    
    # Normalize kernel
    gauss = gauss / (gauss.sum() * dx)
    
    print(f"\nKernel after normalization:")
    print(f"gauss sum * dx: {gauss.sum() * dx:.4f}")
    
    # Simple convolution (no padding for clarity)
    conv = torch.zeros_like(hist)
    for i in range(num_bins):
        for j in range(num_bins):
            if 0 <= i-j+num_bins//2 < num_bins:
                conv[i] += hist[j] * gauss[i-j+num_bins//2]
    
    # Convert to density
    density = conv / (N * dx)
    
    print(f"\nDensity:")
    print(f"density sum: {density.sum()}")
    print(f"density integral: {density.sum() * dx:.4f}")
    
    # Create grid
    edges = torch.linspace(x_min, x_max, num_bins + 1)
    grid = (edges[:-1] + edges[1:]) / 2
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(grid.numpy(), hist.numpy() / (N * dx), width=dx, alpha=0.5, label='Normalized histogram')
    plt.plot(grid.numpy(), density.numpy(), 'r-', linewidth=2, label='KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Simple Convolution')
    plt.legend()
    
    # Now test the actual function
    density2, grid2 = kde_fft_1d(data, x_min, x_max, num_bins, bandwidth)
    dx2 = grid2[1] - grid2[0]
    integral2 = density2.sum() * dx2
    
    plt.subplot(1, 2, 2)
    plt.hist(data.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
    plt.plot(grid2.numpy(), density2.numpy(), 'g-', linewidth=2, label=f'FFT KDE (∫={integral2:.3f})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('FFT Method')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fft_kde_debug.png')
    print("\nPlot saved as 'fft_kde_debug.png'")
    
    return density, grid

if __name__ == "__main__":
    debug_fft_kde() 