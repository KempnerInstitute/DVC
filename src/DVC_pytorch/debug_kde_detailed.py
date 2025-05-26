import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.prob_op import kde, dct1d, find_root_bisection

def debug_kde_detailed():
    """Debug KDE to understand why it's still producing flat densities"""
    
    # Create simple test data
    torch.manual_seed(42)
    data = torch.cat([torch.randn(500) - 2, torch.randn(500) + 2])  # Bimodal distribution
    
    print("=== Data Information ===")
    print(f"Data shape: {data.shape}")
    print(f"Data min: {data.min():.4f}")
    print(f"Data max: {data.max():.4f}")
    print(f"Data mean: {data.mean():.4f}")
    print(f"Data std: {data.std():.4f}")
    print(f"Unique values: {len(torch.unique(data))}")
    
    # Set up KDE parameters
    n = 128
    min_val, max_val = torch.min(data), torch.max(data)
    Range = max_val - min_val
    MIN = min_val - Range / 2
    MAX = max_val + Range / 2
    
    print(f"\n=== KDE Parameters ===")
    print(f"MIN: {MIN:.4f}")
    print(f"MAX: {MAX:.4f}")
    print(f"Range: {MAX - MIN:.4f}")
    print(f"n: {n}")
    
    # Bin the data
    R = MAX - MIN
    dx = R / (n - 1)
    xmesh = MIN + dx * torch.arange(n, dtype=data.dtype)
    
    # Get histogram
    initial_data = torch.histc(data, bins=n, min=MIN.item(), max=MAX.item())
    initial_data = initial_data / torch.sum(initial_data)
    
    print(f"\n=== Histogram Information ===")
    print(f"Histogram sum: {initial_data.sum():.4f}")
    print(f"Histogram max: {initial_data.max():.4f}")
    print(f"Histogram min: {initial_data.min():.4f}")
    print(f"Non-zero bins: {(initial_data > 0).sum()}")
    
    # DCT transform
    a = dct1d(initial_data)
    
    print(f"\n=== DCT Information ===")
    print(f"DCT coefficients shape: {a.shape}")
    print(f"First 5 DCT coefficients: {a[:5]}")
    print(f"DCT max: {a.max():.4f}")
    print(f"DCT min: {a.min():.4f}")
    
    # Bandwidth selection setup
    I = torch.arange(1, n, dtype=data.dtype) ** 2
    a2 = (a[1:] / 2) ** 2
    N = data.shape[0]
    
    print(f"\n=== Bandwidth Selection ===")
    print(f"N (sample size): {N}")
    print(f"I shape: {I.shape}")
    print(f"a2 shape: {a2.shape}")
    print(f"a2 max: {a2.max():.6e}")
    print(f"a2 min: {a2.min():.6e}")
    
    # Test fixed_point function at different values
    def fixed_point_kde(t, N, I, a2):
        l = 7
        t = torch.tensor(t, dtype=data.dtype) if not torch.is_tensor(t) else t
        N = torch.tensor(N, dtype=data.dtype) if not torch.is_tensor(N) else N
        
        pi = torch.tensor(np.pi, dtype=data.dtype)
        
        # Initial f calculation
        f = 2 * torch.pow(pi, 2*l) * torch.sum(torch.pow(I, l) * a2 * torch.exp(-I * pi**2 * t))
        
        for s in range(l-1, 1, -1):
            s_tensor = torch.tensor(s, dtype=data.dtype)
            
            # Use lgamma for numerical stability
            K0 = torch.exp(torch.lgamma(s_tensor + 1) - torch.lgamma(s_tensor/2 + 1) - 0.5 * torch.log(2 * pi))
            
            const = (1 + torch.pow(torch.tensor(0.5, dtype=data.dtype), s_tensor + 0.5)) / 3
            time = torch.pow(2 * const * K0 / N / f, 2 / (3 + 2*s_tensor))
            f = 2 * torch.pow(pi, 2*s_tensor) * torch.sum(torch.pow(I, s_tensor) * a2 * torch.exp(-I * pi**2 * time))
        
        out = t - torch.pow(2 * N * torch.sqrt(pi) * f, -2/5)
        return out, f  # Return both the output and f for debugging
    
    # Test fixed point at various t values
    t_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
    print("\n=== Fixed Point Function Tests ===")
    for t in t_values:
        fp_val, f_val = fixed_point_kde(t, N, I, a2)
        print(f"t={t:.4f}: fixed_point={fp_val:.6f}, f={f_val:.6e}")
    
    # Find optimal t using bisection
    t_star = find_root_bisection(lambda t: fixed_point_kde(t, N, I, a2)[0], 
                                 low=0.0, high=1.0, device=data.device, dtype=data.dtype)
    
    print(f"\n=== Optimal Bandwidth ===")
    print(f"t_star: {t_star:.6f}")
    
    # Apply smoothing
    pi = torch.tensor(np.pi, dtype=data.dtype)
    a_t = a * torch.exp(-torch.arange(n, dtype=data.dtype)**2 * pi**2 * t_star / 2)
    
    print(f"\n=== Smoothed DCT ===")
    print(f"First 5 smoothed coefficients: {a_t[:5]}")
    print(f"Smoothing factors (first 5): {torch.exp(-torch.arange(5, dtype=data.dtype)**2 * pi**2 * t_star / 2)}")
    
    # Get final density
    from utils.prob_op import idct1d
    density = idct1d(a_t) / R
    density = torch.clamp(density, min=0)
    
    print(f"\n=== Final Density ===")
    print(f"Density shape: {density.shape}")
    print(f"Density max: {density.max():.6f}")
    print(f"Density min: {density.min():.6f}")
    print(f"Density variance: {density.var():.6e}")
    print(f"Density sum * dx: {(density * dx).sum():.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Data histogram and KDE
    axes[0, 0].hist(data.numpy(), bins=50, density=True, alpha=0.5, label='Histogram')
    axes[0, 0].plot(xmesh.numpy(), density.numpy(), 'r-', linewidth=2, label='KDE')
    axes[0, 0].set_title('Data and KDE')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    # Plot 2: Initial histogram
    axes[0, 1].bar(xmesh.numpy(), initial_data.numpy(), width=dx.item())
    axes[0, 1].set_title('Initial Histogram (normalized)')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Normalized count')
    
    # Plot 3: DCT coefficients
    axes[1, 0].plot(a.numpy()[:30], 'b.-')
    axes[1, 0].set_title('DCT Coefficients (first 30)')
    axes[1, 0].set_xlabel('Coefficient index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True)
    
    # Plot 4: Fixed point function
    t_range = torch.linspace(0.0001, 0.1, 100)
    fp_values = []
    for t in t_range:
        fp_val, _ = fixed_point_kde(t, N, I, a2)
        fp_values.append(fp_val.item())
    
    axes[1, 1].plot(t_range.numpy(), fp_values)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', label='y=0')
    axes[1, 1].axvline(x=t_star.item(), color='g', linestyle='--', label=f't*={t_star:.4f}')
    axes[1, 1].set_title('Fixed Point Function')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('fixed_point(t)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('kde_debug_detailed.png', dpi=150)
    print("\nPlot saved as 'kde_debug_detailed.png'")

if __name__ == "__main__":
    debug_kde_detailed() 