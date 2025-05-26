import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.prob_op import kernel_pdf2, kde

def debug_integer_kde():
    """Debug why integer data is producing NaN in kernel_pdf2"""
    print("=== Debugging Integer Data KDE ===")
    
    # Create integer data
    torch.manual_seed(42)
    data = torch.randint(0, 10, (500,)).float()
    
    print(f"Data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    print(f"Unique values: {torch.unique(data).numpy()}")
    print(f"Number of unique values: {len(torch.unique(data))}")
    print(f"Data min: {data.min():.2f}, max: {data.max():.2f}")
    
    # Test kernel_pdf2
    print("\n1. Testing kernel_pdf2:")
    try:
        density, mesh = kernel_pdf2(data)
        print(f"   Density shape: {density.shape}")
        print(f"   Mesh shape: {mesh.shape}")
        print(f"   Has NaN in density: {torch.any(torch.isnan(density))}")
        print(f"   Has NaN in mesh: {torch.any(torch.isnan(mesh))}")
        
        if torch.any(torch.isnan(density)):
            print(f"   NaN indices: {torch.where(torch.isnan(density))}")
            print(f"   Non-NaN density values: {density[~torch.isnan(density)][:5]}")
            
        if torch.any(torch.isnan(mesh)):
            print(f"   NaN mesh indices: {torch.where(torch.isnan(mesh))}")
            print(f"   Non-NaN mesh values: {mesh[~torch.isnan(mesh)][:5]}")
            
    except Exception as e:
        print(f"   Error in kernel_pdf2: {e}")
        import traceback
        traceback.print_exc()
    
    # Test kde directly
    print("\n2. Testing kde function directly:")
    try:
        density2, mesh2 = kde(data, n=128)
        print(f"   Density shape: {density2.shape}")
        print(f"   Mesh shape: {mesh2.shape}")
        print(f"   Has NaN in density: {torch.any(torch.isnan(density2))}")
        print(f"   Has NaN in mesh: {torch.any(torch.isnan(mesh2))}")
        print(f"   Density variance: {density2.var():.6f}")
        print(f"   Max density: {density2.max():.6f}")
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(data.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
        if not torch.any(torch.isnan(density2)):
            plt.plot(mesh2.numpy(), density2.numpy(), 'r-', linewidth=2, label='KDE')
        plt.title('Integer Data KDE')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        
    except Exception as e:
        print(f"   Error in kde: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with different ranges
    print("\n3. Testing with forced range:")
    try:
        MIN = data.min() - 1
        MAX = data.max() + 1
        density3, mesh3 = kde(data, n=128, MIN=MIN, MAX=MAX)
        print(f"   Using MIN={MIN:.2f}, MAX={MAX:.2f}")
        print(f"   Has NaN: {torch.any(torch.isnan(density3))}")
        print(f"   Density variance: {density3.var():.6f}")
        
        plt.subplot(1, 2, 2)
        plt.hist(data.numpy(), bins=30, density=True, alpha=0.5, label='Histogram')
        if not torch.any(torch.isnan(density3)):
            plt.plot(mesh3.numpy(), density3.numpy(), 'g-', linewidth=2, label='KDE (forced range)')
        plt.title('Integer Data KDE (Forced Range)')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        
    except Exception as e:
        print(f"   Error with forced range: {e}")
        import traceback
        traceback.print_exc()
    
    plt.tight_layout()
    plt.savefig('integer_kde_debug.png', dpi=150)
    print("\nPlot saved as 'integer_kde_debug.png'")

if __name__ == "__main__":
    debug_integer_kde() 