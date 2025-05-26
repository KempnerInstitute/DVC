import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.prob_op import kde, dct1d, idct1d

# Generate test data
np.random.seed(42)
n_samples = 500
data_np = np.random.randn(n_samples)
data = torch.tensor(data_np, dtype=torch.float32)

print("Debug KDE Implementation:")
print(f"Data: mean={data.mean():.4f}, std={data.std():.4f}")
print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")

# Manual KDE implementation for debugging
n = 128  # Use smaller n for debugging
MIN = data.min() - 0.5
MAX = data.max() + 0.5
R = MAX - MIN

print(f"\nRange setup:")
print(f"  MIN: {MIN:.4f}")
print(f"  MAX: {MAX:.4f}")
print(f"  R: {R:.4f}")

# Create histogram
xmesh = torch.linspace(MIN, MAX, n)
counts = torch.histc(data, bins=n, min=MIN.item(), max=MAX.item())

print(f"\nHistogram:")
print(f"  Counts shape: {counts.shape}")
print(f"  Total counts: {counts.sum()}")
print(f"  Non-zero bins: {(counts > 0).sum()}")
print(f"  Counts range: [{counts.min()}, {counts.max()}]")

# Normalize histogram
initial_data = counts.float() / counts.sum()
print(f"\nNormalized histogram:")
print(f"  Sum: {initial_data.sum():.6f}")
print(f"  Range: [{initial_data.min():.6f}, {initial_data.max():.6f}]")

# DCT
a = dct1d(initial_data)
print(f"\nDCT coefficients:")
print(f"  Shape: {a.shape}")
print(f"  Range: [{a.min():.6f}, {a.max():.6f}]")
print(f"  First 10 coeffs: {a[:10]}")

# Plot to visualize
plt.figure(figsize=(15, 10))

# Plot 1: Original histogram
plt.subplot(2, 3, 1)
plt.bar(xmesh.numpy(), counts.numpy(), width=(xmesh[1]-xmesh[0]).item())
plt.title('Raw Histogram Counts')
plt.xlabel('x')
plt.ylabel('Count')

# Plot 2: Normalized histogram
plt.subplot(2, 3, 2)
plt.bar(xmesh.numpy(), initial_data.numpy(), width=(xmesh[1]-xmesh[0]).item())
plt.title('Normalized Histogram')
plt.xlabel('x')
plt.ylabel('Probability')

# Plot 3: DCT coefficients
plt.subplot(2, 3, 3)
plt.plot(a.numpy())
plt.title('DCT Coefficients')
plt.xlabel('Index')
plt.ylabel('Coefficient')

# Test bandwidth selection with simple values
I = torch.arange(1, n, dtype=torch.float32) ** 2
a2 = (a[1:] / 2) ** 2

# Try different bandwidth values
t_values = [0.001, 0.01, 0.1, 0.5, 1.0]
colors = ['red', 'green', 'blue', 'orange', 'purple']

plt.subplot(2, 3, 4)
for t, color in zip(t_values, colors):
    a_t = a * torch.exp(-torch.arange(n, dtype=torch.float32)**2 * np.pi**2 * t / 2)
    density = idct1d(a_t) / R
    plt.plot(xmesh.numpy(), density.numpy(), color=color, label=f't={t}')

plt.legend()
plt.title('KDE with different bandwidths')
plt.xlabel('x')
plt.ylabel('Density')

# Plot 5: Compare with true normal
plt.subplot(2, 3, 5)
from scipy.stats import norm
x_true = np.linspace(-4, 4, 1000)
y_true = norm.pdf(x_true, loc=0, scale=1)
plt.plot(x_true, y_true, 'k-', label='True N(0,1)', linewidth=2)

# Use optimal bandwidth from the algorithm
density_opt, xmesh_opt = kde(data, n=128, MIN=MIN, MAX=MAX)
plt.plot(xmesh_opt.numpy(), density_opt.numpy(), 'r--', label='KDE result', linewidth=2)
plt.hist(data_np, bins=30, density=True, alpha=0.3, label='Histogram')
plt.legend()
plt.title('Final KDE vs True Distribution')

# Plot 6: Check area under curve
plt.subplot(2, 3, 6)
dx = xmesh_opt[1] - xmesh_opt[0]
cumsum = torch.cumsum(density_opt * dx, dim=0)
plt.plot(xmesh_opt.numpy(), cumsum.numpy())
plt.title(f'Cumulative sum (final={cumsum[-1]:.4f})')
plt.xlabel('x')
plt.ylabel('Cumulative probability')

plt.tight_layout()
plt.savefig('kde_debug.png')
print("\nPlot saved as kde_debug.png")

# Now test the full KDE function
print("\n\nFull KDE test with n=2**14:")
density_full, xmesh_full = kde(data)
print(f"  Output shape: {density_full.shape}")
print(f"  Density range: [{density_full.min():.6f}, {density_full.max():.6f}]")
print(f"  Area under curve: {(density_full * (xmesh_full[1] - xmesh_full[0])).sum():.6f}") 