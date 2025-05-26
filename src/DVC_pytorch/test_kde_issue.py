import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.prob_op import kde, kernel_pdf2

# Generate test data - standard normal
np.random.seed(42)
n_samples = 500
data_np = np.random.randn(n_samples)

# Convert to torch
data = torch.tensor(data_np, dtype=torch.float32)

print("Test 1: Basic KDE test")
print("Data stats:")
print(f"  Mean: {data.mean():.4f}")
print(f"  Std: {data.std():.4f}")
print(f"  Min: {data.min():.4f}")
print(f"  Max: {data.max():.4f}")

# Test kde function directly
density, xmesh = kde(data, n=128, MIN=data.min() - 0.5, MAX=data.max() + 0.5)

print("\nKDE output:")
print(f"  Density shape: {density.shape}")
print(f"  Density range: [{density.min():.6f}, {density.max():.6f}]")
print(f"  Density sum: {density.sum():.6f}")

# Normalize properly
dx = (xmesh[1] - xmesh[0])
area = torch.sum(density * dx)
print(f"  Area under curve: {area:.6f}")

# Plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(data_np, bins=30, density=True, alpha=0.5, label='Histogram')
plt.plot(xmesh.cpu().numpy(), density.cpu().numpy(), 'r-', label='KDE')

# True normal distribution
from scipy.stats import norm
x_true = np.linspace(-4, 4, 100)
y_true = norm.pdf(x_true, loc=0, scale=1)
plt.plot(x_true, y_true, 'k--', label='True N(0,1)')
plt.legend()
plt.title('KDE vs True Distribution')

# Test kernel_pdf2
density2, mesh2 = kernel_pdf2(data)
plt.subplot(1, 2, 2)
plt.hist(data_np, bins=30, density=True, alpha=0.5, label='Histogram')
plt.plot(mesh2.cpu().numpy(), density2.cpu().numpy(), 'g-', label='kernel_pdf2')
plt.plot(x_true, y_true, 'k--', label='True N(0,1)')
plt.legend()
plt.title('kernel_pdf2 vs True Distribution')

plt.tight_layout()
plt.savefig('kde_test.png')
print("\nPlot saved as kde_test.png")

# Calculate error
# Interpolate KDE at true distribution points
from scipy.interpolate import interp1d
f_kde = interp1d(xmesh.cpu().numpy(), density.cpu().numpy(), bounds_error=False, fill_value=0)
kde_at_true = f_kde(x_true)
mae = np.mean(np.abs(kde_at_true - y_true))
print(f"\nMean Absolute Error vs true normal: {mae:.6f}") 