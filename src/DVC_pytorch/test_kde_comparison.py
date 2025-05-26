import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/DVC/src')

# Import PyTorch version
from DVC_pytorch.utils.prob_op import kde as kde_pytorch

# Import TensorFlow version
import tensorflow as tf
from DVC_tensorflow.utils.prob_op import kde as kde_tensorflow

# Generate test data
np.random.seed(42)
n_samples = 500
data_np = np.random.randn(n_samples)

print("Testing KDE implementations:")
print(f"Data: mean={data_np.mean():.4f}, std={data_np.std():.4f}")

# Test 1: PyTorch with n=2**14 (default)
data_torch = torch.tensor(data_np, dtype=torch.float32)
density_pt_full, xmesh_pt_full = kde_pytorch(data_torch)
print(f"\nPyTorch KDE (n=2**14):")
print(f"  Density shape: {density_pt_full.shape}")
print(f"  Density range: [{density_pt_full.min():.6f}, {density_pt_full.max():.6f}]")
print(f"  Density sum: {density_pt_full.sum():.6f}")

# Test 2: PyTorch with n=128 (for comparison)
density_pt_128, xmesh_pt_128 = kde_pytorch(data_torch, n=128)
print(f"\nPyTorch KDE (n=128):")
print(f"  Density shape: {density_pt_128.shape}")
print(f"  Density range: [{density_pt_128.min():.6f}, {density_pt_128.max():.6f}]")

# Test 3: TensorFlow with n=2**14
data_tf = tf.constant(data_np, dtype=tf.float32)
density_tf_full, xmesh_tf_full = kde_tensorflow(data_tf)
print(f"\nTensorFlow KDE (n=2**14):")
print(f"  Density shape: {density_tf_full.shape}")
print(f"  Density range: [{tf.reduce_min(density_tf_full).numpy():.6f}, {tf.reduce_max(density_tf_full).numpy():.6f}]")
print(f"  Density sum: {tf.reduce_sum(density_tf_full).numpy():.6f}")

# Convert TensorFlow outputs to numpy for plotting
density_tf_np = density_tf_full.numpy()
xmesh_tf_np = xmesh_tf_full.numpy()

# Plot comparison
plt.figure(figsize=(12, 8))

# True normal distribution
from scipy.stats import norm
x_true = np.linspace(-4, 4, 100)
y_true = norm.pdf(x_true, loc=0, scale=1)

# Plot 1: PyTorch full resolution
plt.subplot(2, 2, 1)
plt.hist(data_np, bins=30, density=True, alpha=0.5, label='Histogram')
# Downsample for plotting
indices = np.linspace(0, len(xmesh_pt_full)-1, 500).astype(int)
plt.plot(xmesh_pt_full[indices].cpu().numpy(), density_pt_full[indices].cpu().numpy(), 'r-', label='PyTorch KDE')
plt.plot(x_true, y_true, 'k--', label='True N(0,1)')
plt.legend()
plt.title('PyTorch KDE (n=2**14)')

# Plot 2: PyTorch 128 points
plt.subplot(2, 2, 2)
plt.hist(data_np, bins=30, density=True, alpha=0.5, label='Histogram')
plt.plot(xmesh_pt_128.cpu().numpy(), density_pt_128.cpu().numpy(), 'g-', label='PyTorch KDE')
plt.plot(x_true, y_true, 'k--', label='True N(0,1)')
plt.legend()
plt.title('PyTorch KDE (n=128)')

# Plot 3: TensorFlow
plt.subplot(2, 2, 3)
plt.hist(data_np, bins=30, density=True, alpha=0.5, label='Histogram')
# Downsample for plotting
indices_tf = np.linspace(0, len(xmesh_tf_np)-1, 500).astype(int)
plt.plot(xmesh_tf_np[indices_tf], density_tf_np[indices_tf], 'b-', label='TensorFlow KDE')
plt.plot(x_true, y_true, 'k--', label='True N(0,1)')
plt.legend()
plt.title('TensorFlow KDE (n=2**14)')

# Plot 4: Direct comparison
plt.subplot(2, 2, 4)
plt.plot(xmesh_pt_128.cpu().numpy(), density_pt_128.cpu().numpy(), 'g-', label='PyTorch (n=128)', linewidth=2)
# Interpolate TensorFlow to same grid
from scipy.interpolate import interp1d
f_tf = interp1d(xmesh_tf_np, density_tf_np, bounds_error=False, fill_value=0)
density_tf_interp = f_tf(xmesh_pt_128.cpu().numpy())
plt.plot(xmesh_pt_128.cpu().numpy(), density_tf_interp, 'b--', label='TensorFlow (interpolated)', linewidth=2)
plt.plot(x_true, y_true, 'k:', label='True N(0,1)', linewidth=2)
plt.legend()
plt.title('Direct Comparison')

plt.tight_layout()
plt.savefig('kde_comparison.png')
print("\nPlot saved as kde_comparison.png")

# Calculate errors
f_pt = interp1d(xmesh_pt_128.cpu().numpy(), density_pt_128.cpu().numpy(), bounds_error=False, fill_value=0)
kde_pt_at_true = f_pt(x_true)
mae_pt = np.mean(np.abs(kde_pt_at_true - y_true))

kde_tf_at_true = f_tf(x_true)
mae_tf = np.mean(np.abs(kde_tf_at_true - y_true))

print(f"\nMean Absolute Error vs true normal:")
print(f"  PyTorch: {mae_pt:.6f}")
print(f"  TensorFlow: {mae_tf:.6f}") 