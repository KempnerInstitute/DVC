import torch
import numpy as np
from utils.prob_op import dct1d, idct1d

# Test 1: Simple signal
print("Test 1: Simple signal")
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Original: {x}")

# Apply DCT
y = dct1d(x)
print(f"DCT: {y}")

# Apply IDCT
x_reconstructed = idct1d(y)
print(f"Reconstructed: {x_reconstructed}")
print(f"Reconstruction error: {torch.norm(x - x_reconstructed):.6f}")

# Test 2: Random signal
print("\nTest 2: Random signal")
np.random.seed(42)
x = torch.randn(128)
y = dct1d(x)
x_reconstructed = idct1d(y)
print(f"Original shape: {x.shape}")
print(f"DCT shape: {y.shape}")
print(f"Reconstructed shape: {x_reconstructed.shape}")
print(f"Reconstruction error: {torch.norm(x - x_reconstructed):.6f}")

# Test 3: Gaussian-like signal (similar to histogram)
print("\nTest 3: Gaussian-like histogram")
t = torch.linspace(-3, 3, 128)
x = torch.exp(-t**2/2) / np.sqrt(2*np.pi)
x = x / x.sum()  # Normalize like histogram

y = dct1d(x)
x_reconstructed = idct1d(y)

print(f"Original sum: {x.sum():.6f}")
print(f"Reconstructed sum: {x_reconstructed.sum():.6f}")
print(f"Reconstruction error: {torch.norm(x - x_reconstructed):.6f}")

# Compare with numpy FFT-based DCT
print("\nTest 4: Compare with numpy")
x_np = x.numpy()
# Manual DCT using FFT (matching TensorFlow implementation)
n = len(x_np)
x_extended = np.concatenate([x_np, x_np[n-2:0:-1]])
X = np.fft.fft(x_extended)
dct_np = np.real(X[:n])

y_torch = y.numpy()
print(f"DCT difference (torch vs numpy): {np.linalg.norm(y_torch - dct_np):.6f}")

# Test the full cycle with bandwidth smoothing
print("\nTest 5: With bandwidth smoothing")
t_star = 0.01
pi = np.pi
smoothing = torch.exp(-torch.arange(128, dtype=torch.float32)**2 * pi**2 * t_star / 2)
y_smooth = y * smoothing
x_smooth = idct1d(y_smooth)

print(f"Smoothed sum: {x_smooth.sum():.6f}")
print(f"Smoothed max: {x_smooth.max():.6f}")
print(f"Smoothed min: {x_smooth.min():.6f}") 