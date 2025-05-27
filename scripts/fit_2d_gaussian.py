import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm, kendalltau
from pathlib import Path

from DVC_pyolder.config import load_config
from DVC_pyolder.objects import vine_obj_bin, margin_obj, cop_par_obj
from DVC_pyolder.param_copula import fit_gaussian, copulapdf, copulainvccdf
from DVC_pyolder.vine_model import _h_function

# ------------------------------------------------------------
# 1. Generate 2D Gaussian Data
# ------------------------------------------------------------
def generate_gaussian(n_samples=5000, rho=0.7):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    return data, rho

np.random.seed(42)  # For reproducibility
n_samples = 5000
true_rho = 0.7

data, rho = generate_gaussian(n_samples, true_rho)
print(f"Generated {n_samples} samples from 2D Gaussian with rho={rho}")

# Calculate true correlation
true_corr = np.corrcoef(data, rowvar=False)[0, 1]
print(f"True correlation in data: {true_corr:.4f}")

# ------------------------------------------------------------
# 2. Fit Marginal Distributions
# ------------------------------------------------------------
margins = []
uniform_data = np.zeros_like(data)

for i in range(2):
    # Fit normal distribution to margin
    loc, scale = norm.fit(data[:, i])
    margin = margin_obj('norm', [loc, scale], True)
    margins.append(margin)
    
    # Transform to uniform via empirical CDF (for copula fitting)
    uniform_data[:, i] = np.argsort(np.argsort(data[:, i])) / len(data)
    
    print(f"Margin {i}: Normal(μ={loc:.4f}, σ={scale:.4f})")

# Compare original and transformed distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original margins
for i in range(2):
    axes[0, i].hist(data[:, i], bins=30, density=True, alpha=0.7)
    x = np.linspace(data[:, i].min(), data[:, i].max(), 100)
    pdf = norm.pdf(x, loc=margins[i].theta[0], scale=margins[i].theta[1])
    axes[0, i].plot(x, pdf, 'r-', linewidth=2)
    axes[0, i].set_title(f"Margin {i} - Original")
    axes[0, i].grid(True)

# Transformed uniform margins
for i in range(2):
    axes[1, i].hist(uniform_data[:, i], bins=30, density=True, alpha=0.7)
    axes[1, i].plot([0, 1], [1, 1], 'r-', linewidth=2)  # Uniform density is 1
    axes[1, i].set_title(f"Margin {i} - Uniform")
    axes[1, i].set_xlim(0, 1)
    axes[1, i].grid(True)

plt.tight_layout()
plt.savefig('2d_margins.png')
print("Saved margin plots to '2d_margins.png'")

# ------------------------------------------------------------
# 3. Fit Gaussian Copula
# ------------------------------------------------------------
u_tensor = torch.tensor(uniform_data, dtype=torch.float32)

# Method 1: Use DVC's fit_gaussian function
rho_hat, loglik, aic = fit_gaussian(u_tensor)
print(f"\nGaussian copula fit:")
print(f"Estimated rho: {rho_hat:.4f} (true: {rho:.4f})")

# Method 2: Calculate from Kendall's tau
tau, _ = kendalltau(data[:, 0], data[:, 1])
rho_tau = np.sin(np.pi * tau / 2)
print(f"Kendall's tau: {tau:.4f}")
print(f"rho from tau: {rho_tau:.4f}")

# Create copula object
cop = cop_par_obj("gaussian", rho_hat)

# ------------------------------------------------------------
# 4. Sample from Fitted Model
# ------------------------------------------------------------
# Generate samples from the copula
n_samples_new = 5000
cop_samples = np.zeros((n_samples_new, 2))
cop_samples[:, 0] = np.random.rand(n_samples_new)  # U[0,1] for first margin

# Sample second variable conditional on first
for i in range(n_samples_new):
    uv = torch.tensor([[cop_samples[i, 0], 0.5]], dtype=torch.float32)  # Dummy v value
    cop_samples[i, 1] = copulainvccdf(cop, uv).item()

# Transform copula samples back to original scale
data_recon = np.zeros_like(cop_samples)
for i in range(2):
    loc, scale = margins[i].theta
    data_recon[:, i] = norm.ppf(cop_samples[:, i], loc=loc, scale=scale)

# Calculate correlation in reconstructed data
recon_corr = np.corrcoef(data_recon, rowvar=False)[0, 1]
print(f"\nCorrelation in reconstructed data: {recon_corr:.4f}")
print(f"Difference from true correlation: {recon_corr - true_corr:.4f}")

# ------------------------------------------------------------
# 5. Visualize Results
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data
axes[0, 0].scatter(data[:1000, 0], data[:1000, 1], alpha=0.5)
axes[0, 0].set_title(f"Original Data (ρ={true_corr:.4f})")
axes[0, 0].grid(True)

# Copula (uniform) space
axes[0, 1].scatter(uniform_data[:1000, 0], uniform_data[:1000, 1], alpha=0.5)
axes[0, 1].set_title(f"Data in Copula Space (fitted ρ={rho_hat:.4f})")
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(True)

# Sampled copula
axes[1, 0].scatter(cop_samples[:1000, 0], cop_samples[:1000, 1], alpha=0.5)
axes[1, 0].set_title(f"Sampled from Copula")
axes[1, 0].set_xlim(0, 1)
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(True)

# Reconstructed data
axes[1, 1].scatter(data_recon[:1000, 0], data_recon[:1000, 1], alpha=0.5)
axes[1, 1].set_title(f"Reconstructed Data (ρ={recon_corr:.4f})")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('2d_copula_fit.png')
print("Saved copula fit plots to '2d_copula_fit.png'")

# ------------------------------------------------------------
# 6. Test h-function individually (conditional CDF)
# ------------------------------------------------------------
print("\nTesting h-function (conditional CDF)...")

# Test prediction: Given x₁, predict x₂
test_points = np.linspace(-3, 3, 10)  # Test values for x₁

fig, ax = plt.subplots(figsize=(8, 6))

# Plot original data
ax.scatter(data[:500, 0], data[:500, 1], alpha=0.2, color='blue', label='Data')

# For each test point, calculate true conditional mean and variance
for x1 in test_points:
    # True Gaussian conditional mean: μ₂|₁ = μ₂ + ρ(σ₂/σ₁)(x₁-μ₁)
    # Here μ₁=μ₂=0, σ₁=σ₂=1, so: μ₂|₁ = ρx₁
    cond_mean = true_rho * x1
    
    # Mark the conditional mean point
    ax.scatter([x1], [cond_mean], color='red', s=50, zorder=10)
    
    # Draw a line connecting these points
    if x1 == test_points[0]:
        ax.plot(test_points, true_rho * test_points, 'r-', label='True Conditional Mean')

# Test the h-function directly
u_tests = np.linspace(0.1, 0.9, 5)  # Test values for u₁
results = []

for u1 in u_tests:
    # Create grid of u₂ values
    u2_grid = np.linspace(0.01, 0.99, 100)
    
    # Apply h-function: h(u₂|u₁) for each u₂
    h_vals = []
    for u2 in u2_grid:
        u_root = torch.tensor([u1], dtype=torch.float32)
        u_other = torch.tensor([u2], dtype=torch.float32)
        h = _h_function(u_root, u_other, cop, None, side="left").item()
        h_vals.append(h)
    
    # h-function should be roughly diagonal for independent u's
    # or curved for dependent u's
    results.append((u1, u2_grid, h_vals))
    
# Plot h-function results
fig, axes = plt.subplots(1, len(u_tests), figsize=(15, 4))
for i, (u1, u2_grid, h_vals) in enumerate(results):
    ax = axes[i]
    ax.plot(u2_grid, h_vals)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Reference diagonal
    ax.set_title(f"u₁ = {u1:.2f}")
    ax.set_xlabel("u₂")
    ax.set_ylabel("h(u₂|u₁)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

plt.tight_layout()
plt.savefig('h_function_test.png')
print("Saved h-function test to 'h_function_test.png'")

plt.show() 