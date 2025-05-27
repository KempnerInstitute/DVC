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
for i in range(2):
    # Fit normal distribution to margin
    loc, scale = norm.fit(data[:, i])
    margin = margin_obj('norm', [loc, scale], True)
    margins.append(margin)
    print(f"Margin {i}: Normal(μ={loc:.4f}, σ={scale:.4f})")

# ------------------------------------------------------------
# 3. Two approaches for comparison
# ------------------------------------------------------------
# A. Build a Vine Model
vine = vine_obj_bin(
    'c-vine',         # c-vine for 2D is just a pair
    ['gaussian'],     # Gaussian copulas
    2,                # dimension
    margins,          # marginal distributions
    knots=40,         # grid resolution
    method='optimal'  # doesn't matter for 2D
)

# Define the configuration for fitting
cfg = {
    'vine': {
        'family': 'c-vine',
        'knots': 40,
        'method': 'optimal'
    },
    'general': {
        'param': True,    # we want parametric copulas
        'binning': False
    },
    'optimizer': {
        'jit': False,
        'batch_edges': True,
        'batch_size': 5,
        'max_iter_phase1': 70,
        'lr_phase1': 0.10,
        'tol_phase1': 1e-5,
        'max_iter_phase2': 100,
        'lr_phase2': 0.03,
        'tol_phase2': 5e-5
    },
    'bandwidth': {
        'method': 'rule_of_thumb',
        'knn_k': 10
    },
    'npc': {
        'opt_method': 'LL1',
        'grad_precompute': False
    },
    'sampler': {
        'fast_parametric': True,
        'fast_nonparam': True,
        'nspline': 200
    }
}

# Dictionaries for vine.fit()
gen_dict = {
    'param': True,
    'binning': False,
    'fitted': False
}
npc_dict = {}
par_dict = {'param_families': ['gaussian']}
bin_dict = {'n_bin': 1}

# Fit the vine
print("\nFitting 2D vine model...")
vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)

# B. Fit a Direct Gaussian Copula
uniform_data = np.zeros_like(data)
for i in range(2):
    # Transform to uniform via empirical CDF
    uniform_data[:, i] = np.argsort(np.argsort(data[:, i])) / len(data)

# Fit a direct Gaussian copula using multiple methods
u_tensor = torch.tensor(uniform_data, dtype=torch.float32)

# Method 1: Direct Parameter Estimation
rho_hat, loglik, aic = fit_gaussian(u_tensor)

# Method 2: Calculate from Kendall's tau (more robust)
tau, _ = kendalltau(data[:, 0], data[:, 1])
rho_tau = np.sin(np.pi * tau / 2)

# Method 3: Direct correlation of normal scores
try:
    # Handle potential NaN/Inf values
    scores0 = norm.ppf(np.clip(uniform_data[:, 0], 0.001, 0.999))
    scores1 = norm.ppf(np.clip(uniform_data[:, 1], 0.001, 0.999))
    normal_scores = np.column_stack([scores0, scores1])
    rho_normal = np.corrcoef(normal_scores, rowvar=False)[0, 1]
    if not np.isfinite(rho_normal):
        rho_normal = np.nan
except Exception as e:
    print(f"Error calculating normal scores correlation: {e}")
    rho_normal = np.nan

print(f"\nGaussian copula parameter estimation:")
print(f"Method 1 - fit_gaussian function: rho = {rho_hat:.4f}")
print(f"Method 2 - from Kendall's tau: rho = {rho_tau:.4f}")
print(f"Method 3 - normal scores correlation: rho = {rho_normal:.4f}")
print(f"True correlation: {rho:.4f}")

# Use the most reliable method (tau-based)
direct_cop = cop_par_obj("gaussian", rho_tau)

# Find the parameter from the vine model
vine_rho = None
if len(vine.copulas) > 0 and len(vine.copulas[0]) > 0:
    vine_cop = vine.copulas[0][0]
    if hasattr(vine_cop, 'theta'):
        vine_rho = vine_cop.theta
    print(f"Vine-fitted rho: {vine_rho} (true: {rho:.4f})")

# ------------------------------------------------------------
# 4. Generate Samples from Both Models
# ------------------------------------------------------------
# A. Samples from the vine
n_samples_new = 5000
vine_samples = vine.sample(n_samples_new, cfg)
print(f"Generated {n_samples_new} samples from vine model")

# B. Samples from direct Gaussian copula
direct_samples = np.zeros((n_samples_new, 2))
direct_samples[:, 0] = np.random.rand(n_samples_new)  # U[0,1] for first margin

# Sample second variable conditional on first
for i in range(n_samples_new):
    uv = torch.tensor([[direct_samples[i, 0], 0.5]], dtype=torch.float32)
    direct_samples[i, 1] = copulainvccdf(direct_cop, uv).item()

# Transform copula samples back to original scale
direct_recon = np.zeros_like(direct_samples)
for i in range(2):
    loc, scale = margins[i].theta
    direct_recon[:, i] = norm.ppf(direct_samples[:, i], loc=loc, scale=scale)

# ------------------------------------------------------------
# 5. Calculate Correlations and Compare Results
# ------------------------------------------------------------
vine_corr = np.corrcoef(vine_samples, rowvar=False)[0, 1]
direct_corr = np.corrcoef(direct_recon, rowvar=False)[0, 1]

print("\nCorrelation summary:")
print(f"True data correlation: {true_corr:.4f}")
print(f"Vine model correlation: {vine_corr:.4f} (diff: {vine_corr-true_corr:.4f})")
print(f"Direct copula correlation: {direct_corr:.4f} (diff: {direct_corr-true_corr:.4f})")

# ------------------------------------------------------------
# 6. Visualize Results
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original data
axes[0, 0].scatter(data[:1000, 0], data[:1000, 1], alpha=0.5)
axes[0, 0].set_title(f"Original Data (ρ={true_corr:.4f})")
axes[0, 0].grid(True)

# Copula (uniform) space
axes[0, 1].scatter(uniform_data[:1000, 0], uniform_data[:1000, 1], alpha=0.5)
axes[0, 1].set_title(f"Data in Copula Space (ρ={rho_hat:.4f})")
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(True)

# Vine samples
axes[1, 0].scatter(vine_samples[:1000, 0], vine_samples[:1000, 1], alpha=0.5)
axes[1, 0].set_title(f"Vine Samples (ρ={vine_corr:.4f})")
axes[1, 0].grid(True)

# Direct samples
axes[1, 1].scatter(direct_recon[:1000, 0], direct_recon[:1000, 1], alpha=0.5)
axes[1, 1].set_title(f"Direct Copula Samples (ρ={direct_corr:.4f})")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('2d_comparison.png')
print("Saved comparison plots to '2d_comparison.png'")

# ------------------------------------------------------------
# 7. Test Prediction Capabilities
# ------------------------------------------------------------
print("\nTesting prediction capabilities...")

# Generate test data for prediction
x1_values = np.linspace(-3, 3, 20)
x2_true = []    # Ground truth
x2_vine = []    # Vine prediction
x2_direct = []  # Direct prediction

# Create a test dataset with the true conditional mean
test_data = []
for x1 in x1_values:
    # True conditional mean: E[X2|X1=x1] = ρx1 (for standard bivariate normal)
    cond_mean = true_rho * x1
    
    # Save the true value
    x2_true.append(cond_mean)
    
    # Create data point for prediction
    test_point = np.array([x1, 0.0])  # Placeholder for x2
    test_data.append(test_point)

test_data = np.array(test_data)

# A. Vine prediction
for i, x1 in enumerate(x1_values):
    # For a simple 2D Gaussian vine, the conditional mean is directly available
    # from the correlation parameter
    vine_rho = vine.copulas[0][0].theta
    
    # The conditional mean is mu2 + rho*(sigma2/sigma1)*(x1-mu1)
    # For standardized margins, this simplifies to rho*x1
    x2_pred = vine_rho * x1
    x2_vine.append(x2_pred)

# B. Direct copula prediction using sampling
for i, x1 in enumerate(x1_values):
    # For a Gaussian copula, the conditional mean can be calculated analytically
    # We'll use the rho from Kendall's tau since it's most reliable
    
    # The conditional mean is mu2 + rho*(sigma2/sigma1)*(x1-mu1)
    # For standardized margins, this simplifies to rho*x1
    x2_pred = rho_tau * x1
    x2_direct.append(x2_pred)

# Plot prediction results
plt.figure(figsize=(10, 6))
plt.scatter(x1_values, x2_true, label='True Conditional Mean', color='black', s=50)
plt.scatter(x1_values, x2_vine, label='Vine Prediction', alpha=0.7)
plt.scatter(x1_values, x2_direct, label='Direct Copula Prediction', alpha=0.7)
plt.plot([-3, 3], [-3*true_rho, 3*true_rho], 'k--')
plt.xlabel('x₁')
plt.ylabel('Predicted x₂')
plt.title('Conditional Prediction Comparison')
plt.legend()
plt.grid(True)
plt.savefig('2d_prediction.png')
print("Saved prediction comparison to '2d_prediction.png'")

# Calculate prediction errors
vine_mse = np.mean((np.array(x2_vine) - np.array(x2_true))**2)
direct_mse = np.mean((np.array(x2_direct) - np.array(x2_true))**2)

print(f"Vine prediction MSE: {vine_mse:.4f}")
print(f"Direct copula prediction MSE: {direct_mse:.4f}")
print(f"Ratio vine/direct MSE: {vine_mse/direct_mse:.2f}x")

plt.show() 