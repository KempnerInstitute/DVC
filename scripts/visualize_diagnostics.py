import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr

from DVC_pyolder.config import load_config
from DVC_pyolder.objects import vine_obj_bin, margin_obj
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# Load configuration and data
# ------------------------------------------------------------
CFG_PATH = Path(__file__).parent.parent / "configs" / "gauss_nd.yaml"
cfg = load_config(CFG_PATH if CFG_PATH.exists() else None)

# Data generation parameters
n_samples = cfg['data']['n_samples']
dim = cfg['data']['dim']
rho = cfg['data']['rho']

print(f"Generating {n_samples} samples from {dim}D Gaussian with rho={rho}")

# Synthetic Gaussian data
cov_true = np.full((dim, dim), rho)
np.fill_diagonal(cov_true, 1.0)
data = np.random.multivariate_normal(np.zeros(dim), cov_true, size=n_samples)

# ------------------------------------------------------------
# Build and fit vine model
# ------------------------------------------------------------
margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(dim)]

vine = vine_obj_bin(
    cfg['vine']['family'],
    ['gaussian'],
    dim,
    margins,
    knots=cfg['vine']['knots'],
    method=cfg['vine']['method']
)

# Fit dictionaries
gen_dict = {
    'param': cfg['general']['param'],
    'binning': cfg['general']['binning'],
    'fitted': False
}

npc_dict = cfg.get('npc', {})
par_dict = {'param_families': ['gaussian']}
bin_dict = {'n_bin': 1}

print("Fitting vine model...")
vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)

# ------------------------------------------------------------
# Generate samples from vine model
# ------------------------------------------------------------
n_samp = 5000
print(f"Generating {n_samp} samples from fitted vine model...")
samples = vine.sample(n_samp, cfg)

# ------------------------------------------------------------
# Visualization 1: Correlation matrices
# ------------------------------------------------------------
# Calculate correlation matrices
corr_data = np.corrcoef(data, rowvar=False)
corr_samples = np.corrcoef(samples, rowvar=False)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('True Data Correlation Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(corr_samples, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Vine Samples Correlation Matrix')

plt.tight_layout()
plt.savefig('correlation_matrices.png')
print("Saved correlation matrices to 'correlation_matrices.png'")

# ------------------------------------------------------------
# Visualization 2: Pairwise scatter plots
# ------------------------------------------------------------
# Limit to at most first 3 dimensions for visibility
plot_dim = min(3, dim)
fig, axes = plt.subplots(plot_dim, plot_dim, figsize=(12, 12))

# If we're only plotting 1 dimension, axes won't be a 2D array
if plot_dim == 1:
    axes = np.array([[axes]])

# If we're only plotting 2 dimensions, make sure axes is 2D
elif plot_dim == 2:
    if not isinstance(axes[0], np.ndarray):
        axes = axes.reshape(2, 2)

for i in range(plot_dim):
    for j in range(plot_dim):
        ax = axes[i, j]
        if i == j:  # Diagonal: histogram
            # Original data histogram (blue)
            ax.hist(data[:, i], bins=20, alpha=0.5, density=True, color='blue', label='Data')
            # Sampled data histogram (red)
            ax.hist(samples[:, i], bins=20, alpha=0.5, density=True, color='red', label='Vine')
            
            if i == 0:  # Add legend to first diagonal plot only
                ax.legend()
                
            ax.set_title(f'Dim {i}')
            
        elif j > i:  # Upper triangle: scatter
            # Plot fewer points for clarity
            idx_data = np.random.choice(len(data), size=min(500, len(data)), replace=False)
            idx_samples = np.random.choice(len(samples), size=min(500, len(samples)), replace=False)
            
            # Original data (blue dots)
            ax.scatter(data[idx_data, j], data[idx_data, i], alpha=0.5, s=10, 
                      color='blue', label='Data')
            
            # Sampled data (red dots)
            ax.scatter(samples[idx_samples, j], samples[idx_samples, i], alpha=0.5, s=10, 
                       color='red', label='Vine')
            
            if i == 0 and j == 1:  # Add legend to first upper triangle plot
                ax.legend()
                
            # Add correlation coefficients
            r_data = pearsonr(data[:, j], data[:, i])[0]
            r_samples = pearsonr(samples[:, j], samples[:, i])[0]
            ax.set_title(f'r_data={r_data:.2f}, r_vine={r_samples:.2f}')
            
        elif j < i:  # Lower triangle: empty
            ax.axis('off')

plt.tight_layout()
plt.savefig('pairwise_scatter.png')
print("Saved pairwise scatter plots to 'pairwise_scatter.png'")

# ------------------------------------------------------------
# Conditional prediction
# ------------------------------------------------------------
# Predict one variable given others, comparing true vs vine prediction
target_var = 0  # The variable to predict
condition_vars = list(range(1, dim))  # All other variables

# Prepare test data 
n_test = 1000
test_data = np.random.multivariate_normal(np.zeros(dim), cov_true, size=n_test)

# True conditional predictions (analytical for Gaussian)
def gaussian_conditional_mean(x_given, cov, target_idx, given_idx):
    """Compute conditional mean of a multivariate Gaussian."""
    sigma_11 = cov[target_idx, target_idx]
    sigma_12 = cov[target_idx, given_idx]
    sigma_22 = cov[np.ix_(given_idx, given_idx)]
    
    # Compute conditional mean: mu_1|2 = mu_1 + Sigma_12 * Sigma_22^(-1) * (x_2 - mu_2)
    # Since all means are 0, this simplifies to: Sigma_12 * Sigma_22^(-1) * x_2
    sigma_22_inv = np.linalg.inv(sigma_22)
    conditional_mean = sigma_12.dot(sigma_22_inv).dot(x_given)
    
    return conditional_mean

# True predictions
true_predictions = np.zeros(n_test)
for i in range(n_test):
    true_predictions[i] = gaussian_conditional_mean(
        test_data[i, condition_vars], 
        cov_true, 
        target_var, 
        condition_vars
    )

# Vine-based prediction (ML estimate)
vine_predictions = np.zeros(n_test)

# Initialize all variables with the observed values
for i in range(n_test):
    # Start with the observed values for conditioning variables
    x_condition = test_data[i, condition_vars]
    
    # Prepare a search grid for the target variable
    search_range = np.linspace(-3, 3, 100)  # Assuming standard normal margins
    log_probs = np.zeros_like(search_range)
    
    # For each potential value, compute the joint log probability
    for j, x_val in enumerate(search_range):
        # Create a complete data point
        x_full = np.zeros(dim)
        x_full[target_var] = x_val
        x_full[condition_vars] = x_condition
        
        # Compute log probability under the vine model
        log_probs[j] = vine.logpdf(torch.tensor([x_full], dtype=torch.float32)).item()
    
    # Find the value that maximizes the probability (ML estimate)
    best_idx = np.nanargmax(log_probs)
    vine_predictions[i] = search_range[best_idx]

# Evaluate prediction accuracy
mse_true = np.mean((true_predictions - test_data[:, target_var])**2)
mse_vine = np.mean((vine_predictions - test_data[:, target_var])**2)

print(f"\nPrediction of variable {target_var} given variables {condition_vars}:")
print(f"True model MSE: {mse_true:.4f}")
print(f"Vine model MSE: {mse_vine:.4f}")

# Plot prediction results
plt.figure(figsize=(10, 6))
plt.scatter(test_data[:, target_var], true_predictions, alpha=0.5, label='True model')
plt.scatter(test_data[:, target_var], vine_predictions, alpha=0.5, label='Vine model')
plt.plot([-3, 3], [-3, 3], 'k--', label='Perfect prediction')
plt.xlabel(f'True value of variable {target_var}')
plt.ylabel(f'Predicted value of variable {target_var}')
plt.title('Conditional Prediction Comparison')
plt.legend()
plt.grid(True)
plt.savefig('prediction_comparison.png')
print("Saved prediction comparison to 'prediction_comparison.png'")

plt.show() 