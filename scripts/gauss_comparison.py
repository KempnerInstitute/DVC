import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from pathlib import Path

from DVC_pyolder.config import load_config
from DVC_pyolder.objects import vine_obj_bin, margin_obj

# ------------------------------------------------------------
# 1. Generate multivariate Gaussian data
# ------------------------------------------------------------
def generate_gaussian_data(n_samples, dim, rho=0.6, seed=42):
    """Generate samples from a multivariate Gaussian with uniform correlation rho"""
    np.random.seed(seed)
    
    # Create correlation matrix with uniform correlation
    cov = np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)
    
    # Generate samples
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    return data, cov

# Parameters
n_samples = 5000
dim = 5
rho = 0.6

# Generate data
data, cov_true = generate_gaussian_data(n_samples, dim, rho)
print(f"Generated {n_samples} samples from {dim}D Gaussian with rho={rho}")

# Calculate and display true correlation matrix
true_corr = np.corrcoef(data, rowvar=False)
print("\nTrue correlation matrix:")
print(np.round(true_corr, 3))

# ------------------------------------------------------------
# 2. Fit margins
# ------------------------------------------------------------
margins = []
for i in range(dim):
    # Fit normal distribution to margin
    loc, scale = norm.fit(data[:, i])
    margin = margin_obj('norm', [loc, scale], True)
    margins.append(margin)
    print(f"Margin {i}: Normal(μ={loc:.4f}, σ={scale:.4f})")

# ------------------------------------------------------------
# 3. Create and fit different vine structures
# ------------------------------------------------------------
print("\nFitting different vine structures...")

# Base configuration
base_cfg = {
    'vine': {
        'knots': 40
    },
    'general': {
        'param': True,
        'binning': False
    },
    'optimizer': {
        'jit': True,
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
        'grad_precompute': True
    },
    'sampler': {
        'fast_parametric': True,
        'fast_nonparam': True,
        'nspline': 200
    }
}

# Dictionaries for vine.fit()
gen_dict = {'param': True, 'binning': False, 'fitted': False}
npc_dict = {}
par_dict = {'param_families': ['gaussian']}
bin_dict = {'n_bin': 1}

# Define different vine structures to test
vine_structures = [
    {'family': 'c-vine', 'method': 'random', 'name': 'C-Vine (Random)'},
    {'family': 'c-vine', 'method': 'optimal', 'name': 'C-Vine (Optimal)'},
    {'family': 'd-vine', 'method': 'random', 'name': 'D-Vine (Random)'},
    {'family': 'd-vine', 'method': 'optimal', 'name': 'D-Vine (Optimal)'},
    {'family': 'r-vine', 'method': 'random', 'name': 'R-Vine (Random)'},
]

# Create and fit each vine structure
vines = []
for structure in vine_structures:
    print(f"Fitting {structure['name']}...")
    
    # Update configuration
    cfg = base_cfg.copy()
    cfg['vine'] = base_cfg['vine'].copy()
    cfg['vine']['family'] = structure['family']
    cfg['vine']['method'] = structure['method']
    
    # Create vine
    vine = vine_obj_bin(
        structure['family'],
        ['gaussian'],
        dim,
        margins,
        knots=40,
        method=structure['method']
    )
    
    # Fit the vine
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
    
    # Store the vine with its name
    vines.append((vine, structure['name']))
    
    # Print vine structure
    print(f"  Vine structure: {vine.ind_vine}")

# ------------------------------------------------------------
# 4. Generate samples from each vine and compute statistics
# ------------------------------------------------------------
n_samples_new = 5000
results = []

for vine, name in vines:
    # Generate samples
    print(f"Generating samples from {name}...")
    samples = vine.sample(n_samples_new, base_cfg)
    
    # Calculate correlation matrix of samples
    sample_corr = np.corrcoef(samples, rowvar=False)
    
    # Compute metrics:
    # 1. Frobenius norm difference between true and fitted correlation matrices
    corr_diff = np.linalg.norm(true_corr - sample_corr, 'fro')
    
    # 2. Mean absolute difference of individual correlations
    corr_mae = np.mean(np.abs(true_corr - sample_corr))
    
    # Store results
    results.append({
        'name': name,
        'samples': samples, 
        'sample_corr': sample_corr,
        'corr_diff_frob': corr_diff,
        'corr_mae': corr_mae
    })
    
    print(f"  Correlation matrix difference (Frobenius): {corr_diff:.4f}")
    print(f"  Correlation MAE: {corr_mae:.4f}")

# ------------------------------------------------------------
# 5. Visualize correlation matrices
# ------------------------------------------------------------
plt.figure(figsize=(15, 10))
n_vines = len(vines)

# Custom function to plot correlation matrix heatmap (similar to seaborn)
def plot_correlation_heatmap(ax, corr_matrix, title, add_text=True):
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(title)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    if add_text:
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha="center", va="center", 
                        color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")

# Plot true correlation matrix
ax = plt.subplot(2, n_vines, 1)
plot_correlation_heatmap(ax, true_corr, "True Correlation")

# Plot each vine's sample correlation matrix
for i, result in enumerate(results):
    ax = plt.subplot(2, n_vines, i + 2)
    plot_correlation_heatmap(ax, result['sample_corr'], 
                           f"{result['name']}\nMAE: {result['corr_mae']:.4f}")

plt.tight_layout()
plt.savefig("correlation_comparison.png")
print("Saved correlation comparison to 'correlation_comparison.png'")

# ------------------------------------------------------------
# 6. Compare marginal distributions
# ------------------------------------------------------------
# Create a figure with dim rows and n_vines+1 columns
fig, axes = plt.subplots(dim, n_vines+1, figsize=(15, 3*dim))

# First column: original data
for i in range(dim):
    if dim > 1:
        ax = axes[i, 0]
    else:
        ax = axes[0]
        
    ax.hist(data[:, i], bins=30, density=True, alpha=0.7)
    x = np.linspace(data[:, i].min(), data[:, i].max(), 100)
    pdf = norm.pdf(x, loc=margins[i].theta[0], scale=margins[i].theta[1])
    ax.plot(x, pdf, 'r-', linewidth=2)
    ax.set_title(f"Original - Dim {i}")
    ax.grid(True)

# Remaining columns: samples from each vine
for j, result in enumerate(results):
    for i in range(dim):
        if dim > 1:
            ax = axes[i, j+1]
        else:
            ax = axes[j+1]
            
        samples = result['samples']
        ax.hist(samples[:, i], bins=30, density=True, alpha=0.7)
        x = np.linspace(samples[:, i].min(), samples[:, i].max(), 100)
        pdf = norm.pdf(x, loc=margins[i].theta[0], scale=margins[i].theta[1])
        ax.plot(x, pdf, 'r-', linewidth=2)
        if i == 0:
            ax.set_title(result['name'])
        ax.grid(True)

plt.tight_layout()
plt.savefig("margin_comparison.png")
print("Saved marginal comparison to 'margin_comparison.png'")

# ------------------------------------------------------------
# 7. Plot selected scatter plots
# ------------------------------------------------------------
fig, axes = plt.subplots(3, n_vines+1, figsize=(15, 12))

# Pairs to plot
pairs = [(0, 1), (0, 2), (1, 2)] if dim > 2 else [(0, 1)]
pair_titles = [f"Dim {i} vs Dim {j}" for i, j in pairs]

# First column: original data
for p, (i, j) in enumerate(pairs):
    axes[p, 0].scatter(data[:1000, i], data[:1000, j], alpha=0.5)
    axes[p, 0].set_title(f"Original - {pair_titles[p]}")
    axes[p, 0].grid(True)

# Remaining columns: samples from each vine
for v, result in enumerate(results):
    for p, (i, j) in enumerate(pairs):
        samples = result['samples']
        axes[p, v+1].scatter(samples[:1000, i], samples[:1000, j], alpha=0.5)
        if p == 0:
            axes[p, v+1].set_title(result['name'])
        axes[p, v+1].grid(True)
        
        # Add correlation annotation
        corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
        true_c = true_corr[i, j]
        axes[p, v+1].annotate(f"ρ={corr:.2f} (true={true_c:.2f})", 
                             xy=(0.05, 0.95), 
                             xycoords='axes fraction', 
                             fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig("scatter_comparison.png")
print("Saved scatter comparison to 'scatter_comparison.png'")

plt.show() 