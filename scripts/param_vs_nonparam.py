import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from pathlib import Path

from DVC.config import load_config
from DVC.objects import vine_obj_bin, margin_obj

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
# 3. Create and fit different types of vines
# ------------------------------------------------------------
print("\nFitting different vine types...")

# Base configuration
base_cfg = {
    'vine': {
        'family': 'c-vine',
        'method': 'optimal',
        'knots': 40
    },
    'general': {
        'binning': False,
        'fitted': False
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

# Define vines to test
vine_types = [
    {
        'name': 'Parametric Gaussian',
        'param': True,
        'param_families': ['gaussian']
    },
    {
        'name': 'Non-Parametric',
        'param': False,
        'param_families': None
    }
]

# Create and fit each vine type
vines = []
for vine_type in vine_types:
    print(f"Fitting {vine_type['name']}...")
    
    # Update configuration
    cfg = base_cfg.copy()
    cfg['general'] = base_cfg['general'].copy()
    cfg['general']['param'] = vine_type['param']
    
    # Create dictionaries for vine.fit()
    gen_dict = {
        'param': vine_type['param'],
        'binning': False,
        'fitted': False
    }
    par_dict = {'param_families': vine_type['param_families'] or []}
    npc_dict = {}
    bin_dict = {'n_bin': 1}
    
    # Create vine
    vine = vine_obj_bin(
        'c-vine',
        vine_type['param_families'] or [],
        dim,
        margins,
        knots=40,
        method='optimal'
    )
    
    # Fit the vine
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
    
    # Store the vine with its name
    vines.append((vine, vine_type['name']))
    
    # Print vine structure
    print(f"  Vine structure: {vine.ind_vine}")

# ------------------------------------------------------------
# 4. Generate samples and compare correlations
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
plt.figure(figsize=(12, 5))
n_vines = len(vines)

# Custom function to plot correlation matrix heatmap
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
ax = plt.subplot(1, n_vines+1, 1)
plot_correlation_heatmap(ax, true_corr, "True Correlation")

# Plot each vine's sample correlation matrix
for i, result in enumerate(results):
    ax = plt.subplot(1, n_vines+1, i + 2)
    plot_correlation_heatmap(ax, result['sample_corr'], 
                           f"{result['name']}\nMAE: {result['corr_mae']:.4f}")

plt.tight_layout()
plt.savefig("param_vs_nonparam_corr.png")
print("Saved correlation comparison to 'param_vs_nonparam_corr.png'")

# ------------------------------------------------------------
# 6. Compare marginal distributions
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# First subplot: Original data correlation scatter
ax = axes[0]
idx1, idx2 = 0, 1  # Choose two dimensions to visualize
ax.scatter(data[:1000, idx1], data[:1000, idx2], alpha=0.5)
ax.set_title(f"Original Data (ρ={true_corr[idx1, idx2]:.2f})")
ax.set_xlabel(f"Dimension {idx1}")
ax.set_ylabel(f"Dimension {idx2}")
ax.grid(True)

# Compare vine type samples
for i, result in enumerate(results):
    ax = axes[i+1]
    samples = result['samples']
    ax.scatter(samples[:1000, idx1], samples[:1000, idx2], alpha=0.5)
    sample_corr = np.corrcoef(samples[:, idx1], samples[:, idx2])[0, 1]
    ax.set_title(f"{result['name']} (ρ={sample_corr:.2f})")
    ax.set_xlabel(f"Dimension {idx1}")
    ax.set_ylabel(f"Dimension {idx2}")
    ax.grid(True)

# Compare univariate marginals (dimension 0)
dim_idx = 0
ax = axes[3]
# Plot original data histogram
ax.hist(data[:, dim_idx], bins=30, density=True, alpha=0.5, label='Original')
# Plot each vine's samples
for result in results:
    samples = result['samples']
    ax.hist(samples[:, dim_idx], bins=30, density=True, alpha=0.5, label=result['name'])
ax.set_title(f"Dimension {dim_idx} Marginal")
ax.legend()
ax.grid(True)

# Compare univariate marginals (dimension 1)
dim_idx = 1
ax = axes[4]
# Plot original data histogram
ax.hist(data[:, dim_idx], bins=30, density=True, alpha=0.5, label='Original')
# Plot each vine's samples
for result in results:
    samples = result['samples']
    ax.hist(samples[:, dim_idx], bins=30, density=True, alpha=0.5, label=result['name'])
ax.set_title(f"Dimension {dim_idx} Marginal")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("param_vs_nonparam_marginals.png")
print("Saved marginal comparison to 'param_vs_nonparam_marginals.png'")

# ------------------------------------------------------------
# 7. Conditional prediction tests with CORRECT formula
# ------------------------------------------------------------
print("\nRunning conditional prediction tests...")

# Define several test cases where we fix some variables and predict others
test_cases = [
    # Fix the first variable, predict the last
    {'fixed_vars': [0], 'predict_var': dim-1, 'name': 'Predict last given first'},
    
    # Fix two variables, predict the last
    {'fixed_vars': [0, 1], 'predict_var': dim-1, 'name': f'Predict {dim-1} given [0,1]'},
    
    # Fix all but one variable, predict the middle
    {'fixed_vars': list(range(dim)), 'predict_var': dim//2, 'name': f'Predict {dim//2} given rest'},
]

# Remove the prediction variable from fixed_vars in the last test case
test_cases[-1]['fixed_vars'].remove(test_cases[-1]['predict_var'])

# Helper function: Correct conditional mean for multivariate Gaussian with uniform correlation
def true_gaussian_conditional_mean(fixed_vars, fixed_values, predict_var, rho):
    """
    Compute the true conditional mean for uniform correlation Gaussian
    
    For uniform correlation matrix with parameter rho, the conditional mean is:
    E[X_j | X_i1=x_i1, ... X_ik=x_ik] = rho * sum(x_i) / (1 + (k-1)*rho)
    
    where k is the number of conditioning variables
    """
    k = len(fixed_vars)
    fixed_sum = sum(fixed_values)
    
    # In uniform correlation case, this simplifies to:
    if k == 1:
        return rho * fixed_sum
    else:
        # Correct formula for multiple conditioning variables
        return rho * fixed_sum / (1 + (k-1)*rho)

# Create test grid 
n_test_points = 20
test_grid = np.linspace(-2, 2, n_test_points)

# For each test case
prediction_results = {}
for test_case in test_cases:
    print(f"\nRunning test: {test_case['name']}")
    fixed_vars = test_case['fixed_vars']
    predict_var = test_case['predict_var']
    
    # For one or two fixed variables, create a visualization
    if len(fixed_vars) <= 2:
        # Results storage
        results = {
            'fixed_values': [],
            'true_means': [],
        }
        
        # Store results for each vine
        for vine, name in vines:
            results[name] = []
            
        # Create grid of fixed values
        if len(fixed_vars) == 1:
            # 1D grid
            test_points = [(x,) for x in test_grid]
        else:
            # 2D grid
            xx, yy = np.meshgrid(test_grid, test_grid)
            test_points = [(x, y) for x, y in zip(xx.flatten(), yy.flatten())]
        
        # Test each point in the grid
        for test_point in test_points:
            results['fixed_values'].append(test_point)
            
            # Compute true conditional mean with CORRECTED formula
            true_mean = true_gaussian_conditional_mean(
                fixed_vars, test_point, predict_var, rho)
            results['true_means'].append(true_mean)
            
            # Test each vine
            for vine, name in vines:
                # Create a test data point with fixed values
                test_data = np.zeros(dim)
                for i, var_idx in enumerate(fixed_vars):
                    test_data[var_idx] = test_point[i]
                
                # Search for best prediction using maximum likelihood
                # Use WIDER search range (-5 to 5 instead of -3 to 3)
                search_range = np.linspace(-5, 5, 200)
                best_val = None
                best_logp = -np.inf
                
                for val in search_range:
                    # Copy test data and set the prediction variable
                    x = test_data.copy()
                    x[predict_var] = val
                    
                    # Calculate log probability under the vine
                    x_tensor = torch.tensor([x], dtype=torch.float32)
                    logp = vine.logpdf(x_tensor).item()
                    
                    # Update best if higher probability
                    if logp > best_logp and np.isfinite(logp):
                        best_logp = logp
                        best_val = val
                        
                results[name].append(best_val)
        
        # Convert to arrays for plotting
        fixed_values = np.array(results['fixed_values'])
        true_means = np.array(results['true_means'])
        
        # For 1D fixed variable
        if len(fixed_vars) == 1:
            plt.figure(figsize=(10, 6))
            
            # Plot true conditional means
            plt.plot(fixed_values, true_means, 'k-', linewidth=2, label='True Mean')
            
            # Plot vine predictions
            for vine, name in vines:
                vine_preds = np.array(results[name])
                plt.scatter(fixed_values, vine_preds, alpha=0.7, label=name)
                
                # Calculate MSE
                mse = np.mean((vine_preds - true_means) ** 2)
                print(f"  {name} MSE: {mse:.6f}")
                
                # Store in prediction_results
                if name not in prediction_results:
                    prediction_results[name] = []
                prediction_results[name].append({
                    'test_case': test_case['name'],
                    'mse': mse
                })
            
            plt.xlabel(f"Variable {fixed_vars[0]} Value")
            plt.ylabel(f"Predicted Variable {predict_var} Value")
            plt.title(f"Conditional Prediction: {test_case['name']}")
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            plt.savefig(f"param_vs_nonparam_{fixed_vars[0]}_to_{predict_var}.png")

# ------------------------------------------------------------
# 8. Summary of prediction MSE across test cases
# ------------------------------------------------------------
print("\nPrediction MSE Summary:")

for name, results in prediction_results.items():
    print(f"\n{name}:")
    total_mse = 0
    for result in results:
        print(f"  {result['test_case']}: MSE = {result['mse']:.6f}")
        total_mse += result['mse']
    print(f"  Average MSE: {total_mse/len(results):.6f}")

plt.show() 