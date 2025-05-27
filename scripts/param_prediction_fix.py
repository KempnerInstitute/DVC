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

# Parameters - change these as needed
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
# 3. Create and fit parametric vine
# ------------------------------------------------------------
print("\nFitting parametric gaussian vine...")

# Configuration
cfg = {
    'vine': {
        'family': 'c-vine',
        'method': 'optimal',
        'knots': 40
    },
    'general': {
        'param': True,
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

# Dictionaries for vine.fit()
gen_dict = {'param': True, 'binning': False, 'fitted': False}
npc_dict = {}
par_dict = {'param_families': ['gaussian']}
bin_dict = {'n_bin': 1}

# Create vine
vine = vine_obj_bin(
    'c-vine',
    ['gaussian'],
    dim,
    margins,
    knots=40,
    method='optimal'
)

# Fit the vine
vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
print(f"Vine structure: {vine.ind_vine}")

# Print actual copula parameters (Gaussian rho values)
print("\nFitted copula parameters:")
for level, edges in enumerate(vine.ind_vine):
    print(f"Level {level}:")
    for i, edge in enumerate(edges):
        copula = vine.copulas[level][i]
        if hasattr(copula, 'theta'):
            print(f"  Edge {edge}: rho = {copula.theta:.4f}")

# ------------------------------------------------------------
# 4. Generate samples and verify correlation
# ------------------------------------------------------------
print("\nGenerating samples...")
n_samples_new = 5000
samples = vine.sample(n_samples_new, cfg)

# Calculate correlation matrix of samples
sample_corr = np.corrcoef(samples, rowvar=False)

# Compute metrics:
# 1. Frobenius norm difference between true and fitted correlation matrices
corr_diff = np.linalg.norm(true_corr - sample_corr, 'fro')

# 2. Mean absolute difference of individual correlations
corr_mae = np.mean(np.abs(true_corr - sample_corr))

print(f"Correlation matrix difference (Frobenius): {corr_diff:.4f}")
print(f"Correlation MAE: {corr_mae:.4f}")

# ------------------------------------------------------------
# 5. Helper functions for conditional prediction
# ------------------------------------------------------------
# CORRECT conditional mean for multivariate Gaussian with uniform correlation
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

# Function to find ML estimate of conditional mean
def find_conditional_mean_ml(vine, fixed_vars, fixed_values, predict_var, search_range=None):
    """Find conditional mean using maximum likelihood"""
    if search_range is None:
        search_range = np.linspace(-5, 5, 200)  # Wider range
        
    # Create a test data point with fixed values
    # Initialize with the correct dimensions - use vine.n_cop to get the correct dimension
    test_data = np.zeros(vine.n_cop)
    for i, var_idx in enumerate(fixed_vars):
        test_data[var_idx] = fixed_values[i]
    
    # Search for best prediction
    best_val = None
    best_logp = -np.inf
    
    for val in search_range:
        # Copy test data and set the prediction variable
        x = test_data.copy()
        x[predict_var] = val
        
        # Calculate log probability under the vine
        x_tensor = torch.tensor([x], dtype=torch.float32)
        try:
            logp = vine.logpdf(x_tensor).item()
            
            # Update best if higher probability
            if logp > best_logp and np.isfinite(logp):
                best_logp = logp
                best_val = val
        except Exception as e:
            # Skip this value if there's an error
            continue
            
    # If no valid prediction was found, return 0 (neutral prediction)
    if best_val is None:
        return 0.0
            
    return best_val

# Function to compute conditional mean using analytical formula based on copula parameters
def compute_analytical_conditional_mean(vine, fixed_vars, fixed_values, predict_var):
    """
    Compute analytical conditional mean for Gaussian vine.
    
    For a C-vine with Gaussian copulas, this uses the fitted 
    rho parameters directly instead of ML search.
    """
    # For Gaussian vine, we can use the fact that 
    # conditional expectations are linear functions of the conditioning variables
    
    # Basic case: predict y from x using first level copula
    if len(fixed_vars) == 1 and fixed_vars[0] == 0:
        # Get the edge connecting root to predict_var
        for i, edge in enumerate(vine.ind_vine[0]):
            if edge[1] == predict_var:
                # Find the copula object
                cop = vine.copulas[0][i]
                if hasattr(cop, 'theta'):
                    rho = cop.theta
                    return rho * fixed_values[0]
    
    # More complex cases would need to implement paths through the vine
    # This is a simplified version that doesn't handle all cases
    
    # Fallback to ML search
    return find_conditional_mean_ml(vine, fixed_vars, fixed_values, predict_var)

# ------------------------------------------------------------
# 6. Test 1D conditional predictions with CORRECT formula
# ------------------------------------------------------------
print("\nRunning 1D conditional prediction tests...")

# Create test grid
n_test_points = 30
test_grid = np.linspace(-3, 3, n_test_points)

# Different prediction scenarios
test_cases = [
    {'fixed_var': 0, 'predict_var': 1, 'name': "Predict Var 1 given Var 0"},
    {'fixed_var': 1, 'predict_var': 0, 'name': "Predict Var 0 given Var 1"},
    {'fixed_var': 0, 'predict_var': 4, 'name': "Predict Var 4 given Var 0"}
]

# Plot test cases
plt.figure(figsize=(15, 5*len(test_cases)))

for i, test_case in enumerate(test_cases):
    fixed_var = test_case['fixed_var']
    predict_var = test_case['predict_var']
    name = test_case['name']
    
    print(f"\nTest case: {name}")
    
    # Results for different methods
    results = {
        'fixed_values': test_grid,
        'true_means': [],
        'ml_search': [],
        'analytical': []
    }
    
    # Compute predictions for each fixed value
    for fixed_val in test_grid:
        # True conditional mean with correct formula
        true_mean = true_gaussian_conditional_mean(
            [fixed_var], [fixed_val], predict_var, rho)
        results['true_means'].append(true_mean)
        
        # ML search prediction
        ml_pred = find_conditional_mean_ml(
            vine, [fixed_var], [fixed_val], predict_var)
        results['ml_search'].append(ml_pred)
        
        # Analytical prediction
        analytical_pred = compute_analytical_conditional_mean(
            vine, [fixed_var], [fixed_val], predict_var)
        results['analytical'].append(analytical_pred)
    
    # Calculate MSE for valid predictions only
    ml_results = np.array(results['ml_search'])
    analytical_results = np.array(results['analytical'])
    true_results = np.array(results['true_means'])
    
    # Replace any None values with np.nan
    ml_results = np.array([x if x is not None else np.nan for x in ml_results])
    analytical_results = np.array([x if x is not None else np.nan for x in analytical_results])
    
    # Use valid values only
    valid_ml = ~np.isnan(ml_results)
    valid_analytical = ~np.isnan(analytical_results)
    
    if np.any(valid_ml):
        ml_mse = np.mean(np.square(ml_results[valid_ml] - true_results[valid_ml]))
    else:
        ml_mse = np.nan
        
    if np.any(valid_analytical):
        analytical_mse = np.mean(np.square(analytical_results[valid_analytical] - true_results[valid_analytical]))
    else:
        analytical_mse = np.nan
    
    print(f"ML Search MSE: {ml_mse:.6f}")
    print(f"Analytical MSE: {analytical_mse:.6f}")
    
    # Plot results
    plt.subplot(len(test_cases), 1, i+1)
    plt.plot(results['fixed_values'], results['true_means'], 'k-', linewidth=2, label='True Mean')
    plt.scatter(results['fixed_values'], results['ml_search'], alpha=0.7, label=f'ML Search (MSE={ml_mse:.6f})')
    plt.scatter(results['fixed_values'], results['analytical'], alpha=0.7, label=f'Analytical (MSE={analytical_mse:.6f})')
    plt.xlabel(f"Variable {fixed_var}")
    plt.ylabel(f"Predicted Variable {predict_var}")
    plt.title(f"Conditional Prediction: {name}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig("improved_predictions.png")
print("\nSaved improved prediction plots to 'improved_predictions.png'")

# ------------------------------------------------------------
# 7. Test 2D conditional prediction
# ------------------------------------------------------------
print("\nRunning 2D conditional prediction test...")

# Choose two fixed variables and one to predict
fixed_vars = [0, 1]
predict_var = 4
name = f"Predict Var {predict_var} given Vars {fixed_vars}"

# Create 2D grid
n_test_points = 15  # Fewer points for 2D to keep computation reasonable
x_grid = np.linspace(-2, 2, n_test_points)
y_grid = np.linspace(-2, 2, n_test_points)
xx, yy = np.meshgrid(x_grid, y_grid)

# Results storage
true_means = np.zeros_like(xx)
ml_preds = np.zeros_like(xx)

# Compute predictions for each grid point
for i in range(n_test_points):
    for j in range(n_test_points):
        x_val = x_grid[i]
        y_val = y_grid[j]
        fixed_values = [x_val, y_val]
        
        # True conditional mean with correct formula
        true_mean = true_gaussian_conditional_mean(
            fixed_vars, fixed_values, predict_var, rho)
        true_means[j, i] = true_mean  # Note the j,i order for meshgrid
        
        # ML search prediction
        ml_pred = find_conditional_mean_ml(
            vine, fixed_vars, fixed_values, predict_var)
        ml_preds[j, i] = ml_pred if ml_pred is not None else np.nan

# Calculate MSE
valid_mask = ~np.isnan(ml_preds)
if np.any(valid_mask):
    ml_mse = np.mean(np.square(ml_preds[valid_mask] - true_means[valid_mask]))
    print(f"ML Search MSE (2D): {ml_mse:.6f}")
else:
    ml_mse = np.nan
    print("No valid predictions for 2D case")

# Plot 3D surface
fig = plt.figure(figsize=(15, 10))

# True conditional mean surface
ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(xx, yy, true_means, cmap='viridis', alpha=0.8)
ax1.set_xlabel(f'Variable {fixed_vars[0]}')
ax1.set_ylabel(f'Variable {fixed_vars[1]}')
ax1.set_zlabel(f'Variable {predict_var}')
ax1.set_title('True Conditional Mean')

# ML prediction surface
ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(xx, yy, ml_preds, cmap='plasma', alpha=0.8)
ax2.set_xlabel(f'Variable {fixed_vars[0]}')
ax2.set_ylabel(f'Variable {fixed_vars[1]}')
ax2.set_zlabel(f'Variable {predict_var}')
ax2.set_title(f'ML Prediction (MSE={ml_mse:.6f})')

# Error heatmap
ax3 = fig.add_subplot(223)
error = np.abs(ml_preds - true_means)
im = ax3.imshow(error, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], 
                origin='lower', cmap='hot')
plt.colorbar(im, ax=ax3, label='Absolute Error')
ax3.set_xlabel(f'Variable {fixed_vars[0]}')
ax3.set_ylabel(f'Variable {fixed_vars[1]}')
ax3.set_title('Prediction Error')

# Slices comparison
ax4 = fig.add_subplot(224)
mid_idx = n_test_points // 2
ax4.plot(x_grid, true_means[mid_idx, :], 'k-', label=f'True Mean (y={y_grid[mid_idx]:.2f})')
ax4.plot(x_grid, ml_preds[mid_idx, :], 'r--', label=f'ML Pred (y={y_grid[mid_idx]:.2f})')
ax4.set_xlabel(f'Variable {fixed_vars[0]}')
ax4.set_ylabel(f'Predicted Variable {predict_var}')
ax4.set_title('Slice Comparison')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig("2d_improved_prediction.png")
print("Saved 2D improved prediction plot to '2d_improved_prediction.png'")

# Show plots
plt.show() 