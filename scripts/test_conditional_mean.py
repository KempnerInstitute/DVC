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
# 3. Create and fit the parametric vine
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
        'batch_size': 5
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
        'fast_nonparam': True
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

# ------------------------------------------------------------
# 4. Test different prediction scenarios using the new method
# ------------------------------------------------------------
print("\nTesting conditional_mean predictions...")

# Helper function for the true conditional mean
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

# Test cases
test_cases = [
    {'fixed_vars': [0], 'predict_var': 1, 'name': "Root → Child: 0→1"},
    {'fixed_vars': [1], 'predict_var': 0, 'name': "Child → Root: 1→0"},
    {'fixed_vars': [0], 'predict_var': 4, 'name': "Root → Distant: 0→4"},
    {'fixed_vars': [1], 'predict_var': 2, 'name': "Same Level: 1→2"},
    {'fixed_vars': [0, 1], 'predict_var': 4, 'name': "Multiple: [0,1]→4"},
    {'fixed_vars': [0, 2], 'predict_var': 4, 'name': "Multiple: [0,2]→4"},
    {'fixed_vars': [3, 4], 'predict_var': 0, 'name': "Distant: [3,4]→0"}
]

# Create test grid
n_test_points = 20
test_grid = np.linspace(-3, 3, n_test_points)

# Prepare for plotting
plt.figure(figsize=(16, 12))
n_rows = (len(test_cases) + 1) // 2
n_cols = min(2, len(test_cases))

for i, test_case in enumerate(test_cases):
    print(f"\nTest case: {test_case['name']}")
    
    if len(test_case['fixed_vars']) == 1:
        # 1D case - test over a grid
        fixed_var = test_case['fixed_vars'][0]
        predict_var = test_case['predict_var']
        
        # Results
        true_means = []
        vine_preds = []
        
        # For each point in the grid
        for x in test_grid:
            # True mean
            true_mean = true_gaussian_conditional_mean([fixed_var], [x], predict_var, rho)
            true_means.append(true_mean)
            
            # Vine prediction
            pred = vine.conditional_mean([fixed_var], [x], predict_var)
            vine_preds.append(pred)
        
        # Convert to arrays
        true_means = np.array(true_means)
        vine_preds = np.array(vine_preds)
        
        # Calculate MSE
        mse = np.mean((true_means - vine_preds) ** 2)
        print(f"  MSE: {mse:.6f}")
        
        # Plot results
        ax = plt.subplot(n_rows, n_cols, i+1)
        plt.plot(test_grid, true_means, 'k-', linewidth=2, label='True')
        plt.scatter(test_grid, vine_preds, color='r', alpha=0.7, label='Vine')
        plt.xlabel(f"Variable {fixed_var}")
        plt.ylabel(f"Predicted Variable {predict_var}")
        plt.title(f"{test_case['name']} (MSE={mse:.6f})")
        plt.legend()
        plt.grid(True)
        
    elif len(test_case['fixed_vars']) == 2:
        # 2D case - use a smaller grid or random points
        fixed_vars = test_case['fixed_vars']
        predict_var = test_case['predict_var']
        
        # Create grid
        grid_size = 7  # smaller grid for 2D
        grid1 = np.linspace(-2, 2, grid_size)
        grid2 = np.linspace(-2, 2, grid_size)
        xx, yy = np.meshgrid(grid1, grid2)
        
        # Results
        true_means = np.zeros_like(xx)
        vine_preds = np.zeros_like(xx)
        
        # For each point in the grid
        for i in range(grid_size):
            for j in range(grid_size):
                x = grid1[i]
                y = grid2[j]
                
                # True mean
                true_mean = true_gaussian_conditional_mean(fixed_vars, [x, y], predict_var, rho)
                true_means[j, i] = true_mean  # Note j,i ordering for meshgrid
                
                # Vine prediction
                pred = vine.conditional_mean(fixed_vars, [x, y], predict_var)
                vine_preds[j, i] = pred
        
        # Calculate MSE
        mse = np.mean((true_means - vine_preds) ** 2)
        print(f"  MSE: {mse:.6f}")
        
        # Plotting not implemented for 2D case in this simple example
        # Instead just print MSE and continue
        
# Save the plot
plt.tight_layout()
plt.savefig("conditional_mean_test.png")
print("\nSaved conditional mean test plot to 'conditional_mean_test.png'")

# ------------------------------------------------------------
# 5. Create combined visualization of different prediction paths
# ------------------------------------------------------------
plt.figure(figsize=(12, 8))

# Create several paths through the vine
paths = []
for i in range(1, dim):
    paths.append({
        'name': f"0→{i}",
        'fixed_var': 0,
        'predict_var': i
    })
    paths.append({
        'name': f"{i}→0",
        'fixed_var': i,
        'predict_var': 0
    })

# Add some pairs within the same level
for i in range(1, dim-1):
    paths.append({
        'name': f"{i}→{i+1}",
        'fixed_var': i,
        'predict_var': i+1
    })

# Test all paths
results = []
x = 2.0  # Test with a fixed value for conditioning
for path in paths:
    # Get true mean
    true_mean = true_gaussian_conditional_mean([path['fixed_var']], [x], path['predict_var'], rho)
    
    # Get vine prediction
    pred = vine.conditional_mean([path['fixed_var']], [x], path['predict_var'])
    
    # Calculate error
    error = abs(true_mean - pred)
    mse = (true_mean - pred) ** 2
    
    results.append({
        'path': path['name'],
        'true_mean': true_mean,
        'pred': pred,
        'error': error,
        'mse': mse
    })
    
    print(f"Path {path['name']}: True={true_mean:.4f}, Pred={pred:.4f}, Error={error:.4f}, MSE={mse:.6f}")

# Sort by error for visualization
results.sort(key=lambda x: x['error'])

# Visualize the results
plt.figure(figsize=(14, 6))

# Plot MSE by path
paths = [r['path'] for r in results]
errors = [r['error'] for r in results]
plt.bar(range(len(paths)), errors)
plt.xticks(range(len(paths)), paths, rotation=45)
plt.ylabel('Absolute Error')
plt.title('Prediction Error by Path (x=2.0)')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("prediction_paths.png")
print("Saved prediction paths plot to 'prediction_paths.png'")

plt.show() 