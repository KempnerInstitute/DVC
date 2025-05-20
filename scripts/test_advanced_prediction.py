import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from pathlib import Path
import time

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
# 3. Create and fit different vine structures (parametric and non-parametric)
# ------------------------------------------------------------
print("\nFitting different vine structures...")

# Base configuration
base_cfg = {
    'vine': {
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

# Define different vine types to test
vine_configs = [
    {
        'name': 'C-Vine (Parametric)',
        'family': 'c-vine',
        'method': 'optimal',
        'param': True
    },
    {
        'name': 'D-Vine (Parametric)',
        'family': 'd-vine',
        'method': 'optimal',
        'param': True
    },
    # Skip non-parametric for now as it needs more debugging
    # {
    #     'name': 'C-Vine (Non-parametric)',
    #     'family': 'c-vine',
    #     'method': 'optimal',
    #     'param': False
    # }
]

# Create and fit each vine
vines = []
for config in vine_configs:
    print(f"Fitting {config['name']}...")
    
    # Update configuration
    cfg = base_cfg.copy()
    cfg['vine'] = base_cfg['vine'].copy()
    cfg['vine']['family'] = config['family']
    cfg['vine']['method'] = config['method']
    cfg['general'] = base_cfg['general'].copy()
    cfg['general']['param'] = config['param']
    
    # Dictionaries for vine.fit()
    gen_dict = {'param': config['param'], 'binning': False, 'fitted': False}
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    
    # Create vine
    vine = vine_obj_bin(
        config['family'],
        ['gaussian'],
        dim,
        margins,
        knots=40,
        method=config['method']
    )
    
    # Fit the vine
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
    
    # Store the vine with its config
    vines.append((vine, config))
    print(f"  Vine structure: {vine.ind_vine}")

# Helper function for true conditional mean
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

# ------------------------------------------------------------
# 4. Test different prediction paths and compute MSE
# ------------------------------------------------------------
print("\nTesting prediction accuracy for different paths...")

# Define a variety of prediction paths to test
test_paths = [
    # Simple paths
    {'fixed_vars': [0], 'predict_var': 1, 'name': 'Root → Child: 0→1'},
    {'fixed_vars': [1], 'predict_var': 0, 'name': 'Child → Root: 1→0'},
    {'fixed_vars': [0], 'predict_var': 4, 'name': 'Root → Distant: 0→4'},
    
    # Same level
    {'fixed_vars': [1], 'predict_var': 2, 'name': 'Same Level: 1→2'},
    {'fixed_vars': [2], 'predict_var': 3, 'name': 'Same Level: 2→3'},
    
    # Multiple conditioning
    {'fixed_vars': [0, 1], 'predict_var': 2, 'name': 'Multiple: [0,1]→2'},
    {'fixed_vars': [0, 1], 'predict_var': 4, 'name': 'Multiple: [0,1]→4'},
    {'fixed_vars': [1, 2], 'predict_var': 3, 'name': 'Multiple: [1,2]→3'},
    
    # Complex paths
    {'fixed_vars': [1, 3, 4], 'predict_var': 0, 'name': 'Complex: [1,3,4]→0'},
    {'fixed_vars': [0, 2, 4], 'predict_var': 3, 'name': 'Complex: [0,2,4]→3'}
]

# Prepare test data
test_points = 30
test_grid = np.linspace(-3, 3, test_points)

# Store results
results = {}
for vine, config in vines:
    vine_name = config['name']
    results[vine_name] = {}
    
    # Test each path
    for path in test_paths:
        path_name = path['name']
        fixed_vars = path['fixed_vars']
        predict_var = path['predict_var']
        
        # For single conditioning variable, test over a grid
        if len(fixed_vars) == 1:
            fixed_var = fixed_vars[0]
            
            # Track MSE, prediction times, and values
            mse = 0.0
            pred_times = []
            true_values = []
            pred_values = []
            
            # Test each point in the grid
            for x in test_grid:
                # True conditional mean
                true_mean = true_gaussian_conditional_mean([fixed_var], [x], predict_var, rho)
                true_values.append(true_mean)
                
                # Vine prediction with timing
                start_time = time.time()
                pred = vine.conditional_mean([fixed_var], [x], predict_var)
                pred_time = time.time() - start_time
                pred_times.append(pred_time)
                pred_values.append(pred)
                
                # Error for this point
                if pred is not None:
                    mse += (true_mean - pred) ** 2
            
            # Average MSE
            mse /= test_points
            
            # Store results
            results[vine_name][path_name] = {
                'mse': mse,
                'avg_time': np.mean(pred_times),
                'true_values': true_values,
                'pred_values': pred_values,
                'test_grid': test_grid
            }
            
            print(f"  {vine_name} - {path_name}: MSE = {mse:.6f}, Time = {np.mean(pred_times):.6f}s")
        
        # For multiple conditioning variables, use a simpler approach
        else:
            # Use a fixed test value
            fixed_values = [2.0] * len(fixed_vars)
            
            # True conditional mean
            true_mean = true_gaussian_conditional_mean(fixed_vars, fixed_values, predict_var, rho)
            
            # Vine prediction with timing
            start_time = time.time()
            pred = vine.conditional_mean(fixed_vars, fixed_values, predict_var)
            pred_time = time.time() - start_time
            
            # Calculate squared error
            if pred is not None:
                error = (true_mean - pred) ** 2
            else:
                error = float('nan')
            
            # Store result
            results[vine_name][path_name] = {
                'mse': error,
                'avg_time': pred_time,
                'true_value': true_mean,
                'pred_value': pred
            }
            
            print(f"  {vine_name} - {path_name}: Error = {error:.6f}, Time = {pred_time:.6f}s")

# ------------------------------------------------------------
# 5. Create visualization comparing vine structures
# ------------------------------------------------------------

# Collect MSE values and organize them by path
path_results = {}
for path in test_paths:
    path_name = path['name']
    path_results[path_name] = {}
    
    for vine_name in results.keys():
        if path_name in results[vine_name]:
            if 'mse' in results[vine_name][path_name]:
                path_results[path_name][vine_name] = results[vine_name][path_name]['mse']

# Plot MSE by path and vine type
plt.figure(figsize=(14, 8))

# Organize data for bar chart
paths = list(path_results.keys())
vine_names = list(results.keys())
bar_width = 0.35
index = np.arange(len(paths))

for i, vine_name in enumerate(vine_names):
    mse_values = [path_results[path].get(vine_name, float('nan')) for path in paths]
    
    # Plot bar for this vine type, slightly offset
    offset = (i - len(vine_names)/2 + 0.5) * bar_width
    plt.bar(index + offset, mse_values, bar_width, label=vine_name)

plt.xlabel('Prediction Path')
plt.ylabel('Mean Squared Error (log scale)')
plt.title('Prediction Accuracy by Path and Vine Structure')
plt.xticks(index, paths, rotation=45, ha='right')
plt.yscale('log')  # Use log scale to better see differences
plt.legend()
plt.tight_layout()
plt.savefig("advanced_prediction_mse.png")
print("\nSaved MSE comparison to 'advanced_prediction_mse.png'")

# ------------------------------------------------------------
# 6. Visualize prediction accuracy for single-variable paths
# ------------------------------------------------------------
plt.figure(figsize=(15, 10))

# Plot results for 4 representative paths
single_var_paths = [p for p in test_paths if len(p['fixed_vars']) == 1]
n_paths = min(4, len(single_var_paths))

for i, path in enumerate(single_var_paths[:n_paths]):
    path_name = path['name']
    
    plt.subplot(2, 2, i+1)
    
    # Plot true conditional mean
    for vine_name in vine_names:
        if path_name in results[vine_name]:
            res = results[vine_name][path_name]
            if 'true_values' in res and 'pred_values' in res:
                plt.plot(res['test_grid'], res['true_values'], 'k-', linewidth=2, label='True' if i==0 else None)
                plt.scatter(res['test_grid'], res['pred_values'], label=vine_name, alpha=0.7)
                
                # Add MSE annotation
                plt.annotate(f"MSE = {res['mse']:.6f}", xy=(0.05, 0.95-0.05*vine_names.index(vine_name)), 
                             xycoords='axes fraction', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.title(path_name)
    plt.grid(True)
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig("advanced_prediction_paths.png")
print("Saved path visualizations to 'advanced_prediction_paths.png'")

# ------------------------------------------------------------
# 7. Summarize results in a table
# ------------------------------------------------------------
print("\nSummary of prediction MSE by path and vine type:")
print("-" * 80)
print(f"{'Path':<25} | ", end="")
for vine_name in vine_names:
    print(f"{vine_name:<25} | ", end="")
print()
print("-" * 80)

for path in test_paths:
    path_name = path['name']
    print(f"{path_name:<25} | ", end="")
    
    for vine_name in vine_names:
        if path_name in results[vine_name] and 'mse' in results[vine_name][path_name]:
            mse = results[vine_name][path_name]['mse']
            print(f"{mse:<25.6f} | ", end="")
        else:
            print(f"{'N/A':<25} | ", end="")
    print()

# Also print average prediction time
print("\nAverage prediction time (seconds):")
print("-" * 80)
for vine_name in vine_names:
    times = []
    for path_name, path_result in results[vine_name].items():
        if 'avg_time' in path_result:
            times.append(path_result['avg_time'])
    
    avg_time = np.mean(times) if times else float('nan')
    print(f"{vine_name}: {avg_time:.6f}s")

plt.show() 