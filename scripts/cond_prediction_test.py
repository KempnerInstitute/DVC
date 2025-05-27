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
    {'family': 'c-vine', 'method': 'optimal', 'name': 'C-Vine (Optimal)'},
    {'family': 'd-vine', 'method': 'optimal', 'name': 'D-Vine (Optimal)'},
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
# 4. Conditional prediction test
# ------------------------------------------------------------
print("\nRunning conditional prediction tests...")

# Define several test cases where we fix some variables and predict others
test_cases = [
    # Fix the first variable, predict the last
    {'fixed_vars': [0], 'predict_var': dim-1, 'name': 'Predict last given first'},
    
    # Fix the middle variable, predict the first
    {'fixed_vars': [dim//2], 'predict_var': 0, 'name': 'Predict first given middle'},
    
    # Fix the first two variables, predict the last
    {'fixed_vars': [0, 1], 'predict_var': dim-1, 'name': f'Predict {dim-1} given [0,1]'},
    
    # Fix all but one variable, predict the middle
    {'fixed_vars': list(range(dim)), 'predict_var': dim//2, 'name': f'Predict {dim//2} given rest'},
]

# Remove the prediction variable from fixed_vars in the last test case
test_cases[-1]['fixed_vars'].remove(test_cases[-1]['predict_var'])

# Create test grid for fixed variables
n_test_points = 20
test_grid = np.linspace(-2, 2, n_test_points)

# True distribution for reference
mean_true = np.zeros(dim)

# For each test case
for test_case in test_cases:
    print(f"\nRunning test: {test_case['name']}")
    fixed_vars = test_case['fixed_vars']
    predict_var = test_case['predict_var']
    
    # For fixed variables with 1 or 2 dimensions, we'll create a grid
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
            
            # Create conditional mean vector for true distribution
            cond_indices = fixed_vars
            predict_indices = [predict_var]
            
            # For true Gaussian, conditional mean is linear function of fixed values
            # For uniform correlation, all fixed variables equally influence prediction
            # For a uniform correlation matrix, the conditional mean is:
            # μ_predict|fixed = ρ * sum(fixed_values) * len(fixed_values)
            fixed_sum = sum(test_point)
            true_mean = rho * fixed_sum
            results['true_means'].append(true_mean)
            
            # Test each vine
            for vine, name in vines:
                # Create a test data point with fixed values
                test_data = np.zeros(dim)
                for i, var_idx in enumerate(fixed_vars):
                    test_data[var_idx] = test_point[i]
                
                # Search for best prediction using maximum likelihood
                search_range = np.linspace(-3, 3, 100)
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
                    if logp > best_logp:
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
            
            plt.xlabel(f"Variable {fixed_vars[0]} Value")
            plt.ylabel(f"Predicted Variable {predict_var} Value")
            plt.title(f"Conditional Prediction: {test_case['name']}")
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            plt.savefig(f"cond_pred_{fixed_vars[0]}_to_{predict_var}.png")
        
        # For 2D fixed variables
        elif len(fixed_vars) == 2:
            # Plot separate figures for each vine
            for vine, name in vines:
                plt.figure(figsize=(15, 5))
                
                # Get predictions and reshape to 2D grid
                vine_preds = np.array(results[name])
                
                # 3D plot
                ax = plt.subplot(121, projection='3d')
                xx = fixed_values[:, 0].reshape(n_test_points, n_test_points)
                yy = fixed_values[:, 1].reshape(n_test_points, n_test_points)
                zz = vine_preds.reshape(n_test_points, n_test_points)
                ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.8)
                
                # Also plot true means
                zz_true = true_means.reshape(n_test_points, n_test_points)
                ax.plot_surface(xx, yy, zz_true, color='r', alpha=0.3)
                
                ax.set_xlabel(f"Variable {fixed_vars[0]}")
                ax.set_ylabel(f"Variable {fixed_vars[1]}")
                ax.set_zlabel(f"Predicted {predict_var}")
                ax.set_title(f"{name} Predictions")
                
                # 2D heatmap of differences
                ax = plt.subplot(122)
                diffs = np.abs(vine_preds - true_means).reshape(n_test_points, n_test_points)
                im = ax.imshow(diffs, cmap='hot', origin='lower', 
                               extent=[test_grid.min(), test_grid.max(), 
                                       test_grid.min(), test_grid.max()])
                plt.colorbar(im, ax=ax, label="Absolute Error")
                ax.set_xlabel(f"Variable {fixed_vars[0]}")
                ax.set_ylabel(f"Variable {fixed_vars[1]}")
                ax.set_title(f"Prediction Error - MSE: {np.mean(diffs**2):.6f}")
                
                plt.tight_layout()
                plt.savefig(f"cond_pred_{name.replace(' ', '_')}_{fixed_vars[0]}_{fixed_vars[1]}_to_{predict_var}.png")
                
            # Calculate and print MSEs
            for vine, name in vines:
                vine_preds = np.array(results[name])
                mse = np.mean((vine_preds - true_means) ** 2)
                print(f"  {name} MSE: {mse:.6f}")

# Plot prediction errors by vine and by prediction path
plt.figure(figsize=(12, 10))
all_mses = {}

# Define the function to compute conditional prediction mean
def compute_conditional_mean(vine, fixed_vars, fixed_values, predict_var):
    # Create a test data point with fixed values
    test_data = np.zeros(dim)
    for i, var_idx in enumerate(fixed_vars):
        test_data[var_idx] = fixed_values[i]
    
    # Search for best prediction using maximum likelihood
    search_range = np.linspace(-3, 3, 100)
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
        if logp > best_logp:
            best_logp = logp
            best_val = val
            
    return best_val

# More thorough test: measure prediction error for different paths
n_paths = 10
test_points = np.random.randn(n_paths, dim-1)  # Random test points

# List of paths to test (which variables to fix, which to predict)
paths = []
for i in range(dim):
    # Predict variable i, fix all others
    fixed = list(range(dim))
    fixed.remove(i)
    paths.append({
        'fixed': fixed, 
        'predict': i, 
        'name': f"Predict {i}"
    })

# Get results for each path and each vine
results = {}
for vine, name in vines:
    results[name] = []
    
    # For each path
    for path in paths:
        fixed_vars = path['fixed']
        predict_var = path['predict']
        path_mses = []
        
        # For each test point
        for point in test_points:
            # Compute true conditional mean (for Gaussian)
            true_mean = rho * np.sum(point)
            
            # Compute vine prediction
            pred = compute_conditional_mean(vine, fixed_vars, point, predict_var)
            
            # Compute squared error
            sq_error = (pred - true_mean) ** 2
            path_mses.append(sq_error)
        
        # Store average MSE for this path
        path_mse = np.mean(path_mses)
        results[name].append(path_mse)
        print(f"{name} - {path['name']} MSE: {path_mse:.6f}")

# Plot results by dimension
labels = [path['name'] for path in paths]
x = np.arange(len(labels))
width = 0.35 / len(vines)
offsets = np.linspace(-0.15, 0.15, len(vines))

for i, (vine, name) in enumerate(vines):
    plt.bar(x + offsets[i], results[name], width, label=name)

plt.xlabel('Prediction Path')
plt.ylabel('Mean Squared Error')
plt.title('Prediction Error by Path and Vine Structure')
plt.xticks(x, labels, rotation=45)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('prediction_by_dimension.png')

# Show the plots if in interactive mode
plt.show() 