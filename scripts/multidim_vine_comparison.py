import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm, entropy
from pathlib import Path
import time
import os

from DVC_pyolder.config import load_config
from DVC_pyolder.objects import vine_obj_bin, margin_obj

# ------------------------------------------------------------
# Helper functions
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

def calculate_entropy(samples, bins=20):
    """Calculate approximate entropy of samples"""
    hist, _ = np.histogramdd(samples, bins=bins)
    hist = hist / np.sum(hist)  # Normalize
    hist = hist[hist > 0]  # Remove zeros
    return -np.sum(hist * np.log(hist))

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

def plot_correlation_matrix(matrix, title, filename):
    """Plot and save a correlation matrix heatmap"""
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title(title)
    
    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.2f}', 
                     ha='center', va='center', 
                     color='white' if abs(matrix[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_prediction_comparison(true_vals, pred_vals, test_grid, title, filename):
    """Plot and save prediction comparison"""
    plt.figure(figsize=(10, 6))
    plt.plot(test_grid, true_vals, 'k-', linewidth=2, label='True')
    
    for vine_name, vals in pred_vals.items():
        plt.scatter(test_grid, vals, label=vine_name, alpha=0.7)
        
        # Calculate MSE
        mse = np.mean((np.array(true_vals) - np.array(vals))**2)
        plt.annotate(f"{vine_name}: MSE = {mse:.6f}", 
                     xy=(0.05, 0.95-0.05*list(pred_vals.keys()).index(vine_name)), 
                     xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_sample_comparison(original_data, generated_samples, dim, title, filename):
    """Plot and save scatter comparison of original vs generated samples"""
    n_plots = min(dim * (dim - 1) // 2, 6)  # Show at most 6 pairs
    
    # If 2D, do a single special plot
    if dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.5, label='Original')
        plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, label='Generated')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return
    
    # For higher dimensions, determine grid layout based on n_plots
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Handle the case where we have a single row
    if n_rows == 1:
        if n_cols == 1:
            axes = np.array([axes])
        axes = axes.reshape(1, -1)
    
    # Flatten for easier indexing
    axes_flat = axes.flatten()
    
    # Select pairs to plot
    pairs = []
    for i in range(dim):
        for j in range(i+1, dim):
            pairs.append((i, j))
            if len(pairs) >= n_plots:
                break
        if len(pairs) >= n_plots:
            break
    
    for i, (dim1, dim2) in enumerate(pairs):
        if i < len(axes_flat):
            axes_flat[i].scatter(original_data[:, dim1], original_data[:, dim2], alpha=0.3, label='Original')
            axes_flat[i].scatter(generated_samples[:, dim1], generated_samples[:, dim2], alpha=0.3, label='Generated')
            axes_flat[i].set_title(f'Dims {dim1} vs {dim2}')
            axes_flat[i].grid(True)
            if i == 0:
                axes_flat[i].legend()
    
    # Hide any unused subplots
    for i in range(len(pairs), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_histogram_comparison(original_data, generated_samples, dim, title, filename):
    """Plot and save histogram comparison of original vs generated samples"""
    # For higher dimensions, use a grid layout
    n_cols = min(3, dim)
    n_rows = (dim + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Handle the case where we have a single subplot
    if dim == 1:
        axes = np.array([axes])
    
    # Handle different array shapes based on dimensions
    if n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_rows > 1 and n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten for easier indexing
    axes_flat = axes.flatten()
        
    for i in range(dim):
        if i < len(axes_flat):
            axes_flat[i].hist(original_data[:, i], bins=30, alpha=0.5, label='Original')
            axes_flat[i].hist(generated_samples[:, i], bins=30, alpha=0.5, label='Generated')
            axes_flat[i].set_title(f'Dimension {i}')
            axes_flat[i].grid(True)
            if i == 0:
                axes_flat[i].legend()
    
    # Hide any unused subplots
    for i in range(dim, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_mse_comparison(mse_results, title, filename):
    """Plot and save MSE comparison across dimensions"""
    plt.figure(figsize=(12, 6))
    
    # Extract data
    dims = sorted(mse_results.keys())
    vine_types = list(mse_results[dims[0]].keys())
    
    # Plot grouped bars
    bar_width = 0.8 / len(vine_types)
    index = np.arange(len(dims))
    
    for i, vine_type in enumerate(vine_types):
        mse_vals = [mse_results[dim][vine_type] for dim in dims]
        offset = i * bar_width - 0.4 + bar_width/2
        plt.bar(index + offset, mse_vals, bar_width, label=vine_type)
    
    plt.xlabel('Dimensions')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.title(title)
    plt.xticks(index, dims)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ------------------------------------------------------------
# Main experiment function
# ------------------------------------------------------------
def run_experiment(dim, n_train=5000, n_samples=2000, rho=0.6, seed=42, results_dir='results'):
    """Run a complete experiment for a specific dimension"""
    print(f"\n{'='*80}")
    print(f"Running experiment for dimension: {dim}")
    print(f"{'='*80}")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    dim_dir = os.path.join(results_dir, f"{dim}D")
    os.makedirs(dim_dir, exist_ok=True)
    
    # 1. Generate data
    data, cov_true = generate_gaussian_data(n_train, dim, rho, seed)
    print(f"Generated {n_train} samples from {dim}D Gaussian with rho={rho}")
    
    # Calculate and display true correlation matrix
    true_corr = np.corrcoef(data, rowvar=False)
    print("\nTrue correlation matrix:")
    print(np.round(true_corr, 3))
    plot_correlation_matrix(true_corr, f"True Correlation Matrix - {dim}D", 
                           os.path.join(dim_dir, f"true_corr_{dim}d.png"))
    
    # 2. Fit margins
    margins = []
    for i in range(dim):
        loc, scale = norm.fit(data[:, i])
        margin = margin_obj('norm', [loc, scale], True)
        margins.append(margin)
        print(f"Margin {i}: Normal(μ={loc:.4f}, σ={scale:.4f})")
    
    # 3. Create and fit different vine structures
    base_cfg = {
        'vine': {'knots': 40},
        'general': {'binning': False, 'fitted': False},
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
        'bandwidth': {'method': 'rule_of_thumb', 'knn_k': 10},
        'npc': {'opt_method': 'LL1', 'grad_precompute': True},
        'sampler': {'fast_parametric': True, 'fast_nonparam': True, 'nspline': 200}
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
        }
    ]
    
    # Skip non-parametric vines for now due to grid size issues
    # Non-parametric vines require additional configuration and debugging
    
    # Fit vines
    vines = []
    vine_fit_times = {}
    
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
        
        # Fit the vine with timing
        start_time = time.time()
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
        fit_time = time.time() - start_time
        vine_fit_times[config['name']] = fit_time
        
        # Store the vine with its config
        vines.append((vine, config))
        print(f"  Fit time: {fit_time:.2f} seconds")
        print(f"  Vine structure: {vine.ind_vine}")
    
    # 4. Generate samples and analyze
    sample_results = {}
    sample_times = {}
    
    for vine, config in vines:
        vine_name = config['name']
        print(f"Generating samples for {vine_name}...")
        
        # Generate samples with timing
        start_time = time.time()
        samples = vine.sample(n_samples)
        sample_time = time.time() - start_time
        sample_times[vine_name] = sample_time
        print(f"  Sample time: {sample_time:.2f} seconds")
        
        # Calculate correlation matrix of samples
        sample_corr = np.corrcoef(samples, rowvar=False)
        corr_error = np.mean(np.abs(sample_corr - true_corr))
        print(f"  Correlation matrix error: {corr_error:.4f}")
        
        # Calculate approximate entropy
        try:
            sample_entropy = calculate_entropy(samples)
            print(f"  Sample entropy: {sample_entropy:.4f}")
        except Exception as e:
            print(f"  Error calculating entropy: {str(e)}")
            sample_entropy = None
        
        # Store results
        sample_results[vine_name] = {
            'samples': samples,
            'corr_matrix': sample_corr, 
            'corr_error': corr_error,
            'entropy': sample_entropy
        }
        
        # Plot correlation matrix
        plot_correlation_matrix(
            sample_corr, 
            f"Sample Correlation Matrix - {vine_name}", 
            os.path.join(dim_dir, f"sample_corr_{vine_name.replace(' ', '_').lower()}_{dim}d.png")
        )
        
        # Plot sample comparison
        plot_sample_comparison(
            data[:n_samples], samples, dim,
            f"Sample Comparison - {vine_name}",
            os.path.join(dim_dir, f"scatter_{vine_name.replace(' ', '_').lower()}_{dim}d.png")
        )
        
        # Plot histogram comparison
        plot_histogram_comparison(
            data[:n_samples], samples, dim,
            f"Margin Comparison - {vine_name}",
            os.path.join(dim_dir, f"hist_{vine_name.replace(' ', '_').lower()}_{dim}d.png")
        )
    
    # 5. Test conditional prediction
    pred_results = {}
    
    # Define a set of test paths for conditional prediction
    test_paths = []
    
    # Add simple paths (1 conditioning variable)
    for i in range(min(3, dim)):
        for j in range(min(3, dim)):
            if i != j:
                test_paths.append({
                    'fixed_vars': [i], 
                    'predict_var': j, 
                    'name': f'{i}→{j}'
                })
    
    # Add one multiple conditioning path if dimension allows
    if dim >= 3:
        test_paths.append({
            'fixed_vars': [0, 1], 
            'predict_var': 2, 
            'name': '[0,1]→2'
        })
    
    # Test predictions
    prediction_mse = {}
    prediction_times = {}
    
    for path in test_paths:
        path_name = path['name']
        fixed_vars = path['fixed_vars']
        predict_var = path['predict_var']
        
        pred_results[path_name] = {}
        
        # For single conditioning variable, test across a grid
        if len(fixed_vars) == 1:
            fixed_var = fixed_vars[0]
            test_points = 30
            test_grid = np.linspace(-3, 3, test_points)
            
            true_values = []
            all_pred_values = {}
            
            # Calculate true conditional means
            for x in test_grid:
                true_mean = true_gaussian_conditional_mean([fixed_var], [x], predict_var, rho)
                true_values.append(true_mean)
            
            # Test each vine's prediction
            for vine, config in vines:
                vine_name = config['name']
                mse = 0.0
                pred_times = []
                pred_values = []
                
                for x in test_grid:
                    # Vine prediction with timing
                    start_time = time.time()
                    pred = vine.conditional_mean([fixed_var], [x], predict_var)
                    pred_time = time.time() - start_time
                    pred_times.append(pred_time)
                    pred_values.append(pred)
                    
                    # Error for this point
                    if pred is not None:
                        mse += (true_gaussian_conditional_mean([fixed_var], [x], predict_var, rho) - pred) ** 2
                
                # Average MSE and time
                mse /= test_points
                avg_time = np.mean(pred_times)
                
                # Store results
                all_pred_values[vine_name] = pred_values
                prediction_mse.setdefault(vine_name, {})[path_name] = mse
                prediction_times.setdefault(vine_name, {})[path_name] = avg_time
                
                print(f"  {vine_name} - {path_name}: MSE = {mse:.6f}, Time = {avg_time:.6f}s")
            
            # Plot prediction comparison
            plot_prediction_comparison(
                true_values, all_pred_values, test_grid,
                f"Conditional Prediction {path_name} - {dim}D",
                os.path.join(dim_dir, f"pred_{path_name.replace('[', '').replace(']', '').replace(',', '_')}_{dim}d.png")
            )
        
        # For multiple conditioning variables, use fixed test values
        else:
            fixed_values = [1.0] * len(fixed_vars)
            true_mean = true_gaussian_conditional_mean(fixed_vars, fixed_values, predict_var, rho)
            
            for vine, config in vines:
                vine_name = config['name']
                
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
                prediction_mse.setdefault(vine_name, {})[path_name] = error
                prediction_times.setdefault(vine_name, {})[path_name] = pred_time
                
                print(f"  {vine_name} - {path_name}: Error = {error:.6f}, Time = {pred_time:.6f}s")
    
    # Calculate average MSE for each vine type
    avg_mse = {}
    for vine_name in prediction_mse:
        mse_values = [mse for mse in prediction_mse[vine_name].values() if not np.isnan(mse)]
        avg_mse[vine_name] = np.mean(mse_values) if mse_values else float('nan')
        print(f"Average MSE for {vine_name}: {avg_mse[vine_name]:.6f}")
    
    # 6. Summarize results
    summary = {
        'dimension': dim,
        'fit_times': vine_fit_times,
        'sample_times': sample_times,
        'correlation_errors': {name: res['corr_error'] for name, res in sample_results.items()},
        'entropies': {name: res['entropy'] for name, res in sample_results.items()},
        'avg_prediction_mse': avg_mse,
    }
    
    return summary, prediction_mse

# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------
if __name__ == "__main__":
    # Define parameters
    dimensions = [2, 3, 4]
    n_train = 5000
    n_samples = 2000
    rho = 0.6
    results_dir = "multidim_results"
    
    # Run experiments for each dimension
    summaries = {}
    all_mse_results = {}
    
    for dim in dimensions:
        summary, mse_results = run_experiment(dim, n_train, n_samples, rho, 42, results_dir)
        summaries[dim] = summary
        all_mse_results[dim] = {vine: np.mean(list(paths.values())) for vine, paths in mse_results.items()}
    
    # Plot MSE comparison across dimensions
    plot_mse_comparison(
        all_mse_results, 
        "Prediction MSE Comparison Across Dimensions",
        os.path.join(results_dir, "mse_comparison.png")
    )
    
    # Print final summary
    print("\n\nFinal Summary:")
    print("=" * 80)
    print(f"{'Dimension':<10} | {'Vine Type':<25} | {'Fit Time (s)':<12} | {'Sample Time (s)':<15} | {'Corr Error':<10} | {'Pred MSE':<10}")
    print("-" * 80)
    
    for dim in dimensions:
        s = summaries[dim]
        for vine_name in s['fit_times'].keys():
            fit_time = s['fit_times'][vine_name]
            sample_time = s['sample_times'][vine_name]
            corr_error = s['correlation_errors'][vine_name]
            pred_mse = s['avg_prediction_mse'][vine_name]
            
            print(f"{dim:<10} | {vine_name:<25} | {fit_time:<12.2f} | {sample_time:<15.2f} | {corr_error:<10.4f} | {pred_mse:<10.6f}")
    
    print("=" * 80)
    print("Experiment completed successfully!") 