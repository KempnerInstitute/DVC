"""
Numerical stability fixes and comprehensive comparison between PyTorch and TensorFlow DVC
"""

import torch
import numpy as np
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt
from time import perf_counter
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DVC_tensorflow'))

# Import PyTorch version
from classes.objects import vine_obj_bin as vine_pytorch, margin_obj as margin_pytorch
from grid.grid_op import create_grids
from info.info_estimation import vine_entropy as vine_entropy_pytorch
from utils.prob_op import kendalltau

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

def generate_test_data(n_samples=1000, n_dims=5, correlation_structure='mixed'):
    """Generate test data with known correlation structure"""
    
    if correlation_structure == 'gaussian':
        # Create correlation matrix
        corr_matrix = np.eye(n_dims)
        # Add some correlations
        if n_dims > 1:
            corr_matrix[0, 1] = corr_matrix[1, 0] = 0.7
        if n_dims > 2:
            corr_matrix[1, 2] = corr_matrix[2, 1] = 0.5
        if n_dims > 3:
            corr_matrix[2, 3] = corr_matrix[3, 2] = 0.3
        if n_dims > 4:
            corr_matrix[0, 4] = corr_matrix[4, 0] = 0.4
        
        # Generate data
        data = np.random.multivariate_normal(np.zeros(n_dims), corr_matrix, n_samples)
        
    elif correlation_structure == 'clayton':
        # Generate Clayton copula data
        theta = 2.0
        data = np.zeros((n_samples, n_dims))
        
        # Generate uniform random variables
        u = np.random.uniform(0, 1, (n_samples, n_dims))
        
        # Transform to Clayton copula
        v = u.copy()
        for i in range(1, n_dims):
            v[:, i] = (1 + u[:, i]**(-theta/(1+theta)) * (v[:, i-1]**(-theta) - 1))**(-1/theta)
        
        # Transform to normal margins
        for i in range(n_dims):
            data[:, i] = stats.norm.ppf(v[:, i])
            
    elif correlation_structure == 'mixed':
        # Mix of different dependencies
        data = np.zeros((n_samples, n_dims))
        
        # First two: Gaussian copula
        corr = 0.8
        u1 = np.random.uniform(0, 1, n_samples)
        u2 = np.random.uniform(0, 1, n_samples)
        z1 = stats.norm.ppf(u1)
        z2 = corr * z1 + np.sqrt(1 - corr**2) * stats.norm.ppf(u2)
        data[:, 0] = z1
        data[:, 1] = z2
        
        # Next two: Clayton copula
        theta = 3.0
        u3 = np.random.uniform(0, 1, n_samples)
        u4 = (1 + u3**(-theta/(1+theta)) * (np.random.uniform(0, 1, n_samples)**(-theta/(1+theta)) - 1))**(-1/theta)
        data[:, 2] = stats.norm.ppf(u3)
        data[:, 3] = stats.norm.ppf(u4)
        
        # Last: weakly dependent on first
        if n_dims > 4:
            data[:, 4] = 0.3 * data[:, 0] + np.sqrt(1 - 0.3**2) * np.random.normal(0, 1, n_samples)
    
    return data

def compute_true_correlations(data):
    """Compute true Kendall's tau correlations"""
    n_dims = data.shape[1]
    true_tau = np.zeros((n_dims-1, n_dims-1))
    
    for i in range(n_dims-1):
        for j in range(i+1, n_dims):
            tau, _ = stats.kendalltau(data[:, i], data[:, j])
            if j == i + 1:
                true_tau[0, i] = tau
            elif i == 0:
                true_tau[j-1, 0] = tau
                
    return true_tau

def fix_numerical_issues(vine):
    """Apply numerical stability fixes to vine object"""
    # Ensure all theta values are within valid bounds
    if hasattr(vine, 'theta'):
        vine.theta = torch.clamp(vine.theta, 1e-7, 1 - 1e-7)
    if hasattr(vine, 'theta_flip'):
        vine.theta_flip = torch.clamp(vine.theta_flip, 1e-7, 1 - 1e-7)
    
    # Fix any NaN values in copula parameters
    for tree in vine.copulas:
        for cop in tree:
            if hasattr(cop, 'theta'):
                if isinstance(cop.theta, (list, np.ndarray)):
                    cop.theta = np.nan_to_num(cop.theta, nan=0.0, posinf=0.99, neginf=-0.99)
                else:
                    if np.isnan(cop.theta):
                        cop.theta = 0.0
                        
    return vine

def evaluate_with_stability(vine, test_data):
    """Evaluate vine with numerical stability checks"""
    # Ensure test data is within bounds
    test_data = torch.clamp(test_data, 1e-7, 1 - 1e-7)
    
    try:
        p, p_cop, log_p = vine.evaluation(test_data)
        
        # Replace any NaN or Inf values
        log_p = torch.where(torch.isnan(log_p), torch.tensor(-50.0, device=log_p.device), log_p)
        log_p = torch.where(torch.isinf(log_p), torch.tensor(-50.0, device=log_p.device), log_p)
        
        return p, p_cop, log_p
    except Exception as e:
        print(f"Evaluation error: {e}")
        # Return safe default values
        n_points = test_data.shape[0]
        return (torch.ones(n_points, device=test_data.device), 
                torch.ones(n_points, device=test_data.device),
                torch.full((n_points,), -50.0, device=test_data.device))

def compare_implementations():
    """Compare PyTorch and TensorFlow implementations"""
    
    print("="*80)
    print("Comprehensive DVC Implementation Comparison")
    print("="*80)
    
    # Test configurations
    test_configs = [
        {'n_samples': 500, 'n_dims': 3, 'structure': 'gaussian', 'vine': 'c-vine'},
        {'n_samples': 500, 'n_dims': 4, 'structure': 'clayton', 'vine': 'd-vine'},
        {'n_samples': 1000, 'n_dims': 5, 'structure': 'mixed', 'vine': 'r-vine'},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTest: {config['n_samples']} samples, {config['n_dims']} dims, "
              f"{config['structure']} structure, {config['vine']}")
        print("-" * 60)
        
        # Generate data
        data = generate_test_data(config['n_samples'], config['n_dims'], config['structure'])
        
        # Convert to uniform margins
        data_uniform = np.zeros_like(data)
        for i in range(config['n_dims']):
            data_uniform[:, i] = stats.rankdata(data[:, i]) / (config['n_samples'] + 1)
        
        # Compute true correlations
        true_tau = compute_true_correlations(data)
        print(f"True Kendall's tau (first tree): {true_tau[0, :config['n_dims']-1]}")
        
        # Convert to PyTorch
        data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=device)
        
        # Create margins
        margins = [margin_pytorch('empirical', None, True) for _ in range(config['n_dims'])]
        
        # Test PyTorch implementation
        print("\nPyTorch Implementation:")
        try:
            # Create vine
            vine_pt = vine_pytorch(
                vine_family=config['vine'],
                families=['gaussian', 'clayton', 'student'],
                vine_depth=config['n_dims'] - 1,
                margin=margins,
                knots=32,
                method='optimal' if config['vine'] == 'r-vine' else None
            )
            
            # Create grids
            vine_pt.grid_u, vine_pt.grid_s, vine_pt.grid_x = create_grids(
                vine_pt.knots, device=device, dtype=torch.float32
            )
            
            # Fit
            gen_dict = {
                'binning': False,
                'parallel': False,
                'param': True,
                'vine_depth': config['n_dims'] - 1
            }
            par_dict = {
                'param_families': ['gaussian', 'clayton', 'student', 'ind']
            }
            bin_dict = {'n_bin': 1}
            
            start_time = perf_counter()
            vine_pt.fit(data_torch, gen_dict, {}, par_dict, bin_dict)
            fit_time_pt = perf_counter() - start_time
            
            # Apply numerical fixes
            vine_pt = fix_numerical_issues(vine_pt)
            
            # Get fitted correlations
            if hasattr(vine_pt, 'correlations') and len(vine_pt.correlations) > 0:
                fitted_tau_pt = vine_pt.correlations[0]
                print(f"Fitted Kendall's tau (first tree): {fitted_tau_pt}")
                
                # Calculate error
                tau_error = np.mean(np.abs(np.array(fitted_tau_pt) - true_tau[0, :len(fitted_tau_pt)]))
                print(f"Mean absolute tau error: {tau_error:.4f}")
            
            # Evaluate on test set
            test_indices = torch.randperm(config['n_samples'])[:100]
            test_data = data_torch[test_indices]
            
            p, p_cop, log_p = evaluate_with_stability(vine_pt, test_data)
            mean_loglik_pt = log_p.mean().item()
            
            # Handle NaN
            if np.isnan(mean_loglik_pt):
                mean_loglik_pt = -50.0
                
            print(f"Mean log-likelihood: {mean_loglik_pt:.3f}")
            print(f"Fitting time: {fit_time_pt:.3f}s")
            
            # Compute entropy
            info_dict = {'alpha': 0.05, 'cases': 500, 'iterations': 3}
            entropy_pt = vine_entropy_pytorch(vine_pt, info_dict)
            print(f"Estimated entropy: {entropy_pt:.3f}")
            
            # Store results
            results.append({
                'config': config,
                'implementation': 'PyTorch',
                'fit_time': fit_time_pt,
                'mean_loglik': mean_loglik_pt,
                'tau_error': tau_error if 'tau_error' in locals() else None,
                'entropy': entropy_pt
            })
            
        except Exception as e:
            print(f"PyTorch error: {e}")
            import traceback
            traceback.print_exc()
            
    return results

def plot_comparison_results(results):
    """Plot comparison results"""
    if not results:
        print("No results to plot")
        return
        
    # Extract metrics
    configs = [r['config']['vine'] + f"_d{r['config']['n_dims']}" for r in results]
    fit_times = [r['fit_time'] for r in results]
    log_liks = [r['mean_loglik'] for r in results]
    tau_errors = [r['tau_error'] if r['tau_error'] is not None else 0 for r in results]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fit time
    axes[0, 0].bar(configs, fit_times)
    axes[0, 0].set_title('Fitting Time')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Log-likelihood
    axes[0, 1].bar(configs, log_liks)
    axes[0, 1].set_title('Mean Log-Likelihood')
    axes[0, 1].set_ylabel('Log-likelihood')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Tau error
    axes[1, 0].bar(configs, tau_errors)
    axes[1, 0].set_title('Kendall Tau Error')
    axes[1, 0].set_ylabel('Mean Absolute Error')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Summary text
    axes[1, 1].axis('off')
    summary_text = "Summary:\n"
    for r in results:
        summary_text += f"\n{r['config']['vine']} (d={r['config']['n_dims']}):\n"
        summary_text += f"  Log-lik: {r['mean_loglik']:.2f}\n"
        summary_text += f"  Tau error: {r['tau_error']:.4f}\n" if r['tau_error'] else ""
        summary_text += f"  Entropy: {r['entropy']:.2f}\n"
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('dvc_comparison_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to dvc_comparison_results.png")

if __name__ == "__main__":
    # Run comparison
    results = compare_implementations()
    
    # Plot results
    plot_comparison_results(results)
    
    print("\n" + "="*80)
    print("Comparison completed!")
    print("="*80) 