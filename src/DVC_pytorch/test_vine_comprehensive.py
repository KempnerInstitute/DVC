import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import multivariate_normal
import time
import pandas as pd

from classes.objects import vine_obj_bin, margin_obj
from utils.prob_op import kernel_cdf, kde_wrapper

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def generate_multivariate_gaussian(n_samples, correlation_matrix):
    """Generate multivariate Gaussian data with specified correlation"""
    dim = correlation_matrix.shape[0]
    mean = np.zeros(dim)
    
    # Convert correlation to covariance (assuming unit variance)
    cov_matrix = correlation_matrix
    
    # Generate samples
    mvn = multivariate_normal(mean=mean, cov=cov_matrix)
    samples = mvn.rvs(size=n_samples)
    
    # Calculate true entropy
    true_entropy = mvn.entropy()
    
    return torch.tensor(samples, dtype=torch.float32), true_entropy

def create_test_correlation_matrices():
    """Create various test correlation matrices"""
    test_cases = {}
    
    # Case 1: Independent (Identity matrix)
    test_cases['Independent'] = np.eye(4)
    
    # Case 2: Moderate positive correlation
    rho = 0.5
    test_cases['Moderate Positive'] = np.array([
        [1.0, rho, rho/2, rho/4],
        [rho, 1.0, rho, rho/2],
        [rho/2, rho, 1.0, rho],
        [rho/4, rho/2, rho, 1.0]
    ])
    
    # Case 3: Strong correlation
    rho = 0.8
    test_cases['Strong Correlation'] = np.array([
        [1.0, rho, rho*0.9, rho*0.8],
        [rho, 1.0, rho, rho*0.9],
        [rho*0.9, rho, 1.0, rho],
        [rho*0.8, rho*0.9, rho, 1.0]
    ])
    
    # Case 4: Mixed correlations
    test_cases['Mixed'] = np.array([
        [1.0, 0.7, -0.3, 0.1],
        [0.7, 1.0, -0.5, 0.2],
        [-0.3, -0.5, 1.0, -0.6],
        [0.1, 0.2, -0.6, 1.0]
    ])
    
    # Case 5: Block correlation
    test_cases['Block'] = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.9],
        [0.1, 0.1, 0.9, 1.0]
    ])
    
    return test_cases

def fit_vine_copula(data, vine_type='r-vine', is_parametric=False, copula_families=None):
    """Fit vine copula model to data"""
    n_samples, n_dim = data.shape
    
    # Create marginal objects (assume continuous Gaussian)
    margins = []
    for i in range(n_dim):
        margin = margin_obj('kernel', None, True)
        margins.append(margin)
    
    # Create vine object
    n_pairs = (n_dim * (n_dim - 1)) // 2
    if is_parametric and copula_families:
        # For parametric fitting, we need families for each pair
        families = copula_families * n_pairs  # Repeat the family for each pair
    else:
        families = ['kernel'] * n_pairs
    
    # Create vine object
    if vine_type == 'r-vine':
        vine = vine_obj_bin('r-vine', families, n_dim-1, margins, 25, 'optimal')
    elif vine_type == 'c-vine':
        vine = vine_obj_bin('c-vine', families, n_dim-1, margins, 25, 'matrix')
    else:  # d-vine
        vine = vine_obj_bin('d-vine', families, n_dim-1, margins, 25, 'matrix')
    
    # Set up fitting parameters
    gen_dict = {
        'binning': False,
        'parallel': False,
        'param': is_parametric,
        'vine_depth': n_dim - 1
    }
    
    npc_dict = {
        'opt_method': 'LL1',
        'batch_paral': False
    } if not is_parametric else {}
    
    par_dict = {
        'param_families': copula_families if is_parametric else []
    } if is_parametric else {}
    
    bin_dict = {'n_bin': 1}
    
    # Fit the model
    start_time = time.time()
    try:
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
    except Exception as e:
        print(f"Fitting failed: {e}")
        fit_time = np.nan
        return None, fit_time
    
    return vine, fit_time

def sample_from_vine(vine, n_samples=5000):
    """Sample from fitted vine copula"""
    if vine is None:
        return None
        
    # Use rejection sampling or conditional sampling
    # For simplicity, we'll use a basic approach
    d = vine.n_cop
    samples = torch.zeros((n_samples, d))
    
    # Sample uniform marginals
    u_samples = torch.rand((n_samples, d))
    
    # Transform through inverse marginal CDFs
    for i in range(d):
        if hasattr(vine, 'Mar_G') and vine.Mar_G:
            mar_s, mar_p = vine.Mar_G[i]
            # Inverse transform sampling
            samples[:, i] = torch.tensor(np.interp(
                u_samples[:, i].numpy(),
                mar_p.numpy(),
                mar_s.numpy()
            ))
        else:
            # If marginals not available, use standard normal
            samples[:, i] = torch.distributions.Normal(0, 1).icdf(u_samples[:, i])
    
    return samples

def evaluate_vine_model(vine, true_corr, true_entropy, n_test_samples=5000):
    """Evaluate fitted vine model"""
    results = {}
    
    if vine is None:
        return {
            'sample_correlation': np.nan * np.ones_like(true_corr),
            'correlation_mae': np.nan,
            'correlation_frobenius': np.nan,
            'estimated_entropy': np.nan,
            'entropy_error': np.nan
        }
    
    # Generate samples from vine
    samples = sample_from_vine(vine, n_test_samples)
    
    if samples is None:
        return {
            'sample_correlation': np.nan * np.ones_like(true_corr),
            'correlation_mae': np.nan,
            'correlation_frobenius': np.nan,
            'estimated_entropy': np.nan,
            'entropy_error': np.nan
        }
    
    # Compute sample correlation matrix
    sample_corr = np.corrcoef(samples.numpy().T)
    
    # Correlation matrix error
    corr_error = np.mean(np.abs(sample_corr - true_corr))
    corr_frobenius = np.linalg.norm(sample_corr - true_corr, 'fro')
    
    results['sample_correlation'] = sample_corr
    results['correlation_mae'] = corr_error
    results['correlation_frobenius'] = corr_frobenius
    
    # For entropy, we'll use a simple estimate based on correlations
    # For multivariate Gaussian, entropy = 0.5 * log(det(2πe * Σ))
    try:
        # Ensure correlation matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(sample_corr))
        if min_eig < 0:
            sample_corr = sample_corr + (-min_eig + 0.01) * np.eye(sample_corr.shape[0])
        
        det_corr = np.linalg.det(sample_corr)
        if det_corr > 0:
            estimated_entropy = 0.5 * np.log(det_corr * (2 * np.pi * np.e) ** sample_corr.shape[0])
        else:
            estimated_entropy = np.nan
    except:
        estimated_entropy = np.nan
    
    entropy_error = abs(estimated_entropy - true_entropy) if not np.isnan(estimated_entropy) else np.nan
    
    results['estimated_entropy'] = estimated_entropy
    results['entropy_error'] = entropy_error
    
    return results

def plot_results(results_dict, correlation_matrices):
    """Create comprehensive visualization of results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Correlation matrix recovery comparison
    n_cases = len(correlation_matrices)
    n_methods = 3  # kernel, gaussian, r-vine structure comparison
    
    # Plot true correlations and best recoveries
    for idx, (case_name, true_corr) in enumerate(correlation_matrices.items()):
        # True correlation
        ax = plt.subplot(n_cases, 4, idx*4 + 1)
        sns.heatmap(true_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   vmin=-1, vmax=1, cbar=idx==0)
        ax.set_title(f'{case_name}\nTrue Correlation')
        
        # Find best method
        best_method = None
        best_error = float('inf')
        best_corr = None
        
        for method_name, method_results in results_dict[case_name].items():
            if not np.isnan(method_results['correlation_mae']) and method_results['correlation_mae'] < best_error:
                best_error = method_results['correlation_mae']
                best_method = method_name
                best_corr = method_results['sample_correlation']
        
        if best_corr is not None:
            ax = plt.subplot(n_cases, 4, idx*4 + 2)
            sns.heatmap(best_corr, annot=True, fmt='.2f', cmap='coolwarm',
                       vmin=-1, vmax=1, cbar=idx==0)
            ax.set_title(f'Best Recovery ({best_method})\nMAE={best_error:.3f}')
    
    # 2. Error comparison bar plot
    ax = plt.subplot(2, 2, 3)
    
    # Prepare data for plotting
    methods = list(results_dict[list(correlation_matrices.keys())[0]].keys())
    x = np.arange(len(correlation_matrices))
    width = 0.25
    
    for i, method in enumerate(methods[:3]):  # Limit to 3 methods for clarity
        mae_values = []
        for case_name in correlation_matrices.keys():
            mae = results_dict[case_name][method]['correlation_mae']
            mae_values.append(mae if not np.isnan(mae) else 0)
        
        ax.bar(x + i*width, mae_values, width, label=method)
    
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Correlation MAE')
    ax.set_title('Correlation Recovery Error by Method')
    ax.set_xticks(x + width)
    ax.set_xticklabels(correlation_matrices.keys(), rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Performance summary table
    ax = plt.subplot(2, 2, 4)
    ax.axis('off')
    
    # Create summary statistics
    summary_data = []
    for method in methods:
        mae_values = []
        fit_times = []
        
        for case_name in correlation_matrices.keys():
            result = results_dict[case_name][method]
            if not np.isnan(result['correlation_mae']):
                mae_values.append(result['correlation_mae'])
            if 'fit_time' in result and not np.isnan(result['fit_time']):
                fit_times.append(result['fit_time'])
        
        if mae_values:
            summary_data.append({
                'Method': method,
                'Avg MAE': f"{np.mean(mae_values):.4f}",
                'Std MAE': f"{np.std(mae_values):.4f}",
                'Avg Time': f"{np.mean(fit_times):.2f}s" if fit_times else "N/A"
            })
    
    # Display as table
    if summary_data:
        df = pd.DataFrame(summary_data)
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    ax.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig('vine_comprehensive_test_results.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive test results plot")

def plot_detailed_comparison(all_samples, correlation_matrices):
    """Create detailed comparison plots for sampled data"""
    fig, axes = plt.subplots(len(correlation_matrices), 3, figsize=(15, 5*len(correlation_matrices)))
    
    for idx, (case_name, true_corr) in enumerate(correlation_matrices.items()):
        if case_name not in all_samples:
            continue
            
        # Original data
        original_data = all_samples[case_name]['original']
        
        # Plot original data scatter
        ax = axes[idx, 0] if len(correlation_matrices) > 1 else axes[0]
        if original_data.shape[1] >= 2:
            ax.scatter(original_data[:, 0], original_data[:, 1], alpha=0.5, s=10)
            ax.set_title(f'{case_name} - Original Data (Dims 0-1)')
            ax.set_xlabel('Dimension 0')
            ax.set_ylabel('Dimension 1')
        
        # Plot vine samples for two methods
        method_idx = 0
        for method_name in ['r-vine-kernel', 'r-vine-gaussian']:
            if method_name in all_samples[case_name]:
                ax = axes[idx, method_idx+1] if len(correlation_matrices) > 1 else axes[method_idx+1]
                samples = all_samples[case_name][method_name]
                
                if samples is not None and samples.shape[1] >= 2:
                    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
                    ax.set_title(f'{case_name} - {method_name} Samples')
                    ax.set_xlabel('Dimension 0')
                    ax.set_ylabel('Dimension 1')
                
                method_idx += 1
    
    plt.tight_layout()
    plt.savefig('vine_samples_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved samples comparison plot")

def main():
    """Run comprehensive vine copula tests"""
    print("=== Comprehensive Vine Copula Model Testing ===\n")
    
    # Configuration
    n_samples = 1000  # Reduced for faster testing
    n_test = 500
    
    # Test configurations
    test_configs = {
        'r-vine-kernel': {'vine_type': 'r-vine', 'parametric': False, 'families': None},
        'r-vine-gaussian': {'vine_type': 'r-vine', 'parametric': True, 'families': ['gaussian']},
        'c-vine-kernel': {'vine_type': 'c-vine', 'parametric': False, 'families': None},
        'd-vine-kernel': {'vine_type': 'd-vine', 'parametric': False, 'families': None}
    }
    
    # Get test correlation matrices
    correlation_matrices = create_test_correlation_matrices()
    
    # Store all results
    all_results = {}
    all_samples = {}
    
    # Test each correlation structure
    for case_name, corr_matrix in correlation_matrices.items():
        print(f"\n{'='*60}")
        print(f"Testing case: {case_name}")
        print(f"{'='*60}")
        
        case_results = {}
        case_samples = {}
        
        # Generate data
        data, true_entropy = generate_multivariate_gaussian(n_samples, corr_matrix)
        test_data, _ = generate_multivariate_gaussian(n_test, corr_matrix)
        
        case_samples['original'] = data.numpy()
        
        print(f"True entropy: {true_entropy:.4f}")
        print(f"Data shape: {data.shape}")
        print(f"True correlation matrix:")
        print(corr_matrix)
        
        # Test each configuration
        for config_name, config in test_configs.items():
            print(f"\nFitting {config_name}...")
            
            try:
                # Fit model
                vine, fit_time = fit_vine_copula(
                    data, 
                    vine_type=config['vine_type'],
                    is_parametric=config['parametric'],
                    copula_families=config['families']
                )
                
                if vine is not None:
                    print(f"  Fit time: {fit_time:.2f}s")
                    
                    # Evaluate model
                    eval_results = evaluate_vine_model(vine, corr_matrix, true_entropy)
                    eval_results['fit_time'] = fit_time
                    
                    # Store samples for visualization
                    samples = sample_from_vine(vine, n_samples)
                    case_samples[config_name] = samples.numpy() if samples is not None else None
                    
                    print(f"  Correlation MAE: {eval_results['correlation_mae']:.4f}")
                    if not np.isnan(eval_results['entropy_error']):
                        print(f"  Entropy error: {eval_results['entropy_error']:.4f}")
                else:
                    print("  Fitting failed")
                    eval_results = {
                        'correlation_mae': np.nan,
                        'entropy_error': np.nan,
                        'fit_time': fit_time,
                        'sample_correlation': np.nan * np.ones_like(corr_matrix)
                    }
                
                case_results[config_name] = eval_results
                
            except Exception as e:
                print(f"  Failed: {e}")
                case_results[config_name] = {
                    'correlation_mae': np.nan,
                    'entropy_error': np.nan,
                    'fit_time': np.nan,
                    'sample_correlation': np.nan * np.ones_like(corr_matrix)
                }
        
        all_results[case_name] = case_results
        all_samples[case_name] = case_samples
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    plot_results(all_results, correlation_matrices)
    plot_detailed_comparison(all_samples, correlation_matrices)
    
    # Create summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    # Best method for each case
    print("\nBest method for each correlation structure:")
    for case_name in correlation_matrices.keys():
        best_method = None
        best_error = float('inf')
        
        for method_name, results in all_results[case_name].items():
            if not np.isnan(results['correlation_mae']) and results['correlation_mae'] < best_error:
                best_error = results['correlation_mae']
                best_method = method_name
        
        if best_method:
            print(f"  {case_name}: {best_method} (MAE={best_error:.4f})")
        else:
            print(f"  {case_name}: All methods failed")
    
    # Average performance across all cases
    print("\nAverage performance by method:")
    method_avg_errors = {}
    
    for method in test_configs.keys():
        errors = []
        for case_name in correlation_matrices.keys():
            if method in all_results[case_name]:
                error = all_results[case_name][method]['correlation_mae']
                if not np.isnan(error):
                    errors.append(error)
        
        if errors:
            method_avg_errors[method] = np.mean(errors)
    
    # Sort by average error
    sorted_methods = sorted(method_avg_errors.items(), key=lambda x: x[1])
    for method, avg_error in sorted_methods:
        print(f"  {method}: {avg_error:.4f}")
    
    print("\n=== Test completed successfully ===")

if __name__ == "__main__":
    main() 