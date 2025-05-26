import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch modules
from classes.objects import vine_obj_bin, margin_obj
from sampling.vine_sampling import VineSampler
from evalu.vine_entropy import VineEntropyCalculator


def generate_gaussian_data(n_samples=2000, dim=5, seed=42):
    """Generate multivariate Gaussian data with specified correlation structure"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create a correlation matrix with interesting structure
    # Start with identity
    corr_matrix = np.eye(dim)
    
    # Add some strong correlations
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.8
    corr_matrix[0, 2] = corr_matrix[2, 0] = 0.6
    corr_matrix[1, 2] = corr_matrix[2, 1] = 0.5
    corr_matrix[2, 3] = corr_matrix[3, 2] = 0.7
    corr_matrix[3, 4] = corr_matrix[4, 3] = 0.4
    
    # Ensure positive definite
    eigenvalues = np.linalg.eigvals(corr_matrix)
    if np.min(eigenvalues) < 0.01:
        corr_matrix = corr_matrix + (0.01 - np.min(eigenvalues)) * np.eye(dim)
    
    print("True correlation matrix:")
    print(np.round(corr_matrix, 3))
    
    # Generate samples
    mean = np.zeros(dim)
    mvn = multivariate_normal(mean=mean, cov=corr_matrix)
    samples = mvn.rvs(size=n_samples)
    
    # Compute true entropy
    true_entropy = mvn.entropy()
    print(f"\nTrue entropy: {true_entropy:.4f}")
    
    return samples, corr_matrix, true_entropy


def fit_pytorch_vine(data, vine_family='c-vine', copula_families=['gaussian'], param=True):
    """Fit vine copula using PyTorch implementation"""
    print(f"\nFitting PyTorch {vine_family} with {copula_families}...")
    
    data_torch = torch.tensor(data, dtype=torch.float32)
    n_samples, n_dim = data.shape
    
    # Create margin objects
    margins = [margin_obj('kernel', None, True) for _ in range(n_dim)]
    
    # Create vine object
    n_cop = n_dim - 1
    vine = vine_obj_bin(vine_family, copula_families, n_cop, margins, 25, 'matrix')
    
    # Fitting parameters
    gen_dict = {
        'binning': False,
        'parallel': False,
        'param': param,
        'vine_depth': n_cop
    }
    
    par_dict = {
        'param_families': copula_families + ['ind']
    }
    
    npc_dict = {
        'n_npc': 5,
        'bandwidth_selection': 'cv'
    }
    
    bin_dict = {'n_bin': 1}
    
    # Fit the vine
    start_time = time.time()
    vine.fit(data_torch, gen_dict, npc_dict, par_dict, bin_dict)
    fit_time = time.time() - start_time
    print(f"PyTorch fitting time: {fit_time:.2f}s")
    
    return vine, fit_time


def sample_pytorch_vine(vine, n_samples=2000):
    """Sample from PyTorch vine"""
    sampler = VineSampler(vine)
    samples = sampler.sample(n_samples)
    return samples.cpu().numpy()


def compute_correlation_accuracy(true_corr, samples):
    """Compute correlation matrix accuracy"""
    emp_corr = np.corrcoef(samples.T)
    
    # Frobenius norm of difference
    frob_error = np.linalg.norm(true_corr - emp_corr, 'fro')
    
    # Maximum absolute error
    max_error = np.max(np.abs(true_corr - emp_corr))
    
    # Mean absolute error (off-diagonal only)
    mask = ~np.eye(true_corr.shape[0], dtype=bool)
    mae = np.mean(np.abs(true_corr[mask] - emp_corr[mask]))
    
    return emp_corr, frob_error, max_error, mae


def compute_vine_entropy(vine, n_samples=5000):
    """Compute entropy for PyTorch vine"""
    try:
        entropy_calc = VineEntropyCalculator(vine)
        total_entropy = entropy_calc.total_entropy(n_samples=n_samples)
        return total_entropy
    except Exception as e:
        print(f"Entropy calculation failed: {e}")
        # Fallback to empirical entropy
        samples = sample_pytorch_vine(vine, n_samples)
        return compute_empirical_entropy(samples)


def compute_empirical_entropy(samples):
    """Compute entropy using k-NN estimator"""
    from scipy.spatial import distance_matrix
    from scipy.special import digamma
    
    k = 3
    n, d = samples.shape
    
    # Compute distances
    dists = distance_matrix(samples, samples)
    np.fill_diagonal(dists, np.inf)
    
    # k-th nearest neighbor distances
    knn_dists = np.partition(dists, k, axis=1)[:, k]
    
    # Kozachenko-Leonenko estimator
    cd = np.pi**(d/2) / np.math.gamma(d/2 + 1)  # Volume of unit ball
    entropy = digamma(n) - digamma(k) + np.log(cd) + d * np.mean(np.log(2 * knn_dists))
    
    return entropy


def plot_correlation_comparison(true_corr, corr_dict, title="Correlation Matrix Comparison"):
    """Plot correlation matrices side by side"""
    n_plots = len(corr_dict) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
    
    if n_plots == 2:
        axes = [axes]
    
    # Plot true correlation
    ax = axes[0] if n_plots > 2 else axes
    sns.heatmap(true_corr, ax=ax, vmin=-1, vmax=1, cmap='coolwarm', 
                annot=True, fmt='.2f', square=True, cbar=False)
    ax.set_title("True Correlation")
    
    # Plot empirical correlations
    for i, (name, corr) in enumerate(corr_dict.items()):
        ax = axes[i+1] if n_plots > 2 else plt.subplot(1, 2, 2)
        sns.heatmap(corr, ax=ax, vmin=-1, vmax=1, cmap='coolwarm',
                   annot=True, fmt='.2f', square=True, cbar=(i == len(corr_dict)-1))
        ax.set_title(f"{name}")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def test_different_vine_families(data, true_corr, true_entropy):
    """Test different vine families and copula types"""
    results = {}
    
    # Test configurations - focusing on what PyTorch supports well
    vine_families = ['c-vine', 'd-vine']
    copula_configs = [
        (['gaussian'], True, 'Gaussian'),
        (['clayton'], True, 'Clayton'),
        (['gaussian', 'clayton'], True, 'Mixed Gaussian-Clayton'),
    ]
    
    for vine_family in vine_families:
        print(f"\n{'='*60}")
        print(f"Testing {vine_family.upper()}")
        print(f"{'='*60}")
        
        results[vine_family] = {}
        
        for copula_families, param, name in copula_configs:
            print(f"\n--- {name} Copulas ---")
            
            try:
                # Fit PyTorch vine
                vine, fit_time = fit_pytorch_vine(data, vine_family, copula_families, param)
                
                # Sample from vine
                samples = sample_pytorch_vine(vine, data.shape[0])
                
                # Compute metrics
                corr_emp, frob_error, max_error, mae = compute_correlation_accuracy(true_corr, samples)
                entropy = compute_vine_entropy(vine)
                entropy_error = abs(entropy - true_entropy)
                
                results[vine_family][name] = {
                    'corr': corr_emp,
                    'frob_error': frob_error,
                    'max_error': max_error,
                    'mae': mae,
                    'entropy': entropy,
                    'entropy_error': entropy_error,
                    'time': fit_time,
                    'samples': samples
                }
                
                # Print results
                print(f"Fitting time: {fit_time:.3f}s")
                print(f"Correlation MAE: {mae:.4f}")
                print(f"Correlation max error: {max_error:.4f}")
                print(f"Entropy estimate: {entropy:.4f}")
                print(f"Entropy error: {entropy_error:.4f}")
                
            except Exception as e:
                print(f"Failed: {e}")
                import traceback
                traceback.print_exc()
                results[vine_family][name] = None
    
    return results


def create_summary_plots(results, true_corr, vine_family='c-vine'):
    """Create summary plots for a vine family"""
    valid_results = {k: v for k, v in results[vine_family].items() if v is not None}
    
    if not valid_results:
        print(f"No valid results for {vine_family}")
        return
    
    # 1. Correlation comparison plot
    fig = plt.figure(figsize=(15, 5 * len(valid_results)))
    
    for i, (name, res) in enumerate(valid_results.items()):
        # True correlation
        plt.subplot(len(valid_results), 3, i*3 + 1)
        sns.heatmap(true_corr, vmin=-1, vmax=1, cmap='coolwarm', 
                   annot=True, fmt='.2f', square=True, cbar=False)
        plt.title(f"{name} - True Correlation")
        
        # Empirical correlation
        plt.subplot(len(valid_results), 3, i*3 + 2)
        sns.heatmap(res['corr'], vmin=-1, vmax=1, cmap='coolwarm',
                   annot=True, fmt='.2f', square=True, cbar=False)
        plt.title(f"{name} - Empirical Correlation")
        
        # Difference
        plt.subplot(len(valid_results), 3, i*3 + 3)
        diff = res['corr'] - true_corr
        sns.heatmap(diff, vmin=-0.2, vmax=0.2, cmap='RdBu_r',
                   annot=True, fmt='.3f', square=True, cbar=True)
        plt.title(f"{name} - Difference")
    
    plt.suptitle(f"{vine_family.upper()} - Correlation Matrix Comparison")
    plt.tight_layout()
    plt.savefig(f'correlation_comparison_{vine_family}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    names = list(valid_results.keys())
    mae_values = [res['mae'] for res in valid_results.values()]
    entropy_errors = [res['entropy_error'] for res in valid_results.values()]
    fit_times = [res['time'] for res in valid_results.values()]
    
    # MAE bar plot
    axes[0, 0].bar(names, mae_values)
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Correlation MAE by Copula Type')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Entropy error bar plot
    axes[0, 1].bar(names, entropy_errors)
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].set_title('Entropy Estimation Error')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Fitting time bar plot
    axes[1, 0].bar(names, fit_times)
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Fitting Time')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Scatter plot: Entropy vs True
    axes[1, 1].scatter([res['entropy'] for res in valid_results.values()],
                      [true_entropy] * len(valid_results), s=100)
    for i, name in enumerate(names):
        axes[1, 1].annotate(name, (list(valid_results.values())[i]['entropy'], true_entropy))
    axes[1, 1].plot([true_entropy-1, true_entropy+1], [true_entropy-1, true_entropy+1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('Estimated Entropy')
    axes[1, 1].set_ylabel('True Entropy')
    axes[1, 1].set_title('Entropy Estimation Accuracy')
    
    plt.suptitle(f"{vine_family.upper()} - Performance Metrics")
    plt.tight_layout()
    plt.savefig(f'metrics_comparison_{vine_family}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main comparison function"""
    print("="*80)
    print("VINE COPULA SIMULATION AND COMPARISON")
    print("="*80)
    
    # Generate data
    n_samples = 2000
    dim = 5
    data, true_corr, true_entropy = generate_gaussian_data(n_samples, dim)
    
    # Test different configurations
    results = test_different_vine_families(data, true_corr, true_entropy)
    
    # Create plots for each vine family
    for vine_family in ['c-vine', 'd-vine']:
        if vine_family in results:
            create_summary_plots(results, true_corr, vine_family)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    print(f"\n{'Vine Family':<15} {'Copula Type':<25} {'Corr. MAE':<12} {'Entropy Err':<12} {'Fit Time':<10}")
    print("-"*80)
    
    for vine_family in results:
        for copula_type, res in results[vine_family].items():
            if res is not None:
                print(f"{vine_family:<15} {copula_type:<25} {res['mae']:<12.4f} {res['entropy_error']:<12.4f} {res['time']:<10.3f}s")
    
    # Save detailed results
    import pickle
    with open('vine_simulation_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'true_corr': true_corr,
            'true_entropy': true_entropy,
            'data': data
        }, f)
    
    print("\nSimulation complete!")
    print("Results saved to vine_simulation_results.pkl")
    print("Plots saved as correlation_comparison_*.png and metrics_comparison_*.png")
    
    # Additional analysis: best performing configuration
    best_mae = float('inf')
    best_config = None
    
    for vine_family in results:
        for copula_type, res in results[vine_family].items():
            if res is not None and res['mae'] < best_mae:
                best_mae = res['mae']
                best_config = (vine_family, copula_type)
    
    if best_config:
        print(f"\nBest configuration for correlation recovery: {best_config[0]} with {best_config[1]} copulas")
        print(f"Achieved MAE: {best_mae:.4f}")


if __name__ == "__main__":
    main() 