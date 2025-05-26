import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import time
import sys
import os

# Add TensorFlow implementation to path
sys.path.insert(0, '../DVC_tensorflow')

# Import PyTorch modules
from classes.objects import vine_obj_bin, margin_obj
from sampling.vine_sampling import VineSampler
from evalu.vine_entropy import VineEntropyCalculator

# Import TensorFlow modules
import tensorflow as tf
from classes.objects_tf import vine_obj_bin as vine_obj_bin_tf
from classes.objects_tf import margin_obj as margin_obj_tf


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


def fit_tensorflow_vine(data, vine_family='c-vine', copula_families=['gaussian'], param=True):
    """Fit vine copula using TensorFlow implementation"""
    print(f"\nFitting TensorFlow {vine_family} with {copula_families}...")
    
    n_samples, n_dim = data.shape
    
    # Create margin objects
    margins_tf = [margin_obj_tf('kernel', None, True) for _ in range(n_dim)]
    
    # Create vine object
    n_cop = n_dim - 1
    vine_tf = vine_obj_bin_tf(vine_family, copula_families, n_cop, margins_tf, 25, 'matrix')
    
    # Fitting parameters (TF format)
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
        'n_npc': 5
    }
    
    bin_dict = {'n_bin': 1}
    
    # Fit the vine
    start_time = time.time()
    vine_tf.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
    fit_time = time.time() - start_time
    print(f"TensorFlow fitting time: {fit_time:.2f}s")
    
    return vine_tf, fit_time


def sample_pytorch_vine(vine, n_samples=2000):
    """Sample from PyTorch vine"""
    sampler = VineSampler(vine)
    samples = sampler.sample(n_samples)
    return samples.cpu().numpy()


def sample_tensorflow_vine(vine_tf, n_samples=2000):
    """Sample from TensorFlow vine"""
    # TensorFlow vine sampling
    from sampling.vine_sampling_tf import sample_fun
    
    samples = sample_fun(vine_tf, n_samples)
    return samples


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
    entropy_calc = VineEntropyCalculator(vine)
    total_entropy = entropy_calc.total_entropy(n_samples=n_samples)
    return total_entropy


def compute_vine_entropy_tf(vine_tf, n_samples=5000):
    """Compute entropy for TensorFlow vine"""
    # Simplified entropy calculation for TF
    # Sample and use empirical entropy
    samples = sample_tensorflow_vine(vine_tf, n_samples)
    
    # Use k-NN entropy estimator
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
    
    # Plot true correlation
    sns.heatmap(true_corr, ax=axes[0], vmin=-1, vmax=1, cmap='coolwarm', 
                annot=True, fmt='.2f', square=True, cbar=False)
    axes[0].set_title("True Correlation")
    
    # Plot empirical correlations
    for i, (name, corr) in enumerate(corr_dict.items()):
        sns.heatmap(corr, ax=axes[i+1], vmin=-1, vmax=1, cmap='coolwarm',
                   annot=True, fmt='.2f', square=True, cbar=(i == len(corr_dict)-1))
        axes[i+1].set_title(f"{name}")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def main():
    """Main comparison function"""
    print("="*80)
    print("PYTORCH vs TENSORFLOW VINE COPULA COMPARISON")
    print("="*80)
    
    # Generate data
    n_samples = 2000
    dim = 5
    data, true_corr, true_entropy = generate_gaussian_data(n_samples, dim)
    
    # Test configurations
    vine_families = ['c-vine', 'd-vine']
    copula_configs = [
        (['gaussian'], True, 'Gaussian'),
        (['clayton'], True, 'Clayton'),
        (['student'], True, 'Student-t'),
        (['gaussian', 'clayton'], True, 'Mixed'),
        (['kernel'], False, 'Non-parametric')
    ]
    
    results = {}
    
    for vine_family in vine_families:
        print(f"\n{'='*60}")
        print(f"Testing {vine_family.upper()}")
        print(f"{'='*60}")
        
        results[vine_family] = {}
        
        for copula_families, param, name in copula_configs:
            print(f"\n--- {name} Copulas ---")
            
            try:
                # Fit PyTorch vine
                vine_pt, time_pt = fit_pytorch_vine(data, vine_family, copula_families, param)
                
                # Sample from PyTorch vine
                samples_pt = sample_pytorch_vine(vine_pt, n_samples)
                
                # Compute PyTorch metrics
                corr_pt, frob_pt, max_pt, mae_pt = compute_correlation_accuracy(true_corr, samples_pt)
                entropy_pt = compute_vine_entropy(vine_pt)
                
            except Exception as e:
                print(f"PyTorch failed: {e}")
                corr_pt = np.nan * np.ones_like(true_corr)
                frob_pt = max_pt = mae_pt = entropy_pt = time_pt = np.nan
            
            try:
                # Fit TensorFlow vine
                vine_tf, time_tf = fit_tensorflow_vine(data, vine_family, copula_families, param)
                
                # Sample from TensorFlow vine
                samples_tf = sample_tensorflow_vine(vine_tf, n_samples)
                
                # Compute TensorFlow metrics
                corr_tf, frob_tf, max_tf, mae_tf = compute_correlation_accuracy(true_corr, samples_tf)
                entropy_tf = compute_vine_entropy_tf(vine_tf, n_samples)
                
            except Exception as e:
                print(f"TensorFlow failed: {e}")
                corr_tf = np.nan * np.ones_like(true_corr)
                frob_tf = max_tf = mae_tf = entropy_tf = time_tf = np.nan
            
            # Store results
            results[vine_family][name] = {
                'pytorch': {
                    'corr': corr_pt,
                    'frob_error': frob_pt,
                    'max_error': max_pt,
                    'mae': mae_pt,
                    'entropy': entropy_pt,
                    'entropy_error': abs(entropy_pt - true_entropy) if not np.isnan(entropy_pt) else np.nan,
                    'time': time_pt
                },
                'tensorflow': {
                    'corr': corr_tf,
                    'frob_error': frob_tf,
                    'max_error': max_tf,
                    'mae': mae_tf,
                    'entropy': entropy_tf,
                    'entropy_error': abs(entropy_tf - true_entropy) if not np.isnan(entropy_tf) else np.nan,
                    'time': time_tf
                }
            }
            
            # Print comparison
            print(f"\nResults for {name} copulas:")
            print(f"{'Metric':<25} {'PyTorch':<15} {'TensorFlow':<15}")
            print("-"*55)
            print(f"{'Fitting time (s)':<25} {time_pt:<15.3f} {time_tf:<15.3f}")
            print(f"{'Correlation Frob. error':<25} {frob_pt:<15.4f} {frob_tf:<15.4f}")
            print(f"{'Correlation max error':<25} {max_pt:<15.4f} {max_tf:<15.4f}")
            print(f"{'Correlation MAE':<25} {mae_pt:<15.4f} {mae_tf:<15.4f}")
            print(f"{'Entropy estimate':<25} {entropy_pt:<15.4f} {entropy_tf:<15.4f}")
            print(f"{'Entropy error':<25} {abs(entropy_pt - true_entropy):<15.4f} {abs(entropy_tf - true_entropy):<15.4f}")
    
    # Create comparison plots
    for vine_family in vine_families:
        for copula_name in ['Gaussian', 'Non-parametric']:
            if copula_name in results[vine_family]:
                res = results[vine_family][copula_name]
                
                corr_dict = {
                    'PyTorch': res['pytorch']['corr'],
                    'TensorFlow': res['tensorflow']['corr']
                }
                
                fig = plot_correlation_comparison(
                    true_corr, corr_dict, 
                    f"{vine_family.upper()} - {copula_name} Copulas"
                )
                plt.savefig(f'comparison_{vine_family}_{copula_name.lower()}.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Average Performance Across All Configurations")
    print("="*80)
    
    # Compute averages
    pt_mae_avg = []
    tf_mae_avg = []
    pt_entropy_err_avg = []
    tf_entropy_err_avg = []
    pt_time_avg = []
    tf_time_avg = []
    
    for vine_family in results:
        for copula_type in results[vine_family]:
            res = results[vine_family][copula_type]
            if not np.isnan(res['pytorch']['mae']):
                pt_mae_avg.append(res['pytorch']['mae'])
                pt_entropy_err_avg.append(res['pytorch']['entropy_error'])
                pt_time_avg.append(res['pytorch']['time'])
            if not np.isnan(res['tensorflow']['mae']):
                tf_mae_avg.append(res['tensorflow']['mae'])
                tf_entropy_err_avg.append(res['tensorflow']['entropy_error'])
                tf_time_avg.append(res['tensorflow']['time'])
    
    print(f"{'Metric':<30} {'PyTorch':<15} {'TensorFlow':<15}")
    print("-"*60)
    print(f"{'Avg. Correlation MAE':<30} {np.mean(pt_mae_avg):<15.4f} {np.mean(tf_mae_avg):<15.4f}")
    print(f"{'Avg. Entropy Error':<30} {np.mean(pt_entropy_err_avg):<15.4f} {np.mean(tf_entropy_err_avg):<15.4f}")
    print(f"{'Avg. Fitting Time (s)':<30} {np.mean(pt_time_avg):<15.3f} {np.mean(tf_time_avg):<15.3f}")
    print(f"{'Speedup Factor':<30} {np.mean(tf_time_avg)/np.mean(pt_time_avg):<15.2f}x")
    
    # Save detailed results
    import pickle
    with open('comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nComparison complete! Results saved to comparison_results.pkl")
    print("Plots saved as comparison_*.png")


if __name__ == "__main__":
    main() 