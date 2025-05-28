"""
TensorFlow DVC Correlation Prediction Test
==========================================

This test evaluates how well the TensorFlow implementation of Deep Vine Copula (DVC)
can fit to multivariate data, generate samples, and predict pairwise correlations.

The test includes:
1. Generation of multivariate data with known correlation structures
2. Fitting DVC models (both parametric and non-parametric)
3. Sampling from fitted models
4. Correlation recovery analysis
5. Comparison with true correlations
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add TensorFlow DVC to path
tensorflow_path = os.path.join(os.path.dirname(__file__), '..', 'DVC_tensorflow')
if tensorflow_path not in sys.path:
    sys.path.insert(0, tensorflow_path)

# Remove PyTorch path if it exists to avoid conflicts
pytorch_path = os.path.join(os.path.dirname(__file__))
if pytorch_path in sys.path:
    sys.path.remove(pytorch_path)

try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    from classes.objects import vine_obj_bin, margin_obj, cop_par_obj
    from sampling.vine_sample import vine_copula_sample, vine_cop_par_sample
    from pre_proc.preparation import prep_cop
    from utils.prob_op import kendalltau
    
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow DVC modules loaded successfully")
    
except ImportError as e:
    print(f"Failed to import TensorFlow DVC modules: {e}")
    TENSORFLOW_AVAILABLE = False


def generate_multivariate_data(n_samples: int, dim: int, correlation_type: str = 'ar1') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multivariate data with known correlation structure
    
    Args:
        n_samples: Number of samples to generate
        dim: Dimensionality of the data
        correlation_type: Type of correlation structure ('ar1', 'toeplitz', 'block', 'random')
        
    Returns:
        data: Generated data
        true_corr: True correlation matrix
    """
    np.random.seed(42)  # For reproducibility
    
    # Create correlation matrix based on type
    if correlation_type == 'ar1':
        rho = 0.7
        corr = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                corr[i, j] = rho ** abs(i - j)
                
    elif correlation_type == 'toeplitz':
        rho = 0.6
        corr = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                corr[i, j] = rho ** abs(i - j)
                
    elif correlation_type == 'block':
        corr = np.eye(dim)
        block_size = dim // 2
        # First block
        for i in range(block_size):
            for j in range(block_size):
                if i != j:
                    corr[i, j] = 0.8
        # Second block
        for i in range(block_size, dim):
            for j in range(block_size, dim):
                if i != j:
                    corr[i, j] = 0.6
                    
    elif correlation_type == 'random':
        # Generate random positive definite correlation matrix
        A = np.random.randn(dim, dim)
        corr = np.dot(A, A.T)
        # Normalize to correlation matrix
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)
        
    else:
        corr = np.eye(dim)
    
    # Generate multivariate normal data
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, corr, n_samples)
    
    # Transform some margins to make it more interesting
    if dim > 1:
        # Transform second variable to exponential
        data[:, 1] = stats.expon.ppf(stats.norm.cdf(data[:, 1]), scale=2)
    if dim > 2:
        # Transform third variable to uniform
        data[:, 2] = stats.uniform.ppf(stats.norm.cdf(data[:, 2]), loc=-1, scale=2)
    if dim > 3:
        # Transform fourth variable to gamma
        data[:, 3] = stats.gamma.ppf(stats.norm.cdf(data[:, 3]), a=2, scale=1)
    
    return data.astype(np.float32), corr


def compute_correlation_matrix(data: np.ndarray, method: str = 'kendall') -> np.ndarray:
    """Compute correlation matrix using specified method"""
    n, d = data.shape
    corr = np.eye(d)
    
    for i in range(d):
        for j in range(i+1, d):
            if method == 'kendall':
                tau, _ = stats.kendalltau(data[:, i], data[:, j])
                corr[i, j] = tau
                corr[j, i] = tau
            elif method == 'spearman':
                rho, _ = stats.spearmanr(data[:, i], data[:, j])
                corr[i, j] = rho
                corr[j, i] = rho
            else:  # pearson
                r = np.corrcoef(data[:, i], data[:, j])[0, 1]
                corr[i, j] = r
                corr[j, i] = r
    
    return corr


def create_vine_margins(dim: int) -> List:
    """Create margin objects for vine copula"""
    margin_vine = []
    for i in range(dim):
        # Use normal margins for simplicity
        mar_p = margin_obj('norm', [0, 1], True)
        margin_vine.append(mar_p)
    return margin_vine


def fit_tensorflow_vine(data: np.ndarray, vine_type: str = 'c-vine', 
                       parametric: bool = True, vine_depth: Optional[int] = None) -> object:
    """
    Fit TensorFlow DVC vine copula to data
    
    Args:
        data: Input data
        vine_type: Type of vine ('c-vine', 'd-vine', 'r-vine')
        parametric: Whether to use parametric copulas
        vine_depth: Depth of vine (if None, uses full depth)
        
    Returns:
        Fitted vine object
    """
    n_samples, dim = data.shape
    
    if vine_depth is None:
        vine_depth = dim
    
    # Create margins
    margin_vine = create_vine_margins(dim)
    
    # Create vine object
    families = "kercop" if not parametric else "param"
    knots = 50
    method = 'matrix'
    
    # For r-vine, we need an R-matrix
    if vine_type == 'r-vine':
        # Create a simple R-matrix
        r_matrix = np.zeros((dim, dim), dtype=np.int32)
        for i in range(dim):
            r_matrix[i, i] = i + 1
        vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method, r_matrix)
    else:
        vine = vine_obj_bin(vine_type, families, vine_depth, margin_vine, knots, method)
    
    # Prepare data
    sort_n = 'rand'
    prep_cop(data, vine, sort_n)
    
    # Set fitting parameters
    gen_dict = {
        'parallel': True, 
        'binning': False, 
        'param': parametric, 
        'vine_depth': vine_depth - 1,  # TensorFlow uses depth-1 
        'fitted': False
    }
    
    if parametric:
        par_dict = {'param_families': ["ind", "gaussian", "student", "clayton"]}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
    else:
        par_dict = {'param_families': ["ind", "gaussian"]}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
    
    bin_dict = {'n_bin': 3}
    
    # Fit the vine
    print(f"Fitting {vine_type} ({'parametric' if parametric else 'non-parametric'})...")
    start_time = time.time()
    
    try:
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        print(f"Fitting completed in {fit_time:.2f} seconds")
        return vine, fit_time
        
    except Exception as e:
        print(f"Fitting failed: {str(e)}")
        return None, None


def sample_from_vine(vine: object, n_samples: int, parametric: bool = True) -> np.ndarray:
    """Generate samples from fitted vine copula"""
    try:
        if parametric:
            samples = vine_cop_par_sample(vine, n_samples)
        else:
            samples, _, _, _ = vine_copula_sample(vine, n_samples)
        return samples
    except Exception as e:
        print(f"Sampling failed: {str(e)}")
        return None


def analyze_correlation_recovery(true_corr: np.ndarray, data_corr: np.ndarray, 
                               sample_corr: np.ndarray, method_name: str) -> Dict:
    """Analyze how well correlations are recovered"""
    
    # Compute errors
    data_error = np.mean(np.abs(data_corr - true_corr))
    sample_error = np.mean(np.abs(sample_corr - true_corr))
    
    # Compute correlation between true and recovered
    true_flat = true_corr[np.triu_indices_from(true_corr, k=1)]
    sample_flat = sample_corr[np.triu_indices_from(sample_corr, k=1)]
    
    if len(true_flat) > 1:
        recovery_corr, _ = stats.pearsonr(true_flat, sample_flat)
    else:
        recovery_corr = 1.0 if len(true_flat) == 1 and np.abs(true_flat[0] - sample_flat[0]) < 0.1 else 0.0
    
    # Compute maximum absolute error
    max_error = np.max(np.abs(sample_corr - true_corr))
    
    return {
        'method': method_name,
        'data_error': data_error,
        'sample_error': sample_error,
        'recovery_correlation': recovery_corr,
        'max_error': max_error,
        'mean_true_corr': np.mean(np.abs(true_flat)),
        'mean_sample_corr': np.mean(np.abs(sample_flat))
    }


def create_correlation_plots(true_corr: np.ndarray, data_corr: np.ndarray, 
                           results: Dict, save_prefix: str):
    """Create visualization plots for correlation analysis"""
    
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods + 2, figsize=(4 * (n_methods + 2), 8))
    
    if n_methods == 1:
        axes = axes.reshape(2, -1)
    
    # Plot true and data correlations
    matrices = [true_corr, data_corr]
    titles = ['True Correlation', 'Data Correlation']
    
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        sns.heatmap(matrix, ax=axes[0, i], cmap='coolwarm', center=0,
                    vmin=-1, vmax=1, square=True, cbar=True,
                    annot=True, fmt='.2f', annot_kws={'size': 8})
        axes[0, i].set_title(title)
    
    # Plot sample correlations for each method
    col_idx = 2
    for method_name, method_results in results.items():
        if 'sample_correlation' in method_results:
            sample_corr = method_results['sample_correlation']
            sns.heatmap(sample_corr, ax=axes[0, col_idx], cmap='coolwarm', center=0,
                        vmin=-1, vmax=1, square=True, cbar=True,
                        annot=True, fmt='.2f', annot_kws={'size': 8})
            axes[0, col_idx].set_title(f'{method_name}\nSample Correlation')
            
            # Plot error heatmap
            error_matrix = np.abs(sample_corr - true_corr)
            sns.heatmap(error_matrix, ax=axes[1, col_idx], cmap='Reds',
                        square=True, cbar=True,
                        annot=True, fmt='.3f', annot_kws={'size': 8})
            axes[1, col_idx].set_title(f'{method_name}\nAbsolute Error')
            
        col_idx += 1
    
    # Remove unused subplots
    for i in range(col_idx, axes.shape[1]):
        axes[0, i].remove()
        axes[1, i].remove()
    
    # Plot data error
    data_error_matrix = np.abs(data_corr - true_corr)
    sns.heatmap(data_error_matrix, ax=axes[1, 1], cmap='Reds',
                square=True, cbar=True,
                annot=True, fmt='.3f', annot_kws={'size': 8})
    axes[1, 1].set_title('Data Error')
    
    plt.suptitle(f'TensorFlow DVC Correlation Recovery Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_tensorflow_correlation_test():
    """Run comprehensive TensorFlow DVC correlation prediction test"""
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow DVC not available. Skipping test.")
        return
    
    print("TensorFlow DVC Correlation Prediction Test")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {'dim': 3, 'n_samples': 500, 'corr_type': 'ar1'},
        {'dim': 3, 'n_samples': 500, 'corr_type': 'block'},
        {'dim': 4, 'n_samples': 600, 'corr_type': 'toeplitz'},
    ]
    
    vine_types = ['c-vine', 'd-vine']
    methods = [
        {'name': 'Parametric', 'parametric': True},
        {'name': 'Non-parametric', 'parametric': False}
    ]
    
    all_results = []
    
    for config in test_configs:
        dim = config['dim']
        n_samples = config['n_samples']
        corr_type = config['corr_type']
        
        print(f"\nTesting: dim={dim}, n_samples={n_samples}, correlation={corr_type}")
        print("-" * 50)
        
        # Generate test data
        data, true_corr = generate_multivariate_data(n_samples, dim, corr_type)
        data_corr = compute_correlation_matrix(data, method='kendall')
        
        print(f"Data shape: {data.shape}")
        print(f"True correlation range: [{np.min(true_corr):.3f}, {np.max(true_corr):.3f}]")
        print(f"Data correlation range: [{np.min(data_corr):.3f}, {np.max(data_corr):.3f}]")
        
        for vine_type in vine_types:
            print(f"\n  Testing {vine_type}:")
            
            method_results = {}
            
            for method in methods:
                method_name = method['name']
                parametric = method['parametric']
                
                print(f"    {method_name} method...")
                
                try:
                    # Fit vine copula
                    vine, fit_time = fit_tensorflow_vine(
                        data, vine_type=vine_type, 
                        parametric=parametric, vine_depth=dim
                    )
                    
                    if vine is None:
                        print(f"      Failed to fit {method_name} {vine_type}")
                        continue
                    
                    # Generate samples
                    print(f"      Generating samples...")
                    samples = sample_from_vine(vine, 1000, parametric=parametric)
                    
                    if samples is None:
                        print(f"      Failed to sample from {method_name} {vine_type}")
                        continue
                    
                    # Compute sample correlations
                    sample_corr = compute_correlation_matrix(samples, method='kendall')
                    
                    # Analyze correlation recovery
                    analysis = analyze_correlation_recovery(
                        true_corr, data_corr, sample_corr, 
                        f"{method_name} {vine_type}"
                    )
                    analysis['fit_time'] = fit_time
                    analysis['sample_correlation'] = sample_corr
                    analysis['vine_type'] = vine_type
                    analysis['dimension'] = dim
                    analysis['n_samples'] = n_samples
                    analysis['correlation_type'] = corr_type
                    
                    method_results[f"{method_name} {vine_type}"] = analysis
                    
                    print(f"      Success! Error: {analysis['sample_error']:.4f}, "
                          f"Recovery: {analysis['recovery_correlation']:.4f}, "
                          f"Time: {fit_time:.2f}s")
                    
                    all_results.append(analysis)
                    
                except Exception as e:
                    print(f"      Error in {method_name} {vine_type}: {str(e)}")
                    continue
            
            # Create plots for this configuration
            if method_results:
                save_prefix = f"tensorflow_dvc_d{dim}_{corr_type}_{vine_type}"
                create_correlation_plots(true_corr, data_corr, method_results, save_prefix)
    
    # Create summary analysis
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 60)
        print("TENSORFLOW DVC CORRELATION PREDICTION SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal successful tests: {len(df)}")
        
        # Group by method type
        for method_type in ['Parametric', 'Non-parametric']:
            method_df = df[df['method'].str.contains(method_type)]
            if len(method_df) > 0:
                print(f"\n{method_type} Methods:")
                print(f"  Average correlation error:     {method_df['sample_error'].mean():.4f} ± {method_df['sample_error'].std():.4f}")
                print(f"  Average recovery correlation:  {method_df['recovery_correlation'].mean():.4f} ± {method_df['recovery_correlation'].std():.4f}")
                print(f"  Average fit time:              {method_df['fit_time'].mean():.2f} ± {method_df['fit_time'].std():.2f} seconds")
                print(f"  Average max error:             {method_df['max_error'].mean():.4f} ± {method_df['max_error'].std():.4f}")
        
        # Group by vine type
        print(f"\nBy Vine Type:")
        for vine_type in df['vine_type'].unique():
            vine_df = df[df['vine_type'] == vine_type]
            print(f"  {vine_type}:")
            print(f"    Average correlation error:     {vine_df['sample_error'].mean():.4f} ± {vine_df['sample_error'].std():.4f}")
            print(f"    Average recovery correlation:  {vine_df['recovery_correlation'].mean():.4f} ± {vine_df['recovery_correlation'].std():.4f}")
        
        # Save detailed results
        df.to_csv('tensorflow_dvc_correlation_prediction_results.csv', index=False)
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Error by method
        method_types = df['method'].str.extract(r'(Parametric|Non-parametric)')[0]
        df_plot = df.copy()
        df_plot['Method Type'] = method_types
        
        df_plot.boxplot(column='sample_error', by='Method Type', ax=axes[0, 0])
        axes[0, 0].set_title('Correlation Error by Method Type')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        
        # Plot 2: Recovery correlation by method
        df_plot.boxplot(column='recovery_correlation', by='Method Type', ax=axes[0, 1])
        axes[0, 1].set_title('Recovery Correlation by Method Type')
        axes[0, 1].set_ylabel('Correlation with True Values')
        
        # Plot 3: Fit time by method
        df_plot.boxplot(column='fit_time', by='Method Type', ax=axes[1, 0])
        axes[1, 0].set_title('Fit Time by Method Type')
        axes[1, 0].set_ylabel('Time (seconds)')
        
        # Plot 4: Error vs dimension
        for vine_type in df['vine_type'].unique():
            vine_data = df[df['vine_type'] == vine_type]
            axes[1, 1].scatter(vine_data['dimension'], vine_data['sample_error'], 
                             label=vine_type, alpha=0.7, s=60)
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Correlation Error')
        axes[1, 1].set_title('Error vs Dimension')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('TensorFlow DVC Correlation Prediction Summary', fontsize=14)
        plt.tight_layout()
        plt.savefig('tensorflow_dvc_correlation_prediction_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to:")
        print(f"  - tensorflow_dvc_correlation_prediction_results.csv")
        print(f"  - tensorflow_dvc_correlation_prediction_summary.png")
        print(f"  - tensorflow_dvc_d*_*_*.png (individual test plots)")
        
        # Key insights
        print(f"\nKey Insights:")
        best_method = df.loc[df['sample_error'].idxmin(), 'method']
        best_error = df['sample_error'].min()
        print(f"  - Best method: {best_method} (error: {best_error:.4f})")
        
        param_error = df[df['method'].str.contains('Parametric')]['sample_error'].mean()
        nonparam_error = df[df['method'].str.contains('Non-parametric')]['sample_error'].mean()
        
        if param_error < nonparam_error:
            print(f"  - Parametric methods performed better on average ({param_error:.4f} vs {nonparam_error:.4f})")
        else:
            print(f"  - Non-parametric methods performed better on average ({nonparam_error:.4f} vs {param_error:.4f})")
        
        avg_recovery = df['recovery_correlation'].mean()
        print(f"  - Average correlation recovery: {avg_recovery:.4f}")
        
        if avg_recovery > 0.8:
            print(f"  - Excellent correlation recovery achieved")
        elif avg_recovery > 0.6:
            print(f"  - Good correlation recovery achieved")
        else:
            print(f"  - Moderate correlation recovery achieved")


if __name__ == "__main__":
    run_tensorflow_correlation_test() 