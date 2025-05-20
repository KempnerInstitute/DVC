import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats

# Add DVC to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from DVC.vine_model import fit_vine, evaluate_vine, sample_vine
from DVC.objects import vine_obj_bin, margin_obj

# Global variable to store correlation matrix for test data generation
correlation_matrix = None

def generate_correlated_data(N, dimension, corr_matrix=None):
    """
    Generate multivariate normal data with a specified correlation matrix.
    If corr_matrix is None, generate a random positive definite matrix.
    """
    if corr_matrix is None:
        # Create a random correlation matrix
        A = np.random.randn(dimension, dimension)
        corr_matrix = np.dot(A, A.T)
        # Normalize to correlation matrix
        D = np.diag(1.0 / np.sqrt(np.diag(corr_matrix)))
        corr_matrix = np.dot(np.dot(D, corr_matrix), D)
    
    # Generate multivariate normal data
    data = np.random.multivariate_normal(
        mean=np.zeros(dimension),
        cov=corr_matrix,
        size=N
    )
    return data, corr_matrix

def fit_and_evaluate_vine(data, vine_family='c-vine', param=True):
    """
    Fit a vine model to data and evaluate its performance.
    
    Args:
        data: numpy array of shape [N, dimension]
        vine_family: 'c-vine' or 'd-vine'
        param: whether to use parametric copulas
        
    Returns:
        Dictionary with test results
    """
    global correlation_matrix
    
    N, dimension = data.shape
    
    # Create margin objects for each variable
    margins = []
    for i in range(dimension):
        margins.append(margin_obj('norm', [0.0, 1.0], True))
    
    # Create vine object
    vine = vine_obj_bin(vine_family, 'gaussian' if param else 'kercop', dimension, margins, 30, 'matrix')
    
    # Configuration dictionaries for fit_vine
    gen_dict = {
        'binning': False,
        'parallel': False,
        'param': param,
        'fitted': False,
        'vine_depth': dimension
    }
    
    npc_dict = {
        'npc_family': 'locallik',
        'grid_dim': 30,
        'grad_precompute': True if param else False  # Only use for parametric to avoid grid size issues
    }
    
    par_dict = {
        'param_families': ['gaussian']  # Only use Gaussian for simplicity
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit the vine to our data - only use parametric mode for testing
    vine = fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    
    # Generate test samples directly for evaluation
    n_test = 1000
    test_data = np.random.multivariate_normal(
        mean=np.zeros(dimension),
        cov=correlation_matrix,
        size=n_test
    )
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    
    # Evaluate log-likelihood on test data
    log_prob = evaluate_vine(vine, test_tensor)[0]
    mean_log_prob = log_prob.mean().item()
    
    # For parametric models, we can evaluate the copula structure directly
    corr_orig = np.corrcoef(data.T)
    corr_pairs = []
    
    if param:
        # Extract the fitted correlation parameters from the copulas
        fitted_corrs = np.eye(dimension)
        for level in range(dimension-1):
            for i, edge in enumerate(vine.ind_vine[level]):
                if i < len(vine.copulas[level]):
                    cop = vine.copulas[level][i]
                    if hasattr(cop, 'family') and cop.family == 'gaussian' and hasattr(cop, 'theta'):
                        rho = float(cop.theta)
                        i, j = edge
                        fitted_corrs[i, j] = rho
                        fitted_corrs[j, i] = rho
        
        # Compare original vs fitted correlations
        for i in range(dimension):
            for j in range(i+1, dimension):
                corr_orig_ij = corr_orig[i, j]
                corr_fitted_ij = fitted_corrs[i, j]
                diff = abs(corr_orig_ij - corr_fitted_ij)
                corr_pairs.append((i, j, corr_orig_ij, corr_fitted_ij, diff))
        
        # Plot correlation matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im0 = axes[0].imshow(corr_orig, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title('Original Data Correlation')
        axes[0].set_xticks(np.arange(dimension))
        axes[0].set_yticks(np.arange(dimension))
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(fitted_corrs, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title('Fitted Correlation Parameters')
        axes[1].set_xticks(np.arange(dimension))
        axes[1].set_yticks(np.arange(dimension))
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"Vine Model: {vine_family}, Parametric: {param}")
        plt.tight_layout()
        plt.savefig(f'vine_correlation_{vine_family}_param{param}.png')
        
        return {
            'corr_orig': corr_orig,
            'corr_fitted': fitted_corrs,
            'corr_diff': np.linalg.norm(corr_orig - fitted_corrs, 'fro'),
            'corr_pairs': corr_pairs,
            'vine': vine,
            'mean_log_prob': mean_log_prob
        }
    else:
        # For non-parametric, just return original correlations and log probability
        return {
            'corr_orig': corr_orig,
            'vine': vine,
            'mean_log_prob': mean_log_prob
        }

def test_vine_dependence():
    """
    Test if the vine correctly captures dependence structure,
    especially between non-adjacent variables.
    """
    global correlation_matrix
    
    # Generate data with known correlation structure - higher correlation between direct pairs
    # and lower but still significant correlation between non-adjacent pairs
    N = 2000
    dimension = 4
    
    # Define correlation matrix
    # Variables 0-1, 1-2, and 2-3 are directly connected (correlation ~0.7)
    # Variables 0-2 and 1-3 have medium correlation (~0.5)
    # Variables 0-3 have weaker but still significant correlation (~0.3)
    correlation_matrix = np.array([
        [1.0, 0.7, 0.5, 0.3],
        [0.7, 1.0, 0.7, 0.5],
        [0.5, 0.7, 1.0, 0.7],
        [0.3, 0.5, 0.7, 1.0]
    ])
    
    # Generate data
    data, actual_corr = generate_correlated_data(N, dimension, correlation_matrix)
    
    # Test different vine configurations - only use parametric for simplicity
    configs = [
        ('c-vine', True),
        ('d-vine', True)
    ]
    
    results = {}
    for vine_family, param in configs:
        print(f"\nTesting {vine_family} with parametric={param}")
        results[(vine_family, param)] = fit_and_evaluate_vine(data, vine_family, param)
        
        # Print correlation differences
        if 'corr_pairs' in results[(vine_family, param)]:
            corr_pairs = results[(vine_family, param)]['corr_pairs']
            print(f"Correlation differences (orig vs fitted params):")
            for i, j, corr_orig, corr_fitted, diff in corr_pairs:
                print(f"  Vars {i}-{j}: {corr_orig:.4f} vs {corr_fitted:.4f}, diff={diff:.4f}")
            
            overall_diff = results[(vine_family, param)]['corr_diff']
            print(f"Overall correlation matrix difference (Frobenius norm): {overall_diff:.4f}")
        
        print(f"Mean log probability on test data: {results[(vine_family, param)]['mean_log_prob']:.4f}")
    
    # Create comparison plot for key correlations
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract key correlation pairs for comparison
    key_pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    pair_names = ['0-1', '1-2', '2-3', '0-2', '1-3', '0-3']
    
    x = np.arange(len(key_pairs))
    bar_width = 0.2
    offsets = [-bar_width/2, bar_width/2]
    
    # Plot original correlations
    orig_corrs = [actual_corr[i, j] for i, j in key_pairs]
    ax.bar(x - bar_width, orig_corrs, width=bar_width, label='Original', color='black')
    
    # Plot correlations from each configuration
    colors = ['blue', 'red']
    for idx, ((vine_family, param), res) in enumerate(results.items()):
        if 'corr_fitted' in res:
            corr_fitted = res['corr_fitted']
            config_corrs = [corr_fitted[i, j] for i, j in key_pairs]
            ax.bar(x + offsets[idx], config_corrs, width=bar_width, 
                  label=f'{vine_family}', color=colors[idx])
    
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation Parameter Comparison Across Variable Pairs')
    ax.legend()
    ax.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('vine_correlation_comparison.png')
    
    return results

if __name__ == "__main__":
    print("Testing vine dependence structure preservation...")
    results = test_vine_dependence()
    print("\nTests completed. Check the generated PNG files for visualization of results.") 