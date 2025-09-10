import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns

from DVC_pyolder.objects import vine_obj_bin, margin_obj
from DVC_pyolder.d_vine_fix import fix_dvine_uniform_correlations

def generate_gaussian_data(n_samples, dim, rho=0.6, seed=42):
    """Generate samples from a multivariate Gaussian with uniform correlation rho"""
    np.random.seed(seed)
    
    # Create correlation matrix with uniform correlation
    cov = np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)
    
    # Generate samples
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    # Calculate empirical correlation matrix
    empirical_corr = np.corrcoef(data, rowvar=False)
    
    return data, cov, empirical_corr

def fit_vine_and_sample(data, dim, n_samples, fix_correlations=False, target_corr=0.6):
    """Fit a D-vine model to the data and generate samples"""
    # Create margins
    margins = []
    for i in range(dim):
        loc, scale = np.mean(data[:, i]), np.std(data[:, i])
        margin = margin_obj('norm', [loc, scale], True)
        margins.append(margin)
    
    # Create and fit D-vine
    vine = vine_obj_bin('d-vine', ['gaussian'], dim, margins, 40, 'optimal')
    
    # Prepare dictionaries for vine.fit()
    gen_dict = {'param': True, 'binning': False, 'fitted': False}
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    
    # Fit the vine
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, None)
    
    # Print the vine structure and copulas
    print("\nVine structure after fitting:")
    for level in range(len(vine.ind_vine)):
        print(f"Level {level}, edges: {vine.ind_vine[level]}, #copulas stored: {len(vine.copulas[level])}")
        
        print(f"\nLevel {level} correlations (original):")
        for e_idx, edge in enumerate(vine.ind_vine[level]):
            if e_idx < len(vine.copulas[level]):
                cop = vine.copulas[level][e_idx]
                if hasattr(cop, 'family') and cop.family == "gaussian" and hasattr(cop, 'theta'):
                    rho = float(cop.theta) if cop.theta is not None else 0.0
                    print(f"  Edge {edge}: rho = {rho:.4f}")
    
    # Apply correlation fix if requested
    if fix_correlations:
        print("\nApplying correlation fix to D-vine...")
        vine = fix_dvine_uniform_correlations(vine, target_corr)
        
        print("\nVine structure after correlation fix:")
        for level in range(len(vine.ind_vine)):
            print(f"\nLevel {level} correlations (fixed):")
            for e_idx, edge in enumerate(vine.ind_vine[level]):
                if e_idx < len(vine.copulas[level]):
                    cop = vine.copulas[level][e_idx]
                    if hasattr(cop, 'family') and cop.family == "gaussian" and hasattr(cop, 'theta'):
                        rho = float(cop.theta) if cop.theta is not None else 0.0
                        print(f"  Edge {edge}: rho = {rho:.4f}")
    
    # Sample from the vine
    samples = vine.sample(n_samples)
    
    # Calculate empirical correlation
    empirical_corr = np.corrcoef(samples, rowvar=False)
    
    return vine, samples, empirical_corr

def compare_correlations(true_corr, original_corr, fixed_corr, results_dir="fixed_dvine_results"):
    """Compare and plot correlation matrices"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate errors
    error_original = np.abs(true_corr - original_corr)
    error_fixed = np.abs(true_corr - fixed_corr)
    
    # Calculate mean errors
    mae_original = np.mean(error_original)
    mae_fixed = np.mean(error_fixed)
    
    print("\nCorrelation error comparison:")
    print(f"Original D-vine mean absolute error: {mae_original:.4f}")
    print(f"Fixed D-vine mean absolute error: {mae_fixed:.4f}")
    print(f"Improvement: {100 * (mae_original - mae_fixed) / mae_original:.2f}%")
    
    # Non-adjacent correlations
    dim = true_corr.shape[0]
    non_adj_true = []
    non_adj_orig = []
    non_adj_fixed = []
    
    print("\nNon-adjacent variable correlation comparison:")
    print(f"{'Variable Pair':<15} | {'True':<10} | {'Original':<10} | {'Fixed':<10} | {'Orig. Error':<10} | {'Fixed Error':<10}")
    print("-" * 85)
    
    for i in range(dim):
        for j in range(i+2, dim):
            true_val = true_corr[i, j]
            orig_val = original_corr[i, j]
            fixed_val = fixed_corr[i, j]
            
            orig_err = abs(orig_val - true_val)
            fixed_err = abs(fixed_val - true_val)
            
            non_adj_true.append(true_val)
            non_adj_orig.append(orig_val)
            non_adj_fixed.append(fixed_val)
            
            print(f"{f'({i},{j})':<15} | {true_val:<10.4f} | {orig_val:<10.4f} | {fixed_val:<10.4f} | {orig_err:<10.4f} | {fixed_err:<10.4f}")
    
    # Overall statistics for non-adjacent correlations
    non_adj_orig_error = np.mean(np.abs(np.array(non_adj_true) - np.array(non_adj_orig)))
    non_adj_fixed_error = np.mean(np.abs(np.array(non_adj_true) - np.array(non_adj_fixed)))
    
    print(f"\nAverage non-adjacent correlation error:")
    print(f"  Original: {non_adj_orig_error:.4f}")
    print(f"  Fixed: {non_adj_fixed_error:.4f}")
    print(f"  Improvement: {100 * (non_adj_orig_error - non_adj_fixed_error) / non_adj_orig_error:.2f}%")
    
    # Plot correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(true_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("True Correlation")
    
    sns.heatmap(original_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Original D-vine Samples")
    
    sns.heatmap(fixed_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title("Fixed D-vine Samples")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "correlation_comparison.png"))
    plt.close()
    
    # Plot errors
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(error_original, annot=True, fmt=".3f", cmap="Reds", ax=axes[0])
    axes[0].set_title(f"Original D-vine Error (MAE: {mae_original:.4f})")
    
    sns.heatmap(error_fixed, annot=True, fmt=".3f", cmap="Reds", ax=axes[1])
    axes[1].set_title(f"Fixed D-vine Error (MAE: {mae_fixed:.4f})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_comparison.png"))
    plt.close()
    
    # Plot non-adjacent correlations as bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(non_adj_true))
    width = 0.25
    
    plt.bar(x - width, non_adj_true, width, label="True")
    plt.bar(x, non_adj_orig, width, label="Original D-vine")
    plt.bar(x + width, non_adj_fixed, width, label="Fixed D-vine")
    
    plt.xlabel("Variable Pair")
    plt.ylabel("Correlation")
    plt.title("Non-Adjacent Variable Correlations")
    
    # Create custom x-ticks with variable pairs
    pairs = [f"({i},{j})" for i in range(dim) for j in range(i+2, dim)]
    plt.xticks(x, pairs, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_dir, "non_adjacent_correlations.png"))
    plt.close()

if __name__ == "__main__":
    # Parameters
    dim = 6
    rho = 0.6
    n_train = 10000
    n_samples = 5000
    
    print(f"\n{'='*80}")
    print(f"Testing fixed D-vine implementation for {dim}D Gaussian with rho={rho}")
    print(f"{'='*80}")
    
    # Step 1: Generate data from multivariate Gaussian
    data, true_corr, data_corr = generate_gaussian_data(n_train, dim, rho)
    print(f"Generated {n_train} samples from {dim}D Gaussian with rho={rho}")
    
    # Step 2: Fit and sample original D-vine
    print("\nFitting and sampling from original D-vine...")
    _, original_samples, original_corr = fit_vine_and_sample(data, dim, n_samples, fix_correlations=False)
    
    # Step 3: Fit and sample fixed D-vine
    print("\nFitting and sampling from fixed D-vine...")
    _, fixed_samples, fixed_corr = fit_vine_and_sample(data, dim, n_samples, fix_correlations=True, target_corr=rho)
    
    # Step 4: Compare results
    compare_correlations(true_corr, original_corr, fixed_corr)
    
    print("\nTest completed. Results saved in the 'fixed_dvine_results' directory.") 