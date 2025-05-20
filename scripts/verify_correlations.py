import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns

from DVC.objects import vine_obj_bin, margin_obj

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

def fit_vine_and_sample(data, dim, n_samples):
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
    
    # Sample from the vine
    samples = vine.sample(n_samples)
    
    # Calculate empirical correlation
    empirical_corr = np.corrcoef(samples, rowvar=False)
    
    return vine, samples, empirical_corr

def print_correlation_matrices(true_corr, data_corr, samples_corr, dim):
    """Print and compare correlation matrices"""
    print("\nTheoretical correlation matrix:")
    print(np.round(true_corr, 3))
    
    print("\nEmpirical correlation matrix from raw data:")
    print(np.round(data_corr, 3))
    
    print("\nEmpirical correlation matrix from vine samples:")
    print(np.round(samples_corr, 3))
    
    print("\nError: true vs data correlation:")
    print(np.round(np.abs(true_corr - data_corr), 3))
    
    print("\nError: true vs vine samples correlation:")
    print(np.round(np.abs(true_corr - samples_corr), 3))
    
    # Calculate mean absolute errors
    mae_data = np.mean(np.abs(true_corr - data_corr))
    mae_vine = np.mean(np.abs(true_corr - samples_corr))
    
    print(f"\nMean absolute error for data correlation: {mae_data:.4f}")
    print(f"Mean absolute error for vine samples correlation: {mae_vine:.4f}")
    
    # Analyze non-adjacent correlations specifically
    non_adj_true = []
    non_adj_data = []
    non_adj_vine = []
    
    print("\nNon-adjacent variable correlation comparison:")
    print(f"{'Variable Pair':<15} | {'True':<10} | {'Data':<10} | {'Vine':<10} | {'Data Error':<10} | {'Vine Error':<10}")
    print("-" * 85)
    
    for i in range(dim):
        for j in range(i+2, dim):
            true_val = true_corr[i, j]
            data_val = data_corr[i, j]
            vine_val = samples_corr[i, j]
            
            data_err = abs(data_val - true_val)
            vine_err = abs(vine_val - true_val)
            
            non_adj_true.append(true_val)
            non_adj_data.append(data_val)
            non_adj_vine.append(vine_val)
            
            print(f"{f'({i},{j})':<15} | {true_val:<10.4f} | {data_val:<10.4f} | {vine_val:<10.4f} | {data_err:<10.4f} | {vine_err:<10.4f}")
    
    # Overall statistics for non-adjacent correlations
    non_adj_data_error = np.mean(np.abs(np.array(non_adj_true) - np.array(non_adj_data)))
    non_adj_vine_error = np.mean(np.abs(np.array(non_adj_true) - np.array(non_adj_vine)))
    
    print(f"\nAverage non-adjacent correlation error:")
    print(f"  Data: {non_adj_data_error:.4f}")
    print(f"  Vine: {non_adj_vine_error:.4f}")
    
    # Extract and display actual correlations from vine copulas
    # This shows what correlations the vine structure is actually encoding
    print("\nExtracting correlations from vine structure:")
    
    # First level correlations (direct)
    print("\nFirst level correlations (direct):")
    for i, edge in enumerate(vine.ind_vine[0]):
        if i < len(vine.copulas[0]):
            cop = vine.copulas[0][i]
            if hasattr(cop, 'family') and cop.family == "gaussian" and hasattr(cop, 'theta'):
                rho = float(cop.theta) if cop.theta is not None else 0.0
                print(f"Edge {edge}: rho = {rho:.4f}")
    
    # Higher level correlations (non-adjacent)
    for level in range(1, len(vine.ind_vine)):
        print(f"\nLevel {level} correlations (distance {level+1}):")
        for i, edge in enumerate(vine.ind_vine[level]):
            if i < len(vine.copulas[level]):
                cop = vine.copulas[level][i]
                if hasattr(cop, 'family') and cop.family == "gaussian" and hasattr(cop, 'theta'):
                    rho = float(cop.theta) if cop.theta is not None else 0.0
                    print(f"Edge {edge}: rho = {rho:.4f} (true corr between these vars: {true_corr[edge[0], edge[1]]:.4f})")

def plot_correlation_matrices(true_corr, data_corr, samples_corr, results_dir="correlation_verification"):
    """Plot correlation matrices for visual comparison"""
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot theoretical correlation
    sns.heatmap(true_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Theoretical Correlation")
    
    # Plot empirical correlation from data
    sns.heatmap(data_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Empirical Correlation (Data)")
    
    # Plot empirical correlation from vine samples
    sns.heatmap(samples_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title("Empirical Correlation (Vine Samples)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "correlation_comparison.png"))
    plt.close()
    
    # Plot errors
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot data error
    sns.heatmap(np.abs(true_corr - data_corr), annot=True, fmt=".3f", cmap="Reds", ax=axes[0])
    axes[0].set_title("Error: True vs Data Correlation")
    
    # Plot vine error
    sns.heatmap(np.abs(true_corr - samples_corr), annot=True, fmt=".3f", cmap="Reds", ax=axes[1])
    axes[1].set_title("Error: True vs Vine Samples Correlation")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "error_comparison.png"))
    plt.close()

if __name__ == "__main__":
    # Parameters
    dim = 6
    rho = 0.6
    n_train = 10000
    n_samples = 5000
    
    print(f"\n{'='*80}")
    print(f"Verifying correlations for {dim}D Gaussian with rho={rho}")
    print(f"{'='*80}")
    
    # Step 1: Generate data from multivariate Gaussian
    data, true_corr, data_corr = generate_gaussian_data(n_train, dim, rho)
    print(f"Generated {n_train} samples from {dim}D Gaussian with rho={rho}")
    
    # Step 2: Fit vine model and generate samples
    print("\nFitting D-vine and generating samples...")
    vine, samples, samples_corr = fit_vine_and_sample(data, dim, n_samples)
    
    # Step 3: Print and compare correlation matrices
    print_correlation_matrices(true_corr, data_corr, samples_corr, dim)
    
    # Step 4: Plot correlation matrices
    plot_correlation_matrices(true_corr, data_corr, samples_corr)
    
    print(f"\nVerification completed. Plots saved in the 'correlation_verification' directory.") 