import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

from DVC.objects import vine_obj_bin, margin_obj
from DVC.vine_model import sample_vine

# Create a simplified version of the original D-vine sampling that doesn't handle higher-order dependencies
def original_d_vine_sample(vine, nsamples):
    """
    Original D-vine sampling without special handling for non-adjacent variables.
    This simulates the behavior before our fixes.
    """
    d = vine.n_cop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal = torch.distributions.Normal(0., 1.)
    samples = torch.zeros((nsamples, d), dtype=torch.float32)
    
    # First variable - standard sampling
    samples[:, 0] = normal.icdf(torch.rand(nsamples))
    
    # Rest of the variables - only consider direct dependencies
    for i in range(1, d):
        lvl = i-1
        edges = vine.copulas[0]  # Only use first level copulas
        
        # Find the edge connecting variable i-1 and i
        match_idx = -1
        for j, edge in enumerate(vine.ind_vine[0]):
            if (edge[0] == i-1 and edge[1] == i) or (edge[1] == i-1 and edge[0] == i):
                match_idx = j
                break
                
        if match_idx < 0 or match_idx >= len(edges):
            # Fallback to independence
            samples[:, i] = normal.icdf(torch.rand(nsamples))
            continue
            
        # Get the copula
        cop = edges[match_idx]
        
        # Sample conditionally using only direct dependencies
        if hasattr(cop, 'family') and cop.family == "gaussian":
            # Using Gaussian conditional sampling
            root_val = samples[:, i-1]
            root_u = normal.cdf(root_val)
            rand_u = torch.rand(nsamples)
            
            rho = float(cop.theta) if cop.theta is not None else 0.0
            rho = max(min(rho, 0.999999), -0.999999)
            
            z = normal.icdf(torch.clamp(root_u, 1e-9, 1-1e-9))
            e = normal.icdf(torch.clamp(rand_u, 1e-9, 1-1e-9))
            
            denom = 1.0 - rho*rho
            if denom < 1e-12:
                denom = 1e-12
            
            y = rho*z + torch.sqrt(torch.tensor(denom))*e
            vi = normal.cdf(y)
            vi = torch.clamp(vi, 1e-9, 1-1e-9)
            
            samples[:, i] = normal.icdf(vi)
        else:
            # Fallback to independence
            samples[:, i] = normal.icdf(torch.rand(nsamples))
            
    return samples.cpu().numpy()

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

def test_correlation_preservation(dim=6, n_train=10000, n_samples=5000, rho=0.6):
    """Test correlation preservation with our enhanced sampling vs original approach"""
    results_dir = "correlation_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Testing D-vine correlation preservation for dimension: {dim}")
    print(f"{'='*80}")
    
    # Generate data
    data, true_corr = generate_gaussian_data(n_train, dim, rho)
    print(f"Generated {n_train} samples from {dim}D Gaussian with rho={rho}")
    
    # Print true correlation matrix
    print("\nTrue correlation matrix:")
    print(np.round(true_corr, 3))
    
    # Create margins
    margins = []
    for i in range(dim):
        loc, scale = np.mean(data[:, i]), np.std(data[:, i])
        margin = margin_obj('norm', [loc, scale], True)
        margins.append(margin)
    
    # Create and fit D-vine
    print("\nFitting D-vine...")
    vine = vine_obj_bin('d-vine', ['gaussian'], dim, margins, 40, 'optimal')
    
    # Prepare dictionaries for vine.fit()
    gen_dict = {'param': True, 'binning': False, 'fitted': False}
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    
    # Fit the vine
    start_time = time.time()
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, None)
    fit_time = time.time() - start_time
    print(f"Fit time: {fit_time:.2f} seconds")
    
    # Sample with our enhanced sampling
    print("\nGenerating samples with enhanced sampling...")
    start_time = time.time()
    improved_samples = vine.sample(n_samples)
    improved_time = time.time() - start_time
    improved_corr = np.corrcoef(improved_samples, rowvar=False)
    improved_error = np.mean(np.abs(improved_corr - true_corr))
    print(f"Sample time: {improved_time:.2f} seconds")
    print(f"Overall correlation error: {improved_error:.4f}")
    
    # Sample with original sampling
    print("\nGenerating samples with original sampling...")
    start_time = time.time()
    original_samples = original_d_vine_sample(vine, n_samples)
    original_time = time.time() - start_time
    original_corr = np.corrcoef(original_samples, rowvar=False)
    original_error = np.mean(np.abs(original_corr - true_corr))
    print(f"Sample time: {original_time:.2f} seconds")
    print(f"Overall correlation error: {original_error:.4f}")
    
    # Compare non-adjacent variable correlations
    print("\nNon-adjacent variable correlation comparison:")
    print(f"{'Variable Pair':<15} | {'True':<10} | {'Enhanced':<10} | {'Original':<10} | {'Enh. Error':<10} | {'Orig. Error':<10}")
    print("-" * 85)
    
    # Track errors for non-adjacent pairs
    enhanced_nonadj_errors = []
    original_nonadj_errors = []
    
    pairs = []
    true_vals = []
    enhanced_vals = []
    original_vals = []
    
    for i in range(dim):
        for j in range(i+2, dim):  # Only non-adjacent pairs
            true_val = true_corr[i, j]
            enhanced_val = improved_corr[i, j]
            original_val = original_corr[i, j]
            
            enhanced_err = abs(enhanced_val - true_val)
            original_err = abs(original_val - true_val)
            
            enhanced_nonadj_errors.append(enhanced_err)
            original_nonadj_errors.append(original_err)
            
            pairs.append(f"({i},{j})")
            true_vals.append(true_val)
            enhanced_vals.append(enhanced_val)
            original_vals.append(original_val)
            
            print(f"{f'({i},{j})':<15} | {true_val:<10.4f} | {enhanced_val:<10.4f} | {original_val:<10.4f} | {enhanced_err:<10.4f} | {original_err:<10.4f}")
    
    # Calculate average error for non-adjacent pairs
    enhanced_nonadj_avg = np.mean(enhanced_nonadj_errors)
    original_nonadj_avg = np.mean(original_nonadj_errors)
    
    print(f"\nAverage non-adjacent correlation error:")
    print(f"  Enhanced sampling: {enhanced_nonadj_avg:.4f}")
    print(f"  Original sampling: {original_nonadj_avg:.4f}")
    print(f"  Improvement: {(original_nonadj_avg - enhanced_nonadj_avg) / original_nonadj_avg * 100:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    x = np.arange(len(pairs))
    width = 0.2
    
    plt.bar(x - width, true_vals, width, label="True")
    plt.bar(x, enhanced_vals, width, label="Enhanced Sampling")
    plt.bar(x + width, original_vals, width, label="Original Sampling")
    
    plt.xlabel("Variable Pair")
    plt.ylabel("Correlation")
    plt.title(f"{dim}D Gaussian - Non-Adjacent Variable Correlations")
    plt.xticks(x, pairs, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"correlation_comparison_{dim}d.png"))
    
    # Also plot the error comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, enhanced_nonadj_errors, width, label="Enhanced Sampling")
    plt.bar(x + width/2, original_nonadj_errors, width, label="Original Sampling")
    
    plt.xlabel("Variable Pair")
    plt.ylabel("Absolute Error")
    plt.title(f"{dim}D Gaussian - Correlation Error Comparison")
    plt.xticks(x, pairs, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"error_comparison_{dim}d.png"))
    
    print(f"\nPlots saved in {results_dir}/")
    
    return {
        'true_corr': true_corr,
        'enhanced': {
            'samples': improved_samples,
            'corr': improved_corr,
            'error': improved_error,
            'time': improved_time,
            'nonadj_error': enhanced_nonadj_avg
        },
        'original': {
            'samples': original_samples,
            'corr': original_corr,
            'error': original_error,
            'time': original_time,
            'nonadj_error': original_nonadj_avg
        }
    }

if __name__ == "__main__":
    # Run the test
    results = test_correlation_preservation(dim=6, n_train=10000, n_samples=5000) 