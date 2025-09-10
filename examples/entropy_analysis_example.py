#!/usr/bin/env python3
"""
Entropy and information analysis example.

This example demonstrates:
1. Different entropy estimation methods
2. Mutual information computation
3. Information decomposition analysis
4. Comparison across different data types
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, kendalltau
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_entropy_test_datasets(n_samples=1000):
    """Generate datasets with different entropy characteristics."""
    datasets = {}
    
    # Low entropy: highly correlated
    corr_high = np.array([[1.0, 0.9], [0.9, 1.0]])
    datasets['low_entropy'] = np.random.multivariate_normal([0, 0], corr_high, n_samples)
    
    # Medium entropy: moderate correlation
    corr_med = np.array([[1.0, 0.5], [0.5, 1.0]])
    datasets['medium_entropy'] = np.random.multivariate_normal([0, 0], corr_med, n_samples)
    
    # High entropy: independent
    datasets['high_entropy'] = np.random.multivariate_normal([0, 0], np.eye(2), n_samples)
    
    # Non-Gaussian: mixture distribution
    mixture_data = np.vstack([
        np.random.multivariate_normal([-2, -2], np.eye(2) * 0.5, n_samples // 2),
        np.random.multivariate_normal([2, 2], np.eye(2) * 0.5, n_samples // 2)
    ])
    datasets['mixture'] = mixture_data
    
    return datasets


def gaussian_entropy_estimate(data):
    """Estimate entropy assuming Gaussian distribution."""
    cov = np.cov(data.T)
    det_cov = np.linalg.det(cov)
    if det_cov <= 0:
        return float('inf')
    
    d = data.shape[1]
    entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.log(det_cov)
    return entropy


def kernel_entropy_estimate(data):
    """Estimate entropy using kernel density estimation."""
    n_samples, d = data.shape
    
    # Adaptive bandwidth selection
    distances = pdist(data)
    h = np.median(distances) / 2
    
    # Compute kernel density at each point
    dist_matrix = squareform(distances)
    kernel_vals = np.exp(-dist_matrix**2 / (2 * h**2))
    densities = np.sum(kernel_vals, axis=1) / (n_samples * (2 * np.pi * h**2)**(d/2))
    
    # Entropy estimate
    log_densities = np.log(densities + 1e-10)
    entropy = -np.mean(log_densities)
    
    return entropy


def knn_entropy_estimate(data, k=5):
    """Estimate entropy using k-nearest neighbors."""
    n_samples, d = data.shape
    
    # Build KD-tree
    tree = cKDTree(data)
    
    # Find k-th nearest neighbor distances
    distances, _ = tree.query(data, k=k+1)  # +1 because first neighbor is the point itself
    knn_distances = distances[:, k]  # k-th nearest neighbor
    
    # Entropy estimate using Kozachenko-Leonenko estimator
    log_distances = np.log(knn_distances + 1e-10)
    entropy = d * np.mean(log_distances) + np.log(n_samples) + np.log(2) * d - np.log(k)
    
    return entropy


def estimate_mutual_information(data):
    """Estimate mutual information between variables."""
    if data.shape[1] < 2:
        return {}
    
    results = {}
    n_vars = data.shape[1]
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Gaussian MI estimate
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
            mi_gaussian = -0.5 * np.log(1 - corr**2 + 1e-10)
            
            # KNN-based MI estimate (simplified)
            joint_entropy = knn_entropy_estimate(data[:, [i, j]])
            marginal_i = knn_entropy_estimate(data[:, [i]])
            marginal_j = knn_entropy_estimate(data[:, [j]])
            
            mi_knn = marginal_i + marginal_j - joint_entropy
            
            results[f'mi_{i}_{j}'] = {
                'gaussian': mi_gaussian,
                'knn': mi_knn,
                'correlation': corr
            }
    
    return results


def compare_entropy_methods(datasets):
    """Compare different entropy estimation methods."""
    print("\nComparing entropy estimation methods...")
    
    methods = {
        'Gaussian': gaussian_entropy_estimate,
        'Kernel': kernel_entropy_estimate,
        'KNN': knn_entropy_estimate
    }
    
    results = {}
    
    for dataset_name, data in datasets.items():
        print(f"\nDataset: {dataset_name}")
        results[dataset_name] = {}
        
        for method_name, method_func in methods.items():
            try:
                entropy = method_func(data)
                results[dataset_name][method_name] = entropy
                print(f"  {method_name:<8}: {entropy:.4f}")
            except Exception as e:
                print(f"  {method_name:<8}: Error - {e}")
                results[dataset_name][method_name] = float('nan')
        
        # Compute mutual information
        mi_results = estimate_mutual_information(data)
        if mi_results:
            print(f"  Mutual Information:")
            for pair, mi_data in mi_results.items():
                print(f"    {pair}: Gaussian={mi_data['gaussian']:.4f}, KNN={mi_data['knn']:.4f}")
    
    return results


def create_entropy_plots(datasets, entropy_results):
    """Create entropy analysis plots."""
    print("\nCreating entropy analysis plots...")
    
    output_dir = Path("results/entropy_analysis_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot datasets
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(datasets.items()):
        if i < 4:  # Only plot first 4 datasets
            axes[i].scatter(data[:, 0], data[:, 1], alpha=0.6, s=1)
            axes[i].set_title(f'{name.replace("_", " ").title()}')
            axes[i].set_xlabel('Variable 1')
            axes[i].set_ylabel('Variable 2')
    
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_datasets.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot entropy comparison
    methods = ['Gaussian', 'Kernel', 'KNN']
    datasets_list = list(datasets.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(datasets_list))
    width = 0.25
    
    for i, method in enumerate(methods):
        values = []
        for dataset in datasets_list:
            value = entropy_results[dataset].get(method, float('nan'))
            values.append(value if np.isfinite(value) else 0)
        
        ax.bar(x + i * width, values, width, label=method, alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Estimation Method Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([name.replace('_', ' ').title() for name in datasets_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Plots saved to: {output_dir}")


def main():
    """Main function."""
    print("DVC Entropy Analysis Example")
    print("=" * 35)
    
    # Set random seed
    np.random.seed(123)
    
    # Generate test datasets
    print("Generating test datasets with different entropy characteristics...")
    datasets = generate_entropy_test_datasets(n_samples=1500)
    
    for name, data in datasets.items():
        print(f"  {name}: shape {data.shape}, mean correlation = {np.corrcoef(data.T)[0,1]:.3f}")
    
    # Compare entropy estimation methods
    entropy_results = compare_entropy_methods(datasets)
    
    # Create visualizations
    create_entropy_plots(datasets, entropy_results)
    
    # Print summary
    print("\nEntropy Analysis Summary:")
    print("-" * 30)
    
    for dataset_name, results in entropy_results.items():
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        for method, entropy in results.items():
            if np.isfinite(entropy):
                print(f"  {method}: {entropy:.4f}")
            else:
                print(f"  {method}: Failed")
    
    print("\nEntropy analysis example completed!")
    print("Check results/entropy_analysis_example/ for output plots.")


if __name__ == "__main__":
    main()
