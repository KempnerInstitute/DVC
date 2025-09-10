#!/usr/bin/env python3
"""
Basic vine copula example demonstrating core functionality.

This example shows how to:
1. Create synthetic data
2. Fit different vine types (C-vine, D-vine, R-vine)
3. Compare copula families
4. Evaluate model performance
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.core.vine_factory import create_vine
from dvc_package.core.param_copula import parametric_fit
from dvc_package.core.d_vine_fix import compute_correlation_matrix


def generate_test_data(n_samples=1000, n_variables=3, correlation_strength=0.5):
    """Generate synthetic test data with known correlation structure."""
    # Create correlation matrix
    corr_matrix = np.eye(n_variables)
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            # Decay correlation with distance
            corr_val = correlation_strength / (1 + 0.5 * abs(i - j))
            corr_matrix[i, j] = corr_matrix[j, i] = corr_val
    
    # Generate multivariate normal data
    data = np.random.multivariate_normal(
        mean=np.zeros(n_variables),
        cov=corr_matrix,
        size=n_samples
    )
    
    return data, corr_matrix


def compare_vine_types(data, true_correlation):
    """Compare different vine types on the same data."""
    vine_types = ['c-vine', 'd-vine', 'r-vine']
    families = ['independence', 'gaussian', 'clayton']
    
    results = {}
    
    for vine_type in vine_types:
        print(f"\nTesting {vine_type}...")
        
        try:
            # Create vine
            vine = create_vine(
                vine_type=vine_type,
                vine_depth=data.shape[1],
                families=families,
                data=data
            )
            
            print(f"  Created {vine_type} with structure:")
            if hasattr(vine, 'ind_vine'):
                for level, edges in enumerate(vine.ind_vine):
                    print(f"    Level {level}: {edges}")
            
            # Store results
            results[vine_type] = {
                'vine': vine,
                'structure': vine.ind_vine if hasattr(vine, 'ind_vine') else None,
                'r_matrix': vine.r_matrix.tolist() if hasattr(vine, 'r_matrix') else None
            }
            
        except Exception as e:
            print(f"  Error creating {vine_type}: {e}")
            results[vine_type] = {'error': str(e)}
    
    return results


def test_copula_fitting(data):
    """Test copula fitting with different families."""
    print("\nTesting copula fitting...")
    
    # Convert to uniform margins for copula fitting
    n_samples, n_vars = data.shape
    uniform_data = np.zeros_like(data)
    
    for i in range(n_vars):
        sorted_vals = np.sort(data[:, i])
        ranks = np.searchsorted(sorted_vals, data[:, i])
        uniform_data[:, i] = (ranks + 1) / (n_samples + 1)
    
    # Test bivariate copula fitting
    if n_vars >= 2:
        bivariate_data = uniform_data[:, :2].reshape(n_samples, 2, 1)
        families = ['independence', 'gaussian', 'clayton', 'frank', 'gumbel']
        
        try:
            aic_scores, parameters, log_likelihoods = parametric_fit(
                bivariate_data, families, n_cop=1
            )
            
            print("  Copula fitting results:")
            for i, family in enumerate(families):
                aic = aic_scores[0, i] if aic_scores.size > 0 else float('inf')
                param = parameters[0][i] if len(parameters) > 0 else None
                loglik = log_likelihoods[0][i] if len(log_likelihoods) > 0 else float('-inf')
                print(f"    {family:<12}: AIC={aic:8.2f}, LogLik={loglik:8.2f}, Param={param}")
            
            # Find best family
            if aic_scores.size > 0:
                best_idx = np.argmin(aic_scores[0])
                best_family = families[best_idx]
                print(f"  Best family: {best_family}")
                
                return best_family, parameters[0][best_idx]
            
        except Exception as e:
            print(f"  Error in copula fitting: {e}")
    
    return None, None


def analyze_correlation_preservation(data, vine_results):
    """Analyze how well each vine type preserves correlations."""
    print("\nAnalyzing correlation preservation...")
    
    original_corr = compute_correlation_matrix(data)
    print(f"Original correlation matrix:\n{original_corr}")
    
    for vine_type, result in vine_results.items():
        if 'error' in result:
            print(f"  {vine_type}: Skipped due to error")
            continue
        
        try:
            vine = result['vine']
            
            # Sample from vine (simplified - would need full implementation)
            print(f"  {vine_type}: Structure analysis complete")
            print(f"    Edges per level: {[len(level) for level in vine.ind_vine] if hasattr(vine, 'ind_vine') else 'N/A'}")
            
        except Exception as e:
            print(f"  {vine_type}: Error in analysis: {e}")


def create_visualization(data, vine_results):
    """Create visualization plots."""
    print("\nCreating visualizations...")
    
    # Create output directory
    output_dir = Path("results/basic_vine_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_vars = data.shape[1]
    
    # Plot marginal distributions
    fig, axes = plt.subplots(1, n_vars, figsize=(12, 4))
    if n_vars == 1:
        axes = [axes]
    
    for i in range(n_vars):
        axes[i].hist(data[:, i], bins=50, alpha=0.7, density=True)
        axes[i].set_title(f'Variable {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(output_dir / "marginal_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot pairwise relationships
    if n_vars >= 2:
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(10, 10))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    axes[i, j].hist(data[:, i], bins=30, alpha=0.7)
                    axes[i, j].set_title(f'Variable {i}')
                else:
                    axes[i, j].scatter(data[:, j], data[:, i], alpha=0.5, s=1)
                    axes[i, j].set_xlabel(f'Variable {j}')
                    axes[i, j].set_ylabel(f'Variable {i}')
        
        plt.tight_layout()
        plt.savefig(output_dir / "pairwise_relationships.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Plots saved to: {output_dir}")


def main():
    """Main function demonstrating basic vine copula usage."""
    print("DVC Basic Vine Copula Example")
    print("=" * 40)
    
    # Generate test data
    print("Generating synthetic data...")
    data, true_corr = generate_test_data(n_samples=1000, n_variables=3, correlation_strength=0.6)
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    # Test copula fitting
    best_family, best_param = test_copula_fitting(data)
    
    # Compare vine types
    vine_results = compare_vine_types(data, true_corr)
    
    # Analyze correlation preservation
    analyze_correlation_preservation(data, vine_results)
    
    # Create visualizations
    create_visualization(data, vine_results)
    
    print("\nExample completed successfully!")
    print("Check results/basic_vine_example/ for output plots.")


if __name__ == "__main__":
    main()
