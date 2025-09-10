#!/usr/bin/env python3
"""
Time-dependent vine copula example.

This example demonstrates:
1. Generating time-series data with varying correlations
2. Analyzing temporal correlation dynamics
3. Using normalizing flows for time-dependent modeling
4. Evaluating model performance over time
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.time.data import generate_synthetic_time_series, compute_time_varying_correlations
from dvc_package.time.model import TimeDependentVineCopula
from dvc_package.core.vine_factory import create_vine


def analyze_temporal_correlations(time_data, time_indices):
    """Analyze how correlations change over time."""
    print("Analyzing temporal correlation dynamics...")
    
    correlations = compute_time_varying_correlations(time_data)
    n_time_steps, n_vars, _ = correlations.shape
    
    # Print correlation statistics
    print(f"  Time steps: {n_time_steps}")
    print(f"  Variables: {n_vars}")
    
    # Analyze correlation evolution for each pair
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pair_corrs = correlations[:, i, j]
            print(f"  Correlation {i}-{j}: mean={np.mean(pair_corrs):.3f}, "
                  f"std={np.std(pair_corrs):.3f}, range=[{np.min(pair_corrs):.3f}, {np.max(pair_corrs):.3f}]")
    
    return correlations


def test_time_dependent_modeling(time_data, time_indices):
    """Test time-dependent vine copula modeling."""
    print("\nTesting time-dependent modeling...")
    
    n_time_steps, n_samples_per_time, n_variables = time_data.shape
    
    try:
        # Create base vine
        base_vine = create_vine('c-vine', n_variables)
        print(f"  Created base vine with {n_variables} variables")
        
        # Create time-dependent model
        time_model = TimeDependentVineCopula(
            vine=base_vine,
            n_time_steps=n_time_steps,
            hidden_dim=32  # Smaller for example
        )
        print(f"  Created time-dependent model with {len(time_model.edge_flows)} edge flows")
        
        # Test bandwidth prediction
        time_norm = torch.linspace(0, 1, n_time_steps).unsqueeze(-1)
        bandwidths = time_model.forward(time_norm)
        print(f"  Bandwidth predictions shape: {bandwidths.shape}")
        print(f"  Bandwidth range: [{bandwidths.min():.4f}, {bandwidths.max():.4f}]")
        
        # Test loss computation on small sample
        sample_data = torch.tensor(time_data[0][:10], dtype=torch.float32)
        sample_times = torch.zeros(10)
        
        nll_loss = time_model.negative_log_likelihood(sample_data, sample_times)
        entropy_loss = time_model.entropy_based_loss(sample_data, sample_times)
        
        print(f"  Sample NLL loss: {nll_loss:.4f}")
        print(f"  Sample entropy loss: {entropy_loss:.4f}")
        
        return time_model, bandwidths
        
    except Exception as e:
        print(f"  Error in time-dependent modeling: {e}")
        return None, None


def create_time_plots(time_data, time_indices, correlations, bandwidths=None):
    """Create time-dependent analysis plots."""
    print("\nCreating time-dependent plots...")
    
    output_dir = Path("results/time_dependent_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_time_steps, n_samples_per_time, n_variables = time_data.shape
    
    # Plot correlation evolution
    if n_variables >= 2:
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_variables * (n_variables - 1) // 2))
        color_idx = 0
        
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                pair_corrs = correlations[:, i, j]
                plt.plot(time_indices, pair_corrs, 
                        label=f'Variables {i}-{j}', color=colors[color_idx], linewidth=2)
                color_idx += 1
        
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        plt.title('Time-Varying Correlations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Correlation evolution plot saved")
    
    # Plot bandwidth evolution if available
    if bandwidths is not None:
        plt.figure(figsize=(12, 6))
        
        # Plot first few bandwidth dimensions
        n_dims_to_plot = min(4, bandwidths.shape[1])
        for dim in range(n_dims_to_plot):
            plt.plot(time_indices, bandwidths[:, dim].detach().cpu().numpy(), 
                    label=f'Bandwidth Dim {dim}', linewidth=2)
        
        plt.xlabel('Time')
        plt.ylabel('Bandwidth')
        plt.title('Bandwidth Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "bandwidth_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Bandwidth evolution plot saved")
    
    # Plot data snapshots at different time points
    sample_times = np.linspace(0, n_time_steps - 1, 4).astype(int)
    
    if n_variables >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        for i, t in enumerate(sample_times):
            data_t = time_data[t]
            axes[i].scatter(data_t[:, 0], data_t[:, 1], alpha=0.6, s=1)
            axes[i].set_title(f'Time Step {t}')
            axes[i].set_xlabel('Variable 0')
            axes[i].set_ylabel('Variable 1')
        
        plt.tight_layout()
        plt.savefig(output_dir / "data_snapshots.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Data snapshots plot saved")
    
    print(f"  All plots saved to: {output_dir}")


def main():
    """Main function."""
    print("DVC Time-Dependent Vine Copula Example")
    print("=" * 45)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate time-dependent data
    print("Generating time-dependent synthetic data...")
    time_data, time_indices = generate_synthetic_time_series(
        n_time_steps=50,
        n_samples=100,
        n_variables=3,
        correlation_evolution='sinusoidal',
        seed=42
    )
    
    print(f"Generated data shape: {time_data.shape}")
    print(f"Time range: [{time_indices.min():.1f}, {time_indices.max():.1f}]")
    
    # Analyze temporal correlations
    correlations = analyze_temporal_correlations(time_data, time_indices)
    
    # Test time-dependent modeling
    time_model, bandwidths = test_time_dependent_modeling(time_data, time_indices)
    
    # Create plots
    create_time_plots(time_data, time_indices, correlations, bandwidths)
    
    print("\nTime-dependent example completed!")
    print("Check results/time_dependent_example/ for output plots.")


if __name__ == "__main__":
    main()