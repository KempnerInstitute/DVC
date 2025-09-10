#!/usr/bin/env python3
"""
Beyond-Pairwise Interactions Demo

This script demonstrates the advanced capabilities of DVC-NF for modeling
beyond-pairwise (triple) interactions that switch dynamically over time.

Features demonstrated:
- Pairwise correlations that switch between regimes
- Triple interactions: X[k] += strength * X[i] * X[j]
- Testing vine copula's ability to capture higher-order dependencies
- Comparison with simple baseline methods

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for package imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def run_beyond_pairwise_demo():
    """Run comprehensive beyond-pairwise interactions demonstration"""
    
    print("🔗 BEYOND-PAIRWISE INTERACTIONS DEMO")
    print("=" * 80)
    
    from dvc_nf.data.generators import TimeDependentDataGenerator
    from dvc_nf.analysis.comprehensive import ComprehensiveTimeDependentAnalysis
    
    # Configuration for beyond-pairwise analysis
    config = {
        'dim': 4,                    # 4D to show multiple triple interactions
        'n_time_steps': 60,          # Moderate time series length
        'n_samples_per_time': 100,   # Sufficient samples for correlation estimation
        'switch_times': [0.3, 0.7], # Two switch points creating 3 regimes
        'triple_strength': 0.4,     # Strong triple interaction effects
        'corr_range': (0.2, 0.8)    # Wide correlation range
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize data generator
    print(f"\n📊 Generating beyond-pairwise interactions data...")
    generator = TimeDependentDataGenerator(dim=config['dim'], random_seed=42)
    
    # Generate sophisticated beyond-pairwise data
    data, times, metadata = generator.generate_beyond_pairwise_interactions(
        n_time_steps=config['n_time_steps'],
        n_samples_per_time=config['n_samples_per_time'],
        switch_times=config['switch_times'],
        corr_low=config['corr_range'][0],
        corr_high=config['corr_range'][1],
        beyond_pairwise_strength=config['triple_strength']
    )
    
    print(f"✅ Generated data shape: {data.shape}")
    print(f"✅ Number of regimes: {metadata['n_regimes']}")
    print(f"✅ Switch times: {metadata['switch_indices']}")
    print(f"✅ Triple interaction strength: {metadata['beyond_pairwise_strength']}")
    print(f"✅ Mean triple effect: {np.mean(metadata['triple_interaction_evolution']):.4f}")
    
    # Visualize the generated data
    print(f"\n📈 Creating data visualizations...")
    generator.visualize_generated_data('beyond_pairwise', save_plots=True)
    
    # Analyze empirical correlations over time
    print(f"\n📊 Analyzing empirical correlations and triple effects...")
    empirical_analysis = _analyze_empirical_correlations(data, metadata)
    
    # Run comprehensive analysis
    print(f"\n🔮 Running comprehensive analysis with baselines...")
    analyzer = ComprehensiveTimeDependentAnalysis(
        dim=config['dim'], 
        random_seed=42
    )
    
    try:
        analyzer.run_complete_analysis(
            n_time_steps=config['n_time_steps'],
            n_samples_per_time=config['n_samples_per_time'],
            test_scenarios=['beyond_pairwise']
        )
        
        print(f"✅ Comprehensive analysis completed!")
        
        # Create specialized visualizations
        _create_beyond_pairwise_analysis_plots(data, metadata, empirical_analysis)
        
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
        print("Proceeding with empirical analysis only...")
        _create_beyond_pairwise_analysis_plots(data, metadata, empirical_analysis)

def _analyze_empirical_correlations(data, metadata):
    """Analyze empirical correlations and detect triple interaction effects"""
    
    n_time_steps, n_samples, dim = data.shape
    
    # Track pairwise correlations over time
    pairwise_correlations = {}
    for i in range(dim):
        for j in range(i+1, dim):
            pair_name = f"Corr({i},{j})"
            correlations = []
            
            for t in range(n_time_steps):
                xy_data = data[t, :, [i, j]]
                corr = np.corrcoef(xy_data.T)[0, 1]
                correlations.append(corr)
            
            pairwise_correlations[pair_name] = np.array(correlations)
    
    # Analyze triple interaction strength empirically
    triple_effects = []
    if dim >= 3:
        for t in range(n_time_steps):
            # Measure how much X[2] correlates with X[0]*X[1]
            x0 = data[t, :, 0]
            x1 = data[t, :, 1]
            x2 = data[t, :, 2]
            
            x0x1_product = x0 * x1
            triple_corr = np.corrcoef(x2, x0x1_product)[0, 1]
            triple_effects.append(np.abs(triple_corr))
    
    return {
        'pairwise_correlations': pairwise_correlations,
        'triple_effects': np.array(triple_effects) if triple_effects else None,
        'switch_indices': metadata['switch_indices']
    }

def _create_beyond_pairwise_analysis_plots(data, metadata, empirical_analysis):
    """Create comprehensive analysis visualizations for beyond-pairwise interactions"""
    
    print("\n📊 Creating beyond-pairwise analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Beyond-Pairwise Interactions: Comprehensive Analysis', fontsize=16)
    
    n_time_steps = data.shape[0]
    times = np.arange(n_time_steps)
    switch_indices = metadata['switch_indices']
    
    # 1. Regime sequence and triple interaction evolution
    ax = axes[0, 0]
    ax_triple = ax.twinx()
    
    ax.plot(times, metadata['regime_sequence'], 'b-', linewidth=2, label='Regime')
    ax_triple.plot(times, metadata['triple_interaction_evolution'], 'g--', linewidth=2, label='Triple Effect')
    
    # Mark switch times
    for switch_idx in switch_indices:
        ax.axvline(x=switch_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Regime ID', color='b')
    ax_triple.set_ylabel('Triple Effect Magnitude', color='g')
    ax.set_title('Regime Switches & Triple Interactions')
    ax.grid(True, alpha=0.3)
    
    # 2. Pairwise correlations evolution
    ax = axes[0, 1]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    for i, (pair_name, correlations) in enumerate(empirical_analysis['pairwise_correlations'].items()):
        color = colors[i % len(colors)]
        ax.plot(times, correlations, color=color, linewidth=2, label=pair_name)
    
    # Mark switch times
    for switch_idx in switch_indices:
        ax.axvline(x=switch_idx, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Correlation')
    ax.set_title('Pairwise Correlations Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Triple interaction detection
    ax = axes[0, 2]
    if empirical_analysis['triple_effects'] is not None:
        ax.plot(times, empirical_analysis['triple_effects'], 'purple', linewidth=2, label='|Corr(X2, X0*X1)|')
        
        # Mark switch times
        for switch_idx in switch_indices:
            ax.axvline(x=switch_idx, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Triple Interaction Strength')
        ax.set_title('Empirical Triple Interaction Detection')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Correlation matrices at different regimes
    regime_times = [5, switch_indices[0] + 5, switch_indices[1] + 5]
    regime_labels = ['Regime 0', 'Regime 1', 'Regime 2']
    
    for i, (t_idx, regime_label) in enumerate(zip(regime_times, regime_labels)):
        if i >= 3:  # Only show first 3 regimes
            break
            
        row, col = 1, i
        ax = axes[row, col]
        
        if t_idx < n_time_steps:
            corr_matrix = np.corrcoef(data[t_idx].T)
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'{regime_label} (t={t_idx})')
            
            # Add correlation values as text
            for ii in range(corr_matrix.shape[0]):
                for jj in range(corr_matrix.shape[1]):
                    text = ax.text(jj, ii, f'{corr_matrix[ii, jj]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save the comprehensive analysis
    results_dir = os.path.join(parent_dir, 'results', 'beyond_pairwise_demo')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, 'beyond_pairwise_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Analysis plots saved to: {results_dir}")
    
    # Print summary statistics
    print(f"\n📋 Beyond-Pairwise Analysis Summary:")
    print(f"  Data shape: {data.shape}")
    print(f"  Number of regimes: {metadata['n_regimes']}")
    print(f"  Triple interaction strength: {metadata['beyond_pairwise_strength']}")
    print(f"  Mean empirical triple effect: {np.mean(metadata['triple_interaction_evolution']):.4f}")
    
    if empirical_analysis['triple_effects'] is not None:
        print(f"  Mean detected triple correlation: {np.mean(empirical_analysis['triple_effects']):.4f}")
        print(f"  Max detected triple correlation: {np.max(empirical_analysis['triple_effects']):.4f}")

def main():
    """Main demonstration function"""
    
    print("🌟 BEYOND-PAIRWISE INTERACTIONS MODELING")
    print("=" * 80)
    print("Demonstrating advanced time-dependent vine copula capabilities:")
    print("• Pairwise correlations with regime switching")
    print("• Triple interactions: X[k] += strength * X[i] * X[j]") 
    print("• Higher-order dependency detection")
    print("• Vine copula's ability to capture complex interactions")
    print("=" * 80)
    
    run_beyond_pairwise_demo()
    
    print(f"\n🎉 Beyond-pairwise demo complete!")
    print("This demonstration showcases the ability of DVC-NF to model")
    print("and detect higher-order interactions that go beyond simple")
    print("pairwise correlations - a unique capability for vine copulas!")

if __name__ == "__main__":
    main() 