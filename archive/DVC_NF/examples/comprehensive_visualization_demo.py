#!/usr/bin/env python3
"""
Comprehensive Visualization Demo for DVC-NF

This script demonstrates the complete advanced visualization framework
including all simulation scenarios with professional publication-quality plots:

🎨 ADVANCED VISUALIZATION FEATURES:
1. R-vine structure graphs with time-dependent edges
2. 2D copula visualizations with KDE contours  
3. Temporal interaction analysis (line plots & heatmaps)
4. Executive summary dashboards
5. Scenario-specific deep dives
6. Comparative analysis matrices
7. Network topology analysis
8. Statistical properties analysis

🚀 SIMULATION SCENARIOS:
- Ising-like model with time-varying couplings
- Hidden Markov regime switching
- Log-linear synergy model with triple interactions  
- Spatiotemporal image blocks
- Block switching correlations
- Beyond-pairwise interactions
- Financial market dynamics
- Sinusoidal patterns

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_comprehensive_visualization_demo():
    """Run comprehensive visualization demonstration"""
    
    print("🎨 COMPREHENSIVE VISUALIZATION DEMO FOR DVC-NF")
    print("=" * 80)
    print("Demonstrating advanced visualization capabilities including:")
    print("• R-vine structure graphs with enhanced styling")
    print("• 2D copula visualizations with KDE contours")
    print("• Temporal interaction analysis (lines & heatmaps)")
    print("• Executive summary dashboards")
    print("• Scenario-specific deep dives")
    print("• Comparative analysis matrices")
    print("• Network topology analysis")
    print("• Statistical properties analysis")
    print("=" * 80)
    
    from dvc_nf.data.generators import TimeDependentDataGenerator
    from dvc_nf.visualization import (
        VineVisualizer, 
        AdvancedPlotGenerator,
        create_comprehensive_simulation_analysis,
        create_scenario_comparison_suite
    )
    
    # Configuration
    config = {
        'dim': 4,                    # 4D for rich interaction patterns
        'n_time_steps': 30,          # Moderate time series for clear visualization
        'n_samples_per_time': 80,    # Sufficient samples for statistical analysis
        'random_seed': 42
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize components
    generator = TimeDependentDataGenerator(dim=config['dim'], random_seed=config['random_seed'])
    visualizer = VineVisualizer()
    plot_generator = AdvancedPlotGenerator()
    
    # Storage for all results
    all_results = {}
    
    # PHASE 1: GENERATE DIVERSE SIMULATION SCENARIOS
    print("🚀 PHASE 1: GENERATING DIVERSE SIMULATION SCENARIOS")
    print("-" * 60)
    
    scenarios_to_generate = [
        ('ising', '🧲 Ising-like model'),
        ('hmm', '🔄 Hidden Markov switching'),
        ('loglinear', '🧠 Log-linear synergy'),
        ('spatiotemporal', '🌊 Spatiotemporal blocks'),
        ('block_switching', '🔀 Block switching'),
        ('sinusoidal', '〰️ Sinusoidal patterns')
    ]
    
    generation_times = {}
    
    for scenario_key, scenario_name in scenarios_to_generate:
        print(f"\n{scenario_name}...")
        start_time = time.time()
        
        try:
            if scenario_key == 'ising':
                data, times, metadata = generator.generate_ising_time_series(
                    n_time_steps=config['n_time_steps'],
                    n_samples_per_time=config['n_samples_per_time'],
                    mcmc_sweeps=40
                )
                
            elif scenario_key == 'hmm':
                data, times, metadata = generator.generate_hmm_regimes(
                    n_time_steps=config['n_time_steps'],
                    n_samples_per_time=config['n_samples_per_time'],
                    n_regimes=3,
                    regime_transition=0.15
                )
                
            elif scenario_key == 'loglinear':
                data, times, metadata = generator.generate_loglinear_synergy(
                    n_time_steps=config['n_time_steps'],
                    n_samples_per_time=config['n_samples_per_time'],
                    triple_synergy=True,
                    gibbs_sweeps=40
                )
                
            elif scenario_key == 'spatiotemporal':
                data, times, metadata = generator.generate_spatiotemporal_image_blocks(
                    height=12, width=12,
                    n_time_steps=config['n_time_steps'],
                    block_rows=2, block_cols=2,
                    n_frames_per_time=config['n_samples_per_time']
                )
                
            elif scenario_key == 'block_switching':
                data, times, metadata = generator.generate_block_switching_correlation_data(
                    n_time_steps=config['n_time_steps'],
                    n_samples_per_time=config['n_samples_per_time'],
                    n_regimes=3,
                    switch_probability=0.1
                )
                
            elif scenario_key == 'sinusoidal':
                data, times, metadata = generator.generate_sinusoidal_correlation_data(
                    n_time_steps=config['n_time_steps'],
                    n_samples_per_time=config['n_samples_per_time'],
                    base_correlation=0.5,
                    amplitude=0.4,
                    frequency=1.5
                )
            
            generation_time = time.time() - start_time
            generation_times[scenario_key] = generation_time
            
            all_results[scenario_key] = {
                'data': data,
                'times': times,
                'metadata': metadata
            }
            
            print(f"✅ Generated {scenario_key}: {data.shape} in {generation_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to generate {scenario_key}: {e}")
    
    print(f"\n📊 Generation Summary:")
    print(f"  Generated {len(all_results)}/{len(scenarios_to_generate)} scenarios")
    print(f"  Total generation time: {sum(generation_times.values()):.2f}s")
    print(f"  Average generation time: {np.mean(list(generation_times.values())):.2f}s")
    
    # PHASE 2: BASIC VISUALIZATION DEMOS
    print(f"\n🎨 PHASE 2: BASIC VISUALIZATION DEMONSTRATIONS")
    print("-" * 60)
    
    # 2.1 Demonstrate R-vine structure plotting
    print("\n📊 Demonstrating R-vine structure plotting...")
    _demo_rvine_plotting(all_results)
    
    # 2.2 Demonstrate 2D copula plotting
    print("\n📊 Demonstrating 2D copula plotting...")
    _demo_2d_copula_plotting(all_results)
    
    # 2.3 Demonstrate temporal interaction plotting
    print("\n📊 Demonstrating temporal interaction plotting...")
    _demo_temporal_interaction_plotting(all_results)
    
    # PHASE 3: ADVANCED COMPREHENSIVE ANALYSIS
    print(f"\n🔬 PHASE 3: ADVANCED COMPREHENSIVE ANALYSIS")
    print("-" * 60)
    
    # 3.1 Create comprehensive simulation analysis
    print("\n📈 Creating comprehensive simulation analysis...")
    try:
        create_comprehensive_simulation_analysis(all_results, save_plots=True)
        print("✅ Comprehensive simulation analysis complete!")
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
    
    # 3.2 Create scenario comparison suite
    print("\n🔍 Creating scenario comparison suite...")
    try:
        selected_scenarios = ['ising', 'hmm', 'loglinear']
        create_scenario_comparison_suite(
            all_results, 
            scenarios_to_compare=selected_scenarios,
            save_plots=True
        )
        print("✅ Scenario comparison suite complete!")
    except Exception as e:
        print(f"❌ Scenario comparison failed: {e}")
    
    # PHASE 4: ADVANCED VISUALIZATION FEATURES
    print(f"\n✨ PHASE 4: ADVANCED VISUALIZATION FEATURES")
    print("-" * 60)
    
    # 4.1 Create vine visualizer comprehensive analysis
    print("\n🌐 Creating vine visualizer comprehensive analysis...")
    try:
        visualizer.create_comprehensive_analysis(all_results, save_plots=True)
        print("✅ Vine visualizer analysis complete!")
    except Exception as e:
        print(f"❌ Vine visualizer analysis failed: {e}")
    
    # 4.2 Create advanced plot generator suite
    print("\n📊 Creating advanced plot generator suite...")
    try:
        plot_generator.create_comprehensive_analysis_suite(all_results, save_plots=True)
        print("✅ Advanced plot generator suite complete!")
    except Exception as e:
        print(f"❌ Advanced plot generator failed: {e}")
    
    # PHASE 5: PERFORMANCE AND SUMMARY ANALYSIS
    print(f"\n📋 PHASE 5: PERFORMANCE AND SUMMARY ANALYSIS")
    print("-" * 60)
    
    # Create performance summary
    _create_performance_summary(all_results, generation_times)
    
    # Create final summary visualization
    _create_final_summary_visualization(all_results)
    
    print(f"\n🎉 COMPREHENSIVE VISUALIZATION DEMO COMPLETE!")
    print("=" * 80)
    print("Generated comprehensive analysis including:")
    print("• Executive summary dashboards")
    print("• Scenario-specific deep dives")
    print("• R-vine structure analysis")
    print("• 2D copula analysis") 
    print("• Temporal interaction analysis")
    print("• Comparative analysis matrices")
    print("• Network topology analysis")
    print("• Statistical properties analysis")
    print("• Performance benchmarks")
    print(f"All visualizations saved to results directories!")
    print("=" * 80)


def _demo_rvine_plotting(all_results):
    """Demonstrate R-vine structure plotting"""
    
    from dvc_nf.visualization import plot_rvine_graphs
    
    # Create synthetic R-vine structures for demonstration
    print("  Creating synthetic R-vine structures...")
    
    # Example 1: Simple 4-node vine with two tree levels
    r_matrix = np.array([
        [4, 0, 0, 0],
        [3, 3, 0, 0], 
        [2, 2, 2, 0],
        [1, 1, 1, 1]
    ])
    
    # Tree level 0: primary edges
    tree_0_edges = [(0, 1, 0.8), (1, 2, 0.6), (2, 3, 0.4)]
    
    # Tree level 1: conditional edges  
    tree_1_edges = [(0, 2, 0.5), (1, 3, 0.7)]
    
    adjacency_list = [tree_0_edges, tree_1_edges]
    node_labels = ["X₁", "X₂", "X₃", "X₄"]
    
    # Plot R-vine structure
    plot_rvine_graphs(
        r_matrix, 
        adjacency_list,
        node_labels=node_labels,
        title="Demo R-Vine Structure: Time-Dependent Edges"
    )
    
    print("✅ R-vine structure plotting demo complete!")


def _demo_2d_copula_plotting(all_results):
    """Demonstrate 2D copula plotting"""
    
    from dvc_nf.visualization import plot_2d_copula
    from dvc_nf.visualization.vine_visualization import VineVisualizerHelpers
    
    # Create figure for multiple copula demonstrations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('2D Copula Demonstrations', fontsize=16, fontweight='bold')
    
    scenario_names = list(all_results.keys())[:4]  # First 4 scenarios
    
    for idx, scenario in enumerate(scenario_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        data = all_results[scenario]['data']
        
        # Transform to copula data
        u1, u2 = VineVisualizerHelpers._transform_to_copula_data(data, 0, 1)
        
        # Plot 2D copula
        plot_2d_copula(u1, u2, title=f'{scenario.title()} Copula', ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ 2D copula plotting demo complete!")


def _demo_temporal_interaction_plotting(all_results):
    """Demonstrate temporal interaction plotting"""
    
    from dvc_nf.visualization import plot_temporal_interactions, plot_temporal_interactions_heatmap
    from dvc_nf.visualization.vine_visualization import VineVisualizerHelpers
    
    # Select first scenario for demonstration
    scenario_key = list(all_results.keys())[0]
    result = all_results[scenario_key]
    
    data = result['data']
    times = result['times']
    
    # Compute temporal interactions
    interactions = VineVisualizerHelpers._compute_temporal_interactions(data, times)
    
    # Create edge names
    dim = data.shape[2]
    edge_names = []
    for i in range(dim):
        for j in range(i + 1, dim):
            edge_names.append(f'X{i+1}-X{j+1}')
    
    # Create demonstration figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Temporal Interaction Analysis: {scenario_key.title()}', 
                fontsize=16, fontweight='bold')
    
    # Line plot
    plot_temporal_interactions(
        interactions, times, edge_names[:interactions.shape[1]],
        title="Temporal Interactions (Line Plot)", ax=ax1
    )
    
    # Heatmap
    plot_temporal_interactions_heatmap(
        interactions, times, edge_names[:interactions.shape[1]],
        title="Temporal Interactions (Heatmap)", ax=ax2
    )
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Temporal interaction plotting demo complete!")


def _create_performance_summary(all_results, generation_times):
    """Create performance summary analysis"""
    
    print("\n📊 Performance Summary:")
    print("-" * 40)
    
    # Data size analysis
    total_data_points = 0
    for scenario, result in all_results.items():
        data_shape = result['data'].shape
        n_points = np.prod(data_shape)
        total_data_points += n_points
        
        print(f"  {scenario.title()}:")
        print(f"    Shape: {data_shape}")
        print(f"    Data points: {n_points:,}")
        print(f"    Generation time: {generation_times.get(scenario, 0):.2f}s")
        print(f"    Rate: {n_points/generation_times.get(scenario, 1):,.0f} points/sec")
    
    print(f"\n  Total data points: {total_data_points:,}")
    print(f"  Total generation time: {sum(generation_times.values()):.2f}s")
    print(f"  Overall rate: {total_data_points/sum(generation_times.values()):,.0f} points/sec")


def _create_final_summary_visualization(all_results):
    """Create final summary visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Final Summary: Comprehensive Visualization Demo', 
                fontsize=16, fontweight='bold')
    
    scenario_names = list(all_results.keys())
    
    # 1. Data complexity comparison
    ax1 = axes[0, 0]
    complexities = []
    for scenario in scenario_names:
        data = all_results[scenario]['data']
        complexity = np.std(data) / (np.mean(np.abs(data)) + 1e-8)
        complexities.append(complexity)
    
    bars1 = ax1.bar(range(len(scenario_names)), complexities, 
                    color=plt.cm.viridis(np.linspace(0, 1, len(scenario_names))))
    ax1.set_title('Data Complexity (CV)', fontweight='bold')
    ax1.set_xticks(range(len(scenario_names)))
    ax1.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
    ax1.set_ylabel('Coefficient of Variation')
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation strength comparison
    ax2 = axes[0, 1]
    correlation_strengths = []
    for scenario in scenario_names:
        data = all_results[scenario]['data']
        if data.shape[2] >= 2:
            flat_data = data.reshape(-1, data.shape[2])
            corr_matrix = np.corrcoef(flat_data.T)
            mean_abs_corr = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            correlation_strengths.append(mean_abs_corr)
        else:
            correlation_strengths.append(0)
    
    bars2 = ax2.bar(range(len(scenario_names)), correlation_strengths,
                    color=plt.cm.plasma(np.linspace(0, 1, len(scenario_names))))
    ax2.set_title('Mean |Correlation|', fontweight='bold')
    ax2.set_xticks(range(len(scenario_names)))
    ax2.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
    ax2.set_ylabel('Correlation Strength')
    ax2.grid(True, alpha=0.3)
    
    # 3. Temporal variability
    ax3 = axes[1, 0]
    temporal_variabilities = []
    for scenario in scenario_names:
        data = all_results[scenario]['data']
        temporal_means = np.mean(data, axis=1)  # Mean across samples
        overall_temporal_var = np.var(temporal_means, axis=0)  # Variance across time
        mean_temporal_var = np.mean(overall_temporal_var)
        temporal_variabilities.append(mean_temporal_var)
    
    bars3 = ax3.bar(range(len(scenario_names)), temporal_variabilities,
                    color=plt.cm.coolwarm(np.linspace(0, 1, len(scenario_names))))
    ax3.set_title('Temporal Variability', fontweight='bold')
    ax3.set_xticks(range(len(scenario_names)))
    ax3.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
    ax3.set_ylabel('Variance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary metrics radar plot (simplified)
    ax4 = axes[1, 1]
    
    # Normalize metrics for comparison
    norm_complexity = np.array(complexities) / np.max(complexities)
    norm_correlation = np.array(correlation_strengths) / np.max(correlation_strengths)
    norm_temporal = np.array(temporal_variabilities) / np.max(temporal_variabilities)
    
    x = np.arange(len(scenario_names))
    width = 0.25
    
    ax4.bar(x - width, norm_complexity, width, label='Complexity', alpha=0.8)
    ax4.bar(x, norm_correlation, width, label='Correlation', alpha=0.8)
    ax4.bar(x + width, norm_temporal, width, label='Temporal Var', alpha=0.8)
    
    ax4.set_title('Normalized Metrics Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
    ax4.set_ylabel('Normalized Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Final summary visualization complete!")


def main():
    """Main demo function"""
    
    run_comprehensive_visualization_demo()


if __name__ == "__main__":
    main() 