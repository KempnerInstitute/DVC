#!/usr/bin/env python3
"""
Advanced Simulation Scenarios Demo for DVC-NF

This script demonstrates the four advanced simulation scenarios:
1. Ising-like model with time-varying couplings (MCMC-based)
2. Hidden Markov regime switching with higher-order patterns  
3. Log-linear synergy model with triple interactions (Gibbs sampling)
4. Spatiotemporal image blocks for spatial-temporal analysis

Each scenario tests different aspects of time-dependent vine copula modeling:
- Higher-order magnetic interactions (Ising)
- Hidden state dynamics (HMM)
- Information-theoretic synergy patterns (Log-linear)
- Spatial-temporal correlations (Spatiotemporal)

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

def run_advanced_scenarios_demo():
    """Run comprehensive demonstration of all four advanced scenarios"""
    
    print("🚀 ADVANCED SIMULATION SCENARIOS FOR TIME-DEPENDENT VINE COPULAS")
    print("=" * 80)
    print("Demonstrating four cutting-edge simulation scenarios:")
    print("1. 🧲 Ising-like model with time-varying couplings")
    print("2. 🔄 Hidden Markov regime switching")
    print("3. 🧠 Log-linear synergy model with triple interactions")
    print("4. 🌊 Spatiotemporal image blocks")
    print("=" * 80)
    
    from dvc_nf.data.generators import TimeDependentDataGenerator
    from dvc_nf.analysis.comprehensive import ComprehensiveTimeDependentAnalysis
    
    # Configuration for advanced scenarios
    config = {
        'dim': 4,                    # 4D for rich interaction patterns
        'n_time_steps': 40,          # Moderate time series length
        'n_samples_per_time': 100,   # Sufficient samples for analysis
        'random_seed': 42
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize data generator
    generator = TimeDependentDataGenerator(dim=config['dim'], random_seed=config['random_seed'])
    
    # Results storage
    all_results = {}
    
    # 1. ISING-LIKE MODEL DEMONSTRATION
    print("🧲 SCENARIO 1: ISING-LIKE MODEL WITH TIME-VARYING COUPLINGS")
    print("-" * 60)
    print("Features:")
    print("• MCMC sampling with Metropolis acceptance")
    print("• Time-varying pairwise couplings J_ij(t)")
    print("• Optional triple couplings K_ijk(t)")
    print("• Spin variables in {-1, +1}")
    print("• Tests: Higher-order magnetic interactions, phase transitions")
    
    ising_data, ising_times, ising_metadata = generator.generate_ising_time_series(
        n_time_steps=config['n_time_steps'],
        n_samples_per_time=config['n_samples_per_time'],
        mcmc_sweeps=60  # Sufficient MCMC for quality samples
    )
    
    print(f"✅ Generated Ising data: {ising_data.shape}")
    print(f"✅ Coupling statistics: {ising_metadata['coupling_stats']}")
    
    # Visualize Ising data
    generator.visualize_generated_data('ising', save_plots=True)
    all_results['ising'] = {
        'data': ising_data,
        'times': ising_times,
        'metadata': ising_metadata
    }
    
    # 2. HIDDEN MARKOV MODEL DEMONSTRATION
    print(f"\n🔄 SCENARIO 2: HIDDEN MARKOV REGIME SWITCHING")
    print("-" * 60)
    print("Features:")
    print("• Multiple hidden regimes with distinct correlation structures")
    print("• Markov transitions between regimes")
    print("• Regime-specific correlation matrices")
    print("• Tests: Hidden state dynamics, regime-dependent correlations")
    
    hmm_data, hmm_times, hmm_metadata = generator.generate_hmm_regimes(
        n_time_steps=config['n_time_steps'],
        n_samples_per_time=config['n_samples_per_time'],
        n_regimes=4,
        regime_transition=0.15  # Moderate transition rate
    )
    
    print(f"✅ Generated HMM data: {hmm_data.shape}")
    print(f"✅ Regime statistics: {hmm_metadata['regime_stats']}")
    
    # Visualize HMM data
    generator.visualize_generated_data('hmm', save_plots=True)
    all_results['hmm'] = {
        'data': hmm_data,
        'times': hmm_times,
        'metadata': hmm_metadata
    }
    
    # 3. LOG-LINEAR SYNERGY MODEL DEMONSTRATION
    print(f"\n🧠 SCENARIO 3: LOG-LINEAR SYNERGY MODEL")
    print("-" * 60)
    print("Features:")
    print("• Log-linear probability: log P(X) ~ Σ θ_ij(t) X_i X_j + Σ α_ijk(t) X_i X_j X_k")
    print("• Time-varying pairwise and triple synergy parameters")
    print("• Gibbs sampling for binary variables X_i ∈ {0,1}")
    print("• Tests: Information-theoretic synergy, higher-order interactions")
    
    loglinear_data, loglinear_times, loglinear_metadata = generator.generate_loglinear_synergy(
        n_time_steps=config['n_time_steps'],
        n_samples_per_time=config['n_samples_per_time'],
        triple_synergy=True,
        gibbs_sweeps=60  # Sufficient Gibbs sampling
    )
    
    print(f"✅ Generated log-linear data: {loglinear_data.shape}")
    print(f"✅ Synergy statistics: {loglinear_metadata['synergy_stats']}")
    
    # Visualize log-linear data
    generator.visualize_generated_data('loglinear', save_plots=True)
    all_results['loglinear'] = {
        'data': loglinear_data,
        'times': loglinear_times,
        'metadata': loglinear_metadata
    }
    
    # 4. SPATIOTEMPORAL IMAGE BLOCKS DEMONSTRATION
    print(f"\n🌊 SCENARIO 4: SPATIOTEMPORAL IMAGE BLOCKS")
    print("-" * 60)
    print("Features:")
    print("• Synthetic 'video' frames with spatial-temporal dynamics")
    print("• Block-based aggregation to reduce dimensionality")
    print("• Wave and swirl patterns evolving over time")
    print("• Tests: Spatial-temporal correlations, block-level interactions")
    
    spatiotemporal_data, spatiotemporal_times, spatiotemporal_metadata = generator.generate_spatiotemporal_image_blocks(
        height=12, width=12,  # Moderate image size
        n_time_steps=config['n_time_steps'],
        block_rows=2, block_cols=2,  # 4 blocks total
        n_frames_per_time=config['n_samples_per_time']
    )
    
    print(f"✅ Generated spatiotemporal data: {spatiotemporal_data.shape}")
    print(f"✅ Spatial statistics: blocks={spatiotemporal_metadata['n_blocks']}")
    
    # Visualize spatiotemporal data
    generator.visualize_generated_data('spatiotemporal', save_plots=True)
    all_results['spatiotemporal'] = {
        'data': spatiotemporal_data,
        'times': spatiotemporal_times,
        'metadata': spatiotemporal_metadata
    }
    
    # 5. COMPARATIVE ANALYSIS
    print(f"\n📊 COMPARATIVE ANALYSIS ACROSS SCENARIOS")
    print("-" * 60)
    _create_comparative_analysis(all_results)
    
    # 6. COMPREHENSIVE VINE COPULA ANALYSIS (Optional)
    print(f"\n🔮 COMPREHENSIVE VINE COPULA ANALYSIS")
    print("-" * 60)
    print("Running time-dependent vine copula fitting on advanced scenarios...")
    
    try:
        analyzer = ComprehensiveTimeDependentAnalysis(
            dim=config['dim'], 
            random_seed=config['random_seed']
        )
        
        # Run analysis on selected scenarios (reduced for performance)
        selected_scenarios = ['ising', 'hmm']  # Start with two most interesting
        
        analyzer.run_complete_analysis(
            n_time_steps=config['n_time_steps'],
            n_samples_per_time=config['n_samples_per_time'],
            test_scenarios=selected_scenarios
        )
        
        print(f"✅ Comprehensive analysis completed for {selected_scenarios}!")
        
    except Exception as e:
        print(f"❌ Comprehensive analysis encountered issues: {e}")
        print("Proceeding with data analysis only...")
    
    # 7. SUMMARY AND INSIGHTS
    print(f"\n🎯 SUMMARY AND INSIGHTS")
    print("-" * 60)
    _print_scenario_insights(all_results)
    
    print(f"\n🎉 ADVANCED SCENARIOS DEMO COMPLETE!")
    print("=" * 80)
    print("Generated high-quality data and visualizations for:")
    print("• Ising-like magnetic interactions with MCMC")
    print("• Hidden Markov regime switching dynamics")
    print("• Log-linear synergy with triple interactions")
    print("• Spatiotemporal block patterns")
    print("All results saved to results/time_dependent_data/")
    print("=" * 80)

def _create_comparative_analysis(all_results):
    """Create comparative analysis across all four scenarios"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Advanced Scenarios: Comparative Analysis', fontsize=16)
    
    scenario_names = list(all_results.keys())
    scenario_colors = ['red', 'blue', 'green', 'purple']
    
    # 1. Data type and range comparison
    ax = axes[0, 0]
    for i, (scenario, color) in enumerate(zip(scenario_names, scenario_colors)):
        data = all_results[scenario]['data']
        
        # Compute data statistics
        data_mean = np.mean(data)
        data_std = np.std(data)
        data_min = np.min(data)
        data_max = np.max(data)
        
        ax.errorbar(i, data_mean, yerr=data_std, 
                   color=color, marker='o', markersize=8, 
                   capsize=5, capthick=2, label=scenario.title())
        ax.scatter(i, data_min, color=color, marker='v', alpha=0.7, s=30)
        ax.scatter(i, data_max, color=color, marker='^', alpha=0.7, s=30)
    
    ax.set_title('Data Statistics Comparison')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Value')
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels([s.title() for s in scenario_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Temporal correlation patterns
    ax = axes[0, 1]
    for scenario, color in zip(scenario_names, scenario_colors):
        data = all_results[scenario]['data']
        times = all_results[scenario]['times']
        
        # Compute time-dependent correlation (first two variables)
        if data.shape[2] >= 2:
            correlations = []
            for t in range(len(times)):
                if data[t].shape[0] > 1:  # Ensure we have multiple samples
                    corr = np.corrcoef(data[t][:, 0], data[t][:, 1])[0, 1]
                    correlations.append(corr)
                else:
                    correlations.append(0)
            
            ax.plot(times, correlations, color=color, linewidth=2, 
                   label=f'{scenario.title()}', alpha=0.8)
    
    ax.set_title('Temporal Correlation Evolution')
    ax.set_xlabel('Time')
    ax.set_ylabel('Correlation (Var 0 vs 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Complexity measures
    ax = axes[1, 0]
    complexity_measures = []
    scenario_labels = []
    
    for scenario in scenario_names:
        data = all_results[scenario]['data']
        metadata = all_results[scenario]['metadata']
        
        # Different complexity measures based on scenario type
        if metadata['type'] == 'ising_time_series':
            complexity = metadata['coupling_stats']['pairwise_coupling_mean']
            measure_name = 'Coupling Strength'
        elif metadata['type'] == 'hmm_regimes':
            complexity = metadata['regime_stats']['regime_switches'] / len(all_results[scenario]['times'])
            measure_name = 'Switch Rate'
        elif metadata['type'] == 'loglinear_synergy':
            complexity = metadata['synergy_stats']['mean_pairwise_synergy']
            measure_name = 'Synergy Strength'
        elif metadata['type'] == 'spatiotemporal_image_blocks':
            complexity = metadata['spatial_stats']['pattern_evolution_variance']
            measure_name = 'Pattern Variance'
        else:
            complexity = np.var(np.mean(data, axis=1))
            measure_name = 'Temporal Variance'
        
        complexity_measures.append(complexity)
        scenario_labels.append(scenario.title())
    
    bars = ax.bar(range(len(scenario_names)), complexity_measures, color=scenario_colors, alpha=0.7)
    ax.set_title('Scenario Complexity Measures')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Complexity')
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels(scenario_labels, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, complexity_measures):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Sample distributions comparison
    ax = axes[1, 1]
    for i, (scenario, color) in enumerate(zip(scenario_names, scenario_colors)):
        data = all_results[scenario]['data']
        
        # Flatten data for distribution analysis
        flat_data = data.flatten()
        
        # Create histogram
        ax.hist(flat_data, bins=30, alpha=0.6, color=color, 
               label=scenario.title(), density=True)
    
    ax.set_title('Data Distribution Comparison')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparative analysis
    results_dir = os.path.join(parent_dir, 'results', 'advanced_scenarios')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, 'comparative_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comparative analysis saved to: {results_dir}")

def _print_scenario_insights(all_results):
    """Print insights and characteristics for each scenario"""
    
    for scenario_name, result in all_results.items():
        data = result['data']
        metadata = result['metadata']
        
        print(f"\n📈 {scenario_name.upper()} INSIGHTS:")
        print(f"  Data shape: {data.shape}")
        print(f"  Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"  Data mean: {np.mean(data):.3f} ± {np.std(data):.3f}")
        
        if metadata['type'] == 'ising_time_series':
            print(f"  Magnetic properties:")
            print(f"    • Average magnetization: {np.mean(data):.3f}")
            print(f"    • Coupling strength: {metadata['coupling_stats']['pairwise_coupling_mean']:.3f}")
            print(f"    • Has triple couplings: {metadata['coupling_stats']['has_triple_couplings']}")
            
        elif metadata['type'] == 'hmm_regimes':
            print(f"  Regime dynamics:")
            print(f"    • Number of regimes: {metadata['n_regimes']}")
            print(f"    • Regime switches: {metadata['regime_stats']['regime_switches']}")
            print(f"    • Average regime duration: {metadata['regime_stats']['avg_regime_duration']:.1f}")
            print(f"    • Regime distribution: {metadata['regime_stats']['regime_distribution']}")
            
        elif metadata['type'] == 'loglinear_synergy':
            print(f"  Synergy properties:")
            print(f"    • Activity level: {np.mean(data):.3f}")
            print(f"    • Pairwise synergy: {metadata['synergy_stats']['mean_pairwise_synergy']:.3f}")
            print(f"    • Triple synergy: {metadata['synergy_stats']['mean_triple_synergy']:.3f}")
            print(f"    • Has triple interactions: {metadata['triple_synergy']}")
            
        elif metadata['type'] == 'spatiotemporal_image_blocks':
            print(f"  Spatial properties:")
            print(f"    • Number of blocks: {metadata['n_blocks']}")
            print(f"    • Image size: {metadata['height']}x{metadata['width']}")
            print(f"    • Block grid: {metadata['block_rows']}x{metadata['block_cols']}")
            spatial_stats = metadata['spatial_stats']
            print(f"    • Correlation range: [{np.min(spatial_stats['block_correlations']):.3f}, {np.max(spatial_stats['block_correlations']):.3f}]")

def main():
    """Main demo function"""
    
    run_advanced_scenarios_demo()

if __name__ == "__main__":
    main() 