#!/usr/bin/env python3
"""
Comprehensive Advanced Analysis for DVC-NF

This script provides a unified framework for testing all advanced simulations, 
visualizations, and experiments with configurable parameters suitable for 
large-scale model testing.

Usage:
    # Quick test
    python comprehensive_advanced_analysis.py --config quick_test
    
    # Standard analysis  
    python comprehensive_advanced_analysis.py --config standard_analysis
    
    # Large scale test
    python comprehensive_advanced_analysis.py --config comprehensive_large

🚀 INTEGRATED CAPABILITIES:
- All advanced simulation scenarios (Ising, HMM, Log-linear, Spatiotemporal, etc.)
- Complete visualization suite (R-vine, 2D copula, temporal analysis)
- Time-dependent vine copula modeling with normalizing flows
- Entropy-based R-vine optimization
- Performance benchmarking and scalability analysis
- Comprehensive reporting and result storage
- Configurable parameters for different model scales

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

# Add DVC_NF to path  
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@dataclass
class AnalysisConfig:
    """Configuration class for comprehensive analysis"""
    
    # Data dimensions and scale
    dim: int = 4
    n_time_steps: int = 30
    n_samples_per_time: int = 100
    
    # Scenario selection
    scenarios_to_run: List[str] = None
    
    # Advanced analysis options
    run_time_dependent_vines: bool = True
    run_entropy_optimization: bool = True
    run_comprehensive_visualization: bool = True
    run_performance_analysis: bool = True
    
    # Parameters
    mcmc_sweeps: int = 40
    gibbs_sweeps: int = 40
    
    # Visualization parameters
    save_plots: bool = True
    plot_dpi: int = 300
    show_plots: bool = False
    
    # Random seed
    random_seed: int = 42
    
    # Results directory
    results_dir: str = None
    
    def __post_init__(self):
        if self.scenarios_to_run is None:
            self.scenarios_to_run = [
                'ising', 'hmm', 'loglinear', 'spatiotemporal',
                'block_switching', 'sinusoidal'
            ]
        
        if self.results_dir is None:
            self.results_dir = os.path.join(parent_dir, 'results', 'comprehensive_advanced_analysis')
        
        os.makedirs(self.results_dir, exist_ok=True)


class ComprehensiveAdvancedAnalysis:
    """
    Comprehensive analysis framework integrating all advanced capabilities
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results = {}
        self.performance_metrics = {}
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        # Initialize components
        self._initialize_components()
        
        print("🚀 COMPREHENSIVE ADVANCED ANALYSIS FOR DVC-NF")
        print("=" * 80)
        print("Integrated framework for advanced simulations, visualizations, and analysis")
        print(f"Configuration: {config.dim}D, {config.n_time_steps} time steps, {config.n_samples_per_time} samples")
        print(f"Scenarios: {config.scenarios_to_run}")
        print(f"Results directory: {config.results_dir}")
        print("=" * 80)
    
    def _initialize_components(self):
        """Initialize all analysis components"""
        
        try:
            from dvc_nf.data.generators import TimeDependentDataGenerator
            from dvc_nf.visualization import VineVisualizer, AdvancedPlotGenerator
            
            self.generator = TimeDependentDataGenerator(
                dim=self.config.dim, 
                random_seed=self.config.random_seed
            )
            self.visualizer = VineVisualizer(self.config.results_dir)
            self.plot_generator = AdvancedPlotGenerator(self.config.results_dir)
            
            print("✅ All components initialized successfully")
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            raise
    
    def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis"""
        
        start_time = time.time()
        
        try:
            # Phase 1: Generate all simulation scenarios
            self._generate_all_scenarios()
            
            # Phase 2: Run advanced visualizations
            if self.config.run_comprehensive_visualization:
                self._run_comprehensive_visualizations()
            
            # Phase 3: Time-dependent vine copula analysis
            if self.config.run_time_dependent_vines:
                self._run_time_dependent_vine_analysis()
            
            # Phase 4: Entropy-based optimization analysis
            if self.config.run_entropy_optimization:
                self._run_entropy_optimization_analysis()
            
            # Phase 5: Performance and scalability analysis
            if self.config.run_performance_analysis:
                self._run_performance_analysis()
            
            # Phase 6: Generate comprehensive reports
            self._generate_comprehensive_reports()
            
            total_time = time.time() - start_time
            
            print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
            print("=" * 80)
            print(f"Total execution time: {total_time:.2f}s")
            print(f"Results saved to: {self.config.results_dir}")
            print("=" * 80)
            
            return self.results, self.performance_metrics
            
        except Exception as e:
            print(f"❌ Comprehensive analysis failed: {e}")
            raise
    
    def _generate_all_scenarios(self):
        """Generate all configured simulation scenarios"""
        
        print("\n🚀 PHASE 1: GENERATING ALL SIMULATION SCENARIOS")
        print("-" * 60)
        
        generation_times = {}
        
        for scenario in self.config.scenarios_to_run:
            print(f"\n📊 Generating {scenario.upper()} scenario...")
            
            start_time = time.time()
            
            try:
                data, times, metadata = self._generate_single_scenario(scenario)
                
                generation_time = time.time() - start_time
                generation_times[scenario] = generation_time
                
                self.results[scenario] = {
                    'data': data,
                    'times': times,
                    'metadata': metadata,
                    'generation_time': generation_time
                }
                
                print(f"✅ {scenario}: {data.shape} in {generation_time:.3f}s")
                
            except Exception as e:
                print(f"❌ Failed to generate {scenario}: {e}")
                continue
        
        # Store performance metrics
        self.performance_metrics['generation_times'] = generation_times
        
        print(f"\n📊 Generation Summary:")
        print(f"  Successfully generated: {len(self.results)}/{len(self.config.scenarios_to_run)} scenarios")
        print(f"  Total generation time: {sum(generation_times.values()):.2f}s")
        print(f"  Average generation time: {np.mean(list(generation_times.values())):.2f}s")
    
    def _generate_single_scenario(self, scenario: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate a single simulation scenario"""
        
        if scenario == 'ising':
            return self.generator.generate_ising_time_series(
                n_time_steps=self.config.n_time_steps,
                n_samples_per_time=self.config.n_samples_per_time,
                mcmc_sweeps=self.config.mcmc_sweeps
            )
        
        elif scenario == 'hmm':
            return self.generator.generate_hmm_regimes(
                n_time_steps=self.config.n_time_steps,
                n_samples_per_time=self.config.n_samples_per_time,
                n_regimes=3,
                regime_transition=0.15
            )
        
        elif scenario == 'loglinear':
            return self.generator.generate_loglinear_synergy(
                n_time_steps=self.config.n_time_steps,
                n_samples_per_time=self.config.n_samples_per_time,
                triple_synergy=True,
                gibbs_sweeps=self.config.gibbs_sweeps
            )
        
        elif scenario == 'spatiotemporal':
            return self.generator.generate_spatiotemporal_image_blocks(
                height=12, width=12,
                n_time_steps=self.config.n_time_steps,
                block_rows=2, block_cols=2,
                n_frames_per_time=self.config.n_samples_per_time
            )
        
        elif scenario == 'block_switching':
            return self.generator.generate_block_switching_correlation_data(
                n_time_steps=self.config.n_time_steps,
                n_samples_per_time=self.config.n_samples_per_time,
                n_regimes=3,
                switch_probability=0.1
            )
        
        elif scenario == 'sinusoidal':
            return self.generator.generate_sinusoidal_correlation_data(
                n_time_steps=self.config.n_time_steps,
                n_samples_per_time=self.config.n_samples_per_time,
                base_correlation=0.5,
                amplitude=0.4,
                frequency=1.5
            )
        
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def _run_comprehensive_visualizations(self):
        """Run comprehensive visualization analysis"""
        
        print("\n🎨 PHASE 2: COMPREHENSIVE VISUALIZATION ANALYSIS")
        print("-" * 60)
        
        try:
            # Basic visualization demonstrations
            print("\n📊 Running basic visualization demos...")
            self._run_basic_visualization_demos()
            
            # Advanced visualization suite
            print("\n✨ Running advanced visualization suite...")
            
            # Create comprehensive simulation analysis
            from dvc_nf.visualization import create_comprehensive_simulation_analysis
            create_comprehensive_simulation_analysis(
                self.results, 
                results_dir=os.path.join(self.config.results_dir, 'comprehensive_simulation'),
                save_plots=self.config.save_plots
            )
            
            print("✅ Comprehensive visualization analysis complete!")
            
        except Exception as e:
            print(f"❌ Visualization analysis failed: {e}")
    
    def _run_basic_visualization_demos(self):
        """Run basic visualization demonstrations"""
        
        try:
            from dvc_nf.visualization import plot_rvine_graphs, plot_2d_copula
            from dvc_nf.visualization.vine_visualization import VineVisualizerHelpers
            
            # R-vine structure demonstration
            print("  • R-vine structure plotting...")
            r_matrix = np.array([
                [4, 0, 0, 0],
                [3, 3, 0, 0],
                [2, 2, 2, 0],
                [1, 1, 1, 1]
            ])
            adjacency_list = [[(0, 1, 0.8), (1, 2, 0.6), (2, 3, 0.4)], [(0, 2, 0.5), (1, 3, 0.7)]]
            plot_rvine_graphs(r_matrix, adjacency_list, title="Demo R-Vine Structure")
            
            # 2D copula demonstration for first scenario
            if self.results:
                scenario_key = list(self.results.keys())[0]
                data = self.results[scenario_key]['data']
                
                print("  • 2D copula plotting...")
                u1, u2 = VineVisualizerHelpers._transform_to_copula_data(data, 0, 1)
                plot_2d_copula(u1, u2, title=f"{scenario_key.title()} Copula Demo")
            
            print("    ✅ Basic visualization demos complete!")
            
        except Exception as e:
            print(f"    ❌ Basic visualization demos failed: {e}")
    
    def _run_time_dependent_vine_analysis(self):
        """Run time-dependent vine copula analysis"""
        
        print("\n🌐 PHASE 3: TIME-DEPENDENT VINE COPULA ANALYSIS")
        print("-" * 60)
        
        try:
            from dvc_nf.core.flows import TimeDependentVineCopula
            
            vine_results = {}
            
            # Analyze first two scenarios for vine modeling
            scenarios_for_vine = list(self.results.keys())[:2]
            
            for scenario_key in scenarios_for_vine:
                print(f"\n📊 Analyzing {scenario_key} with time-dependent vine copula...")
                
                try:
                    vine_model = TimeDependentVineCopula(
                        dim=self.config.dim,
                        vine_type='r-vine',
                        n_time_steps=self.config.n_time_steps
                    )
                    
                    data = self.results[scenario_key]['data']
                    times = self.results[scenario_key]['times']
                    
                    # Initialize and fit (with reduced epochs for speed)
                    vine_model.initialize_vine_structure(data.reshape(-1, data.shape[2]))
                    vine_model.initialize_flows(hidden_dim=32)
                    
                    vine_model.fit(
                        data, times,
                        learning_rate=1e-3,
                        num_epochs=min(100, 200),  # Reduced for speed
                        patience=20
                    )
                    
                    predictions = vine_model.predict_bandwidth_evolution()
                    
                    vine_results[scenario_key] = {
                        'predictions': predictions,
                        'training_history': vine_model.training_history
                    }
                    
                    print(f"  ✅ {scenario_key}: vine copula analysis complete")
                    
                except Exception as e:
                    print(f"  ❌ {scenario_key}: vine analysis failed: {e}")
                    continue
            
            self.results['vine_analysis'] = vine_results
            print(f"\n✅ Time-dependent vine analysis complete for {len(vine_results)} scenarios!")
            
        except Exception as e:
            print(f"❌ Time-dependent vine analysis failed: {e}")
    
    def _run_entropy_optimization_analysis(self):
        """Run entropy-based optimization analysis"""
        
        print("\n🧠 PHASE 4: ENTROPY-BASED OPTIMIZATION ANALYSIS")
        print("-" * 60)
        
        try:
            from dvc_nf.optimization.entropy import EntropyBasedRVineOptimizer
            
            entropy_results = {}
            
            # Select first scenario for entropy analysis (computationally intensive)
            scenario_key = list(self.results.keys())[0]
            print(f"\n📊 Running entropy optimization for {scenario_key}...")
            
            entropy_optimizer = EntropyBasedRVineOptimizer(
                dim=self.config.dim,
                n_samples=min(1000, self.config.n_samples_per_time * self.config.n_time_steps)
            )
            
            data = self.results[scenario_key]['data']
            flat_data = data.reshape(-1, data.shape[2])
            
            # Run entropy optimization comparison
            comparison_results = entropy_optimizer.compare_optimization_methods(flat_data)
            
            entropy_results[scenario_key] = {
                'comparison_results': comparison_results
            }
            
            self.results['entropy_analysis'] = entropy_results
            print(f"✅ Entropy optimization analysis complete!")
            
        except Exception as e:
            print(f"❌ Entropy optimization analysis failed: {e}")
    
    def _run_performance_analysis(self):
        """Run performance and scalability analysis"""
        
        print("\n⚡ PHASE 5: PERFORMANCE AND SCALABILITY ANALYSIS")
        print("-" * 60)
        
        performance_analysis = {}
        
        # Data size analysis
        print("\n📊 Analyzing data sizes and generation performance...")
        total_data_points = 0
        generation_rates = {}
        
        for scenario, result in self.results.items():
            if 'data' in result:
                data_shape = result['data'].shape
                n_points = np.prod(data_shape)
                total_data_points += n_points
                
                generation_time = result.get('generation_time', 1.0)
                rate = n_points / generation_time if generation_time > 0 else 0
                generation_rates[scenario] = rate
                
                print(f"  {scenario}: {data_shape} → {n_points:,} points → {rate:,.0f} points/sec")
        
        performance_analysis['total_data_points'] = total_data_points
        performance_analysis['generation_rates'] = generation_rates
        performance_analysis['mean_generation_rate'] = np.mean(list(generation_rates.values()))
        
        # Scalability projections
        print("\n📈 Scalability projections...")
        scalability_projections = self._compute_scalability_projections()
        performance_analysis['scalability_projections'] = scalability_projections
        
        self.performance_metrics['performance_analysis'] = performance_analysis
        print("✅ Performance analysis complete!")
    
    def _compute_scalability_projections(self) -> Dict:
        """Compute scalability projections for larger problem sizes"""
        
        projections = {}
        
        # Current configuration
        current_config = {
            'dim': self.config.dim,
            'n_time_steps': self.config.n_time_steps,
            'n_samples_per_time': self.config.n_samples_per_time
        }
        
        # Projected configurations
        projected_configs = [
            {'dim': 8, 'n_time_steps': 50, 'n_samples_per_time': 200},
            {'dim': 16, 'n_time_steps': 100, 'n_samples_per_time': 500},
            {'dim': 32, 'n_time_steps': 200, 'n_samples_per_time': 1000}
        ]
        
        for i, proj_config in enumerate(projected_configs):
            scale_factor = (proj_config['dim'] / current_config['dim']) * \
                          (proj_config['n_time_steps'] / current_config['n_time_steps']) * \
                          (proj_config['n_samples_per_time'] / current_config['n_samples_per_time'])
            
            # Estimate times based on current performance
            mean_generation_time = np.mean([r.get('generation_time', 0) for r in self.results.values()])
            projected_time = mean_generation_time * scale_factor
            
            projections[f'scale_{i+1}'] = {
                'config': proj_config,
                'scale_factor': scale_factor,
                'estimated_generation_time_per_scenario': projected_time,
                'estimated_total_time_hours': projected_time * len(self.config.scenarios_to_run) / 3600
            }
            
            print(f"  Scale {i+1}: {proj_config['dim']}D × {proj_config['n_time_steps']}T × {proj_config['n_samples_per_time']}S")
            print(f"    → Est. {projected_time:.1f}s per scenario ({projected_time * len(self.config.scenarios_to_run) / 3600:.1f}h total)")
        
        return projections
    
    def _generate_comprehensive_reports(self):
        """Generate comprehensive analysis reports"""
        
        print("\n📋 PHASE 6: GENERATING COMPREHENSIVE REPORTS")
        print("-" * 60)
        
        # Generate text report
        self._generate_text_report()
        
        # Generate JSON results
        self._save_json_results()
        
        # Generate performance visualization
        self._create_performance_visualization()
        
        print("✅ Comprehensive reports generated!")
    
    def _generate_text_report(self):
        """Generate detailed text report"""
        
        report_path = os.path.join(self.config.results_dir, 'comprehensive_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE ADVANCED ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Configuration summary
            f.write("1. ANALYSIS CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Dimensions: {self.config.dim}\n")
            f.write(f"Time steps: {self.config.n_time_steps}\n")
            f.write(f"Samples per time: {self.config.n_samples_per_time}\n")
            f.write(f"Scenarios analyzed: {len(self.results)}\n")
            f.write(f"Random seed: {self.config.random_seed}\n\n")
            
            # Generation performance
            f.write("2. DATA GENERATION PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            generation_times = self.performance_metrics.get('generation_times', {})
            for scenario, time_val in generation_times.items():
                data_shape = self.results[scenario]['data'].shape if scenario in self.results else "N/A"
                f.write(f"{scenario}: {data_shape} in {time_val:.3f}s\n")
            
            if generation_times:
                f.write(f"Average generation time: {np.mean(list(generation_times.values())):.3f}s\n")
                f.write(f"Total generation time: {sum(generation_times.values()):.3f}s\n\n")
            
            # Performance projections
            if 'performance_analysis' in self.performance_metrics:
                perf_analysis = self.performance_metrics['performance_analysis']
                if 'scalability_projections' in perf_analysis:
                    f.write("3. SCALABILITY PROJECTIONS\n")
                    f.write("-" * 40 + "\n")
                    projections = perf_analysis['scalability_projections']
                    for scale_key, projection in projections.items():
                        config = projection['config']
                        f.write(f"{scale_key}: {config['dim']}D × {config['n_time_steps']}T × {config['n_samples_per_time']}S\n")
                        f.write(f"  Estimated time: {projection['estimated_total_time_hours']:.1f} hours\n")
                    f.write("\n")
            
            f.write("4. ANALYSIS SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"✅ Successfully analyzed {len(self.results)} simulation scenarios\n")
            f.write(f"✅ Generated comprehensive visualizations\n")
            if 'vine_analysis' in self.results:
                f.write(f"✅ Completed time-dependent vine analysis\n")
            if 'entropy_analysis' in self.results:
                f.write(f"✅ Completed entropy-based optimization analysis\n")
            f.write(f"✅ Generated performance benchmarks and scalability projections\n")
        
        print(f"  📄 Text report saved: {report_path}")
    
    def _save_json_results(self):
        """Save results in JSON format"""
        
        # Prepare serializable results
        serializable_results = {}
        
        for key, value in self.results.items():
            if key in ['vine_analysis', 'entropy_analysis']:
                serializable_results[key] = value
            elif isinstance(value, dict) and 'data' in value:
                # Save metadata and summary statistics instead of full data arrays
                serializable_results[key] = {
                    'data_shape': value['data'].shape,
                    'generation_time': value.get('generation_time', 0),
                    'metadata': value.get('metadata', {}),
                    'data_summary': {
                        'mean': float(np.mean(value['data'])),
                        'std': float(np.std(value['data'])),
                        'min': float(np.min(value['data'])),
                        'max': float(np.max(value['data']))
                    }
                }
        
        # Add performance metrics
        serializable_results['performance_metrics'] = self.performance_metrics
        
        # Add configuration
        serializable_results['configuration'] = asdict(self.config)
        
        # Save to JSON
        json_path = os.path.join(self.config.results_dir, 'comprehensive_analysis_results.json')
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"  📊 JSON results saved: {json_path}")
    
    def _create_performance_visualization(self):
        """Create performance visualization dashboard"""
        
        if not self.config.save_plots:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Comprehensive Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Generation times
        ax = axes[0, 0]
        if 'generation_times' in self.performance_metrics:
            times = self.performance_metrics['generation_times']
            scenarios = list(times.keys())
            values = list(times.values())
            
            bars = ax.bar(range(len(scenarios)), values, color=plt.cm.viridis(np.linspace(0, 1, len(scenarios))))
            ax.set_title('Data Generation Times')
            ax.set_ylabel('Time (seconds)')
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}s', ha='center', va='bottom')
        
        # 2. Data sizes
        ax = axes[0, 1]
        scenario_results = {k: v for k, v in self.results.items() if isinstance(v, dict) and 'data' in v}
        if scenario_results:
            scenarios = list(scenario_results.keys())
            data_points = [np.prod(result['data'].shape) for result in scenario_results.values()]
            
            bars = ax.bar(range(len(scenarios)), data_points, color=plt.cm.plasma(np.linspace(0, 1, len(scenarios))))
            ax.set_title('Total Data Points')
            ax.set_ylabel('Number of Data Points')
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.set_yscale('log')
        
        # 3. Generation rates
        ax = axes[1, 0]
        if 'performance_analysis' in self.performance_metrics:
            perf_analysis = self.performance_metrics['performance_analysis']
            if 'generation_rates' in perf_analysis:
                rates = perf_analysis['generation_rates']
                scenarios = list(rates.keys())
                values = [rates[s] for s in scenarios]
                
                bars = ax.bar(range(len(scenarios)), values, color=plt.cm.tab10(np.linspace(0, 1, len(scenarios))))
                ax.set_title('Generation Rates')
                ax.set_ylabel('Points/Second')
                ax.set_xticks(range(len(scenarios)))
                ax.set_xticklabels(scenarios, rotation=45, ha='right')
                ax.set_yscale('log')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary text
        summary_text = "PERFORMANCE SUMMARY\n\n"
        if 'generation_times' in self.performance_metrics:
            times = self.performance_metrics['generation_times']
            summary_text += f"Total scenarios: {len(times)}\n"
            summary_text += f"Total generation time: {sum(times.values()):.2f}s\n"
            summary_text += f"Average generation time: {np.mean(list(times.values())):.2f}s\n\n"
        
        summary_text += f"Configuration:\n"
        summary_text += f"• Dimensions: {self.config.dim}\n"
        summary_text += f"• Time steps: {self.config.n_time_steps}\n"
        summary_text += f"• Samples/time: {self.config.n_samples_per_time}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.config.results_dir, 'performance_dashboard.png')
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Performance dashboard saved: {plot_path}")


def create_configuration_presets() -> Dict[str, AnalysisConfig]:
    """Create predefined configuration presets for different use cases"""
    
    presets = {
        'quick_test': AnalysisConfig(
            dim=3,
            n_time_steps=10,
            n_samples_per_time=50,
            scenarios_to_run=['ising', 'hmm', 'sinusoidal'],
            run_time_dependent_vines=False,
            run_entropy_optimization=False,
            mcmc_sweeps=20,
            gibbs_sweeps=20
        ),
        
        'standard_analysis': AnalysisConfig(
            dim=4,
            n_time_steps=30,
            n_samples_per_time=100,
            scenarios_to_run=['ising', 'hmm', 'loglinear', 'spatiotemporal', 'block_switching', 'sinusoidal'],
            run_time_dependent_vines=True,
            run_entropy_optimization=True,
            mcmc_sweeps=40,
            gibbs_sweeps=40
        ),
        
        'comprehensive_large': AnalysisConfig(
            dim=8,
            n_time_steps=50,
            n_samples_per_time=200,
            scenarios_to_run=['ising', 'hmm', 'loglinear', 'spatiotemporal', 'block_switching', 'sinusoidal'],
            run_time_dependent_vines=True,
            run_entropy_optimization=True,
            mcmc_sweeps=60,
            gibbs_sweeps=60
        ),
        
        'scalability_test': AnalysisConfig(
            dim=16,
            n_time_steps=100,
            n_samples_per_time=500,
            scenarios_to_run=['ising', 'hmm', 'sinusoidal'],  # Reduced scenarios for large scale
            run_time_dependent_vines=False,  # Too computationally expensive
            run_entropy_optimization=False,
            mcmc_sweeps=40,
            gibbs_sweeps=40
        )
    }
    
    return presets


def run_comprehensive_analysis_with_config(config_name: str = 'standard_analysis', 
                                         custom_config: AnalysisConfig = None):
    """Run comprehensive analysis with specified configuration"""
    
    if custom_config is not None:
        config = custom_config
    else:
        presets = create_configuration_presets()
        if config_name not in presets:
            raise ValueError(f"Unknown config preset: {config_name}. Available: {list(presets.keys())}")
        config = presets[config_name]
    
    print(f"🚀 Running comprehensive analysis with configuration: {config_name}")
    print(f"Configuration details:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    
    # Create and run analysis
    analysis = ComprehensiveAdvancedAnalysis(config)
    results, performance_metrics = analysis.run_comprehensive_analysis()
    
    return results, performance_metrics, analysis


def main():
    """Main function for running comprehensive analysis"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Advanced Analysis for DVC-NF')
    parser.add_argument('--config', type=str, default='standard_analysis',
                       choices=['quick_test', 'standard_analysis', 'comprehensive_large', 'scalability_test'],
                       help='Configuration preset to use')
    parser.add_argument('--dim', type=int, help='Override dimensions')
    parser.add_argument('--time_steps', type=int, help='Override number of time steps')
    parser.add_argument('--samples', type=int, help='Override samples per time step')
    parser.add_argument('--scenarios', nargs='+', help='Override scenarios to run')
    parser.add_argument('--no_plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively')
    parser.add_argument('--results_dir', type=str, help='Override results directory')
    
    args = parser.parse_args()
    
    # Get base configuration
    presets = create_configuration_presets()
    config = presets[args.config]
    
    # Apply overrides
    if args.dim:
        config.dim = args.dim
    if args.time_steps:
        config.n_time_steps = args.time_steps
    if args.samples:
        config.n_samples_per_time = args.samples
    if args.scenarios:
        config.scenarios_to_run = args.scenarios
    if args.no_plots:
        config.save_plots = False
    if args.show_plots:
        config.show_plots = True
    if args.results_dir:
        config.results_dir = args.results_dir
    
    # Run analysis
    results, performance_metrics, analysis = run_comprehensive_analysis_with_config(
        config_name=args.config,
        custom_config=config
    )
    
    print(f"\n✅ Analysis complete! Check results in: {config.results_dir}")


if __name__ == "__main__":
    main() 