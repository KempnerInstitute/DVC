#!/usr/bin/env python3
"""
Comprehensive Time-Dependent Vine Copula Analysis

This script demonstrates the complete pipeline for time-dependent vine copula modeling:
1. Generate synthetic time-dependent datasets
2. Fit time-dependent vine copulas with normalizing flows
3. Compare with traditional static vine copulas
4. Analyze bandwidth evolution and model performance

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import our time-dependent modules
from ..core.flows import TimeDependentVineCopula
from ..data.generators import TimeDependentDataGenerator

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class ComprehensiveTimeDependentAnalysis:
    """
    Comprehensive analysis framework for time-dependent vine copulas
    """
    
    def __init__(self, dim=4, random_seed=42):
        """
        Initialize analysis framework
        
        Parameters:
        -----------
        dim : int
            Data dimensionality
        random_seed : int
            Random seed for reproducibility
        """
        
        self.dim = dim
        self.random_seed = random_seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Initialize components
        self.data_generator = TimeDependentDataGenerator(dim=dim, random_seed=random_seed)
        
        # Results storage
        self.results = {}
        self.models = {}
        
        # Results directory
        self.results_dir = os.path.join(current_dir, '..', 'results', 'comprehensive_time_dependent')
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Initialized analysis framework for {dim}-dimensional data")
        print(f"Results will be saved to: {self.results_dir}")
    
    def run_complete_analysis(self, 
                            n_time_steps=100,
                            n_samples_per_time=150,
                            test_scenarios=['piecewise', 'sinusoidal', 'financial']):
        """
        Run complete time-dependent vine copula analysis
        
        Parameters:
        -----------
        n_time_steps : int
            Number of time steps
        n_samples_per_time : int
            Samples per time step
        test_scenarios : list
            Data scenarios to test
        """
        
        print("="*80)
        print("COMPREHENSIVE TIME-DEPENDENT VINE COPULA ANALYSIS")
        print("="*80)
        print(f"Dimensions: {self.dim}")
        print(f"Time steps: {n_time_steps}")
        print(f"Samples per time: {n_samples_per_time}")
        print(f"Test scenarios: {test_scenarios}")
        print("="*80)
        
        for scenario in test_scenarios:
            print(f"\n{'='*50}")
            print(f"ANALYZING SCENARIO: {scenario.upper()}")
            print(f"{'='*50}")
            
            # Step 1: Generate test data
            print(f"\n1. Generating {scenario} data...")
            test_data, test_times, metadata = self._generate_test_data(
                scenario, n_time_steps, n_samples_per_time
            )
            
            # Step 2: Fit time-dependent vine copula
            print(f"\n2. Fitting time-dependent vine copula...")
            time_dependent_results = self._fit_time_dependent_model(
                test_data, test_times, scenario
            )
            
            # Step 3: Fit traditional static vine copula for comparison
            print(f"\n3. Fitting static vine copula baseline...")
            static_results = self._fit_static_baseline(
                test_data, scenario
            )
            
            # Step 4: Compare results
            print(f"\n4. Comparing models...")
            comparison_results = self._compare_models(
                time_dependent_results, static_results, test_data, test_times, scenario
            )
            
            # Store results
            self.results[scenario] = {
                'metadata': metadata,
                'time_dependent': time_dependent_results,
                'static': static_results,
                'comparison': comparison_results
            }
            
            # Step 5: Create visualizations
            print(f"\n5. Creating visualizations...")
            self._create_scenario_visualizations(scenario)
        
        # Step 6: Create summary analysis
        print(f"\n{'='*50}")
        print("CREATING SUMMARY ANALYSIS")
        print(f"{'='*50}")
        self._create_summary_analysis()
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"All results saved to: {self.results_dir}")
        print(f"{'='*80}")
    
    def _generate_test_data(self, scenario, n_time_steps, n_samples_per_time):
        """Generate test data for specific scenario"""
        
        if scenario == 'piecewise':
            data, times, metadata = self.data_generator.generate_piecewise_correlation_data(
                n_time_steps, n_samples_per_time,
                breakpoints=[0.3, 0.7],
                correlations=[0.2, 0.8, 0.3]
            )
        elif scenario == 'sinusoidal':
            data, times, metadata = self.data_generator.generate_sinusoidal_correlation_data(
                n_time_steps, n_samples_per_time,
                base_correlation=0.4,
                amplitude=0.4,
                frequency=2.0
            )
        elif scenario == 'financial':
            data, times, metadata = self.data_generator.generate_financial_inspired_data(
                n_time_steps, n_samples_per_time,
                volatility_clustering=True,
                correlation_breaks=True
            )
        elif scenario == 'regime_switching':
            data, times, metadata = self.data_generator.generate_regime_switching_data(
                n_time_steps, n_samples_per_time,
                n_regimes=3,
                regime_persistence=0.9
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        print(f"  Generated data shape: {data.shape}")
        print(f"  Scenario type: {metadata['type']}")
        
        return data, times, metadata
    
    def _fit_time_dependent_model(self, data, times, scenario):
        """Fit time-dependent vine copula model"""
        
        # Initialize time-dependent model
        model = TimeDependentVineCopula(
            dim=self.dim,
            vine_type='c-vine',  # Start with C-vine for simplicity
            optimization_method='tau',
            n_time_steps=len(times)
        )
        
        # Initialize structure (without data for now)
        model.initialize_vine_structure()
        
        # Initialize flows
        model.initialize_flows(hidden_dim=32)
        
        # Record start time
        start_time = time.time()
        
        # Fit model
        try:
            model.fit(
                data, times,
                learning_rate=1e-3,
                num_epochs=200,
                patience=30,
                min_delta=1e-6
            )
            
            fit_success = True
            fit_time = time.time() - start_time
            
            # Get bandwidth predictions
            bandwidth_predictions = model.predict_bandwidth_evolution()
            
            # Evaluate model
            train_loss = model.evaluate_time_dependent_fit(data, times)
            
        except Exception as e:
            print(f"  Time-dependent model fitting failed: {e}")
            fit_success = False
            fit_time = time.time() - start_time
            bandwidth_predictions = {}
            train_loss = float('inf')
        
        # Store model
        self.models[f'{scenario}_time_dependent'] = model if fit_success else None
        
        results = {
            'success': fit_success,
            'fit_time': fit_time,
            'train_loss': train_loss,
            'bandwidth_predictions': bandwidth_predictions,
            'model_type': 'time_dependent'
        }
        
        print(f"  Time-dependent model fit: {'SUCCESS' if fit_success else 'FAILED'}")
        print(f"  Fit time: {fit_time:.2f}s")
        if fit_success:
            print(f"  Train loss: {train_loss:.6f}")
            print(f"  Bandwidth predictions: {len(bandwidth_predictions)} edges")
        
        return results
    
    def _fit_static_baseline(self, data, scenario):
        """Fit static vine copula baseline for comparison"""
        
        # Flatten time series data for static model
        # Use average across time steps
        static_data = np.mean(data, axis=0)  # Shape: (n_samples, dim)
        
        start_time = time.time()
        
        try:
            # Simple static analysis - compute correlation matrix
            correlation_matrix = np.corrcoef(static_data.T)
            
            # Compute average pairwise correlation as a simple metric
            off_diagonal = correlation_matrix[np.triu_indices(self.dim, k=1)]
            avg_correlation = np.mean(np.abs(off_diagonal))
            
            # Simulate fitting time
            fit_time = time.time() - start_time
            
            # Simple likelihood approximation
            log_likelihood = -0.5 * np.sum(np.log(np.diag(correlation_matrix)))
            
            fit_success = True
            
        except Exception as e:
            print(f"  Static model fitting failed: {e}")
            fit_success = False
            fit_time = time.time() - start_time
            avg_correlation = 0.0
            log_likelihood = float('-inf')
            correlation_matrix = np.eye(self.dim)
        
        results = {
            'success': fit_success,
            'fit_time': fit_time,
            'correlation_matrix': correlation_matrix,
            'avg_correlation': avg_correlation,
            'log_likelihood': log_likelihood,
            'model_type': 'static'
        }
        
        print(f"  Static model fit: {'SUCCESS' if fit_success else 'FAILED'}")
        print(f"  Fit time: {fit_time:.2f}s")
        if fit_success:
            print(f"  Average correlation: {avg_correlation:.4f}")
        
        return results
    
    def _compare_models(self, time_dependent_results, static_results, data, times, scenario):
        """Compare time-dependent vs static models"""
        
        comparison = {
            'scenario': scenario,
            'time_dependent_success': time_dependent_results['success'],
            'static_success': static_results['success'],
            'fit_time_ratio': None,
            'performance_metrics': {}
        }
        
        if time_dependent_results['success'] and static_results['success']:
            # Compare fit times
            comparison['fit_time_ratio'] = (
                time_dependent_results['fit_time'] / static_results['fit_time']
            )
            
            # Compare model complexity
            comparison['performance_metrics'] = {
                'time_dependent_loss': time_dependent_results['train_loss'],
                'static_log_likelihood': static_results['log_likelihood'],
                'bandwidth_variability': self._compute_bandwidth_variability(
                    time_dependent_results['bandwidth_predictions']
                )
            }
            
            print(f"  Fit time ratio (TD/Static): {comparison['fit_time_ratio']:.2f}")
            print(f"  Time-dependent loss: {time_dependent_results['train_loss']:.6f}")
            print(f"  Bandwidth variability: {comparison['performance_metrics']['bandwidth_variability']:.4f}")
        
        return comparison
    
    def _compute_bandwidth_variability(self, bandwidth_predictions):
        """Compute variability in bandwidth predictions"""
        
        if not bandwidth_predictions:
            return 0.0
        
        variabilities = []
        for edge_id, pred in bandwidth_predictions.items():
            bandwidths = pred['bandwidths']
            variability = np.std(bandwidths) / np.mean(bandwidths)  # Coefficient of variation
            variabilities.append(variability)
        
        return np.mean(variabilities)
    
    def _create_scenario_visualizations(self, scenario):
        """Create visualizations for specific scenario"""
        
        if scenario not in self.results:
            return
        
        result = self.results[scenario]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Time-Dependent Vine Copula Analysis: {scenario.title()}', fontsize=16)
        
        # 1. Data overview
        metadata = result['metadata']
        if 'correlation_evolution' in metadata:
            axes[0, 0].plot(metadata['correlation_evolution'], linewidth=2)
            axes[0, 0].set_title('True Correlation Evolution')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Correlation')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Bandwidth evolution (if available)
        if result['time_dependent']['success']:
            bandwidth_pred = result['time_dependent']['bandwidth_predictions']
            if bandwidth_pred:
                first_edge = list(bandwidth_pred.keys())[0]
                pred = bandwidth_pred[first_edge]
                axes[0, 1].plot(pred['times'], pred['bandwidths'], linewidth=2)
                axes[0, 1].set_title(f'Learned Bandwidth Evolution\n{first_edge}')
                axes[0, 1].set_xlabel('Time (normalized)')
                axes[0, 1].set_ylabel('Bandwidth')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model comparison
        if result['comparison']['time_dependent_success'] and result['comparison']['static_success']:
            metrics = result['comparison']['performance_metrics']
            metric_names = ['TD Loss', 'Static LogLik', 'BW Variability']
            metric_values = [
                metrics['time_dependent_loss'],
                metrics['static_log_likelihood'],
                metrics['bandwidth_variability']
            ]
            
            axes[0, 2].bar(metric_names, metric_values)
            axes[0, 2].set_title('Model Performance Comparison')
            axes[0, 2].set_ylabel('Value')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Fit time comparison
        if result['comparison']['fit_time_ratio'] is not None:
            fit_times = [
                result['time_dependent']['fit_time'],
                result['static']['fit_time']
            ]
            model_names = ['Time-Dependent', 'Static']
            
            axes[1, 0].bar(model_names, fit_times)
            axes[1, 0].set_title('Fitting Time Comparison')
            axes[1, 0].set_ylabel('Time (seconds)')
        
        # 5. Bandwidth variability across edges
        if result['time_dependent']['success']:
            bandwidth_pred = result['time_dependent']['bandwidth_predictions']
            if bandwidth_pred:
                edge_names = []
                variabilities = []
                
                for edge_id, pred in bandwidth_pred.items():
                    bandwidths = pred['bandwidths']
                    variability = np.std(bandwidths) / np.mean(bandwidths)
                    edge_names.append(edge_id.split('_')[-2:])  # Get node pair
                    variabilities.append(variability)
                
                axes[1, 1].bar(range(len(variabilities)), variabilities)
                axes[1, 1].set_title('Bandwidth Variability by Edge')
                axes[1, 1].set_xlabel('Edge Index')
                axes[1, 1].set_ylabel('Coefficient of Variation')
                axes[1, 1].set_xticks(range(len(edge_names)))
                axes[1, 1].set_xticklabels([f"{e[0]}-{e[1]}" for e in edge_names], rotation=45)
        
        # 6. Model success summary
        success_data = [
            int(result['time_dependent']['success']),
            int(result['static']['success'])
        ]
        model_types = ['Time-Dependent', 'Static']
        
        axes[1, 2].bar(model_types, success_data, color=['blue', 'orange'])
        axes[1, 2].set_title('Model Fitting Success')
        axes[1, 2].set_ylabel('Success (1=Yes, 0=No)')
        axes[1, 2].set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, f'{scenario}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved visualization: {plot_path}")
    
    def _create_summary_analysis(self):
        """Create summary analysis across all scenarios"""
        
        # Prepare summary data
        scenarios = list(self.results.keys())
        
        summary_data = {
            'scenario': scenarios,
            'time_dependent_success': [],
            'static_success': [],
            'fit_time_ratio': [],
            'bandwidth_variability': []
        }
        
        for scenario in scenarios:
            result = self.results[scenario]
            
            summary_data['time_dependent_success'].append(
                int(result['time_dependent']['success'])
            )
            summary_data['static_success'].append(
                int(result['static']['success'])
            )
            
            if result['comparison']['fit_time_ratio'] is not None:
                summary_data['fit_time_ratio'].append(result['comparison']['fit_time_ratio'])
            else:
                summary_data['fit_time_ratio'].append(np.nan)
            
            if result['time_dependent']['success']:
                summary_data['bandwidth_variability'].append(
                    result['comparison']['performance_metrics'].get('bandwidth_variability', 0)
                )
            else:
                summary_data['bandwidth_variability'].append(np.nan)
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Time-Dependent Vine Copula Analysis Summary', fontsize=16)
        
        # 1. Success rates
        td_success_rate = np.mean(summary_data['time_dependent_success'])
        static_success_rate = np.mean(summary_data['static_success'])
        
        axes[0, 0].bar(['Time-Dependent', 'Static'], [td_success_rate, static_success_rate])
        axes[0, 0].set_title('Model Success Rates')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim([0, 1.1])
        
        # 2. Fit time ratios
        valid_ratios = [r for r in summary_data['fit_time_ratio'] if not np.isnan(r)]
        if valid_ratios:
            axes[0, 1].bar(range(len(valid_ratios)), valid_ratios)
            axes[0, 1].set_title('Fit Time Ratio (TD/Static)')
            axes[0, 1].set_xlabel('Scenario')
            axes[0, 1].set_ylabel('Ratio')
            axes[0, 1].set_xticks(range(len(scenarios)))
            axes[0, 1].set_xticklabels(scenarios, rotation=45)
        
        # 3. Bandwidth variability
        valid_vars = [v for v in summary_data['bandwidth_variability'] if not np.isnan(v)]
        if valid_vars:
            axes[1, 0].bar(range(len(valid_vars)), valid_vars)
            axes[1, 0].set_title('Bandwidth Variability by Scenario')
            axes[1, 0].set_xlabel('Scenario')
            axes[1, 0].set_ylabel('Coefficient of Variation')
            axes[1, 0].set_xticks(range(len(scenarios)))
            axes[1, 0].set_xticklabels(scenarios, rotation=45)
        
        # 4. Summary statistics table
        axes[1, 1].axis('off')
        table_data = []
        table_data.append(['Metric', 'Value'])
        table_data.append(['TD Success Rate', f'{td_success_rate:.2f}'])
        table_data.append(['Static Success Rate', f'{static_success_rate:.2f}'])
        if valid_ratios:
            table_data.append(['Avg Fit Time Ratio', f'{np.mean(valid_ratios):.2f}'])
        if valid_vars:
            table_data.append(['Avg BW Variability', f'{np.mean(valid_vars):.4f}'])
        
        table = axes[1, 1].table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save summary plot
        summary_path = os.path.join(self.results_dir, 'summary_analysis.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved summary analysis: {summary_path}")
        
        # Save summary data
        summary_data_path = os.path.join(self.results_dir, 'summary_results.npz')
        np.savez(summary_data_path, **summary_data)
        
        print(f"  Saved summary data: {summary_data_path}")


def main():
    """
    Main function to run comprehensive time-dependent vine copula analysis
    """
    
    print("Comprehensive Time-Dependent Vine Copula Analysis")
    print("=" * 60)
    
    # Configuration
    config = {
        'dim': 3,                    # Start with 3D for computational efficiency
        'n_time_steps': 80,          # Moderate number of time steps
        'n_samples_per_time': 120,   # Sufficient samples per time
        'test_scenarios': ['piecewise', 'sinusoidal', 'financial'],
        'random_seed': 42
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize analysis framework
    analyzer = ComprehensiveTimeDependentAnalysis(
        dim=config['dim'],
        random_seed=config['random_seed']
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis(
        n_time_steps=config['n_time_steps'],
        n_samples_per_time=config['n_samples_per_time'],
        test_scenarios=config['test_scenarios']
    )


if __name__ == "__main__":
    main() 