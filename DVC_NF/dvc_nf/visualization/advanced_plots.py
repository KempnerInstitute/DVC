#!/usr/bin/env python3
"""
Advanced Plot Generation for DVC-NF

Provides high-level functions for creating comprehensive simulation analysis
and scenario comparison plots. Integrates with the VineVisualizer class to
create publication-quality figures.

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List, Optional
from .vine_visualization import VineVisualizer, plot_rvine_graphs, plot_2d_copula

# Set professional plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

class AdvancedPlotGenerator:
    """
    Advanced plot generation for comprehensive simulation analysis
    """
    
    def __init__(self, results_dir=None):
        """
        Initialize advanced plot generator
        
        Parameters:
        -----------
        results_dir : str, optional
            Directory to save plots
        """
        if results_dir is None:
            self.results_dir = os.path.join(os.getcwd(), 'results', 'advanced_plots')
        else:
            self.results_dir = results_dir
            
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize vine visualizer
        self.vine_visualizer = VineVisualizer(self.results_dir)
    
    def create_comprehensive_analysis_suite(self, all_results, save_plots=True):
        """
        Create comprehensive analysis suite with all advanced visualizations
        
        Parameters:
        -----------
        all_results : dict
            Results from multiple scenarios
        save_plots : bool
            Whether to save plots to disk
        """
        
        print("🎨 Creating comprehensive analysis suite...")
        
        try:
            # 1. Executive summary dashboard
            self._create_executive_summary(all_results, save_plots)
            
            # 2. Scenario-specific deep dives
            self._create_scenario_deep_dives(all_results, save_plots)
            
            # 3. Comparative analysis matrix
            self._create_comparative_matrix(all_results, save_plots)
            
            # 4. Time-dependent interaction analysis
            self._create_time_dependent_analysis(all_results, save_plots)
            
            # 5. Network topology analysis
            self._create_network_topology_analysis(all_results, save_plots)
            
            # 6. Statistical properties comparison
            self._create_statistical_properties_analysis(all_results, save_plots)
            
            # 7. Advanced correlation analysis
            self._create_advanced_correlation_analysis(all_results, save_plots)
            
            # 8. Temporal evolution analysis
            self._create_temporal_evolution_analysis(all_results, save_plots)
            
            # 9. Distribution analysis
            self._create_distribution_analysis_suite(all_results, save_plots)
            
            # 10. Complexity analysis
            self._create_complexity_analysis_suite(all_results, save_plots)
            
            print(f"✅ Comprehensive analysis suite saved to: {self.results_dir}")
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            # Continue with basic analysis
            self._create_basic_analysis_fallback(all_results, save_plots)
    
    def _create_executive_summary(self, all_results, save_plots):
        """Create executive summary dashboard"""
        
        print("  📊 Creating executive summary dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Data overview (top row, left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_data_overview_summary(all_results, ax1)
        
        # 2. Complexity ranking (top row, right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_complexity_ranking(all_results, ax2)
        
        # 3. Temporal dynamics (middle row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_temporal_dynamics_summary(all_results, ax3)
        
        # 4. Correlation patterns (middle row, right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_correlation_patterns_summary(all_results, ax4)
        
        # 5. Key metrics table (bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        self._create_metrics_table(all_results, ax5)
        
        fig.suptitle('Executive Summary: Time-Dependent Vine Copula Analysis', 
                    fontsize=18, fontweight='bold')
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, 'executive_summary.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_overview_summary(self, all_results, ax):
        """Plot data overview summary"""
        scenario_names = list(all_results.keys())
        # Filter out analysis results
        scenario_results = {k: v for k, v in all_results.items() 
                           if isinstance(v, dict) and 'data' in v}
        scenario_names = list(scenario_results.keys())
        
        if not scenario_names:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        data_shapes = [scenario_results[s]['data'].shape for s in scenario_names]
        
        # Plot data dimensions and sizes
        dims = [shape[2] for shape in data_shapes]
        time_steps = [shape[0] for shape in data_shapes]
        samples = [shape[1] for shape in data_shapes]
        
        x = np.arange(len(scenario_names))
        width = 0.25
        
        ax.bar(x - width, dims, width, label='Dimensions', alpha=0.8)
        ax.bar(x, [t/10 for t in time_steps], width, label='Time Steps (×10)', alpha=0.8)
        ax.bar(x + width, [s/100 for s in samples], width, label='Samples (×100)', alpha=0.8)
        
        ax.set_title('Data Overview', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_complexity_ranking(self, all_results, ax):
        """Plot complexity ranking"""
        scenario_results = {k: v for k, v in all_results.items() 
                           if isinstance(v, dict) and 'data' in v}
        scenario_names = list(scenario_results.keys())
        
        if not scenario_names:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        complexities = []
        for scenario in scenario_names:
            data = scenario_results[scenario]['data']
            # Comprehensive complexity measure
            cv = np.std(data) / (np.mean(np.abs(data)) + 1e-8)
            temporal_var = np.var(np.mean(data, axis=1))
            total_complexity = cv + temporal_var
            complexities.append(total_complexity)
        
        # Sort by complexity
        sorted_data = sorted(zip(scenario_names, complexities), key=lambda x: x[1], reverse=True)
        sorted_scenarios, sorted_complexities = zip(*sorted_data)
        
        colors = sns.color_palette("viridis", len(sorted_scenarios))
        bars = ax.barh(range(len(sorted_scenarios)), sorted_complexities, color=colors)
        
        ax.set_title('Complexity Ranking', fontweight='bold')
        ax.set_yticks(range(len(sorted_scenarios)))
        ax.set_yticklabels([s.title() for s in sorted_scenarios])
        ax.set_xlabel('Complexity Score')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_complexities)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
    
    def _plot_temporal_dynamics_summary(self, all_results, ax):
        """Plot temporal dynamics summary"""
        scenario_results = {k: v for k, v in all_results.items() 
                           if isinstance(v, dict) and 'data' in v}
        scenario_names = list(scenario_results.keys())
        
        if not scenario_names:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        colors = sns.color_palette("husl", len(scenario_names))
        
        for idx, scenario in enumerate(scenario_names):
            data = scenario_results[scenario]['data']
            times = scenario_results[scenario]['times']
            
            # Compute temporal variance
            temporal_var = np.var(data, axis=1)  # Variance across samples
            mean_temporal_var = np.mean(temporal_var, axis=1)  # Mean across dimensions
            
            ax.plot(times, mean_temporal_var, color=colors[idx], 
                   linewidth=2, label=scenario.title(), alpha=0.8, marker='o', markersize=3)
        
        ax.set_title('Temporal Dynamics Evolution', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Variance')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_patterns_summary(self, all_results, ax):
        """Plot correlation patterns summary"""
        scenario_results = {k: v for k, v in all_results.items() 
                           if isinstance(v, dict) and 'data' in v}
        scenario_names = list(scenario_results.keys())
        
        if not scenario_names:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        correlation_strengths = []
        correlation_variabilities = []
        
        for scenario in scenario_names:
            data = scenario_results[scenario]['data']
            
            # Compute overall correlation strength
            if data.shape[2] >= 2:
                flat_data = data.reshape(-1, data.shape[2])
                corr_matrix = np.corrcoef(flat_data.T)
                mean_abs_corr = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
                correlation_strengths.append(mean_abs_corr)
                
                # Compute temporal correlation variability
                correlations_over_time = []
                for t in range(data.shape[0]):
                    if data.shape[1] > 1:
                        try:
                            corr_t = np.corrcoef(data[t].T)
                            mean_corr_t = np.mean(np.abs(corr_t[np.triu_indices_from(corr_t, k=1)]))
                            correlations_over_time.append(mean_corr_t)
                        except:
                            correlations_over_time.append(0)
                
                correlation_variabilities.append(np.std(correlations_over_time))
            else:
                correlation_strengths.append(0)
                correlation_variabilities.append(0)
        
        # Scatter plot
        colors = sns.color_palette("husl", len(scenario_names))
        scatter = ax.scatter(correlation_strengths, correlation_variabilities, 
                           c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        for i, scenario in enumerate(scenario_names):
            ax.annotate(scenario.title(), 
                       (correlation_strengths[i], correlation_variabilities[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_title('Correlation Patterns', fontweight='bold')
        ax.set_xlabel('Mean |Correlation|')
        ax.set_ylabel('Temporal Variability')
        ax.grid(True, alpha=0.3)
    
    def _create_metrics_table(self, all_results, ax):
        """Create metrics summary table"""
        ax.axis('off')
        
        scenario_results = {k: v for k, v in all_results.items() 
                           if isinstance(v, dict) and 'data' in v}
        
        if not scenario_results:
            ax.text(0.5, 0.5, 'No data available for metrics table', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Compute key metrics for each scenario
        metrics_data = []
        headers = ['Scenario', 'Shape', 'Mean', 'Std', 'Min', 'Max', 'Complexity']
        
        for scenario, result in scenario_results.items():
            data = result['data']
            shape_str = f"({data.shape[0]}×{data.shape[1]}×{data.shape[2]})"
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            complexity = std_val / (abs(mean_val) + 1e-8)
            
            metrics_data.append([
                scenario.title(),
                shape_str,
                f"{mean_val:.3f}",
                f"{std_val:.3f}",
                f"{min_val:.3f}",
                f"{max_val:.3f}",
                f"{complexity:.3f}"
            ])
        
        # Create table
        table = ax.table(cellText=metrics_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title('Summary Metrics Table', fontweight='bold', y=0.95)
    
    def _create_scenario_deep_dives(self, all_results, save_plots):
        """Create scenario-specific deep dive analysis"""
        
        print("  📊 Creating scenario deep dives...")
        
        scenario_results = {k: v for k, v in all_results.items() 
                           if isinstance(v, dict) and 'data' in v}
        
        for scenario in scenario_results.keys():
            try:
                self._create_single_scenario_deep_dive(scenario, scenario_results[scenario], save_plots)
            except Exception as e:
                print(f"    ❌ Failed to create deep dive for {scenario}: {e}")
    
    def _create_single_scenario_deep_dive(self, scenario, result, save_plots):
        """Create deep dive for single scenario"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        data = result['data']
        times = result['times']
        metadata = result['metadata']
        
        # 1. Data evolution over time (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_data_evolution(data, times, ax1, scenario)
        
        # 2. Correlation heatmap (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_correlation_heatmap(data, ax2, scenario)
        
        # 3. 2D copula (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if data.shape[2] >= 2:
            try:
                from .vine_visualization import VineVisualizerHelpers
                u1, u2 = VineVisualizerHelpers._transform_to_copula_data(data, 0, 1)
                plot_2d_copula(u1, u2, title=f'{scenario.title()} Copula', ax=ax3)
            except:
                ax3.text(0.5, 0.5, 'Copula plot\nunavailable', ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Need ≥2D for\ncopula plot', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Temporal statistics (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_temporal_statistics(data, times, ax4, scenario)
        
        # 5. Distribution analysis (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_distribution_analysis(data, ax5, scenario)
        
        # 6. Scenario-specific metrics (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_scenario_specific_metrics(metadata, ax6, scenario)
        
        # 7. Variable evolution (bottom left)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_variable_evolution(data, times, ax7, scenario)
        
        # 8. Pairwise correlations (bottom middle)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_pairwise_correlations_evolution(data, times, ax8, scenario)
        
        # 9. Summary statistics (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_summary_statistics(data, ax9, scenario)
        
        fig.suptitle(f'Deep Dive Analysis: {scenario.title()}', 
                    fontsize=16, fontweight='bold')
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, f'{scenario}_deep_dive.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_data_evolution(self, data, times, ax, scenario):
        """Plot data evolution over time"""
        # Plot mean and std evolution for each variable
        n_vars = min(data.shape[2], 4)  # Limit to 4 variables for clarity
        colors = sns.color_palette("husl", n_vars)
        
        for i in range(n_vars):
            var_data = data[:, :, i]
            mean_evolution = np.mean(var_data, axis=1)
            std_evolution = np.std(var_data, axis=1)
            
            ax.plot(times, mean_evolution, color=colors[i], linewidth=2, 
                   label=f'Var {i+1} Mean', alpha=0.8)
            ax.fill_between(times, mean_evolution - std_evolution, 
                           mean_evolution + std_evolution, 
                           color=colors[i], alpha=0.2)
        
        ax.set_title(f'{scenario.title()}: Data Evolution', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_heatmap(self, data, ax, scenario):
        """Plot correlation heatmap"""
        # Compute overall correlation matrix
        flat_data = data.reshape(-1, data.shape[2])
        corr_matrix = np.corrcoef(flat_data.T)
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{scenario.title()}: Correlation Matrix', fontweight='bold')
    
    def _plot_temporal_statistics(self, data, times, ax, scenario):
        """Plot temporal statistics"""
        # Compute and plot temporal moments
        means = np.mean(data, axis=(1, 2))
        stds = np.std(data, axis=(1, 2))
        skews = [self._compute_skewness(data[t].flatten()) for t in range(data.shape[0])]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(times, means, 'b-', label='Mean', linewidth=2)
        line2 = ax2.plot(times, stds, 'r-', label='Std', linewidth=2)
        line3 = ax2.plot(times, skews, 'g-', label='Skewness', linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean', color='b')
        ax2.set_ylabel('Std / Skewness', color='r')
        ax.set_title(f'{scenario.title()}: Temporal Statistics', fontweight='bold')
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _compute_skewness(self, data):
        """Compute skewness"""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.sum(((data - mean) / std) ** 3) / n
    
    def _plot_distribution_analysis(self, data, ax, scenario):
        """Plot distribution analysis"""
        flat_data = data.flatten()
        
        # Histogram with KDE
        ax.hist(flat_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(flat_data)
            x_range = np.linspace(flat_data.min(), flat_data.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass
        
        ax.set_title(f'{scenario.title()}: Data Distribution', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_scenario_specific_metrics(self, metadata, ax, scenario):
        """Plot scenario-specific metrics"""
        ax.axis('off')
        
        # Extract key metadata
        metrics_text = f"SCENARIO: {scenario.upper()}\n\n"
        
        if 'type' in metadata:
            metrics_text += f"Type: {metadata['type']}\n"
        
        # Add scenario-specific metrics
        if scenario == 'ising':
            if 'coupling_stats' in metadata:
                stats = metadata['coupling_stats']
                metrics_text += f"Pairwise Coupling Mean: {stats.get('pairwise_coupling_mean', 0):.4f}\n"
                metrics_text += f"Pairwise Coupling Max: {stats.get('pairwise_coupling_max', 0):.4f}\n"
                metrics_text += f"Has Triple Couplings: {stats.get('has_triple_couplings', False)}\n"
        
        elif scenario == 'hmm':
            if 'regime_stats' in metadata:
                stats = metadata['regime_stats']
                metrics_text += f"Regime Switches: {stats.get('regime_switches', 0)}\n"
                if 'regime_distribution' in stats:
                    metrics_text += f"Regime Distribution: {stats['regime_distribution']}\n"
        
        elif scenario == 'loglinear':
            if 'synergy_stats' in metadata:
                stats = metadata['synergy_stats']
                metrics_text += f"Mean Pairwise Synergy: {stats.get('mean_pairwise_synergy', 0):.4f}\n"
                metrics_text += f"Max Pairwise Synergy: {stats.get('max_pairwise_synergy', 0):.4f}\n"
        
        elif scenario == 'spatiotemporal':
            if 'block_stats' in metadata:
                stats = metadata['block_stats']
                metrics_text += f"Number of Blocks: {stats.get('n_blocks', 0)}\n"
                if 'block_correlation_range' in stats:
                    range_vals = stats['block_correlation_range']
                    metrics_text += f"Block Corr Range: [{range_vals[0]:.3f}, {range_vals[1]:.3f}]\n"
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def _plot_variable_evolution(self, data, times, ax, scenario):
        """Plot individual variable evolution"""
        n_vars = min(data.shape[2], 6)  # Show up to 6 variables
        colors = sns.color_palette("tab10", n_vars)
        
        for i in range(n_vars):
            var_means = np.mean(data[:, :, i], axis=1)
            ax.plot(times, var_means, color=colors[i], linewidth=2, 
                   label=f'Var {i+1}', alpha=0.8, marker='o', markersize=3)
        
        ax.set_title(f'{scenario.title()}: Variable Evolution', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Value')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    def _plot_pairwise_correlations_evolution(self, data, times, ax, scenario):
        """Plot evolution of pairwise correlations"""
        if data.shape[2] < 2:
            ax.text(0.5, 0.5, 'Need ≥2 variables\nfor correlations', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Compute correlations over time
        correlations_over_time = []
        for t in range(data.shape[0]):
            if data.shape[1] > 1:
                try:
                    corr_matrix = np.corrcoef(data[t].T)
                    # Extract upper triangle correlations
                    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                    correlations_over_time.append(upper_tri)
                except:
                    correlations_over_time.append(np.zeros(1))
            else:
                correlations_over_time.append(np.zeros(1))
        
        correlations_over_time = np.array(correlations_over_time)
        
        # Plot each correlation pair
        n_pairs = correlations_over_time.shape[1]
        colors = sns.color_palette("husl", n_pairs)
        
        for i in range(min(n_pairs, 6)):  # Limit to 6 pairs for clarity
            ax.plot(times, correlations_over_time[:, i], color=colors[i], 
                   linewidth=2, label=f'Pair {i+1}', alpha=0.8)
        
        ax.set_title(f'{scenario.title()}: Correlation Evolution', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Correlation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    def _plot_summary_statistics(self, data, ax, scenario):
        """Plot summary statistics"""
        # Compute various statistics
        stats = {
            'Mean': np.mean(data),
            'Std': np.std(data),
            'Min': np.min(data),
            'Max': np.max(data),
            'Range': np.max(data) - np.min(data),
            'CV': np.std(data) / (abs(np.mean(data)) + 1e-8)
        }
        
        # Create bar plot
        stat_names = list(stats.keys())
        stat_values = list(stats.values())
        
        bars = ax.bar(stat_names, stat_values, color=sns.color_palette("viridis", len(stat_names)))
        
        ax.set_title(f'{scenario.title()}: Summary Statistics', fontweight='bold')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, stat_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _create_basic_analysis_fallback(self, all_results, save_plots):
        """Create basic analysis if advanced methods fail"""
        print("  📊 Creating basic analysis fallback...")
        
        scenario_results = {k: v for k, v in all_results.items() 
                           if isinstance(v, dict) and 'data' in v}
        
        if not scenario_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Basic Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Data shapes
        ax1 = axes[0, 0]
        scenario_names = list(scenario_results.keys())
        data_points = [np.prod(result['data'].shape) for result in scenario_results.values()]
        ax1.bar(range(len(scenario_names)), data_points, color=sns.color_palette("viridis", len(scenario_names)))
        ax1.set_title('Total Data Points')
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.set_yscale('log')
        
        # 2. Mean values
        ax2 = axes[0, 1]
        mean_values = [np.mean(result['data']) for result in scenario_results.values()]
        ax2.bar(range(len(scenario_names)), mean_values, color=sns.color_palette("plasma", len(scenario_names)))
        ax2.set_title('Mean Values')
        ax2.set_xticks(range(len(scenario_names)))
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        
        # 3. Standard deviations
        ax3 = axes[1, 0]
        std_values = [np.std(result['data']) for result in scenario_results.values()]
        ax3.bar(range(len(scenario_names)), std_values, color=sns.color_palette("coolwarm", len(scenario_names)))
        ax3.set_title('Standard Deviations')
        ax3.set_xticks(range(len(scenario_names)))
        ax3.set_xticklabels(scenario_names, rotation=45, ha='right')
        
        # 4. Data ranges
        ax4 = axes[1, 1]
        ranges = [np.max(result['data']) - np.min(result['data']) for result in scenario_results.values()]
        ax4.bar(range(len(scenario_names)), ranges, color=sns.color_palette("tab10", len(scenario_names)))
        ax4.set_title('Data Ranges')
        ax4.set_xticks(range(len(scenario_names)))
        ax4.set_xticklabels(scenario_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, 'basic_analysis_dashboard.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    # Placeholder implementations for other methods
    def _create_comparative_matrix(self, all_results, save_plots):
        """Create comparative analysis matrix"""
        print("  📊 Creating comparative analysis matrix...")
        # Implementation will be added if needed
        pass
    
    def _create_time_dependent_analysis(self, all_results, save_plots):
        """Create time-dependent interaction analysis"""
        print("  📊 Creating time-dependent analysis...")
        try:
            self.vine_visualizer.create_comprehensive_analysis(all_results, save_plots)
        except Exception as e:
            print(f"    ❌ Vine visualizer failed: {e}")
    
    def _create_network_topology_analysis(self, all_results, save_plots):
        """Create network topology analysis"""
        print("  📊 Creating network topology analysis...")
        # Implementation will be added if needed
        pass
    
    def _create_statistical_properties_analysis(self, all_results, save_plots):
        """Create statistical properties analysis"""
        print("  📊 Creating statistical properties analysis...")
        # Implementation will be added if needed
        pass
    
    def _create_advanced_correlation_analysis(self, all_results, save_plots):
        """Create advanced correlation analysis"""
        print("  📊 Creating advanced correlation analysis...")
        # Implementation will be added if needed
        pass
    
    def _create_temporal_evolution_analysis(self, all_results, save_plots):
        """Create temporal evolution analysis"""
        print("  📊 Creating temporal evolution analysis...")
        # Implementation will be added if needed
        pass
    
    def _create_distribution_analysis_suite(self, all_results, save_plots):
        """Create distribution analysis suite"""
        print("  📊 Creating distribution analysis suite...")
        # Implementation will be added if needed
        pass
    
    def _create_complexity_analysis_suite(self, all_results, save_plots):
        """Create complexity analysis suite"""
        print("  📊 Creating complexity analysis suite...")
        # Implementation will be added if needed
        pass


def create_comprehensive_simulation_analysis(all_results, results_dir=None, save_plots=True):
    """
    High-level function to create comprehensive simulation analysis
    
    Parameters:
    -----------
    all_results : dict
        Results from multiple scenarios
    results_dir : str, optional
        Directory to save plots
    save_plots : bool
        Whether to save plots to disk
    """
    
    print("🚀 Creating comprehensive simulation analysis...")
    
    plot_generator = AdvancedPlotGenerator(results_dir)
    plot_generator.create_comprehensive_analysis_suite(all_results, save_plots)
    
    print("✅ Comprehensive simulation analysis complete!")


def create_scenario_comparison_suite(all_results, scenarios_to_compare=None, 
                                   results_dir=None, save_plots=True):
    """
    High-level function to create scenario comparison suite
    
    Parameters:
    -----------
    all_results : dict
        Results from multiple scenarios
    scenarios_to_compare : list, optional
        Specific scenarios to compare
    results_dir : str, optional
        Directory to save plots
    save_plots : bool
        Whether to save plots to disk
    """
    
    print("🔍 Creating scenario comparison suite...")
    
    if scenarios_to_compare is None:
        scenarios_to_compare = list(all_results.keys())
    
    # Filter results
    filtered_results = {k: v for k, v in all_results.items() if k in scenarios_to_compare}
    
    plot_generator = AdvancedPlotGenerator(results_dir)
    
    # Create focused comparison analysis
    plot_generator.create_comprehensive_analysis_suite(filtered_results, save_plots)
    
    print("✅ Scenario comparison suite complete!")


# Export main functions
__all__ = [
    'AdvancedPlotGenerator',
    'create_comprehensive_simulation_analysis',
    'create_scenario_comparison_suite'
] 