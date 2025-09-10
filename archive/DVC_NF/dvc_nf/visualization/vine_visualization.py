#!/usr/bin/env python3
"""
Advanced Vine Visualization Utilities for DVC-NF

Integrates advanced visualization capabilities including:
1. R-vine adjacency graphs for each tree level
2. 2D copula visualizations with scatter, hexbin, and KDE contours
3. Temporal interaction analysis and animation
4. Professional publication-quality plots

Enhanced for time-dependent vine copula analysis.


"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import itertools
import os
from typing import List, Tuple, Optional, Dict, Any

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class VineVisualizer:
    """
    Advanced visualization class for time-dependent vine copulas
    """
    
    def __init__(self, results_dir=None):
        """
        Initialize vine visualizer
        
        Parameters:
        -----------
        results_dir : str, optional
            Directory to save plots. If None, creates default.
        """
        if results_dir is None:
            self.results_dir = os.path.join(os.getcwd(), 'results', 'vine_visualization')
        else:
            self.results_dir = results_dir
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Professional color schemes
        self.scenario_colors = {
            'ising': '#E74C3C',      # Red
            'hmm': '#3498DB',        # Blue  
            'loglinear': '#2ECC71',  # Green
            'spatiotemporal': '#9B59B6',  # Purple
            'piecewise': '#F39C12',  # Orange
            'sinusoidal': '#1ABC9C', # Teal
            'financial': '#34495E',  # Dark Gray
            'block_switching': '#E67E22',  # Orange
            'beyond_pairwise': '#8E44AD'   # Purple
        }
    
    def create_comprehensive_analysis(self, all_results, save_plots=True):
        """
        Create comprehensive visualization analysis for all scenarios
        
        Parameters:
        -----------
        all_results : dict
            Results from multiple scenarios
        save_plots : bool
            Whether to save plots to disk
        """
        
        print("🎨 Creating comprehensive visualization analysis...")
        
        # 1. Create R-vine structure analysis
        self._create_rvine_structure_analysis(all_results, save_plots)
        
        # 2. Create 2D copula analysis
        self._create_2d_copula_analysis(all_results, save_plots)
        
        # 3. Create temporal interaction analysis
        self._create_temporal_interaction_analysis(all_results, save_plots)
        
        # 4. Create advanced comparative analysis
        self._create_advanced_comparative_analysis(all_results, save_plots)
        
        print(f"✅ Comprehensive analysis saved to: {self.results_dir}")
    
    def _create_rvine_structure_analysis(self, all_results, save_plots):
        """Create R-vine structure analysis for different scenarios"""
        
        print("  📊 Creating R-vine structure analysis...")
        
        # Create synthetic R-vine structures for demonstration
        # In practice, these would come from fitted vine models
        scenario_names = list(all_results.keys())
        n_scenarios = len(scenario_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('R-Vine Structure Analysis Across Scenarios', fontsize=16, fontweight='bold')
        
        for idx, scenario in enumerate(scenario_names[:4]):  # Show first 4 scenarios
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            data = all_results[scenario]['data']
            metadata = all_results[scenario]['metadata']
            dim = data.shape[2]
            
            # Create synthetic R-vine structure based on scenario characteristics
            adjacency_list = self._generate_scenario_vine_structure(scenario, metadata, dim)
            
            # Plot vine graph
            self._plot_single_vine_graph(adjacency_list, ax, scenario, dim)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, 'rvine_structure_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_2d_copula_analysis(self, all_results, save_plots):
        """Create 2D copula analysis for variable pairs"""
        
        print("  📊 Creating 2D copula analysis...")
        
        scenario_names = list(all_results.keys())
        n_scenarios = len(scenario_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('2D Copula Analysis: Variable Relationships', fontsize=16, fontweight='bold')
        
        for idx, scenario in enumerate(scenario_names[:4]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            data = all_results[scenario]['data']
            
            # Transform data to uniform margins (pseudo-copula)
            u1, u2 = self._transform_to_copula_data(data, var1=0, var2=1)
            
            # Plot 2D copula
            self._plot_2d_copula_enhanced(u1, u2, ax, scenario)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, '2d_copula_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_temporal_interaction_analysis(self, all_results, save_plots):
        """Create temporal interaction analysis"""
        
        print("  📊 Creating temporal interaction analysis...")
        
        # Create line plot version
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Interaction Evolution', fontsize=16, fontweight='bold')
        
        scenario_names = list(all_results.keys())
        
        for idx, scenario in enumerate(scenario_names[:4]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            data = all_results[scenario]['data']
            times = all_results[scenario]['times']
            
            # Compute temporal interactions (correlations, entropy, etc.)
            interactions = self._compute_temporal_interactions(data, times)
            
            # Plot temporal evolution
            self._plot_temporal_interactions_enhanced(interactions, times, ax, scenario)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, 'temporal_interactions_line.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create heatmap version
        self._create_temporal_heatmap_analysis(all_results, save_plots)
    
    def _create_temporal_heatmap_analysis(self, all_results, save_plots):
        """Create temporal heatmap analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Interaction Heatmaps', fontsize=16, fontweight='bold')
        
        scenario_names = list(all_results.keys())
        
        for idx, scenario in enumerate(scenario_names[:4]):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            data = all_results[scenario]['data']
            times = all_results[scenario]['times']
            
            # Compute interaction matrix over time
            interaction_matrix = self._compute_interaction_matrix(data, times)
            
            # Plot heatmap
            self._plot_temporal_heatmap_enhanced(interaction_matrix, ax, scenario)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, 'temporal_interactions_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_advanced_comparative_analysis(self, all_results, save_plots):
        """Create advanced comparative analysis across scenarios"""
        
        print("  📊 Creating advanced comparative analysis...")
        
        # Multi-panel comparative figure
        fig = plt.figure(figsize=(20, 16))
        
        # Create complex subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Correlation structure comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_correlation_structure_comparison(all_results, ax1)
        
        # 2. Entropy evolution comparison  
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_entropy_evolution_comparison(all_results, ax2)
        
        # 3. Complexity measures
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_complexity_measures(all_results, ax3)
        
        # 4. Data distribution comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_data_distribution_comparison(all_results, ax4)
        
        # 5. Performance metrics
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_performance_metrics(all_results, ax5)
        
        # 6. Network analysis
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_network_analysis_comparison(all_results, ax6)
        
        fig.suptitle('Advanced Comparative Analysis: Time-Dependent Vine Copulas', 
                    fontsize=18, fontweight='bold')
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, 'advanced_comparative_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()


def plot_rvine_graphs(r_matrix, adjacency_list_per_level, 
                      node_labels=None, 
                      figsize=(16,10),
                      title="R-Vine Structure"):
    """
    Plot each level of an R-vine as a graph with enhanced styling
    
    Parameters:
    -----------
    r_matrix : np.ndarray
        The R-matrix for the vine
    adjacency_list_per_level : list of lists
        List of edges for each tree level
    node_labels : list of str, optional
        Custom labels for nodes
    figsize : tuple
        Figure size
    title : str
        Figure title
    """
    
    n_levels = len(adjacency_list_per_level)
    fig, axes = plt.subplots(1, n_levels, figsize=figsize, squeeze=False)
    axes = axes[0]

    # Node labels
    num_nodes = r_matrix.shape[0] if r_matrix is not None else None
    if node_labels is None and num_nodes is not None:
        node_labels = [f"X{i+1}" for i in range(num_nodes)]
    
    # Color scheme for different tree levels
    level_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
    
    for t in range(n_levels):
        ax = axes[t]
        edges = adjacency_list_per_level[t]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        if num_nodes is not None:
            G.add_nodes_from(range(num_nodes))
        else:
            node_set = set()
            for e in edges:
                node_set.update([e[0], e[1]])
            G.add_nodes_from(node_set)
        
        # Add edges with weights
        for e in edges:
            if len(e) == 3:
                i, j, val = e
                G.add_edge(i, j, weight=val)
            else:
                i, j = e
                G.add_edge(i, j, weight=1.0)
        
        # Enhanced layout
        if num_nodes <= 6:
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=2.0, iterations=50)

        # Enhanced visualization
        color = level_colors[t % len(level_colors)]
        
        # Edge widths based on weights
        widths = []
        for (u, v, w) in G.edges(data='weight'):
            if w is not None:
                widths.append(1.0 + 4.0 * abs(float(w)))
            else:
                widths.append(2.0)

        # Draw network with enhanced styling
        nx.draw_networkx_nodes(G, pos, node_color=color, node_size=800, 
                              alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.6, 
                              edge_color='gray', ax=ax)
        
        # Labels
        if node_labels and len(node_labels) >= num_nodes:
            label_dict = {i: node_labels[i] for i in range(num_nodes)}
            nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=12, 
                                   font_weight='bold', ax=ax)

        # Edge labels
        edge_labels = {}
        for (u, v, w) in G.edges(data='weight'):
            if w is not None and w != 1.0:
                edge_labels[(u, v)] = f"{float(w):.2f}"
        
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                        font_color='red', ax=ax, font_size=10)
        
        ax.set_title(f"Tree Level {t}", fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_2d_copula(u1, u2, 
                   bins=50, 
                   scatter_sample=2000,
                   title="2D Copula",
                   ax=None):
    """
    Enhanced 2D copula visualization
    
    Parameters:
    -----------
    u1, u2 : array-like
        Data in [0,1] representing copula variables
    bins : int
        Number of bins for hexbin/histogram
    scatter_sample : int
        Number of points for scatter plot
    title : str
        Plot title
    ax : matplotlib axis, optional
        Axis to plot on
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
        show_plot = True
    else:
        show_plot = False

    # Subsample if needed
    N = len(u1)
    if N > scatter_sample:
        idx = np.random.choice(N, scatter_sample, replace=False)
        x_sub = u1[idx]
        y_sub = u2[idx]
    else:
        x_sub = u1
        y_sub = u2
    
    # Hexbin plot
    hb = ax.hexbin(x_sub, y_sub, gridsize=bins, cmap='Blues', 
                   extent=(0, 1, 0, 1), alpha=0.8)
    
    # KDE contours
    try:
        sns.kdeplot(x=x_sub, y=y_sub, levels=5, colors='red', 
                   alpha=0.6, ax=ax, linewidths=2)
    except:
        pass  # Skip if KDE fails
    
    # Independence line (diagonal)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Independence')
    
    # Styling
    ax.set_xlabel("U₁", fontsize=12)
    ax.set_ylabel("U₂", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    if show_plot:
        plt.colorbar(hb, ax=ax, label='Density')
        plt.show()


def plot_temporal_interactions(interaction_array, 
                               time_indices=None,
                               edge_names=None,
                               title="Temporal Interactions",
                               figsize=(12, 7),
                               ax=None):
    """
    Enhanced temporal interaction visualization
    
    Parameters:
    -----------
    interaction_array : np.ndarray
        Shape (n_time, n_edges) containing interaction values
    time_indices : array-like, optional
        Time indices
    edge_names : list, optional
        Names for each edge
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : matplotlib axis, optional
        Axis to plot on
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False
    
    n_time, n_edges = interaction_array.shape
    if time_indices is None:
        time_indices = np.arange(n_time)

    # Color palette
    colors = sns.color_palette("husl", n_edges)
    
    # Plot lines for each edge
    for e in range(n_edges):
        label = edge_names[e] if edge_names else f"Edge {e+1}"
        ax.plot(time_indices, interaction_array[:, e], 
               color=colors[e], linewidth=2.5, alpha=0.8, 
               marker='o', markersize=3, label=label)

    # Styling
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Interaction Strength", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if n_edges <= 10:
        ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add trend line if requested
    if n_edges == 1:
        z = np.polyfit(time_indices, interaction_array[:, 0], 1)
        p = np.poly1d(z)
        ax.plot(time_indices, p(time_indices), "r--", alpha=0.6, linewidth=2)
    
    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_temporal_interactions_heatmap(interaction_array, 
                                       time_indices=None,
                                       edge_names=None,
                                       title="Temporal Interactions (Heatmap)",
                                       figsize=(10, 7),
                                       ax=None):
    """
    Enhanced temporal interaction heatmap
    
    Parameters:
    -----------
    interaction_array : np.ndarray
        Shape (n_time, n_edges) containing interaction values
    time_indices : array-like, optional
        Time indices
    edge_names : list, optional
        Names for each edge
    title : str
        Plot title
    figsize : tuple
        Figure size
    ax : matplotlib axis, optional
        Axis to plot on
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_plot = True
    else:
        show_plot = False
    
    n_time, n_edges = interaction_array.shape
    if time_indices is None:
        time_indices = np.arange(n_time)
    
    # Create heatmap
    im = ax.imshow(interaction_array.T, aspect='auto', cmap='RdBu_r', 
                   origin='lower', interpolation='nearest')
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Index", fontsize=12)
    ax.set_ylabel("Edge Index", fontsize=12)
    
    # Ticks and labels
    if edge_names is not None and len(edge_names) <= 20:
        ax.set_yticks(range(n_edges))
        ax.set_yticklabels(edge_names, fontsize=10)
    
    # Time labels (subsample if too many)
    if n_time <= 20:
        ax.set_xticks(range(n_time))
        ax.set_xticklabels([f"{t:.1f}" for t in time_indices], rotation=45)
    
    # Colorbar
    if show_plot:
        plt.colorbar(im, ax=ax, label='Interaction Strength')
        plt.tight_layout()
        plt.show()


# Helper methods for VineVisualizer class
def _generate_scenario_vine_structure(scenario, metadata, dim):
    """Generate synthetic vine structure based on scenario characteristics"""
    
    adjacency_list = []
    
    if scenario == 'ising':
        # Chain-like structure for Ising model
        edges = []
        for i in range(dim - 1):
            strength = np.random.uniform(0.3, 0.8)
            edges.append((i, i+1, strength))
        adjacency_list.append(edges)
        
    elif scenario == 'hmm':
        # Star-like structure for HMM
        edges = []
        for i in range(1, dim):
            strength = np.random.uniform(0.4, 0.9)
            edges.append((0, i, strength))
        adjacency_list.append(edges)
        
    elif scenario == 'loglinear':
        # Random structure for log-linear
        edges = []
        n_edges = min(dim, 4)
        pairs = [(i, j) for i in range(dim) for j in range(i+1, dim)]
        selected_pairs = np.random.choice(len(pairs), n_edges, replace=False)
        for idx in selected_pairs:
            i, j = pairs[idx]
            strength = np.random.uniform(0.2, 0.7)
            edges.append((i, j, strength))
        adjacency_list.append(edges)
        
    else:
        # Default structure
        edges = []
        for i in range(min(dim-1, 3)):
            strength = np.random.uniform(0.3, 0.8)
            edges.append((i, i+1, strength))
        adjacency_list.append(edges)
    
    return adjacency_list


# Additional helper methods for VineVisualizer class
class VineVisualizerHelpers:
    """Helper methods for VineVisualizer"""
    
    @staticmethod
    def _plot_single_vine_graph(adjacency_list, ax, scenario, dim):
        """Plot a single vine graph on given axis"""
        
        edges = adjacency_list[0] if len(adjacency_list) > 0 else []
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(range(dim))
        
        # Add edges
        for e in edges:
            if len(e) == 3:
                i, j, val = e
                G.add_edge(i, j, weight=val)
            else:
                i, j = e
                G.add_edge(i, j, weight=1.0)
        
        # Layout
        pos = nx.circular_layout(G) if dim <= 6 else nx.spring_layout(G, seed=42)
        
        # Colors
        scenario_colors = {
            'ising': '#E74C3C', 'hmm': '#3498DB', 'loglinear': '#2ECC71', 
            'spatiotemporal': '#9B59B6', 'piecewise': '#F39C12',
            'sinusoidal': '#1ABC9C', 'financial': '#34495E'
        }
        color = scenario_colors.get(scenario, '#95A5A6')
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=color, node_size=600, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title(f'{scenario.title()} Structure', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    @staticmethod
    def _transform_to_copula_data(data, var1=0, var2=1):
        """Transform data to uniform margins for copula analysis"""
        
        # Flatten time series
        flat_data = data.reshape(-1, data.shape[2])
        
        # Extract variables
        x1 = flat_data[:, var1]
        x2 = flat_data[:, var2]
        
        # Transform to uniform using empirical CDF
        from scipy.stats import rankdata
        
        u1 = rankdata(x1) / (len(x1) + 1)
        u2 = rankdata(x2) / (len(x2) + 1)
        
        return u1, u2
    
    @staticmethod
    def _plot_2d_copula_enhanced(u1, u2, ax, scenario):
        """Enhanced 2D copula plot"""
        
        plot_2d_copula(u1, u2, title=f'{scenario.title()} Copula', ax=ax)
    
    @staticmethod
    def _compute_temporal_interactions(data, times):
        """Compute temporal interactions (correlations between variable pairs)"""
        
        n_time, n_samples, dim = data.shape
        n_pairs = dim * (dim - 1) // 2
        
        interactions = np.zeros((n_time, n_pairs))
        
        pair_idx = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                for t in range(n_time):
                    if n_samples > 1:
                        corr = np.corrcoef(data[t, :, i], data[t, :, j])[0, 1]
                        interactions[t, pair_idx] = corr
                    else:
                        interactions[t, pair_idx] = 0
                pair_idx += 1
        
        return interactions
    
    @staticmethod
    def _plot_temporal_interactions_enhanced(interactions, times, ax, scenario):
        """Enhanced temporal interactions plot"""
        
        # Generate edge names
        dim = int((1 + np.sqrt(1 + 8 * interactions.shape[1])) / 2)
        edge_names = []
        for i in range(dim):
            for j in range(i + 1, dim):
                edge_names.append(f'({i},{j})')
        
        plot_temporal_interactions(interactions, times, edge_names[:interactions.shape[1]], 
                                 f'{scenario.title()} Interactions', ax=ax)
    
    @staticmethod
    def _compute_interaction_matrix(data, times):
        """Compute interaction matrix over time"""
        
        interactions = VineVisualizerHelpers._compute_temporal_interactions(data, times)
        return interactions.T  # Transpose for heatmap (edges x time)
    
    @staticmethod
    def _plot_temporal_heatmap_enhanced(interaction_matrix, ax, scenario):
        """Enhanced temporal heatmap"""
        
        plot_temporal_interactions_heatmap(interaction_matrix.T, 
                                         title=f'{scenario.title()} Heatmap', ax=ax)
    
    @staticmethod
    def _plot_correlation_structure_comparison(all_results, ax):
        """Plot correlation structure comparison"""
        
        scenario_names = list(all_results.keys())
        colors = sns.color_palette("husl", len(scenario_names))
        
        for idx, scenario in enumerate(scenario_names):
            data = all_results[scenario]['data']
            times = all_results[scenario]['times']
            
            # Compute mean correlation over time
            correlations = []
            for t in range(len(times)):
                if data.shape[2] >= 2 and data.shape[1] > 1:
                    corr = np.corrcoef(data[t, :, 0], data[t, :, 1])[0, 1]
                    correlations.append(abs(corr))
                else:
                    correlations.append(0)
            
            ax.plot(times, correlations, color=colors[idx], linewidth=2, 
                   label=scenario.title(), alpha=0.8)
        
        ax.set_title('Correlation Structure Evolution', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('|Correlation|')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_entropy_evolution_comparison(all_results, ax):
        """Plot entropy evolution comparison"""
        
        scenario_names = list(all_results.keys())
        entropies = []
        labels = []
        
        for scenario in scenario_names:
            data = all_results[scenario]['data']
            
            # Compute empirical entropy (simplified)
            flat_data = data.reshape(-1, data.shape[2])
            entropy = -np.sum(np.log(np.var(flat_data, axis=0) + 1e-8))
            entropies.append(entropy)
            labels.append(scenario.title())
        
        ax.bar(range(len(entropies)), entropies, color=sns.color_palette("husl", len(entropies)))
        ax.set_title('Entropy Comparison', fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Entropy')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_complexity_measures(all_results, ax):
        """Plot complexity measures"""
        
        scenario_names = list(all_results.keys())
        complexities = []
        
        for scenario in scenario_names:
            data = all_results[scenario]['data']
            metadata = all_results[scenario]['metadata']
            
            # Different complexity measures based on scenario type
            if metadata.get('type') == 'ising_time_series':
                complexity = metadata['coupling_stats']['pairwise_coupling_mean']
            elif metadata.get('type') == 'hmm_regimes':
                complexity = metadata['regime_stats']['regime_switches'] / len(all_results[scenario]['times'])
            else:
                # Use temporal variance as complexity measure
                complexity = np.var(np.mean(data, axis=1))
            
            complexities.append(complexity)
        
        ax.bar(range(len(complexities)), complexities, 
               color=sns.color_palette("viridis", len(complexities)))
        ax.set_title('Complexity Measures', fontweight='bold')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
        ax.set_ylabel('Complexity')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_data_distribution_comparison(all_results, ax):
        """Plot data distribution comparison"""
        
        scenario_names = list(all_results.keys())
        colors = sns.color_palette("husl", len(scenario_names))
        
        for idx, scenario in enumerate(scenario_names):
            data = all_results[scenario]['data']
            flat_data = data.flatten()
            
            # Create histogram
            ax.hist(flat_data, bins=30, alpha=0.6, color=colors[idx], 
                   label=scenario.title(), density=True)
        
        ax.set_title('Data Distribution Comparison', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_performance_metrics(all_results, ax):
        """Plot performance metrics"""
        
        scenario_names = list(all_results.keys())
        metrics = []
        
        for scenario in scenario_names:
            data = all_results[scenario]['data']
            
            # Simple performance metric: coefficient of variation
            mean_val = np.mean(data)
            std_val = np.std(data)
            cv = std_val / (abs(mean_val) + 1e-8)
            metrics.append(cv)
        
        ax.bar(range(len(metrics)), metrics, 
               color=sns.color_palette("plasma", len(metrics)))
        ax.set_title('Coefficient of Variation', fontweight='bold')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
        ax.set_ylabel('CV')
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_network_analysis_comparison(all_results, ax):
        """Plot network analysis comparison"""
        
        scenario_names = list(all_results.keys())
        
        # Create network metrics comparison
        network_data = []
        for scenario in scenario_names:
            data = all_results[scenario]['data']
            
            # Simple network metric: mean absolute correlation
            if data.shape[2] >= 2:
                flat_data = data.reshape(-1, data.shape[2])
                corr_matrix = np.corrcoef(flat_data.T)
                mean_abs_corr = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
                network_data.append(mean_abs_corr)
            else:
                network_data.append(0)
        
        # Bar plot
        bars = ax.bar(range(len(network_data)), network_data, 
                     color=sns.color_palette("coolwarm", len(network_data)))
        
        ax.set_title('Network Connectivity (Mean |Correlation|)', fontweight='bold')
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels([s.title() for s in scenario_names], rotation=45, ha='right')
        ax.set_ylabel('Mean |Correlation|')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, network_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')


# Attach helper methods to VineVisualizer class
for method_name in dir(VineVisualizerHelpers):
    if not method_name.startswith('__'):
        setattr(VineVisualizer, method_name, getattr(VineVisualizerHelpers, method_name))


# Export the enhanced functions
__all__ = [
    'VineVisualizer',
    'plot_rvine_graphs', 
    'plot_2d_copula',
    'plot_temporal_interactions',
    'plot_temporal_interactions_heatmap'
] 