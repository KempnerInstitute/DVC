#!/usr/bin/env python3
"""
Comprehensive Vine Copula Algorithms Analysis & Visualization

This script provides:
1. Detailed explanation of exact algorithms used for different vine optimizations
2. Professional visualization of vine structures at different tree levels
3. Analysis of interaction structures and copula depths
4. Performance comparison with detailed algorithm insights

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add DVC_tensorflow to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dvc_tensorflow_dir = os.path.join(project_root, 'src', 'DVC_tensorflow')
sys.path.append(dvc_tensorflow_dir)

from classes.objects import *
from vine_tree.tree_op import *
from param.generate_rvine import *
from pre_proc.preparation import prep_cop
from sampling.vine_sample import *
from scipy.stats import kendalltau

# Results directory
results_dir = os.path.join(current_dir, '..', 'results', 'vine_algorithms_analysis')
os.makedirs(results_dir, exist_ok=True)

class VineAlgorithmAnalyzer:
    """Comprehensive analysis of vine copula algorithms and structures"""
    
    def __init__(self, dim=4):
        self.dim = dim
        
        print("="*80)
        print("VINE COPULA ALGORITHMS: DEEP ANALYSIS & VISUALIZATION")
        print("="*80)
        print(f"• Analyzing {dim}D vine structures")
        print("• Explaining exact algorithms used")
        print("• Creating professional visualizations")
        print("• Showing interaction structures at multiple depths")
        print("="*80)
    
    def explain_algorithms(self):
        """Provide detailed explanation of vine algorithms"""
        
        algorithms_explanation = {
            "Traditional R-vine (Optimal)": {
                "algorithm": "Prim's Minimum Spanning Tree with Kendall's Tau",
                "criterion": "Maximize |τ| (absolute Kendall's tau correlation)",
                "process": [
                    "1. Initialize: V = all variables, Q = empty, start with random variable",
                    "2. For each tree level:",
                    "   - Tree 0: Compute τ(Xi, Xj) for all variable pairs",
                    "   - Higher trees: Compute τ on conditional/transformed data",
                    "   - Select edge (i,j) that maximizes |τ|",
                    "   - Add to spanning tree, update Q and V",
                    "3. Ensure vine structure constraints (parent variables)",
                    "4. Build R-matrix from edge sequence"
                ],
                "complexity": "O(d²) per tree level, O(d³) total",
                "advantages": ["Systematic optimization", "Theoretical foundation"],
                "disadvantages": ["Linear correlation bias", "Local optima"]
            },
            
            "Random R-vine": {
                "algorithm": "Prim's MST with Random Weights",
                "criterion": "Random uniform weights U[-1,1] instead of correlations",
                "process": [
                    "1. Same Prim's algorithm structure as optimal",
                    "2. Replace Kendall's tau with random weights:",
                    "   - Tree 0: τ = random.uniform(-1, 1)",
                    "   - Higher trees: Same random weight generation",
                    "3. Still respects vine structure constraints",
                    "4. Explores structure space randomly"
                ],
                "complexity": "O(d²) per tree level, O(d³) total",
                "advantages": ["Structure space exploration", "Avoids correlation bias"],
                "disadvantages": ["No systematic optimization", "High variance"]
            },
            
            "C-vine (Canonical)": {
                "algorithm": "Fixed Star/Hub Structure",
                "criterion": "Predetermined central variable structure",
                "process": [
                    "1. Choose central variable (usually variable 0)",
                    "2. Tree 0: Connect central variable to all others",
                    "3. Higher trees: Conditional dependencies through center",
                    "4. R-matrix: Lower triangular with specific pattern:",
                    "   r_matrix = tril(tile([d, d-1, ..., 1]))"
                ],
                "complexity": "O(d) - structure is predetermined",
                "advantages": ["Simple structure", "One dominant variable"],
                "disadvantages": ["Assumes hub structure", "Limited flexibility"]
            },
            
            "D-vine (Drawable)": {
                "algorithm": "Fixed Path/Chain Structure", 
                "criterion": "Sequential variable connections",
                "process": [
                    "1. Tree 0: Connect variables in sequence (0-1, 1-2, 2-3, ...)",
                    "2. Tree 1: Connect variables with one gap (0-2|1, 1-3|2, ...)",
                    "3. Higher trees: Longer-range dependencies",
                    "4. R-matrix: Specific pattern for path structure"
                ],
                "complexity": "O(d) - structure is predetermined", 
                "advantages": ["Natural path structure", "Easy interpretation"],
                "disadvantages": ["Assumes sequential dependencies", "Limited flexibility"]
            },
            
            "Entropy-based R-vine": {
                "algorithm": "Prim's MST with Entropy Maximization",
                "criterion": "Maximize H(copula) = -∫∫ c(u,v) log c(u,v) du dv",
                "process": [
                    "1. Initialize: V = all variables, Q = empty, start with random variable",
                    "2. For each tree level:",
                    "   - Transform data to copula domain [0,1]",
                    "   - For each edge (i,j): estimate copula entropy H(Xi, Xj)",
                    "   - Methods: KDE, histogram, or fallback to τ-based proxy",
                    "   - Select edge that maximizes entropy H",
                    "   - Add to spanning tree, update Q and V",
                    "3. Ensure vine structure constraints",
                    "4. Build R-matrix from entropy-optimal edge sequence"
                ],
                "complexity": "O(d²·n) per tree level, O(d³·n) total (n = entropy estimation cost)",
                "advantages": ["Information-theoretic criterion", "Captures nonlinear dependencies", "Principled optimization"],
                "disadvantages": ["Computational overhead", "Entropy estimation complexity", "Numerical stability"]
            }
        }
        
        return algorithms_explanation
    
    def generate_vine_structures(self):
        """Generate different vine structures for visualization"""
        
        print("Generating vine structures...")
        
        # Generate test data
        np.random.seed(42)
        corr_matrix = np.eye(self.dim)
        for i in range(self.dim-1):
            corr_matrix[i, i+1] = 0.7
            corr_matrix[i+1, i] = 0.7
        if self.dim >= 4:
            corr_matrix[0, 2] = 0.5
            corr_matrix[2, 0] = 0.5
        
        data = np.random.multivariate_normal(np.zeros(self.dim), corr_matrix, 1000)
        
        structures = {}
        
        # 1. C-vine structure
        print("Building C-vine structure...")
        c_r_matrix, c_ind_vine, c_nodes, c_edges = prepare_vine('c-vine', self.dim)
        structures['C-vine'] = {
            'r_matrix': c_r_matrix,
            'ind_vine': c_ind_vine,
            'edges': c_edges,
            'type': 'fixed',
            'description': 'Star/hub structure with central variable'
        }
        
        # 2. D-vine structure  
        print("Building D-vine structure...")
        d_r_matrix, d_ind_vine, d_nodes, d_edges = prepare_vine('d-vine', self.dim)
        structures['D-vine'] = {
            'r_matrix': d_r_matrix,
            'ind_vine': d_ind_vine,
            'edges': d_edges,
            'type': 'fixed',
            'description': 'Path/chain structure with sequential connections'
        }
        
        # 3. Optimal R-vine structure
        print("Building Optimal R-vine structure...")
        opt_edges_per_tree = []
        opt_weights_per_tree = []
        opt_ind_vine = []
        
        for tr in range(self.dim-1):
            if tr == 0:
                edges, weights = optimal_tree(data.T, None, opt_ind_vine, tr, rand=False)
            else:
                # For higher trees, we'd need conditional data - simplified here
                edges, weights = optimal_tree(data.T, data.T, opt_ind_vine, tr, rand=False)
            opt_edges_per_tree.append(edges)
            opt_weights_per_tree.append(weights)
            opt_ind_vine.append(edges)
        
        opt_r_matrix, opt_E, opt_nodes = prepare_optimal(self.dim, opt_ind_vine)
        structures['Optimal R-vine'] = {
            'r_matrix': opt_r_matrix,
            'ind_vine': opt_ind_vine,
            'edges_per_tree': opt_edges_per_tree,
            'weights_per_tree': opt_weights_per_tree,
            'type': 'optimized',
            'description': 'Kendall tau optimized using Prim algorithm'
        }
        
        # 4. Random R-vine structure
        print("Building Random R-vine structure...")
        rand_r_matrix, rand_ind_vine, rand_nodes, rand_E = random_r_matrix_gen(self.dim)
        structures['Random R-vine'] = {
            'r_matrix': rand_r_matrix,
            'ind_vine': rand_ind_vine,
            'type': 'random',
            'description': 'Random structure exploration using uniform weights'
        }
        
        return structures, data, corr_matrix
    
    def visualize_vine_structure(self, structure_name, structure_info, ax):
        """Create professional visualization of vine structure"""
        
        r_matrix = structure_info['r_matrix']
        ind_vine = structure_info['ind_vine']
        
        # Clear the axis
        ax.clear()
        ax.set_xlim(-1, self.dim)
        ax.set_ylim(-1, self.dim-1)
        ax.set_aspect('equal')
        
        # Color scheme for different tree levels
        tree_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Plot each tree level
        for tree_level in range(len(ind_vine)):
            edges = ind_vine[tree_level]
            color = tree_colors[tree_level % len(tree_colors)]
            
            # Calculate vertical position for this tree level
            y_pos = self.dim - 2 - tree_level
            
            for edge_idx, edge in enumerate(edges):
                var1, var2 = edge[0], edge[1]
                
                # Calculate positions
                x1 = var1 + 0.1 * tree_level
                x2 = var2 + 0.1 * tree_level
                
                # Draw edge
                ax.plot([x1, x2], [y_pos, y_pos], 
                       color=color, linewidth=3, alpha=0.8,
                       label=f'Tree {tree_level+1}' if edge_idx == 0 else "")
                
                # Draw nodes
                ax.scatter([x1, x2], [y_pos, y_pos], 
                          c=color, s=100, alpha=0.9, edgecolors='black', linewidth=1)
                
                # Add edge labels
                mid_x = (x1 + x2) / 2
                if structure_name == 'Optimal R-vine' and 'weights_per_tree' in structure_info:
                    if tree_level < len(structure_info['weights_per_tree']):
                        weight = structure_info['weights_per_tree'][tree_level][edge_idx]
                        ax.text(mid_x, y_pos + 0.15, f'τ={weight:.2f}', 
                               ha='center', va='bottom', fontsize=8, 
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                # Add variable labels
                ax.text(x1, y_pos - 0.2, f'X{var1+1}', ha='center', va='top', fontsize=10, weight='bold')
                ax.text(x2, y_pos - 0.2, f'X{var2+1}', ha='center', va='top', fontsize=10, weight='bold')
        
        # Customize the plot
        ax.set_title(f'{structure_name}\n{structure_info["description"]}', 
                    fontsize=14, weight='bold', pad=20)
        ax.set_xlabel('Variables', fontsize=12, weight='bold')
        ax.set_ylabel('Tree Levels (Bottom to Top)', fontsize=12, weight='bold')
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Remove grid and ticks for cleaner look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add algorithm info box
        info_text = f"Type: {structure_info['type'].title()}\n"
        if structure_name == 'C-vine':
            info_text += "Central hub: X1\nComplexity: O(d)"
        elif structure_name == 'D-vine':
            info_text += "Path structure\nComplexity: O(d)"
        elif 'R-vine' in structure_name:
            info_text += f"Tree optimization\nComplexity: O(d³)"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    def create_r_matrix_visualization(self, structures):
        """Create professional R-matrix visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, (name, structure) in enumerate(structures.items()):
            ax = axes[idx]
            r_matrix = structure['r_matrix']
            
            # Create heatmap
            im = ax.imshow(r_matrix, cmap='viridis', aspect='equal')
            
            # Add text annotations
            for i in range(self.dim):
                for j in range(self.dim):
                    text = ax.text(j, i, str(r_matrix[i, j]), 
                                  ha="center", va="center", color="white", fontsize=12, weight='bold')
            
            ax.set_title(f'{name} R-Matrix', fontsize=14, weight='bold', pad=10)
            ax.set_xlabel('Column Index', fontsize=12)
            ax.set_ylabel('Row Index', fontsize=12)
            
            # Set ticks
            ax.set_xticks(range(self.dim))
            ax.set_yticks(range(self.dim))
            ax.set_xticklabels([f'C{i}' for i in range(self.dim)])
            ax.set_yticklabels([f'R{i}' for i in range(self.dim)])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Variable Index', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'r_matrices_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ R-matrix visualizations saved")
    
    def create_algorithm_flowchart(self, algorithms_explanation):
        """Create professional algorithm flowchart"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for idx, (alg_name, alg_info) in enumerate(algorithms_explanation.items()):
            ax = axes[idx]
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Title
            ax.text(5, 9.5, alg_name, ha='center', va='center', fontsize=16, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # Algorithm description
            ax.text(5, 8.5, f"Algorithm: {alg_info['algorithm']}", ha='center', va='center', 
                   fontsize=12, weight='bold')
            
            # Criterion
            ax.text(5, 7.8, f"Criterion: {alg_info['criterion']}", ha='center', va='center', 
                   fontsize=11, style='italic')
            
            # Process steps
            y_start = 7
            for i, step in enumerate(alg_info['process'][:4]):  # Limit to 4 steps for space
                ax.text(1, y_start - i*0.8, step, ha='left', va='center', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
            
            # Complexity
            ax.text(5, 3, f"Complexity: {alg_info['complexity']}", ha='center', va='center', 
                   fontsize=11, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
            
            # Advantages and Disadvantages
            adv_text = "Advantages:\n" + "\n".join([f"• {adv}" for adv in alg_info['advantages']])
            dis_text = "Disadvantages:\n" + "\n".join([f"• {dis}" for dis in alg_info['disadvantages']])
            
            ax.text(2, 1.5, adv_text, ha='left', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            ax.text(8, 1.5, dis_text, ha='right', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightpink', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'algorithms_flowchart.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Algorithm flowchart saved")
    
    def create_tree_depth_analysis(self, structures, data, corr_matrix):
        """Analyze and visualize copula interactions at different tree depths"""
        
        print("Analyzing tree depth interactions...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # For each structure, analyze tree-level dependencies
        for struct_idx, (name, structure) in enumerate(list(structures.items())[:2]):  # C-vine and D-vine
            ind_vine = structure['ind_vine']
            
            # Analyze dependencies at each tree level
            for tree_level in range(min(3, len(ind_vine))):  # Show up to 3 trees
                ax = axes[struct_idx, tree_level]
                
                edges = ind_vine[tree_level]
                
                # Calculate empirical correlations for this tree level
                correlations = []
                for edge in edges:
                    var1, var2 = edge[0], edge[1]
                    if var1 < data.shape[1] and var2 < data.shape[1]:
                        corr = np.corrcoef(data[:, var1], data[:, var2])[0, 1]
                        correlations.append(abs(corr))
                
                # Create network graph for this tree level
                G = nx.Graph()
                for i, edge in enumerate(edges):
                    var1, var2 = edge[0], edge[1]
                    weight = correlations[i] if i < len(correlations) else 0.5
                    G.add_edge(f'X{var1+1}', f'X{var2+1}', weight=weight)
                
                # Draw network
                pos = nx.spring_layout(G, seed=42)
                
                # Draw edges with thickness proportional to correlation
                for edge in G.edges():
                    weight = G[edge[0]][edge[1]]['weight']
                    nx.draw_networkx_edges(G, pos, edgelist=[edge], width=weight*5, 
                                         alpha=0.7, edge_color='gray', ax=ax)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                     node_size=1000, alpha=0.8, ax=ax)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
                
                ax.set_title(f'{name}\nTree {tree_level + 1}', fontsize=12, weight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tree_depth_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Tree depth analysis saved")
    
    def create_performance_comparison(self):
        """Create performance comparison based on previous results"""
        
        # Load previous results if available
        results_file = os.path.join(current_dir, '..', 'results', 'comprehensive_analysis', 'comprehensive_results.json')
        
        if os.path.exists(results_file):
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract performance data
            data_types = list(results.keys())
            methods = []
            errors = []
            times = []
            data_type_labels = []
            
            for data_type, data_result in results.items():
                for method, method_result in data_result['vine_results'].items():
                    if method_result['status'] == 'success':
                        methods.append(method)
                        errors.append(method_result['correlation_error'])
                        times.append(method_result['fit_time'])
                        data_type_labels.append(data_type)
            
            df = pd.DataFrame({
                'Method': methods,
                'Error': errors,
                'Time': times,
                'Data_Type': data_type_labels
            })
            
            # Performance by method
            axes[0, 0].boxplot([df[df['Method'] == method]['Error'].values 
                               for method in df['Method'].unique()],
                              labels=[m.replace('_', '\n') for m in df['Method'].unique()])
            axes[0, 0].set_title('Correlation Error by Method', fontsize=14, weight='bold')
            axes[0, 0].set_ylabel('Correlation Error')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Speed by method
            axes[0, 1].boxplot([df[df['Method'] == method]['Time'].values 
                               for method in df['Method'].unique()],
                              labels=[m.replace('_', '\n') for m in df['Method'].unique()])
            axes[0, 1].set_title('Fitting Time by Method', fontsize=14, weight='bold')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Error vs Time scatter
            colors = {'c-vine_matrix': 'blue', 'd-vine_matrix': 'green', 
                     'r-vine_optimal': 'red', 'r-vine_random': 'orange',
                     'r-vine_sequential': 'purple', 'r-vine_entropy': 'brown'}
            
            for method in df['Method'].unique():
                method_data = df[df['Method'] == method]
                axes[1, 0].scatter(method_data['Time'], method_data['Error'], 
                                  label=method.replace('_', ' '), alpha=0.7, s=100,
                                  c=colors.get(method, 'gray'))
            
            axes[1, 0].set_xlabel('Fitting Time (seconds)')
            axes[1, 0].set_ylabel('Correlation Error')
            axes[1, 0].set_title('Performance vs Speed Trade-off', fontsize=14, weight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Algorithm summary
            axes[1, 1].axis('off')
            summary_text = """ALGORITHM PERFORMANCE SUMMARY

Traditional Optimal R-vine:
• Uses Kendall's τ + Prim's MST
• Systematic but correlation-biased
• Variable performance across data types

Random R-vine:
• Explores structure space randomly
• Surprisingly effective on complex data
• High variance in performance

C-vine & D-vine:
• Fixed structures, O(d) complexity
• Reliable baseline performance
• Limited flexibility for complex dependencies

Key Insights:
• No single method dominates all scenarios
• Data complexity determines optimal choice
• Random exploration can outperform optimization"""
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'performance_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Performance comparison saved")
        else:
            print("Previous results not found, skipping performance comparison")
    
    def run_complete_analysis(self):
        """Run complete vine algorithm analysis"""
        
        print("\n" + "="*60)
        print("RUNNING COMPLETE VINE ALGORITHM ANALYSIS")
        print("="*60)
        
        # 1. Explain algorithms
        print("\n1. Analyzing algorithms...")
        algorithms_explanation = self.explain_algorithms()
        
        # 2. Generate vine structures
        print("\n2. Generating vine structures...")
        structures, data, corr_matrix = self.generate_vine_structures()
        
        # 3. Create main vine structure visualization
        print("\n3. Creating vine structure visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for idx, (name, structure) in enumerate(structures.items()):
            self.visualize_vine_structure(name, structure, axes[idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'vine_structures_complete.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create R-matrix visualization
        print("\n4. Creating R-matrix visualizations...")
        self.create_r_matrix_visualization(structures)
        
        # 5. Create algorithm flowchart
        print("\n5. Creating algorithm flowcharts...")
        self.create_algorithm_flowchart(algorithms_explanation)
        
        # 6. Create tree depth analysis
        print("\n6. Creating tree depth analysis...")
        self.create_tree_depth_analysis(structures, data, corr_matrix)
        
        # 7. Create performance comparison
        print("\n7. Creating performance comparison...")
        self.create_performance_comparison()
        
        # 8. Generate comprehensive report
        self.generate_comprehensive_report(algorithms_explanation, structures)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📁 Results saved to: {results_dir}")
        print("📊 Visualizations created:")
        print("  • vine_structures_complete.png - Main vine structure visualization")
        print("  • r_matrices_comparison.png - R-matrix comparisons")
        print("  • algorithms_flowchart.png - Algorithm explanations")
        print("  • tree_depth_analysis.png - Tree-level interaction analysis")
        print("  • performance_comparison.png - Performance analysis")
        print("  • comprehensive_report.txt - Detailed text report")
        print("="*60)
    
    def generate_comprehensive_report(self, algorithms_explanation, structures):
        """Generate comprehensive text report"""
        
        report_path = os.path.join(results_dir, 'comprehensive_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE VINE COPULA ALGORITHMS ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. ALGORITHM EXPLANATIONS\n")
            f.write("-"*30 + "\n\n")
            
            for alg_name, alg_info in algorithms_explanation.items():
                f.write(f"{alg_name}:\n")
                f.write(f"  Algorithm: {alg_info['algorithm']}\n")
                f.write(f"  Criterion: {alg_info['criterion']}\n")
                f.write(f"  Complexity: {alg_info['complexity']}\n")
                f.write(f"  Advantages: {', '.join(alg_info['advantages'])}\n")
                f.write(f"  Disadvantages: {', '.join(alg_info['disadvantages'])}\n\n")
            
            f.write("2. VINE STRUCTURE ANALYSIS\n")
            f.write("-"*30 + "\n\n")
            
            for name, structure in structures.items():
                f.write(f"{name}:\n")
                f.write(f"  Type: {structure['type']}\n")
                f.write(f"  Description: {structure['description']}\n")
                f.write(f"  R-matrix shape: {np.array(structure['r_matrix']).shape}\n")
                f.write(f"  Number of trees: {len(structure['ind_vine'])}\n\n")
            
            f.write("3. KEY INSIGHTS\n")
            f.write("-"*30 + "\n\n")
            f.write("• Traditional optimal R-vine uses systematic Prim's MST with Kendall's tau\n")
            f.write("• Random R-vine explores structure space without optimization bias\n") 
            f.write("• C-vine and D-vine have predetermined structures with O(d) complexity\n")
            f.write("• R-vine optimization has O(d³) complexity due to tree-by-tree construction\n")
            f.write("• No single method dominates - performance depends on data complexity\n")
            f.write("• Random exploration can surprisingly outperform systematic optimization\n")
        
        print("✓ Comprehensive report saved")


def main():
    """Main function"""
    print("Starting comprehensive vine algorithms analysis...")
    
    analyzer = VineAlgorithmAnalyzer(dim=4)
    analyzer.run_complete_analysis()
    
    print("\n🎉 Vine algorithms analysis completed successfully!")
    print(f"Check {results_dir} for all visualizations and reports.")


if __name__ == "__main__":
    main() 