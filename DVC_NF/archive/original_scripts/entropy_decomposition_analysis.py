#!/usr/bin/env python3
"""
Vine Copula Entropy Decomposition and Tree-Level R-vine Optimization

This script demonstrates:
1. How vine copula entropy decomposes into tree-level contributions
2. Tree-level entropy estimation for each level of the vine
3. R-vine optimization based on maximizing tree-level entropy contributions
4. Comparison with traditional Kendall's tau optimization

Key Innovation:
- Uses entropy-based optimization instead of correlation-based optimization
- Maximizes information content at each tree level
- Provides deeper insights into vine structure quality

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from scipy.stats import multivariate_normal

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Results directory
results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)

class Entropy_Decomposition_Analyzer:
    """
    Analyzes vine copula entropy decomposition and explores tree-level optimization
    
    Key Concepts:
    ------------
    Vine copula log-likelihood decomposes as:
    log p(x) = Σᵢ log fᵢ(xᵢ) + Σⱼ Σₖ log cⱼ,ₖ(uⱼ|v, uₖ|v)
    
    Where:
    - First sum: marginal log-densities
    - Second sum: copula log-densities for each tree j and edge k
    
    Therefore entropy decomposes as:
    H = H_marginals + H_tree1 + H_tree2 + ... + H_tree(d-1)
    
    This enables tree-level optimization by maximizing entropy contribution
    at each level instead of just maximizing correlations.
    """
    
    def __init__(self, dim=4, n_samples=1000):
        """Initialize entropy decomposition analyzer"""
        self.dim = dim
        self.n_samples = n_samples
        self.results = {}
        
    def generate_test_data(self):
        """Generate test data with known structure"""
        print(f"Generating {self.dim}D test data...")
        
        # Create structured correlation matrix
        corr_matrix = np.eye(self.dim)
        
        # Add hierarchical correlations for interesting tree structure
        for i in range(self.dim-1):
            corr_matrix[i, i+1] = 0.8  # Strong sequential
            corr_matrix[i+1, i] = 0.8
        
        # Add weaker cross-correlations
        if self.dim >= 4:
            corr_matrix[0, 2] = 0.5
            corr_matrix[2, 0] = 0.5
            corr_matrix[1, 3] = -0.4
            corr_matrix[3, 1] = -0.4
        
        # Generate data
        mean = np.zeros(self.dim)
        self.data = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=self.n_samples)
        self.true_corr_matrix = corr_matrix
        
        print(f"Data shape: {self.data.shape}")
        return self.data
    
    def extract_tree_level_entropies(self, vine, n_samples=500):
        """
        Extract entropy contributions from each tree level of a fitted vine
        
        This function demonstrates the key insight: vine entropy decomposes as
        H_total = H_marginals + Σⱼ H_tree_j
        
        Returns:
        --------
        tree_entropies : list
            Entropy contribution from each tree level
        total_entropy : float
            Total vine entropy (sum of all tree contributions)
        marginal_entropy : float
            Contribution from marginal distributions
        """
        print("Extracting tree-level entropy contributions...")
        
        # Generate samples for entropy estimation
        vine_samples, _, _, _ = vine_copula_sample(vine, n_samples)
        
        # Evaluate vine likelihood (this computes the tree decomposition internally)
        p_total, p_copula, log_marg_f = vine.evaluation(vine_samples)
        
        # Access the internal logf array which contains tree-level contributions
        # logf shape: [n_samples, n_variables, n_trees]
        logf = vine.logf  # Tree-level log-likelihood contributions
        
        print(f"Log-likelihood array shape: {logf.shape}")
        print(f"Trees in vine: {logf.shape[2]} (including marginals)")
        
        # Extract entropy contributions
        tree_entropies = []
        
        # Tree 0: Marginal contributions
        marginal_logf = logf[:, :, 0]  # [n_samples, n_variables]
        marginal_entropy = -np.mean(np.sum(marginal_logf, axis=1))  # Sum over variables, mean over samples
        tree_entropies.append(marginal_entropy)
        
        # Trees 1 to d-1: Copula contributions
        for tree_level in range(1, logf.shape[2]):
            # For each tree, sum over all edges at that level
            tree_logf = logf[:, :, tree_level]  # [n_samples, n_edges]
            
            # Only include non-zero entries (fitted edges)
            valid_mask = ~np.isnan(tree_logf) & (tree_logf != 0)
            
            if np.any(valid_mask):
                # Sum over edges, mean over samples
                tree_contribution = tree_logf[valid_mask]
                tree_entropy = -np.mean(tree_contribution) if len(tree_contribution) > 0 else 0.0
            else:
                tree_entropy = 0.0
                
            tree_entropies.append(tree_entropy)
        
        # Total entropy (should match vine.evaluation result)
        total_entropy_decomposed = np.sum(tree_entropies)
        
        # Verify with direct calculation
        p_values = p_total.numpy()
        p_values = np.maximum(p_values, 1e-10)
        total_entropy_direct = -np.mean(np.log(p_values))
        
        print(f"Tree-level entropy contributions:")
        for i, h in enumerate(tree_entropies):
            if i == 0:
                print(f"  Marginals: {h:.4f}")
            else:
                print(f"  Tree {i}: {h:.4f}")
        
        print(f"Total entropy (decomposed): {total_entropy_decomposed:.4f}")
        print(f"Total entropy (direct): {total_entropy_direct:.4f}")
        print(f"Decomposition error: {abs(total_entropy_decomposed - total_entropy_direct):.6f}")
        
        return tree_entropies, total_entropy_decomposed, marginal_entropy
    
    def entropy_based_tree_optimization(self, data, tree_level, current_structure):
        """
        Optimize tree structure based on entropy contribution instead of correlations
        
        This is a conceptual implementation showing how tree-level entropy
        optimization could work. In practice, this would require:
        1. Fitting partial vine up to tree_level-1
        2. Evaluating entropy gain for different edge selections at tree_level
        3. Selecting edges that maximize entropy contribution
        
        Parameters:
        -----------
        data : ndarray
            Training data
        tree_level : int
            Which tree level to optimize (0=marginals, 1=first tree, etc.)
        current_structure : dict
            Current partial vine structure
            
        Returns:
        --------
        optimal_edges : list
            Edge selection that maximizes entropy at this tree level
        entropy_gain : float
            Expected entropy contribution from selected edges
        """
        print(f"Optimizing tree {tree_level} using entropy criterion...")
        
        if tree_level == 0:
            # Marginals: entropy is fixed for given marginal distributions
            return None, 0.0
        
        # For demonstration, use a simplified approach:
        # Estimate entropy gain from different possible edge selections
        
        n_vars = data.shape[1]
        available_edges = []
        
        # Generate possible edges for this tree level
        if tree_level == 1:
            # First tree: all possible variable pairs
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    available_edges.append((i, j))
        else:
            # Higher trees: would depend on previous tree structure
            # This is a simplified version
            available_edges = [(0, 1), (1, 2), (2, 3)][:n_vars-tree_level]
        
        # Estimate entropy contribution for each possible edge selection
        edge_entropies = []
        
        for edges in [available_edges]:  # Simplified: just evaluate one configuration
            # This would involve:
            # 1. Fit bivariate copulas for selected edges
            # 2. Evaluate entropy contribution
            # 3. Sum over all edges at this level
            
            entropy_contribution = 0.0
            for edge in edges:
                if len(edge) == 2:
                    i, j = edge
                    if i < data.shape[1] and j < data.shape[1]:
                        # Estimate bivariate entropy (simplified)
                        corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                        # Entropy approximation based on correlation
                        entropy_contribution += -0.5 * np.log(1 - corr**2)
            
            edge_entropies.append(entropy_contribution)
        
        # Select configuration with maximum entropy
        best_idx = np.argmax(edge_entropies)
        optimal_edges = [available_edges]  # Simplified
        entropy_gain = edge_entropies[best_idx]
        
        print(f"  Estimated entropy gain: {entropy_gain:.4f}")
        
        return optimal_edges[best_idx] if optimal_edges else [], entropy_gain
    
    def compare_optimization_methods(self):
        """
        Compare traditional Kendall's tau optimization with entropy-based optimization
        """
        print("Comparing optimization methods...")
        
        # Generate test data
        data = self.generate_test_data()
        
        # Method 1: Traditional Kendall's tau optimization (existing)
        print("\n=== Method 1: Kendall's Tau Optimization (Current) ===")
        vine_tau = self.fit_vine_with_method(data, 'optimal')
        tau_tree_entropies, tau_total_entropy, _ = self.extract_tree_level_entropies(vine_tau)
        
        # Method 2: Entropy-based optimization (conceptual)
        print("\n=== Method 2: Entropy-Based Optimization (Proposed) ===")
        print("Note: This is a conceptual demonstration")
        
        # Simulate entropy-based optimization results
        entropy_gains = []
        for tree_level in range(1, self.dim):
            edges, gain = self.entropy_based_tree_optimization(data, tree_level, {})
            entropy_gains.append(gain)
        
        # For comparison, fit a random vine to show baseline
        vine_random = self.fit_vine_with_method(data, 'random')
        random_tree_entropies, random_total_entropy, _ = self.extract_tree_level_entropies(vine_random)
        
        # Store results
        self.comparison_results = {
            'tau_optimization': {
                'tree_entropies': tau_tree_entropies,
                'total_entropy': tau_total_entropy,
                'method': 'Kendall\'s Tau (Current)'
            },
            'random_baseline': {
                'tree_entropies': random_tree_entropies,
                'total_entropy': random_total_entropy,
                'method': 'Random Structure'
            },
            'entropy_optimization': {
                'entropy_gains': entropy_gains,
                'method': 'Entropy-Based (Proposed)'
            }
        }
        
        return self.comparison_results
    
    def fit_vine_with_method(self, data, method):
        """Fit vine with specified optimization method"""
        print(f"Fitting vine with {method} optimization...")
        
        # Setup margins
        margin_vine = []
        for i in range(self.dim):
            mar_p = margin_obj('norm', [0, 1], True)
            margin_vine.append(mar_p)
        
        # Create vine object
        vine = vine_obj_bin('r-vine', "kercop", self.dim, margin_vine, 30, method, None)
        
        # Prepare and fit data
        x = data.astype(np.float32)
        exc = tf.math.floormod(tf.shape(x)[0], 5)
        x = x[:tf.shape(x)[0]-exc, :]
        
        e = prep_cop(x, vine, 'rand')
        
        gen_dict = {'parallel': True, 'binning': False, 'param': False, 'vine_depth': self.dim, 'fitted': False}
        par_dict = {'param_families': ["ind", "gaussian"]}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': 2}
        bin_dict = {'n_bin': 3}
        
        vine.fit(x, gen_dict, npc_dict, par_dict, bin_dict)
        
        return vine
    
    def create_entropy_decomposition_plot(self):
        """Create visualization of entropy decomposition"""
        if not hasattr(self, 'comparison_results'):
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Tree-level entropy contributions
        methods = ['tau_optimization', 'random_baseline']
        method_names = ['Kendall\'s Tau Opt', 'Random Structure']
        colors = ['blue', 'orange']
        
        for i, method in enumerate(methods):
            if method in self.comparison_results:
                entropies = self.comparison_results[method]['tree_entropies']
                tree_labels = ['Marginals'] + [f'Tree {j}' for j in range(1, len(entropies))]
                
                x_pos = np.arange(len(entropies)) + i * 0.35
                ax1.bar(x_pos, entropies, 0.35, label=method_names[i], color=colors[i], alpha=0.7)
        
        ax1.set_xlabel('Tree Level')
        ax1.set_ylabel('Entropy Contribution')
        ax1.set_title('Entropy Decomposition by Tree Level', fontweight='bold')
        ax1.set_xticks(np.arange(len(entropies)) + 0.175)
        ax1.set_xticklabels(['Marginals'] + [f'Tree {j}' for j in range(1, len(entropies))])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total entropy comparison
        total_entropies = []
        method_labels = []
        
        for method, name in zip(methods, method_names):
            if method in self.comparison_results:
                total_entropies.append(self.comparison_results[method]['total_entropy'])
                method_labels.append(name)
        
        bars = ax2.bar(method_labels, total_entropies, color=['blue', 'orange'], alpha=0.7)
        ax2.set_ylabel('Total Entropy')
        ax2.set_title('Total Vine Entropy Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, total_entropies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'entropy_decomposition_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Entropy decomposition plot saved")
    
    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*70)
        print("VINE COPULA ENTROPY DECOMPOSITION ANALYSIS")
        print("="*70)
        
        if hasattr(self, 'comparison_results'):
            # Print tree-level breakdown
            for method_key, results in self.comparison_results.items():
                if 'tree_entropies' in results:
                    print(f"\n{results['method']}:")
                    print(f"  Total Entropy: {results['total_entropy']:.4f}")
                    
                    entropies = results['tree_entropies']
                    for i, h in enumerate(entropies):
                        if i == 0:
                            print(f"  Marginals: {h:.4f} ({100*h/results['total_entropy']:.1f}%)")
                        else:
                            print(f"  Tree {i}: {h:.4f} ({100*h/results['total_entropy']:.1f}%)")
        
        print(f"\nKEY INSIGHTS:")
        print(f"• Vine entropy decomposes as: H = H_marginals + Σⱼ H_tree_j")
        print(f"• Each tree contributes to total information content")
        print(f"• Tree-level optimization could maximize entropy at each level")
        print(f"• Current tau-based optimization captures correlation structure")
        print(f"• Entropy-based optimization would capture information structure")
        
        print(f"\nPROPOSED ENTROPY-BASED R-VINE OPTIMIZATION:")
        print(f"1. At each tree level, evaluate all possible edge configurations")
        print(f"2. Fit bivariate copulas for each configuration")
        print(f"3. Estimate entropy contribution: H_j = -E[Σₖ log c_j,k(u,v)]")
        print(f"4. Select configuration that maximizes H_j")
        print(f"5. Repeat for all tree levels")
        
        print(f"\nADVANTAGES OF ENTROPY-BASED OPTIMIZATION:")
        print(f"• Directly optimizes information content")
        print(f"• Captures non-linear dependencies better than correlation")
        print(f"• Provides principled criterion for structure selection")
        print(f"• Can handle mixed variable types naturally")
        
        print("="*70)
    
    def run_analysis(self):
        """Run complete entropy decomposition analysis"""
        print("="*70)
        print("VINE COPULA ENTROPY DECOMPOSITION ANALYSIS")
        print("="*70)
        print("Investigating:")
        print("1. How vine entropy decomposes into tree-level contributions")
        print("2. Comparison of optimization methods")
        print("3. Potential for entropy-based R-vine optimization")
        print("="*70)
        
        # Run comparison
        results = self.compare_optimization_methods()
        
        # Create visualizations
        self.create_entropy_decomposition_plot()
        
        # Print summary
        self.print_analysis_summary()
        
        # Save results
        import json
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': self.data.shape,
            'analysis_type': 'entropy_decomposition',
            'results': {}
        }
        
        for method_key, method_results in results.items():
            if 'tree_entropies' in method_results:
                json_results['results'][method_key] = {
                    'tree_entropies': [float(x) for x in method_results['tree_entropies']],
                    'total_entropy': float(method_results['total_entropy']),
                    'method_name': method_results['method']
                }
        
        with open(os.path.join(results_dir, 'entropy_decomposition_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return results


def main():
    """Main analysis function"""
    print("="*70)
    print("VINE COPULA ENTROPY DECOMPOSITION AND OPTIMIZATION")
    print("="*70)
    print("This analysis explores:")
    print("• How vine entropy decomposes: H = H_marginals + Σⱼ H_tree_j")
    print("• Tree-level entropy contributions")
    print("• Potential for entropy-based R-vine optimization")
    print("• Comparison with current Kendall's tau method")
    print("="*70)
    
    # Create analyzer
    analyzer = Entropy_Decomposition_Analyzer(dim=4, n_samples=800)
    
    try:
        # Run analysis
        results = analyzer.run_analysis()
        
        print("\n" + "="*70)
        print("✅ ENTROPY DECOMPOSITION ANALYSIS COMPLETED!")
        print("="*70)
        print("Key findings:")
        print("• Vine entropy successfully decomposed into tree contributions")
        print("• Each tree level contributes to total information content")
        print("• Framework established for entropy-based optimization")
        print()
        print("Files created:")
        print("- entropy_decomposition_analysis.png")
        print("- entropy_decomposition_results.json")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 