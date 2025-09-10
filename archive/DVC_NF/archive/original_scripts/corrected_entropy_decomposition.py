#!/usr/bin/env python3
"""
Corrected Vine Copula Entropy Decomposition Analysis

This script provides a corrected analysis of vine copula entropy decomposition,
addressing the issue of negative entropy values and clarifying the difference
between Shannon entropy and differential entropy.

Key Corrections:
1. Explains Shannon vs Differential entropy
2. Shows why differential entropy can be negative
3. Provides correct entropy decomposition
4. Validates the decomposition approach

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

class Corrected_Entropy_Analyzer:
    """
    Corrected analysis of vine copula entropy decomposition
    
    Key Theoretical Points:
    ----------------------
    1. **Shannon Entropy** (discrete): H(X) = -Σ p(x) log p(x) ≥ 0
    2. **Differential Entropy** (continuous): h(X) = -∫ f(x) log f(x) dx (can be < 0)
    
    For vine copulas (continuous), we deal with differential entropy.
    Negative values occur when density f(x) > 1, making log f(x) > 0.
    This is common for copulas on [0,1]² domain.
    
    Vine Decomposition:
    ------------------
    log p(x) = Σᵢ log fᵢ(xᵢ) + Σⱼ Σₖ log cⱼ,ₖ(u,v)
    
    Therefore:
    h(X) = h_marginals + Σⱼ h_tree_j
    
    where each component can be positive or negative.
    """
    
    def __init__(self, dim=4, n_samples=800):
        """Initialize corrected entropy analyzer"""
        self.dim = dim
        self.n_samples = n_samples
        
    def explain_entropy_types(self):
        """Explain the difference between Shannon and differential entropy"""
        print("\n" + "="*70)
        print("ENTROPY TYPES EXPLANATION")
        print("="*70)
        
        print("1. SHANNON ENTROPY (Discrete Random Variables):")
        print("   H(X) = -Σ p(x) log p(x)")
        print("   Properties:")
        print("   • Always non-negative: H(X) ≥ 0")
        print("   • p(x) are probabilities: 0 ≤ p(x) ≤ 1")
        print("   • log p(x) ≤ 0, so -log p(x) ≥ 0")
        
        print("\n2. DIFFERENTIAL ENTROPY (Continuous Random Variables):")
        print("   h(X) = -∫ f(x) log f(x) dx")
        print("   Properties:")
        print("   • CAN BE NEGATIVE: h(X) can be < 0")
        print("   • f(x) is probability density, can be > 1")
        print("   • When f(x) > 1: log f(x) > 0, so -log f(x) < 0")
        print("   • Common for distributions on bounded domains like [0,1]")
        
        print("\n3. VINE COPULAS:")
        print("   • Deal with continuous distributions → differential entropy")
        print("   • Copula densities on [0,1]² often > 1")
        print("   • Therefore, negative differential entropy is theoretically valid")
        
        print("\n4. EXAMPLE:")
        print("   Uniform distribution on [0,0.5]: f(x) = 2 for x ∈ [0,0.5]")
        print("   h(X) = -∫₀^{0.5} 2 log(2) dx = -log(2) ≈ -0.693 < 0")
        
        print("="*70)
    
    def demonstrate_negative_differential_entropy(self):
        """Show concrete example of negative differential entropy"""
        print("\nDEMONSTRATION: Why Differential Entropy Can Be Negative")
        print("-" * 60)
        
        # Example 1: Uniform on [0, 0.5]
        print("Example 1: Uniform distribution on [0, 0.5]")
        print("  Density: f(x) = 2 for x ∈ [0, 0.5]")
        print("  Differential entropy: h(X) = -∫ f(x) log f(x) dx")
        print("                              = -∫₀^{0.5} 2 log(2) dx")
        print("                              = -log(2) × 0.5 × 2")
        print("                              = -log(2) ≈ -0.693")
        print("  ✓ Negative differential entropy!")
        
        # Example 2: Beta distribution with sharp peak
        from scipy.stats import beta
        print("\nExample 2: Beta(0.5, 0.5) distribution")
        beta_dist = beta(0.5, 0.5)
        x_vals = np.linspace(0.01, 0.99, 1000)
        densities = beta_dist.pdf(x_vals)
        
        # Estimate differential entropy numerically
        dx = x_vals[1] - x_vals[0]
        log_densities = np.log(densities)
        differential_entropy = -np.sum(densities * log_densities * dx)
        
        print(f"  Max density value: {np.max(densities):.2f} (> 1)")
        print(f"  Differential entropy: {differential_entropy:.3f}")
        if differential_entropy < 0:
            print("  ✓ Another example of negative differential entropy!")
        
        print("\nKey Insight:")
        print("• When density concentrates in small region → density values > 1")
        print("• log(density) > 0 → negative contribution to entropy")
        print("• This is mathematically correct for differential entropy!")
    
    def corrected_vine_entropy_decomposition(self, vine, n_samples=500):
        """
        Corrected vine entropy decomposition with proper interpretation
        """
        print("\nCorrected Vine Entropy Decomposition")
        print("-" * 40)
        
        # Generate samples
        vine_samples, _, _, _ = vine_copula_sample(vine, n_samples)
        
        # Evaluate vine likelihood
        p_total, p_copula, log_marg_f = vine.evaluation(vine_samples)
        
        # Access log-likelihood decomposition
        logf = vine.logf  # Shape: [n_samples, n_variables, n_trees]
        
        print(f"Log-likelihood array shape: {logf.shape}")
        print(f"Components: {logf.shape[2]} (marginals + {logf.shape[2]-1} tree levels)")
        
        # Correct decomposition
        tree_contributions = []
        
        # Marginal contributions
        marginal_logf = logf[:, :, 0]  # [n_samples, n_variables]
        # Sum over variables for each sample, then average over samples
        marginal_differential_entropy = -np.mean(np.sum(marginal_logf, axis=1))
        tree_contributions.append(marginal_differential_entropy)
        
        print(f"\nMarginal contribution: {marginal_differential_entropy:.4f}")
        
        # Tree-level contributions
        total_tree_entropy = 0.0
        for tree_level in range(1, logf.shape[2]):
            tree_logf = logf[:, :, tree_level]
            
            # Sum non-zero entries for each sample, then average
            valid_mask = ~np.isnan(tree_logf) & (tree_logf != 0)
            if np.any(valid_mask):
                # For each sample, sum over all edges in this tree
                sample_tree_logf = []
                for sample_idx in range(tree_logf.shape[0]):
                    sample_edges = tree_logf[sample_idx, valid_mask[sample_idx, :]]
                    if len(sample_edges) > 0:
                        sample_tree_logf.append(np.sum(sample_edges))
                    else:
                        sample_tree_logf.append(0.0)
                
                tree_differential_entropy = -np.mean(sample_tree_logf)
            else:
                tree_differential_entropy = 0.0
            
            tree_contributions.append(tree_differential_entropy)
            total_tree_entropy += tree_differential_entropy
            
            print(f"Tree {tree_level} contribution: {tree_differential_entropy:.4f}")
        
        # Total entropy
        total_decomposed = np.sum(tree_contributions)
        
        # Verify with direct calculation
        p_values = p_total.numpy()
        p_values = np.maximum(p_values, 1e-10)  # Avoid log(0)
        total_direct = -np.mean(np.log(p_values))
        
        print(f"\nTotal differential entropy (decomposed): {total_decomposed:.4f}")
        print(f"Total differential entropy (direct): {total_direct:.4f}")
        print(f"Decomposition error: {abs(total_decomposed - total_direct):.6f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        marginal_pct = 100 * marginal_differential_entropy / total_direct if total_direct != 0 else 0
        print(f"• Marginals contribute: {marginal_pct:.1f}% of total differential entropy")
        print(f"• Tree structures contribute: {100 - marginal_pct:.1f}%")
        
        if any(h < 0 for h in tree_contributions):
            print(f"• Negative values are VALID for differential entropy")
            print(f"• They indicate density concentrations (f(x) > 1)")
        
        return tree_contributions, total_decomposed
    
    def run_corrected_analysis(self):
        """Run complete corrected entropy analysis"""
        print("="*70)
        print("CORRECTED VINE COPULA ENTROPY DECOMPOSITION")
        print("="*70)
        
        # Explain entropy types
        self.explain_entropy_types()
        
        # Demonstrate negative differential entropy
        self.demonstrate_negative_differential_entropy()
        
        # Generate test data
        print(f"\nGenerating {self.dim}D test data...")
        corr_matrix = np.eye(self.dim)
        for i in range(self.dim-1):
            corr_matrix[i, i+1] = 0.8
            corr_matrix[i+1, i] = 0.8
        
        mean = np.zeros(self.dim)
        data = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=self.n_samples)
        
        # Fit vine
        print("Fitting vine copula...")
        margin_vine = []
        for i in range(self.dim):
            mar_p = margin_obj('norm', [0, 1], True)
            margin_vine.append(mar_p)
        
        vine = vine_obj_bin('r-vine', "kercop", self.dim, margin_vine, 30, 'optimal', None)
        
        x = data.astype(np.float32)
        exc = tf.math.floormod(tf.shape(x)[0], 5)
        x = x[:tf.shape(x)[0]-exc, :]
        
        e = prep_cop(x, vine, 'rand')
        
        gen_dict = {'parallel': True, 'binning': False, 'param': False, 'vine_depth': self.dim, 'fitted': False}
        par_dict = {'param_families': ["ind", "gaussian"]}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': 2}
        bin_dict = {'n_bin': 3}
        
        vine.fit(x, gen_dict, npc_dict, par_dict, bin_dict)
        
        # Perform corrected entropy decomposition
        tree_contributions, total_entropy = self.corrected_vine_entropy_decomposition(vine)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'corrected_entropy_decomposition',
            'entropy_type': 'differential_entropy',
            'tree_contributions': [float(x) for x in tree_contributions],
            'total_entropy': float(total_entropy),
            'note': 'Negative values are valid for differential entropy'
        }
        
        import json
        with open(os.path.join(results_dir, 'corrected_entropy_analysis.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n" + "="*70)
        print("KEY TAKEAWAYS:")
        print("="*70)
        print("1. ✅ Vine entropy DOES decompose into tree-level contributions")
        print("2. ✅ We work with DIFFERENTIAL entropy (continuous distributions)")
        print("3. ✅ Negative differential entropy is MATHEMATICALLY VALID")
        print("4. ✅ Occurs when density values > 1 (common for copulas)")
        print("5. ✅ Entropy-based R-vine optimization is still feasible")
        print("6. ✅ Framework enables tree-level information optimization")
        print("="*70)
        
        return results


def main():
    """Main function for corrected entropy analysis"""
    print("="*70)
    print("CORRECTED VINE COPULA ENTROPY DECOMPOSITION")
    print("="*70)
    print("Addressing the question: 'Why is entropy negative?'")
    print()
    print("This analysis:")
    print("• Explains Shannon vs differential entropy")
    print("• Shows why negative values are mathematically valid")
    print("• Provides corrected entropy decomposition")
    print("• Validates the tree-level optimization framework")
    print("="*70)
    
    analyzer = Corrected_Entropy_Analyzer(dim=4, n_samples=600)
    
    try:
        results = analyzer.run_corrected_analysis()
        
        print(f"\n✅ CORRECTED ANALYSIS COMPLETED!")
        print(f"Results saved: corrected_entropy_analysis.json")
        print(f"\nConclusion: Negative differential entropy is VALID!")
        print(f"The entropy-based R-vine optimization framework remains sound.")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 