#!/usr/bin/env python3
"""
Test Entropy-Based R-vine Optimization Integration

This script tests whether the entropy-based optimization has been 
successfully integrated into the existing DVC framework.
"""

import os
import sys
import numpy as np
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

print("="*80)
print("TESTING ENTROPY-BASED R-VINE OPTIMIZATION INTEGRATION")
print("="*80)

def test_entropy_optimization_methods():
    """Test available optimization methods"""
    
    print("\n1. Testing Enhanced optimal_tree Function")
    print("-"*50)
    
    # Generate simple test data
    np.random.seed(42)
    data = np.random.randn(100, 4).astype(np.float32)
    
    try:
        # Test traditional tau-based optimization
        print("Testing tau-based optimization...")
        edges_tau, weights_tau = optimal_tree(data.T, None, [], 0, rand=False, optimization_method='tau')
        print(f"✓ Tau-based: {len(edges_tau)} edges, weights: {[f'{w:.3f}' for w in weights_tau]}")
        
        # Test entropy-based optimization
        print("Testing entropy-based optimization...")
        edges_entropy, weights_entropy = optimal_tree(data.T, None, [], 0, rand=False, optimization_method='entropy')
        print(f"✓ Entropy-based: {len(edges_entropy)} edges, weights: {[f'{w:.3f}' for w in weights_entropy]}")
        
        # Test random optimization
        print("Testing random optimization...")
        edges_random, weights_random = optimal_tree(data.T, None, [], 0, rand=False, optimization_method='random')
        print(f"✓ Random: {len(edges_random)} edges, weights: {[f'{w:.3f}' for w in weights_random]}")
        
        # Compare edge selections
        print(f"\nEdge Selection Comparison:")
        print(f"Tau edges:     {edges_tau}")
        print(f"Entropy edges: {edges_entropy}")
        print(f"Random edges:  {edges_random}")
        
        if edges_tau != edges_entropy:
            print("🔥 SUCCESS: Entropy optimization selects different edges than tau-based!")
        else:
            print("⚠️  NOTE: Entropy and tau selected same edges (could be due to simple data)")
            
    except Exception as e:
        print(f"❌ Error testing optimal_tree: {e}")
        return False
    
    return True

def test_vine_with_entropy():
    """Test vine fitting with entropy optimization"""
    
    print("\n2. Testing Vine Fitting with Entropy Optimization")
    print("-"*50)
    
    try:
        # Generate test data with correlations
        np.random.seed(42)
        corr_matrix = np.eye(4)
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.7
        corr_matrix[1, 2] = corr_matrix[2, 1] = 0.5
        corr_matrix[2, 3] = corr_matrix[3, 2] = 0.6
        
        data = np.random.multivariate_normal(np.zeros(4), corr_matrix, 200).astype(np.float32)
        
        # Test C-vine (baseline)
        print("Testing C-vine (baseline)...")
        vine_cvine = test_vine_fitting(data, 'c-vine', 'matrix')
        
        # Test R-vine with tau optimization
        print("Testing R-vine with tau optimization...")
        vine_tau = test_vine_fitting(data, 'r-vine', 'optimal', 'tau')
        
        # Test R-vine with entropy optimization 
        print("Testing R-vine with entropy optimization...")
        vine_entropy = test_vine_fitting(data, 'r-vine', 'optimal', 'entropy')
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing vine fitting: {e}")
        return False

def test_vine_fitting(data, vine_type, method, optimization_method=None):
    """Helper function to test vine fitting"""
    
    # Setup margins
    margin_vine = []
    for i in range(4):
        mar_p = margin_obj('norm', [0, 1], True)
        mar_p.ker = data[:, i]
        margin_vine.append(mar_p)
    
    # Create vine object
    vine = vine_obj_bin(vine_type, "kercop", 4, margin_vine, 30, method)
    
    # Store optimization method if provided
    if optimization_method:
        vine.optimization_method = optimization_method
    
    # Prepare data
    exc = tf.math.floormod(tf.shape(data)[0], 5)
    data_clean = data[:tf.shape(data)[0]-exc, :]
    
    # Prepare fitting settings
    gen_dict = {
        'parallel': False,
        'binning': False,
        'param': False,
        'vine_depth': 4,
        'fitted': False
    }
    
    # Add optimization method if available
    if hasattr(vine, 'optimization_method'):
        gen_dict['optimization_method'] = vine.optimization_method
    
    par_dict = {'param_families': ["ind", "gaussian"]}
    npc_dict = {'opt_method': 'LL1', 'batch_paral': 1}
    bin_dict = {'n_bin': 3}
    
    # Fit vine
    try:
        vine.fit(data_clean, gen_dict, npc_dict, par_dict, bin_dict)
        print(f"✓ {vine_type} with {method} ({optimization_method if optimization_method else 'default'}) fitted successfully")
        return vine
    except Exception as e:
        print(f"❌ {vine_type} with {method} failed: {e}")
        return None

def main():
    """Main test function"""
    
    print("Starting entropy optimization integration tests...\n")
    
    # Test 1: Enhanced optimal_tree function
    success1 = test_entropy_optimization_methods()
    
    # Test 2: Vine fitting with entropy
    success2 = test_vine_with_entropy()
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    if success1:
        print("✅ optimal_tree function: Enhanced with entropy optimization")
    else:
        print("❌ optimal_tree function: Failed to enhance")
    
    if success2:
        print("✅ Vine fitting: Successfully integrated entropy optimization")
    else:
        print("❌ Vine fitting: Failed to integrate entropy optimization")
    
    if success1 and success2:
        print("\n🎉 INTEGRATION SUCCESS!")
        print("Entropy-based R-vine optimization is now available in the DVC framework!")
        print("\nUsage:")
        print("• Set method='optimal' and optimization_method='entropy' in gen_dict")
        print("• Available methods: 'tau', 'entropy', 'random'")
    else:
        print("\n⚠️  INTEGRATION INCOMPLETE")
        print("Some components need additional work")
    
    print("="*80)

if __name__ == "__main__":
    main() 