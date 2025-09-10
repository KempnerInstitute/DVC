"""
Test script for PyTorch DVC implementation
Tests basic functionality including fitting, evaluation, entropy and MI estimation
"""

import numpy as np
import torch
import sys
sys.path.append('src')

from DVC_pyolder import (
    vine_obj_bin, margin_obj, prep_cop,
    vine_entropy, mutual_information
)

def generate_test_data(n_samples=1000, dim=4):
    """Generate synthetic test data with known dependencies"""
    # Generate correlated Gaussian data
    mean = np.zeros(dim)
    
    # Create correlation matrix
    rho = 0.7
    cov = rho * np.ones((dim, dim))
    np.fill_diagonal(cov, 1.0)
    
    # Generate samples
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    
    return data.astype(np.float32)

def test_vine_fitting():
    """Test basic vine fitting functionality"""
    print("Testing vine fitting...")
    
    # Generate test data
    data = generate_test_data(n_samples=500, dim=4)
    print(f"Data shape: {data.shape}")
    
    # Define margins (normal distributions)
    margins = []
    for i in range(data.shape[1]):
        margin = margin_obj('norm', [0, 1], True)
        margins.append(margin)
    
    # Create vine object
    vine_family = 'd-vine'
    families = 'kercop'
    vine_depth = data.shape[1]
    knots = 30
    
    vine = vine_obj_bin(vine_family, families, vine_depth, margins, knots)
    
    # Prepare copula data
    vine_data = prep_cop(data, vine, 'rand')
    
    # Set up fitting parameters
    gen_dict = {
        'parallel': False,
        'binning': False,
        'param': True,  # Use parametric copulas for simplicity
        'vine_depth': vine_depth,
        'fitted': False
    }
    
    par_dict = {
        'param_families': ['gaussian', 'ind']
    }
    
    npc_dict = {
        'opt_method': 'LL1',
        'batch_paral': 4
    }
    
    bin_dict = {
        'n_bin': 1
    }
    
    # Fit the vine
    print("Fitting vine copula...")
    try:
        vine.fit(vine_data, gen_dict, npc_dict, par_dict, bin_dict)
        print("✓ Vine fitting successful")
        
        # Check if copulas were fitted
        if len(vine.copulas) > 0:
            print(f"✓ Number of tree levels: {len(vine.copulas)}")
            for i, level in enumerate(vine.copulas):
                print(f"  Tree {i}: {len(level)} copulas")
        else:
            print("✗ No copulas fitted")
            
    except Exception as e:
        print(f"✗ Error during fitting: {e}")
        import traceback
        traceback.print_exc()
    
    return vine

def test_entropy_estimation(vine):
    """Test entropy estimation"""
    print("\nTesting entropy estimation...")
    
    info_dict = {
        'alpha': 0.05,
        'cases': 100,  # Small for testing
        'iterations': 5
    }
    
    try:
        # Check if vine has required attributes
        if not hasattr(vine, 'sample') or vine.sample is None:
            print("✗ Vine does not have sampling method implemented")
            return
        
        # Estimate entropy
        entropy = vine_entropy(vine, info_dict)
        print(f"✓ Estimated entropy: {entropy:.4f}")
        
    except Exception as e:
        print(f"✗ Error during entropy estimation: {e}")
        import traceback
        traceback.print_exc()

def test_evaluation(vine):
    """Test vine evaluation on new points"""
    print("\nTesting vine evaluation...")
    
    # Generate test points
    test_points = generate_test_data(n_samples=10, dim=vine.n_cop)
    test_points_tensor = torch.from_numpy(test_points).float()
    
    try:
        # Evaluate vine
        p, p_copula, log_marg = vine.evaluation(test_points_tensor)
        
        print(f"✓ Evaluation successful")
        print(f"  Density shape: {p.shape}")
        print(f"  Copula density shape: {p_copula.shape}")
        print(f"  Log marginal shape: {log_marg.shape}")
        
        # Check for reasonable values
        if torch.all(p >= 0):
            print("✓ All density values are non-negative")
        else:
            print("✗ Some density values are negative")
            
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("="*60)
    print("PyTorch DVC Implementation Test")
    print("="*60)
    
    # Test vine fitting
    vine = test_vine_fitting()
    
    if vine is not None and vine.fitted:
        # Test evaluation
        test_evaluation(vine)
        
        # Test entropy estimation (may fail if sampling not fully implemented)
        test_entropy_estimation(vine)
    else:
        print("\nSkipping evaluation and entropy tests due to fitting failure")
    
    print("\n" + "="*60)
    print("Test completed")
    print("="*60)

if __name__ == "__main__":
    main() 