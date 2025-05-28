#!/usr/bin/env python3
"""
Test script to verify PyTorch DVC correlation fixes
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import torch
from scipy.stats import multivariate_normal

# Import DVC modules
try:
    from DVC_pyolder import fit_vine, vine_obj_bin, margin_obj
    from DVC_pyolder.vine_model import sample_vine
    print("✓ Successfully imported PyTorch DVC modules")
except ImportError as e:
    print(f"✗ Failed to import DVC modules: {e}")
    sys.exit(1)

def test_correlation_recovery():
    """Test if the fixed PyTorch implementation recovers correlations properly"""
    
    print("\n" + "="*60)
    print("Testing PyTorch DVC Correlation Recovery")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate correlated data
    n_samples = 1000
    d = 5  # 5 dimensions
    
    # Create correlation matrix
    target_corr = np.array([
        [1.0, 0.7, 0.5, 0.3, 0.2],
        [0.7, 1.0, 0.6, 0.4, 0.3],
        [0.5, 0.6, 1.0, 0.5, 0.4],
        [0.3, 0.4, 0.5, 1.0, 0.6],
        [0.2, 0.3, 0.4, 0.6, 1.0]
    ])
    
    # Generate data
    mean = np.zeros(d)
    data = np.random.multivariate_normal(mean, target_corr, n_samples)
    
    print(f"\nGenerated {n_samples} samples with {d} dimensions")
    print("\nTarget correlation matrix:")
    print(target_corr)
    
    # Initialize vine
    vine = vine_obj_bin()
    vine.vine_family = 'd-vine'
    vine.param = True
    vine.fitted = False
    vine.n_cop = d
    vine.knots = 50
    
    # Initialize margins
    vine.margin = []
    for i in range(d):
        m = margin_obj()
        m.family = 'norm'
        m.theta = (0, 1)  # Standard normal
        vine.margin.append(m)
    
    # Fit vine
    gen_dict = {'param': True, 'binning': False, 'fitted': False}
    npc_dict = {'ker': None}
    par_dict = {'param_families': ['gaussian', 'clayton', 'ind']}
    bin_dict = {'n_bin': 5}
    
    print("\nFitting D-vine copula...")
    try:
        vine = fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        print("✓ Vine fitting successful")
    except Exception as e:
        print(f"✗ Vine fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check if flip_flag was initialized
    if hasattr(vine, 'flip_flag'):
        print(f"✓ flip_flag initialized with {len(vine.flip_flag)} levels")
        for i, flags in enumerate(vine.flip_flag):
            print(f"  Level {i}: {len(flags)} edges, {sum(flags)} flipped")
    else:
        print("✗ flip_flag not found - fix not applied correctly")
    
    # Check for NaN values in theta
    if hasattr(vine, 'theta'):
        nan_count = torch.isnan(vine.theta).sum().item()
        print(f"\nTheta matrix NaN count: {nan_count}")
        if nan_count > 0:
            print("⚠ Warning: Theta matrix contains NaN values")
            # Find where NaNs occur
            nan_locations = torch.where(torch.isnan(vine.theta))
            print(f"  NaN locations (first 10): {list(zip(nan_locations[0][:10].tolist(), nan_locations[1][:10].tolist(), nan_locations[2][:10].tolist()))}")
    
    # Check selected copula families
    print("\nSelected copula families by level:")
    for level, copulas in enumerate(vine.copulas):
        if len(copulas) > 0:
            if hasattr(copulas[0], 'family'):
                # Parametric copulas
                families = [cop.family for cop in copulas]
                params = [cop.theta for cop in copulas]
                print(f"  Level {level}: {families}")
                print(f"    Parameters: {params}")
            else:
                # Non-parametric
                print(f"  Level {level}: Non-parametric ({len(copulas)} copulas)")
    
    # Sample from the fitted vine
    print("\nSampling from fitted vine...")
    try:
        samples = sample_vine(vine, n_samples)
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        print(f"✓ Generated {samples.shape[0]} samples")
        
        # Calculate sample correlation
        sample_corr = np.corrcoef(samples.T)
        print("\nSample correlation matrix:")
        print(sample_corr)
        
        # Calculate correlation error
        corr_error = np.abs(sample_corr - target_corr)
        mae = np.mean(corr_error[np.triu_indices(d, k=1)])
        
        print(f"\nCorrelation recovery MAE: {mae:.4f}")
        
        # Performance assessment
        if mae < 0.05:
            print("✅ Excellent: Correlation recovery is very good (MAE < 0.05)")
        elif mae < 0.1:
            print("✓ Good: Correlation recovery is acceptable (MAE < 0.1)")
        elif mae < 0.2:
            print("⚠ Fair: Correlation recovery needs improvement (MAE < 0.2)")
        else:
            print("❌ Poor: Correlation recovery is not working properly (MAE >= 0.2)")
        
        # Show correlation differences
        print("\nCorrelation differences (target - recovered):")
        diff_matrix = target_corr - sample_corr
        print(diff_matrix)
        
    except Exception as e:
        print(f"✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    
    print("PyTorch DVC Correlation Fix Verification")
    print("========================================")
    
    # Check Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Run correlation recovery test
    test_correlation_recovery()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    main() 