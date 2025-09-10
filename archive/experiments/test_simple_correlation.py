#!/usr/bin/env python3
"""
Simplified Correlation Recovery Test
==================================

Focus on testing just the correlation recovery without parametric copula selection.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

def test_simple_correlation_recovery():
    """Test correlation recovery with a simple 2D case"""
    print("Testing simple 2D correlation recovery...")
    
    try:
        from DVC_pyolder.vine_model import fit_vine
        from DVC_pyolder.objects import vine_obj_bin, margin_obj
        from DVC_pyolder.sampling import vine_copula_sample
        
        # Generate simple 2D correlated data
        np.random.seed(42)
        n_samples = 300
        correlation = 0.7
        
        # Create 2D correlated normal data
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]
        
        data = np.random.multivariate_normal(mean, cov, n_samples)
        
        # Transform to uniform margins
        from scipy.stats import norm
        data_uniform = norm.cdf(data)
        
        print(f"Original correlation: {np.corrcoef(data_uniform.T)[0,1]:.4f}")
        
        # Create simple vine model  
        margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(2)]
        vine = vine_obj_bin('c-vine', 'gaussian', 2, margins, 20, 'matrix')
        
        # Simple configuration for fitting
        gen_dict = {
            'binning': False,
            'parallel': False,
            'param': True,
            'fitted': False,
            'vine_depth': 1
        }
        
        npc_dict = {
            'npc_family': 'locallik',
            'grid_dim': 20
        }
        
        par_dict = {
            'param_families': ['gaussian']  # Only Gaussian to avoid AIC issues
        }
        
        bin_dict = {
            'n_bin': 1
        }
        
        # Fit vine
        vine = fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        print("✓ Vine fitting successful")
        
        # Check fitted copula
        if hasattr(vine, 'copulas') and len(vine.copulas) > 0:
            if len(vine.copulas[0]) > 0:
                cop = vine.copulas[0][0]
                print(f"Fitted copula: {cop.family if hasattr(cop, 'family') else 'non-parametric'}")
                if hasattr(cop, 'theta'):
                    print(f"Fitted parameter: {cop.theta}")
        
        # Test sampling
        print("Testing sampling...")
        
        # Add debug for vine structure before sampling
        debug_vine_structure(vine)
        
        try:
            samples, u_samples, _, _ = vine_copula_sample(vine, 1000)
            
            # Check sample correlations
            sample_corr = np.corrcoef(u_samples.T)[0,1]
            print(f"Sample correlation: {sample_corr:.4f}")
            
            # Check if we recovered correlation
            correlation_diff = abs(correlation - sample_corr)
            print(f"Correlation difference: {correlation_diff:.4f}")
            
            if correlation_diff < 0.3:
                print("✓ Correlation recovery successful")
                return True
            else:
                print("✗ Correlation recovery failed")
                return False
                
        except Exception as e:
            print(f"✗ Sampling failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_vine_structure(vine):
    """Debug function to examine vine structure"""
    print("\n=== Vine Structure Debug ===")
    print(f"Vine family: {vine.vine_family}")
    print(f"Number of levels: {len(vine.ind_vine) if hasattr(vine, 'ind_vine') else 'N/A'}")
    
    if hasattr(vine, 'ind_vine'):
        for i, level in enumerate(vine.ind_vine):
            print(f"Level {i} edges: {level}")
    
    if hasattr(vine, 'copulas'):
        print(f"Number of copula levels: {len(vine.copulas)}")
        for i, level in enumerate(vine.copulas):
            print(f"Level {i} copulas: {len(level)} copulas")
            for j, cop in enumerate(level):
                if hasattr(cop, 'family'):
                    print(f"  Copula {j}: {cop.family}, theta={cop.theta}")
                else:
                    print(f"  Copula {j}: Non-parametric")
    
    if hasattr(vine, 'theta'):
        print(f"Theta matrix shape: {vine.theta.shape}")
        print(f"Theta matrix NaN count: {np.isnan(vine.theta.cpu().numpy()).sum()}")
    
    print("=== End Debug ===\n")

if __name__ == "__main__":
    success = test_simple_correlation_recovery()
    sys.exit(0 if success else 1) 