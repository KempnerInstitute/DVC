import torch
import numpy as np
from scipy.stats import multivariate_normal

from classes.objects import vine_obj_bin, margin_obj

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

def test_simple_vine():
    """Test simple 2D Gaussian copula fitting"""
    
    # Generate 2D correlated Gaussian data
    correlation = 0.7
    cov_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
    
    mvn = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    samples = mvn.rvs(size=500)
    data = torch.tensor(samples, dtype=torch.float32)
    
    print(f"Data shape: {data.shape}")
    print(f"True correlation: {correlation}")
    print(f"Sample correlation: {np.corrcoef(data.T)[0,1]:.3f}")
    
    # Create marginal objects
    margins = [margin_obj('kernel', None, True) for _ in range(2)]
    
    # Test 1: Non-parametric kernel copula
    print("\n1. Testing non-parametric kernel copula...")
    try:
        vine_kernel = vine_obj_bin('c-vine', ['kernel'], 1, margins, 25, 'matrix')
        
        gen_dict = {'binning': False, 'parallel': False, 'param': False, 'vine_depth': 1}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': False}
        par_dict = {}
        bin_dict = {'n_bin': 1}
        
        vine_kernel.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        print("   Success!")
        
        # Check correlation
        if hasattr(vine_kernel, 'correlations') and vine_kernel.correlations:
            print(f"   Estimated correlation: {vine_kernel.correlations[0][0]:.3f}")
            
    except Exception as e:
        print(f"   Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Parametric Gaussian copula
    print("\n2. Testing parametric Gaussian copula...")
    try:
        vine_gauss = vine_obj_bin('c-vine', ['gaussian'], 1, margins, 25, 'matrix')
        
        gen_dict = {'binning': False, 'parallel': False, 'param': True, 'vine_depth': 1}
        npc_dict = {}
        par_dict = {'param_families': ['gaussian', 'ind']}  # Families to try
        bin_dict = {'n_bin': 1}
        
        vine_gauss.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        print("   Success!")
        
        # Check fitted parameters
        if hasattr(vine_gauss, 'copulas') and vine_gauss.copulas:
            cop = vine_gauss.copulas[0][0]
            if hasattr(cop, 'theta'):
                print(f"   Fitted parameter: {cop.theta}")
                # For Gaussian copula, theta is correlation
                print(f"   Estimated correlation: {cop.theta:.3f}")
                
    except Exception as e:
        print(f"   Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test with 3D data
    print("\n3. Testing 3D vine copula...")
    try:
        # Generate 3D correlated data
        cov_3d = np.array([[1.0, 0.5, 0.3],
                          [0.5, 1.0, 0.6],
                          [0.3, 0.6, 1.0]])
        
        mvn_3d = multivariate_normal(mean=[0, 0, 0], cov=cov_3d)
        samples_3d = mvn_3d.rvs(size=500)
        data_3d = torch.tensor(samples_3d, dtype=torch.float32)
        
        margins_3d = [margin_obj('kernel', None, True) for _ in range(3)]
        
        # Need 3 copulas for 3D C-vine
        vine_3d = vine_obj_bin('c-vine', ['gaussian', 'gaussian', 'gaussian'], 2, margins_3d, 25, 'matrix')
        
        gen_dict = {'binning': False, 'parallel': False, 'param': True, 'vine_depth': 2}
        npc_dict = {}
        par_dict = {'param_families': ['gaussian', 'ind']}
        bin_dict = {'n_bin': 1}
        
        vine_3d.fit(data_3d, gen_dict, npc_dict, par_dict, bin_dict)
        print("   Success!")
        
        # Check correlations
        sample_corr_3d = np.corrcoef(data_3d.T)
        print("   Sample correlations:")
        print(sample_corr_3d)
        
    except Exception as e:
        print(f"   Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_vine() 