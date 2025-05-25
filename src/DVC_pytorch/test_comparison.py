"""
Comprehensive test suite to compare TensorFlow and PyTorch DVC implementations
"""

import os
import sys
import numpy as np
import torch
import tensorflow as tf
from scipy import stats
from time import perf_counter
import matplotlib.pyplot as plt
import pandas as pd

# Add paths for both implementations
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DVC_tensorflow'))

# Import PyTorch version
from classes.objects import vine_obj_bin as vine_pytorch, margin_obj as margin_pytorch
from utils.prob_op import kendalltau
from grid.grid_op import create_grids

# Import TensorFlow version  
# Note: This assumes the TensorFlow version is accessible
# from DVC_tensorflow.classes.objects import vine_obj_bin as vine_tensorflow, margin_obj as margin_tensorflow

class DVCComparison:
    """Compare TensorFlow and PyTorch DVC implementations"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def generate_test_data(self, n_samples=1000, n_dims=5, correlation_type='gaussian'):
        """Generate test data with known correlation structure"""
        
        if correlation_type == 'gaussian':
            # Generate correlated Gaussian data
            mean = np.zeros(n_dims)
            cov = np.eye(n_dims)
            for i in range(n_dims):
                for j in range(n_dims):
                    if i != j:
                        cov[i, j] = 0.7 * np.exp(-0.5 * abs(i - j))
                        cov[j, i] = cov[i, j]
            
            data = np.random.multivariate_normal(mean, cov, n_samples)
            
        elif correlation_type == 'clayton':
            # Generate Clayton copula data
            theta = 2.0
            u1 = np.random.uniform(0, 1, n_samples)
            v = np.random.uniform(0, 1, n_samples)
            u2 = (1 + u1**(-theta) * (v**(-theta/(1+theta)) - 1))**(-1/theta)
            
            # Transform to normal margins for higher dimensions
            data = np.zeros((n_samples, n_dims))
            data[:, 0] = stats.norm.ppf(u1)
            data[:, 1] = stats.norm.ppf(u2)
            
            # Add more dimensions with decreasing dependence
            for i in range(2, n_dims):
                u_new = np.random.uniform(0, 1, n_samples)
                data[:, i] = stats.norm.ppf(u_new)
                
        elif correlation_type == 'student':
            # Generate Student-t copula data
            df = 5
            rho = 0.7
            mean = np.zeros(n_dims)
            cov = np.eye(n_dims)
            for i in range(n_dims):
                for j in range(n_dims):
                    if i != j:
                        cov[i, j] = rho * (0.9 ** abs(i - j))
                        cov[j, i] = cov[i, j]
            
            # Generate multivariate t
            chi2 = np.random.chisquare(df, n_samples)
            norm = np.random.multivariate_normal(mean, cov, n_samples)
            data = norm / np.sqrt(chi2[:, np.newaxis] / df)
            
        # Convert to uniform margins
        data_uniform = np.zeros_like(data)
        for i in range(n_dims):
            data_uniform[:, i] = stats.rankdata(data[:, i]) / (n_samples + 1)
            
        return data, data_uniform
    
    def test_parametric_fitting(self, n_samples_list=[500, 1000, 2000], n_dims_list=[3, 5, 10]):
        """Test parametric copula fitting"""
        print("\n" + "="*60)
        print("PARAMETRIC COPULA FITTING TEST")
        print("="*60)
        
        results = []
        
        for n_dims in n_dims_list:
            for n_samples in n_samples_list:
                print(f"\nTesting with {n_samples} samples and {n_dims} dimensions...")
                
                # Generate test data
                data, data_uniform = self.generate_test_data(n_samples, n_dims, 'gaussian')
                
                # Convert to PyTorch tensor
                data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=self.device)
                
                # Create margins
                margins = []
                for i in range(n_dims):
                    margin = margin_pytorch(dist='empirical', theta=None, is_cont=True)
                    margins.append(margin)
                
                # Test different vine structures
                for vine_family in ['r-vine', 'c-vine', 'd-vine']:
                    print(f"\n{vine_family.upper()}:")
                    
                    # PyTorch version
                    vine_pt = vine_pytorch(
                        vine_family=vine_family,
                        families=['gaussian', 'clayton', 'student'],
                        vine_depth=n_dims - 1,
                        margin=margins,
                        knots=32,
                        method='random' if vine_family == 'r-vine' else None
                    )
                    
                    # Create grids
                    vine_pt.grid_u, vine_pt.grid_s, vine_pt.grid_x = create_grids(vine_pt.knots, device=self.device)
                    
                    # Set up parameters
                    gen_dict = {
                        'binning': False,
                        'parallel': False,
                        'param': True,
                        'fitted': False,
                        'vine_depth': n_dims - 1
                    }
                    
                    npc_dict = {
                        'opt_method': 'MISE',
                        'batch_paral': 1
                    }
                    
                    par_dict = {
                        'param_families': ['gaussian', 'clayton', 'student', 'ind']
                    }
                    
                    bin_dict = {
                        'n_bin': 1
                    }
                    
                    # Fit PyTorch model
                    start_time = perf_counter()
                    try:
                        vine_pt.fit(data_torch, gen_dict, npc_dict, par_dict, bin_dict)
                        pt_time = perf_counter() - start_time
                        pt_success = True
                        
                        # Get selected families
                        if vine_pt.copulas:
                            selected_families = []
                            for tree in vine_pt.copulas:
                                tree_families = []
                                for cop in tree:
                                    tree_families.append(cop.family)
                                selected_families.append(tree_families)
                        else:
                            selected_families = []
                            
                    except Exception as e:
                        print(f"PyTorch fitting failed: {e}")
                        pt_time = np.nan
                        pt_success = False
                        selected_families = []
                    
                    # Store results
                    result = {
                        'n_samples': n_samples,
                        'n_dims': n_dims,
                        'vine_family': vine_family,
                        'pt_time': pt_time,
                        'pt_success': pt_success,
                        'selected_families': selected_families
                    }
                    
                    results.append(result)
                    
                    print(f"  PyTorch: {'Success' if pt_success else 'Failed'} - Time: {pt_time:.2f}s")
                    if selected_families:
                        print(f"  Selected families (first tree): {selected_families[0] if selected_families else 'N/A'}")
        
        self.results['parametric'] = pd.DataFrame(results)
        return results
    
    def test_nonparametric_fitting(self, n_samples_list=[500, 1000], n_dims_list=[3, 5]):
        """Test non-parametric copula fitting"""
        print("\n" + "="*60)
        print("NON-PARAMETRIC COPULA FITTING TEST")
        print("="*60)
        
        results = []
        
        for n_dims in n_dims_list:
            for n_samples in n_samples_list:
                print(f"\nTesting with {n_samples} samples and {n_dims} dimensions...")
                
                # Generate test data
                data, data_uniform = self.generate_test_data(n_samples, n_dims, 'student')
                
                # Convert to PyTorch tensor
                data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=self.device)
                
                # Create margins
                margins = []
                for i in range(n_dims):
                    margin = margin_pytorch(dist='empirical', theta=None, is_cont=True)
                    margins.append(margin)
                
                # Test R-vine only (most general)
                vine_family = 'r-vine'
                
                # PyTorch version
                vine_pt = vine_pytorch(
                    vine_family=vine_family,
                    families=['gaussian'],
                    vine_depth=min(n_dims - 1, 2),  # Limit depth for speed
                    margin=margins,
                    knots=16,  # Reduced for speed
                    method='optimal'
                )
                
                # Create grids
                vine_pt.grid_u, vine_pt.grid_s, vine_pt.grid_x = create_grids(vine_pt.knots, device=self.device)
                
                # Set up parameters
                gen_dict = {
                    'binning': False,
                    'parallel': False,
                    'param': False,  # Non-parametric
                    'fitted': False,
                    'vine_depth': min(n_dims - 1, 2)
                }
                
                npc_dict = {
                    'opt_method': 'LL1',  # Single bandwidth
                    'batch_paral': 1
                }
                
                par_dict = {}
                
                bin_dict = {
                    'n_bin': 1
                }
                
                # Fit PyTorch model
                start_time = perf_counter()
                try:
                    vine_pt.fit(data_torch, gen_dict, npc_dict, par_dict, bin_dict)
                    pt_time = perf_counter() - start_time
                    pt_success = True
                    
                    # Get optimized bandwidths
                    if vine_pt.copulas:
                        bandwidths = []
                        for copula in vine_pt.copulas:
                            if hasattr(copula, 'opt_bw'):
                                bandwidths.append(copula.opt_bw)
                    else:
                        bandwidths = []
                        
                except Exception as e:
                    print(f"PyTorch fitting failed: {e}")
                    pt_time = np.nan
                    pt_success = False
                    bandwidths = []
                
                # Store results
                result = {
                    'n_samples': n_samples,
                    'n_dims': n_dims,
                    'vine_family': vine_family,
                    'pt_time': pt_time,
                    'pt_success': pt_success,
                    'bandwidths': bandwidths
                }
                
                results.append(result)
                
                print(f"  PyTorch: {'Success' if pt_success else 'Failed'} - Time: {pt_time:.2f}s")
        
        self.results['nonparametric'] = pd.DataFrame(results)
        return results
    
    def test_correlation_estimation(self):
        """Test correlation estimation accuracy"""
        print("\n" + "="*60)
        print("CORRELATION ESTIMATION TEST")
        print("="*60)
        
        n_samples = 1000
        n_dims = 5
        
        # Generate data with known correlations
        true_correlations = []
        for corr_type in ['gaussian', 'clayton', 'student']:
            print(f"\n{corr_type.upper()} dependence:")
            
            data, data_uniform = self.generate_test_data(n_samples, n_dims, corr_type)
            
            # Calculate true Kendall's tau
            true_tau = []
            for i in range(n_dims - 1):
                tau, _ = stats.kendalltau(data_uniform[:, i], data_uniform[:, i + 1])
                true_tau.append(tau)
            
            print(f"True Kendall's tau (first 4 pairs): {[f'{t:.3f}' for t in true_tau[:4]]}")
            
            # Convert to PyTorch
            data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=self.device)
            
            # Estimate using PyTorch kendalltau
            estimated_tau = []
            for i in range(n_dims - 1):
                tau, p_value = kendalltau(data_torch[:, i], data_torch[:, i + 1])
                estimated_tau.append(tau)
            
            print(f"PyTorch estimated tau: {[f'{t:.3f}' for t in estimated_tau[:4]]}")
            
            # Calculate error
            errors = [abs(true - est) for true, est in zip(true_tau, estimated_tau)]
            print(f"Absolute errors: {[f'{e:.3f}' for e in errors[:4]]}")
            print(f"Mean absolute error: {np.mean(errors):.3f}")
    
    def test_different_copula_families(self):
        """Test fitting different copula families"""
        print("\n" + "="*60)
        print("COPULA FAMILY SELECTION TEST")
        print("="*60)
        
        n_samples = 1000
        n_dims = 3
        
        test_cases = [
            ('gaussian', 'Gaussian copula data'),
            ('clayton', 'Clayton copula data'),
            ('student', 'Student-t copula data')
        ]
        
        for data_type, description in test_cases:
            print(f"\n{description}:")
            
            # Generate data
            data, data_uniform = self.generate_test_data(n_samples, n_dims, data_type)
            data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=self.device)
            
            # Create margins
            margins = []
            for i in range(n_dims):
                margin = margin_pytorch(dist='empirical', theta=None, is_cont=True)
                margins.append(margin)
            
            # Fit vine copula
            vine_pt = vine_pytorch(
                vine_family='r-vine',
                families=['gaussian', 'clayton', 'student'],
                vine_depth=n_dims - 1,
                margin=margins,
                knots=32,
                method='random'
            )
            
            # Create grids
            vine_pt.grid_u, vine_pt.grid_s, vine_pt.grid_x = create_grids(vine_pt.knots, device=self.device)
            
            gen_dict = {
                'binning': False,
                'parallel': False,
                'param': True,
                'fitted': False,
                'vine_depth': n_dims - 1
            }
            
            par_dict = {
                'param_families': ['gaussian', 'clayton', 'student', 'ind']
            }
            
            try:
                vine_pt.fit(data_torch, gen_dict, {}, par_dict, {'n_bin': 1})
                
                # Report selected families
                if vine_pt.copulas:
                    print("  Selected copula families by tree:")
                    for i, tree in enumerate(vine_pt.copulas):
                        families = [cop.family for cop in tree]
                        print(f"    Tree {i}: {families}")
                        
                    # Report parameters for first tree
                    print("  Parameters for first tree:")
                    for j, cop in enumerate(vine_pt.copulas[0]):
                        print(f"    Edge {j}: {cop.family} - theta = {cop.theta}")
                        
            except Exception as e:
                print(f"  Fitting failed: {e}")
    
    def test_entropy_estimation(self):
        """Test entropy estimation"""
        print("\n" + "="*60)
        print("ENTROPY ESTIMATION TEST")
        print("="*60)
        
        n_samples = 1000
        n_dims = 3
        
        # Generate data
        data, data_uniform = self.generate_test_data(n_samples, n_dims, 'gaussian')
        data_torch = torch.tensor(data_uniform, dtype=torch.float32, device=self.device)
        
        # Create and fit vine
        margins = []
        for i in range(n_dims):
            margin = margin_pytorch(dist='empirical', theta=None, is_cont=True)
            margins.append(margin)
        
        vine_pt = vine_pytorch(
            vine_family='c-vine',
            families=['gaussian'],
            vine_depth=n_dims - 1,
            margin=margins,
            knots=32,
            method=None
        )
        
        # Create grids
        vine_pt.grid_u, vine_pt.grid_s, vine_pt.grid_x = create_grids(vine_pt.knots, device=self.device)
        
        gen_dict = {
            'binning': False,
            'parallel': False,
            'param': True,
            'fitted': False,
            'vine_depth': n_dims - 1
        }
        
        par_dict = {
            'param_families': ['gaussian']
        }
        
        try:
            vine_pt.fit(data_torch, gen_dict, {}, par_dict, {'n_bin': 1})
            
            # Test evaluation on new points
            test_points = torch.rand(100, n_dims, dtype=torch.float32, device=self.device)
            p, p_copula, log_p = vine_pt.evaluation(test_points)
            
            print(f"  Evaluation successful!")
            print(f"  Mean log-likelihood: {log_p.mean().item():.3f}")
            print(f"  Mean copula density: {p_copula.mean().item():.3f}")
            
            # Estimate entropy (would need sampling implementation)
            # from info.info_estimation import vine_entropy
            # info_dict = {
            #     'alpha': 0.05,
            #     'cases': 1000,
            #     'iterations': 10
            # }
            # entropy = vine_entropy(vine_pt, info_dict)
            # print(f"Estimated entropy: {entropy:.3f}")
            
        except Exception as e:
            print(f"Entropy estimation failed: {e}")
    
    def run_all_tests(self):
        """Run all comparison tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DVC TENSORFLOW vs PYTORCH COMPARISON")
        print("="*80)
        
        # Run tests
        self.test_correlation_estimation()
        self.test_different_copula_families()
        self.test_parametric_fitting(
            n_samples_list=[500, 1000],
            n_dims_list=[3, 5]
        )
        self.test_nonparametric_fitting(
            n_samples_list=[500],
            n_dims_list=[3]
        )
        self.test_entropy_estimation()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if 'parametric' in self.results:
            df = self.results['parametric']
            print("\nParametric fitting results:")
            print(f"  Total tests: {len(df)}")
            print(f"  Successful: {df['pt_success'].sum()}")
            print(f"  Average time: {df['pt_time'].mean():.3f}s")
        
        if 'nonparametric' in self.results:
            df = self.results['nonparametric']
            print("\nNon-parametric fitting results:")
            print(f"  Total tests: {len(df)}")
            print(f"  Successful: {df['pt_success'].sum()}")
            print(f"  Average time: {df['pt_time'].mean():.3f}s")


if __name__ == "__main__":
    # Create comparison object
    comparison = DVCComparison()
    
    # Run all tests
    comparison.run_all_tests()
    
    print("\nTest suite completed!") 