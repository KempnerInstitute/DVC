"""
Comprehensive DVC Comparison Test: PyTorch vs TensorFlow
=======================================================

This test compares PyTorch and TensorFlow implementations of Deep Vine Copula (DVC)
across multiple dimensions:
- Correlation structure recovery
- Different vine types (C-vine, D-vine)
- Parametric vs non-parametric methods
- Performance metrics and computational efficiency
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup paths for both implementations
current_dir = os.path.dirname(os.path.abspath(__file__))
tensorflow_path = os.path.join(current_dir, 'src', 'DVC_tensorflow')
pytorch_path = os.path.join(current_dir, 'src', 'DVC_pytorch')

# Global flags for availability
TENSORFLOW_AVAILABLE = False
PYTORCH_AVAILABLE = False

def setup_tensorflow():
    """Setup TensorFlow DVC imports"""
    global TENSORFLOW_AVAILABLE
    
    # Clean path
    paths_to_remove = [p for p in sys.path if 'DVC' in p and 'tensorflow' not in p.lower()]
    for path in paths_to_remove:
        if path in sys.path:
            sys.path.remove(path)
    
    if tensorflow_path not in sys.path:
        sys.path.insert(0, tensorflow_path)
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        from classes.objects import vine_obj_bin, margin_obj
        from sampling.vine_sample import vine_copula_sample, vine_cop_par_sample
        from pre_proc.preparation import prep_cop
        
        TENSORFLOW_AVAILABLE = True
        print("✓ TensorFlow DVC modules loaded successfully")
        
        return {
            'vine_obj_bin': vine_obj_bin,
            'margin_obj': margin_obj,
            'vine_sample': vine_copula_sample,
            'vine_par_sample': vine_cop_par_sample,
            'prep_cop': prep_cop,
            'tf': tf
        }
        
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow DVC: {e}")
        TENSORFLOW_AVAILABLE = False
        return None

def setup_pytorch():
    """Setup PyTorch DVC imports"""
    global PYTORCH_AVAILABLE
    
    # Clean path
    paths_to_remove = [p for p in sys.path if 'tensorflow' in p.lower()]
    for path in paths_to_remove:
        if path in sys.path:
            sys.path.remove(path)
    
    if pytorch_path not in sys.path:
        sys.path.insert(0, pytorch_path)
    
    try:
        import torch
        
        from classes.objects import vine_obj_bin, margin_obj
        from pre_proc.preparation import prep_cop
        from info.info_estimation import vine_entropy
        
        PYTORCH_AVAILABLE = True
        print("✓ PyTorch DVC modules loaded successfully")
        
        return {
            'vine_obj_bin': vine_obj_bin,
            'margin_obj': margin_obj,
            'prep_cop': prep_cop,
            'vine_entropy': vine_entropy,
            'torch': torch
        }
        
    except ImportError as e:
        print(f"✗ Failed to import PyTorch DVC: {e}")
        PYTORCH_AVAILABLE = False
        return None

def generate_multivariate_data(n_samples: int, dim: int, correlation_type: str = 'ar1', 
                             seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate multivariate data with known correlation structure"""
    np.random.seed(seed)
    
    # Create correlation matrix
    if correlation_type == 'ar1':
        rho = 0.7
        corr = np.array([[rho ** abs(i - j) for j in range(dim)] for i in range(dim)])
    elif correlation_type == 'toeplitz':
        rho = 0.6
        corr = np.array([[rho ** abs(i - j) for j in range(dim)] for i in range(dim)])
    elif correlation_type == 'block':
        corr = np.eye(dim)
        block_size = dim // 2
        for i in range(block_size):
            for j in range(block_size):
                if i != j:
                    corr[i, j] = 0.8
        for i in range(block_size, dim):
            for j in range(block_size, dim):
                if i != j:
                    corr[i, j] = 0.6
    elif correlation_type == 'random':
        A = np.random.randn(dim, dim)
        corr = np.dot(A, A.T)
        D = np.sqrt(np.diag(corr))
        corr = corr / np.outer(D, D)
    else:
        corr = np.eye(dim)
    
    # Generate multivariate normal data
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, corr, n_samples)
    
    # Transform margins for complexity
    if dim > 1:
        data[:, 1] = stats.expon.ppf(stats.norm.cdf(data[:, 1]), scale=2)
    if dim > 2:
        data[:, 2] = stats.uniform.ppf(stats.norm.cdf(data[:, 2]), loc=-1, scale=2)
    if dim > 3:
        data[:, 3] = stats.gamma.ppf(stats.norm.cdf(data[:, 3]), a=2, scale=1)
    
    # Calculate ground truth entropy (differential entropy of multivariate normal)
    det_corr = np.linalg.det(corr)
    ground_truth_entropy = 0.5 * dim * (1 + np.log(2 * np.pi)) + 0.5 * np.log(det_corr)
    
    return data.astype(np.float32), corr, ground_truth_entropy

def compute_correlation_matrix(data: np.ndarray, method: str = 'kendall') -> np.ndarray:
    """Compute correlation matrix"""
    n, d = data.shape
    corr = np.eye(d)
    
    for i in range(d):
        for j in range(i+1, d):
            if method == 'kendall':
                tau, _ = stats.kendalltau(data[:, i], data[:, j])
                corr[i, j] = tau
                corr[j, i] = tau
            elif method == 'spearman':
                rho, _ = stats.spearmanr(data[:, i], data[:, j])
                corr[i, j] = rho
                corr[j, i] = rho
            else:  # pearson
                r = np.corrcoef(data[:, i], data[:, j])[0, 1]
                corr[i, j] = r
                corr[j, i] = r
    
    return corr

def create_margins(dim: int, framework: str):
    """Create margin objects for the specified framework"""
    margins = []
    
    if framework == 'tensorflow':
        tf_modules = setup_tensorflow()
        if tf_modules:
            for i in range(dim):
                mar = tf_modules['margin_obj']('norm', [0, 1], True)
                margins.append(mar)
    elif framework == 'pytorch':
        pt_modules = setup_pytorch()
        if pt_modules:
            for i in range(dim):
                mar = pt_modules['margin_obj']('norm', [0, 1], True)
                margins.append(mar)
    
    return margins

def fit_tensorflow_vine(data: np.ndarray, vine_type: str, parametric: bool, 
                       vine_depth: int) -> Tuple[Any, float, bool]:
    """Fit TensorFlow vine copula"""
    tf_modules = setup_tensorflow()
    if not tf_modules:
        return None, 0.0, False
    
    try:
        n_samples, dim = data.shape
        margins = create_margins(dim, 'tensorflow')
        
        families = "param" if parametric else "kercop"
        knots = 50
        method = 'matrix'
        
        if vine_type == 'r-vine':
            r_matrix = np.array([[i+1 if i == j else 0 for j in range(dim)] for i in range(dim)], dtype=np.int32)
            vine = tf_modules['vine_obj_bin'](vine_type, families, vine_depth, margins, knots, method, r_matrix)
        else:
            vine = tf_modules['vine_obj_bin'](vine_type, families, vine_depth, margins, knots, method)
        
        tf_modules['prep_cop'](data, vine, 'rand')
        
        gen_dict = {
            'parallel': True, 
            'binning': False, 
            'param': parametric, 
            'vine_depth': vine_depth - 1,
            'fitted': False
        }
        
        if parametric:
            par_dict = {'param_families': ["ind", "gaussian", "student", "clayton"]}
            npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
        else:
            par_dict = {'param_families': ["ind", "gaussian"]}
            npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
        
        bin_dict = {'n_bin': 3}
        
        start_time = time.time()
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        return vine, fit_time, True
        
    except Exception as e:
        print(f"TensorFlow fitting failed: {str(e)}")
        return None, 0.0, False

def fit_pytorch_vine(data: np.ndarray, vine_type: str, parametric: bool, 
                    vine_depth: int) -> Tuple[Any, float, bool]:
    """Fit PyTorch vine copula using the working pattern from the example"""
    pt_modules = setup_pytorch()
    if not pt_modules:
        return None, 0.0, False
    
    try:
        n_samples, dim = data.shape
        
        # Create margin objects
        margins = []
        for i in range(dim):
            margins.append(pt_modules['margin_obj']('norm', [0, 1], True))
        
        # Create vine object with simplified config (following the working example)
        if vine_type == 'r-vine':
            vine = pt_modules['vine_obj_bin']('r-vine', ['gaussian'], dim, margins, 11, 'random')
        else:
            vine = pt_modules['vine_obj_bin'](vine_type, ['gaussian'], dim, margins, 11, 'matrix')
        
        # Use numpy data directly and let prep_cop handle the conversion
        data_prep = pt_modules['prep_cop'](data, vine, 'no_sort')
        
        # Convert to torch tensor after prep_cop
        device = pt_modules['torch'].device('cpu')
        if isinstance(data_prep, np.ndarray):
            data_prep = pt_modules['torch'].from_numpy(data_prep.astype(np.float32)).to(device)
        elif not isinstance(data_prep, pt_modules['torch'].Tensor):
            data_prep = pt_modules['torch'].tensor(data_prep, dtype=pt_modules['torch'].float32, device=device)
        
        # Configure fitting parameters based on parametric vs non-parametric
        if parametric:
            gen_dict = {
                'param': True,
                'binning': False,
                'fitted': False,
                'parallel': True,
                'vine_depth': min(dim - 1, 4)  # Allow up to 4 trees for larger dimensions
            }
            
            par_dict = {
                'param_families': ['gaussian', 'student', 'clayton']  # Multiple families for better fit
            }
        else:
            gen_dict = {
                'param': False,  # Use non-parametric
                'binning': True,
                'fitted': False,
                'parallel': True,
                'vine_depth': min(dim - 1, 3)  # Reduced depth for non-parametric
            }
            
            par_dict = {
                'param_families': ['gaussian']  # Fallback for non-parametric
            }
        
        npc_dict = {
            'opt_method': 'trust-exact',
            'batch_paral': 10
        }
        
        bin_dict = {
            'n_bin': 10 if not parametric else 1  # More bins for non-parametric
        }
        
        start_time = time.time()
        vine.fit(data_prep, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        return vine, fit_time, True
        
    except Exception as e:
        print(f"PyTorch fitting failed: {str(e)}")
        return None, 0.0, False

def sample_tensorflow_vine(vine: Any, n_samples: int, parametric: bool) -> Tuple[np.ndarray, bool]:
    """Sample from TensorFlow vine"""
    tf_modules = setup_tensorflow()
    if not tf_modules or vine is None:
        return None, False
    
    try:
        if parametric:
            samples = tf_modules['vine_par_sample'](vine, n_samples)
        else:
            samples, _, _, _ = tf_modules['vine_sample'](vine, n_samples)
        return samples, True
    except Exception as e:
        print(f"TensorFlow sampling failed: {str(e)}")
        return None, False

def sample_pytorch_vine(vine: Any, n_samples: int, parametric: bool) -> Tuple[np.ndarray, bool]:
    """Sample from PyTorch vine"""
    pt_modules = setup_pytorch()
    if not pt_modules or vine is None:
        return None, False
    
    try:
        # Use the sample method from the vine object
        samples = vine.sample(n_samples)
        if isinstance(samples, pt_modules['torch'].Tensor):
            samples = samples.detach().cpu().numpy()
        return samples, True
    except Exception as e:
        print(f"PyTorch sampling failed: {str(e)}")
        return None, False

def compute_tensorflow_entropy(vine: Any, data: np.ndarray) -> Tuple[float, bool]:
    """Compute entropy using TensorFlow implementation"""
    tf_modules = setup_tensorflow()
    if not tf_modules or vine is None:
        return 0.0, False
    
    try:
        # TensorFlow entropy computation would go here
        # For now, return a placeholder
        return 0.0, False
    except Exception as e:
        print(f"TensorFlow entropy computation failed: {str(e)}")
        return 0.0, False

def compute_pytorch_entropy(vine: Any, data: np.ndarray) -> Tuple[float, bool]:
    """Compute entropy using PyTorch implementation"""
    pt_modules = setup_pytorch()
    if not pt_modules or vine is None:
        return 0.0, False
    
    try:
        # Use the entropy function from the working example
        info_dict = {'alpha': 0.05, 'cases': 1000, 'iterations': 10}
        entropy = pt_modules['vine_entropy'](vine, info_dict)
        return float(entropy), True
    except Exception as e:
        print(f"PyTorch entropy computation failed: {str(e)}")
        return 0.0, False

def analyze_results(true_corr: np.ndarray, data_corr: np.ndarray, 
                   sample_corr: np.ndarray, method_name: str) -> Dict:
    """Analyze correlation recovery results"""
    
    data_error = np.mean(np.abs(data_corr - true_corr))
    sample_error = np.mean(np.abs(sample_corr - true_corr))
    
    true_flat = true_corr[np.triu_indices_from(true_corr, k=1)]
    sample_flat = sample_corr[np.triu_indices_from(sample_corr, k=1)]
    
    if len(true_flat) > 1:
        recovery_corr, _ = stats.pearsonr(true_flat, sample_flat)
    else:
        recovery_corr = 1.0 if len(true_flat) == 1 and np.abs(true_flat[0] - sample_flat[0]) < 0.1 else 0.0
    
    max_error = np.max(np.abs(sample_corr - true_corr))
    
    return {
        'method': method_name,
        'data_error': data_error,
        'sample_error': sample_error,
        'recovery_correlation': recovery_corr,
        'max_error': max_error,
        'mean_true_corr': np.mean(np.abs(true_flat)),
        'mean_sample_corr': np.mean(np.abs(sample_flat))
    }

def run_comprehensive_comparison():
    """Run comprehensive comparison between PyTorch and TensorFlow DVC"""
    
    print("Comprehensive DVC Comparison: PyTorch vs TensorFlow")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        {'dim': 3, 'n_samples': 300, 'corr_type': 'ar1'},
        {'dim': 3, 'n_samples': 300, 'corr_type': 'block'},
        {'dim': 4, 'n_samples': 400, 'corr_type': 'toeplitz'},
    ]
    
    vine_types = ['c-vine', 'd-vine']
    methods = [
        {'name': 'Parametric', 'parametric': True},
        {'name': 'Non-parametric', 'parametric': False}
    ]
    
    frameworks = ['tensorflow', 'pytorch']
    all_results = []
    
    for config in test_configs:
        dim = config['dim']
        n_samples = config['n_samples']
        corr_type = config['corr_type']
        
        print(f"\nTesting: dim={dim}, n_samples={n_samples}, correlation={corr_type}")
        print("-" * 70)
        
        # Generate test data
        data, true_corr, ground_truth_entropy = generate_multivariate_data(n_samples, dim, corr_type)
        data_corr = compute_correlation_matrix(data, method='kendall')
        
        print(f"Data shape: {data.shape}")
        print(f"True correlation range: [{np.min(true_corr):.3f}, {np.max(true_corr):.3f}]")
        print(f"Ground truth entropy: {ground_truth_entropy:.4f}")
        
        for vine_type in vine_types:
            print(f"\n  Testing {vine_type}:")
            
            for method in methods:
                method_name = method['name']
                parametric = method['parametric']
                
                print(f"    {method_name} method:")
                
                for framework in frameworks:
                    print(f"      {framework.upper()}...", end=' ')
                    
                    try:
                        # Fit vine
                        if framework == 'tensorflow':
                            vine, fit_time, fit_success = fit_tensorflow_vine(
                                data, vine_type, parametric, dim
                            )
                        else:  # pytorch
                            vine, fit_time, fit_success = fit_pytorch_vine(
                                data, vine_type, parametric, dim
                            )
                        
                        if not fit_success:
                            print("Fit failed")
                            continue
                        
                        # Sample from vine
                        if framework == 'tensorflow':
                            samples, sample_success = sample_tensorflow_vine(vine, 500, parametric)
                        else:  # pytorch
                            samples, sample_success = sample_pytorch_vine(vine, 500, parametric)
                        
                        if not sample_success:
                            print("Sampling failed")
                            continue
                        
                        # Compute sample correlations
                        sample_corr = compute_correlation_matrix(samples, method='kendall')
                        
                        # Compute entropy
                        if framework == 'tensorflow':
                            estimated_entropy, entropy_success = compute_tensorflow_entropy(vine, data)
                        else:  # pytorch
                            estimated_entropy, entropy_success = compute_pytorch_entropy(vine, data)
                        
                        if not entropy_success:
                            estimated_entropy = np.nan
                        
                        # Analyze results
                        analysis = analyze_results(true_corr, data_corr, sample_corr, 
                                                 f"{framework.upper()} {method_name} {vine_type}")
                        
                        # Add additional metrics
                        analysis.update({
                            'framework': framework,
                            'fit_time': fit_time,
                            'vine_type': vine_type,
                            'dimension': dim,
                            'n_samples': n_samples,
                            'correlation_type': corr_type,
                            'ground_truth_entropy': ground_truth_entropy,
                            'estimated_entropy': estimated_entropy,
                            'entropy_error': abs(estimated_entropy - ground_truth_entropy) if not np.isnan(estimated_entropy) else np.nan,
                            'sample_correlation': sample_corr
                        })
                        
                        all_results.append(analysis)
                        
                        entropy_str = f"Entropy Error: {analysis['entropy_error']:.4f}, " if not np.isnan(estimated_entropy) else ""
                        print(f"Success! Corr Error: {analysis['sample_error']:.4f}, "
                              f"Recovery: {analysis['recovery_correlation']:.4f}, "
                              f"{entropy_str}"
                              f"Time: {fit_time:.2f}s")
                        
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        continue
    
    # Create comprehensive analysis
    if all_results:
        df = pd.DataFrame(all_results)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DVC COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal successful tests: {len(df)}")
        
        # Framework comparison
        print(f"\nFramework Comparison:")
        for framework in df['framework'].unique():
            fw_df = df[df['framework'] == framework]
            print(f"  {framework.upper()}:")
            print(f"    Tests completed:               {len(fw_df)}")
            print(f"    Average correlation error:     {fw_df['sample_error'].mean():.4f} ± {fw_df['sample_error'].std():.4f}")
            print(f"    Average recovery correlation:  {fw_df['recovery_correlation'].mean():.4f} ± {fw_df['recovery_correlation'].std():.4f}")
            print(f"    Average fit time:              {fw_df['fit_time'].mean():.2f} ± {fw_df['fit_time'].std():.2f} seconds")
            print(f"    Average max error:             {fw_df['max_error'].mean():.4f} ± {fw_df['max_error'].std():.4f}")
            
            # Entropy analysis
            entropy_errors = fw_df['entropy_error'].dropna()
            if len(entropy_errors) > 0:
                print(f"    Average entropy error:         {entropy_errors.mean():.4f} ± {entropy_errors.std():.4f}")
        
        # Method comparison
        print(f"\nMethod Comparison:")
        for method_type in ['Parametric', 'Non-parametric']:
            method_df = df[df['method'].str.contains(method_type)]
            if len(method_df) > 0:
                print(f"  {method_type}:")
                print(f"    Average correlation error:     {method_df['sample_error'].mean():.4f} ± {method_df['sample_error'].std():.4f}")
                print(f"    Average recovery correlation:  {method_df['recovery_correlation'].mean():.4f} ± {method_df['recovery_correlation'].std():.4f}")
                print(f"    Average fit time:              {method_df['fit_time'].mean():.2f} ± {method_df['fit_time'].std():.2f} seconds")
                
                entropy_errors = method_df['entropy_error'].dropna()
                if len(entropy_errors) > 0:
                    print(f"    Average entropy error:         {entropy_errors.mean():.4f} ± {entropy_errors.std():.4f}")
        
        # Vine type comparison
        print(f"\nVine Type Comparison:")
        for vine_type in df['vine_type'].unique():
            vine_df = df[df['vine_type'] == vine_type]
            print(f"  {vine_type}:")
            print(f"    Average correlation error:     {vine_df['sample_error'].mean():.4f} ± {vine_df['sample_error'].std():.4f}")
            print(f"    Average recovery correlation:  {vine_df['recovery_correlation'].mean():.4f} ± {vine_df['recovery_correlation'].std():.4f}")
        
        # Save results
        df.to_csv('comprehensive_dvc_comparison_results.csv', index=False)
        
        # Create visualizations
        create_comparison_plots(df)
        
        print(f"\nResults saved to:")
        print(f"  - comprehensive_dvc_comparison_results.csv")
        print(f"  - comprehensive_dvc_comparison_plots.png")
        
        # Key insights
        print_key_insights(df)

def create_comparison_plots(df: pd.DataFrame):
    """Create comprehensive comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Correlation error by framework
    df.boxplot(column='sample_error', by='framework', ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Error by Framework')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    
    # Plot 2: Recovery correlation by framework
    df.boxplot(column='recovery_correlation', by='framework', ax=axes[0, 1])
    axes[0, 1].set_title('Recovery Correlation by Framework')
    axes[0, 1].set_ylabel('Correlation with True Values')
    
    # Plot 3: Fit time by framework
    df.boxplot(column='fit_time', by='framework', ax=axes[0, 2])
    axes[0, 2].set_title('Fit Time by Framework')
    axes[0, 2].set_ylabel('Time (seconds)')
    
    # Plot 4: Entropy error by framework
    entropy_df = df.dropna(subset=['entropy_error'])
    if len(entropy_df) > 0:
        entropy_df.boxplot(column='entropy_error', by='framework', ax=axes[1, 0])
        axes[1, 0].set_title('Entropy Error by Framework')
        axes[1, 0].set_ylabel('Absolute Entropy Error')
    
    # Plot 5: Error by vine type and framework
    for framework in df['framework'].unique():
        fw_data = df[df['framework'] == framework]
        axes[1, 1].scatter(fw_data['dimension'], fw_data['sample_error'], 
                          label=f'{framework.upper()}', alpha=0.7, s=60)
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Correlation Error')
    axes[1, 1].set_title('Error vs Dimension by Framework')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Framework vs vine type heatmap
    pivot_data = df.pivot_table(values='sample_error', index='framework', columns='vine_type', aggfunc='mean')
    sns.heatmap(pivot_data, ax=axes[1, 2], annot=True, fmt='.4f', cmap='YlOrRd')
    axes[1, 2].set_title('Mean Error: Framework vs Vine Type')
    
    plt.suptitle('Comprehensive DVC Comparison: PyTorch vs TensorFlow', fontsize=16)
    plt.tight_layout()
    plt.savefig('comprehensive_dvc_comparison_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

def print_key_insights(df: pd.DataFrame):
    """Print key insights from the comparison"""
    
    print(f"\nKey Insights:")
    
    # Best overall method
    best_idx = df['sample_error'].idxmin()
    best_method = df.loc[best_idx, 'method']
    best_framework = df.loc[best_idx, 'framework']
    best_error = df.loc[best_idx, 'sample_error']
    print(f"  - Best overall: {best_framework.upper()} {best_method} (error: {best_error:.4f})")
    
    # Framework comparison
    tf_results = df[df['framework'] == 'tensorflow']
    pt_results = df[df['framework'] == 'pytorch']
    
    if len(tf_results) > 0 and len(pt_results) > 0:
        tf_error = tf_results['sample_error'].mean()
        pt_error = pt_results['sample_error'].mean()
        
        if tf_error < pt_error:
            print(f"  - TensorFlow performed better on average ({tf_error:.4f} vs {pt_error:.4f})")
        else:
            print(f"  - PyTorch performed better on average ({pt_error:.4f} vs {tf_error:.4f})")
        
        # Speed comparison
        tf_time = tf_results['fit_time'].mean()
        pt_time = pt_results['fit_time'].mean()
        
        if tf_time < pt_time:
            print(f"  - TensorFlow was faster on average ({tf_time:.2f}s vs {pt_time:.2f}s)")
        else:
            print(f"  - PyTorch was faster on average ({pt_time:.2f}s vs {tf_time:.2f}s)")
        
        # Entropy comparison
        entropy_df = df.dropna(subset=['entropy_error'])
        if len(entropy_df) > 0:
            tf_entropy = entropy_df[entropy_df['framework'] == 'tensorflow']['entropy_error'].mean()
            pt_entropy = entropy_df[entropy_df['framework'] == 'pytorch']['entropy_error'].mean()
            
            if not np.isnan(tf_entropy) and not np.isnan(pt_entropy):
                if tf_entropy < pt_entropy:
                    print(f"  - TensorFlow had better entropy estimation ({tf_entropy:.4f} vs {pt_entropy:.4f})")
                else:
                    print(f"  - PyTorch had better entropy estimation ({pt_entropy:.4f} vs {tf_entropy:.4f})")
    
    elif len(tf_results) > 0:
        print(f"  - Only TensorFlow results available (avg error: {tf_results['sample_error'].mean():.4f})")
    elif len(pt_results) > 0:
        print(f"  - Only PyTorch results available (avg error: {pt_results['sample_error'].mean():.4f})")
    
    # Overall assessment
    avg_recovery = df['recovery_correlation'].mean()
    print(f"  - Average correlation recovery: {avg_recovery:.4f}")
    
    if avg_recovery > 0.9:
        print(f"  - Excellent overall performance achieved")
    elif avg_recovery > 0.8:
        print(f"  - Good overall performance achieved")
    else:
        print(f"  - Moderate overall performance achieved")
    
    # Success rates
    total_possible = len(df['framework'].unique()) * 3 * 2 * 2  # frameworks * configs * vine_types * methods
    actual_tests = len(df)
    success_rate = actual_tests / total_possible * 100
    print(f"  - Test success rate: {success_rate:.1f}% ({actual_tests}/{total_possible})")

if __name__ == "__main__":
    run_comprehensive_comparison() 