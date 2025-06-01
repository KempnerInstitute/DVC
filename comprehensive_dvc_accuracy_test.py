#!/usr/bin/env python3
"""
Comprehensive DVC Accuracy Test

This test evaluates PyTorch and TensorFlow Deep Vine Copula implementations on:
1. Pairwise interaction estimation accuracy (correlation recovery)
2. Entropy estimation accuracy compared to ground truth
3. Overall model performance comparison

The test uses standardized data generation and evaluation metrics to provide
a fair comparison between the two implementations.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DVCAccuracyTester:
    """Comprehensive accuracy tester for DVC implementations."""
    
    def __init__(self, results_dir="dvc_accuracy_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = []
    
    def generate_test_data(self, n_samples=1000, n_dims=4, correlation_type='ar1', seed=42):
        """Generate test data with known correlation structure and true entropy."""
        np.random.seed(seed)
        
        # Create correlation matrix based on type
        if correlation_type == 'ar1':
            rho = 0.7
            corr_matrix = np.eye(n_dims)
            for i in range(n_dims):
                for j in range(n_dims):
                    corr_matrix[i, j] = rho ** abs(i - j)
                    
        elif correlation_type == 'block':
            corr_matrix = np.eye(n_dims)
            block_size = n_dims // 2
            # First block
            for i in range(block_size):
                for j in range(block_size):
                    if i != j:
                        corr_matrix[i, j] = 0.8
            # Second block
            for i in range(block_size, n_dims):
                for j in range(block_size, n_dims):
                    if i != j:
                        corr_matrix[i, j] = 0.6
                        
        elif correlation_type == 'toeplitz':
            rho = 0.6
            corr_matrix = np.zeros((n_dims, n_dims))
            for i in range(n_dims):
                for j in range(n_dims):
                    corr_matrix[i, j] = rho ** abs(i - j)
        else:
            corr_matrix = np.eye(n_dims)
        
        # Generate multivariate normal data
        data = np.random.multivariate_normal(
            mean=np.zeros(n_dims),
            cov=corr_matrix,
            size=n_samples
        )
        
        # Calculate true entropy (for multivariate normal)
        true_entropy = 0.5 * n_dims * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.linalg.det(corr_matrix))
        
        return data.astype(np.float32), corr_matrix, true_entropy
    
    def compute_pairwise_interactions(self, data):
        """Compute pairwise correlation matrix using Kendall's tau."""
        n_samples, n_dims = data.shape
        corr_matrix = np.eye(n_dims)
        
        for i in range(n_dims):
            for j in range(i+1, n_dims):
                tau, _ = stats.kendalltau(data[:, i], data[:, j])
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
        
        return corr_matrix
    
    def estimate_entropy_from_samples(self, samples):
        """Estimate entropy from samples using various methods."""
        try:
            # Method 1: Differential entropy estimation using determinant approach
            sample_cov = np.cov(samples.T)
            if np.linalg.det(sample_cov) > 0:
                entropy_kde = 0.5 * samples.shape[1] * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.linalg.det(sample_cov))
            else:
                entropy_kde = np.nan
            
            # Method 2: Empirical entropy using histogram (for comparison)
            hist_entropy = 0.0
            for dim in range(samples.shape[1]):
                hist, bin_edges = np.histogram(samples[:, dim], bins=50, density=True)
                bin_width = bin_edges[1] - bin_edges[0]
                # Remove zero probabilities
                hist = hist[hist > 0]
                hist_entropy += -np.sum(hist * np.log(hist) * bin_width)
            
            return {
                'kde_entropy': entropy_kde,
                'histogram_entropy': hist_entropy,
                'primary_entropy': entropy_kde
            }
        except Exception as e:
            print(f"    Entropy estimation failed: {e}")
            return {
                'kde_entropy': np.nan,
                'histogram_entropy': np.nan,
                'primary_entropy': np.nan
            }
    
    def test_pytorch_dvc(self, data, vine_type='d-vine', parametric=True):
        """Test PyTorch DVC implementation."""
        try:
            # Temporarily remove TensorFlow path to avoid conflicts
            tensorflow_path = os.path.join(os.getcwd(), 'src', 'DVC_tensorflow')
            if tensorflow_path in sys.path:
                sys.path.remove(tensorflow_path)
            
            # Add PyTorch path
            pytorch_path = os.path.join(os.getcwd(), 'src', 'DVC_pytorch')
            if pytorch_path not in sys.path:
                sys.path.insert(0, pytorch_path)
            
            # Clear any cached modules that might cause conflicts
            modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith(('sampling', 'param', 'classes'))]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]
            
            import torch
            from classes.objects import vine_obj_bin, margin_obj
            from sampling.vine_sampler import VineSampler
            
            print(f"  Testing PyTorch DVC: {vine_type}, {'parametric' if parametric else 'non-parametric'}")
            
            n_samples, dim = data.shape
            start_time = time.time()
            
            # Convert data to PyTorch tensor
            data_torch = torch.tensor(data, dtype=torch.float32)
            
            # Create margins
            margin_vine = []
            for i in range(dim):
                mar_p = margin_obj('norm', [0, 1], True)
                margin_vine.append(mar_p)
            
            # Create vine object
            families = ['gaussian', 'clayton'] if parametric else 'kercop'
            vine = vine_obj_bin(
                vine_family=vine_type,
                families=families,
                vine_depth=dim-1,  # PyTorch uses depth-1
                margin=margin_vine,
                knots=50,
                method='matrix'
            )
            
            # Configuration dictionaries based on working example
            gen_dict = {
                "parallel": False,
                "param": parametric,
                "binning": False,
                "fitted": False,
                "vine_depth": dim-1
            }
            
            if parametric:
                par_dict = {"param_families": ["gaussian", "clayton", "student"]}
            else:
                par_dict = {"param_families": ["gaussian"]}
                
            npc_dict = {"opt_method": "LL1", "batch_paral": False}
            bin_dict = {"n_bin": 1}
            
            # Fit the vine
            vine.fit(data_torch, gen_dict, npc_dict, par_dict, bin_dict)
            fit_time = time.time() - start_time
            
            # Generate samples using VineSampler
            start_time = time.time()
            sampler = VineSampler(vine)
            samples, u_samples = sampler.sample(n_samples)
            sample_time = time.time() - start_time
            
            # Convert back to numpy
            if isinstance(samples, torch.Tensor):
                samples = samples.numpy()
            
            return {
                'success': True,
                'samples': samples,
                'fit_time': fit_time,
                'sample_time': sample_time,
                'error': None
            }
            
        except Exception as e:
            print(f"    PyTorch failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'samples': None,
                'fit_time': None,
                'sample_time': None,
                'error': str(e)
            }
        finally:
            # Restore TensorFlow path for subsequent tests
            if tensorflow_path not in sys.path:
                sys.path.insert(0, tensorflow_path)
    
    def test_tensorflow_dvc(self, data, vine_type='d-vine', parametric=True):
        """Test TensorFlow DVC implementation."""
        try:
            # Add TensorFlow path
            tensorflow_path = os.path.join(os.getcwd(), 'src', 'DVC_tensorflow')
            if tensorflow_path not in sys.path:
                sys.path.insert(0, tensorflow_path)
            
            from classes.objects import vine_obj_bin, margin_obj
            from sampling.vine_sample import vine_copula_sample, vine_cop_par_sample
            from pre_proc.preparation import prep_cop
            
            print(f"  Testing TensorFlow DVC: {vine_type}, {'parametric' if parametric else 'non-parametric'}")
            
            n_samples, dim = data.shape
            start_time = time.time()
            
            # Create margins with proper setup
            margin_vine = []
            for i in range(dim):
                mar_p = margin_obj('norm', [0.0, 1.0], True)
                mar_p.ker = data[:, i]  # Critical for TensorFlow
                margin_vine.append(mar_p)
            
            # Create vine object
            families = 'param' if parametric else 'npc'
            vine = vine_obj_bin(
                vine_family=vine_type,
                families=families,
                vine_depth=dim,
                margin=margin_vine,
                knots=50,
                method='matrix'
            )
            
            # Prepare data
            prep_cop(data, vine, 'rand')
            
            # Configuration based on working examples
            gen_dict = {
                'parallel': False,
                'binning': False,
                'param': parametric,
                'vine_depth': dim,
                'fitted': False
            }
            
            if parametric:
                par_dict = {'param_families': ["ind", "gaussian", "clayton"]}
            else:
                par_dict = {}
                
            npc_dict = {'opt_method': 'LL1', 'batch_paral': False}
            bin_dict = {'n_bin': 1}
            
            # Fit the vine
            vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
            fit_time = time.time() - start_time
            
            # Generate samples
            start_time = time.time()
            if parametric:
                samples = vine_cop_par_sample(vine, n_samples)
            else:
                result = vine_copula_sample(vine, n_samples)
                samples = result[0] if isinstance(result, tuple) else result
            sample_time = time.time() - start_time
            
            return {
                'success': True,
                'samples': samples,
                'fit_time': fit_time,
                'sample_time': sample_time,
                'error': None
            }
            
        except Exception as e:
            print(f"    TensorFlow failed: {e}")
            return {
                'success': False,
                'samples': None,
                'fit_time': None,
                'sample_time': None,
                'error': str(e)
            }
    
    def evaluate_accuracy(self, original_data, samples, true_corr, true_entropy, method_name):
        """Evaluate accuracy of pairwise interactions and entropy estimation."""
        if samples is None:
            return None
        
        # Ensure samples is numpy array
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)
        
        if samples.size == 0:
            return None
        
        print(f"    Evaluating {method_name}: samples shape {samples.shape}")
        
        # 1. Pairwise interaction accuracy
        sample_corr = self.compute_pairwise_interactions(samples)
        original_corr = self.compute_pairwise_interactions(original_data)
        
        # Correlation metrics
        corr_diff = sample_corr - true_corr
        mae_corr = np.mean(np.abs(corr_diff))
        rmse_corr = np.sqrt(np.mean(corr_diff**2))
        max_abs_diff = np.max(np.abs(corr_diff))
        
        # Correlation of correlations (how well structure is preserved)
        true_upper = true_corr[np.triu_indices_from(true_corr, k=1)]
        sample_upper = sample_corr[np.triu_indices_from(sample_corr, k=1)]
        
        if len(true_upper) > 1:
            structure_recovery, _ = pearsonr(true_upper, sample_upper)
        else:
            structure_recovery = 1.0 if len(true_upper) == 1 else np.nan
        
        # 2. Entropy estimation accuracy
        entropy_metrics = self.estimate_entropy_from_samples(samples)
        entropy_error = abs(entropy_metrics['primary_entropy'] - true_entropy)
        entropy_relative_error = entropy_error / abs(true_entropy) if true_entropy != 0 else np.inf
        
        return {
            'method': method_name,
            'samples': samples,
            'sample_correlation': sample_corr,
            'original_correlation': original_corr,
            'true_correlation': true_corr,
            
            # Pairwise interaction metrics
            'correlation_mae': mae_corr,
            'correlation_rmse': rmse_corr,
            'correlation_max_diff': max_abs_diff,
            'structure_recovery': structure_recovery,
            
            # Entropy metrics
            'estimated_entropy': entropy_metrics['primary_entropy'],
            'true_entropy': true_entropy,
            'entropy_error': entropy_error,
            'entropy_relative_error': entropy_relative_error,
            'entropy_details': entropy_metrics
        }
    
    def create_visualization(self, results, scenario_name):
        """Create comprehensive visualization of results."""
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("    No valid results to plot")
            return
            
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'DVC Accuracy Comparison - {scenario_name}', fontsize=16)
        
        methods = [r['method'] for r in valid_results]
        
        # 1. Correlation MAE
        correlation_maes = [r['correlation_mae'] for r in valid_results]
        if correlation_maes:
            colors = ['skyblue' if 'TensorFlow' in m else 'lightcoral' for m in methods]
            axes[0, 0].bar(range(len(methods)), correlation_maes, alpha=0.7, color=colors)
            axes[0, 0].set_title('Pairwise Interaction MAE')
            axes[0, 0].set_ylabel('Mean Absolute Error')
            axes[0, 0].set_xticks(range(len(methods)))
            axes[0, 0].set_xticklabels(methods, rotation=45)
        
        # 2. Structure Recovery
        structure_recoveries = [r['structure_recovery'] for r in valid_results]
        if structure_recoveries:
            colors = ['skyblue' if 'TensorFlow' in m else 'lightcoral' for m in methods]
            axes[0, 1].bar(range(len(methods)), structure_recoveries, alpha=0.7, color=colors)
            axes[0, 1].set_title('Correlation Structure Recovery')
            axes[0, 1].set_ylabel('Correlation with True Structure')
            axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect')
            axes[0, 1].legend()
            axes[0, 1].set_xticks(range(len(methods)))
            axes[0, 1].set_xticklabels(methods, rotation=45)
        
        # 3. Entropy Error
        entropy_errors = [r['entropy_relative_error'] for r in valid_results]
        if entropy_errors:
            colors = ['skyblue' if 'TensorFlow' in m else 'lightcoral' for m in methods]
            axes[0, 2].bar(range(len(methods)), entropy_errors, alpha=0.7, color=colors)
            axes[0, 2].set_title('Entropy Estimation Relative Error')
            axes[0, 2].set_ylabel('Relative Error')
            axes[0, 2].set_xticks(range(len(methods)))
            axes[0, 2].set_xticklabels(methods, rotation=45)
        
        # 4. Performance comparison
        fit_times = [r.get('fit_time', 0) for r in valid_results]
        sample_times = [r.get('sample_time', 0) for r in valid_results]
        if fit_times:
            x = np.arange(len(methods))
            width = 0.35
            colors_fit = ['darkblue' if 'TensorFlow' in m else 'darkred' for m in methods]
            colors_sample = ['lightblue' if 'TensorFlow' in m else 'pink' for m in methods]
            
            axes[0, 3].bar(x - width/2, fit_times, width, label='Fit Time', color=colors_fit, alpha=0.7)
            axes[0, 3].bar(x + width/2, sample_times, width, label='Sample Time', color=colors_sample, alpha=0.7)
            axes[0, 3].set_title('Performance Comparison')
            axes[0, 3].set_ylabel('Time (seconds)')
            axes[0, 3].set_xticks(x)
            axes[0, 3].set_xticklabels(methods, rotation=45)
            axes[0, 3].legend()
        
        # 5-7. Correlation heatmaps for best results from each implementation
        tf_results = [r for r in valid_results if 'TensorFlow' in r['method']]
        pt_results = [r for r in valid_results if 'PyTorch' in r['method']]
        
        if tf_results:
            best_tf = min(tf_results, key=lambda x: x['correlation_mae'])
            im = axes[1, 0].imshow(best_tf['sample_correlation'], cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title(f'TensorFlow Best - Sample Correlations')
            plt.colorbar(im, ax=axes[1, 0])
        
        if pt_results:
            best_pt = min(pt_results, key=lambda x: x['correlation_mae'])
            im = axes[1, 1].imshow(best_pt['sample_correlation'], cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_title(f'PyTorch Best - Sample Correlations')
            plt.colorbar(im, ax=axes[1, 1])
        
        # 8. True correlation for reference
        if valid_results:
            im = axes[1, 2].imshow(valid_results[0]['true_correlation'], cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 2].set_title('True Correlations')
            plt.colorbar(im, ax=axes[1, 2])
        
        # 9. Implementation comparison plot
        if tf_results and pt_results:
            tf_mae = np.mean([r['correlation_mae'] for r in tf_results])
            pt_mae = np.mean([r['correlation_mae'] for r in pt_results])
            tf_recovery = np.mean([r['structure_recovery'] for r in tf_results])
            pt_recovery = np.mean([r['structure_recovery'] for r in pt_results])
            
            implementations = ['TensorFlow', 'PyTorch']
            mae_values = [tf_mae, pt_mae]
            recovery_values = [tf_recovery, pt_recovery]
            
            x = np.arange(len(implementations))
            ax2 = axes[1, 3].twinx()
            
            bars1 = axes[1, 3].bar(x - 0.2, mae_values, 0.4, label='MAE', color=['skyblue', 'lightcoral'], alpha=0.7)
            bars2 = ax2.bar(x + 0.2, recovery_values, 0.4, label='Recovery', color=['darkblue', 'darkred'], alpha=0.7)
            
            axes[1, 3].set_xlabel('Implementation')
            axes[1, 3].set_ylabel('MAE', color='blue')
            ax2.set_ylabel('Structure Recovery', color='red')
            axes[1, 3].set_title('Implementation Comparison')
            axes[1, 3].set_xticks(x)
            axes[1, 3].set_xticklabels(implementations)
            
            # Add value labels on bars
            for bar, value in zip(bars1, mae_values):
                axes[1, 3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
            for bar, value in zip(bars2, recovery_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.results_dir, f"dvc_comparison_{scenario_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    Plots saved to: {plot_path}")
        plt.show()
    
    def generate_final_report(self, all_results):
        """Generate final comprehensive accuracy report."""
        report_path = os.path.join(self.results_dir, "comprehensive_dvc_accuracy_report.md")
        
        pytorch_results = [r for r in all_results if 'PyTorch' in r['method']]
        tensorflow_results = [r for r in all_results if 'TensorFlow' in r['method']]
        
        with open(report_path, 'w') as f:
            f.write("# Comprehensive DVC Accuracy Test Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- Total scenarios tested: {len(set(r['scenario']['name'] for r in all_results))}\n")
            f.write(f"- PyTorch successful runs: {len(pytorch_results)}\n")
            f.write(f"- TensorFlow successful runs: {len(tensorflow_results)}\n")
            f.write(f"- Total successful runs: {len(all_results)}\n\n")
            
            if pytorch_results and tensorflow_results:
                # Performance comparison
                f.write("## Implementation Comparison\n\n")
                f.write("| Implementation | Avg Correlation MAE | Avg Structure Recovery | Avg Entropy Rel Error | Avg Fit Time | Avg Sample Time |\n")
                f.write("|---------------|-------------------|---------------------|---------------------|-------------|----------------|\n")
                
                for impl_name, results in [('PyTorch', pytorch_results), ('TensorFlow', tensorflow_results)]:
                    avg_mae = np.mean([r['correlation_mae'] for r in results])
                    avg_recovery = np.mean([r['structure_recovery'] for r in results])
                    avg_entropy_err = np.mean([r['entropy_relative_error'] for r in results if not np.isinf(r['entropy_relative_error'])])
                    avg_fit_time = np.mean([r['fit_time'] for r in results])
                    avg_sample_time = np.mean([r['sample_time'] for r in results])
                    
                    f.write(f"| **{impl_name}** | {avg_mae:.4f} | {avg_recovery:.4f} | {avg_entropy_err:.4f} | {avg_fit_time:.3f}s | {avg_sample_time:.3f}s |\n")
                
                f.write("\n")
                
                # Winner analysis
                f.write("## Winner Analysis\n\n")
                pytorch_mae = np.mean([r['correlation_mae'] for r in pytorch_results])
                tensorflow_mae = np.mean([r['correlation_mae'] for r in tensorflow_results])
                f.write(f"**🎯 Pairwise Interaction Accuracy**: {'🥇 TensorFlow' if tensorflow_mae < pytorch_mae else '🥇 PyTorch'} ")
                f.write(f"(MAE: {min(pytorch_mae, tensorflow_mae):.4f} vs {max(pytorch_mae, tensorflow_mae):.4f})\n\n")
                
                pytorch_recovery = np.mean([r['structure_recovery'] for r in pytorch_results])
                tensorflow_recovery = np.mean([r['structure_recovery'] for r in tensorflow_results])
                f.write(f"**🔗 Structure Recovery**: {'🥇 TensorFlow' if tensorflow_recovery > pytorch_recovery else '🥇 PyTorch'} ")
                f.write(f"(Recovery: {max(pytorch_recovery, tensorflow_recovery):.4f} vs {min(pytorch_recovery, tensorflow_recovery):.4f})\n\n")
                
                pytorch_entropy = np.mean([r['entropy_relative_error'] for r in pytorch_results if not np.isinf(r['entropy_relative_error'])])
                tensorflow_entropy = np.mean([r['entropy_relative_error'] for r in tensorflow_results if not np.isinf(r['entropy_relative_error'])])
                f.write(f"**📊 Entropy Estimation**: {'🥇 TensorFlow' if tensorflow_entropy < pytorch_entropy else '🥇 PyTorch'} ")
                f.write(f"(Rel Error: {min(pytorch_entropy, tensorflow_entropy):.4f} vs {max(pytorch_entropy, tensorflow_entropy):.4f})\n\n")
                
                pytorch_fit = np.mean([r['fit_time'] for r in pytorch_results])
                tensorflow_fit = np.mean([r['fit_time'] for r in tensorflow_results])
                f.write(f"**⚡ Fit Speed**: {'🥇 PyTorch' if pytorch_fit < tensorflow_fit else '🥇 TensorFlow'} ")
                f.write(f"(Fit time: {min(pytorch_fit, tensorflow_fit):.3f}s vs {max(pytorch_fit, tensorflow_fit):.3f}s)\n\n")
                
                pytorch_sample = np.mean([r['sample_time'] for r in pytorch_results])
                tensorflow_sample = np.mean([r['sample_time'] for r in tensorflow_results])
                f.write(f"**🚀 Sample Speed**: {'🥇 PyTorch' if pytorch_sample < tensorflow_sample else '🥇 TensorFlow'} ")
                f.write(f"(Sample time: {min(pytorch_sample, tensorflow_sample):.3f}s vs {max(pytorch_sample, tensorflow_sample):.3f}s)\n\n")
                
                # Overall winner
                pytorch_score = 0
                tensorflow_score = 0
                
                if pytorch_mae < tensorflow_mae: pytorch_score += 1
                else: tensorflow_score += 1
                
                if pytorch_recovery > tensorflow_recovery: pytorch_score += 1
                else: tensorflow_score += 1
                
                if pytorch_entropy < tensorflow_entropy: pytorch_score += 1
                else: tensorflow_score += 1
                
                if pytorch_fit < tensorflow_fit: pytorch_score += 1
                else: tensorflow_score += 1
                
                if pytorch_sample < tensorflow_sample: pytorch_score += 1
                else: tensorflow_score += 1
                
                if pytorch_score > tensorflow_score:
                    f.write(f"## 🏆 Overall Winner: PyTorch DVC ({pytorch_score}/5 metrics)\n\n")
                elif tensorflow_score > pytorch_score:
                    f.write(f"## 🏆 Overall Winner: TensorFlow DVC ({tensorflow_score}/5 metrics)\n\n")
                else:
                    f.write(f"## 🤝 Tie: Both implementations excel in different areas\n\n")
                
            elif all_results:
                # Single implementation results
                f.write("## Overall Performance\n\n")
                
                avg_mae = np.mean([r['correlation_mae'] for r in all_results])
                avg_recovery = np.mean([r['structure_recovery'] for r in all_results])
                avg_entropy_err = np.mean([r['entropy_relative_error'] for r in all_results if not np.isinf(r['entropy_relative_error'])])
                avg_fit_time = np.mean([r['fit_time'] for r in all_results])
                avg_sample_time = np.mean([r['sample_time'] for r in all_results])
                
                f.write(f"- **Average Correlation MAE**: {avg_mae:.4f}\n")
                f.write(f"- **Average Structure Recovery**: {avg_recovery:.4f}\n")
                f.write(f"- **Average Entropy Relative Error**: {avg_entropy_err:.4f}\n")
                f.write(f"- **Average Fit Time**: {avg_fit_time:.3f}s\n")
                f.write(f"- **Average Sample Time**: {avg_sample_time:.3f}s\n\n")
                
                # Best and worst results
                best_mae = min(all_results, key=lambda x: x['correlation_mae'])
                best_recovery = max(all_results, key=lambda x: x['structure_recovery'])
                
                f.write("## Best Results\n\n")
                f.write(f"**Best Correlation Accuracy**: {best_mae['method']} on {best_mae['scenario']['name']} (MAE: {best_mae['correlation_mae']:.4f})\n\n")
                f.write(f"**Best Structure Recovery**: {best_recovery['method']} on {best_recovery['scenario']['name']} (Recovery: {best_recovery['structure_recovery']:.4f})\n\n")
            
            # Detailed results
            f.write("## Detailed Results by Scenario\n\n")
            for result in all_results:
                scenario = result['scenario']
                f.write(f"### {scenario['name']} - {result['method']}\n\n")
                f.write(f"**Data**: {scenario['n_samples']} samples, {scenario['n_dims']} dimensions, {scenario['correlation_type']} correlation\n\n")
                f.write(f"**Performance**:\n")
                f.write(f"- Fit time: {result['fit_time']:.3f}s\n")
                f.write(f"- Sample time: {result['sample_time']:.3f}s\n\n")
                f.write(f"**Accuracy**:\n")
                f.write(f"- Correlation MAE: {result['correlation_mae']:.4f}\n")
                f.write(f"- Structure Recovery: {result['structure_recovery']:.4f}\n")
                f.write(f"- Entropy Relative Error: {result['entropy_relative_error']:.4f}\n")
                f.write(f"- Max Correlation Difference: {result['correlation_max_diff']:.4f}\n\n")
                f.write("---\n\n")
        
        print(f"\nComprehensive report saved to: {report_path}")
    
    def run_comprehensive_test(self):
        """Run comprehensive accuracy test comparing PyTorch and TensorFlow DVC."""
        print("="*80)
        print("COMPREHENSIVE DVC ACCURACY TEST")
        print("Comparing PyTorch vs TensorFlow: Pairwise Interactions & Entropy Estimation")
        print("="*80)
        
        # Test scenarios
        scenarios = [
            {'name': 'AR1_4D', 'n_samples': 800, 'n_dims': 4, 'correlation_type': 'ar1'},
            {'name': 'Block_4D', 'n_samples': 800, 'n_dims': 4, 'correlation_type': 'block'},
            {'name': 'Toeplitz_3D', 'n_samples': 600, 'n_dims': 3, 'correlation_type': 'toeplitz'},
        ]
        
        all_results = []
        
        for i, scenario in enumerate(scenarios):
            print(f"\n--- Scenario {i+1}: {scenario['name']} ---")
            print(f"Samples: {scenario['n_samples']}, Dims: {scenario['n_dims']}, Type: {scenario['correlation_type']}")
            
            # Generate test data
            data, true_corr, true_entropy = self.generate_test_data(
                n_samples=scenario['n_samples'],
                n_dims=scenario['n_dims'],
                correlation_type=scenario['correlation_type'],
                seed=42 + i
            )
            
            print(f"True entropy: {true_entropy:.4f}")
            print(f"True correlation range: [{np.min(true_corr):.3f}, {np.max(true_corr):.3f}]")
            
            scenario_results = []
            
            # Test TensorFlow DVC with different configurations
            for vine_type in ['c-vine', 'd-vine']:
                print(f"\n  Testing TensorFlow DVC - {vine_type}...")
                tensorflow_result = self.test_tensorflow_dvc(data, vine_type=vine_type, parametric=True)
                
                if tensorflow_result['success']:
                    print(f"    ✓ Success! Fit: {tensorflow_result['fit_time']:.3f}s, Sample: {tensorflow_result['sample_time']:.3f}s")
                    tensorflow_accuracy = self.evaluate_accuracy(
                        data, tensorflow_result['samples'], true_corr, true_entropy, f'TensorFlow {vine_type}'
                    )
                    if tensorflow_accuracy:
                        print(f"    Correlation MAE: {tensorflow_accuracy['correlation_mae']:.4f}")
                        print(f"    Structure Recovery: {tensorflow_accuracy['structure_recovery']:.4f}")
                        print(f"    Entropy Error: {tensorflow_accuracy['entropy_relative_error']:.4f}")
                        
                        tensorflow_accuracy.update({
                            'fit_time': tensorflow_result['fit_time'],
                            'sample_time': tensorflow_result['sample_time'],
                            'scenario': scenario,
                            'vine_type': vine_type
                        })
                        scenario_results.append(tensorflow_accuracy)
                else:
                    print(f"    ✗ Failed: {tensorflow_result['error']}")
            
            # Test PyTorch DVC with different configurations
            for vine_type in ['c-vine', 'd-vine']:
                print(f"\n  Testing PyTorch DVC - {vine_type}...")
                pytorch_result = self.test_pytorch_dvc(data, vine_type=vine_type, parametric=True)
                
                if pytorch_result['success']:
                    print(f"    ✓ Success! Fit: {pytorch_result['fit_time']:.3f}s, Sample: {pytorch_result['sample_time']:.3f}s")
                    pytorch_accuracy = self.evaluate_accuracy(
                        data, pytorch_result['samples'], true_corr, true_entropy, f'PyTorch {vine_type}'
                    )
                    if pytorch_accuracy:
                        print(f"    Correlation MAE: {pytorch_accuracy['correlation_mae']:.4f}")
                        print(f"    Structure Recovery: {pytorch_accuracy['structure_recovery']:.4f}")
                        print(f"    Entropy Error: {pytorch_accuracy['entropy_relative_error']:.4f}")
                        
                        pytorch_accuracy.update({
                            'fit_time': pytorch_result['fit_time'],
                            'sample_time': pytorch_result['sample_time'],
                            'scenario': scenario,
                            'vine_type': vine_type
                        })
                        scenario_results.append(pytorch_accuracy)
                else:
                    print(f"    ✗ Failed: {pytorch_result['error']}")
            
            # Create visualization for this scenario
            if scenario_results:
                # Take the best result for visualization
                best_result = min(scenario_results, key=lambda x: x['correlation_mae'])
                self.create_visualization([best_result], scenario['name'])
                all_results.extend(scenario_results)
        
        # Generate comprehensive report
        self.generate_final_report(all_results)
        
        return all_results


def main():
    """Run comprehensive DVC accuracy test."""
    print("Initializing Comprehensive DVC Accuracy Test...")
    print("This test compares PyTorch vs TensorFlow on pairwise interactions and entropy estimation.")
    
    tester = DVCAccuracyTester()
    results = tester.run_comprehensive_test()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DVC ACCURACY TEST COMPLETE")
    print("="*80)
    
    if results:
        print(f"Total successful evaluations: {len(results)}")
        
        # Separate results by implementation
        pytorch_results = [r for r in results if 'PyTorch' in r['method']]
        tensorflow_results = [r for r in results if 'TensorFlow' in r['method']]
        
        if pytorch_results and tensorflow_results:
            print(f"\n🔥 HEAD-TO-HEAD COMPARISON:")
            print(f"PyTorch runs: {len(pytorch_results)}")
            print(f"TensorFlow runs: {len(tensorflow_results)}")
            
            # Accuracy comparison
            pytorch_mae = np.mean([r['correlation_mae'] for r in pytorch_results])
            tensorflow_mae = np.mean([r['correlation_mae'] for r in tensorflow_results])
            
            pytorch_recovery = np.mean([r['structure_recovery'] for r in pytorch_results])
            tensorflow_recovery = np.mean([r['structure_recovery'] for r in tensorflow_results])
            
            pytorch_entropy = np.mean([r['entropy_relative_error'] for r in pytorch_results if not np.isinf(r['entropy_relative_error'])])
            tensorflow_entropy = np.mean([r['entropy_relative_error'] for r in tensorflow_results if not np.isinf(r['entropy_relative_error'])])
            
            # Performance comparison
            pytorch_fit = np.mean([r['fit_time'] for r in pytorch_results])
            tensorflow_fit = np.mean([r['fit_time'] for r in tensorflow_results])
            
            pytorch_sample = np.mean([r['sample_time'] for r in pytorch_results])
            tensorflow_sample = np.mean([r['sample_time'] for r in tensorflow_results])
            
            print(f"\n📊 ACCURACY METRICS:")
            print(f"- Correlation MAE:     PyTorch {pytorch_mae:.4f} vs TensorFlow {tensorflow_mae:.4f} {'🥇' if pytorch_mae < tensorflow_mae else '🥈'} vs {'🥇' if tensorflow_mae < pytorch_mae else '🥈'}")
            print(f"- Structure Recovery:  PyTorch {pytorch_recovery:.4f} vs TensorFlow {tensorflow_recovery:.4f} {'🥇' if pytorch_recovery > tensorflow_recovery else '🥈'} vs {'🥇' if tensorflow_recovery > pytorch_recovery else '🥈'}")
            print(f"- Entropy Rel Error:   PyTorch {pytorch_entropy:.4f} vs TensorFlow {tensorflow_entropy:.4f} {'🥇' if pytorch_entropy < tensorflow_entropy else '🥈'} vs {'🥇' if tensorflow_entropy < pytorch_entropy else '🥈'}")
            
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"- Fit Time:    PyTorch {pytorch_fit:.3f}s vs TensorFlow {tensorflow_fit:.3f}s {'🥇' if pytorch_fit < tensorflow_fit else '🥈'} vs {'🥇' if tensorflow_fit < pytorch_fit else '🥈'}")
            print(f"- Sample Time: PyTorch {pytorch_sample:.3f}s vs TensorFlow {tensorflow_sample:.3f}s {'🥇' if pytorch_sample < tensorflow_sample else '🥈'} vs {'🥇' if tensorflow_sample < pytorch_sample else '🥈'}")
            
            # Determine overall winner
            pytorch_wins = 0
            tensorflow_wins = 0
            
            if pytorch_mae < tensorflow_mae: pytorch_wins += 1
            else: tensorflow_wins += 1
            
            if pytorch_recovery > tensorflow_recovery: pytorch_wins += 1
            else: tensorflow_wins += 1
            
            if pytorch_entropy < tensorflow_entropy: pytorch_wins += 1
            else: tensorflow_wins += 1
            
            if pytorch_fit < tensorflow_fit: pytorch_wins += 1
            else: tensorflow_wins += 1
            
            if pytorch_sample < tensorflow_sample: pytorch_wins += 1
            else: tensorflow_wins += 1
            
            print(f"\n🏆 OVERALL WINNER:")
            if pytorch_wins > tensorflow_wins:
                print(f"🥇 PyTorch DVC wins {pytorch_wins}/5 metrics!")
                print("   Best for: Speed and efficiency")
            elif tensorflow_wins > pytorch_wins:
                print(f"🥇 TensorFlow DVC wins {tensorflow_wins}/5 metrics!")
                print("   Best for: Accuracy and precision")
            else:
                print("🤝 It's a tie! Both implementations excel in different areas.")
                
        elif pytorch_results:
            avg_mae = np.mean([r['correlation_mae'] for r in pytorch_results])
            avg_recovery = np.mean([r['structure_recovery'] for r in pytorch_results])
            print(f"\nPyTorch Results:")
            print(f"- Average correlation MAE: {avg_mae:.4f}")
            print(f"- Average structure recovery: {avg_recovery:.4f}")
            
        elif tensorflow_results:
            avg_mae = np.mean([r['correlation_mae'] for r in tensorflow_results])
            avg_recovery = np.mean([r['structure_recovery'] for r in tensorflow_results])
            print(f"\nTensorFlow Results:")
            print(f"- Average correlation MAE: {avg_mae:.4f}")
            print(f"- Average structure recovery: {avg_recovery:.4f}")
    
    return results


if __name__ == "__main__":
    results = main() 