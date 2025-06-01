#!/usr/bin/env python3
"""
Comprehensive DVC Experiment: PyTorch vs TensorFlow
Multivariate Gaussian Data with Different Marginals

This experiment:
1. Generates multivariate Gaussian data with known correlation structures
2. Applies different marginal transformations (normal, exponential, uniform, etc.)
3. Fits both PyTorch and TensorFlow DVC models
4. Samples from fitted models
5. Compares recovered correlation structures with ground truth
6. Estimates and compares entropies
7. Creates comprehensive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tensorflow as tf
from scipy import stats
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# Import TensorFlow DVC components
import sys
import os
sys.path.append('src/DVC_tensorflow')
from classes.objects import vine_obj_bin as vine_obj_bin_tf
from sampling.vine_sample import vine_copula_sample as vine_sample_tf
from info.info_estimation import vine_entropy as entropy_est_tf

# Import PyTorch DVC components  
sys.path.append('src/DVC_pytorch')
from classes.objects import vine_obj_bin as vine_obj_bin_pt
from sampling.vine_sampler import VineSampler
from info.info_estimation import vine_entropy as entropy_est_pt


class GaussianDataGenerator:
    """Generate multivariate Gaussian data with different marginal transformations"""
    
    def __init__(self, d=4, n_samples=1000):
        self.d = d
        self.n_samples = n_samples
        
    def generate_correlation_matrix(self, structure_type='ar1', param=0.7):
        """Generate different correlation structures"""
        if structure_type == 'ar1':
            # AR(1) correlation structure
            corr = np.zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    corr[i, j] = param ** abs(i - j)
        elif structure_type == 'block':
            # Block correlation structure
            corr = np.eye(self.d)
            block_size = self.d // 2
            corr[:block_size, :block_size] = param
            corr[block_size:, block_size:] = param
            np.fill_diagonal(corr, 1.0)
        elif structure_type == 'toeplitz':
            # Toeplitz correlation structure
            corr = np.zeros((self.d, self.d))
            for i in range(self.d):
                for j in range(self.d):
                    corr[i, j] = param ** abs(i - j)
        elif structure_type == 'compound':
            # Compound symmetry
            corr = np.full((self.d, self.d), param)
            np.fill_diagonal(corr, 1.0)
        else:
            # Identity (independence)
            corr = np.eye(self.d)
            
        return corr
    
    def apply_marginal_transforms(self, gaussian_data, marginal_types):
        """Apply different marginal transformations to Gaussian data"""
        transformed_data = np.zeros_like(gaussian_data)
        
        for i, marginal in enumerate(marginal_types):
            # Convert to uniform using Gaussian CDF
            uniform_data = stats.norm.cdf(gaussian_data[:, i])
            
            if marginal == 'normal':
                # Keep as normal with different parameters
                transformed_data[:, i] = stats.norm.ppf(uniform_data, loc=i, scale=0.5 + i*0.2)
            elif marginal == 'exponential':
                # Transform to exponential
                transformed_data[:, i] = stats.expon.ppf(uniform_data, scale=1 + i*0.5)
            elif marginal == 'uniform':
                # Transform to uniform
                transformed_data[:, i] = stats.uniform.ppf(uniform_data, loc=i, scale=2)
            elif marginal == 'gamma':
                # Transform to gamma
                transformed_data[:, i] = stats.gamma.ppf(uniform_data, a=2 + i, scale=1)
            elif marginal == 'beta':
                # Transform to beta
                transformed_data[:, i] = stats.beta.ppf(uniform_data, a=2 + i, b=3)
            else:
                # Default to standard normal
                transformed_data[:, i] = gaussian_data[:, i]
                
        return transformed_data
    
    def generate_data(self, correlation_structure='ar1', marginal_types=None):
        """Generate complete dataset with specified correlation and marginals"""
        if marginal_types is None:
            marginal_types = ['normal'] * self.d
            
        # Generate correlation matrix
        true_corr = self.generate_correlation_matrix(correlation_structure)
        
        # Generate multivariate Gaussian data
        gaussian_data = multivariate_normal.rvs(
            mean=np.zeros(self.d), 
            cov=true_corr, 
            size=self.n_samples
        )
        
        # Apply marginal transformations
        transformed_data = self.apply_marginal_transforms(gaussian_data, marginal_types)
        
        # Calculate true entropy (for Gaussian copula)
        true_entropy = 0.5 * np.log((2 * np.pi * np.e) ** self.d * np.linalg.det(true_corr))
        
        return transformed_data, true_corr, true_entropy


def fit_tensorflow_vine(data, vine_type='cvine', method='parametric'):
    """Fit TensorFlow DVC model"""
    try:
        # Create vine object
        vine = vine_obj_bin_tf(
            data=data,
            vine_type=vine_type,
            param=method == 'parametric',
            vine_depth=data.shape[1] - 1
        )
        
        # Fit the model
        vine.fit()
        
        return vine, True
    except Exception as e:
        print(f"TensorFlow fitting failed: {e}")
        return None, False


def fit_pytorch_vine(data, vine_type='cvine', method='parametric'):
    """Fit PyTorch DVC model"""
    try:
        # Convert to torch tensor
        device = torch.device('cpu')
        dtype = torch.float32
        
        # Prepare data following working pattern
        data_tensor = torch.tensor(data, dtype=dtype, device=device)
        
        # Create vine object
        vine = vine_obj_bin_pt(
            data=data_tensor,
            vine_type=vine_type,
            param=method == 'parametric',
            vine_depth=data.shape[1] - 1
        )
        
        # Fit the model
        vine.fit()
        
        return vine, True
    except Exception as e:
        print(f"PyTorch fitting failed: {e}")
        return None, False


def sample_from_vine(vine, n_samples, framework='tensorflow'):
    """Sample from fitted vine model"""
    try:
        if framework == 'tensorflow':
            samples, _ = vine_sample_tf(vine, n_samples)
            return samples, True
        else:  # pytorch
            sampler = VineSampler(vine)
            samples, _ = sampler.sample(n_samples)
            if torch.is_tensor(samples):
                samples = samples.detach().cpu().numpy()
            return samples, True
    except Exception as e:
        print(f"{framework} sampling failed: {e}")
        return None, False


def estimate_entropy(data, vine=None, framework='tensorflow'):
    """Estimate entropy using DVC model"""
    try:
        # Prepare info_dict for entropy estimation
        info_dict = {
            'alpha': 0.05,  # 95% confidence level
            'cases': 100,   # Samples per iteration
            'iterations': 10  # Maximum iterations
        }
        
        if framework == 'tensorflow':
            entropy = entropy_est_tf(vine, info_dict)
        else:  # pytorch
            entropy = entropy_est_pt(vine, info_dict)
        return entropy, True
    except Exception as e:
        print(f"{framework} entropy estimation failed: {e}")
        return None, False


def calculate_correlation_error(true_corr, estimated_corr):
    """Calculate correlation matrix error metrics"""
    # Frobenius norm error
    frobenius_error = np.linalg.norm(true_corr - estimated_corr, 'fro')
    
    # Element-wise absolute error
    abs_error = np.mean(np.abs(true_corr - estimated_corr))
    
    # Correlation coefficient between matrices
    corr_coeff = np.corrcoef(true_corr.flatten(), estimated_corr.flatten())[0, 1]
    
    return {
        'frobenius_error': frobenius_error,
        'abs_error': abs_error,
        'correlation_coeff': corr_coeff
    }


def run_single_experiment(d=4, n_samples=1000, correlation_structure='ar1', 
                         marginal_types=None, vine_type='cvine', method='parametric'):
    """Run a single experiment comparing PyTorch and TensorFlow"""
    
    print(f"\nRunning experiment: d={d}, structure={correlation_structure}, "
          f"vine={vine_type}, method={method}")
    
    # Generate data
    generator = GaussianDataGenerator(d=d, n_samples=n_samples)
    data, true_corr, true_entropy = generator.generate_data(
        correlation_structure=correlation_structure,
        marginal_types=marginal_types
    )
    
    results = {
        'dimension': d,
        'n_samples': n_samples,
        'correlation_structure': correlation_structure,
        'vine_type': vine_type,
        'method': method,
        'true_entropy': true_entropy,
        'true_correlation': true_corr
    }
    
    # Test TensorFlow
    print("  Fitting TensorFlow model...")
    tf_vine, tf_fit_success = fit_tensorflow_vine(data, vine_type, method)
    
    if tf_fit_success:
        print("  Sampling from TensorFlow model...")
        tf_samples, tf_sample_success = sample_from_vine(tf_vine, n_samples, 'tensorflow')
        
        if tf_sample_success:
            # Calculate correlation from samples
            tf_estimated_corr = np.corrcoef(tf_samples.T)
            tf_corr_error = calculate_correlation_error(true_corr, tf_estimated_corr)
            
            # Estimate entropy
            tf_entropy, tf_entropy_success = estimate_entropy(data, tf_vine, 'tensorflow')
            
            results.update({
                'tf_fit_success': True,
                'tf_sample_success': True,
                'tf_estimated_correlation': tf_estimated_corr,
                'tf_correlation_error': tf_corr_error,
                'tf_entropy': tf_entropy if tf_entropy_success else None,
                'tf_entropy_error': abs(true_entropy - tf_entropy) if tf_entropy_success else None
            })
        else:
            results.update({
                'tf_fit_success': True,
                'tf_sample_success': False,
                'tf_estimated_correlation': None,
                'tf_correlation_error': None,
                'tf_entropy': None,
                'tf_entropy_error': None
            })
    else:
        results.update({
            'tf_fit_success': False,
            'tf_sample_success': False,
            'tf_estimated_correlation': None,
            'tf_correlation_error': None,
            'tf_entropy': None,
            'tf_entropy_error': None
        })
    
    # Test PyTorch
    print("  Fitting PyTorch model...")
    pt_vine, pt_fit_success = fit_pytorch_vine(data, vine_type, method)
    
    if pt_fit_success:
        print("  Sampling from PyTorch model...")
        pt_samples, pt_sample_success = sample_from_vine(pt_vine, n_samples, 'pytorch')
        
        if pt_sample_success:
            # Calculate correlation from samples
            pt_estimated_corr = np.corrcoef(pt_samples.T)
            pt_corr_error = calculate_correlation_error(true_corr, pt_estimated_corr)
            
            # Estimate entropy
            pt_entropy, pt_entropy_success = estimate_entropy(data, pt_vine, 'pytorch')
            
            results.update({
                'pt_fit_success': True,
                'pt_sample_success': True,
                'pt_estimated_correlation': pt_estimated_corr,
                'pt_correlation_error': pt_corr_error,
                'pt_entropy': pt_entropy if pt_entropy_success else None,
                'pt_entropy_error': abs(true_entropy - pt_entropy) if pt_entropy_success else None
            })
        else:
            results.update({
                'pt_fit_success': True,
                'pt_sample_success': False,
                'pt_estimated_correlation': None,
                'pt_correlation_error': None,
                'pt_entropy': None,
                'pt_entropy_error': None
            })
    else:
        results.update({
            'pt_fit_success': False,
            'pt_sample_success': False,
            'pt_estimated_correlation': None,
            'pt_correlation_error': None,
            'pt_entropy': None,
            'pt_entropy_error': None
        })
    
    return results


def create_visualizations(all_results):
    """Create comprehensive visualizations of results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Success Rate Comparison
    ax1 = plt.subplot(3, 4, 1)
    success_data = []
    for result in all_results:
        success_data.append({
            'Framework': 'TensorFlow',
            'Success': result['tf_fit_success'] and result['tf_sample_success'],
            'Method': result['method'],
            'Structure': result['correlation_structure']
        })
        success_data.append({
            'Framework': 'PyTorch', 
            'Success': result['pt_fit_success'] and result['pt_sample_success'],
            'Method': result['method'],
            'Structure': result['correlation_structure']
        })
    
    success_df = pd.DataFrame(success_data)
    success_rates = success_df.groupby(['Framework', 'Method'])['Success'].mean().reset_index()
    
    sns.barplot(data=success_rates, x='Method', y='Success', hue='Framework', ax=ax1)
    ax1.set_title('Success Rate by Framework and Method')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1.1)
    
    # 2. Correlation Error Comparison
    ax2 = plt.subplot(3, 4, 2)
    error_data = []
    for result in all_results:
        if result['tf_correlation_error'] is not None:
            error_data.append({
                'Framework': 'TensorFlow',
                'Frobenius_Error': result['tf_correlation_error']['frobenius_error'],
                'Method': result['method'],
                'Structure': result['correlation_structure']
            })
        if result['pt_correlation_error'] is not None:
            error_data.append({
                'Framework': 'PyTorch',
                'Frobenius_Error': result['pt_correlation_error']['frobenius_error'], 
                'Method': result['method'],
                'Structure': result['correlation_structure']
            })
    
    if error_data:
        error_df = pd.DataFrame(error_data)
        sns.boxplot(data=error_df, x='Method', y='Frobenius_Error', hue='Framework', ax=ax2)
        ax2.set_title('Correlation Matrix Frobenius Error')
        ax2.set_ylabel('Frobenius Error')
    
    # 3. Entropy Error Comparison
    ax3 = plt.subplot(3, 4, 3)
    entropy_data = []
    for result in all_results:
        if result['tf_entropy_error'] is not None:
            entropy_data.append({
                'Framework': 'TensorFlow',
                'Entropy_Error': result['tf_entropy_error'],
                'Method': result['method']
            })
        if result['pt_entropy_error'] is not None:
            entropy_data.append({
                'Framework': 'PyTorch',
                'Entropy_Error': result['pt_entropy_error'],
                'Method': result['method']
            })
    
    if entropy_data:
        entropy_df = pd.DataFrame(entropy_data)
        sns.boxplot(data=entropy_df, x='Method', y='Entropy_Error', hue='Framework', ax=ax3)
        ax3.set_title('Entropy Estimation Error')
        ax3.set_ylabel('Absolute Entropy Error')
    
    # 4. Correlation Structure Recovery by Structure Type
    ax4 = plt.subplot(3, 4, 4)
    if error_data:
        sns.boxplot(data=error_df, x='Structure', y='Frobenius_Error', hue='Framework', ax=ax4)
        ax4.set_title('Error by Correlation Structure')
        ax4.set_ylabel('Frobenius Error')
        ax4.tick_params(axis='x', rotation=45)
    
    # 5-8. Example Correlation Matrices
    example_results = [r for r in all_results if r['tf_estimated_correlation'] is not None 
                      and r['pt_estimated_correlation'] is not None]
    
    if example_results:
        example = example_results[0]  # Take first successful example
        
        # True correlation
        ax5 = plt.subplot(3, 4, 5)
        sns.heatmap(example['true_correlation'], annot=True, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax5, cbar_kws={'shrink': 0.8})
        ax5.set_title('True Correlation Matrix')
        
        # TensorFlow estimated
        ax6 = plt.subplot(3, 4, 6)
        sns.heatmap(example['tf_estimated_correlation'], annot=True, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax6, cbar_kws={'shrink': 0.8})
        ax6.set_title('TensorFlow Estimated')
        
        # PyTorch estimated
        ax7 = plt.subplot(3, 4, 7)
        sns.heatmap(example['pt_estimated_correlation'], annot=True, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax7, cbar_kws={'shrink': 0.8})
        ax7.set_title('PyTorch Estimated')
        
        # Error matrix (TF - True)
        ax8 = plt.subplot(3, 4, 8)
        tf_error_matrix = example['tf_estimated_correlation'] - example['true_correlation']
        sns.heatmap(tf_error_matrix, annot=True, cmap='RdBu_r', center=0,
                   ax=ax8, cbar_kws={'shrink': 0.8})
        ax8.set_title('TensorFlow Error Matrix')
    
    # 9. Framework Performance Summary
    ax9 = plt.subplot(3, 4, 9)
    if error_data:
        framework_summary = error_df.groupby('Framework')['Frobenius_Error'].agg(['mean', 'std']).reset_index()
        x_pos = np.arange(len(framework_summary))
        ax9.bar(x_pos, framework_summary['mean'], yerr=framework_summary['std'], 
               capsize=5, alpha=0.7)
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels(framework_summary['Framework'])
        ax9.set_title('Average Correlation Error by Framework')
        ax9.set_ylabel('Mean Frobenius Error')
    
    # 10. Method Performance Summary
    ax10 = plt.subplot(3, 4, 10)
    if error_data:
        method_summary = error_df.groupby('Method')['Frobenius_Error'].agg(['mean', 'std']).reset_index()
        x_pos = np.arange(len(method_summary))
        ax10.bar(x_pos, method_summary['mean'], yerr=method_summary['std'],
                capsize=5, alpha=0.7)
        ax10.set_xticks(x_pos)
        ax10.set_xticklabels(method_summary['Method'])
        ax10.set_title('Average Error by Method')
        ax10.set_ylabel('Mean Frobenius Error')
    
    # 11. Correlation Recovery Quality
    ax11 = plt.subplot(3, 4, 11)
    recovery_data = []
    for result in all_results:
        if result['tf_correlation_error'] is not None:
            recovery_data.append({
                'Framework': 'TensorFlow',
                'Recovery_Quality': result['tf_correlation_error']['correlation_coeff'],
                'Method': result['method']
            })
        if result['pt_correlation_error'] is not None:
            recovery_data.append({
                'Framework': 'PyTorch',
                'Recovery_Quality': result['pt_correlation_error']['correlation_coeff'],
                'Method': result['method']
            })
    
    if recovery_data:
        recovery_df = pd.DataFrame(recovery_data)
        sns.boxplot(data=recovery_df, x='Method', y='Recovery_Quality', hue='Framework', ax=ax11)
        ax11.set_title('Correlation Recovery Quality')
        ax11.set_ylabel('Correlation Coefficient')
        ax11.set_ylim(0, 1.1)
    
    # 12. Summary Statistics Table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Create summary statistics
    summary_text = "EXPERIMENT SUMMARY\n\n"
    total_experiments = len(all_results)
    tf_successes = sum(1 for r in all_results if r['tf_fit_success'] and r['tf_sample_success'])
    pt_successes = sum(1 for r in all_results if r['pt_fit_success'] and r['pt_sample_success'])
    
    summary_text += f"Total Experiments: {total_experiments}\n"
    summary_text += f"TensorFlow Success: {tf_successes}/{total_experiments} ({tf_successes/total_experiments*100:.1f}%)\n"
    summary_text += f"PyTorch Success: {pt_successes}/{total_experiments} ({pt_successes/total_experiments*100:.1f}%)\n\n"
    
    if error_data:
        tf_errors = [d['Frobenius_Error'] for d in error_data if d['Framework'] == 'TensorFlow']
        pt_errors = [d['Frobenius_Error'] for d in error_data if d['Framework'] == 'PyTorch']
        
        if tf_errors:
            summary_text += f"TensorFlow Avg Error: {np.mean(tf_errors):.4f} ± {np.std(tf_errors):.4f}\n"
        if pt_errors:
            summary_text += f"PyTorch Avg Error: {np.mean(pt_errors):.4f} ± {np.std(pt_errors):.4f}\n"
    
    ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('comprehensive_gaussian_dvc_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run comprehensive DVC experiment"""
    print("Starting Comprehensive DVC Gaussian Experiment")
    print("=" * 60)
    
    # Experiment configurations
    experiments = [
        # Different correlation structures
        {'d': 3, 'correlation_structure': 'ar1', 'marginal_types': ['normal', 'exponential', 'uniform']},
        {'d': 3, 'correlation_structure': 'block', 'marginal_types': ['normal', 'gamma', 'beta']},
        {'d': 4, 'correlation_structure': 'toeplitz', 'marginal_types': ['normal', 'exponential', 'uniform', 'gamma']},
        {'d': 4, 'correlation_structure': 'compound', 'marginal_types': ['normal', 'normal', 'exponential', 'uniform']},
        
        # Different dimensions with same structure
        {'d': 3, 'correlation_structure': 'ar1', 'marginal_types': ['normal', 'normal', 'normal']},
        {'d': 4, 'correlation_structure': 'ar1', 'marginal_types': ['normal', 'normal', 'normal', 'normal']},
        {'d': 5, 'correlation_structure': 'ar1', 'marginal_types': ['normal', 'exponential', 'uniform', 'gamma', 'beta']},
    ]
    
    vine_types = ['cvine', 'dvine']
    methods = ['parametric', 'nonparametric']
    
    all_results = []
    
    # Run all experiments
    for exp_config in experiments:
        for vine_type in vine_types:
            for method in methods:
                try:
                    result = run_single_experiment(
                        d=exp_config['d'],
                        n_samples=1000,
                        correlation_structure=exp_config['correlation_structure'],
                        marginal_types=exp_config['marginal_types'],
                        vine_type=vine_type,
                        method=method
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"Experiment failed: {e}")
                    continue
    
    # Save results
    print(f"\nCompleted {len(all_results)} experiments")
    
    # Create summary DataFrame for successful experiments
    summary_data = []
    for result in all_results:
        summary_row = {
            'dimension': result['dimension'],
            'correlation_structure': result['correlation_structure'],
            'vine_type': result['vine_type'],
            'method': result['method'],
            'tf_success': result['tf_fit_success'] and result['tf_sample_success'],
            'pt_success': result['pt_fit_success'] and result['pt_sample_success'],
            'tf_frobenius_error': result['tf_correlation_error']['frobenius_error'] if result['tf_correlation_error'] else None,
            'pt_frobenius_error': result['pt_correlation_error']['frobenius_error'] if result['pt_correlation_error'] else None,
            'tf_entropy_error': result['tf_entropy_error'],
            'pt_entropy_error': result['pt_entropy_error'],
            'true_entropy': result['true_entropy']
        }
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('gaussian_dvc_experiment_results.csv', index=False)
    print("Results saved to 'gaussian_dvc_experiment_results.csv'")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 60)
    
    total_exp = len(all_results)
    tf_success_rate = summary_df['tf_success'].mean()
    pt_success_rate = summary_df['pt_success'].mean()
    
    print(f"Total experiments: {total_exp}")
    print(f"TensorFlow success rate: {tf_success_rate:.2%}")
    print(f"PyTorch success rate: {pt_success_rate:.2%}")
    
    # Error statistics for successful experiments
    tf_errors = summary_df[summary_df['tf_success']]['tf_frobenius_error'].dropna()
    pt_errors = summary_df[summary_df['pt_success']]['pt_frobenius_error'].dropna()
    
    if len(tf_errors) > 0:
        print(f"TensorFlow avg correlation error: {tf_errors.mean():.4f} ± {tf_errors.std():.4f}")
    if len(pt_errors) > 0:
        print(f"PyTorch avg correlation error: {pt_errors.mean():.4f} ± {pt_errors.std():.4f}")
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main() 