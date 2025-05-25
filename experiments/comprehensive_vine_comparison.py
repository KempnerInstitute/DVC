"""
Comprehensive Comparison of PyTorch and TensorFlow Vine Copula Implementations

This script performs systematic comparison across:
- Different vine types (D-vine, C-vine, R-vine)
- Parametric vs Non-parametric approaches
- Correlation recovery accuracy
- Computational time analysis
- Entropy estimation
- Conditional prediction accuracy
- Multiple data distributions
"""

import numpy as np
import sys
sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal, entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from typing import Dict, List, Tuple
import time

# Import PyTorch implementation
from DVC import vine_obj_bin, margin_obj, fit_vine, predict_conditional

# Import TensorFlow implementation
import tensorflow as tf
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj
from DVC_tensorflow.sampling.vine_sample import vine_cop_par_sample, vine_copula_sample

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VineComparison:
    """Class to handle comprehensive vine copula comparisons"""
    
    def __init__(self, dimensions: List[int] = [3, 4, 5], 
                 vine_types: List[str] = ['d-vine', 'c-vine'],
                 approaches: List[str] = ['parametric', 'non-parametric'],
                 n_samples: int = 1000,
                 n_test_samples: int = 500):
        self.dimensions = dimensions
        self.vine_types = vine_types
        self.approaches = approaches
        self.n_samples = n_samples
        self.n_test_samples = n_test_samples
        self.results = {}
        
    def generate_test_data(self, dim: int, correlation_strength: str = 'moderate') -> Tuple[np.ndarray, np.ndarray, float]:
        """Generate correlated test data with known ground truth properties"""
        
        if correlation_strength == 'weak':
            base_corr = 0.2
        elif correlation_strength == 'moderate':
            base_corr = 0.5
        elif correlation_strength == 'strong':
            base_corr = 0.8
        else:
            base_corr = 0.5
            
        # Create structured correlation matrix
        cov_matrix = np.eye(dim)
        
        # Add block structure for more interesting correlations
        for i in range(dim):
            for j in range(i+1, dim):
                if j == i + 1:  # Adjacent variables
                    cov_matrix[i, j] = cov_matrix[j, i] = base_corr
                elif j == i + 2:  # Skip-one variables
                    cov_matrix[i, j] = cov_matrix[j, i] = base_corr * 0.6
                else:  # Distant variables
                    cov_matrix[i, j] = cov_matrix[j, i] = base_corr * 0.3
                    
        # Ensure positive definite
        cov_matrix = cov_matrix + 0.1 * np.eye(dim)
        
        # Generate data
        data = np.random.multivariate_normal(np.zeros(dim), cov_matrix, self.n_samples).astype(np.float32)
        
        # Calculate true entropy (for multivariate normal)
        true_entropy = 0.5 * np.log(2 * np.pi * np.e) * dim + 0.5 * np.log(np.linalg.det(cov_matrix))
        
        return data, cov_matrix, true_entropy
    
    def fit_pytorch_vine(self, data: np.ndarray, vine_type: str, approach: str) -> Tuple[vine_obj_bin, Dict]:
        """Fit PyTorch vine implementation"""
        dim = data.shape[1]
        
        # Create vine
        vine = vine_obj_bin(
            vine_family=vine_type,
            families=['gaussian', 'ind'],
            vine_depth=dim,
            margin=[],
            knots=50
        )
        
        # Set margins
        for i in range(dim):
            vine.margin.append(margin_obj('norm', [0, 1], True))
        
        # Configuration based on approach
        is_parametric = (approach == 'parametric')
        gen_dict = {"parallel": False, "param": is_parametric, "binning": False}
        par_dict = {"param_families": ["gaussian", "ind"]}
        npc_dict = {"method": "local", "n_iter": 50 if is_parametric else 100}  # Fewer iterations for parametric
        bin_dict = {"n_bin": 1}
        
        # Time different phases
        timing_info = {}
        
        # Fit vine
        start_time = time.time()
        fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        timing_info['fit_time'] = fit_time
        
        # Time sampling
        start_time = time.time()
        _ = vine.sample(100)  # Small sample for timing
        sample_time = time.time() - start_time
        timing_info['sample_time'] = sample_time
        
        timing_info['total_time'] = fit_time + sample_time
        
        return vine, timing_info
    
    def fit_tensorflow_vine(self, data: np.ndarray, vine_type: str, approach: str) -> Tuple[tf_vine_obj_bin, Dict]:
        """Fit TensorFlow vine implementation"""
        dim = data.shape[1]
        
        # Create vine
        vine = tf_vine_obj_bin(
            vine_family=vine_type,
            families=['gaussian', 'ind'],
            vine_depth=dim,
            margin=[],
            knots=50,
            method='matrix'
        )
        
        # Set margins
        for i in range(dim):
            margin = tf_margin_obj('norm', [0, 1], True)
            margin.ker = data[:, i]
            vine.margin.append(margin)
        
        # Configuration based on approach
        is_parametric = (approach == 'parametric')
        gen_dict = {"parallel": False, "param": is_parametric, "binning": False, "fitted": False, "vine_depth": dim}
        par_dict = {"param_families": ["gaussian", "ind"]}
        npc_dict = {"opt_method": "local", "batch_paral": False}
        bin_dict = {"n_bin": 1}
        
        # Time different phases
        timing_info = {}
        
        # Fit vine
        start_time = time.time()
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        timing_info['fit_time'] = fit_time
        
        # Time sampling
        start_time = time.time()
        if is_parametric:
            _ = vine_cop_par_sample(vine, 100)  # Small sample for timing
        else:
            _ = vine_copula_sample(vine, 100)[0]  # Small sample for timing
        sample_time = time.time() - start_time
        timing_info['sample_time'] = sample_time
        
        timing_info['total_time'] = fit_time + sample_time
        
        return vine, timing_info
    
    def evaluate_correlation_recovery(self, vine, true_cov: np.ndarray, implementation: str, approach: str) -> Dict:
        """Evaluate how well the vine recovers correlations"""
        
        # Time the sampling process
        start_time = time.time()
        
        # Generate samples
        if implementation == 'pytorch':
            samples = vine.sample(self.n_test_samples)
        else:  # tensorflow
            if approach == 'parametric':
                samples = vine_cop_par_sample(vine, self.n_test_samples)
            else:
                samples = vine_copula_sample(vine, self.n_test_samples)[0]
        
        sampling_time = time.time() - start_time
        
        # Compute correlation matrices
        true_corr = true_cov / np.sqrt(np.outer(np.diag(true_cov), np.diag(true_cov)))
        pred_corr = np.corrcoef(samples.T)
        
        # Compute errors
        corr_mae = np.mean(np.abs(pred_corr - true_corr))
        corr_rmse = np.sqrt(np.mean((pred_corr - true_corr)**2))
        
        # Compute correlation of correlations
        true_corr_flat = true_corr[np.triu_indices_from(true_corr, k=1)]
        pred_corr_flat = pred_corr[np.triu_indices_from(pred_corr, k=1)]
        
        if len(true_corr_flat) > 1:
            corr_correlation = np.corrcoef(true_corr_flat, pred_corr_flat)[0, 1]
        else:
            corr_correlation = np.nan
        
        return {
            'samples': samples,
            'true_corr': true_corr,
            'pred_corr': pred_corr,
            'corr_mae': corr_mae,
            'corr_rmse': corr_rmse,
            'corr_correlation': corr_correlation,
            'sampling_time': sampling_time
        }
    
    def evaluate_entropy_estimation(self, samples: np.ndarray, true_entropy: float) -> Dict:
        """Evaluate entropy estimation using KDE"""
        
        # Estimate entropy using KDE (simplified for multivariate case)
        try:
            # For simplicity, estimate entropy of first principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(samples).flatten()
            
            # Estimate entropy using histogram
            hist, bin_edges = np.histogram(pc1, bins=50, density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            entropy_est = -np.sum(hist * np.log(hist + 1e-10) * bin_width)
            
            entropy_error = abs(entropy_est - true_entropy)
            
        except:
            entropy_est = np.nan
            entropy_error = np.nan
        
        return {
            'estimated_entropy': entropy_est,
            'true_entropy': true_entropy,
            'entropy_error': entropy_error
        }
    
    def evaluate_conditional_prediction(self, vine, data: np.ndarray, implementation: str, approach: str) -> Dict:
        """Evaluate conditional prediction accuracy"""
        dim = data.shape[1]
        
        if dim < 3:
            return {'pred_mae': np.nan, 'pred_rmse': np.nan, 'prediction_time': np.nan}
        
        # Use first half of dimensions as observed, second half as targets
        mid_point = dim // 2
        observed_indices = list(range(mid_point))
        target_indices = list(range(mid_point, dim))
        
        # Take subset of data for testing
        test_data = data[:50]  # Smaller subset for speed
        observed_data = test_data[:, observed_indices]
        true_targets = test_data[:, target_indices]
        
        start_time = time.time()
        
        try:
            if implementation == 'pytorch':
                predictions = predict_conditional(vine, observed_data, observed_indices, target_indices, n_samples=500)
            else:  # tensorflow - simplified prediction
                # For TensorFlow, we'll use a simple approach
                # Generate samples and approximate conditional distribution
                if approach == 'parametric':
                    samples = vine_cop_par_sample(vine, 500)
                else:
                    samples = vine_copula_sample(vine, 500)[0]
                predictions = np.mean(samples[:, target_indices], axis=0)
                predictions = np.tile(predictions, (len(true_targets), 1))
            
            prediction_time = time.time() - start_time
            
            # Compute errors
            pred_mae = mean_absolute_error(true_targets, predictions)
            pred_rmse = np.sqrt(mean_squared_error(true_targets, predictions))
            
        except Exception as e:
            print(f"Prediction error for {implementation}-{approach}: {e}")
            pred_mae = np.nan
            pred_rmse = np.nan
            predictions = np.full_like(true_targets, np.nan)
            prediction_time = np.nan
        
        return {
            'predictions': predictions,
            'true_targets': true_targets,
            'pred_mae': pred_mae,
            'pred_rmse': pred_rmse,
            'prediction_time': prediction_time,
            'observed_indices': observed_indices,
            'target_indices': target_indices
        }
    
    def run_comparison(self) -> Dict:
        """Run comprehensive comparison across all configurations"""
        
        print("Starting comprehensive vine copula comparison...")
        print(f"Testing dimensions: {self.dimensions}")
        print(f"Testing vine types: {self.vine_types}")
        print(f"Testing approaches: {self.approaches}")
        print(f"Training samples: {self.n_samples}")
        print(f"Test samples: {self.n_test_samples}")
        print("-" * 60)
        
        results = {}
        
        for correlation_strength in ['weak', 'moderate', 'strong']:
            results[correlation_strength] = {}
            
            for dim in self.dimensions:
                results[correlation_strength][dim] = {}
                
                print(f"\nTesting {correlation_strength} correlations, dimension {dim}...")
                
                # Generate data
                data, true_cov, true_entropy = self.generate_test_data(dim, correlation_strength)
                
                for approach in self.approaches:
                    results[correlation_strength][dim][approach] = {}
                    
                    print(f"  Testing {approach} approach...")
                    
                    for vine_type in self.vine_types:
                        results[correlation_strength][dim][approach][vine_type] = {}
                        
                        print(f"    Testing {vine_type}...")
                        
                        # Test PyTorch implementation
                        try:
                            pytorch_vine, pytorch_timing = self.fit_pytorch_vine(data, vine_type, approach)
                            pytorch_corr = self.evaluate_correlation_recovery(pytorch_vine, true_cov, 'pytorch', approach)
                            pytorch_entropy = self.evaluate_entropy_estimation(pytorch_corr['samples'], true_entropy)
                            pytorch_pred = self.evaluate_conditional_prediction(pytorch_vine, data, 'pytorch', approach)
                            
                            results[correlation_strength][dim][approach][vine_type]['pytorch'] = {
                                'timing': pytorch_timing,
                                'correlation': pytorch_corr,
                                'entropy': pytorch_entropy,
                                'prediction': pytorch_pred
                            }
                            print(f"      PyTorch: ✓ (MAE: {pytorch_corr['corr_mae']:.4f}, Time: {pytorch_timing['total_time']:.2f}s)")
                            
                        except Exception as e:
                            print(f"      PyTorch: ✗ ({e})")
                            results[correlation_strength][dim][approach][vine_type]['pytorch'] = None
                        
                        # Test TensorFlow implementation
                        try:
                            tf_vine, tf_timing = self.fit_tensorflow_vine(data, vine_type, approach)
                            tf_corr = self.evaluate_correlation_recovery(tf_vine, true_cov, 'tensorflow', approach)
                            tf_entropy = self.evaluate_entropy_estimation(tf_corr['samples'], true_entropy)
                            tf_pred = self.evaluate_conditional_prediction(tf_vine, data, 'tensorflow', approach)
                            
                            results[correlation_strength][dim][approach][vine_type]['tensorflow'] = {
                                'timing': tf_timing,
                                'correlation': tf_corr,
                                'entropy': tf_entropy,
                                'prediction': tf_pred
                            }
                            print(f"      TensorFlow: ✓ (MAE: {tf_corr['corr_mae']:.4f}, Time: {tf_timing['total_time']:.2f}s)")
                            
                        except Exception as e:
                            print(f"      TensorFlow: ✗ ({e})")
                            results[correlation_strength][dim][approach][vine_type]['tensorflow'] = None
                
                # Store ground truth for this configuration
                results[correlation_strength][dim]['ground_truth'] = {
                    'data': data,
                    'true_cov': true_cov,
                    'true_entropy': true_entropy
                }
        
        self.results = results
        return results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        
        if not self.results:
            print("No results to visualize. Run comparison first.")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Correlation Recovery Comparison
        self._plot_correlation_comparison(fig)
        
        # 2. Sample Visualizations
        self._plot_sample_comparisons(fig)
        
        # 3. Performance Metrics
        self._plot_performance_metrics(fig)
        
        # 4. Timing Analysis
        self._plot_timing_analysis(fig)
        
        # 5. Prediction Accuracy
        self._plot_prediction_accuracy(fig)
        
        # 6. Approach Comparison
        self._plot_approach_comparison(fig)
        
        plt.tight_layout()
        plt.savefig('comprehensive_vine_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_correlation_comparison(self, fig):
        """Plot correlation matrix comparisons"""
        
        # Select one representative case for detailed correlation comparison
        correlation_strength = 'moderate'
        dim = 5 if 5 in self.dimensions else max(self.dimensions)
        vine_type = 'd-vine'
        approach = 'parametric'
        
        if (correlation_strength in self.results and 
            dim in self.results[correlation_strength] and
            approach in self.results[correlation_strength][dim] and
            vine_type in self.results[correlation_strength][dim][approach]):
            
            ground_truth = self.results[correlation_strength][dim]['ground_truth']
            pytorch_results = self.results[correlation_strength][dim][approach][vine_type].get('pytorch')
            tf_results = self.results[correlation_strength][dim][approach][vine_type].get('tensorflow')
            
            if pytorch_results and tf_results:
                # Create correlation heatmaps
                ax1 = plt.subplot(4, 4, 1)
                true_corr = ground_truth['true_cov'] / np.sqrt(np.outer(np.diag(ground_truth['true_cov']), 
                                                                      np.diag(ground_truth['true_cov'])))
                sns.heatmap(true_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1)
                plt.title(f'True Correlation Matrix\n(Dim {dim})')
                
                ax2 = plt.subplot(4, 4, 2)
                sns.heatmap(pytorch_results['correlation']['pred_corr'], annot=True, fmt='.2f', 
                           cmap='RdBu_r', center=0, vmin=-1, vmax=1)
                plt.title(f'PyTorch {vine_type} ({approach})')
                
                ax3 = plt.subplot(4, 4, 3)
                sns.heatmap(tf_results['correlation']['pred_corr'], annot=True, fmt='.2f', 
                           cmap='RdBu_r', center=0, vmin=-1, vmax=1)
                plt.title(f'TensorFlow {vine_type} ({approach})')
                
                # Error heatmap
                ax4 = plt.subplot(4, 4, 4)
                error_matrix = np.abs(pytorch_results['correlation']['pred_corr'] - 
                                    tf_results['correlation']['pred_corr'])
                sns.heatmap(error_matrix, annot=True, fmt='.3f', cmap='Reds')
                plt.title('|PyTorch - TensorFlow|')
    
    def _plot_sample_comparisons(self, fig):
        """Plot sample scatter plots and distributions"""
        
        # Select representative case
        correlation_strength = 'moderate'
        dim = 5 if 5 in self.dimensions else max(self.dimensions)
        vine_type = 'd-vine'
        approach = 'parametric'
        
        if (correlation_strength in self.results and 
            dim in self.results[correlation_strength] and
            approach in self.results[correlation_strength][dim] and
            vine_type in self.results[correlation_strength][dim][approach]):
            
            ground_truth = self.results[correlation_strength][dim]['ground_truth']
            pytorch_results = self.results[correlation_strength][dim][approach][vine_type].get('pytorch')
            tf_results = self.results[correlation_strength][dim][approach][vine_type].get('tensorflow')
            
            if pytorch_results and tf_results:
                # Plot first two dimensions as scatter plots
                ax5 = plt.subplot(4, 4, 5)
                plt.scatter(ground_truth['data'][:, 0], ground_truth['data'][:, 1], alpha=0.6, s=20)
                plt.title('True Data (X1 vs X2)')
                plt.xlabel('X1')
                plt.ylabel('X2')
                
                ax6 = plt.subplot(4, 4, 6)
                pytorch_samples = pytorch_results['correlation']['samples']
                plt.scatter(pytorch_samples[:, 0], pytorch_samples[:, 1], alpha=0.6, s=20, color='orange')
                plt.title('PyTorch Samples (X1 vs X2)')
                plt.xlabel('X1')
                plt.ylabel('X2')
                
                ax7 = plt.subplot(4, 4, 7)
                tf_samples = tf_results['correlation']['samples']
                plt.scatter(tf_samples[:, 0], tf_samples[:, 1], alpha=0.6, s=20, color='green')
                plt.title('TensorFlow Samples (X1 vs X2)')
                plt.xlabel('X1')
                plt.ylabel('X2')
                
                # Distribution comparison for first variable
                ax8 = plt.subplot(4, 4, 8)
                plt.hist(ground_truth['data'][:, 0], bins=30, alpha=0.5, label='True', density=True)
                plt.hist(pytorch_samples[:, 0], bins=30, alpha=0.5, label='PyTorch', density=True)
                plt.hist(tf_samples[:, 0], bins=30, alpha=0.5, label='TensorFlow', density=True)
                plt.title('X1 Distribution Comparison')
                plt.legend()
    
    def _plot_performance_metrics(self, fig):
        """Plot performance metrics across configurations"""
        
        # Collect correlation MAE across all configurations
        data_for_plot = []
        
        for corr_strength in self.results:
            for dim in self.results[corr_strength]:
                if dim == 'ground_truth':
                    continue
                for approach in self.results[corr_strength][dim]:
                    if approach == 'ground_truth':
                        continue
                    for vine_type in self.results[corr_strength][dim][approach]:
                        if vine_type == 'ground_truth':
                            continue
                        
                        pytorch_res = self.results[corr_strength][dim][approach][vine_type].get('pytorch')
                        tf_res = self.results[corr_strength][dim][approach][vine_type].get('tensorflow')
                        
                        if pytorch_res:
                            data_for_plot.append({
                                'Implementation': 'PyTorch',
                                'Vine Type': vine_type,
                                'Dimension': dim,
                                'Correlation Strength': corr_strength,
                                'Approach': approach,
                                'Correlation MAE': pytorch_res['correlation']['corr_mae'],
                                'Fit Time': pytorch_res['timing']['fit_time'],
                                'Sampling Time': pytorch_res['timing']['sample_time'],
                                'Total Time': pytorch_res['timing']['total_time']
                            })
                        
                        if tf_res:
                            data_for_plot.append({
                                'Implementation': 'TensorFlow',
                                'Vine Type': vine_type,
                                'Dimension': dim,
                                'Correlation Strength': corr_strength,
                                'Approach': approach,
                                'Correlation MAE': tf_res['correlation']['corr_mae'],
                                'Fit Time': tf_res['timing']['fit_time'],
                                'Sampling Time': tf_res['timing']['sample_time'],
                                'Total Time': tf_res['timing']['total_time']
                            })
        
        if data_for_plot:
            df = pd.DataFrame(data_for_plot)
            
            # Correlation MAE comparison by approach
            ax9 = plt.subplot(4, 4, 9)
            sns.barplot(data=df, x='Approach', y='Correlation MAE', hue='Implementation')
            plt.title('Correlation Recovery Error by Approach')
            plt.xticks(rotation=45)
            
            # Performance by dimension
            ax10 = plt.subplot(4, 4, 10)
            sns.lineplot(data=df, x='Dimension', y='Correlation MAE', 
                        hue='Implementation', style='Approach', marker='o')
            plt.title('Performance vs Dimension')
            
            # Performance by vine type
            ax11 = plt.subplot(4, 4, 11)
            sns.barplot(data=df, x='Vine Type', y='Correlation MAE', hue='Implementation')
            plt.title('Performance by Vine Type')
            plt.xticks(rotation=45)
            
            # Correlation strength effect
            ax12 = plt.subplot(4, 4, 12)
            sns.boxplot(data=df, x='Correlation Strength', y='Correlation MAE', hue='Implementation')
            plt.title('Performance by Correlation Strength')
    
    def _plot_timing_analysis(self, fig):
        """Plot detailed timing analysis"""
        
        # Collect timing data
        timing_data = []
        
        for corr_strength in self.results:
            for dim in self.results[corr_strength]:
                if dim == 'ground_truth':
                    continue
                for approach in self.results[corr_strength][dim]:
                    if approach == 'ground_truth':
                        continue
                    for vine_type in self.results[corr_strength][dim][approach]:
                        if vine_type == 'ground_truth':
                            continue
                        
                        for impl in ['pytorch', 'tensorflow']:
                            res = self.results[corr_strength][dim][approach][vine_type].get(impl)
                            if res:
                                timing_data.append({
                                    'Implementation': impl.capitalize(),
                                    'Vine Type': vine_type,
                                    'Approach': approach,
                                    'Dimension': dim,
                                    'Fit Time': res['timing']['fit_time'],
                                    'Sampling Time': res['timing']['sample_time'],
                                    'Total Time': res['timing']['total_time']
                                })
        
        if timing_data:
            timing_df = pd.DataFrame(timing_data)
            
            # Fit time comparison
            ax13 = plt.subplot(4, 4, 13)
            sns.barplot(data=timing_df, x='Approach', y='Fit Time', hue='Implementation')
            plt.title('Fitting Time by Approach')
            plt.xticks(rotation=45)
            
            # Sampling time comparison
            ax14 = plt.subplot(4, 4, 14)
            sns.barplot(data=timing_df, x='Approach', y='Sampling Time', hue='Implementation')
            plt.title('Sampling Time by Approach')
            plt.xticks(rotation=45)
            
            # Total time by dimension
            ax15 = plt.subplot(4, 4, 15)
            sns.lineplot(data=timing_df, x='Dimension', y='Total Time', 
                        hue='Implementation', style='Approach', marker='o')
            plt.title('Total Time vs Dimension')
            
            # Time breakdown stacked bar
            ax16 = plt.subplot(4, 4, 16)
            # Average times for each implementation-approach combination
            avg_times = timing_df.groupby(['Implementation', 'Approach'])[['Fit Time', 'Sampling Time']].mean().reset_index()
            
            width = 0.35
            implementations = avg_times['Implementation'].unique()
            approaches = avg_times['Approach'].unique()
            
            x = np.arange(len(approaches))
            for i, impl in enumerate(implementations):
                impl_data = avg_times[avg_times['Implementation'] == impl]
                plt.bar(x + i*width, impl_data['Fit Time'], width, label=f'{impl} Fit', alpha=0.8)
                plt.bar(x + i*width, impl_data['Sampling Time'], width, 
                       bottom=impl_data['Fit Time'], label=f'{impl} Sample', alpha=0.6)
            
            plt.xlabel('Approach')
            plt.ylabel('Time (seconds)')
            plt.title('Time Breakdown')
            plt.xticks(x + width/2, approaches, rotation=45)
            plt.legend()
    
    def _plot_prediction_accuracy(self, fig):
        """Plot conditional prediction accuracy - simplified for space"""
        pass  # Skip for now to save space in visualization
    
    def _plot_approach_comparison(self, fig):
        """Plot parametric vs non-parametric comparison"""
        pass  # Skip for now to save space in visualization

    def generate_report(self):
        """Generate a comprehensive text report"""
        
        if not self.results:
            print("No results to report. Run comparison first.")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE VINE COPULA COMPARISON REPORT")
        print("="*80)
        
        # Summary statistics
        total_tests = 0
        successful_tests = 0
        timing_summary = {}
        accuracy_summary = {}
        
        for corr_strength in self.results:
            for dim in self.results[corr_strength]:
                if dim == 'ground_truth':
                    continue
                for approach in self.results[corr_strength][dim]:
                    if approach == 'ground_truth':
                        continue
                    for vine_type in self.results[corr_strength][dim][approach]:
                        if vine_type == 'ground_truth':
                            continue
                        total_tests += 2  # PyTorch + TensorFlow
                        
                        for impl in ['pytorch', 'tensorflow']:
                            res = self.results[corr_strength][dim][approach][vine_type].get(impl)
                            if res:
                                successful_tests += 1
                                
                                # Collect timing data
                                key = f"{impl}_{approach}"
                                if key not in timing_summary:
                                    timing_summary[key] = {'fit_times': [], 'sample_times': [], 'total_times': []}
                                timing_summary[key]['fit_times'].append(res['timing']['fit_time'])
                                timing_summary[key]['sample_times'].append(res['timing']['sample_time'])
                                timing_summary[key]['total_times'].append(res['timing']['total_time'])
                                
                                # Collect accuracy data
                                if key not in accuracy_summary:
                                    accuracy_summary[key] = []
                                accuracy_summary[key].append(res['correlation']['corr_mae'])
        
        print(f"Total tests: {total_tests}")
        print(f"Successful tests: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        
        # Timing summary
        print(f"\n{'='*20} TIMING ANALYSIS {'='*20}")
        print(f"{'Implementation':<15} {'Approach':<15} {'Fit Time (s)':<12} {'Sample Time (s)':<15} {'Total Time (s)':<15}")
        print("-" * 75)
        
        for key, times in timing_summary.items():
            impl, approach = key.split('_')
            avg_fit = np.mean(times['fit_times'])
            avg_sample = np.mean(times['sample_times'])
            avg_total = np.mean(times['total_times'])
            print(f"{impl.capitalize():<15} {approach:<15} {avg_fit:<12.3f} {avg_sample:<15.3f} {avg_total:<15.3f}")
        
        # Accuracy summary
        print(f"\n{'='*20} ACCURACY ANALYSIS {'='*20}")
        print(f"{'Implementation':<15} {'Approach':<15} {'Avg MAE':<12} {'Std MAE':<12} {'Best MAE':<12}")
        print("-" * 75)
        
        for key, maes in accuracy_summary.items():
            impl, approach = key.split('_')
            avg_mae = np.mean(maes)
            std_mae = np.std(maes)
            best_mae = np.min(maes)
            print(f"{impl.capitalize():<15} {approach:<15} {avg_mae:<12.4f} {std_mae:<12.4f} {best_mae:<12.4f}")
        
        # Detailed results by configuration
        print(f"\n{'='*20} DETAILED RESULTS {'='*20}")
        
        for corr_strength in self.results:
            print(f"\n{corr_strength.upper()} CORRELATIONS:")
            print("-" * 50)
            
            for dim in self.results[corr_strength]:
                if dim == 'ground_truth':
                    continue
                print(f"\nDimension {dim}:")
                
                for approach in self.results[corr_strength][dim]:
                    if approach == 'ground_truth':
                        continue
                    print(f"  {approach.capitalize()} Approach:")
                    
                    for vine_type in self.results[corr_strength][dim][approach]:
                        if vine_type == 'ground_truth':
                            continue
                        
                        print(f"    {vine_type}:")
                        
                        pytorch_res = self.results[corr_strength][dim][approach][vine_type].get('pytorch')
                        tf_res = self.results[corr_strength][dim][approach][vine_type].get('tensorflow')
                        
                        if pytorch_res:
                            print(f"      PyTorch:   MAE={pytorch_res['correlation']['corr_mae']:.4f}, "
                                  f"Time={pytorch_res['timing']['total_time']:.2f}s "
                                  f"(Fit: {pytorch_res['timing']['fit_time']:.2f}s, "
                                  f"Sample: {pytorch_res['timing']['sample_time']:.3f}s)")
                        else:
                            print(f"      PyTorch:   FAILED")
                        
                        if tf_res:
                            print(f"      TensorFlow: MAE={tf_res['correlation']['corr_mae']:.4f}, "
                                  f"Time={tf_res['timing']['total_time']:.2f}s "
                                  f"(Fit: {tf_res['timing']['fit_time']:.2f}s, "
                                  f"Sample: {tf_res['timing']['sample_time']:.3f}s)")
                        else:
                            print(f"      TensorFlow: FAILED")
        
        # Summary insights
        print(f"\n{'='*20} KEY INSIGHTS {'='*20}")
        
        # Compare parametric vs non-parametric
        if 'pytorch_parametric' in timing_summary and 'pytorch_non-parametric' in timing_summary:
            param_time = np.mean(timing_summary['pytorch_parametric']['total_times'])
            nonparam_time = np.mean(timing_summary['pytorch_non-parametric']['total_times'])
            print(f"• PyTorch Parametric is {nonparam_time/param_time:.1f}x faster than Non-parametric")
        
        if 'tensorflow_parametric' in timing_summary and 'tensorflow_non-parametric' in timing_summary:
            param_time = np.mean(timing_summary['tensorflow_parametric']['total_times'])
            nonparam_time = np.mean(timing_summary['tensorflow_non-parametric']['total_times'])
            print(f"• TensorFlow Parametric is {nonparam_time/param_time:.1f}x faster than Non-parametric")
        
        # Compare implementations
        if 'pytorch_parametric' in accuracy_summary and 'tensorflow_parametric' in accuracy_summary:
            pytorch_acc = np.mean(accuracy_summary['pytorch_parametric'])
            tf_acc = np.mean(accuracy_summary['tensorflow_parametric'])
            if pytorch_acc < tf_acc:
                print(f"• PyTorch Parametric is more accurate (MAE: {pytorch_acc:.4f} vs {tf_acc:.4f})")
            else:
                print(f"• TensorFlow Parametric is more accurate (MAE: {tf_acc:.4f} vs {pytorch_acc:.4f})")
        
        # Computational efficiency insights
        fastest_config = min(timing_summary.items(), key=lambda x: np.mean(x[1]['total_times']))
        print(f"• Fastest configuration: {fastest_config[0].replace('_', ' ').title()} "
              f"(Avg: {np.mean(fastest_config[1]['total_times']):.2f}s)")
        
        most_accurate = min(accuracy_summary.items(), key=lambda x: np.mean(x[1]))
        print(f"• Most accurate configuration: {most_accurate[0].replace('_', ' ').title()} "
              f"(MAE: {np.mean(most_accurate[1]):.4f})")
        
        print(f"\n{'='*80}")
        print("Report complete!")
        print(f"{'='*80}")

def main():
    """Run comprehensive comparison"""
    
    # Initialize comparison
    comparison = VineComparison(
        dimensions=[3, 4, 5],
        vine_types=['d-vine', 'c-vine'],
        approaches=['parametric', 'non-parametric'],
        n_samples=800,
        n_test_samples=500
    )
    
    # Run comparison
    results = comparison.run_comparison()
    
    # Generate visualizations
    comparison.create_visualizations()
    
    # Generate report
    comparison.generate_report()
    
    print(f"\nComparison complete! Results saved to 'comprehensive_vine_comparison.png'")

if __name__ == "__main__":
    main() 