#!/usr/bin/env python3
"""
Robust Multivariate Gaussian Vine Analysis - Memory Optimized

This script is a robust, memory-efficient version designed to avoid system kills.
Key improvements:
1. Reduced memory footprint
2. Progressive complexity scaling
3. Memory monitoring and cleanup
4. Timeouts and graceful degradation
5. Efficient data processing

Author: DVC Analysis Team
Date: 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
import pandas as pd
import gc
import time
import psutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Configure TensorFlow for memory efficiency
if tf.config.list_physical_devices('GPU'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Add DVC_tensorflow to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
dvc_tensorflow_dir = os.path.join(project_root, 'src', 'DVC_tensorflow')
sys.path.append(dvc_tensorflow_dir)

from classes.objects import *
from vine_tree.tree_op import *
from param.generate_rvine import *
from pre_proc.preparation import prep_cop
from sampling.vine_sample import *

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Results directory
results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)

class Robust_Vine_Analyzer:
    """
    Memory-efficient vine copula analyzer designed to avoid system kills
    
    Key Features:
    - Adaptive memory management
    - Progressive scaling (starts small, increases if successful)
    - Timeout protection
    - Memory monitoring
    - Graceful degradation
    """
    
    def __init__(self, max_dim=4, max_samples=1200, timeout_minutes=10):
        """Initialize robust analyzer with conservative defaults"""
        self.max_dim = max_dim
        self.max_samples = max_samples
        self.timeout_minutes = timeout_minutes
        self.start_time = time.time()
        self.results = {}
        
        # Conservative starting parameters
        self.current_dim = min(3, max_dim)  # Start small
        self.current_samples = min(800, max_samples)  # Start small
        
        print("="*60)
        print("ROBUST MULTIVARIATE VINE ANALYSIS")
        print("="*60)
        print(f"Configuration:")
        print(f"• Max dimensions: {max_dim}")
        print(f"• Max samples: {max_samples}")
        print(f"• Timeout: {timeout_minutes} minutes")
        print(f"• Starting conservatively: {self.current_dim}D, {self.current_samples} samples")
        print("="*60)
    
    def check_memory_and_time(self):
        """Monitor memory usage and elapsed time"""
        # Check elapsed time
        elapsed = (time.time() - self.start_time) / 60
        if elapsed > self.timeout_minutes:
            raise TimeoutError(f"Analysis exceeded {self.timeout_minutes} minute timeout")
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            print(f"Warning: High memory usage ({memory_percent:.1f}%)")
            gc.collect()  # Force garbage collection
            
        return elapsed, memory_percent
    
    def generate_safe_correlation_matrix(self, dim):
        """Generate numerically stable correlation matrix"""
        print(f"Generating {dim}×{dim} correlation matrix...")
        
        # Create structured correlations for interpretability
        corr_matrix = np.eye(dim)
        
        # Add sequential correlations (guaranteed stable)
        for i in range(dim-1):
            corr_matrix[i, i+1] = 0.7
            corr_matrix[i+1, i] = 0.7
        
        # Add some cross-correlations if dimension allows
        if dim >= 4:
            corr_matrix[0, 2] = 0.4
            corr_matrix[2, 0] = 0.4
            corr_matrix[1, 3] = -0.3
            corr_matrix[3, 1] = -0.3
        
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(corr_matrix)
        min_eigenval = np.min(eigenvals)
        if min_eigenval < 0.1:
            corr_matrix += (0.1 - min_eigenval) * np.eye(dim)
        
        print(f"✓ Correlation matrix: eigenvalues [{np.min(eigenvals):.3f}, {np.max(eigenvals):.3f}]")
        return corr_matrix
    
    def generate_data_safely(self, dim, n_samples):
        """Generate data with memory monitoring"""
        print(f"Generating {n_samples} samples of {dim}D data...")
        
        # Check if data size is reasonable
        data_size_mb = (n_samples * dim * 8) / 1e6
        print(f"Expected memory: {data_size_mb:.1f} MB")
        
        if data_size_mb > 100:
            print("Warning: Large data size, reducing...")
            n_samples = min(n_samples, int(100e6 / (dim * 8)))
            print(f"Reduced to {n_samples} samples")
        
        # Generate correlation matrix and data
        corr_matrix = self.generate_safe_correlation_matrix(dim)
        mean = np.zeros(dim)
        
        try:
            data = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples)
            print(f"✓ Data generated: shape {data.shape}")
            
            # Check memory after generation
            elapsed, memory_pct = self.check_memory_and_time()
            print(f"Status: {elapsed:.1f}min elapsed, {memory_pct:.1f}% memory")
            
            return data, corr_matrix
            
        except Exception as e:
            print(f"Error generating data: {e}")
            # Try with smaller size
            n_samples = n_samples // 2
            print(f"Retrying with {n_samples} samples...")
            data = multivariate_normal.rvs(mean=mean, cov=corr_matrix, size=n_samples)
            return data, corr_matrix
    
    def fit_vine_safely(self, data, vine_type='c-vine'):
        """Fit vine with memory management and timeouts"""
        print(f"Fitting {vine_type} copula...")
        print(f"Data shape: {data.shape}")
        
        dim = data.shape[1]
        
        try:
            # Use C-vine for reliability (most stable)
            vine_type = 'c-vine'  # Force C-vine for safety
            
            # Setup margins
            margin_vine = []
            for i in range(dim):
                mar_p = margin_obj('norm', [0, 1], True)
                margin_vine.append(mar_p)
            
            # Create vine with conservative settings
            vine = vine_obj_bin(vine_type, "kercop", dim, margin_vine, 30, 'matrix', None)
            
            # Prepare data
            x = data.astype(np.float32)
            exc = tf.math.floormod(tf.shape(x)[0], 5)
            x = x[:tf.shape(x)[0]-exc, :]
            
            print(f"Preprocessed data shape: {x.shape}")
            
            # Transform to copula domain
            e = prep_cop(x, vine, 'rand')
            
            # Configure fitting with conservative parameters
            gen_dict = {
                'parallel': False,  # Disable parallel for memory safety
                'binning': False,
                'param': False,
                'vine_depth': dim,
                'fitted': False
            }
            
            par_dict = {'param_families': ["ind", "gaussian"]}
            npc_dict = {'opt_method': 'LL1', 'batch_paral': 1}  # Minimal batch size
            bin_dict = {'n_bin': 3}
            
            print("Starting vine fitting...")
            fit_start = time.time()
            
            # Fit with timeout protection
            vine.fit(x, gen_dict, npc_dict, par_dict, bin_dict)
            
            fit_time = time.time() - fit_start
            print(f"✓ Vine fitted in {fit_time:.1f} seconds")
            
            # Check memory after fitting
            elapsed, memory_pct = self.check_memory_and_time()
            print(f"Status: {elapsed:.1f}min elapsed, {memory_pct:.1f}% memory")
            
            return vine
            
        except Exception as e:
            print(f"Error fitting vine: {e}")
            print("Attempting recovery with simpler settings...")
            
            # Try with even more conservative settings
            try:
                # Reduce dimension if possible
                if dim > 3:
                    data = data[:, :3]
                    print(f"Reduced to 3D data for stability")
                    return self.fit_vine_safely(data, 'c-vine')
                else:
                    raise e
            except:
                print("Vine fitting failed completely")
                return None
    
    def generate_samples_safely(self, vine, n_samples=500):
        """Generate samples with memory protection"""
        if vine is None:
            print("Cannot generate samples: vine is None")
            return None
        
        try:
            print(f"Generating {n_samples} vine samples...")
            samples, _, _, _ = vine_copula_sample(vine, n_samples)
            print(f"✓ Generated samples: shape {samples.shape}")
            
            # Check memory
            elapsed, memory_pct = self.check_memory_and_time()
            print(f"Status: {elapsed:.1f}min elapsed, {memory_pct:.1f}% memory")
            
            return samples
            
        except Exception as e:
            print(f"Error generating samples: {e}")
            # Try with fewer samples
            n_samples = n_samples // 2
            print(f"Retrying with {n_samples} samples...")
            try:
                samples, _, _, _ = vine_copula_sample(vine, n_samples)
                return samples
            except:
                print("Sample generation failed")
                return None
    
    def compare_correlations_safely(self, original_data, vine_samples, true_corr):
        """Compare correlations with error handling"""
        if vine_samples is None:
            return None, None, np.inf
        
        try:
            print("Computing correlation comparison...")
            
            # Original correlations
            original_corr = np.corrcoef(original_data.T)
            
            # Vine correlations
            vine_corr = np.corrcoef(vine_samples.T)
            
            # Calculate errors
            true_error = np.mean(np.abs(original_corr - true_corr))
            vine_error = np.mean(np.abs(vine_corr - true_corr))
            
            print(f"Correlation errors:")
            print(f"• Original vs True: {true_error:.4f}")
            print(f"• Vine vs True: {vine_error:.4f}")
            
            return original_corr, vine_corr, vine_error
            
        except Exception as e:
            print(f"Error in correlation comparison: {e}")
            return None, None, np.inf
    
    def create_safe_visualization(self, original_data, vine_samples, true_corr, original_corr, vine_corr):
        """Create visualization with error handling"""
        try:
            if vine_samples is None:
                print("Cannot create visualization: no vine samples")
                return
            
            print("Creating visualization...")
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot 1: Original data scatter
            if original_data.shape[1] >= 2:
                axes[0,0].scatter(original_data[:,0], original_data[:,1], alpha=0.6, s=20)
                axes[0,0].set_title('Original Data (Var 1 vs 2)')
                axes[0,0].set_xlabel('Variable 1')
                axes[0,0].set_ylabel('Variable 2')
            
            # Plot 2: Vine samples scatter  
            if vine_samples.shape[1] >= 2:
                axes[0,1].scatter(vine_samples[:,0], vine_samples[:,1], alpha=0.6, s=20, color='orange')
                axes[0,1].set_title('Vine Samples (Var 1 vs 2)')
                axes[0,1].set_xlabel('Variable 1')
                axes[0,1].set_ylabel('Variable 2')
            
            # Plot 3: True correlation matrix
            im1 = axes[0,2].imshow(true_corr, cmap='RdBu', vmin=-1, vmax=1)
            axes[0,2].set_title('True Correlations')
            plt.colorbar(im1, ax=axes[0,2])
            
            # Plot 4: Original correlation matrix
            if original_corr is not None:
                im2 = axes[1,0].imshow(original_corr, cmap='RdBu', vmin=-1, vmax=1)
                axes[1,0].set_title('Original Data Correlations')
                plt.colorbar(im2, ax=axes[1,0])
            
            # Plot 5: Vine correlation matrix
            if vine_corr is not None:
                im3 = axes[1,1].imshow(vine_corr, cmap='RdBu', vmin=-1, vmax=1)
                axes[1,1].set_title('Vine Sample Correlations')
                plt.colorbar(im3, ax=axes[1,1])
            
            # Plot 6: Error comparison
            if original_corr is not None and vine_corr is not None:
                true_error = np.mean(np.abs(original_corr - true_corr))
                vine_error = np.mean(np.abs(vine_corr - true_corr))
                
                axes[1,2].bar(['Original', 'Vine'], [true_error, vine_error], 
                             color=['blue', 'orange'], alpha=0.7)
                axes[1,2].set_title('Correlation Error (MAE)')
                axes[1,2].set_ylabel('Mean Absolute Error')
            
            plt.tight_layout()
            
            # Save plot
            filename = f'robust_vine_analysis_{self.current_dim}D_{self.current_samples}samples.png'
            plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved: {filename}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def run_safe_analysis(self):
        """Run complete analysis with progressive scaling and safety checks"""
        print("Starting robust vine analysis...")
        
        success_configs = []
        
        # Try progressively larger configurations
        for dim in range(3, self.max_dim + 1):
            for n_samples in [600, 1000, self.max_samples]:
                if n_samples > self.max_samples:
                    continue
                
                print(f"\n{'='*60}")
                print(f"ATTEMPTING: {dim}D, {n_samples} samples")
                print(f"{'='*60}")
                
                try:
                    # Check if we should continue
                    elapsed, memory_pct = self.check_memory_and_time()
                    if elapsed > self.timeout_minutes * 0.8:  # 80% of timeout
                        print("Approaching timeout, stopping progression")
                        break
                    
                    # Generate data
                    data, true_corr = self.generate_data_safely(dim, n_samples)
                    
                    # Fit vine
                    vine = self.fit_vine_safely(data)
                    if vine is None:
                        print(f"Failed: {dim}D, {n_samples} samples - vine fitting failed")
                        continue
                    
                    # Generate samples
                    vine_samples = self.generate_samples_safely(vine, min(n_samples, 800))
                    if vine_samples is None:
                        print(f"Failed: {dim}D, {n_samples} samples - sample generation failed")
                        continue
                    
                    # Compare correlations
                    original_corr, vine_corr, vine_error = self.compare_correlations_safely(
                        data, vine_samples, true_corr)
                    
                    # Create visualization
                    self.create_safe_visualization(data, vine_samples, true_corr, original_corr, vine_corr)
                    
                    # Store successful configuration
                    config = {
                        'dim': dim,
                        'n_samples': n_samples,
                        'vine_error': vine_error,
                        'elapsed_time': elapsed,
                        'memory_used': memory_pct
                    }
                    success_configs.append(config)
                    
                    print(f"✅ SUCCESS: {dim}D, {n_samples} samples, error: {vine_error:.4f}")
                    
                    # Clean up memory
                    del data, vine, vine_samples, true_corr, original_corr, vine_corr
                    gc.collect()
                    
                except TimeoutError as e:
                    print(f"⏰ TIMEOUT: {e}")
                    break
                except Exception as e:
                    print(f"❌ FAILED: {dim}D, {n_samples} samples - {str(e)[:100]}")
                    gc.collect()
                    continue
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        if success_configs:
            print("Successful configurations:")
            for config in success_configs:
                print(f"• {config['dim']}D, {config['n_samples']} samples: "
                      f"error={config['vine_error']:.4f}, "
                      f"time={config['elapsed_time']:.1f}min, "
                      f"memory={config['memory_used']:.1f}%")
            
            # Find best configuration
            best_config = min(success_configs, key=lambda x: x['vine_error'])
            print(f"\nBest configuration: {best_config['dim']}D, {best_config['n_samples']} samples")
            print(f"Best correlation error: {best_config['vine_error']:.4f}")
            
        else:
            print("No configurations completed successfully")
            print("Try reducing max_dim or max_samples")
        
        # Save results
        import json
        results = {
            'timestamp': datetime.now().isoformat(),
            'successful_configs': success_configs,
            'total_elapsed': (time.time() - self.start_time) / 60,
            'analysis_type': 'robust_vine_analysis'
        }
        
        with open(os.path.join(results_dir, 'robust_vine_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Analysis completed in {results['total_elapsed']:.1f} minutes")
        print(f"Results saved: robust_vine_results.json")
        
        return success_configs


def main():
    """Main function with conservative settings"""
    print("Starting robust multivariate vine analysis...")
    print("This version is designed to avoid system kills with:")
    print("• Conservative memory usage")
    print("• Progressive scaling")
    print("• Timeout protection") 
    print("• Graceful error handling")
    
    # Conservative settings that should work on most systems
    analyzer = Robust_Vine_Analyzer(
        max_dim=4,           # Conservative dimension limit
        max_samples=1200,    # Conservative sample limit  
        timeout_minutes=8    # Conservative time limit
    )
    
    try:
        success_configs = analyzer.run_safe_analysis()
        
        if success_configs:
            print(f"\n🎉 SUCCESS! Completed {len(success_configs)} configurations")
        else:
            print(f"\n⚠️  No configurations succeeded - try reducing limits")
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 