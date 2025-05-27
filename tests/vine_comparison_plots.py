#!/usr/bin/env python3
"""
Comprehensive Visual Comparison of PyTorch vs TensorFlow Vine Copulas
Tests both C-vine and D-vine structures with detailed plotting.
"""

import sys, os, numpy as np, pandas as pd, time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal, pearsonr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'DVC_tensorflow'))

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_test_data(n_samples=800, scenario='mixed'):
    """Create different correlation scenarios for testing."""
    np.random.seed(42)
    
    if scenario == 'mixed':
        # Mixed correlations - good for both C and D vines
        true_corr = np.array([
            [1.00, 0.75, 0.40, 0.20, 0.10],
            [0.75, 1.00, 0.65, 0.30, 0.15],
            [0.40, 0.65, 1.00, 0.55, 0.35],
            [0.20, 0.30, 0.55, 1.00, 0.70],
            [0.10, 0.15, 0.35, 0.70, 1.00]
        ])
        title = "Mixed Correlation Structure"
        
    elif scenario == 'chain':
        # Chain-like correlations - ideal for D-vines
        true_corr = np.array([
            [1.00, 0.80, 0.40, 0.20, 0.10],
            [0.80, 1.00, 0.75, 0.35, 0.15],
            [0.40, 0.75, 1.00, 0.70, 0.30],
            [0.20, 0.35, 0.70, 1.00, 0.65],
            [0.10, 0.15, 0.30, 0.65, 1.00]
        ])
        title = "Chain Correlation Structure (D-vine favorable)"
        
    elif scenario == 'star':
        # Star-like correlations - ideal for C-vines (made positive definite)
        true_corr = np.array([
            [1.00, 0.70, 0.60, 0.50, 0.40],
            [0.70, 1.00, 0.30, 0.25, 0.20],
            [0.60, 0.30, 1.00, 0.25, 0.20],
            [0.50, 0.25, 0.25, 1.00, 0.20],
            [0.40, 0.20, 0.20, 0.20, 1.00]
        ])
        title = "Star Correlation Structure (C-vine favorable)"
    
    # Generate data
    data = multivariate_normal.rvs(mean=np.zeros(5), cov=true_corr, size=n_samples)
    empirical_corr = np.corrcoef(data, rowvar=False)
    
    return data, true_corr, empirical_corr, title

def fit_pytorch_vine(data, vine_type='c-vine'):
    """Fit PyTorch vine copula."""
    try:
        from DVC_pyolder.objects import vine_obj_bin, margin_obj
        
        start_time = time.time()
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(5)]
        vine = vine_obj_bin(vine_type, ['gaussian'], 5, margins, 25)
        
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        par_dict = {'param_families': ['gaussian']}
        npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
        bin_dict = {'n_bin': 5}
        
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        # Generate samples
        samples = vine.sample(1000)
        pred_corr = np.corrcoef(samples, rowvar=False)
        
        # Extract fitted parameters
        fitted_params = []
        for level, copulas in enumerate(vine.copulas):
            level_params = []
            for cop in copulas:
                if hasattr(cop, 'theta'):
                    level_params.append(cop.theta)
                else:
                    level_params.append(0.0)
            fitted_params.append(level_params)
        
        return {
            'success': True,
            'samples': samples,
            'pred_corr': pred_corr,
            'fit_time': fit_time,
            'fitted_params': fitted_params
        }
        
    except Exception as e:
        print(f"PyTorch {vine_type} failed: {e}")
        return {'success': False, 'error': str(e)}

def fit_tensorflow_vine(data, vine_type='c-vine'):
    """Fit TensorFlow vine copula."""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        from classes.objects import vine_obj_bin as tf_vine_obj, margin_obj as tf_margin_obj
        from sampling.vine_sample import vine_cop_par_sample
        
        start_time = time.time()
        tf_margins = []
        for i in range(5):
            tf_margin = tf_margin_obj('norm', (0.0, 1.0), True)
            tf_margin.ker = data[:, i].astype(np.float32)
            tf_margins.append(tf_margin)
        
        tf_vine = tf_vine_obj(vine_type, ['gaussian'], 5, tf_margins, 25, None)
        
        gen_dict = {'param': True, 'binning': False, 'fitted': False, 'parallel': False, 'vine_depth': 5}
        par_dict = {'param_families': ['gaussian']}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': False}
        bin_dict = {'n_bin': 5}
        
        tf_vine.fit(data.astype(np.float32), gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        # Generate samples
        tf_samples = vine_cop_par_sample(tf_vine, 1000)
        pred_corr = np.corrcoef(tf_samples, rowvar=False)
        
        # Extract fitted parameters
        fitted_params = []
        if hasattr(tf_vine, 'copulas') and tf_vine.copulas:
            for level, copulas in enumerate(tf_vine.copulas):
                level_params = []
                if isinstance(copulas, list):
                    for cop in copulas:
                        if hasattr(cop, 'theta'):
                            level_params.append(cop.theta)
                        else:
                            level_params.append(0.0)
                fitted_params.append(level_params)
        
        return {
            'success': True,
            'samples': tf_samples,
            'pred_corr': pred_corr,
            'fit_time': fit_time,
            'fitted_params': fitted_params
        }
        
    except Exception as e:
        print(f"TensorFlow {vine_type} failed: {e}")
        return {'success': False, 'error': str(e)}

def plot_correlation_comparison(true_corr, results_dict, scenario_title, save_path=None):
    """Create comprehensive correlation comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{scenario_title}\nCorrelation Matrix Comparison', fontsize=16, fontweight='bold')
    
    # True correlation matrix
    im1 = axes[0,0].imshow(true_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    axes[0,0].set_title('True Correlations', fontweight='bold')
    for i in range(5):
        for j in range(5):
            axes[0,0].text(j, i, f'{true_corr[i,j]:.2f}', ha='center', va='center', 
                          color='white' if abs(true_corr[i,j]) > 0.5 else 'black', fontweight='bold')
    axes[0,0].set_xticks(range(5))
    axes[0,0].set_yticks(range(5))
    axes[0,0].set_xticklabels([f'X{i+1}' for i in range(5)])
    axes[0,0].set_yticklabels([f'X{i+1}' for i in range(5)])
    
    # PyTorch results
    if 'pytorch_c' in results_dict and results_dict['pytorch_c']['success']:
        pt_c_corr = results_dict['pytorch_c']['pred_corr']
        im2 = axes[0,1].imshow(pt_c_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        axes[0,1].set_title('PyTorch C-vine\nPredicted Correlations', fontweight='bold')
        for i in range(5):
            for j in range(5):
                axes[0,1].text(j, i, f'{pt_c_corr[i,j]:.2f}', ha='center', va='center',
                              color='white' if abs(pt_c_corr[i,j]) > 0.5 else 'black', fontweight='bold')
        axes[0,1].set_xticks(range(5))
        axes[0,1].set_yticks(range(5))
        axes[0,1].set_xticklabels([f'X{i+1}' for i in range(5)])
        axes[0,1].set_yticklabels([f'X{i+1}' for i in range(5)])
    else:
        axes[0,1].text(0.5, 0.5, 'PyTorch C-vine\nFailed', ha='center', va='center', 
                       transform=axes[0,1].transAxes, fontsize=14, color='red')
        axes[0,1].set_xticks([])
        axes[0,1].set_yticks([])
    
    # TensorFlow results
    if 'tensorflow_c' in results_dict and results_dict['tensorflow_c']['success']:
        tf_c_corr = results_dict['tensorflow_c']['pred_corr']
        im3 = axes[0,2].imshow(tf_c_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        axes[0,2].set_title('TensorFlow C-vine\nPredicted Correlations', fontweight='bold')
        for i in range(5):
            for j in range(5):
                axes[0,2].text(j, i, f'{tf_c_corr[i,j]:.2f}', ha='center', va='center',
                              color='white' if abs(tf_c_corr[i,j]) > 0.5 else 'black', fontweight='bold')
        axes[0,2].set_xticks(range(5))
        axes[0,2].set_yticks(range(5))
        axes[0,2].set_xticklabels([f'X{i+1}' for i in range(5)])
        axes[0,2].set_yticklabels([f'X{i+1}' for i in range(5)])
    else:
        axes[0,2].text(0.5, 0.5, 'TensorFlow C-vine\nFailed', ha='center', va='center',
                       transform=axes[0,2].transAxes, fontsize=14, color='red')
        axes[0,2].set_xticks([])
        axes[0,2].set_yticks([])
    
    # Add colorbar for top row
    cbar1 = plt.colorbar(im1, ax=axes[0,:], shrink=0.8, aspect=20)
    cbar1.set_label('Correlation Coefficient', fontweight='bold')
    
    # D-vine results (bottom row)
    if 'pytorch_d' in results_dict and results_dict['pytorch_d']['success']:
        pt_d_corr = results_dict['pytorch_d']['pred_corr']
        im4 = axes[1,1].imshow(pt_d_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        axes[1,1].set_title('PyTorch D-vine\nPredicted Correlations', fontweight='bold')
        for i in range(5):
            for j in range(5):
                axes[1,1].text(j, i, f'{pt_d_corr[i,j]:.2f}', ha='center', va='center',
                              color='white' if abs(pt_d_corr[i,j]) > 0.5 else 'black', fontweight='bold')
        axes[1,1].set_xticks(range(5))
        axes[1,1].set_yticks(range(5))
        axes[1,1].set_xticklabels([f'X{i+1}' for i in range(5)])
        axes[1,1].set_yticklabels([f'X{i+1}' for i in range(5)])
    else:
        axes[1,1].text(0.5, 0.5, 'PyTorch D-vine\nFailed', ha='center', va='center',
                       transform=axes[1,1].transAxes, fontsize=14, color='red')
        axes[1,1].set_xticks([])
        axes[1,1].set_yticks([])
    
    if 'tensorflow_d' in results_dict and results_dict['tensorflow_d']['success']:
        tf_d_corr = results_dict['tensorflow_d']['pred_corr']
        im5 = axes[1,2].imshow(tf_d_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        axes[1,2].set_title('TensorFlow D-vine\nPredicted Correlations', fontweight='bold')
        for i in range(5):
            for j in range(5):
                axes[1,2].text(j, i, f'{tf_d_corr[i,j]:.2f}', ha='center', va='center',
                              color='white' if abs(tf_d_corr[i,j]) > 0.5 else 'black', fontweight='bold')
        axes[1,2].set_xticks(range(5))
        axes[1,2].set_yticks(range(5))
        axes[1,2].set_xticklabels([f'X{i+1}' for i in range(5)])
        axes[1,2].set_yticklabels([f'X{i+1}' for i in range(5)])
    else:
        axes[1,2].text(0.5, 0.5, 'TensorFlow D-vine\nFailed', ha='center', va='center',
                       transform=axes[1,2].transAxes, fontsize=14, color='red')
        axes[1,2].set_xticks([])
        axes[1,2].set_yticks([])
    
    # Empty the middle-left plot
    axes[1,0].axis('off')
    
    # Add colorbar for bottom row
    if 'pytorch_d' in results_dict and results_dict['pytorch_d']['success']:
        cbar2 = plt.colorbar(im4, ax=axes[1,1:], shrink=0.8, aspect=20)
        cbar2.set_label('Correlation Coefficient', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation comparison saved to {save_path}")
    
    return fig

def plot_scatter_comparison(data, results_dict, scenario_title, save_path=None):
    """Create scatter plot comparison of original data vs samples."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'{scenario_title}\nScatter Plot Comparison: X1 vs X2', fontsize=16, fontweight='bold')
    
    # Original data
    axes[0,0].scatter(data[:,0], data[:,1], alpha=0.6, s=20, color='black')
    axes[0,0].set_title('Original Data', fontweight='bold')
    axes[0,0].set_xlabel('X1')
    axes[0,0].set_ylabel('X2')
    r_orig = pearsonr(data[:,0], data[:,1])[0]
    axes[0,0].text(0.05, 0.95, f'r = {r_orig:.3f}', transform=axes[0,0].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # PyTorch C-vine
    if 'pytorch_c' in results_dict and results_dict['pytorch_c']['success']:
        samples = results_dict['pytorch_c']['samples']
        axes[0,1].scatter(samples[:,0], samples[:,1], alpha=0.6, s=20, color='blue')
        axes[0,1].set_title('PyTorch C-vine Samples', fontweight='bold')
        axes[0,1].set_xlabel('X1')
        axes[0,1].set_ylabel('X2')
        r_pt_c = pearsonr(samples[:,0], samples[:,1])[0]
        axes[0,1].text(0.05, 0.95, f'r = {r_pt_c:.3f}', transform=axes[0,1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axes[0,1].transAxes, 
                       fontsize=14, color='red')
    
    # TensorFlow C-vine
    if 'tensorflow_c' in results_dict and results_dict['tensorflow_c']['success']:
        samples = results_dict['tensorflow_c']['samples']
        axes[0,2].scatter(samples[:,0], samples[:,1], alpha=0.6, s=20, color='green')
        axes[0,2].set_title('TensorFlow C-vine Samples', fontweight='bold')
        axes[0,2].set_xlabel('X1')
        axes[0,2].set_ylabel('X2')
        r_tf_c = pearsonr(samples[:,0], samples[:,1])[0]
        axes[0,2].text(0.05, 0.95, f'r = {r_tf_c:.3f}', transform=axes[0,2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontweight='bold')
    else:
        axes[0,2].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axes[0,2].transAxes,
                       fontsize=14, color='red')
    
    # PyTorch D-vine
    if 'pytorch_d' in results_dict and results_dict['pytorch_d']['success']:
        samples = results_dict['pytorch_d']['samples']
        axes[0,3].scatter(samples[:,0], samples[:,1], alpha=0.6, s=20, color='orange')
        axes[0,3].set_title('PyTorch D-vine Samples', fontweight='bold')
        axes[0,3].set_xlabel('X1')
        axes[0,3].set_ylabel('X2')
        r_pt_d = pearsonr(samples[:,0], samples[:,1])[0]
        axes[0,3].text(0.05, 0.95, f'r = {r_pt_d:.3f}', transform=axes[0,3].transAxes,
                       bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.8), fontweight='bold')
    else:
        axes[0,3].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axes[0,3].transAxes,
                       fontsize=14, color='red')
    
    # TensorFlow D-vine
    if 'tensorflow_d' in results_dict and results_dict['tensorflow_d']['success']:
        samples = results_dict['tensorflow_d']['samples']
        axes[0,4].scatter(samples[:,0], samples[:,1], alpha=0.6, s=20, color='purple')
        axes[0,4].set_title('TensorFlow D-vine Samples', fontweight='bold')
        axes[0,4].set_xlabel('X1')
        axes[0,4].set_ylabel('X2')
        r_tf_d = pearsonr(samples[:,0], samples[:,1])[0]
        axes[0,4].text(0.05, 0.95, f'r = {r_tf_d:.3f}', transform=axes[0,4].transAxes,
                       bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8), fontweight='bold')
    else:
        axes[0,4].text(0.5, 0.5, 'Failed', ha='center', va='center', transform=axes[0,4].transAxes,
                       fontsize=14, color='red')
    
    # Second row: X3 vs X4
    axes[1,0].scatter(data[:,2], data[:,3], alpha=0.6, s=20, color='black')
    axes[1,0].set_title('Original Data (X3 vs X4)', fontweight='bold')
    axes[1,0].set_xlabel('X3')
    axes[1,0].set_ylabel('X4')
    r_orig_34 = pearsonr(data[:,2], data[:,3])[0]
    axes[1,0].text(0.05, 0.95, f'r = {r_orig_34:.3f}', transform=axes[1,0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontweight='bold')
    
    # Add other X3 vs X4 plots for each method
    methods = ['pytorch_c', 'tensorflow_c', 'pytorch_d', 'tensorflow_d']
    colors = ['blue', 'green', 'orange', 'purple']
    titles = ['PyTorch C-vine', 'TensorFlow C-vine', 'PyTorch D-vine', 'TensorFlow D-vine']
    
    for idx, (method, color, title) in enumerate(zip(methods, colors, titles)):
        if method in results_dict and results_dict[method]['success']:
            samples = results_dict[method]['samples']
            axes[1,idx+1].scatter(samples[:,2], samples[:,3], alpha=0.6, s=20, color=color)
            axes[1,idx+1].set_title(f'{title} (X3 vs X4)', fontweight='bold')
            axes[1,idx+1].set_xlabel('X3')
            axes[1,idx+1].set_ylabel('X4')
            r = pearsonr(samples[:,2], samples[:,3])[0]
            axes[1,idx+1].text(0.05, 0.95, f'r = {r:.3f}', transform=axes[1,idx+1].transAxes,
                               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3), fontweight='bold')
        else:
            axes[1,idx+1].text(0.5, 0.5, 'Failed', ha='center', va='center', 
                               transform=axes[1,idx+1].transAxes, fontsize=14, color='red')
    
    # Third row: Performance metrics
    for i in range(5):
        axes[2,i].axis('off')
    
    # Add performance summary in the bottom row
    axes[2,2].text(0.5, 0.8, 'Performance Summary', ha='center', va='top',
                   transform=axes[2,2].transAxes, fontsize=14, fontweight='bold')
    
    summary_text = ""
    for method, title in zip(methods, titles):
        if method in results_dict and results_dict[method]['success']:
            fit_time = results_dict[method]['fit_time']
            summary_text += f"{title}: {fit_time:.2f}s\n"
        else:
            summary_text += f"{title}: Failed\n"
    
    axes[2,2].text(0.5, 0.6, summary_text, ha='center', va='top',
                   transform=axes[2,2].transAxes, fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter comparison saved to {save_path}")
    
    return fig

def calculate_metrics(true_corr, pred_corr):
    """Calculate correlation prediction metrics."""
    if pred_corr is None:
        return {'mae': np.nan, 'rmse': np.nan, 'recovery': np.nan}
    
    # Extract upper triangular elements (excluding diagonal)
    mask = np.triu(np.ones_like(true_corr, dtype=bool), k=1)
    true_vals = true_corr[mask]
    pred_vals = pred_corr[mask]
    
    # Check for valid values
    valid_mask = np.isfinite(pred_vals)
    if not np.any(valid_mask):
        return {'mae': np.nan, 'rmse': np.nan, 'recovery': np.nan}
    
    true_valid = true_vals[valid_mask]
    pred_valid = pred_vals[valid_mask]
    
    mae = np.mean(np.abs(true_valid - pred_valid))
    rmse = np.sqrt(np.mean((true_valid - pred_valid)**2))
    recovery = np.corrcoef(true_valid, pred_valid)[0,1] if len(true_valid) > 1 else np.nan
    
    return {'mae': mae, 'rmse': rmse, 'recovery': recovery}

def plot_performance_metrics(scenarios_results, save_path=None):
    """Create performance comparison bar plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Performance Comparison Across Scenarios', fontsize=16, fontweight='bold')
    
    scenarios = list(scenarios_results.keys())
    methods = ['pytorch_c', 'tensorflow_c', 'pytorch_d', 'tensorflow_d']
    method_labels = ['PyTorch C-vine', 'TensorFlow C-vine', 'PyTorch D-vine', 'TensorFlow D-vine']
    colors = ['lightblue', 'lightgreen', 'orange', 'plum']
    
    # Collect metrics
    mae_data = {method: [] for method in methods}
    rmse_data = {method: [] for method in methods}
    recovery_data = {method: [] for method in methods}
    fit_time_data = {method: [] for method in methods}
    
    for scenario in scenarios:
        results = scenarios_results[scenario]
        true_corr = results['true_corr']
        
        for method in methods:
            if method in results and results[method]['success']:
                metrics = calculate_metrics(true_corr, results[method]['pred_corr'])
                mae_data[method].append(metrics['mae'])
                rmse_data[method].append(metrics['rmse'])
                recovery_data[method].append(metrics['recovery'])
                fit_time_data[method].append(results[method]['fit_time'])
            else:
                mae_data[method].append(np.nan)
                rmse_data[method].append(np.nan)
                recovery_data[method].append(np.nan)
                fit_time_data[method].append(np.nan)
    
    # Plot MAE
    x = np.arange(len(scenarios))
    width = 0.2
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        axes[0,0].bar(x + i*width, mae_data[method], width, label=label, color=color)
    axes[0,0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
    axes[0,0].set_xlabel('Scenario')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].set_xticks(x + width*1.5)
    axes[0,0].set_xticklabels(scenarios, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot RMSE
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        axes[0,1].bar(x + i*width, rmse_data[method], width, label=label, color=color)
    axes[0,1].set_title('Root Mean Square Error (RMSE)', fontweight='bold')
    axes[0,1].set_xlabel('Scenario')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].set_xticks(x + width*1.5)
    axes[0,1].set_xticklabels(scenarios, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot Recovery
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        axes[0,2].bar(x + i*width, recovery_data[method], width, label=label, color=color)
    axes[0,2].set_title('Recovery Correlation', fontweight='bold')
    axes[0,2].set_xlabel('Scenario')
    axes[0,2].set_ylabel('Recovery Correlation')
    axes[0,2].set_xticks(x + width*1.5)
    axes[0,2].set_xticklabels(scenarios, rotation=45)
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim(0, 1)
    
    # Plot Fit Time
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        axes[1,0].bar(x + i*width, fit_time_data[method], width, label=label, color=color)
    axes[1,0].set_title('Fitting Time', fontweight='bold')
    axes[1,0].set_xlabel('Scenario')
    axes[1,0].set_ylabel('Time (seconds)')
    axes[1,0].set_xticks(x + width*1.5)
    axes[1,0].set_xticklabels(scenarios, rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Summary table
    axes[1,1].axis('off')
    axes[1,2].axis('off')
    
    # Create summary table
    summary_data = []
    for method, label in zip(methods, method_labels):
        avg_mae = np.nanmean(mae_data[method])
        avg_recovery = np.nanmean(recovery_data[method])
        avg_time = np.nanmean(fit_time_data[method])
        summary_data.append([label, f'{avg_mae:.3f}', f'{avg_recovery:.3f}', f'{avg_time:.2f}s'])
    
    table = axes[1,1].table(cellText=summary_data,
                           colLabels=['Method', 'Avg MAE', 'Avg Recovery', 'Avg Time'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0.0, 0.3, 1.0, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1,1].set_title('Average Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics saved to {save_path}")
    
    return fig

def run_comprehensive_experiment():
    """Run comprehensive comparison experiment."""
    print("="*80)
    print("COMPREHENSIVE PYTORCH vs TENSORFLOW VINE COPULA COMPARISON")
    print("Testing C-vine and D-vine structures across multiple scenarios")
    print("="*80)
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    scenarios = ['mixed', 'chain', 'star']
    scenarios_results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*60}")
        
        # Generate test data
        data, true_corr, empirical_corr, title = create_test_data(scenario=scenario)
        print(f"Generated {title}")
        print(f"Data shape: {data.shape}")
        
        # Test all combinations
        results = {'true_corr': true_corr, 'empirical_corr': empirical_corr, 'title': title}
        
        print("\nTesting PyTorch C-vine...")
        results['pytorch_c'] = fit_pytorch_vine(data, 'c-vine')
        
        print("Testing TensorFlow C-vine...")
        results['tensorflow_c'] = fit_tensorflow_vine(data, 'c-vine')
        
        print("Testing PyTorch D-vine...")
        results['pytorch_d'] = fit_pytorch_vine(data, 'd-vine')
        
        print("Testing TensorFlow D-vine...")
        results['tensorflow_d'] = fit_tensorflow_vine(data, 'd-vine')
        
        scenarios_results[scenario] = results
        
        # Create plots for this scenario
        print(f"\nGenerating plots for {scenario} scenario...")
        
        # Correlation matrix comparison
        corr_fig = plot_correlation_comparison(
            true_corr, results, title,
            save_path=f'plots/{scenario}_correlation_comparison.png'
        )
        plt.close(corr_fig)
        
        # Scatter plot comparison
        scatter_fig = plot_scatter_comparison(
            data, results, title,
            save_path=f'plots/{scenario}_scatter_comparison.png'
        )
        plt.close(scatter_fig)
        
        # Print summary for this scenario
        print(f"\n--- {scenario.upper()} SCENARIO SUMMARY ---")
        for method_key, method_name in [('pytorch_c', 'PyTorch C-vine'), 
                                       ('tensorflow_c', 'TensorFlow C-vine'),
                                       ('pytorch_d', 'PyTorch D-vine'), 
                                       ('tensorflow_d', 'TensorFlow D-vine')]:
            if method_key in results and results[method_key]['success']:
                metrics = calculate_metrics(true_corr, results[method_key]['pred_corr'])
                fit_time = results[method_key]['fit_time']
                print(f"{method_name:20}: MAE={metrics['mae']:.3f}, Recovery={metrics['recovery']:.3f}, Time={fit_time:.2f}s")
            else:
                print(f"{method_name:20}: FAILED")
    
    # Create overall performance comparison
    print(f"\nGenerating overall performance comparison...")
    perf_fig = plot_performance_metrics(
        scenarios_results,
        save_path='plots/overall_performance_comparison.png'
    )
    plt.close(perf_fig)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print("Generated plots:")
    print("  - plots/mixed_correlation_comparison.png")
    print("  - plots/mixed_scatter_comparison.png")
    print("  - plots/chain_correlation_comparison.png")
    print("  - plots/chain_scatter_comparison.png")
    print("  - plots/star_correlation_comparison.png")
    print("  - plots/star_scatter_comparison.png")
    print("  - plots/overall_performance_comparison.png")
    print("\nCheck the plots/ directory for all visualizations!")
    
    return scenarios_results

if __name__ == "__main__":
    results = run_comprehensive_experiment() 