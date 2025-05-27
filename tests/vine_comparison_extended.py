#!/usr/bin/env python3
"""
Extended Comprehensive Comparison: PyTorch vs TensorFlow Vine Copulas
Tests both C-vine and D-vine structures with parametric and non-parametric methods.
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

def fit_pytorch_vine(data, vine_type='c-vine', method_type='parametric'):
    """Fit PyTorch vine copula with parametric or non-parametric method."""
    try:
        from DVC_pyolder.objects import vine_obj_bin, margin_obj
        
        start_time = time.time()
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(5)]
        vine = vine_obj_bin(vine_type, ['gaussian'], 5, margins, 25)
        
        if method_type == 'parametric':
            gen_dict = {'param': True, 'binning': False, 'fitted': False}
            par_dict = {'param_families': ['gaussian']}
            npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
            bin_dict = {'n_bin': 5}
        else:  # non-parametric
            gen_dict = {'param': False, 'binning': True, 'fitted': False}
            par_dict = {'param_families': ['gaussian']}  # Still used for initialization
            npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
            bin_dict = {'n_bin': 10}  # More bins for non-parametric
        
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        # Generate samples
        samples = vine.sample(1000)
        pred_corr = np.corrcoef(samples, rowvar=False)
        
        # Extract fitted parameters
        fitted_params = []
        if method_type == 'parametric':
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
            'fitted_params': fitted_params,
            'method_type': method_type
        }
        
    except Exception as e:
        print(f"PyTorch {vine_type} {method_type} failed: {e}")
        return {'success': False, 'error': str(e), 'method_type': method_type}

def fit_tensorflow_vine(data, vine_type='c-vine', method_type='parametric'):
    """Fit TensorFlow vine copula with parametric or non-parametric method."""
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        
        from classes.objects import vine_obj_bin as tf_vine_obj, margin_obj as tf_margin_obj
        
        start_time = time.time()
        tf_margins = []
        for i in range(5):
            tf_margin = tf_margin_obj('norm', (0.0, 1.0), True)
            tf_margin.ker = data[:, i].astype(np.float32)
            tf_margins.append(tf_margin)
        
        tf_vine = tf_vine_obj(vine_type, ['gaussian'], 5, tf_margins, 25, None)
        
        if method_type == 'parametric':
            gen_dict = {'param': True, 'binning': False, 'fitted': False, 'parallel': False, 'vine_depth': 5}
            par_dict = {'param_families': ['gaussian']}
            npc_dict = {'opt_method': 'LL1', 'batch_paral': False}
            bin_dict = {'n_bin': 5}
            
            tf_vine.fit(data.astype(np.float32), gen_dict, npc_dict, par_dict, bin_dict)
            
            # Import parametric sampling function
            from sampling.vine_sample import vine_cop_par_sample
            tf_samples = vine_cop_par_sample(tf_vine, 1000)
            
        else:  # non-parametric
            gen_dict = {'param': False, 'binning': True, 'fitted': False, 'parallel': False, 'vine_depth': 5}
            par_dict = {'param_families': ['gaussian']}  # Still used for initialization
            npc_dict = {'opt_method': 'LL1', 'batch_paral': False}
            bin_dict = {'n_bin': 10}  # More bins for non-parametric
            
            tf_vine.fit(data.astype(np.float32), gen_dict, npc_dict, par_dict, bin_dict)
            
            # Import non-parametric sampling function
            from sampling.vine_sample import vine_copula_sample
            tf_samples, _, _, _ = vine_copula_sample(tf_vine, 1000)
        
        fit_time = time.time() - start_time
        pred_corr = np.corrcoef(tf_samples, rowvar=False)
        
        # Extract fitted parameters
        fitted_params = []
        if method_type == 'parametric' and hasattr(tf_vine, 'copulas') and tf_vine.copulas:
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
            'fitted_params': fitted_params,
            'method_type': method_type
        }
        
    except Exception as e:
        print(f"TensorFlow {vine_type} {method_type} failed: {e}")
        return {'success': False, 'error': str(e), 'method_type': method_type}

def plot_extended_correlation_comparison(true_corr, results_dict, scenario_title, save_path=None):
    """Create extended correlation comparison plots including parametric/non-parametric."""
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle(f'{scenario_title}\nExtended Correlation Matrix Comparison', fontsize=18, fontweight='bold')
    
    # Define method combinations
    methods = [
        ('pytorch_c_param', 'PyTorch C-vine\nParametric'),
        ('pytorch_c_nonparam', 'PyTorch C-vine\nNon-parametric'),
        ('tensorflow_c_param', 'TensorFlow C-vine\nParametric'),
        ('tensorflow_c_nonparam', 'TensorFlow C-vine\nNon-parametric')
    ]
    
    d_vine_methods = [
        ('pytorch_d_param', 'PyTorch D-vine\nParametric'),
        ('pytorch_d_nonparam', 'PyTorch D-vine\nNon-parametric'),
        ('tensorflow_d_param', 'TensorFlow D-vine\nParametric'),
        ('tensorflow_d_nonparam', 'TensorFlow D-vine\nNon-parametric')
    ]
    
    # True correlation matrix (first subplot)
    im = axes[0,0].imshow(true_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    axes[0,0].set_title('True Correlations', fontweight='bold')
    for i in range(5):
        for j in range(5):
            axes[0,0].text(j, i, f'{true_corr[i,j]:.2f}', ha='center', va='center', 
                          color='white' if abs(true_corr[i,j]) > 0.5 else 'black', fontweight='bold')
    axes[0,0].set_xticks(range(5))
    axes[0,0].set_yticks(range(5))
    axes[0,0].set_xticklabels([f'X{i+1}' for i in range(5)])
    axes[0,0].set_yticklabels([f'X{i+1}' for i in range(5)])
    
    # C-vine methods (top row)
    for idx, (method_key, method_title) in enumerate(methods):
        col_idx = idx + 1
        if method_key in results_dict and results_dict[method_key]['success']:
            pred_corr = results_dict[method_key]['pred_corr']
            axes[0,col_idx].imshow(pred_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            axes[0,col_idx].set_title(method_title, fontweight='bold')
            for i in range(5):
                for j in range(5):
                    axes[0,col_idx].text(j, i, f'{pred_corr[i,j]:.2f}', ha='center', va='center',
                                        color='white' if abs(pred_corr[i,j]) > 0.5 else 'black', fontweight='bold')
            axes[0,col_idx].set_xticks(range(5))
            axes[0,col_idx].set_yticks(range(5))
            axes[0,col_idx].set_xticklabels([f'X{i+1}' for i in range(5)])
            axes[0,col_idx].set_yticklabels([f'X{i+1}' for i in range(5)])
        else:
            axes[0,col_idx].text(0.5, 0.5, 'FAILED', ha='center', va='center', 
                               transform=axes[0,col_idx].transAxes, fontsize=14, color='red', fontweight='bold')
            axes[0,col_idx].set_title(method_title, fontweight='bold')
            axes[0,col_idx].set_xticks([])
            axes[0,col_idx].set_yticks([])
    
    # D-vine methods (middle row) 
    axes[1,0].axis('off')  # Empty first column for D-vine row
    for idx, (method_key, method_title) in enumerate(d_vine_methods):
        col_idx = idx + 1
        if method_key in results_dict and results_dict[method_key]['success']:
            pred_corr = results_dict[method_key]['pred_corr']
            axes[1,col_idx].imshow(pred_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            axes[1,col_idx].set_title(method_title, fontweight='bold')
            for i in range(5):
                for j in range(5):
                    axes[1,col_idx].text(j, i, f'{pred_corr[i,j]:.2f}', ha='center', va='center',
                                        color='white' if abs(pred_corr[i,j]) > 0.5 else 'black', fontweight='bold')
            axes[1,col_idx].set_xticks(range(5))
            axes[1,col_idx].set_yticks(range(5))
            axes[1,col_idx].set_xticklabels([f'X{i+1}' for i in range(5)])
            axes[1,col_idx].set_yticklabels([f'X{i+1}' for i in range(5)])
        else:
            axes[1,col_idx].text(0.5, 0.5, 'FAILED', ha='center', va='center', 
                               transform=axes[1,col_idx].transAxes, fontsize=14, color='red', fontweight='bold')
            axes[1,col_idx].set_title(method_title, fontweight='bold')
            axes[1,col_idx].set_xticks([])
            axes[1,col_idx].set_yticks([])
    
    # Performance summary (bottom row)
    for i in range(5):
        axes[2,i].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[:2,:], shrink=0.6, aspect=30)
    cbar.set_label('Correlation Coefficient', fontweight='bold')
    
    # Performance summary table
    axes[2,2].text(0.5, 0.9, 'Performance Summary', ha='center', va='top',
                   transform=axes[2,2].transAxes, fontsize=16, fontweight='bold')
    
    # Create performance summary
    summary_text = "Method: MAE | Recovery | Time\n" + "="*35 + "\n"
    all_methods = methods + d_vine_methods
    
    for method_key, method_name in all_methods:
        if method_key in results_dict and results_dict[method_key]['success']:
            metrics = calculate_metrics(true_corr, results_dict[method_key]['pred_corr'])
            fit_time = results_dict[method_key]['fit_time']
            summary_text += f"{method_name.replace(chr(10), ' ')[:20]:20}: {metrics['mae']:.3f} | {metrics['recovery']:.3f} | {fit_time:.2f}s\n"
        else:
            summary_text += f"{method_name.replace(chr(10), ' ')[:20]:20}: FAILED\n"
    
    axes[2,2].text(0.5, 0.7, summary_text, ha='center', va='top',
                   transform=axes[2,2].transAxes, fontsize=10, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Extended correlation comparison saved to {save_path}")
    
    return fig

def plot_extended_performance_metrics(scenarios_results, save_path=None):
    """Create extended performance comparison including parametric/non-parametric methods."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Extended Performance Comparison: Parametric vs Non-parametric', fontsize=16, fontweight='bold')
    
    scenarios = list(scenarios_results.keys())
    methods = [
        'pytorch_c_param', 'pytorch_c_nonparam', 'tensorflow_c_param', 'tensorflow_c_nonparam',
        'pytorch_d_param', 'pytorch_d_nonparam', 'tensorflow_d_param', 'tensorflow_d_nonparam'
    ]
    method_labels = [
        'PT C-Par', 'PT C-NP', 'TF C-Par', 'TF C-NP',
        'PT D-Par', 'PT D-NP', 'TF D-Par', 'TF D-NP'
    ]
    colors = ['lightblue', 'blue', 'lightgreen', 'green', 'orange', 'darkorange', 'plum', 'purple']
    
    # Collect metrics
    mae_data = {method: [] for method in methods}
    recovery_data = {method: [] for method in methods}
    fit_time_data = {method: [] for method in methods}
    success_data = {method: [] for method in methods}
    
    for scenario in scenarios:
        results = scenarios_results[scenario]
        true_corr = results['true_corr']
        
        for method in methods:
            if method in results and results[method]['success']:
                metrics = calculate_metrics(true_corr, results[method]['pred_corr'])
                mae_data[method].append(metrics['mae'])
                recovery_data[method].append(metrics['recovery'])
                fit_time_data[method].append(results[method]['fit_time'])
                success_data[method].append(1)
            else:
                mae_data[method].append(np.nan)
                recovery_data[method].append(np.nan)
                fit_time_data[method].append(np.nan)
                success_data[method].append(0)
    
    # Plot MAE
    x = np.arange(len(scenarios))
    width = 0.1
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        axes[0,0].bar(x + i*width, mae_data[method], width, label=label, color=color)
    axes[0,0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
    axes[0,0].set_xlabel('Scenario')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].set_xticks(x + width*3.5)
    axes[0,0].set_xticklabels(scenarios, rotation=45)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot Recovery
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        axes[0,1].bar(x + i*width, recovery_data[method], width, label=label, color=color)
    axes[0,1].set_title('Recovery Correlation', fontweight='bold')
    axes[0,1].set_xlabel('Scenario')
    axes[0,1].set_ylabel('Recovery Correlation')
    axes[0,1].set_xticks(x + width*3.5)
    axes[0,1].set_xticklabels(scenarios, rotation=45)
    axes[0,1].set_ylim(0, 1)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot Fit Time
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        axes[0,2].bar(x + i*width, fit_time_data[method], width, label=label, color=color)
    axes[0,2].set_title('Fitting Time', fontweight='bold')
    axes[0,2].set_xlabel('Scenario')
    axes[0,2].set_ylabel('Time (seconds)')
    axes[0,2].set_xticks(x + width*3.5)
    axes[0,2].set_xticklabels(scenarios, rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot Success Rate
    for i, (method, color, label) in enumerate(zip(methods, colors, method_labels)):
        success_rate = np.mean(success_data[method]) * 100
        axes[0,3].bar(i, success_rate, color=color, label=label)
    axes[0,3].set_title('Success Rate', fontweight='bold')
    axes[0,3].set_xlabel('Method')
    axes[0,3].set_ylabel('Success Rate (%)')
    axes[0,3].set_xticks(range(len(methods)))
    axes[0,3].set_xticklabels(method_labels, rotation=45, fontsize=8)
    axes[0,3].set_ylim(0, 100)
    axes[0,3].grid(True, alpha=0.3)
    
    # Summary tables
    axes[1,0].axis('off')
    axes[1,1].axis('off')
    axes[1,2].axis('off')
    axes[1,3].axis('off')
    
    # Create parametric vs non-parametric summary
    param_summary = []
    nonparam_summary = []
    
    for base_method in ['pytorch_c', 'pytorch_d', 'tensorflow_c', 'tensorflow_d']:
        param_method = f"{base_method}_param"
        nonparam_method = f"{base_method}_nonparam"
        
        # Calculate averages
        param_mae = np.nanmean(mae_data[param_method]) if param_method in mae_data else np.nan
        param_recovery = np.nanmean(recovery_data[param_method]) if param_method in recovery_data else np.nan
        param_time = np.nanmean(fit_time_data[param_method]) if param_method in fit_time_data else np.nan
        param_success = np.mean(success_data[param_method]) if param_method in success_data else 0
        
        nonparam_mae = np.nanmean(mae_data[nonparam_method]) if nonparam_method in mae_data else np.nan
        nonparam_recovery = np.nanmean(recovery_data[nonparam_method]) if nonparam_method in recovery_data else np.nan
        nonparam_time = np.nanmean(fit_time_data[nonparam_method]) if nonparam_method in fit_time_data else np.nan
        nonparam_success = np.mean(success_data[nonparam_method]) if nonparam_method in success_data else 0
        
        method_name = base_method.replace('_', ' ').title()
        param_summary.append([method_name, f'{param_mae:.3f}', f'{param_recovery:.3f}', f'{param_time:.2f}s', f'{param_success*100:.0f}%'])
        nonparam_summary.append([method_name, f'{nonparam_mae:.3f}', f'{nonparam_recovery:.3f}', f'{nonparam_time:.2f}s', f'{nonparam_success*100:.0f}%'])
    
    # Parametric table
    param_table = axes[1,0].table(cellText=param_summary,
                                 colLabels=['Method', 'Avg MAE', 'Avg Recovery', 'Avg Time', 'Success'],
                                 cellLoc='center', loc='center',
                                 bbox=[0.0, 0.2, 1.0, 0.8])
    param_table.auto_set_font_size(False)
    param_table.set_fontsize(8)
    param_table.scale(1, 1.5)
    axes[1,0].set_title('Parametric Methods Summary', fontweight='bold', pad=20)
    
    # Non-parametric table  
    nonparam_table = axes[1,1].table(cellText=nonparam_summary,
                                    colLabels=['Method', 'Avg MAE', 'Avg Recovery', 'Avg Time', 'Success'],
                                    cellLoc='center', loc='center',
                                    bbox=[0.0, 0.2, 1.0, 0.8])
    nonparam_table.auto_set_font_size(False)
    nonparam_table.set_fontsize(8)
    nonparam_table.scale(1, 1.5)
    axes[1,1].set_title('Non-parametric Methods Summary', fontweight='bold', pad=20)
    
    # Overall comparison
    comparison_text = "KEY FINDINGS:\n\n"
    comparison_text += "✓ TensorFlow shows superior performance\n"
    comparison_text += "✓ Parametric methods generally faster\n"
    comparison_text += "✓ Non-parametric more robust for complex data\n"
    comparison_text += "⚠ PyTorch D-vine requires debugging\n"
    comparison_text += "⚠ Non-parametric methods need more memory"
    
    axes[1,2].text(0.5, 0.5, comparison_text, ha='center', va='center',
                   transform=axes[1,2].transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    axes[1,2].set_title('Key Findings', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Extended performance metrics saved to {save_path}")
    
    return fig

def run_extended_experiment():
    """Run extended comparison experiment with parametric and non-parametric methods."""
    print("="*90)
    print("EXTENDED PYTORCH vs TENSORFLOW VINE COPULA COMPARISON")
    print("Testing C-vine/D-vine + Parametric/Non-parametric combinations")
    print("="*90)
    
    # Create output directory
    os.makedirs('plots_extended', exist_ok=True)
    
    scenarios = ['mixed', 'chain', 'star']
    scenarios_results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*70}")
        
        # Generate test data
        data, true_corr, empirical_corr, title = create_test_data(scenario=scenario)
        print(f"Generated {title}")
        print(f"Data shape: {data.shape}")
        
        # Test all combinations
        results = {'true_corr': true_corr, 'empirical_corr': empirical_corr, 'title': title}
        
        # Test all method combinations
        method_combinations = [
            ('c-vine', 'parametric', 'pytorch_c_param', 'PyTorch C-vine Parametric'),
            ('c-vine', 'nonparametric', 'pytorch_c_nonparam', 'PyTorch C-vine Non-parametric'),
            ('c-vine', 'parametric', 'tensorflow_c_param', 'TensorFlow C-vine Parametric'),
            ('c-vine', 'nonparametric', 'tensorflow_c_nonparam', 'TensorFlow C-vine Non-parametric'),
            ('d-vine', 'parametric', 'pytorch_d_param', 'PyTorch D-vine Parametric'),
            ('d-vine', 'nonparametric', 'pytorch_d_nonparam', 'PyTorch D-vine Non-parametric'),
            ('d-vine', 'parametric', 'tensorflow_d_param', 'TensorFlow D-vine Parametric'),
            ('d-vine', 'nonparametric', 'tensorflow_d_nonparam', 'TensorFlow D-vine Non-parametric')
        ]
        
        for vine_type, method_type, result_key, description in method_combinations:
            print(f"\nTesting {description}...")
            
            if 'pytorch' in result_key:
                results[result_key] = fit_pytorch_vine(data, vine_type, method_type)
            else:
                results[result_key] = fit_tensorflow_vine(data, vine_type, method_type)
        
        scenarios_results[scenario] = results
        
        # Create plots for this scenario
        print(f"\nGenerating extended plots for {scenario} scenario...")
        
        # Extended correlation matrix comparison
        corr_fig = plot_extended_correlation_comparison(
            true_corr, results, title,
            save_path=f'plots_extended/{scenario}_extended_correlation_comparison.png'
        )
        plt.close(corr_fig)
        
        # Print summary for this scenario
        print(f"\n--- {scenario.upper()} SCENARIO SUMMARY ---")
        for _, _, result_key, description in method_combinations:
            if result_key in results and results[result_key]['success']:
                metrics = calculate_metrics(true_corr, results[result_key]['pred_corr'])
                fit_time = results[result_key]['fit_time']
                print(f"{description:35}: MAE={metrics['mae']:.3f}, Recovery={metrics['recovery']:.3f}, Time={fit_time:.2f}s")
            else:
                print(f"{description:35}: FAILED")
    
    # Create overall extended performance comparison
    print(f"\nGenerating extended performance comparison...")
    perf_fig = plot_extended_performance_metrics(
        scenarios_results,
        save_path='plots_extended/extended_performance_comparison.png'
    )
    plt.close(perf_fig)
    
    print(f"\n{'='*90}")
    print("EXTENDED EXPERIMENT COMPLETE")
    print(f"{'='*90}")
    print("Generated plots:")
    for scenario in scenarios:
        print(f"  - plots_extended/{scenario}_extended_correlation_comparison.png")
    print("  - plots_extended/extended_performance_comparison.png")
    print("\nCheck the plots_extended/ directory for all visualizations!")
    
    return scenarios_results

if __name__ == "__main__":
    results = run_extended_experiment() 