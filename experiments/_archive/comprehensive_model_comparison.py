"""
Comprehensive Model Comparison: PyTorch vs TensorFlow DVC

This script performs a detailed comparison between PyTorch and TensorFlow DVC implementations,
including correlation recovery, sampling quality, predictions, and entropy calculations.
"""

import numpy as np
import torch
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import rel_entr

sys.path.append('src')
sys.path.append('src/DVC_tensorflow')

# PyTorch imports
from DVC_pyolder import vine_obj_bin, margin_obj
from DVC_pyolder.vine_model import fit_vine, evaluate_vine, pdf_vine, cdf_vine
from DVC_pyolder.vine_model import conditional_mean_vine

# TensorFlow imports
from DVC_tensorflow.classes.objects import vine_obj_bin as tf_vine_obj_bin
from DVC_tensorflow.classes.objects import margin_obj as tf_margin_obj


def generate_ground_truth_data(n=1000, d=4, corr_type='decreasing', rho=0.6):
    """Generate ground truth data with known correlation structure"""
    np.random.seed(42)
    
    if corr_type == 'decreasing':
        # Decreasing correlation with distance
        corr = np.eye(d)
        for i in range(d):
            for j in range(i+1, d):
                corr[i, j] = corr[j, i] = rho ** abs(i-j)
    
    elif corr_type == 'block':
        # Block correlation structure
        corr = np.eye(d)
        # First block
        for i in range(min(2, d)):
            for j in range(i+1, min(2, d)):
                corr[i, j] = corr[j, i] = 0.8
        # Second block
        for i in range(2, d):
            for j in range(i+1, d):
                corr[i, j] = corr[j, i] = 0.7
                
    elif corr_type == 'hub':
        # Hub structure - first variable correlated with all others
        corr = np.eye(d)
        for i in range(1, d):
            corr[0, i] = corr[i, 0] = 0.7
            
    else:  # uniform
        corr = np.full((d, d), rho)
        np.fill_diagonal(corr, 1.0)
    
    # Generate data
    mean = np.zeros(d)
    data = np.random.multivariate_normal(mean, corr, n)
    
    # Calculate ground truth entropy (multivariate normal)
    ground_truth_entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(corr)[1]
    
    return data.astype(np.float32), corr, ground_truth_entropy


def fit_pytorch_vine(data, vine_type='d-vine', families=['gaussian', 'ind']):
    """Fit PyTorch vine model"""
    n, d = data.shape
    
    vine = vine_obj_bin(
        vine_family=vine_type,
        families=families,
        vine_depth=d,
        margin=[margin_obj('norm', [0, 1], True) for _ in range(d)],
        knots=50
    )
    
    gen_dict = {"parallel": False, "param": True, "binning": False, "fitted": False}
    par_dict = {"param_families": families}
    npc_dict = {}
    bin_dict = {"n_bin": 1}
    
    start_time = time.time()
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    fit_time = time.time() - start_time
    
    return vine, fit_time


def fit_tensorflow_vine(data, vine_type='d-vine', families=['gaussian', 'ind']):
    """Fit TensorFlow vine model"""
    n, d = data.shape
    
    # Create margins
    margins_tf = []
    for i in range(d):
        margin = tf_margin_obj('norm', [0.0, 1.0], True)
        margin.ker = data[:, i]
        margins_tf.append(margin)
    
    vine = tf_vine_obj_bin(
        vine_family=vine_type,
        families=families,
        vine_depth=d,
        margin=margins_tf,
        knots=50,
        method='matrix'
    )
    
    gen_dict_tf = {"parallel": False, "param": True, "binning": False, 
                   "fitted": False, "vine_depth": d}
    par_dict_tf = {"param_families": families}
    npc_dict_tf = {"opt_method": "local", "batch_paral": False}
    bin_dict_tf = {"n_bin": 1}
    
    start_time = time.time()
    vine.fit(data, gen_dict_tf, npc_dict_tf, par_dict_tf, bin_dict_tf)
    fit_time = time.time() - start_time
    
    return vine, fit_time


def evaluate_samples(vine, n_samples=1000, framework='pytorch'):
    """Generate samples from fitted vine"""
    if framework == 'pytorch':
        try:
            samples = vine.sample(n_samples)
            return samples
        except Exception as e:
            print(f"PyTorch sampling error: {e}")
            return None
    else:  # tensorflow
        try:
            # Try different methods for TensorFlow sampling
            if hasattr(vine, 'sampling'):
                samples = vine.sampling(n_samples)
            elif hasattr(vine, 'sim'):
                samples = vine.sim(n_samples)
            else:
                print("TensorFlow vine doesn't have a recognized sampling method")
                return None
            return samples
        except Exception as e:
            print(f"TensorFlow sampling error: {e}")
            return None


def calculate_entropy(vine, framework='pytorch', n_mc_samples=5000):
    """Calculate entropy using Monte Carlo integration"""
    # Generate samples for Monte Carlo integration
    if framework == 'pytorch':
        samples = evaluate_samples(vine, n_mc_samples, 'pytorch')
        if samples is None:
            return None
        
        # Calculate log-pdf for each sample
        try:
            log_pdfs = []
            for i in range(0, n_mc_samples, 100):  # Process in batches
                batch = samples[i:min(i+100, n_mc_samples)]
                # Convert numpy array to torch tensor if needed
                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch).float()
                pdf_vals, _, _ = evaluate_vine(vine, batch)
                log_pdfs.extend(np.log(pdf_vals.cpu().numpy() + 1e-10))
            
            # Monte Carlo estimate of entropy: -E[log p(x)]
            entropy = -np.mean(log_pdfs)
            return entropy
        except Exception as e:
            print(f"PyTorch entropy calculation error: {e}")
            return None
    else:
        # For TensorFlow, we need to implement entropy calculation
        return None


def evaluate_predictions(vine, test_data, framework='pytorch'):
    """Evaluate conditional predictions"""
    n, d = test_data.shape
    predictions = []
    
    if framework == 'pytorch':
        try:
            # Test prediction of last variable given others
            for i in range(min(100, n)):  # Test on subset
                fixed_vars = list(range(d-1))
                fixed_values = test_data[i, :-1]
                
                # Get conditional mean
                pred = conditional_mean_vine(vine, fixed_vars, fixed_values, d-1)
                predictions.append(pred)
            
            return np.array(predictions)
        except Exception as e:
            print(f"PyTorch prediction error: {e}")
            return None
    else:
        # TensorFlow prediction would go here
        return None


def compare_vine_structures(vine_pt, vine_tf):
    """Compare the vine structures between PyTorch and TensorFlow"""
    print("\n" + "="*80)
    print("VINE STRUCTURE COMPARISON")
    print("="*80)
    
    # Compare edges
    print("\nPyTorch vine edges:")
    if hasattr(vine_pt, 'ind_vine'):
        for level, edges in enumerate(vine_pt.ind_vine):
            print(f"  Level {level}: {edges}")
    
    print("\nTensorFlow vine edges:")
    # TensorFlow structure would be printed here
    
    # Compare copula parameters
    print("\nPyTorch copula parameters:")
    if hasattr(vine_pt, 'copulas'):
        for level, copulas in enumerate(vine_pt.copulas):
            for i, cop in enumerate(copulas):
                if hasattr(cop, 'family') and hasattr(cop, 'theta'):
                    print(f"  Level {level}, Edge {i}: {cop.family}, theta={cop.theta}")


def visualize_comprehensive_comparison(results):
    """Create comprehensive visualization of results"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Correlation Recovery Heatmaps
    ax1 = plt.subplot(3, 4, 1)
    sns.heatmap(results['ground_truth_corr'], annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, ax=ax1)
    ax1.set_title('Ground Truth Correlation', fontsize=12)
    
    ax2 = plt.subplot(3, 4, 2)
    if results.get('pytorch_sample_corr') is not None:
        sns.heatmap(results['pytorch_sample_corr'], annot=True, fmt='.2f', cmap='coolwarm',
                    vmin=-1, vmax=1, ax=ax2)
    ax2.set_title('PyTorch Sample Correlation', fontsize=12)
    
    ax3 = plt.subplot(3, 4, 3)
    if results.get('tensorflow_sample_corr') is not None:
        sns.heatmap(results['tensorflow_sample_corr'], annot=True, fmt='.2f', cmap='coolwarm',
                    vmin=-1, vmax=1, ax=ax3)
    ax3.set_title('TensorFlow Sample Correlation', fontsize=12)
    
    # 2. Correlation Error Heatmaps
    ax4 = plt.subplot(3, 4, 4)
    if results.get('pytorch_sample_corr') is not None:
        pytorch_error = results['pytorch_sample_corr'] - results['ground_truth_corr']
        sns.heatmap(pytorch_error, annot=True, fmt='.3f', cmap='RdBu_r',
                    vmin=-0.5, vmax=0.5, ax=ax4)
    ax4.set_title('PyTorch Correlation Error', fontsize=12)
    
    # 3. Sample Distributions
    if results.get('pytorch_samples') is not None:
        for i in range(min(4, results['pytorch_samples'].shape[1])):
            ax = plt.subplot(3, 4, 5 + i)
            
            # Plot ground truth
            ax.hist(results['ground_truth_data'][:, i], bins=30, alpha=0.5, 
                   density=True, label='Ground Truth')
            
            # Plot PyTorch samples
            ax.hist(results['pytorch_samples'][:, i], bins=30, alpha=0.5,
                   density=True, label='PyTorch')
            
            ax.set_title(f'Variable {i+1} Distribution', fontsize=10)
            ax.legend()
    
    # 4. Performance Metrics Bar Chart
    ax9 = plt.subplot(3, 2, 5)
    metrics = ['Fit Time', 'Corr MAE', 'Sample Time']
    pytorch_vals = [
        results.get('pytorch_fit_time', 0),
        results.get('pytorch_corr_mae', 0),
        results.get('pytorch_sample_time', 0)
    ]
    tensorflow_vals = [
        results.get('tensorflow_fit_time', 0),
        results.get('tensorflow_corr_mae', 0),
        results.get('tensorflow_sample_time', 0)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax9.bar(x - width/2, pytorch_vals, width, label='PyTorch', alpha=0.8)
    ax9.bar(x + width/2, tensorflow_vals, width, label='TensorFlow', alpha=0.8)
    ax9.set_xlabel('Metrics')
    ax9.set_ylabel('Values')
    ax9.set_title('Performance Comparison')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics)
    ax9.legend()
    
    # 5. Entropy Comparison
    ax10 = plt.subplot(3, 2, 6)
    entropy_labels = ['Ground Truth', 'PyTorch', 'TensorFlow']
    entropy_values = [
        results.get('ground_truth_entropy', 0),
        results.get('pytorch_entropy', 0),
        results.get('tensorflow_entropy', 0)
    ]
    
    bars = ax10.bar(entropy_labels, entropy_values, alpha=0.8)
    ax10.set_ylabel('Entropy')
    ax10.set_title('Entropy Comparison')
    
    # Add value labels on bars
    for bar, val in zip(bars, entropy_values):
        if val > 0:
            ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to comprehensive_model_comparison.png")


def run_comprehensive_comparison(vine_type='d-vine', d=4, n_train=1000, n_test=500):
    """Run comprehensive comparison between PyTorch and TensorFlow"""
    print(f"\n{'='*80}")
    print(f"TESTING {vine_type.upper()} WITH {d} DIMENSIONS")
    print(f"{'='*80}")
    
    # Generate ground truth data
    train_data, true_corr, ground_truth_entropy = generate_ground_truth_data(
        n=n_train, d=d, corr_type='decreasing', rho=0.6
    )
    test_data, _, _ = generate_ground_truth_data(
        n=n_test, d=d, corr_type='decreasing', rho=0.6
    )
    
    results = {
        'ground_truth_data': train_data,
        'ground_truth_corr': true_corr,
        'ground_truth_entropy': ground_truth_entropy,
        'vine_type': vine_type,
        'dimensions': d
    }
    
    print(f"\nGround truth entropy: {ground_truth_entropy:.4f}")
    
    # Fit PyTorch model
    print("\n--- PYTORCH ---")
    try:
        vine_pt, fit_time_pt = fit_pytorch_vine(train_data, vine_type)
        results['pytorch_fit_time'] = fit_time_pt
        print(f"Fit time: {fit_time_pt:.3f}s")
        
        # Generate samples
        start = time.time()
        samples_pt = evaluate_samples(vine_pt, n_test, 'pytorch')
        sample_time_pt = time.time() - start
        results['pytorch_sample_time'] = sample_time_pt
        
        if samples_pt is not None:
            results['pytorch_samples'] = samples_pt
            
            # Calculate correlations
            sample_corr_pt = np.corrcoef(samples_pt.T)
            results['pytorch_sample_corr'] = sample_corr_pt
            
            # Calculate MAE
            corr_mae_pt = np.mean(np.abs(sample_corr_pt - true_corr))
            results['pytorch_corr_mae'] = corr_mae_pt
            print(f"Correlation MAE: {corr_mae_pt:.4f}")
            
            # Calculate entropy
            entropy_pt = calculate_entropy(vine_pt, 'pytorch')
            if entropy_pt is not None:
                results['pytorch_entropy'] = entropy_pt
                print(f"Entropy: {entropy_pt:.4f}")
            
            # Test predictions
            predictions_pt = evaluate_predictions(vine_pt, test_data, 'pytorch')
            if predictions_pt is not None:
                results['pytorch_predictions'] = predictions_pt
                pred_mse = np.mean((predictions_pt - test_data[:len(predictions_pt), -1])**2)
                print(f"Prediction MSE: {pred_mse:.4f}")
        
        # Store vine for structure comparison
        results['pytorch_vine'] = vine_pt
        
    except Exception as e:
        print(f"PyTorch failed: {e}")
        results['pytorch_error'] = str(e)
    
    # Fit TensorFlow model
    print("\n--- TENSORFLOW ---")
    try:
        vine_tf, fit_time_tf = fit_tensorflow_vine(train_data, vine_type)
        results['tensorflow_fit_time'] = fit_time_tf
        print(f"Fit time: {fit_time_tf:.3f}s")
        
        # Generate samples
        start = time.time()
        samples_tf = evaluate_samples(vine_tf, n_test, 'tensorflow')
        sample_time_tf = time.time() - start
        results['tensorflow_sample_time'] = sample_time_tf
        
        if samples_tf is not None:
            results['tensorflow_samples'] = samples_tf
            
            # Calculate correlations
            sample_corr_tf = np.corrcoef(samples_tf.T)
            results['tensorflow_sample_corr'] = sample_corr_tf
            
            # Calculate MAE
            corr_mae_tf = np.mean(np.abs(sample_corr_tf - true_corr))
            results['tensorflow_corr_mae'] = corr_mae_tf
            print(f"Correlation MAE: {corr_mae_tf:.4f}")
            
            # Calculate entropy
            entropy_tf = calculate_entropy(vine_tf, 'tensorflow')
            if entropy_tf is not None:
                results['tensorflow_entropy'] = entropy_tf
                print(f"Entropy: {entropy_tf:.4f}")
        
        # Store vine for structure comparison
        results['tensorflow_vine'] = vine_tf
        
    except Exception as e:
        print(f"TensorFlow failed: {e}")
        results['tensorflow_error'] = str(e)
    
    # Compare structures if both successful
    if 'pytorch_vine' in results and 'tensorflow_vine' in results:
        compare_vine_structures(results['pytorch_vine'], results['tensorflow_vine'])
    
    return results


def diagnose_remaining_differences():
    """Diagnose remaining differences between PyTorch and TensorFlow"""
    print("\n" + "="*80)
    print("DIAGNOSING REMAINING DIFFERENCES")
    print("="*80)
    
    # Test simple 3D case for detailed analysis
    data, true_corr, _ = generate_ground_truth_data(n=500, d=3, rho=0.5)
    
    # Fit both models
    vine_pt, _ = fit_pytorch_vine(data, 'd-vine', families=['gaussian'])
    vine_tf, _ = fit_tensorflow_vine(data, 'd-vine', families=['gaussian'])
    
    print("\n1. THETA VALUES COMPARISON")
    print("-" * 40)
    
    # Compare theta values for first few samples
    if hasattr(vine_pt, 'theta'):
        theta_pt = vine_pt.theta[:5].cpu().numpy()
        print("PyTorch theta (first 5 samples):")
        for i in range(5):
            print(f"  Sample {i}: {theta_pt[i]}")
    
    print("\n2. COPULA PARAMETERS COMPARISON")
    print("-" * 40)
    
    # Compare fitted copula parameters
    if hasattr(vine_pt, 'copulas'):
        print("PyTorch copula parameters:")
        for level, copulas in enumerate(vine_pt.copulas):
            for i, cop in enumerate(copulas):
                if hasattr(cop, 'theta'):
                    print(f"  Level {level}, Edge {i}: {cop.theta}")
    
    print("\n3. H-FUNCTION OUTPUT COMPARISON")
    print("-" * 40)
    
    # Test h-function on same inputs
    u1, u2 = torch.tensor([0.3, 0.5, 0.7]), torch.tensor([0.4, 0.6, 0.8])
    if hasattr(vine_pt.copulas[0][0], 'theta'):
        from DVC_pyolder.vine_model import _h_function
        h_out = _h_function(u1, u2, vine_pt.copulas[0][0], vine_pt.grid_u, side="left")
        print(f"PyTorch h-function output: {h_out.numpy()}")


def main():
    """Run comprehensive model comparison"""
    print("="*80)
    print("COMPREHENSIVE DVC MODEL COMPARISON")
    print("="*80)
    
    all_results = {}
    
    # Test different vine types
    vine_types = ['d-vine', 'c-vine']  # Skip r-vine for now due to complexity
    dimensions = [3, 4, 5]
    
    for vine_type in vine_types:
        for d in dimensions:
            key = f"{vine_type}_{d}d"
            print(f"\n{'='*80}")
            print(f"Testing {vine_type} with {d} dimensions")
            
            results = run_comprehensive_comparison(
                vine_type=vine_type,
                d=d,
                n_train=800,
                n_test=500
            )
            
            all_results[key] = results
            
            # Create visualization for this configuration
            if results.get('pytorch_samples') is not None:
                visualize_comprehensive_comparison(results)
    
    # Run detailed diagnosis
    diagnose_remaining_differences()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nCorrelation MAE Comparison:")
    print(f"{'Configuration':<15} {'PyTorch':<10} {'TensorFlow':<10} {'Difference':<10}")
    print("-" * 50)
    
    for key, results in all_results.items():
        pt_mae = results.get('pytorch_corr_mae', np.nan)
        tf_mae = results.get('tensorflow_corr_mae', np.nan)
        diff = pt_mae - tf_mae if not np.isnan(pt_mae) and not np.isnan(tf_mae) else np.nan
        
        print(f"{key:<15} {pt_mae:<10.4f} {tf_mae:<10.4f} {diff:<10.4f}")
    
    return all_results


if __name__ == "__main__":
    results = main() 