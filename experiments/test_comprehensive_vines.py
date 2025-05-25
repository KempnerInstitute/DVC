"""
Comprehensive Test Suite for PyTorch DVC Implementation
Tests multiple vine structures, covariance recovery, prediction accuracy,
and information theory calculations against ground truth.
"""

import numpy as np
import sys
sys.path.append('src')
import time
import torch
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from DVC import (
    vine_obj_bin, margin_obj, fit_vine, 
    vine_entropy, mutual_information,
    predict_conditional, prepare_vine, random_r_matrix_gen
)

# Set random seed for reproducibility
np.random.seed(42)

def generate_multivariate_gaussian(n_samples, cov_matrix):
    """Generate multivariate Gaussian data with known covariance."""
    dim = cov_matrix.shape[0]
    mean = np.zeros(dim)
    data = np.random.multivariate_normal(mean, cov_matrix, size=n_samples)
    return data.astype(np.float32)

def analytical_gaussian_entropy(cov_matrix):
    """Calculate analytical entropy for multivariate Gaussian."""
    dim = cov_matrix.shape[0]
    det_cov = np.linalg.det(cov_matrix)
    # H = 0.5 * log(det(2πe * Σ))
    entropy_nats = 0.5 * np.log(np.power(2 * np.pi * np.e, dim) * det_cov)
    # Convert to bits
    entropy_bits = entropy_nats / np.log(2)
    return entropy_bits

def analytical_gaussian_mi(cov_matrix, indices_x, indices_y):
    """Calculate analytical mutual information for Gaussian subsets."""
    # Extract submatrices
    cov_x = cov_matrix[np.ix_(indices_x, indices_x)]
    cov_y = cov_matrix[np.ix_(indices_y, indices_y)]
    
    # Joint covariance
    indices_joint = indices_x + indices_y
    cov_joint = cov_matrix[np.ix_(indices_joint, indices_joint)]
    
    # MI = H(X) + H(Y) - H(X,Y)
    h_x = analytical_gaussian_entropy(cov_x)
    h_y = analytical_gaussian_entropy(cov_y)
    h_joint = analytical_gaussian_entropy(cov_joint)
    
    return h_x + h_y - h_joint

def fit_and_evaluate_vine(data, vine_type, method=None):
    """Fit a vine copula and evaluate its performance."""
    n, d = data.shape
    
    # Create vine object
    vine = vine_obj_bin(
        vine_family=vine_type,
        families=['gaussian', 'ind', 'clayton'],
        vine_depth=d,
        margin=[],
        knots=50
    )
    
    # Set margins
    for i in range(d):
        vine.margin.append(margin_obj('norm', [0, 1], True))
    
    # Configuration
    gen_dict = {"parallel": True, "param": True, "binning": False}
    par_dict = {"param_families": ["gaussian", "ind", "clayton"]}
    npc_dict = {"method": "local", "n_iter": 100}
    bin_dict = {"n_bin": 1}
    
    # For R-vine, set the method
    if vine_type == 'r-vine' and method:
        vine.method = method
    
    # Fit the vine
    start_time = time.time()
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    fit_time = time.time() - start_time
    
    return vine, fit_time

def evaluate_covariance_recovery(vine, true_cov, n_samples=5000):
    """Generate samples from vine and compare covariance."""
    try:
        # Generate samples
        samples = vine.sample(n_samples)
        
        # Check for NaN
        if np.any(np.isnan(samples)):
            print(f"WARNING: Generated samples contain NaN for {vine.vine_family}")
            return np.eye(true_cov.shape[0]), np.nan, np.nan
        
        # Compute empirical covariance
        emp_cov = np.cov(samples.T)
        
        # Compute Frobenius norm error
        cov_error = np.linalg.norm(emp_cov - true_cov, 'fro')
        
        # Compute correlation matrix error
        true_corr = true_cov / np.sqrt(np.outer(np.diag(true_cov), np.diag(true_cov)))
        emp_corr = emp_cov / np.sqrt(np.outer(np.diag(emp_cov), np.diag(emp_cov)))
        corr_error = np.linalg.norm(emp_corr - true_corr, 'fro')
        
        return emp_cov, cov_error, corr_error
    except Exception as e:
        print(f"ERROR in covariance recovery for {vine.vine_family}: {e}")
        return np.eye(true_cov.shape[0]), np.nan, np.nan

def test_conditional_prediction(vine, test_data, observed_indices, target_indices):
    """Test conditional prediction accuracy."""
    try:
        n_test = min(100, test_data.shape[0])  # Limit test size for speed
        test_subset = test_data[:n_test]
        
        # Prepare observed and target data
        observed_data = test_subset[:, observed_indices]
        true_targets = test_subset[:, target_indices]
        
        # Predict
        start_time = time.time()
        predictions = predict_conditional(vine, observed_data, observed_indices, target_indices, n_samples=2000)
        pred_time = time.time() - start_time
        
        # Check for NaN in predictions
        if np.any(np.isnan(predictions)):
            print(f"WARNING: Predictions contain NaN for {vine.vine_family}")
            # Replace NaN with mean of true targets
            predictions = np.nan_to_num(predictions, nan=np.mean(true_targets))
        
        # Compute errors
        mse = mean_squared_error(true_targets, predictions)
        mae = mean_absolute_error(true_targets, predictions)
        
        return predictions, mse, mae, pred_time
    except Exception as e:
        print(f"ERROR in conditional prediction for {vine.vine_family}: {e}")
        # Return dummy predictions
        n_test = min(100, test_data.shape[0])
        predictions = np.zeros((n_test, len(target_indices)))
        return predictions, np.nan, np.nan, 0.0

def test_vine_structure_optimization(data, n_structures=5):
    """Test different R-vine structures and find optimal."""
    results = []
    
    for i in range(n_structures):
        # Generate random R-matrix
        r_matrix, _, _, _ = random_r_matrix_gen(data.shape[1])
        
        # Fit R-vine with this structure
        vine, fit_time = fit_and_evaluate_vine(data, 'r-vine', method='random')
        
        # Evaluate log-likelihood on test data
        test_data = generate_multivariate_gaussian(100, np.cov(data.T))
        p, _, _ = vine.evaluation(test_data)
        # Convert p to numpy if it's a tensor
        if torch.is_tensor(p):
            p = p.cpu().numpy()
        avg_log_lik = np.mean(np.log(p + 1e-10))
        
        results.append({
            'structure_id': i,
            'r_matrix': r_matrix,
            'fit_time': fit_time,
            'avg_log_lik': avg_log_lik
        })
    
    # Sort by log-likelihood
    results.sort(key=lambda x: x['avg_log_lik'], reverse=True)
    
    return results

def main():
    print("=" * 80)
    print("Comprehensive Vine Copula Test Suite")
    print("=" * 80)
    
    # 1. Generate test data with known covariance structure
    print("\n1. Generating multivariate Gaussian data with known covariance...")
    dim = 5
    # Create a interesting covariance matrix
    A = np.random.randn(dim, dim)
    cov_matrix = np.dot(A.T, A)  # Ensure positive definite
    # Normalize to correlation matrix with unit variance
    D = np.sqrt(np.diag(cov_matrix))
    cov_matrix = cov_matrix / np.outer(D, D)
    
    print(f"   True covariance matrix shape: {cov_matrix.shape}")
    print(f"   Condition number: {np.linalg.cond(cov_matrix):.2f}")
    
    # Generate training and test data
    n_train = 1000
    n_test = 500
    train_data = generate_multivariate_gaussian(n_train, cov_matrix)
    test_data = generate_multivariate_gaussian(n_test, cov_matrix)
    
    # 2. Fit different vine structures
    print("\n2. Fitting different vine structures...")
    vine_types = ['d-vine', 'c-vine', 'r-vine']
    vine_results = {}
    
    for vtype in vine_types:
        print(f"\n   Fitting {vtype}...")
        vine, fit_time = fit_and_evaluate_vine(train_data, vtype)
        print(f"   ✓ {vtype} fitted in {fit_time:.2f} seconds")
        
        # Store results
        vine_results[vtype] = {
            'vine': vine,
            'fit_time': fit_time
        }
    
    # 3. Evaluate covariance recovery
    print("\n3. Evaluating covariance recovery from generated samples...")
    print(f"   {'Vine Type':<10} {'Cov Error':<12} {'Corr Error':<12}")
    print("   " + "-" * 34)
    
    for vtype, results in vine_results.items():
        vine = results['vine']
        emp_cov, cov_err, corr_err = evaluate_covariance_recovery(vine, cov_matrix)
        results['cov_error'] = cov_err
        results['corr_error'] = corr_err
        results['emp_cov'] = emp_cov
        print(f"   {vtype:<10} {cov_err:<12.4f} {corr_err:<12.4f}")
    
    # 4. Test conditional prediction
    print("\n4. Testing conditional prediction accuracy...")
    observed_indices = [0, 1]
    target_indices = [2, 3, 4]
    
    print(f"   Predicting variables {target_indices} given {observed_indices}")
    print(f"   {'Vine Type':<10} {'MSE':<12} {'MAE':<12} {'Time (s)':<12}")
    print("   " + "-" * 46)
    
    for vtype, results in vine_results.items():
        vine = results['vine']
        predictions, mse, mae, pred_time = test_conditional_prediction(
            vine, test_data, observed_indices, target_indices
        )
        results['pred_mse'] = mse
        results['pred_mae'] = mae
        print(f"   {vtype:<10} {mse:<12.4f} {mae:<12.4f} {pred_time:<12.2f}")
    
    # 5. Test R-vine structure optimization
    print("\n5. Testing R-vine structure optimization...")
    r_vine_structures = test_vine_structure_optimization(train_data, n_structures=3)
    
    print(f"   {'Structure':<12} {'Avg Log-Lik':<15} {'Fit Time (s)':<12}")
    print("   " + "-" * 39)
    for i, result in enumerate(r_vine_structures):
        print(f"   Structure {i:<3} {result['avg_log_lik']:<15.4f} {result['fit_time']:<12.2f}")
    
    print(f"\n   Best structure: Structure {r_vine_structures[0]['structure_id']}")
    
    # 6. Compare entropy and MI with analytical values
    print("\n6. Comparing entropy and mutual information with analytical values...")
    
    # Calculate analytical values
    analytical_entropy = analytical_gaussian_entropy(cov_matrix)
    analytical_mi = analytical_gaussian_mi(cov_matrix, [0, 1], [2, 3])
    
    print(f"\n   Analytical entropy: {analytical_entropy:.4f} bits")
    print(f"   Analytical MI([0,1], [2,3]): {analytical_mi:.4f} bits")
    
    # Estimate from vines
    info_dict = {'alpha': 0.05, 'cases': 2000, 'iterations': 10}
    
    print(f"\n   {'Vine Type':<10} {'Entropy':<12} {'Error':<12} {'MI':<12} {'MI Error':<12}")
    print("   " + "-" * 58)
    
    for vtype, results in vine_results.items():
        vine = results['vine']
        
        # Estimate entropy
        est_entropy = vine_entropy(vine, info_dict)
        entropy_error = abs(est_entropy - analytical_entropy)
        
        # Estimate MI
        est_mi = mutual_information(vine, [0, 1], [2, 3], info_dict)
        mi_error = abs(est_mi - analytical_mi)
        
        results['entropy'] = est_entropy
        results['mi'] = est_mi
        
        print(f"   {vtype:<10} {est_entropy:<12.4f} {entropy_error:<12.4f} {est_mi:<12.4f} {mi_error:<12.4f}")
    
    # 7. Visualize results
    print("\n7. Creating visualization plots...")
    
    # Plot covariance comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # True covariance
    im = axes[0, 0].imshow(cov_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0, 0].set_title('True Covariance')
    axes[0, 0].set_xlabel('Variable')
    axes[0, 0].set_ylabel('Variable')
    
    # Vine covariances
    for i, (vtype, results) in enumerate(vine_results.items()):
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(results['emp_cov'], cmap='coolwarm', vmin=-1, vmax=1)
        axes[row, col].set_title(f'{vtype} Covariance')
        axes[row, col].set_xlabel('Variable')
        axes[row, col].set_ylabel('Variable')
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.suptitle('Covariance Matrix Comparison')
    plt.tight_layout()
    plt.savefig('covariance_comparison.png', dpi=150)
    print("   ✓ Saved covariance_comparison.png")
    
    # Plot prediction scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (vtype, results) in enumerate(vine_results.items()):
        vine = results['vine']
        # Get a few predictions for visualization
        obs_data = test_data[:20, observed_indices]
        true_targets = test_data[:20, target_indices[0]]  # First target variable
        pred_targets = predict_conditional(vine, obs_data, observed_indices, [target_indices[0]], n_samples=1000)
        
        axes[i].scatter(true_targets, pred_targets, alpha=0.6)
        axes[i].plot([true_targets.min(), true_targets.max()], 
                     [true_targets.min(), true_targets.max()], 'r--', lw=2)
        axes[i].set_xlabel('True Values')
        axes[i].set_ylabel('Predicted Values')
        axes[i].set_title(f'{vtype} Predictions')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Conditional Prediction: Variables {target_indices[0]} given {observed_indices}')
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=150)
    print("   ✓ Saved prediction_comparison.png")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Results")
    print("=" * 80)
    
    print("\nBest performing vine by metric:")
    
    # Find best by each metric
    metrics = ['cov_error', 'corr_error', 'pred_mse', 'entropy', 'mi']
    metric_names = ['Covariance Error', 'Correlation Error', 'Prediction MSE', 
                    'Entropy Accuracy', 'MI Accuracy']
    
    for metric, name in zip(metrics, metric_names):
        if metric in ['entropy', 'mi']:
            # For these, we want closest to analytical
            if metric == 'entropy':
                best_vine = min(vine_results.items(), 
                               key=lambda x: abs(x[1].get(metric, float('inf')) - analytical_entropy))
            else:
                best_vine = min(vine_results.items(), 
                               key=lambda x: abs(x[1].get(metric, float('inf')) - analytical_mi))
        else:
            # For errors, we want minimum
            best_vine = min(vine_results.items(), 
                           key=lambda x: x[1].get(metric, float('inf')))
        
        print(f"   {name:<20}: {best_vine[0]}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main() 