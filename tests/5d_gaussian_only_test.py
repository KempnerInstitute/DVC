#!/usr/bin/env python3
"""
5D Gaussian-only correlation test to validate TensorFlow alignment fixes.

This test forces the use of Gaussian copulas only to see if the poor
correlation recovery is due to inappropriate copula family selection.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, rankdata

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_5d_test_data(n_samples=600):
    """Create 5D Gaussian data with known correlation structure."""
    np.random.seed(42)  # Reproducible results
    
    # Define correlation matrix with realistic structure
    true_corr = np.array([
        [1.00, 0.70, 0.50, 0.30, 0.20],  # X1: strong with X2
        [0.70, 1.00, 0.60, 0.25, 0.15],  # X2: connected to X1,X3
        [0.50, 0.60, 1.00, 0.40, 0.35],  # X3: hub variable
        [0.30, 0.25, 0.40, 1.00, 0.65],  # X4: strong with X5
        [0.20, 0.15, 0.35, 0.65, 1.00]   # X5: strong with X4
    ])
    
    # Generate data
    data = multivariate_normal.rvs(mean=np.zeros(5), cov=true_corr, size=n_samples)
    empirical_corr = np.corrcoef(data, rowvar=False)
    
    print("=== 5D GAUSSIAN-ONLY TEST DATA ===")
    print(f"Samples: {n_samples}")
    print("\nTrue correlation matrix:")
    print(pd.DataFrame(true_corr, 
                      index=['X1','X2','X3','X4','X5'], 
                      columns=['X1','X2','X3','X4','X5']).round(3))
    
    print("\nEmpirical correlation matrix:")
    print(pd.DataFrame(empirical_corr, 
                      index=['X1','X2','X3','X4','X5'], 
                      columns=['X1','X2','X3','X4','X5']).round(3))
    
    return data, true_corr, empirical_corr

def test_gaussian_pairwise_fitting(data):
    """Test Gaussian copula fitting on all pairs."""
    print("\n=== GAUSSIAN COPULA PAIRWISE FITTING ===")
    
    from DVC.param_copula import parametric_fit
    
    n_samples = data.shape[0]
    results = []
    
    # Test all pairs with Gaussian copula only
    for i in range(5):
        for j in range(i+1, 5):
            # Extract pair
            pair_data = data[:, [i, j]]
            
            # Convert to uniform margins
            u_pair = np.zeros_like(pair_data)
            for k in range(2):
                ranks = rankdata(pair_data[:, k])
                u_pair[:, k] = ranks / (n_samples + 1)
            u_pair = u_pair.reshape(n_samples, 2, 1)
            
            # Test ONLY Gaussian copula (no independence or Clayton)
            families = ["gaussian"]
            aic_vals, theta_vals, logp_vals = parametric_fit(u_pair, families, n_cop=1)
            
            # Results
            gauss_aic = aic_vals[0][0]
            gauss_theta = theta_vals[0][0]
            gauss_logp = logp_vals[0][0]
            
            emp_corr = np.corrcoef(pair_data[:, 0], pair_data[:, 1])[0, 1]
            
            results.append({
                'pair': f'X{i+1}-X{j+1}',
                'emp_corr': emp_corr,
                'gauss_theta': gauss_theta,
                'gauss_aic': gauss_aic,
                'gauss_logp': gauss_logp
            })
            
            print(f"X{i+1}-X{j+1}: ρ_emp={emp_corr:.3f}, ρ_est={gauss_theta:.3f}, "
                  f"AIC={gauss_aic:.1f}, LogP={gauss_logp:.1f}")
    
    # Summary
    print(f"\nGaussian Parameter Estimation Summary:")
    theta_errors = [abs(r['emp_corr'] - r['gauss_theta']) for r in results]
    mean_theta_error = np.mean(theta_errors)
    print(f"Mean |ρ_empirical - ρ_estimated|: {mean_theta_error:.4f}")
    
    return results

def test_gaussian_vine_fitting(data):
    """Test PyTorch vine fitting with ONLY Gaussian copulas."""
    print("\n=== GAUSSIAN VINE FITTING TEST ===")
    
    try:
        from DVC.objects import vine_obj_bin, margin_obj
        
        # Create vine with ONLY Gaussian copulas
        margins = [margin_obj("norm", (0.0, 1.0)) for _ in range(5)]
        vine = vine_obj_bin(
            vine_family='c-vine',
            families=['gaussian'],  # ONLY Gaussian copulas
            vine_depth=5,
            margin=margins,
            knots=25
        )
        
        # Fit with ONLY Gaussian family
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
        par_dict = {'param_families': ['gaussian']}  # ONLY Gaussian
        bin_dict = {'n_bin': 5}
        
        print("Fitting 5D Gaussian vine copula...")
        import time
        start_time = time.time()
        
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        
        fit_time = time.time() - start_time
        print(f"✓ Gaussian vine fitting completed in {fit_time:.2f}s")
        
        # Extract fitted parameters for inspection
        print("\nFitted Gaussian parameters by level:")
        for level, copulas in enumerate(vine.copulas):
            print(f"Level {level}:")
            for i, cop in enumerate(copulas):
                if hasattr(cop, 'theta'):
                    print(f"  Edge {i}: ρ = {cop.theta:.4f}")
                else:
                    print(f"  Edge {i}: Non-parametric")
        
        # Generate samples
        print("\nGenerating samples for correlation prediction...")
        samples = vine.sample(1000)
        
        # Compute predicted correlations
        pred_corr = np.corrcoef(samples, rowvar=False)
        
        print("✓ Gaussian sampling and correlation prediction completed")
        
        return {
            'success': True,
            'fit_time': fit_time,
            'predicted_corr': pred_corr,
            'samples': samples,
            'vine': vine
        }
        
    except Exception as e:
        print(f"✗ Gaussian vine fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def numerical_correlation_comparison_robust(true_corr, pred_corr):
    """Detailed numerical comparison with robust NaN handling."""
    print("\n=== ROBUST NUMERICAL CORRELATION COMPARISON ===")
    
    # Extract pairwise correlations
    pairs = []
    true_vals = []
    pred_vals = []
    
    for i in range(5):
        for j in range(i+1, 5):
            pairs.append(f'X{i+1}-X{j+1}')
            true_vals.append(true_corr[i,j])
            pred_vals.append(pred_corr[i,j])
    
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    
    # Handle NaN values
    valid_mask = np.isfinite(pred_vals) & np.isfinite(true_vals)
    n_valid = np.sum(valid_mask)
    n_total = len(pairs)
    
    print(f"Valid correlation pairs: {n_valid}/{n_total}")
    
    if n_valid == 0:
        print("❌ No valid correlations found!")
        return None
    
    # Compute statistics only on valid pairs
    valid_true = true_vals[valid_mask]
    valid_pred = pred_vals[valid_mask]
    valid_errors = np.abs(valid_true - valid_pred)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Pair': np.array(pairs)[valid_mask],
        'True_ρ': valid_true,
        'Predicted_ρ': valid_pred,
        'Error': valid_errors,
        'Rel_Error_%': valid_errors / np.abs(valid_true) * 100
    })
    
    print("Valid Pairwise Correlation Comparison:")
    print(comparison_df.round(4))
    
    # Summary statistics
    mae = np.mean(valid_errors)
    rmse = np.sqrt(np.mean(valid_errors**2))
    max_error = np.max(valid_errors)
    mean_rel_error = np.mean(valid_errors / np.abs(valid_true) * 100)
    
    # Recovery correlation
    if n_valid > 1:
        recovery_corr = np.corrcoef(valid_true, valid_pred)[0,1]
    else:
        recovery_corr = np.nan
    
    print(f"\n--- GAUSSIAN VINE ACCURACY METRICS ---")
    print(f"Valid pairs: {n_valid}/{n_total}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Maximum Error:                 {max_error:.4f}")
    print(f"Mean Relative Error:           {mean_rel_error:.2f}%")
    if n_valid > 1:
        print(f"Recovery Correlation:          {recovery_corr:.4f}")
    
    # Performance assessment
    if n_valid < n_total:
        assessment = f"⚠ PARTIAL ({n_valid}/{n_total} valid)"
    elif not np.isfinite(recovery_corr):
        assessment = "❌ POOR (Invalid recovery)"
    elif recovery_corr > 0.95:
        assessment = "🎉 EXCELLENT"
    elif recovery_corr > 0.90:
        assessment = "✅ VERY GOOD"
    elif recovery_corr > 0.80:
        assessment = "✅ GOOD"
    elif recovery_corr > 0.60:
        assessment = "⚠ FAIR"
    else:
        assessment = "❌ POOR"
    
    print(f"Overall Assessment: {assessment}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'mean_rel_error': mean_rel_error,
        'recovery_correlation': recovery_corr,
        'valid_pairs': n_valid,
        'total_pairs': n_total,
        'comparison_table': comparison_df
    }

def run_gaussian_only_analysis():
    """Run Gaussian-only 5D analysis."""
    print("="*80)
    print("5D GAUSSIAN-ONLY CORRELATION PREDICTION ANALYSIS")
    print("Testing PyTorch with Gaussian Copulas Only")
    print("="*80)
    
    # Generate test data
    data, true_corr, empirical_corr = create_5d_test_data()
    
    # Test pairwise Gaussian fitting
    pairwise_results = test_gaussian_pairwise_fitting(data)
    
    # Test vine fitting with Gaussian copulas only
    vine_results = test_gaussian_vine_fitting(data)
    
    if vine_results['success']:
        # Numerical comparison
        numerical_results = numerical_correlation_comparison_robust(true_corr, vine_results['predicted_corr'])
        
        if numerical_results:
            # Final assessment
            print("\n" + "="*80)
            print("GAUSSIAN COPULA EFFECTIVENESS SUMMARY")
            print("="*80)
            
            # Parameter estimation accuracy
            theta_errors = [abs(r['emp_corr'] - r['gauss_theta']) for r in pairwise_results]
            param_accuracy = 1.0 - np.mean(theta_errors)
            print(f"✅ Parameter Estimation Accuracy: {param_accuracy:.1%}")
            
            # Vine fitting success
            print(f"✅ Vine Fitting: Successful in {vine_results['fit_time']:.2f}s")
            
            # Correlation prediction accuracy
            recovery_score = numerical_results['recovery_correlation']
            mae_score = numerical_results['mae']
            valid_ratio = numerical_results['valid_pairs'] / numerical_results['total_pairs']
            
            print(f"✅ Correlation Prediction:")
            print(f"   Valid pairs: {numerical_results['valid_pairs']}/{numerical_results['total_pairs']}")
            print(f"   Recovery Score: {recovery_score:.4f}")
            print(f"   Mean Absolute Error: {mae_score:.4f}")
            
            # Overall assessment
            overall_score = (param_accuracy + min(recovery_score if recovery_score > 0 else 0, 1.0) + valid_ratio) / 3
            
            print(f"\n📊 GAUSSIAN COPULA EFFECTIVENESS: {overall_score:.1%}")
            
            if overall_score >= 0.8:
                print("🎉 EXCELLENT: Gaussian copulas work very well!")
            elif overall_score >= 0.6:
                print("✅ GOOD: Gaussian copulas show strong performance")
            elif overall_score >= 0.4:
                print("⚠ FAIR: Gaussian copulas show moderate performance")
            else:
                print("❌ POOR: Issues remain even with correct copula family")
            
            return {
                'param_accuracy': param_accuracy,
                'vine_fit_time': vine_results['fit_time'],
                'correlation_recovery': recovery_score,
                'mae': mae_score,
                'valid_ratio': valid_ratio,
                'overall_score': overall_score
            }
    
    else:
        print(f"\n❌ Vine fitting failed: {vine_results['error']}")
        return None

if __name__ == "__main__":
    results = run_gaussian_only_analysis() 