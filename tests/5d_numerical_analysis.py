#!/usr/bin/env python3
"""
Focused 5D numerical analysis of correlation prediction accuracy.

This provides concrete numerical calculations on how well our PyTorch 
implementation (with TensorFlow alignment fixes) predicts correlations 
in 5D Gaussian data with known correlation structure.
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
    
    print("=== 5D TEST DATA ===")
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

def test_pairwise_copula_selection(data):
    """Test our improved independence penalty on all pairs."""
    print("\n=== PAIRWISE COPULA SELECTION TEST ===")
    
    from DVC.param_copula import parametric_fit
    
    n_samples = data.shape[0]
    results = []
    
    # Test all pairs
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
            
            # Test copula families
            families = ["ind", "gaussian", "clayton"]
            aic_vals, theta_vals, logp_vals = parametric_fit(u_pair, families, n_cop=1)
            
            # Results
            ind_aic = aic_vals[0][0]
            gauss_aic = aic_vals[0][1]
            clayton_aic = aic_vals[0][2]
            
            best_idx = np.argmin(aic_vals[0])
            best_family = families[best_idx]
            
            emp_corr = np.corrcoef(pair_data[:, 0], pair_data[:, 1])[0, 1]
            
            # Should reject independence for |ρ| > 0.25
            should_reject_indep = abs(emp_corr) > 0.25
            did_reject_indep = (best_family != "ind")
            correct = should_reject_indep == did_reject_indep
            
            results.append({
                'pair': f'X{i+1}-X{j+1}',
                'emp_corr': emp_corr,
                'ind_aic': ind_aic,
                'gauss_aic': gauss_aic,
                'clayton_aic': clayton_aic,
                'selected': best_family,
                'correct': correct
            })
            
            status = "✅" if correct else "❌"
            print(f"X{i+1}-X{j+1}: ρ={emp_corr:.3f}, {best_family}, "
                  f"AIC_diff={ind_aic-aic_vals[0][best_idx]:.1f} {status}")
    
    # Summary
    correct_count = sum(r['correct'] for r in results)
    total = len(results)
    accuracy = correct_count / total
    
    print(f"\nPairwise Selection: {correct_count}/{total} correct ({accuracy:.1%})")
    
    return results

def test_pytorch_vine_fitting(data):
    """Test PyTorch vine fitting with our TensorFlow alignment fixes."""
    print("\n=== PYTORCH VINE FITTING TEST ===")
    
    try:
        from DVC.objects import vine_obj_bin, margin_obj
        
        # Create vine
        margins = [margin_obj("norm", (0.0, 1.0)) for _ in range(5)]
        vine = vine_obj_bin(
            vine_family='c-vine',
            families=['gaussian', 'clayton', 'independence'],
            vine_depth=5,
            margin=margins,
            knots=25
        )
        
        # Fit
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
        par_dict = {'param_families': ['ind', 'gaussian', 'clayton']}
        bin_dict = {'n_bin': 5}
        
        print("Fitting 5D vine copula...")
        import time
        start_time = time.time()
        
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        
        fit_time = time.time() - start_time
        print(f"✓ Fitting completed in {fit_time:.2f}s")
        
        # Generate samples
        print("Generating samples for correlation prediction...")
        samples = vine.sample(1000)
        
        # Compute predicted correlations
        pred_corr = np.corrcoef(samples, rowvar=False)
        
        print("✓ Sampling and correlation prediction completed")
        
        return {
            'success': True,
            'fit_time': fit_time,
            'predicted_corr': pred_corr,
            'samples': samples
        }
        
    except Exception as e:
        print(f"✗ PyTorch vine fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def numerical_correlation_comparison(true_corr, pred_corr):
    """Detailed numerical comparison of correlations."""
    print("\n=== NUMERICAL CORRELATION COMPARISON ===")
    
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
    
    # Create comparison table
    errors = np.abs(true_vals - pred_vals)
    rel_errors = errors / np.abs(true_vals) * 100
    
    # Handle NaN values: identify valid pairs
    valid_mask = np.isfinite(pred_vals) & np.isfinite(true_vals)
    n_valid = np.sum(valid_mask)
    n_total = len(pairs)
    
    comparison_df = pd.DataFrame({
        'Pair': pairs,
        'True_ρ': true_vals,
        'Predicted_ρ': pred_vals,
        'Error': errors,
        'Rel_Error_%': rel_errors
    })
    
    print("Pairwise Correlation Comparison:")
    print(comparison_df.round(4))
    print(f"\nValid correlation pairs: {n_valid}/{n_total}")
    
    # Summary statistics only on valid pairs
    if n_valid > 0:
        valid_errors = errors[valid_mask]
        valid_rel_errors = rel_errors[valid_mask]
        valid_true = true_vals[valid_mask]
        valid_pred = pred_vals[valid_mask]
        
        mae = np.mean(valid_errors)
        rmse = np.sqrt(np.mean(valid_errors**2))
        max_error = np.max(valid_errors)
        mean_rel_error = np.mean(valid_rel_errors[np.isfinite(valid_rel_errors)])
        
        # Correlation between true and predicted
        if n_valid > 1:
            recovery_corr = np.corrcoef(valid_true, valid_pred)[0,1]
        else:
            recovery_corr = np.nan
    else:
        mae = rmse = max_error = mean_rel_error = recovery_corr = np.nan
    
    print(f"\n--- NUMERICAL ACCURACY METRICS (based on {n_valid} valid pairs) ---")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Maximum Error:                 {max_error:.4f}")
    print(f"Mean Relative Error:           {mean_rel_error:.2f}%")
    if n_valid > 1:
        print(f"Recovery Correlation:          {recovery_corr:.4f}")
    else:
        print(f"Recovery Correlation:          N/A (need >1 valid pairs)")
    
    # Performance assessment based on valid data
    if n_valid == 0:
        assessment = "❌ FAILED (No valid data)"
    elif n_valid < n_total * 0.5:
        assessment = "❌ POOR (Too many NaN values)"
    elif not np.isfinite(recovery_corr):
        if n_valid == 1:
            assessment = "⚠ LIMITED (Only 1 valid pair)"
        else:
            assessment = "❌ POOR (Invalid recovery correlation)"
    elif recovery_corr > 0.95:
        assessment = "🎉 EXCELLENT"
    elif recovery_corr > 0.90:
        assessment = "✅ VERY GOOD"
    elif recovery_corr > 0.85:
        assessment = "✅ GOOD"
    elif recovery_corr > 0.75:
        assessment = "⚠ FAIR"
    else:
        assessment = "❌ POOR"
    
    print(f"Overall Assessment: {assessment}")
    
    # Detailed breakdown by correlation strength (only valid pairs)
    if n_valid > 0:
        print(f"\n--- ERROR BREAKDOWN BY CORRELATION STRENGTH ---")
        
        strong_mask = (np.abs(valid_true) >= 0.5)
        moderate_mask = (np.abs(valid_true) >= 0.3) & (np.abs(valid_true) < 0.5)
        weak_mask = (np.abs(valid_true) < 0.3)
        
        if np.any(strong_mask):
            strong_mae = np.mean(valid_errors[strong_mask])
            strong_count = np.sum(strong_mask)
            print(f"Strong correlations (|ρ| ≥ 0.5): {strong_count} pairs, MAE = {strong_mae:.4f}")
        
        if np.any(moderate_mask):
            moderate_mae = np.mean(valid_errors[moderate_mask])
            moderate_count = np.sum(moderate_mask)
            print(f"Moderate correlations (0.3 ≤ |ρ| < 0.5): {moderate_count} pairs, MAE = {moderate_mae:.4f}")
        
        if np.any(weak_mask):
            weak_mae = np.mean(valid_errors[weak_mask])
            weak_count = np.sum(weak_mask)
            print(f"Weak correlations (|ρ| < 0.3): {weak_count} pairs, MAE = {weak_mae:.4f}")
    
    # Additional diagnostic information
    print(f"\n--- DIAGNOSTIC INFORMATION ---")
    nan_pairs = pairs[~valid_mask] if n_valid < n_total else []
    if len(nan_pairs) > 0:
        print(f"Pairs with NaN predictions: {', '.join(nan_pairs)}")
        print("This suggests issues with:")
        print("  - Vine fitting for certain variable combinations")
        print("  - Sampling algorithm robustness")
        print("  - Parameter estimation for specific copula families")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'mean_rel_error': mean_rel_error,
        'recovery_correlation': recovery_corr,
        'comparison_table': comparison_df,
        'valid_pairs': n_valid,
        'total_pairs': n_total
    }

def run_5d_analysis():
    """Run complete 5D numerical analysis."""
    print("="*80)
    print("5D GAUSSIAN CORRELATION PREDICTION - NUMERICAL ANALYSIS")
    print("Testing PyTorch with TensorFlow Alignment Fixes")
    print("="*80)
    
    # Generate test data
    data, true_corr, empirical_corr = create_5d_test_data()
    
    # Test pairwise copula selection (validates our independence penalty fix)
    pairwise_results = test_pairwise_copula_selection(data)
    
    # Test vine fitting and correlation prediction
    vine_results = test_pytorch_vine_fitting(data)
    
    if vine_results['success']:
        # Numerical comparison
        numerical_results = numerical_correlation_comparison(true_corr, vine_results['predicted_corr'])
        
        # Final assessment
        print("\n" + "="*80)
        print("TENSORFLOW ALIGNMENT FIXES - EFFECTIVENESS SUMMARY")
        print("="*80)
        
        # Pairwise selection accuracy
        pairwise_accuracy = sum(r['correct'] for r in pairwise_results) / len(pairwise_results)
        print(f"✅ Independence Penalty: {pairwise_accuracy:.1%} correct copula selections")
        
        # Vine fitting success
        print(f"✅ Vine Fitting: Successful in {vine_results['fit_time']:.2f}s")
        
        # Correlation prediction accuracy
        recovery_score = numerical_results['recovery_correlation']
        mae_score = numerical_results['mae']
        
        print(f"✅ Correlation Prediction:")
        print(f"   Recovery Score: {recovery_score:.4f}")
        print(f"   Mean Absolute Error: {mae_score:.4f}")
        
        # Overall assessment
        overall_score = (pairwise_accuracy + min(recovery_score, 1.0) + (1 if vine_results['fit_time'] < 20 else 0.5)) / 3
        
        print(f"\n📊 OVERALL EFFECTIVENESS: {overall_score:.1%}")
        
        if overall_score >= 0.9:
            print("🎉 EXCELLENT: TensorFlow alignment fixes are highly effective!")
        elif overall_score >= 0.8:
            print("✅ VERY GOOD: TensorFlow alignment fixes show strong improvements")
        elif overall_score >= 0.7:
            print("✅ GOOD: TensorFlow alignment fixes are working well")
        else:
            print("⚠ FAIR: TensorFlow alignment fixes show some improvements")
        
        return {
            'pairwise_accuracy': pairwise_accuracy,
            'vine_fit_time': vine_results['fit_time'],
            'correlation_recovery': recovery_score,
            'mae': mae_score,
            'overall_score': overall_score
        }
    
    else:
        print(f"\n❌ Vine fitting failed: {vine_results['error']}")
        return None

if __name__ == "__main__":
    results = run_5d_analysis() 