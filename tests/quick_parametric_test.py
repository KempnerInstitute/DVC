#!/usr/bin/env python3
"""
Quick test comparing PyTorch vs TensorFlow parametric vine performance.
"""

import sys, os, numpy as np, time
from scipy.stats import multivariate_normal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'DVC_tensorflow'))

def test_scenario(scenario_name, true_corr, data):
    """Test both PyTorch and TensorFlow on a given scenario."""
    print(f"\n=== {scenario_name.upper()} SCENARIO ===")
    
    results = {}
    
    # PyTorch C-vine Parametric
    print("Testing PyTorch C-vine Parametric...")
    try:
        from DVC_pyolder.objects import vine_obj_bin, margin_obj
        
        start_time = time.time()
        margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(5)]
        vine = vine_obj_bin('c-vine', ['gaussian'], 5, margins, 25)
        
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        par_dict = {'param_families': ['gaussian']}
        npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
        bin_dict = {'n_bin': 5}
        
        vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        samples = vine.sample(1000)
        pred_corr = np.corrcoef(samples, rowvar=False)
        
        # Calculate metrics
        mask = np.triu(np.ones_like(true_corr, dtype=bool), k=1)
        true_vals = true_corr[mask]
        pred_vals = pred_corr[mask]
        mae = np.mean(np.abs(true_vals - pred_vals))
        recovery = np.corrcoef(true_vals, pred_vals)[0, 1]
        
        results['pytorch_c_param'] = {
            'mae': mae,
            'recovery': recovery,
            'fit_time': fit_time,
            'success': True
        }
        print(f"  ✓ MAE: {mae:.3f}, Recovery: {recovery:.3f}, Time: {fit_time:.2f}s")
        
    except Exception as e:
        results['pytorch_c_param'] = {'success': False, 'error': str(e)}
        print(f"  ✗ Failed: {e}")
    
    # TensorFlow C-vine Parametric
    print("Testing TensorFlow C-vine Parametric...")
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
        
        tf_vine = tf_vine_obj('c-vine', ['gaussian'], 5, tf_margins, 25, None)
        
        gen_dict = {'param': True, 'binning': False, 'fitted': False, 'parallel': False, 'vine_depth': 5}
        par_dict = {'param_families': ['gaussian']}
        npc_dict = {'opt_method': 'LL1', 'batch_paral': False}
        bin_dict = {'n_bin': 5}
        
        tf_vine.fit(data.astype(np.float32), gen_dict, npc_dict, par_dict, bin_dict)
        fit_time = time.time() - start_time
        
        tf_samples = vine_cop_par_sample(tf_vine, 1000)
        tf_pred_corr = np.corrcoef(tf_samples, rowvar=False)
        
        # Calculate metrics
        mask = np.triu(np.ones_like(true_corr, dtype=bool), k=1)
        true_vals = true_corr[mask]
        pred_vals = tf_pred_corr[mask]
        mae = np.mean(np.abs(true_vals - pred_vals))
        recovery = np.corrcoef(true_vals, pred_vals)[0, 1]
        
        results['tensorflow_c_param'] = {
            'mae': mae,
            'recovery': recovery,
            'fit_time': fit_time,
            'success': True
        }
        print(f"  ✓ MAE: {mae:.3f}, Recovery: {recovery:.3f}, Time: {fit_time:.2f}s")
        
    except Exception as e:
        results['tensorflow_c_param'] = {'success': False, 'error': str(e)}
        print(f"  ✗ Failed: {e}")
    
    return results

def main():
    """Run comparison tests on multiple scenarios."""
    print("PyTorch vs TensorFlow Parametric Vine Comparison")
    print("=" * 60)
    
    # Test scenarios
    scenarios = {
        'mixed': np.array([
            [1.00, 0.75, 0.40, 0.20, 0.10],
            [0.75, 1.00, 0.65, 0.30, 0.15],
            [0.40, 0.65, 1.00, 0.55, 0.35],
            [0.20, 0.30, 0.55, 1.00, 0.70],
            [0.10, 0.15, 0.35, 0.70, 1.00]
        ]),
        'star': np.array([
            [1.00, 0.70, 0.60, 0.50, 0.40],
            [0.70, 1.00, 0.30, 0.25, 0.20],
            [0.60, 0.30, 1.00, 0.25, 0.20],
            [0.50, 0.25, 0.25, 1.00, 0.20],
            [0.40, 0.20, 0.20, 0.20, 1.00]
        ])
    }
    
    all_results = {}
    
    for scenario_name, true_corr in scenarios.items():
        # Generate data
        np.random.seed(42)
        data = multivariate_normal.rvs(mean=np.zeros(5), cov=true_corr, size=800)
        
        # Test scenario
        results = test_scenario(scenario_name, true_corr, data)
        all_results[scenario_name] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*60}")
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name.upper()} Scenario:")
        for method, result in results.items():
            if result['success']:
                print(f"  {method:25}: MAE={result['mae']:.3f}, Recovery={result['recovery']:.3f}, Time={result['fit_time']:.2f}s")
            else:
                print(f"  {method:25}: FAILED")
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    for scenario_name, results in all_results.items():
        pt_result = results.get('pytorch_c_param', {})
        tf_result = results.get('tensorflow_c_param', {})
        
        if pt_result.get('success') and tf_result.get('success'):
            pt_mae = pt_result['mae']
            tf_mae = tf_result['mae']
            pt_recovery = pt_result['recovery']
            tf_recovery = tf_result['recovery']
            
            mae_improvement = ((tf_mae - pt_mae) / tf_mae) * 100 if tf_mae > 0 else 0
            recovery_improvement = ((pt_recovery - tf_recovery) / abs(tf_recovery)) * 100 if tf_recovery != 0 else 0
            
            print(f"\n{scenario_name.upper()}:")
            print(f"  MAE: PyTorch {pt_mae:.3f} vs TensorFlow {tf_mae:.3f} ({'improvement' if mae_improvement > 0 else 'decline'}: {abs(mae_improvement):.1f}%)")
            print(f"  Recovery: PyTorch {pt_recovery:.3f} vs TensorFlow {tf_recovery:.3f} ({'improvement' if recovery_improvement > 0 else 'decline'}: {abs(recovery_improvement):.1f}%)")
            
            if pt_mae < tf_mae * 1.5 and pt_recovery > tf_recovery * 0.5:
                print(f"  🎯 GOOD: PyTorch performance is competitive with TensorFlow!")
            else:
                print(f"  ⚠️ NEEDS WORK: Significant performance gap remains")

if __name__ == "__main__":
    main() 