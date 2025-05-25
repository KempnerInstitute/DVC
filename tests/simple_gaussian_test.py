#!/usr/bin/env python3
import sys, os, numpy as np
from scipy.stats import multivariate_normal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Generate the same 5D data
np.random.seed(42)
true_corr = np.array([
    [1.00, 0.70, 0.50, 0.30, 0.20],
    [0.70, 1.00, 0.60, 0.25, 0.15],
    [0.50, 0.60, 1.00, 0.40, 0.35],
    [0.30, 0.25, 0.40, 1.00, 0.65],
    [0.20, 0.15, 0.35, 0.65, 1.00]
])
data = multivariate_normal.rvs(mean=np.zeros(5), cov=true_corr, size=600)

print('=== GAUSSIAN-ONLY VINE TEST ===')
from DVC.objects import vine_obj_bin, margin_obj

# Create vine with ONLY Gaussian copulas
margins = [margin_obj('norm', (0.0, 1.0)) for _ in range(5)]
vine = vine_obj_bin(
    vine_family='c-vine',
    families=['gaussian'],
    vine_depth=5,
    margin=margins,
    knots=25
)

# Fit with ONLY Gaussian family
gen_dict = {'param': True, 'binning': False, 'fitted': False}
par_dict = {'param_families': ['gaussian']}  # ONLY Gaussian
npc_dict = {'opt_method': 'LL1', 'grad_precompute': False}
bin_dict = {'n_bin': 5}

print('Fitting Gaussian-only vine...')
try:
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict)
    print('✓ Fitting completed')

    print('Generating samples...')
    samples = vine.sample(1000)
    pred_corr = np.corrcoef(samples, rowvar=False)
    print('✓ Sampling completed')

    # Extract pairwise correlations for comparison
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

    # Check for valid correlations
    valid_mask = np.isfinite(pred_vals)
    n_valid = np.sum(valid_mask)

    print(f'\\nGaussian-only results:')
    print(f'Valid correlations: {n_valid}/{len(pairs)}')

    if n_valid > 1:
        valid_true = true_vals[valid_mask]
        valid_pred = pred_vals[valid_mask]
        mae = np.mean(np.abs(valid_true - valid_pred))
        recovery = np.corrcoef(valid_true, valid_pred)[0,1]
        
        print(f'MAE: {mae:.4f}')
        print(f'Recovery correlation: {recovery:.4f}')
        
        # Show some examples
        print('\\nSample predictions:')
        for i in range(min(5, len(pairs))):
            if valid_mask[i]:
                print(f'  {pairs[i]}: True={true_vals[i]:.3f}, Pred={pred_vals[i]:.3f}, Error={abs(true_vals[i]-pred_vals[i]):.3f}')
        
        if recovery > 0.8:
            print('🎉 EXCELLENT: Gaussian copulas work well!')
        elif recovery > 0.5:
            print('✅ GOOD: Significant improvement with Gaussian copulas')
        elif recovery > 0:
            print('⚠ FAIR: Some improvement with Gaussian copulas')
        else:
            print('❌ Still having issues even with correct copula family')
    else:
        print('❌ Insufficient valid correlations')

except Exception as e:
    print(f'✗ Test failed: {e}')
    import traceback
    traceback.print_exc() 