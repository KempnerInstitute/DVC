#!/usr/bin/env python3
"""
Test script to verify that the parametric C-vine implementation
correctly preserves correlation structure in sampled data.

This tests the key improvement: that vine-sampled data has correlations
close to the true correlations, rather than near-zero correlations.
"""

import numpy as np
import torch

from DVC_pyolder.objects import vine_obj_bin, margin_obj
# Import the new parametric C-vine implementation
from DVC_pyolder.vine_model import fit_vine, evaluate_vine, sample_vine, logpdf_vine

def generate_gaussian_data(n_samples=2000, dim=5, corr=0.5, seed=42):
    """Generate multivariate Gaussian data with uniform correlation."""
    np.random.seed(seed)
    mean = np.zeros(dim)
    # uniform correlation for all pairs
    cov = (1-corr)*np.eye(dim) + corr*np.ones((dim,dim))
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    return data

def correlation_matrix(x):
    """Compute correlation matrix."""
    return np.corrcoef(x.T)

def main():
    """Main test function."""
    # hyper-params
    n_train=2000
    n_test=1000
    d=5
    true_corr = 0.5

    print("=== Parametric C-Vine Correlation Preservation Test ===\n")

    # Generate data
    data_train = generate_gaussian_data(n_train, d, true_corr, seed=42)
    data_test  = generate_gaussian_data(n_test,  d, true_corr, seed=99)
    
    # show correlation
    print("=== True correlation on training data ===")
    train_corr = correlation_matrix(data_train)
    print(np.round(train_corr, 3))
    
    # Build vine
    margins=[]
    for i in range(d):
        margins.append(margin_obj('norm',(0.,1.),True))
    
    vine = vine_obj_bin(
        vine_family="c-vine",
        families=["gaussian"]*(d-1),
        vine_depth=d,
        margin=margins,
        knots=30
    )
    vine.method=None

    # Fit
    gen_dict = {"param": True, "binning":False, "fitted":False}
    npc_dict={}
    par_dict={"param_families":["ind","gaussian","clayton"]}
    bin_dict={"n_bin":4}

    print(f"\n=== Fitting C-VINE with param=True ===")
    vine.fit(data_train, gen_dict, npc_dict, par_dict, bin_dict, cfg=None)
    print("=== Fit complete ===")

    # Check what copulas were fitted
    print(f"\n=== Fitted Copula Families ===")
    for level, cops in enumerate(vine.copulas):
        families = [cop.family for cop in cops]
        print(f"Level {level}: {families}")

    # Evaluate on test set
    test_t = torch.tensor(data_test, dtype=torch.float32)
    logp_test = vine.logpdf(test_t)
    mean_logp = torch.mean(logp_test).item()
    print(f"\nMean logpdf on test set: {mean_logp:.4f}")

    # Sample and compare correlation
    nsamp=2000
    print(f"\n=== Sampling {nsamp} points from fitted vine ===")
    x_samp = vine.sample(nsamp)
    corr_samp = correlation_matrix(x_samp)
    corr_test = correlation_matrix(data_test)

    print("\n=== Empirical correlation from vine-sampled data ===")
    print(np.round(corr_samp,3))
    print("=== Empirical correlation from true test data ===")
    print(np.round(corr_test,3))

    # Compute correlation preservation metrics
    # Compare off-diagonal elements (correlations)
    mask = ~np.eye(d, dtype=bool)  # Off-diagonal mask
    true_corrs = train_corr[mask]
    samp_corrs = corr_samp[mask]
    test_corrs = corr_test[mask]

    # Correlation preservation metrics
    mae_samp = np.mean(np.abs(true_corrs - samp_corrs))
    mae_test = np.mean(np.abs(true_corrs - test_corrs))
    
    print(f"\n=== Correlation Preservation Metrics ===")
    print(f"Target correlation: {true_corr:.3f}")
    print(f"Mean sampled correlation: {np.mean(samp_corrs):.3f}")
    print(f"Mean test correlation: {np.mean(test_corrs):.3f}")
    print(f"MAE (sampled vs target): {mae_samp:.4f}")
    print(f"MAE (test vs target): {mae_test:.4f}")

    # Success criteria
    success = mae_samp < 0.15  # Correlations should be within 0.15 of target
    correlation_preserved = np.mean(samp_corrs) > 0.3  # Should have meaningful correlation

    print(f"\n=== Test Results ===")
    print(f"Correlation preservation: {'PASS' if success else 'FAIL'}")
    print(f"Meaningful correlation: {'PASS' if correlation_preserved else 'FAIL'}")
    
    if success and correlation_preserved:
        print("✅ SUCCESS: Parametric C-vine correctly preserves correlation structure!")
    else:
        print("❌ FAILURE: Correlation structure not preserved")
        if not correlation_preserved:
            print("   Issue: Correlations are too weak (near independence)")
        if not success:
            print("   Issue: Correlations deviate too much from target")

    print("\nDONE.\n")

    return success and correlation_preserved

if __name__=="__main__":
    success = main()
    exit(0 if success else 1) 