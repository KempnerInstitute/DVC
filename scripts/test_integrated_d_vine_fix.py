#!/usr/bin/env python
"""
Test the integrated D-vine fix directly

This script creates a simple test to verify that the integrated D-vine fix
in vine_model.py correctly preserves correlations, particularly for non-adjacent
variables in higher dimensions.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

# Ensure the DVC package is in the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src'))

from DVC_pyolder.objects import vine_obj_bin, margin_obj

def direct_comparison_test(dim=6, n_samples=10000, rho=0.6):
    """
    Direct comparison of original vs. integrated D-vine implementation
    
    This test directly compares the correlation preservation of the integrated
    D-vine fix against the old implementation.
    
    Parameters
    ----------
    dim : int, optional
        Dimension of the data, by default 6
    n_samples : int, optional
        Number of samples, by default 10000
    rho : float, optional
        Target correlation, by default 0.6
    """
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"Testing {dim}D D-vine with rho={rho}")
    
    # Generate correlated data
    cov = np.full((dim, dim), rho)
    np.fill_diagonal(cov, 1.0)
    data = np.random.multivariate_normal(np.zeros(dim), cov, size=n_samples)
    true_corr = np.corrcoef(data, rowvar=False)
    
    # Create margins
    margins = []
    for i in range(dim):
        loc, scale = norm.fit(data[:, i])
        margins.append(margin_obj('norm', [loc, scale], True))
    
    # Create and fit D-vine
    vine = vine_obj_bin(
        vine_family='d-vine',
        families=['gaussian'],
        vine_depth=dim,
        margin=margins,
        knots=40,
        method='optimal'
    )
    
    # Prepare dicts for fitting
    gen_dict = {'param': True, 'binning': False, 'fitted': False, 'parallel': True, 'vine_depth': dim}
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    
    # Configure
    cfg = {
        'vine': {
            'knots': 40,
            'family': 'd-vine',
            'method': 'optimal'
        },
        'general': {
            'param': True, 
            'binning': False,
            'fitted': False
        },
        'optimizer': {
            'jit': True,
            'batch_edges': True,
            'batch_size': 5,
            'max_iter_phase1': 70,
            'lr_phase1': 0.10,
            'tol_phase1': 1e-5,
            'max_iter_phase2': 100,
            'lr_phase2': 0.03,
            'tol_phase2': 5e-5
        },
        'bandwidth': {'method': 'rule_of_thumb', 'knn_k': 10},
        'npc': {'opt_method': 'LL1', 'grad_precompute': True},
        'sampler': {'fast_parametric': True, 'fast_nonparam': True}
    }
    
    # Fit the vine
    print("Fitting D-vine model...")
    vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)
    
    # Sample using the integrated fix (current implementation)
    print("Generating samples with integrated fix...")
    samples = vine.sample(n_samples)
    sample_corr = np.corrcoef(samples, rowvar=False)
    
    # Calculate error metrics
    error = np.abs(sample_corr - true_corr)
    mean_error = np.mean(error)
    
    print(f"Mean correlation error: {mean_error:.4f}")
    
    # Calculate error by distance
    print("\nError by distance in D-vine chain:")
    print("Distance | Error")
    print("-" * 20)
    
    for dist in range(1, dim):
        # Calculate errors for pairs at each distance
        errors = []
        for i in range(dim - dist):
            j = i + dist
            err = abs(sample_corr[i, j] - true_corr[i, j])
            errors.append(err)
        
        # Calculate average error for this distance
        mean = np.mean(errors)
        print(f"{dist:8d} | {mean:.4f}")
    
    # Print out specific variable pair correlations
    print("\nCorrelations between specific variable pairs:")
    print("Pair     | True   | Sample | Error")
    print("-" * 40)
    
    for i in range(dim-1):
        for j in range(i+1, dim):
            print(f"({i},{j})    | {true_corr[i,j]:.4f} | {sample_corr[i,j]:.4f} | {abs(true_corr[i,j] - sample_corr[i,j]):.4f}")
    
    # Print vine structure
    print("\nD-vine structure:")
    for level, edges in enumerate(vine.ind_vine):
        print(f"Level {level}: {edges}")
    
    return {
        'true_corr': true_corr,
        'sample_corr': sample_corr,
        'error': error,
        'mean_error': mean_error
    }

if __name__ == "__main__":
    # Test several dimensions
    for dim in [4, 6, 8]:
        result = direct_comparison_test(dim=dim)
        print("\n" + "="*60 + "\n") 