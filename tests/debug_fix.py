#!/usr/bin/env python3

import numpy as np
import torch

from DVC_pyolder.objects import margin_obj, vine_obj_bin
from DVC_pyolder.config import DEFAULT_CFG
from DVC_pyolder.vine_model import fit_vine, evaluate_vine, sample_vine, logpdf_vine

def generate_gaussian_data(n_samples=1000, dim=5, correlation=0.5, seed=42):
    np.random.seed(seed)
    mean = np.zeros(dim)
    # build a correlation matrix with uniform off-diagonal = correlation
    cov = (1-correlation)*np.eye(dim) + correlation*np.ones((dim,dim))
    data = np.random.multivariate_normal(mean, cov, size=n_samples)
    return data

def correlation_matrix(x):
    return np.corrcoef(x.T)

def main():
    n_samples_train=2000
    n_samples_test=1000
    d=5
    correlation=0.5

    # Generate data
    data_train = generate_gaussian_data(n_samples_train,d,correlation)
    data_test  = generate_gaussian_data(n_samples_test,d,correlation,seed=99)

    print("=== True correlation on training data ===")
    print(correlation_matrix(data_train))

    # Setup for c-vine param
    param_vine = True
    vine_family='c-vine'
    fit_config = {
        "optimizer":{
            "batch_edges":True,
            "batch_size":5,
            "max_iter_phase1":70,
            "lr_phase1":0.1,
            "tol_phase1":1e-5,
            "max_iter_phase2":100,
            "lr_phase2":0.03,
            "tol_phase2":5e-5,
            "jit":False,
            "max_edges_per_batch":1,
        },
        "bandwidth":{
            "method":"rule_of_thumb",
            "knn_k":10,
        },
        "npc":{
            "opt_method":"LL1",
            "grad_precompute":False
        },
        "sampler":{
            "fast_parametric":True,
            "fast_nonparam":True,
            "noise_scale":0.0
        }
    }

    # Make margins
    margin_list = []
    for i in range(d):
        margin_list.append(margin_obj('norm',(0.0,1.0),True))

    # Build vine
    vine = vine_obj_bin(
        vine_family=vine_family,
        families=["gaussian"]*(d-1),
        vine_depth=d,
        margin=margin_list,
        knots=30
    )
    vine.method = None  # not used for c-vine

    gen_dict={
        "param":param_vine,
        "binning":False,
        "fitted":False
    }
    npc_dict={}
    par_dict={
        "param_families":["ind","gaussian","clayton"]  # or whichever param families you want
    }
    bin_dict={"n_bin":4}

    print(f"\n=== Fitting {vine_family.upper()} with param={param_vine} ===")
    vine.fit(data_train, gen_dict, npc_dict, par_dict, bin_dict, cfg=fit_config)
    print("=== Fit complete ===\n")

    # Evaluate on test
    test_t = torch.tensor(data_test,dtype=torch.float32)
    logp_test = vine.logpdf(test_t)
    avg_logp = torch.mean(logp_test).item()
    print(f"Mean logpdf on test set: {avg_logp:.4f}")

    # Sample from vine, check correlation
    nsample_vine=2000
    sample_from_vine = vine.sample(nsample_vine)
    corr_vine = correlation_matrix(sample_from_vine)
    corr_data = correlation_matrix(data_test)

    print("\n=== Empirical correlation from vine-sampled data ===")
    print(corr_vine)
    print("=== Empirical correlation from true test data ===")
    print(corr_data)

    print("\n[DEBUG SUMMARY]")
    print(f"  Param vine? {param_vine}")
    print(f"  Vine type: {vine_family}")
    print(f"  Mean test logpdf   : {avg_logp:.5f}")
    print("\nDONE.\n")

if __name__=="__main__":
    main()