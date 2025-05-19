"""
Generate synthetic data with systematically controlled 2nd-, 3rd-, and 4th-order interactions,
fit a vine-copula, and compare how the tree-level log-likelihood increments line up with 
the 'ground truth' synergy level.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from classes.objects import vine_obj_bin, margin_obj
from pre_proc.define_copulas import define_copulas
from pre_proc.preparation import prep_cop
from sampling.vine_sample import vine_copula_sample
from info.info_estimation import vine_entropy


def gen_pairwise_data(N, d):
    """
    Generate data with only pairwise dependencies. 
    E.g. a multivariate Gaussian with a random correlation matrix that ensures no real triple synergy.
    """
    # create a random correlation matrix with no hidden synergy
    rnd = np.random.randn(d, d)
    sym = np.dot(rnd, rnd.T)
    # make pos def
    vals, vecs = np.linalg.eigh(sym)
    vals = np.abs(vals) + 0.1
    sym_pd = np.dot(vecs, np.diag(vals)).dot(vecs.T)
    # scale to correlation
    diag = np.sqrt(np.diag(sym_pd))
    corr = sym_pd / np.outer(diag, diag)
    # sample from a correlated Gaussian
    mean = np.zeros(d)
    data = np.random.multivariate_normal(mean, corr, size=N)
    # data shape = [N, d]
    return data

def gen_3way_data(N, d):
    """
    Generate data with a strong 3-way synergy among X1, X2, X3,
    beyond pairwise correlations. For simplicity, embed a 'trivariate mixing' 
    that can't be explained by only pairwise correlation, then fill the rest with random normal.
    """
    # approach: we can do something like 
    #   X1 ~ N(0,1), X2 ~ N(0,1), X3 = X1*X2 (some non-linear) + noise
    # plus the rest are just normal or correlated.
    # This is purely demonstration; many ways to do it.
    
    X1 = np.random.randn(N)
    X2 = np.random.randn(N)
    # let X3 = X1 * X2 + small noise => a "triple synergy" not captured by pairwise alone
    X3 = X1 * X2 + 0.1*np.random.randn(N)
    
    out = np.zeros((N, d))
    out[:, 0] = X1
    out[:, 1] = X2
    out[:, 2] = X3
    
    # fill the rest with random combos
    for col in range(3, d):
        out[:, col] = 0.5*X1 + 0.2*X2 + 0.3*np.random.randn(N)
    return out

def gen_4way_data(N, d):
    """
    Generate data with a 4-way synergy among X1, X2, X3, X4.
    For demonstration, let X4 = X1 * X2 * X3 + noise, ignoring pairwise sufficiency.
    """
    X1 = np.random.randn(N)
    X2 = np.random.randn(N)
    X3 = np.random.randn(N)
    # 4th var depends on the product => 4-way synergy 
    # (since any triple subset is insufficient to predict X4 well).
    X4 = (X1 * X2 * X3) + 0.05*np.random.randn(N)
    
    out = np.zeros((N, d))
    out[:, 0] = X1
    out[:, 1] = X2
    out[:, 2] = X3
    out[:, 3] = X4
    
    # fill the rest with random combos
    for col in range(4, d):
        out[:, col] = 0.1*X1 + 0.1*X2 + 0.1*X3 + 0.1*X4 + 0.2*np.random.randn(N)
    return out

def fit_and_entropy(data_np, vine_type, method, device, param=False):
    """
    Fit a vine to data_np (numpy array), return total entropy, 
    and possibly partial contributions from each tree level if you want.
    """
    data_t = torch.tensor(data_np, dtype=torch.float32, device=device)
    d = data_t.shape[1]
    # define margins
    margin_list = [ margin_obj('norm', [0.0, 1.0], True ) for _ in range(d)]
    # define the vine struct
    r_matrix, cop_vine, ind_vine, nodes, matrix_edges, _ = define_copulas(vine_type, method, binning=False, n_bin=1, dim=d)
    vine = vine_obj_bin(vine_type, "kercop", d, margin_list, 50, method, r_matrix)
    
    # prep
    data_prep = prep_cop(data_t, vine, 'rand')
    gen_dict = {'parallel': True, 'binning': False, 'param': param, 'vine_depth': d, 'fitted': False}
    par_dict = {'param_families': ["ind", "gaussian"]}  # or add more families
    npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
    bin_dict = {'n_bin': 1}
    
    vine.fit(data_prep, gen_dict, npc_dict, par_dict, bin_dict)
    # compute entropy
    info_dict = {'cases': 1000, 'iterations': 10, 'alpha': 0.05}
    H_est = vine_entropy(vine, info_dict)
    return vine, H_est

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 5000
    d = 6
    vine_type = 'c-vine'
    method = 'matrix'
    
    # 1) purely pairwise
    data_2w = gen_pairwise_data(N, d)
    vine2, H2 = fit_and_entropy(data_2w, vine_type, method, device, param=False)
    print("Pairwise-only synergy: fitted Entropy=", H2)
    
    # 2) 3way synergy
    data_3w = gen_3way_data(N, d)
    vine3, H3 = fit_and_entropy(data_3w, vine_type, method, device, param=False)
    print("3-way synergy: fitted Entropy=", H3)
    
    # 3) 4way synergy
    data_4w = gen_4way_data(N, d)
    vine4, H4 = fit_and_entropy(data_4w, vine_type, method, device, param=False)
    print("4-way synergy: fitted Entropy=", H4)
    
    # You can see how H2, H3, H4 differ. 
    # Optionally, also inspect partial log-likelihood contributions from each tree level 
    # if you have that recorded in vine. e.g. vine.copulas etc.
    
    # optional: visualize or compare
    print("Done main synergy experiment.")

if __name__=="__main__":
    main()