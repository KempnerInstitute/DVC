# src/experiment.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pickle
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch

from pre_proc.define_copulas import define_copulas
from param.generate_rvine import generate_r_samples
from pre_proc.preparation import prep_cop
from sampling.vine_sample import vine_copula_sample
from utils.tensor_op import create_points, replace_nan_inf
from classes.objects import vine_obj_bin, margin_obj
from pred.prediction import predict_vine
from info.info_estimation import vine_entropy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1) Define vine for generating data
    cases = 1000
    vine_type = 'c-vine'  # can be 'c-vine', 'd-vine', or 'r-vine'
    method = 'matrix'
    binning = False
    n_bin = 3
    dim = 5

    r_matrix, cop_vine, ind_vine, nodes, matrix_edges, margin_vine = define_copulas(vine_type, method, binning, n_bin, dim)
    sample, v, v_flip, tau_corr, tau_bins = generate_r_samples(cases, r_matrix, ind_vine, nodes, margin_vine, cop_vine, n_bin, binning)
    print("Generated sample shape:", sample.shape)
    plt.figure()
    plt.plot(sample[:, 0], sample[:, 1], '.')
    plt.title("Generated Data from Synthetic Vine")
    plt.show()

    # 2) Optionally load from .mat or pickle (set flags accordingly)
    load_mat = False
    load_pickle = False
    if load_mat:
        mat_contents = sio.loadmat('stu_ex_01.mat')
        dat = mat_contents.get('x')
        sample = torch.tensor(dat, dtype=torch.float32)
    if load_pickle:
        with open("clay_20_ale", "rb") as f:
            dict_save = pickle.load(f)
            vine_copulas = dict_save["vine_copulas"]
            x = dict_save["data"]
            r_matrix = dict_save["r_matrix"]
            vine_depth = 20
            sample = torch.tensor(x, dtype=torch.float32)

    # 3) Define vine for fitting
    vine_depth = dim
    margin_list = [margin_obj('norm', [0.0, 1.0], True) for _ in range(vine_depth)]
    knots = 50
    vine = vine_obj_bin(vine_type, "kercop", vine_depth, margin_list, knots, method, r_matrix)
    x_data = sample.clone().detach().to(device=device, dtype=torch.float32)
    
    # 4) Preprocess data
    sort_n = 'rand'
    x_prep = prep_cop(x_data, vine, sort_n)
    print("Preprocessed data shape:", x_prep.shape)

    # 5) Fit the vine model
    gen_dict = {'parallel': True, 'binning': binning, 'param': False, 'vine_depth': vine_depth, 'fitted': False}
    par_dict = {'param_families': ["ind", "gaussian"]}
    npc_dict = {'opt_method': 'LL1', 'batch_paral': 3}
    bin_dict = {'n_bin': n_bin}
    vine.fit(x_prep, gen_dict, npc_dict, par_dict, bin_dict)

    # 6) Sample from the fitted vine
    samp_fit = vine_copula_sample(vine, 2000)
    print("Sampled from fitted vine shape:", samp_fit.shape)

    # 7) Evaluate the vine density
    exp_dim = 100
    dim_col = 0
    pts = create_points(x_prep, dim_col, exp_dim)
    p, p_cop, logf = vine.evaluation(pts)
    print("Evaluated density shapes:", p.shape, p_cop.shape)

    # 8) Predict using the vine model
    p_pred, y_ml, y_em = predict_vine(x_prep, vine, dim_col, exp_dim)
    y_em = replace_nan_inf(y_em)
    x_col = x_prep[:, dim_col].cpu().numpy()
    y_em_np = y_em.cpu().numpy()
    corr = np.corrcoef(x_col, y_em_np)[0, 1]
    print("Correlation (dim 0 vs y_em):", corr)
    plt.figure()
    plt.plot(x_col, y_ml.cpu().numpy(), 'r.', label='ML')
    plt.plot(x_col, y_em_np, 'b.', label='EM')
    plt.title(f"Correlation: {corr:.3f}")
    plt.legend()
    plt.show()

    # 9) Estimate the mutual information / entropy
    info_dict = {'cases': 1000, 'iterations': 10, 'alpha': 0.05}
    H = vine_entropy(vine, info_dict)
    print("Estimated vine entropy:", H)

    # Optionally, save the vine (e.g., via pickle)
    # with open("clay_20_ale", "wb") as f:
    #     dict_save = {"vine_copulas": vine.copulas, "r_matrix": vine.r_matrix, "data": x_prep.cpu().numpy()}
    #     pickle.dump(dict_save, f)

if __name__ == "__main__":
    main()