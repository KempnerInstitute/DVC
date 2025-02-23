###############################################
# src/torch_vine/vine_model.py
###############################################

import torch
import numpy as np
from .objects import vine_obj_bin, copula_obj, cop_par_obj
from .transformation import Transform
from .utils_bandwidth import bandwidth_rule_of_thumb
from .utils_locallik import loclik_batch_eval
from .dataset_ops import kfold
from .param_copula import parametric_fit, copulainvccdf
from .vine_tree import parent_var, flip_check_all
from .utils_prob import kernel_cdf
import random
from scipy.stats import kendalltau

def fit_vine(vine: vine_obj_bin,
             x: np.ndarray,
             gen_dict: dict,
             npc_dict: dict,
             par_dict: dict,
             bin_dict: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)
    vine.param = gen_dict['param']
    vine.binning = gen_dict['binning']
    vine.fitted = gen_dict['fitted']
    vine.n_bin = bin_dict['n_bin'] if vine.binning else 1

    d = x.shape[1]

    if vine.param:
        families = par_dict['param_families']
        n_cop = d-1
        data_np = []
        for i in range(n_cop):
            pair = x_torch[:, [i, i+1]]
            data_np.append(pair.unsqueeze(2))
        data_cat = torch.cat(data_np, dim=2).cpu().numpy()  # [N,2,n_cop]
        aic2, theta_list, logp_list = parametric_fit(data_cat, families, n_cop)
        best_fams = []
        best_ths = []
        for i in range(n_cop):
            aic_i = aic2[i,:]
            idx_best = np.argmin(aic_i)
            fam_best = families[idx_best]
            param_best = theta_list[i][idx_best]
            best_fams.append(fam_best)
            best_ths.append(param_best)
        vine.copulas = []
        for i in range(n_cop):
            cpo = cop_par_obj(best_fams[i], best_ths[i])
            vine.copulas.append([cpo])
    else:
        n_cop = d-1
        data_np = []
        for i in range(n_cop):
            pair = x_torch[:, [i, i+1]]
            data_np.append(pair.unsqueeze(2))
        data_cat = torch.cat(data_np, dim=2)
        bw = bandwidth_rule_of_thumb(data_cat, 2, n_cop)
        cobj = copula_obj(bw)
        vine.copulas = [cobj]

def evaluate_vine(vine: vine_obj_bin, points: torch.Tensor):
    d = vine.n_cop
    device = points.device
    normal = torch.distributions.Normal(0.,1.)
    logf_marginal = torch.zeros(points.shape[0], points.shape[1], device=device)
    for i in range(d):
        z = points[:,i]
        logpdf = normal.log_prob(z)
        logf_marginal[:,i] = logpdf
    sum_log_marg = logf_marginal.sum(dim=1)
    if vine.param:
        logc_sum = torch.zeros(points.shape[0], device=device)
        n_cop = len(vine.copulas)
        for i in range(n_cop):
            # gather columns i, i+1
            if i+1< points.shape[1]:
                uv = points[:, [i, i+1]]
                # skip real PDF => just do 0
                logpdf_c = torch.zeros(points.shape[0], device=device)
                logc_sum += logpdf_c
        p = (sum_log_marg + logc_sum).exp()
        return p, p, sum_log_marg
    else:
        p = sum_log_marg.exp()
        return p, p, sum_log_marg

def sample_vine(vine: vine_obj_bin, nsamples: int):
    d = vine.n_cop
    if vine.param:
        sample_u = np.random.rand(nsamples, d)
        return sample_u
    else:
        sample_u = np.random.rand(nsamples, d)
        return sample_u

# Attach
vine_obj_bin.fit = fit_vine
vine_obj_bin.evaluation = evaluate_vine
vine_obj_bin.sample = sample_vine