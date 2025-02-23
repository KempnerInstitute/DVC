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
from .param_copula import parametric_fit, copulainvccdf, copulapdf
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
    """
    Fit the vine on data x, either param or nonparam, binning or not,
    storing the final 'vine.copulas'.
    
    Args:
      vine: vine_obj_bin to store the result in
      x: shape [N, d]
      gen_dict: general flags => { 'param': bool, 'binning': bool, 'fitted': bool, ... }
      npc_dict: nonparam config => e.g. { 'opt_method':..., 'batch_paral':... }
      par_dict: param config => e.g. { 'param_families': [...] }
      bin_dict: bin config => e.g. { 'n_bin': ... }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.tensor(x, dtype=torch.float32, device=device)

    # parse flags
    vine.param = gen_dict['param']
    vine.binning = gen_dict['binning']
    vine.fitted = gen_dict['fitted']
    vine.n_bin = bin_dict['n_bin'] if vine.binning else 1

    # dimension
    d = x.shape[1]
    # number of bivariate edges is typically d-1 for a simple chain structure
    # but in an R-vine it could be more. For now, we do a chain approach => d-1 edges
    n_cop = d - 1

    if vine.param:
        families = par_dict['param_families']  # e.g. ["ind","gaussian","student","clayton","claytonrot90"]
        # prepare data shape => [N,2,n_cop]
        data_np = []
        for i in range(n_cop):
            # gather columns i, i+1
            pair = x_torch[:, [i, i+1]]  # shape [N,2]
            data_np.append(pair.unsqueeze(2))  # shape [N,2,1]
        data_cat = torch.cat(data_np, dim=2).cpu().numpy()  # shape [N,2,n_cop]

        # parametric_fit => returns aic2,theta_list,logp_list for each edge
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

        # store in vine.copulas => each edge => list with one cop_par_obj
        vine.copulas = []
        for i in range(n_cop):
            cpo = cop_par_obj(best_fams[i], best_ths[i])
            vine.copulas.append([cpo])

    else:
        # nonparam approach => just do a single copula_obj with a 'rule_of_thumb' bandwidth
        data_np = []
        for i in range(n_cop):
            pair = x_torch[:, [i, i+1]]
            data_np.append(pair.unsqueeze(2))
        data_cat = torch.cat(data_np, dim=2)   # shape [N,2,n_cop]
        bw = bandwidth_rule_of_thumb(data_cat, 2, n_cop)  # shape [2,n_cop]
        cobj = copula_obj(bw)
        vine.copulas = [cobj]


def evaluate_vine(vine: vine_obj_bin, points: torch.Tensor):
    """
    Evaluate the vine's PDF at 'points' => returns (p, p_cop, log_marg).
    If param => we incorporate param copula factors. If nonparam => we do a partial approach.

    Args:
      vine: vine_obj_bin
      points: shape [N,d], typically the 's' space (normal).
    Returns:
      p: the final PDF at each row => shape [N]
      p_cop: the copula part => shape [N]
      log_marg: the sum of log marginal (like sum of Normal logs if we interpret margins).
    """
    d = vine.n_cop  # often same as dimension
    device = points.device

    # We'll interpret the margins as standard normal if nonparam => a partial approach
    normal_dist = torch.distributions.Normal(0.,1.)
    # sum of log marginal
    log_marg = torch.zeros(points.shape[0], device=device)
    for i in range(d):
        zcol = points[:, i]
        logpdf_i = normal_dist.log_prob(zcol)
        log_marg += logpdf_i
    # => log_marg is sum_{i=1..d} log of standard normal

    if vine.param:
        # param approach => we sum the copula log-likelihood
        n_cop = len(vine.copulas)
        log_cop = torch.zeros(points.shape[0], device=device)

        # We assume a chain => edge i => columns (i, i+1)
        for i in range(n_cop):
            if i+1 < points.shape[1]:
                uv = points[:, [i, i+1]]
                cop_p = vine.copulas[i][0]  # cop_par_obj
                # compute pdf => shape [N]
                pdf_val = copulapdf(cop_p, uv)
                # add log
                log_cop += torch.log(torch.clamp(pdf_val, 1e-30, 1e30))
        # final => p = exp( log_marg + log_cop )
        logp = log_marg + log_cop
        p = torch.exp(logp)
        return p, torch.exp(log_cop), log_marg
    else:
        # nonparam => the vine is not well-defined for multi edges in this minimal code,
        # but let's do a partial => p= exp(log_marg). No extra copula factor
        p = torch.exp(log_marg)
        return p, p, log_marg


def sample_vine(vine: vine_obj_bin, nsamples: int):
    """
    Sample from the fitted vine. 
    If param => we do a chain approach: 
      U1= uniform(0,1)
      for i in [1..d-1]:
         U_{i+1} = inverseConditional( a random in [0,1], given U_i, via copulainvccdf ).
    If nonparam => we do a trivial approach => random in [0,1].
    Returns shape [nsamples,d].
    """
    d = vine.n_cop
    if vine.param:
        # param approach => chain-based
        samples = np.zeros((nsamples, d), dtype=np.float64)
        # for each row
        for n in range(nsamples):
            # first uniform
            Urow = np.zeros(d, dtype=np.float64)
            Urow[0] = np.random.rand()
            for i in range(d-1):
                c_random = np.random.rand()  # for the second variable
                cop_p = vine.copulas[i][0]   # cop_par_obj
                uv = np.array([[Urow[i], c_random]], dtype=np.float32)  # shape [1,2]
                uv_torch = torch.from_numpy(uv)
                # compute inverse conditional for second
                U2 = copulainvccdf(cop_p, uv_torch)  # shape [1]
                Urow[i+1] = float(U2.item())
            samples[n,:] = Urow
        return samples
    else:
        # nonparam => trivial
        return np.random.rand(nsamples, d)


# Attach to vine_obj_bin
vine_obj_bin.fit = fit_vine
vine_obj_bin.evaluation = evaluate_vine
vine_obj_bin.sample = sample_vine