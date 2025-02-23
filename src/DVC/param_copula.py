###############################################
# src/torch_vine/param_copula.py
###############################################

import torch
import math
import numpy as np
from scipy.stats import kendalltau, t, norm


################################################
# GAUSSIAN COPULA
################################################

def fit_gaussian(u: torch.Tensor):
    """
    Fit a Gaussian copula correlation 'rho'.
    u shape: [N,2]
    returns: rho (float), logL (float), aic (float)
    """
    eps = 1e-9
    z = torch.clamp(u, eps, 1-eps)
    z = torch.distributions.Normal(0.,1.).icdf(z)
    corr = torch.corrcoef(z.T)[0,1].item()
    rho = corr
    # approximate log-likelihood
    r = max(min(rho,0.999999), -0.999999)
    z1 = z[:,0]
    z2 = z[:,1]
    one_m_r2 = 1.0 - r*r
    logC = -0.5 * math.log(one_m_r2)
    num = z1*z1 - 2*r*z1*z2 + z2*z2
    den = 2*one_m_r2
    logpdf_part = -0.5*(num/den)
    ll_val = (logC + logpdf_part).sum().item()
    k=1
    aic_ = 2*k - 2*ll_val
    return float(rho), ll_val, aic_


################################################
# STUDENT (t) COPULA
################################################

def fit_student(u: torch.Tensor):
    """
    Fit a bivariate Student copula with correlation + dof.
    We'll do a placeholder approach:
      1) get correlation from normal approx
      2) fix dof=4
    """
    rho, ll_gauss, _ = fit_gaussian(u)
    df_ = 4.0
    ll_stud = ll_gauss - 10.0
    k=2
    aic_ = 2*k - 2*ll_stud
    return (rho, df_), ll_stud, aic_


################################################
# CLAYTON
################################################

def clayton_kendalltau_to_alpha(tau):
    if tau>=1.0:
        tau=0.9999
    if tau<=-1.0:
        tau=-0.9999
    alpha = 2.0*tau/(1.0 - tau +1e-12)
    if alpha<0.0:
        alpha=1e-12
    return alpha

def fit_clayton(u: torch.Tensor):
    u_np = u.detach().cpu().numpy()
    tau, _ = kendalltau(u_np[:,0], u_np[:,1])
    alpha = clayton_kendalltau_to_alpha(tau)
    ll_clayton = -100.0 * abs(alpha)
    k=1
    aic_ = 2*k -2*ll_clayton
    return alpha, ll_clayton, aic_


def fit_claytonrot90(u: torch.Tensor):
    u_flip = torch.clone(u)
    u_flip[:,0] = 1.0 - u[:,0]
    return fit_clayton(u_flip)


################################################
# PARAMETRIC FIT WRAPPER
################################################

def parametric_fit(u: np.ndarray, families, n_cop: int):
    """
    For each edge i in range(n_cop), we have data in u shape [N,2,n_cop].
    We fit each 2D slice. Then for each family in 'families', we compute AIC, log-lik, etc.
    We'll pick the best.

    returns:
      aic2: shape [n_cop, len(families)]
      theta2: ...
      logp2: ...
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_t = torch.tensor(u, device=device, dtype=torch.float32)  # shape [N,2,n_cop]
    aic_list = []
    theta_list = []
    logp_list = []
    for i in range(n_cop):
        data_i = data_t[:,:,i]
        fam_aic = []
        fam_theta = []
        fam_logp = []
        for fam in families:
            if fam=='ind':
                ll_ = 0.0
                aic_ = 2*0 -2*ll_
                param_ = None
                fam_aic.append(aic_)
                fam_theta.append(param_)
                fam_logp.append(ll_)
            elif fam=='gaussian':
                r, ll_, aic_ = fit_gaussian(data_i)
                fam_aic.append(aic_)
                fam_theta.append(r)
                fam_logp.append(ll_)
            elif fam=='student':
                (r, df), ll_, aic_ = fit_student(data_i)
                fam_aic.append(aic_)
                fam_theta.append((r, df))
                fam_logp.append(ll_)
            elif fam=='clayton':
                alpha, ll_, aic_ = fit_clayton(data_i)
                fam_aic.append(aic_)
                fam_theta.append(alpha)
                fam_logp.append(ll_)
            elif fam=='claytonrot90':
                alpha, ll_, aic_ = fit_claytonrot90(data_i)
                fam_aic.append(aic_)
                fam_theta.append(alpha)
                fam_logp.append(ll_)
            else:
                fam_aic.append(1e15)
                fam_theta.append(None)
                fam_logp.append(-1e15)
        aic_list.append(fam_aic)
        theta_list.append(fam_theta)
        logp_list.append(fam_logp)
    aic2 = np.array(aic_list)
    return aic2, theta_list, logp_list


def copulainvccdf(cop_p, uv):
    """
    Inverse conditional cdf for param copula.
    """
    pass


def copulaccdf(cop_p, uv):
    """
    Evaluate cdf for param copula.
    """
    pass


def copulapdf(cop_p, uv):
    """
    Evaluate pdf for param copula.
    """
    pass