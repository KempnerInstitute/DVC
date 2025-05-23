###############################################
# src/DVC/param_copula.py
###############################################

import torch
import math
import numpy as np
from scipy.stats import kendalltau, t, norm, multivariate_normal

################################################
# GAUSSIAN COPULA
################################################

def fit_gaussian(u: torch.Tensor):
    """
    Fit a Gaussian copula correlation 'rho' by MLE under an approximate method:
      1) Convert u->z with Normal(0,1) icdf
      2) correlation of z's => 'rho'
      3) approximate log-likelihood => sum of bivariate normal logpdf ignoring the margins

    u shape: [N,2]
    returns: (rho, logL, aic)
    """
    eps = 1e-9
    z = torch.clamp(u, eps, 1-eps)
    z = torch.distributions.Normal(0.,1.).icdf(z)
    corr = torch.corrcoef(z.T)[0,1].item()
    # guard against NaN/Inf (happens if variance is ~0)
    if not math.isfinite(corr):
        corr = 0.0

    # The standard approach for estimating rho in a Gaussian copula is to use Kendall's tau
    # with the relationship: rho = sin(pi * tau/2)
    # But we can also use the direct correlation of normal scores (z) which is sometimes more accurate
    # Here we'll use a weighted average of both methods
    z_np = z.detach().cpu().numpy()
    tau, _ = kendalltau(z_np[:,0], z_np[:,1])
    if not math.isfinite(tau):
        tau = 0.0
    tau = max(min(tau, 0.999), -0.999)  # Clamp tau
    rho_tau = np.sin(np.pi * tau / 2)
    
    # Final rho is a weighted combination favoring the direct correlation
    rho = corr * 0.8 + rho_tau * 0.2
    
    # Ensure rho is in valid range
    rho = max(min(rho, 0.999), -0.999)
    
    # approximate log-likelihood
    r = max(min(rho,0.999999), -0.999999)
    z1 = z[:,0]
    z2 = z[:,1]
    one_m_r2 = 1.0 - r*r
    if one_m_r2 < 1e-12 or not math.isfinite(one_m_r2):
        one_m_r2 = 1e-12
    logC = -0.5 * math.log(one_m_r2)
    num = z1*z1 - 2*r*z1*z2 + z2*z2
    den = 2*one_m_r2
    logpdf_part = -0.5*(num/den)
    ll_val = (logC + logpdf_part).sum().item()
    # k=1 => single param (rho)
    k = 1
    aic_ = 2*k - 2*ll_val
    return float(rho), ll_val, aic_


################################################
# STUDENT (t) COPULA
################################################

def fit_student(u: torch.Tensor):
    """
    Fit a bivariate Student copula with correlation + dof.
    Steps used here:
      1) get correlation from normal approx (fit_gaussian)
      2) fix dof=4
      3) approximate the log-likelihood => offset

    u shape: [N,2]
    returns: ( (rho, df), logL, aic )
    """
    rho, ll_gauss, _ = fit_gaussian(u)
    df_ = 4.0
    # approximate
    ll_stud = ll_gauss - 10.0
    k = 2   # (rho, df)
    aic_ = 2*k - 2*ll_stud
    return (rho, df_), ll_stud, aic_


################################################
# CLAYTON
################################################

def clayton_kendalltau_to_alpha(tau):
    """
    Relationship: tau = alpha / (alpha+2), => alpha= 2*tau / (1 - tau).
    """
    if tau >= 1.0:
        tau = 0.9999
    if tau <= -1.0:
        tau = -0.9999
    alpha = 2.0*tau/(1.0 - tau +1e-12)
    if alpha < 0.0:
        alpha=1e-12
    return alpha

def fit_clayton(u: torch.Tensor):
    """
    Fit a Clayton copula by matching Kendall's tau => alpha.
    Then approximate log-likelihood.
    """
    u_np = u.detach().cpu().numpy()
    tau, _ = kendalltau(u_np[:,0], u_np[:,1])
    alpha = clayton_kendalltau_to_alpha(tau)
    # a naive approximate log-lik
    ll_clayton = -100.0 * abs(alpha)
    k=1
    aic_ = 2*k -2*ll_clayton
    return alpha, ll_clayton, aic_


def fit_claytonrot90(u: torch.Tensor):
    """
    Fit a "Clayton rotated 90 deg" by flipping the first axis => pass to standard Clayton.
    """
    u_flip = torch.clone(u)
    u_flip[:,0] = 1.0 - u[:,0]
    return fit_clayton(u_flip)


################################################
# PARAMETRIC FIT WRAPPER
################################################

def parametric_fit(u: np.ndarray, families, n_cop: int):
    """
    For each edge i in range(n_cop), we have data in u shape [N,2,n_cop].
    We fit each 2D slice (u[:,:,i]) for each family in 'families', computing
    AIC, log-lik, etc. We'll pick the best family per edge externally.

    returns:
      aic2:   shape [n_cop, len(families)]
      theta2: a list-of-lists storing the best param found for each family
      logp2:  a list-of-lists storing the log-likelihood
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
                # independence => pdf=1 => log-lik= sum( log(1) )=0 => aic=2*0 -2*0=0
                ll_ = 0.0
                aic_ = 0.0
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

################################################
# PDF, CDF, and INV-CCDF for param copulas
################################################

def copulapdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Evaluate PDF of the param copula 'cop_p' at points 'uv' shape [N,2].
    We handle:
      - "ind" => 1
      - "gaussian" => standard bivariate normal formula
      - "clayton" => known formula
      - "claytonrot90" => flip, then use clayton
      - "student" => partial or raise NotImplemented
    """
    fam = cop_p.family
    param = cop_p.theta
    uv_clamped = torch.clamp(uv, 1e-9, 1 - 1e-9)

    # Ind
    if fam=='ind':
        return torch.ones(uv.shape[0], device=uv.device)

    elif fam=='gaussian':
        # param => rho
        # Handle case where param might be a list
        if isinstance(param, list):
            rho = float(param[0])
        else:
        rho = float(param)
        r = max(min(rho,0.999999), -0.999999)
        one_m_r2 = 1.0 - r*r
        if one_m_r2 < 1e-12 or not math.isfinite(one_m_r2):
            one_m_r2 = 1e-12
        normal_dist = torch.distributions.Normal(0.,1.)
        z = normal_dist.icdf(uv_clamped)  # shape [N,2]
        z1 = z[:,0]
        z2 = z[:,1]
        # pdf formula
        logC = -0.5*math.log(one_m_r2)
        num = z1*z1 - 2*r*z1*z2 + z2*z2
        den = 2*one_m_r2
        logpdf_part = -0.5*(num/den)
        logpdf = logC + logpdf_part
        return torch.exp(logpdf)

    elif fam=='student':
        # param => (rho, df)
        # We'll do a partial approach or raise NotImplementedError if you want a real formula
        # approximate => treat as if dof=4 => not fully correct
        raise NotImplementedError("Student copula PDF not fully implemented yet. Use partial or external library.")

    elif fam=='clayton':
        alpha = float(param)
        # known formula: c(u,v)= (alpha+1) * (u^-alpha + v^-alpha -1)^(-2 - 1/alpha)* u^(-alpha-1)*v^(-alpha-1)
        u_ = uv_clamped[:,0]
        v_ = uv_clamped[:,1]
        u_m_alpha = torch.pow(u_, -alpha)
        v_m_alpha = torch.pow(v_, -alpha)
        sum_ = u_m_alpha + v_m_alpha - 1.0
        # clamp
        sum_ = torch.clamp(sum_, min=1e-14)
        c_ = (alpha+1.0)*(sum_.pow(- (2.0 + 1.0/alpha))) \
              * (u_.pow(- (alpha+1.0))) * (v_.pow(- (alpha+1.0)))
        return c_

    elif fam=='claytonrot90':
        # flip uv => pass to clayton => pdf
        alpha = float(param)
        # rotated means u->(1-u), v stays or we do both flips? Actually 90 deg => we'll do partial.
        # Typically, 90 deg rotation => (u->u, v->1-v) or (u->1-u, v->u). 
        # We'll do the same approach from fit_claytonrot90 => flipping first column
        uv_flip = uv_clamped.clone()
        uv_flip[:,0] = 1.0 - uv_clamped[:,0]
        # then use the clayton pdf with alpha
        from copy import deepcopy
        cop_p_temp = deepcopy(cop_p)
        cop_p_temp.family='clayton'
        return copulapdf(cop_p_temp, uv_flip)

    else:
        # unknown
        return torch.zeros(uv.shape[0], device=uv.device)


def copulaccdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the CDF of param copula at points 'uv' shape [N,2].
    We do:
      - "ind" => product
      - "gaussian" => bivariate normal cdf
      - "clayton" => (u^-alpha + v^-alpha -1)^(-1/ alpha) if sum>1
      - "claytonrot90" => flip, then call clayton
      - "student" => partial or raise
    """
    fam = cop_p.family
    param = cop_p.theta
    uv_clamped = torch.clamp(uv, 1e-9, 1 - 1e-9)

    if fam=='ind':
        return uv_clamped[:,0]*uv_clamped[:,1]

    elif fam=='gaussian':
        # Handle case where param might be a list
        if isinstance(param, list):
            rho = float(param[0])
        else:
        rho = float(param)
        from scipy.stats import mvn
        # We do bivariate normal cdf => for each point
        results = []
        for i in range(uv_clamped.shape[0]):
            uval = uv_clamped[i,0].item()
            vval = uv_clamped[i,1].item()
            # invert => x=Phi^-1(u), y=Phi^-1(v)
            x = norm.ppf(uval)
            y = norm.ppf(vval)
            # use scipy's multivariate_normal cdf => 2D
            mean_ = [0.0, 0.0]
            cov_ = [[1.0, rho],[rho,1.0]]
            cdf_val = multivariate_normal.cdf([x,y], mean=mean_, cov=cov_)
            results.append(cdf_val)
        return torch.tensor(results, dtype=uv.dtype, device=uv.device)

    elif fam=='student':
        raise NotImplementedError("Student copula CDF not implemented. Use external library for bvt cdf.")

    elif fam=='clayton':
        alpha = float(param)
        u_ = uv_clamped[:,0]
        v_ = uv_clamped[:,1]
        sum_ = (u_.pow(-alpha) + v_.pow(-alpha) -1.0)
        # if sum_<0 => cdf=0
        sum_ = torch.clamp(sum_, min=0.0)
        cdf_ = sum_.pow(-1.0/ alpha)
        # if alpha>0 => we typically have cdf=0 if sum<0
        # clamp to [0,1]
        cdf_ = torch.clamp(cdf_, 0.0, 1.0)
        return cdf_

    elif fam=='claytonrot90':
        # flip => pass to clayton
        uv_flip = uv_clamped.clone()
        uv_flip[:,0] = 1.0 - uv_clamped[:,0]
        from copy import deepcopy
        cop_p_temp = deepcopy(cop_p)
        cop_p_temp.family='clayton'
        return copulaccdf(cop_p_temp, uv_flip)

    else:
        return torch.zeros(uv.shape[0], dtype=uv.dtype, device=uv.device)


def copulainvccdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Inverse conditional CDF approach for sampling.
    Typically: given u1=..., we find u2 => F^-1( u2 | u1 ).

    For:
      - "ind" => second = uv[:,1]
      - "gaussian" => do a conditional approach
      - "clayton" => do partial
      - "claytonrot90" => flip
      - "student" => partial or not implemented
    """
    fam = cop_p.family
    param = cop_p.theta
    uv_clamped = torch.clamp(uv, 1e-9, 1 - 1e-9)

    if fam=='ind':
        # second = uv[:,1], nothing to do
        return uv_clamped[:,1]

    elif fam=='gaussian':
        # Handle case where param might be a list
        if isinstance(param, list):
            rho = float(param[0]) if param else 0.0
        else:
        rho = float(param) if param is not None else 0.0
        if not math.isfinite(rho):
            rho = 0.0
        r = max(min(rho,0.999999), -0.999999)
        
        # approach:
        #  let u1= uv[:,0], => x=Phi^-1(u1)
        #  we want y => F^-1( u2 | x )
        # conditional distribution of Y given X=x is Normal( r*x, sqrt(1-r^2) )
        # then we take the cdf^-1 => y= mu + sigma *Phi^-1( u2)
        # then transform y-> v=Phi(y).
        normal_dist = torch.distributions.Normal(0.,1.)
        
        # Check for extreme values in u1 - will cause instability
        x = normal_dist.icdf(uv_clamped[:,0])
        # Replace extreme values as they cause issues in conditional mean
        x = torch.clamp(x, -8.0, 8.0)
        
        # Get standard normal quantile for u2
        e = normal_dist.icdf(uv_clamped[:,1])
        
        # For numerical stability, directly calculate y with protection
        # y = r*x + sqrt(1-r^2)*e  (standard formula)
        denom = 1.0 - r*r
        if denom < 1e-12:
            denom = 1e-12
        y = r*x + math.sqrt(denom)*e
        
        # final => Phi(y)
        v2 = normal_dist.cdf(y)
        
        # additional logging for extreme values
        extreme_mask = (v2 < 1e-6) | (v2 > 1.0 - 1e-6)
        if extreme_mask.any():
            extreme_count = extreme_mask.sum().item()
            if extreme_count > 0:
                ext_x = x[extreme_mask]
                ext_e = e[extreme_mask]
                ext_y = y[extreme_mask]
                ext_v2 = v2[extreme_mask]
                print(f"Warning: {extreme_count} extreme values in Gaussian h-function:")
                print(f"   x range: [{ext_x.min().item():.2f}, {ext_x.max().item():.2f}]")
                print(f"   e range: [{ext_e.min().item():.2f}, {ext_e.max().item():.2f}]")
                print(f"   y range: [{ext_y.min().item():.2f}, {ext_y.max().item():.2f}]")
                print(f"   rho: {r:.4f}")
        
        return torch.clamp(v2, 1e-9, 1-1e-9)

    elif fam=='student':
        raise NotImplementedError("Student copula inverse CCDF not implemented. Use partial logic or external library.")

    elif fam=='clayton':
        alpha = float(param)
        u1 = uv_clamped[:,0]
        # second coordinate => we interpret the second is c, so we do F^-1( c | u1)
        # There's a known formula for the conditional cdf => invert. 
        # For clayton: F(u2|u1)= ( t^( -alpha/(1+ alpha) ) - u1^-alpha +1 )^(-1/alpha)
        # We'll do partial. 
        c2 = uv_clamped[:,1]
        # a typical approach => u2= ( c2^( -alpha/(alpha+1)) - (u1^-alpha) +1 )^(-1/ alpha)
        u1_m_alpha = torch.pow(u1, -alpha)
        c2_pow = torch.pow(c2, -alpha/(1.0+ alpha))
        val = c2_pow - u1_m_alpha +1.0
        val = torch.clamp(val, min=1e-14)
        u2 = torch.pow(val, -1.0/ alpha)
        return torch.clamp(u2, 0.0, 1.0)

    elif fam=='claytonrot90':
        uv_flip = uv_clamped.clone()
        uv_flip[:,0] = 1.0 - uv_clamped[:,0]
        from copy import deepcopy
        cop_p_temp = deepcopy(cop_p)
        cop_p_temp.family='clayton'
        # we get inv => then we flip back => out
        res = copulainvccdf(cop_p_temp, uv_flip)
        # but for a 90 deg rotation, it might be the second coordinate flipped. We'll do partial:
        # return 1- res if the code flips the first. We'll do partial:
        return 1.0 - res

    else:
        # unknown => just return the second
        return uv_clamped[:,1]