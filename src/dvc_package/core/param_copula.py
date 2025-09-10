###############################################
# src/DVC/param_copula.py
###############################################

import torch
import math
import numpy as np
from scipy.stats import kendalltau, t, norm, multivariate_normal

from ..utils.utils_tensor import replace_nan_inf, check_finite, safe_log, handle_small_sample_size

################################################
# GAUSSIAN COPULA
################################################

def fit_gaussian(u: torch.Tensor):
    """
    Fit a Gaussian copula using gradient-based optimization (matching TensorFlow).
    
    Uses Nadam optimizer to minimize negative log-likelihood.
    """
    device = u.device
    dtype = u.dtype
    n_samples = u.shape[0]
    
    # Initialize parameter (correlation) - match TensorFlow's pos_trace = 0.5
    rho_init = torch.tensor([0.5], dtype=dtype, device=device, requires_grad=True)
    
    # Optimization parameters from TensorFlow
    lr = 0.005
    conv_tol = 1e-3
    max_iter = 200 if u.shape[0] > 100 else 100
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6
    
    # Initialize Nadam parameters
    m = torch.zeros_like(rho_init)
    v = torch.zeros_like(rho_init)
    
    # Previous error for convergence check
    prev_err = torch.tensor(float('inf'), dtype=dtype, device=device)
    
    for iter_num in range(max_iter):
        # Zero gradients
        if rho_init.grad is not None:
            rho_init.grad.zero_()
            
        # Compute negative log-likelihood (matching TensorFlow's gaussian_cost)
        rho = torch.clamp(rho_init, -0.999, 0.999)
        
        # Convert to normal scores
        z = torch.distributions.Normal(0, 1).icdf(torch.clamp(u, 1e-9, 1-1e-9))
        z1, z2 = z[:, 0], z[:, 1]
        
        # Compute copula PDF (not joint PDF)
        one_minus_rho2 = 1 - rho**2
        one_minus_rho2 = torch.clamp(one_minus_rho2, min=1e-12)
        
        # Log copula density
        log_copula_pdf = -0.5 * torch.log(one_minus_rho2) - \
                         (rho**2 * (z1**2 + z2**2) - 2*rho*z1*z2) / (2*one_minus_rho2)
        
        # Negative log-likelihood
        err = -torch.sum(log_copula_pdf)
        
        # Check convergence
        if torch.abs(err - prev_err) < conv_tol:
            break
            
        # Backward pass
        err.backward()
        
        # Nadam update (matching TensorFlow)
        grad = rho_init.grad
        iter1 = float(iter_num + 1)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**iter1) + (1 - beta1) * grad / (1 - beta1**iter1)
        v_hat = v / (1 - beta2**iter1)
        
        # Update parameter
        with torch.no_grad():
            rho_init -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            rho_init.clamp_(-0.999, 0.999)
            
        prev_err = err.clone()
    
    # Final parameter value
    rho_final = torch.clamp(rho_init, -0.999, 0.999).item()
    
    # Compute final log-likelihood for AIC
    with torch.no_grad():
        z = torch.distributions.Normal(0, 1).icdf(torch.clamp(u, 1e-9, 1-1e-9))
        z1, z2 = z[:, 0], z[:, 1]
        one_minus_rho2 = 1 - rho_final**2
        one_minus_rho2 = max(one_minus_rho2, 1e-12)
        
        log_copula_pdf = -0.5 * math.log(one_minus_rho2) - \
                         ((rho_final**2 * (z1**2 + z2**2) - 2*rho_final*z1*z2) / (2*one_minus_rho2)).sum()
        
        ll_val = log_copula_pdf.item()
    
    k = 1  # One parameter
    aic = 2*k - 2*ll_val
    
    return float(rho_final), ll_val, aic


################################################
# STUDENT (t) COPULA
################################################

def fit_student(u: torch.Tensor):
    """
    Fit a bivariate Student copula using gradient-based optimization (matching TensorFlow).
    
    Uses Nadam optimizer to minimize negative log-likelihood of bivariate t-copula.
    Parameters: correlation (rho) and degrees of freedom (nu).
    
    u shape: [N,2]
    returns: ( (rho, df), logL, aic )
    """
    device = u.device
    dtype = u.dtype
    n_samples = u.shape[0]
    
    # Initialize parameters: [rho, log(nu-2)] to ensure nu > 2
    # Start with Gaussian fit for rho, and nu=4
    rho_init, _, _ = fit_gaussian(u)
    params_init = torch.tensor([rho_init, math.log(2.0)], dtype=dtype, device=device, requires_grad=True)
    
    # Optimization parameters from TensorFlow
    lr = 0.01
    conv_tol = 1e-3
    max_iter = 200 if u.shape[0] > 100 else 100
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6
    
    # Initialize Nadam parameters
    m = torch.zeros_like(params_init)
    v = torch.zeros_like(params_init)
    
    # Previous error for convergence check
    prev_err = torch.tensor(float('inf'), dtype=dtype, device=device)
    
    for iter_num in range(max_iter):
        # Zero gradients
        if params_init.grad is not None:
            params_init.grad.zero_()
            
        # Extract parameters
        rho = torch.clamp(params_init[0], -0.999, 0.999)
        nu = torch.exp(params_init[1]) + 2.0  # Ensure nu > 2
        nu = torch.clamp(nu, 2.1, 50.0)  # Reasonable bounds
        
        # Convert to t-distribution quantiles
        u_clamped = torch.clamp(u, 1e-9, 1-1e-9)
        
        # Use scipy's t.ppf for quantiles (more stable than torch implementation)
        t_quantiles = torch.zeros_like(u_clamped)
        for i in range(u_clamped.shape[0]):
            for j in range(2):
                t_quantiles[i, j] = torch.tensor(
                    t.ppf(u_clamped[i, j].cpu().numpy(), nu.cpu().numpy()),
                    dtype=dtype, device=device
                )
        
        t1, t2 = t_quantiles[:, 0], t_quantiles[:, 1]
        
        # Compute copula log-density for bivariate t-copula
        # Formula: log c(u,v) = log(Gamma((nu+2)/2) * Gamma(nu/2)) - log(Gamma((nu+1)/2))^2 
        #                      - 0.5 * log(1-rho^2) + (nu+1)/2 * log(1 + (t1^2 + t2^2 - 2*rho*t1*t2)/(nu*(1-rho^2)))
        #                      - log(1 + t1^2/nu)^((nu+1)/2) - log(1 + t2^2/nu)^((nu+1)/2)
        
        one_minus_rho2 = 1 - rho**2
        one_minus_rho2 = torch.clamp(one_minus_rho2, min=1e-12)
        
        # Gamma function terms
        log_gamma_term = (torch.lgamma((nu + 2) / 2) + torch.lgamma(nu / 2) 
                         - 2 * torch.lgamma((nu + 1) / 2))
        
        # Quadratic form
        quad_form = (t1**2 + t2**2 - 2*rho*t1*t2) / (nu * one_minus_rho2)
        
        # Log copula density
        log_copula_pdf = (log_gamma_term - 0.5 * torch.log(one_minus_rho2) 
                         + (nu + 1) / 2 * torch.log(1 + quad_form)
                         - (nu + 1) / 2 * torch.log(1 + t1**2 / nu)
                         - (nu + 1) / 2 * torch.log(1 + t2**2 / nu))
        
        # Handle potential NaN/Inf values
        log_copula_pdf = torch.where(torch.isfinite(log_copula_pdf), 
                                   log_copula_pdf, torch.tensor(-30.0, dtype=dtype, device=device))
        
        # Negative log-likelihood
        err = -torch.sum(log_copula_pdf)
        
        # Check convergence
        if torch.abs(err - prev_err) < conv_tol:
            break
            
        # Backward pass
        err.backward()
        
        # Nadam update (matching TensorFlow)
        grad = params_init.grad
        iter1 = float(iter_num + 1)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**iter1) + (1 - beta1) * grad / (1 - beta1**iter1)
        v_hat = v / (1 - beta2**iter1)
        
        # Update parameters
        with torch.no_grad():
            params_init -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            
        prev_err = err.clone()
    
    # Final parameter values
    rho_final = torch.clamp(params_init[0], -0.999, 0.999).item()
    nu_final = (torch.exp(params_init[1]) + 2.0).item()
    nu_final = max(min(nu_final, 50.0), 2.1)
    
    # Compute final log-likelihood for AIC
    ll_val = -err.item()
    
    k = 2  # Two parameters: rho and nu
    aic = 2*k - 2*ll_val
    
    return (float(rho_final), float(nu_final)), ll_val, aic


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
    Fit a Clayton copula using gradient-based optimization (matching TensorFlow).
    
    Uses Nadam optimizer to minimize negative log-likelihood.
    """
    device = u.device
    dtype = u.dtype
    
    # Initialize parameter - match TensorFlow's pos_trace = 3.0
    alpha_init = torch.tensor([3.0], dtype=dtype, device=device, requires_grad=True)
    
    # Optimization parameters from TensorFlow
    lr = 0.2
    conv_tol = 1e-3
    max_iter = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6
    
    # Initialize Nadam parameters
    m = torch.zeros_like(alpha_init)
    v = torch.zeros_like(alpha_init)
    
    # Previous error for convergence check
    prev_err = torch.tensor(float('inf'), dtype=dtype, device=device)
    
    for iter_num in range(max_iter):
        # Zero gradients
        if alpha_init.grad is not None:
            alpha_init.grad.zero_()
            
        # Compute negative log-likelihood
        alpha = torch.clamp(alpha_init, 0.1, 20.0)
        
        # Clayton copula PDF: c(u,v) = (1+alpha) * (u^-alpha + v^-alpha - 1)^(-2-1/alpha) * u^(-alpha-1) * v^(-alpha-1)
        u_clamped = torch.clamp(u, 1e-9, 1-1e-9)
        u1, u2 = u_clamped[:, 0], u_clamped[:, 1]
        
        u1_neg_alpha = torch.pow(u1, -alpha)
        u2_neg_alpha = torch.pow(u2, -alpha)
        sum_term = u1_neg_alpha + u2_neg_alpha - 1.0
        sum_term = torch.clamp(sum_term, min=1e-14)
        
        # Log PDF
        log_pdf = torch.log(1 + alpha) + \
                  (-2 - 1/alpha) * torch.log(sum_term) + \
                  (-alpha - 1) * (torch.log(u1) + torch.log(u2))
        
        # Handle special case where alpha = 0 (independence)
        if alpha.item() < 0.01:
            log_pdf = torch.zeros_like(log_pdf)
        
        # Negative log-likelihood
        err = -torch.sum(log_pdf)
        
        # Check convergence
        if torch.abs(err - prev_err) < conv_tol:
            break
            
        # Backward pass
        err.backward()
        
        # Nadam update
        grad = alpha_init.grad
        iter1 = float(iter_num + 1)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**iter1) + (1 - beta1) * grad / (1 - beta1**iter1)
        v_hat = v / (1 - beta2**iter1)
        
        # Update parameter
        with torch.no_grad():
            alpha_init -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            alpha_init.clamp_(0.1, 20.0)
            
        prev_err = err.clone()
    
    # Final parameter value
    alpha_final = torch.clamp(alpha_init, 0.1, 20.0).item()
    ll_val = -err.item()
    
    k = 1  # One parameter
    aic = 2*k - 2*ll_val
    
    return float(alpha_final), ll_val, aic


def fit_claytonrot90(u: torch.Tensor):
    """
    Fit a "Clayton rotated 90 deg" by flipping the first axis => pass to standard Clayton.
    """
    u_flip = torch.clone(u)
    u_flip[:,0] = 1.0 - u[:,0]
    return fit_clayton(u_flip)


################################################
# FRANK COPULA
################################################

def frank_kendalltau_to_theta(tau):
    """
    Relationship between Kendall's tau and Frank copula parameter theta.
    tau = 1 - 4/theta + 4*D_1(theta)/theta, where D_1 is Debye function.
    For practical purposes, use approximation: theta ≈ 4*tau/(1-tau) for small tau.
    """
    if abs(tau) < 1e-6:
        return 0.0
    if tau >= 0.99:
        tau = 0.99
    if tau <= -0.99:
        tau = -0.99
    
    # Initial approximation
    if abs(tau) < 0.5:
        theta_init = 4.0 * tau / (1.0 - abs(tau))
    else:
        theta_init = 10.0 * np.sign(tau)
    
    return theta_init

def fit_frank(u: torch.Tensor):
    """
    Fit a Frank copula using gradient-based optimization.
    
    Frank copula: C(u,v) = -1/theta * log(1 + (exp(-theta*u)-1)(exp(-theta*v)-1)/(exp(-theta)-1))
    """
    device = u.device
    dtype = u.dtype
    
    # Initialize parameter using Kendall's tau
    u_np = u.cpu().numpy()
    tau_kendall, _ = kendalltau(u_np[:, 0], u_np[:, 1])
    theta_init_val = frank_kendalltau_to_theta(tau_kendall)
    
    theta_init = torch.tensor([theta_init_val], dtype=dtype, device=device, requires_grad=True)
    
    # Optimization parameters
    lr = 0.1
    conv_tol = 1e-3
    max_iter = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6
    
    # Initialize Nadam parameters
    m = torch.zeros_like(theta_init)
    v = torch.zeros_like(theta_init)
    
    prev_err = torch.tensor(float('inf'), dtype=dtype, device=device)
    
    for iter_num in range(max_iter):
        if theta_init.grad is not None:
            theta_init.grad.zero_()
            
        theta = torch.clamp(theta_init, -20.0, 20.0)
        
        # Handle theta ≈ 0 (independence)
        if torch.abs(theta) < 1e-6:
            log_pdf = torch.zeros(u.shape[0], dtype=dtype, device=device)
        else:
            u_clamped = torch.clamp(u, 1e-9, 1-1e-9)
            u1, u2 = u_clamped[:, 0], u_clamped[:, 1]
            
            # Frank copula PDF computation
            exp_theta = torch.exp(theta)
            exp_theta_u1 = torch.exp(theta * u1)
            exp_theta_u2 = torch.exp(theta * u2)
            
            numerator = theta * (exp_theta - 1) * exp_theta_u1 * exp_theta_u2
            denominator = (exp_theta - exp_theta_u1) * (exp_theta - exp_theta_u2) + (exp_theta - 1)
            
            # Clamp to avoid numerical issues
            denominator = torch.clamp(denominator, min=1e-15)
            
            log_pdf = torch.log(torch.abs(numerator)) - torch.log(denominator)
        
        # Handle potential NaN/Inf values
        log_pdf = torch.where(torch.isfinite(log_pdf), 
                             log_pdf, torch.tensor(-30.0, dtype=dtype, device=device))
        
        err = -torch.sum(log_pdf)
        
        if torch.abs(err - prev_err) < conv_tol:
            break
            
        err.backward()
        
        # Nadam update
        grad = theta_init.grad
        iter1 = float(iter_num + 1)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**iter1) + (1 - beta1) * grad / (1 - beta1**iter1)
        v_hat = v / (1 - beta2**iter1)
        
        with torch.no_grad():
            theta_init -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            
        prev_err = err.clone()
    
    theta_final = torch.clamp(theta_init, -20.0, 20.0).item()
    ll_val = -err.item()
    
    k = 1  # One parameter
    aic = 2*k - 2*ll_val
    
    return float(theta_final), ll_val, aic


################################################
# GUMBEL COPULA
################################################

def gumbel_kendalltau_to_theta(tau):
    """
    Relationship: tau = (theta-1)/theta, => theta = 1/(1-tau).
    """
    if tau >= 0.99:
        tau = 0.99
    if tau <= 0.0:
        return 1.0001  # Near independence
    theta = 1.0 / (1.0 - tau)
    return max(theta, 1.0001)

def fit_gumbel(u: torch.Tensor):
    """
    Fit a Gumbel copula using gradient-based optimization.
    
    Gumbel copula: C(u,v) = exp(-[(-log u)^theta + (-log v)^theta]^(1/theta))
    """
    device = u.device
    dtype = u.dtype
    
    # Initialize parameter using Kendall's tau
    u_np = u.cpu().numpy()
    tau_kendall, _ = kendalltau(u_np[:, 0], u_np[:, 1])
    theta_init_val = gumbel_kendalltau_to_theta(max(tau_kendall, 0.01))
    
    theta_init = torch.tensor([theta_init_val], dtype=dtype, device=device, requires_grad=True)
    
    # Optimization parameters
    lr = 0.05
    conv_tol = 1e-3
    max_iter = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6
    
    # Initialize Nadam parameters
    m = torch.zeros_like(theta_init)
    v = torch.zeros_like(theta_init)
    
    prev_err = torch.tensor(float('inf'), dtype=dtype, device=device)
    
    for iter_num in range(max_iter):
        if theta_init.grad is not None:
            theta_init.grad.zero_()
            
        theta = torch.clamp(theta_init, 1.001, 20.0)  # theta > 1 for Gumbel
        
        u_clamped = torch.clamp(u, 1e-9, 1-1e-9)
        u1, u2 = u_clamped[:, 0], u_clamped[:, 1]
        
        # Gumbel copula PDF computation
        log_u1 = torch.log(u1)
        log_u2 = torch.log(u2)
        
        neg_log_u1_pow = torch.pow(-log_u1, theta)
        neg_log_u2_pow = torch.pow(-log_u2, theta)
        sum_pow = neg_log_u1_pow + neg_log_u2_pow
        
        # Clamp to avoid numerical issues
        sum_pow = torch.clamp(sum_pow, min=1e-15)
        sum_pow_1_theta = torch.pow(sum_pow, 1.0/theta)
        
        # Log PDF formula for Gumbel
        log_pdf = (torch.log(sum_pow_1_theta) + (theta - 1) * torch.log(-log_u1) 
                  + (theta - 1) * torch.log(-log_u2) + torch.log(theta - 1 + sum_pow_1_theta)
                  - sum_pow_1_theta - (2 - 1/theta) * torch.log(sum_pow))
        
        # Handle potential NaN/Inf values
        log_pdf = torch.where(torch.isfinite(log_pdf), 
                             log_pdf, torch.tensor(-30.0, dtype=dtype, device=device))
        
        err = -torch.sum(log_pdf)
        
        if torch.abs(err - prev_err) < conv_tol:
            break
            
        err.backward()
        
        # Nadam update
        grad = theta_init.grad
        iter1 = float(iter_num + 1)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**iter1) + (1 - beta1) * grad / (1 - beta1**iter1)
        v_hat = v / (1 - beta2**iter1)
        
        with torch.no_grad():
            theta_init -= lr * m_hat / (torch.sqrt(v_hat) + eps)
            
        prev_err = err.clone()
    
    theta_final = torch.clamp(theta_init, 1.001, 20.0).item()
    ll_val = -err.item()
    
    k = 1  # One parameter
    aic = 2*k - 2*ll_val
    
    return float(theta_final), ll_val, aic


################################################
# PARAMETRIC FIT WRAPPER
################################################

def parametric_fit(u: np.ndarray, families, n_cop: int):
    """
    For each edge i in range(n_cop), we have data in u shape [N,2,n_cop].
    We fit each 2D slice (u[:,:,i]) for each family in 'families', computing
    AIC, log-lik, etc. We'll pick the best family per edge externally.

    Enhanced with robustness checks for small samples and numerical stability.

    returns:
      aic2:   shape [n_cop, len(families)]
      theta2: a list-of-lists storing the best param found for each family
      logp2:  a list-of-lists storing the log-likelihood
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_t = torch.tensor(u, device=device, dtype=torch.float32)  # shape [N,2,n_cop]
    
    # Check for NaN/Inf values in input data
    data_t = replace_nan_inf(data_t)
    
    aic_list = []
    theta_list = []
    logp_list = []
    for i in range(n_cop):
        data_i = data_t[:,:,i]
        
        # Handle small sample sizes
        data_i, is_small_sample = handle_small_sample_size(data_i, min_samples=30)
        
        fam_aic = []
        fam_theta = []
        fam_logp = []
        for fam in families:
            if fam=='ind':
                # Independence copula: c(u,v) = 1 for all (u,v)
                # Log-likelihood = sum(log(1)) = 0
                # But we need a fair comparison with other copulas
                # For independence, the copula density is 1, so log-density = 0
                # Total log-likelihood = n_samples * log(1) = 0
                n_samples = data_i.shape[0]
                ll_ = 0.0  # Log-likelihood is indeed 0 for independence
                k = 0  # No parameters
                # The issue is that we're comparing copula likelihoods
                # For a fair comparison, we should penalize lack of fit
                # One approach: use the empirical copula deviation
                
                # Improved independence penalty to match TensorFlow behavior
                u_vals = data_i.cpu().numpy()
                emp_corr = np.corrcoef(u_vals[:, 0], u_vals[:, 1])[0, 1]
                
                # More sophisticated penalty that matches TensorFlow's implicit behavior
                # TensorFlow tends to select Gaussian over independence when correlation exists
                correlation_strength = abs(emp_corr)
                
                if correlation_strength > 0.1:
                    # Strong penalty for independence when clear correlation exists
                    penalty = n_samples * (correlation_strength ** 2) * 10.0
                elif correlation_strength > 0.05:
                    # Moderate penalty for weak correlation
                    penalty = n_samples * (correlation_strength ** 2) * 5.0
                else:
                    # Minimal penalty for very weak correlation
                    penalty = n_samples * (correlation_strength ** 2) * 1.0
                
                # AIC for independence with penalty
                aic_ = 2*k + penalty
                
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
            elif fam=='frank':
                theta, ll_, aic_ = fit_frank(data_i)
                fam_aic.append(aic_)
                fam_theta.append(theta)
                fam_logp.append(ll_)
            elif fam=='gumbel':
                theta, ll_, aic_ = fit_gumbel(data_i)
                fam_aic.append(aic_)
                fam_theta.append(theta)
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
        if isinstance(param, (list, tuple)) and len(param) >= 2:
            rho, nu = float(param[0]), float(param[1])
        else:
            rho, nu = 0.0, 4.0
        
        rho = max(min(rho, 0.999999), -0.999999)
        nu = max(nu, 2.1)
        
        # Convert to t-distribution quantiles
        t_quantiles = torch.zeros_like(uv_clamped)
        for i in range(uv_clamped.shape[0]):
            for j in range(2):
                t_quantiles[i, j] = torch.tensor(
                    t.ppf(uv_clamped[i, j].cpu().numpy(), nu),
                    dtype=uv_clamped.dtype, device=uv_clamped.device
                )
        
        t1, t2 = t_quantiles[:, 0], t_quantiles[:, 1]
        one_minus_rho2 = 1 - rho**2
        
        # Student t copula PDF
        log_gamma_term = (torch.lgamma(torch.tensor((nu + 2) / 2)) + torch.lgamma(torch.tensor(nu / 2)) 
                         - 2 * torch.lgamma(torch.tensor((nu + 1) / 2)))
        
        quad_form = (t1**2 + t2**2 - 2*rho*t1*t2) / (nu * one_minus_rho2)
        
        log_pdf = (log_gamma_term - 0.5 * math.log(one_minus_rho2) 
                  + (nu + 1) / 2 * torch.log(1 + quad_form)
                  - (nu + 1) / 2 * torch.log(1 + t1**2 / nu)
                  - (nu + 1) / 2 * torch.log(1 + t2**2 / nu))
        
        return torch.exp(torch.clamp(log_pdf, -30.0, 30.0))

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

    elif fam=='frank':
        theta = float(param)
        if abs(theta) < 1e-6:
            return torch.ones(uv.shape[0], device=uv.device)
        
        u1, u2 = uv_clamped[:, 0], uv_clamped[:, 1]
        exp_theta = torch.exp(torch.tensor(theta))
        exp_theta_u1 = torch.exp(theta * u1)
        exp_theta_u2 = torch.exp(theta * u2)
        
        numerator = theta * (exp_theta - 1) * exp_theta_u1 * exp_theta_u2
        denominator = (exp_theta - exp_theta_u1) * (exp_theta - exp_theta_u2) + (exp_theta - 1)
        
        denominator = torch.clamp(denominator, min=1e-15)
        pdf = torch.abs(numerator) / denominator
        
        return torch.clamp(pdf, 1e-15, 1e15)

    elif fam=='gumbel':
        theta = float(param)
        theta = max(theta, 1.001)
        
        u1, u2 = uv_clamped[:, 0], uv_clamped[:, 1]
        log_u1 = torch.log(u1)
        log_u2 = torch.log(u2)
        
        neg_log_u1_pow = torch.pow(-log_u1, theta)
        neg_log_u2_pow = torch.pow(-log_u2, theta)
        sum_pow = neg_log_u1_pow + neg_log_u2_pow
        sum_pow = torch.clamp(sum_pow, min=1e-15)
        
        sum_pow_1_theta = torch.pow(sum_pow, 1.0/theta)
        
        # Gumbel copula PDF
        pdf = (sum_pow_1_theta * torch.pow(-log_u1, theta - 1) * torch.pow(-log_u2, theta - 1) 
               * (theta - 1 + sum_pow_1_theta) / (u1 * u2 * torch.pow(sum_pow, 2 - 1/theta)))
        
        return torch.clamp(pdf, 1e-15, 1e15)

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

    elif fam=='frank':
        theta = float(param)
        if abs(theta) < 1e-6:
            return uv_clamped[:, 0] * uv_clamped[:, 1]
        
        u1, u2 = uv_clamped[:, 0], uv_clamped[:, 1]
        exp_theta = torch.exp(torch.tensor(theta))
        exp_theta_u1 = torch.exp(theta * u1)
        exp_theta_u2 = torch.exp(theta * u2)
        
        # Frank copula CDF: C(u,v) = -1/theta * log(1 + (exp(-theta*u)-1)(exp(-theta*v)-1)/(exp(-theta)-1))
        numerator = (exp_theta_u1 - 1) * (exp_theta_u2 - 1)
        denominator = exp_theta - 1
        
        cdf = -torch.log(1 + numerator / denominator) / theta
        return torch.clamp(cdf, 0.0, 1.0)

    elif fam=='gumbel':
        theta = float(param)
        theta = max(theta, 1.001)
        
        u1, u2 = uv_clamped[:, 0], uv_clamped[:, 1]
        log_u1 = torch.log(u1)
        log_u2 = torch.log(u2)
        
        neg_log_u1_pow = torch.pow(-log_u1, theta)
        neg_log_u2_pow = torch.pow(-log_u2, theta)
        sum_pow = neg_log_u1_pow + neg_log_u2_pow
        sum_pow = torch.clamp(sum_pow, min=1e-15)
        
        # Gumbel copula CDF: C(u,v) = exp(-[(-log u)^theta + (-log v)^theta]^(1/theta))
        cdf = torch.exp(-torch.pow(sum_pow, 1.0/theta))
        return torch.clamp(cdf, 0.0, 1.0)

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

    elif fam=='frank':
        theta = float(param)
        if abs(theta) < 1e-6:
            return uv_clamped[:, 1]
        
        u1 = uv_clamped[:, 0]
        c2 = uv_clamped[:, 1]
        
        # Frank copula conditional CDF inverse (approximate)
        # For simplicity, use numerical approximation
        exp_theta = torch.exp(torch.tensor(theta))
        exp_theta_u1 = torch.exp(theta * u1)
        
        # Approximate inverse using iterative approach or direct formula
        # Direct formula is complex, so use approximation
        if theta > 0:
            # Positive dependence case
            u2_approx = -torch.log(1 - c2 * (1 - torch.exp(-theta)) / (exp_theta_u1 - c2 * (exp_theta_u1 - 1))) / theta
        else:
            # Negative dependence case
            u2_approx = -torch.log(1 + c2 * (torch.exp(-theta) - 1) / (1 - c2 + c2 * exp_theta_u1)) / theta
            
        return torch.clamp(u2_approx, 1e-9, 1-1e-9)

    elif fam=='gumbel':
        theta = float(param)
        theta = max(theta, 1.001)
        
        u1 = uv_clamped[:, 0]
        c2 = uv_clamped[:, 1]
        
        # Gumbel copula conditional CDF inverse
        log_u1 = torch.log(u1)
        neg_log_u1_pow = torch.pow(-log_u1, theta)
        
        # Conditional CDF: h(u2|u1) = C(u1,u2) * [(-log u1)^(theta-1) + (-log u2)^(theta-1)]^(1-theta) / u1
        # Inverse is complex, use approximation
        log_c2 = torch.log(torch.clamp(c2, 1e-9, 1-1e-9))
        
        # Approximate inverse
        term = -log_c2 - neg_log_u1_pow**(1.0/theta)
        term = torch.clamp(term, 1e-15, 1e15)
        
        u2_approx = torch.exp(-torch.pow(torch.clamp(term, 1e-15), theta))
        
        return torch.clamp(u2_approx, 1e-9, 1-1e-9)

    else:
        # unknown => just return the second
        return uv_clamped[:,1]