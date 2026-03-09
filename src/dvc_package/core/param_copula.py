###############################################
# src/DVC/param_copula.py
###############################################

import logging
import torch
import math
import numpy as np
from scipy.stats import kendalltau, t, norm, multivariate_normal

from ..utils.utils_tensor import replace_nan_inf, check_finite, safe_log, handle_small_sample_size

logger = logging.getLogger("DVC.param_copula")

_FAMILY_ALIASES = {
    "independence": "ind",
    "independent": "ind",
    "gauss": "gaussian",
    "student-t": "student",
    "t": "student",
}


def _normalize_family_name(family: str) -> str:
    """Normalize external family aliases to canonical internal names."""
    fam = family.lower().strip()
    return _FAMILY_ALIASES.get(fam, fam)


def _safe_empirical_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Numerically stable Pearson correlation with finite fallback."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0

    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    sx = np.std(x_centered, ddof=0)
    sy = np.std(y_centered, ddof=0)
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0

    corr = float(np.mean((x_centered / sx) * (y_centered / sy)))
    if not np.isfinite(corr):
        return 0.0
    return max(min(corr, 1.0), -1.0)

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
    Fit a bivariate Student-t copula.

    Notes
    -----
    The copula density is
      c(u,v) = f_{t,2}(t^{-1}(u), t^{-1}(v); rho, nu) / (f_{t,1}(t^{-1}(u);nu) f_{t,1}(t^{-1}(v);nu)).

    We estimate rho from Kendall's tau (elliptical copulas) and select nu by
    a lightweight grid-search on AIC. This avoids unstable gradients through
    scipy's `t.ppf` and is robust for typical sample sizes.

    Parameters
    ----------
    u : torch.Tensor
        Pseudo-observations in [0,1], shape (N,2).

    Returns
    -------
    (rho, nu), ll, aic
    """
    # Clamp to open interval (0,1) for stability in quantile transforms.
    u_np = torch.clamp(u.detach().cpu(), 1e-9, 1.0 - 1e-9).numpy()
    if u_np.shape[0] < 10:
        # Too few samples to reliably fit nu; fall back to a near-Gaussian t-copula.
        tau, _ = kendalltau(u_np[:, 0], u_np[:, 1])
        tau = 0.0 if not np.isfinite(tau) else float(tau)
        rho = math.sin(0.5 * math.pi * max(min(tau, 0.999), -0.999))
        nu = 50.0
        ll_val = 0.0
        k = 2
        aic = 2 * k - 2 * ll_val
        return (float(rho), float(nu)), float(ll_val), float(aic)

    tau, _ = kendalltau(u_np[:, 0], u_np[:, 1])
    tau = 0.0 if not np.isfinite(tau) else float(tau)
    rho = math.sin(0.5 * math.pi * max(min(tau, 0.999), -0.999))
    rho = float(np.clip(rho, -0.999999, 0.999999))

    # Candidate nu grid; favor lower nu resolution near 2 where tails change rapidly.
    nu_grid = np.array([2.2, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0], dtype=np.float64)

    one_minus_rho2 = max(1.0 - rho * rho, 1e-12)
    log_const_nu = None

    best = {"aic": float("inf"), "ll": float("-inf"), "nu": 50.0}
    x_u = u_np[:, 0]
    y_u = u_np[:, 1]

    for nu in nu_grid:
        nu = float(np.clip(nu, 2.1, 50.0))
        x = t.ppf(x_u, nu)
        y = t.ppf(y_u, nu)

        # Guard against numerical issues in extreme tails.
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            continue

        quad = (x * x + y * y - 2.0 * rho * x * y) / (nu * one_minus_rho2)
        quad = np.maximum(quad, 0.0)

        # log constant term:
        # log Γ((nu+2)/2) + log Γ(nu/2) - 2 log Γ((nu+1)/2) - 0.5 log(1-rho^2)
        log_gamma_term = (
            math.lgamma((nu + 2.0) / 2.0)
            + math.lgamma(nu / 2.0)
            - 2.0 * math.lgamma((nu + 1.0) / 2.0)
        )
        log_const = log_gamma_term - 0.5 * math.log(one_minus_rho2)

        log_c = (
            log_const
            - 0.5 * (nu + 2.0) * np.log1p(quad)
            + 0.5 * (nu + 1.0) * (np.log1p((x * x) / nu) + np.log1p((y * y) / nu))
        )
        if not np.all(np.isfinite(log_c)):
            continue

        ll = float(np.sum(log_c))
        k = 2
        aic = 2.0 * k - 2.0 * ll
        if aic < best["aic"]:
            best = {"aic": aic, "ll": ll, "nu": nu}

    # If all candidates failed, fall back.
    nu_best = float(best["nu"])
    ll_val = float(best["ll"]) if np.isfinite(best["ll"]) else 0.0
    aic = float(best["aic"]) if np.isfinite(best["aic"]) else float("inf")
    return (float(rho), float(nu_best)), ll_val, aic


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
        
        # Keep a valid gradient path near independence so optimization can
        # recover gracefully instead of crashing on a constant objective.
        if float(torch.abs(theta).item()) < 1e-6:
            log_pdf = torch.zeros(u.shape[0], dtype=dtype, device=device) + 0.0 * theta.squeeze()
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
        
        # Gumbel copula log-density computation.
        # Use a stable log-form consistent with `copulapdf` below.
        log_u1 = torch.log(u1)
        log_u2 = torch.log(u2)
        
        neg_log_u1_pow = torch.pow(-log_u1, theta)
        neg_log_u2_pow = torch.pow(-log_u2, theta)
        sum_pow = neg_log_u1_pow + neg_log_u2_pow
        
        # Clamp to avoid numerical issues
        sum_pow = torch.clamp(sum_pow, min=1e-15)
        sum_pow_1_theta = torch.pow(sum_pow, 1.0/theta)
        
        # Density:
        #   c(u,v) = exp(-s) * t^(2/theta - 2) * (-log u)^(theta-1) * (-log v)^(theta-1)
        #            * (theta - 1 + s) / (u v),
        # where t = (-log u)^theta + (-log v)^theta and s = t^(1/theta).
        #
        # Log-density (stable form):
        #   log c = -s - (2 - 1/theta) log(t)
        #           + (theta-1)[log(-log u)+log(-log v)] + log(theta-1+s) - log u - log v.
        log_pdf = (
            -(2.0 - 1.0 / theta) * torch.log(sum_pow)
            + (theta - 1.0) * (torch.log(-log_u1) + torch.log(-log_u2))
            + torch.log(theta - 1.0 + sum_pow_1_theta)
            - sum_pow_1_theta
            - log_u1
            - log_u2
        )
        
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
# JOE COPULA
################################################

def joe_kendalltau_to_theta(tau):
    """
    Approximate relationship between Kendall's tau and Joe copula parameter.

    For the Joe copula with parameter theta >= 1:
      tau = 1 + 4/theta^2 * integral_0^1 t*log(t)*(1-t)^{2(1-theta)/theta} dt.

    No closed-form inversion exists; we use a Gumbel-like approximation
    theta ~ 1/(1-tau) as a starting point for optimisation.
    """
    if tau <= 0.0:
        return 1.0001
    if tau >= 0.99:
        tau = 0.99
    theta = 1.0 / (1.0 - tau)
    return max(theta, 1.0001)


def fit_joe(u: torch.Tensor):
    """
    Fit a Joe copula using gradient-based optimization.

    Joe copula CDF:
        C(u,v) = 1 - [(1-u)^theta + (1-v)^theta - (1-u)^theta (1-v)^theta]^{1/theta}
    where theta >= 1. theta=1 gives independence.

    Density (log-form used for stability):
        log c = (1/theta - 2) log S + (theta-1)[log(1-u) + log(1-v)]
                + log(theta - 1 + S)
    where S = (1-u)^theta + (1-v)^theta - (1-u)^theta (1-v)^theta.
    """
    device = u.device
    dtype = u.dtype

    u_np = u.cpu().numpy()
    tau_kendall, _ = kendalltau(u_np[:, 0], u_np[:, 1])
    theta_init_val = joe_kendalltau_to_theta(max(tau_kendall, 0.01))

    theta_init = torch.tensor([theta_init_val], dtype=dtype, device=device, requires_grad=True)

    lr = 0.05
    conv_tol = 1e-3
    max_iter = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-6

    m = torch.zeros_like(theta_init)
    v = torch.zeros_like(theta_init)
    prev_err = torch.tensor(float('inf'), dtype=dtype, device=device)

    for iter_num in range(max_iter):
        if theta_init.grad is not None:
            theta_init.grad.zero_()

        theta = torch.clamp(theta_init, 1.001, 30.0)

        uv_clamped = torch.clamp(u, 1e-9, 1 - 1e-9)
        u1_bar = 1.0 - uv_clamped[:, 0]
        u2_bar = 1.0 - uv_clamped[:, 1]

        a = torch.pow(u1_bar, theta)
        b = torch.pow(u2_bar, theta)
        S = a + b - a * b
        S = torch.clamp(S, min=1e-14)

        log_pdf = (
            (1.0 / theta - 2.0) * torch.log(S)
            + (theta - 1.0) * (torch.log(u1_bar) + torch.log(u2_bar))
            + torch.log(theta - 1.0 + S)
        )

        log_pdf = torch.where(
            torch.isfinite(log_pdf),
            log_pdf,
            torch.tensor(-30.0, dtype=dtype, device=device),
        )

        err = -torch.sum(log_pdf)

        if torch.abs(err - prev_err) < conv_tol:
            break

        err.backward()

        grad = theta_init.grad
        iter1 = float(iter_num + 1)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        m_hat = m / (1 - beta1 ** iter1) + (1 - beta1) * grad / (1 - beta1 ** iter1)
        v_hat = v / (1 - beta2 ** iter1)

        with torch.no_grad():
            theta_init -= lr * m_hat / (torch.sqrt(v_hat) + eps)

        prev_err = err.clone()

    theta_final = torch.clamp(theta_init, 1.001, 30.0).item()
    ll_val = -err.item()

    k = 1
    aic = 2 * k - 2 * ll_val

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
        for family in families:
            fam = _normalize_family_name(family)
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
                emp_corr = _safe_empirical_corr(u_vals[:, 0], u_vals[:, 1])
                
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
            elif fam=='joe':
                theta, ll_, aic_ = fit_joe(data_i)
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
    fam = _normalize_family_name(cop_p.family)
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
        # Gaussian copula density:
        #   c(u,v) = 1/sqrt(1-r^2) * exp((2 r z1 z2 - r^2(z1^2+z2^2)) / (2(1-r^2))).
        denom = max(one_m_r2, 1e-12)
        logc = -0.5 * math.log(denom)
        quad = (2.0 * r * z1 * z2 - (r * r) * (z1 * z1 + z2 * z2)) / (2.0 * denom)
        logc_t = torch.tensor(logc, dtype=uv_clamped.dtype, device=uv_clamped.device)
        return torch.exp(torch.clamp(logc_t + quad, -30.0, 30.0))

    elif fam=='student':
        # param => (rho, df)
        if isinstance(param, (list, tuple)) and len(param) >= 2:
            rho, nu = float(param[0]), float(param[1])
        else:
            rho, nu = 0.0, 4.0
        
        rho = max(min(rho, 0.999999), -0.999999)
        nu = max(nu, 2.1)
        
        # Convert to t-distribution quantiles (vectorized via SciPy).
        u_np = uv_clamped.detach().cpu().numpy()
        t_quantiles_np = t.ppf(u_np, nu)
        t_quantiles = torch.tensor(t_quantiles_np, dtype=uv_clamped.dtype, device=uv_clamped.device)
        
        t1, t2 = t_quantiles[:, 0], t_quantiles[:, 1]
        one_minus_rho2 = 1 - rho**2
        
        # Student t copula PDF
        nu_t = torch.tensor(nu, dtype=uv_clamped.dtype, device=uv_clamped.device)
        log_gamma_term = (
            torch.lgamma((nu_t + 2.0) / 2.0)
            + torch.lgamma(nu_t / 2.0)
            - 2.0 * torch.lgamma((nu_t + 1.0) / 2.0)
        )
        
        quad_form = (t1**2 + t2**2 - 2*rho*t1*t2) / (nu * one_minus_rho2)
        
        # Correct log copula density:
        # log c(u,v) = const - (nu+2)/2 * log(1 + quad) + (nu+1)/2 * [log(1+x^2/nu) + log(1+y^2/nu)]
        log_pdf = (
            log_gamma_term
            - 0.5 * math.log(max(one_minus_rho2, 1e-12))
            - 0.5 * (nu + 2.0) * torch.log1p(torch.clamp(quad_form, min=0.0))
            + 0.5 * (nu + 1.0) * (torch.log1p((t1**2) / nu) + torch.log1p((t2**2) / nu))
        )
        
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
        
        # Gumbel copula density includes the copula value C(u,v)=exp(-s) factor.
        C = torch.exp(-sum_pow_1_theta)
        # Correct density (matches symbolic differentiation):
        #   c(u,v) = exp(-s) * (-log u)^(theta-1) * (-log v)^(theta-1)
        #            * (theta - 1 + s) / (u v * t^(2 - 1/theta)),
        # with t = (-log u)^theta + (-log v)^theta and s = t^(1/theta).
        pdf = (
            C
            * torch.pow(-log_u1, theta - 1.0)
            * torch.pow(-log_u2, theta - 1.0)
            * (theta - 1.0 + sum_pow_1_theta)
            / (u1 * u2 * torch.pow(sum_pow, 2.0 - 1.0 / theta))
        )
        return torch.clamp(pdf, 1e-15, 1e15)

    elif fam=='joe':
        theta = float(param)
        theta = max(theta, 1.001)

        u1, u2 = uv_clamped[:, 0], uv_clamped[:, 1]
        u1_bar = 1.0 - u1
        u2_bar = 1.0 - u2

        a = torch.pow(u1_bar, theta)
        b = torch.pow(u2_bar, theta)
        S = torch.clamp(a + b - a * b, min=1e-14)

        log_pdf = (
            (1.0 / theta - 2.0) * torch.log(S)
            + (theta - 1.0) * (torch.log(u1_bar) + torch.log(u2_bar))
            + torch.log(theta - 1.0 + S)
        )
        return torch.exp(torch.clamp(log_pdf, -30.0, 30.0))

    else:
        # unknown
        return torch.zeros(uv.shape[0], device=uv.device)


def copulaccdf(cop_p, uv: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the conditional CDF / h-function h(v|u) of a bivariate copula.

    Notes
    -----
    For continuous marginals, the h-function is
      h(v|u) = ∂C(u,v) / ∂u,
    and is required for vine construction (pseudo-observations propagation) and
    sequential sampling.

    Parameters
    ----------
    cop_p:
        Copula object with fields `.family` and `.theta`.
    uv:
        Tensor of shape [N,2] with values in [0,1].
    """
    fam = _normalize_family_name(cop_p.family)
    param = cop_p.theta
    uv_clamped = torch.clamp(uv, 1e-9, 1 - 1e-9)

    if fam=='ind':
        # Independence: h(v|u) = v.
        return uv_clamped[:, 1]

    elif fam=='gaussian':
        # Handle case where param might be a list
        # Analytical h-function for Gaussian copula:
        # h(v|u) = Phi( (Phi^{-1}(v) - rho * Phi^{-1}(u)) / sqrt(1 - rho^2) )
        if isinstance(param, list):
            rho = float(param[0])
        else:
            rho = float(param)
        r = max(min(rho, 0.999999), -0.999999)
        normal_dist = torch.distributions.Normal(0., 1.)

        # Transform to normal space
        z1 = normal_dist.icdf(uv_clamped[:, 0])
        z2 = normal_dist.icdf(uv_clamped[:, 1])

        # Conditional distribution: Z2|Z1 ~ N(rho*Z1, sqrt(1-rho^2))
        one_minus_r2 = 1.0 - r * r
        if one_minus_r2 < 1e-12:
            one_minus_r2 = 1e-12
        z_std = (z2 - r * z1) / math.sqrt(one_minus_r2)

        return normal_dist.cdf(z_std).clamp(1e-9, 1 - 1e-9)

    elif fam=='student':
        # Student-t copula conditional CDF (h-function):
        # h(v|u; rho, nu) = t_{nu+1}( (t_nu^{-1}(v) - rho * t_nu^{-1}(u)) /
        #                              sqrt((nu + t_nu^{-1}(u)^2) * (1 - rho^2) / (nu + 1)) )
        if isinstance(param, (list, tuple)) and len(param) >= 2:
            rho, nu = float(param[0]), float(param[1])
        else:
            rho, nu = 0.0, 4.0
        rho = max(min(rho, 0.999999), -0.999999)
        nu = max(nu, 2.1)

        u_np = uv_clamped.cpu().numpy()
        # t_nu^{-1}(u) and t_nu^{-1}(v)
        x = t.ppf(u_np[:, 0], nu)
        y = t.ppf(u_np[:, 1], nu)

        # Numerator: t_nu^{-1}(v) - rho * t_nu^{-1}(u)
        numerator = y - rho * x

        # Denominator: sqrt((nu + x^2) * (1 - rho^2) / (nu + 1))
        denominator = np.sqrt((nu + x**2) * (1 - rho**2) / (nu + 1))
        denominator = np.maximum(denominator, 1e-15)

        # h(v|u) = t_{nu+1}(numerator / denominator)
        z = numerator / denominator
        h_val = t.cdf(z, nu + 1)

        return torch.tensor(h_val, dtype=uv.dtype, device=uv.device).clamp(1e-9, 1 - 1e-9)

    elif fam=='clayton':
        # Clayton: C(u,v) = (u^{-a} + v^{-a} - 1)^(-1/a), a>0.
        # h(v|u) = ∂C/∂u = u^{-a-1} * (u^{-a} + v^{-a} - 1)^(-1/a - 1).
        alpha = max(float(param), 1e-6)
        u_ = uv_clamped[:, 0]
        v_ = uv_clamped[:, 1]
        u_m_alpha = torch.pow(u_, -alpha)
        v_m_alpha = torch.pow(v_, -alpha)
        sum_ = torch.clamp(u_m_alpha + v_m_alpha - 1.0, min=1e-14)
        h_ = torch.pow(u_, -(alpha + 1.0)) * torch.pow(sum_, -(1.0 / alpha + 1.0))
        return torch.clamp(h_, 1e-9, 1.0 - 1e-9)

    elif fam=='claytonrot90':
        # 90-degree rotation: C^{90}(u,v) = v - C(1-u, v).
        # Then h^{90}(v|u) = ∂C^{90}/∂u = h(v|1-u) for the base copula.
        uv_flip = uv_clamped.clone()
        uv_flip[:, 0] = 1.0 - uv_clamped[:, 0]
        from copy import deepcopy
        cop_p_temp = deepcopy(cop_p)
        cop_p_temp.family = 'clayton'
        return copulaccdf(cop_p_temp, uv_flip)

    elif fam=='frank':
        theta = float(param)
        if abs(theta) < 1e-6:
            return uv_clamped[:, 1]

        u1, v = uv_clamped[:, 0], uv_clamped[:, 1]

        exp_neg_theta_u1 = torch.exp(-theta * u1)
        expm1_neg_theta_v = torch.expm1(-theta * v)
        expm1_neg_theta = torch.expm1(torch.tensor(-theta, dtype=uv.dtype, device=uv.device))

        num = exp_neg_theta_u1 * expm1_neg_theta_v
        den = expm1_neg_theta + (exp_neg_theta_u1 - 1.0) * expm1_neg_theta_v
        den = torch.where(
            den.abs() < 1e-15,
            torch.where(den >= 0, torch.full_like(den, 1e-15), torch.full_like(den, -1e-15)),
            den,
        )
        h_ = num / den
        return torch.clamp(h_, 1e-9, 1.0 - 1e-9)

    elif fam=='gumbel':
        theta = float(param)
        if theta <= 1.0 + 1e-6:
            return uv_clamped[:, 1]
        theta = max(theta, 1.0 + 1e-6)

        u, v = uv_clamped[:, 0], uv_clamped[:, 1]
        log_u = torch.log(u)
        log_v = torch.log(v)

        a = torch.pow(-log_u, theta)
        b = torch.pow(-log_v, theta)
        sum_pow = torch.clamp(a + b, min=1e-15)
        sum_pow_1_theta = torch.pow(sum_pow, 1.0 / theta)
        C = torch.exp(-sum_pow_1_theta)

        # h(v|u) = ∂C/∂u = C * (a+b)^{1/theta - 1} * (-log u)^{theta-1} / u.
        term1 = torch.pow(sum_pow, 1.0 / theta - 1.0)
        term2 = torch.pow(-log_u, theta - 1.0) / torch.clamp(u, min=1e-12)
        h_ = C * term1 * term2
        return torch.clamp(h_, 1e-9, 1.0 - 1e-9)

    elif fam=='joe':
        # Joe copula h-function:
        # h(v|u) = S^{1/theta - 1} * (1-u)^{theta-1} * (1 - (1-v)^theta)
        # where S = (1-u)^theta + (1-v)^theta - (1-u)^theta (1-v)^theta.
        theta = float(param)
        if theta <= 1.0 + 1e-6:
            return uv_clamped[:, 1]
        theta = max(theta, 1.0 + 1e-6)

        u_, v_ = uv_clamped[:, 0], uv_clamped[:, 1]
        u_bar = 1.0 - u_
        v_bar = 1.0 - v_

        a = torch.pow(u_bar, theta)
        b = torch.pow(v_bar, theta)
        S = torch.clamp(a + b - a * b, min=1e-14)

        h_ = torch.pow(S, 1.0 / theta - 1.0) * torch.pow(u_bar, theta - 1.0) * (1.0 - b)
        return torch.clamp(h_, 1e-9, 1.0 - 1e-9)

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
    fam = _normalize_family_name(cop_p.family)
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
                logger.warning(
                    "%d extreme values in Gaussian h-function: "
                    "x range: [%.2f, %.2f], e range: [%.2f, %.2f], "
                    "y range: [%.2f, %.2f], rho: %.4f",
                    extreme_count,
                    ext_x.min().item(), ext_x.max().item(),
                    ext_e.min().item(), ext_e.max().item(),
                    ext_y.min().item(), ext_y.max().item(),
                    r
                )
        
        return torch.clamp(v2, 1e-9, 1-1e-9)

    elif fam=='student':
        # Inverse h-function for Student-t copula:
        # v = t_nu( rho * t_nu^{-1}(u1) + t_{nu+1}^{-1}(w) *
        #           sqrt((nu + t_nu^{-1}(u1)^2) * (1 - rho^2) / (nu + 1)) )
        if isinstance(param, (list, tuple)) and len(param) >= 2:
            rho, nu = float(param[0]), float(param[1])
        else:
            rho, nu = 0.0, 4.0
        rho = max(min(rho, 0.999999), -0.999999)
        nu = max(nu, 2.1)

        u_np = uv_clamped.cpu().numpy()
        u1_np = u_np[:, 0]
        w_np = u_np[:, 1]

        # t_nu^{-1}(u1)
        x = t.ppf(u1_np, nu)

        # t_{nu+1}^{-1}(w)
        e = t.ppf(w_np, nu + 1)

        # sqrt((nu + x^2) * (1 - rho^2) / (nu + 1))
        scale = np.sqrt((nu + x**2) * (1 - rho**2) / (nu + 1))

        # y = rho * x + e * scale
        y = rho * x + e * scale

        # v = t_nu(y)
        v = t.cdf(y, nu)

        return torch.tensor(v, dtype=uv.dtype, device=uv.device).clamp(1e-9, 1 - 1e-9)

    elif fam=='clayton':
        # Invert h(v|u) = d/du C(u,v).
        #
        # Clayton copula:
        #   C(u,v) = (u^{-a} + v^{-a} - 1)^{-1/a},  a>0
        # h(v|u) = u^{-a-1} (u^{-a} + v^{-a} - 1)^{-1/a - 1}
        #
        # Solving w = h(v|u) for v gives:
        #   v = [ 1 + u^{-a} ( w^{-a/(1+a)} - 1 ) ]^{-1/a}
        alpha = max(float(param), 1e-6)
        u1 = uv_clamped[:, 0]
        w = uv_clamped[:, 1]

        u1_m_alpha = torch.pow(u1, -alpha)
        w_pow = torch.pow(w, -alpha / (1.0 + alpha))
        val = 1.0 + u1_m_alpha * (w_pow - 1.0)
        val = torch.clamp(val, min=1e-14)
        u2 = torch.pow(val, -1.0 / alpha)
        return torch.clamp(u2, 1e-9, 1 - 1e-9)

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
        w = uv_clamped[:, 1]
        exp_neg_theta_u1 = torch.exp(-theta * u1)
        expm1_neg_theta = torch.expm1(torch.tensor(-theta, dtype=u1.dtype, device=u1.device))

        # Frank h-function:
        #   h(v|u) = A B / (C + (A - 1) B),
        # where A = exp(-theta u), B = expm1(-theta v), C = expm1(-theta).
        # Solving for B yields
        #   B = w C / (A (1 - w) + w),
        # then v = -log1p(B) / theta.
        denom = exp_neg_theta_u1 * (1.0 - w) + w
        denom = torch.clamp(denom, min=1e-15)
        b_term = (w * expm1_neg_theta) / denom
        b_term = torch.clamp(b_term, min=-1.0 + 1e-12)
        u2 = -torch.log1p(b_term) / theta

        return torch.clamp(u2, 1e-9, 1-1e-9)

    elif fam=='gumbel':
        theta = float(param)
        if theta <= 1.0 + 1e-6:
            # Near-independence.
            return uv_clamped[:, 1]

        # No simple closed form for the inverse h-function; use bisection on v in (0,1)
        # to solve copulaccdf([u, v]) = w.
        u1 = uv_clamped[:, 0]
        w = uv_clamped[:, 1]

        lo = torch.full_like(w, 1e-9)
        hi = torch.full_like(w, 1.0 - 1e-9)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            uv_mid = torch.stack([u1, mid], dim=1)
            h = copulaccdf(cop_p, uv_mid)
            h = torch.nan_to_num(h, nan=0.5, posinf=1.0 - 1e-9, neginf=1e-9).clamp(1e-9, 1 - 1e-9)
            lo = torch.where(h < w, mid, lo)
            hi = torch.where(h >= w, mid, hi)

        u2 = 0.5 * (lo + hi)
        return torch.clamp(u2, 1e-9, 1 - 1e-9)

    elif fam=='joe':
        theta = float(param)
        if theta <= 1.0 + 1e-6:
            return uv_clamped[:, 1]

        # No closed form; use bisection on v in (0,1) to solve h(v|u) = w.
        u1 = uv_clamped[:, 0]
        w = uv_clamped[:, 1]

        lo = torch.full_like(w, 1e-9)
        hi = torch.full_like(w, 1.0 - 1e-9)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            uv_mid = torch.stack([u1, mid], dim=1)
            h = copulaccdf(cop_p, uv_mid)
            h = torch.nan_to_num(h, nan=0.5, posinf=1.0 - 1e-9, neginf=1e-9).clamp(1e-9, 1 - 1e-9)
            lo = torch.where(h < w, mid, lo)
            hi = torch.where(h >= w, mid, hi)

        u2 = 0.5 * (lo + hi)
        return torch.clamp(u2, 1e-9, 1 - 1e-9)

    else:
        # unknown => just return the second
        return uv_clamped[:,1]
