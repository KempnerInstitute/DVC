# src/param/cond_copula.py
import torch
import math
from torch.distributions import Normal, StudentT

def copulapdf(family: str, theta, uv: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the bivariate copula PDF at uv (shape [N,2]) for the specified family.
    Supported families: 'ind', 'gaussian', 'clayton', 'claytonrot90', 'student'.
    """
    uv = torch.clamp(uv, 1e-7, 1 - 1e-7)
    N = uv.shape[0]
    device = uv.device
    if family.lower() == 'ind':
        return torch.ones(N, device=device, dtype=uv.dtype)
    elif family.lower() == 'gaussian':
        r = theta
        dist = Normal(0.0, 1.0)
        x = dist.icdf(uv[:, 0])
        y = dist.icdf(uv[:, 1])
        denom = torch.sqrt(1 - r ** 2)
        exponent = - (r ** 2 * (x ** 2 + y ** 2) - 2 * r * x * y) / (2 * (1 - r ** 2))
        c = torch.exp(exponent) / denom
        return c
    elif family.lower() == 'clayton':
        t = theta
        u_val = uv[:, 0]
        v_val = uv[:, 1]
        num = (1 + t) * (u_val * v_val) ** (-1 - t)
        denom = (u_val ** (-t) + v_val ** (-t) - 1) ** (1 / t + 2)
        return num / denom
    elif family.lower() == 'claytonrot90':
        t = theta
        u_val = uv[:, 0]
        v_val = 1 - uv[:, 1]
        num = (1 + t) * (u_val * v_val) ** (-1 - t)
        denom = (u_val ** (-t) + v_val ** (-t) - 1) ** (1 / t + 2)
        return num / denom
    elif family.lower() == 'student':
        # theta = [rho, nu]
        rho = theta[0]
        nu = theta[1]
        dist = StudentT(nu)
        x = dist.icdf(uv[:, 0])
        y = dist.icdf(uv[:, 1])
        # Compute the bivariate t copula density using standard formula:
        factor = torch.lgamma((nu + 2) / 2) - torch.lgamma(nu / 2)
        log_denom = 0.5 * torch.log(1 - rho ** 2)
        z = - ((nu + 2) / 2) * torch.log(1 + (x ** 2 - 2 * rho * x * y + y ** 2) / (nu * (1 - rho ** 2)))
        log_pdf = factor - log_denom + z
        return torch.exp(log_pdf)
    else:
        raise ValueError("Unknown copula family.")

def copulaccdf(family: str, theta, uv: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the conditional copula CDF (i.e. the cdf of the conditional density)
    for the given family.
    For gaussian: F(u|v)=Phi((Phi^{-1}(u)-rho*Phi^{-1}(v))/sqrt(1-rho^2)).
    For clayton-type, use the corresponding formulation.
    """
    uv = torch.clamp(uv, 1e-7, 1 - 1e-7)
    device = uv.device
    if family.lower() == 'ind':
        return uv[:, 0]
    elif family.lower() == 'gaussian':
        r = theta
        dist = Normal(0.0, 1.0)
        x = dist.icdf(uv[:, 0])
        y = dist.icdf(uv[:, 1])
        denom = torch.sqrt(1 - r ** 2)
        tmp = (x - r * y) / denom
        return dist.cdf(tmp)
    elif family.lower() == 'clayton':
        t = theta
        u_val = uv[:, 0]
        v_val = uv[:, 1]
        # Using known formula for conditional distribution of Clayton copula:
        return v_val ** (-1 - t) * (u_val ** (-t) + v_val ** (-t) - 1) ** (-1 / t - 1)
    elif family.lower() == 'claytonrot90':
        t = theta
        u_val = uv[:, 0]
        v_val = 1 - uv[:, 1]
        return v_val ** (-1 - t) * (u_val ** (-t) + v_val ** (-t) - 1) ** (-1 / t - 1)
    elif family.lower() == 'student':
        # Use a simplified conditional t-CDF
        rho = theta[0]
        nu = theta[1]
        dist = StudentT(nu)
        x = dist.icdf(uv[:, 0])
        y = dist.icdf(uv[:, 1])
        denom = torch.sqrt((nu + y ** 2) * (1 - rho ** 2) / (nu + 1))
        tmp = (x - rho * y) / denom
        return dist.cdf(tmp)
    else:
        raise ValueError("Unknown copula family.")