import torch
import torch.distributions as dist
from scipy.special import gammaln
from scipy.stats import t
from utils.tensor_op import replace_nan_inf, update_tensor2D
import math as m
import numpy as np

############################################ GAUSSIAN MARGIN PDF ######################################

def gaussian_pdf(u, theta_par):
    """Compute Gaussian copula PDF"""
    device = u.device
    dtype = u.dtype
    
    norm_dist = dist.Normal(torch.tensor(0., dtype=dtype, device=device), 
                           torch.tensor(1., dtype=dtype, device=device))
    x = norm_dist.icdf(u)
    
    # Handle dimensions properly
    if u.dim() == 2:
        u = u.unsqueeze(-1)
        x = x.unsqueeze(-1)
    
    p = torch.exp((2 * theta_par * x[:, 0, :] * x[:, 1, :] - 
                   (theta_par**2) * (x[:, 0, :]**2 + x[:, 1, :]**2)) / 
                  (2 * (1 - theta_par**2))) / torch.sqrt(1 - theta_par**2)
    return p

############################################ CLAYTON MARGIN PDF ######################################

def clayton_pdf(u, theta):
    """Compute Clayton copula PDF"""
    device = u.device
    dtype = u.dtype
    
    # Handle dimensions
    if u.dim() == 2:
        u = u.unsqueeze(-1)
    
    p = (1 + theta) * (u[:, 0, :] * u[:, 1, :])**(-1 - theta) * \
        (u[:, 0, :]**(-theta) + u[:, 1, :]**(-theta) - 1)**(-1/theta - 2)
    
    # Handle theta=0 case
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=dtype, device=device)
    
    if theta.dim() == 0:
        theta = theta.unsqueeze(-1)
    
    # Set p=1 where theta=0
    cond = theta == 0
    if torch.any(cond):
        ind = torch.where(cond)[0]
        for i in ind:
            newval = torch.ones(u.shape[0], dtype=dtype, device=device)
            p = update_tensor2D(p, i.item(), newval)
    
    return p

############################################ CLAYTON ROT 90 MARGIN PDF ######################################

def claytonrot90_pdf(u, theta):
    """Compute Clayton rotated 90 degrees copula PDF"""
    device = u.device
    dtype = u.dtype
    
    # Handle dimensions
    if u.dim() == 2:
        u = u.unsqueeze(-1)
    
    p = (1 + theta) * (u[:, 0, :] * (1 - u[:, 1, :]))**(-1 - theta) * \
        (u[:, 0, :]**(-theta) + (1 - u[:, 1, :])**(-theta) - 1)**(-1/theta - 2)
    
    # Handle theta=0 case
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=dtype, device=device)
    
    if theta.dim() == 0:
        theta = theta.unsqueeze(-1)
    
    # Set p=1 where theta=0
    cond = theta == 0
    if torch.any(cond):
        ind = torch.where(cond)[0]
        for i in ind:
            newval = torch.ones(u.shape[0], dtype=dtype, device=device)
            p = update_tensor2D(p, i.item(), newval)
    
    return p

############################################ STUDENT MARGIN PDF ######################################

def gammaln1(x):
    """Compute log gamma function"""
    if torch.is_tensor(x):
        x_np = x.cpu().numpy()
    else:
        x_np = x
    result = gammaln(x_np)
    if torch.is_tensor(x):
        return torch.tensor(result, dtype=x.dtype, device=x.device)
    else:
        return result

def tpdf(x, vk):
    """Compute Student-t PDF"""
    device = x.device
    dtype = x.dtype
    
    term = torch.exp(gammaln1((vk + 1) / 2) - gammaln1(vk / 2))
    pi = torch.tensor(m.pi, dtype=dtype, device=device)
    y = term / (torch.sqrt(vk * pi) * (1 + (x**2) / vk) ** ((vk + 1) / 2))
    return y

def student_pdf(u, theta):
    """Compute Student-t copula PDF"""
    device = u.device
    dtype = u.dtype
    
    # Handle dimensions
    if u.dim() == 2:
        u = u.unsqueeze(-1)
    
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=dtype, device=device)
    
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    
    df = theta[:, 1]
    loc = 0
    scale = 1
    pi = torch.tensor(m.pi, dtype=dtype, device=device)
    
    # Convert to numpy for scipy.stats.t.ppf
    u_np = u.cpu().numpy()
    theta_np = theta.cpu().numpy()
    x_np = t.ppf(u_np, theta_np[:, 1], loc=loc, scale=scale)
    x = torch.tensor(x_np, dtype=dtype, device=device)
    
    factor1 = gammaln1(theta[:, 1] / 2 + 1)
    factor2 = -gammaln1(theta[:, 1] / 2) - torch.log(pi) - torch.log(theta[:, 1]) - \
              torch.log(1 - theta[:, 0]**2) / 2 - \
              torch.log(tpdf(x[:, 0, :], theta[:, 1])) - \
              torch.log(tpdf(x[:, 1, :], theta[:, 1]))
    
    factor3 = (-(theta[:, 1] + 2) / 2) * \
              torch.log(1 + (x[:, 0, :]**2 + x[:, 1, :]**2 - 
                            theta[:, 0] * x[:, 0, :] * x[:, 1, :]) / 
                       (theta[:, 1] * (1 - theta[:, 0]**2)))
    
    p = torch.exp(factor1 + factor2 + factor3)
    p = replace_nan_inf(p)
    
    return p 