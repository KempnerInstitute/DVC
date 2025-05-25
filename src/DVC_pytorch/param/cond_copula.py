import torch
import torch.distributions as dist
import numpy as np
from scipy import stats

from utils.bijector import NormalCDF
from param.margin_pdf import *

############################################ COPULA PDF ############################################

def copulapdf(vine_par, u):
    """
    Compute copula PDF
    
    Args:
        vine_par: Vine parameter object with family and theta attributes
        u: Data tensor of shape (n_samples, 2)
        
    Returns:
        c: Copula PDF values
    """
    device = u.device if torch.is_tensor(u) else torch.device('cpu')
    dtype = u.dtype if torch.is_tensor(u) else torch.float32
    
    if not torch.is_tensor(u):
        u = torch.tensor(u, dtype=dtype, device=device)
    
    if vine_par.family == 'ind':
        c = torch.ones(u.shape[0], dtype=dtype, device=device)
    elif vine_par.family == 'gaussian':
        theta = torch.tensor(vine_par.theta, dtype=dtype, device=device)
        c = gaussian_pdf(u, theta)
    elif vine_par.family == 'student':
        theta = torch.tensor(vine_par.theta, dtype=dtype, device=device)
        c = student_pdf(u, theta)
    elif vine_par.family == 'clayton':
        theta = torch.tensor(vine_par.theta, dtype=dtype, device=device)
        c = clayton_pdf(u, theta)
    elif vine_par.family == 'claytonrot90':
        theta = torch.tensor(vine_par.theta, dtype=dtype, device=device)
        c = claytonrot90_pdf(u, theta)
    else:
        raise ValueError(f"Unknown copula family: {vine_par.family}")
    
    return c

############################################ COPULA CONDITIONED CDF ############################################

def copulaccdf(vine_par, u):
    """
    Compute conditional CDF: C(u1|u2)
    
    Args:
        vine_par: Vine parameter object with family and theta attributes
        u: Data array of shape (n_samples, 2)
        
    Returns:
        c: Conditional CDF values
    """
    # Handle bounds
    u = np.clip(u, 1e-7, 1 - 1e-7)
    
    loc = 0
    scale = 1
    c = np.zeros(u.shape[0], dtype=u.dtype)
    
    if vine_par.family == 'ind':
        c = u[:, 0]
    elif vine_par.family == 'gaussian':
        bijector = NormalCDF(loc, scale)
        x_tensor = bijector.forward(u)
        x = x_tensor.cpu().numpy() if torch.is_tensor(x_tensor) else x_tensor
        theta = vine_par.theta
        tmp = (x[:, 0] - theta * x[:, 1]) / np.sqrt(1 - theta**2)
        c_tensor = bijector.inverse(tmp)
        c = c_tensor.cpu().numpy() if torch.is_tensor(c_tensor) else c_tensor
    elif vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
        tmp = np.sqrt((theta2 + 1) / (theta2 + x[:, 1]**2)) * \
              (x[:, 0] - theta1 * x[:, 1]) / np.sqrt(1 - theta1**2)
        c = stats.t.cdf(tmp, theta2 + 1, loc, scale)
    elif vine_par.family == 'clayton':
        theta = vine_par.theta
        if theta == 0:
            c = u[:, 0]
        else:
            c = np.maximum(u[:, 1]**(-1 - theta) * 
                          (u[:, 0]**(-theta) + u[:, 1]**(-theta) - 1)**(-1 - 1/theta), 0)
    elif vine_par.family == 'claytonrot90':
        theta = vine_par.theta
        if theta == 0:
            c = u[:, 0]
        else:
            c = np.maximum((1 - u[:, 1])**(-1 - theta) * 
                          (u[:, 0]**(-theta) + (1 - u[:, 1])**(-theta) - 1)**(-1 - 1/theta), 0)
    else:
        raise ValueError(f"Unknown copula family: {vine_par.family}")
    
    return c

############################################ COPULA INVERSE CONDITIONED CDF ############################################

def copulainvccdf(vine_par, u):
    """
    Compute inverse conditional CDF: C^{-1}(u1|u2)
    
    Args:
        vine_par: Vine parameter object with family and theta attributes
        u: Data array of shape (n_samples, 2)
        
    Returns:
        c: Inverse conditional CDF values
    """
    # Handle bounds
    u = np.clip(u, 1e-7, 1 - 1e-7)
    
    loc = 0
    scale = 1
    c = np.zeros(u.shape[0], dtype=u.dtype)
    
    if vine_par.family == 'ind':
        c = u[:, 0]
    elif vine_par.family == 'gaussian':
        bijector = NormalCDF(loc, scale)
        x_tensor = bijector.forward(u)
        x = x_tensor.cpu().numpy() if torch.is_tensor(x_tensor) else x_tensor
        theta = vine_par.theta
        tmp = x[:, 0] * np.sqrt(1 - theta**2) + theta * x[:, 1]
        c_tensor = bijector.inverse(tmp)
        c = c_tensor.cpu().numpy() if torch.is_tensor(c_tensor) else c_tensor
    elif vine_par.family == 'student':
        theta1 = vine_par.theta[0]
        theta2 = vine_par.theta[1]
        x = stats.t.ppf(u, theta2, loc, scale)
        param = theta2 + 1
        tmp_inv = stats.t.ppf(u[:, 0], param, loc, scale)
        tmp = np.sqrt(((1 - theta1**2) * (theta2 + x[:, 1]**2)) / (theta2 + 1)) * tmp_inv + theta1 * x[:, 1]
        c = stats.t.cdf(tmp, theta2, loc, scale)
    elif vine_par.family == 'clayton':
        theta = vine_par.theta
        if theta == 0:
            c = u[:, 0]
        else:
            c = (1 - u[:, 1]**(-theta) + 
                 (u[:, 0] * (u[:, 1]**(1 + theta)))**(-theta/(1 + theta)))**(-1/theta)
    elif vine_par.family == 'claytonrot90':
        theta = vine_par.theta
        if theta == 0:
            c = u[:, 0]
        else:
            c = (1 - (1 - u[:, 1])**(-theta) + 
                 (u[:, 0] * ((1 - u[:, 1])**(1 + theta)))**(-theta/(1 + theta)))**(-1/theta)
    else:
        raise ValueError(f"Unknown copula family: {vine_par.family}")
    
    return c

# PyTorch versions for when u is a tensor

def copulaccdf_torch(vine_par, u):
    """
    PyTorch version of copulaccdf
    
    Args:
        vine_par: Vine parameter object
        u: Data tensor of shape (n_samples, 2)
        
    Returns:
        c: Conditional CDF values as tensor
    """
    device = u.device
    dtype = u.dtype
    
    # Convert to numpy, compute, then convert back
    u_np = u.cpu().numpy()
    c_np = copulaccdf(vine_par, u_np)
    c = torch.tensor(c_np, dtype=dtype, device=device)
    
    return c

def copulainvccdf_torch(vine_par, u):
    """
    PyTorch version of copulainvccdf
    
    Args:
        vine_par: Vine parameter object
        u: Data tensor of shape (n_samples, 2)
        
    Returns:
        c: Inverse conditional CDF values as tensor
    """
    device = u.device
    dtype = u.dtype
    
    # Convert to numpy, compute, then convert back
    u_np = u.cpu().numpy()
    c_np = copulainvccdf(vine_par, u_np)
    c = torch.tensor(c_np, dtype=dtype, device=device)
    
    return c 