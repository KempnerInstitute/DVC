import torch
from utils.tensor_op import replace_nan_with, replace_inf_with
from param.margin_pdf import *

############################################ GAUSSIAN COST FUNCTION ######################################

def gaussian_cost(u, theta_par):
    """Compute negative log-likelihood for Gaussian copula"""
    p = gaussian_pdf(u, theta_par)
    p = replace_nan_with(p, torch.tensor(1.0, dtype=u.dtype, device=u.device))
    eps = torch.finfo(u.dtype).eps
    err = -torch.sum(torch.log(p + eps), dim=0)
    return err

############################################ STUDENT COST FUNCTION ######################################

def student_cost(u, theta):
    """Compute negative log-likelihood for Student-t copula"""
    p = student_pdf(u, theta)
    eps = torch.finfo(u.dtype).eps
    err = -torch.sum(torch.log(p + eps), dim=0)
    return err

############################################ CLAYTON COST FUNCTION ######################################

def clayton_cost(u, theta_cla1):
    """Compute negative log-likelihood for Clayton copula"""
    p = clayton_pdf(u, theta_cla1)
    p = replace_nan_with(p, torch.tensor(1.0, dtype=u.dtype, device=u.device))
    p = replace_inf_with(p, torch.finfo(u.dtype).max)
    eps = torch.finfo(u.dtype).eps
    err = -torch.sum(torch.log(p + eps), dim=0)
    return err

############################################ CLAYTON ROT 90 COST FUNCTION ######################################

def claytonrot90_cost(u, theta_cla1):
    """Compute negative log-likelihood for Clayton rotated 90 degrees copula"""
    p = claytonrot90_pdf(u, theta_cla1)
    p = replace_nan_with(p, torch.tensor(1.0, dtype=u.dtype, device=u.device))
    p = replace_inf_with(p, torch.finfo(u.dtype).max)
    eps = torch.finfo(u.dtype).eps
    err = -torch.sum(torch.log(p + eps), dim=0)
    return err 