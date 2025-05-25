import torch
import torch.distributions as dist
import numpy as np
from utils.prob_op import kernel_pdf2
from utils.tensor_op import update_tensor, update_tensor2D, replace_nan_inf
from pre_proc.preparation import prep_copula

########################## INFO ESTIMATION ##################################

def vine_entropy(vine, info_dict):
    """
    Estimate entropy of vine copula using Monte Carlo sampling
    
    Args:
        vine: Fitted vine copula object
        info_dict: Dictionary with:
            - alpha: Confidence level
            - cases: Number of samples per iteration
            - iterations: Maximum iterations
            
    Returns:
        infoc1: Estimated entropy
    """
    alpha = info_dict['alpha']
    cases = info_dict['cases']
    max_iter = info_dict['iterations']
    d = vine.n_cop
    
    device = vine.grid_u.ex.device
    dtype = vine.grid_u.ex.dtype
    
    # Normal distribution for confidence intervals
    norm_dist = dist.Normal(0., 1.)
    conf = norm_dist.icdf(torch.tensor(1 - alpha))
    
    mo = 0
    varsum1 = 0
    infoc1 = 0
    stderr1 = 1e+6
    stderr2 = 1e+6
    stderr_tot = 1e+6
    erreps = 1e-3
    
    mag = torch.max(vine.grid_u.ex)
    mig = torch.min(vine.grid_u.ex)
    
    while ((stderr1 >= erreps) or (stderr2 >= erreps) or (stderr_tot >= erreps)) and (mo < max_iter):
        mo = mo + 1
        
        if not vine.param:
            # Non-parametric case
            w = torch.rand(cases, d, dtype=dtype, device=device)
            w = (mag - mig) * (w - torch.min(w)) / (torch.max(w) - torch.min(w)) + mig
            
            # Generate samples from vine copula
            # Note: vine_copula_sample needs to be implemented in sampling module
            # For now, we'll use uniform samples as placeholder
            sample = torch.rand(cases, d, dtype=dtype, device=device)
            
            # Evaluate PDF at samples
            p, p_copula, plog = vine.evaluation(sample)
            
            # Convert to numpy for log2 (PyTorch doesn't have log2)
            log2pp = np.log2(p_copula.cpu().numpy())
            log2pp[p_copula.cpu().numpy() == 0] = 0
            
            # Update running average
            infoc1 = infoc1 + (np.mean(log2pp) - infoc1) / mo
            
            # Update variance sum
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf.item() * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
        else:
            # Parametric case
            # Generate samples (placeholder - needs vine_cop_par_sample implementation)
            sample = torch.rand(cases, d, dtype=dtype, device=device)
            
            # Compute PDF of samples
            p, pcop, _ = vine.evaluation(sample)
            
            log2pp = np.log2(pcop.cpu().numpy())
            log2pp[pcop.cpu().numpy() == 0] = 0
            
            infoc1 = infoc1 + (np.mean(log2pp) - infoc1) / mo
            varsum1 = varsum1 + np.sum((log2pp - infoc1)**2)
            stderr1 = conf.item() * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
    
    return infoc1 