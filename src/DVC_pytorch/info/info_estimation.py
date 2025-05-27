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
    d = len(vine.margin) if hasattr(vine, 'margin') else vine.n_cop
    
    device = vine.grid_u.ex.device if hasattr(vine, 'grid_u') else torch.device('cpu')
    dtype = vine.grid_u.ex.dtype if hasattr(vine, 'grid_u') else torch.float32
    
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
    
    # No need for mag/mig as vine.sample handles the transformation
    
    while ((stderr1 >= erreps) or (stderr2 >= erreps) or (stderr_tot >= erreps)) and (mo < max_iter):
        mo = mo + 1
        
        # CRITICAL FIX: Use actual vine sampling instead of random uniform
        # The vine.sample method returns samples in the original data space
        sample = vine.sample(cases)
        
        # Ensure sample is on the correct device
        if not torch.is_tensor(sample):
            sample = torch.tensor(sample, dtype=dtype, device=device)
        elif sample.device != device:
            sample = sample.to(device)
        
        # Evaluate PDF at samples
        p, p_copula, plog = vine.evaluation(sample)
        
        # Convert to numpy for log2 (PyTorch doesn't have log2)
        p_copula_np = p_copula.cpu().numpy()
        
        # Compute log2 of copula density, handling zeros
        log2pp = np.log2(p_copula_np)
        log2pp[p_copula_np == 0] = 0
        
        # Update running average for entropy estimation
        # Note: Negative expectation of log density gives entropy
        infoc1 = infoc1 + (-np.mean(log2pp) - infoc1) / mo
        
        # Update variance sum for confidence interval
        varsum1 = varsum1 + np.sum(((-log2pp) - infoc1)**2)
        
        # Compute standard error
        if mo > 1:
            stderr1 = conf.item() * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
        
        # For compatibility, keep stderr2 and stderr_tot as placeholders
        # In the TensorFlow version, these might be used for other metrics
        stderr2 = stderr1  # Placeholder
        stderr_tot = stderr1  # Placeholder
    
    return infoc1 