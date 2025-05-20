###############################################
# src/DVC/info_estimation.py
###############################################

import torch
import numpy as np

def vine_entropy(vine, info_dict: dict):
    """
    Compute vine entropy using Monte Carlo sampling.
    
    Args:
        vine: fitted vine object
        info_dict: dictionary with parameters
            'alpha': confidence level (e.g., 0.05)
            'cases': number of samples per iteration
            'iterations': maximum number of iterations
    
    Returns:
        entropy estimate (H_est)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract parameters
    alpha = info_dict.get('alpha', 0.05)
    cases = info_dict.get('cases', 1000)
    max_iter = info_dict.get('iterations', 10)
    d = vine.n_cop
    
    # Get confidence interval multiplier
    normal_dist = torch.distributions.Normal(0., 1.)
    conf = normal_dist.icdf(torch.tensor([1 - alpha], device=device)).item()
    
    # Initialization
    mo = 0              # iteration counter
    varsum1 = 0.0       # sum of squared deviations
    H_est = 0.0         # running entropy estimate
    stderr1 = 1e6       # standard error
    erreps = 1e-3       # convergence threshold
    
    # Find min/max for grid
    if hasattr(vine, 'grid_u') and vine.grid_u is not None:
        mag = vine.grid_u.ex.max().item()
        mig = vine.grid_u.ex.min().item()
    else:
        mag = 1.0
        mig = 0.0
    
    # Monte Carlo iterations
    while (stderr1 >= erreps) and (mo < max_iter):
        mo += 1
        
        # Generate samples
        if not vine.param:
            # Non-parametric vine
            sample = vine.sample(cases)
            
            # Convert to torch tensor if needed
            if isinstance(sample, np.ndarray):
                sample_t = torch.tensor(sample, dtype=torch.float32, device=device)
            else:
                sample_t = sample
                
            # Evaluate PDF
            p, p_copula, _ = vine.evaluation(sample_t)
            
            # Convert to log2 and handle zeros
            p_copula_np = p_copula.cpu().numpy()
            log2pp = np.log2(p_copula_np)
            log2pp[p_copula_np == 0] = 0
            
            # Update running average
            old_H_est = H_est
            H_est += (np.mean(log2pp) - H_est) / mo
            
            # Update variance sum for standard error
            varsum1 += np.sum((log2pp - H_est) * (log2pp - old_H_est))
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
            
        else:
            # Parametric vine
            sample = vine.sample(cases)
            
            # Convert to torch tensor if needed
            if isinstance(sample, np.ndarray):
                sample_t = torch.tensor(sample, dtype=torch.float32, device=device)
            else:
                sample_t = sample
                
            # Evaluate PDF
            p, p_copula, _ = vine.evaluation(sample_t)
            
            # Convert to log2 and handle zeros
            p_copula_np = p_copula.cpu().numpy()
            log2pp = np.log2(p_copula_np)
            log2pp[p_copula_np == 0] = 0
            
            # Update running average
            old_H_est = H_est
            H_est += (np.mean(log2pp) - H_est) / mo
            
            # Update variance sum for standard error
            varsum1 += np.sum((log2pp - H_est) * (log2pp - old_H_est))
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1)))
    
    return H_est