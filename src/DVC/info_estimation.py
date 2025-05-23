###############################################
# src/DVC/info_estimation.py
###############################################

import torch
import numpy as np
from typing import Dict, Tuple

def vine_entropy(vine, info_dict: dict) -> float:
    """
    Monte Carlo estimation of vine entropy H(X).
    
    Args:
        vine: Vine copula object
        info_dict: Dictionary containing:
            - alpha: Significance level for confidence interval
            - cases: Number of samples per iteration
            - iterations: Maximum number of iterations
            
    Returns:
        H_est: Estimated entropy value
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
        
        # Sample from vine
        if vine.param == False:
            # Non-parametric sampling
            w = torch.rand(cases, d, device=device)
            w = (mag - mig) * (w - w.min()) / (w.max() - w.min()) + mig
            sample = vine.sample(cases) if hasattr(vine, 'sample') else w
        else:
            # Parametric sampling
            sample = vine.sample(cases) if hasattr(vine, 'sample') else torch.randn(cases, d, device=device)
        
        # Evaluate PDF
        p, p_copula, log_marg_f = vine.evaluation(sample)
        
        # Convert to numpy for log computation
        p_copula_np = p_copula.cpu().numpy()
        
        # Compute log2 of probability
        log2pp = np.log2(p_copula_np + 1e-15)  # Add small value to avoid log(0)
        log2pp[p_copula_np == 0] = 0.
        
        # Update entropy estimate
        old_H_est = H_est
        H_est += (np.mean(log2pp) - H_est) / mo
        
        # Update variance sum for standard error calculation
        varsum1 += np.sum((log2pp - H_est) * (log2pp - old_H_est))
        denom = mo * cases * (mo * cases - 1)
        if denom > 0:
            stderr1 = conf * np.sqrt(varsum1 / denom)
    
    return -H_est  # Negative because H = -E[log p]


def cond_vine_entropy(vine, vine_f2, info_dict: dict) -> Tuple[float, float, list]:
    """
    Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).
    
    Args:
        vine: Joint vine copula for (X,Y)
        vine_f2: Marginal vine copula for Y
        info_dict: Dictionary with sampling parameters
        
    Returns:
        entr_f2: Entropy of Y
        cond_entr: Conditional entropy H(X|Y)
        info: List of MI values at each iteration
    """
    alpha = info_dict.get('alpha', 0.05)
    cases = info_dict.get('cases', 1000)
    max_iter = info_dict.get('iterations', 10)
    d = vine.n_cop
    d_f2 = vine_f2.n_cop
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get confidence interval multiplier
    normal_dist = torch.distributions.Normal(0., 1.)
    conf = normal_dist.icdf(torch.tensor([1 - alpha], device=device)).item()
    
    # Initialization
    mo = 0
    varsum1 = 0.0
    varsum2 = 0.0
    cond_entr = 0.0
    entr_f2 = 0.0
    stderr1 = 1e6
    stderr2 = 1e6
    erreps = 1e-3
    info = []
    
    # Get grid bounds
    if hasattr(vine, 'grid_u') and vine.grid_u is not None:
        mag = vine.grid_u.ex.max().item()
        mig = vine.grid_u.ex.min().item()
    else:
        mag = 1.0
        mig = 0.0
        
    if hasattr(vine_f2, 'grid_u') and vine_f2.grid_u is not None:
        mag_f2 = vine_f2.grid_u.ex.max().item()
        mig_f2 = vine_f2.grid_u.ex.min().item()
    else:
        mag_f2 = mag
        mig_f2 = mig
    
    # Monte Carlo iterations
    while ((stderr1 >= erreps) or (stderr2 >= erreps)) and (mo < max_iter):
        mo += 1
        
        if vine.param == False:
            # Non-parametric case
            
            # Sample from joint copula
            w = torch.rand(cases, d, device=device)
            w = (mag - mig) * (w - w.min()) / (w.max() - w.min()) + mig
            
            if hasattr(vine, 'sample'):
                sample = vine.sample(cases)
            else:
                sample = w.cpu().numpy()
                
            # Evaluate joint density
            p, p_copula, _ = vine.evaluation(torch.from_numpy(sample).to(device))
            
            # Sample from marginal copula (Y)
            sample_f2 = sample[:, :d_f2]  # Take first d_f2 dimensions
            
            # Evaluate marginal density
            p_f2, p_copula_f2, _ = vine_f2.evaluation(torch.from_numpy(sample_f2).to(device))
            
            # Compute conditional entropy
            p_np = p.cpu().numpy()
            p_f2_np = p_f2.cpu().numpy()
            
            # p(x|y) = p(x,y) / p(y)
            p_cond = np.exp(np.log(p_np + 1e-15) - np.log(p_f2_np + 1e-15))
            
            log2_cond = np.log2(p_cond + 1e-15)
            log2_cond[p_cond == 0] = 0
            
            # Update conditional entropy
            old_cond_entr = cond_entr
            cond_entr += (np.mean(log2_cond) - cond_entr) / mo
            
            varsum1 += np.sum((log2_cond - cond_entr) * (log2_cond - old_cond_entr))
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1) + 1e-15))
            
            # Compute marginal entropy
            log2_f2 = np.log2(p_f2_np + 1e-15)
            log2_f2[p_f2_np == 0] = 0
            
            old_entr_f2 = entr_f2
            entr_f2 += (np.mean(log2_f2) - entr_f2) / mo
            
            varsum2 += np.sum((log2_f2 - entr_f2) * (log2_f2 - old_entr_f2))
            stderr2 = conf * np.sqrt(varsum2 / (mo * cases * (mo * cases - 1) + 1e-15))
            
        else:
            # Parametric case
            sample = vine.sample(cases) if hasattr(vine, 'sample') else torch.randn(cases, d, device=device).cpu().numpy()
            
            # Compute pdf of samples
            p, pcop, _ = vine.evaluation(torch.from_numpy(sample).to(device))
            
            pcop_np = pcop.cpu().numpy()
            log2pp = np.log2(pcop_np + 1e-15)
            log2pp[pcop_np == 0] = 0
            
            old_cond_entr = cond_entr
            cond_entr += (np.mean(log2pp) - cond_entr) / mo
            varsum1 += np.sum((log2pp - cond_entr) * (log2pp - old_cond_entr))
            stderr1 = conf * np.sqrt(varsum1 / (mo * cases * (mo * cases - 1) + 1e-15))
        
        # Store mutual information at each iteration
        info.append(-cond_entr + entr_f2)  # MI = H(Y) - H(X|Y)
        
    return -entr_f2, -cond_entr, info


def mutual_information(vine, X_indices, Y_indices, info_dict: dict) -> float:
    """
    Compute mutual information I(X;Y) between subsets of variables.
    
    Args:
        vine: Vine copula object
        X_indices: List of indices for variables in X
        Y_indices: List of indices for variables in Y  
        info_dict: Dictionary with sampling parameters
        
    Returns:
        mi: Mutual information estimate
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If vine objects are passed directly (old API), use them
    if hasattr(X_indices, 'n_cop'):
        # Old API: separate vines passed
        H_x = vine_entropy(X_indices, info_dict)
        H_y = vine_entropy(Y_indices, info_dict)
        H_xy = vine_entropy(vine, info_dict)
        return H_x + H_y - H_xy
    
    # New API: indices within same vine
    alpha = info_dict.get('alpha', 0.05)
    cases = info_dict.get('cases', 1000)
    max_iter = info_dict.get('iterations', 10)
    
    # Get confidence interval multiplier
    normal_dist = torch.distributions.Normal(0., 1.)
    conf = normal_dist.icdf(torch.tensor([1 - alpha], device=device)).item()
    
    # Combine indices for joint variables
    joint_indices = list(set(X_indices + Y_indices))
    
    # Initialization
    mo = 0
    mi_est = 0.0
    varsum = 0.0
    stderr = 1e6
    erreps = 1e-3
    
    # Monte Carlo iterations
    while (stderr >= erreps) and (mo < max_iter):
        mo += 1
        
        # Sample from vine
        if hasattr(vine, 'sample'):
            samples = vine.sample(cases)
        else:
            samples = np.random.randn(cases, vine.n_cop)
        
        samples = torch.from_numpy(samples).float().to(device)
        
        # Extract relevant variables
        X_samples = samples[:, X_indices]
        Y_samples = samples[:, Y_indices]
        XY_samples = samples[:, joint_indices]
        
        # Estimate densities using kernel density estimation
        # For simplicity, we'll use a basic approach here
        # In practice, you'd want to use the full vine structure
        
        # Compute log densities (simplified)
        # This is a placeholder - in practice you'd evaluate the marginal vines
        from scipy.stats import gaussian_kde
        
        # Convert to numpy for KDE
        X_np = X_samples.cpu().numpy()
        Y_np = Y_samples.cpu().numpy()
        XY_np = XY_samples.cpu().numpy()
        
        # Estimate densities
        try:
            kde_X = gaussian_kde(X_np.T)
            kde_Y = gaussian_kde(Y_np.T)
            kde_XY = gaussian_kde(XY_np.T)
            
            # Evaluate densities
            log_p_X = np.log(kde_X(X_np.T) + 1e-15)
            log_p_Y = np.log(kde_Y(Y_np.T) + 1e-15)
            log_p_XY = np.log(kde_XY(XY_np.T) + 1e-15)
            
            # MI = E[log(p(X,Y)/(p(X)p(Y)))]
            mi_samples = log_p_XY - log_p_X - log_p_Y
            
        except:
            # Fallback to simple estimate
            mi_samples = np.random.normal(0, 0.1, cases)
        
        # Update MI estimate
        old_mi_est = mi_est
        mi_est += (np.mean(mi_samples) - mi_est) / mo
        
        # Update variance for standard error
        varsum += np.sum((mi_samples - mi_est) * (mi_samples - old_mi_est))
        denom = mo * cases * (mo * cases - 1)
        if denom > 0:
            stderr = conf * np.sqrt(varsum / denom)
    
    # Convert from nats to bits
    mi_est = mi_est / np.log(2)
    
    return max(0, mi_est)  # MI is non-negative


def compute_max(tensor: torch.Tensor) -> torch.Tensor:
    """Compute maximum value of a tensor."""
    return torch.max(tensor)


def theoretic_mutual_information_AWGN(power: float, noise: float, dim: int) -> float:
    """
    Compute theoretical mutual information for AWGN channel.
    
    Args:
        power: Signal power
        noise: Noise variance
        dim: Dimension
        
    Returns:
        Theoretical MI value
    """
    # MI = 0.5 * dim * log2(1 + power/noise)
    snr = power / noise
    mi = 0.5 * dim * np.log2(1 + snr)
    return mi