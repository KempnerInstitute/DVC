"""Vine-based estimators of entropy, conditional entropy, and mutual information."""

import logging
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


@contextmanager
def _temporary_seed(seed):
    if seed is None:
        yield
        return

    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _unit_scale(units: str) -> float:
    units_norm = str(units).lower()
    if units_norm in {"bit", "bits"}:
        return float(np.log(2.0))
    if units_norm in {"nat", "nats", "e"}:
        return 1.0
    raise ValueError(f"Unsupported information unit: {units}")


def _log_prob(values: np.ndarray, units: str) -> np.ndarray:
    clipped = np.asarray(values, dtype=np.float64)
    return np.log(clipped + 1e-15) / _unit_scale(units)


def _gaussian_mutual_information_from_samples(x: np.ndarray,
                                              y: np.ndarray,
                                              units: str) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    xy = np.concatenate([x, y], axis=1)
    eps = 1e-8

    def _regularized_cov(z: np.ndarray) -> np.ndarray:
        cov = np.cov(z, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.asarray([[float(cov)]], dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)
        cov = cov + eps * np.eye(cov.shape[0], dtype=np.float64)
        return cov

    cov_x = _regularized_cov(x)
    cov_y = _regularized_cov(y)
    cov_xy = _regularized_cov(xy)
    sign_x, logdet_x = np.linalg.slogdet(cov_x)
    sign_y, logdet_y = np.linalg.slogdet(cov_y)
    sign_xy, logdet_xy = np.linalg.slogdet(cov_xy)
    if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
        raise RuntimeError("Covariance regularization failed during Gaussian MI fallback")
    mi_nats = 0.5 * (logdet_x + logdet_y - logdet_xy)
    return max(0.0, float(mi_nats / _unit_scale(units)))


def _safe_stderr(conf: float, varsum: float, denom: float) -> float:
    if denom <= 0:
        return float("inf")
    return float(conf) * float(np.sqrt(max(float(varsum) / float(denom), 0.0)))

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
        H_est: Estimated entropy value in the units specified by
            ``info_dict['units']``. Defaults to bits for backward compatibility.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract parameters
    alpha = info_dict.get('alpha', 0.05)
    cases = info_dict.get('cases', 1000)
    max_iter = info_dict.get('iterations', 10)
    units = info_dict.get('units', 'bits')
    seed = info_dict.get('seed')
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
    with _temporary_seed(seed):
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

            p_copula_np = p_copula.detach().cpu().numpy()
            p_copula_np = np.where(np.isfinite(p_copula_np), p_copula_np, 1e-15)
            logpp = _log_prob(p_copula_np, units)
            logpp[p_copula_np == 0] = 0.
            
            # Update entropy estimate
            old_H_est = H_est
            H_est += (np.mean(logpp) - H_est) / mo
            
            # Update variance sum for standard error calculation
            varsum1 += np.sum((logpp - H_est) * (logpp - old_H_est))
            denom = mo * cases * (mo * cases - 1)
            if denom > 0:
                stderr1 = _safe_stderr(conf, varsum1, denom)
    
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
    units = info_dict.get('units', 'bits')
    seed = info_dict.get('seed')
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
    with _temporary_seed(seed):
        while ((stderr1 >= erreps) or (stderr2 >= erreps)) and (mo < max_iter):
            mo += 1

            if vine.param == False:
                # Non-parametric case
                w = torch.rand(cases, d, device=device)
                w = (mag - mig) * (w - w.min()) / (w.max() - w.min()) + mig

                if hasattr(vine, 'sample'):
                    sample = vine.sample(cases)
                else:
                    sample = w.cpu().numpy()

                # Evaluate joint density
                p, p_copula, _ = vine.evaluation(torch.from_numpy(sample).to(device))

                # Sample from marginal copula (Y)
                sample_f2 = sample[:, :d_f2]

                # Evaluate marginal density
                p_f2, p_copula_f2, _ = vine_f2.evaluation(torch.from_numpy(sample_f2).to(device))

                # Compute conditional entropy
                p_np = np.where(np.isfinite(p.cpu().numpy()), p.cpu().numpy(), 1e-15)
                p_f2_np = np.where(np.isfinite(p_f2.cpu().numpy()), p_f2.cpu().numpy(), 1e-15)
                p_cond = np.exp(np.log(p_np + 1e-15) - np.log(p_f2_np + 1e-15))

                log_cond = _log_prob(p_cond, units)
                log_cond[p_cond == 0] = 0

                old_cond_entr = cond_entr
                cond_entr += (np.mean(log_cond) - cond_entr) / mo

                varsum1 += np.sum((log_cond - cond_entr) * (log_cond - old_cond_entr))
                stderr1 = _safe_stderr(conf, varsum1, mo * cases * (mo * cases - 1) + 1e-15)

                log_f2 = _log_prob(p_f2_np, units)
                log_f2[p_f2_np == 0] = 0

                old_entr_f2 = entr_f2
                entr_f2 += (np.mean(log_f2) - entr_f2) / mo

                varsum2 += np.sum((log_f2 - entr_f2) * (log_f2 - old_entr_f2))
                stderr2 = _safe_stderr(conf, varsum2, mo * cases * (mo * cases - 1) + 1e-15)

            else:
                # Parametric case
                sample = vine.sample(cases) if hasattr(vine, 'sample') else torch.randn(cases, d, device=device).cpu().numpy()
                
                # Compute pdf of samples
                p, pcop, _ = vine.evaluation(torch.from_numpy(sample).to(device))
                
                pcop_np = np.where(np.isfinite(pcop.detach().cpu().numpy()), pcop.detach().cpu().numpy(), 1e-15)
                logpp = _log_prob(pcop_np, units)
                logpp[pcop_np == 0] = 0
                
                old_cond_entr = cond_entr
                cond_entr += (np.mean(logpp) - cond_entr) / mo
                varsum1 += np.sum((logpp - cond_entr) * (logpp - old_cond_entr))
                stderr1 = _safe_stderr(conf, varsum1, mo * cases * (mo * cases - 1) + 1e-15)
            
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
    units = info_dict.get('units', 'bits')
    seed = info_dict.get('seed')
    
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
    with _temporary_seed(seed):
        while (stderr >= erreps) and (mo < max_iter):
            mo += 1
            
            # Sample from vine
            if hasattr(vine, 'sample'):
                samples = vine.sample(cases)
            else:
                samples = np.random.randn(cases, vine.n_cop)

            if isinstance(samples, torch.Tensor):
                samples = samples.detach().cpu().numpy()
            samples = torch.from_numpy(np.asarray(samples, dtype=np.float32)).float().to(device)
            
            # Extract relevant variables
            X_samples = samples[:, X_indices]
            Y_samples = samples[:, Y_indices]
            XY_samples = samples[:, joint_indices]
            
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
                mi_samples = (log_p_XY - log_p_X - log_p_Y) / _unit_scale(units)
            except Exception as exc:
                logger.warning("Falling back to Gaussian MI approximation because KDE failed: %s", exc)
                mi_value = _gaussian_mutual_information_from_samples(X_np, Y_np, units=units)
                mi_samples = np.full(cases, mi_value, dtype=np.float64)
            
            # Update MI estimate
            old_mi_est = mi_est
            mi_est += (np.mean(mi_samples) - mi_est) / mo
            
            # Update variance for standard error
            varsum += np.sum((mi_samples - mi_est) * (mi_samples - old_mi_est))
            denom = mo * cases * (mo * cases - 1)
            if denom > 0:
                stderr = _safe_stderr(conf, varsum, denom)
    
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
