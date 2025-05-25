import torch
import torch.distributions as dist
import torch.fft as fft
import math
import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
from utils.tensor_op import *
from utils.interpolation import interp1d_np

def biv_norm(x1_s, x2_s):
    """Compute bivariate normal PDF"""
    norm = dist.Normal(loc=torch.tensor(0.0, dtype=x1_s.dtype, device=x1_s.device),
                      scale=torch.tensor(1.0, dtype=x1_s.dtype, device=x1_s.device))
    
    P1 = norm.log_prob(x1_s).exp().unsqueeze(-1)
    P2 = norm.log_prob(x2_s).exp().unsqueeze(-1)
    
    NORM = P1 * P2.t()
    return NORM

def op_cdf(data, margin_s_exc):
    """Compute empirical CDF indices"""
    data_ti = data.unsqueeze(-1).repeat(1, margin_s_exc.shape[0])
    margin_s_ti = margin_s_exc.unsqueeze(0).repeat(data.shape[0], 1)
    
    dif1 = margin_s_ti - data_ti
    dif1 = torch.maximum(dif1, torch.zeros_like(dif1))
    dif1 = torch.sign(dif1)
    kka1 = torch.sum(dif1, dim=0)
    kka1 = kka1.to(torch.int32)
    
    return kka1

def kernel_cdf_batch(data, y, ex, batch_size):
    """Compute kernel CDF with batching"""
    margin_s, _ = torch.unique(data, sorted=True, return_inverse=True)
    
    # Calculate the batch length
    batch_len = margin_s.shape[0] // batch_size
    
    kka_list = []
    
    for i in range(batch_size):
        start_idx = batch_len * i
        end_idx = batch_len * (i + 1) if i < batch_size - 1 else margin_s.shape[0]
        
        data_ti = data.unsqueeze(-1).repeat(1, end_idx - start_idx)
        margin_s_ti = margin_s[start_idx:end_idx].unsqueeze(0).repeat(data.shape[0], 1)
        
        dif1 = margin_s_ti - data_ti
        dif1 = torch.maximum(dif1, torch.zeros_like(dif1))
        dif1 = torch.sign(dif1)
        kka1 = torch.sum(dif1, dim=0)
        
        kka_list.append(kka1)
    
    kka = torch.cat(kka_list)
    
    margin_p = kka.float() / (data.shape[0] + 1)
    margin_p = margin_p.to(data.dtype)
    
    interp_cdf = interp1d_np(y, margin_s, margin_p)
    interp_cdf = constraints_bound(interp_cdf, ex)
    interp_cdf = check_bound_and_nan(interp_cdf, torch.max(margin_s), torch.min(margin_s))
    
    return interp_cdf, margin_s, margin_p

def kernel_cdf(data, y, ex):
    """Compute kernel CDF without batching"""
    margin_s, _ = torch.unique(data, sorted=True, return_inverse=True)
    
    data_ti = data.unsqueeze(-1).repeat(1, margin_s.shape[0])
    dif1 = margin_s - data_ti
    dif1 = torch.maximum(dif1, torch.zeros_like(dif1))
    dif1 = torch.sign(dif1)
    kka = torch.sum(dif1, dim=0) + 1  # Important: add 1 to account for own value
    
    margin_p = kka.float() / (data.shape[0] + 1)
    margin_p = margin_p.to(data.dtype)
    
    interp_cdf = interp1d_np(y, margin_s, margin_p)
    interp_cdf = constraints_bound(interp_cdf, ex)
    interp_cdf = check_bound_and_nan(interp_cdf, torch.max(margin_s), torch.min(margin_s))
    
    return interp_cdf, margin_s, margin_p

def kernel_pdf2(x):
    """Kernel density estimation"""
    x_ker = x
    iscont = torch.tensor(1, dtype=torch.int32)
    density = torch.zeros(128, dtype=x.dtype, device=x.device)
    mesh = torch.zeros(128, dtype=x.dtype, device=x.device)
    
    if not torch.any(x_ker < 0):
        # BIMODAL DISTRIBUTION
        indpp1 = torch.where(x_ker < 1e-6)[0]
        indpp2 = torch.where(x_ker >= 1e-6)[0]
        
        if len(indpp1) > 1 and len(indpp2) > 1:
            pow1 = x_ker[indpp1]
            pow2 = x_ker[indpp2]
            
            # Uniform distribution for pow1
            max_pow1 = torch.max(pow1)
            min_pow1 = torch.min(pow1)
            p_uni = 1 / (max_pow1 - min_pow1)
            den2 = torch.full((128,), p_uni, dtype=pow1.dtype, device=pow1.device)
            
            R = max_pow1 + 2e-16 - min_pow1
            mden2 = torch.linspace(0, R, 128, dtype=pow1.dtype, device=pow1.device) + min_pow1
            
            # KDE for pow2
            den3, mden3 = kde(pow2, 128, torch.min(pow2), torch.max(pow2) + 2e-16)
            
            # Normalize den2
            m_diff = mden2[1:] - mden2[:-1]
            m_diff = torch.cat([m_diff, m_diff[-1:]])
            norm = torch.sum(den2 * m_diff)
            den2 = den2 / norm
            
            # Normalize den3
            m_diff = mden3[1:] - mden3[:-1]
            m_diff = torch.cat([m_diff, m_diff[-1:]])
            norm = torch.sum(den3 * m_diff)
            den3 = den3 / norm
            
            # Combine distributions
            SM = torch.linspace(torch.max(mden2) + 1e-6, torch.min(mden3) - 1e-6, 100,
                               dtype=x_ker.dtype, device=x_ker.device)
            mesh = torch.cat([mden2, SM, mden3])
            
            part1 = len(indpp1) / x_ker.shape[0]
            part2 = len(indpp2) / x_ker.shape[0]
            density = torch.cat([den2 * part1, torch.zeros(100, dtype=x_ker.dtype, device=x_ker.device), den3 * part2])
        else:
            density, mesh = kde(x_ker, 128, torch.min(x_ker), torch.max(x_ker))
            m_diff = mesh[1:] - mesh[:-1]
            m_diff = torch.cat([m_diff, m_diff[-1:]])
            area = torch.sum(density * m_diff)
            density = density / area
    else:
        density, mesh = kde(x_ker, 128, torch.min(x_ker), torch.max(x_ker))
        m_diff = mesh[1:] - mesh[:-1]
        m_diff = torch.cat([m_diff, m_diff[-1:]])
        area = torch.sum(density * m_diff)
        density = density / area
    
    return density, mesh

######## FUNCTION USED IN THE KDE:
# fixed_point: Function to evaluate best point, when it is equal zero.
# dct1d: Discrete cosine transform 1-D
# idct1d: Inverse discrete cosine transform 1-D
# histc: Python function to count how many times a value goes into predefined intervals
# kde: Main kernel density estimation

def fixed_point(xx, N, I, a2):
    """Fixed point iteration for bandwidth selection"""
    dtype = a2.dtype
    device = a2.device
    
    xx = xx.to(dtype)
    N = torch.tensor(N, dtype=dtype, device=device) if not torch.is_tensor(N) else N.to(dtype)
    I = I.to(dtype)
    
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    l = torch.tensor(7, dtype=dtype, device=device)
    
    # Compute f with numerical stability
    exp_term = -I * torch.square(pi) * xx
    # Clamp to prevent underflow
    exp_term = torch.clamp(exp_term, min=-50.0)
    f = 2 * torch.pow(pi, 2*l) * torch.sum(torch.pow(I, l) * a2 * torch.exp(exp_term))
    
    # Add small epsilon to prevent division by zero
    eps = torch.finfo(dtype).eps
    f = torch.clamp(f, min=eps)
    
    for i in range(6, 1, -1):
        i_tensor = torch.tensor(i, dtype=dtype, device=device)
        K0 = torch.prod(torch.arange(1, 2*i, 2, dtype=dtype, device=device)) / torch.sqrt(2*pi)
        const = (1 + torch.pow(torch.tensor(0.5, dtype=dtype, device=device), i_tensor + 0.5)) / 3
        time_arg = 2 * const * K0 / N / f
        # Prevent negative values in pow
        time_arg = torch.clamp(time_arg, min=eps)
        time = torch.pow(time_arg, 2 / (3 + 2*i_tensor))
        
        # Compute f with numerical stability
        exp_term = -I * torch.square(pi) * time
        exp_term = torch.clamp(exp_term, min=-50.0)
        f = 2 * torch.pow(pi, 2*i_tensor) * torch.sum(torch.pow(I, i_tensor) * a2 * torch.exp(exp_term))
        f = torch.clamp(f, min=eps)
    
    # Final calculation with numerical stability
    final_arg = 2 * N * torch.sqrt(pi) * f
    final_arg = torch.clamp(final_arg, min=eps)
    out = xx - torch.pow(final_arg, -2/5)
    
    return out

def dct1d(data):
    """Discrete cosine transform using PyTorch"""
    # PyTorch doesn't have a built-in DCT, so we'll use the FFT-based implementation
    N = data.shape[0]
    device = data.device
    dtype = data.dtype
    
    # Prepare data for FFT-based DCT
    y = torch.zeros(2*N, dtype=dtype, device=device)
    y[:N] = data
    y[N:] = torch.flip(data, dims=[0])
    
    # Compute FFT
    Y = torch.fft.fft(y.to(torch.complex64))
    
    # Extract DCT coefficients
    dct_result = torch.real(Y[:N]) / 2
    dct_result[0] /= torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    
    # Apply normalization
    dct_result *= torch.sqrt(torch.tensor(2.0/N, dtype=dtype, device=device))
    
    return dct_result.to(dtype)

def idct1d(data):
    """Inverse discrete cosine transform using PyTorch"""
    N = data.shape[0]
    device = data.device
    dtype = data.dtype
    
    # Denormalize
    data_copy = data.clone()
    data_copy *= torch.sqrt(torch.tensor(N/2.0, dtype=dtype, device=device))
    data_copy[0] *= torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    
    # Prepare for IFFT
    Y = torch.zeros(2*N, dtype=torch.complex64, device=device)
    Y[:N] = data_copy.to(torch.complex64)
    # For the second half, we need N-1 elements (excluding the first one)
    if N > 1:
        Y[N:2*N-1] = torch.flip(data_copy[1:], dims=[0]).to(torch.complex64)
    
    # Compute IFFT
    y = torch.fft.ifft(Y)
    
    # Extract result
    result = torch.real(y[:N]) * 2
    
    return result.to(dtype)

def histc(X, bins):
    """Histogram counts using numpy (for compatibility)"""
    X_np = X.cpu().numpy() if torch.is_tensor(X) else X
    bins_np = bins.cpu().numpy() if torch.is_tensor(bins) else bins
    
    map_to_bins = np.digitize(X_np, bins_np)
    r = np.zeros(bins_np.shape)
    for i in map_to_bins:
        if i > 0 and i <= len(bins_np):
            r[i-1] += 1
    return r, map_to_bins

def histc1(X, bins):
    """Histogram counts using PyTorch"""
    # Use torch.histc or torch.histogram
    counts = torch.histc(X, bins=len(bins)-1, min=bins[0], max=bins[-1])
    return counts

def kde(data, N, MIN, MAX):
    """Kernel density estimation"""
    device = data.device
    dtype = data.dtype
    
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    R = MAX - MIN
    
    nbins = N
    xmesh = torch.linspace(0, R, 128, dtype=dtype, device=device) + MIN
    
    # Get unique values
    data_unique, _ = torch.unique(data, sorted=True, return_inverse=True)
    N_samples = torch.ceil(torch.tensor((data.shape[0] - 1) / 2, dtype=dtype, device=device)) * 2
    N_samples = N_samples.to(torch.int32)
    
    # Compute histogram
    counts = torch.histc(data, bins=128, min=MIN, max=MAX)
    init_data = counts / N_samples.float()
    init_data = init_data / torch.sum(init_data)
    
    # Apply DCT
    a = dct1d(init_data)
    I = torch.square(torch.arange(1, 128, 1, dtype=dtype, device=device))
    a2 = torch.square(a[1:] / 2)
    N_float = N_samples.float()
    
    # Find optimal bandwidth using fixed point iteration
    tol = 1e-12 + 0.01 * (N_float - 50) / 1000
    
    # Simple secant method implementation for finding root
    t_star = find_root_secant(lambda t: fixed_point(t, N_float, I, a2), tol, device=device, dtype=dtype)
    
    # Apply bandwidth
    a_1 = torch.exp(-(torch.arange(0, nbins, dtype=dtype, device=device)**2 * (pi**2) * t_star) / 2)
    a_t = a * a_1
    
    # Apply inverse DCT
    density = idct1d(a_t) / R
    
    # Ensure non-negative
    if torch.any(density < 0):
        eps = torch.finfo(dtype).eps
        density = replace_negative(density, eps)
    
    return density, xmesh

def find_root_secant(func, x0, device='cpu', dtype=torch.float32, tol=1e-8, max_iter=50):
    """Simple secant method for root finding"""
    x0 = torch.tensor(0.01, dtype=dtype, device=device)  # Start with a reasonable initial value
    x1 = x0 + torch.tensor(0.01, dtype=dtype, device=device)
    
    for i in range(max_iter):
        f0 = func(x0)
        f1 = func(x1)
        
        # Check for convergence
        if torch.abs(f1) < tol:
            return x1
        
        # Check if denominator is too small
        if torch.abs(f1 - f0) < 1e-10:
            # Try a different step
            x1 = x1 + torch.tensor(0.01, dtype=dtype, device=device)
            continue
            
        # Secant update
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        
        # Ensure x_new is positive and reasonable
        x_new = torch.clamp(x_new, min=1e-6, max=1.0)
        
        # Check convergence
        if torch.abs(x_new - x1) < tol:
            return x_new
            
        x0 = x1
        x1 = x_new
    
    # If no convergence, return last value
    return x1

# Additional utility functions for copula operations
def kendalltau(x, y):
    """Compute Kendall's tau correlation"""
    # Convert to numpy for scipy
    x_np = x.cpu().numpy() if torch.is_tensor(x) else x
    y_np = y.cpu().numpy() if torch.is_tensor(y) else y
    
    tau, p_value = stats.kendalltau(x_np, y_np)
    return tau, p_value 