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
            
            # Handle case where all values in pow1 are identical
            if torch.abs(max_pow1 - min_pow1) < 1e-16:
                # Use a small range around the value
                center = (max_pow1 + min_pow1) / 2
                R = 1e-6
                min_pow1 = center - R/2
                max_pow1 = center + R/2
            else:
                R = max_pow1 - min_pow1
            
            p_uni = 1 / R
            den2 = torch.full((128,), p_uni, dtype=pow1.dtype, device=pow1.device)
            
            mden2 = torch.linspace(min_pow1, max_pow1, 128, dtype=pow1.dtype, device=pow1.device)
            
            # KDE for pow2
            den3, mden3 = kde(pow2, 128, torch.min(pow2), torch.max(pow2) + 2e-16)
            
            # Normalize den2
            m_diff = mden2[1:] - mden2[:-1]
            m_diff = torch.cat([m_diff, m_diff[-1:]])
            norm = torch.sum(den2 * m_diff)
            if norm > 0:
                den2 = den2 / norm
            
            # Normalize den3
            m_diff = mden3[1:] - mden3[:-1]
            m_diff = torch.cat([m_diff, m_diff[-1:]])
            norm = torch.sum(den3 * m_diff)
            if norm > 0:
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
            if area > 0:
                density = density / area
    else:
        density, mesh = kde(x_ker, 128, torch.min(x_ker), torch.max(x_ker))
        m_diff = mesh[1:] - mesh[:-1]
        m_diff = torch.cat([m_diff, m_diff[-1:]])
        area = torch.sum(density * m_diff)
        if area > 0:
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
    """Discrete cosine transform using FFT - matching TensorFlow implementation"""
    n = data.shape[0]
    device = data.device
    dtype = data.dtype
    
    if n == 1:
        return data
    
    # Extend the data by mirroring (matching TensorFlow's approach)
    # TensorFlow does: [data, reverse(data[1:n-1])]
    extended = torch.cat([data, torch.flip(data[1:n-1], dims=[0])], dim=0)
    
    # Perform FFT
    if dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        complex_dtype = torch.complex64
    
    result = torch.fft.fft(extended.to(complex_dtype))
    result = torch.real(result)
    
    # Return first n elements
    return result[:n]

def idct1d(data):
    """Inverse discrete cosine transform using FFT - matching TensorFlow implementation"""
    n = data.shape[0]
    device = data.device
    dtype = data.dtype
    
    if n == 1:
        return data
    
    # Extend the data by mirroring (matching TensorFlow's approach)
    # TensorFlow does: [data, reverse(data[1:])]
    extended = torch.cat([data, torch.flip(data[1:], dims=[0])], dim=0)
    
    # Perform inverse FFT
    if dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        complex_dtype = torch.complex64
    
    result = torch.fft.ifft(extended.to(complex_dtype))
    result = torch.real(result)
    
    # Return first n elements
    return result[:n]

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

def kde(data, n=128, use_fft=False, bounded=False, bounds=None):
    """
    Compute kernel density estimate.
    
    Args:
        data: Input data tensor
        n: Number of points in output
        use_fft: Whether to use FFT-based method (faster)
        bounded: Whether to enforce zero probability outside data range
        bounds: Explicit bounds (min, max). If None and bounded=True, uses data range
    
    Returns:
        density: Probability density values
        mesh: Evaluation points
    """
    if use_fft:
        from .kde_simple import kde_gaussian
        return kde_gaussian(data, n=n, method='fft')
    
    if bounded:
        from .kde_bounded import bounded_kde_gaussian
        return bounded_kde_gaussian(data, n=n, bounds=bounds, boundary_correction='renormalize')
    
    # Original implementation continues below...
    
    device = data.device
    dtype = data.dtype
    
    # Set up range
    if bounds is None:
        min_val, max_val = torch.min(data), torch.max(data)
        Range = max_val - min_val
        bounds = (min_val - Range / 2, max_val + Range / 2)
    
    bounds = (bounds[0].to(dtype), bounds[1].to(dtype))
    
    # Set up the grid over which the density estimate is computed
    R = bounds[1] - bounds[0]
    dx = R / (n - 1)
    xmesh = bounds[0] + dx * torch.arange(n, dtype=dtype, device=device)
    
    # Get actual sample size instead of number of unique values
    N = data.shape[0]
    
    # Bin the data uniformly using the defined grid
    initial_data = torch.histc(data, bins=n, min=bounds[0].item(), max=bounds[1].item())
    initial_data = initial_data / torch.sum(initial_data)  # Normalize by total count
    initial_data = initial_data.to(dtype)
    
    # Discrete cosine transform of initial data
    a = dct1d(initial_data)
    
    # Optimal bandwidth selection
    I = torch.arange(1, n, dtype=dtype, device=device) ** 2
    a2 = (a[1:] / 2) ** 2
    
    # Define the fixed-point equation
    def fixed_point_kde(t, N, I, a2):
        l = 7
        t = t.to(dtype) if torch.is_tensor(t) else torch.tensor(t, dtype=dtype, device=device)
        N = torch.tensor(N, dtype=dtype, device=device) if not torch.is_tensor(N) else N.to(dtype)
        
        pi = torch.tensor(math.pi, dtype=dtype, device=device)
        
        # Initial f calculation
        f = 2 * torch.pow(pi, 2*l) * torch.sum(torch.pow(I, l) * a2 * torch.exp(-I * pi**2 * t))
        
        for s in range(l-1, 1, -1):
            s_tensor = torch.tensor(s, dtype=dtype, device=device)
            
            # Use lgamma for numerical stability
            K0 = torch.exp(torch.lgamma(s_tensor + 1) - torch.lgamma(s_tensor/2 + 1) - 0.5 * torch.log(2 * pi))
            
            const = (1 + torch.pow(torch.tensor(0.5, dtype=dtype, device=device), s_tensor + 0.5)) / 3
            time = torch.pow(2 * const * K0 / N / f, 2 / (3 + 2*s_tensor))
            f = 2 * torch.pow(pi, 2*s_tensor) * torch.sum(torch.pow(I, s_tensor) * a2 * torch.exp(-I * pi**2 * time))
        
        out = t - torch.pow(2 * N * torch.sqrt(pi) * f, -2/5)
        return out
    
    # Find the root of the fixed-point equation using a simple bisection method
    t_star = find_root_bisection(lambda t: fixed_point_kde(t, N, I, a2), 
                                 low=0.0, high=1.0, device=device, dtype=dtype)
    
    # Smooth the DCT of initial data using t_star
    pi = torch.tensor(math.pi, dtype=dtype, device=device)
    a_t = a * torch.exp(-torch.arange(n, dtype=dtype, device=device)**2 * pi**2 * t_star / 2)
    
    # Apply the inverse DCT
    density = idct1d(a_t) / R
    
    # Ensure non-negative
    density = torch.clamp(density, min=0)
    
    # Normalize to ensure integral = 1
    # When called with small n (like 128), downsample first then normalize
    if n > 128:
        # Downsample to 128 points
        indices = torch.linspace(0, len(xmesh)-1, 128, dtype=torch.long, device=device)
        xmesh = xmesh[indices]
        density = density[indices]
    
    # Calculate dx for the final grid
    dx_final = (xmesh[-1] - xmesh[0]) / (len(xmesh) - 1)
    integral = torch.sum(density) * dx_final
    if integral > 0:
        density = density / integral
    
    return density, xmesh

def find_root_bisection(func, low=0.0, high=1.0, device='cpu', dtype=torch.float32, tol=1e-8, max_iter=50):
    """Improved bisection method for root finding with better numerical stability"""
    low = torch.tensor(low, dtype=dtype, device=device)
    high = torch.tensor(high, dtype=dtype, device=device)
    
    # First check if we can evaluate the function at the bounds
    try:
        f_low = func(low)
        f_high = func(high)
    except:
        # If function evaluation fails at bounds, try slightly interior points
        low = low + 1e-10
        high = high - 1e-10
        f_low = func(low)
        f_high = func(high)
    
    # Check if root is at the bounds
    if torch.abs(f_low) < tol:
        return low
    if torch.abs(f_high) < tol:
        return high
    
    # If signs are the same, try to find a better interval
    if torch.sign(f_low) == torch.sign(f_high):
        # Try multiple starting points
        test_points = torch.linspace(low, high, 20, dtype=dtype, device=device)
        f_values = []
        valid_points = []
        
        for t in test_points:
            try:
                f_val = func(t)
                if torch.isfinite(f_val):
                    f_values.append(f_val)
                    valid_points.append(t)
            except:
                continue
        
        if len(f_values) > 0:
            f_values = torch.stack(f_values)
            valid_points = torch.stack(valid_points)
            
            # Find where sign changes
            for i in range(len(f_values) - 1):
                if torch.sign(f_values[i]) != torch.sign(f_values[i+1]):
                    low = valid_points[i]
                    high = valid_points[i+1]
                    f_low = f_values[i]
                    f_high = f_values[i+1]
                    break
            else:
                # No sign change found, return point with minimum absolute value
                min_idx = torch.argmin(torch.abs(f_values))
                return valid_points[min_idx]
    
    # Standard bisection
    for i in range(max_iter):
        mid = (low + high) / 2
        try:
            f_mid = func(mid)
        except:
            # If evaluation fails, try a different point
            mid = (low + high) / 2 + (high - low) * 0.1 * (torch.rand(1, device=device) - 0.5)
            f_mid = func(mid)
        
        if not torch.isfinite(f_mid):
            # If we get inf or nan, adjust the interval
            if not torch.isfinite(f_low):
                low = mid
                f_low = f_mid
            else:
                high = mid
                f_high = f_mid
            continue
            
        if torch.abs(f_mid) < tol:
            return mid
            
        if torch.sign(f_mid) == torch.sign(f_low):
            low = mid
            f_low = f_mid
        else:
            high = mid
            f_high = f_mid
            
        if torch.abs(high - low) < tol:
            return mid
    
    return (low + high) / 2

# Additional utility functions for copula operations
def kendalltau(x, y):
    """Compute Kendall's tau correlation"""
    # Convert to numpy for scipy
    x_np = x.cpu().numpy() if torch.is_tensor(x) else x
    y_np = y.cpu().numpy() if torch.is_tensor(y) else y
    
    tau, p_value = stats.kendalltau(x_np, y_np)
    return tau, p_value

def kde_wrapper(data, n=128, method='fft', bounded=False, bounds=None):
    """
    Wrapper function for different KDE methods.
    
    Args:
        data: Input data
        n: Number of evaluation points
        method: KDE method ('original', 'fft', 'cdist', 'fft_bounded', etc.)
        bounded: Whether to enforce bounds
        bounds: Explicit bounds
    
    Returns:
        density: Probability density
        mesh: Evaluation points
    """
    if bounded or '_bounded' in method:
        from .kde_bounded import bounded_kde_wrapper
        return bounded_kde_wrapper(data, n=n, method=method, 
                                 enforce_bounds=True, bounds=bounds)
    
    if method == 'original':
        return kde(data, n=n, use_fft=False)
    elif method == 'kernel_pdf2':
        return kernel_pdf2(data)
    else:
        from .kde_simple import kde_gaussian
        return kde_gaussian(data, n=n, method=method) 