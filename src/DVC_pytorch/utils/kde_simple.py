"""
Simple and efficient KDE implementations for PyTorch
These are alternatives to the complex DCT-based approach
"""

import torch
import math


def silverman_bandwidth(data):
    """
    Compute bandwidth using Silverman's rule of thumb
    h = 0.9 * min(σ, IQR/1.349) * n^(-1/5)
    
    Args:
        data: torch.Tensor of shape [N], 1D data
        
    Returns:
        bandwidth: float
    """
    N = data.shape[0]
    sd = torch.std(data)
    
    # Compute IQR (interquartile range)
    q1 = torch.quantile(data, 0.25)
    q3 = torch.quantile(data, 0.75)
    iqr = (q3 - q1).clamp_min(1e-15)
    
    # Silverman's rule
    sigma = torch.min(sd, iqr / 1.349)
    bandwidth = 0.9 * sigma * (N ** (-0.2))
    
    return bandwidth.item()


def scott_bandwidth(data):
    """
    Compute bandwidth using Scott's rule
    h = σ * n^(-1/5)
    
    Args:
        data: torch.Tensor of shape [N], 1D data
        
    Returns:
        bandwidth: float
    """
    N = data.shape[0]
    sd = torch.std(data)
    bandwidth = sd * (N ** (-0.2))
    return bandwidth.item()


def kde_1d_cdist(data, grid, bandwidth=None):
    """
    Simple 1D KDE using cdist (distance matrix approach)
    
    Args:
        data: torch.Tensor of shape [N], the 1D samples
        grid: torch.Tensor of shape [M], evaluation points
        bandwidth: float, if None uses Silverman's rule
        
    Returns:
        density: torch.Tensor of shape [M], estimated density
    """
    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)
    
    N = data.shape[0]
    
    # Reshape for cdist
    data_2d = data.view(-1, 1)
    grid_2d = grid.view(-1, 1)
    
    # Compute distances [M, N]
    dists = torch.cdist(grid_2d, data_2d, p=2.0)
    
    # Gaussian kernel: exp(-0.5 * (x-xi)^2 / h^2) / (sqrt(2π) * h)
    var = bandwidth ** 2
    kernel_vals = torch.exp(-0.5 * (dists ** 2) / var)
    kernel_vals = kernel_vals / (math.sqrt(2 * math.pi) * bandwidth)
    
    # Average over all data points
    density = kernel_vals.mean(dim=1)
    
    return density


def kde_1d_cdist_chunked(data, grid, bandwidth=None, chunk_size=20000):
    """
    Chunked version of kde_1d_cdist for large datasets
    
    Args:
        data: torch.Tensor of shape [N], the 1D samples
        grid: torch.Tensor of shape [M], evaluation points
        bandwidth: float, if None uses Silverman's rule
        chunk_size: int, number of data points per chunk
        
    Returns:
        density: torch.Tensor of shape [M], estimated density
    """
    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)
    
    M = grid.shape[0]
    N = data.shape[0]
    device = grid.device
    
    density = torch.zeros(M, device=device)
    n_chunks = (N + chunk_size - 1) // chunk_size
    
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, N)
        data_chunk = data[start:end]
        
        # Compute distances for this chunk
        dists = torch.cdist(grid.view(-1, 1), data_chunk.view(-1, 1))
        
        # Apply Gaussian kernel
        kernel_vals = torch.exp(-0.5 * (dists ** 2) / (bandwidth ** 2))
        kernel_vals /= (math.sqrt(2 * math.pi) * bandwidth)
        
        # Sum contributions from this chunk
        density += kernel_vals.sum(dim=1)
    
    # Normalize by total number of data points
    density /= float(N)
    
    return density


def kde_fft_1d(data, x_min=None, x_max=None, num_bins=512, bandwidth=None):
    """
    Fast 1D KDE using FFT convolution
    
    Args:
        data: torch.Tensor of shape [N], 1D data
        x_min: float, minimum x value (default: data.min() - 3*bandwidth)
        x_max: float, maximum x value (default: data.max() + 3*bandwidth)
        num_bins: int, number of bins for discretization
        bandwidth: float, if None uses Silverman's rule
        
    Returns:
        density: torch.Tensor of shape [num_bins], estimated density
        grid: torch.Tensor of shape [num_bins], evaluation points
    """
    device = data.device
    dtype = data.dtype
    N = data.shape[0]
    
    # Compute bandwidth if not provided
    if bandwidth is None:
        bandwidth = silverman_bandwidth(data)
    
    # Set boundaries if not provided
    if x_min is None:
        x_min = data.min().item() - 3 * bandwidth
    if x_max is None:
        x_max = data.max().item() + 3 * bandwidth
    
    # Create histogram
    hist = torch.histc(data, bins=num_bins, min=x_min, max=x_max).to(dtype)
    dx = (x_max - x_min) / num_bins
    
    # Create Gaussian kernel centered at 0
    x_kernel = torch.arange(num_bins, device=device, dtype=dtype) - num_bins // 2
    x_kernel = x_kernel * dx
    
    # Gaussian kernel: exp(-0.5 * x^2 / h^2) / (sqrt(2π) * h)
    gauss = torch.exp(-0.5 * (x_kernel ** 2) / (bandwidth ** 2))
    gauss = gauss / (math.sqrt(2 * math.pi) * bandwidth)
    
    # Perform convolution using scipy's fftconvolve approach
    # For circular convolution, shift the kernel
    gauss_shifted = torch.zeros_like(gauss)
    gauss_shifted[:num_bins//2] = gauss[num_bins//2:]  # Positive part
    gauss_shifted[num_bins//2:] = gauss[:num_bins//2]  # Negative part
    
    # FFT convolution
    Hist = torch.fft.fft(hist)
    Gauss = torch.fft.fft(gauss_shifted)
    conv = torch.fft.ifft(Hist * Gauss)
    density = torch.real(conv)
    
    # Convert from counts to density
    # The convolution preserves the sum, so we just need to normalize by N
    density = density / N
    
    # Ensure non-negative
    density = torch.clamp(density, min=0)
    
    # Create grid points (bin centers)
    edges = torch.linspace(x_min, x_max, num_bins + 1, device=device, dtype=dtype)
    grid = (edges[:-1] + edges[1:]) / 2
    
    return density, grid


def kde_simple(data, grid=None, method='cdist', bandwidth=None, **kwargs):
    """
    Simple unified interface for different KDE methods
    
    Args:
        data: torch.Tensor of shape [N], 1D data
        grid: torch.Tensor of shape [M], evaluation points (optional for FFT method)
        method: str, one of ['cdist', 'cdist_chunked', 'fft']
        bandwidth: float, if None uses Silverman's rule
        **kwargs: additional arguments for specific methods
        
    Returns:
        density: torch.Tensor, estimated density
        grid: torch.Tensor, evaluation points (same as input for cdist methods)
    """
    if method == 'cdist':
        if grid is None:
            # Create default grid
            bw = bandwidth or silverman_bandwidth(data)
            x_min = data.min().item() - 3 * bw
            x_max = data.max().item() + 3 * bw
            grid = torch.linspace(x_min, x_max, 128, device=data.device)
        density = kde_1d_cdist(data, grid, bandwidth)
        return density, grid
        
    elif method == 'cdist_chunked':
        if grid is None:
            # Create default grid
            bw = bandwidth or silverman_bandwidth(data)
            x_min = data.min().item() - 3 * bw
            x_max = data.max().item() + 3 * bw
            grid = torch.linspace(x_min, x_max, 128, device=data.device)
        chunk_size = kwargs.get('chunk_size', 20000)
        density = kde_1d_cdist_chunked(data, grid, bandwidth, chunk_size)
        return density, grid
        
    elif method == 'fft':
        x_min = kwargs.get('x_min', None)
        x_max = kwargs.get('x_max', None)
        num_bins = kwargs.get('num_bins', 512)
        return kde_fft_1d(data, x_min, x_max, num_bins, bandwidth)
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['cdist', 'cdist_chunked', 'fft']")


# Wrapper to make it compatible with existing code
def kde_gaussian(data, n=128, MIN=None, MAX=None, method='fft', bandwidth=None):
    """
    Gaussian KDE wrapper that matches the interface of the existing kde function
    
    Args:
        data: torch.Tensor of shape [N], 1D data
        n: int, number of evaluation points
        MIN: float, minimum value for grid
        MAX: float, maximum value for grid
        method: str, KDE method to use
        bandwidth: float, if None uses Silverman's rule
        
    Returns:
        density: torch.Tensor of shape [n]
        xmesh: torch.Tensor of shape [n]
    """
    if method == 'fft':
        density, xmesh = kde_fft_1d(data, x_min=MIN, x_max=MAX, num_bins=n, bandwidth=bandwidth)
    else:
        # For cdist methods, create grid
        if MIN is None:
            bw = bandwidth or silverman_bandwidth(data)
            MIN = data.min().item() - 3 * bw
        if MAX is None:
            bw = bandwidth or silverman_bandwidth(data) 
            MAX = data.max().item() + 3 * bw
            
        xmesh = torch.linspace(MIN, MAX, n, device=data.device, dtype=data.dtype)
        
        if method == 'cdist':
            density = kde_1d_cdist(data, xmesh, bandwidth)
        elif method == 'cdist_chunked':
            density = kde_1d_cdist_chunked(data, xmesh, bandwidth)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return density, xmesh 