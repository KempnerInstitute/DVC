import torch
import numpy as np
from utils.tensor_op import moving_average, create_points, replace_nan_inf

################## PREDICT VINE ########################

def predict_vine(x, vine, dim, exp_dim):
    """
    Predict using vine copula
    
    Args:
        x: Input data (n_samples, n_dims)
        vine: Fitted vine copula object
        dim: Dimension to predict
        exp_dim: Number of expansion points
        
    Returns:
        p: Probability densities
        y_ml: Maximum likelihood predictions
        y_em: Expectation maximization predictions
    """
    device = x.device if torch.is_tensor(x) else torch.device('cpu')
    dtype = x.dtype if torch.is_tensor(x) else torch.float32
    
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=dtype, device=device)
    
    # Create points for evaluation
    points = create_points(x, dim, exp_dim)
    
    # Evaluate vine copula
    p, p_cop, logp = vine.evaluation(points)
    
    # Reshape probabilities
    p1 = p.reshape(x.shape[0], exp_dim)
    p1 = replace_nan_inf(p1)
    
    # Create y vector
    min_dim = torch.min(x[:, dim])
    max_dim = torch.max(x[:, dim])
    y_vec = torch.linspace(min_dim - 2e-16 + 1e-5, max_dim + 2e-16, exp_dim, 
                          dtype=dtype, device=device)
    
    # Smooth probabilities
    mov_p = torch.zeros_like(p1)
    for i in range(p1.shape[0]):
        movag = smooth(p1[i, :].cpu().numpy(), 4, 'flat')
        mov_p[i, :] = torch.tensor(movag, dtype=dtype, device=device)
    
    mov_p = mov_p[:, 3:]
    y_vec_adj = y_vec[3:]
    
    ############### Y MAXIMUM LIKELIHOOD  ##################
    
    # Find maximum probability indices
    ind_max1 = torch.argmax(mov_p, dim=1)
    y_ml = y_vec_adj[ind_max1]
    
    ############### Y EXPECTATION MAXIMIZATION  ##################
    
    # Compute differences
    y_diff = torch.diff(y_vec_adj, dim=0)
    y_diff = torch.cat([y_diff, y_diff[-1:]], dim=0)
    
    # Normalize probabilities
    q1 = torch.sum(mov_p * y_diff.unsqueeze(0), dim=1, keepdim=True)
    q = mov_p / (q1 + 1e-10)  # Add small value to avoid division by zero
    
    # Compute expectation
    y_tmp = y_vec_adj * y_diff
    y_em = torch.sum(q * y_tmp.unsqueeze(0), dim=1)
    
    return p, y_ml, y_em

###################  PREDICT RESPONSE   ######################

def predict_response(p1, y_vec):
    """
    Predict response from probability densities
    
    Args:
        p1: Probability densities (n_samples, n_points)
        y_vec: Y values corresponding to probabilities
        
    Returns:
        y_ml: Maximum likelihood predictions
        y_em: Expectation maximization predictions
    """
    device = p1.device
    dtype = p1.dtype
    
    if not torch.is_tensor(y_vec):
        y_vec = torch.tensor(y_vec, dtype=dtype, device=device)
    
    # Smooth probabilities
    mov_p = torch.zeros_like(p1)
    for i in range(p1.shape[0]):
        movag = smooth(p1[i, :].cpu().numpy(), 4, 'flat')
        mov_p[i, :] = torch.tensor(movag, dtype=dtype, device=device)
    
    mov_p = mov_p[:, 3:]
    y_vec_adj = y_vec[3:]
    
    ############### Y MAXIMUM LIKELIHOOD  ##################
    
    # Find maximum probability indices
    ind_max1 = torch.argmax(mov_p, dim=1)
    y_ml = y_vec_adj[ind_max1]
    
    ############### Y EXPECTATION MAXIMIZATION  ##################
    
    # Compute differences
    y_diff = torch.diff(y_vec_adj, dim=0)
    y_diff = torch.cat([y_diff, y_diff[-1:]], dim=0)
    
    # Normalize probabilities
    q1 = torch.sum(mov_p * y_diff.unsqueeze(0), dim=1, keepdim=True)
    q = mov_p / (q1 + 1e-10)  # Add small value to avoid division by zero
    
    # Compute expectation
    y_tmp = y_vec_adj * y_diff
    y_em = torch.sum(q * y_tmp.unsqueeze(0), dim=1)
    
    return y_ml, y_em


def smooth(x, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    Args:
        x: The input signal (1D array)
        window_len: The dimension of the smoothing window; should be an odd integer
        window: The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    
    Returns:
        The smoothed signal
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len < 3:
        return x
    
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    # Prepare signal with reflected ends
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    
    # Create window
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval(f'np.{window}({window_len})')
    
    # Convolve
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y 