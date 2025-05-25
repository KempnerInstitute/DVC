import torch
from utils.tensor_op import check_bound3, replace_nan_inf
from optim.MISE import MISE_mul

############################# NADAM OPTIMIZATION #################################

def fit_ban(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test, 
            n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    """
    Fit bandwidth using Nadam optimizer (single bandwidth per copula)
    
    Args:
        a: Initial bandwidth multiplier
        bw: Base bandwidth
        adu11, adu22: Grid differences
        step_s: Grid step size
        min_s, max_s: Grid bounds
        grid_x: Grid points
        data_x: All data in x-space
        data_x_train: Training data
        data_s_test: Test data
        n_cop: Number of copulas
        batch: Batch size
        NORM: Normalization constant
        norm_flag: Whether to normalize
        pos_trace: Previous position trace
        max_iter: Maximum iterations
        convergence_tol: Convergence tolerance
        lr: Learning rate
        
    Returns:
        a: Optimized bandwidth multiplier
        err: Final error
        iter_err: Number of iterations
        converged: Convergence flag
    """
    device = a.device
    dtype = a.dtype
    eps = 1e-6
    
    iter_err = 1
    err_trace = torch.ones(n_cop, dtype=dtype, device=device)
    err = err_trace + 10 * convergence_tol
    
    # Initial error
    err = MISE_mul(pos_trace, bw, adu11, adu22, step_s, min_s, max_s, grid_x, 
                   data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
    
    # Initialize Adam parameters
    m = torch.zeros(bw.shape[1], dtype=dtype, device=device)
    v = torch.zeros(bw.shape[1], dtype=dtype, device=device)
    beta_1 = 0.9
    beta_2 = 0.999
    
    while iter_err < max_iter and torch.any(torch.abs(err - err_trace) > convergence_tol):
        err_trace = err.clone()
        err_trace = err_trace.reshape(bw.shape[1])
        
        # Compute new error
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, 
                       data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        # Compute gradient using finite differences
        grad = (err - err_trace) / (a - pos_trace)
        grad = replace_nan_inf(grad)
        
        pos_trace = a.clone()
        iter_float = float(iter_err)
        
        # Nadam update
        m = beta_1 * m + (1 - beta_1) * grad
        m = m.reshape(bw.shape[1])
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = v.reshape(bw.shape[1])
        
        m_hat = m / (1 - beta_1**iter_float) + (1 - beta_1) * grad / (1 - beta_1**iter_float)
        v_hat = v / (1 - beta_2**iter_float)
        diff = -lr * m_hat / (torch.sqrt(v_hat) + eps)
        
        # Check bandwidth constraints
        bw_n = torch.abs(a * bw)
        ind = torch.where(bw_n[1, :] < 1e-2)[0]
        if len(ind) > 0:
            bu1 = torch.full((len(ind),), 5e-3, dtype=dtype, device=device)
            gat = bw[1, ind]
            aa1 = bu1 / gat
            a[ind] = aa1
        
        a = a + diff
        
        # Apply bounds
        a_new = check_bound3(a, torch.tensor(4, dtype=dtype, device=device),
                           torch.tensor(1e-2, dtype=dtype, device=device))
        a = a_new
        a = a.reshape(bw.shape[1])
        
        iter_err += 1
    
    converged = torch.all(torch.abs(err - err_trace) < convergence_tol)
    return a, err, iter_err, converged

def fit_banLL2(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, data_x_train, data_s_test,
                n_cop, batch, NORM, norm_flag, pos_trace, max_iter, convergence_tol, lr):
    """
    Fit bandwidth using Nadam optimizer (two bandwidths per copula)
    
    Args:
        a: Initial bandwidth multipliers (2, n_cop)
        bw: Base bandwidth
        Other args same as fit_ban
        
    Returns:
        Same as fit_ban
    """
    device = a.device
    dtype = a.dtype
    eps = 1e-6
    
    iter_err = 1
    err_trace_x1y = torch.ones(n_cop, dtype=dtype, device=device)
    err_trace_xy1 = torch.ones(n_cop, dtype=dtype, device=device)
    
    err = err_trace_x1y + 10 * convergence_tol
    
    # Initialize Adam parameters
    m = torch.zeros_like(bw)
    v = torch.zeros_like(bw)
    beta_1 = 0.9
    beta_2 = 0.999
    
    while (iter_err < max_iter and 
           (torch.any(torch.abs(err - err_trace_x1y) > convergence_tol) or
            torch.any(torch.abs(err - err_trace_xy1) > convergence_tol))):
        
        # Create partial updates
        x1y = torch.cat([pos_trace[0:1], a[1:2]], dim=0)
        xy1 = torch.cat([a[0:1], pos_trace[1:2]], dim=0)
        
        # Compute errors for partial updates
        err_x1y = MISE_mul(x1y, bw, adu11, adu22, step_s, min_s, max_s, grid_x,
                          data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        err_xy1 = MISE_mul(xy1, bw, adu11, adu22, step_s, min_s, max_s, grid_x,
                          data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        
        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        err_trace_x1y = err_trace_x1y.reshape(bw.shape[1])
        err_trace_xy1 = err_trace_xy1.reshape(bw.shape[1])
        
        # Compute full error
        err = MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x,
                      data_x, data_x_train, data_s_test, n_cop, batch, NORM, norm_flag)
        err = err.reshape(bw.shape[1])
        
        # Compute gradients
        grad_x1y = (err - err_trace_x1y) / (a[0] - x1y[0])
        grad_xy1 = (err - err_trace_xy1) / (a[1] - xy1[1])
        
        if grad_x1y.dim() == 1:
            grad_x1y = grad_x1y.unsqueeze(-1)
            grad_xy1 = grad_xy1.unsqueeze(-1)
        
        grad = torch.cat([grad_x1y, grad_xy1], dim=0)
        grad = replace_nan_inf(grad)
        
        pos_trace = a.clone()
        iter_float = float(iter_err)
        
        # Nadam update
        m = beta_1 * m + (1 - beta_1) * grad
        m = m.reshape(bw.shape)
        v = beta_2 * v + (1 - beta_2) * grad**2
        v = v.reshape(bw.shape)
        
        m_hat = m / (1 - beta_1**iter_float) + (1 - beta_1) * grad / (1 - beta_1**iter_float)
        v_hat = v / (1 - beta_2**iter_float)
        diff = -lr * m_hat / (torch.sqrt(v_hat) + eps)
        
        a = a + diff
        
        # Apply bounds
        a_new = check_bound3(a, torch.tensor(2, dtype=dtype, device=device),
                           torch.tensor(1e-2, dtype=dtype, device=device))
        a = a_new
        a = a.reshape(bw.shape)
        
        iter_err += 1
    
    converged = (torch.all(torch.abs(err - err_trace_x1y) < convergence_tol) or
                torch.all(torch.abs(err - err_trace_xy1) < convergence_tol))
    
    return a, err, iter_err, converged 