import torch
import torch.distributions as dist
from utils.tensor_op import check_bound3, replace_nan_inf
from param.margin_cost import *

################################# GAUSSIAN FITTING ###############################################

def fit_gaussian(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    """Fit Gaussian copula parameters using Adam optimizer"""
    device = u.device
    dtype = u.dtype
    eps = 1e-6
    
    iter_err = 1
    err_trace = torch.ones(n_cop, dtype=dtype, device=device)
    err = err_trace + 10 * convergence_tol
    
    # Initialize Adam parameters
    m = torch.zeros_like(a)
    v = torch.zeros_like(a)
    beta_1 = 0.9
    beta_2 = 0.999
    
    # Make a require gradient for autograd
    a = a.clone().detach().requires_grad_(True)
    
    while iter_err < max_iter and torch.any(torch.abs(err - err_trace) > convergence_tol):
        err_trace = err.clone()
        
        # Compute cost
        err = gaussian_cost(u, a)
        
        # Compute gradient manually (finite differences)
        with torch.no_grad():
            grad = (err - err_trace) / (a - pos_trace)
            grad = replace_nan_inf(grad)
            
            pos_trace = a.clone()
            
            # Adam update
            iter_float = float(iter_err)
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * grad**2
            
            m_hat = m / (1 - beta_1**iter_float)
            v_hat = v / (1 - beta_2**iter_float)
            
            diff = -lr * m_hat / (torch.sqrt(v_hat) + eps)
            a = a + diff
            
            # Bound constraints
            a = check_bound3(a, torch.tensor(1 - 1e-3, dtype=dtype, device=device),
                           torch.tensor(0 + 1e-3, dtype=dtype, device=device))
        
        iter_err += 1
    
    converged = torch.abs(err - err_trace) < convergence_tol
    return a, err, iter_err, converged

################################# STUDENT FITTING ###############################################

def fit_student(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    """Fit Student-t copula parameters using Adam optimizer"""
    device = u.device
    dtype = u.dtype
    eps = 1e-6
    
    iter_err = 1
    err_trace_x1y = torch.ones(n_cop, dtype=dtype, device=device)
    err_trace_xy1 = torch.ones(n_cop, dtype=dtype, device=device)
    
    # Initialize Adam parameters
    m = torch.zeros_like(a)
    v = torch.zeros_like(a)
    beta_1 = 0.9
    beta_2 = 0.999
    
    while (iter_err < max_iter and 
           (torch.any(torch.abs(student_cost(u, a) - err_trace_x1y) > convergence_tol) or
            torch.any(torch.abs(student_cost(u, a) - err_trace_xy1) > convergence_tol))):
        
        # Create partial updates
        x1y = torch.cat([pos_trace[:, 0:1], a[:, 1:2]], dim=1)
        xy1 = torch.cat([a[:, 0:1], pos_trace[:, 1:2]], dim=1)
        
        # Compute costs
        err_x1y = student_cost(u, x1y)
        err_xy1 = student_cost(u, xy1)
        err = student_cost(u, a)
        
        err_trace_x1y = err_x1y
        err_trace_xy1 = err_xy1
        
        # Compute gradients
        with torch.no_grad():
            grad_x1y = (err - err_trace_x1y) / (a[:, 0] - x1y[:, 0])
            grad_xy1 = (err - err_trace_xy1) / (a[:, 1] - xy1[:, 1])
            
            if grad_x1y.dim() == 1:
                grad_x1y = grad_x1y.unsqueeze(-1)
                grad_xy1 = grad_xy1.unsqueeze(-1)
            
            grad = torch.cat([grad_x1y, grad_xy1], dim=1)
            grad = replace_nan_inf(grad)
            
            pos_trace = a.clone()
            
            # Adam update
            iter_float = float(iter_err)
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * grad**2
            
            m_hat = m / (1 - beta_1**iter_float)
            v_hat = v / (1 - beta_2**iter_float)
            
            diff = -lr * m_hat / (torch.sqrt(v_hat) + eps)
            a = a + diff
            
            # Apply bounds
            a[:, 0] = check_bound3(a[:, 0], torch.tensor(1.0, dtype=dtype, device=device),
                                  torch.tensor(-1.0, dtype=dtype, device=device))
            a[:, 1] = check_bound3(a[:, 1], torch.tensor(1000.0, dtype=dtype, device=device),
                                  torch.tensor(1e-3, dtype=dtype, device=device))
        
        iter_err += 1
    
    converged = ((torch.abs(err - err_trace_x1y) < convergence_tol).all() or
                 (torch.abs(err - err_trace_xy1) < convergence_tol).all())
    
    return a, err, iter_err, converged

################################# CLAYTON FITTING ###############################################

def fit_clayton(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    """Fit Clayton copula parameters using Adam optimizer"""
    device = u.device
    dtype = u.dtype
    eps = 1e-6
    
    iter_err = 1
    err_trace = torch.ones(n_cop, dtype=dtype, device=device)
    err = err_trace + 10 * convergence_tol
    
    # Initialize Adam parameters
    m = torch.zeros_like(a)
    v = torch.zeros_like(a)
    beta_1 = 0.9
    beta_2 = 0.999
    
    while iter_err < max_iter and torch.any(torch.abs(err - err_trace) > convergence_tol):
        err_trace = err.clone()
        
        # Compute cost
        err = clayton_cost(u, a)
        
        # Compute gradient
        with torch.no_grad():
            grad = (err - err_trace) / (a - pos_trace)
            grad = replace_nan_inf(grad)
            
            pos_trace = a.clone()
            
            # Adam update
            iter_float = float(iter_err)
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * grad**2
            
            m_hat = m / (1 - beta_1**iter_float)
            v_hat = v / (1 - beta_2**iter_float)
            
            diff = -lr * m_hat / (torch.sqrt(v_hat) + eps)
            a = a + diff
            
            # Apply bounds
            a = check_bound3(a, torch.tensor(20.0, dtype=dtype, device=device),
                           torch.tensor(0.1, dtype=dtype, device=device))
        
        iter_err += 1
    
    converged = torch.abs(err - err_trace) < convergence_tol
    return a, err, iter_err, converged

################################# CLAYTON ROT 90 FITTING ###############################################

def fit_claytonrot90(u, a, pos_trace, convergence_tol, lr, max_iter, n_cop):
    """Fit Clayton rotated 90 degrees copula parameters using Adam optimizer"""
    device = u.device
    dtype = u.dtype
    eps = 1e-6
    
    iter_err = 1
    err_trace = torch.ones(n_cop, dtype=dtype, device=device)
    err = err_trace + 10 * convergence_tol
    
    # Initialize Adam parameters
    m = torch.zeros_like(a)
    v = torch.zeros_like(a)
    beta_1 = 0.9
    beta_2 = 0.999
    
    while iter_err < max_iter and torch.any(torch.abs(err - err_trace) > convergence_tol):
        err_trace = err.clone()
        
        # Compute cost
        err = claytonrot90_cost(u, a)
        
        # Compute gradient
        with torch.no_grad():
            grad = (err - err_trace) / (a - pos_trace)
            grad = replace_nan_inf(grad)
            
            pos_trace = a.clone()
            
            # Adam update
            iter_float = float(iter_err)
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * grad**2
            
            m_hat = m / (1 - beta_1**iter_float)
            v_hat = v / (1 - beta_2**iter_float)
            
            diff = -lr * m_hat / (torch.sqrt(v_hat) + eps)
            a = a + diff
            
            # Apply bounds
            a = check_bound3(a, torch.tensor(20.0, dtype=dtype, device=device),
                           torch.tensor(0.1, dtype=dtype, device=device))
        
        iter_err += 1
    
    converged = torch.abs(err - err_trace) < convergence_tol
    return a, err, iter_err, converged 