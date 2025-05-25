import torch
import numpy as np
from utils.tensor_op import check_bound3
from utils.prob_op import biv_norm
from utils.dataset_op import kfold, data_split
from param.copula_fit import fit_gaussian, fit_student, fit_clayton, fit_claytonrot90
from optim.bandwidth import bandwidth_mul
from time import perf_counter

########################## PARAMETRIC FITTING #############################

def parametric_fit(u, families, n_cop):
    """
    Fit parametric copula families and select best based on AIC
    
    Args:
        u: Data tensor of shape (n_samples, 2, n_cop)
        families: List of copula families to fit
        n_cop: Number of copulas
        
    Returns:
        aic2: AIC values for each copula and family
        theta2: Parameter estimates for each copula and family
        logp2: Log-likelihood values for each copula and family
    """
    device = u.device
    dtype = u.dtype
    
    u = check_bound3(u, torch.tensor(1-1e-7, dtype=dtype, device=device),
                    torch.tensor(1e-7, dtype=dtype, device=device))
    
    theta = []
    logp = []
    aic = []
    
    for j in range(len(families)):
        fam = families[j]
        
        if fam == 'ind':
            # Independence copula
            theta_est = []
            for i in range(n_cop):
                theta_est.append([])
            theta.append(theta_est)
            
            p = torch.ones((1, n_cop), dtype=dtype, device=device)
            err = -torch.sum(torch.log(p), dim=0)
            err = err.cpu().numpy()
            logp.append(err)
            
            aic1 = 2 * len(theta_est) + 2 * err
            aic.append(aic1)
        
        elif fam == 'gaussian':
            # Gaussian copula
            pos_trace = torch.full((n_cop,), 0.5, dtype=dtype, device=device)
            lr = torch.tensor(0.005, dtype=dtype, device=device)
            conv_tol = torch.tensor(1e-3, dtype=dtype, device=device)
            max_iter = 100 if u.shape[2] == 1 else 200
            a = pos_trace + lr
            
            theta_est, err, n_iter, conv_flag = fit_gaussian(u, a, pos_trace, conv_tol, lr, max_iter, n_cop)
            
            theta_est = theta_est.cpu().numpy()
            err = err.cpu().numpy()
            n_iter = n_iter if isinstance(n_iter, (int, np.integer)) else n_iter.cpu().numpy()
            conv_flag = conv_flag.cpu().numpy()
            
            theta.append(theta_est)
            logp.append(err)
            aic1 = 2 * theta_est.shape[0] + 2 * err
            aic.append(aic1)
        
        elif fam == 'student':
            # Student-t copula
            n_cop_int = n_cop if isinstance(n_cop, int) else n_cop.item()
            pos_trace = torch.tensor([[0.5, 3.0]], dtype=dtype, device=device)
            pos_trace = pos_trace.repeat(n_cop_int, 1)
            lr = torch.tensor(0.1, dtype=dtype, device=device)
            conv_tol = torch.tensor(5e-1, dtype=dtype, device=device)
            max_iter = 100 if u.shape[2] == 1 else 200
            a = pos_trace + lr
            
            theta_est, err, n_iter, conv_flag = fit_student(u, a, pos_trace, conv_tol, lr, max_iter, n_cop_int)
            
            theta_est = theta_est.cpu().numpy()
            err = err.cpu().numpy()
            n_iter = n_iter if isinstance(n_iter, (int, np.integer)) else n_iter.cpu().numpy()
            conv_flag = conv_flag.cpu().numpy()
            
            theta.append(theta_est)
            logp.append(err)
            aic1 = 2 * theta_est.shape[1] + 2 * err
            aic.append(aic1)
        
        elif fam in ['clayton', 'claytonrot90']:
            # Clayton copula (regular or rotated)
            pos_trace = torch.full((n_cop,), 3.0, dtype=dtype, device=device)
            lr = torch.tensor(0.2, dtype=dtype, device=device)
            conv_tol = torch.tensor(1e-3, dtype=dtype, device=device)
            max_iter = 200
            a = pos_trace + lr
            
            if fam == 'clayton':
                theta_est, err, n_iter, conv_flag = fit_clayton(u, a, pos_trace, conv_tol, lr, max_iter, n_cop)
            else:  # claytonrot90
                theta_est, err, n_iter, conv_flag = fit_claytonrot90(u, a, pos_trace, conv_tol, lr, max_iter, n_cop)
            
            theta_est = theta_est.cpu().numpy()
            err = err.cpu().numpy()
            n_iter = n_iter if isinstance(n_iter, (int, np.integer)) else n_iter.cpu().numpy()
            conv_flag = conv_flag.cpu().numpy()
            
            theta.append(theta_est)
            logp.append(err)
            aic1 = 2 * theta_est.shape[0] + 2 * err
            aic.append(aic1)
    
    # Reorganize results by copula
    aic2 = []
    theta2 = []
    logp2 = []
    
    n_cop_int = n_cop if isinstance(n_cop, int) else n_cop.item()
    
    for i in range(n_cop_int):
        aic22 = []
        theta22 = []
        logp22 = []
        for j in range(len(families)):
            aic22.append(aic[j][i])
            theta22.append(theta[j][i])
            logp22.append(logp[j][i])
        aic2.append(aic22)
        theta2.append(theta22)
        logp2.append(logp22)
    
    return aic2, theta2, logp2

########################## NON-PARAMETRIC FITTING #############################

def optimization(grid_dict, data_dict, par_dict):
    """
    Non-parametric optimization using bandwidth selection
    
    Args:
        grid_dict: Dictionary with grid objects
        data_dict: Dictionary with data arrays
        par_dict: Dictionary with parameters
        
    Returns:
        opt1: Optimized bandwidth multipliers
    """
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s']
    grid_x = grid_dict['grid_x']
    adu11, adu22 = grid_u.diff()
    step_s = grid_s.step_grid()
    min_s = grid_s.min_grid()
    max_s = grid_s.max_grid()
    
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    
    n_cop = par_dict['n_cop']
    batch = par_dict['batch']
    max_iter = par_dict['max_iter']
    lr = par_dict['lr']
    conv_tol = par_dict['conv_tol']
    opt_method = par_dict['opt_method']
    
    device = data_x.device
    dtype = data_x.dtype
    
    # Convert parameters to tensors
    batch = torch.tensor(batch, dtype=torch.int32)
    lr = torch.tensor(lr, dtype=dtype)
    conv_tol = torch.tensor(conv_tol, dtype=dtype)
    max_iter = torch.tensor(max_iter, dtype=torch.int32)
    
    # Bivariate normal for normalization
    x1_s, x2_s = grid_s.axis()
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1)
    NORM = NORM.repeat(1, 1, n_cop)
    
    # Compute bandwidth
    bw = bandwidth_mul(data_x, 2, n_cop)
    
    # Split data for cross-validation
    from utils.dataset_op import kfold, data_split
    train_ind, test_ind = kfold(data_x, 5)
    
    data_s_train = data_split(data_s, train_ind)
    data_s_test = data_split(data_s, test_ind)
    
    data_x_train = data_split(data_x, train_ind)
    data_x_test = data_split(data_x, test_ind)
    
    norm_flag = False
    
    # First stage optimization (without normalization)
    start_time = perf_counter()
    
    if opt_method == 'LL1':
        # Single bandwidth per copula
        a = torch.rand(n_cop, dtype=dtype, device=device) * 1.8 + 0.1
        pos_trace = torch.rand(n_cop, dtype=dtype, device=device) * 1.8 + 0.1
        
        from optim.nadam import fit_ban
        opt1, opt2, opt3, opt4 = fit_ban(
            a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, 
            data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, 
            pos_trace, max_iter[0], conv_tol[0], lr[0]
        )
    elif opt_method == 'LL2':
        # Two bandwidths per copula
        a = torch.rand(2, n_cop, dtype=dtype, device=device) * 1.8 + 0.1
        pos_trace = torch.rand(2, n_cop, dtype=dtype, device=device) * 1.8 + 0.1
        
        from optim.nadam import fit_banLL2
        opt1, opt2, opt3, opt4 = fit_banLL2(
            a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, 
            data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, 
            pos_trace, max_iter[0], conv_tol[0], lr[0]
        )
    
    time_fit = perf_counter() - start_time
    print(f'Time fit stage 1: {time_fit:.2f}s')
    
    # Second stage optimization (with normalization)
    norm_flag = True
    pos_trace = opt1.clone()
    a = opt1 - lr[1]
    
    start_time = perf_counter()
    
    if opt_method == 'LL1':
        opt1, opt2, opt3, opt4 = fit_ban(
            a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, 
            data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, 
            pos_trace, max_iter[1], conv_tol[1], lr[1]
        )
    elif opt_method == 'LL2':
        opt1, opt2, opt3, opt4 = fit_banLL2(
            a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, data_x, 
            data_x_train, data_s_test, n_cop, batch, NORM, norm_flag, 
            pos_trace, max_iter[1], conv_tol[1], lr[1]
        )
    
    time_fit = perf_counter() - start_time
    print(f'Time fit stage 2: {time_fit:.2f}s')
    
    return opt1 