
def evaluate_fit_fixed(data_dict: dict, grid_dict: dict, par_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fixed version of evaluate_fit with kernel_cdf transformation.
    
    This includes the critical fix that was missing in PyTorch implementation.
    """
    # Import kernel_cdf from TensorFlow or use PyTorch version
    try:
        from DVC_tensorflow.utils.prob_op import kernel_cdf as tf_kernel_cdf
        use_tf_kernel_cdf = True
    except ImportError:
        use_tf_kernel_cdf = False
    
    # Get inputs
    data_s = data_dict['data_s']
    data_x = data_dict['data_x']
    
    grid_u = grid_dict['grid_u']
    grid_s = grid_dict['grid_s'] 
    grid_x = grid_dict['grid_x']
    
    bw = par_dict['bw']
    n_cop = par_dict['n_cop']
    batch_size = par_dict['batch']
    
    device = data_s.device
    dtype = data_s.dtype
    
    # Create grid differentials
    adu11, adu22 = grid_u.diff()
    
    # Create bivariate normal reference
    x1_s, x2_s = grid_s.axis()
    from DVC.utils_prob import biv_norm
    NORM = biv_norm(x1_s, x2_s)
    NORM = NORM.unsqueeze(-1).repeat(1, 1, n_cop).to(device)
    
    # Evaluate local likelihood
    from DVC.utils_locallik import loclik_batch_eval
    ker_grid_fin = loclik_batch_eval(bw, data_x, grid_x, n_cop, batch_size)
    
    # Reshape to grid format
    K = int(np.sqrt(ker_grid_fin.shape[0]))
    ker_grid_all = ker_grid_fin.view(K, K, n_cop).permute(1, 0, 2)
    
    # Add small epsilon for numerical stability (matching TensorFlow)
    ker_grid_all = ker_grid_all + 1e-15 * NORM  # TensorFlow uses 1e-15
    
    # Normalize to get copula density
    from DVC.cop_eval import eval_rs_cop
    pd_grid = eval_rs_cop(adu11, adu22, ker_grid_all, NORM, n_cop)
    pd_grid_uv = pd_grid / NORM
    
    # Compute CDF
    from DVC.vine_eval import cdf_grid_fun
    cdf_grid = cdf_grid_fun(pd_grid_uv, grid_u.ex, adu11, adu22, n_cop)
    
    # Initialize theta updates
    theta_update = torch.zeros((data_s.shape[0], n_cop), device=device, dtype=dtype)
    
    # Interpolate CDF at data points
    from DVC.utils_interpolation import interp_regular_nd_grid
    
    for i in range(n_cop):
        ccdf_data = interp_regular_nd_grid(
            data_s[:, :, i],
            grid_s.min,
            grid_s.max,
            cdf_grid[:, :, i]
        )
        
        # CRITICAL FIX: Apply kernel_cdf transformation
        if use_tf_kernel_cdf:
            # Use TensorFlow's kernel_cdf
            interp_cdf, _, _ = tf_kernel_cdf(
                ccdf_data.cpu().numpy(),
                ccdf_data.cpu().numpy(),
                np.linspace(0, 1, 50)
            )
            theta_update[:, i] = torch.tensor(interp_cdf, device=device, dtype=dtype)
        else:
            # Use PyTorch implementation
            theta_update[:, i] = kernel_cdf_pytorch(ccdf_data)
    
    # Compute gradients if requested
    grad_u, grad_v = None, None
    if par_dict.get('grad_precompute', False):
        # Compute gradients using finite differences
        eps = 1e-4
        grad_u = (cdf_grid[1:, :, :] - cdf_grid[:-1, :, :]) / eps
        grad_v = (cdf_grid[:, 1:, :] - cdf_grid[:, :-1, :]) / eps
    
    return pd_grid_uv, cdf_grid, theta_update, grad_u, grad_v
