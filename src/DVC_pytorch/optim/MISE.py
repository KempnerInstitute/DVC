import torch
from evalu.cop_eval import eval_rs_p
from optim.local_lik import loclik_batch
from utils.interpolation import interp_regular_nd_grid

############################## MISE COST FUNCTION ################################

def MISE_mul(a, bw, adu11, adu22, step_s, min_s, max_s, grid_x, all_x, data_x_train, data_s_test, n_cop, batch_size, NORM1, norm_flag):
    """
    Mean Integrated Squared Error (MISE) cost function for bandwidth selection
    
    Args:
        a: Bandwidth multiplier
        bw: Base bandwidth
        adu11, adu22: Grid differences
        step_s: Grid step size
        min_s, max_s: Grid bounds
        grid_x: Grid points in x-space
        all_x: All data in x-space
        data_x_train: Training data in x-space
        data_s_test: Test data in s-space
        n_cop: Number of copulas
        batch_size: Batch size for processing
        NORM1: Normalization constant (bivariate normal)
        norm_flag: Whether to normalize
        
    Returns:
        err: MISE error for each copula
    """
    device = bw.device
    dtype = bw.dtype
    
    # Handle 2D data
    if all_x.dim() == 2:
        all_x = all_x.unsqueeze(-1)
    
    bw1 = torch.abs(a * bw)
    n_splits = data_x_train.shape[3]
    
    if grid_x.dim() == 2:
        grid_x = grid_x.unsqueeze(-1)
    
    # Compute kernel density on grid using all data
    ker_grid_all = loclik_batch(bw1, all_x, grid_x, n_cop, batch_size)
    
    if norm_flag:
        ker_grid_all = ker_grid_all.reshape(adu11.shape[0], adu11.shape[0], n_cop).permute(1, 0, 2)
        pd_grid = eval_rs_p(adu11, adu22, ker_grid_all, NORM1, n_cop)
    else:
        pd_grid = torch.zeros((adu11.shape[0], adu11.shape[0], n_cop), dtype=dtype, device=device)
    
    # Cross-validation: evaluate on each fold
    kkk_fin_list = []
    
    for k in range(n_splits):
        # Compute kernel density for this fold
        ker_grid_fin = loclik_batch(bw1, data_x_train[:, :, :, k], grid_x, n_cop, batch_size)
        pd_grid1 = ker_grid_fin.reshape(adu11.shape[0], adu11.shape[0], n_cop).permute(1, 0, 2)
        
        if norm_flag:
            pd_grid1 = eval_rs_p(adu11, adu22, pd_grid1, NORM1, n_cop)
        
        # Interpolate at test points
        interp_data_list = []
        for kk in range(n_cop):
            # Interpolate density at test points
            interp_data1 = interp_regular_nd_grid(
                data_s_test[:, :, kk, k], 
                min_s, max_s, 
                pd_grid1[:, :, kk]
            )
            interp_data_list.append(interp_data1)
        
        interp_data = torch.stack(interp_data_list, dim=1)
        
        # Normalize
        if norm_flag:
            interp_data = interp_data / torch.sum(pd_grid * step_s, dim=[0, 1])
        else:
            interp_data = interp_data / torch.sum(ker_grid_fin * step_s, dim=0)
        
        kkk_fin_list.append(interp_data)
    
    # Stack and reshape
    kkk_fin = torch.cat(kkk_fin_list, dim=0)
    
    # Compute MISE
    if norm_flag:
        err = torch.sum(pd_grid**2 * step_s, dim=[0, 1]) - 2 * torch.mean(kkk_fin, dim=0)
    else:
        pd_grid = ker_grid_all / torch.sum(ker_grid_all * step_s, dim=0)
        err = torch.sum(pd_grid**2 * step_s, dim=0) - 2 * torch.mean(kkk_fin, dim=0)
    
    # Penalize out-of-bounds values
    ind_err = torch.where((a <= 1e-4) | (a >= 2))[0]
    
    if len(ind_err) > 0:
        penalty = 0.001
        for ind in ind_err:
            if err[ind] > 0:
                err[ind] = err[ind] + err[ind] * penalty
            else:
                err[ind] = err[ind] - err[ind] * penalty
    
    # Ensure err is 1D
    if err.dim() == 0:
        err = err.unsqueeze(0)
    
    return err 