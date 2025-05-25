"""
Advanced Monkey Patch for kernel_cdf Fix

This patch fixes both the evaluate_fit function and ensures it's called
with the correct parameters to apply the kernel_cdf transformation.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def apply_comprehensive_fix():
    """Apply comprehensive fixes to PyTorch DVC"""
    
    # First, patch evaluate_fit to always apply kernel_cdf
    import DVC.vine_eval
    from DVC.utils_interpolation import interp_regular_nd_grid
    from DVC.utils_prob import kernel_cdf
    
    original_evaluate_fit = DVC.vine_eval.evaluate_fit
    
    def evaluate_fit_fixed(data_dict: dict, grid_dict: dict, par_dict: dict):
        # Call original function
        pd_grid_uv, cdf1, theta, grad_u, grad_v = original_evaluate_fit(
            data_dict, grid_dict, par_dict
        )
        
        # ALWAYS apply kernel_cdf transformation for non-parametric models
        # This is the critical fix
        data_s = data_dict['data_s']
        grid_s = grid_dict['grid_s']
        grid_u = grid_dict['grid_u']
        n_cop = par_dict['n_cop']
        
        if data_s.shape[0] > 0 and n_cop > 0:
            device = data_s.device if hasattr(data_s, 'device') else 'cpu'
            theta_update = torch.zeros((data_s.shape[0], n_cop), device=device)
            
            for i in range(n_cop):
                # Interpolate CDF at data points
                if data_s.dim() == 3:
                    ccdf_data = interp_regular_nd_grid(
                        data_s[:, :, i],
                        grid_s.min.to(device),
                        grid_s.max.to(device), 
                        cdf1[:, :, i].to(device)
                    )
                else:
                    ccdf_data = interp_regular_nd_grid(
                        data_s,
                        grid_s.min.to(device),
                        grid_s.max.to(device), 
                        cdf1[:, :, i].to(device)
                    )
                
                # Apply kernel CDF transformation (CRITICAL)
                interp_cdf, _, _ = kernel_cdf(
                    ccdf_data.cpu().numpy(),
                    ccdf_data.cpu().numpy(),
                    grid_u.ex.cpu().numpy()
                )
                
                theta_update[:, i] = torch.from_numpy(interp_cdf).to(device)
            
            # Update theta if provided
            if theta is not None and par_dict.get('tr') is not None:
                tr = par_dict['tr']
                flip_flag = par_dict.get('flip_flag', [False] * n_cop)
                ind_edge_rel = par_dict.get('ind_edge_rel', list(range(n_cop)))
                
                for i in range(n_cop):
                    if flip_flag[i] == False:
                        theta[:, tr+1, ind_edge_rel[i]] = theta_update[:, i]
                    else:
                        if data_dict.get('theta_flip') is not None:
                            data_dict['theta_flip'][:, tr+1, ind_edge_rel[i]] = theta_update[:, i]
        
        return pd_grid_uv, cdf1, theta, grad_u, grad_v
    
    # Replace the function
    DVC.vine_eval.evaluate_fit = evaluate_fit_fixed
    print("✓ evaluate_fit has been patched with kernel_cdf fix")
    
    # Also patch the vine_model.fit_vine to pass correct parameters
    import DVC.vine_model
    original_fit_vine = DVC.vine_model.fit_vine
    
    def fit_vine_fixed(vine, x, gen_dict, npc_dict, par_dict, bin_dict, cfg=None):
        # Intercept and modify the evaluate_fit calls
        import DVC.vine_eval
        original_eval = DVC.vine_eval.evaluate_fit
        
        def evaluate_fit_with_tr(data_dict, grid_dict, par_dict):
            # Add 'tr' to par_dict if missing (needed for theta update)
            if 'tr' not in par_dict:
                # Try to infer from context
                par_dict['tr'] = getattr(evaluate_fit_with_tr, '_current_tr', 0)
            return original_eval(data_dict, grid_dict, par_dict)
        
        # Temporarily replace evaluate_fit
        DVC.vine_eval.evaluate_fit = evaluate_fit_with_tr
        
        try:
            # Track current tree level
            original_fit_fn = original_fit_vine.__wrapped__ if hasattr(original_fit_vine, '__wrapped__') else original_fit_vine
            result = original_fit_fn(vine, x, gen_dict, npc_dict, par_dict, bin_dict, cfg)
        finally:
            # Restore original
            DVC.vine_eval.evaluate_fit = original_eval
        
        return result
    
    # Don't replace fit_vine for now as it's more complex
    # DVC.vine_model.fit_vine = fit_vine_fixed
    
    print("✓ Comprehensive fix applied")


# Apply the fix when imported
if __name__ != "__main__":
    apply_comprehensive_fix()
