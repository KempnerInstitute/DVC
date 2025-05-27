
"""
Monkey Patch Script for kernel_cdf Fix

This script can be imported to apply the kernel_cdf fix to the existing
PyTorch DVC implementation without modifying the source files.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def kernel_cdf_pytorch(data: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of kernel CDF transformation"""
    n = data.shape[0]
    if n <= 1:
        return torch.full_like(data, 0.5)
    
    sorted_data, sort_indices = torch.sort(data)
    ranks = torch.arange(1, n + 1, device=data.device, dtype=data.dtype)
    unsort_indices = torch.argsort(sort_indices)
    return ranks[unsort_indices] / (n + 1)


# Monkey patch the evaluate_fit function
def patch_evaluate_fit():
    """Apply the kernel_cdf fix to evaluate_fit"""
    import DVC_pyolder.vine_eval
    
    # Store original function
    original_evaluate_fit = DVC_pyolder.vine_eval.evaluate_fit
    
    def evaluate_fit_patched(data_dict: dict, grid_dict: dict, par_dict: dict):
        # Call original function
        pd_grid_uv, cdf_grid, theta_update, grad_u, grad_v = original_evaluate_fit(
            data_dict, grid_dict, par_dict
        )
        
        # Apply kernel_cdf fix to theta_update if it exists
        if theta_update is not None:
            device = theta_update.device
            dtype = theta_update.dtype
            
            # Process each column
            for i in range(theta_update.shape[1]):
                col_data = theta_update[:, i]
                
                # Apply kernel_cdf transformation
                try:
                    from DVC_tensorflow.utils.prob_op import kernel_cdf
                    # Use TensorFlow kernel_cdf
                    fixed_data, _, _ = kernel_cdf(
                        col_data.cpu().numpy(),
                        col_data.cpu().numpy(),
                        np.linspace(0, 1, 50)
                    )
                    theta_update[:, i] = torch.tensor(fixed_data, device=device, dtype=dtype)
                except ImportError:
                    # Use PyTorch implementation
                    theta_update[:, i] = kernel_cdf_pytorch(col_data)
        
        return pd_grid_uv, cdf_grid, theta_update, grad_u, grad_v
    
    # Replace the function
    DVC_pyolder.vine_eval.evaluate_fit = evaluate_fit_patched
    print("✓ evaluate_fit has been patched with kernel_cdf fix")


# Apply the patch when imported
if __name__ != "__main__":
    patch_evaluate_fit()
