"""
Core fixes for the DVC implementation to match TensorFlow behavior.

This module contains fixed implementations of critical functions that affect correlation
preservation in the vine copula model. Each function is designed to exactly match the
TensorFlow implementation's behavior.
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple, List, Union
import logging

from .objects import vine_obj_bin, copula_obj, cop_par_obj
from .grid_ops import grid_obj
from .utils_prob import kernel_cdf
from .utils_interpolation import bilinearInterp2d, nearestInterp2d

logger = logging.getLogger(__name__)

def eval_rs_cop_fixed(pd_grid: torch.Tensor,
                     adu11: torch.Tensor,
                     adu22: torch.Tensor,
                     n_cop: int,
                     max_iter: int = 500) -> torch.Tensor:
    """
    Fixed row/column normalization matching TensorFlow's 500 iterations.
    
    Args:
        pd_grid: Input PDF grid to normalize
        adu11, adu22: Grid differentials
        n_cop: Number of copulas
        max_iter: Number of iterations (default 500 to match TF)
    """
    # Initialize
    t2 = pd_grid.clone()
    eps = 1e-30  # Match TF's epsilon
    
    # Do 500 iterations of row/column normalization
    for _ in range(max_iter):
        # Row sums (matching TF's eval1)
        I1 = torch.sum(adu22.unsqueeze(-1) * t2, dim=1)  # [K, n_cop]
        I1 = torch.where(I1 > eps, I1, torch.ones_like(I1))
        
        # Column sums
        I2 = torch.sum(adu11.unsqueeze(-1) * t2, dim=0)  # [K, n_cop]
        I2 = torch.where(I2 > eps, I2, torch.ones_like(I2))
        
        # Update (exactly as TF does)
        t2 = t2 / (torch.outer(I1, I2) + eps)
        t2 = torch.where(torch.isfinite(t2), t2, torch.zeros_like(t2))
        t2 = t2.clamp(min=eps)
    
    return t2

def cdf_grid_fun_fixed(pd_grid_uv: torch.Tensor,
                       ex_u: torch.Tensor,
                       u1d: torch.Tensor,
                       u2d: torch.Tensor,
                       n_cop: int) -> torch.Tensor:
    """
    Fixed CDF grid computation matching TensorFlow exactly.
    
    Args:
        pd_grid_uv: Normalized PDF grid
        ex_u: Grid points
        u1d, u2d: Grid coordinates
        n_cop: Number of copulas
    """
    # Get grid steps (matching TF)
    dx = (ex_u[1] - ex_u[0]).item()
    dy = dx  # Square grid assumed
    
    # Compute CDF (matching TF's cumsum approach)
    cdf = torch.cumsum(torch.cumsum(pd_grid_uv * dy, dim=1) * dx, dim=0)
    
    # Normalize (with TF's epsilon)
    norm = cdf[-1, -1].clamp(min=1e-15)
    cdf = cdf / norm
    
    # Ensure [0,1] range (matching TF's clamp)
    cdf = torch.clamp(cdf, 0.0, 1.0)
    
    return cdf

def h_function_fixed(u_root: torch.Tensor,
                    u_other: torch.Tensor,
                    cobj: Union[copula_obj, cop_par_obj],
                    grid_u: Optional[grid_obj] = None,
                    side: str = "left") -> torch.Tensor:
    """
    Fixed h-function implementation matching TensorFlow's behavior.
    
    Args:
        u_root: Conditioning variable (N,)
        u_other: Variable to condition (N,)
        cobj: Copula object
        grid_u: Grid for non-parametric interpolation
        side: "left" for h(other|root), "right" for h(root|other)
    """
    # Ensure 1D inputs (matching TF)
    if u_root.dim() == 2:
        u_root = u_root.squeeze(1)
    if u_other.dim() == 2:
        u_other = u_other.squeeze(1)
    
    # Clamp inputs (matching TF's bounds)
    ur = torch.clamp(u_root, 1e-9, 1-1e-9)
    vo = torch.clamp(u_other, 1e-9, 1e-9)
    
    # Swap for right side (matching TF's approach)
    if side == "right":
        ur, vo = vo, ur
    
    # Handle parametric copulas
    if hasattr(cobj, "family"):
        fam = cobj.family
        param = cobj.theta
        
        if fam == "ind":
            return vo.clone()
        
        elif fam == "gaussian":
            # Match TF's Gaussian implementation exactly
            rho = float(param) if param is not None else 0.0
            if not math.isfinite(rho):
                rho = 0.0
            rho = max(min(rho, 0.999999), -0.999999)
            
            # Normal scores (with TF's bounds)
            normal = torch.distributions.Normal(0., 1.)
            x = normal.icdf(ur)
            y = normal.icdf(vo)
            x = torch.clamp(x, -8.0, 8.0)
            y = torch.clamp(y, -8.0, 8.0)
            
            # Conditional calculation (matching TF's epsilon)
            denom = max(1.0 - rho*rho, 1e-12)
            z = (y - rho*x) / math.sqrt(denom)
            
            # Handle invalid values (as TF does)
            if torch.isnan(z).any() or torch.isinf(z).any():
                z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
            
            return torch.clamp(normal.cdf(z), 1e-9, 1-1e-9)
        
        elif fam == "clayton":
            # Match TF's Clayton implementation
            alpha = float(param)
            u_m = ur.pow(-alpha-1.0)
            common = (ur.pow(-alpha) + vo.pow(-alpha) - 1.0).pow(-1.0/alpha - 1.0)
            h = u_m * common
            return torch.clamp(h, 1e-9, 1-1e-9)
        
        elif fam == "claytonrot90":
            # Match TF's rotated Clayton
            ur_f = 1.0 - ur
            alpha = float(param)
            u_m = ur_f.pow(-alpha-1.0)
            common = (ur_f.pow(-alpha) + vo.pow(-alpha) - 1.0).pow(-1.0/alpha - 1.0)
            h = u_m * common
            return torch.clamp(1.0 - h, 1e-9, 1-1e-9)
        
        else:
            # Fallback to numerical derivative (matching TF's epsilon)
            eps = 1e-4
            ur2 = torch.clamp(ur + eps, 1e-9, 1-1e-9)
            uv1 = torch.stack([ur, vo], dim=1)
            uv2 = torch.stack([ur2, vo], dim=1)
            from .utils_prob import copulaccdf
            c1 = copulaccdf(cobj, uv2)
            c0 = copulaccdf(cobj, uv1)
            h = (c1 - c0) / eps
            return torch.clamp(h, 1e-9, 1-1e-9)
    
    # Non-parametric case
    else:
        if hasattr(cobj, 'grad_u') and cobj.grad_u is not None:
            # Use precomputed gradients (matching TF's interpolation)
            x_axis, y_axis = grid_u.axis()
            points = torch.stack([ur, vo], dim=1)
            if side == "left":
                return bilinearInterp2d(points, x_axis, y_axis, cobj.grad_u)
            else:
                return bilinearInterp2d(points, x_axis, y_axis, cobj.grad_v)
        
        # Fallback to finite difference (matching TF's grid handling)
        if grid_u is None or cobj.cdf is None:
            raise RuntimeError("Grid information required for nonparam h-function")
        
        x_axis, y_axis = grid_u.axis()
        step = (x_axis[1]-x_axis[0]).item() if x_axis.numel()>1 else 1e-3
        eps = step
        
        points0 = torch.stack([ur, vo], dim=1)
        points1 = torch.stack([torch.clamp(ur+eps, 0.0, 1.0), vo], dim=1)
        
        c0 = nearestInterp2d(points0, x_axis, y_axis, cobj.cdf)
        c1 = nearestInterp2d(points1, x_axis, y_axis, cobj.cdf)
        
        h = (c1 - c0)/(eps+1e-12)  # Match TF's epsilon
        return torch.clamp(h, 1e-9, 1-1e-9)

def update_theta_fixed(vine: vine_obj_bin,
                      tr: int,
                      edge: List[int],
                      cobj: Union[copula_obj, cop_par_obj],
                      u_i: torch.Tensor,
                      u_j: torch.Tensor,
                      parent: int,
                      is_flip: bool) -> None:
    """
    Fixed theta/theta_flip update matching TensorFlow's logic.
    
    Args:
        vine: Vine object
        tr: Current tree level
        edge: Current edge [i,j]
        cobj: Copula object
        u_i, u_j: Input values
        parent: Parent variable
        is_flip: Whether to use flipped version
    """
    next_level = tr + 1
    i, j = edge
    
    if is_flip:
        # Compute h-function with proper side
        h_val = h_function_fixed(u_j, u_i, cobj, vine.grid_u, side="right")
        
        # Apply kernel_cdf to maintain uniform margins (matching TF)
        h_np = h_val.cpu().numpy()
        h_transformed, _, _ = kernel_cdf(h_np, h_np, np.linspace(0, 1, vine.knots))
        
        # Store in theta_flip
        vine.theta_flip[:, next_level, i] = torch.from_numpy(h_transformed).to(h_val.device)
    else:
        # Regular h-function
        h_val = h_function_fixed(u_i, u_j, cobj, vine.grid_u, side="left")
        
        # Apply kernel_cdf
        h_np = h_val.cpu().numpy()
        h_transformed, _, _ = kernel_cdf(h_np, h_np, np.linspace(0, 1, vine.knots))
        
        # Store in theta
        vine.theta[:, next_level, j] = torch.from_numpy(h_transformed).to(h_val.device)

def sample_vine_fixed(vine: vine_obj_bin,
                     nsamples: int,
                     cfg: Optional[dict] = None) -> np.ndarray:
    """
    Fixed vine sampling matching TensorFlow's chain-of-conditionals.
    
    Args:
        vine: Fitted vine object
        nsamples: Number of samples
        cfg: Optional configuration
    """
    d = vine.n_cop
    device = next(iter(vine.copulas[0][0].parameters())).device
    
    # Initialize samples (matching TF's approach)
    samples = torch.zeros((nsamples, d), dtype=torch.float32, device=device)
    normal = torch.distributions.Normal(0., 1.)
    
    # First variable from U(0,1)
    samples[:,0] = normal.icdf(torch.rand(nsamples, device=device))
    
    # Chain of conditionals (matching TF exactly)
    for i in range(1, d):
        lvl = i-1
        edges = vine.copulas[lvl]
        struct_edges = vine.ind_vine[lvl]
        
        # Find the edge connecting to variable i
        edge_idx = None
        for idx, edge in enumerate(struct_edges):
            if edge[1] == i:  # Found connection
                edge_idx = idx
                break
        
        if edge_idx is None:
            continue
        
        cobj = edges[edge_idx]
        edge = struct_edges[edge_idx]
        parent = edge[0]
        
        # Get parent value
        parent_val = samples[:, parent]
        parent_u = normal.cdf(parent_val)
        
        # Generate new value (matching TF's approach)
        rand_u = torch.rand(nsamples, device=device)
        
        if vine.binning:
            # Handle binning (matching TF)
            bins = vine.bins[lvl][edge_idx]
            val_to_bin = torch.bucketize(parent_u, bins) - 1
            val_to_bin = torch.clamp(val_to_bin, 0, len(vine.copulas[lvl][edge_idx])-1)
            
            # Use appropriate bin's copula
            vi = torch.zeros_like(parent_u)
            for bb in range(len(vine.copulas[lvl][edge_idx])):
                mask = (val_to_bin == bb)
                if mask.any():
                    bin_cop = vine.copulas[lvl][edge_idx][bb]
                    vi[mask] = generate_conditional(
                        bin_cop, parent_u[mask], rand_u[mask], is_flip=False
                    )
        else:
            # Regular sampling
            vi = generate_conditional(cobj, parent_u, rand_u, is_flip=False)
        
        # Convert to normal margins
        samples[:,i] = normal.icdf(torch.clamp(vi, 1e-9, 1-1e-9))
    
    return samples.cpu().numpy()

def generate_conditional(cobj: Union[copula_obj, cop_par_obj],
                       u_parent: torch.Tensor,
                       rand_u: torch.Tensor,
                       is_flip: bool) -> torch.Tensor:
    """
    Generate conditional samples matching TensorFlow's approach.
    
    Args:
        cobj: Copula object
        u_parent: Parent values
        rand_u: Random uniforms
        is_flip: Whether to use flipped version
    """
    if hasattr(cobj, 'family'):
        # Parametric copulas
        if cobj.family == "gaussian":
            # Direct method for Gaussian (matching TF)
            rho = float(cobj.theta) if cobj.theta is not None else 0.0
            rho = max(min(rho, 0.999999), -0.999999)
            
            normal = torch.distributions.Normal(0., 1.)
            z = normal.icdf(torch.clamp(u_parent, 1e-9, 1-1e-9))
            e = normal.icdf(torch.clamp(rand_u, 1e-9, 1-1e-9))
            
            denom = max(1.0 - rho*rho, 1e-12)
            y = rho*z + math.sqrt(denom)*e
            
            return normal.cdf(y)
        
        elif cobj.family == "clayton":
            # Clayton sampling (matching TF)
            alpha = float(cobj.theta)
            if is_flip:
                u_parent = 1.0 - u_parent
            
            val = (rand_u.pow(-alpha/(1+alpha)) - u_parent.pow(-alpha) + 1.0).clamp_min(1e-12)
            vi = val.pow(-1.0/alpha)
            
            return 1.0 - vi if is_flip else vi
        
        else:
            # Fallback to copulainvccdf
            from .utils_prob import copulainvccdf
            uv = torch.stack([u_parent, rand_u], dim=1)
            if is_flip:
                uv = torch.stack([rand_u, u_parent], dim=1)
            return copulainvccdf(cobj, uv)
    
    else:
        # Non-parametric sampling (matching TF's grid approach)
        if hasattr(cobj, 'cdf'):
            x_axis, y_axis = cobj.cdf_xlin, cobj.cdf_ylin
            row_idx = torch.bucketize(u_parent, x_axis)
            row_idx = torch.clamp(row_idx, 1, x_axis.numel()-1) - 1
            
            cdf_rows = cobj.cdf[row_idx]
            from .utils_interpolation import inverse_cdf_row
            vi = inverse_cdf_row(rand_u, cdf_rows, y_axis)
            return torch.clamp(vi, 1e-9, 1-1e-9)
        
        else:
            # Fallback to independence
            return rand_u 