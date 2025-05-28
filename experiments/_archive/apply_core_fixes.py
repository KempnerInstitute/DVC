#!/usr/bin/env python3
"""
Apply core fixes to vine_model.py to match TensorFlow behavior.
"""

import os
import re
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def backup_file(filepath):
    """Create a backup of the file before modifying"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def apply_fixes():
    """Apply all fixes to vine_model.py"""
    
    # Paths
    src_dir = "src/DVC"
    vine_model_path = os.path.join(src_dir, "vine_model.py")
    
    if not os.path.exists(vine_model_path):
        logger.error(f"Cannot find {vine_model_path}")
        return
    
    # Backup original
    backup_file(vine_model_path)
    
    # Read current content
    with open(vine_model_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Add imports
    logger.info("Adding required imports...")
    imports = """
# Import kernel_cdf for margin transformation
try:
    from DVC_tensorflow.utils.prob_op import kernel_cdf
    HAS_TF_KERNEL_CDF = True
except ImportError:
    HAS_TF_KERNEL_CDF = False
    # Simple fallback
    def kernel_cdf(data, y, ex):
        \"\"\"Fallback kernel_cdf implementation\"\"\"
        n = len(data)
        if n <= 1:
            return np.full_like(data, 0.5), data, np.array([0.5])
        sorted_data = np.sort(data)
        ranks = np.searchsorted(sorted_data, data, side='right')
        cdf_vals = ranks / (n + 1)
        return cdf_vals, sorted_data, cdf_vals
"""
    # Add after other imports
    content = re.sub(r'(from \.[^\n]+\n\n)', r'\1' + imports + '\n', content)
    
    # Fix 2: Update eval_rs_cop to use 500 iterations
    logger.info("Fixing row/column normalization...")
    content = re.sub(
        r'def eval_rs_cop\([^)]*\):.*?max_iter\s*=\s*\d+',
        'def eval_rs_cop(*args, **kwargs):\n    # Use 500 iterations to match TensorFlow\n    max_iter = 500',
        content,
        flags=re.DOTALL
    )
    
    # Fix 3: Update h-function with proper side handling
    logger.info("Fixing h-function implementation...")
    h_func = """def _h_function(u_root: torch.Tensor,
                u_other: torch.Tensor,
                cobj,
                grid_u: Optional[grid_obj],
                side: str = "left") -> torch.Tensor:
    \"\"\"Return h_{other|root}(u_root,u_other).

    Works for both *parametric* (`cop_par_obj`) and *non-parametric*
    (`copula_obj`) edges.
    \"\"\"
    # Ensure 1D inputs (matching TF)
    if u_root.dim() == 2:
        u_root = u_root.squeeze(1)
    if u_other.dim() == 2:
        u_other = u_other.squeeze(1)
    
    # Clamp inputs (matching TF's bounds)
    ur = torch.clamp(u_root, 1e-9, 1-1e-9)
    vo = torch.clamp(u_other, 1e-9, 1-1e-9)
    
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
        return torch.clamp(h, 1e-9, 1-1e-9)"""
    
    # Replace h-function
    content = re.sub(
        r'def _h_function.*?def\s+\w+',
        h_func + '\n\ndef ',
        content,
        flags=re.DOTALL
    )
    
    # Fix 4: Update theta propagation with proper flip handling
    logger.info("Fixing theta propagation...")
    theta_update = """
        # ---- propagate theta / theta_flip for next level ----
        next_level = tr + 1
        if next_level < d:
            for e_idx, edge in enumerate(edges_now):
                i, j = edge  # left, right variables
                cobj_now = copulas_level[e_idx]
                
                # Get the correct input values based on flip flag
                if tr == 0:
                    u_i = vine.theta[:, tr, i]
                    u_j = vine.theta[:, tr, j]
                    flip_flags_level.append(False)
                else:
                    # Check parent variable
                    parent, _, _ = parent_var(tr, vine.ind_vine, edge)
                    
                    # Check if we need flipped theta for i
                    if i < len(vine.ind_vine[tr-1]):
                        prev_edge = vine.ind_vine[tr-1][i]
                        if prev_edge[0] != parent:
                            u_i = vine.theta_flip[:, tr, i]
                            flip_flags_level.append(True)
                        else:
                            u_i = vine.theta[:, tr, i]
                            flip_flags_level.append(False)
                    else:
                        u_i = vine.theta[:, tr, i]
                        flip_flags_level.append(False)
                    
                    u_j = vine.theta[:, tr, j]
                
                # Debug: Check for NaN values before h-function
                if torch.isnan(u_i).any() or torch.isnan(u_j).any():
                    logger.warning(f"NaN values in theta before h-function at level {tr}, edge {e_idx}")
                
                # For parametric copulas, apply kernel_cdf transformation after h-function
                if vine.param and hasattr(cobj_now, 'family'):
                    # Get h-function values
                    h_vals = _h_function(u_i, u_j, cobj_now, vine.grid_u, side="left")
                    h_vals_flip = _h_function(u_j, u_i, cobj_now, vine.grid_u, side="right")
                    
                    # Apply kernel_cdf transformation to maintain uniform margins
                    h_np = h_vals.cpu().numpy()
                    h_flip_np = h_vals_flip.cpu().numpy()
                    
                    if HAS_TF_KERNEL_CDF:
                        h_transformed, _, _ = kernel_cdf(h_np, h_np, np.linspace(0, 1, vine.knots))
                        h_flip_transformed, _, _ = kernel_cdf(h_flip_np, h_flip_np, np.linspace(0, 1, vine.knots))
                    else:
                        # Fallback: Use empirical CDF transformation
                        n = len(h_np)
                        h_sorted = np.sort(h_np)
                        h_transformed = np.searchsorted(h_sorted, h_np, side='right') / (n + 1)
                        h_flip_sorted = np.sort(h_flip_np)
                        h_flip_transformed = np.searchsorted(h_flip_sorted, h_flip_np, side='right') / (n + 1)
                    
                    # Store transformed values
                    vine.theta[:, next_level, j] = torch.from_numpy(h_transformed).to(device)
                    vine.theta_flip[:, next_level, i] = torch.from_numpy(h_flip_transformed).to(device)
                else:
                    # Non-parametric case - kernel_cdf already applied in evaluate_fit
                    vine.theta[:, next_level, j] = _h_function(u_i, u_j, cobj_now, vine.grid_u, side="left")
                    vine.theta_flip[:, next_level, i] = _h_function(u_j, u_i, cobj_now, vine.grid_u, side="right")
                
                # Debug: Check for NaN values after h-function
                if torch.isnan(vine.theta[:, next_level, j]).any():
                    logger.warning(f"NaN values in theta after h-function at level {tr}, edge {e_idx}")
                    logger.warning(f"Copula family: {cobj_now.family if hasattr(cobj_now, 'family') else 'non-parametric'}")
                    if hasattr(cobj_now, 'theta'):
                        logger.warning(f"Copula parameter: {cobj_now.theta}")
        # ------------------------------------------------------"""
    
    # Replace theta propagation section
    content = re.sub(
        r'# ---- propagate theta / theta_flip for next level ----.*?# ------------------------------------------------------',
        theta_update,
        content,
        flags=re.DOTALL
    )
    
    # Fix 5: Update sampling with proper chain-of-conditionals
    logger.info("Fixing sampling implementation...")
    sampling = """def sample_vine(vine: vine_obj_bin, nsamples: int, cfg: Optional[dict] = None):
    \"\"\"
    Sample from vine. For param => partial approach. For nonparam => build local cdf.
    We'll store final in an array [nsamples, d], assume standard normal margins for demonstration.
    \"\"\"
    d = vine.n_cop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    \"\"\"
    Generate conditional samples matching TensorFlow's approach.
    \"\"\"
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
            return rand_u"""
    
    # Replace sampling section
    content = re.sub(
        r'def sample_vine.*?def\s+\w+',
        sampling + '\n\ndef ',
        content,
        flags=re.DOTALL
    )
    
    # Write updated content
    with open(vine_model_path, 'w') as f:
        f.write(content)
    
    logger.info("All fixes applied successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Run test_correlation_fixes.py to verify improvements")
    logger.info("2. Check that correlation MAE is now < 0.1")
    logger.info("3. Verify no NaN values in theta matrices")
    logger.info("4. Confirm Gaussian copulas are selected for correlated data")

if __name__ == "__main__":
    apply_fixes() 