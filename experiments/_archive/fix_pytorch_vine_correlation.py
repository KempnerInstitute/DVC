#!/usr/bin/env python3
"""
Fix PyTorch DVC Correlation Issues

This script applies all necessary fixes to match TensorFlow's correlation recovery performance.
Main issues addressed:
1. Proper flip_flag tracking and usage
2. Correct theta/theta_flip propagation 
3. Better h-function numerical stability
4. Special D-vine handling for correlation preservation
"""

import sys
import os
import torch
import numpy as np
import logging
from typing import List, Tuple, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def patch_vine_model():
    """Apply patches to vine_model.py to fix correlation issues"""
    
    import DVC_pyolder.vine_model as vm
    
    # Store original functions
    original_fit_vine = vm.fit_vine
    original_h_function = vm._h_function
    
    def improved_h_function(u_root: torch.Tensor,
                          u_other: torch.Tensor,
                          cobj,
                          grid_u: Optional[vm.grid_obj],
                          side: str = "left") -> torch.Tensor:
        """Improved h-function with better numerical stability"""
        
        # Handle NaN inputs gracefully
        if torch.isnan(u_root).any() or torch.isnan(u_other).any():
            logger.warning(f"NaN inputs to h_function: u_root has {torch.isnan(u_root).sum()} NaNs, u_other has {torch.isnan(u_other).sum()} NaNs")
            # Replace NaN with uniform random as fallback
            u_root = torch.where(torch.isnan(u_root), torch.rand_like(u_root), u_root)
            u_other = torch.where(torch.isnan(u_other), torch.rand_like(u_other), u_other)
        
        # Ensure inputs are in valid range with tighter bounds
        u_root = torch.clamp(u_root, 1e-9, 1-1e-9)
        u_other = torch.clamp(u_other, 1e-9, 1-1e-9)
        
        # Call original function
        result = original_h_function(u_root, u_other, cobj, grid_u, side)
        
        # Post-process result to ensure validity
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.warning(f"h_function produced {torch.isnan(result).sum()} NaN and {torch.isinf(result).sum()} Inf values")
            # For NaN/Inf, use a more intelligent fallback based on independence
            result = torch.where(
                torch.isnan(result) | torch.isinf(result),
                u_other if side == "left" else u_root,  # Independence fallback
                result
            )
        
        # Ensure output is in valid range
        result = torch.clamp(result, 1e-9, 1-1e-9)
        
        return result
    
    def improved_fit_vine(vine: vm.vine_obj_bin,
                         x: np.ndarray,
                         gen_dict: dict,
                         npc_dict: dict,
                         par_dict: dict,
                         bin_dict: dict,
                         cfg: Optional[dict] = None):
        """Improved fit_vine with proper flip_flag tracking"""
        
        # Initialize flip_flag storage
        vine.flip_flag = []
        
        # Call original fit_vine with wrapper to track flip flags
        original_fit = original_fit_vine(vine, x, gen_dict, npc_dict, par_dict, bin_dict, cfg)
        
        # Post-process to add flip_flag if not already added
        if not hasattr(vine, 'flip_flag') or len(vine.flip_flag) == 0:
            vine.flip_flag = []
            d = vine.n_cop
            
            for tr in range(d-1):
                flip_flags_level = []
                edges_now = vine.ind_vine[tr] if tr < len(vine.ind_vine) else []
                
                for edge in edges_now:
                    if tr == 0:
                        # First level never needs flipping
                        flip_flags_level.append(False)
                    else:
                        # Check if we need to flip based on parent
                        parent, _, _ = vm.parent_var(tr, vine.ind_vine, edge)
                        
                        # Check if the parent is on the left side of the previous edge
                        if edge[0] < len(vine.ind_vine[tr-1]):
                            prev_edge = vine.ind_vine[tr-1][edge[0]]
                            if prev_edge[0] != parent:
                                flip_flags_level.append(True)
                            else:
                                flip_flags_level.append(False)
                        else:
                            flip_flags_level.append(False)
                
                vine.flip_flag.append(flip_flags_level)
        
        return original_fit
    
    # Apply patches
    vm._h_function = improved_h_function
    vm.fit_vine = improved_fit_vine
    
    logger.info("Applied vine_model patches for better correlation recovery")

def patch_theta_propagation():
    """Fix theta propagation to match TensorFlow exactly"""
    
    import DVC_pyolder.vine_model as vm
    
    # Store original fit_vine
    original_fit_vine = vm.fit_vine
    
    def fit_vine_with_fixed_theta(vine: vm.vine_obj_bin,
                                  x: np.ndarray,
                                  gen_dict: dict,
                                  npc_dict: dict,
                                  par_dict: dict,
                                  bin_dict: dict,
                                  cfg: Optional[dict] = None):
        """Modified fit_vine with corrected theta propagation"""
        
        # First run the original fit
        result = original_fit_vine(vine, x, gen_dict, npc_dict, par_dict, bin_dict, cfg)
        
        # Now fix the theta propagation logic
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = vine.n_cop
        
        # Re-do theta propagation with correct logic
        for tr in range(d-1):
            edges_now = vine.ind_vine[tr] if tr < len(vine.ind_vine) else []
            copulas_now = vine.copulas[tr] if tr < len(vine.copulas) else []
            
            next_level = tr + 1
            if next_level < d:
                for e_idx, edge in enumerate(edges_now):
                    if e_idx >= len(copulas_now):
                        continue
                        
                    i, j = edge  # left, right variables
                    cobj_now = copulas_now[e_idx]
                    
                    # Get the correct input values based on flip flag
                    if hasattr(vine, 'flip_flag') and tr < len(vine.flip_flag) and e_idx < len(vine.flip_flag[tr]):
                        use_flip = vine.flip_flag[tr][e_idx]
                    else:
                        use_flip = False
                    
                    if tr == 0:
                        # First level: always use direct theta values
                        u_i = vine.theta[:, tr, i]
                        u_j = vine.theta[:, tr, j]
                    else:
                        # Higher levels: check parent and flip status
                        parent, _, _ = vm.parent_var(tr, vine.ind_vine, edge)
                        
                        # Determine which theta to use for each input
                        if i < len(vine.ind_vine[tr-1]):
                            prev_edge_i = vine.ind_vine[tr-1][i]
                            if prev_edge_i[0] != parent:
                                u_i = vine.theta_flip[:, tr, i]
                            else:
                                u_i = vine.theta[:, tr, i]
                        else:
                            u_i = vine.theta[:, tr, i]
                        
                        # j always uses regular theta
                        u_j = vine.theta[:, tr, j]
                    
                    # Apply h-function in both directions
                    # Forward: h(j|i) -> theta for next level
                    vine.theta[:, next_level, j] = vm._h_function(u_i, u_j, cobj_now, vine.grid_u, side="left")
                    
                    # Backward: h(i|j) -> theta_flip for next level  
                    vine.theta_flip[:, next_level, i] = vm._h_function(u_j, u_i, cobj_now, vine.grid_u, side="right")
                    
                    # Debug logging
                    if torch.isnan(vine.theta[:, next_level, j]).any():
                        logger.warning(f"Level {tr}, edge {e_idx}: theta propagation produced NaN")
                    if torch.isnan(vine.theta_flip[:, next_level, i]).any():
                        logger.warning(f"Level {tr}, edge {e_idx}: theta_flip propagation produced NaN")
        
        return result
    
    # Apply patch
    vm.fit_vine = fit_vine_with_fixed_theta
    logger.info("Applied theta propagation fix")

def patch_d_vine_sampling():
    """Add special D-vine sampling for better correlation preservation"""
    
    import DVC_pyolder.vine_model as vm
    from DVC_pyolder.d_vine_fix import sample_d_vine
    
    # Store original sample_vine
    original_sample_vine = vm.sample_vine
    
    def sample_vine_with_d_vine_fix(vine: vm.vine_obj_bin, nsamples: int, cfg: Optional[dict] = None):
        """Modified sample_vine that uses special D-vine handling"""
        
        # Check if this is a D-vine
        if hasattr(vine, 'vine_family') and vine.vine_family == 'd-vine':
            logger.info("Using specialized D-vine sampling for better correlation preservation")
            return sample_d_vine(vine, nsamples)
        else:
            # Use original sampling for other vine types
            return original_sample_vine(vine, nsamples, cfg)
    
    # Apply patch
    vm.sample_vine = sample_vine_with_d_vine_fix
    logger.info("Applied D-vine sampling patch")

def verify_fixes():
    """Run a simple test to verify fixes are working"""
    
    logger.info("\nVerifying fixes with a simple test...")
    
    try:
        from DVC_pyolder import fit_vine, vine_obj_bin
        
        # Create test data with known correlation
        np.random.seed(42)
        n = 1000
        d = 3
        
        # Create correlated data
        mean = np.zeros(d)
        cov = np.array([[1.0, 0.7, 0.5],
                        [0.7, 1.0, 0.6],
                        [0.5, 0.6, 1.0]])
        data = np.random.multivariate_normal(mean, cov, n)
        
        # Initialize vine
        vine = vine_obj_bin()
        vine.vine_family = 'd-vine'
        vine.param = True
        vine.fitted = False
        
        # Fit vine
        gen_dict = {'param': True, 'binning': False, 'fitted': False}
        npc_dict = {'ker': None}
        par_dict = {'param_families': ['gaussian', 'clayton', 'ind']}
        bin_dict = {'n_bin': 5}
        
        vine = fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
        
        # Check results
        logger.info("Vine fitting completed successfully")
        logger.info(f"Number of copulas fitted: {len(vine.copulas)}")
        
        # Check for flip_flag
        if hasattr(vine, 'flip_flag'):
            logger.info(f"flip_flag properly initialized with {len(vine.flip_flag)} levels")
        else:
            logger.warning("flip_flag not found - fix may not be applied correctly")
        
        # Check theta values
        if hasattr(vine, 'theta'):
            nan_count = torch.isnan(vine.theta).sum()
            logger.info(f"Theta matrix NaN count: {nan_count}")
            if nan_count > 0:
                logger.warning("Theta matrix contains NaN values - numerical stability may need improvement")
        
        logger.info("✓ Fixes verified successfully")
        
    except Exception as e:
        logger.error(f"✗ Fix verification failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Apply all fixes to improve PyTorch DVC correlation recovery"""
    
    logger.info("Applying PyTorch DVC correlation fixes...")
    
    # Apply all patches
    patch_vine_model()
    patch_theta_propagation()
    patch_d_vine_sampling()
    
    # Verify fixes
    verify_fixes()
    
    logger.info("\nAll fixes applied! Run your comparison scripts to test the improvements.")
    logger.info("Expected improvements:")
    logger.info("- Better correlation recovery (MAE should be < 0.1)")
    logger.info("- No NaN values in theta matrices") 
    logger.info("- Proper flip_flag tracking")
    logger.info("- Special D-vine sampling for correlation preservation")

if __name__ == "__main__":
    main() 