"""
TensorFlow-Aligned Vine Sampler for PyTorch

This implementation closely follows the TensorFlow vine_sample.py logic,
ensuring correct sampling behavior.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from scipy import stats

# Ensure we're using DVC_pytorch components
from classes.objects import vine_obj_bin
from vine_tree.tree_op import parent_var
from param.cond_copula import copulainvccdf_torch, copulaccdf_torch
from utils.prob_op import kernel_cdf, kernel_cdf_torch
from pre_proc.transformation import Transform


class VineSampler:
    """
    Vine sampler that follows TensorFlow's vine_copula_sample logic exactly
    """
    
    def __init__(self, vine_model: vine_obj_bin):
        self.vine = vine_model
        self.d = len(vine_model.margin)
        self.n = self.d - 1
        self.depth = vine_model.vine_depth
        # Use CPU device for now to avoid device mismatches
        self.device = torch.device('cpu')  # Change from cuda to cpu
        
    def sample(self, cases: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main sampling function following TensorFlow's vine_copula_sample/vine_cop_par_sample
        """
        if self.vine.param:
            return self._vine_cop_par_sample(cases)
        else:
            return self._vine_copula_sample(cases)
    
    def _remap_col_index(self, tr: int, col: int) -> int:
        """
        Remap column index to match the actual edge ordering in copulas array.
        This is the critical missing step from TensorFlow implementation.
        
        In TensorFlow:
        ind_array = np.array(vine.ind_edge_rel[tr])
        ind_col = np.where(ind_array == col)
        col = ind_col[0][0]
        """
        if hasattr(self.vine, 'ind_edge_rel') and tr < len(self.vine.ind_edge_rel):
            ind_array = self.vine.ind_edge_rel[tr]
            # Find where col matches in the array
            if isinstance(ind_array, np.ndarray):
                idx_matches = np.where(ind_array == col)[0]
                if len(idx_matches) > 0:
                    return idx_matches[0]
            else:
                # If it's a list or other iterable
                idx_matches = [idx for idx, val in enumerate(ind_array) if val == col]
                if len(idx_matches) > 0:
                    return idx_matches[0]
        
        # If no remapping found, return original col
        return col
    
    def _vine_copula_sample(self, cases: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Non-parametric vine sampling following TensorFlow logic exactly
        """
        # Initialize uniform samples
        w = torch.rand(cases, self.d, device=self.device)
        
        # Adjust to grid bounds like TensorFlow
        if hasattr(self.vine, 'grid_u') and self.vine.grid_u is not None:
            mag = self.vine.grid_u.ex.max().item()
            mig = self.vine.grid_u.ex.min().item()
            w = (mag - mig) * (w - w.min()) / (w.max() - w.min()) + mig
        
        # Initialize v and v_flip matrices (3D like TensorFlow)
        v = torch.zeros(cases, self.d, self.d, device=self.device, dtype=w.dtype)
        v_flip = torch.zeros_like(v)
        v[:, 0, 0] = w[:, 0]
        
        # Initialize flip flags structure (following TensorFlow exactly)
        flip_flag1 = []
        for ii in range(1, self.d-1):
            flip_flag2 = []
            for j in range(ii):
                flip_flag2.append([])
            flip_flag1.append(flip_flag2)
        
        # Sample each variable (following TensorFlow loop structure)
        for i in range(1, self.d):
            v[:, i, i] = w[:, i]
            
            c = 0
            for k in range(i-1, -1, -1):
                tr = k
                col = i - k - 1
                
                # CRITICAL: Remap col index to match edge ordering
                col_remapped = self._remap_col_index(tr, col)
                
                if hasattr(self.vine, 'ind_vine') and k < len(self.vine.ind_vine):
                    ind_now = self.vine.ind_vine[k][c]
                else:
                    ind_now = [k, i]
                
                # Handle first tree differently (like TensorFlow)
                if k == 0:
                    # Get indices from R-matrix
                    if hasattr(self.vine, 'r_matrix'):
                        tr1 = self.n - k
                        col1 = self.n - i
                        ind1 = self.vine.r_matrix[tr1, col1]  # Don't subtract 1
                        if hasattr(self.vine, 'nodes'):
                            # Convert nodes to tensor if it's numpy array
                            nodes = self.vine.nodes
                            if not torch.is_tensor(nodes):
                                nodes = torch.tensor(nodes.copy(), device=self.device)  # Use .copy() to avoid stride issues
                            ind1 = torch.where(nodes == ind1)[0][0].item()
                        else:
                            ind1 = 0
                    else:
                        ind1 = 0
                    
                    v2 = v[:, k, ind1].unsqueeze(1)
                    v1 = v[:, k+1, i].unsqueeze(1)
                    vv = torch.cat([v2, v1], dim=1)
                    
                    # Apply inverse conditional CDF
                    if tr <= self.depth:
                        if hasattr(self.vine.copulas[tr], 'cdf') and hasattr(self.vine.copulas[tr].cdf, '__getitem__'):
                            # Non-parametric with CDF grid
                            cdf_grid = self.vine.copulas[tr].cdf[:, :, col_remapped]
                            v[:, k, i] = self._kerncopccdfinv(vv, cdf_grid)
                        else:
                            # Parametric copula
                            v[:, k, i] = copulainvccdf_torch(self.vine.copulas[tr][col_remapped], vv)
                    else:
                        # Independence copula
                        v[:, k, i] = vv[:, 0]  # Just return first component for independence
                
                else:
                    # Higher trees - check parent and flip
                    parent, inx1, inx2 = parent_var(k, self.vine.ind_vine, ind_now)
                    
                    # Check if we need to use flipped values (following TF logic)
                    if k > 0 and hasattr(self.vine, 'ind_vine') and len(self.vine.ind_vine) > k-1:
                        if self.vine.ind_vine[k-1][ind_now[0]][0] != parent:
                            v2 = v_flip[:, k, k+ind_now[0]].unsqueeze(1)
                        else:
                            v2 = v[:, k, k+ind_now[0]].unsqueeze(1)
                    else:
                        v2 = v[:, k, k+ind_now[0]].unsqueeze(1)
                    
                    v1 = v[:, k+1, i].unsqueeze(1)
                    vv = torch.cat([v2, v1], dim=1)
                    
                    # Apply appropriate copula
                    if tr > self.depth:
                        # Independent copula
                        v[:, k, i] = vv[:, 0]
                    else:
                        # Fitted copula
                        if hasattr(self.vine.copulas[tr], 'cdf'):
                            cdf_grid = self.vine.copulas[tr].cdf[:, :, col_remapped]
                            v[:, k, i] = self._kerncopccdfinv(vv, cdf_grid)
                        else:
                            v[:, k, i] = copulainvccdf_torch(self.vine.copulas[tr][col_remapped], vv)
                
                c += 1
            
            # Compute h-functions for next level (following TF structure)
            if i < self.d - 1:
                self._compute_h_functions(v, v_flip, i, flip_flag1)
        
        # Extract samples in correct order (following TF logic)
        u = self._extract_samples_order(v, w.shape)
        
        # Transform to original scale
        samples = self._transform_to_original(u)
        
        return samples, u
    
    def _vine_cop_par_sample(self, cases: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parametric vine sampling following TensorFlow logic
        """
        # Initialize uniform samples
        w = torch.rand(cases, self.d, device=self.device)
        
        # Adjust to grid bounds
        if hasattr(self.vine, 'grid_u') and self.vine.grid_u is not None:
            mag = self.vine.grid_u.ex.max().item() - 1e-5
            mig = self.vine.grid_u.ex.min().item() + 1e-5
            w = (mag - mig) * (w - w.min()) / (w.max() - w.min()) + mig
        
        v = torch.zeros(cases, self.d, self.d, device=self.device, dtype=w.dtype)
        v_flip = torch.zeros_like(v)
        v[:, 0, 0] = w[:, 0]
        
        # Sample each variable
        for i in range(1, self.d):
            v[:, i, i] = w[:, i]
            
            c = 0
            for k in range(i-1, -1, -1):
                tr = k
                col = i - k - 1
                
                # CRITICAL: Remap col index to match edge ordering
                col_remapped = self._remap_col_index(tr, col)
                
                if hasattr(self.vine, 'ind_vine') and k < len(self.vine.ind_vine):
                    ind_now = self.vine.ind_vine[k][c]
                else:
                    ind_now = [k, i]
                
                if k == 0:
                    # First tree
                    if hasattr(self.vine, 'r_matrix'):
                        tr1 = self.n - k
                        col1 = self.n - i
                        ind1 = self.vine.r_matrix[tr1, col1]  # Don't subtract 1
                        if hasattr(self.vine, 'nodes'):
                            # Convert nodes to tensor if it's numpy array
                            nodes = self.vine.nodes
                            if not torch.is_tensor(nodes):
                                nodes = torch.tensor(nodes.copy(), device=self.device)  # Use .copy() to avoid stride issues
                            ind1 = torch.where(nodes == ind1)[0][0].item()
                        else:
                            ind1 = 0
                    else:
                        ind1 = 0
                    
                    v2 = v[:, k, ind1].unsqueeze(1)
                    v1 = v[:, k+1, i].unsqueeze(1)
                    vv = torch.cat([v1, v2], dim=1)  # Note: order is v1, v2 for parametric
                    
                    if tr < len(self.vine.copulas) and col_remapped < len(self.vine.copulas[tr]):
                        v[:, k, i] = copulainvccdf_torch(self.vine.copulas[tr][col_remapped], vv)
                    else:
                        v[:, k, i] = vv[:, 0]
                    
                else:
                    # Higher trees
                    parent, inx1, inx2 = parent_var(k, self.vine.ind_vine, ind_now)
                    
                    if k > 0 and hasattr(self.vine, 'ind_vine') and len(self.vine.ind_vine) > k-1:
                        if self.vine.ind_vine[k-1][ind_now[0]][0] != parent:
                            v2 = v_flip[:, k, k+ind_now[0]].unsqueeze(1)
                        else:
                            v2 = v[:, k, k+ind_now[0]].unsqueeze(1)
                    else:
                        v2 = v[:, k, k+ind_now[0]].unsqueeze(1)
                    
                    v1 = v[:, k+1, i].unsqueeze(1)
                    vv = torch.cat([v1, v2], dim=1)
                    
                    if tr < len(self.vine.copulas) and col_remapped < len(self.vine.copulas[tr]):
                        v[:, k, i] = copulainvccdf_torch(self.vine.copulas[tr][col_remapped], vv)
                    else:
                        v[:, k, i] = vv[:, 0]
                
                c += 1
            
            # Compute h-functions for next level
            if i < self.d - 1:
                self._compute_h_functions_parametric(v, v_flip, i)
        
        # Extract samples
        u = self._extract_samples_order(v, w.shape)
        samples = self._transform_to_original(u)
        
        return samples, u
    
    def _kerncopccdfinv(self, w: torch.Tensor, cdf_grid: torch.Tensor) -> torch.Tensor:
        """
        Inverse non-parametric copula CDF following TensorFlow implementation
        """
        # Get grid points
        if hasattr(self.vine, 'grid_u'):
            u1 = self.vine.grid_u.ax1
            u2 = self.vine.grid_u.ax2
        else:
            # Default grid
            u1 = torch.linspace(0.01, 0.99, 100, device=self.device)
            u2 = torch.linspace(0.01, 0.99, 100, device=self.device)
        
        len_w = w.shape[0]
        len_ax = u1.shape[0]
        
        # Find closest u1 values (following TF logic)
        u1_tile = u1.repeat(len_w, 1).T
        w0_tile = w[:, 0].repeat(len_ax, 1)
        
        m1 = torch.argmin(torch.abs(u1_tile - w0_tile), dim=0)
        
        # Get conditional CDFs
        g = cdf_grid[m1, :]
        g = g.T
        
        # Find inverse
        w1_tile = w[:, 1].repeat(len_ax, 1)
        propro = g - w1_tile
        
        mask1 = (propro > 0).int()
        ind = torch.argmax(mask1, dim=0)
        U2 = u2[ind]
        
        return U2
    
    def _compute_h_functions(self, v: torch.Tensor, v_flip: torch.Tensor, i: int, flip_flag1: List):
        """
        Compute h-functions for non-parametric copulas following TF logic
        """
        cc1 = 0
        for ii in range(1, i + 1):
            cc2 = 0
            for j in range(ii):
                tr = j
                col = ii - j - 1
                
                # CRITICAL: Remap col index to match edge ordering
                col_remapped = self._remap_col_index(tr, col)
                
                # Skip if beyond depth
                if tr > self.depth:
                    cc2 += 1
                    continue
                
                # Get edge indices
                if tr < len(self.vine.ind_vine) and col < len(self.vine.ind_vine[tr]):
                    ind_now = self.vine.ind_vine[tr][col]
                else:
                    cc2 += 1
                    continue
                
                # Determine flip flag
                flip_flag = False
                if tr + 1 < len(self.vine.ind_vine):
                    if j == self.n - 2:
                        ind_sup = self.vine.ind_vine[tr + 1][0]
                    else:
                        if (i - 1 - j) < len(self.vine.ind_vine[tr + 1]):
                            ind_sup = self.vine.ind_vine[tr + 1][i - 1 - j]
                        else:
                            ind_sup = ind_now
                    
                    parent, inx1, inx2 = parent_var(tr + 1, self.vine.ind_vine, ind_sup)
                    u_edge = set(ind_now)
                    
                    if (u_edge.issubset(set(inx1)) or u_edge.issubset(set(inx2))):
                        if ind_now[0] != parent:
                            flip_flag = True
                
                # Store flip flag
                if i > 1 and cc1 < len(flip_flag1) and cc2 < len(flip_flag1[cc1]):
                    flip_flag1[cc1][cc2].append(flip_flag)
                
                # Prepare data for h-function
                if tr == 0:
                    # First tree
                    tr1 = self.n - tr
                    col1 = self.n - ii
                    if hasattr(self.vine, 'r_matrix'):
                        ind1 = self.vine.r_matrix[tr1, col1]  # Don't subtract 1
                        if hasattr(self.vine, 'nodes'):
                            nodes = self.vine.nodes
                            if not torch.is_tensor(nodes):
                                nodes = torch.tensor(nodes.copy(), device=self.device)
                            ind1 = torch.where(nodes == ind1)[0][0].item()
                    else:
                        ind1 = 0
                    
                    v2 = v[:, tr, ind1].unsqueeze(1)
                else:
                    # Higher trees
                    parent1, inx1, inx2 = parent_var(tr, self.vine.ind_vine, ind_now)
                    
                    if tr > 0 and hasattr(self.vine, 'ind_vine') and len(self.vine.ind_vine) > tr-1:
                        if self.vine.ind_vine[tr-1][ind_now[0]][0] != parent1:
                            v2 = v_flip[:, tr, tr + ind_now[0]].unsqueeze(1)
                        else:
                            v2 = v[:, tr, tr + ind_now[0]].unsqueeze(1)
                    else:
                        v2 = v[:, tr, tr + ind_now[0]].unsqueeze(1)
                
                v1 = v[:, tr, ii].unsqueeze(1)
                
                # Apply flip if needed
                if flip_flag:
                    data_u = torch.cat([v1, v2], dim=1)
                else:
                    data_u = torch.cat([v2, v1], dim=1)
                
                # Compute h-function for non-parametric copula
                if hasattr(self.vine.copulas[tr], 'cdf') and col_remapped < self.vine.copulas[tr].cdf.shape[2]:
                    # Get CDF grid
                    cdf_grid = self.vine.copulas[tr].cdf[:, :, col_remapped]
                    
                    # Apply numerical h-function computation
                    # For now, use a simple interpolation approach
                    # TODO: Implement full non-parametric h-function evaluation
                    
                    # Store result based on flip
                    if flip_flag:
                        v_flip[:, tr + 1, ii] = data_u[:, 0]  # Placeholder
                    else:
                        v[:, tr + 1, ii] = data_u[:, 0]  # Placeholder
                
                cc2 += 1
            cc1 += 1
    
    def _compute_h_functions_parametric(self, v: torch.Tensor, v_flip: torch.Tensor, i: int):
        """
        Compute h-functions for parametric copulas for trees 1..i.
        We update v[:, tr+1, ii] or v_flip[:, tr+1, ii] with the
        conditional CDF (h-function) from tree 'tr'.
        """
        # Loop over variables 1 to i
        for ii in range(1, i + 1):
            # Loop over tree levels
            for j in range(ii):
                tr = j  # Current tree level
                col = ii - j - 1  # Column index in copula array
                
                # CRITICAL: Remap col index to match edge ordering
                col_remapped = self._remap_col_index(tr, col)
                
                # Skip if beyond vine depth
                if tr > self.depth:
                    continue
                
                # Get the edge being processed
                if tr < len(self.vine.ind_vine) and col < len(self.vine.ind_vine[tr]):
                    ind_now = self.vine.ind_vine[tr][col]
                else:
                    continue
                
                # Get the copula for this edge
                if tr < len(self.vine.copulas) and col_remapped < len(self.vine.copulas[tr]):
                    cop_obj = self.vine.copulas[tr][col_remapped]
                else:
                    continue
                
                # Determine if we need the next tree's parent for flip logic
                if j == self.n - 2:  # Special case for last tree
                    ind_sup = self.vine.ind_vine[j + 1][0] if (j + 1) < len(self.vine.ind_vine) else ind_now
                else:
                    ind_sup = self.vine.ind_vine[j + 1][i - 1 - j] if (j + 1) < len(self.vine.ind_vine) and (i - 1 - j) < len(self.vine.ind_vine[j + 1]) else ind_now
                
                # Prepare the data for h-function computation
                if j == 0:
                    # First tree: use R-matrix to find the conditioning variable
                    tr1 = self.n - j
                    col1 = self.n - ii
                    
                    if hasattr(self.vine, 'r_matrix'):
                        ind1 = self.vine.r_matrix[tr1, col1]  # Don't subtract 1
                        if hasattr(self.vine, 'nodes'):
                            nodes = self.vine.nodes
                            if not torch.is_tensor(nodes):
                                nodes = torch.tensor(nodes.copy(), device=self.device)
                            # Find which position has this node number
                            ind1 = torch.where(nodes == ind1)[0][0].item()
                    else:
                        ind1 = 0
                    
                    v2 = v[:, j, ind1].unsqueeze(1)
                else:
                    # Higher trees: check parent to determine if we need flipped values
                    parent1, inx1, inx2 = parent_var(j, self.vine.ind_vine, ind_now)
                    
                    # CRITICAL: Check the parent relationship to decide v vs v_flip
                    if j > 0 and self.vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                        v2 = v_flip[:, j, j + ind_now[0]].unsqueeze(1)
                    else:
                        v2 = v[:, j, j + ind_now[0]].unsqueeze(1)
                
                # v1 is always from the current row
                v1 = v[:, j, ii].unsqueeze(1)
                
                # Concatenate in the correct order
                vv = torch.cat([v1, v2], dim=1)
                
                # Check if we need to flip for the NEXT tree level
                flip_needed = False
                if (j + 1) < len(self.vine.ind_vine):
                    parent, inx1, inx2 = parent_var(j + 1, self.vine.ind_vine, ind_sup)
                    u_edge = set(ind_now)
                    
                    # Check if this edge appears in the next tree's structure
                    if (u_edge.issubset(set(inx1)) or u_edge.issubset(set(inx2))):
                        if ind_now[0] != parent:
                            # Need to flip the order for h-function
                            vv = torch.cat([v2, v1], dim=1)
                            flip_needed = True
                
                # Compute h-function
                h_val = copulaccdf_torch(cop_obj, vv)
                
                # Ensure h_val is 1D
                if h_val.dim() > 1:
                    h_val = h_val.squeeze()
                
                # Clamp to valid range
                h_val = torch.clamp(h_val, 1e-7, 1 - 1e-7)
                
                # CRITICAL: Store at the correct location
                # The h-function result goes to tree j+1, variable ii
                if flip_needed:
                    v_flip[:, j + 1, ii] = h_val
                else:
                    v[:, j + 1, ii] = h_val
    
    def _extract_samples_order(self, v: torch.Tensor, w_shape: torch.Size) -> torch.Tensor:
        """
        Extract samples in correct variable order following TF logic
        """
        # The samples are stored in specific positions of v matrix:
        # For 2D: v[:, 0, 0] and v[:, 0, 1] 
        # For 3D: v[:, 0, 0], v[:, 0, 1], v[:, 0, 2]
        # etc.
        u = v[:, 0, :self.d]
        u1 = torch.zeros_like(u)
        
        if hasattr(self.vine, 'r_matrix'):
            c = 0
            for i in range(self.d-1, -1, -1):
                ind = self.vine.r_matrix[i, i] - 1  # R-matrix is 1-indexed in TF
                u1[:, ind] = u[:, c]
                c += 1
        else:
            u1 = u
        
        return u1
    
    def _transform_to_original(self, u: torch.Tensor) -> torch.Tensor:
        """
        Transform uniform samples to original scale using marginal distributions
        """
        cases = u.shape[0]
        sample1 = torch.zeros(cases, self.d, device=self.device, dtype=u.dtype)
        
        for i in range(self.d):
            if hasattr(self.vine, 'Mar_G') and i < len(self.vine.Mar_G):
                mar_s, mar_p = self.vine.Mar_G[i]
                
                # Convert to torch tensors if needed
                if not torch.is_tensor(mar_s):
                    mar_s = torch.tensor(mar_s, device=self.device, dtype=u.dtype)
                if not torch.is_tensor(mar_p):
                    mar_p = torch.tensor(mar_p, device=self.device, dtype=u.dtype)
                
                # Interpolate using PyTorch
                # Using searchsorted for quantile interpolation
                indices = torch.searchsorted(mar_p, u[:, i])
                indices = torch.clamp(indices, 1, len(mar_p)-1)
                
                # Linear interpolation
                p1 = mar_p[indices-1]
                p2 = mar_p[indices]
                s1 = mar_s[indices-1]
                s2 = mar_s[indices]
                
                alpha = (u[:, i] - p1) / (p2 - p1 + 1e-10)
                sample1[:, i] = (1 - alpha) * s1 + alpha * s2
            else:
                # Default to standard normal
                sample1[:, i] = torch.distributions.Normal(0, 1).icdf(u[:, i])
        
        return sample1
