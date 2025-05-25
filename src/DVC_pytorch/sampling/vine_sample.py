import torch
import numpy as np
from utils.prob_op import kernel_cdf
from evalu.vine_eval import evaluate_points
from utils.interpolation import interp1d_torch, interp_regular_nd_grid
from utils.dataset_op import create_bins, check_bins
from vine_tree.tree_op import parent_var
from pre_proc.preparation import prep_copula
from pre_proc.transformation import Transform
from param.cond_copula import copulaccdf, copulainvccdf, copulapdf

#################################### SAMPLING FROM NON-PARAMETRIC COPULA #######################################

############################ INVERSE NON-PARAMETRIC COPULA CDF ####################################

def kerncopccdfinv(w, cdf_grid, u1, u2):
    """
    Inverse CDF for non-parametric copula (sampling)
    
    Args:
        w: Random uniform values (n_samples, 2)
        cdf_grid: CDF values on grid
        u1, u2: Grid axes
        
    Returns:
        U2: Sampled values
    """
    device = w.device
    dtype = w.dtype
    
    # Get dimensions
    len_w = w.shape[0]
    len_ax = u1.shape[0]
    
    # Find nearest index in u1 for w[:,0]
    u1_expanded = u1.unsqueeze(0).expand(len_w, -1)
    w0_expanded = w[:, 0].unsqueeze(1).expand(-1, len_ax)
    
    # Find minimum absolute difference
    m1 = torch.argmin(torch.abs(u1_expanded - w0_expanded), dim=1)
    
    # Gather CDF values
    g = cdf_grid[m1]  # Shape: (len_w, len_ax)
    
    # Find where CDF exceeds w[:,1]
    w1_expanded = w[:, 1].unsqueeze(1).expand(-1, len_ax)
    propro = g - w1_expanded
    
    # Find first positive value
    mask1 = (propro > 0).long()
    ind = torch.argmax(mask1, dim=1)
    
    # Handle case where no positive value found
    ind = torch.clamp(ind, max=len_ax-1)
    
    # Gather U2 values
    U2 = u2[ind]
    
    return U2

def vine_copula_sample(vine, cases):
    """
    Sample from non-parametric vine copula
    
    Args:
        vine: Fitted vine copula object
        cases: Number of samples to generate
        
    Returns:
        sample1: Samples in original space
        u: Samples in uniform space
        sample_pdf: PDF values
        sample_pds: PDS values
    """
    device = vine.grid_u.ex.device
    dtype = vine.grid_u.ex.dtype
    
    d = vine.r_matrix.shape[0]
    n = d - 1
    depth = vine.vine_depth
    
    # Generate uniform random values
    w = torch.rand(cases, d, device=device, dtype=dtype)
    
    # Scale to grid bounds
    mag = torch.max(vine.grid_u.ex).item()
    mig = torch.min(vine.grid_u.ex).item()
    w = (mag - mig) * (w - torch.min(w)) / (torch.max(w) - torch.min(w)) + mig
    
    # Initialize storage
    v = torch.zeros(cases, d, d, device=device, dtype=dtype)
    v_flip = torch.zeros(cases, d, d, device=device, dtype=dtype)
    v[:, 0, 0] = w[:, 0]
    
    # Get grid axes
    u1 = vine.grid_u.ax1
    u2 = vine.grid_u.ax2
    
    # Initialize flip flags
    flip_flag1 = []
    for ii in range(1, d-1):
        flip_flag2 = []
        for j in range(ii):
            flip_flag2.append([])
        flip_flag1.append(flip_flag2)
    
    # Start from first column
    for i in range(1, d):
        v[:, i, i] = w[:, i]
        
        c = 0
        for k in range(i-1, -1, -1):
            tr = k
            col = i - k - 1
            
            if k < len(vine.ind_vine):
                ind_now = vine.ind_vine[k][c]
                
                if hasattr(vine, 'ind_edge_rel') and tr < len(vine.ind_edge_rel):
                    ind_array = np.array(vine.ind_edge_rel[tr])
                    ind_col = np.where(ind_array == col)[0]
                    if len(ind_col) > 0:
                        col = ind_col[0]
                
                if k == 0:
                    # First tree
                    tr1 = n - k
                    col1 = n - i
                    ind1 = vine.r_matrix[tr1, col1].item()
                    ind1 = torch.where(vine.nodes == ind1)[0][0].item()
                    
                    v2 = v[:, k, ind1].unsqueeze(1)
                    v1 = v[:, k+1, i].unsqueeze(1)
                    vv = torch.cat([v2, v1], dim=1)
                    
                    # Sample from copula
                    if hasattr(vine.copulas[tr], 'cdf') and col < vine.copulas[tr].cdf.shape[2]:
                        cdf_grid = vine.copulas[tr].cdf[:, :, col]
                        v[:, k, i] = kerncopccdfinv(vv, cdf_grid, u1, u2)
                    else:
                        # Independence copula
                        v[:, k, i] = torch.rand(cases, device=device, dtype=dtype)
                else:
                    # Higher trees
                    parent, inx1, inx2 = parent_var(k, vine.ind_vine, ind_now)
                    
                    if k > 0 and ind_now[0] < len(vine.ind_vine[k-1]):
                        if vine.ind_vine[k-1][ind_now[0]][0] != parent:
                            v2 = v_flip[:, k, k+ind_now[0]].unsqueeze(1)
                        else:
                            v2 = v[:, k, k+ind_now[0]].unsqueeze(1)
                    else:
                        v2 = v[:, k, k].unsqueeze(1)
                    
                    v1 = v[:, k+1, i].unsqueeze(1)
                    vv = torch.cat([v2, v1], dim=1)
                
                # Sample based on copula type
                if tr > depth:
                    # Independence copula
                    v[:, k, i] = torch.rand(cases, device=device, dtype=dtype)
                else:
                    # Fitted copula
                    if vine.param:
                        # Parametric copula
                        if tr < len(vine.copulas) and col < len(vine.copulas[tr]):
                            cop_p = vine.copulas[tr][col]
                            vv_flip = torch.flip(vv, dims=[1])
                            v[:, k, i] = torch.tensor(
                                copulainvccdf(cop_p, vv_flip.cpu().numpy()),
                                device=device, dtype=dtype
                            )
                        else:
                            v[:, k, i] = torch.rand(cases, device=device, dtype=dtype)
                    else:
                        # Non-parametric copula
                        if hasattr(vine.copulas[tr], 'cdf') and col < vine.copulas[tr].cdf.shape[2]:
                            cdf_grid = vine.copulas[tr].cdf[:, :, col]
                            v[:, k, i] = kerncopccdfinv(vv, cdf_grid, u1, u2)
                        else:
                            v[:, k, i] = torch.rand(cases, device=device, dtype=dtype)
            c += 1
        
        # Compute h-functions for next tree if needed
        if i < d - 1:
            cc1 = 0
            for ii in range(1, i+1):
                cc2 = 0
                for j in range(ii):
                    tr = j
                    col = ii - j - 1
                    
                    if j < len(vine.ind_vine) and ii-1-j < len(vine.ind_vine[j]):
                        ind_now = vine.ind_vine[j][ii-1-j]
                        
                        # Determine flip flag
                        flip_flag = False
                        if j+1 < len(vine.ind_vine) and i-1-j < len(vine.ind_vine[j+1]):
                            ind_sup = vine.ind_vine[j+1][i-1-j]
                            parent, inx1, inx2 = parent_var(j+1, vine.ind_vine, ind_sup)
                            u_edge = {ind_now[0], ind_now[1]}
                            if u_edge.issubset(inx1) or u_edge.issubset(inx2):
                                if ind_now[0] != parent:
                                    flip_flag = True
                        
                        # Store flip flag
                        if cc1 < len(flip_flag1) and cc2 < len(flip_flag1[cc1]):
                            if flip_flag not in flip_flag1[cc1][cc2]:
                                flip_flag1[cc1][cc2].append(flip_flag)
                        
                        # Compute h-function
                        if j == 0:
                            tr1 = n - j
                            col1 = n - ii
                            ind1 = vine.r_matrix[tr1, col1].item()
                            ind1 = torch.where(vine.nodes == ind1)[0][0].item()
                            v2 = v[:, j, ind1].unsqueeze(1)
                        else:
                            parent1, inx1, inx2 = parent_var(j, vine.ind_vine, ind_now)
                            if j > 0 and ind_now[0] < len(vine.ind_vine[j-1]):
                                if vine.ind_vine[j-1][ind_now[0]][0] != parent1:
                                    v2 = v_flip[:, j, j+ind_now[0]].unsqueeze(1)
                                else:
                                    v2 = v[:, j, j+ind_now[0]].unsqueeze(1)
                            else:
                                v2 = v[:, j, j].unsqueeze(1)
                        
                        v1 = v[:, j, ii].unsqueeze(1)
                        
                        if flip_flag:
                            data_u = torch.cat([v1, v2], dim=1)
                        else:
                            data_u = torch.cat([v2, v1], dim=1)
                        
                        # Compute h-function
                        if j > depth:
                            # Independence copula
                            if flip_flag:
                                v_flip[:, j+1, ii] = torch.rand(cases, device=device, dtype=dtype)
                            else:
                                v[:, j+1, ii] = torch.rand(cases, device=device, dtype=dtype)
                        else:
                            # Fitted copula
                            if vine.param:
                                # Parametric
                                if tr < len(vine.copulas) and col < len(vine.copulas[tr]):
                                    cop_p = vine.copulas[tr][col]
                                    vv_flip = torch.flip(data_u, dims=[1])
                                    h_val = torch.tensor(
                                        copulaccdf(cop_p, vv_flip.cpu().numpy()),
                                        device=device, dtype=dtype
                                    )
                                    if flip_flag:
                                        v_flip[:, j+1, ii] = h_val
                                    else:
                                        v[:, j+1, ii] = h_val
                                else:
                                    if flip_flag:
                                        v_flip[:, j+1, ii] = data_u[:, 1]
                                    else:
                                        v[:, j+1, ii] = data_u[:, 0]
                            else:
                                # Non-parametric
                                # Transform data
                                data_u_3d = data_u.unsqueeze(-1)
                                trans = Transform(1)
                                data_s = trans.forward_u(data_u_3d)
                                
                                # Evaluate CDF
                                if hasattr(vine, 'grid_s') and hasattr(vine.copulas[tr], 'cdf'):
                                    if col < vine.copulas[tr].cdf.shape[2]:
                                        cdf1 = vine.copulas[tr].cdf[:, :, col]
                                        
                                        # Interpolate CDF
                                        ccdf_data = interp_regular_nd_grid(
                                            data_s[:, :, 0],
                                            vine.grid_s.min_grid(),
                                            vine.grid_s.max_grid(),
                                            cdf1
                                        )
                                        
                                        # Force to uniform
                                        interp_cdf_poi, _, _ = kernel_cdf(
                                            ccdf_data, ccdf_data, vine.grid_u.ex
                                        )
                                        
                                        if flip_flag:
                                            v_flip[:, j+1, ii] = interp_cdf_poi
                                        else:
                                            v[:, j+1, ii] = interp_cdf_poi
                                    else:
                                        if flip_flag:
                                            v_flip[:, j+1, ii] = data_u[:, 1]
                                        else:
                                            v[:, j+1, ii] = data_u[:, 0]
                                else:
                                    if flip_flag:
                                        v_flip[:, j+1, ii] = data_u[:, 1]
                                    else:
                                        v[:, j+1, ii] = data_u[:, 0]
                    cc2 += 1
                cc1 += 1
    
    # Reorder samples according to R-matrix
    u = v[:, 0, :].reshape(w.shape)
    u1 = torch.zeros_like(u)
    
    c = 0
    for i in range(d-1, -1, -1):
        ind = vine.r_matrix[i, i].item() - 1
        u1[:, ind] = u[:, c]
        
        # Add small random perturbation
        if hasattr(vine.grid_u, 'diff1'):
            u_ax = vine.grid_u.ax1
            gr_diff = vine.grid_u.diff1
            
            # Find nearest grid point
            u_expanded = u[:, c].unsqueeze(1)
            u_ax_expanded = u_ax.unsqueeze(0).expand(cases, -1)
            u_diff = torch.abs(u_ax_expanded - u_expanded)
            ind1 = torch.argmin(u_diff, dim=1)
            diff_val = gr_diff[ind1]
            
            # Add random perturbation
            u1[:, ind] = u[:, c] + diff_val * torch.rand(cases, device=device, dtype=dtype)
        
        c += 1
    
    u = u1
    
    # Transform to original space
    sample1 = torch.zeros(cases, d, device=device, dtype=dtype)
    sample_pdf = torch.zeros(len(vine.Mar_G[0][0]), d, device=device, dtype=dtype)
    sample_pds = torch.zeros(len(vine.Mar_G[0][0]), d, device=device, dtype=dtype)
    
    for i in range(d):
        if i < len(vine.Mar_G) and vine.Mar_G[i] is not None:
            mar_s1 = torch.tensor(vine.Mar_G[i][0], device=device, dtype=dtype)
            mar_p1 = torch.tensor(vine.Mar_G[i][1], device=device, dtype=dtype)
            
            # Interpolate to get samples in original space
            sample1_pro = interp1d_torch(u[:, i], mar_p1, mar_s1)
            sample1[:, i] = prep_copula(sample1_pro, 0)
            
            sample_pdf[:, i] = mar_p1
            sample_pds[:, i] = mar_s1
        else:
            # If margin not available, keep uniform
            sample1[:, i] = u[:, i]
    
    return sample1, u, sample_pdf, sample_pds

def vine_cop_par_sample(vine, cases):
    """
    Sample from parametric vine copula
    
    Args:
        vine: Fitted parametric vine copula object
        cases: Number of samples to generate
        
    Returns:
        sample1: Samples in original space
        u: Samples in uniform space
        sample_pdf: PDF values
        sample_pds: PDS values
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    if hasattr(vine, 'grid_u') and vine.grid_u is not None:
        device = vine.grid_u.ex.device
        dtype = vine.grid_u.ex.dtype
    
    d = vine.r_matrix.shape[0]
    n = d - 1
    
    # Generate uniform random values
    w = torch.rand(cases, d, device=device, dtype=dtype)
    
    # Scale to grid bounds (with small margin)
    if hasattr(vine, 'grid_u') and vine.grid_u is not None:
        mag = torch.max(vine.grid_u.ex).item() - 1e-5
        mig = torch.min(vine.grid_u.ex).item() + 1e-5
        w = (mag - mig) * (w - torch.min(w)) / (torch.max(w) - torch.min(w)) + mig
    
    # Initialize storage
    v = torch.zeros(cases, d, d, device=device, dtype=dtype)
    v[:, 0, 0] = w[:, 0]
    
    # Sample through vine structure
    for i in range(1, d):
        v[:, i, i] = w[:, i]
        
        for k in range(i-1, -1, -1):
            if k == 0:
                # First tree - direct sampling
                ind1 = vine.r_matrix[n, n-i].item() - 1
                ind2 = vine.r_matrix[i-1, i-1].item() - 1
                
                # Get conditional values
                v2 = v[:, k, ind1].unsqueeze(1)
                v1 = v[:, k+1, i].unsqueeze(1)
                vv = torch.cat([v2, v1], dim=1)
                
                # Sample from copula
                if k < len(vine.copulas) and i-k-1 < len(vine.copulas[k]):
                    cop_p = vine.copulas[k][i-k-1]
                    vv_flip = torch.flip(vv, dims=[1])
                    v[:, k, i] = torch.tensor(
                        copulainvccdf(cop_p, vv_flip.cpu().numpy()),
                        device=device, dtype=dtype
                    )
                else:
                    # Independence
                    v[:, k, i] = w[:, i]
            else:
                # Higher trees - use h-functions
                if k < len(vine.ind_vine) and i-k-1 < len(vine.ind_vine[k]):
                    ind_now = vine.ind_vine[k][i-k-1]
                    
                    # Get parent
                    parent, inx1, inx2 = parent_var(k, vine.ind_vine, ind_now)
                    
                    # Get conditional values
                    if k > 0 and ind_now[0] < len(vine.ind_vine[k-1]):
                        if vine.ind_vine[k-1][ind_now[0]][0] != parent:
                            v2 = v[:, k, k+ind_now[0]].unsqueeze(1)  # Use flipped
                        else:
                            v2 = v[:, k, k+ind_now[0]].unsqueeze(1)
                    else:
                        v2 = v[:, k, k].unsqueeze(1)
                    
                    v1 = v[:, k+1, i].unsqueeze(1)
                    vv = torch.cat([v2, v1], dim=1)
                    
                    # Sample from copula
                    if k < len(vine.copulas) and i-k-1 < len(vine.copulas[k]):
                        cop_p = vine.copulas[k][i-k-1]
                        vv_flip = torch.flip(vv, dims=[1])
                        v[:, k, i] = torch.tensor(
                            copulainvccdf(cop_p, vv_flip.cpu().numpy()),
                            device=device, dtype=dtype
                        )
                    else:
                        # Independence
                        v[:, k, i] = w[:, i]
                else:
                    v[:, k, i] = w[:, i]
        
        # Compute h-functions for next level
        if i < d - 1:
            for ii in range(1, i+1):
                for j in range(ii):
                    if j < len(vine.ind_vine) and ii-j-1 < len(vine.ind_vine[j]):
                        ind_now = vine.ind_vine[j][ii-j-1]
                        
                        # Get data for h-function
                        if j == 0:
                            ind1 = vine.r_matrix[n, n-ii].item() - 1
                            v2 = v[:, j, ind1].unsqueeze(1)
                        else:
                            parent, inx1, inx2 = parent_var(j, vine.ind_vine, ind_now)
                            if j > 0 and ind_now[0] < len(vine.ind_vine[j-1]):
                                v2 = v[:, j, j+ind_now[0]].unsqueeze(1)
                            else:
                                v2 = v[:, j, j].unsqueeze(1)
                        
                        v1 = v[:, j, ii].unsqueeze(1)
                        vv = torch.cat([v2, v1], dim=1)
                        
                        # Compute h-function
                        if j < len(vine.copulas) and ii-j-1 < len(vine.copulas[j]):
                            cop_p = vine.copulas[j][ii-j-1]
                            vv_flip = torch.flip(vv, dims=[1])
                            v[:, j+1, ii] = torch.tensor(
                                copulaccdf(cop_p, vv_flip.cpu().numpy()),
                                device=device, dtype=dtype
                            )
                        else:
                            v[:, j+1, ii] = v[:, j, ii]
    
    # Reorder samples according to R-matrix
    u = v[:, 0, :].reshape(w.shape)
    u1 = torch.zeros_like(u)
    
    c = 0
    for i in range(d-1, -1, -1):
        ind = vine.r_matrix[i, i].item() - 1
        u1[:, ind] = u[:, c]
        c += 1
    
    u = u1
    
    # Transform to original space if margins available
    sample1 = u.clone()
    sample_pdf = torch.zeros(1, d, device=device, dtype=dtype)
    sample_pds = torch.zeros(1, d, device=device, dtype=dtype)
    
    if hasattr(vine, 'Mar_G') and vine.Mar_G is not None:
        for i in range(d):
            if i < len(vine.Mar_G) and vine.Mar_G[i] is not None:
                mar_s1 = torch.tensor(vine.Mar_G[i][0], device=device, dtype=dtype)
                mar_p1 = torch.tensor(vine.Mar_G[i][1], device=device, dtype=dtype)
                
                # Interpolate to get samples in original space
                sample1_pro = interp1d_torch(u[:, i], mar_p1, mar_s1)
                sample1[:, i] = prep_copula(sample1_pro, 0)
                
                if sample_pdf.shape[0] < len(mar_p1):
                    sample_pdf = torch.zeros(len(mar_p1), d, device=device, dtype=dtype)
                    sample_pds = torch.zeros(len(mar_s1), d, device=device, dtype=dtype)
                
                sample_pdf[:len(mar_p1), i] = mar_p1
                sample_pds[:len(mar_s1), i] = mar_s1
    
    return sample1, u, sample_pdf, sample_pds 