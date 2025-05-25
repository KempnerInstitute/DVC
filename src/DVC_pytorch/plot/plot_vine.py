import matplotlib.pyplot as plt
import numpy as np
import torch
from vine_tree.tree_op import edges_index

def plot_vine(typ, vine):
    """
    Plot vine copula structure and densities
    
    Args:
        typ: Plot type - 'cdf', 'pdf', or 'structure'
        vine: Fitted vine copula object
        
    Returns:
        fig: Matplotlib figure object
    """
    d = vine.n_cop
    
    if typ == 'cdf':
        fig, ax = plt.subplots(d, d, sharex='col', sharey='row', figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle('VINE - CDF', weight='bold')
    elif typ == 'pdf':
        fig, ax = plt.subplots(d, d, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.7, wspace=0.7)
        fig.suptitle('VINE - COPULA PDF', weight='bold')
    elif typ == 'structure':
        fig, ax = plt.subplots(d, d, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle('VINE STRUCTURE', weight='bold')
    
    n = vine.r_matrix.shape[0] - 1
    
    # Convert to numpy for plotting
    if torch.is_tensor(vine.r_matrix):
        r_matrix_np = vine.r_matrix.cpu().numpy()
    else:
        r_matrix_np = vine.r_matrix
    
    # PLOT CDF, PDF, etc.
    if typ == 'cdf':
        tr = 0
        for i in range(n, 0, -1):
            if hasattr(vine, 'E'):
                ind_ee = edges_index(vine.E, vine.r_matrix, tr)
            c = 0
            for j in range(i-1, -1, -1):
                if hasattr(vine, 'E') and c < len(ind_ee):
                    edg = ind_ee[c]
                    if hasattr(vine, 'theta') and tr < vine.theta.shape[1]:
                        theta_np = vine.theta.cpu().numpy() if torch.is_tensor(vine.theta) else vine.theta
                        ax[i, j].scatter(theta_np[:, tr, edg[0]], 
                                       theta_np[:, tr, edg[1]], 
                                       s=0.1, marker='.')
                        ax[i, j].set_xlabel('u1')
                        ax[i, j].set_ylabel('u2')
                c += 1
            tr += 1
            
    elif typ == 'pdf':
        tr = 0
        for i in range(n, 0, -1):
            c = 0
            for j in range(i-1, -1, -1):
                if tr < len(vine.copulas):
                    if hasattr(vine.copulas[tr], 'pd_grid_uv'):
                        # Non-parametric copula with grid PDF
                        if c < vine.copulas[tr].pd_grid_uv.shape[2]:
                            pd_grid = vine.copulas[tr].pd_grid_uv[:, :, c]
                            if torch.is_tensor(pd_grid):
                                pd_grid = pd_grid.cpu().numpy()
                            im = ax[i, j].imshow(pd_grid, cmap="jet", aspect='auto',
                                               extent=[0, 1, 0, 1], origin='lower')
                            plt.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
                    else:
                        # Parametric copula
                        if c < len(vine.copulas[tr]):
                            cop = vine.copulas[tr][c]
                            # Create a grid to evaluate parametric copula
                            u_grid = torch.linspace(0.01, 0.99, 50)
                            u1, u2 = torch.meshgrid(u_grid, u_grid, indexing='xy')
                            u_points = torch.stack([u1.flatten(), u2.flatten()], dim=1)
                            
                            # Evaluate copula PDF
                            from param.cond_copula import copulapdf
                            pdf_vals = copulapdf(cop, u_points.numpy())
                            pdf_grid = pdf_vals.reshape(50, 50)
                            
                            im = ax[i, j].imshow(pdf_grid, cmap="jet", aspect='auto',
                                               extent=[0, 1, 0, 1], origin='lower')
                            plt.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
                            
                            # Add copula info
                            ax[i, j].text(0.05, 0.95, f"{cop.family}", 
                                        transform=ax[i, j].transAxes,
                                        fontsize=8, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                c += 1
            tr += 1
    
    # PLOT SUBTITLE - pair labels
    for i in range(n, -1, -1):
        for j in range(i-1, -1, -1):
            str1 = '(' + str(r_matrix_np[i, j]) + ',' + str(r_matrix_np[j, j])
            c = 0
            for ii in range(i+1, n+1):
                if c == 0:
                    str1 = str1 + '|' + str(r_matrix_np[ii, j])
                else:
                    str1 = str1 + ',' + str(r_matrix_np[ii, j])
                c += 1
            str1 = str1 + ')'
            ax[i, j].title.set_text(str1)
            ax[i, j].title.set_fontsize(10)
    
    # PLOT INSIDE SQUARES - diagonal elements
    r1 = np.flip(r_matrix_np)
    for i in range(0, n+1):
        for j in range(i, n+1):
            str1 = str(r1[i, j])
            ax[i, j].text(0.5, 0.5, str1,
                         fontsize=16, ha='center', va='center', weight='bold')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
    
    # Remove unused subplots
    for i in range(d):
        for j in range(d):
            if j > i:
                ax[i, j].axis('off')
    
    plt.tight_layout()
    return fig


def plot_copula_contour(vine, tree_level=0, copula_index=0, n_points=100):
    """
    Plot contour plot of a specific copula in the vine
    
    Args:
        vine: Fitted vine copula object
        tree_level: Tree level (0 for first tree)
        copula_index: Index of copula in that tree
        n_points: Number of grid points
        
    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create grid
    u = torch.linspace(0.01, 0.99, n_points)
    u1, u2 = torch.meshgrid(u, u, indexing='xy')
    u_points = torch.stack([u1.flatten(), u2.flatten()], dim=1)
    
    if tree_level < len(vine.copulas) and copula_index < len(vine.copulas[tree_level]):
        cop = vine.copulas[tree_level][copula_index]
        
        if hasattr(cop, 'pd_grid_uv'):
            # Non-parametric copula
            pdf_vals = cop.pd_grid_uv[:, :, 0] if cop.pd_grid_uv.ndim > 2 else cop.pd_grid_uv
            if torch.is_tensor(pdf_vals):
                pdf_vals = pdf_vals.cpu().numpy()
        else:
            # Parametric copula
            from param.cond_copula import copulapdf
            pdf_vals = copulapdf(cop, u_points.numpy()).reshape(n_points, n_points)
        
        # Create contour plot
        contour = ax.contour(u1.numpy(), u2.numpy(), pdf_vals, levels=15)
        ax.clabel(contour, inline=True, fontsize=8)
        im = ax.imshow(pdf_vals, extent=[0, 1, 0, 1], origin='lower', 
                      cmap='viridis', alpha=0.6, aspect='auto')
        plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('u1')
        ax.set_ylabel('u2')
        ax.set_title(f'Copula PDF - Tree {tree_level}, Copula {copula_index}')
        
        if hasattr(cop, 'family'):
            ax.text(0.05, 0.95, f"Family: {cop.family}", 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_vine_matrix(vine):
    """
    Plot the R-vine matrix structure
    
    Args:
        vine: Fitted vine copula object
        
    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Convert to numpy
    if torch.is_tensor(vine.r_matrix):
        r_matrix = vine.r_matrix.cpu().numpy()
    else:
        r_matrix = vine.r_matrix
    
    n = r_matrix.shape[0]
    
    # Create text representation of matrix
    for i in range(n):
        for j in range(n):
            if j <= i:
                val = r_matrix[i, j]
                if val > 0:
                    ax.text(j, n-1-i, str(int(val)), ha='center', va='center',
                           fontsize=14, weight='bold')
                    # Draw box
                    rect = plt.Rectangle((j-0.4, n-1-i-0.4), 0.8, 0.8, 
                                       fill=False, edgecolor='black', linewidth=2)
                    ax.add_patch(rect)
    
    # Set limits and remove axes
    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.5, n-0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title(f'{vine.vine_family.upper()} Matrix Structure', 
                fontsize=16, weight='bold', pad=20)
    
    # Add labels
    for i in range(n):
        ax.text(-1, n-1-i, f'Tree {i}', ha='right', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig 