import torch
import torch.distributions as dist
from utils.bijector import NormalCDF
from grid.grid_class import grid_obj
from pre_proc.transformation import Transform

def mk_grid(knots, dtype):
    """Create matrix grid and expanded grid:
    Args:
        knots: num. of knots of the grid.
        dtype: data type for the grid
    Returns:
        coordinates: matrix grid
        expanded: expanded grid
    """
    # Get device from dtype if it's a torch dtype
    device = 'cpu'
    if hasattr(dtype, 'device'):
        device = dtype.device
    
    loc = torch.tensor(0.0, dtype=dtype)
    scale = torch.tensor(1.0, dtype=dtype)
    
    # Create normal CDF bijector
    normal_cdf = NormalCDF(loc, scale)
    
    # Create linearly spaced points and apply inverse CDF
    linear_points = torch.linspace(-3.2, 3.2, knots, dtype=dtype, device=device)
    points = normal_cdf.inverse(linear_points)
    
    # Create meshgrid
    x_grid, y_grid = torch.meshgrid(points, points, indexing='xy')
    
    # Create coordinates (edge points)
    coordinates = torch.stack([x_grid[0, :], y_grid[:, 0]], dim=1)
    
    # Create expanded grid (all points)
    x_grid_flat = x_grid.reshape(-1)
    y_grid_flat = y_grid.reshape(-1)
    expanded = torch.stack([x_grid_flat, y_grid_flat], dim=1)
    
    return coordinates, expanded

def create_grids(knots, device='cpu', dtype=torch.float32):
    """Create all three grids needed for vine copula fitting
    
    Args:
        knots: Number of knots for the grid
        device: Device to create grids on
        dtype: Data type for grids
        
    Returns:
        grid_u: Grid in uniform space
        grid_s: Grid in normal space  
        grid_x: Grid in transformed space
    """
    # Create base grid in uniform space
    u_1, ex_u = mk_grid(knots, dtype)
    
    # Move to specified device
    ex_u = ex_u.to(device)
    
    # Create grid objects
    grid_u = grid_obj(ex_u)
    
    # Transform to normal space
    # The expanded grid has shape (knots*knots, 2), need to reshape for Transform
    # Transform expects (n_samples, 2, n_cop), so add a dimension
    ex_u_3d = ex_u.unsqueeze(-1)  # Shape: (knots*knots, 2, 1)
    trans = Transform(1)  # Single copula
    ex_s_3d = trans.forward_u(ex_u_3d)
    ex_s = ex_s_3d.squeeze(-1)  # Back to 2D
    grid_s = grid_obj(ex_s)
    
    # Transform to x space
    ex_x_3d = trans.forward_s(ex_s_3d)
    ex_x = ex_x_3d.squeeze(-1)  # Back to 2D
    grid_x = grid_obj(ex_x)
    
    return grid_u, grid_s, grid_x 