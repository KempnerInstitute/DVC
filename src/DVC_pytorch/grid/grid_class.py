import torch
from utils.tensor_op import uniquetol

class grid_obj(object):
    """Grid object."""
    
    def __init__(self, ex):
        """Create a grid object.
        Args:
            ex: Expanded grid.
        """
        self.ex = ex
        self.ax1 = None
        self.ax2 = None
        self.step = None
        self.min = None
        self.max = None
        self.diff1 = None
        self.diff2 = None
    
    def axis(self):
        """Compute axis of the grid"""
        self.ax1 = torch.unique(self.ex[:, 0])
        self.ax2 = torch.unique(self.ex[:, 1])
        return self.ax1, self.ax2
    
    def diff(self):
        """Compute diff vector of axis"""
        if self.ax1 is None:
            self.ax1, self.ax2 = self.axis()
        
        # Calculate differences between consecutive elements
        ad1 = self.ax1[1:] - self.ax1[:-1]
        self.diff1 = torch.cat([ad1, ad1[-1:]], dim=0)
        
        ad2 = self.ax2[1:] - self.ax2[:-1]
        self.diff2 = torch.cat([ad2, ad2[-1:]], dim=0)
        
        return self.diff1, self.diff2
    
    def step_grid(self):
        """Compute the step of the grid"""
        if self.diff1 is None:
            self.diff1, self.diff2 = self.diff()
        
        # Calculate step sizes
        dx = self.diff1 * self.diff2
        dx1 = uniquetol(dx, 1e-5)
        
        if dx1.numel() == 1:
            self.step = dx1
        else:
            raise Exception(f'The grid is not uniform. There are different steps that exceed tolerance: {dx1}')
        
        return self.step
    
    def min_grid(self):
        """Get minimum values of grid"""
        return torch.min(self.ex, dim=0)[0]
    
    def max_grid(self):
        """Get maximum values of grid"""
        return torch.max(self.ex, dim=0)[0] 