###############################################
# src/DVC/grid_ops.py
###############################################

import torch

class grid_obj:
    """
    A simple object to hold a 2D grid 'ex' of shape [K^2, 2],
    plus methods:
      - axis() => unique x and y coordinates
      - diff() => difference arrays
      - min_grid(), max_grid()
      - step_grid() => checks if the grid is uniformly spaced
    """

    def __init__(self, ex: torch.Tensor):
        """
        Args:
          ex: A tensor of shape [K^2, 2], holding (x,y) points of the grid.
        """
        self.ex = ex            # shape [K^2, 2]
        self.ax1 = None         # unique x coords
        self.ax2 = None         # unique y coords
        self.min = None         # (x_min, y_min)
        self.max = None         # (x_max, y_max)
        self.diff1 = None       # diffs along x
        self.diff2 = None       # diffs along y
        self.step = None        # (step_x, step_y) if uniform

    def axis(self):
        """
        Extract unique x and y from 'ex' by looking at columns 0,1.
        Stores them in self.ax1, self.ax2. Returns (ax1, ax2).
        """
        ax1 = torch.unique(self.ex[:, 0])
        ax2 = torch.unique(self.ex[:, 1])
        self.ax1 = ax1
        self.ax2 = ax2
        return ax1, ax2

    def diff(self):
        """
        Compute difference arrays for self.ax1, self.ax2.
        If not yet set, calls axis() first.
        Stores in self.diff1, self.diff2, each of shape [K].
        The last element is duplicated to maintain shape
        consistent with original code logic.
        Returns (d1, d2).
        """
        if self.ax1 is None or self.ax2 is None:
            self.axis()
        d1 = self.ax1.diff(dim=0)   # shape [K-1]
        d2 = self.ax2.diff(dim=0)   # shape [K-1]
        if len(d1) > 0:
            d1 = torch.cat([d1, d1[-1:]], dim=0)
        else:
            d1 = torch.tensor([1.0], device=self.ax1.device)
        if len(d2) > 0:
            d2 = torch.cat([d2, d2[-1:]], dim=0)
        else:
            d2 = torch.tensor([1.0], device=self.ax2.device)
        self.diff1 = d1
        self.diff2 = d2
        return d1, d2

    def min_grid(self):
        """
        Determine the minimum x and y in ex, store in self.min.
        Returns a tensor [2] => (min_x, min_y).
        """
        mi1 = self.ex[:, 0].min()
        mi2 = self.ex[:, 1].min()
        self.min = torch.stack([mi1, mi2], dim=0)
        return self.min

    def max_grid(self):
        """
        Determine the maximum x and y in ex, store in self.max.
        Returns a tensor [2] => (max_x, max_y).
        """
        ma1 = self.ex[:, 0].max()
        ma2 = self.ex[:, 1].max()
        self.max = torch.stack([ma1, ma2], dim=0)
        return self.max

    def step_grid(self, tolerance=1e-7):
        """
        Check if the grid is uniformly spaced in x and y.
        If yes, store (step_x, step_y) in self.step; else None.

        Returns self.step => (step_x, step_y) or None
        """
        if self.diff1 is None or self.diff2 is None:
            self.diff()
        # self.diff1,2 each shape [K], but last element is repeated
        d1_core = self.diff1[:-1]
        d2_core = self.diff2[:-1]
        d1_min, d1_max = d1_core.min(), d1_core.max()
        d2_min, d2_max = d2_core.min(), d2_core.max()

        if (d1_max - d1_min).abs() < tolerance and (d2_max - d2_min).abs() < tolerance:
            # uniform
            step_x = d1_core[0].item()
            step_y = d2_core[0].item()
            self.step = (step_x, step_y)
        else:
            self.step = None
        return self.step


def mk_grid(knots: int, dtype=torch.float32):
    """
    A minimal function to create a [knots^2,2] grid
    over some domain, e.g. [-3.2..3.2]^2.
    If your code references mk_grid, import from this file.

    Returns:
      A tensor ex of shape [knots^2,2].
    """
    x_lin = torch.linspace(-3.2, 3.2, knots, dtype=dtype)
    xx, yy = torch.meshgrid(x_lin, x_lin, indexing='ij')
    ex = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return ex