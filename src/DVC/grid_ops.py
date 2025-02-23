###############################################
# src/torch_vine/grid_ops.py
###############################################

import torch

class grid_obj:
    """
    Simple object to hold an expanded 2D grid ex: shape [K^2,2],
    plus axis, diffs, etc.
    """

    def __init__(self, ex: torch.Tensor):
        self.ex = ex
        self.ax1 = None
        self.ax2 = None
        self.min = None
        self.max = None
        self.diff1 = None
        self.diff2 = None
        self.step = None

    def axis(self):
        ax1 = torch.unique(self.ex[:,0])
        ax2 = torch.unique(self.ex[:,1])
        self.ax1 = ax1
        self.ax2 = ax2
        return ax1, ax2

    def diff(self):
        if self.ax1 is None or self.ax2 is None:
            self.axis()
        d1 = self.ax1.diff(dim=0)
        d2 = self.ax2.diff(dim=0)
        if len(d1)>0:
            d1 = torch.cat([d1, d1[-1:]],dim=0)
        else:
            d1 = torch.tensor([1.0], device=self.ax1.device)
        if len(d2)>0:
            d2 = torch.cat([d2, d2[-1:]],dim=0)
        else:
            d2 = torch.tensor([1.0], device=self.ax2.device)
        self.diff1 = d1
        self.diff2 = d2
        return d1, d2

    def min_grid(self):
        mi1 = self.ex[:,0].min()
        mi2 = self.ex[:,1].min()
        self.min = torch.stack([mi1, mi2],dim=0)
        return self.min

    def max_grid(self):
        ma1 = self.ex[:,0].max()
        ma2 = self.ex[:,1].max()
        self.max = torch.stack([ma1, ma2],dim=0)
        return self.max

    def step_grid(self):
        """
        check if we have uniform step
        """
        pass