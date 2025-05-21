##################################################
# DVC/grid_ops.py
##################################################
import torch

class grid_obj:
    """
    Simple object to hold a 2D grid ex of shape [K^2, 2].
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
        ax1 = torch.unique(self.ex[:, 0])
        ax2 = torch.unique(self.ex[:, 1])
        self.ax1 = ax1
        self.ax2 = ax2
        return ax1, ax2

    def diff(self):
        if self.ax1 is None or self.ax2 is None:
            self.axis()
        d1 = self.ax1.diff(dim=0)
        d2 = self.ax2.diff(dim=0)
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
        mi1 = self.ex[:, 0].min()
        mi2 = self.ex[:, 1].min()
        self.min = torch.stack([mi1, mi2], dim=0)
        return self.min

    def max_grid(self):
        ma1 = self.ex[:, 0].max()
        ma2 = self.ex[:, 1].max()
        self.max = torch.stack([ma1, ma2], dim=0)
        return self.max

    def step_grid(self, tolerance=1e-7):
        if self.diff1 is None or self.diff2 is None:
            self.diff()
        d1_core = self.diff1[:-1]
        d2_core = self.diff2[:-1]
        d1_min, d1_max = d1_core.min(), d1_core.max()
        d2_min, d2_max = d2_core.min(), d2_core.max()
        if (d1_max - d1_min).abs() < tolerance and (d2_max - d2_min).abs() < tolerance:
            step_x = d1_core[0].item()
            step_y = d2_core[0].item()
            self.step = (step_x, step_y)
        else:
            self.step = None
        return self.step

def mk_grid(knots: int, dtype=torch.float32):
    """
    Create a 2D grid with shape [knots^2, 2], mimicking the TF logic.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal_dist = torch.distributions.Normal(0., 1.)

    # define range in normal scale
    start_val = normal_dist.cdf(torch.tensor(-3.2, dtype=dtype, device=device))
    end_val   = normal_dist.cdf(torch.tensor( 3.2, dtype=dtype, device=device))

    points = normal_dist.icdf(torch.linspace(start_val, end_val, knots, dtype=dtype, device=device))
    xx, yy = torch.meshgrid(points, points, indexing='ij')

    expanded = torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=1)
    return expanded