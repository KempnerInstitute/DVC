##################################################
# DVC/objects.py
##################################################
import torch
import numpy as np
from typing import List, Optional, Union


class copula_obj:
    """
    Nonparametric copula object:
      - 'opt_bw' can store bandwidth parameters
      - 'pd_grid_uv', 'cdf', etc. used for grids
    """
    def __init__(self, opt_bw: Union[torch.Tensor, np.ndarray]):
        if isinstance(opt_bw, np.ndarray):
            opt_bw = torch.from_numpy(opt_bw).float()
        self.opt_bw = opt_bw
        self.pd_grid_uv = None
        self.cdf = None

class cop_par_obj:
    """
    Parametric copula object:
      - 'family': e.g. "gaussian", "student", "clayton", ...
      - 'theta': parameters (could be correlation, df, etc.)
    """
    def __init__(self, family: str, theta: float):
        self.family = family
        self.theta = theta

class margin_obj:
    """
    Marginal distribution object:
      - 'dist': e.g. "norm", "gamma", ...
      - 'theta': parameters like (loc, scale)
      - 'is_cont': boolean
    """
    def __init__(self, dist: str, theta, is_cont=True):
        self.dist = dist
        self.theta = theta
        self.is_cont = is_cont
        self.ker = None  # for nonparam kernel, if needed

class vine_obj_bin:
    """
    A general vine object:
      - 'vine_family': "d-vine", "c-vine", "r-vine"
      - 'families': list of copula families
      - 'n_cop': dimension
      - 'margin': list of margin_obj
      - 'knots': for the numeric grid
      - 'ind_vine': adjacency or edge structure
      - 'copulas': each level's list of copula objects
    """
    def __init__(self, vine_family: str, families: List[str], vine_depth: int,
                 margin: List[margin_obj], knots: int, *args):
        self.vine_family = vine_family
        self.families = families
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots

        self.ind_vine = []
        for i in range(self.n_cop - 1):
            self.ind_vine.append([])

        # Possibly handle method="matrix" or "optimal" if r-vine
        self.method = None
        if self.vine_family == "r-vine":
            if len(args) > 0:
                self.method = args[0]
            # etc.

        self.copulas = []       # list of list-of-copulas per tree level
        self.data_u = None      # in [N, 2, #edge]
        self.grid_u = None      # for the uv grid
        self.grid_s = None      # sometimes the s-t transform
        self.param = False      # if param or nonparam
        self.binning = False
        self.n_bin = 1
        self.fitted = False
        self.sample = None      # We'll patch this method later with correct sampling

        self.d_vine_patched = False

