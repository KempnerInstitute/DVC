###############################################
# src/DVC/objects.py
###############################################

import torch
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt

class copula_obj:
    """
    Copula object for non-parametric (local-likelihood) fits.
    Holds the optimized bandwidth and optional cdf/pdf on a grid.
    
    In the original code, we often store:
      self.pd_grid_uv (pdf on a 2D grid)
      self.cdf        (cdf on a 2D grid)
      self.opt_bw     (bandwidth)
    """
    def __init__(self, opt_bw: torch.Tensor):
        """
        Args:
            opt_bw: Optimized bandwidth array. Possible shapes:
                - (2, n_cop)
                - (2, n_cop, n_bin) if binning used
        """
        self.opt_bw = opt_bw
        self.pd_grid_uv = None  # 2D PDF, shape [knots, knots, n_cop] if used
        self.cdf = None         # 2D CDF, same shape if used


class cop_par_obj:
    """
    Copula param object for parametric families, e.g. "gaussian", "student", "clayton", etc.
    with 'theta' storing correlation or other parameters.
    """
    def __init__(self, family: str, theta):
        """
        Args:
            family: e.g. "gaussian", "student", "clayton", "claytonrot90", "ind", ...
            theta:  numeric or tuple storing the copula parameter(s)
        """
        self.family = family
        self.theta = theta


class margin_obj:
    """
    Margin object representing a univariate distribution or raw kernel data.
    
    Typically:
      self.dist = 'norm' or 'gamma', etc.
      self.theta = distribution parameters
      self.is_cont = True for continuous
      self.ker = the actual raw data if using a nonparam approach
    """
    def __init__(self, dist: str, theta, is_cont: bool):
        """
        Args:
            dist: e.g. 'norm', 'gamma', etc.
            theta: distribution parameters, e.g. [mu, sigma] for normal
            is_cont: True if continuous
        """
        self.dist = dist
        self.theta = theta
        self.is_cont = is_cont
        self.ker = None  # If storing raw data (like ranks) for kernel-based approach


class vine_obj_bin:
    """
    Main Vine object (can be R-vine, C-vine, or D-vine). It can store:
      - param vs nonparam edges
      - binning info
      - margins
      - adjacency/structure (r_matrix, ind_vine, nodes, etc.)
      - final fitted copulas (copulas)
      - the 'theta' arrays used if flipping or for iterative building
      - optional grid references for CDF/PDF evaluation
      - etc.

    The methods .fit, .evaluation, .sample typically delegate to vine_model.py 
    """

    def __init__(self,
                 vine_family: str,
                 families,
                 vine_depth: int,
                 margin: List[margin_obj],
                 knots: int,
                 method: str,
                 r_matrix=None):
        """
        Args:
            vine_family: 'r-vine', 'c-vine', or 'd-vine'
            families:    If nonparam => 'kercop'; if param => list of possible families
            vine_depth:  dimension of the vine (d)
            margin:      list of margin_obj, one per dimension
            knots:       number of knots for the grid
            method:      'matrix', 'optimal', 'random', ...
            r_matrix:    optional adjacency for R-vine
        """
        self.vine_family = vine_family
        self.families = families
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots
        self.method = method
        self.r_matrix = r_matrix

        # Adjacency / structure storage
        self.ind_vine = []   # e.g. list of edges in each tree level
        self.nodes = None
        self.matrix_edges = None

        # Copulas: for each level we store either param or nonparam objects
        self.copulas = None

        # Additional flags and binning info
        self.param = False          # whether edges are param or nonparam
        self.binning = False        # whether binning is used
        self.n_bin = 1             # number of bins if binning
        self.fitted = False         # if we've run the .fit

        # We store references to possible "flipped" or "theta" arrays
        self.theta = None
        self.theta_flip = None

        # For PDF/CDF evaluation or partial usage
        self.grid_u = None
        self.grid_s = None
        self.grid_x = None

        # In the original code, we might store correlations, flip flags, etc.
        self.correlations = []
        self.correlations_bins = []
        self.flip_flag = []

        # final Fp arrays or logf arrays if we do partial expansions
        self.Fp = None
        self.Fp_flip = None
        self.logf = None
        self.logf_flip = None

    def fit(self,
            x: np.ndarray,
            gen_dict: dict,
            npc_dict: dict,
            par_dict: dict,
            bin_dict: dict,
            cfg: Optional[dict] = None):
        """
        Fit the vine on data x (shape [N,d]) with the given dictionaries:
          gen_dict => general flags (parallel, param, binning, etc.)
          npc_dict => nonparam config (opt_method, batch_parallel, etc.)
          par_dict => param config   (list of families, etc.)
          bin_dict => bin config     (n_bin=..., etc.)

        Implementation is typically in vine_model.py; we just forward.
        """
        # e.g.:
        from .vine_model import fit_vine
        fit_vine(self, x, gen_dict, npc_dict, par_dict, bin_dict, cfg)

        for lvl, edges in enumerate(self.ind_vine):
            print(f"Level {lvl}, edges: {edges}, #copulas stored: {len(self.copulas[lvl])}")

    def evaluation(self, points: torch.Tensor):
        """
        Evaluate the fitted vine PDF at 'points'. 
        """
        from .vine_model import evaluate_vine
        return evaluate_vine(self, points)

    def sample(self, nsamples: int):
        """
        Sample from the fitted vine. 
        """
        from .vine_model import sample_vine
        return sample_vine(self, nsamples)

    def plot_first_level_copulas(self):
        n_first = len(self.copulas[0])
        if n_first == 0:
            print("No copulas were fitted on the first tree level – skipping PDF plots.")
        else:
            fig, axes = plt.subplots(1, min(3, n_first), figsize=(12, 3))
            for ax, cobj in zip(axes, self.copulas[0][:3]):
                if getattr(cobj, "pd_grid_uv", None) is not None:
                    ax.imshow(cobj.pd_grid_uv.cpu().numpy(), origin="lower", cmap="magma")
                ax.axis("off")
            plt.suptitle("First-level copula PDFs")
            plt.show()