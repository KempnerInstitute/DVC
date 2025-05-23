##################################################
# src/DVC/objects.py
##################################################
import torch
import numpy as np
from typing import List, Tuple, Optional

class copula_obj:
    def __init__(self, opt_bw):
        if isinstance(opt_bw, np.ndarray):
            opt_bw = torch.from_numpy(opt_bw).float()
        elif isinstance(opt_bw, float):
            opt_bw = torch.tensor(opt_bw, dtype=torch.float32)
        self.opt_bw = opt_bw
        self.pd_grid_uv = None
        self.cdf = None

class cop_par_obj:
    def __init__(self, family: str, theta):
        self.family = family
        self.theta = theta

class margin_obj:
    def __init__(self, dist: str, theta, is_cont: bool = True):
        self.dist = dist
        self.theta = theta
        self.is_cont = is_cont
        self.ker = None

class vine_obj_bin:
    def __init__(self, vine_family: str,
                 families: List[str],
                 vine_depth: int,
                 margin: List[margin_obj],
                 knots: int,
                 *args):
        self.vine_family = vine_family
        self.families = families
        self.n_cop = vine_depth
        self.margin = margin
        self.knots = knots
        self.ind_vine = [[] for _ in range(self.n_cop - 1)]
        self.method = None
        if self.vine_family == 'r-vine' and len(args) > 0:
            self.method = args[0]
        self.copulas = []
        self.data_u = None
        self.data_s = None
        self.data_x = None
        self.grid_u = None
        self.grid_s = None
        self.grid_x = None
        self.binning = False
        self.n_bin = 1
        self.param = False
        self.fitted = False
        self.sample = None
        self.d_vine_patched = False
        
        # Additional attributes from TF implementation
        self.theta = None
        self.theta_flip = None
        self.Mar_G = None
        self.points_u = None
        self.points_s = None
        self.points_x = None
        self.Fp = None
        self.logf = None
        self.r_matrix = None
        self.nodes = None
        self.matrix_edges = None
        self.ind_edge_rel = []
        self.flip_flag = []
        self.correlations = []
        self.correlations_bins = []

    def select_batch_size_cdf(self, x: torch.Tensor) -> int:
        """Select appropriate batch size for CDF computation based on data size."""
        n_samples = x.shape[0]
        if n_samples > 500000:
            return 2000
        elif n_samples > 200000:
            return 1000
        elif n_samples > 100000:
            return 500
        elif n_samples > 50000:
            return 200
        elif n_samples > 10000:
            return 100
        elif n_samples > 5000:
            return 10
        else:
            return 1

    def select_batch_size(self, data: torch.Tensor) -> int:
        """Select appropriate batch size based on data size."""
        n_samples = data.shape[0]
        if n_samples > 500000:
            return 200
        elif n_samples > 200000:
            return 100
        elif n_samples > 100000:
            return 50
        elif n_samples > 50000:
            return 20
        elif n_samples > 10000:
            return 10
        elif n_samples > 2000:
            return 5
        else:
            return 1

    def evaluation(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the fitted vine on data points.
        
        Args:
            data: Input data, shape [N, d], can be numpy array or torch tensor
            
        Returns:
            p: Joint PDF values
            p_copula: Copula PDF values
            log_marg: Log marginal values
        """
        if not self.fitted:
            raise ValueError("Vine must be fitted before evaluation")
        
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Initialize arrays for results
        N = data.shape[0]
        d = data.shape[1]
        
        # Convert to uniform space [0,1]
        data_u = torch.zeros_like(data)
        for i in range(d):
            col_data = data[:, i].cpu().numpy()
            ranks = col_data.argsort().argsort() + 1
            data_u[:, i] = torch.tensor(ranks / (N + 1.0))
        
        # Evaluate copula density
        p_copula = torch.ones(N)
        
        # For each tree level
        for level, level_copulas in enumerate(self.copulas):
            edges = self.ind_vine[level]
            
            for edge_idx, (i, j) in enumerate(edges):
                # Get copula for this edge
                cop = level_copulas[edge_idx]
                
                # Get data for this edge
                edge_data = torch.stack([data_u[:, i], data_u[:, j]], dim=1)
                
                # Evaluate copula PDF
                if hasattr(cop, 'family'):  # Parametric copula
                    from .param_copula import copulapdf
                    pdf_vals = copulapdf(cop, edge_data)
                else:  # Non-parametric copula
                    # Simple approximation for now
                    pdf_vals = torch.ones(N)
                
                p_copula *= pdf_vals
        
        # For margins, assume standard normal for simplicity
        log_marg = torch.zeros(N)
        p = p_copula  # Joint PDF is copula * margins
        
        return p, p_copula, log_marg

    def fit(self, x, gen_dict, npc_dict, par_dict, bin_dict):
        from .vine_model import fit_vine
        self.binning = gen_dict.get('binning', False)
        self.param = gen_dict.get('param', False)
        self.fitted = gen_dict.get('fitted', False)
        # dimension from data
        if hasattr(x, 'shape'):
            self.n_cop = x.shape[1]
        fit_vine(self, x, gen_dict, npc_dict, par_dict, bin_dict)