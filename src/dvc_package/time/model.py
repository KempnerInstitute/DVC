import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from .flows import TimeBandwidthFlow, MLPEdgeFlow
from .nf import Softplus, compute_kl_divergence
from ..core.objects import vine_obj_bin
from ..core.utils_prob import biv_norm
from ..core.grid_ops import mk_grid, grid_obj
from ..core.transformation import Transform
from ..core.utils_locallik import loclik_batch_eval

logger = logging.getLogger("DVC.time")


def create_grids(knots, device=None, dtype=torch.float32):
    """Create U, S, and X grids for vine copula evaluation.

    Args:
        knots: Number of grid knots per dimension.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        (grid_u, grid_s, grid_x) tuple of grid_obj instances.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex_u = mk_grid(knots, dtype=dtype).to(device)
    grid_u = grid_obj(ex_u)
    grid_u.axis()
    grid_u.diff()
    grid_u.min_grid()
    grid_u.max_grid()

    transformer = Transform(1)
    ex_s = transformer.forward_u(ex_u)
    grid_s = grid_obj(ex_s)
    grid_s.axis()
    grid_s.diff()
    grid_s.min_grid()
    grid_s.max_grid()

    # X-space: identity transform for default (no rotation)
    grid_x = grid_obj(ex_s.clone())
    grid_x.axis()
    grid_x.diff()
    grid_x.min_grid()
    grid_x.max_grid()

    return grid_u, grid_s, grid_x


class TimeDependentVineCopula(nn.Module):
    """Wrap a vine structure with per-edge time-dependent bandwidth flows."""

    def __init__(self, vine: vine_obj_bin, n_time_steps: int, hidden_dim: int = 64,
                 edge_flow_cls=MLPEdgeFlow):
        super().__init__()
        self.vine = vine
        self.n_time_steps = n_time_steps
        self.hidden_dim = hidden_dim

        # Build one flow per edge per tree up to vine.vine_depth
        self.edge_flows: nn.ModuleList = nn.ModuleList()
        self.edge_index_map: List[tuple] = []  # (tree, edge_index)
        for tr in range(min(self.vine.n_cop - 1, getattr(self.vine, 'vine_depth', self.vine.n_cop - 1) + 1)):
            edges = self.vine.ind_vine[tr] if tr < len(self.vine.ind_vine) else []
            for ei, _ in enumerate(edges):
                # Create flow per edge
                self.edge_flows.append(edge_flow_cls(hidden_dim=hidden_dim, out_dim=2, dropout=0.1))
                self.edge_index_map.append((tr, ei))

        # Grids for evaluation (shared)
        knots = self.vine.knots
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32
        self.grid_u, self.grid_s, self.grid_x = create_grids(knots, device=device, dtype=dtype)

    def forward(self, times_norm: torch.Tensor) -> torch.Tensor:
        """Return concatenated bandwidths for all edges at given times."""
        bws = []
        for flow in self.edge_flows:
            bws.append(flow(times_norm))  # (batch, 2)
        return torch.stack(bws, dim=2)  # (batch, 2, n_edges)

    def negative_log_likelihood(self, data_t: torch.Tensor, times_t: torch.Tensor) -> torch.Tensor:
        """Compute NLL across a batch of (data, time)."""
        device = data_t.device
        dtype = data_t.dtype
        batch = data_t.shape[0]
        # Normalize time to [0,1]
        t_min = times_t.min()
        t_max = times_t.max()
        t_norm = (times_t - t_min) / (t_max - t_min + 1e-8)
        # Predict bandwidths for all edges
        bw_edges = self.forward(t_norm)  # (batch, 2, n_edges)

        # Prepare per-time evaluation on a deduplicated grid.
        grid_eval_base = self.grid_x.ex
        if grid_eval_base.dim() == 3 and grid_eval_base.shape[-1] == 1:
            grid_eval_base = grid_eval_base.squeeze(-1)
        if grid_eval_base.dim() != 2 or grid_eval_base.shape[1] != 2:
            raise ValueError(f"Unexpected grid shape for likelihood evaluation: {tuple(grid_eval_base.shape)}")

        grid_eval_unique = torch.unique(grid_eval_base, dim=0)
        if grid_eval_unique.shape[0] == 0:
            raise ValueError("No grid points available for likelihood evaluation.")
        grid_eval = grid_eval_unique.unsqueeze(-1).to(device=device, dtype=dtype)  # (M, 2, 1)

        x_vals = torch.sort(torch.unique(grid_eval_unique[:, 0]))[0]
        y_vals = torch.sort(torch.unique(grid_eval_unique[:, 1]))[0]
        NORM = biv_norm(x_vals, y_vals).unsqueeze(-1).to(device=device, dtype=dtype)
        grid_h, grid_w = NORM.shape[0], NORM.shape[1]

        nll_total = torch.zeros((), device=device, dtype=dtype)

        # Map edge index to tree
        # We evaluate each tree separately to re-use existing kernels; this is simplified
        for edge_idx, (tr, rel_idx) in enumerate(self.edge_index_map):
            if self.vine.data_x is None:
                raise ValueError("vine.data_x is required for negative_log_likelihood but is not initialized.")
            if self.vine.data_x.dim() != 3:
                raise ValueError(f"Expected vine.data_x to have shape [N, 2, E], got {tuple(self.vine.data_x.shape)}")

            # Prefer global edge indexing; fallback to rel_idx for older layouts.
            if edge_idx < self.vine.data_x.shape[2]:
                data_x = self.vine.data_x[:, :, edge_idx]
            elif rel_idx < self.vine.data_x.shape[2]:
                data_x = self.vine.data_x[:, :, rel_idx]
            else:
                raise IndexError(
                    f"No data for edge_idx={edge_idx}, rel_idx={rel_idx}, available={self.vine.data_x.shape[2]}"
                )
            data_x = data_x.to(device)

            # For each time in batch, evaluate loc-lik and accumulate NLL via nearest interpolation
            bw_pair = bw_edges[:, :, edge_idx]  # (batch, 2)
            for b in range(batch):
                B = bw_pair[b].unsqueeze(1)  # (2, 1)
                ker_grid = loclik_batch_eval(B, data_x.unsqueeze(-1), grid_eval, 1, batch_size=1)

                # Normalize possible output layouts from local-likelihood code to [M, 1].
                if ker_grid.dim() == 3 and ker_grid.shape[0] == data_x.shape[0]:
                    ker_grid = ker_grid.mean(dim=0)
                elif ker_grid.dim() == 3 and ker_grid.shape[1] == data_x.shape[0]:
                    ker_grid = ker_grid.mean(dim=1)
                elif ker_grid.dim() > 2:
                    ker_grid = ker_grid.reshape(ker_grid.shape[0], -1).mean(dim=1, keepdim=True)
                elif ker_grid.dim() == 1:
                    ker_grid = ker_grid.unsqueeze(-1)

                target = grid_h * grid_w
                if ker_grid.shape[0] < target:
                    pad_rows = target - ker_grid.shape[0]
                    ker_grid = torch.cat([ker_grid, ker_grid[-1:].repeat(pad_rows, 1)], dim=0)
                elif ker_grid.shape[0] > target:
                    ker_grid = ker_grid[:target]

                ker_grid = ker_grid.reshape(grid_h, grid_w, 1).permute(1, 0, 2)
                pdf_uv = ker_grid / (NORM + 1e-12)
                avg_pdf = torch.clamp(pdf_uv.mean(), 1e-10)
                nll_total = nll_total - torch.log(avg_pdf)

        return nll_total / (batch * max(len(self.edge_index_map), 1))

    def entropy_based_loss(self, 
                          data_t: torch.Tensor, 
                          times_t: torch.Tensor,
                          entropy_weight: float = 0.1) -> torch.Tensor:
        """
        Compute entropy-based loss for time-dependent vine copula optimization.
        
        This implements the entropy-based optimization approach from DVC_NF,
        which balances likelihood maximization with entropy regularization
        to prevent overfitting and encourage smooth time-dependent behavior.
        
        Args:
            data_t: Data tensor, shape (batch, n_vars)
            times_t: Time tensor, shape (batch,)
            entropy_weight: Weight for entropy regularization term
            
        Returns:
            Combined loss (negative log-likelihood + entropy penalty)
        """
        device = data_t.device
        dtype = data_t.dtype
        batch = data_t.shape[0]
        
        # Compute standard negative log-likelihood
        nll = self.negative_log_likelihood(data_t, times_t)
        
        # Normalize time to [0,1]
        t_min = times_t.min()
        t_max = times_t.max()
        t_norm = (times_t - t_min) / (t_max - t_min + 1e-8)
        
        # Compute entropy regularization
        entropy_penalty = torch.tensor(0.0, device=device, dtype=dtype)
        
        # For each edge flow, compute bandwidth variation entropy
        for edge_idx, (tr, rel_idx) in enumerate(self.edge_index_map):
            bw_predictions = self.edge_flows[edge_idx](t_norm.unsqueeze(-1))  # (batch, 2)
            
            # Compute entropy of bandwidth predictions
            # Higher entropy = more uniform/smooth bandwidth evolution
            # Lower entropy = more variable/spiky bandwidth evolution
            
            # Method 1: Variance-based entropy approximation
            bw_var = torch.var(bw_predictions, dim=0)  # Variance across time
            bw_entropy = torch.log(2 * np.pi * np.e * bw_var + 1e-8).sum()
            
            # Method 2: Differential entropy approximation using kernel density
            # This encourages smooth bandwidth evolution over time
            if batch > 1:
                # Compute pairwise distances in bandwidth space
                bw_expanded = bw_predictions.unsqueeze(1)  # (batch, 1, 2)
                bw_distances = torch.norm(bw_expanded - bw_predictions.unsqueeze(0), dim=2)  # (batch, batch)
                
                # Gaussian kernel with adaptive bandwidth
                h = torch.median(bw_distances) * 0.5  # Adaptive bandwidth
                kernel_vals = torch.exp(-bw_distances**2 / (2 * h**2))
                
                # Estimate density and entropy
                density_est = kernel_vals.mean(dim=1)
                log_density = torch.log(density_est + 1e-8)
                differential_entropy = -log_density.mean()
                
                # Combine entropy measures
                combined_entropy = 0.7 * bw_entropy + 0.3 * differential_entropy
            else:
                combined_entropy = bw_entropy
            
            entropy_penalty += combined_entropy
        
        # Temporal smoothness penalty
        # Encourage smooth bandwidth evolution over time
        smoothness_penalty = torch.tensor(0.0, device=device, dtype=dtype)
        
        if batch > 2:
            # Sort by time for smoothness computation
            time_sorted_indices = torch.argsort(t_norm)
            
            for edge_idx in range(len(self.edge_flows)):
                bw_sorted = self.edge_flows[edge_idx](t_norm[time_sorted_indices].unsqueeze(-1))
                
                # Compute second derivatives (curvature) as smoothness measure
                if bw_sorted.shape[0] > 2:
                    # Central differences for second derivative
                    bw_diff1 = bw_sorted[1:] - bw_sorted[:-1]  # First differences
                    bw_diff2 = bw_diff1[1:] - bw_diff1[:-1]   # Second differences
                    
                    # L2 penalty on curvature
                    curvature_penalty = torch.mean(bw_diff2**2)
                    smoothness_penalty += curvature_penalty
        
        # Information-theoretic regularization
        # Encourage bandwidth values to be informative but not extreme
        info_penalty = torch.tensor(0.0, device=device, dtype=dtype)
        
        for edge_idx in range(len(self.edge_flows)):
            bw_pred = self.edge_flows[edge_idx](t_norm.unsqueeze(-1))
            
            # Penalty for extreme bandwidth values
            min_bw, max_bw = 0.01, 2.0
            extreme_penalty = (
                torch.mean(torch.relu(min_bw - bw_pred)) +  # Too small
                torch.mean(torch.relu(bw_pred - max_bw))    # Too large
            )
            
            # Information content penalty (encourage moderate variance)
            bw_mean = torch.mean(bw_pred, dim=0)
            bw_std = torch.std(bw_pred, dim=0)
            
            # Penalty for too little variation (uninformative)
            low_var_penalty = torch.mean(torch.relu(0.01 - bw_std))
            
            # Penalty for too much variation (unstable)
            high_var_penalty = torch.mean(torch.relu(bw_std - 0.5))
            
            info_penalty += extreme_penalty + low_var_penalty + high_var_penalty
        
        # Combine all terms
        total_entropy_penalty = (
            entropy_penalty +           # Encourage high entropy (smoothness)
            0.1 * smoothness_penalty +  # Encourage temporal smoothness
            0.05 * info_penalty         # Encourage informative but stable bandwidths
        )
        
        # Final loss: NLL - entropy_weight * entropy (we want to maximize entropy)
        total_loss = nll - entropy_weight * total_entropy_penalty
        
        return total_loss

    def compute_bandwidth_evolution_metrics(self, 
                                          times_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for bandwidth evolution analysis.
        
        Args:
            times_t: Time tensor
            
        Returns:
            Dictionary of metrics
        """
        device = times_t.device
        
        # Normalize time
        t_min = times_t.min()
        t_max = times_t.max()
        t_norm = (times_t - t_min) / (t_max - t_min + 1e-8)
        
        metrics = {}
        
        for edge_idx, (tr, rel_idx) in enumerate(self.edge_index_map):
            edge_id = f"tree_{tr}_edge_{rel_idx}"
            
            # Get bandwidth predictions
            bw_pred = self.edge_flows[edge_idx](t_norm.unsqueeze(-1))  # (batch, 2)
            
            # Compute various metrics
            metrics[f"{edge_id}_mean_bw"] = torch.mean(bw_pred, dim=0)
            metrics[f"{edge_id}_std_bw"] = torch.std(bw_pred, dim=0)
            metrics[f"{edge_id}_min_bw"] = torch.min(bw_pred, dim=0)[0]
            metrics[f"{edge_id}_max_bw"] = torch.max(bw_pred, dim=0)[0]
            
            # Temporal variation
            if bw_pred.shape[0] > 1:
                bw_diff = bw_pred[1:] - bw_pred[:-1]
                metrics[f"{edge_id}_temporal_variation"] = torch.mean(torch.norm(bw_diff, dim=1))
        
        return metrics
