"""
Time-Dependent Vine Copula Models

Implements neural network models for time-dependent vine copula parameters,
including bandwidth flows and dynamic entropy estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence
import logging

from ..core.objects import vine_obj_bin
from ..core.utils_locallik import loclik_batch_eval
from .flows import TimeBandwidthFlow
from .trajectory_models import create_trajectory_model

logger = logging.getLogger(__name__)


class TimeDependentVine(nn.Module):
    """
    Time-dependent vine copula that combines a base vine structure
    with time-varying bandwidth parameters.

    Current implementation details:
    - The local-likelihood path uses level-0 vine edges.
    - Time conditioning is used for NLL evaluation through per-edge bandwidths.
    - Sampling remains delegated to the fitted base vine.
    
    Parameters
    ----------
    base_vine : vine_obj_bin
        Base vine copula structure
    bandwidth_flow : TimeBandwidthFlow
        Neural network for time-dependent bandwidths
    device : str or torch.device
        Device for computation
    """
    
    def __init__(self, 
                 base_vine: vine_obj_bin,
                 bandwidth_flow: Optional[TimeBandwidthFlow] = None,
                 device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        
        self.base_vine = base_vine
        self.device = torch.device(device)
        
        self.edge_index = self._get_level0_edges()
        n_edges = max(len(self.edge_index), 1)
        
        # Create bandwidth flow if not provided
        if bandwidth_flow is None:
            self.bandwidth_flow = TimeBandwidthFlow(n_edges)
        else:
            self.bandwidth_flow = bandwidth_flow
        
        self._time_grid: Optional[torch.Tensor] = None
        self._fit_pairs_by_time: List[torch.Tensor] = []
        self._reference_ready = False
        self._normal = torch.distributions.Normal(0.0, 1.0)
        self.to(self.device)

    def _get_level0_edges(self) -> List[Tuple[int, int]]:
        """Use level-0 vine edges (or default root-star edges) for KDE likelihood."""
        if hasattr(self.base_vine, "ind_vine") and self.base_vine.ind_vine and self.base_vine.ind_vine[0]:
            return [(int(i), int(j)) for (i, j) in self.base_vine.ind_vine[0]]
        d = int(getattr(self.base_vine, "n_cop", 2))
        if d < 2:
            return [(0, 1)]
        return [(0, j) for j in range(1, d)]
    
    def _count_vine_edges(self) -> int:
        """Count total number of edges in vine structure."""
        return max(len(self.edge_index), 1)

    def _rank_normal_scores(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Rank-normalize a matrix to approximately N(0,1) marginals."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {x.shape}")
        n, d = x.shape
        u = np.zeros((n, d), dtype=np.float64)
        for j in range(d):
            ranks = np.argsort(np.argsort(x[:, j], kind="mergesort"), kind="mergesort") + 1
            u[:, j] = ranks / (n + 1.0)
        u = np.clip(u, eps, 1.0 - eps)
        z = np.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * u - 1.0, dtype=torch.float32))
        return z.numpy()

    def _pairs_from_matrix(self, x: np.ndarray) -> torch.Tensor:
        """Stack level-0 edge pairs into tensor [N,2,E]."""
        x = np.asarray(x, dtype=np.float32)
        pairs = [x[:, [i, j]] for (i, j) in self.edge_index]
        if not pairs:
            pairs = [x[:, [0, 1]]]
        arr = np.stack(pairs, axis=2)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def set_reference_data(
        self,
        data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
    ) -> None:
        """Store per-time reference data used by local-likelihood copula evaluation."""
        if isinstance(data_by_time, np.ndarray) and data_by_time.ndim == 3:
            x_list = [np.asarray(data_by_time[t], dtype=np.float32) for t in range(data_by_time.shape[0])]
        else:
            x_list = [np.asarray(x, dtype=np.float32) for x in data_by_time]
        if not x_list:
            raise ValueError("data_by_time must be non-empty")

        if time_points is None:
            t = np.linspace(0.0, 1.0, len(x_list), dtype=np.float32)
        else:
            t = np.asarray(time_points, dtype=np.float32).reshape(-1)
            if t.size != len(x_list):
                raise ValueError("time_points length must match number of time slices")

        self.bandwidth_flow.set_time_range(float(t.min()), float(t.max()))
        if hasattr(self.bandwidth_flow, "set_reference_time_grid"):
            self.bandwidth_flow.set_reference_time_grid(torch.tensor(t, dtype=torch.float32, device=self.device))
        self._time_grid = torch.tensor(t, dtype=torch.float32, device=self.device)
        self._fit_pairs_by_time = [self._pairs_from_matrix(self._rank_normal_scores(x_t)) for x_t in x_list]
        self._reference_ready = True

    def _nearest_time_index(self, t: torch.Tensor) -> int:
        if self._time_grid is None or self._time_grid.numel() == 0:
            raise ValueError("Reference time grid is not initialized")
        idx = int(torch.argmin(torch.abs(self._time_grid - t)).item())
        return idx

    def _edge_nll(
        self,
        fit_pairs: torch.Tensor,
        eval_pairs: torch.Tensor,
        bw_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample truncated vine NLL from level-0 edge densities."""
        if bw_edge.dim() != 1:
            raise ValueError(f"Expected bw_edge shape [E], got {tuple(bw_edge.shape)}")
        bw_edge = bw_edge.clamp(min=1e-4, max=5.0)
        B = torch.stack([bw_edge, bw_edge], dim=0)  # [2,E], isotropic per edge
        ker = loclik_batch_eval(B, fit_pairs, eval_pairs, n_cop=fit_pairs.shape[2], batch_size=1).clamp_min(1e-30)
        log_joint = torch.log(ker)
        log_norm = self._normal.log_prob(eval_pairs[:, 0, :]) + self._normal.log_prob(eval_pairs[:, 1, :])
        log_c = log_joint - log_norm
        return -log_c.sum(dim=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for time-dependent vine.
        
        Parameters
        ----------
        x : torch.Tensor
            Data samples, shape (batch_size, n_features)
        time : torch.Tensor
            Time points, shape (batch_size,)
            
        Returns
        -------
        torch.Tensor
            Negative log-likelihood values, shape (batch_size,)
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(time, np.ndarray):
            time = torch.tensor(time, dtype=torch.float32)

        x = x.to(self.device, dtype=torch.float32)
        time = time.to(self.device, dtype=torch.float32).reshape(-1)
        if x.dim() != 2:
            raise ValueError(f"Expected x shape [N,d], got {tuple(x.shape)}")
        if time.shape[0] != x.shape[0]:
            raise ValueError("time and x must have same batch size")

        if not self._reference_ready:
            # Fallback path when reference data has not been initialized.
            try:
                if hasattr(self.base_vine, "logpdf"):
                    log_probs = self.base_vine.logpdf(x).to(self.device)
                    return -log_probs
                if hasattr(self.base_vine, "evaluation"):
                    p, _, _ = self.base_vine.evaluation(x.detach().cpu().numpy())
                    if isinstance(p, torch.Tensor):
                        return -torch.log(p.clamp_min(1e-30)).to(self.device)
                    return -torch.log(torch.tensor(p, dtype=torch.float32, device=self.device).clamp_min(1e-30))
            except Exception as exc:
                logger.warning("TimeDependentVine fallback evaluation failed: %s", exc)
            return torch.full((x.shape[0],), 10.0, device=self.device)

        nll_values = torch.zeros(x.shape[0], dtype=torch.float32, device=self.device)
        uniq_t = torch.unique(time)
        for t_val in uniq_t:
            sel = (time == t_val)
            if not torch.any(sel):
                continue
            idx_ref = self._nearest_time_index(t_val)
            fit_pairs = self._fit_pairs_by_time[idx_ref]
            x_sel = x[sel].detach().cpu().numpy()
            z_sel = self._rank_normal_scores(x_sel)
            eval_pairs = self._pairs_from_matrix(z_sel)
            bw = self.bandwidth_flow(t_val.view(1, 1)).squeeze(0)
            nll_values[sel] = self._edge_nll(fit_pairs, eval_pairs, bw)

        return nll_values

    def fit_bandwidth_flow(
        self,
        train_data_by_time: Union[np.ndarray, Sequence[np.ndarray]],
        time_points: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
        *,
        val_fraction: float = 0.2,
        n_epochs: int = 150,
        lr: float = 1e-2,
        batch_time_steps: int = 8,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Train time->bandwidth flow by minimizing validation copula NLL."""
        if isinstance(train_data_by_time, np.ndarray) and train_data_by_time.ndim == 3:
            x_list = [np.asarray(train_data_by_time[t], dtype=np.float32) for t in range(train_data_by_time.shape[0])]
        else:
            x_list = [np.asarray(x, dtype=np.float32) for x in train_data_by_time]
        if len(x_list) < 2:
            raise ValueError("Need at least two time slices to train bandwidth flow")

        if time_points is None:
            t = np.linspace(0.0, 1.0, len(x_list), dtype=np.float32)
        else:
            t = np.asarray(time_points, dtype=np.float32).reshape(-1)
            if t.size != len(x_list):
                raise ValueError("time_points length must match number of time slices")

        self.bandwidth_flow.set_time_range(float(t.min()), float(t.max()))
        if hasattr(self.bandwidth_flow, "set_reference_time_grid"):
            self.bandwidth_flow.set_reference_time_grid(torch.tensor(t, dtype=torch.float32, device=self.device))
        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).unsqueeze(-1)

        val_fraction = float(np.clip(val_fraction, 0.05, 0.5))
        rng = np.random.default_rng(int(seed))
        fit_pairs: List[torch.Tensor] = []
        val_pairs: List[torch.Tensor] = []
        full_pairs: List[torch.Tensor] = []
        for x_t in x_list:
            z_t = self._rank_normal_scores(x_t)
            pairs = self._pairs_from_matrix(z_t)
            n = pairs.shape[0]
            if n < 20:
                raise ValueError("Need at least ~20 samples per time slice")
            perm = rng.permutation(n)
            n_fit = max(int(round((1.0 - val_fraction) * n)), 10)
            n_fit = min(n_fit, n - 5)
            idx_fit = perm[:n_fit]
            idx_val = perm[n_fit:]
            fit_pairs.append(pairs[idx_fit])
            val_pairs.append(pairs[idx_val])
            full_pairs.append(pairs)

        opt = torch.optim.Adam(self.bandwidth_flow.parameters(), lr=float(lr))
        history: List[float] = []
        T = len(x_list)
        bsz = min(int(batch_time_steps), T)

        for ep in range(int(n_epochs)):
            opt.zero_grad(set_to_none=True)
            rng_ep = np.random.default_rng(int(seed + 97 * ep))
            t_sel = rng_ep.choice(T, size=bsz, replace=False)
            loss = torch.zeros((), dtype=torch.float32, device=self.device)
            for ti in t_sel:
                bw = self.bandwidth_flow(t_tensor[ti:ti + 1]).squeeze(0)
                loss = loss + self._edge_nll(fit_pairs[ti], val_pairs[ti], bw).mean()
            loss = loss / float(bsz)

            bw_all = self.bandwidth_flow(t_tensor)
            bw_reg = 1e-3 * torch.mean((bw_all[1:] - bw_all[:-1]) ** 2) if bw_all.shape[0] > 1 else 0.0
            if hasattr(self.bandwidth_flow, "regularization_loss"):
                bw_reg = bw_reg + self.bandwidth_flow.regularization_loss()
            loss = loss + bw_reg
            loss.backward()
            opt.step()
            history.append(float(loss.detach().cpu()))

        self._time_grid = torch.tensor(t, dtype=torch.float32, device=self.device)
        self._fit_pairs_by_time = full_pairs
        self._reference_ready = True
        return {
            "history": history,
            "time_points": t.tolist(),
            "n_edges": int(self._count_vine_edges()),
        }
    
    def sample(self, n_samples: int, time_points: torch.Tensor) -> torch.Tensor:
        """
        Sample from the model at specified time points.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        time_points : torch.Tensor
            Time points for sampling, shape (n_samples,).
            Currently used for API consistency and validation. The sampling
            path remains the fitted base vine and does not yet condition on
            time-varying bandwidth predictions.
            
        Returns
        -------
        torch.Tensor
            Generated samples, shape (n_samples, n_features)
        """
        if time_points is None:
            raise ValueError("time_points must be provided with shape (n_samples,)")
        t = torch.as_tensor(time_points, dtype=torch.float32, device=self.device).reshape(-1)
        if int(t.numel()) != int(n_samples):
            raise ValueError(f"time_points length ({int(t.numel())}) must equal n_samples ({int(n_samples)})")

        with torch.no_grad():
            # Sampling path remains parametric (base vine). Flow is used in NLL evaluation.
            try:
                if hasattr(self.base_vine, 'sample'):
                    samples = self.base_vine.sample(n_samples)
                    samples = torch.from_numpy(samples).float().to(self.device)
                else:
                    # Fallback: Gaussian samples
                    n_features = getattr(self.base_vine, 'n_cop', 2)
                    samples = torch.randn(n_samples, n_features, device=self.device)
                
                return samples
                
            except Exception as e:
                logger.warning(f"Error in time-dependent vine sampling: {e}")
                n_features = getattr(self.base_vine, 'n_cop', 2)
                return torch.randn(n_samples, n_features, device=self.device)
    
    def get_bandwidths_over_time(self, time_range: torch.Tensor) -> torch.Tensor:
        """Get bandwidth evolution over a time range."""
        with torch.no_grad():
            return self.bandwidth_flow(time_range)


class DynamicEntropyEstimator(nn.Module):
    """
    Estimator for dynamic entropy in time-dependent vine copulas.
    
    Uses Monte Carlo sampling with time-dependent vine models to estimate
    entropy as a function of time.
    
    Parameters
    ----------
    time_vine : TimeDependentVine
        Time-dependent vine copula model
    n_samples : int
        Number of Monte Carlo samples for entropy estimation
    n_time_points : int
        Number of time points for entropy trajectory
    """
    
    def __init__(self, 
                 time_vine: TimeDependentVine,
                 n_samples: int = 1000,
                 n_time_points: int = 100):
        super().__init__()
        
        self.time_vine = time_vine
        self.n_samples = n_samples
        self.n_time_points = n_time_points
        
    def estimate_entropy_trajectory(self, 
                                  time_start: float = 0.0,
                                  time_end: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate entropy as a function of time.
        
        Parameters
        ----------
        time_start : float
            Start time for trajectory
        time_end : float
            End time for trajectory
            
        Returns
        -------
        tuple of torch.Tensor
            (time_points, entropy_values)
        """
        device = next(self.time_vine.parameters()).device
        
        # Create time grid
        time_points = torch.linspace(time_start, time_end, self.n_time_points, device=device)
        entropy_values = torch.zeros(self.n_time_points, device=device)
        
        with torch.no_grad():
            for i, t in enumerate(time_points):
                # Sample from vine at time t
                t_batch = t.repeat(self.n_samples)
                samples = self.time_vine.sample(self.n_samples, t_batch)
                
                # Compute log probabilities
                log_probs = -self.time_vine(samples, t_batch)
                
                # Estimate entropy: H = -E[log p(x)]
                entropy_values[i] = -torch.mean(log_probs)
        
        return time_points.cpu(), entropy_values.cpu()
    
    def estimate_mutual_information_trajectory(self,
                                             var_indices_x: List[int],
                                             var_indices_y: List[int],
                                             time_start: float = 0.0,
                                             time_end: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate mutual information between variable subsets over time.
        
        Parameters
        ----------
        var_indices_x : list of int
            Indices of variables in set X
        var_indices_y : list of int
            Indices of variables in set Y
        time_start : float
            Start time
        time_end : float
            End time
            
        Returns
        -------
        tuple of torch.Tensor
            (time_points, mi_values)
        """
        device = next(self.time_vine.parameters()).device
        
        time_points = torch.linspace(time_start, time_end, self.n_time_points, device=device)
        mi_values = torch.zeros(self.n_time_points, device=device)
        
        with torch.no_grad():
            for i, t in enumerate(time_points):
                # Sample from vine at time t
                t_batch = t.repeat(self.n_samples)
                samples = self.time_vine.sample(self.n_samples, t_batch)
                
                # Extract variable subsets
                x_samples = samples[:, var_indices_x]
                y_samples = samples[:, var_indices_y]
                xy_samples = samples[:, var_indices_x + var_indices_y]
                
                # Estimate MI using density ratios (simplified approach)
                # MI(X;Y) = E[log(p(x,y) / (p(x)p(y)))]
                
                # This is a placeholder - in practice you'd need proper density estimation
                # for marginals and joint distributions
                mi_values[i] = self._estimate_mi_from_samples(x_samples, y_samples, xy_samples)
        
        return time_points.cpu(), mi_values.cpu()
    
    def _estimate_mi_from_samples(self, x_samples: torch.Tensor, 
                                y_samples: torch.Tensor, 
                                xy_samples: torch.Tensor) -> torch.Tensor:
        """Estimate MI from samples using a Gaussian plug-in estimator.

        For multivariate continuous variables, this computes:
            I(X;Y) = 0.5 * (log |Sigma_x| + log |Sigma_y| - log |Sigma_xy|)
        with covariance regularization for numerical stability.
        """
        try:
            x_np = np.asarray(x_samples.detach().cpu().numpy(), dtype=np.float64)
            y_np = np.asarray(y_samples.detach().cpu().numpy(), dtype=np.float64)

            if x_np.ndim == 1:
                x_np = x_np[:, None]
            if y_np.ndim == 1:
                y_np = y_np[:, None]
            if x_np.ndim != 2 or y_np.ndim != 2:
                return torch.tensor(0.0)
            if x_np.shape[0] != y_np.shape[0]:
                return torch.tensor(0.0)
            n = x_np.shape[0]
            dx = x_np.shape[1]
            dy = y_np.shape[1]
            if n < max(dx + dy + 2, 8):
                return torch.tensor(0.0)

            # Standardize to reduce scale pathologies in covariance estimation.
            z = np.concatenate([x_np, y_np], axis=1)
            z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)

            cov = np.cov(z, rowvar=False)
            if cov.ndim != 2:
                return torch.tensor(0.0)

            cov_x = cov[:dx, :dx]
            cov_y = cov[dx:, dx:]
            cov_xy = cov

            ridge = 1e-6
            cov_x = cov_x + ridge * np.eye(dx)
            cov_y = cov_y + ridge * np.eye(dy)
            cov_xy = cov_xy + ridge * np.eye(dx + dy)

            sx, ldx = np.linalg.slogdet(cov_x)
            sy, ldy = np.linalg.slogdet(cov_y)
            sxy, ldxy = np.linalg.slogdet(cov_xy)
            if sx <= 0 or sy <= 0 or sxy <= 0:
                return torch.tensor(0.0)

            mi = 0.5 * (ldx + ldy - ldxy)
            if not np.isfinite(mi):
                return torch.tensor(0.0)
            return torch.tensor(float(max(mi, 0.0)))
        except Exception:
            return torch.tensor(0.0)
    
    def compute_entropy_rate(self, time_points: torch.Tensor, 
                           entropy_values: torch.Tensor) -> torch.Tensor:
        """Compute entropy rate (derivative of entropy with respect to time)."""
        # Numerical differentiation
        dt = time_points[1] - time_points[0]
        entropy_rate = torch.gradient(entropy_values, spacing=dt.item())[0]
        return entropy_rate


def create_time_dependent_vine(base_vine: vine_obj_bin,
                             hidden_dims: Optional[List[int]] = None,
                             time_embedding_dim: int = 16,
                             trajectory_type: str = "mlp",
                             trajectory_kwargs: Optional[Dict[str, Any]] = None,
                             device: Union[str, torch.device] = 'cpu') -> TimeDependentVine:
    """
    Factory function to create a time-dependent vine copula.
    
    Parameters
    ----------
    base_vine : vine_obj_bin
        Base vine copula structure
    hidden_dims : list of int
        Hidden dimensions for bandwidth flow network
    time_embedding_dim : int
        Time embedding dimension
    trajectory_type : str
        Temporal parameterization for the bandwidth path. One of
        ``"mlp"`` (default), ``"basis"``, or ``"state_space"``.
    trajectory_kwargs : dict
        Additional keyword arguments forwarded to the chosen trajectory model.
    device : str or torch.device
        Computation device
        
    Returns
    -------
    TimeDependentVine
        Time-dependent vine copula model using level-0 edge bandwidths
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]

    # Use level-0 edges, matching TimeDependentVine local-likelihood path.
    if hasattr(base_vine, 'ind_vine') and base_vine.ind_vine and base_vine.ind_vine[0]:
        n_edges = len(base_vine.ind_vine[0])
    else:
        n_edges = max(int(getattr(base_vine, "n_cop", 2)) - 1, 1)
    
    trajectory_kwargs = dict(trajectory_kwargs or {})
    if trajectory_type == "mlp":
        bandwidth_flow = TimeBandwidthFlow(
            n_edges=n_edges,
            hidden_dims=hidden_dims,
            time_embedding_dim=time_embedding_dim,
            **trajectory_kwargs,
        )
    else:
        default_kwargs: Dict[str, Any] = {
            "constraint": "bounded",
            "min_value": 0.01,
            "max_value": 2.0,
        }
        if trajectory_type == "basis":
            default_kwargs.setdefault("n_basis", 4)
        if trajectory_type == "state_space":
            default_kwargs.setdefault("latent_dim", 3)
            default_kwargs.setdefault("n_steps", 8)
            default_kwargs.setdefault("transition_penalty", 1e-2)
        default_kwargs.update(trajectory_kwargs)
        bandwidth_flow = create_trajectory_model(
            trajectory_type,
            output_dim=n_edges,
            **default_kwargs,
        )
    
    # Create time-dependent vine
    time_vine = TimeDependentVine(
        base_vine=base_vine,
        bandwidth_flow=bandwidth_flow,
        device=device
    )
    
    return time_vine
