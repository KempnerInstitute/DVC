r"""Time-varying Gaussian graphical model baselines.

This implements a lightweight TVGL-style baseline:

  min_{Theta_t \succ 0} sum_t [ -log det(Theta_t) + tr(S_t Theta_t) ]
      + alpha * sum_t ||Theta_t||_{1,off}
      + beta  * sum_{t>0} ||Theta_t - Theta_{t-1}||_F^2

The solver here is a pragmatic proximal-gradient loop with SPD projection.
It is not a drop-in replacement for the full Hallac et al. (2017) TVGL ADMM,
but captures the key baseline behavior: sparse precision matrices with
temporal smoothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class TVGLResult:
    precision: List[np.ndarray]
    objective_history: List[float]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _soft_threshold_offdiag(A: np.ndarray, lam: float) -> np.ndarray:
    """Soft-threshold only off-diagonal entries."""
    out = A.copy()
    d = out.shape[0]
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            v = out[i, j]
            if v > lam:
                out[i, j] = v - lam
            elif v < -lam:
                out[i, j] = v + lam
            else:
                out[i, j] = 0.0
    return out


def _project_spd(A: np.ndarray, eps: float) -> np.ndarray:
    """Project a symmetric matrix to SPD by eigenvalue clipping."""
    w, V = np.linalg.eigh(_sym(A))
    w = np.clip(w, eps, None)
    return V @ np.diag(w) @ V.T


def _objective(
    Thetas: List[np.ndarray],
    covs: List[np.ndarray],
    alpha: float,
    beta: float,
) -> float:
    T = len(Thetas)
    obj = 0.0
    for t in range(T):
        Theta = Thetas[t]
        S = covs[t]
        sign, logdet = np.linalg.slogdet(Theta)
        if sign <= 0 or not np.isfinite(logdet):
            return float("inf")
        obj += -logdet + float(np.trace(S @ Theta))
        # L1 off-diagonal
        obj += alpha * float(np.sum(np.abs(Theta)) - np.sum(np.abs(np.diag(Theta))))
        if t > 0:
            diff = Theta - Thetas[t - 1]
            obj += beta * float(np.sum(diff * diff))
    return float(obj)


def tvgl_frobenius(
    covariances: List[np.ndarray],
    *,
    alpha: float = 0.02,
    beta: float = 1.0,
    max_iter: int = 200,
    step_size: float = 0.05,
    eps: float = 1e-4,
    backtrack: bool = True,
    backtrack_shrink: float = 0.5,
    backtrack_min_step: float = 1e-4,
    verbose: bool = False,
) -> TVGLResult:
    """Fit a temporally smoothed sparse precision sequence.

    Parameters
    ----------
    covariances:
        List of empirical covariances S_t (SPD-ish), one per time step.
    alpha:
        L1 penalty on off-diagonal entries (sparsity).
    beta:
        Frobenius smoothing penalty between consecutive precision matrices.
    max_iter:
        Maximum proximal-gradient iterations.
    step_size:
        Initial gradient step size.
    eps:
        Minimum eigenvalue for SPD projection.
    backtrack:
        Whether to backtrack step size if objective increases.
    """
    covs = [np.asarray(S, dtype=np.float64) for S in covariances]
    if not covs:
        raise ValueError("covariances must be a non-empty list")
    d = covs[0].shape[0]
    if any(S.shape != (d, d) for S in covs):
        raise ValueError("All covariances must have same square shape")

    # Initialize with per-time ridge inverse.
    Thetas = []
    for S in covs:
        S = _sym(S)
        S = S + 1e-3 * np.eye(d)
        Theta0 = np.linalg.inv(S)
        Thetas.append(_project_spd(Theta0, eps))

    obj_hist: List[float] = []
    eta = float(step_size)
    prev_obj = _objective(Thetas, covs, alpha, beta)
    obj_hist.append(prev_obj)

    for it in range(max_iter):
        invs = [np.linalg.inv(Theta) for Theta in Thetas]

        # Gradient step + proximal operator.
        cand: List[np.ndarray] = []
        for t in range(len(Thetas)):
            Theta = Thetas[t]
            S = covs[t]
            grad = S - invs[t]
            if beta > 0:
                if t == 0:
                    grad = grad + 2.0 * beta * (Theta - Thetas[t + 1])
                elif t == len(Thetas) - 1:
                    grad = grad + 2.0 * beta * (Theta - Thetas[t - 1])
                else:
                    grad = grad + 2.0 * beta * (2.0 * Theta - Thetas[t - 1] - Thetas[t + 1])

            Theta_step = _sym(Theta - eta * grad)
            Theta_step = _soft_threshold_offdiag(Theta_step, eta * alpha)
            Theta_step = _project_spd(Theta_step, eps)
            cand.append(Theta_step)

        cand_obj = _objective(cand, covs, alpha, beta)
        if backtrack and cand_obj > prev_obj and eta > backtrack_min_step:
            eta = max(backtrack_min_step, eta * backtrack_shrink)
            if verbose:
                print(f"[tvgl] backtrack step -> {eta:.4g} (obj {prev_obj:.4g} -> {cand_obj:.4g})")
            continue

        Thetas = cand
        prev_obj = cand_obj
        obj_hist.append(prev_obj)

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            print(f"[tvgl] iter={it} obj={prev_obj:.6g} step={eta:.4g}")

        # Simple convergence check: relative improvement.
        if len(obj_hist) >= 5:
            rel = abs(obj_hist[-2] - obj_hist[-1]) / (abs(obj_hist[-2]) + 1e-12)
            if rel < 1e-5:
                break

    return TVGLResult(precision=Thetas, objective_history=obj_hist)


def precision_to_partial_corr(P: np.ndarray) -> np.ndarray:
    """Convert a precision matrix to partial correlations."""
    P = np.asarray(P, dtype=np.float64)
    denom = np.sqrt(np.outer(np.diag(P), np.diag(P)))
    denom = np.clip(denom, 1e-12, None)
    pc = -P / denom
    np.fill_diagonal(pc, 0.0)
    return pc


def hub_from_precision(P: np.ndarray, edge_threshold: float = 0.05) -> int:
    """Heuristic hub estimate: node with max degree in thresholded partial-corr graph."""
    pc = precision_to_partial_corr(P)
    adj = (np.abs(pc) >= float(edge_threshold)).astype(np.int32)
    deg = adj.sum(axis=1)
    return int(np.argmax(deg))
