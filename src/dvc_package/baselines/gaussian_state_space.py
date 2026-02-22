"""State-space Gaussian copula baseline for time-varying dependence.

This module fits a simple dynamical model for Gaussian-copula parameters:
- latent Fisher-z edge correlations follow a random walk
- observed per-time empirical Fisher-z correlations are noisy measurements
- a Kalman filter estimates the latent trajectory for each edge

The fitted latent edge trajectories are converted back to correlation matrices,
projected to valid SPD correlation matrices, and evaluated by held-out copula NLL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class GaussianCopulaStateSpaceResult:
    """Outputs of the Gaussian copula state-space fit."""

    correlations: List[np.ndarray]
    process_variance: float
    observation_variance: List[float]
    latent_fisher_z: np.ndarray
    observed_fisher_z: np.ndarray
    edge_index: List[Tuple[int, int]]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _rank_normal_scores(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normal scores via rank pseudo-observations."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    n, d = x.shape
    u = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        ranks = np.argsort(np.argsort(x[:, j], kind="mergesort"), kind="mergesort") + 1
        u[:, j] = ranks / (n + 1.0)
    u = np.clip(u, eps, 1.0 - eps)
    return norm.ppf(u)


def _ensure_spd_correlation(R: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Project to a valid SPD correlation matrix."""
    R = _sym(np.asarray(R, dtype=np.float64))
    d = R.shape[0]
    R = R + float(ridge) * np.eye(d)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = V @ np.diag(w) @ V.T
    std = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    R = R / np.outer(std, std)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = _sym(R)
    np.fill_diagonal(R, 1.0)
    return R


def _gaussian_copula_nll_given_corr(z: np.ndarray, R: np.ndarray) -> float:
    """Mean negative log copula density for Gaussian copula with fixed R."""
    z = np.asarray(z, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Expected z shape [N,d], got {z.shape}")
    n, d = z.shape
    if n < 5 or R.shape != (d, d):
        return float("nan")

    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0 or not np.isfinite(logdet):
        return float("nan")

    invR = np.linalg.inv(R)
    A = invR - np.eye(d)
    quad = np.einsum("ni,ij,nj->n", z, A, z)
    log_c = -0.5 * logdet - 0.5 * quad
    return float(-np.mean(log_c))


def _kalman_filter_random_walk(
    y: np.ndarray,
    obs_var: np.ndarray,
    q: float,
    init_var: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """1D Kalman filter for random-walk latent state.

    State model: x_t = x_{t-1} + w_t, w_t ~ N(0, q)
    Obs model:   y_t = x_t + v_t,     v_t ~ N(0, r_t)
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    r = np.asarray(obs_var, dtype=np.float64).reshape(-1)
    if y.shape[0] != r.shape[0]:
        raise ValueError("y and obs_var must have same length")
    if np.any(r <= 0.0):
        raise ValueError("obs_var must be positive")
    q = float(max(q, 0.0))

    T = y.shape[0]
    m = np.zeros(T, dtype=np.float64)
    P = np.zeros(T, dtype=np.float64)

    m_prev = float(y[0])
    P_prev = float(max(init_var, 1e-6))

    nll = 0.0
    for t in range(T):
        m_pred = m_prev
        P_pred = P_prev + q

        S = P_pred + r[t]
        S = float(max(S, 1e-10))
        innov = float(y[t] - m_pred)

        nll += 0.5 * (math.log(2.0 * math.pi * S) + (innov * innov) / S)

        K = P_pred / S
        m_t = m_pred + K * innov
        P_t = (1.0 - K) * P_pred

        m[t] = m_t
        P[t] = max(P_t, 1e-10)
        m_prev = m_t
        P_prev = P_t

    return m, P, float(nll)


def fit_gaussian_copula_state_space(
    x_train_by_time: Sequence[np.ndarray],
    *,
    q_grid: Optional[Sequence[float]] = None,
) -> GaussianCopulaStateSpaceResult:
    """Fit a state-space model over Gaussian-copula edge correlations."""
    x_list = [np.asarray(x, dtype=np.float64) for x in x_train_by_time]
    if not x_list:
        raise ValueError("x_train_by_time must be non-empty")
    d = int(x_list[0].shape[1])
    if any(x.ndim != 2 or x.shape[1] != d for x in x_list):
        raise ValueError("All time slices must have shape [N,d] with fixed d")

    T = len(x_list)
    edges = [(i, j) for i in range(d) for j in range(i + 1, d)]
    E = len(edges)

    y = np.zeros((T, E), dtype=np.float64)
    obs_var = np.zeros(T, dtype=np.float64)

    for t, x_t in enumerate(x_list):
        z_t = _rank_normal_scores(x_t)
        R_t = np.corrcoef(z_t, rowvar=False)
        R_t = np.nan_to_num(R_t, nan=0.0, posinf=0.0, neginf=0.0)
        R_t = _sym(R_t)
        np.fill_diagonal(R_t, 1.0)

        for k, (i, j) in enumerate(edges):
            r_ij = float(np.clip(R_t[i, j], -0.995, 0.995))
            y[t, k] = np.arctanh(r_ij)

        n_t = int(x_t.shape[0])
        obs_var[t] = 1.0 / float(max(n_t - 3, 1))

    if q_grid is None:
        q_grid = np.concatenate([[0.0], np.logspace(-6, -1, 10)])
    q_candidates = [float(max(q, 0.0)) for q in q_grid]

    best_q = q_candidates[0]
    best_obj = float("inf")
    for q in q_candidates:
        obj = 0.0
        for k in range(E):
            _m, _P, nll = _kalman_filter_random_walk(y[:, k], obs_var, q)
            obj += nll
        if obj < best_obj:
            best_obj = obj
            best_q = q

    latent = np.zeros_like(y)
    for k in range(E):
        m, _P, _nll = _kalman_filter_random_walk(y[:, k], obs_var, best_q)
        latent[:, k] = m

    corr_seq: List[np.ndarray] = []
    for t in range(T):
        R_t = np.eye(d, dtype=np.float64)
        for k, (i, j) in enumerate(edges):
            r = float(np.tanh(latent[t, k]))
            R_t[i, j] = r
            R_t[j, i] = r
        corr_seq.append(_ensure_spd_correlation(R_t))

    return GaussianCopulaStateSpaceResult(
        correlations=corr_seq,
        process_variance=float(best_q),
        observation_variance=obs_var.tolist(),
        latent_fisher_z=latent,
        observed_fisher_z=y,
        edge_index=edges,
    )


def gaussian_copula_state_space_nll_fit_eval(
    x_train_by_time: Sequence[np.ndarray],
    x_test_by_time: Sequence[np.ndarray],
    *,
    q_grid: Optional[Sequence[float]] = None,
) -> Tuple[List[float], GaussianCopulaStateSpaceResult]:
    """Fit state-space Gaussian copula and return held-out NLL per time."""
    x_train_list = [np.asarray(x, dtype=np.float64) for x in x_train_by_time]
    x_test_list = [np.asarray(x, dtype=np.float64) for x in x_test_by_time]
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_by_time and x_test_by_time must have same length")

    fit = fit_gaussian_copula_state_space(x_train_list, q_grid=q_grid)

    nll: List[float] = []
    for R_t, xte in zip(fit.correlations, x_test_list):
        zte = _rank_normal_scores(xte)
        nll.append(_gaussian_copula_nll_given_corr(zte, R_t))

    return nll, fit


__all__ = [
    "GaussianCopulaStateSpaceResult",
    "fit_gaussian_copula_state_space",
    "gaussian_copula_state_space_nll_fit_eval",
]
