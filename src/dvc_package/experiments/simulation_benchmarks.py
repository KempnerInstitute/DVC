"""
Simulation benchmark suite for Dynamic Vine Copulas (DVC).

This module provides:
- synthetic generators designed to isolate higher-order and non-Gaussian dependence
- minimal baselines (Gaussian copula) and metrics (NLL, tail dependence, structure recovery)
- publication-style multi-panel figures saved under an experiment output directory

Design constraints:
- outputs must be file-based (Overleaf cannot run Python)
- figures are generated under `results/<experiment>/plots/*.png` and vendored into `drafts/`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t as student_t

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baselines.gaussian_state_space import gaussian_copula_state_space_nll_fit_eval
from ..baselines.tvgl import hub_from_precision, tvgl_frobenius
from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.vine_factory import create_vine
from ..core.param_copula import copulaccdf, copulainvccdf, copulapdf, parametric_fit
from ..core.utils_locallik import loclik_batch_eval
from ..core.vine_model import fit_vine
from ..optimization.structure import optimize_vine_structure
from ..time.joint_dynamic_cvine import JointDynamicCVine, JointDynamicCVineResult
from ..time.regularized_cvine import RegularizedDynamicCVine, RegularizedDynamicCVineResult
from ..time.latent_state_dynamic_cvine import LatentStateDynamicCVine, LatentStateDynamicCVineResult
from ..time.nonparametric_dynamic_cvine import (
    JointDynamicNonparametricCVine,
    JointDynamicNonparametricCVineResult,
    WindowedNonparametricCVine,
    WindowedNonparametricCVineResult,
)


def _set_seaborn_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _normalize_family_name(family: str) -> str:
    fam = str(family).lower().strip()
    if fam in {"independence", "independent"}:
        return "ind"
    if fam in {"gauss"}:
        return "gaussian"
    if fam in {"student-t", "t"}:
        return "student"
    return fam


def _pseudo_obs_from_gaussianized(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert approximately N(0,1) marginals to pseudo-observations via N(0,1) CDF."""
    u = norm.cdf(np.asarray(x, dtype=np.float64))
    return np.clip(u, eps, 1.0 - eps).astype(np.float32)


def _pseudo_obs_rank(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Rank-based pseudo-observations U in (0,1), per column (copula-preserving)."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    n, d = x.shape
    u = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        col = x[:, j]
        ranks = np.argsort(np.argsort(col, kind="mergesort"), kind="mergesort").astype(np.float64) + 1.0
        u[:, j] = ranks / (n + 1.0)
    return np.clip(u, eps, 1.0 - eps).astype(np.float32)

def _normal_scores_from_rank_pobs(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert data to normal scores via rank pseudo-observations.

    This is a standard Gaussian-copula preprocessing step when marginals are unknown:
      u_j = rank(x_j) / (n+1),  z_j = Phi^{-1}(u_j).
    """
    u = _pseudo_obs_rank(x, eps=eps).astype(np.float64)
    return norm.ppf(u).astype(np.float64)


def _fit_best_bivariate_copula(
    u_train: np.ndarray,
    families: List[str],
) -> cop_par_obj:
    """Fit a bivariate copula by AIC over candidate families.

    Parameters
    ----------
    u_train:
        Pseudo-observations in (0,1), shape (N,2).
    families:
        Candidate family names.
    """
    u_train = np.asarray(u_train, dtype=np.float32)
    if u_train.ndim != 2 or u_train.shape[1] != 2:
        raise ValueError(f"Expected u_train shape (N,2), got {u_train.shape}")

    # parametric_fit expects shape [N,2,n_edges].
    u3 = u_train[:, :, None]
    aic2, theta_list, _logp_list = parametric_fit(u3, families=families, n_cop=1)
    aic = np.asarray(aic2[0], dtype=np.float64)
    best_idx = int(np.nanargmin(aic))
    fam = _normalize_family_name(families[best_idx])
    theta = theta_list[0][best_idx]
    return cop_par_obj(fam, theta)


def _mean_bivariate_copula_nll(cop: cop_par_obj, u_test: np.ndarray) -> float:
    """Mean bivariate copula NLL (nats) on pseudo-observations u_test[:,2]."""
    u_test = np.asarray(u_test, dtype=np.float32)
    uv = torch.tensor(u_test, dtype=torch.float32)
    pdf = copulapdf(cop, uv).clamp_min(1e-30)
    return float((-torch.log(pdf)).mean().detach().cpu())


def gaussianize_columns(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Monotone per-column transform to standard normal marginals (copula-preserving)."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    n, d = x.shape
    out = np.zeros_like(x, dtype=np.float64)
    for j in range(d):
        col = x[:, j]
        # Rank-based pseudo-observations (continuous-data assumption).
        ranks = np.argsort(np.argsort(col, kind="mergesort"), kind="mergesort").astype(np.float64) + 1.0
        u = ranks / (n + 1.0)
        u = np.clip(u, eps, 1.0 - eps)
        out[:, j] = norm.ppf(u)
    # Standardize numerically (optional but stabilizes plotting).
    out = (out - out.mean(axis=0, keepdims=True)) / (out.std(axis=0, keepdims=True) + 1e-12)
    return out.astype(np.float32)


def _build_cvine_edges(order: List[int]) -> List[List[List[int]]]:
    """Return `ind_vine` for a C-vine given a variable/root order."""
    d = len(order)
    ind_vine: List[List[List[int]]] = []
    for level in range(d - 1):
        root = int(order[level])
        edges_level: List[List[int]] = []
        for j in range(level + 1, d):
            edges_level.append([root, int(order[j])])
        ind_vine.append(edges_level)
    return ind_vine


def _make_levelwise_cvine(
    d: int,
    order: List[int],
    level_families: List[str],
    level_thetas: List[Any],
) -> vine_obj_bin:
    """Create a parametric C-vine with explicit per-level copula families/parameters."""
    if len(level_families) != d - 1 or len(level_thetas) != d - 1:
        raise ValueError("level_families/thetas must have length d-1 for a C-vine.")

    vine = create_vine("c-vine", d, families=list(sorted(set(level_families))))
    vine.ind_vine = _build_cvine_edges(order)
    vine.copulas = []
    for level in range(d - 1):
        fam = level_families[level]
        theta = level_thetas[level]
        cops: List[cop_par_obj] = []
        for _ in vine.ind_vine[level]:
            cops.append(cop_par_obj(fam, theta))
        vine.copulas.append(cops)
    vine.param = True
    vine.fitted = True
    return vine


def _sample_gaussian_star_cvine(
    n_samples: int,
    order: List[int],
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a Gaussian C-vine with one hub and independent higher trees.

    For a Gaussian C-vine whose level-0 edges all share the same correlation and
    whose higher trees are independent, the implied joint distribution is a
    simple one-factor Gaussian model:

      X_root ~ N(0, 1)
      X_leaf = rho * X_root + sqrt(1-rho^2) * eps_leaf

    where the leaves are conditionally independent given the root.
    """
    order = [int(v) for v in order]
    d = len(order)
    if d < 2:
        return rng.standard_normal((n_samples, d)).astype(np.float32)

    rho = float(np.clip(rho, -0.999, 0.999))
    root_scores = rng.standard_normal(n_samples).astype(np.float32)
    samples = np.zeros((n_samples, d), dtype=np.float32)
    samples[:, order[0]] = root_scores
    scale = float(np.sqrt(max(1.0 - rho * rho, 1e-8)))
    for leaf in order[1:]:
        eps = rng.standard_normal(n_samples).astype(np.float32)
        samples[:, int(leaf)] = rho * root_scores + scale * eps
    return samples


def _mean_copula_nll(vine: vine_obj_bin, x: np.ndarray) -> float:
    """Mean negative log copula density (copula NLL), in nats.

    Important: compute in log-space to avoid overflow when multiplying many edge PDFs.
    Uses rank-based pseudo-observations (pseudo-likelihood), so it does not assume
    a parametric form for the marginals.
    """
    from ..core.nonparametric_vine import _build_edge_input_pairs, _build_internal_edge_structure
    from ..core.vine_tree import flip_check_all

    x_t = torch.tensor(x, dtype=torch.float32)
    n, d = x_t.shape
    device = x_t.device

    # Base pseudo-observations via ranks (per-dimension).
    u_state = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    u_state_flip = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    for i in range(d):
        vals = x_t[:, i].contiguous()
        sorted_col = torch.sort(vals)[0]
        ranks = torch.searchsorted(sorted_col, vals).float() + 1.0
        u_state[:, 0, i] = (ranks / (n + 1.0)).clamp(1e-6, 1.0 - 1e-6)

    edge_refs = getattr(vine, "_internal_ind_vine", None)
    if edge_refs is None:
        edge_refs = _build_internal_edge_structure(vine, d)
        vine._internal_ind_vine = edge_refs

    log_cop = torch.zeros(n, dtype=torch.float32, device=device)
    for level in range(d - 1):
        if not getattr(vine, "flip_flag", None) or level >= len(vine.flip_flag):
            flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(edge_refs, level, False, 1)
        else:
            flip_flag1 = vine.flip_flag[level]
            ind_edge_rel1 = vine.ind_edge_rel[level]

        point_u = _build_edge_input_pairs(
            state=u_state,
            state_flip=u_state_flip,
            edge_refs=edge_refs,
            level=level,
            device=device,
        )
        cops_now = vine.copulas[level] if level < len(vine.copulas) else []
        density_edges_seen = set()
        for j, ind_edge in enumerate(ind_edge_rel1):
            if ind_edge >= len(cops_now):
                continue
            cobj = cops_now[ind_edge]
            uv = point_u[:, :, ind_edge]
            if int(ind_edge) not in density_edges_seen:
                density_edges_seen.add(int(ind_edge))
                pdf_val = copulapdf(cobj, uv).clamp_min(1e-30)
                log_cop = log_cop + torch.log(pdf_val)
            if flip_flag1[j]:
                hval = copulaccdf(cobj, uv[:, [1, 0]]).clamp(1e-6, 1.0 - 1e-6)
                u_state_flip[:, level + 1, ind_edge] = torch.where(torch.isfinite(hval), hval, uv[:, 0])
            else:
                hval = copulaccdf(cobj, uv).clamp(1e-6, 1.0 - 1e-6)
                u_state[:, level + 1, ind_edge] = torch.where(torch.isfinite(hval), hval, uv[:, 1])

    return float((-log_cop).mean().detach().cpu())


def _mean_level0_edge_nll(vine: vine_obj_bin, x: np.ndarray) -> float:
    """Mean NLL (nats) of only the level-0 edges of a fitted C-vine.

    This is useful for comparing against a pairwise-only baseline on the same edge set.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected x shape (N,d), got {x.shape}")

    if not getattr(vine, "copulas", None) or not vine.copulas or not vine.copulas[0]:
        return float("nan")
    if not getattr(vine, "ind_vine", None) or not vine.ind_vine or not vine.ind_vine[0]:
        return float("nan")

    u = _pseudo_obs_from_gaussianized(x)
    nlls: List[float] = []
    for e_idx, cop in enumerate(vine.copulas[0]):
        if e_idx >= len(vine.ind_vine[0]):
            continue
        i, j = vine.ind_vine[0][e_idx]
        u_pair = u[:, [int(i), int(j)]]
        nlls.append(_mean_bivariate_copula_nll(cop, u_pair))
    if not nlls:
        return float("nan")
    return float(np.mean(nlls))


def _gaussian_copula_nll(x: np.ndarray, ridge: float = 1e-4) -> float:
    """Mean negative log copula density under Gaussian copula fit, in nats.

    Uses rank pseudo-observations + normal scores (so it does not assume known marginals).
    """
    z = _normal_scores_from_rank_pobs(np.asarray(x, dtype=np.float64))
    n, d = z.shape
    if n < 5:
        return float("nan")

    # Correlation fit on normal scores derived from rank pseudo-observations.
    R = np.corrcoef(z, rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    # Stabilize: ridge + eigenvalue clip; re-normalize diag to 1.
    R = R + ridge * np.eye(d)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = V @ np.diag(w) @ V.T
    dstd = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    R = R / np.outer(dstd, dstd)
    np.fill_diagonal(R, 1.0)

    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0 or not np.isfinite(logdet):
        return float("nan")
    invR = np.linalg.inv(R)
    A = invR - np.eye(d)
    quad = np.einsum("ni,ij,nj->n", z, A, z)
    log_c = -0.5 * logdet - 0.5 * quad
    return float(-np.mean(log_c))


def _gaussian_copula_fit_corr(x_train: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Fit a stabilized correlation matrix for a Gaussian copula."""
    z_train = _normal_scores_from_rank_pobs(np.asarray(x_train, dtype=np.float64))
    n, d = z_train.shape
    if n < 5:
        return np.eye(d, dtype=np.float64)

    R = np.corrcoef(z_train, rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)

    R = R + ridge * np.eye(d)
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-6, None)
    R = V @ np.diag(w) @ V.T
    dstd = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    R = R / np.outer(dstd, dstd)
    np.fill_diagonal(R, 1.0)
    return R


def _gaussian_copula_nll_given_corr(x: np.ndarray, R: np.ndarray) -> float:
    """Mean negative log copula density under a Gaussian copula with fixed correlation."""
    x = np.asarray(x, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n, d = x.shape
    if n < 5:
        return float("nan")

    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0 or not np.isfinite(logdet):
        return float("nan")
    invR = np.linalg.inv(R)
    A = invR - np.eye(d)
    quad = np.einsum("ni,ij,nj->n", x, A, x)
    log_c = -0.5 * logdet - 0.5 * quad
    return float(-np.mean(log_c))


def _gaussian_copula_nll_fit_eval(x_train: np.ndarray, x_test: np.ndarray, ridge: float = 1e-4) -> float:
    """Fit Gaussian-copula correlation on train and evaluate copula NLL on test."""
    R = _gaussian_copula_fit_corr(x_train, ridge=ridge)
    z_test = _normal_scores_from_rank_pobs(np.asarray(x_test, dtype=np.float64))
    return _gaussian_copula_nll_given_corr(z_test, R)


def _corr_from_cov(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    dstd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    R = C / np.outer(dstd, dstd)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def _glasso_gaussian_copula_nll_fit_eval(
    x_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 0.02,
) -> float:
    """GraphicalLasso Gaussian-copula baseline: fit sparse covariance on train, evaluate copula NLL on test."""
    from sklearn.covariance import GraphicalLasso

    x_train = _normal_scores_from_rank_pobs(np.asarray(x_train, dtype=np.float64))
    x_test = _normal_scores_from_rank_pobs(np.asarray(x_test, dtype=np.float64))
    if x_train.shape[0] < 5 or x_test.shape[0] < 5:
        return float("nan")
    model = GraphicalLasso(alpha=float(alpha), max_iter=200)
    model.fit(x_train)
    R = _corr_from_cov(np.asarray(model.covariance_, dtype=np.float64))
    return _gaussian_copula_nll_given_corr(x_test, R)


def _tvgl_gaussian_copula_nll_fit_eval(
    x_train_by_time: List[np.ndarray],
    x_test_by_time: List[np.ndarray],
    alpha: float,
    beta: float,
    max_iter: int = 200,
    step_size: float = 0.05,
    eps: float = 1e-4,
) -> List[float]:
    """TVGL-style Gaussian-copula baseline across time.

    Fits a precision-matrix sequence on train covariances, then evaluates per-time copula NLL on test.
    """
    covs = []
    for xtr in x_train_by_time:
        xtr = _normal_scores_from_rank_pobs(np.asarray(xtr, dtype=np.float64))
        C = np.cov(xtr, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        C = 0.5 * (C + C.T) + 1e-6 * np.eye(C.shape[0])
        covs.append(C)

    tvgl = tvgl_frobenius(
        covs,
        alpha=float(alpha),
        beta=float(beta),
        max_iter=int(max_iter),
        step_size=float(step_size),
        eps=float(eps),
        verbose=False,
    )
    out = []
    for P, xte in zip(tvgl.precision, x_test_by_time):
        cov = np.linalg.inv(P)
        R = _corr_from_cov(cov)
        zte = _normal_scores_from_rank_pobs(np.asarray(xte, dtype=np.float64))
        out.append(_gaussian_copula_nll_given_corr(zte, R))
    return out


def _estimate_hub_by_correlation(x: np.ndarray) -> int:
    """Heuristic hub estimate: argmax_i sum_j |corr(i,j)|."""
    x = np.asarray(x, dtype=np.float64)
    R = np.corrcoef(x, rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(R, 0.0)
    score = np.sum(np.abs(R), axis=1)
    return int(np.argmax(score))


def _estimate_hub_by_glasso(
    x: np.ndarray,
    alpha: float = 0.02,
    edge_threshold: float = 0.05,
) -> int:
    """Heuristic hub estimate from Graphical Lasso partial-correlation degrees."""
    from sklearn.covariance import GraphicalLasso

    x = np.asarray(x, dtype=np.float64)
    model = GraphicalLasso(alpha=float(alpha), max_iter=200)
    model.fit(x)
    P = np.asarray(model.precision_, dtype=np.float64)
    denom = np.sqrt(np.outer(np.diag(P), np.diag(P)))
    denom = np.clip(denom, 1e-12, None)
    pcor = -P / denom
    np.fill_diagonal(pcor, 0.0)
    adj = (np.abs(pcor) >= float(edge_threshold)).astype(np.int32)
    deg = adj.sum(axis=1)
    # Break ties deterministically by lowest index.
    return int(np.argmax(deg))


def _empirical_tail_dependence(u: np.ndarray, q: float, tail: str) -> float:
    """Empirical tail dependence for a bivariate pseudo-observation array u[:,2]."""
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"Expected shape [N,2], got {u.shape}")
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'")
    if tail == "upper":
        a = u[:, 0] > q
        b = u[:, 1] > q
    else:
        a = u[:, 0] < q
        b = u[:, 1] < q
    denom = max(int(a.sum()), 1)
    return float((a & b).sum() / denom)


def _taildep_student(rho: float, nu: float) -> float:
    """Upper/lower tail dependence coefficient for Student-t copula (symmetric)."""
    rho = float(np.clip(rho, -0.999999, 0.999999))
    nu = float(max(nu, 2.1))
    arg = -math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * student_t.cdf(arg, df=nu + 1.0))


def _taildeps_from_copula(cop: cop_par_obj) -> Tuple[float, float]:
    fam = str(cop.family).lower().strip()
    theta = cop.theta
    if fam in {"ind", "independence", "gaussian", "frank"}:
        return 0.0, 0.0
    if fam == "student":
        try:
            rho, nu = float(theta[0]), float(theta[1])
        except Exception:
            rho, nu = 0.0, 4.0
        lam = _taildep_student(rho, nu)
        return lam, lam
    if fam == "clayton":
        th = float(theta)
        th = max(th, 1e-6)
        return float(2.0 ** (-1.0 / th)), 0.0
    if fam == "gumbel":
        th = float(theta)
        th = max(th, 1.0 + 1e-6)
        return 0.0, float(2.0 - 2.0 ** (1.0 / th))
    return 0.0, 0.0


def generate_multiplicative_triplet(
    n_samples: int,
    noise_std: float,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, 1))
    y = rng.standard_normal((n_samples, 1))
    z = x * y + noise_std * rng.standard_normal((n_samples, 1))
    raw = np.concatenate([x, y, z], axis=1)
    data = gaussianize_columns(raw)
    return {"data": data, "raw": raw.astype(np.float32)}


def generate_dynamic_tail_df(
    n_time_steps: int,
    n_samples_per_time: int,
    n_variables: int,
    rho: float,
    nu_low: float,
    nu_high: float,
    schedule: str,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    time_idx = np.arange(n_time_steps, dtype=np.float32)
    if schedule == "linear":
        nu_schedule = np.linspace(nu_low, nu_high, n_time_steps, dtype=np.float32)
        change_point = None
    else:
        # default: piecewise step.
        cp = n_time_steps // 2
        nu_schedule = np.full(n_time_steps, nu_low, dtype=np.float32)
        nu_schedule[cp:] = nu_high
        change_point = cp

    # Generate a t-copula time series with constant correlation and time-varying nu,
    # then map to standard normal marginals via probability integral transform.
    d = n_variables
    rho = float(np.clip(rho, -0.99, 0.99))
    if rho <= -1.0 / max(d - 1, 1):
        raise ValueError(f"Equicorrelation rho={rho} is not PSD for d={d}")
    Sigma = (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))

    data = np.zeros((n_time_steps, n_samples_per_time, d), dtype=np.float32)
    eps = 1e-6
    for t in range(n_time_steps):
        nu_t = float(nu_schedule[t])
        z = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n_samples_per_time).astype(np.float64)
        g = rng.chisquare(df=nu_t, size=n_samples_per_time).astype(np.float64) / nu_t
        x_t = z / np.sqrt(g)[:, None]  # multivariate t with df=nu_t
        u_t = student_t.cdf(x_t, df=nu_t)
        u_t = np.clip(u_t, eps, 1.0 - eps)
        data[t] = norm.ppf(u_t).astype(np.float32)

    return {
        "time_data": data,
        "time_indices": time_idx,
        "nu_schedule": nu_schedule,
        "change_point": change_point,
        "rho": float(rho),
    }


def generate_tail_switch_clayton_gumbel(
    n_time_steps: int,
    n_samples_per_time: int,
    n_variables: int,
    kendall_tau: float,
    seed: int,
) -> Dict[str, Any]:
    """Regime-switching dependence with matched Kendall tau but opposite tail asymmetry."""
    rng = np.random.default_rng(seed)
    time_idx = np.arange(n_time_steps, dtype=np.float32)
    cp = n_time_steps // 2
    tau = float(np.clip(kendall_tau, 0.05, 0.9))

    # Match Kendall tau:
    # Clayton: tau = theta / (theta + 2)  => theta = 2*tau/(1-tau)
    theta_clayton = 2.0 * tau / (1.0 - tau)
    # Gumbel: tau = 1 - 1/theta => theta = 1/(1-tau)
    theta_gumbel = 1.0 / (1.0 - tau)

    data = np.zeros((n_time_steps, n_samples_per_time, n_variables), dtype=np.float32)
    fam_schedule: List[str] = []
    eps = 1e-6
    for t in range(n_time_steps):
        if t < cp:
            fam = "clayton"
            theta = theta_clayton
        else:
            fam = "gumbel"
            theta = theta_gumbel
        fam_schedule.append(fam)

        # Stable star construction: only level-0 dependence (root 0), no deep inverse-h recursion.
        u0 = rng.uniform(eps, 1.0 - eps, size=n_samples_per_time).astype(np.float32)
        U = np.zeros((n_samples_per_time, n_variables), dtype=np.float32)
        U[:, 0] = u0
        cobj = cop_par_obj(fam, theta)
        for j in range(1, n_variables):
            w = rng.uniform(eps, 1.0 - eps, size=n_samples_per_time).astype(np.float32)
            uv = torch.tensor(np.stack([u0, w], axis=1), dtype=torch.float32)
            uj = copulainvccdf(cobj, uv).detach().cpu().numpy().astype(np.float32)
            U[:, j] = np.clip(uj, eps, 1.0 - eps)

        data[t] = norm.ppf(U).astype(np.float32)

    return {
        "time_data": data,
        "time_indices": time_idx,
        "change_point": cp,
        "kendall_tau_target": tau,
        "theta_clayton": float(theta_clayton),
        "theta_gumbel": float(theta_gumbel),
        "family_schedule": fam_schedule,
    }


def generate_hub_switch(
    n_time_steps: int,
    n_samples_per_time: int,
    n_variables: int,
    hub_a: int,
    hub_b: int,
    rho_hub: float,
    seed: int,
) -> Dict[str, Any]:
    """Switch the C-vine root (hub) variable half-way through time."""
    rng = np.random.default_rng(seed)
    time_idx = np.arange(n_time_steps, dtype=np.float32)
    cp = n_time_steps // 2

    hub_a = int(hub_a)
    hub_b = int(hub_b)
    if not (0 <= hub_a < n_variables and 0 <= hub_b < n_variables):
        raise ValueError("hub indices out of range")

    def order_for_hub(h: int) -> List[int]:
        rest = [i for i in range(n_variables) if i != h]
        return [h] + rest

    data = np.zeros((n_time_steps, n_samples_per_time, n_variables), dtype=np.float32)
    true_hubs: List[int] = []
    for t in range(n_time_steps):
        hub = hub_a if t < cp else hub_b
        true_hubs.append(hub)
        order = order_for_hub(hub)

        # Avoid benchmark contamination from generic recursive sampling here:
        # this scenario is analytically a Gaussian star with independent higher trees.
        data[t] = _sample_gaussian_star_cvine(
            n_samples=n_samples_per_time,
            order=order,
            rho=rho_hub,
            rng=rng,
        )
        data[t] += 1e-3 * rng.standard_normal(data[t].shape).astype(np.float32)

    return {
        "time_data": data,
        "time_indices": time_idx,
        "change_point": cp,
        "true_hubs": true_hubs,
        "rho_hub": float(rho_hub),
    }


def generate_agent_interaction_episodes(
    n_time_steps: int,
    n_samples_per_time: int,
    n_agents: int,
    rho_pairwise: float,
    rho_higher: float,
    nu_higher: float,
    seed: int,
) -> Dict[str, Any]:
    """Generate time series with episodic interaction bursts among agents.

    Background state is independence. Several separated interaction episodes
    are embedded so the benchmark tests recurrence across the same motifs:
    1. Pairwise Gaussian-copula bursts
    2. Higher-order Student-t/Clayton C-vine bursts
    3. Mixed bursts with disjoint pairwise and higher-order groups

    The key demonstration: a 1-truncated vine suffices for pairwise episodes,
    but higher-order episodes require the full vine (deeper tree levels).
    """
    if n_agents < 6:
        raise ValueError("agent interaction episodes require n_agents >= 6")

    rng = np.random.default_rng(seed)
    time_idx = np.arange(n_time_steps, dtype=np.float32)

    # ----- episode schedule -------------------------------------------------
    # label codes: 0=independence, 1=pairwise, 2=higher_order, 3=mixed
    episode_labels = np.zeros(n_time_steps, dtype=np.int32)
    episode_agents: List[List[int]] = [[] for _ in range(n_time_steps)]
    episode_specs: List[Optional[Dict[str, Any]]] = [None for _ in range(n_time_steps)]
    episode_schedule: List[Dict[str, Any]] = []

    # Duration weights are scaled to n_time_steps. Short unit tests keep the
    # older compact schedule; paper-scale runs use recurrent episodes.
    legacy_template: List[Dict[str, Any]] = [
        {"weight": 5.0, "type": "independence"},
        {"weight": 5.0, "type": "pairwise", "agents": [0, 1]},
        {"weight": 4.0, "type": "independence"},
        {"weight": 6.0, "type": "higher_order", "agents": [2, 3, 4]},
        {"weight": 3.0, "type": "independence"},
        {"weight": 5.0, "type": "mixed", "agents_pairwise": [0, 1], "agents_higher": [3, 4, 5]},
    ]
    recurrent_template: List[Dict[str, Any]] = [
        {"weight": 5.0, "type": "independence"},
        {"weight": 6.0, "type": "pairwise", "agents": [0, 1]},
        {"weight": 4.0, "type": "independence"},
        {"weight": 7.0, "type": "higher_order", "agents": [2, 3, 4]},
        {"weight": 4.0, "type": "independence"},
        {"weight": 6.0, "type": "mixed", "agents_pairwise": [0, 1], "agents_higher": [3, 4, 5]},
        {"weight": 4.0, "type": "independence"},
        {"weight": 6.0, "type": "pairwise", "agents": [0, 1]},
        {"weight": 7.0, "type": "higher_order", "agents": [2, 3, 4]},
        {"weight": 5.0, "type": "mixed", "agents_pairwise": [0, 1], "agents_higher": [3, 4, 5]},
        {"weight": 4.0, "type": "independence"},
    ]
    schedule_template = recurrent_template if n_time_steps >= 36 else legacy_template
    weights = np.asarray([float(entry["weight"]) for entry in schedule_template], dtype=np.float64)
    cum = np.round(np.cumsum(weights / np.sum(weights)) * n_time_steps).astype(int)
    cum[-1] = n_time_steps
    boundaries = np.concatenate([[0], cum])

    label_code = {"pairwise": 1, "higher_order": 2, "mixed": 3}

    for seg_idx, spec in enumerate(schedule_template):
        t_start = int(boundaries[seg_idx])
        t_end = int(boundaries[seg_idx + 1])
        if t_end <= t_start or spec["type"] == "independence":
            continue

        spec_clean = {k: v for k, v in spec.items() if k != "weight"}
        spec_clean["t_start"] = t_start
        spec_clean["t_end"] = t_end
        episode_schedule.append(spec_clean)

        label = label_code[str(spec["type"])]
        if spec["type"] == "mixed":
            agents = sorted(set(spec["agents_pairwise"]) | set(spec["agents_higher"]))
        else:
            agents = list(spec["agents"])

        for t in range(t_start, t_end):
            episode_labels[t] = label
            episode_agents[t] = agents
            episode_specs[t] = spec_clean

    # ----- data generation ---------------------------------------------------
    eps = 1e-6
    data = np.zeros((n_time_steps, n_samples_per_time, n_agents), dtype=np.float32)

    for t in range(n_time_steps):
        # Start with all-independent standard normals.
        data[t] = rng.standard_normal((n_samples_per_time, n_agents)).astype(np.float32)

        label = int(episode_labels[t])

        if label == 0:
            # Independence — already done.
            pass

        elif label == 1:
            # Pairwise interaction via Gaussian copula.
            spec = episode_specs[t] or {"agents": [0, 1]}
            data[t] = _embed_pairwise(
                data[t], agents=spec["agents"], rho=rho_pairwise, rng=rng, eps=eps,
            )

        elif label == 2:
            # Higher-order interaction via 2-level C-vine.
            spec = episode_specs[t] or {"agents": [2, 3, 4]}
            data[t] = _embed_higher_order_vine(
                data[t], agents=spec["agents"],
                rho=rho_higher, nu=nu_higher, rng=rng, eps=eps,
            )

        elif label == 3:
            # Mixed: pairwise + higher-order on disjoint subsets.
            spec = episode_specs[t] or {"agents_pairwise": [0, 1], "agents_higher": [3, 4, 5]}
            data[t] = _embed_pairwise(
                data[t], agents=spec["agents_pairwise"], rho=rho_pairwise, rng=rng, eps=eps,
            )
            data[t] = _embed_higher_order_vine(
                data[t], agents=spec["agents_higher"],
                rho=rho_higher, nu=nu_higher, rng=rng, eps=eps,
            )

        # Small jitter for numerical stability.
        data[t] += 1e-3 * rng.standard_normal(data[t].shape).astype(np.float32)

    return {
        "time_data": data,
        "time_indices": time_idx,
        "episode_labels": episode_labels,
        "episode_agents": episode_agents,
        "episode_schedule": episode_schedule,
    }


def _sample_modsum_higher_order_triplet(
    n_samples: int,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> np.ndarray:
    """Sample a continuous XOR-style triplet with pairwise-independent marginals.

    Let U, V ~ Unif(0,1) and W = (U + V) mod 1. Each pair is marginally
    independent, but the triplet has deterministic higher-order dependence.
    After Gaussianization, pairwise Gaussian statistics remain near zero while
    the full-vine conditional edge carries substantial signal.
    """
    uv = rng.uniform(eps, 1.0 - eps, size=(n_samples, 2)).astype(np.float64)
    w = np.mod(uv[:, 0] + uv[:, 1], 1.0)
    w = np.clip(w, eps, 1.0 - eps)
    xyz = np.column_stack([uv[:, 0], uv[:, 1], w])
    return norm.ppf(xyz).astype(np.float32)


def generate_higher_order_only_switch(
    n_time_steps: int,
    n_samples_per_time: int,
    seed: int,
) -> Dict[str, Any]:
    """Generate a dynamic benchmark where pairwise marginals stay matched.

    Pre-change: fully independent Gaussianized triplets.
    Post-change: continuous XOR-style triplets with the same pairwise marginals
    but strong third-order dependence.
    """
    rng = np.random.default_rng(seed)
    time_idx = np.arange(n_time_steps, dtype=np.float32)
    change_point = n_time_steps // 2
    regime_labels = np.zeros(n_time_steps, dtype=np.int32)
    regime_labels[change_point:] = 1

    data = np.zeros((n_time_steps, n_samples_per_time, 3), dtype=np.float32)
    for t in range(n_time_steps):
        if t < change_point:
            u = rng.uniform(1e-6, 1.0 - 1e-6, size=(n_samples_per_time, 3))
            data[t] = norm.ppf(u).astype(np.float32)
        else:
            data[t] = _sample_modsum_higher_order_triplet(
                n_samples=n_samples_per_time,
                rng=rng,
                eps=1e-6,
            )
        data[t] += 1e-3 * rng.standard_normal(data[t].shape).astype(np.float32)

    return {
        "time_data": data,
        "time_indices": time_idx,
        "change_point": int(change_point),
        "regime_labels": regime_labels,
        "regime_schedule": [
            {"t_start": 0, "t_end": int(change_point), "type": "pairwise_matched_control"},
            {"t_start": int(change_point), "t_end": int(n_time_steps), "type": "higher_order_only"},
        ],
    }


def _embed_pairwise(
    x: np.ndarray,
    agents: List[int],
    rho: float,
    rng: np.random.Generator,
    eps: float,
) -> np.ndarray:
    """Replace columns for *agents* (len 2) with Gaussian-copula-correlated samples."""
    n = x.shape[0]
    i, j = agents[0], agents[1]
    u_i = rng.uniform(eps, 1.0 - eps, size=n).astype(np.float32)
    w = rng.uniform(eps, 1.0 - eps, size=n).astype(np.float32)
    cobj = cop_par_obj("gaussian", float(rho))
    uv = torch.tensor(np.stack([u_i, w], axis=1), dtype=torch.float32)
    u_j = copulainvccdf(cobj, uv).detach().cpu().numpy().astype(np.float32)
    u_j = np.clip(u_j, eps, 1.0 - eps)
    x[:, i] = norm.ppf(u_i).astype(np.float32)
    x[:, j] = norm.ppf(u_j).astype(np.float32)
    return x


def _embed_higher_order_vine(
    x: np.ndarray,
    agents: List[int],
    rho: float,
    nu: float,
    rng: np.random.Generator,
    eps: float,
) -> np.ndarray:
    """Replace columns for *agents* (len 3) with samples from a 2-level C-vine.

    Level 0: Student-t copula (rho, nu) — creates tail dependence.
    Level 1: Clayton copula (theta=2.0) — genuine conditional/higher-order dependence
    not captured by any pairwise-only model.
    """
    d_sub = len(agents)
    order = list(range(d_sub))
    # level 0 = Student-t, levels 1+ = Clayton
    level_fams = ["student"] + ["clayton"] * (d_sub - 2)
    level_thetas: List[Any] = [(float(rho), float(nu))] + [2.0] * (d_sub - 2)
    vine = _make_levelwise_cvine(
        d_sub, order=order, level_families=level_fams, level_thetas=level_thetas,
    )
    samples = vine.sample(x.shape[0])
    # The vine.sample() returns standard-normal-marginal data.
    for local_idx, global_idx in enumerate(agents):
        x[:, global_idx] = samples[:, local_idx]
    return x


def _fit_parametric_vine(
    x_train: np.ndarray,
    families: List[str],
    optimize_structure: bool,
    seed: int,
) -> vine_obj_bin:
    if optimize_structure:
        opt = optimize_vine_structure(
            x_train,
            vine_type="c-vine",
            method="sequential",
            criterion="kendall_tau",
            max_iterations=1,
            verbose=False,
        )
        vine = opt.best_vine
    else:
        vine = create_vine("c-vine", x_train.shape[1])

    gen_dict = {"param": True, "binning": False, "fitted": True}
    npc_dict: Dict[str, Any] = {}
    par_dict = {"param_families": families, "seed": seed}
    bin_dict: Dict[str, Any] = {}

    fit_vine(vine, x_train, gen_dict, npc_dict, par_dict, bin_dict)
    return vine


def _fit_truncated_cvine_level0(
    x_train: np.ndarray,
    families: List[str],
    order: Optional[List[int]] = None,
) -> vine_obj_bin:
    """Fit a 1-truncated C-vine: level-0 edges are fitted; higher trees are set to independence.

    This provides a coherent multivariate baseline that uses only pairwise dependence terms.
    """
    x_train = np.asarray(x_train, dtype=np.float32)
    if x_train.ndim != 2:
        raise ValueError(f"Expected x_train shape (N,d), got {x_train.shape}")
    d = int(x_train.shape[1])
    if order is None:
        order = list(range(d))

    fams = sorted({_normalize_family_name(f) for f in families} | {"ind"})
    vine = create_vine("c-vine", d, families=fams)
    vine.ind_vine = _build_cvine_edges(order)

    u = _pseudo_obs_rank(x_train)
    copulas: List[List[cop_par_obj]] = []

    # Level 0: fit each edge independently on rank pseudo-observations.
    cops0: List[cop_par_obj] = []
    for edge in vine.ind_vine[0]:
        i, j = int(edge[0]), int(edge[1])
        u_pair = u[:, [i, j]]
        cops0.append(_fit_best_bivariate_copula(u_pair, families=families))
    copulas.append(cops0)

    # Levels 1..: independence.
    for level in range(1, d - 1):
        cops_level = [cop_par_obj("ind", None) for _ in vine.ind_vine[level]]
        copulas.append(cops_level)

    vine.copulas = copulas
    vine.param = True
    vine.fitted = True
    return vine


def _fit_regularized_dynamic_cvine_from_splits(
    x_train_list: List[np.ndarray],
    x_test_list: List[np.ndarray],
    families: List[str],
    *,
    root_switch_penalty: float,
    family_switch_penalty: float,
    parameter_drift_penalty: float,
    parameter_smoothing: float,
    root_score_method: str,
) -> Tuple[RegularizedDynamicCVineResult, List[float]]:
    """Fit a temporally regularized dynamic C-vine on train windows and evaluate on test windows."""
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_list and x_test_list must have the same length")
    model = RegularizedDynamicCVine(
        families=families,
        root_switch_penalty=root_switch_penalty,
        family_switch_penalty=family_switch_penalty,
        parameter_drift_penalty=parameter_drift_penalty,
        parameter_smoothing=parameter_smoothing,
        root_score_method=root_score_method,
    )
    result = model.fit(x_train_list)
    test_nll = model.evaluate(x_test_list).tolist()
    return result, test_nll


def _fit_joint_dynamic_cvine_from_splits(
    x_train_list: List[np.ndarray],
    x_test_list: List[np.ndarray],
    families: List[str],
    *,
    order: Optional[List[int]] = None,
    n_basis: int = 4,
    smoothness_penalty: float = 5.0,
    ridge_penalty: float = 1e-3,
    maxiter: int = 80,
) -> Tuple[JointDynamicCVineResult, List[float]]:
    """Fit a jointly parameterized dynamic C-vine on train windows and evaluate on test windows."""
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_list and x_test_list must have the same length")
    model = JointDynamicCVine(
        families=families,
        order=order,
        n_basis=n_basis,
        smoothness_penalty=smoothness_penalty,
        ridge_penalty=ridge_penalty,
        maxiter=maxiter,
    )
    result = model.fit(x_train_list)
    test_nll = model.evaluate(x_test_list).tolist()
    return result, test_nll


def _fit_latent_state_dynamic_cvine_from_splits(
    x_train_list: List[np.ndarray],
    x_test_list: List[np.ndarray],
    families: List[str],
    *,
    order: Optional[List[int]] = None,
    selection_n_basis: int = 4,
    selection_smoothness_penalty: float = 5.0,
    latent_dim: int = 2,
    transition_penalty: float = 1e-2,
    n_epochs: int = 250,
    lr: float = 2e-2,
) -> Tuple[LatentStateDynamicCVineResult, List[float]]:
    """Fit a shared-latent dynamic C-vine on train windows and evaluate on test windows."""
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_list and x_test_list must have the same length")
    model = LatentStateDynamicCVine(
        families=families,
        order=order,
        selection_n_basis=selection_n_basis,
        selection_smoothness_penalty=selection_smoothness_penalty,
        latent_dim=latent_dim,
        transition_penalty=transition_penalty,
        n_epochs=n_epochs,
        lr=lr,
    )
    result = model.fit(x_train_list)
    test_nll = model.evaluate(x_test_list).tolist()
    return result, test_nll


def _fit_windowed_nonparametric_cvine_from_splits(
    x_train_list: List[np.ndarray],
    x_test_list: List[np.ndarray],
    *,
    order: Optional[List[int]] = None,
    knots: int = 9,
    npc_dict: Optional[Dict[str, Any]] = None,
    temporal_smoothing_bandwidth: float = 0.0,
) -> Tuple[WindowedNonparametricCVineResult, List[float]]:
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_list and x_test_list must have the same length")
    model = WindowedNonparametricCVine(
        order=order,
        knots=knots,
        npc_dict=npc_dict,
        temporal_smoothing_bandwidth=temporal_smoothing_bandwidth,
    )
    result = model.fit(x_train_list)
    test_nll = model.evaluate(x_test_list).tolist()
    return result, test_nll


def _fit_joint_dynamic_nonparametric_cvine_from_splits(
    x_train_list: List[np.ndarray],
    x_test_list: List[np.ndarray],
    *,
    order: Optional[List[int]] = None,
    knots: int = 9,
    trajectory_type: str = "basis",
    trajectory_kwargs: Optional[Dict[str, Any]] = None,
    n_epochs: int = 20,
    lr: float = 3e-2,
    smoothness_penalty: float = 1e-2,
    batch_size: int = 2,
    normalization_iters: int = 10,
    final_normalization_iters: int = 50,
    density_smoothing_bandwidth: float = 0.0,
) -> Tuple[JointDynamicNonparametricCVineResult, List[float]]:
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_list and x_test_list must have the same length")
    model = JointDynamicNonparametricCVine(
        order=order,
        knots=knots,
        trajectory_type=trajectory_type,
        trajectory_kwargs=trajectory_kwargs,
        n_epochs=n_epochs,
        lr=lr,
        smoothness_penalty=smoothness_penalty,
        batch_size=batch_size,
        normalization_iters=normalization_iters,
        final_normalization_iters=final_normalization_iters,
        density_smoothing_bandwidth=density_smoothing_bandwidth,
    )
    result = model.fit(x_train_list)
    test_nll = model.evaluate(x_test_list).tolist()
    return result, test_nll


class _TimeBandwidthMLP(nn.Module):
    """Small per-edge time->bandwidth network for the KDE-flow baseline."""

    def __init__(self, hidden_dim: int = 32, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return F.softplus(self.net(t)) + 1e-4


def _stack_root_edge_pairs(x: np.ndarray, root: int = 0) -> np.ndarray:
    """Return [N,2,E] array stacking pairs (root, j) for j != root."""
    x = np.asarray(x, dtype=np.float32)
    d = x.shape[1]
    pairs = []
    for j in range(d):
        if j == root:
            continue
        pairs.append(x[:, [root, j]])
    return np.stack(pairs, axis=2)  # [N,2,E]


def _kde_flow_truncated_level0_nll_from_splits(
    x_train_list: List[np.ndarray],
    x_test_list: List[np.ndarray],
    seed: int,
    *,
    root: int = 0,
    val_fraction: float = 0.2,
    n_epochs: int = 200,
    lr: float = 1e-2,
    hidden_dim: int = 32,
    batch_time_steps: int = 8,
    device: str = "auto",
) -> Tuple[List[float], List[float], np.ndarray]:
    """KDE-flow baseline: a time->bandwidth flow for a 1-truncated C-vine in normal space.

    Each (root, j) edge is modeled by a local-likelihood KDE on (X_root, X_j), with a
    learned bandwidth as a smooth function of time. The multivariate copula density is
    approximated by a 1-truncated C-vine: sum of log copula densities over level-0 edges.

    Notes
    -----
    - Assumes each marginal is approximately standard normal, so copula log-density can
      be computed as log p(x_i, x_j) - log phi(x_i) - log phi(x_j).
    - Uses a held-out validation split (within each time step's train set) to prevent
      bandwidth collapse.
    """
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_list and x_test_list must have the same length")
    T = len(x_train_list)
    if T < 2:
        raise ValueError("Need at least 2 time steps for KDE-flow baseline")
    d = int(np.asarray(x_train_list[0]).shape[1])
    if d < 2:
        raise ValueError("Need at least 2 variables for KDE-flow baseline")

    # Time normalization to [0,1] for stable training.
    t_idx = np.arange(T, dtype=np.float32)
    t_norm = (t_idx - t_idx.min()) / (t_idx.max() - t_idx.min() + 1e-8)

    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    torch.manual_seed(seed)

    n_edges = d - 1

    # Within each time step's train set: fit + val.
    val_fraction = float(val_fraction)
    val_fraction = float(np.clip(val_fraction, 0.05, 0.5))

    fit_pairs_t: List[torch.Tensor] = []
    val_pairs_t: List[torch.Tensor] = []
    test_pairs_t: List[torch.Tensor] = []
    train_full_pairs_t: List[torch.Tensor] = []
    for t in range(T):
        tr = np.asarray(x_train_list[t], dtype=np.float32)
        te = np.asarray(x_test_list[t], dtype=np.float32)
        if tr.ndim != 2 or te.ndim != 2 or tr.shape[1] != d or te.shape[1] != d:
            raise ValueError("All train/test arrays must have shape (N,d) with fixed d")
        n_tr = tr.shape[0]
        if n_tr < 20:
            raise ValueError("Need at least ~20 training samples per time step for KDE-flow baseline")

        rng_t = np.random.default_rng(int(seed + 10007 * t))
        idx = rng_t.permutation(n_tr)
        tr = tr[idx]
        n_fit = max(int(round((1.0 - val_fraction) * n_tr)), 10)
        n_fit = min(n_fit, n_tr - 5)
        fit = tr[:n_fit]
        val = tr[n_fit:]

        fit_pairs_t.append(torch.tensor(_stack_root_edge_pairs(fit, root=root), dtype=torch.float32, device=device_t))
        val_pairs_t.append(torch.tensor(_stack_root_edge_pairs(val, root=root), dtype=torch.float32, device=device_t))
        test_pairs_t.append(torch.tensor(_stack_root_edge_pairs(te, root=root), dtype=torch.float32, device=device_t))
        train_full_pairs_t.append(torch.tensor(_stack_root_edge_pairs(tr, root=root), dtype=torch.float32, device=device_t))

    t_norm_t = torch.tensor(t_norm, dtype=torch.float32, device=device_t).unsqueeze(-1)  # [T,1]

    flows = nn.ModuleList(
        [_TimeBandwidthMLP(hidden_dim=hidden_dim, out_dim=2) for _ in range(n_edges)]
    ).to(device_t)
    opt = torch.optim.Adam(flows.parameters(), lr=lr)
    normal = torch.distributions.Normal(0.0, 1.0)

    def _predict_B(t_batch: torch.Tensor) -> torch.Tensor:
        # returns [B,2,E]
        bws = []
        for flow in flows:
            bws.append(flow(t_batch))  # [B,2]
        return torch.stack(bws, dim=2)

    def _nll_for_time(
        data_fit: torch.Tensor,  # [Nfit,2,E]
        data_eval: torch.Tensor,  # [Neval,2,E]
        B: torch.Tensor,  # [2,E]
    ) -> torch.Tensor:
        # loclik expects [N,2,E] and [M,2,E]
        ker = loclik_batch_eval(B, data_fit, data_eval, n_cop=n_edges, batch_size=1).clamp_min(1e-30)
        log_joint = torch.log(ker)  # [M,E]
        log_norm = normal.log_prob(data_eval[:, 0, :]) + normal.log_prob(data_eval[:, 1, :])
        log_c = log_joint - log_norm
        # mean over samples per edge, then sum edges => truncated vine log-density
        return -(log_c.mean(dim=0).sum())

    # Train on validation NLL to avoid bandwidth collapse on training data.
    history = []
    for ep in range(int(n_epochs)):
        opt.zero_grad(set_to_none=True)
        bsz = min(int(batch_time_steps), T)
        rng_ep = np.random.default_rng(int(seed + 9973 * ep))
        t_sel = rng_ep.choice(T, size=bsz, replace=False)
        t_sel_t = torch.tensor(t_sel, dtype=torch.long, device=device_t)
        t_batch = t_norm_t[t_sel_t]  # [B,1]
        bws = _predict_B(t_batch)  # [B,2,E]

        loss = torch.zeros((), dtype=torch.float32, device=device_t)
        for bi, t_idx_i in enumerate(t_sel):
            B = bws[bi]  # [2,E]
            loss = loss + _nll_for_time(fit_pairs_t[t_idx_i], val_pairs_t[t_idx_i], B)

        loss = loss / float(bsz)

        # Mild regularization: discourage extreme bandwidths.
        bw_mean = bws.mean()
        bw_pen = 1e-3 * torch.relu(0.02 - bw_mean) + 1e-3 * torch.relu(bw_mean - 2.0)
        loss = loss + bw_pen

        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))

    # Predict bandwidths over all time steps.
    with torch.no_grad():
        bws_all = _predict_B(t_norm_t).detach().cpu().numpy()  # [T,2,E]

    # Evaluate per time on val/test, using full train as KDE data.
    val_nll = []
    test_nll = []
    with torch.no_grad():
        for t in range(T):
            B = torch.tensor(bws_all[t], dtype=torch.float32, device=device_t)  # [2,E]
            val_nll.append(float(_nll_for_time(train_full_pairs_t[t], val_pairs_t[t], B).detach().cpu()))
            test_nll.append(float(_nll_for_time(train_full_pairs_t[t], test_pairs_t[t], B).detach().cpu()))

    return test_nll, val_nll, bws_all


def _kde_flow_truncated_level0_nll(
    time_data: np.ndarray,
    seed: int,
    *,
    root: int = 0,
    val_fraction: float = 0.2,
    n_epochs: int = 200,
    lr: float = 1e-2,
    hidden_dim: int = 32,
    batch_time_steps: int = 8,
    device: str = "auto",
) -> Tuple[List[float], List[float], np.ndarray]:
    """Convenience wrapper around `_kde_flow_truncated_level0_nll_from_splits` using an internal split."""
    time_data = np.asarray(time_data, dtype=np.float32)
    if time_data.ndim != 3:
        raise ValueError(f"Expected time_data shape [T,N,d], got {time_data.shape}")
    T, N, _d = time_data.shape
    n_train = int(0.8 * N)
    x_train_list: List[np.ndarray] = []
    x_test_list: List[np.ndarray] = []
    for t in range(T):
        idx = np.random.default_rng(int(seed + 37 * t)).permutation(N)
        x_t = time_data[t]
        x_train_list.append(x_t[idx[:n_train]])
        x_test_list.append(x_t[idx[n_train:]])
    return _kde_flow_truncated_level0_nll_from_splits(
        x_train_list,
        x_test_list,
        seed=seed,
        root=root,
        val_fraction=val_fraction,
        n_epochs=n_epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        batch_time_steps=batch_time_steps,
        device=device,
    )


def _plot_multiplicative_triplet(
    out_png: Path,
    x: np.ndarray,
    title: str,
) -> None:
    _set_seaborn_style()
    df = pd.DataFrame(x, columns=["X", "Y", "Z"])
    fig, axes = plt.subplots(
        1, 3, figsize=(11.5, 3.2), sharex=False, sharey=False,
        constrained_layout=True,
    )

    axes[0].scatter(df["Y"], df["Z"], s=3, alpha=0.3, color="#1f77b4", edgecolors="none")
    axes[0].set_title(r"All samples: $Y$ vs $Z$")
    axes[0].set_xlabel(r"$Y$")
    axes[0].set_ylabel(r"$Z$")

    mask_pos = df["X"] > 0
    axes[1].scatter(
        df.loc[mask_pos, "Y"], df.loc[mask_pos, "Z"],
        s=3, alpha=0.3, color="#d62728", edgecolors="none",
    )
    axes[1].set_title(r"Conditioned: $X > 0$")
    axes[1].set_xlabel(r"$Y$")
    axes[1].set_ylabel(r"$Z$")

    mask_neg = df["X"] < 0
    axes[2].scatter(
        df.loc[mask_neg, "Y"], df.loc[mask_neg, "Z"],
        s=3, alpha=0.3, color="#2ca02c", edgecolors="none",
    )
    axes[2].set_title(r"Conditioned: $X < 0$")
    axes[2].set_xlabel(r"$Y$")
    axes[2].set_ylabel(r"$Z$")

    fig.suptitle(title, y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_higher_order_only_switch_panel(
    out_png: Path,
    time: np.ndarray,
    pairwise_abs_corr: np.ndarray,
    tc_higher_order: np.ndarray,
    detection_threshold: float,
    nll_gaps: Dict[str, np.ndarray],
    change_point: int,
    title: str,
) -> None:
    _set_seaborn_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)

    # Panel 1: pairwise summary should stay flat across the regime switch.
    ax = axes[0]
    ax.plot(time, pairwise_abs_corr, color="#1f77b4", linewidth=2.2)
    ax.axvline(float(time[change_point]), color="red", linewidth=2, alpha=0.8)
    ax.set_title("Pairwise mean |corr|")
    ax.set_xlabel("time")
    ax.set_ylabel(r"mean $|\rho_{ij}|$")

    # Panel 2: higher-order score should rise only after the switch.
    ax = axes[1]
    ax.plot(time, tc_higher_order, color="#d62728", linewidth=2.2, label=r"$\mathrm{TC}_{\mathrm{higher}}$")
    ax.axhline(float(detection_threshold), color="gray", linewidth=1.3, linestyle="--", label="threshold")
    ax.axvline(float(time[change_point]), color="red", linewidth=2, alpha=0.8)
    ax.fill_between(
        time,
        detection_threshold,
        tc_higher_order,
        where=tc_higher_order >= detection_threshold,
        color="#d62728",
        alpha=0.15,
    )
    ax.set_title("Higher-order score")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\mathrm{NLL}(1\mathrm{-trunc}) - \mathrm{NLL}(\mathrm{DVC})$")
    ax.legend(frameon=True)

    # Panel 3: NLL gaps vs baselines.
    ax = axes[2]
    ax.plot(time, nll_gaps["nll_gap"], color="black", linewidth=2.0, label="Gaussian copula")
    if "nll_gap_truncated_level0" in nll_gaps:
        ax.plot(time, nll_gaps["nll_gap_truncated_level0"], color="#1f77b4", linewidth=1.8, linestyle="--", label="1-truncated C-vine")
    if "nll_gap_glasso" in nll_gaps:
        ax.plot(time, nll_gaps["nll_gap_glasso"], color="#2ca02c", linewidth=1.6, linestyle=":", label="Graphical Lasso")
    if "nll_gap_tvgl" in nll_gaps:
        ax.plot(time, nll_gaps["nll_gap_tvgl"], color="#d62728", linewidth=1.6, linestyle="-.", label="TVGL (Frobenius)")
    if "nll_gap_state_space" in nll_gaps:
        ax.plot(time, nll_gaps["nll_gap_state_space"], color="#9467bd", linewidth=1.6, linestyle=(0, (3, 1, 1, 1)), label="Gaussian SSM")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax.axvline(float(time[change_point]), color="red", linewidth=2, alpha=0.8)
    ax.set_title("Held-out NLL gap")
    ax.set_xlabel("time")
    ax.set_ylabel("baseline - DVC")
    ax.legend(frameon=True)

    fig.suptitle(title, y=1.03)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic_panel(
    out_png: Path,
    time: np.ndarray,
    series: Dict[str, np.ndarray],
    change_point: Optional[int],
    title: str,
    family_heatmap: Optional[np.ndarray] = None,
    family_labels: Optional[List[str]] = None,
) -> None:
    _set_seaborn_style()
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Curated color palette: distinct colors that read well on white background.
    _corr_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]

    # Panel 1: second-order summaries.
    ci = 0
    for k, y in series.items():
        if not k.startswith("corr_"):
            continue
        ax1.plot(
            time, y,
            label=k.replace("corr_", "").replace("_", " "),
            linewidth=2.0,
            color=_corr_colors[ci % len(_corr_colors)],
        )
        ci += 1
    ax1.set_title("Pairwise summaries")
    ax1.set_xlabel("time")
    ax1.set_ylabel("mean statistic")
    ax1.legend(frameon=True)

    # Panel 2: tail dependence.
    ci = 0
    for k, y in series.items():
        if not k.startswith("tail_"):
            continue
        ax2.plot(
            time, y,
            label=k.replace("tail_", "").replace("_", " "),
            linewidth=2.0,
            color=_corr_colors[ci % len(_corr_colors)],
        )
        ci += 1
    ax2.set_title("Tail dependence")
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\lambda$")
    ax2.legend(frameon=True)

    # Panel 3: family heatmap (optional).
    if family_heatmap is not None and family_labels is not None:
        cmap = sns.color_palette("tab10", n_colors=max(3, len(family_labels)))
        sns.heatmap(
            family_heatmap,
            ax=ax3,
            cmap=cmap,
            cbar=True,
            vmin=0,
            vmax=max(len(family_labels) - 1, 1),
            xticklabels=False,
            yticklabels=False,
        )
        ax3.set_title("Edge family selection (coded)")
        ax3.set_xlabel("time")
        ax3.set_ylabel("edge")
    else:
        ax3.axis("off")

    # Panel 4: NLL gaps vs baselines (positive = DVC better).
    ax4.plot(time, series["nll_gap"], color="black", linewidth=2.0, label="Gaussian copula")
    if "nll_gap_truncated_level0" in series:
        ax4.plot(
            time,
            series["nll_gap_truncated_level0"],
            color="#1f77b4",
            linewidth=1.5,
            linestyle="--",
            label="1-truncated C-vine",
        )
    if "nll_gap_glasso" in series:
        ax4.plot(
            time,
            series["nll_gap_glasso"],
            color="#2ca02c",
            linewidth=1.5,
            linestyle=":",
            label="Graphical Lasso",
        )
    if "nll_gap_tvgl" in series:
        ax4.plot(
            time,
            series["nll_gap_tvgl"],
            color="#d62728",
            linewidth=1.5,
            linestyle="-.",
            label="TVGL (Frobenius)",
        )
    if "nll_gap_state_space" in series:
        ax4.plot(
            time,
            series["nll_gap_state_space"],
            color="#9467bd",
            linewidth=1.5,
            linestyle=(0, (3, 1, 1, 1)),
            label="Gaussian SSM",
        )
    if "nll_gap_kde_flow" in series:
        ax4.plot(
            time,
            series["nll_gap_kde_flow"],
            color="#ff7f0e",
            linewidth=1.5,
            linestyle="-",
            label="KDE-flow (time BW)",
        )
    if "nll_gap_regularized_dvc" in series:
        ax4.plot(
            time,
            series["nll_gap_regularized_dvc"],
            color="#8c564b",
            linewidth=1.8,
            linestyle=(0, (5, 2)),
            label="Regularized DVC",
        )
    if "nll_gap_joint_dynamic_dvc" in series:
        ax4.plot(
            time,
            series["nll_gap_joint_dynamic_dvc"],
            color="#000000",
            linewidth=1.9,
            linestyle=(0, (7, 1.5)),
            label="Joint Dynamic DVC",
        )
    if "nll_gap_latent_state_dvc" in series:
        ax4.plot(
            time,
            series["nll_gap_latent_state_dvc"],
            color="#6a3d9a",
            linewidth=1.9,
            linestyle=(0, (3, 1.5)),
            label="Latent-State DVC",
        )
    if "nll_gap_windowed_nonparametric_dvc" in series:
        ax4.plot(
            time,
            series["nll_gap_windowed_nonparametric_dvc"],
            color="#17becf",
            linewidth=1.7,
            linestyle=(0, (2, 1)),
            label="Windowed NP-DVC",
        )
    if "nll_gap_joint_nonparametric_dvc" in series:
        ax4.plot(
            time,
            series["nll_gap_joint_nonparametric_dvc"],
            color="#bc5090",
            linewidth=1.8,
            linestyle=(0, (6, 2)),
            label="Joint NP-DVC",
        )
    ax4.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax4.set_title("Held-out copula NLL gap (positive = DVC better)")
    ax4.set_xlabel("time")
    ax4.set_ylabel("NLL(baseline) $-$ NLL(DVC)")
    ax4.legend(frameon=True)

    if change_point is not None:
        for ax in (ax1, ax2, ax4):
            ax.axvline(float(time[change_point]), color="red", linewidth=2, alpha=0.8)

    fig.suptitle(title, y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_hub_switch_panel(
    out_png: Path,
    time: np.ndarray,
    true_hub: List[int],
    est_hub: List[int],
    reg_hub: Optional[List[int]],
    corr_hub: Optional[List[int]],
    glasso_hub: Optional[List[int]],
    tvgl_hub: Optional[List[int]],
    change_point: int,
    title: str,
    nll_gaps: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    _set_seaborn_style()

    n_rows = 2 if nll_gaps else 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(11.2, 3.8 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    # --- Top panel: hub identity over time ---
    ax_hub = axes[0]
    ax_hub.plot(
        time, true_hub, label="true hub", linewidth=2.5,
        color="black", marker="s", markersize=4, markevery=2,
    )
    ax_hub.plot(
        time, est_hub, label="DVC (C-vine root)", linewidth=2.5,
        color="#1f77b4", marker="o", markersize=4, markevery=2,
    )
    if reg_hub is not None:
        ax_hub.plot(
            time, reg_hub, label="Regularized DVC", linewidth=2.2,
            color="#8c564b", linestyle=(0, (5, 2)), marker="P", markersize=3, markevery=2,
        )
    if corr_hub is not None:
        ax_hub.plot(
            time, corr_hub, label="corr hub", linewidth=2.0,
            linestyle="--", color="#d62728", marker="^", markersize=3, markevery=3,
        )
    if glasso_hub is not None:
        ax_hub.plot(
            time, glasso_hub, label="glasso hub", linewidth=2.0,
            linestyle=":", color="#2ca02c", marker="v", markersize=3, markevery=3,
        )
    if tvgl_hub is not None:
        ax_hub.plot(
            time, tvgl_hub, label="tvgl hub", linewidth=2.0,
            linestyle="-.", color="#9467bd", marker="D", markersize=3, markevery=3,
        )
    ax_hub.axvline(float(time[change_point]), color="red", linewidth=2, alpha=0.8)
    ax_hub.set_title("Hub identity over time")
    ax_hub.set_xlabel("time")
    ax_hub.set_ylabel("hub index")

    # Add accuracy annotation.
    acc = float(np.mean(np.asarray(true_hub) == np.asarray(est_hub)))
    acc_parts = [f"DVC: {acc:.3f}"]
    if reg_hub is not None:
        acc_parts.append(f"Reg-DVC: {float(np.mean(np.asarray(true_hub) == np.asarray(reg_hub))):.3f}")
    if corr_hub is not None:
        acc_parts.append(f"Corr: {float(np.mean(np.asarray(true_hub) == np.asarray(corr_hub))):.3f}")
    if glasso_hub is not None:
        acc_parts.append(f"GLasso: {float(np.mean(np.asarray(true_hub) == np.asarray(glasso_hub))):.3f}")
    if tvgl_hub is not None:
        acc_parts.append(f"TVGL: {float(np.mean(np.asarray(true_hub) == np.asarray(tvgl_hub))):.3f}")
    acc_text = "Accuracy  " + " | ".join(acc_parts)
    ax_hub.annotate(
        acc_text, xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )
    ax_hub.legend(frameon=True, loc="lower right")

    # --- Bottom panel: NLL gap subplot (if data provided) ---
    if nll_gaps and n_rows == 2:
        ax_nll = axes[1]
        _nll_styles = [
            ("nll_gap", "Gaussian copula", "black", "-"),
            ("nll_gap_truncated_level0", "1-truncated C-vine", "#1f77b4", "--"),
            ("nll_gap_glasso", "Graphical Lasso", "#2ca02c", ":"),
            ("nll_gap_tvgl", "TVGL (Frobenius)", "#d62728", "-."),
            ("nll_gap_state_space", "Gaussian SSM", "#9467bd", (0, (3, 1, 1, 1))),
            ("nll_gap_regularized_dvc", "Regularized DVC", "#8c564b", (0, (5, 2))),
        ]
        for key, label, color, ls in _nll_styles:
            if key in nll_gaps:
                ax_nll.plot(
                    time, nll_gaps[key], label=label,
                    color=color, linewidth=2.0, linestyle=ls,
                )
        ax_nll.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
        ax_nll.axvline(float(time[change_point]), color="red", linewidth=2, alpha=0.8)
        ax_nll.set_title("Held-out copula NLL gap (positive = DVC better)")
        ax_nll.set_xlabel("time")
        ax_nll.set_ylabel("NLL(baseline) $-$ NLL(DVC)")
        ax_nll.legend(frameon=True)

    fig.suptitle(title, y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_agent_interaction_episodes_panel(
    out_png: Path,
    time: np.ndarray,
    episode_labels: np.ndarray,
    episode_schedule: List[Dict[str, Any]],
    nll_gaps: Dict[str, np.ndarray],
    tc_pairwise: np.ndarray,
    tc_higher_order: np.ndarray,
    corr_matrices: np.ndarray,
    n_agents: int,
    title: str,
    method_detections: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Publication-quality 4x2 panel for agent interaction episodes scenario."""
    _set_seaborn_style()
    from matplotlib.patches import Patch

    has_det = method_detections is not None and len(method_detections) > 0
    n_rows = 4 if has_det else 3
    fig = plt.figure(figsize=(14, 3.5 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 2, height_ratios=[0.6, 1.0, 1.0] + ([1.0] if has_det else []))

    ax_timeline = fig.add_subplot(gs[0, :])  # span both columns
    ax_nll = fig.add_subplot(gs[1, 0])
    ax_tc = fig.add_subplot(gs[1, 1])
    ax_trunc_gap = fig.add_subplot(gs[2, 0])
    ax_corr = fig.add_subplot(gs[2, 1])
    if has_det:
        ax_f1 = fig.add_subplot(gs[3, 0])
        ax_det_timeline = fig.add_subplot(gs[3, 1])

    # Color scheme for episode types.
    _ep_colors = {0: "#cccccc", 1: "#1f77b4", 2: "#d62728", 3: "#9467bd"}
    _ep_names = {0: "Independence", 1: "Pairwise", 2: "Higher-order", 3: "Mixed"}

    # ----- Panel A: Ground-truth episode timeline -----
    for t_idx in range(len(time)):
        label = int(episode_labels[t_idx])
        ax_timeline.axvspan(
            float(time[t_idx]) - 0.5, float(time[t_idx]) + 0.5,
            alpha=0.5, color=_ep_colors[label],
        )
    legend_patches = [
        Patch(facecolor=_ep_colors[k], alpha=0.5, label=_ep_names[k])
        for k in sorted(_ep_colors.keys())
    ]
    ax_timeline.legend(handles=legend_patches, frameon=True, loc="upper right", fontsize=8)
    for ep in episode_schedule:
        t_mid = (ep["t_start"] + ep["t_end"]) / 2.0
        if ep["type"] == "mixed":
            agents_str = f"P:{ep['agents_pairwise']}\nH:{ep['agents_higher']}"
        else:
            agents_str = str(ep["agents"])
        ax_timeline.text(
            t_mid, 0.5, agents_str, ha="center", va="center",
            fontsize=7, transform=ax_timeline.get_xaxis_transform(),
        )
    ax_timeline.set_xlim(float(time[0]) - 0.5, float(time[-1]) + 0.5)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("time step")
    ax_timeline.set_title("(A) Ground truth: interaction episodes")

    # ----- Panel B: NLL gap vs baselines -----
    _nll_styles = [
        ("nll_gap", "Gaussian copula", "black", "-"),
        ("nll_gap_glasso", "Graphical Lasso", "#2ca02c", ":"),
        ("nll_gap_tvgl", "TVGL (Frobenius)", "#d62728", "-."),
        ("nll_gap_state_space", "Gaussian SSM", "#9467bd", (0, (3, 1, 1, 1))),
        ("nll_gap_regularized_dvc", "Regularized DVC", "#8c564b", (0, (5, 2))),
    ]
    for key, label, color, ls in _nll_styles:
        if key in nll_gaps:
            ax_nll.plot(time, nll_gaps[key], label=label, color=color, linewidth=2.0, linestyle=ls)
    ax_nll.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    for ep in episode_schedule:
        c = _ep_colors.get({"pairwise": 1, "higher_order": 2, "mixed": 3}[ep["type"]], "#ccc")
        ax_nll.axvspan(ep["t_start"] - 0.5, ep["t_end"] - 0.5, alpha=0.12, color=c)
    ax_nll.set_title("(B) NLL gap vs. baselines (positive = DVC better)")
    ax_nll.set_xlabel("time step")
    ax_nll.set_ylabel("NLL(baseline) $-$ NLL(DVC)")
    ax_nll.legend(frameon=True, fontsize=8)

    # ----- Panel C: TC decomposition (stacked area) -----
    tc_p = np.clip(tc_pairwise, 0, None)
    tc_h = np.clip(tc_higher_order, 0, None)
    ax_tc.fill_between(time, 0, tc_p, alpha=0.5, color="#1f77b4", label=r"$\mathrm{TC}_{\mathrm{pair}}$")
    ax_tc.fill_between(time, tc_p, tc_p + tc_h, alpha=0.5, color="#d62728", label=r"$\mathrm{TC}_{\mathrm{higher}}$")
    ax_tc.plot(time, tc_p + tc_h, color="black", linewidth=1.0, alpha=0.5)
    for ep in episode_schedule:
        c = _ep_colors.get({"pairwise": 1, "higher_order": 2, "mixed": 3}[ep["type"]], "#ccc")
        ax_tc.axvspan(ep["t_start"] - 0.5, ep["t_end"] - 0.5, alpha=0.08, color=c)
    ax_tc.set_title(r"(C) Total correlation decomposition")
    ax_tc.set_xlabel("time step")
    ax_tc.set_ylabel("TC (nats)")
    ax_tc.legend(frameon=True, fontsize=8)

    # ----- Panel D: DVC vs 1-truncated vine gap (higher-order indicator) -----
    if "nll_gap_truncated_level0" in nll_gaps:
        ax_trunc_gap.bar(
            time, nll_gaps["nll_gap_truncated_level0"],
            width=0.8, color="#d62728", alpha=0.6, label="Full vine $-$ 1-truncated",
        )
    ax_trunc_gap.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    for ep in episode_schedule:
        c = _ep_colors.get({"pairwise": 1, "higher_order": 2, "mixed": 3}[ep["type"]], "#ccc")
        ax_trunc_gap.axvspan(ep["t_start"] - 0.5, ep["t_end"] - 0.5, alpha=0.12, color=c)
    ax_trunc_gap.set_title(r"(D) Higher-order signal: NLL(1-trunc) $-$ NLL(DVC)")
    ax_trunc_gap.set_xlabel("time step")
    ax_trunc_gap.set_ylabel("NLL gap (nats)")
    ax_trunc_gap.legend(frameon=True, fontsize=8)

    # ----- Panel E: Pairwise correlation heatmap -----
    pair_labels_list: List[str] = []
    pair_corrs_list: List[np.ndarray] = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            pair_labels_list.append(f"{i}-{j}")
            pair_corrs_list.append(np.array([np.abs(corr_matrices[t, i, j]) for t in range(len(time))]))
    pair_mat = np.stack(pair_corrs_list, axis=0)
    sns.heatmap(
        pair_mat,
        ax=ax_corr,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.7,
        cbar_kws={"label": r"$|\rho|$"},
        xticklabels=5,
        yticklabels=pair_labels_list,
    )
    ax_corr.set_title("(E) Pairwise absolute correlation")
    ax_corr.set_xlabel("time step")
    ax_corr.set_ylabel("agent pair")

    # ----- Panels F & G: Detection comparison (if method_detections provided) -----
    if has_det:
        # Panel F: F1 score comparison (horizontal bar chart).
        method_names = list(method_detections.keys())
        f1_scores = [method_detections[m]["f1"] for m in method_names]
        prec_scores = [method_detections[m]["precision"] for m in method_names]
        rec_scores = [method_detections[m]["recall"] for m in method_names]

        y_pos = np.arange(len(method_names))
        bar_h = 0.25
        ax_f1.barh(y_pos - bar_h, prec_scores, bar_h, label="Precision", color="#1f77b4", alpha=0.7)
        ax_f1.barh(y_pos, rec_scores, bar_h, label="Recall", color="#ff7f0e", alpha=0.7)
        ax_f1.barh(y_pos + bar_h, f1_scores, bar_h, label="F1", color="#2ca02c", alpha=0.7)
        ax_f1.set_yticks(y_pos)
        ax_f1.set_yticklabels(method_names, fontsize=8)
        ax_f1.set_xlim(0, 1.05)
        ax_f1.set_xlabel("Score")
        ax_f1.set_title("(F) Episode detection: binary (interaction vs independence)")
        ax_f1.legend(frameon=True, fontsize=8, loc="lower right")
        # Add F1 value annotations.
        for i, v in enumerate(f1_scores):
            ax_f1.text(v + 0.01, i + bar_h, f"{v:.2f}", va="center", fontsize=7)

        # Panel G: Per-method detection timeline (raster).
        det_matrix = []
        det_names = []
        for m in method_names:
            det_matrix.append(method_detections[m]["detected"])
            det_names.append(m)
        det_arr = np.array(det_matrix, dtype=np.float32)  # (n_methods, T)
        # Show detected (1) vs not (0) as colored cells.
        cmap_det = sns.color_palette(["#f0f0f0", "#2ca02c"], as_cmap=True)
        sns.heatmap(
            det_arr,
            ax=ax_det_timeline,
            cmap=cmap_det,
            vmin=0, vmax=1,
            cbar=False,
            xticklabels=5,
            yticklabels=det_names,
            linewidths=0.3, linecolor="white",
        )
        # Overlay ground truth as top row border.
        gt_binary = (episode_labels > 0).astype(np.float32)
        for t_idx in range(len(time)):
            if gt_binary[t_idx] > 0:
                ax_det_timeline.axvspan(t_idx, t_idx + 1, ymin=1.0, ymax=1.05,
                                        color="#d62728", clip_on=False)
        ax_det_timeline.set_title("(G) Detection timeline (green = interaction detected)")
        ax_det_timeline.set_xlabel("time step")
        ax_det_timeline.tick_params(axis='y', labelsize=7)

    fig.suptitle(title, y=1.02, fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    payload: Dict[str, Any]


def run_simulation_benchmark_suite(
    output_dir: Path,
    seed: int,
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the simulation suite and write figures under output_dir/plots/."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "experiment_type": "simulation_benchmarks",
        "seed": int(seed),
        "scenarios": {},
    }

    for sc in scenarios:
        name = str(sc.get("name", "")).strip()
        if not name:
            raise ValueError("Each scenario entry must have a non-empty 'name'")

        if name == "multiplicative_triplet":
            n_samples = int(sc.get("n_samples", 4000))
            noise_std = float(sc.get("noise_std", 0.25))
            gen = generate_multiplicative_triplet(n_samples=n_samples, noise_std=noise_std, seed=seed)
            x = gen["data"]

            corr_yz = float(np.corrcoef(x[:, 1], x[:, 2])[0, 1])
            corr_yz_pos = float(np.corrcoef(x[x[:, 0] > 0, 1], x[x[:, 0] > 0, 2])[0, 1])
            corr_yz_neg = float(np.corrcoef(x[x[:, 0] < 0, 1], x[x[:, 0] < 0, 2])[0, 1])

            families = ["independence", "gaussian", "student", "joe", "frank"]
            # Train/test split for likelihood-based comparisons.
            n = x.shape[0]
            n_train = int(0.8 * n)
            idx = np.random.default_rng(seed).permutation(n)
            tr = x[idx[:n_train]]
            te = x[idx[n_train:]]

            vine = _fit_parametric_vine(tr, families=families, optimize_structure=False, seed=seed)
            dvc_nll = _mean_copula_nll(vine, te)
            gauss_nll = _gaussian_copula_nll_fit_eval(tr, te)
            trunc_vine = _fit_truncated_cvine_level0(tr, families=families, order=[0, 1, 2])
            trunc_nll = _mean_copula_nll(trunc_vine, te)

            # Extract the single tree-2 edge (conditional copula) for d=3.
            cond_family = None
            cond_theta = None
            if len(vine.copulas) >= 2 and len(vine.copulas[1]) >= 1:
                cond_family = vine.copulas[1][0].family
                cond_theta = vine.copulas[1][0].theta

            # Pairwise-only fit for (Y,Z) as a baseline (no conditioning).
            u_tr_yz = _pseudo_obs_from_gaussianized(tr[:, [1, 2]])
            u_te_yz = _pseudo_obs_from_gaussianized(te[:, [1, 2]])
            pair_cop = _fit_best_bivariate_copula(u_tr_yz, families=families)
            pair_nll_yz = _mean_bivariate_copula_nll(pair_cop, u_te_yz)

            out_png = plots_dir / "multiplicative_triplet_panel.png"
            _plot_multiplicative_triplet(
                out_png, x, title="Beyond pairwise: multiplicative interaction (Gaussianized)"
            )

            results["scenarios"][name] = {
                "n_samples": n_samples,
                "noise_std": noise_std,
                "corr_yz": corr_yz,
                "corr_yz_xpos": corr_yz_pos,
                "corr_yz_xneg": corr_yz_neg,
                "dvc_nll": float(dvc_nll),
                "gaussian_nll": float(gauss_nll),
                "nll_gap": float(gauss_nll - dvc_nll),
                "truncated_level0_nll": float(trunc_nll),
                "nll_gap_truncated_level0": float(trunc_nll - dvc_nll),
                "conditional_edge_family": cond_family,
                "conditional_edge_theta": cond_theta,
                "pairwise_yz_best_family": str(pair_cop.family),
                "pairwise_yz_best_theta": pair_cop.theta,
                "pairwise_yz_nll": float(pair_nll_yz),
            }
            continue

        if name == "higher_order_only_switch":
            cfg = {
                "n_time_steps": int(sc.get("n_time_steps", 8)),
                "n_samples_per_time": int(sc.get("n_samples_per_time", 3000)),
            }
            gen = generate_higher_order_only_switch(seed=seed, **cfg)
            time_data = gen["time_data"]
            time = gen["time_indices"]
            change_point = int(gen["change_point"])
            regime_labels = np.asarray(gen["regime_labels"], dtype=np.int32)
            regime_schedule = gen["regime_schedule"]

            families = ["independence", "gaussian", "frank"]

            dvc_nll: List[float] = []
            gauss_nll: List[float] = []
            trunc_nll: List[float] = []
            glasso_nll: List[float] = []
            pairwise_abs_corr: List[float] = []
            x_train_list: List[np.ndarray] = []
            x_test_list: List[np.ndarray] = []

            for t in range(time_data.shape[0]):
                x_t = time_data[t]
                n = x_t.shape[0]
                n_train = int(0.8 * n)
                idx = np.random.default_rng(seed + 19 * t).permutation(n)
                tr = x_t[idx[:n_train]]
                te = x_t[idx[n_train:]]

                vine = _fit_parametric_vine(tr, families=families, optimize_structure=False, seed=seed + 19 * t)
                dvc_nll.append(_mean_copula_nll(vine, te))
                trunc_vine = _fit_truncated_cvine_level0(tr, families=families, order=[0, 1, 2])
                trunc_nll.append(_mean_copula_nll(trunc_vine, te))
                gauss_nll.append(_gaussian_copula_nll_fit_eval(tr, te))
                glasso_nll.append(_glasso_gaussian_copula_nll_fit_eval(tr, te, alpha=0.02))

                R = np.corrcoef(x_t, rowvar=False)
                tri = np.triu_indices_from(R, k=1)
                pairwise_abs_corr.append(float(np.mean(np.abs(R[tri]))))

                x_train_list.append(tr)
                x_test_list.append(te)

            tvgl_nll = _tvgl_gaussian_copula_nll_fit_eval(
                x_train_list,
                x_test_list,
                alpha=0.02,
                beta=1.0,
                max_iter=200,
                step_size=0.05,
                eps=1e-4,
            )
            ssm_nll, ssm_fit = gaussian_copula_state_space_nll_fit_eval(
                x_train_list,
                x_test_list,
            )

            _dvc_arr = np.asarray(dvc_nll, dtype=np.float64)
            _trunc_arr = np.asarray(trunc_nll, dtype=np.float64)
            _gauss_arr = np.asarray(gauss_nll, dtype=np.float64)
            _glasso_arr = np.asarray(glasso_nll, dtype=np.float64)
            _tvgl_arr = np.asarray(tvgl_nll, dtype=np.float64)
            _ssm_arr = np.asarray(ssm_nll, dtype=np.float64)
            tc_higher_order = _trunc_arr - _dvc_arr
            pairwise_abs_corr_arr = np.asarray(pairwise_abs_corr, dtype=np.float64)

            indep_mask = regime_labels == 0
            higher_mask = regime_labels == 1
            tc_higher_ind_mean = float(np.mean(tc_higher_order[indep_mask])) if np.any(indep_mask) else float("nan")
            tc_higher_higher_mean = float(np.mean(tc_higher_order[higher_mask])) if np.any(higher_mask) else float("nan")
            higher_order_regime_contrast = float(tc_higher_higher_mean - tc_higher_ind_mean)
            pairwise_abs_corr_ind_mean = float(np.mean(pairwise_abs_corr_arr[indep_mask])) if np.any(indep_mask) else float("nan")
            pairwise_abs_corr_higher_mean = float(np.mean(pairwise_abs_corr_arr[higher_mask])) if np.any(higher_mask) else float("nan")

            if np.any(indep_mask):
                thresh_higher = float(np.mean(tc_higher_order[indep_mask]) + 2.0 * max(np.std(tc_higher_order[indep_mask]), 0.02))
            else:
                thresh_higher = 0.05
            higher_detected = (tc_higher_order > thresh_higher).astype(np.int32)
            higher_detect_acc = float(np.mean(higher_detected == regime_labels))
            if np.any(higher_detected > 0):
                change_point_hat = int(np.argmax(higher_detected > 0))
            else:
                change_point_hat = int(np.argmax(tc_higher_order))
            change_point_abs_error = int(abs(change_point_hat - change_point))

            try:
                from sklearn.metrics import average_precision_score, roc_auc_score
            except Exception:
                average_precision_score = None
                roc_auc_score = None
            higher_auroc = None
            higher_avg_prec = None
            if np.unique(regime_labels).size >= 2 and roc_auc_score is not None and average_precision_score is not None:
                try:
                    higher_auroc = float(roc_auc_score(regime_labels, tc_higher_order))
                except Exception:
                    higher_auroc = None
                try:
                    higher_avg_prec = float(average_precision_score(regime_labels, tc_higher_order))
                except Exception:
                    higher_avg_prec = None

            nll_gaps = {
                "nll_gap": (_gauss_arr - _dvc_arr),
                "nll_gap_truncated_level0": (_trunc_arr - _dvc_arr),
                "nll_gap_glasso": (_glasso_arr - _dvc_arr),
                "nll_gap_tvgl": (_tvgl_arr - _dvc_arr),
                "nll_gap_state_space": (_ssm_arr - _dvc_arr),
            }

            out_png = plots_dir / "higher_order_only_switch_panel.png"
            _plot_higher_order_only_switch_panel(
                out_png,
                time=time,
                pairwise_abs_corr=pairwise_abs_corr_arr,
                tc_higher_order=tc_higher_order,
                detection_threshold=thresh_higher,
                nll_gaps=nll_gaps,
                change_point=change_point,
                title="Higher-order-only switch with matched pairwise marginals",
            )

            results["scenarios"][name] = {
                **cfg,
                "change_point": change_point,
                "change_point_hat_higher_order": change_point_hat,
                "change_point_abs_error_higher_order": change_point_abs_error,
                "regime_labels": regime_labels.tolist(),
                "regime_schedule": regime_schedule,
                "pairwise_abs_corr_by_time": pairwise_abs_corr_arr.tolist(),
                "pairwise_abs_corr_independence_mean": pairwise_abs_corr_ind_mean,
                "pairwise_abs_corr_higher_order_mean": pairwise_abs_corr_higher_mean,
                "pairwise_abs_corr_shift": float(abs(pairwise_abs_corr_higher_mean - pairwise_abs_corr_ind_mean)),
                "higher_order_detection_threshold": thresh_higher,
                "higher_order_detection_accuracy": higher_detect_acc,
                "higher_order_detection_auroc": higher_auroc,
                "higher_order_detection_average_precision": higher_avg_prec,
                "higher_order_detected": higher_detected.tolist(),
                "tc_higher_independence_mean": tc_higher_ind_mean,
                "tc_higher_higher_order_mean": tc_higher_higher_mean,
                "higher_order_regime_contrast": higher_order_regime_contrast,
                "tc_higher_order": tc_higher_order.tolist(),
                "dvc_nll": dvc_nll,
                "gaussian_nll": gauss_nll,
                "truncated_level0_nll": trunc_nll,
                "glasso_gaussian_nll": glasso_nll,
                "tvgl_gaussian_nll": tvgl_nll,
                "state_space_gaussian_nll": ssm_nll,
                "nll_gap": nll_gaps["nll_gap"].tolist(),
                "nll_gap_truncated_level0": nll_gaps["nll_gap_truncated_level0"].tolist(),
                "nll_gap_glasso": nll_gaps["nll_gap_glasso"].tolist(),
                "nll_gap_tvgl": nll_gaps["nll_gap_tvgl"].tolist(),
                "nll_gap_state_space": nll_gaps["nll_gap_state_space"].tolist(),
                "state_space_process_variance": float(ssm_fit.process_variance),
            }
            continue

        if name == "dynamic_tail_df":
            cfg = {
                "n_time_steps": int(sc.get("n_time_steps", 30)),
                "n_samples_per_time": int(sc.get("n_samples_per_time", 250)),
                "n_variables": int(sc.get("n_variables", 5)),
                "rho": float(sc.get("rho", 0.6)),
                "nu_low": float(sc.get("nu_low", 3.0)),
                "nu_high": float(sc.get("nu_high", 30.0)),
                "schedule": str(sc.get("schedule", "piecewise")),
            }
            gen = generate_dynamic_tail_df(seed=seed, **cfg)
            time_data = gen["time_data"]
            time = gen["time_indices"]
            cp = gen["change_point"]
            tail_true = np.asarray(
                [_taildep_student(float(cfg["rho"]), float(nu_t)) for nu_t in gen["nu_schedule"]],
                dtype=np.float32,
            )

            families = ["gaussian", "student", "clayton", "gumbel", "joe", "independence"]
            smooth_dynamic_families = ["gaussian", "student", "independence"]
            dvc_nll = []
            reg_dvc_nll: List[float] = []
            joint_dvc_nll: List[float] = []
            latent_dvc_nll: List[float] = []
            windowed_np_dvc_nll: List[float] = []
            joint_np_dvc_nll: List[float] = []
            gauss_nll = []
            trunc_nll = []
            glasso_nll = []
            tail_emp = []
            tail_fit = []
            corr_mean = []
            tau_mean = []
            df_hat = []
            x_train_list: List[np.ndarray] = []
            x_test_list: List[np.ndarray] = []

            for t in range(time_data.shape[0]):
                x_t = time_data[t]
                # Train/test split
                n = x_t.shape[0]
                n_train = int(0.8 * n)
                idx = np.random.default_rng(seed + t).permutation(n)
                tr = x_t[idx[:n_train]]
                te = x_t[idx[n_train:]]

                vine = _fit_parametric_vine(tr, families=families, optimize_structure=False, seed=seed + t)
                dvc_nll.append(_mean_copula_nll(vine, te))
                gauss_nll.append(_gaussian_copula_nll_fit_eval(tr, te))
                trunc_vine = _fit_truncated_cvine_level0(tr, families=families)
                trunc_nll.append(_mean_copula_nll(trunc_vine, te))
                glasso_nll.append(_glasso_gaussian_copula_nll_fit_eval(tr, te, alpha=0.02))
                x_train_list.append(tr)
                x_test_list.append(te)

                # Second-order summaries: mean abs corr/tau for level-0 edges.
                root = 0
                edges0 = [(root, j) for j in range(1, cfg["n_variables"])]
                corrs = []
                taus = []
                tails = []
                for (i, j) in edges0:
                    corrs.append(abs(float(np.corrcoef(te[:, i], te[:, j])[0, 1])))

                    # Kendall tau on pseudo-observations (robust summary).
                    ui = norm.cdf(te[:, i])
                    uj = norm.cdf(te[:, j])
                    tau_ij = float(pd.Series(ui).corr(pd.Series(uj), method="kendall"))
                    taus.append(abs(tau_ij))

                    u_pair = np.stack([ui, uj], axis=1)
                    tails.append(_empirical_tail_dependence(u_pair, q=0.95, tail="upper"))
                corr_mean.append(float(np.mean(corrs)))
                tau_mean.append(float(np.mean(taus)))
                tail_emp.append(float(np.mean(tails)))

                # Fitted tail dep from level-0 copulas.
                tails_fit = []
                nus = []
                if vine.copulas and len(vine.copulas[0]) == len(edges0):
                    for cop in vine.copulas[0]:
                        lam_l, lam_u = _taildeps_from_copula(cop)
                        tails_fit.append(lam_u)
                        if str(cop.family).lower().strip() == "student":
                            try:
                                nus.append(float(cop.theta[1]))
                            except Exception:
                                pass
                tail_fit.append(float(np.mean(tails_fit)) if tails_fit else 0.0)
                df_hat.append(float(np.mean(nus)) if nus else float("nan"))

            tvgl_nll = _tvgl_gaussian_copula_nll_fit_eval(
                x_train_list,
                x_test_list,
                alpha=0.02,
                beta=1.0,
                max_iter=200,
                step_size=0.05,
                eps=1e-4,
            )
            ssm_nll, ssm_fit = gaussian_copula_state_space_nll_fit_eval(
                x_train_list,
                x_test_list,
            )
            reg_result, reg_dvc_nll = _fit_regularized_dynamic_cvine_from_splits(
                x_train_list,
                x_test_list,
                families=smooth_dynamic_families,
                root_switch_penalty=0.10,
                family_switch_penalty=0.15,
                parameter_drift_penalty=0.15,
                parameter_smoothing=0.30,
                root_score_method="kendall_tau",
            )
            joint_result, joint_dvc_nll = _fit_joint_dynamic_cvine_from_splits(
                x_train_list,
                x_test_list,
                families=smooth_dynamic_families,
                n_basis=2,
                smoothness_penalty=0.50,
                ridge_penalty=1e-3,
                maxiter=20,
            )
            latent_result, latent_dvc_nll = _fit_latent_state_dynamic_cvine_from_splits(
                x_train_list,
                x_test_list,
                families=smooth_dynamic_families,
                order=list(joint_result.order),
                selection_n_basis=2,
                selection_smoothness_penalty=0.50,
                latent_dim=1,
                transition_penalty=5e-2,
                n_epochs=25,
                lr=2e-2,
            )
            windowed_np_result, windowed_np_dvc_nll = _fit_windowed_nonparametric_cvine_from_splits(
                x_train_list,
                x_test_list,
                order=list(joint_result.order),
                knots=7,
                npc_dict={
                    "opt_method": "LL1",
                    "max_iter_phase1": 1,
                    "max_iter_phase2": 1,
                    "normal_iters_phase1": 5,
                    "normal_iters_phase2": 5,
                    "final_normalization_iters": 50,
                    "batch_size": 1,
                },
                temporal_smoothing_bandwidth=0.12,
            )
            joint_np_result, joint_np_dvc_nll = _fit_joint_dynamic_nonparametric_cvine_from_splits(
                x_train_list,
                x_test_list,
                order=list(joint_result.order),
                knots=7,
                trajectory_type="basis",
                trajectory_kwargs={"n_basis": 2},
                n_epochs=3,
                lr=5e-2,
                smoothness_penalty=5e-3,
                batch_size=1,
                normalization_iters=5,
                final_normalization_iters=50,
                density_smoothing_bandwidth=0.12,
            )
            kde_flow_nll, kde_flow_val_nll, kde_flow_bandwidths = _kde_flow_truncated_level0_nll_from_splits(
                x_train_list,
                x_test_list,
                seed=seed + 1701,
                root=0,
                val_fraction=0.2,
                n_epochs=120,
                lr=1e-2,
                hidden_dim=32,
                batch_time_steps=8,
                device="auto",
            )

            series = {
                "corr_mean_abs": np.asarray(corr_mean, dtype=np.float32),
                "tau_mean_abs": np.asarray(tau_mean, dtype=np.float32),
                "tail_emp_upper_q95": np.asarray(tail_emp, dtype=np.float32),
                "tail_fit_upper": np.asarray(tail_fit, dtype=np.float32),
                "tail_true_upper": tail_true,
                "tail_true_lower": tail_true,
                "nll_gap": np.asarray(gauss_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_truncated_level0": np.asarray(trunc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_glasso": np.asarray(glasso_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_tvgl": np.asarray(tvgl_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_state_space": np.asarray(ssm_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_kde_flow": np.asarray(kde_flow_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_regularized_dvc": np.asarray(reg_dvc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_joint_dynamic_dvc": np.asarray(joint_dvc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_latent_state_dvc": np.asarray(latent_dvc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_windowed_nonparametric_dvc": np.asarray(windowed_np_dvc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_joint_nonparametric_dvc": np.asarray(joint_np_dvc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
            }
            out_png = plots_dir / "dynamic_tail_df_panel.png"
            _plot_dynamic_panel(
                out_png,
                time=time,
                series=series,
                change_point=cp,
                title="Dynamic tail dependence at stable second-order summaries (Student-t)",
            )

            cp_hat = int(np.argmax(np.abs(np.diff(np.asarray(tail_emp, dtype=np.float64))))) + 1 if len(tail_emp) > 1 else None
            cp_err = None if (cp is None or cp_hat is None) else int(abs(int(cp_hat) - int(cp)))

            results["scenarios"][name] = {
                **cfg,
                "change_point": cp,
                "change_point_hat_tail_emp": cp_hat,
                "change_point_abs_error_tail_emp": cp_err,
                "nu_schedule": gen["nu_schedule"].tolist(),
                "tail_true_upper": tail_true.tolist(),
                "tail_true_lower": tail_true.tolist(),
                "smooth_dynamic_family_candidates": list(smooth_dynamic_families),
                "dvc_nll": dvc_nll,
                "regularized_dvc_nll": reg_dvc_nll,
                "joint_dynamic_dvc_nll": joint_dvc_nll,
                "latent_state_dvc_nll": latent_dvc_nll,
                "windowed_nonparametric_dvc_nll": windowed_np_dvc_nll,
                "joint_nonparametric_dvc_nll": joint_np_dvc_nll,
                "gaussian_nll": gauss_nll,
                "nll_gap": (np.asarray(gauss_nll) - np.asarray(dvc_nll)).tolist(),
                "truncated_level0_nll": trunc_nll,
                "nll_gap_truncated_level0": (np.asarray(trunc_nll) - np.asarray(dvc_nll)).tolist(),
                "glasso_gaussian_nll": glasso_nll,
                "nll_gap_glasso": (np.asarray(glasso_nll) - np.asarray(dvc_nll)).tolist(),
                "tvgl_gaussian_nll": tvgl_nll,
                "nll_gap_tvgl": (np.asarray(tvgl_nll) - np.asarray(dvc_nll)).tolist(),
                "state_space_gaussian_nll": ssm_nll,
                "nll_gap_state_space": (np.asarray(ssm_nll) - np.asarray(dvc_nll)).tolist(),
                "state_space_process_variance": float(ssm_fit.process_variance),
                "kde_flow_truncated_level0_nll": kde_flow_nll,
                "kde_flow_val_nll": kde_flow_val_nll,
                "kde_flow_bandwidths": kde_flow_bandwidths.tolist(),
                "nll_gap_kde_flow": (np.asarray(kde_flow_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_regularized_dvc": (np.asarray(reg_dvc_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_joint_dynamic_dvc": (np.asarray(joint_dvc_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_latent_state_dvc": (np.asarray(latent_dvc_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_windowed_nonparametric_dvc": (np.asarray(windowed_np_dvc_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_joint_nonparametric_dvc": (np.asarray(joint_np_dvc_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_improvement_regularized_over_dvc": (np.asarray(dvc_nll) - np.asarray(reg_dvc_nll)).tolist(),
                "nll_improvement_joint_over_dvc": (np.asarray(dvc_nll) - np.asarray(joint_dvc_nll)).tolist(),
                "nll_improvement_latent_over_dvc": (np.asarray(dvc_nll) - np.asarray(latent_dvc_nll)).tolist(),
                "joint_dynamic_order": list(joint_result.order),
                "joint_dynamic_selected_families": [str(edge_fit.family) for edge_fit in joint_result.edge_fits],
                "latent_state_order": list(latent_result.order),
                "latent_state_selected_families": [str(edge_fit.family) for edge_fit in latent_result.edge_fits],
                "windowed_nonparametric_order": list(windowed_np_result.order),
                "joint_nonparametric_order": list(joint_np_result.order),
                "windowed_nonparametric_config": dict(windowed_np_result.config),
                "joint_nonparametric_config": dict(joint_np_result.config),
                "regularized_family_switch_count": int(reg_result.total_family_switches()),
                "regularized_parameter_drift_total": float(reg_result.total_parameter_drift()),
                "corr_mean_abs": corr_mean,
                "tau_mean_abs": tau_mean,
                "tail_emp_upper_q95": tail_emp,
                "tail_fit_upper": tail_fit,
                "df_hat_mean_level0": df_hat,
            }
            continue

        if name == "tail_switch":
            cfg = {
                "n_time_steps": int(sc.get("n_time_steps", 30)),
                "n_samples_per_time": int(sc.get("n_samples_per_time", 250)),
                "n_variables": int(sc.get("n_variables", 5)),
                "kendall_tau": float(sc.get("kendall_tau", 0.4)),
            }
            gen = generate_tail_switch_clayton_gumbel(seed=seed, **cfg)
            time_data = gen["time_data"]
            time = gen["time_indices"]
            cp = int(gen["change_point"])
            theta_clayton = float(gen["theta_clayton"])
            theta_gumbel = float(gen["theta_gumbel"])
            true_tail_lower = np.asarray(
                [2.0 ** (-1.0 / theta_clayton) if fam == "clayton" else 0.0 for fam in gen["family_schedule"]],
                dtype=np.float32,
            )
            true_tail_upper = np.asarray(
                [0.0 if fam == "clayton" else 2.0 - 2.0 ** (1.0 / theta_gumbel) for fam in gen["family_schedule"]],
                dtype=np.float32,
            )

            families = ["gaussian", "clayton", "gumbel", "joe", "independence"]
            dvc_nll = []
            gauss_nll = []
            trunc_nll = []
            glasso_nll = []
            tau_mean = []
            tail_u = []
            tail_l = []
            fam_codes = []
            fam_labels = ["ind", "gaussian", "clayton", "gumbel", "student", "frank", "joe"]
            fam_to_code = {k: i for i, k in enumerate(fam_labels)}
            x_train_list: List[np.ndarray] = []
            x_test_list: List[np.ndarray] = []

            for t in range(time_data.shape[0]):
                x_t = time_data[t]
                n = x_t.shape[0]
                n_train = int(0.8 * n)
                idx = np.random.default_rng(seed + 11 * t).permutation(n)
                tr = x_t[idx[:n_train]]
                te = x_t[idx[n_train:]]

                vine = _fit_parametric_vine(tr, families=families, optimize_structure=False, seed=seed + 11 * t)
                dvc_nll.append(_mean_copula_nll(vine, te))
                gauss_nll.append(_gaussian_copula_nll_fit_eval(tr, te))
                trunc_vine = _fit_truncated_cvine_level0(tr, families=families)
                trunc_nll.append(_mean_copula_nll(trunc_vine, te))
                glasso_nll.append(_glasso_gaussian_copula_nll_fit_eval(tr, te, alpha=0.02))
                x_train_list.append(tr)
                x_test_list.append(te)

                # Track Kendall tau stability and tail asymmetry for level-0 edges.
                edges0 = [(0, j) for j in range(1, cfg["n_variables"])]
                taus = []
                tailu = []
                taill = []
                for i, j in edges0:
                    ui = norm.cdf(te[:, i])
                    uj = norm.cdf(te[:, j])
                    tau_ij = float(pd.Series(ui).corr(pd.Series(uj), method="kendall"))
                    taus.append(abs(tau_ij))
                    u_pair = np.stack([ui, uj], axis=1)
                    tailu.append(_empirical_tail_dependence(u_pair, q=0.95, tail="upper"))
                    taill.append(_empirical_tail_dependence(u_pair, q=0.05, tail="lower"))
                tau_mean.append(float(np.mean(taus)))
                tail_u.append(float(np.mean(tailu)))
                tail_l.append(float(np.mean(taill)))

                # Family codes for level-0 edges (for heatmap).
                codes_t = []
                if vine.copulas and vine.copulas[0]:
                    for cop in vine.copulas[0]:
                        fam = str(cop.family).lower().strip()
                        fam = "ind" if fam in {"independence", "independent"} else fam
                        codes_t.append(fam_to_code.get(fam, 0))
                fam_codes.append(codes_t)

            tvgl_nll = _tvgl_gaussian_copula_nll_fit_eval(
                x_train_list,
                x_test_list,
                alpha=0.02,
                beta=1.0,
                max_iter=200,
                step_size=0.05,
                eps=1e-4,
            )
            ssm_nll, ssm_fit = gaussian_copula_state_space_nll_fit_eval(
                x_train_list,
                x_test_list,
            )
            kde_flow_nll, kde_flow_val_nll, kde_flow_bandwidths = _kde_flow_truncated_level0_nll_from_splits(
                x_train_list,
                x_test_list,
                seed=seed + 1709,
                root=0,
                val_fraction=0.2,
                n_epochs=120,
                lr=1e-2,
                hidden_dim=32,
                batch_time_steps=8,
                device="auto",
            )

            truth_codes = np.asarray(
                [fam_to_code.get(str(fam).lower().strip(), 0) for fam in gen["family_schedule"]],
                dtype=np.int64,
            )
            heat = np.asarray(fam_codes, dtype=np.int64).T if fam_codes else None
            if heat is not None:
                heat = np.vstack([truth_codes.reshape(1, -1), heat])
            series = {
                "corr_tau_mean_abs": np.asarray(tau_mean, dtype=np.float32),
                "tail_upper_q95": np.asarray(tail_u, dtype=np.float32),
                "tail_lower_q05": np.asarray(tail_l, dtype=np.float32),
                "tail_true_upper": true_tail_upper,
                "tail_true_lower": true_tail_lower,
                "nll_gap": np.asarray(gauss_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_truncated_level0": np.asarray(trunc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_glasso": np.asarray(glasso_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_tvgl": np.asarray(tvgl_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_state_space": np.asarray(ssm_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_kde_flow": np.asarray(kde_flow_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
            }
            out_png = plots_dir / "tail_switch_panel.png"
            _plot_dynamic_panel(
                out_png,
                time=time,
                series=series,
                change_point=cp,
                title="Matched Kendall tau, switching tail asymmetry (Clayton ↔ Gumbel)",
                family_heatmap=heat,
                family_labels=fam_labels,
            )

            delta_tail = np.asarray(tail_u, dtype=np.float64) - np.asarray(tail_l, dtype=np.float64)
            cp_hat = int(np.argmax(np.abs(np.diff(delta_tail)))) + 1 if len(delta_tail) > 1 else None
            cp_err = None if cp_hat is None else int(abs(int(cp_hat) - int(cp)))

            results["scenarios"][name] = {
                **cfg,
                "change_point": cp,
                "change_point_hat_tail_asym": cp_hat,
                "change_point_abs_error_tail_asym": cp_err,
                "family_schedule": gen["family_schedule"],
                "tail_true_upper": true_tail_upper.tolist(),
                "tail_true_lower": true_tail_lower.tolist(),
                "level0_family_truth_codes": truth_codes.tolist(),
                "dvc_nll": dvc_nll,
                "gaussian_nll": gauss_nll,
                "nll_gap": (np.asarray(gauss_nll) - np.asarray(dvc_nll)).tolist(),
                "truncated_level0_nll": trunc_nll,
                "nll_gap_truncated_level0": (np.asarray(trunc_nll) - np.asarray(dvc_nll)).tolist(),
                "glasso_gaussian_nll": glasso_nll,
                "nll_gap_glasso": (np.asarray(glasso_nll) - np.asarray(dvc_nll)).tolist(),
                "tvgl_gaussian_nll": tvgl_nll,
                "nll_gap_tvgl": (np.asarray(tvgl_nll) - np.asarray(dvc_nll)).tolist(),
                "state_space_gaussian_nll": ssm_nll,
                "nll_gap_state_space": (np.asarray(ssm_nll) - np.asarray(dvc_nll)).tolist(),
                "state_space_process_variance": float(ssm_fit.process_variance),
                "kde_flow_truncated_level0_nll": kde_flow_nll,
                "kde_flow_val_nll": kde_flow_val_nll,
                "kde_flow_bandwidths": kde_flow_bandwidths.tolist(),
                "nll_gap_kde_flow": (np.asarray(kde_flow_nll) - np.asarray(dvc_nll)).tolist(),
                "tau_mean_abs": tau_mean,
                "tail_upper_q95": tail_u,
                "tail_lower_q05": tail_l,
                "level0_family_codes": fam_codes,
                "family_codebook": fam_labels,
            }
            continue

        if name == "hub_switch":
            cfg = {
                "n_time_steps": int(sc.get("n_time_steps", 30)),
                "n_samples_per_time": int(sc.get("n_samples_per_time", 250)),
                "n_variables": int(sc.get("n_variables", 8)),
                "hub_a": int(sc.get("hub_a", 0)),
                "hub_b": int(sc.get("hub_b", 1)),
                "rho_hub": float(sc.get("rho_hub", 0.7)),
            }
            gen = generate_hub_switch(seed=seed, **cfg)
            time_data = gen["time_data"]
            time = gen["time_indices"]
            cp = int(gen["change_point"])
            true_hubs = gen["true_hubs"]

            est_hubs: List[int] = []
            corr_hubs: List[int] = []
            glasso_hubs: List[int] = []
            tvgl_hubs: List[int] = []

            families = ["gaussian", "independence"]
            dvc_nll: List[float] = []
            reg_dvc_nll: List[float] = []
            gauss_nll: List[float] = []
            trunc_nll: List[float] = []
            glasso_nll: List[float] = []
            x_train_list: List[np.ndarray] = []
            x_test_list: List[np.ndarray] = []

            for t in range(time_data.shape[0]):
                x_t = time_data[t]
                n = x_t.shape[0]
                n_train = int(0.8 * n)
                idx = np.random.default_rng(seed + 101 * t).permutation(n)
                tr = x_t[idx[:n_train]]
                te = x_t[idx[n_train:]]

                # DVC: optimize structure + fit Gaussian/independence families, then evaluate NLL.
                vine = _fit_parametric_vine(tr, families=families, optimize_structure=True, seed=seed + 101 * t)
                dvc_nll.append(_mean_copula_nll(vine, te))

                # Baselines: Gaussian copula, truncated vine, Graphical Lasso.
                gauss_nll.append(_gaussian_copula_nll_fit_eval(tr, te))
                order = getattr(vine, "variable_order", None) or list(range(cfg["n_variables"]))
                trunc_vine = _fit_truncated_cvine_level0(tr, families=families, order=order)
                trunc_nll.append(_mean_copula_nll(trunc_vine, te))
                glasso_nll.append(_glasso_gaussian_copula_nll_fit_eval(tr, te, alpha=0.02))

                # Hub estimates from different methods.
                dvc_root = int(order[0]) if order else int(vine.ind_vine[0][0][0])
                est_hubs.append(dvc_root)

                corr_hub = _estimate_hub_by_correlation(tr)
                corr_hubs.append(corr_hub)
                try:
                    glasso_hubs.append(_estimate_hub_by_glasso(tr))
                except Exception:
                    glasso_hubs.append(corr_hub)

                x_train_list.append(tr)
                x_test_list.append(te)

            reg_result, reg_dvc_nll = _fit_regularized_dynamic_cvine_from_splits(
                x_train_list,
                x_test_list,
                families=families,
                root_switch_penalty=0.25,
                family_switch_penalty=0.0,
                parameter_drift_penalty=0.0,
                parameter_smoothing=0.25,
                root_score_method="kendall_tau",
            )
            reg_est_hubs = [int(v) for v in reg_result.root_sequence]

            # TVGL baseline (fit across time).
            covs = []
            for xtr in x_train_list:
                ztr = _normal_scores_from_rank_pobs(np.asarray(xtr, dtype=np.float64))
                C = np.cov(ztr, rowvar=False)
                C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                C = 0.5 * (C + C.T) + 1e-6 * np.eye(C.shape[0])
                covs.append(C)

            tvgl = tvgl_frobenius(
                covs,
                alpha=0.02,
                beta=1.0,
                max_iter=200,
                step_size=0.05,
                eps=1e-4,
                verbose=False,
            )
            tvgl_nll: List[float] = []
            for P, xte in zip(tvgl.precision, x_test_list):
                cov = np.linalg.inv(P)
                R = _corr_from_cov(cov)
                zte = _normal_scores_from_rank_pobs(np.asarray(xte, dtype=np.float64))
                tvgl_nll.append(_gaussian_copula_nll_given_corr(zte, R))
                tvgl_hubs.append(hub_from_precision(P, edge_threshold=0.05))
            ssm_nll, ssm_fit = gaussian_copula_state_space_nll_fit_eval(
                x_train_list,
                x_test_list,
            )

            _dvc_arr = np.asarray(dvc_nll)
            hub_nll_gaps: Dict[str, np.ndarray] = {
                "nll_gap": np.asarray(gauss_nll) - _dvc_arr,
                "nll_gap_truncated_level0": np.asarray(trunc_nll) - _dvc_arr,
                "nll_gap_glasso": np.asarray(glasso_nll) - _dvc_arr,
                "nll_gap_tvgl": np.asarray(tvgl_nll) - _dvc_arr,
                "nll_gap_state_space": np.asarray(ssm_nll) - _dvc_arr,
                "nll_gap_regularized_dvc": np.asarray(reg_dvc_nll) - _dvc_arr,
            }

            out_png = plots_dir / "hub_switch_panel.png"
            _plot_hub_switch_panel(
                out_png,
                time=time,
                true_hub=true_hubs,
                est_hub=est_hubs,
                reg_hub=reg_est_hubs,
                corr_hub=corr_hubs,
                glasso_hub=glasso_hubs,
                tvgl_hub=tvgl_hubs,
                change_point=cp,
                title="Hub switching (C-vine root) via structure optimization",
                nll_gaps=hub_nll_gaps,
            )

            acc = float(np.mean(np.asarray(true_hubs) == np.asarray(est_hubs)))
            acc_reg = float(np.mean(np.asarray(true_hubs) == np.asarray(reg_est_hubs)))
            acc_corr = float(np.mean(np.asarray(true_hubs) == np.asarray(corr_hubs)))
            acc_glasso = float(np.mean(np.asarray(true_hubs) == np.asarray(glasso_hubs)))
            acc_tvgl = float(np.mean(np.asarray(true_hubs) == np.asarray(tvgl_hubs)))

            def _first_change(seq: List[int]) -> Optional[int]:
                if not seq:
                    return None
                for k in range(1, len(seq)):
                    if seq[k] != seq[k - 1]:
                        return k
                return None

            cp_hat_dvc = _first_change(est_hubs)
            cp_hat_reg = _first_change(reg_est_hubs)
            cp_hat_corr = _first_change(corr_hubs)
            cp_hat_glasso = _first_change(glasso_hubs)
            cp_hat_tvgl = _first_change(tvgl_hubs)

            results["scenarios"][name] = {
                **cfg,
                "change_point": cp,
                "change_point_hat_dvc": cp_hat_dvc,
                "change_point_abs_error_dvc": None if cp_hat_dvc is None else int(abs(int(cp_hat_dvc) - int(cp))),
                "change_point_hat_regularized_dvc": cp_hat_reg,
                "change_point_abs_error_regularized_dvc": None if cp_hat_reg is None else int(abs(int(cp_hat_reg) - int(cp))),
                "change_point_hat_corr": cp_hat_corr,
                "change_point_abs_error_corr": None if cp_hat_corr is None else int(abs(int(cp_hat_corr) - int(cp))),
                "change_point_hat_glasso": cp_hat_glasso,
                "change_point_abs_error_glasso": None if cp_hat_glasso is None else int(abs(int(cp_hat_glasso) - int(cp))),
                "change_point_hat_tvgl": cp_hat_tvgl,
                "change_point_abs_error_tvgl": None if cp_hat_tvgl is None else int(abs(int(cp_hat_tvgl) - int(cp))),
                "true_hubs": true_hubs,
                "estimated_hubs": est_hubs,
                "root_recovery_accuracy": acc,
                "regularized_estimated_hubs": reg_est_hubs,
                "regularized_root_recovery_accuracy": acc_reg,
                "corr_hub_estimated_hubs": corr_hubs,
                "corr_hub_recovery_accuracy": acc_corr,
                "glasso_hub_estimated_hubs": glasso_hubs,
                "glasso_hub_recovery_accuracy": acc_glasso,
                "tvgl_hub_estimated_hubs": tvgl_hubs,
                "tvgl_hub_recovery_accuracy": acc_tvgl,
                "dvc_nll": dvc_nll,
                "regularized_dvc_nll": reg_dvc_nll,
                "gaussian_nll": gauss_nll,
                "truncated_level0_nll": trunc_nll,
                "glasso_gaussian_nll": glasso_nll,
                "tvgl_gaussian_nll": tvgl_nll,
                "state_space_gaussian_nll": ssm_nll,
                "nll_gap": (np.asarray(gauss_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_truncated_level0": (np.asarray(trunc_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_glasso": (np.asarray(glasso_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_tvgl": (np.asarray(tvgl_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_state_space": (np.asarray(ssm_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_gap_regularized_dvc": (np.asarray(reg_dvc_nll) - np.asarray(dvc_nll)).tolist(),
                "nll_improvement_regularized_over_dvc": (np.asarray(dvc_nll) - np.asarray(reg_dvc_nll)).tolist(),
                "regularized_family_switch_count": int(reg_result.total_family_switches()),
                "regularized_parameter_drift_total": float(reg_result.total_parameter_drift()),
                "state_space_process_variance": float(ssm_fit.process_variance),
            }
            continue

        if name == "agent_interaction_episodes":
            cfg = {
                "n_time_steps": int(sc.get("n_time_steps", 28)),
                "n_samples_per_time": int(sc.get("n_samples_per_time", 300)),
                "n_agents": int(sc.get("n_agents", 6)),
                "rho_pairwise": float(sc.get("rho_pairwise", 0.7)),
                "rho_higher": float(sc.get("rho_higher", 0.5)),
                "nu_higher": float(sc.get("nu_higher", 3.0)),
            }
            gen = generate_agent_interaction_episodes(seed=seed, **cfg)
            time_data = gen["time_data"]
            time = gen["time_indices"]
            ep_labels = gen["episode_labels"]
            ep_agents = gen["episode_agents"]
            ep_schedule = gen["episode_schedule"]

            families = ["gaussian", "student", "clayton", "gumbel", "joe", "independence"]

            dvc_nll: List[float] = []
            reg_dvc_nll: List[float] = []
            gauss_nll: List[float] = []
            trunc_nll: List[float] = []
            glasso_nll: List[float] = []
            corr_matrices: List[np.ndarray] = []
            x_train_list: List[np.ndarray] = []
            x_test_list: List[np.ndarray] = []

            for t in range(time_data.shape[0]):
                x_t = time_data[t]
                n = x_t.shape[0]
                n_train = int(0.8 * n)
                idx = np.random.default_rng(seed + 13 * t).permutation(n)
                tr = x_t[idx[:n_train]]
                te = x_t[idx[n_train:]]

                vine = _fit_parametric_vine(tr, families=families, optimize_structure=False, seed=seed + 13 * t)
                dvc_nll.append(_mean_copula_nll(vine, te))
                gauss_nll.append(_gaussian_copula_nll_fit_eval(tr, te))
                trunc_vine = _fit_truncated_cvine_level0(tr, families=families)
                trunc_nll.append(_mean_copula_nll(trunc_vine, te))
                glasso_nll.append(_glasso_gaussian_copula_nll_fit_eval(tr, te, alpha=0.02))

                R = np.corrcoef(te, rowvar=False)
                corr_matrices.append(R)

                x_train_list.append(tr)
                x_test_list.append(te)

            reg_result, reg_dvc_nll = _fit_regularized_dynamic_cvine_from_splits(
                x_train_list,
                x_test_list,
                families=families,
                root_switch_penalty=0.10,
                family_switch_penalty=0.20,
                parameter_drift_penalty=0.10,
                parameter_smoothing=0.25,
                root_score_method="aic",
            )

            # Cross-time baselines: TVGL.
            covs = []
            for xtr in x_train_list:
                ztr = _normal_scores_from_rank_pobs(np.asarray(xtr, dtype=np.float64))
                C = np.cov(ztr, rowvar=False)
                C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                C = 0.5 * (C + C.T) + 1e-6 * np.eye(C.shape[0])
                covs.append(C)

            tvgl = tvgl_frobenius(
                covs,
                alpha=0.02,
                beta=1.0,
                max_iter=200,
                step_size=0.05,
                eps=1e-4,
                verbose=False,
            )
            tvgl_nll: List[float] = []
            for P, xte in zip(tvgl.precision, x_test_list):
                cov = np.linalg.inv(P)
                R = _corr_from_cov(cov)
                zte = _normal_scores_from_rank_pobs(np.asarray(xte, dtype=np.float64))
                tvgl_nll.append(_gaussian_copula_nll_given_corr(zte, R))

            # Cross-time baselines: Gaussian state-space.
            ssm_nll, ssm_fit = gaussian_copula_state_space_nll_fit_eval(
                x_train_list,
                x_test_list,
            )

            # NLL gaps.
            _dvc_arr = np.asarray(dvc_nll)
            ep_nll_gaps: Dict[str, np.ndarray] = {
                "nll_gap": np.asarray(gauss_nll) - _dvc_arr,
                "nll_gap_truncated_level0": np.asarray(trunc_nll) - _dvc_arr,
                "nll_gap_glasso": np.asarray(glasso_nll) - _dvc_arr,
                "nll_gap_tvgl": np.asarray(tvgl_nll) - _dvc_arr,
                "nll_gap_state_space": np.asarray(ssm_nll) - _dvc_arr,
                "nll_gap_regularized_dvc": np.asarray(reg_dvc_nll) - _dvc_arr,
            }

            # TC decomposition relative to the independence copula.  Since
            # copula NLL is the negative log-density contribution, the
            # 1-truncated vine gives the pairwise contribution and the
            # full-vs-truncated gap gives the higher-tree residual.
            tc_pairwise = -np.asarray(trunc_nll)
            tc_higher_order = np.asarray(trunc_nll) - _dvc_arr
            reg_tc_higher_order = np.asarray(trunc_nll) - np.asarray(reg_dvc_nll)
            tc_pairwise_flexible_over_gaussian = np.asarray(gauss_nll) - np.asarray(trunc_nll)

            # --- Episode detection across all methods ---
            # For each method, compute per-time NLL and detect episodes via
            # thresholding the NLL drop relative to a simple independence baseline.
            # DVC uniquely can also distinguish pairwise from higher-order via
            # the truncated-vine gap.

            # Independence-baseline NLL: fit a Gaussian copula with identity
            # correlation (= sum of marginal NLLs, i.e. 0 for standard normals).
            # Since data is Gaussianized, the independence NLL ≈ 0.
            # We use each method's own NLL trajectory and detect departures.
            indep_mask = ep_labels == 0

            def _detect_episodes_binary(nll_trajectory: np.ndarray) -> np.ndarray:
                """Detect non-independence episodes from an NLL trajectory.

                Lower NLL = better fit = more dependence detected.
                Returns binary array: 0=independence, 1=interaction detected.
                """
                arr = np.asarray(nll_trajectory, dtype=np.float64)
                if np.any(indep_mask):
                    indep_vals = arr[indep_mask]
                    # Interaction detected when NLL drops substantially below
                    # the independence-period level (more negative = better fit).
                    thresh = float(np.mean(indep_vals) - 2.0 * max(np.std(indep_vals), 0.01))
                else:
                    thresh = float(np.median(arr))
                return (arr < thresh).astype(np.int32)

            # Binary detection (0=indep, 1=any interaction) for each method.
            gt_binary = (ep_labels > 0).astype(np.int32)
            method_detections: Dict[str, Dict[str, Any]] = {}
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score
            except Exception:
                average_precision_score = None
                roc_auc_score = None

            for mname, nll_arr in [
                ("DVC", dvc_nll),
                ("Regularized DVC", reg_dvc_nll),
                ("Gaussian copula", gauss_nll),
                ("1-truncated C-vine", trunc_nll),
                ("Graphical Lasso", glasso_nll),
                ("TVGL", tvgl_nll),
                ("Gaussian SSM", ssm_nll),
            ]:
                det = _detect_episodes_binary(np.asarray(nll_arr))
                acc_bin = float(np.mean(det == gt_binary))
                # Precision/recall for interaction class.
                tp = int(np.sum((det == 1) & (gt_binary == 1)))
                fp = int(np.sum((det == 1) & (gt_binary == 0)))
                fn = int(np.sum((det == 0) & (gt_binary == 1)))
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)
                # Threshold-free ranking metrics avoid tuning a detector on the same sequence.
                score = -np.asarray(nll_arr, dtype=np.float64)
                auroc = None
                avg_prec = None
                if np.unique(gt_binary).size >= 2 and roc_auc_score is not None and average_precision_score is not None:
                    try:
                        auroc = float(roc_auc_score(gt_binary, score))
                    except Exception:
                        auroc = None
                    try:
                        avg_prec = float(average_precision_score(gt_binary, score))
                    except Exception:
                        avg_prec = None
                method_detections[mname] = {
                    "detected": det.tolist(),
                    "accuracy": acc_bin,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auroc": auroc,
                    "average_precision": avg_prec,
                }

            # DVC 3-class detection (independence / pairwise / higher-order).
            # Use total-correlation evidence to detect *any* interaction: a
            # Gaussian-compatible pairwise episode should not disappear simply
            # because the Gaussian copula is also a strong fit. Then use the
            # full-vs-1-truncated gap to decide whether the signal is higher
            # tree rather than level-0 pairwise.
            total_tc = -_dvc_arr
            reg_total_tc = -np.asarray(reg_dvc_nll, dtype=np.float64)
            if np.any(indep_mask):
                indep_tc = total_tc[indep_mask]
                thresh_tc = float(np.mean(indep_tc) + 2.0 * max(np.std(indep_tc), 0.01))
                indep_trunc = tc_higher_order[indep_mask]
                thresh_higher = float(np.mean(indep_trunc) + 2.0 * max(np.std(indep_trunc), 0.005))
            else:
                thresh_tc = 0.02
                thresh_higher = 0.01

            detected = np.zeros(len(time), dtype=np.int32)
            for t_idx in range(len(time)):
                if total_tc[t_idx] < thresh_tc:
                    detected[t_idx] = 0  # independence
                elif tc_higher_order[t_idx] < thresh_higher:
                    detected[t_idx] = 1  # pairwise
                else:
                    detected[t_idx] = 2  # higher-order (or mixed)

            # Map mixed (label=3) to higher-order (2) for accuracy comparison.
            gt_collapsed = np.where(ep_labels == 3, 2, ep_labels)
            ep_detect_acc = float(np.mean(detected == gt_collapsed))
            reg_detected = np.zeros(len(time), dtype=np.int32)
            if np.any(indep_mask):
                reg_indep_tc = reg_total_tc[indep_mask]
                reg_thresh_tc = float(np.mean(reg_indep_tc) + 2.0 * max(np.std(reg_indep_tc), 0.01))
                reg_indep_trunc = reg_tc_higher_order[indep_mask]
                reg_thresh_higher = float(np.mean(reg_indep_trunc) + 2.0 * max(np.std(reg_indep_trunc), 0.005))
            else:
                reg_thresh_tc = 0.02
                reg_thresh_higher = 0.01
            for t_idx in range(len(time)):
                if reg_total_tc[t_idx] < reg_thresh_tc:
                    reg_detected[t_idx] = 0
                elif reg_tc_higher_order[t_idx] < reg_thresh_higher:
                    reg_detected[t_idx] = 1
                else:
                    reg_detected[t_idx] = 2
            reg_ep_detect_acc = float(np.mean(reg_detected == gt_collapsed))
            pair_mask = ep_labels == 1
            higher_mask = ep_labels == 2
            mixed_mask = ep_labels == 3
            tc_higher_pairwise_mean = float(np.mean(tc_higher_order[pair_mask])) if np.any(pair_mask) else float("nan")
            tc_higher_higher_order_mean = float(np.mean(tc_higher_order[higher_mask])) if np.any(higher_mask) else float("nan")
            tc_higher_mixed_mean = float(np.mean(tc_higher_order[mixed_mask])) if np.any(mixed_mask) else float("nan")
            reg_tc_higher_pairwise_mean = float(np.mean(reg_tc_higher_order[pair_mask])) if np.any(pair_mask) else float("nan")
            reg_tc_higher_higher_order_mean = float(np.mean(reg_tc_higher_order[higher_mask])) if np.any(higher_mask) else float("nan")
            reg_tc_higher_mixed_mean = float(np.mean(reg_tc_higher_order[mixed_mask])) if np.any(mixed_mask) else float("nan")

            # Plot.
            corr_mat_arr = np.stack(corr_matrices, axis=0)  # (T, d, d)
            out_png = plots_dir / "agent_interaction_episodes_panel.png"
            _plot_agent_interaction_episodes_panel(
                out_png,
                time=time,
                episode_labels=ep_labels,
                episode_schedule=ep_schedule,
                nll_gaps=ep_nll_gaps,
                tc_pairwise=tc_pairwise,
                tc_higher_order=tc_higher_order,
                corr_matrices=corr_mat_arr,
                n_agents=cfg["n_agents"],
                method_detections=method_detections,
                title="Agent interaction episodes: pairwise vs higher-order detection",
            )

            results["scenarios"][name] = {
                **cfg,
                "episode_schedule": ep_schedule,
                "episode_labels": ep_labels.tolist(),
                "episode_detection_accuracy": ep_detect_acc,
                "order_classification_accuracy": ep_detect_acc,
                "regularized_order_classification_accuracy": reg_ep_detect_acc,
                "method_detection_metrics": method_detections,
                "detection_threshold_nll": thresh_tc,
                "detection_threshold_total_tc": thresh_tc,
                "detection_threshold_higher": thresh_higher,
                "regularized_detection_threshold_nll": reg_thresh_tc,
                "regularized_detection_threshold_total_tc": reg_thresh_tc,
                "regularized_detection_threshold_higher": reg_thresh_higher,
                "tc_higher_pairwise_mean": tc_higher_pairwise_mean,
                "tc_higher_higher_order_mean": tc_higher_higher_order_mean,
                "tc_higher_mixed_mean": tc_higher_mixed_mean,
                "regularized_tc_higher_pairwise_mean": reg_tc_higher_pairwise_mean,
                "regularized_tc_higher_higher_order_mean": reg_tc_higher_higher_order_mean,
                "regularized_tc_higher_mixed_mean": reg_tc_higher_mixed_mean,
                "tc_pairwise": tc_pairwise.tolist(),
                "tc_pairwise_flexible_over_gaussian": tc_pairwise_flexible_over_gaussian.tolist(),
                "tc_higher_order": tc_higher_order.tolist(),
                "dvc_nll": dvc_nll,
                "regularized_dvc_nll": reg_dvc_nll,
                "gaussian_nll": gauss_nll,
                "truncated_level0_nll": trunc_nll,
                "glasso_gaussian_nll": glasso_nll,
                "tvgl_gaussian_nll": tvgl_nll,
                "state_space_gaussian_nll": ssm_nll,
                "nll_gap": (np.asarray(gauss_nll) - _dvc_arr).tolist(),
                "nll_gap_truncated_level0": (np.asarray(trunc_nll) - _dvc_arr).tolist(),
                "nll_gap_glasso": (np.asarray(glasso_nll) - _dvc_arr).tolist(),
                "nll_gap_tvgl": (np.asarray(tvgl_nll) - _dvc_arr).tolist(),
                "nll_gap_state_space": (np.asarray(ssm_nll) - _dvc_arr).tolist(),
                "nll_gap_regularized_dvc": (np.asarray(reg_dvc_nll) - _dvc_arr).tolist(),
                "nll_improvement_regularized_over_dvc": (np.asarray(dvc_nll) - np.asarray(reg_dvc_nll)).tolist(),
                "regularized_tc_higher_order": reg_tc_higher_order.tolist(),
                "regularized_family_switch_count": int(reg_result.total_family_switches()),
                "regularized_parameter_drift_total": float(reg_result.total_parameter_drift()),
                "state_space_process_variance": float(ssm_fit.process_variance),
            }
            continue

        raise ValueError(f"Unknown simulation benchmark scenario: {name}")

    # Minimal master summary for table export.
    summary_rows = []
    for name, payload in results["scenarios"].items():
        row = {"scenario": name}
        if "root_recovery_accuracy" in payload:
            row["root_recovery_accuracy"] = payload["root_recovery_accuracy"]
        if "regularized_root_recovery_accuracy" in payload:
            row["regularized_root_recovery_accuracy"] = payload["regularized_root_recovery_accuracy"]
        if "corr_hub_recovery_accuracy" in payload:
            row["corr_hub_recovery_accuracy"] = payload["corr_hub_recovery_accuracy"]
        if "glasso_hub_recovery_accuracy" in payload:
            row["glasso_hub_recovery_accuracy"] = payload["glasso_hub_recovery_accuracy"]
        if "tvgl_hub_recovery_accuracy" in payload:
            row["tvgl_hub_recovery_accuracy"] = payload["tvgl_hub_recovery_accuracy"]
        if "nll_gap" in payload:
            gap = np.asarray(payload["nll_gap"], dtype=np.float64)
            row["nll_gap_mean"] = float(np.nanmean(gap))
            row["nll_gap_std"] = float(np.nanstd(gap))
        if "nll_gap_truncated_level0" in payload:
            gap = np.asarray(payload["nll_gap_truncated_level0"], dtype=np.float64)
            row["nll_gap_truncated_level0_mean"] = float(np.nanmean(gap))
            row["nll_gap_truncated_level0_std"] = float(np.nanstd(gap))
        if "nll_gap_glasso" in payload:
            gap = np.asarray(payload["nll_gap_glasso"], dtype=np.float64)
            row["nll_gap_glasso_mean"] = float(np.nanmean(gap))
            row["nll_gap_glasso_std"] = float(np.nanstd(gap))
        if "nll_gap_tvgl" in payload:
            gap = np.asarray(payload["nll_gap_tvgl"], dtype=np.float64)
            row["nll_gap_tvgl_mean"] = float(np.nanmean(gap))
            row["nll_gap_tvgl_std"] = float(np.nanstd(gap))
        if "nll_gap_state_space" in payload:
            gap = np.asarray(payload["nll_gap_state_space"], dtype=np.float64)
            row["nll_gap_state_space_mean"] = float(np.nanmean(gap))
            row["nll_gap_state_space_std"] = float(np.nanstd(gap))
        if "nll_gap_kde_flow" in payload:
            gap = np.asarray(payload["nll_gap_kde_flow"], dtype=np.float64)
            row["nll_gap_kde_flow_mean"] = float(np.nanmean(gap))
            row["nll_gap_kde_flow_std"] = float(np.nanstd(gap))
        if "nll_gap_regularized_dvc" in payload:
            gap = np.asarray(payload["nll_gap_regularized_dvc"], dtype=np.float64)
            row["nll_gap_regularized_dvc_mean"] = float(np.nanmean(gap))
            row["nll_gap_regularized_dvc_std"] = float(np.nanstd(gap))
        if "nll_gap_joint_dynamic_dvc" in payload:
            gap = np.asarray(payload["nll_gap_joint_dynamic_dvc"], dtype=np.float64)
            row["nll_gap_joint_dynamic_dvc_mean"] = float(np.nanmean(gap))
            row["nll_gap_joint_dynamic_dvc_std"] = float(np.nanstd(gap))
        if "nll_gap_latent_state_dvc" in payload:
            gap = np.asarray(payload["nll_gap_latent_state_dvc"], dtype=np.float64)
            row["nll_gap_latent_state_dvc_mean"] = float(np.nanmean(gap))
            row["nll_gap_latent_state_dvc_std"] = float(np.nanstd(gap))
        if "nll_gap_windowed_nonparametric_dvc" in payload:
            gap = np.asarray(payload["nll_gap_windowed_nonparametric_dvc"], dtype=np.float64)
            row["nll_gap_windowed_nonparametric_dvc_mean"] = float(np.nanmean(gap))
            row["nll_gap_windowed_nonparametric_dvc_std"] = float(np.nanstd(gap))
        if "nll_gap_joint_nonparametric_dvc" in payload:
            gap = np.asarray(payload["nll_gap_joint_nonparametric_dvc"], dtype=np.float64)
            row["nll_gap_joint_nonparametric_dvc_mean"] = float(np.nanmean(gap))
            row["nll_gap_joint_nonparametric_dvc_std"] = float(np.nanstd(gap))
        if "nll_improvement_regularized_over_dvc" in payload:
            gap = np.asarray(payload["nll_improvement_regularized_over_dvc"], dtype=np.float64)
            row["nll_improvement_regularized_over_dvc_mean"] = float(np.nanmean(gap))
            row["nll_improvement_regularized_over_dvc_std"] = float(np.nanstd(gap))
        if "nll_improvement_joint_over_dvc" in payload:
            gap = np.asarray(payload["nll_improvement_joint_over_dvc"], dtype=np.float64)
            row["nll_improvement_joint_over_dvc_mean"] = float(np.nanmean(gap))
            row["nll_improvement_joint_over_dvc_std"] = float(np.nanstd(gap))
        if "nll_improvement_latent_over_dvc" in payload:
            gap = np.asarray(payload["nll_improvement_latent_over_dvc"], dtype=np.float64)
            row["nll_improvement_latent_over_dvc_mean"] = float(np.nanmean(gap))
            row["nll_improvement_latent_over_dvc_std"] = float(np.nanstd(gap))
        for k in [
            "episode_detection_accuracy",
            "order_classification_accuracy",
            "regularized_order_classification_accuracy",
            "higher_order_detection_accuracy",
            "higher_order_detection_auroc",
            "higher_order_detection_average_precision",
            "pairwise_abs_corr_independence_mean",
            "pairwise_abs_corr_higher_order_mean",
            "pairwise_abs_corr_shift",
            "higher_order_regime_contrast",
            "tc_higher_independence_mean",
            "tc_higher_higher_order_mean",
            "tc_higher_pairwise_mean",
            "tc_higher_higher_order_mean",
            "tc_higher_mixed_mean",
            "regularized_tc_higher_pairwise_mean",
            "regularized_tc_higher_higher_order_mean",
            "regularized_tc_higher_mixed_mean",
        ]:
            if k in payload:
                try:
                    row[k] = float(payload[k])
                except Exception:
                    pass
        method_metrics = payload.get("method_detection_metrics")
        if isinstance(method_metrics, dict):
            for method_name, prefix in [
                ("DVC", "dvc"),
                ("Gaussian copula", "gaussian"),
                ("1-truncated C-vine", "truncated"),
                ("Graphical Lasso", "glasso"),
                ("TVGL", "tvgl"),
                ("Gaussian SSM", "state_space"),
                ("Regularized DVC", "regularized_dvc"),
            ]:
                mp = method_metrics.get(method_name)
                if not isinstance(mp, dict):
                    continue
                for src_key, dst_key in [
                    ("f1", f"{prefix}_binary_f1"),
                    ("auroc", f"{prefix}_binary_auroc"),
                    ("average_precision", f"{prefix}_binary_average_precision"),
                ]:
                    val = mp.get(src_key)
                    if val is not None:
                        try:
                            row[dst_key] = float(val)
                        except Exception:
                            pass

        # Change-point localization diagnostics (optional per scenario).
        for k in [
            "change_point_abs_error_tail_emp",
            "change_point_abs_error_tail_asym",
            "change_point_abs_error_higher_order",
            "change_point_abs_error_dvc",
            "change_point_abs_error_regularized_dvc",
            "change_point_abs_error_corr",
            "change_point_abs_error_glasso",
            "change_point_abs_error_tvgl",
        ]:
            if k in payload:
                row[k] = payload[k]
        summary_rows.append(row)
    results["summary_table"] = summary_rows

    return results
