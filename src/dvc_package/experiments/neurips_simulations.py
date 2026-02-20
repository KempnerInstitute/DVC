"""
NeurIPS 2026 simulation suite for Dynamic Vine Copulas (DVC).

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

from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.vine_factory import create_vine
from ..core.param_copula import copulaccdf, copulainvccdf, copulapdf, parametric_fit
from ..core.vine_model import fit_vine
from ..optimization.structure import optimize_vine_structure


def _set_seaborn_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
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


def _mean_copula_nll(vine: vine_obj_bin, x: np.ndarray) -> float:
    """Mean negative log copula density (copula NLL), in nats.

    Important: compute in log-space to avoid overflow when multiplying many edge PDFs.
    Uses rank-based pseudo-observations (pseudo-likelihood), so it does not assume
    a parametric form for the marginals.
    """
    x_t = torch.tensor(x, dtype=torch.float32)
    n, d = x_t.shape
    device = x_t.device

    # Base pseudo-observations via ranks (per-dimension).
    u_ = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    for i in range(d):
        vals = x_t[:, i].contiguous()
        sorted_col = torch.sort(vals)[0]
        ranks = torch.searchsorted(sorted_col, vals).float() + 1.0
        u_[:, 0, i] = (ranks / (n + 1.0)).clamp(1e-6, 1.0 - 1e-6)

    log_cop = torch.zeros(n, dtype=torch.float32, device=device)
    for level in range(d - 1):
        edges_now = vine.ind_vine[level] if level < len(vine.ind_vine) else []
        cops_now = vine.copulas[level] if level < len(vine.copulas) else []
        for e_idx, edge in enumerate(edges_now):
            if e_idx >= len(cops_now):
                continue
            cobj = cops_now[e_idx]
            i, j = int(edge[0]), int(edge[1])
            ui = u_[:, level, i]
            uj = u_[:, level, j]
            uv = torch.stack([ui, uj], dim=1)
            pdf_val = copulapdf(cobj, uv).clamp_min(1e-30)
            log_cop = log_cop + torch.log(pdf_val)
            # forward h-function to build conditional uniforms for higher trees
            u_[:, level + 1, j] = copulaccdf(cobj, uv).clamp(1e-6, 1.0 - 1e-6)

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

    Assumes x has (approximately) standard normal marginals.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n < 5:
        return float("nan")

    # Correlation fit on z-scores (here z == x due to normal marginals).
    R = np.corrcoef(x, rowvar=False)
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
    quad = np.einsum("ni,ij,nj->n", x, A, x)
    log_c = -0.5 * logdet - 0.5 * quad
    return float(-np.mean(log_c))


def _gaussian_copula_fit_corr(x_train: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Fit a stabilized correlation matrix for a Gaussian copula."""
    x_train = np.asarray(x_train, dtype=np.float64)
    n, d = x_train.shape
    if n < 5:
        return np.eye(d, dtype=np.float64)

    R = np.corrcoef(x_train, rowvar=False)
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
    return _gaussian_copula_nll_given_corr(x_test, R)


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

        # Star dependence: strong Gaussian at level 0, independence at higher trees.
        level_fams = ["gaussian"] + ["ind"] * (n_variables - 2)
        level_thetas: List[Any] = [float(rho_hub)] + [None] * (n_variables - 2)
        vine = _make_levelwise_cvine(
            n_variables,
            order=order,
            level_families=level_fams,
            level_thetas=level_thetas,
        )
        data[t] = vine.sample(n_samples_per_time)
        data[t] += 1e-3 * rng.standard_normal(data[t].shape).astype(np.float32)

    return {
        "time_data": data,
        "time_indices": time_idx,
        "change_point": cp,
        "true_hubs": true_hubs,
        "rho_hub": float(rho_hub),
    }


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


def _plot_multiplicative_triplet(
    out_png: Path,
    x: np.ndarray,
    title: str,
) -> None:
    _set_seaborn_style()
    df = pd.DataFrame(x, columns=["X", "Y", "Z"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharex=False, sharey=False)

    axes[0].scatter(df["Y"], df["Z"], s=4, alpha=0.35)
    axes[0].set_title("All samples: Y vs Z")
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("Z")

    mask_pos = df["X"] > 0
    axes[1].scatter(df.loc[mask_pos, "Y"], df.loc[mask_pos, "Z"], s=4, alpha=0.35)
    axes[1].set_title("Conditioned: X > 0")
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("Z")

    mask_neg = df["X"] < 0
    axes[2].scatter(df.loc[mask_neg, "Y"], df.loc[mask_neg, "Z"], s=4, alpha=0.35)
    axes[2].set_title("Conditioned: X < 0")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
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
    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel 1: second-order summaries.
    for k, y in series.items():
        if not k.startswith("corr_"):
            continue
        ax1.plot(time, y, label=k.replace("corr_", "").replace("_", " "))
    ax1.set_title("Pairwise summaries")
    ax1.set_xlabel("time")
    ax1.set_ylabel("mean statistic")
    ax1.legend(frameon=True)

    # Panel 2: tail dependence.
    for k, y in series.items():
        if not k.startswith("tail_"):
            continue
        ax2.plot(time, y, label=k.replace("tail_", "").replace("_", " "))
    ax2.set_title("Tail dependence")
    ax2.set_xlabel("time")
    ax2.set_ylabel("lambda")
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
    ax4.plot(time, series["nll_gap"], color="black", linewidth=1.6, label="Gaussian copula")
    if "nll_gap_truncated_level0" in series:
        ax4.plot(
            time,
            series["nll_gap_truncated_level0"],
            color="tab:blue",
            linewidth=1.4,
            linestyle="--",
            label="1-truncated C-vine",
        )
    ax4.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax4.set_title("Held-out copula NLL gap (positive = DVC better)")
    ax4.set_xlabel("time")
    ax4.set_ylabel("NLL(baseline) - NLL(DVC)")
    ax4.legend(frameon=True)

    if change_point is not None:
        for ax in (ax1, ax2, ax4):
            ax.axvline(float(time[change_point]), color="crimson", linewidth=1.2, alpha=0.8)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _plot_hub_switch_panel(
    out_png: Path,
    time: np.ndarray,
    true_hub: List[int],
    est_hub: List[int],
    corr_hub: Optional[List[int]],
    glasso_hub: Optional[List[int]],
    change_point: int,
    title: str,
) -> None:
    _set_seaborn_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.4))

    axes[0].plot(time, true_hub, label="true hub", linewidth=2.0)
    axes[0].plot(time, est_hub, label="DVC (C-vine root)", linewidth=1.6)
    if corr_hub is not None:
        axes[0].plot(time, corr_hub, label="corr hub", linewidth=1.2, linestyle="--")
    if glasso_hub is not None:
        axes[0].plot(time, glasso_hub, label="glasso hub", linewidth=1.2, linestyle=":")
    axes[0].axvline(float(time[change_point]), color="crimson", linewidth=1.2)
    axes[0].set_title("Hub identity over time")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("hub index")
    axes[0].legend(frameon=True)

    acc = np.mean(np.asarray(true_hub) == np.asarray(est_hub))
    acc_corr = None if corr_hub is None else float(np.mean(np.asarray(true_hub) == np.asarray(corr_hub)))
    acc_glasso = None if glasso_hub is None else float(np.mean(np.asarray(true_hub) == np.asarray(glasso_hub)))
    axes[1].axis("off")
    extra = ""
    if acc_corr is not None:
        extra += f"\nCorr hub accuracy: {acc_corr:.3f}"
    if acc_glasso is not None:
        extra += f"\nGlasso hub accuracy: {acc_glasso:.3f}"
    axes[1].text(
        0.0,
        0.8,
        f"DVC root accuracy: {acc:.3f}{extra}\nChange point: t={int(change_point)}",
        fontsize=11,
    )

    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    payload: Dict[str, Any]


def run_neurips_simulation_suite(
    output_dir: Path,
    seed: int,
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the simulation suite and write figures under output_dir/plots/."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "experiment_type": "neurips_simulations",
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

            families = ["independence", "gaussian", "student"]
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

            families = ["gaussian", "student", "independence"]
            dvc_nll = []
            gauss_nll = []
            trunc_nll = []
            tail_emp = []
            tail_fit = []
            corr_mean = []
            tau_mean = []
            df_hat = []

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

            series = {
                "corr_mean_abs": np.asarray(corr_mean, dtype=np.float32),
                "tau_mean_abs": np.asarray(tau_mean, dtype=np.float32),
                "tail_emp_upper_q95": np.asarray(tail_emp, dtype=np.float32),
                "tail_fit_upper": np.asarray(tail_fit, dtype=np.float32),
                "nll_gap": np.asarray(gauss_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_truncated_level0": np.asarray(trunc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
            }
            out_png = plots_dir / "dynamic_tail_df_panel.png"
            _plot_dynamic_panel(
                out_png,
                time=time,
                series=series,
                change_point=cp,
                title="Dynamic tail dependence at stable second-order summaries (Student-t)",
            )

            results["scenarios"][name] = {
                **cfg,
                "change_point": cp,
                "nu_schedule": gen["nu_schedule"].tolist(),
                "dvc_nll": dvc_nll,
                "gaussian_nll": gauss_nll,
                "nll_gap": (np.asarray(gauss_nll) - np.asarray(dvc_nll)).tolist(),
                "truncated_level0_nll": trunc_nll,
                "nll_gap_truncated_level0": (np.asarray(trunc_nll) - np.asarray(dvc_nll)).tolist(),
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

            families = ["gaussian", "clayton", "gumbel", "independence"]
            dvc_nll = []
            gauss_nll = []
            trunc_nll = []
            tau_mean = []
            tail_u = []
            tail_l = []
            fam_codes = []
            fam_labels = ["ind", "gaussian", "clayton", "gumbel", "student", "frank"]
            fam_to_code = {k: i for i, k in enumerate(fam_labels)}

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

            heat = np.asarray(fam_codes, dtype=np.int64).T if fam_codes else None
            series = {
                "corr_tau_mean_abs": np.asarray(tau_mean, dtype=np.float32),
                "tail_upper_q95": np.asarray(tail_u, dtype=np.float32),
                "tail_lower_q05": np.asarray(tail_l, dtype=np.float32),
                "nll_gap": np.asarray(gauss_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
                "nll_gap_truncated_level0": np.asarray(trunc_nll, dtype=np.float32) - np.asarray(dvc_nll, dtype=np.float32),
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

            results["scenarios"][name] = {
                **cfg,
                "change_point": cp,
                "family_schedule": gen["family_schedule"],
                "dvc_nll": dvc_nll,
                "gaussian_nll": gauss_nll,
                "nll_gap": (np.asarray(gauss_nll) - np.asarray(dvc_nll)).tolist(),
                "truncated_level0_nll": trunc_nll,
                "nll_gap_truncated_level0": (np.asarray(trunc_nll) - np.asarray(dvc_nll)).tolist(),
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
            for t in range(time_data.shape[0]):
                x_t = time_data[t]

                corr_hub = _estimate_hub_by_correlation(x_t)
                corr_hubs.append(corr_hub)
                try:
                    glasso_hubs.append(_estimate_hub_by_glasso(x_t))
                except Exception:
                    glasso_hubs.append(corr_hub)

                # Structure only (fast): optimize C-vine order on pseudo-observations.
                opt = optimize_vine_structure(
                    x_t,
                    vine_type="c-vine",
                    method="sequential",
                    criterion="kendall_tau",
                    max_iterations=1,
                    verbose=False,
                )
                vine = opt.best_vine
                order = getattr(vine, "variable_order", None)
                if not order:
                    # Fallback: infer root from first edge list.
                    root = int(vine.ind_vine[0][0][0]) if vine.ind_vine and vine.ind_vine[0] else 0
                else:
                    root = int(order[0])
                est_hubs.append(root)

            out_png = plots_dir / "hub_switch_panel.png"
            _plot_hub_switch_panel(
                out_png,
                time=time,
                true_hub=true_hubs,
                est_hub=est_hubs,
                corr_hub=corr_hubs,
                glasso_hub=glasso_hubs,
                change_point=cp,
                title="Hub switching (C-vine root) via structure optimization",
            )

            acc = float(np.mean(np.asarray(true_hubs) == np.asarray(est_hubs)))
            acc_corr = float(np.mean(np.asarray(true_hubs) == np.asarray(corr_hubs)))
            acc_glasso = float(np.mean(np.asarray(true_hubs) == np.asarray(glasso_hubs)))
            results["scenarios"][name] = {
                **cfg,
                "change_point": cp,
                "true_hubs": true_hubs,
                "estimated_hubs": est_hubs,
                "root_recovery_accuracy": acc,
                "corr_hub_estimated_hubs": corr_hubs,
                "corr_hub_recovery_accuracy": acc_corr,
                "glasso_hub_estimated_hubs": glasso_hubs,
                "glasso_hub_recovery_accuracy": acc_glasso,
            }
            continue

        raise ValueError(f"Unknown NeurIPS simulation scenario: {name}")

    # Minimal master summary for table export.
    summary_rows = []
    for name, payload in results["scenarios"].items():
        row = {"scenario": name}
        if "root_recovery_accuracy" in payload:
            row["root_recovery_accuracy"] = payload["root_recovery_accuracy"]
        if "corr_hub_recovery_accuracy" in payload:
            row["corr_hub_recovery_accuracy"] = payload["corr_hub_recovery_accuracy"]
        if "glasso_hub_recovery_accuracy" in payload:
            row["glasso_hub_recovery_accuracy"] = payload["glasso_hub_recovery_accuracy"]
        if "nll_gap" in payload:
            gap = np.asarray(payload["nll_gap"], dtype=np.float64)
            row["nll_gap_mean"] = float(np.nanmean(gap))
            row["nll_gap_std"] = float(np.nanstd(gap))
        if "nll_gap_truncated_level0" in payload:
            gap = np.asarray(payload["nll_gap_truncated_level0"], dtype=np.float64)
            row["nll_gap_truncated_level0_mean"] = float(np.nanmean(gap))
            row["nll_gap_truncated_level0_std"] = float(np.nanstd(gap))
        summary_rows.append(row)
    results["summary_table"] = summary_rows

    return results
