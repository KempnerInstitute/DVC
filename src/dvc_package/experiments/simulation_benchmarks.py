"""Simulation generators, baselines, and metrics for Dynamic Vine Copulas.

Provides:
- synthetic generators designed to isolate higher-order and non-Gaussian
  dependence,
- minimal baselines (Gaussian copula, truncated C-vine, GLasso, TVGL,
  Gaussian SSM, KDE-flow),
- shared fit/evaluation helpers reused by `dvc_package.experiments.real_world`
  and the benchmark test suite.

The multi-panel diagnostic figure orchestration that consumes these primitives
lives outside the public package, under
`drafts/projects/paper_benchmarks/run_suite.py`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, t as student_t

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baselines.tvgl import tvgl_frobenius
from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.vine_factory import create_vine
from ..core.param_copula import copulaccdf, copulainvccdf, copulapdf, parametric_fit
from ..core.utils_locallik import loclik_batch_eval
from ..core.vine_model import fit_vine
from ..optimization.structure import optimize_vine_structure
from ..time.joint_dynamic_cvine import JointDynamicCVine, JointDynamicCVineResult
from ..time.switching_dynamic_cvine import SwitchingDynamicCVine, SwitchingDynamicCVineResult
from ..time.regularized_cvine import RegularizedDynamicCVine, RegularizedDynamicCVineResult
from ..time.latent_state_dynamic_cvine import LatentStateDynamicCVine, LatentStateDynamicCVineResult
from ..time.nonparametric_dynamic_cvine import (
    JointDynamicNonparametricCVine,
    JointDynamicNonparametricCVineResult,
    WindowedNonparametricCVine,
    WindowedNonparametricCVineResult,
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


def _level0_upper_tail_from_dynamic_edge_fits(edge_fits: List[Any], n_time_steps: int) -> List[float]:
    """Average upper-tail coefficient from jointly fit level-0 dynamic edges."""
    tails_by_time: List[List[float]] = [[] for _ in range(int(n_time_steps))]
    for edge_fit in edge_fits:
        if int(getattr(edge_fit, "level", -1)) != 0:
            continue
        family = str(getattr(edge_fit, "family", "ind"))
        theta_path = list(getattr(edge_fit, "theta_trajectory", []))
        for t_idx, theta in enumerate(theta_path[:n_time_steps]):
            try:
                _, lam_u = _taildeps_from_copula(cop_par_obj(family, theta))
            except Exception:
                lam_u = 0.0
            tails_by_time[t_idx].append(float(lam_u))
    return [float(np.mean(vals)) if vals else 0.0 for vals in tails_by_time]


def _level0_upper_tail_from_switching_edge_fits(edge_fits: List[Any], n_time_steps: int) -> List[float]:
    """Average upper-tail coefficient from jointly selected level-0 switching edges."""
    tails_by_time: List[List[float]] = [[] for _ in range(int(n_time_steps))]
    for edge_fit in edge_fits:
        if int(getattr(edge_fit, "level", -1)) != 0:
            continue
        family_path = list(getattr(edge_fit, "family_path", []))
        theta_path = list(getattr(edge_fit, "theta_path", []))
        for t_idx, (family, theta) in enumerate(zip(family_path[:n_time_steps], theta_path[:n_time_steps])):
            try:
                _, lam_u = _taildeps_from_copula(cop_par_obj(str(family), theta))
            except Exception:
                lam_u = 0.0
            tails_by_time[t_idx].append(float(lam_u))
    return [float(np.mean(vals)) if vals else 0.0 for vals in tails_by_time]


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

    # Duration weights are scaled to n_time_steps. Short runs (unit tests)
    # keep a compact schedule; longer runs reuse recurrent episodes.
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
    dynamic_student_df: bool = False,
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
        dynamic_student_df=dynamic_student_df,
        maxiter=maxiter,
    )
    result = model.fit(x_train_list)
    test_nll = model.evaluate(x_test_list).tolist()
    return result, test_nll


def _fit_switching_dynamic_cvine_from_splits(
    x_train_list: List[np.ndarray],
    x_test_list: List[np.ndarray],
    families: List[str],
    *,
    order: Optional[List[int]] = None,
    family_switch_penalty: float = 0.08,
    parameter_drift_penalty: float = 0.0,
    activation_penalty: float = 0.0,
) -> Tuple[SwitchingDynamicCVineResult, List[float]]:
    """Fit a jointly selected switching-state dynamic C-vine and evaluate it."""
    if len(x_train_list) != len(x_test_list):
        raise ValueError("x_train_list and x_test_list must have the same length")
    model = SwitchingDynamicCVine(
        families=families,
        order=order,
        family_switch_penalty=family_switch_penalty,
        parameter_drift_penalty=parameter_drift_penalty,
        activation_penalty=activation_penalty,
        normalize_cost=True,
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

