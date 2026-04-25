#!/usr/bin/env python3
"""Shared utilities for the final standalone showcase benchmark scripts."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from scipy.stats import kendalltau, norm

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-showcase-utils-"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.baselines.gaussian_state_space import (  # noqa: E402
    gaussian_copula_state_space_nll_fit_eval,
)
from dvc_package.baselines.nf_copula import nf_copula_nll_fit_eval  # noqa: E402
from dvc_package.core.objects import cop_par_obj  # noqa: E402
from dvc_package.core.param_copula import copulaccdf, copulainvccdf  # noqa: E402
from dvc_package.experiments.simulation_benchmarks import (  # noqa: E402
    _embed_higher_order_vine,
    _fit_parametric_vine,
    _fit_regularized_dynamic_cvine_from_splits,
    _fit_truncated_cvine_level0,
    _gaussian_copula_nll_fit_eval,
    _mean_copula_nll,
    gaussianize_columns,
)
from scripts.run_showcase_benchmark import (  # noqa: E402
    D,
    FAMILIES,
    N_PER_TIME,
    PHASE_BOUNDARIES,
    PHASES,
    T,
    TRAIN_FRAC,
    _gen_independent,
    _gen_pairwise_star_block,
    _gen_tail_block,
)


DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "showcase_ale_final"


@dataclass
class ShowcaseConfig:
    d: int = D
    t: int = T
    n_per_time: int = N_PER_TIME
    train_frac: float = TRAIN_FRAC
    phase_boundaries: tuple[int, ...] = field(default_factory=lambda: tuple(PHASE_BOUNDARIES))
    phases: tuple[str, ...] = field(default_factory=lambda: tuple(PHASES))
    pair_root: int = 0
    pair_leaves: tuple[int, ...] = (1, 2, 3, 4)
    pair_rho: float = 0.7
    phase3_mode: str = "fixed_triplet"
    triplet_blocks: tuple[tuple[int, ...], ...] = ((0, 5, 6),)
    triplet_rho: float = 0.65
    triplet_nu: float = 4.5
    triplet_clayton_theta: float = 2.0
    multiplicative_noise_std: float = 0.15
    xor_jitter_std: float = 1e-3
    tail_block: tuple[int, ...] = (0, 1, 2, 3)
    tail_theta: float = 1.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_seed_list(n_seeds: int = 5, base_seed: int = 2026, stride: int = 97) -> List[int]:
    return [int(base_seed + stride * i) for i in range(n_seeds)]


def phase_index_of_window(t_idx: int, config: ShowcaseConfig) -> int:
    for phase_idx in range(len(config.phase_boundaries) - 1):
        lo = config.phase_boundaries[phase_idx]
        hi = config.phase_boundaries[phase_idx + 1]
        if lo <= t_idx < hi:
            return phase_idx
    return len(config.phases) - 1


def representative_windows(config: ShowcaseConfig) -> Dict[str, int]:
    starts = config.phase_boundaries[:-1]
    return {
        phase_name: int(starts[phase_idx])
        for phase_idx, phase_name in enumerate(config.phases)
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def gaussian_mi_from_tau(tau: float) -> float:
    if not np.isfinite(tau):
        return float("nan")
    tau = float(np.clip(tau, -0.999, 0.999))
    rho = float(np.clip(np.sin(np.pi * tau / 2.0), -0.999, 0.999))
    return float(-0.5 * np.log(max(1e-12, 1.0 - rho * rho)))


def matrix_to_csv(path: Path, matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savetxt(path, np.asarray(matrix, dtype=np.float64), delimiter=",", fmt="%.6f")


def pairwise_stat_matrices(x: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    tau = np.eye(d, dtype=np.float64)
    pearson = np.corrcoef(x, rowvar=False)
    pearson = np.nan_to_num(pearson, nan=0.0, posinf=0.0, neginf=0.0)
    pearson = 0.5 * (pearson + pearson.T)
    np.fill_diagonal(pearson, 1.0)
    mi_proxy = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1, d):
            tau_ij = kendalltau(x[:, i], x[:, j]).statistic
            tau_ij = 0.0 if tau_ij is None or not np.isfinite(tau_ij) else float(tau_ij)
            tau[i, j] = tau_ij
            tau[j, i] = tau_ij
            mi = gaussian_mi_from_tau(tau_ij)
            mi_proxy[i, j] = mi
            mi_proxy[j, i] = mi
    return {
        "kendall_tau": tau,
        "pearson": pearson,
        "gaussian_mi_proxy": mi_proxy,
    }


def top_edges(matrix: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    d = matrix.shape[0]
    edges: List[Dict[str, Any]] = []
    for i in range(d):
        for j in range(i + 1, d):
            value = float(matrix[i, j])
            edges.append(
                {
                    "pair": [int(i), int(j)],
                    "value": value,
                    "abs_value": abs(value),
                }
            )
    edges.sort(key=lambda row: row["abs_value"], reverse=True)
    return edges[:top_k]


def selected_pair_rows(x: np.ndarray, pairs: Sequence[tuple[int, int]]) -> List[Dict[str, Any]]:
    stats = pairwise_stat_matrices(x)
    tau = stats["kendall_tau"]
    pearson = stats["pearson"]
    mi_proxy = stats["gaussian_mi_proxy"]
    rows: List[Dict[str, Any]] = []
    for i, j in pairs:
        rows.append(
            {
                "pair": [int(i), int(j)],
                "kendall_tau": float(tau[i, j]),
                "pearson": float(pearson[i, j]),
                "gaussian_mi_proxy": float(mi_proxy[i, j]),
            }
        )
    return rows


def _sample_conditional_triplet_from_root(
    root_scores: np.ndarray,
    *,
    rho: float,
    nu: float,
    clayton_theta: float,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample two leaves from a C-vine conditional on an existing root."""
    u_root = np.clip(norm.cdf(np.asarray(root_scores, dtype=np.float64)), eps, 1.0 - eps)
    n = u_root.shape[0]
    student_cop = cop_par_obj("student", (float(rho), float(nu)))
    cond_cop = cop_par_obj("clayton", float(clayton_theta))

    w_leaf1 = rng.uniform(eps, 1.0 - eps, size=n).astype(np.float32)
    uv_root_leaf1 = torch.tensor(np.column_stack([u_root, w_leaf1]), dtype=torch.float32)
    u_leaf1 = copulainvccdf(student_cop, uv_root_leaf1).detach().cpu().numpy().astype(np.float64)
    u_leaf1 = np.clip(u_leaf1, eps, 1.0 - eps)

    uv_h_root_leaf1 = torch.tensor(np.column_stack([u_root, u_leaf1]), dtype=torch.float32)
    h_leaf1_given_root = (
        copulaccdf(student_cop, uv_h_root_leaf1).detach().cpu().numpy().astype(np.float64)
    )
    h_leaf1_given_root = np.clip(h_leaf1_given_root, eps, 1.0 - eps)

    w_leaf2 = rng.uniform(eps, 1.0 - eps, size=n).astype(np.float32)
    uv_cond = torch.tensor(
        np.column_stack([h_leaf1_given_root, w_leaf2]),
        dtype=torch.float32,
    )
    h_leaf2_given_root = copulainvccdf(cond_cop, uv_cond).detach().cpu().numpy().astype(np.float64)
    h_leaf2_given_root = np.clip(h_leaf2_given_root, eps, 1.0 - eps)

    uv_root_leaf2 = torch.tensor(np.column_stack([u_root, h_leaf2_given_root]), dtype=torch.float32)
    u_leaf2 = copulainvccdf(student_cop, uv_root_leaf2).detach().cpu().numpy().astype(np.float64)
    u_leaf2 = np.clip(u_leaf2, eps, 1.0 - eps)

    return norm.ppf(u_leaf1).astype(np.float32), norm.ppf(u_leaf2).astype(np.float32)


def inject_fixed_triplet_blocks(
    x: np.ndarray,
    *,
    blocks: Sequence[Sequence[int]],
    rho: float,
    nu: float,
    clayton_theta: float,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> np.ndarray:
    """Preserve each block's root column and sample the leaves conditionally."""
    x = np.asarray(x, dtype=np.float32).copy()
    for block in blocks:
        if len(block) != 3:
            raise ValueError(
                "Fixed-semantics triplet injection currently supports triplets of length 3 only."
            )
        root_idx, leaf1_idx, leaf2_idx = [int(idx) for idx in block]
        leaf1, leaf2 = _sample_conditional_triplet_from_root(
            x[:, root_idx],
            rho=rho,
            nu=nu,
            clayton_theta=clayton_theta,
            rng=rng,
            eps=eps,
        )
        x[:, leaf1_idx] = leaf1
        x[:, leaf2_idx] = leaf2
    return x


def inject_overwrite_blocks(
    x: np.ndarray,
    *,
    blocks: Sequence[Sequence[int]],
    rho: float,
    nu: float,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    for block in blocks:
        x = _embed_higher_order_vine(
            x,
            agents=[int(idx) for idx in block],
            rho=rho,
            nu=nu,
            rng=rng,
            eps=eps,
        )
    return x


def _sample_modsum_higher_order_triplet(
    n_samples: int,
    rng: np.random.Generator,
    eps: float = 1e-6,
) -> np.ndarray:
    """Continuous XOR-style triplet with near-zero pairwise marginals."""
    uv = rng.uniform(eps, 1.0 - eps, size=(n_samples, 2)).astype(np.float64)
    w = np.mod(uv[:, 0] + uv[:, 1], 1.0)
    w = np.clip(w, eps, 1.0 - eps)
    return norm.ppf(np.column_stack([uv[:, 0], uv[:, 1], w])).astype(np.float32)


def inject_xor_triplet_blocks(
    x: np.ndarray,
    *,
    blocks: Sequence[Sequence[int]],
    rng: np.random.Generator,
    jitter_std: float = 1e-3,
    eps: float = 1e-6,
) -> np.ndarray:
    """Inject pairwise-matched higher-order triplets on disjoint 3-variable blocks."""
    x = np.asarray(x, dtype=np.float32).copy()
    for block in blocks:
        if len(block) != 3:
            raise ValueError("XOR triplet blocks must have length 3.")
        samples = _sample_modsum_higher_order_triplet(
            n_samples=x.shape[0],
            rng=rng,
            eps=eps,
        )
        if jitter_std > 0.0:
            samples = samples + float(jitter_std) * rng.standard_normal(samples.shape).astype(np.float32)
        for local_idx, global_idx in enumerate(block):
            x[:, int(global_idx)] = samples[:, local_idx]
    return x


def _sample_multiplicative_triplet(
    n_samples: int,
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    """Gaussianized multiplicative triplet with strong higher-order dependence."""
    x = rng.standard_normal((n_samples, 1)).astype(np.float64)
    y = rng.standard_normal((n_samples, 1)).astype(np.float64)
    z = x * y + float(noise_std) * rng.standard_normal((n_samples, 1)).astype(np.float64)
    raw = np.concatenate([x, y, z], axis=1)
    return gaussianize_columns(raw).astype(np.float32)


def inject_multiplicative_triplet_blocks(
    x: np.ndarray,
    *,
    blocks: Sequence[Sequence[int]],
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    for block in blocks:
        if len(block) != 3:
            raise ValueError("Multiplicative triplet blocks must have length 3.")
        samples = _sample_multiplicative_triplet(
            n_samples=x.shape[0],
            rng=rng,
            noise_std=noise_std,
        )
        for local_idx, global_idx in enumerate(block):
            x[:, int(global_idx)] = samples[:, local_idx]
    return x


def generate_window(
    t_idx: int,
    rng: np.random.Generator,
    *,
    config: ShowcaseConfig,
    variant: str = "current",
) -> np.ndarray:
    phase = phase_index_of_window(t_idx, config)
    if phase == 0:
        return _gen_independent(config.n_per_time, config.d, rng).astype(np.float32)
    if phase == 1:
        return _gen_pairwise_star_block(
            config.n_per_time,
            config.d,
            root_index=config.pair_root,
            leaf_indices=list(config.pair_leaves),
            rho=config.pair_rho,
            rng=rng,
        ).astype(np.float32)
    if phase == 2:
        phase3_mode = variant if variant is not None else config.phase3_mode
        if phase3_mode == "triplet_only":
            x = _gen_independent(config.n_per_time, config.d, rng).astype(np.float32)
            return inject_fixed_triplet_blocks(
                x,
                blocks=config.triplet_blocks,
                rho=config.triplet_rho,
                nu=config.triplet_nu,
                clayton_theta=config.triplet_clayton_theta,
                rng=rng,
            )
        if phase3_mode == "xor_only":
            x = _gen_independent(config.n_per_time, config.d, rng).astype(np.float32)
            return inject_xor_triplet_blocks(
                x,
                blocks=config.triplet_blocks,
                rng=rng,
                jitter_std=config.xor_jitter_std,
            )
        if phase3_mode == "multiplicative_only":
            x = _gen_independent(config.n_per_time, config.d, rng).astype(np.float32)
            return inject_multiplicative_triplet_blocks(
                x,
                blocks=config.triplet_blocks,
                rng=rng,
                noise_std=config.multiplicative_noise_std,
            )
        x = _gen_pairwise_star_block(
            config.n_per_time,
            config.d,
            root_index=config.pair_root,
            leaf_indices=list(config.pair_leaves),
            rho=config.pair_rho,
            rng=rng,
        ).astype(np.float32)
        if phase3_mode == "current":
            return inject_overwrite_blocks(
                x,
                blocks=config.triplet_blocks,
                rho=config.triplet_rho,
                nu=config.triplet_nu,
                rng=rng,
            )
        if phase3_mode == "fixed_phase3":
            return inject_fixed_triplet_blocks(
                x,
                blocks=config.triplet_blocks,
                rho=config.triplet_rho,
                nu=config.triplet_nu,
                clayton_theta=config.triplet_clayton_theta,
                rng=rng,
            )
        if phase3_mode == "xor_triplets":
            return inject_xor_triplet_blocks(
                x,
                blocks=config.triplet_blocks,
                rng=rng,
                jitter_std=config.xor_jitter_std,
            )
        if phase3_mode == "multiplicative_triplets":
            return inject_multiplicative_triplet_blocks(
                x,
                blocks=config.triplet_blocks,
                rng=rng,
                noise_std=config.multiplicative_noise_std,
            )
        raise ValueError(f"Unknown showcase variant: {phase3_mode}")
    return _gen_tail_block(
        config.n_per_time,
        config.d,
        block_indices=list(config.tail_block),
        theta=config.tail_theta,
        rng=rng,
    ).astype(np.float32)


def generate_sequence(
    *,
    seed: int,
    config: ShowcaseConfig,
    variant: str = "current",
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        generate_window(t_idx, rng, config=config, variant=variant)
        for t_idx in range(config.t)
    ]


def split_train_test(
    windows: Sequence[np.ndarray],
    train_frac: float,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    x_train: List[np.ndarray] = []
    x_test: List[np.ndarray] = []
    for x in windows:
        split = int(round(train_frac * x.shape[0]))
        x_train.append(np.asarray(x[:split], dtype=np.float32))
        x_test.append(np.asarray(x[split:], dtype=np.float32))
    return x_train, x_test


def evaluate_static_baselines(
    x_train_by_t: Sequence[np.ndarray],
    x_test_by_t: Sequence[np.ndarray],
    *,
    config: ShowcaseConfig,
    seed: int,
    skip_nf: bool = False,
    nf_epochs: int = 40,
    nf_hidden_dim: int = 32,
    nf_blocks: int = 4,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for t_idx, (tr, te) in enumerate(zip(x_train_by_t, x_test_by_t)):
        eval_seed = int(seed + 1000 + 17 * t_idx)
        row: Dict[str, Any] = {
            "t": int(t_idx),
            "phase": int(phase_index_of_window(t_idx, config)),
            "phase_name": config.phases[phase_index_of_window(t_idx, config)],
        }
        vine = _fit_parametric_vine(tr, families=FAMILIES, optimize_structure=False, seed=eval_seed)
        nll_dvc = float(_mean_copula_nll(vine, te))
        trunc_vine = _fit_truncated_cvine_level0(tr, families=FAMILIES, order=list(range(config.d)))
        nll_trunc = float(_mean_copula_nll(trunc_vine, te))
        nll_gauss = float(_gaussian_copula_nll_fit_eval(tr, te))
        row["nll_dvc"] = nll_dvc
        row["nll_trunc_level0"] = nll_trunc
        row["nll_gauss"] = nll_gauss
        row["tc_total_dvc"] = -nll_dvc
        row["tc_pair_dvc"] = -nll_trunc
        row["tc_higher_dvc"] = row["tc_total_dvc"] - row["tc_pair_dvc"]
        row["tc_gauss"] = -nll_gauss
        if skip_nf:
            row["nll_nf"] = float("nan")
            row["tc_total_nf"] = float("nan")
        else:
            try:
                nll_nf = float(
                    nf_copula_nll_fit_eval(
                        tr,
                        te,
                        n_epochs=nf_epochs,
                        hidden_dim=nf_hidden_dim,
                        n_blocks=nf_blocks,
                        seed=eval_seed,
                    )
                )
            except Exception:
                nll_nf = float("nan")
            row["nll_nf"] = nll_nf
            row["tc_total_nf"] = -nll_nf
        rows.append(row)

    ssm_result: Dict[str, Any] = {
        "process_variance": float("nan"),
        "nll": [float("nan")] * len(rows),
    }
    try:
        ssm_nll, fit = gaussian_copula_state_space_nll_fit_eval(x_train_by_t, x_test_by_t)
        ssm_result = {
            "process_variance": float(fit.process_variance),
            "nll": [float(v) for v in ssm_nll],
        }
    except Exception:
        pass

    for t_idx, nll_ssm in enumerate(ssm_result["nll"]):
        rows[t_idx]["nll_ssm"] = float(nll_ssm)
        rows[t_idx]["tc_total_ssm"] = -float(nll_ssm)

    return {
        "rows": rows,
        "ssm_process_variance": float(ssm_result["process_variance"]),
    }


def evaluate_regularized_dynamic_dvc(
    x_train_by_t: Sequence[np.ndarray],
    x_test_by_t: Sequence[np.ndarray],
    *,
    root_switch_penalty: float = 0.2,
    family_switch_penalty: float = 0.25,
    parameter_drift_penalty: float = 0.2,
    parameter_smoothing: float = 0.35,
    root_score_method: str = "kendall_tau",
) -> Dict[str, Any]:
    result, nll = _fit_regularized_dynamic_cvine_from_splits(
        list(x_train_by_t),
        list(x_test_by_t),
        families=FAMILIES,
        root_switch_penalty=root_switch_penalty,
        family_switch_penalty=family_switch_penalty,
        parameter_drift_penalty=parameter_drift_penalty,
        parameter_smoothing=parameter_smoothing,
        root_score_method=root_score_method,
    )
    return {
        "nll": [float(v) for v in nll],
        "tc_total_reg_dvc": [float(-v) for v in nll],
        "root_sequence": [int(v) for v in result.root_sequence],
        "root_local_costs_shape": list(result.root_local_costs.shape),
        "total_family_switches": int(result.total_family_switches()),
        "total_parameter_drift": float(result.total_parameter_drift()),
    }


def aggregate_seed_runs(
    seed_runs: Sequence[Dict[str, Any]],
    *,
    config: ShowcaseConfig,
) -> Dict[str, Any]:
    if not seed_runs:
        raise ValueError("seed_runs must be non-empty")

    numeric_keys = sorted(
        {
            key
            for run in seed_runs
            for row in run["rows"]
            for key, value in row.items()
            if isinstance(value, (int, float, np.floating))
        }
        - {"t", "phase"}
    )

    rows: List[Dict[str, Any]] = []
    for t_idx in range(config.t):
        ref = seed_runs[0]["rows"][t_idx]
        agg_row: Dict[str, Any] = {
            "t": int(ref["t"]),
            "phase": int(ref["phase"]),
            "phase_name": ref["phase_name"],
        }
        for key in numeric_keys:
            vals = np.asarray(
                [run["rows"][t_idx].get(key, np.nan) for run in seed_runs],
                dtype=np.float64,
            )
            if np.isfinite(vals).any():
                agg_row[key] = float(np.nanmean(vals))
                agg_row[f"{key}_std"] = float(np.nanstd(vals))
            else:
                agg_row[key] = float("nan")
                agg_row[f"{key}_std"] = float("nan")
        rows.append(agg_row)

    ssm_q = np.asarray([run.get("ssm_process_variance", np.nan) for run in seed_runs], dtype=np.float64)
    return {
        "config": config.to_dict(),
        "n_seeds": len(seed_runs),
        "seeds": [int(run["seed"]) for run in seed_runs],
        "ssm_process_variance": float(np.nanmean(ssm_q)),
        "ssm_process_variance_std": float(np.nanstd(ssm_q)),
        "rows": rows,
    }


def phasewise_metric_means(
    rows: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
    *,
    config: ShowcaseConfig,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for phase_name in config.phases:
        phase_rows = [row for row in rows if row["phase_name"] == phase_name]
        out[phase_name] = {}
        for key in metric_keys:
            vals = np.asarray(
                [row.get(key, np.nan) for row in phase_rows],
                dtype=np.float64,
            )
            if np.isfinite(vals).any():
                out[phase_name][key] = float(np.nanmean(vals))
            else:
                out[phase_name][key] = float("nan")
    return out


def per_phase_rows(rows: Sequence[Dict[str, Any]], *, config: ShowcaseConfig) -> Dict[str, List[Dict[str, Any]]]:
    return {
        phase_name: [row for row in rows if row["phase_name"] == phase_name]
        for phase_name in config.phases
    }


def enrich_phasewise_deltas(
    rows: Sequence[Dict[str, Any]],
    *,
    config: ShowcaseConfig,
) -> Dict[str, Dict[str, float]]:
    metric_keys = [
        "tc_total_dvc",
        "tc_total_np_windowed",
        "tc_pair_dvc",
        "tc_higher_dvc",
        "tc_total_ssm",
        "tc_gauss",
        "tc_total_nf",
    ]
    means = phasewise_metric_means(rows, metric_keys, config=config)
    out: Dict[str, Dict[str, float]] = {}
    for phase_name, phase_metrics in means.items():
        dvc = phase_metrics["tc_total_dvc"]
        np_windowed = phase_metrics.get("tc_total_np_windowed", float("nan"))
        gauss = phase_metrics["tc_gauss"]
        ssm = phase_metrics["tc_total_ssm"]
        out[phase_name] = {
            **phase_metrics,
            "dvc_minus_gauss": float(dvc - gauss),
            "dvc_minus_ssm": float(dvc - ssm),
            "ssm_minus_gauss": float(ssm - gauss),
            "np_minus_gauss": float(np_windowed - gauss) if math.isfinite(np_windowed) else float("nan"),
            "np_minus_ssm": float(np_windowed - ssm) if math.isfinite(np_windowed) else float("nan"),
            "np_minus_dvc": float(np_windowed - dvc) if math.isfinite(np_windowed) else float("nan"),
        }
    return out


def summarize_current_summary(summary_payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = summary_payload["rows"]
    config = ShowcaseConfig(
        d=int(summary_payload.get("d", D)),
        t=int(summary_payload.get("T", T)),
        n_per_time=int(summary_payload.get("n_per_time", N_PER_TIME)),
        phase_boundaries=tuple(summary_payload.get("phase_boundaries", PHASE_BOUNDARIES)),
        phases=tuple(summary_payload.get("phase_names", PHASES)),
    )
    return {
        "config": config.to_dict(),
        "n_seeds": int(summary_payload.get("n_seeds", 0)),
        "seeds": list(summary_payload.get("seeds", [])),
        "phasewise": enrich_phasewise_deltas(rows, config=config),
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float("nan")


def metric_delta_rows(
    phasewise: Dict[str, Dict[str, float]],
    *,
    labels: Sequence[str] = ("tc_total_dvc", "tc_total_ssm", "tc_gauss", "tc_higher_dvc"),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for phase_name, metrics in phasewise.items():
        row = {"phase_name": phase_name}
        for label in labels:
            row[label] = safe_float(metrics.get(label))
        row["dvc_minus_ssm"] = safe_float(metrics.get("dvc_minus_ssm"))
        row["dvc_minus_gauss"] = safe_float(metrics.get("dvc_minus_gauss"))
        row["ssm_minus_gauss"] = safe_float(metrics.get("ssm_minus_gauss"))
        rows.append(row)
    return rows


def write_simple_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for key in headers:
            val = row.get(key, "")
            if isinstance(val, float):
                if math.isnan(val):
                    values.append("nan")
                else:
                    values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n")


def run_static_seed(
    *,
    seed: int,
    config: ShowcaseConfig,
    variant: str,
    skip_nf: bool,
    nf_epochs: int = 40,
) -> Dict[str, Any]:
    windows = generate_sequence(seed=seed, config=config, variant=variant)
    x_train, x_test = split_train_test(windows, config.train_frac)
    evaluated = evaluate_static_baselines(
        x_train,
        x_test,
        config=config,
        seed=seed,
        skip_nf=skip_nf,
        nf_epochs=nf_epochs,
    )
    return {
        "seed": int(seed),
        **evaluated,
    }


def phase_acceptance_flags(phasewise: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
    pairwise = phasewise.get("pairwise-block", {})
    higher = phasewise.get("pairwise+higher-order", {})
    tail = phasewise.get("tail-block", {})
    return {
        "pairwise_close_to_ssm": abs(safe_float(pairwise.get("dvc_minus_ssm"))) <= 0.10,
        "higher_order_clear_gap": safe_float(higher.get("dvc_minus_ssm")) >= 0.25,
        "higher_order_tc_higher_large": safe_float(higher.get("tc_higher_dvc")) >= 0.50,
        "tail_gap_clear": safe_float(tail.get("dvc_minus_ssm")) >= 0.20,
    }


def score_sweep_candidate(phasewise: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    flags = phase_acceptance_flags(phasewise)
    satisfied = float(sum(flags.values()))
    higher = phasewise.get("pairwise+higher-order", {})
    tail = phasewise.get("tail-block", {})
    return {
        "targets_satisfied": satisfied,
        "higher_order_gap": safe_float(higher.get("dvc_minus_ssm")),
        "higher_order_tc_higher": safe_float(higher.get("tc_higher_dvc")),
        "tail_gap": safe_float(tail.get("dvc_minus_ssm")),
    }


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
