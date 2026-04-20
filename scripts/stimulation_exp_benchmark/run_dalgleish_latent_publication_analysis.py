#!/usr/bin/env python3
"""Publication-ready latent-state analysis for the Dalgleish dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.stimulation_exp_benchmark.build_dalgleish_dvc_dataset as builder
from scripts.stimulation_exp_benchmark.common import (
    FAMILY_VARIANTS,
    _apply_train_only_ecdf,
    _build_split_plan,
    _fit_train_only_ecdf,
    _prepare_session_cache,
    _score_gaussian_from_pobs,
    _score_vine_on_uniforms,
    _split_positions_random,
    _winsorize_train_apply,
    _with_quieter_repo_logging,
    _write_json,
    configure_logging,
)
from dvc_package.baselines.gaussian_state_space import gaussian_copula_state_space_nll_fit_eval
from dvc_package.experiments.simulation_benchmarks import (
    _estimate_hub_by_correlation,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
    _glasso_gaussian_copula_nll_fit_eval,
    _tvgl_gaussian_copula_nll_fit_eval,
)
from scipy.stats import norm


LOGGER = logging.getLogger("dalgleish_latent_publication")

SOURCE_VARIANTS: List[Dict[str, Any]] = [
    {"variant": "targeted_2bin_pca4", "source_space": "targeted", "n_components": 4},
    {"variant": "mixed_2bin_pca6", "source_space": "mixed", "n_components": 6},
    {"variant": "non_targeted_2bin_pca6", "source_space": "non_targeted", "n_components": 6},
]
MAIN_VARIANT = "non_targeted_2bin_pca6"
MAIN_SOURCE_SPACE = "non_targeted"
MAIN_N_COMPONENTS = 6
BASELINE_ORDER = ["graphical_lasso", "tvgl", "gaussian_ssm", "gaussian_copula", "truncated_vine"]
BASELINE_LABELS = {
    "graphical_lasso": "Graphical Lasso",
    "tvgl": "TVGL",
    "gaussian_ssm": "Gaussian SSM",
    "gaussian_copula": "Gaussian Copula",
    "truncated_vine": "1-trunc vine",
}


def _source_indices(targeted_mask: np.ndarray, source_space: str) -> np.ndarray:
    if source_space == "non_targeted":
        return np.flatnonzero(~targeted_mask).astype(int)
    if source_space == "targeted":
        return np.flatnonzero(targeted_mask).astype(int)
    if source_space == "mixed":
        return np.arange(targeted_mask.size, dtype=int)
    raise ValueError(f"Unknown source_space={source_space}")


def _build_source_matrix(cache: Dict[str, Any], source_indices: np.ndarray) -> np.ndarray:
    delayed = np.asarray(cache["delayed"][:, source_indices], dtype=np.float64)
    post = np.asarray(cache["post"][:, source_indices], dtype=np.float64)
    return np.concatenate([delayed, post], axis=1).astype(np.float64)


def _fit_pca_with_components(
    train_x: np.ndarray,
    test_x: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    mean = np.nanmean(train_x, axis=0)
    centered_train = train_x - mean
    centered_test = test_x - mean
    std = np.nanstd(centered_train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    z_train = np.nan_to_num(centered_train / std, nan=0.0, posinf=0.0, neginf=0.0)
    z_test = np.nan_to_num(centered_test / std, nan=0.0, posinf=0.0, neginf=0.0)
    _u, s, vt = np.linalg.svd(z_train, full_matrices=False)
    k = min(int(n_components), vt.shape[0], z_train.shape[0], z_train.shape[1])
    if k < int(n_components):
        raise ValueError(f"Only {k} PCA components available, need {n_components}")
    components = np.asarray(vt[:k], dtype=np.float64)
    train_scores = np.asarray(z_train @ components.T, dtype=np.float64)
    test_scores = np.asarray(z_test @ components.T, dtype=np.float64)
    denom = max(z_train.shape[0] - 1, 1)
    eigvals = (s ** 2) / denom
    total_var = float(np.sum(eigvals))
    explained = np.zeros(k, dtype=np.float64) if total_var <= 0.0 else np.asarray(eigvals[:k] / total_var, dtype=np.float64)
    return train_scores, test_scores, explained, components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dalgleish latent publication analysis.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--results_root", default="results/stimulation_exp_benchmark")
    parser.add_argument(
        "--window_backbone",
        choices=sorted(builder.WINDOW_BACKBONES.keys()),
        default=builder.DEFAULT_WINDOW_BACKBONE,
        help="Named Dalgleish response-window backbone used to rebuild trial summaries.",
    )
    parser.add_argument("--family_variant", choices=sorted(FAMILY_VARIANTS.keys()), default="stable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--min_trials_floor", type=int, default=18)
    parser.add_argument("--dynamic_min_block_trials", type=int, default=12)
    parser.add_argument("--control_min_train", type=int, default=5)
    parser.add_argument("--control_min_test", type=int, default=3)
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _fit_pca_basis(x: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    centered = x - mean
    std = np.nanstd(centered, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    z = np.nan_to_num(centered / std, nan=0.0, posinf=0.0, neginf=0.0)
    _u, s, vt = np.linalg.svd(z, full_matrices=False)
    k = min(int(n_components), vt.shape[0], z.shape[0], z.shape[1])
    if k < int(n_components):
        raise ValueError(f"Only {k} PCA components available, need {n_components}")
    components = np.asarray(vt[:k], dtype=np.float64)
    denom = max(z.shape[0] - 1, 1)
    eigvals = (s ** 2) / denom
    total_var = float(np.sum(eigvals))
    explained = np.zeros(k, dtype=np.float64) if total_var <= 0 else np.asarray(eigvals[:k] / total_var, dtype=np.float64)
    return mean, std, components, explained


def _project_pca(x: np.ndarray, mean: np.ndarray, std: np.ndarray, components: np.ndarray) -> np.ndarray:
    z = np.nan_to_num((np.asarray(x, dtype=np.float64) - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(z @ components.T, dtype=np.float64)


def _group_family(fam: str) -> str:
    fam = str(fam).lower().strip()
    if fam in {"ind", "independence"}:
        return "independence"
    if fam == "gaussian":
        return "gaussian_like_elliptical"
    if fam == "student":
        return "heavy_tailed_elliptical"
    if fam == "clayton":
        return "lower_tail_asymmetric"
    return fam


def _iter_family_rows(vine: Any, common: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if vine is None:
        return rows
    for level, cops in enumerate(getattr(vine, "copulas", [])):
        for edge_index, cop in enumerate(cops):
            fam = str(getattr(cop, "family", "ind")).lower().strip()
            rows.append(
                {
                    **common,
                    "tree_level": int(level),
                    "edge_index": int(edge_index),
                    "family_raw": fam,
                    "family_group": _group_family(fam),
                }
            )
    return rows


def _fit_transform_uniforms(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_clip, test_clip = _winsorize_train_apply(train_x, test_x)
    mappings = _fit_train_only_ecdf(train_clip)
    u_train = _apply_train_only_ecdf(train_clip, mappings)
    u_test = _apply_train_only_ecdf(test_clip, mappings)
    return u_train, u_test


def _fit_publication_models(
    train_x: np.ndarray,
    test_x: np.ndarray,
    families: Sequence[str],
    seed: int,
    run_extended_baselines: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "gaussian_copula_nll": np.nan,
        "graphical_lasso_nll": np.nan,
        "tvgl_nll": np.nan,
        "gaussian_ssm_nll": np.nan,
        "truncated_vine_nll": np.nan,
        "full_vine_nll": np.nan,
        "gaussian_status": "not_run",
        "graphical_lasso_status": "not_run",
        "tvgl_status": "not_run",
        "gaussian_ssm_status": "not_run",
        "truncated_vine_status": "not_run",
        "full_vine_status": "not_run",
        "truncated_vine": None,
        "full_vine": None,
    }
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    try:
        u_train, u_test = _fit_transform_uniforms(train_x, test_x)
        hub = int(_estimate_hub_by_correlation(norm.ppf(np.clip(u_train, 1e-6, 1.0 - 1e-6))))
        order = [hub] + [idx for idx in range(u_train.shape[1]) if idx != hub]
        out["gaussian_copula_nll"] = float(_score_gaussian_from_pobs(u_train, u_test))
        out["gaussian_status"] = "success"
        try:
            trunc_vine = _with_quieter_repo_logging(
                _fit_truncated_cvine_level0,
                x_train=u_train.astype(np.float32),
                families=list(families),
                order=order,
            )
            out["truncated_vine_nll"] = float(_score_vine_on_uniforms(trunc_vine, u_test))
            out["truncated_vine_status"] = "success"
            out["truncated_vine"] = trunc_vine
        except Exception as exc:  # pragma: no cover
            out["truncated_vine_status"] = f"failed:{exc}"
        try:
            full_vine = _with_quieter_repo_logging(
                _fit_parametric_vine,
                x_train=u_train.astype(np.float32),
                families=list(families),
                optimize_structure=False,
                seed=int(seed),
            )
            out["full_vine_nll"] = float(_score_vine_on_uniforms(full_vine, u_test))
            out["full_vine_status"] = "success"
            out["full_vine"] = full_vine
        except Exception as exc:  # pragma: no cover
            out["full_vine_status"] = f"failed:{exc}"
    except Exception as exc:  # pragma: no cover
        out["gaussian_status"] = f"failed:{exc}"
        return out

    if not run_extended_baselines:
        return out

    try:
        out["graphical_lasso_nll"] = float(_glasso_gaussian_copula_nll_fit_eval(train_x, test_x, alpha=0.02))
        out["graphical_lasso_status"] = "success"
    except Exception as exc:  # pragma: no cover
        out["graphical_lasso_status"] = f"failed:{exc}"
    try:
        seq = _tvgl_gaussian_copula_nll_fit_eval(
            [train_x],
            [test_x],
            alpha=0.02,
            beta=1.0,
            max_iter=200,
            step_size=0.05,
            eps=1e-4,
        )
        out["tvgl_nll"] = float(seq[0]) if seq else np.nan
        out["tvgl_status"] = "success" if seq else "failed:empty_sequence"
    except Exception as exc:  # pragma: no cover
        out["tvgl_status"] = f"failed:{exc}"
    try:
        seq, fit = gaussian_copula_state_space_nll_fit_eval([train_x], [test_x])
        out["gaussian_ssm_nll"] = float(seq[0]) if seq else np.nan
        out["gaussian_ssm_status"] = "success" if seq else "failed:empty_sequence"
    except Exception as exc:  # pragma: no cover
        out["gaussian_ssm_status"] = f"failed:{exc}"
    return out


def _family_mix_summary(frame: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    counts = frame.groupby(group_cols + ["family_raw", "family_group"]).size().reset_index(name="edge_count")
    totals = counts.groupby(group_cols)["edge_count"].sum().reset_index(name="total_edges")
    out = counts.merge(totals, on=group_cols, how="left")
    out["edge_fraction"] = out["edge_count"] / out["total_edges"]
    return out


def _mean_bootstrap_ci(values: Sequence[float], seed: int, draws: int) -> Tuple[float, float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(draws), arr.size))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_pvalue(values: Sequence[float], alternative: str = "greater") -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.nan
    obs = float(np.mean(arr))
    if n <= 16:
        means = []
        for mask in range(1 << n):
            signs = np.ones(n, dtype=np.float64)
            for bit in range(n):
                if (mask >> bit) & 1:
                    signs[bit] = -1.0
            means.append(float(np.mean(arr * signs)))
        means_arr = np.asarray(means, dtype=np.float64)
    else:
        rng = np.random.default_rng(0)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(50000, n), replace=True)
        means_arr = np.mean(signs * arr[None, :], axis=1)
    if alternative == "greater":
        return float(np.mean(means_arr >= obs))
    if alternative == "less":
        return float(np.mean(means_arr <= obs))
    return float(np.mean(np.abs(means_arr) >= abs(obs)))


def _rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return np.nan
    xr = pd.Series(x_arr[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y_arr[mask]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def _plot_main_figure(
    baseline_session_df: pd.DataFrame,
    decomposition_session_df: pd.DataFrame,
    source_session_df: pd.DataFrame,
    dose_pooled_df: pd.DataFrame,
    use_control_panel: bool,
    control_panel_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))

    # Panel A: full vine vs baselines.
    x = np.arange(len(BASELINE_ORDER))
    for session_id, group in baseline_session_df.groupby("session_id"):
        group = group.set_index("baseline").reindex(BASELINE_ORDER).reset_index()
        axes[0, 0].plot(x, group["delta_vs_full"], color="#c7c7c7", alpha=0.6, marker="o", linewidth=1.0)
    agg = baseline_session_df.groupby("baseline")["delta_vs_full"].mean().reindex(BASELINE_ORDER)
    axes[0, 0].plot(x, agg.to_numpy(dtype=float), color="#1f77b4", marker="o", linewidth=2.5)
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_xticks(x, [BASELINE_LABELS[b] for b in BASELINE_ORDER], rotation=25, ha="right")
    axes[0, 0].set_ylabel("Baseline NLL - Full-vine NLL")
    axes[0, 0].set_title("A. Full vine versus baselines")

    # Panel B: decomposition.
    decomp_order = ["low_level_pairwise_gain", "higher_order_gain"]
    decomp_labels = ["Gaussian -> 1-trunc", "1-trunc -> full"]
    dx = np.arange(len(decomp_order))
    for session_id, group in decomposition_session_df.groupby("session_id"):
        group = group.set_index("component").reindex(decomp_order).reset_index()
        axes[0, 1].plot(dx, group["value"], color="#c7c7c7", alpha=0.6, marker="o", linewidth=1.0)
    dagg = decomposition_session_df.groupby("component")["value"].mean().reindex(decomp_order)
    axes[0, 1].plot(dx, dagg.to_numpy(dtype=float), color="#d62728", marker="o", linewidth=2.5)
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_xticks(dx, decomp_labels, rotation=15, ha="right")
    axes[0, 1].set_ylabel("Held-out NLL gain")
    axes[0, 1].set_title("B. Pairwise-flexible vs higher-order gain")

    # Panel C: source space.
    source_order = [cfg["variant"] for cfg in SOURCE_VARIANTS]
    source_labels = ["Targeted", "Mixed", "Non-targeted"]
    sx = np.arange(len(source_order))
    for session_id, group in source_session_df.groupby("session_id"):
        group = group.set_index("variant").reindex(source_order).reset_index()
        axes[1, 0].plot(sx, group["full_vs_gaussian"], color="#c7c7c7", alpha=0.65, marker="o", linewidth=1.0)
    sagg = source_session_df.groupby("variant")["full_vs_gaussian"].mean().reindex(source_order)
    axes[1, 0].plot(sx, sagg.to_numpy(dtype=float), color="#2ca02c", marker="o", linewidth=2.5)
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_xticks(sx, source_labels)
    axes[1, 0].set_ylabel("Full-vs-Gaussian")
    axes[1, 0].set_title("C. Source-space comparison")

    # Panel D: dose or control
    if use_control_panel and not control_panel_df.empty:
        ctrl_order = ["catch_control", "stimulated"]
        ctrl_labels = ["Catch/control", "Stimulated"]
        cx = np.arange(len(ctrl_order))
        for session_id, group in control_panel_df.groupby("session_id"):
            group = group.set_index("condition_group").reindex(ctrl_order).reset_index()
            axes[1, 1].plot(cx, group["full_vs_gaussian"], color="#c7c7c7", alpha=0.6, marker="o", linewidth=1.0)
        cagg = control_panel_df.groupby("condition_group")["full_vs_gaussian"].mean().reindex(ctrl_order)
        axes[1, 1].plot(cx, cagg.to_numpy(dtype=float), color="#9467bd", marker="o", linewidth=2.5)
        axes[1, 1].axhline(0.0, color="black", linewidth=1)
        axes[1, 1].set_xticks(cx, ctrl_labels)
        axes[1, 1].set_ylabel("Full-vs-Gaussian")
        axes[1, 1].set_title("D. Control vs stimulated")
    else:
        for session_id, group in dose_pooled_df[dose_pooled_df["scope"] == "session"].groupby("session_id"):
            group = group.sort_values("dose")
            axes[1, 1].plot(group["dose"], group["full_vs_gaussian"], color="#c7c7c7", alpha=0.5)
        pooled = dose_pooled_df[dose_pooled_df["scope"] == "pooled"].sort_values("dose")
        axes[1, 1].plot(pooled["dose"], pooled["mean_full_vs_gaussian"], color="#9467bd", marker="o", linewidth=2.5)
        axes[1, 1].fill_between(pooled["dose"], pooled["ci_low_full_vs_gaussian"], pooled["ci_high_full_vs_gaussian"], color="#9467bd", alpha=0.2)
        axes[1, 1].axhline(0.0, color="black", linewidth=1)
        axes[1, 1].set_xlabel("Dose")
        axes[1, 1].set_ylabel("Full-vs-Gaussian")
        axes[1, 1].set_title("D. Dose robustness")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic_supplement(dynamic_block_df: pd.DataFrame, family_df: pd.DataFrame, out_path: Path) -> None:
    if dynamic_block_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    block_order = ["early", "middle", "late"]
    bx = np.arange(len(block_order))
    for basis_mode, color in [("window_train_basis", "#1f77b4"), ("common_basis", "#d62728")]:
        sub = dynamic_block_df[dynamic_block_df["basis_mode"] == basis_mode].copy()
        for session_id, group in sub.groupby("session_id"):
            group = group.set_index("block_id").reindex(block_order).reset_index()
            axes[0].plot(bx, group["full_vs_gaussian"], color=color, alpha=0.18)
            axes[1].plot(bx, group["tc_higher"], color=color, alpha=0.18)
        agg = sub.groupby("block_id")[["full_vs_gaussian", "tc_higher"]].mean().reindex(block_order)
        axes[0].plot(bx, agg["full_vs_gaussian"], color=color, marker="o", linewidth=2.5, label=basis_mode)
        axes[1].plot(bx, agg["tc_higher"], color=color, marker="o", linewidth=2.5, label=basis_mode)
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(bx, ["Early", "Middle", "Late"])
        ax.legend(frameon=False)
    axes[0].set_ylabel("Full-vs-Gaussian")
    axes[0].set_title("Dynamic gain")
    axes[1].set_ylabel("TC_higher")
    axes[1].set_title("Dynamic higher-order gain")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_family_supplement(family_summary_df: pd.DataFrame, out_path: Path) -> None:
    if family_summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    static = family_summary_df[(family_summary_df["analysis_scope"] == "static_source_space") & (family_summary_df["summary_level"] == "grouped_by_source")].copy()
    order = ["targeted_2bin_pca4", "mixed_2bin_pca6", "non_targeted_2bin_pca6"]
    fam_order = ["independence", "gaussian_like_elliptical", "heavy_tailed_elliptical", "lower_tail_asymmetric"]
    xpos = np.arange(len(order))
    bottoms = np.zeros(len(order), dtype=float)
    pivot = static.pivot_table(index="variant", columns="family_group", values="edge_fraction", aggfunc="mean", fill_value=0.0).reindex(order)
    for fam in [f for f in fam_order if f in pivot.columns]:
        vals = pivot[fam].to_numpy(dtype=float)
        axes[0].bar(xpos, vals, bottom=bottoms, label=fam)
        bottoms += vals
    axes[0].set_xticks(xpos, ["Targeted", "Mixed", "Non-targeted"])
    axes[0].set_ylabel("Edge fraction")
    axes[0].set_title("Static family mix by source space")
    axes[0].legend(frameon=False, fontsize=8)

    dyn = family_summary_df[(family_summary_df["analysis_scope"] == "dynamic") & (family_summary_df["summary_level"] == "grouped_by_block_basis")].copy()
    if not dyn.empty:
        dyn = dyn[dyn["basis_mode"] == "window_train_basis"]
        pivot = dyn.pivot_table(index="block_id", columns="family_group", values="edge_fraction", aggfunc="mean", fill_value=0.0).reindex(["early", "middle", "late"])
        xpos = np.arange(len(pivot))
        bottoms = np.zeros(len(pivot), dtype=float)
        for fam in [f for f in fam_order if f in pivot.columns]:
            vals = pivot[fam].to_numpy(dtype=float)
            axes[1].bar(xpos, vals, bottom=bottoms, label=fam)
            bottoms += vals
        axes[1].set_xticks(xpos, ["Early", "Middle", "Late"])
        axes[1].set_ylabel("Edge fraction")
        axes[1].set_title("Dynamic family mix by block")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    active_windows = builder.configure_windows(window_backbone=args.window_backbone)
    LOGGER.info(
        "Using Dalgleish window backbone %s with windows (seconds): %s",
        builder.get_window_backbone_name(),
        active_windows,
    )

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    results_root = Path(args.results_root).resolve()
    data_dir = results_root / "data"
    plots_dir = results_root / "plots"
    for path in [out_root, results_root, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    families = list(FAMILY_VARIANTS[args.family_variant])

    manifest = builder.build_manifest(data_root)
    trials_df, neural_data, _ = builder.build_trials(data_root, manifest)
    session_cache: Dict[str, Dict[str, Any]] = {}
    for session_id, payload in neural_data.items():
        if payload.get("used", False):
            cache = _prepare_session_cache(session_id, payload, data_root)
            cache["roi_lookup"] = payload["roi_lookup"].reset_index(drop=True).copy()
            session_cache[session_id] = cache

    static_rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    baseline_notes = {
        "gaussian_copula": "Validated Gaussian copula baseline used in the latent benchmark path.",
        "truncated_vine": "Validated 1-truncated vine baseline used in the latent benchmark path.",
        "graphical_lasso": "Repository GraphicalLasso Gaussian-copula baseline run on latent coordinates.",
        "tvgl": "Repository TVGL baseline attempted as a singleton-time adaptation on each static latent slice.",
        "gaussian_ssm": "Repository Gaussian state-space baseline run as a singleton-time adaptation on each static latent slice.",
    }

    for cfg in SOURCE_VARIANTS:
        variant = str(cfg["variant"])
        source_space = str(cfg["source_space"])
        n_components = int(cfg["n_components"])
        for session_id, cache in session_cache.items():
            source_idx = _source_indices(np.asarray(cache["targeted_mask"], dtype=bool), source_space)
            if source_idx.size < n_components:
                continue
            source_matrix = _build_source_matrix(cache, source_idx)
            split_plan = _build_split_plan(
                session_df=cache["session_df"],
                feature_dim=n_components,
                seed=int(args.seed),
                session_id=session_id,
                n_repeats=int(args.n_repeats),
                train_fraction=float(args.train_fraction),
                min_trials_floor=int(args.min_trials_floor),
            )
            for split in split_plan:
                train_idx = np.asarray(split["train_idx"], dtype=int)
                test_idx = np.asarray(split["test_idx"], dtype=int)
                try:
                    train_scores, test_scores, explained, _components = _fit_pca_with_components(
                        train_x=source_matrix[train_idx],
                        test_x=source_matrix[test_idx],
                        n_components=n_components,
                    )
                except Exception as exc:
                    LOGGER.debug("Static PCA failed for %s %s: %s", variant, split["slice_key"], exc)
                    continue
                scored = _fit_publication_models(
                    train_scores,
                    test_scores,
                    families=families,
                    seed=int(args.seed) + int(split["repeat_id"]) + n_components,
                    run_extended_baselines=(variant == MAIN_VARIANT),
                )
                tc = scored["truncated_vine_nll"] - scored["full_vine_nll"] if np.isfinite(scored["truncated_vine_nll"]) and np.isfinite(scored["full_vine_nll"]) else np.nan
                common = {
                    "analysis_scope": "static",
                    "variant": variant,
                    "source_space": source_space,
                    "n_components": n_components,
                    "session_id": session_id,
                    "dose": float(split["dose"]),
                    "repeat_id": int(split["repeat_id"]),
                    "slice_key": str(split["slice_key"]),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "pca_variance_retained": float(np.sum(explained)),
                }
                static_rows.extend(
                    [
                        {**common, "model": "gaussian_copula", "heldout_nll": scored["gaussian_copula_nll"], "fit_status": scored["gaussian_status"]},
                        {**common, "model": "truncated_vine", "heldout_nll": scored["truncated_vine_nll"], "fit_status": scored["truncated_vine_status"]},
                        {**common, "model": "full_vine", "heldout_nll": scored["full_vine_nll"], "fit_status": scored["full_vine_status"]},
                    ]
                )
                if variant == MAIN_VARIANT:
                    static_rows.extend(
                        [
                            {**common, "model": "graphical_lasso", "heldout_nll": scored["graphical_lasso_nll"], "fit_status": scored["graphical_lasso_status"]},
                            {**common, "model": "tvgl", "heldout_nll": scored["tvgl_nll"], "fit_status": scored["tvgl_status"]},
                            {**common, "model": "gaussian_ssm", "heldout_nll": scored["gaussian_ssm_nll"], "fit_status": scored["gaussian_ssm_status"]},
                        ]
                    )
                fam_common = {
                    "analysis_scope": "static_source_space",
                    "summary_level": "raw_static",
                    "variant": variant,
                    "source_space": source_space,
                    "session_id": session_id,
                    "dose": float(split["dose"]),
                    "repeat_id": int(split["repeat_id"]),
                    "slice_key": str(split["slice_key"]),
                }
                family_rows.extend(_iter_family_rows(scored["full_vine"], fam_common))

    static_df = pd.DataFrame(static_rows)

    # Source-space common slice intersection on gaussian/trunc/full only.
    source_success_sets = []
    for variant in [cfg["variant"] for cfg in SOURCE_VARIANTS]:
        full_success = static_df[
            (static_df["variant"] == variant)
            & (static_df["model"] == "full_vine")
            & (static_df["fit_status"] == "success")
        ]["slice_key"]
        trunc_success = static_df[
            (static_df["variant"] == variant)
            & (static_df["model"] == "truncated_vine")
            & (static_df["fit_status"] == "success")
        ]["slice_key"]
        gauss_success = static_df[
            (static_df["variant"] == variant)
            & (static_df["model"] == "gaussian_copula")
            & (static_df["fit_status"] == "success")
        ]["slice_key"]
        source_success_sets.append(set(full_success.tolist()) & set(trunc_success.tolist()) & set(gauss_success.tolist()))
    common_slice_keys = set.intersection(*source_success_sets) if source_success_sets else set()
    static_df["common_source_slice"] = static_df["slice_key"].isin(common_slice_keys)

    # Panel A baseline summary on main variant.
    main_rows = static_df[static_df["variant"] == MAIN_VARIANT].copy()
    pivot = main_rows.pivot_table(index=["session_id", "dose", "repeat_id", "slice_key"], columns="model", values="heldout_nll", aggfunc="first").reset_index()
    baseline_session_rows: List[Dict[str, Any]] = []
    for baseline in BASELINE_ORDER:
        if baseline not in pivot.columns or "full_vine" not in pivot.columns:
            continue
        tmp = pivot[["session_id", baseline, "full_vine"]].copy()
        tmp = tmp[np.isfinite(tmp[baseline]) & np.isfinite(tmp["full_vine"])].copy()
        tmp["delta_vs_full"] = tmp[baseline] - tmp["full_vine"]
        sess = tmp.groupby("session_id")["delta_vs_full"].mean().reset_index()
        sess["baseline"] = baseline
        baseline_session_rows.extend(sess.to_dict("records"))
    baseline_session_df = pd.DataFrame(baseline_session_rows)
    baseline_feasibility = []
    for baseline in BASELINE_ORDER:
        n_sessions = 0
        if not baseline_session_df.empty:
            n_sessions = int(baseline_session_df.loc[baseline_session_df["baseline"] == baseline, "session_id"].nunique())
        status = "run"
        note = baseline_notes[baseline]
        if n_sessions == 0:
            status = "attempted_no_valid_session_output"
            note = "No usable session-level latent-static comparison was produced in this publication pass. " + note
        baseline_feasibility.append(
            {
                "baseline": baseline,
                "status": status,
                "n_sessions": n_sessions,
                "note": note,
            }
        )

    # Source-space biological summary.
    source_pivot = static_df[static_df["common_source_slice"] & static_df["model"].isin(["gaussian_copula", "truncated_vine", "full_vine"])].pivot_table(
        index=["variant", "source_space", "session_id", "dose", "repeat_id", "slice_key"],
        columns="model",
        values="heldout_nll",
        aggfunc="first",
    ).reset_index()
    source_pivot = source_pivot[np.isfinite(source_pivot["gaussian_copula"]) & np.isfinite(source_pivot["truncated_vine"]) & np.isfinite(source_pivot["full_vine"])].copy()
    source_pivot["full_vs_gaussian"] = source_pivot["gaussian_copula"] - source_pivot["full_vine"]
    source_pivot["gaussian_to_trunc"] = source_pivot["gaussian_copula"] - source_pivot["truncated_vine"]
    source_pivot["tc_higher"] = source_pivot["truncated_vine"] - source_pivot["full_vine"]
    source_session_df = source_pivot.groupby(["variant", "source_space", "session_id"]).agg(
        full_vs_gaussian=("full_vs_gaussian", "mean"),
        gaussian_to_trunc=("gaussian_to_trunc", "mean"),
        tc_higher=("tc_higher", "mean"),
        n_slices=("slice_key", "nunique"),
    ).reset_index()

    # Decomposition for main variant only.
    main_source_session = source_session_df[source_session_df["variant"] == MAIN_VARIANT].copy()
    decomposition_session_df = pd.concat(
        [
            main_source_session[["session_id", "gaussian_to_trunc"]].rename(columns={"gaussian_to_trunc": "value"}).assign(component="low_level_pairwise_gain"),
            main_source_session[["session_id", "tc_higher"]].rename(columns={"tc_higher": "value"}).assign(component="higher_order_gain"),
        ],
        ignore_index=True,
    )

    # Dose summary for main variant.
    dose_session_df = source_pivot[source_pivot["variant"] == MAIN_VARIANT].groupby(["session_id", "dose"]).agg(
        full_vs_gaussian=("full_vs_gaussian", "mean"),
        gaussian_to_trunc=("gaussian_to_trunc", "mean"),
        tc_higher=("tc_higher", "mean"),
        n_slices=("slice_key", "nunique"),
    ).reset_index()
    dose_rows: List[Dict[str, Any]] = []
    for row in dose_session_df.to_dict("records"):
        row["scope"] = "session"
        dose_rows.append(row)
    for dose, group in dose_session_df.groupby("dose"):
        mean_fg, low_fg, high_fg = _mean_bootstrap_ci(group["full_vs_gaussian"], seed=args.seed + int(round(float(dose))), draws=args.bootstrap_draws)
        mean_gt, low_gt, high_gt = _mean_bootstrap_ci(group["gaussian_to_trunc"], seed=args.seed + 1000 + int(round(float(dose))), draws=args.bootstrap_draws)
        mean_tc, low_tc, high_tc = _mean_bootstrap_ci(group["tc_higher"], seed=args.seed + 2000 + int(round(float(dose))), draws=args.bootstrap_draws)
        dose_rows.append(
            {
                "scope": "pooled",
                "session_id": "",
                "dose": float(dose),
                "mean_full_vs_gaussian": mean_fg,
                "ci_low_full_vs_gaussian": low_fg,
                "ci_high_full_vs_gaussian": high_fg,
                "mean_gaussian_to_trunc": mean_gt,
                "ci_low_gaussian_to_trunc": low_gt,
                "ci_high_gaussian_to_trunc": high_gt,
                "mean_tc_higher": mean_tc,
                "ci_low_tc_higher": low_tc,
                "ci_high_tc_higher": high_tc,
                "p_value_full_vs_gaussian_positive": _sign_flip_pvalue(group["full_vs_gaussian"], alternative="greater"),
                "p_value_tc_positive": _sign_flip_pvalue(group["tc_higher"], alternative="greater"),
                "n_sessions": int(group["session_id"].nunique()),
            }
        )
    dose_summary_df = pd.DataFrame(dose_rows)

    # Control/catch feasibility and reduced-rank exploratory screen.
    control_rows: List[Dict[str, Any]] = []
    control_clean_enough = False
    control_session_panel = pd.DataFrame()
    for source_space, n_components in [("non_targeted", 2), ("mixed", 2), ("non_targeted", 3), ("mixed", 3)]:
        variant = f"{source_space}_control_pca{n_components}"
        usable_sessions = 0
        session_results = []
        for session_id, cache in session_cache.items():
            source_idx = _source_indices(np.asarray(cache["targeted_mask"], dtype=bool), source_space)
            if source_idx.size < n_components:
                continue
            source_matrix = _build_source_matrix(cache, source_idx)
            sdf = cache["session_df"].copy()
            catch_idx = sdf.index[sdf["dose"] == 0.0].to_numpy(dtype=int)
            stim_idx = sdf.index[sdf["dose"] > 0.0].to_numpy(dtype=int)
            if len(catch_idx) < (int(args.control_min_train) + int(args.control_min_test)) or len(stim_idx) < (int(args.control_min_train) + int(args.control_min_test)):
                continue
            rng = np.random.default_rng(int(args.seed) + len(session_id) + n_components)
            catch_train_pos, catch_test_pos = _split_positions_random(len(catch_idx), float(args.train_fraction), rng)
            stim_train_pos, stim_test_pos = _split_positions_random(len(stim_idx), float(args.train_fraction), rng)
            if len(catch_train_pos) < int(args.control_min_train) or len(catch_test_pos) < int(args.control_min_test):
                continue
            if len(stim_train_pos) < int(args.control_min_train) or len(stim_test_pos) < int(args.control_min_test):
                continue
            usable_sessions += 1
            for cond_name, train_idx, test_idx in [
                ("catch_control", catch_idx[catch_train_pos], catch_idx[catch_test_pos]),
                ("stimulated", stim_idx[stim_train_pos], stim_idx[stim_test_pos]),
            ]:
                try:
                    train_scores, test_scores, explained, _ = _fit_pca_with_components(
                        train_x=source_matrix[train_idx],
                        test_x=source_matrix[test_idx],
                        n_components=n_components,
                    )
                except Exception:
                    continue
                scored = _fit_publication_models(
                    train_scores,
                    test_scores,
                    families=families,
                    seed=int(args.seed) + (0 if cond_name == "catch_control" else 100),
                    run_extended_baselines=False,
                )
                if not np.isfinite(scored["full_vine_nll"]) or not np.isfinite(scored["gaussian_copula_nll"]) or not np.isfinite(scored["truncated_vine_nll"]):
                    continue
                session_results.append(
                    {
                        "variant": variant,
                        "source_space": source_space,
                        "n_components": n_components,
                        "session_id": session_id,
                        "condition_group": cond_name,
                        "full_vs_gaussian": scored["gaussian_copula_nll"] - scored["full_vine_nll"],
                        "gaussian_to_trunc": scored["gaussian_copula_nll"] - scored["truncated_vine_nll"],
                        "tc_higher": scored["truncated_vine_nll"] - scored["full_vine_nll"],
                        "pca_variance_retained": float(np.sum(explained)),
                    }
                )
        status = "not_clean_enough"
        if usable_sessions >= 6:
            status = "clean_enough_for_session_level_supporting_analysis"
            control_clean_enough = True
        control_rows.append(
            {
                "variant": variant,
                "source_space": source_space,
                "n_components": n_components,
                "usable_sessions": usable_sessions,
                "status": status,
                "note": "Catch/control remains sparse; reduced-rank exploratory comparison only.",
            }
        )
        if status == "clean_enough_for_session_level_supporting_analysis" and not control_session_panel.empty:
            pass
        if status == "clean_enough_for_session_level_supporting_analysis" and not control_session_panel.size:
            control_session_panel = pd.DataFrame(session_results)
    control_summary_df = pd.DataFrame(control_rows)
    if control_session_panel.empty:
        control_clean_enough = False

    # Dynamic early/middle/late exploratory analysis with family usage.
    dynamic_rows: List[Dict[str, Any]] = []
    dynamic_family_rows: List[Dict[str, Any]] = []
    for session_id, cache in session_cache.items():
        source_idx = _source_indices(np.asarray(cache["targeted_mask"], dtype=bool), MAIN_SOURCE_SPACE)
        if source_idx.size < MAIN_N_COMPONENTS:
            continue
        source_matrix = _build_source_matrix(cache, source_idx)
        session_df = cache["session_df"].copy()
        dose_counts = session_df.groupby("dose").size().sort_index()
        eligible = dose_counts[dose_counts >= 3 * int(args.dynamic_min_block_trials)]
        if eligible.empty:
            continue
        dose = float(np.max(eligible.index.to_numpy(dtype=float)))
        dose_df = session_df[session_df["dose"] == dose].sort_values("trial_order_within_session").copy()
        idx = dose_df.index.to_numpy(dtype=int)
        blocks = np.array_split(idx, 3)
        if min(len(b) for b in blocks) < int(args.dynamic_min_block_trials):
            continue
        all_x = source_matrix[idx]
        try:
            common_mean, common_std, common_components, common_explained = _fit_pca_basis(all_x, MAIN_N_COMPONENTS)
        except Exception:
            continue
        for block_id, block_idx in zip(["early", "middle", "late"], blocks):
            rng = np.random.default_rng(int(args.seed) + int(round(dose)) + len(block_id))
            train_pos, test_pos = _split_positions_random(len(block_idx), float(args.train_fraction), rng)
            if len(train_pos) < 5 or len(test_pos) < 3:
                continue
            train_idx = np.asarray(block_idx[train_pos], dtype=int)
            test_idx = np.asarray(block_idx[test_pos], dtype=int)
            for basis_mode in ["window_train_basis", "common_basis"]:
                try:
                    if basis_mode == "window_train_basis":
                        train_scores, test_scores, explained, _ = _fit_pca_with_components(
                            train_x=source_matrix[train_idx],
                            test_x=source_matrix[test_idx],
                            n_components=MAIN_N_COMPONENTS,
                        )
                    else:
                        train_scores = _project_pca(source_matrix[train_idx], common_mean, common_std, common_components)
                        test_scores = _project_pca(source_matrix[test_idx], common_mean, common_std, common_components)
                        explained = common_explained
                except Exception:
                    continue
                scored = _fit_publication_models(
                    train_scores,
                    test_scores,
                    families=families,
                    seed=int(args.seed) + int(round(dose)) + len(block_id),
                    run_extended_baselines=False,
                )
                if not np.isfinite(scored["full_vine_nll"]) or not np.isfinite(scored["gaussian_copula_nll"]) or not np.isfinite(scored["truncated_vine_nll"]):
                    continue
                dynamic_rows.append(
                    {
                        "session_id": session_id,
                        "dose": float(dose),
                        "block_id": block_id,
                        "basis_mode": basis_mode,
                        "full_vs_gaussian": scored["gaussian_copula_nll"] - scored["full_vine_nll"],
                        "gaussian_to_trunc": scored["gaussian_copula_nll"] - scored["truncated_vine_nll"],
                        "tc_higher": scored["truncated_vine_nll"] - scored["full_vine_nll"],
                        "pca_variance_retained": float(np.sum(explained)),
                    }
                )
                fam_common = {
                    "analysis_scope": "dynamic",
                    "summary_level": "raw_dynamic_block",
                    "variant": MAIN_VARIANT,
                    "source_space": MAIN_SOURCE_SPACE,
                    "session_id": session_id,
                    "dose": float(dose),
                    "block_id": block_id,
                    "basis_mode": basis_mode,
                }
                dynamic_family_rows.extend(_iter_family_rows(scored["full_vine"], fam_common))
    dynamic_block_df = pd.DataFrame(dynamic_rows)

    # Family summaries.
    family_df = pd.DataFrame(family_rows + dynamic_family_rows)
    family_summary_frames = []
    if not family_df.empty:
        family_summary_frames.append(_family_mix_summary(family_df[(family_df["analysis_scope"] == "static_source_space")].assign(summary_level="grouped_by_source"), ["analysis_scope", "summary_level", "variant", "source_space"]))
        family_summary_frames.append(_family_mix_summary(family_df[(family_df["analysis_scope"] == "static_source_space") & (family_df["variant"] == MAIN_VARIANT)].assign(analysis_scope="static_main", summary_level="grouped_by_dose"), ["analysis_scope", "summary_level", "dose"]))
        family_summary_frames.append(_family_mix_summary(family_df[(family_df["analysis_scope"] == "static_source_space") & (family_df["variant"] == MAIN_VARIANT)].assign(analysis_scope="static_main", summary_level="grouped_by_tree"), ["analysis_scope", "summary_level", "tree_level"]))
        if not dynamic_block_df.empty:
            dynfam = family_df[family_df["analysis_scope"] == "dynamic"].assign(summary_level="grouped_by_block_basis")
            family_summary_frames.append(_family_mix_summary(dynfam, ["analysis_scope", "summary_level", "basis_mode", "block_id"]))
    family_summary_df = pd.concat(family_summary_frames, ignore_index=True) if family_summary_frames else pd.DataFrame()

    # PC interpretability summary from existing validated follow-up outputs.
    latent_interpret_path = data_dir / "latent_state_interpretability.csv"
    latent_interpret = pd.read_csv(latent_interpret_path) if latent_interpret_path.exists() else pd.DataFrame()
    pc_main = latent_interpret[latent_interpret["variant"] == MAIN_VARIANT].copy() if not latent_interpret.empty else pd.DataFrame()
    pc_mixed = latent_interpret[latent_interpret["variant"] == "mixed_2bin_pca6"].copy() if not latent_interpret.empty else pd.DataFrame()
    pc_rows: List[Dict[str, Any]] = []
    if not pc_main.empty:
        prof = pc_main.groupby("pc_index").agg(
            mean_explained_variance=("explained_variance", "mean"),
            mean_loading_stability=("loading_stability_abs_cosine", "mean"),
            mean_post_weight_fraction=("post_weight_fraction", "mean"),
        ).reset_index()
        for row in prof.to_dict("records"):
            row["row_type"] = "main_variant_pc_profile"
            pc_rows.append(row)
        session_gain = source_session_df[source_session_df["variant"] == MAIN_VARIANT][["session_id", "full_vs_gaussian"]].copy()
        main_pc1 = pc_main[pc_main["pc_index"] == 1].groupby("session_id").agg(
            pc1_loading_stability=("loading_stability_abs_cosine", "mean"),
            pc1_post_weight_fraction=("post_weight_fraction", "mean"),
        ).reset_index()
        merged = main_pc1.merge(session_gain, on="session_id", how="inner")
        pc_rows.append(
            {
                "row_type": "association",
                "association": "full_vs_gaussian_vs_pc1_stability",
                "estimate": _rank_corr(merged["pc1_loading_stability"], merged["full_vs_gaussian"]),
                "interpretation": "positive means stronger gains with more stable recruited-population modes",
            }
        )
    if not pc_mixed.empty:
        mixed_pc1 = pc_mixed[pc_mixed["pc_index"] == 1].groupby("session_id").agg(
            pc1_targeted_enrichment=("targeted_enrichment", "mean"),
            pc1_post_weight_fraction=("post_weight_fraction", "mean"),
            pc1_target_proximity_delta_px=("top_loading_distance_delta_px", "mean"),
        ).reset_index()
        mixed_gain = source_session_df[source_session_df["variant"] == "mixed_2bin_pca6"][["session_id", "full_vs_gaussian"]].copy()
        merged = mixed_pc1.merge(mixed_gain, on="session_id", how="inner")
        pc_rows.extend(
            [
                {
                    "row_type": "association",
                    "association": "mixed_full_vs_gaussian_vs_pc1_targeted_enrichment",
                    "estimate": _rank_corr(merged["pc1_targeted_enrichment"], merged["full_vs_gaussian"]),
                    "interpretation": "positive means stronger gains with more targeted-enriched mixed latent modes",
                },
                {
                    "row_type": "association",
                    "association": "mixed_full_vs_gaussian_vs_pc1_target_proximity",
                    "estimate": _rank_corr(merged["pc1_target_proximity_delta_px"], merged["full_vs_gaussian"]),
                    "interpretation": "negative means stronger gains with more target-proximal mixed latent modes",
                },
            ]
        )
    if latent_interpret.empty:
        pc_rows.append(
            {
                "row_type": "status",
                "association": "latent_interpretability_missing",
                "estimate": np.nan,
                "interpretation": "Optional latent_state_interpretability.csv was not found, so PC interpretability summaries were skipped.",
            }
        )
    pc_summary_df = pd.DataFrame(pc_rows)

    # Stats summary.
    stats_rows: List[Dict[str, Any]] = []
    for baseline in BASELINE_ORDER:
        vals = baseline_session_df.loc[baseline_session_df["baseline"] == baseline, "delta_vs_full"].to_numpy(dtype=float)
        mean_v, low, high = _mean_bootstrap_ci(vals, seed=args.seed + len(baseline), draws=args.bootstrap_draws)
        stats_rows.append(
            {
                "analysis_scope": "panel_a_baseline",
                "comparison": f"full vine vs {baseline}",
                "metric": "baseline_minus_full",
                "estimate": mean_v,
                "ci_low": low,
                "ci_high": high,
                "p_value": _sign_flip_pvalue(vals, alternative="greater"),
                "replication_unit": "session",
                "n_sessions": int(np.sum(np.isfinite(vals))),
            }
        )
    for component in ["low_level_pairwise_gain", "higher_order_gain"]:
        vals = decomposition_session_df.loc[decomposition_session_df["component"] == component, "value"].to_numpy(dtype=float)
        mean_v, low, high = _mean_bootstrap_ci(vals, seed=args.seed + len(component), draws=args.bootstrap_draws)
        stats_rows.append(
            {
                "analysis_scope": "panel_b_decomposition",
                "comparison": component,
                "metric": "gain",
                "estimate": mean_v,
                "ci_low": low,
                "ci_high": high,
                "p_value": _sign_flip_pvalue(vals, alternative="greater"),
                "replication_unit": "session",
                "n_sessions": int(np.sum(np.isfinite(vals))),
            }
        )
    for metric in ["full_vs_gaussian", "tc_higher", "gaussian_to_trunc"]:
        for a, b, label in [
            ("non_targeted_2bin_pca6", "targeted_2bin_pca4", "non_targeted minus targeted"),
            ("non_targeted_2bin_pca6", "mixed_2bin_pca6", "non_targeted minus mixed"),
        ]:
            a_df = source_session_df[source_session_df["variant"] == a][["session_id", metric]].rename(columns={metric: "a"})
            b_df = source_session_df[source_session_df["variant"] == b][["session_id", metric]].rename(columns={metric: "b"})
            merged = a_df.merge(b_df, on="session_id", how="inner")
            diff = (merged["a"] - merged["b"]).to_numpy(dtype=float)
            mean_v, low, high = _mean_bootstrap_ci(diff, seed=args.seed + len(metric) + len(label), draws=args.bootstrap_draws)
            stats_rows.append(
                {
                    "analysis_scope": "panel_c_source_space",
                    "comparison": label,
                    "metric": metric,
                    "estimate": mean_v,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": _sign_flip_pvalue(diff, alternative="two-sided"),
                    "replication_unit": "session",
                    "n_sessions": int(np.sum(np.isfinite(diff))),
                }
            )
    for dose, group in dose_session_df.groupby("dose"):
        vals = group["full_vs_gaussian"].to_numpy(dtype=float)
        mean_v, low, high = _mean_bootstrap_ci(vals, seed=args.seed + int(round(float(dose))), draws=args.bootstrap_draws)
        stats_rows.append(
            {
                "analysis_scope": "panel_d_dose",
                "comparison": f"dose_{int(round(float(dose)))}",
                "metric": "full_vs_gaussian",
                "estimate": mean_v,
                "ci_low": low,
                "ci_high": high,
                "p_value": _sign_flip_pvalue(vals, alternative="greater"),
                "replication_unit": "session",
                "n_sessions": int(np.sum(np.isfinite(vals))),
            }
        )
    if not dynamic_block_df.empty:
        # basis agreement by sign
        merged = dynamic_block_df.pivot_table(index=["session_id", "block_id"], columns="basis_mode", values="full_vs_gaussian")
        if {"window_train_basis", "common_basis"} <= set(merged.columns):
            agree = np.sign(merged["window_train_basis"]) == np.sign(merged["common_basis"])
            stats_rows.append(
                {
                    "analysis_scope": "dynamic",
                    "comparison": "full_vs_gaussian_basis_sign_agreement",
                    "metric": "agreement_fraction",
                    "estimate": float(np.mean(agree)),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": np.nan,
                    "replication_unit": "session_block",
                    "n_sessions": int(agree.shape[0]),
                }
            )
    stats_df = pd.DataFrame(stats_rows)

    # Final figure choice for panel D.
    use_control_panel = bool(control_clean_enough and not control_session_panel.empty)

    # Outputs.
    static_out = pd.concat(
        [
            source_session_df.assign(row_type="source_space_session"),
            baseline_session_df.assign(row_type="baseline_session"),
            decomposition_session_df.assign(row_type="decomposition_session"),
        ],
        ignore_index=True,
        sort=False,
    )
    dynamic_out = dynamic_block_df.copy()
    control_out = control_summary_df.copy()
    if not control_session_panel.empty:
        control_out = pd.concat([control_out, control_session_panel], ignore_index=True, sort=False)

    for name, frame in {
        "latent_publication_static_summary.csv": static_out,
        "latent_publication_control_summary.csv": control_out,
        "latent_publication_family_summary.csv": family_summary_df,
        "latent_publication_dynamic_summary.csv": dynamic_out,
        "latent_publication_pc_summary.csv": pc_summary_df,
        "latent_publication_stats_summary.csv": stats_df,
        "latent_publication_baseline_feasibility.csv": pd.DataFrame(baseline_feasibility),
    }.items():
        frame.to_csv(data_dir / name, index=False)
        frame.to_csv(out_root / name, index=False)

    _plot_main_figure(
        baseline_session_df=baseline_session_df,
        decomposition_session_df=decomposition_session_df,
        source_session_df=source_session_df,
        dose_pooled_df=dose_summary_df,
        use_control_panel=use_control_panel,
        control_panel_df=control_session_panel,
        out_path=plots_dir / "fig_latent_publication_final.png",
    )
    _plot_dynamic_supplement(dynamic_block_df, family_summary_df, plots_dir / "fig_latent_publication_dynamic_supplement.png")
    _plot_family_supplement(family_summary_df, plots_dir / "fig_latent_publication_family_supplement.png")

    main_mean_full_vs_gaussian = float(main_source_session["full_vs_gaussian"].mean()) if not main_source_session.empty else float("nan")
    main_mean_tc_higher = float(main_source_session["tc_higher"].mean()) if not main_source_session.empty else float("nan")
    paper_decision = "main-text ready"
    if not (
        np.isfinite(main_mean_full_vs_gaussian)
        and np.isfinite(main_mean_tc_higher)
        and main_mean_full_vs_gaussian > 0.0
        and main_mean_tc_higher > 0.0
    ):
        paper_decision = "supporting-only / re-audit required"
    metadata = {
        "family_variant": args.family_variant,
        "window_backbone": builder.get_window_backbone_name(),
        "window_backbone_metadata": builder.get_window_backbone_metadata(),
        "main_variant": MAIN_VARIANT,
        "main_source_space": MAIN_SOURCE_SPACE,
        "panel_d_used": "control" if use_control_panel else "dose",
        "control_clean_enough": bool(use_control_panel),
        "control_constraint": "catch trials are sparse and were screened with reduced-rank exploratory comparisons",
        "dynamic_is_exploratory": True,
        "main_mean_full_vs_gaussian": main_mean_full_vs_gaussian,
        "main_mean_tc_higher": main_mean_tc_higher,
        "paper_decision": paper_decision,
    }
    _write_json(data_dir / "latent_publication_metadata.json", metadata)
    _write_json(out_root / "latent_publication_metadata.json", metadata)
    LOGGER.info("Latent publication analysis complete | Panel D = %s | paper decision: %s", metadata["panel_d_used"], paper_decision)


if __name__ == "__main__":
    main()
