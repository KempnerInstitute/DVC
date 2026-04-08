#!/usr/bin/env python3
"""Publication-oriented latent-state follow-up analysis for the Dalgleish dataset."""

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
from dvc_package.experiments.simulation_benchmarks import (
    _estimate_hub_by_correlation,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
)
from scipy.stats import norm
from scripts.debug_stimulation_exp.run_dalgleish_formulation_viability import _build_split_plan, _prepare_session_cache
from scripts.debug_stimulation_exp.run_dalgleish_latent_state_full_run import (
    _build_source_matrix,
    _fit_pca_with_components,
    _source_indices,
)
from scripts.debug_stimulation_exp.run_dalgleish_real_data_benchmark import (
    FAMILY_VARIANTS,
    _apply_train_only_ecdf,
    _fit_train_only_ecdf,
    _score_gaussian_from_pobs,
    _score_vine_on_uniforms,
    _winsorize_train_apply,
    _with_quieter_repo_logging,
    _write_json,
    configure_logging,
)


LOGGER = logging.getLogger("dalgleish_latent_followup")


STATIC_VARIANTS: List[Dict[str, Any]] = [
    {"variant": "targeted_2bin_pca4", "source_space": "targeted", "n_components": 4},
    {"variant": "mixed_2bin_pca6", "source_space": "mixed", "n_components": 6},
    {"variant": "non_targeted_2bin_pca6", "source_space": "non_targeted", "n_components": 6},
]
MAIN_VARIANT = "non_targeted_2bin_pca6"
MAIN_SOURCE_SPACE = "non_targeted"
MAIN_N_COMPONENTS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run publication-oriented Dalgleish latent follow-up analysis.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--results_root", default="results/stimulation_exp_benchmark")
    parser.add_argument("--family_variant", choices=sorted(FAMILY_VARIANTS.keys()), default="stable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--min_trials_floor", type=int, default=18)
    parser.add_argument("--dynamic_window_size", type=int, default=18)
    parser.add_argument("--dynamic_step", type=int, default=6)
    parser.add_argument("--dynamic_min_windows", type=int, default=3)
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _fit_and_score_with_vines(
    train_x: np.ndarray,
    test_x: np.ndarray,
    families: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    out: Dict[str, Any] = {
        "gaussian_nll": np.nan,
        "trunc_nll": np.nan,
        "full_nll": np.nan,
        "gaussian_status": "not_run",
        "trunc_status": "not_run",
        "full_status": "not_run",
        "trunc_vine": None,
        "full_vine": None,
    }
    train_clip, test_clip = _winsorize_train_apply(train_x, test_x)
    mappings = _fit_train_only_ecdf(train_clip)
    u_train = _apply_train_only_ecdf(train_clip, mappings)
    u_test = _apply_train_only_ecdf(test_clip, mappings)
    hub = int(_estimate_hub_by_correlation(norm.ppf(np.clip(u_train, 1e-6, 1.0 - 1e-6))))
    order = [hub] + [idx for idx in range(u_train.shape[1]) if idx != hub]

    out["gaussian_nll"] = float(_score_gaussian_from_pobs(u_train, u_test))
    out["gaussian_status"] = "success"
    try:
        trunc_vine = _with_quieter_repo_logging(
            _fit_truncated_cvine_level0,
            x_train=u_train.astype(np.float32),
            families=list(families),
            order=order,
        )
        out["trunc_nll"] = float(_score_vine_on_uniforms(trunc_vine, u_test))
        out["trunc_status"] = "success"
        out["trunc_vine"] = trunc_vine
    except Exception as exc:  # pragma: no cover - runtime dependent
        out["trunc_status"] = f"failed:{exc}"
    try:
        full_vine = _with_quieter_repo_logging(
            _fit_parametric_vine,
            x_train=u_train.astype(np.float32),
            families=list(families),
            optimize_structure=False,
            seed=int(seed),
        )
        out["full_nll"] = float(_score_vine_on_uniforms(full_vine, u_test))
        out["full_status"] = "success"
        out["full_vine"] = full_vine
    except Exception as exc:  # pragma: no cover - runtime dependent
        out["full_status"] = f"failed:{exc}"
    return out


def _fit_pca_basis(x: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _iter_vine_family_rows(
    vine: Any,
    common: Dict[str, Any],
) -> List[Dict[str, Any]]:
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


def _session_seed(session_id: str, base_seed: int, extra: int = 0) -> int:
    session_term = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(session_id)))
    return int(base_seed + session_term + extra)


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


def _family_mix_summary(frame: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    counts = frame.groupby(group_cols + ["family_raw", "family_group"]).size().reset_index(name="edge_count")
    totals = counts.groupby(group_cols)["edge_count"].sum().reset_index(name="total_edges")
    out = counts.merge(totals, on=group_cols, how="left")
    out["edge_fraction"] = out["edge_count"] / out["total_edges"]
    return out


def _plot_main_figure(static_session_df: pd.DataFrame, dose_pooled_df: pd.DataFrame, pc_plot_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    order = ["targeted_2bin_pca4", "mixed_2bin_pca6", "non_targeted_2bin_pca6"]
    labels = ["Targeted", "Mixed", "Non-targeted"]
    x = np.arange(len(order))

    for session_id, group in static_session_df.groupby("session_id"):
        group = group.set_index("variant").reindex(order).reset_index()
        axes[0, 0].plot(x, group["full_delta_vs_gaussian"], color="#c7c7c7", alpha=0.75, marker="o")
        axes[0, 1].plot(x, group["tc_higher"], color="#c7c7c7", alpha=0.75, marker="o")
    agg = static_session_df.groupby("variant").agg(
        mean_full_delta=("full_delta_vs_gaussian", "mean"),
        mean_tc=("tc_higher", "mean"),
    ).reindex(order)
    axes[0, 0].plot(x, agg["mean_full_delta"], color="#1f77b4", linewidth=2.5, marker="o")
    axes[0, 1].plot(x, agg["mean_tc"], color="#d62728", linewidth=2.5, marker="o")
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_xticks(x, labels)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Full-vine delta vs Gaussian")
    axes[0, 1].set_ylabel("TC_higher")
    axes[0, 0].set_title("Source-space comparison")
    axes[0, 1].set_title("Higher-order gain by source space")

    for session_id, group in dose_pooled_df[dose_pooled_df["scope"] == "session"].groupby("session_id"):
        group = group.sort_values("dose")
        axes[1, 0].plot(group["dose"], group["full_delta_vs_gaussian"], color="#c7c7c7", alpha=0.6)
    pooled = dose_pooled_df[dose_pooled_df["scope"] == "pooled"].sort_values("dose")
    axes[1, 0].plot(pooled["dose"], pooled["mean_full_delta_vs_gaussian"], color="#1f77b4", linewidth=2.5, marker="o")
    axes[1, 0].fill_between(
        pooled["dose"],
        pooled["ci_low_full_delta_vs_gaussian"],
        pooled["ci_high_full_delta_vs_gaussian"],
        color="#1f77b4",
        alpha=0.2,
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_xlabel("Dose")
    axes[1, 0].set_ylabel("Full-vine delta vs Gaussian")
    axes[1, 0].set_title("Dose robustness in latent space")

    if not pc_plot_df.empty:
        for _, row in pc_plot_df.iterrows():
            axes[1, 1].plot([row["pc_index"] - 0.12, row["pc_index"] + 0.12], [row["mean_explained_variance"], row["mean_loading_stability"]], color="#bdbdbd", alpha=0.5)
        axes[1, 1].plot(pc_plot_df["pc_index"], pc_plot_df["mean_explained_variance"], color="#4c78a8", marker="o", linewidth=2.0, label="Variance explained")
        axes[1, 1].plot(pc_plot_df["pc_index"], pc_plot_df["mean_loading_stability"], color="#54a24b", marker="o", linewidth=2.0, label="Loading stability")
        axes[1, 1].set_xlabel("PC")
        axes[1, 1].set_ylabel("Mean value")
        axes[1, 1].set_title("Latent PC interpretability")
        axes[1, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic(dynamic_agg_df: pd.DataFrame, out_path: Path) -> None:
    if dynamic_agg_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for basis_mode, color in [("window_train_basis", "#1f77b4"), ("common_basis_descriptive", "#d62728")]:
        sub = dynamic_agg_df[dynamic_agg_df["basis_mode"] == basis_mode].sort_values("window_rank")
        for session_id, group in sub.groupby("session_id"):
            group = group.sort_values("window_rank")
            axes[0].plot(group["window_rank"], group["full_delta_vs_gaussian"], color=color, alpha=0.18)
            axes[1].plot(group["window_rank"], group["tc_higher"], color=color, alpha=0.18)
        pooled = sub.groupby("window_rank").agg(
            mean_full_delta=("full_delta_vs_gaussian", "mean"),
            mean_tc=("tc_higher", "mean"),
        ).reset_index()
        axes[0].plot(pooled["window_rank"], pooled["mean_full_delta"], color=color, linewidth=2.5, marker="o", label=basis_mode)
        axes[1].plot(pooled["window_rank"], pooled["mean_tc"], color=color, linewidth=2.5, marker="o", label=basis_mode)
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Rolling window rank")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Full-vine delta vs Gaussian")
    axes[1].set_ylabel("TC_higher")
    axes[0].set_title("Dynamic latent gain")
    axes[1].set_title("Dynamic higher-order gain")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_family_usage(fam_df: pd.DataFrame, out_path: Path) -> None:
    if fam_df.empty:
        return
    plot_df = fam_df[(fam_df["analysis_scope"] == "static_main") & (fam_df["summary_level"] == "grouped_by_tree")].copy()
    if plot_df.empty:
        return
    order = ["independence", "gaussian_like_elliptical", "heavy_tailed_elliptical", "lower_tail_asymmetric"]
    pivot = plot_df.pivot_table(index="tree_level", columns="family_group", values="edge_fraction", aggfunc="mean", fill_value=0.0)
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns], fill_value=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bottoms = np.zeros(len(pivot), dtype=float)
    for fam in pivot.columns:
        axes[0].bar(pivot.index.astype(int), pivot[fam].to_numpy(dtype=float), bottom=bottoms, label=fam)
        bottoms += pivot[fam].to_numpy(dtype=float)
    axes[0].set_xlabel("Tree level")
    axes[0].set_ylabel("Mean edge fraction")
    axes[0].set_title("Full-vine family usage by tree")
    axes[0].legend(frameon=False, fontsize=8)

    dose_df = fam_df[(fam_df["analysis_scope"] == "static_main") & (fam_df["summary_level"] == "grouped_by_dose")].copy()
    if not dose_df.empty:
        for fam in order:
            fam_sub = dose_df[dose_df["family_group"] == fam].sort_values("dose")
            if fam_sub.empty:
                continue
            axes[1].plot(fam_sub["dose"], fam_sub["edge_fraction"], marker="o", linewidth=2.0, label=fam)
        axes[1].set_xlabel("Dose")
        axes[1].set_ylabel("Mean edge fraction")
        axes[1].set_title("Family mix by dose")
        axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_interpretability_extra(pc_session_df: pd.DataFrame, out_path: Path) -> None:
    if pc_session_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].scatter(pc_session_df["pc1_target_proximity_delta_px"], pc_session_df["full_delta_vs_gaussian"], color="#1f77b4")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("PC1 top-loading distance delta to targets (px)")
    axes[0].set_ylabel("Session mean full-vs-Gaussian")
    axes[0].set_title("Target proximity vs latent DVC gain")

    axes[1].scatter(pc_session_df["pc1_loading_stability"], pc_session_df["full_delta_vs_gaussian"], color="#d62728")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("PC1 loading stability")
    axes[1].set_ylabel("Session mean full-vs-Gaussian")
    axes[1].set_title("Stability vs latent DVC gain")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    results_root = Path(args.results_root).resolve()
    data_dir = results_root / "data"
    plots_dir = results_root / "plots"
    for path in [out_root, results_root, data_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    families = list(FAMILY_VARIANTS[args.family_variant])
    base_results_dir = data_dir
    latent_metrics_existing = pd.read_csv(base_results_dir / "latent_state_metrics_table.csv")
    latent_interpret_existing = pd.read_csv(base_results_dir / "latent_state_interpretability.csv")

    manifest = builder.build_manifest(data_root)
    _trials_df, neural_data, _ = builder.build_trials(data_root, manifest)
    session_cache: Dict[str, Dict[str, Any]] = {}
    for session_id, payload in neural_data.items():
        if payload.get("used", False):
            cache = _prepare_session_cache(session_id, payload, data_root)
            cache["roi_lookup"] = payload["roi_lookup"].reset_index(drop=True).copy()
            session_cache[session_id] = cache

    static_metric_rows: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []

    for cfg in STATIC_VARIANTS:
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
                scored = _fit_and_score_with_vines(
                    train_x=train_scores,
                    test_x=test_scores,
                    families=families,
                    seed=int(args.seed) + int(split["repeat_id"]) + n_components,
                )
                tc = scored["trunc_nll"] - scored["full_nll"] if np.isfinite(scored["trunc_nll"]) and np.isfinite(scored["full_nll"]) else np.nan
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
                static_metric_rows.extend(
                    [
                        {**common, "model": "gaussian", "heldout_nll": scored["gaussian_nll"], "delta_vs_gaussian": 0.0, "full_delta_vs_trunc": np.nan, "tc_higher": np.nan, "fit_status": scored["gaussian_status"]},
                        {**common, "model": "truncated_vine", "heldout_nll": scored["trunc_nll"], "delta_vs_gaussian": scored["gaussian_nll"] - scored["trunc_nll"] if np.isfinite(scored["gaussian_nll"]) and np.isfinite(scored["trunc_nll"]) else np.nan, "full_delta_vs_trunc": np.nan, "tc_higher": tc, "fit_status": scored["trunc_status"]},
                        {**common, "model": "full_vine", "heldout_nll": scored["full_nll"], "delta_vs_gaussian": scored["gaussian_nll"] - scored["full_nll"] if np.isfinite(scored["gaussian_nll"]) and np.isfinite(scored["full_nll"]) else np.nan, "full_delta_vs_trunc": scored["trunc_nll"] - scored["full_nll"] if np.isfinite(scored["trunc_nll"]) and np.isfinite(scored["full_nll"]) else np.nan, "tc_higher": tc, "fit_status": scored["full_status"]},
                    ]
                )
                family_common = {
                    "analysis_scope": "static_source_space",
                    "variant": variant,
                    "source_space": source_space,
                    "n_components": n_components,
                    "session_id": session_id,
                    "dose": float(split["dose"]),
                    "repeat_id": int(split["repeat_id"]),
                    "slice_key": str(split["slice_key"]),
                }
                family_rows.extend(_iter_vine_family_rows(scored["full_vine"], family_common))

    static_metrics_df = pd.DataFrame(static_metric_rows)
    successful_sets = []
    for variant in [cfg["variant"] for cfg in STATIC_VARIANTS]:
        rows = static_metrics_df[
            (static_metrics_df["variant"] == variant)
            & (static_metrics_df["model"] == "full_vine")
            & (static_metrics_df["fit_status"] == "success")
        ]["slice_key"]
        successful_sets.append(set(rows.tolist()))
    common_slice_keys = set.intersection(*successful_sets) if successful_sets else set()
    static_metrics_df["common_slice"] = static_metrics_df["slice_key"].isin(common_slice_keys)

    static_full = static_metrics_df[
        (static_metrics_df["model"] == "full_vine") & (static_metrics_df["common_slice"])
    ].copy()
    static_session_df = static_full.groupby(["variant", "source_space", "session_id"]).agg(
        full_delta_vs_gaussian=("delta_vs_gaussian", "mean"),
        full_delta_vs_trunc=("full_delta_vs_trunc", "mean"),
        tc_higher=("tc_higher", "mean"),
        pca_variance_retained=("pca_variance_retained", "mean"),
        n_slices=("slice_key", "nunique"),
    ).reset_index()
    static_variant_df = static_session_df.groupby(["variant", "source_space"]).agg(
        mean_full_delta_vs_gaussian=("full_delta_vs_gaussian", "mean"),
        mean_full_delta_vs_trunc=("full_delta_vs_trunc", "mean"),
        mean_tc_higher=("tc_higher", "mean"),
        session_count=("session_id", "nunique"),
        mean_n_slices=("n_slices", "mean"),
    ).reset_index()

    stats_rows: List[Dict[str, Any]] = []
    for metric_name in ["full_delta_vs_gaussian", "tc_higher", "full_delta_vs_trunc"]:
        for variant, label in [
            ("non_targeted_2bin_pca6", "non_targeted positivity"),
            ("mixed_2bin_pca6", "mixed positivity"),
            ("targeted_2bin_pca4", "targeted positivity"),
        ]:
            vals = static_session_df.loc[static_session_df["variant"] == variant, metric_name].to_numpy(dtype=float)
            mean_v, ci_low, ci_high = _mean_bootstrap_ci(vals, seed=args.seed + len(metric_name), draws=args.bootstrap_draws)
            stats_rows.append(
                {
                    "analysis_scope": "static",
                    "comparison": label,
                    "metric": metric_name,
                    "estimate": mean_v,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": _sign_flip_pvalue(vals, alternative="greater"),
                    "replication_unit": "session",
                    "n_sessions": int(np.sum(np.isfinite(vals))),
                }
            )
        for a, b, label in [
            ("non_targeted_2bin_pca6", "targeted_2bin_pca4", "non_targeted minus targeted"),
            ("non_targeted_2bin_pca6", "mixed_2bin_pca6", "non_targeted minus mixed"),
        ]:
            a_df = static_session_df[static_session_df["variant"] == a][["session_id", metric_name]].rename(columns={metric_name: "a"})
            b_df = static_session_df[static_session_df["variant"] == b][["session_id", metric_name]].rename(columns={metric_name: "b"})
            merged = a_df.merge(b_df, on="session_id", how="inner")
            diff = (merged["a"] - merged["b"]).to_numpy(dtype=float)
            mean_v, ci_low, ci_high = _mean_bootstrap_ci(diff, seed=args.seed + len(label), draws=args.bootstrap_draws)
            stats_rows.append(
                {
                    "analysis_scope": "static",
                    "comparison": label,
                    "metric": metric_name,
                    "estimate": mean_v,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": _sign_flip_pvalue(diff, alternative="two-sided"),
                    "replication_unit": "session",
                    "n_sessions": int(np.sum(np.isfinite(diff))),
                }
            )

    # Dose summary from the chosen main latent variant.
    main_full = static_metrics_df[
        (static_metrics_df["variant"] == MAIN_VARIANT)
        & (static_metrics_df["model"] == "full_vine")
        & (static_metrics_df["fit_status"] == "success")
    ].copy()
    trunc_df = static_metrics_df[(static_metrics_df["variant"] == MAIN_VARIANT) & (static_metrics_df["model"] == "truncated_vine")][["session_id", "dose", "slice_key", "heldout_nll"]].rename(columns={"heldout_nll": "trunc_nll"})
    gauss_df = static_metrics_df[(static_metrics_df["variant"] == MAIN_VARIANT) & (static_metrics_df["model"] == "gaussian")][["session_id", "dose", "slice_key", "heldout_nll"]].rename(columns={"heldout_nll": "gaussian_nll"})
    full_df = main_full[["session_id", "dose", "slice_key", "heldout_nll", "delta_vs_gaussian", "full_delta_vs_trunc", "tc_higher"]].rename(columns={"heldout_nll": "full_nll"})
    dose_join = gauss_df.merge(trunc_df, on=["session_id", "dose", "slice_key"]).merge(full_df, on=["session_id", "dose", "slice_key"])
    dose_session = dose_join.groupby(["session_id", "dose"]).agg(
        full_delta_vs_gaussian=("delta_vs_gaussian", "mean"),
        full_delta_vs_trunc=("full_delta_vs_trunc", "mean"),
        tc_higher=("tc_higher", "mean"),
        gaussian_nll=("gaussian_nll", "mean"),
        trunc_nll=("trunc_nll", "mean"),
        full_nll=("full_nll", "mean"),
        n_slices=("slice_key", "nunique"),
    ).reset_index()
    dose_rows: List[Dict[str, Any]] = []
    for row in dose_session.to_dict("records"):
        row["scope"] = "session"
        dose_rows.append(row)
    for dose, group in dose_session.groupby("dose"):
        mean_fg, low_fg, high_fg = _mean_bootstrap_ci(group["full_delta_vs_gaussian"], seed=args.seed + int(round(float(dose))), draws=args.bootstrap_draws)
        mean_tc, low_tc, high_tc = _mean_bootstrap_ci(group["tc_higher"], seed=args.seed + 1000 + int(round(float(dose))), draws=args.bootstrap_draws)
        dose_rows.append(
            {
                "scope": "pooled",
                "session_id": "",
                "dose": float(dose),
                "mean_full_delta_vs_gaussian": mean_fg,
                "ci_low_full_delta_vs_gaussian": low_fg,
                "ci_high_full_delta_vs_gaussian": high_fg,
                "mean_tc_higher": mean_tc,
                "ci_low_tc_higher": low_tc,
                "ci_high_tc_higher": high_tc,
                "p_value_full_delta_positive": _sign_flip_pvalue(group["full_delta_vs_gaussian"], alternative="greater"),
                "p_value_tc_positive": _sign_flip_pvalue(group["tc_higher"], alternative="greater"),
                "n_sessions": int(group["session_id"].nunique()),
            }
        )
        stats_rows.extend(
            [
                {
                    "analysis_scope": "dose",
                    "comparison": f"dose {int(round(float(dose)))} positivity",
                    "metric": "full_delta_vs_gaussian",
                    "estimate": mean_fg,
                    "ci_low": low_fg,
                    "ci_high": high_fg,
                    "p_value": _sign_flip_pvalue(group["full_delta_vs_gaussian"], alternative="greater"),
                    "replication_unit": "session",
                    "n_sessions": int(group["session_id"].nunique()),
                },
                {
                    "analysis_scope": "dose",
                    "comparison": f"dose {int(round(float(dose)))} positivity",
                    "metric": "tc_higher",
                    "estimate": mean_tc,
                    "ci_low": low_tc,
                    "ci_high": high_tc,
                    "p_value": _sign_flip_pvalue(group["tc_higher"], alternative="greater"),
                    "replication_unit": "session",
                    "n_sessions": int(group["session_id"].nunique()),
                },
            ]
        )
    dose_summary_df = pd.DataFrame(dose_rows)

    # Family summaries from the static runs.
    family_df = pd.DataFrame(family_rows)
    if not family_df.empty:
        # source-space grouped at static level
        source_grouped = _family_mix_summary(
            family_df.assign(summary_level="grouped_by_source"),
            ["analysis_scope", "summary_level", "variant", "source_space"],
        )
        tree_grouped = _family_mix_summary(
            family_df[family_df["variant"] == MAIN_VARIANT].assign(analysis_scope="static_main", summary_level="grouped_by_tree"),
            ["analysis_scope", "summary_level", "tree_level"],
        )
        dose_grouped = _family_mix_summary(
            family_df[family_df["variant"] == MAIN_VARIANT].assign(analysis_scope="static_main", summary_level="grouped_by_dose"),
            ["analysis_scope", "summary_level", "dose"],
        )
        family_summary_df = pd.concat([source_grouped, tree_grouped, dose_grouped], axis=0, ignore_index=True)
    else:
        family_summary_df = pd.DataFrame()

    # Dynamic rolling-window sensitivity on the main variant.
    dynamic_rows: List[Dict[str, Any]] = []
    dynamic_family_rows: List[Dict[str, Any]] = []
    for session_id, cache in session_cache.items():
        source_idx = _source_indices(np.asarray(cache["targeted_mask"], dtype=bool), MAIN_SOURCE_SPACE)
        if source_idx.size < MAIN_N_COMPONENTS:
            continue
        source_matrix = _build_source_matrix(cache, source_idx)
        session_df = cache["session_df"].copy()
        dose_counts = session_df.groupby("dose").size().sort_index()
        min_trials_needed = int(args.dynamic_window_size) + int(args.dynamic_step) * (int(args.dynamic_min_windows) - 1)
        eligible = dose_counts[dose_counts >= min_trials_needed]
        if eligible.empty:
            continue
        dose = float(np.max(eligible.index.to_numpy(dtype=float)))
        dose_df = session_df[session_df["dose"] == dose].sort_values("trial_order_within_session").copy()
        idx = dose_df.index.to_numpy(dtype=int)
        starts = list(range(0, len(idx) - int(args.dynamic_window_size) + 1, int(args.dynamic_step)))
        if len(starts) < int(args.dynamic_min_windows):
            continue

        all_x = source_matrix[idx]
        common_mean, common_std, common_components, common_explained = _fit_pca_basis(all_x, MAIN_N_COMPONENTS)
        for window_rank, start in enumerate(starts):
            window_idx = idx[start : start + int(args.dynamic_window_size)]
            rng = np.random.default_rng(_session_seed(session_id, int(args.seed), extra=int(round(dose)) + 19 * window_rank))
            perm = rng.permutation(len(window_idx))
            n_train = min(max(int(math.floor(float(args.train_fraction) * len(window_idx))), 5), len(window_idx) - 1)
            train_pos = np.sort(perm[:n_train])
            test_pos = np.sort(perm[n_train:])
            if len(train_pos) < 5 or len(test_pos) < 3:
                continue
            train_idx = np.asarray(window_idx[train_pos], dtype=int)
            test_idx = np.asarray(window_idx[test_pos], dtype=int)

            for basis_mode in ["window_train_basis", "common_basis_descriptive"]:
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
                except Exception as exc:
                    LOGGER.debug("Dynamic PCA failed for %s %s %s: %s", session_id, dose, basis_mode, exc)
                    continue
                scored = _fit_and_score_with_vines(
                    train_x=train_scores,
                    test_x=test_scores,
                    families=families,
                    seed=int(args.seed) + window_rank + (0 if basis_mode == "window_train_basis" else 1000),
                )
                tc = scored["trunc_nll"] - scored["full_nll"] if np.isfinite(scored["trunc_nll"]) and np.isfinite(scored["full_nll"]) else np.nan
                latent_var = np.nanmean(np.var(np.vstack([train_scores, test_scores]), axis=0, ddof=0))
                common = {
                    "analysis_scope": "dynamic",
                    "variant": MAIN_VARIANT,
                    "basis_mode": basis_mode,
                    "session_id": session_id,
                    "dose": float(dose),
                    "window_rank": int(window_rank),
                    "window_center_trial_order": float(dose_df.iloc[start : start + int(args.dynamic_window_size)]["trial_order_within_session"].mean()),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "pca_variance_retained": float(np.sum(explained)),
                    "latent_marginal_variance": float(latent_var),
                }
                dynamic_rows.extend(
                    [
                        {**common, "model": "gaussian", "heldout_nll": scored["gaussian_nll"], "delta_vs_gaussian": 0.0, "tc_higher": np.nan, "fit_status": scored["gaussian_status"]},
                        {**common, "model": "truncated_vine", "heldout_nll": scored["trunc_nll"], "delta_vs_gaussian": scored["gaussian_nll"] - scored["trunc_nll"] if np.isfinite(scored["gaussian_nll"]) and np.isfinite(scored["trunc_nll"]) else np.nan, "tc_higher": tc, "fit_status": scored["trunc_status"]},
                        {**common, "model": "full_vine", "heldout_nll": scored["full_nll"], "delta_vs_gaussian": scored["gaussian_nll"] - scored["full_nll"] if np.isfinite(scored["gaussian_nll"]) and np.isfinite(scored["full_nll"]) else np.nan, "tc_higher": tc, "fit_status": scored["full_status"]},
                    ]
                )
                dynamic_family_common = {
                    "analysis_scope": "dynamic",
                    "summary_level": "raw_dynamic",
                    "variant": MAIN_VARIANT,
                    "basis_mode": basis_mode,
                    "session_id": session_id,
                    "dose": float(dose),
                    "window_rank": int(window_rank),
                }
                dynamic_family_rows.extend(_iter_vine_family_rows(scored["full_vine"], dynamic_family_common))

    dynamic_df = pd.DataFrame(dynamic_rows)
    if not dynamic_df.empty:
        dynamic_full = dynamic_df[(dynamic_df["model"] == "full_vine") & (dynamic_df["fit_status"] == "success")].copy()
        dynamic_agg_df = dynamic_full.groupby(["basis_mode", "session_id", "window_rank"]).agg(
            full_delta_vs_gaussian=("delta_vs_gaussian", "mean"),
            tc_higher=("tc_higher", "mean"),
            heldout_nll=("heldout_nll", "mean"),
            pca_variance_retained=("pca_variance_retained", "mean"),
            latent_marginal_variance=("latent_marginal_variance", "mean"),
            dose=("dose", "mean"),
            window_center_trial_order=("window_center_trial_order", "mean"),
        ).reset_index()
        agree_rows = []
        for (session_id, window_rank), group in dynamic_agg_df.groupby(["session_id", "window_rank"]):
            if {"window_train_basis", "common_basis_descriptive"} <= set(group["basis_mode"]):
                a = float(group.loc[group["basis_mode"] == "window_train_basis", "full_delta_vs_gaussian"].iloc[0])
                b = float(group.loc[group["basis_mode"] == "common_basis_descriptive", "full_delta_vs_gaussian"].iloc[0])
                at = float(group.loc[group["basis_mode"] == "window_train_basis", "tc_higher"].iloc[0])
                bt = float(group.loc[group["basis_mode"] == "common_basis_descriptive", "tc_higher"].iloc[0])
                agree_rows.append(
                    {
                        "analysis_scope": "dynamic",
                        "comparison": "basis agreement",
                        "metric": "full_delta_vs_gaussian_sign_agreement",
                        "estimate": float(np.sign(a) == np.sign(b)),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                        "replication_unit": "session_window",
                        "n_sessions": 1,
                    }
                )
                agree_rows.append(
                    {
                        "analysis_scope": "dynamic",
                        "comparison": "basis agreement",
                        "metric": "tc_higher_sign_agreement",
                        "estimate": float(np.sign(at) == np.sign(bt)),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "p_value": np.nan,
                        "replication_unit": "session_window",
                        "n_sessions": 1,
                    }
                )
        stats_rows.extend(agree_rows)
    else:
        dynamic_agg_df = pd.DataFrame()

    if dynamic_family_rows:
        dynamic_family_df = pd.DataFrame(dynamic_family_rows)
        dynamic_grouped = _family_mix_summary(
            dynamic_family_df.assign(summary_level="grouped_by_dynamic"),
            ["analysis_scope", "summary_level", "basis_mode", "window_rank"],
        )
        family_summary_df = pd.concat([family_summary_df, dynamic_grouped], axis=0, ignore_index=True) if not family_summary_df.empty else dynamic_grouped

    # PC interpretability summaries and session associations.
    pc_main = latent_interpret_existing[latent_interpret_existing["variant"] == MAIN_VARIANT].copy()
    pc_plot_df = pc_main.groupby("pc_index").agg(
        mean_explained_variance=("explained_variance", "mean"),
        mean_loading_stability=("loading_stability_abs_cosine", "mean"),
    ).reset_index()
    pc_session_df = pc_main[pc_main["pc_index"] == 1].groupby("session_id").agg(
        pc1_explained_variance=("explained_variance", "mean"),
        pc1_loading_stability=("loading_stability_abs_cosine", "mean"),
        pc1_post_weight_fraction=("post_weight_fraction", "mean"),
        pc1_target_proximity_delta_px=("top_loading_distance_delta_px", "mean"),
    ).reset_index()
    session_gain = static_session_df[static_session_df["variant"] == MAIN_VARIANT][["session_id", "full_delta_vs_gaussian", "tc_higher"]].copy()
    pc_session_df = pc_session_df.merge(session_gain, on="session_id", how="left")

    pc_summary_rows: List[Dict[str, Any]] = []
    for row in pc_plot_df.to_dict("records"):
        row["row_type"] = "pc_profile"
        pc_summary_rows.append(row)
    if not pc_session_df.empty:
        pc_summary_rows.extend(
            [
                {
                    "row_type": "association",
                    "association": "full_delta_vs_target_proximity",
                    "estimate": _rank_corr(pc_session_df["pc1_target_proximity_delta_px"], pc_session_df["full_delta_vs_gaussian"]),
                    "interpretation": "positive means stronger gains with more distributed/farther-from-target PC1 modes; negative means stronger gains with more target-proximal modes",
                },
                {
                    "row_type": "association",
                    "association": "full_delta_vs_pc1_stability",
                    "estimate": _rank_corr(pc_session_df["pc1_loading_stability"], pc_session_df["full_delta_vs_gaussian"]),
                    "interpretation": "positive means stronger gains with more stable leading latent modes",
                },
            ]
        )
    pc_summary_df = pd.DataFrame(pc_summary_rows)

    # Final paper recommendation.
    static_nt = static_variant_df[static_variant_df["variant"] == MAIN_VARIANT]
    static_mean = float(static_nt["mean_full_delta_vs_gaussian"].iloc[0]) if not static_nt.empty else np.nan
    static_tc = float(static_nt["mean_tc_higher"].iloc[0]) if not static_nt.empty else np.nan
    targeted_mean = float(static_variant_df[static_variant_df["variant"] == "targeted_2bin_pca4"]["mean_full_delta_vs_gaussian"].iloc[0]) if not static_variant_df[static_variant_df["variant"] == "targeted_2bin_pca4"].empty else np.nan
    dynamic_agreement = np.nan
    if dynamic_agg_df is not None and not dynamic_agg_df.empty:
        merged = dynamic_agg_df.pivot_table(index=["session_id", "window_rank"], columns="basis_mode", values="full_delta_vs_gaussian")
        if {"window_train_basis", "common_basis_descriptive"} <= set(merged.columns):
            dynamic_agreement = float(np.mean(np.sign(merged["window_train_basis"]) == np.sign(merged["common_basis_descriptive"])))
    paper_decision = "supplement/control only"
    targeted_gap_p = np.nan
    targeted_gap = np.nan
    targeted_gap_row = stats_df[
        (stats_df["analysis_scope"] == "static")
        & (stats_df["comparison"] == "non_targeted minus targeted")
        & (stats_df["metric"] == "full_delta_vs_gaussian")
    ]
    if not targeted_gap_row.empty:
        targeted_gap = float(targeted_gap_row["estimate"].iloc[0])
        targeted_gap_p = float(targeted_gap_row["p_value"].iloc[0])
    if np.isfinite(static_mean) and np.isfinite(static_tc) and np.isfinite(targeted_gap):
        if static_mean > 0.30 and static_tc > 0.08 and targeted_gap > 0.15 and (not np.isfinite(targeted_gap_p) or targeted_gap_p < 0.05):
            paper_decision = "main-text ready"
        elif static_mean < 0.05:
            paper_decision = "not worth carrying forward"

    stats_df = pd.DataFrame(stats_rows)

    # Write outputs.
    static_out = pd.concat(
        [
            static_session_df.assign(row_type="session_summary"),
            static_variant_df.assign(row_type="variant_summary"),
        ],
        ignore_index=True,
        sort=False,
    )
    outputs = {
        "latent_followup_static_summary.csv": static_out,
        "latent_followup_dose_summary.csv": dose_summary_df,
        "latent_followup_dynamic_summary.csv": dynamic_agg_df,
        "latent_followup_family_summary.csv": family_summary_df,
        "latent_followup_pc_summary.csv": pc_summary_df,
        "latent_followup_stats_summary.csv": stats_df,
    }
    for filename, frame in outputs.items():
        frame.to_csv(data_dir / filename, index=False)
        frame.to_csv(out_root / filename, index=False)

    _plot_main_figure(static_session_df, dose_summary_df, pc_plot_df, plots_dir / "fig_latent_publication_main.png")
    _plot_dynamic(dynamic_agg_df, plots_dir / "fig_latent_publication_dynamic.png")
    _plot_family_usage(family_summary_df, plots_dir / "fig_latent_publication_family_usage.png")
    _plot_interpretability_extra(pc_session_df, plots_dir / "fig_latent_publication_interpretability_extra.png")

    metadata = {
        "family_variant": args.family_variant,
        "static_variants": [cfg["variant"] for cfg in STATIC_VARIANTS],
        "main_variant": MAIN_VARIANT,
        "main_source_space": MAIN_SOURCE_SPACE,
        "dynamic_window_size": int(args.dynamic_window_size),
        "dynamic_step": int(args.dynamic_step),
        "dynamic_basis_modes": ["window_train_basis", "common_basis_descriptive"],
        "behavior_status": "not feasible with current validated data path",
        "paper_decision": paper_decision,
        "dynamic_basis_agreement_full_delta": dynamic_agreement,
    }
    _write_json(data_dir / "latent_followup_metadata.json", metadata)
    _write_json(out_root / "latent_followup_metadata.json", metadata)
    LOGGER.info("Latent follow-up complete | paper decision: %s", paper_decision)


if __name__ == "__main__":
    main()
