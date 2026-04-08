#!/usr/bin/env python3
"""Full latent-state DVC run for the Dalgleish photostimulation dataset."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.stimulation_exp_benchmark.build_dalgleish_dvc_dataset as builder
from scripts.debug_stimulation_exp.run_dalgleish_real_data_benchmark import FAMILY_VARIANTS, _write_json, configure_logging
from scripts.debug_stimulation_exp.run_dalgleish_formulation_viability import (
    _prepare_session_cache,
    _build_split_plan,
    _fit_pca_train_apply,
    _evaluate_features_with_status,
    DELAYED_WINDOW,
    POST_WINDOW,
)


LOGGER = logging.getLogger("dalgleish_latent_state_full_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dalgleish latent-state full analysis.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--results_root", default="results/stimulation_exp_benchmark")
    parser.add_argument("--family_variant", choices=sorted(FAMILY_VARIANTS.keys()), default="stable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--min_trials_floor", type=int, default=18)
    parser.add_argument("--dynamic_min_block_trials", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _latent_feature_semantics_row(variant: str, source_space: str, n_components: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    source_note = {
        "non_targeted": "all non-targeted usable neurons",
        "targeted": "all targeted neurons mapped in the session",
        "mixed": "all targeted and non-targeted usable neurons",
    }[source_space]
    for pc in range(1, n_components + 1):
        rows.append(
            {
                "variant": variant,
                "source_space": source_space,
                "feature_index": pc,
                "feature_name": f"PC{pc}",
                "semantics": f"train-fit PCA latent population-state component {pc} from {source_note} using delayed and early-post spks bins",
                "start_s": float(DELAYED_WINDOW[0]),
                "end_s": float(POST_WINDOW[1]),
                "source_pool": source_note,
                "latent_component": pc,
            }
        )
    return rows


def _behavior_feasibility(data_root: Path) -> Dict[str, Any]:
    varfiles = sorted(data_root.glob("*/targets/*BhvTraining*VarFile*.mat"))
    processed_summary = sorted(data_root.glob("**/DalgleishHausser2020_imaging_raw.mat"))
    if processed_summary:
        status = "behavior is available and reliable enough to include"
        reason = "Processed imaging summary with documented trial-level behavior appears to be present."
    elif not varfiles:
        status = "behavior is not feasible without a new extraction layer"
        reason = "No behavior sidecar files were found under the raw session directories."
    else:
        # Raw sidecars exist but current validated builder does not export trial-level responses or reaction times.
        status = "behavior is not feasible without a new extraction layer"
        reason = (
            "Behavior/training VarFiles are present and expose protocol-level arrays such as laser_trials, pyb, and stim, "
            "but the current validated builder does not export trial-level response or reaction-time variables, and the "
            "processed DalgleishHausser2020_imaging_raw.mat summary documented in the authors' repo is not present locally."
        )
    return {
        "status": status,
        "n_varfiles_found": int(len(varfiles)),
        "n_processed_behavior_summaries_found": int(len(processed_summary)),
        "reason": reason,
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


def _component_summaries(
    components: np.ndarray,
    n_source: int,
    source_space: str,
    source_targeted_mask: np.ndarray,
    roi_lookup: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if components.size == 0:
        return rows
    source_targeted_mask = np.asarray(source_targeted_mask, dtype=bool)
    roi_lookup = roi_lookup.reset_index(drop=True)
    if source_space == "targeted":
        source_targeted_mask = np.ones(n_source, dtype=bool)
    elif source_space == "non_targeted":
        source_targeted_mask = np.zeros(n_source, dtype=bool)
    for comp_idx in range(components.shape[0]):
        w = np.asarray(components[comp_idx], dtype=np.float64)
        delayed_w = w[:n_source]
        post_w = w[n_source:]
        delayed_abs = float(np.sum(np.abs(delayed_w)))
        post_abs = float(np.sum(np.abs(post_w)))
        total_abs = delayed_abs + post_abs
        post_weight_fraction = post_abs / total_abs if total_abs > 0 else np.nan
        per_neuron_abs = np.abs(delayed_w) + np.abs(post_w)
        total_neuron_abs = float(np.sum(per_neuron_abs))
        if total_neuron_abs > 0:
            targeted_weight_fraction = float(np.sum(per_neuron_abs[source_targeted_mask]) / total_neuron_abs)
        else:
            targeted_weight_fraction = np.nan
        targeted_fraction_base = float(np.mean(source_targeted_mask)) if source_targeted_mask.size else np.nan
        if np.isfinite(targeted_weight_fraction) and np.isfinite(targeted_fraction_base) and targeted_fraction_base > 0:
            targeted_enrichment = float(targeted_weight_fraction / targeted_fraction_base)
        else:
            targeted_enrichment = np.nan

        top_loading_distance_delta = np.nan
        if source_space in {"non_targeted", "mixed"} and not roi_lookup.empty:
            target_df = roi_lookup[source_targeted_mask].copy()
            if not target_df.empty and {"x_center", "y_center"}.issubset(roi_lookup.columns):
                src_df = roi_lookup.copy()
                src_xy = src_df[["x_center", "y_center"]].to_numpy(dtype=float)
                target_xy = target_df[["x_center", "y_center"]].to_numpy(dtype=float)
                if np.isfinite(src_xy).all() and np.isfinite(target_xy).all():
                    diff = src_xy[:, None, :] - target_xy[None, :, :]
                    min_dist = np.sqrt(np.sum(diff ** 2, axis=2)).min(axis=1)
                    k_top = min(max(10, int(math.ceil(0.05 * n_source))), n_source)
                    top_idx = np.argsort(per_neuron_abs)[::-1][:k_top]
                    top_loading_distance_delta = float(np.mean(min_dist[top_idx]) - np.mean(min_dist))

        rows.append(
            {
                "pc_index": int(comp_idx + 1),
                "post_weight_fraction": post_weight_fraction,
                "targeted_weight_fraction": targeted_weight_fraction,
                "targeted_enrichment": targeted_enrichment,
                "top_loading_distance_delta_px": top_loading_distance_delta,
            }
        )
    return rows


def _component_stability(components_a: np.ndarray, components_b: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    k = min(components_a.shape[0], components_b.shape[0])
    for comp_idx in range(k):
        a = np.asarray(components_a[comp_idx], dtype=np.float64)
        b = np.asarray(components_b[comp_idx], dtype=np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        sim = np.nan if denom <= 0 else float(np.abs(np.dot(a, b) / denom))
        rows.append({"pc_index": int(comp_idx + 1), "loading_stability_abs_cosine": sim})
    return rows


def _score_summary(metrics_df: pd.DataFrame) -> Dict[str, float]:
    gaussian = metrics_df[metrics_df["model"] == "gaussian"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "g"})
    trunc = metrics_df[metrics_df["model"] == "truncated_vine"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "t"})
    full = metrics_df[metrics_df["model"] == "full_vine"][["split_id", "heldout_nll", "tc_higher"]].rename(columns={"heldout_nll": "f"})
    merged = gaussian.merge(trunc, on="split_id", how="inner").merge(full, on="split_id", how="inner")
    merged = merged[np.isfinite(merged["g"]) & np.isfinite(merged["t"]) & np.isfinite(merged["f"])].copy()
    if merged.empty:
        return {"score": -1e9, "mean_full_delta_vs_gaussian": np.nan, "mean_tc_higher": np.nan}
    full_delta = float(np.mean(merged["g"] - merged["f"]))
    tc_mean = float(np.mean(merged["tc_higher"]))
    prop_g = float(np.mean(merged["f"] < merged["g"]))
    prop_t = float(np.mean(merged["f"] < merged["t"]))
    score = full_delta + 0.35 * tc_mean + 0.10 * (prop_g - 0.5) + 0.05 * (prop_t - 0.5)
    return {"score": score, "mean_full_delta_vs_gaussian": full_delta, "mean_tc_higher": tc_mean}


def _summarize_latent_variant(metrics_df: pd.DataFrame, source_space: str, n_components: int, variant: str, scope: str) -> Dict[str, Any]:
    sub = metrics_df[(metrics_df["variant"] == variant) & (metrics_df["comparison_scope"] == scope)].copy()
    if sub.empty:
        return {
            "comparison_scope": scope,
            "variant": variant,
            "source_space": source_space,
            "n_components": n_components,
            "session_count": 0,
            "slice_count": 0,
            "gaussian_mean_nll": np.nan,
            "trunc_mean_nll": np.nan,
            "full_mean_nll": np.nan,
            "mean_full_delta_vs_gaussian": np.nan,
            "prop_full_beats_gaussian": np.nan,
            "prop_full_beats_trunc": np.nan,
            "mean_tc_higher": np.nan,
            "median_tc_higher": np.nan,
            "pca_mean_explained_variance_retained": np.nan,
            "full_failure_rate": np.nan,
        }
    gaussian = sub[sub["model"] == "gaussian"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "g"})
    trunc = sub[sub["model"] == "truncated_vine"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "t"})
    full = sub[sub["model"] == "full_vine"][["split_id", "heldout_nll", "tc_higher"]].rename(columns={"heldout_nll": "f"})
    merged = gaussian.merge(trunc, on="split_id").merge(full, on="split_id")
    merged = merged[np.isfinite(merged["g"]) & np.isfinite(merged["t"]) & np.isfinite(merged["f"])].copy()
    if merged.empty:
        full_delta = prop_g = prop_t = tc_mean = tc_median = np.nan
    else:
        full_delta = float(np.mean(merged["g"] - merged["f"]))
        prop_g = float(np.mean(merged["f"] < merged["g"]))
        prop_t = float(np.mean(merged["f"] < merged["t"]))
        tc_mean = float(np.mean(merged["tc_higher"]))
        tc_median = float(np.median(merged["tc_higher"]))
    full_rows = sub[sub["model"] == "full_vine"]
    failure_rate = 1.0 - float(np.mean(full_rows["fit_status"] == "success")) if not full_rows.empty else np.nan
    return {
        "comparison_scope": scope,
        "variant": variant,
        "source_space": source_space,
        "n_components": n_components,
        "session_count": int(sub["session_id"].nunique()),
        "slice_count": int(sub["split_id"].nunique()),
        "gaussian_mean_nll": float(sub[sub["model"] == "gaussian"]["heldout_nll"].mean()),
        "trunc_mean_nll": float(sub[sub["model"] == "truncated_vine"]["heldout_nll"].mean()),
        "full_mean_nll": float(sub[sub["model"] == "full_vine"]["heldout_nll"].mean()),
        "mean_full_delta_vs_gaussian": full_delta,
        "prop_full_beats_gaussian": prop_g,
        "prop_full_beats_trunc": prop_t,
        "mean_tc_higher": tc_mean,
        "median_tc_higher": tc_median,
        "pca_mean_explained_variance_retained": float(sub["pca_variance_retained"].dropna().mean()),
        "full_failure_rate": failure_rate,
    }


def _source_variant_name(source_space: str, n_components: int) -> str:
    return f"{source_space}_2bin_pca{n_components}"


def _plot_source_space(summary_df: pd.DataFrame, out_path: Path) -> None:
    sub = summary_df[(summary_df["comparison_scope"] == "native") & (summary_df["n_components"] == 4)].copy()
    order = ["non_targeted", "targeted", "mixed"]
    x = np.arange(len(order))
    full_delta = [float(sub.loc[sub["source_space"] == src, "mean_full_delta_vs_gaussian"].iloc[0]) for src in order]
    tc = [float(sub.loc[sub["source_space"] == src, "mean_tc_higher"].iloc[0]) for src in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x, full_delta, color="#4c78a8")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x, ["Non-targeted", "Targeted", "Mixed"])
    axes[0].set_ylabel("Full-vine delta vs Gaussian")
    axes[0].set_title("Source-space comparison at PCA rank 4")
    axes[1].bar(x, tc, color="#f58518")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x, ["Non-targeted", "Targeted", "Mixed"])
    axes[1].set_ylabel("TC_higher")
    axes[1].set_title("Higher-order gain by source space")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dose_summary(dose_df: pd.DataFrame, out_path: Path) -> None:
    pooled = dose_df[dose_df["scope"] == "pooled"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(pooled["dose"], pooled["full_delta_vs_gaussian"], marker="o", color="#1f77b4")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("Dose")
    axes[0].set_ylabel("Full-vine delta vs Gaussian")
    axes[0].set_title("Latent-state gain by dose")
    axes[1].plot(pooled["dose"], pooled["tc_higher"], marker="o", color="#d62728")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Dose")
    axes[1].set_ylabel("TC_higher")
    axes[1].set_title("Latent higher-order gain by dose")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_session_dose(dose_df: pd.DataFrame, out_path: Path) -> None:
    pooled = dose_df[dose_df["scope"] == "session"].copy()
    if pooled.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for session_id, group in pooled.groupby("session_id"):
        group = group.sort_values("dose")
        ax.plot(group["dose"], group["full_delta_vs_gaussian"], color="#bdbdbd", alpha=0.7)
    agg = pooled.groupby("dose")["full_delta_vs_gaussian"].mean().reset_index()
    ax.plot(agg["dose"], agg["full_delta_vs_gaussian"], color="#1f77b4", linewidth=2.5, marker="o")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Dose")
    ax.set_ylabel("Full-vine delta vs Gaussian")
    ax.set_title("Session-level robustness in latent space")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic(dynamic_df: pd.DataFrame, out_path: Path) -> None:
    if dynamic_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = {"early": 0, "middle": 1, "late": 2}
    dynamic_df = dynamic_df.copy()
    dynamic_df["block_order"] = dynamic_df["block_id"].map(order)
    full = dynamic_df[dynamic_df["model"] == "full_vine"].copy()
    for session_id, group in full.groupby("session_id"):
        group = group.sort_values("block_order")
        axes[0].plot(group["block_order"], group["delta_vs_gaussian"], color="#bdbdbd", alpha=0.7)
        axes[1].plot(group["block_order"], group["tc_higher"], color="#bdbdbd", alpha=0.7)
    agg = full.groupby("block_order")[["delta_vs_gaussian", "tc_higher"]].mean().reset_index()
    axes[0].plot(agg["block_order"], agg["delta_vs_gaussian"], color="#1f77b4", linewidth=2.5, marker="o")
    axes[1].plot(agg["block_order"], agg["tc_higher"], color="#d62728", linewidth=2.5, marker="o")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks([0, 1, 2], ["early", "middle", "late"])
    axes[0].set_ylabel("Full-vine delta vs Gaussian")
    axes[0].set_title("Latent dynamic gain")
    axes[1].set_ylabel("TC_higher")
    axes[1].set_title("Latent dynamic higher-order gain")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_interpretability(int_df: pd.DataFrame, out_path: Path) -> None:
    if int_df.empty:
        return
    summary = int_df.groupby("pc_index").agg(
        explained_variance=("explained_variance", "mean"),
        loading_stability=("loading_stability_abs_cosine", "mean"),
        post_weight_fraction=("post_weight_fraction", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    axes[0].bar(summary["pc_index"], summary["explained_variance"], color="#4c78a8")
    axes[0].set_title("Variance Explained")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Mean explained variance")
    axes[1].bar(summary["pc_index"], summary["loading_stability"], color="#54a24b")
    axes[1].set_title("Loading Stability")
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("Abs cosine across repeats")
    axes[2].bar(summary["pc_index"], summary["post_weight_fraction"], color="#f58518")
    axes[2].axhline(0.5, color="black", linewidth=1)
    axes[2].set_title("Temporal Weight Balance")
    axes[2].set_xlabel("PC")
    axes[2].set_ylabel("Post-bin weight fraction")
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

    behavior_status = _behavior_feasibility(data_root)
    LOGGER.info("Behavior feasibility: %s", behavior_status["status"])
    behavior_df = pd.DataFrame([behavior_status])
    behavior_df.to_csv(data_dir / "latent_state_behavior_summary.csv", index=False)
    behavior_df.to_csv(out_root / "latent_state_behavior_summary.csv", index=False)

    manifest = builder.build_manifest(data_root)
    _trials_df, neural_data, _ = builder.build_trials(data_root, manifest)
    families = list(FAMILY_VARIANTS[args.family_variant])

    session_cache: Dict[str, Dict[str, Any]] = {}
    for session_id, payload in neural_data.items():
        if payload.get("used", False):
            cache = _prepare_session_cache(session_id, payload, data_root)
            cache["roi_lookup"] = payload["roi_lookup"].reset_index(drop=True).copy()
            session_cache[session_id] = cache

    # Mandatory source-space comparison first.
    source_screen = [
        {"source_space": "non_targeted", "n_components": 4},
        {"source_space": "targeted", "n_components": 4},
        {"source_space": "mixed", "n_components": 4},
    ]

    metrics_rows: List[Dict[str, Any]] = []
    loadings_meta: List[Dict[str, Any]] = []
    semantics_rows: List[Dict[str, Any]] = []
    source_variant_keys: List[str] = []

    def run_variant(source_space: str, n_components: int) -> str:
        variant = _source_variant_name(source_space, n_components)
        source_variant_keys.append(variant)
        semantics_rows.extend(_latent_feature_semantics_row(variant, source_space, n_components))
        for session_id, cache in session_cache.items():
            source_idx = _source_indices(np.asarray(cache["targeted_mask"], dtype=bool), source_space)
            if source_idx.size < n_components:
                continue
            source_matrix = _build_source_matrix(cache, source_idx)
            source_targeted_mask = np.asarray(cache["targeted_mask"][source_idx], dtype=bool)
            split_plan = _build_split_plan(
                session_df=cache["session_df"],
                feature_dim=int(n_components),
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
                    train_scores, test_scores, explained, components = _fit_pca_with_components(
                        train_x=source_matrix[train_idx],
                        test_x=source_matrix[test_idx],
                        n_components=int(n_components),
                    )
                except Exception as exc:
                    LOGGER.debug("PCA failed for %s %s: %s", variant, split["slice_key"], exc)
                    continue
                score = _evaluate_features_with_status(
                    train_x=train_scores,
                    test_x=test_scores,
                    families=families,
                    seed=int(args.seed) + int(split["repeat_id"]) + int(n_components),
                )
                common = {
                    "analysis_view": "dose_static",
                    "comparison_scope": "native",
                    "variant": variant,
                    "source_space": source_space,
                    "n_components": int(n_components),
                    "session_id": session_id,
                    "dose": float(split["dose"]),
                    "repeat_id": int(split["repeat_id"]),
                    "slice_key": str(split["slice_key"]),
                    "split_id": f"{variant}__{split['slice_key']}",
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "pca_variance_retained": float(np.sum(explained)),
                    "fit_status": "",
                }
                tc = score["trunc_nll"] - score["full_nll"] if np.isfinite(score["trunc_nll"]) and np.isfinite(score["full_nll"]) else np.nan
                metrics_rows.extend(
                    [
                        {**common, "model": "gaussian", "heldout_nll": score["gaussian_nll"], "delta_vs_gaussian": 0.0, "tc_higher": np.nan, "fit_status": score["gaussian_status"]},
                        {**common, "model": "truncated_vine", "heldout_nll": score["trunc_nll"], "delta_vs_gaussian": score["gaussian_nll"] - score["trunc_nll"] if np.isfinite(score["gaussian_nll"]) and np.isfinite(score["trunc_nll"]) else np.nan, "tc_higher": tc, "fit_status": score["trunc_status"]},
                        {**common, "model": "full_vine", "heldout_nll": score["full_nll"], "delta_vs_gaussian": score["gaussian_nll"] - score["full_nll"] if np.isfinite(score["gaussian_nll"]) and np.isfinite(score["full_nll"]) else np.nan, "tc_higher": tc, "fit_status": score["full_status"]},
                    ]
                )
                comp_rows = _component_summaries(
                    components=components,
                    n_source=source_idx.size,
                    source_space=source_space,
                    source_targeted_mask=source_targeted_mask,
                    roi_lookup=cache["roi_lookup"].iloc[source_idx].reset_index(drop=True),
                )
                for row in comp_rows:
                    loadings_meta.append(
                        {
                            "variant": variant,
                            "source_space": source_space,
                            "n_components": int(n_components),
                            "session_id": session_id,
                            "dose": float(split["dose"]),
                            "repeat_id": int(split["repeat_id"]),
                            "slice_base_key": f"{session_id}__dose_{int(round(float(split['dose']))):03d}",
                            "split_id": f"{variant}__{split['slice_key']}",
                            "pc_index": row["pc_index"],
                            "explained_variance": float(explained[row["pc_index"] - 1]),
                            "cumulative_variance_retained": float(np.sum(explained[: row["pc_index"]])),
                            "post_weight_fraction": row["post_weight_fraction"],
                            "targeted_weight_fraction": row["targeted_weight_fraction"],
                            "targeted_enrichment": row["targeted_enrichment"],
                            "top_loading_distance_delta_px": row["top_loading_distance_delta_px"],
                            "components": components[row["pc_index"] - 1].tolist(),
                        }
                    )
        LOGGER.info("Source variant complete: %s", variant)
        return variant

    for cfg in source_screen:
        run_variant(cfg["source_space"], cfg["n_components"])

    metrics_df = pd.DataFrame(metrics_rows)
    loading_df = pd.DataFrame(loadings_meta)

    source_summary_rows: List[Dict[str, Any]] = []
    for cfg in source_screen:
        variant = _source_variant_name(cfg["source_space"], cfg["n_components"])
        source_summary_rows.append(_summarize_latent_variant(metrics_df, cfg["source_space"], cfg["n_components"], variant, "native"))
    source_summary_df = pd.DataFrame(source_summary_rows)

    # Choose best source and then rank sensitivity on best source plus non-targeted baseline if different.
    best_source_row = source_summary_df.sort_values(
        ["mean_full_delta_vs_gaussian", "mean_tc_higher", "pca_mean_explained_variance_retained"],
        ascending=[False, False, False],
    ).iloc[0]
    best_source_space = str(best_source_row["source_space"])
    rank_sources = [best_source_space]
    if best_source_space != "non_targeted":
        rank_sources.append("non_targeted")

    for src in rank_sources:
        if src == best_source_space:
            run_variant(src, 6)
        elif src == "non_targeted":
            run_variant(src, 6)

    metrics_df = pd.DataFrame(metrics_rows)
    loading_df = pd.DataFrame(loadings_meta)

    # Common-slice comparison mandatory across pca4 source-space screen.
    rep_variants = [_source_variant_name("non_targeted", 4), _source_variant_name("targeted", 4), _source_variant_name("mixed", 4)]
    common_sets = []
    for variant in rep_variants:
        successful = metrics_df[(metrics_df["variant"] == variant) & (metrics_df["model"] == "full_vine") & (metrics_df["fit_status"] == "success")]["slice_key"]
        common_sets.append(set(successful.tolist()))
    common_intersection = set.intersection(*common_sets) if common_sets else set()
    metrics_df["on_common_slice"] = metrics_df["slice_key"].isin(common_intersection)

    source_summary_rows = []
    for variant in sorted(set(source_variant_keys)):
        src, rank_str = variant.rsplit("_pca", 1)
        n_comp = int(rank_str)
        source_summary_rows.append(_summarize_latent_variant(metrics_df, src.replace("_2bin", ""), n_comp, variant, "native"))
        tmp = metrics_df[(metrics_df["variant"] == variant) & (metrics_df["on_common_slice"])].copy()
        if not tmp.empty:
            tmp.loc[:, "comparison_scope"] = "common_slice"
            source_summary_rows.append(_summarize_latent_variant(tmp, src.replace("_2bin", ""), n_comp, variant, "common_slice"))
    source_summary_df = pd.DataFrame(source_summary_rows)

    # Choose main latent variant among executed variants.
    variant_scores: Dict[str, Dict[str, float]] = {}
    for variant in sorted(set(source_variant_keys)):
        variant_scores[variant] = _score_summary(metrics_df[metrics_df["variant"] == variant].copy())
    main_variant = max(variant_scores.items(), key=lambda kv: kv[1]["score"])[0]
    main_row = source_summary_df[(source_summary_df["variant"] == main_variant) & (source_summary_df["comparison_scope"] == "native")].iloc[0]
    main_source_space = str(main_row["source_space"])
    main_n_components = int(main_row["n_components"])
    LOGGER.info("Chosen main latent variant: %s", main_variant)

    # Loading stability across repeats for executed variants.
    interpret_rows: List[Dict[str, Any]] = []
    if not loading_df.empty:
        component_store = {}
        for row in loadings_meta:
            key = (row["variant"], row["session_id"], row["dose"], row["repeat_id"])
            component_store.setdefault(key, {})[int(row["pc_index"])] = np.asarray(row["components"], dtype=np.float64)
        for variant in sorted(set(source_variant_keys)):
            var_rows = loading_df[loading_df["variant"] == variant].copy()
            src = str(var_rows["source_space"].iloc[0]) if not var_rows.empty else ""
            n_comp = int(var_rows["n_components"].iloc[0]) if not var_rows.empty else 0
            for (session_id, dose), group in var_rows.groupby(["session_id", "dose"]):
                reps = sorted(group["repeat_id"].unique().tolist())
                if len(reps) < 2:
                    continue
                comps_a = np.vstack([component_store[(variant, session_id, dose, reps[0])][pc] for pc in range(1, n_comp + 1)])
                comps_b = np.vstack([component_store[(variant, session_id, dose, reps[1])][pc] for pc in range(1, n_comp + 1)])
                stability = _component_stability(comps_a, comps_b)
                for item in stability:
                    group_pc = group[group["pc_index"] == item["pc_index"]]
                    interpret_rows.append(
                        {
                            "variant": variant,
                            "source_space": src,
                            "n_components": n_comp,
                            "session_id": session_id,
                            "dose": float(dose),
                            "pc_index": int(item["pc_index"]),
                            "explained_variance": float(group_pc["explained_variance"].mean()),
                            "cumulative_variance_retained": float(group_pc["cumulative_variance_retained"].mean()),
                            "loading_stability_abs_cosine": item["loading_stability_abs_cosine"],
                            "post_weight_fraction": float(group_pc["post_weight_fraction"].mean()),
                            "targeted_weight_fraction": float(group_pc["targeted_weight_fraction"].mean()) if group_pc["targeted_weight_fraction"].notna().any() else np.nan,
                            "targeted_enrichment": float(group_pc["targeted_enrichment"].mean()) if group_pc["targeted_enrichment"].notna().any() else np.nan,
                            "top_loading_distance_delta_px": float(group_pc["top_loading_distance_delta_px"].mean()) if group_pc["top_loading_distance_delta_px"].notna().any() else np.nan,
                        }
                    )
    interpret_df = pd.DataFrame(interpret_rows)

    # Dose summaries for chosen main latent variant.
    main_metrics = metrics_df[metrics_df["variant"] == main_variant].copy()
    dose_rows: List[Dict[str, Any]] = []
    if not main_metrics.empty:
        gaussian = main_metrics[main_metrics["model"] == "gaussian"][["session_id", "dose", "split_id", "heldout_nll"]].rename(columns={"heldout_nll": "g"})
        trunc = main_metrics[main_metrics["model"] == "truncated_vine"][["session_id", "dose", "split_id", "heldout_nll"]].rename(columns={"heldout_nll": "t"})
        full = main_metrics[main_metrics["model"] == "full_vine"][["session_id", "dose", "split_id", "heldout_nll", "tc_higher"]].rename(columns={"heldout_nll": "f"})
        merged = gaussian.merge(trunc, on=["session_id", "dose", "split_id"]).merge(full, on=["session_id", "dose", "split_id"])
        merged = merged[np.isfinite(merged["g"]) & np.isfinite(merged["t"]) & np.isfinite(merged["f"])].copy()
        merged["full_delta_vs_gaussian"] = merged["g"] - merged["f"]
        merged["trunc_delta_vs_gaussian"] = merged["g"] - merged["t"]
        pooled = merged.groupby("dose").agg(
            full_delta_vs_gaussian=("full_delta_vs_gaussian", "mean"),
            trunc_delta_vs_gaussian=("trunc_delta_vs_gaussian", "mean"),
            tc_higher=("tc_higher", "mean"),
            gaussian_mean_nll=("g", "mean"),
            trunc_mean_nll=("t", "mean"),
            full_mean_nll=("f", "mean"),
            n_slices=("split_id", "nunique"),
        ).reset_index()
        for row in pooled.to_dict("records"):
            row["scope"] = "pooled"
            row["session_id"] = ""
            dose_rows.append(row)
        by_session = merged.groupby(["session_id", "dose"]).agg(
            full_delta_vs_gaussian=("full_delta_vs_gaussian", "mean"),
            trunc_delta_vs_gaussian=("trunc_delta_vs_gaussian", "mean"),
            tc_higher=("tc_higher", "mean"),
            gaussian_mean_nll=("g", "mean"),
            trunc_mean_nll=("t", "mean"),
            full_mean_nll=("f", "mean"),
            n_slices=("split_id", "nunique"),
        ).reset_index()
        for row in by_session.to_dict("records"):
            row["scope"] = "session"
            dose_rows.append(row)
    dose_df = pd.DataFrame(dose_rows)

    # Dynamic analysis for main variant at strongest eligible dose per session.
    dynamic_rows: List[Dict[str, Any]] = []
    for session_id, cache in session_cache.items():
        source_idx = _source_indices(np.asarray(cache["targeted_mask"], dtype=bool), main_source_space)
        if source_idx.size < main_n_components:
            continue
        source_matrix = _build_source_matrix(cache, source_idx)
        session_df = cache["session_df"].copy()
        dose_counts = session_df.groupby("dose").size().sort_index()
        eligible = dose_counts[dose_counts >= 3 * int(args.dynamic_min_block_trials)]
        if eligible.empty:
            continue
        dose = float(np.max(eligible.index.to_numpy(dtype=float)))
        dose_df_sess = session_df[session_df["dose"] == dose].sort_values("trial_order_within_session").copy()
        idx = dose_df_sess.index.to_numpy(dtype=int)
        blocks = np.array_split(idx, 3)
        if min(len(b) for b in blocks) < int(args.dynamic_min_block_trials):
            continue
        for block_id, block_idx in zip(["early", "middle", "late"], blocks):
            rng = np.random.default_rng(int(args.seed) + int(round(dose)) + len(block_id))
            # Reuse validated split helper semantics.
            from scripts.debug_stimulation_exp.run_dalgleish_real_data_benchmark import _split_positions_random as _split_block
            train_pos, test_pos = _split_block(len(block_idx), float(args.train_fraction), rng)
            train_idx = np.asarray(block_idx[train_pos], dtype=int)
            test_idx = np.asarray(block_idx[test_pos], dtype=int)
            if len(train_idx) < 5 or len(test_idx) < 3:
                continue
            try:
                train_scores, test_scores, explained, _components = _fit_pca_with_components(
                    train_x=source_matrix[train_idx],
                    test_x=source_matrix[test_idx],
                    n_components=int(main_n_components),
                )
            except Exception:
                continue
            score = _evaluate_features_with_status(
                train_x=train_scores,
                test_x=test_scores,
                families=families,
                seed=int(args.seed) + int(round(dose)) + int(main_n_components),
            )
            tc = score["trunc_nll"] - score["full_nll"] if np.isfinite(score["trunc_nll"]) and np.isfinite(score["full_nll"]) else np.nan
            common = {
                "variant": main_variant,
                "source_space": main_source_space,
                "n_components": int(main_n_components),
                "analysis_view": "within_session_dynamic",
                "session_id": session_id,
                "dose": float(dose),
                "block_id": block_id,
                "pca_variance_retained": float(np.sum(explained)),
            }
            dynamic_rows.extend(
                [
                    {**common, "model": "gaussian", "heldout_nll": score["gaussian_nll"], "delta_vs_gaussian": 0.0, "tc_higher": np.nan},
                    {**common, "model": "truncated_vine", "heldout_nll": score["trunc_nll"], "delta_vs_gaussian": score["gaussian_nll"] - score["trunc_nll"] if np.isfinite(score["gaussian_nll"]) and np.isfinite(score["trunc_nll"]) else np.nan, "tc_higher": tc},
                    {**common, "model": "full_vine", "heldout_nll": score["full_nll"], "delta_vs_gaussian": score["gaussian_nll"] - score["full_nll"] if np.isfinite(score["gaussian_nll"]) and np.isfinite(score["full_nll"]) else np.nan, "tc_higher": tc},
                ]
            )
    dynamic_df = pd.DataFrame(dynamic_rows)

    # Paper recommendation for latent-state formulation.
    paper_recommendation = "supplement/control only"
    if not main_metrics.empty and not dynamic_df.empty:
        full_delta = float(main_row["mean_full_delta_vs_gaussian"])
        tc_mean = float(main_row["mean_tc_higher"])
        prop_g = float(main_row["prop_full_beats_gaussian"])
        prop_t = float(main_row["prop_full_beats_trunc"])
        stability = float(interpret_df[interpret_df["variant"] == main_variant]["loading_stability_abs_cosine"].mean()) if not interpret_df.empty else np.nan
        if full_delta > 0.20 and tc_mean > 0.05 and prop_g >= 0.70 and prop_t >= 0.60 and (not np.isfinite(stability) or stability >= 0.60):
            paper_recommendation = "main-text viable"
        elif full_delta > 0.05 and prop_g >= 0.55:
            paper_recommendation = "supplement/control only"
        else:
            paper_recommendation = "not worth carrying forward"

    # Write outputs.
    source_summary_df.to_csv(data_dir / "latent_state_source_space_summary.csv", index=False)
    dose_df.to_csv(data_dir / "latent_state_dose_summary.csv", index=False)
    dynamic_df.to_csv(data_dir / "latent_state_dynamic_summary.csv", index=False)
    interpret_df.to_csv(data_dir / "latent_state_interpretability.csv", index=False)
    pd.DataFrame(semantics_rows).to_csv(data_dir / "latent_state_feature_semantics.csv", index=False)
    metrics_df.to_csv(data_dir / "latent_state_metrics_table.csv", index=False)

    source_summary_df.to_csv(out_root / "latent_state_source_space_summary.csv", index=False)
    dose_df.to_csv(out_root / "latent_state_dose_summary.csv", index=False)
    dynamic_df.to_csv(out_root / "latent_state_dynamic_summary.csv", index=False)
    interpret_df.to_csv(out_root / "latent_state_interpretability.csv", index=False)
    pd.DataFrame(semantics_rows).to_csv(out_root / "latent_state_feature_semantics.csv", index=False)
    metrics_df.to_csv(out_root / "latent_state_metrics_table.csv", index=False)

    _plot_source_space(source_summary_df, plots_dir / "fig_latent_source_space_comparison.png")
    _plot_dose_summary(dose_df, plots_dir / "fig_latent_dose_summary.png")
    _plot_session_dose(dose_df, plots_dir / "fig_latent_session_dose_robustness.png")
    _plot_dynamic(dynamic_df, plots_dir / "fig_latent_dynamic_blocks.png")
    _plot_interpretability(interpret_df[interpret_df["variant"] == main_variant].copy(), plots_dir / "fig_latent_interpretability.png")

    metadata = {
        "behavior_feasibility": behavior_status["status"],
        "behavior_reason": behavior_status["reason"],
        "source_space_screen_variants": [_source_variant_name(cfg["source_space"], cfg["n_components"]) for cfg in source_screen],
        "rank_sensitivity_sources": rank_sources,
        "chosen_main_variant": main_variant,
        "chosen_main_source_space": main_source_space,
        "chosen_main_n_components": int(main_n_components),
        "common_slice_count": int(len(common_intersection)),
        "paper_recommendation": paper_recommendation,
    }
    _write_json(data_dir / "latent_state_full_run_metadata.json", metadata)
    _write_json(out_root / "latent_state_full_run_metadata.json", metadata)
    LOGGER.info("Main latent variant: %s | paper recommendation: %s", main_variant, paper_recommendation)


if __name__ == "__main__":
    main()
