#!/usr/bin/env python3
"""Compare formulation families for the Dalgleish photostimulation benchmark."""

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
from scripts.debug_stimulation_exp.run_dalgleish_real_data_benchmark import (
    FAMILY_VARIANTS,
    _apply_train_only_ecdf,
    _fit_train_only_ecdf,
    _score_gaussian_from_pobs,
    _score_vine_on_uniforms,
    _session_seed,
    _split_positions_random,
    _winsorize_train_apply,
    _with_quieter_repo_logging,
    _write_json,
    configure_logging,
)
from scripts.debug_stimulation_exp.run_dalgleish_representation_audit import _compute_single_window, _load_signal_matrix
from dvc_package.experiments.simulation_benchmarks import (
    _estimate_hub_by_correlation,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
)
from scipy.stats import norm


LOGGER = logging.getLogger("dalgleish_formulation_viability")
BASELINE_WINDOW = (-1.0, -0.1)
DELAYED_WINDOW = (0.2, 0.7)
POST_WINDOW = (0.7, 1.4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dalgleish formulation viability study.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--results_root", default="results/stimulation_exp_benchmark")
    parser.add_argument("--family_variant", choices=sorted(FAMILY_VARIANTS.keys()), default="stable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--selection_mode", choices=["responsive_random", "topk_responsive"], default="topk_responsive")
    parser.add_argument("--min_trials_floor", type=int, default=18)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _variant_configs() -> List[Dict[str, Any]]:
    return [
        {
            "family": "A_raw_neuron",
            "variant": "A1_raw_4n_post_scalar",
            "variant_role": "core",
            "feature_dim": 4,
            "selection_kind": "raw",
            "n_neurons": 4,
            "bins": [("post_0p7_1p4", POST_WINDOW)],
            "selection_response": "post",
            "interpretability_note": "Local neuron-level post-stim responses from a small responsive non-targeted subset.",
        },
        {
            "family": "A_raw_neuron",
            "variant": "A2_raw_4n_2bin",
            "variant_role": "core",
            "feature_dim": 8,
            "selection_kind": "raw",
            "n_neurons": 4,
            "bins": [("delayed_0p2_0p7", DELAYED_WINDOW), ("post_0p7_1p4", POST_WINDOW)],
            "selection_response": "mean_delayed_post",
            "interpretability_note": "Local neuron-level delayed and early-post responses from a small responsive non-targeted subset.",
        },
        {
            "family": "A_raw_neuron",
            "variant": "A3_raw_8n_post_scalar",
            "variant_role": "stretch",
            "feature_dim": 8,
            "selection_kind": "raw",
            "n_neurons": 8,
            "bins": [("post_0p7_1p4", POST_WINDOW)],
            "selection_response": "post",
            "interpretability_note": "Stretch test using a somewhat larger local neuron subset; still post-stim scalar only.",
        },
        {
            "family": "B_population_summary",
            "variant": "B1_pop_post_4d",
            "variant_role": "core",
            "feature_dim": 4,
            "selection_kind": "population",
            "summary_kind": "post_4d",
            "interpretability_note": "Interpretable recruitment summary using targeted and non-targeted post-stim population statistics.",
        },
        {
            "family": "B_population_summary",
            "variant": "B2_pop_temporal_6d",
            "variant_role": "core",
            "feature_dim": 6,
            "selection_kind": "population",
            "summary_kind": "temporal_6d",
            "interpretability_note": "Interpretable delayed-versus-post recruitment summary using targeted and non-targeted population means and spread.",
        },
        {
            "family": "B_population_summary",
            "variant": "B3_pop_recruitment_6d",
            "variant_role": "core",
            "feature_dim": 6,
            "selection_kind": "population",
            "summary_kind": "recruitment_6d",
            "interpretability_note": "Interpretable recruitment-shape summary using means, tails, and delayed-to-post change from large neuron pools.",
        },
        {
            "family": "C_latent_state",
            "variant": "C1_latent_post_pca4",
            "variant_role": "core",
            "feature_dim": 4,
            "selection_kind": "latent",
            "latent_kind": "post_pca",
            "n_components": 4,
            "interpretability_note": "Low-dimensional latent population state from post-stim non-targeted activity.",
        },
        {
            "family": "C_latent_state",
            "variant": "C2_latent_2bin_pca4",
            "variant_role": "core",
            "feature_dim": 4,
            "selection_kind": "latent",
            "latent_kind": "twobin_pca",
            "n_components": 4,
            "interpretability_note": "Low-dimensional latent population state from delayed and post non-targeted activity.",
        },
        {
            "family": "C_latent_state",
            "variant": "C3_latent_2bin_pca6",
            "variant_role": "core",
            "feature_dim": 6,
            "selection_kind": "latent",
            "latent_kind": "twobin_pca",
            "n_components": 6,
            "interpretability_note": "Higher-rank latent population state from delayed and post non-targeted activity.",
        },
    ]


def _targeted_union_mask(payload: Dict[str, Any], n_neurons: int) -> np.ndarray:
    mask = np.zeros(n_neurons, dtype=bool)
    for roi_indices in payload.get("targeted_roi_by_program", {}).values():
        for idx in roi_indices:
            if 0 <= int(idx) < n_neurons:
                mask[int(idx)] = True
    return mask


def _select_raw_indices(
    baseline_train: np.ndarray,
    response_train: np.ndarray,
    targeted_mask: np.ndarray,
    n_neurons: int,
    selection_mode: str,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scores = builder._compute_responsiveness_scores(baseline_train, response_train)
    mean_effect = np.nanmean(response_train - baseline_train, axis=0)
    finite_mask = np.isfinite(scores) & np.isfinite(mean_effect)
    positive_mask = finite_mask & (mean_effect > 0.0)
    pool = np.flatnonzero(positive_mask & ~targeted_mask)
    if pool.size < n_neurons:
        pool = np.flatnonzero(finite_mask & ~targeted_mask)
    meta: Dict[str, Any] = {"candidate_count": int(pool.size), "policy": "non_targeted"}
    if pool.size < n_neurons:
        meta["warning"] = f"Only {pool.size} non-targeted neurons for n={n_neurons}."
        return np.array([], dtype=int), meta
    rng = np.random.default_rng(seed)
    if selection_mode == "responsive_random":
        picked = np.sort(rng.choice(pool, size=n_neurons, replace=False)).astype(int)
    else:
        order = np.argsort(scores[pool])[::-1]
        picked = np.sort(pool[order[:n_neurons]]).astype(int)
    meta["selected_scores"] = scores[picked].astype(float).tolist()
    meta["selected_effects"] = mean_effect[picked].astype(float).tolist()
    return picked, meta


def _prepare_session_cache(session_id: str, payload: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
    traces = _load_signal_matrix(data_root / session_id, "spks")
    session_df = payload["trial_table"].reset_index(drop=True).copy()
    stim_frames = session_df["stim_frame"].to_numpy(dtype=int)
    frame_rate_hz = float(pd.to_numeric(session_df["frame_rate_hz"], errors="coerce").dropna().iloc[0])
    baseline, valid_base = _compute_single_window(traces, stim_frames, frame_rate_hz, BASELINE_WINDOW)
    delayed, valid_delayed = _compute_single_window(traces, stim_frames, frame_rate_hz, DELAYED_WINDOW)
    post, valid_post = _compute_single_window(traces, stim_frames, frame_rate_hz, POST_WINDOW)
    valid = (
        valid_base
        & valid_delayed
        & valid_post
        & session_df["is_valid"].to_numpy(dtype=bool)
        & session_df["dose"].notna().to_numpy()
    )
    session_df = session_df.loc[valid].copy().reset_index(drop=True)
    baseline = np.asarray(baseline[valid], dtype=np.float64)
    delayed = np.asarray(delayed[valid], dtype=np.float64)
    post = np.asarray(post[valid], dtype=np.float64)
    targeted_mask = _targeted_union_mask(payload, baseline.shape[1])
    return {
        "session_df": session_df,
        "baseline": baseline,
        "delayed": delayed,
        "post": post,
        "targeted_mask": targeted_mask,
        "frame_rate_hz": frame_rate_hz,
        "n_total_rois": int(baseline.shape[1]),
        "n_targeted_rois": int(np.sum(targeted_mask)),
        "n_non_targeted_rois": int(np.sum(~targeted_mask)),
    }


def _build_split_plan(
    session_df: pd.DataFrame,
    feature_dim: int,
    seed: int,
    session_id: str,
    n_repeats: int,
    train_fraction: float,
    min_trials_floor: int,
) -> List[Dict[str, Any]]:
    min_trials_required = max(int(min_trials_floor), 2 * int(feature_dim) + 4)
    plans: List[Dict[str, Any]] = []
    dose_counts = session_df.groupby("dose").size()
    eligible_doses = sorted(float(d) for d, n in dose_counts.items() if int(n) >= min_trials_required)
    for repeat_id in range(int(n_repeats)):
        rng = np.random.default_rng(_session_seed(seed, session_id, extra=7919 * repeat_id + feature_dim))
        for dose in eligible_doses:
            idx = session_df.index[session_df["dose"] == dose].to_numpy(dtype=int)
            train_pos, test_pos = _split_positions_random(len(idx), train_fraction, rng)
            train_idx = idx[train_pos]
            test_idx = idx[test_pos]
            if len(train_idx) < 5 or len(test_idx) < 3:
                continue
            plans.append(
                {
                    "session_id": session_id,
                    "dose": float(dose),
                    "repeat_id": int(repeat_id),
                    "slice_key": f"{session_id}__dose_{int(round(float(dose))):03d}__repeat_{repeat_id:02d}",
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    return plans


def _evaluate_features_with_status(
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
        "warning": "",
    }
    try:
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
        except Exception as exc:  # pragma: no cover - runtime dependent
            out["full_status"] = f"failed:{exc}"
    except Exception as exc:  # pragma: no cover - runtime dependent
        out["warning"] = str(exc)
        out["gaussian_status"] = f"failed:{exc}"
    return out


def _population_features(kind: str, delayed: np.ndarray, post: np.ndarray, targeted_mask: np.ndarray) -> np.ndarray:
    targ = np.asarray(targeted_mask, dtype=bool)
    non = ~targ
    if np.sum(targ) == 0 or np.sum(non) == 0:
        return np.empty((delayed.shape[0], 0), dtype=np.float64)

    targ_delayed = delayed[:, targ]
    targ_post = post[:, targ]
    non_delayed = delayed[:, non]
    non_post = post[:, non]

    targeted_mean_delayed = np.nanmean(targ_delayed, axis=1)
    targeted_mean_post = np.nanmean(targ_post, axis=1)
    non_mean_delayed = np.nanmean(non_delayed, axis=1)
    non_mean_post = np.nanmean(non_post, axis=1)
    non_std_delayed = np.nanstd(non_delayed, axis=1)
    non_std_post = np.nanstd(non_post, axis=1)
    non_q90_post = np.nanpercentile(non_post, 90.0, axis=1)
    non_q10_post = np.nanpercentile(non_post, 10.0, axis=1)

    if kind == "post_4d":
        feats = np.column_stack(
            [targeted_mean_post, non_mean_post, non_std_post, non_q90_post]
        )
    elif kind == "temporal_6d":
        feats = np.column_stack(
            [
                targeted_mean_delayed,
                targeted_mean_post,
                non_mean_delayed,
                non_mean_post,
                non_std_delayed,
                non_std_post,
            ]
        )
    elif kind == "recruitment_6d":
        feats = np.column_stack(
            [
                targeted_mean_post,
                non_mean_post,
                non_q90_post,
                non_q10_post,
                non_mean_post - non_mean_delayed,
                targeted_mean_post - targeted_mean_delayed,
            ]
        )
    else:
        raise ValueError(f"Unknown population summary kind {kind}")
    return np.asarray(feats, dtype=np.float64)


def _fit_pca_train_apply(
    train_x: np.ndarray,
    test_x: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    mean = np.nanmean(train_x, axis=0)
    centered_train = train_x - mean
    centered_test = test_x - mean
    std = np.nanstd(centered_train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    z_train = centered_train / std
    z_test = centered_test / std
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0)
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=0.0, neginf=0.0)
    u, s, vt = np.linalg.svd(z_train, full_matrices=False)
    k = min(int(n_components), vt.shape[0], z_train.shape[0], z_train.shape[1])
    if k < int(n_components):
        raise ValueError(f"Only {k} PCA components available, need {n_components}")
    components = vt[:k]
    train_scores = z_train @ components.T
    test_scores = z_test @ components.T
    denom = max(z_train.shape[0] - 1, 1)
    eigvals = (s ** 2) / denom
    total_var = float(np.sum(eigvals))
    if total_var <= 0.0:
        explained = np.zeros(k, dtype=np.float64)
    else:
        explained = eigvals[:k] / total_var
    return np.asarray(train_scores, dtype=np.float64), np.asarray(test_scores, dtype=np.float64), np.asarray(explained, dtype=np.float64)


def _raw_feature_matrix(
    cache: Dict[str, Any],
    selected: np.ndarray,
    bins: Sequence[Tuple[str, Tuple[float, float]]],
) -> np.ndarray:
    pieces: List[np.ndarray] = []
    for label, _window in bins:
        source = cache["post"] if "post" in label else cache["delayed"]
        pieces.append(np.asarray(source[:, selected], dtype=np.float64))
    return np.concatenate(pieces, axis=1).astype(np.float64)


def _variant_semantics_rows(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    family = str(cfg["family"])
    variant = str(cfg["variant"])
    if family == "A_raw_neuron":
        idx = 1
        for bin_label, secs in cfg["bins"]:
            for neuron_slot in range(1, int(cfg["n_neurons"]) + 1):
                rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "feature_index": idx,
                        "feature_name": f"x{idx}",
                        "feature_group": "raw_neuron",
                        "semantics": f"selected non-targeted responsive neuron slot {neuron_slot} mean spks from {secs[0]:.1f} s to {secs[1]:.1f} s",
                        "start_s": float(secs[0]),
                        "end_s": float(secs[1]),
                        "source_pool": "selected responsive non-targeted neurons",
                        "latent_component": np.nan,
                    }
                )
                idx += 1
    elif variant == "B1_pop_post_4d":
        names = [
            ("targeted_mean_post", "mean post-stim spks across the session-level union of targeted neurons", POST_WINDOW),
            ("nontargeted_mean_post", "mean post-stim spks across non-targeted usable neurons", POST_WINDOW),
            ("nontargeted_std_post", "standard deviation of post-stim spks across non-targeted usable neurons", POST_WINDOW),
            ("nontargeted_q90_post", "90th percentile of post-stim spks across non-targeted usable neurons", POST_WINDOW),
        ]
        for idx, (name, semantics, secs) in enumerate(names, start=1):
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "feature_index": idx,
                    "feature_name": name,
                    "feature_group": "population_summary",
                    "semantics": semantics,
                    "start_s": float(secs[0]),
                    "end_s": float(secs[1]),
                    "source_pool": "session-level targeted/non-targeted pools",
                    "latent_component": np.nan,
                }
            )
    elif variant == "B2_pop_temporal_6d":
        names = [
            ("targeted_mean_delayed", "mean delayed spks across the session-level union of targeted neurons", DELAYED_WINDOW),
            ("targeted_mean_post", "mean early-post spks across the session-level union of targeted neurons", POST_WINDOW),
            ("nontargeted_mean_delayed", "mean delayed spks across non-targeted usable neurons", DELAYED_WINDOW),
            ("nontargeted_mean_post", "mean early-post spks across non-targeted usable neurons", POST_WINDOW),
            ("nontargeted_std_delayed", "standard deviation of delayed spks across non-targeted usable neurons", DELAYED_WINDOW),
            ("nontargeted_std_post", "standard deviation of early-post spks across non-targeted usable neurons", POST_WINDOW),
        ]
        for idx, (name, semantics, secs) in enumerate(names, start=1):
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "feature_index": idx,
                    "feature_name": name,
                    "feature_group": "population_summary",
                    "semantics": semantics,
                    "start_s": float(secs[0]),
                    "end_s": float(secs[1]),
                    "source_pool": "session-level targeted/non-targeted pools",
                    "latent_component": np.nan,
                }
            )
    elif variant == "B3_pop_recruitment_6d":
        names = [
            ("targeted_mean_post", "mean early-post spks across the session-level union of targeted neurons", POST_WINDOW),
            ("nontargeted_mean_post", "mean early-post spks across non-targeted usable neurons", POST_WINDOW),
            ("nontargeted_q90_post", "90th percentile of early-post spks across non-targeted usable neurons", POST_WINDOW),
            ("nontargeted_q10_post", "10th percentile of early-post spks across non-targeted usable neurons", POST_WINDOW),
            ("nontargeted_post_minus_delayed", "change in non-targeted mean spks from delayed to early-post windows", (DELAYED_WINDOW[0], POST_WINDOW[1])),
            ("targeted_post_minus_delayed", "change in targeted mean spks from delayed to early-post windows", (DELAYED_WINDOW[0], POST_WINDOW[1])),
        ]
        for idx, (name, semantics, secs) in enumerate(names, start=1):
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "feature_index": idx,
                    "feature_name": name,
                    "feature_group": "population_summary",
                    "semantics": semantics,
                    "start_s": float(secs[0]),
                    "end_s": float(secs[1]),
                    "source_pool": "session-level targeted/non-targeted pools",
                    "latent_component": np.nan,
                }
            )
    else:
        n_components = int(cfg["n_components"])
        for idx in range(1, n_components + 1):
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "feature_index": idx,
                    "feature_name": f"PC{idx}",
                    "feature_group": "latent_state",
                    "semantics": f"train-fit PCA latent population-state component {idx} from non-targeted spks activity",
                    "start_s": float(DELAYED_WINDOW[0] if "2bin" in variant else POST_WINDOW[0]),
                    "end_s": float(POST_WINDOW[1]),
                    "source_pool": "all non-targeted usable neurons",
                    "latent_component": int(idx),
                }
            )
    return rows


def _variant_score(summary_row: pd.Series) -> float:
    full_delta = float(summary_row.get("mean_full_delta_vs_gaussian", np.nan))
    tc_mean = float(summary_row.get("mean_tc_higher", np.nan))
    prop_g = float(summary_row.get("prop_full_beats_gaussian", np.nan))
    prop_t = float(summary_row.get("prop_full_beats_trunc", np.nan))
    failure_rate = float(summary_row.get("full_failure_rate", np.nan))
    if not np.isfinite(full_delta):
        return -1e9
    return (
        full_delta
        + 0.35 * (tc_mean if np.isfinite(tc_mean) else 0.0)
        + 0.10 * ((prop_g - 0.5) if np.isfinite(prop_g) else 0.0)
        + 0.05 * ((prop_t - 0.5) if np.isfinite(prop_t) else 0.0)
        - 0.25 * (failure_rate if np.isfinite(failure_rate) else 0.0)
    )


def _summarize_variant(
    cfg: Dict[str, Any],
    metrics_df: pd.DataFrame,
    comparison_scope: str,
) -> Dict[str, Any]:
    sub = metrics_df[metrics_df["comparison_scope"] == comparison_scope].copy()
    if sub.empty:
        return {
            "row_type": "variant_summary",
            "comparison_scope": comparison_scope,
            "family": cfg["family"],
            "variant": cfg["variant"],
            "variant_role": cfg["variant_role"],
            "feature_dim": cfg["feature_dim"],
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
            "full_failure_rate": np.nan,
            "interpretability_rating": "unknown",
            "interpretability_note": cfg["interpretability_note"],
            "pca_mean_explained_variance_retained": np.nan,
            "pca_median_explained_variance_retained": np.nan,
            "family_classification": "",
        }

    gaussian = sub[sub["model"] == "gaussian"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "g"})
    trunc = sub[sub["model"] == "truncated_vine"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "t"})
    full = sub[sub["model"] == "full_vine"][["split_id", "heldout_nll", "tc_higher"]].rename(columns={"heldout_nll": "f"})
    merged = gaussian.merge(trunc, on="split_id", how="inner").merge(full, on="split_id", how="inner")
    usable = merged[np.isfinite(merged["g"]) & np.isfinite(merged["t"]) & np.isfinite(merged["f"])].copy()
    full_rows = sub[sub["model"] == "full_vine"].copy()
    failure_rate = 1.0 - float(np.mean(full_rows["fit_status"] == "success")) if not full_rows.empty else np.nan
    if usable.empty:
        full_delta = np.nan
        prop_g = np.nan
        prop_t = np.nan
        tc_mean = np.nan
        tc_median = np.nan
    else:
        full_delta = float(np.mean(usable["g"] - usable["f"]))
        prop_g = float(np.mean(usable["f"] < usable["g"]))
        prop_t = float(np.mean(usable["f"] < usable["t"]))
        tc_mean = float(np.mean(usable["tc_higher"]))
        tc_median = float(np.median(usable["tc_higher"]))
    if cfg["family"] == "B_population_summary":
        rating = "high"
    elif cfg["family"] == "A_raw_neuron":
        rating = "medium"
    else:
        rating = "medium"
    return {
        "row_type": "variant_summary",
        "comparison_scope": comparison_scope,
        "family": cfg["family"],
        "variant": cfg["variant"],
        "variant_role": cfg["variant_role"],
        "feature_dim": cfg["feature_dim"],
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
        "full_failure_rate": failure_rate,
        "interpretability_rating": rating,
        "interpretability_note": cfg["interpretability_note"],
        "pca_mean_explained_variance_retained": float(sub["pca_variance_retained"].dropna().mean()) if "pca_variance_retained" in sub else np.nan,
        "pca_median_explained_variance_retained": float(sub["pca_variance_retained"].dropna().median()) if "pca_variance_retained" in sub else np.nan,
        "family_classification": "",
    }


def _choose_representatives(summary_df: pd.DataFrame) -> Dict[str, str]:
    reps: Dict[str, str] = {}
    native = summary_df[(summary_df["row_type"] == "variant_summary") & (summary_df["comparison_scope"] == "native")].copy()
    for family, group in native.groupby("family"):
        use_group = group.copy()
        if family == "A_raw_neuron":
            core = use_group[use_group["variant_role"] == "core"].copy()
            if not core.empty:
                use_group = core
        use_group = use_group.assign(score=use_group.apply(_variant_score, axis=1))
        reps[str(family)] = str(use_group.sort_values(["score", "slice_count"], ascending=[False, False]).iloc[0]["variant"])
    return reps


def _family_classification(
    family: str,
    row: pd.Series,
    chosen_direction_family: str,
) -> str:
    if row["slice_count"] <= 0 or (np.isfinite(row["full_failure_rate"]) and row["full_failure_rate"] > 0.3):
        return "not_viable"
    if family == chosen_direction_family:
        return "best_current_direction"
    full_delta = float(row["mean_full_delta_vs_gaussian"]) if pd.notna(row["mean_full_delta_vs_gaussian"]) else np.nan
    tc_mean = float(row["mean_tc_higher"]) if pd.notna(row["mean_tc_higher"]) else np.nan
    if family == "B_population_summary":
        return "scientifically_interpretable_but_not_competitive"
    if family == "A_raw_neuron":
        if np.isfinite(full_delta) and full_delta > 0.0:
            return "technically_viable_but_scientifically_weak"
        return "not_viable"
    if np.isfinite(full_delta) and full_delta > 0.0 and (not np.isfinite(tc_mean) or tc_mean <= 0.0):
        return "technically_viable_but_scientifically_weak"
    return "not_viable"


def _plot_family_comparison(summary_df: pd.DataFrame, reps: Dict[str, str], out_path: Path) -> None:
    native = summary_df[(summary_df["row_type"] == "variant_summary") & (summary_df["comparison_scope"] == "native")]
    common = summary_df[(summary_df["row_type"] == "variant_summary") & (summary_df["comparison_scope"] == "common_slice")]
    native = native[native["variant"].isin(reps.values())].copy()
    common = common[common["variant"].isin(reps.values())].copy()
    families = ["A_raw_neuron", "B_population_summary", "C_latent_state"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = np.arange(len(families))
    width = 0.35
    native_vals = [float(native.loc[native["family"] == fam, "mean_full_delta_vs_gaussian"].iloc[0]) for fam in families]
    common_vals = [float(common.loc[common["family"] == fam, "mean_full_delta_vs_gaussian"].iloc[0]) for fam in families]
    axes[0].bar(x - width / 2, native_vals, width, label="native")
    axes[0].bar(x + width / 2, common_vals, width, label="common slice")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks(x, ["Raw", "Pop", "Latent"])
    axes[0].set_ylabel("Full-vine delta vs Gaussian")
    axes[0].set_title("Best variant per formulation family")
    axes[0].legend(frameon=False)

    native_tc = [float(native.loc[native["family"] == fam, "mean_tc_higher"].iloc[0]) for fam in families]
    common_tc = [float(common.loc[common["family"] == fam, "mean_tc_higher"].iloc[0]) for fam in families]
    axes[1].bar(x - width / 2, native_tc, width, label="native")
    axes[1].bar(x + width / 2, common_tc, width, label="common slice")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x, ["Raw", "Pop", "Latent"])
    axes[1].set_ylabel("TC_higher")
    axes[1].set_title("Higher-order gain on matched slices")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_family_variants(summary_df: pd.DataFrame, family: str, out_path: Path, title: str) -> None:
    sub = summary_df[(summary_df["row_type"] == "variant_summary") & (summary_df["comparison_scope"] == "native") & (summary_df["family"] == family)].copy()
    if sub.empty:
        return
    sub = sub.sort_values("mean_full_delta_vs_gaussian", ascending=False)
    labels = sub["variant"].tolist()
    full_delta = sub["mean_full_delta_vs_gaussian"].astype(float).to_numpy()
    tc_vals = sub["mean_tc_higher"].astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    y = np.arange(len(labels))
    axes[0].barh(y, full_delta, color="#4c78a8")
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("Full-vine delta vs Gaussian")
    axes[0].set_title(title)
    axes[1].barh(y, tc_vals, color="#f58518")
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("TC_higher")
    if family == "C_latent_state":
        explained = sub["pca_mean_explained_variance_retained"].astype(float).to_numpy()
        for idx, val in enumerate(explained):
            if np.isfinite(val):
                axes[1].text(tc_vals[idx], idx, f"  EVR={val:.2f}", va="center", fontsize=9)
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

    manifest = builder.build_manifest(data_root)
    _trials_df, neural_data, _ = builder.build_trials(data_root, manifest)
    families = list(FAMILY_VARIANTS[args.family_variant])
    configs = _variant_configs()

    session_cache: Dict[str, Dict[str, Any]] = {}
    for session_id, payload in neural_data.items():
        if payload.get("used", False):
            session_cache[session_id] = _prepare_session_cache(session_id, payload, data_root)

    metrics_rows: List[Dict[str, Any]] = []
    semantics_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        semantics_rows.extend(_variant_semantics_rows(cfg))

    for cfg in configs:
        family = str(cfg["family"])
        variant = str(cfg["variant"])
        feature_dim = int(cfg["feature_dim"])
        for session_id, cache in session_cache.items():
            session_df = cache["session_df"]
            split_plan = _build_split_plan(
                session_df=session_df,
                feature_dim=feature_dim,
                seed=int(args.seed),
                session_id=session_id,
                n_repeats=int(args.n_repeats),
                train_fraction=float(args.train_fraction),
                min_trials_floor=int(args.min_trials_floor),
            )
            if not split_plan:
                continue
            targeted_mask = np.asarray(cache["targeted_mask"], dtype=bool)
            for split in split_plan:
                train_idx = np.asarray(split["train_idx"], dtype=int)
                test_idx = np.asarray(split["test_idx"], dtype=int)
                pca_retained = np.nan
                feature_note = ""
                candidate_count = np.nan

                if cfg["selection_kind"] == "raw":
                    if cfg["selection_response"] == "post":
                        select_response = np.asarray(cache["post"], dtype=np.float64)
                    else:
                        select_response = 0.5 * (np.asarray(cache["delayed"], dtype=np.float64) + np.asarray(cache["post"], dtype=np.float64))
                    selected, meta = _select_raw_indices(
                        baseline_train=np.asarray(cache["baseline"][train_idx], dtype=np.float64),
                        response_train=np.asarray(select_response[train_idx], dtype=np.float64),
                        targeted_mask=targeted_mask,
                        n_neurons=int(cfg["n_neurons"]),
                        selection_mode=args.selection_mode,
                        seed=_session_seed(int(args.seed), session_id, extra=feature_dim + 101 * int(split["repeat_id"])),
                    )
                    candidate_count = meta.get("candidate_count", np.nan)
                    if selected.size != int(cfg["n_neurons"]):
                        continue
                    all_x = _raw_feature_matrix(cache, selected, cfg["bins"])
                    train_x = all_x[train_idx]
                    test_x = all_x[test_idx]
                    feature_note = f"{selected.size} selected non-targeted neurons"
                elif cfg["selection_kind"] == "population":
                    all_x = _population_features(
                        kind=str(cfg["summary_kind"]),
                        delayed=np.asarray(cache["delayed"], dtype=np.float64),
                        post=np.asarray(cache["post"], dtype=np.float64),
                        targeted_mask=targeted_mask,
                    )
                    if all_x.shape[1] != int(cfg["feature_dim"]):
                        continue
                    train_x = all_x[train_idx]
                    test_x = all_x[test_idx]
                    feature_note = "session-level targeted versus non-targeted population summaries"
                else:
                    non_target_idx = np.flatnonzero(~targeted_mask)
                    if non_target_idx.size < int(cfg["n_components"]):
                        continue
                    if str(cfg["latent_kind"]) == "post_pca":
                        source_all = np.asarray(cache["post"][:, non_target_idx], dtype=np.float64)
                    else:
                        source_all = np.concatenate(
                            [
                                np.asarray(cache["delayed"][:, non_target_idx], dtype=np.float64),
                                np.asarray(cache["post"][:, non_target_idx], dtype=np.float64),
                            ],
                            axis=1,
                        )
                    try:
                        train_x, test_x, explained = _fit_pca_train_apply(
                            train_x=source_all[train_idx],
                            test_x=source_all[test_idx],
                            n_components=int(cfg["n_components"]),
                        )
                        pca_retained = float(np.sum(explained))
                    except Exception:
                        continue
                    feature_note = f"PCA on {non_target_idx.size} non-targeted neurons"

                score = _evaluate_features_with_status(train_x=train_x, test_x=test_x, families=families, seed=_session_seed(int(args.seed), session_id, extra=feature_dim + 19 * int(split['repeat_id'])))
                common = {
                    "family": family,
                    "variant": variant,
                    "variant_role": cfg["variant_role"],
                    "comparison_scope": "native",
                    "session_id": session_id,
                    "dose": float(split["dose"]),
                    "repeat_id": int(split["repeat_id"]),
                    "slice_key": str(split["slice_key"]),
                    "split_id": f"{variant}__{split['slice_key']}",
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "feature_dim": feature_dim,
                    "candidate_count": candidate_count,
                    "feature_note": feature_note,
                    "pca_variance_retained": pca_retained,
                    "on_common_slice": False,
                    "fit_status": "",
                }
                metrics_rows.extend(
                    [
                        {
                            **common,
                            "model": "gaussian",
                            "heldout_nll": score["gaussian_nll"],
                            "delta_vs_gaussian": 0.0,
                            "tc_higher": np.nan,
                            "fit_status": score["gaussian_status"],
                        },
                        {
                            **common,
                            "model": "truncated_vine",
                            "heldout_nll": score["trunc_nll"],
                            "delta_vs_gaussian": score["gaussian_nll"] - score["trunc_nll"] if np.isfinite(score["gaussian_nll"]) and np.isfinite(score["trunc_nll"]) else np.nan,
                            "tc_higher": score["trunc_nll"] - score["full_nll"] if np.isfinite(score["trunc_nll"]) and np.isfinite(score["full_nll"]) else np.nan,
                            "fit_status": score["trunc_status"],
                        },
                        {
                            **common,
                            "model": "full_vine",
                            "heldout_nll": score["full_nll"],
                            "delta_vs_gaussian": score["gaussian_nll"] - score["full_nll"] if np.isfinite(score["gaussian_nll"]) and np.isfinite(score["full_nll"]) else np.nan,
                            "tc_higher": score["trunc_nll"] - score["full_nll"] if np.isfinite(score["trunc_nll"]) and np.isfinite(score["full_nll"]) else np.nan,
                            "fit_status": score["full_status"],
                        },
                    ]
                )
        LOGGER.info("%s complete", variant)

    metrics_df = pd.DataFrame(metrics_rows)
    semantics_df = pd.DataFrame(semantics_rows)

    summary_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        sub = metrics_df[metrics_df["variant"] == cfg["variant"]].copy()
        summary_rows.append(_summarize_variant(cfg, sub, comparison_scope="native"))
    summary_df = pd.DataFrame(summary_rows)

    reps = _choose_representatives(summary_df)

    common_slice_sets: List[set] = []
    for family, variant in reps.items():
        successful = metrics_df[
            (metrics_df["variant"] == variant)
            & (metrics_df["model"] == "full_vine")
            & (metrics_df["fit_status"] == "success")
        ]["slice_key"]
        common_slice_sets.append(set(successful.tolist()))
    common_slice_intersection = set.intersection(*common_slice_sets) if common_slice_sets else set()
    metrics_df.loc[metrics_df["slice_key"].isin(common_slice_intersection), "on_common_slice"] = True

    for cfg in configs:
        if cfg["variant"] not in reps.values():
            continue
        sub = metrics_df[(metrics_df["variant"] == cfg["variant"]) & (metrics_df["on_common_slice"])].copy()
        if not sub.empty:
            sub.loc[:, "comparison_scope"] = "common_slice"
        summary_rows.append(_summarize_variant(cfg, sub, comparison_scope="common_slice"))
    summary_df = pd.DataFrame(summary_rows)

    native_rep_rows = summary_df[
        (summary_df["row_type"] == "variant_summary")
        & (summary_df["comparison_scope"] == "native")
        & (summary_df["variant"].isin(reps.values()))
    ].copy()
    common_rep_rows = summary_df[
        (summary_df["row_type"] == "variant_summary")
        & (summary_df["comparison_scope"] == "common_slice")
        & (summary_df["variant"].isin(reps.values()))
    ].copy()

    # Prefer interpretable population summaries when numerically close to latent states.
    chosen_direction_family = "B_population_summary"
    if not native_rep_rows.empty:
        b_row = native_rep_rows[native_rep_rows["family"] == "B_population_summary"]
        c_row = native_rep_rows[native_rep_rows["family"] == "C_latent_state"]
        if not b_row.empty and not c_row.empty:
            b_row = b_row.iloc[0]
            c_row = c_row.iloc[0]
            close_full = abs(float(b_row["mean_full_delta_vs_gaussian"]) - float(c_row["mean_full_delta_vs_gaussian"])) <= 0.10
            close_tc = abs(float(b_row["mean_tc_higher"]) - float(c_row["mean_tc_higher"])) <= 0.10
            if not (close_full and close_tc):
                b_score = _variant_score(b_row)
                c_score = _variant_score(c_row)
                chosen_direction_family = "B_population_summary" if b_score >= c_score else "C_latent_state"
        else:
            scored = native_rep_rows.assign(score=native_rep_rows.apply(_variant_score, axis=1)).sort_values("score", ascending=False)
            if not scored.empty:
                chosen_direction_family = str(scored.iloc[0]["family"])

    family_summary_rows: List[Dict[str, Any]] = []
    for scope, rep_frame in [("native", native_rep_rows), ("common_slice", common_rep_rows)]:
        for family, variant in reps.items():
            row = rep_frame[rep_frame["family"] == family]
            if row.empty:
                continue
            rep = row.iloc[0].to_dict()
            rep["row_type"] = "family_summary"
            rep["comparison_scope"] = scope
            rep["family_classification"] = _family_classification(family, row.iloc[0], chosen_direction_family)
            family_summary_rows.append(rep)
    summary_df = pd.concat([summary_df, pd.DataFrame(family_summary_rows)], axis=0, ignore_index=True)
    summary_df = summary_df.sort_values(["row_type", "comparison_scope", "family", "variant"]).reset_index(drop=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = data_dir / "formulation_metrics_table.csv"
    summary_path = data_dir / "formulation_viability_summary.csv"
    semantics_path = data_dir / "formulation_feature_semantics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    semantics_df.to_csv(semantics_path, index=False)
    metrics_df.to_csv(out_root / "formulation_metrics_table.csv", index=False)
    summary_df.to_csv(out_root / "formulation_viability_summary.csv", index=False)
    semantics_df.to_csv(out_root / "formulation_feature_semantics.csv", index=False)

    _plot_family_comparison(summary_df, reps, plots_dir / "fig_formulation_family_comparison.png")
    _plot_family_variants(summary_df, "A_raw_neuron", plots_dir / "fig_formulation_best_raw_neuron.png", "Raw-neuron family")
    _plot_family_variants(summary_df, "B_population_summary", plots_dir / "fig_formulation_best_population_summary.png", "Population-summary family")
    _plot_family_variants(summary_df, "C_latent_state", plots_dir / "fig_formulation_best_latent_state.png", "Latent-state family")

    metadata = {
        "seed": int(args.seed),
        "n_repeats": int(args.n_repeats),
        "train_fraction": float(args.train_fraction),
        "selection_mode": args.selection_mode,
        "family_variant": args.family_variant,
        "common_slice_count": int(len(common_slice_intersection)),
        "representative_variants": reps,
        "preferred_direction_family": chosen_direction_family,
        "family_preference_rule": "Prefer B_population_summary over C_latent_state when native metrics are numerically close in full-vs-Gaussian and TC_higher.",
    }
    _write_json(data_dir / "formulation_viability_metadata.json", metadata)
    _write_json(out_root / "formulation_viability_metadata.json", metadata)
    LOGGER.info("Representative variants: %s", reps)
    LOGGER.info("Preferred family direction: %s", chosen_direction_family)


if __name__ == "__main__":
    main()
