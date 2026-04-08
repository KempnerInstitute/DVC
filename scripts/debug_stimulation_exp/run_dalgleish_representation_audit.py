#!/usr/bin/env python3
"""Focused representation and artifact audit for the Dalgleish real-data benchmark."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.stimulation_exp_benchmark.build_dalgleish_dvc_dataset as builder
from dvc_package.experiments.simulation_benchmarks import (
    _estimate_hub_by_correlation,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
)
from scripts.debug_stimulation_exp.run_dalgleish_real_data_benchmark import (
    DEFAULT_FAMILY_VARIANT,
    FAMILY_VARIANTS,
    _apply_train_only_ecdf,
    _feature_lookup_rows,
    _fit_train_only_ecdf,
    _score_gaussian_from_pobs,
    _score_vine_on_uniforms,
    _session_seed,
    _split_positions_random,
    _with_quieter_repo_logging,
    _winsorize_train_apply,
    _write_json,
    configure_logging,
)
from scripts.debug_stimulation_exp.run_dalgleish_real_data_decision_rerun import _dynamic_axis_label, _status_label


LOGGER = logging.getLogger("dalgleish_representation_audit")

BASELINE_WINDOW = (-1.0, -0.1)
WINDOW_DEFS: Dict[str, Tuple[float, float]] = {
    "current_stim_mean": (0.0, 1.0),
    "delayed_stim_mean": (0.2, 1.2),
    "post_stim_mean": (1.0, 2.0),
    "stim_skip_early": (0.2, 1.0),
}
BIN_DEFS: List[Tuple[str, Tuple[float, float]]] = [
    ("late_stim_bin", (0.2, 0.7)),
    ("early_post_bin", (0.7, 1.4)),
]
SIGNAL_TYPES = ("spks", "F", "Fcorr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dalgleish representation / artifact audit.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--results_root", default="results/stimulation_exp_benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--selection_mode", choices=["responsive_random", "topk_responsive"], default="topk_responsive")
    parser.add_argument("--family_variant", choices=sorted(FAMILY_VARIANTS.keys()), default=DEFAULT_FAMILY_VARIANT)
    parser.add_argument("--min_trials_per_slice", type=int, default=18)
    parser.add_argument("--dynamic_min_block_trials", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(1.96 * np.std(vals, ddof=1) / math.sqrt(vals.size))


def _align_and_filter_trace(trace: np.ndarray, iscell: np.ndarray) -> np.ndarray:
    if iscell.ndim == 1:
        iscell_mask = iscell.astype(np.float64) > 0.5
    else:
        iscell_mask = iscell[:, 0].astype(np.float64) > 0.5
    trace = np.asarray(trace, dtype=np.float32)
    if trace.ndim != 2:
        raise ValueError(f"Trace matrix is not 2D: shape={trace.shape}")
    if trace.shape[0] != iscell_mask.size and trace.shape[1] == iscell_mask.size:
        trace = trace.T
    if trace.shape[0] != iscell_mask.size:
        raise ValueError(f"Could not align trace matrix with iscell mask: shape={trace.shape}, n_iscell={iscell_mask.size}")
    return trace[iscell_mask]


def _load_signal_matrix(session_path: Path, signal_type: str) -> np.ndarray:
    fall_path = session_path / "Fall.mat"
    iscell, _stat, _ops = builder._load_suite2p_metadata(fall_path)
    if signal_type == "spks":
        loaded = builder._safe_loadmat(fall_path, variable_names=["spks"])
        return _align_and_filter_trace(np.asarray(loaded["spks"], dtype=np.float32), iscell)
    if signal_type == "F":
        loaded = builder._safe_loadmat(fall_path, variable_names=["F"])
        return _align_and_filter_trace(np.asarray(loaded["F"], dtype=np.float32), iscell)
    if signal_type == "Fcorr":
        loaded = builder._safe_loadmat(fall_path, variable_names=["F", "Fneu"])
        f = _align_and_filter_trace(np.asarray(loaded["F"], dtype=np.float32), iscell)
        fneu = _align_and_filter_trace(np.asarray(loaded["Fneu"], dtype=np.float32), iscell)
        return (f - 0.7 * fneu).astype(np.float32)
    raise ValueError(f"Unknown signal_type={signal_type}")


def _compute_single_window(
    traces: np.ndarray,
    stim_frames: np.ndarray,
    frame_rate_hz: float,
    window_seconds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    start_off = int(round(window_seconds[0] * frame_rate_hz))
    end_off = int(round(window_seconds[1] * frame_rate_hz))
    if end_off <= start_off:
        raise ValueError(f"Invalid window {window_seconds}")
    n_trials = len(stim_frames)
    n_neurons, n_frames = traces.shape
    out = np.full((n_trials, n_neurons), np.nan, dtype=np.float32)
    valid = np.zeros(n_trials, dtype=bool)
    for idx, stim_frame in enumerate(np.asarray(stim_frames, dtype=int)):
        start = int(stim_frame + start_off)
        end = int(stim_frame + end_off)
        if start < 0 or end > n_frames or end <= start:
            continue
        out[idx] = np.nanmean(traces[:, start:end], axis=1).astype(np.float32)
        valid[idx] = True
    return out, valid


def _targeted_union_mask(payload: Dict[str, Any], n_neurons: int) -> np.ndarray:
    mask = np.zeros(n_neurons, dtype=bool)
    for roi_indices in payload.get("targeted_roi_by_program", {}).values():
        for idx in roi_indices:
            if 0 <= int(idx) < n_neurons:
                mask[int(idx)] = True
    return mask


def _materialize_session_signal(
    session_id: str,
    payload: Dict[str, Any],
    signal_type: str,
    data_root: Path,
) -> Dict[str, Any]:
    traces = _load_signal_matrix(data_root / session_id, signal_type)
    session_df = payload["trial_table"].reset_index(drop=True).copy()
    stim_frames = session_df["stim_frame"].to_numpy(dtype=int)
    frame_rate_hz = float(pd.to_numeric(session_df["frame_rate_hz"], errors="coerce").dropna().iloc[0])

    baseline, baseline_valid = _compute_single_window(traces, stim_frames, frame_rate_hz, BASELINE_WINDOW)
    scalar_windows: Dict[str, np.ndarray] = {}
    scalar_valids: Dict[str, np.ndarray] = {}
    for window_name, window_seconds in WINDOW_DEFS.items():
        arr, valid = _compute_single_window(traces, stim_frames, frame_rate_hz, window_seconds)
        scalar_windows[window_name] = arr
        scalar_valids[window_name] = baseline_valid & valid & session_df["is_valid"].to_numpy(dtype=bool) & session_df["dose"].notna().to_numpy()

    bin_arrays: Dict[str, np.ndarray] = {}
    bin_valids: List[np.ndarray] = []
    for bin_name, window_seconds in BIN_DEFS:
        arr, valid = _compute_single_window(traces, stim_frames, frame_rate_hz, window_seconds)
        bin_arrays[bin_name] = arr
        bin_valids.append(valid)
    binned_valid = baseline_valid.copy()
    for valid in bin_valids:
        binned_valid &= valid
    binned_valid &= session_df["is_valid"].to_numpy(dtype=bool) & session_df["dose"].notna().to_numpy()
    selection_response = np.mean(np.stack([bin_arrays[name] for name, _window in BIN_DEFS], axis=0), axis=0).astype(np.float32)

    return {
        "baseline": baseline,
        "scalar_windows": scalar_windows,
        "scalar_valids": scalar_valids,
        "binned_windows": bin_arrays,
        "binned_valid": binned_valid,
        "binned_selection_response": selection_response,
    }


def _prepare_representation_data(
    signal_cache: Dict[str, Any],
    representation: str,
    window_definition: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    baseline = np.asarray(signal_cache["baseline"], dtype=np.float32)
    if representation == "scalar":
        response = np.asarray(signal_cache["scalar_windows"][window_definition], dtype=np.float32)
        valid = np.asarray(signal_cache["scalar_valids"][window_definition], dtype=bool)
        feature_matrix = response.astype(np.float32)
        feature_labels = ["mean"]
        return baseline, response, feature_matrix, valid, feature_labels
    if representation == "binned":
        bin_names = [name for name, _window in BIN_DEFS]
        bins = [np.asarray(signal_cache["binned_windows"][name], dtype=np.float32) for name in bin_names]
        selection_response = np.asarray(signal_cache["binned_selection_response"], dtype=np.float32)
        valid = np.asarray(signal_cache["binned_valid"], dtype=bool)
        feature_labels = bin_names
        feature_matrix = np.concatenate(bins, axis=1).astype(np.float32)
        return baseline, selection_response, feature_matrix, valid, feature_labels
    raise ValueError(f"Unknown representation={representation}")


def _select_indices(
    baseline_train: np.ndarray,
    response_train: np.ndarray,
    targeted_mask: np.ndarray,
    d: int,
    selection_mode: str,
    targeted_policy: str,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scores = builder._compute_responsiveness_scores(baseline_train, response_train)
    mean_effect = np.nanmean(response_train - baseline_train, axis=0)
    finite_mask = np.isfinite(scores) & np.isfinite(mean_effect)
    positive_mask = finite_mask & (mean_effect > 0.0)

    rng = np.random.default_rng(seed)

    def _pick(pool: np.ndarray, k: int) -> np.ndarray:
        if pool.size < k:
            return np.array([], dtype=int)
        if selection_mode == "responsive_random":
            return np.sort(rng.choice(pool, size=k, replace=False)).astype(int)
        order = np.argsort(scores[pool])[::-1]
        return np.sort(pool[order[:k]]).astype(int)

    if targeted_policy == "exclude":
        pool = np.flatnonzero(positive_mask & ~targeted_mask)
        if pool.size < d:
            pool = np.flatnonzero(finite_mask & ~targeted_mask)
        picked = _pick(pool, d)
        return picked, {"candidate_count": int(pool.size), "policy": targeted_policy}

    if targeted_policy == "include":
        pool = np.flatnonzero(positive_mask)
        if pool.size < d:
            pool = np.flatnonzero(finite_mask)
        picked = _pick(pool, d)
        return picked, {"candidate_count": int(pool.size), "policy": targeted_policy}

    if targeted_policy == "mixed":
        n_targeted = max(1, d // 2)
        n_non = d - n_targeted
        target_pool = np.flatnonzero(positive_mask & targeted_mask)
        non_pool = np.flatnonzero(positive_mask & ~targeted_mask)
        if target_pool.size < n_targeted:
            target_pool = np.flatnonzero(finite_mask & targeted_mask)
        if non_pool.size < n_non:
            non_pool = np.flatnonzero(finite_mask & ~targeted_mask)
        targ = _pick(target_pool, n_targeted)
        non = _pick(non_pool, n_non)
        if targ.size != n_targeted or non.size != n_non:
            return np.array([], dtype=int), {
                "candidate_count_targeted": int(target_pool.size),
                "candidate_count_nontargeted": int(non_pool.size),
                "policy": targeted_policy,
                "warning": f"Need {n_targeted} targeted and {n_non} non-targeted neurons.",
            }
        return np.concatenate([targ, non]).astype(int), {
            "candidate_count_targeted": int(target_pool.size),
            "candidate_count_nontargeted": int(non_pool.size),
            "policy": targeted_policy,
        }

    raise ValueError(f"Unknown targeted_policy={targeted_policy}")


def _build_feature_matrix_for_selected(
    feature_matrix: np.ndarray,
    selected: np.ndarray,
    representation: str,
    n_neurons_selected: int,
    feature_labels: Sequence[str],
) -> np.ndarray:
    if representation == "scalar":
        return np.asarray(feature_matrix[:, selected], dtype=np.float64)
    n_trials = feature_matrix.shape[0]
    n_total_neurons = feature_matrix.shape[1] // len(feature_labels)
    reshaped = feature_matrix.reshape(n_trials, len(feature_labels), n_total_neurons)
    pieces = [reshaped[:, bin_idx, selected] for bin_idx in range(len(feature_labels))]
    out = np.concatenate(pieces, axis=1).astype(np.float64)
    return out


def _evaluate_raw_features(train_x: np.ndarray, test_x: np.ndarray, families: Sequence[str], seed: int) -> Tuple[float, float, float]:
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_x, test_x = _winsorize_train_apply(train_x, test_x)
    mappings = _fit_train_only_ecdf(train_x)
    u_train = _apply_train_only_ecdf(train_x, mappings)
    u_test = _apply_train_only_ecdf(test_x, mappings)

    hub = int(_estimate_hub_by_correlation(norm.ppf(np.clip(u_train, 1e-6, 1.0 - 1e-6))))
    order = [hub] + [idx for idx in range(train_x.shape[1]) if idx != hub]
    gaussian_nll = float(_score_gaussian_from_pobs(u_train, u_test))
    trunc_vine = _with_quieter_repo_logging(
        _fit_truncated_cvine_level0,
        x_train=u_train.astype(np.float32),
        families=list(families),
        order=order,
    )
    trunc_nll = float(_score_vine_on_uniforms(trunc_vine, u_test))
    full_vine = _with_quieter_repo_logging(
        _fit_parametric_vine,
        x_train=u_train.astype(np.float32),
        families=list(families),
        optimize_structure=False,
        seed=seed,
    )
    full_nll = float(_score_vine_on_uniforms(full_vine, u_test))
    return gaussian_nll, trunc_nll, full_nll


def _evaluate_static_variant(
    variant_cfg: Dict[str, Any],
    neural_data: Dict[str, Any],
    data_root: Path,
    families: Sequence[str],
    seed: int,
    n_repeats: int,
    train_fraction: float,
    min_trials_per_slice: int,
    selection_mode: str,
    session_signal_cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    lookup_rows: List[Dict[str, Any]] = []

    variant = str(variant_cfg["variant"])
    signal_type = str(variant_cfg["signal_type"])
    representation = str(variant_cfg["representation"])
    window_definition = str(variant_cfg["window_definition"])
    targeted_policy = str(variant_cfg["targeted_policy"])
    d_total = int(variant_cfg["d"])
    n_neurons_selected = int(variant_cfg["n_neurons_selected"])

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        cache_key = (session_id, signal_type)
        if cache_key not in session_signal_cache:
            session_signal_cache[cache_key] = _materialize_session_signal(session_id, payload, signal_type, data_root)
        signal_cache = session_signal_cache[cache_key]

        baseline_all, selection_response_all, feature_matrix_all, valid_mask, feature_labels = _prepare_representation_data(
            signal_cache=signal_cache,
            representation=representation,
            window_definition=window_definition,
        )
        session_df = payload["trial_table"].reset_index(drop=True).copy()
        session_df = session_df[valid_mask].copy().reset_index(drop=True)
        if session_df.empty:
            continue
        baseline = baseline_all[valid_mask]
        selection_response = selection_response_all[valid_mask]
        feature_matrix = feature_matrix_all[valid_mask]
        targeted_mask = _targeted_union_mask(payload, baseline.shape[1])

        dose_counts = session_df.groupby("dose").size()
        eligible_doses = sorted(float(dose) for dose, n in dose_counts.items() if int(n) >= int(min_trials_per_slice))
        if not eligible_doses:
            continue

        for repeat_id in range(int(n_repeats)):
            rng = np.random.default_rng(_session_seed(seed, session_id, extra=1999 * repeat_id + d_total))
            dose_split_map: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
            train_pool: List[int] = []

            for dose in eligible_doses:
                group_idx = session_df.index[session_df["dose"] == dose].to_numpy(dtype=int)
                train_pos, test_pos = _split_positions_random(len(group_idx), train_fraction, rng)
                train_idx = group_idx[train_pos]
                test_idx = group_idx[test_pos]
                if len(train_idx) < 5 or len(test_idx) < 3:
                    continue
                dose_split_map[float(dose)] = (train_idx, test_idx)
                train_pool.extend(train_idx.tolist())

            if not dose_split_map:
                continue

            selected, selection_meta = _select_indices(
                baseline_train=baseline[np.array(sorted(set(train_pool)), dtype=int)],
                response_train=selection_response[np.array(sorted(set(train_pool)), dtype=int)],
                targeted_mask=targeted_mask,
                d=n_neurons_selected,
                selection_mode=selection_mode,
                targeted_policy=targeted_policy,
                seed=_session_seed(seed, session_id, extra=repeat_id + 31 * d_total),
            )
            if selected.size != n_neurons_selected:
                continue

            for dose, (train_idx, test_idx) in sorted(dose_split_map.items()):
                raw_matrix = _build_feature_matrix_for_selected(
                    feature_matrix=feature_matrix,
                    selected=selected,
                    representation=representation,
                    n_neurons_selected=n_neurons_selected,
                    feature_labels=feature_labels,
                )
                train_x = raw_matrix[train_idx]
                test_x = raw_matrix[test_idx]
                slice_seed = _session_seed(seed, session_id, extra=repeat_id + int(100 * dose) + d_total)
                try:
                    gaussian_nll, trunc_nll, full_nll = _evaluate_raw_features(train_x, test_x, families, slice_seed)
                except Exception as exc:
                    LOGGER.warning("%s %s dose=%s repeat=%s failed: %s", variant, session_id, dose, repeat_id, exc)
                    continue

                split_id = f"{variant}__{session_id}__dose_{int(dose):03d}__repeat_{repeat_id:02d}"
                train_df = session_df.loc[train_idx].copy()
                test_df = session_df.loc[test_idx].copy()
                train_wins, test_wins = _winsorize_train_apply(train_x, test_x)
                mappings = _fit_train_only_ecdf(train_wins)
                u_train = _apply_train_only_ecdf(train_wins, mappings)
                u_test = _apply_train_only_ecdf(test_wins, mappings)

                for frame, u_slice, split_role in [(train_df, u_train, "train"), (test_df, u_test, "test")]:
                    for row_idx, (_, row) in enumerate(frame.iterrows()):
                        out = {
                            "variant": variant,
                            "analysis_view": "dose_static",
                            "session_id": row["session_id"],
                            "trial_id": row["trial_id"],
                            "dose": float(row["dose"]),
                            "condition_label": row["condition"],
                            "trial_order": int(row["trial_order_within_session"]),
                            "split": split_role,
                            "split_id": split_id,
                            "selection_mode": selection_mode,
                            "targeted_policy": targeted_policy,
                            "window_definition": window_definition,
                            "signal_type": signal_type,
                            "representation": representation,
                            "block_id": "",
                        }
                        for j in range(u_slice.shape[1]):
                            out[f"x{j + 1}"] = float(u_slice[row_idx, j])
                        benchmark_rows.append(out)

                lookup_rows.extend(
                    _feature_lookup_rows(
                        payload=payload,
                        session_id=session_id,
                        selected=selected,
                        analysis_view="dose_static",
                        split_id=split_id,
                        selection_mode=selection_mode,
                        targeted_policy=targeted_policy,
                    )
                )
                tc_higher = trunc_nll - full_nll
                common = {
                    "variant": variant,
                    "analysis_view": "dose_static",
                    "session_id": session_id,
                    "dose": float(dose),
                    "block_id": "",
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "selection_mode": selection_mode,
                    "targeted_policy": targeted_policy,
                    "window_definition": window_definition,
                    "signal_type": signal_type,
                    "representation": representation,
                    "d": d_total,
                    "repeat_id": int(repeat_id),
                    "split_id": split_id,
                    "tc_higher": float(tc_higher),
                }
                metrics_rows.extend(
                    [
                        {**common, "model": "gaussian", "heldout_nll": gaussian_nll, "delta_vs_gaussian": 0.0},
                        {**common, "model": "truncated_vine", "heldout_nll": trunc_nll, "delta_vs_gaussian": gaussian_nll - trunc_nll},
                        {**common, "model": "full_vine", "heldout_nll": full_nll, "delta_vs_gaussian": gaussian_nll - full_nll},
                    ]
                )

    benchmark_df = pd.DataFrame(benchmark_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    lookup_df = pd.DataFrame(lookup_rows).drop_duplicates() if lookup_rows else pd.DataFrame()
    return benchmark_df, metrics_df, lookup_df


def _summarize_variant(metrics_df: pd.DataFrame, variant_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if metrics_df.empty:
        return {
            "variant": variant_cfg["variant"],
            "window_definition": variant_cfg["window_definition"],
            "signal_type": variant_cfg["signal_type"],
            "targeted_policy": variant_cfg["targeted_policy"],
            "d": variant_cfg["d"],
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
            "status": "not_useful",
        }

    gaussian = metrics_df[metrics_df["model"] == "gaussian"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "gaussian_nll"})
    trunc = metrics_df[metrics_df["model"] == "truncated_vine"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "trunc_nll"})
    full = metrics_df[metrics_df["model"] == "full_vine"][["split_id", "heldout_nll", "tc_higher", "session_id"]].rename(columns={"heldout_nll": "full_nll"})
    merged = gaussian.merge(trunc, on="split_id").merge(full, on="split_id")
    full_delta = merged["gaussian_nll"] - merged["full_nll"]
    trunc_delta = merged["gaussian_nll"] - merged["trunc_nll"]
    full_beats_gauss = float(np.mean(merged["full_nll"] < merged["gaussian_nll"]))
    full_beats_trunc = float(np.mean(merged["full_nll"] < merged["trunc_nll"]))
    tc_mean = float(np.mean(merged["tc_higher"]))
    tc_median = float(np.median(merged["tc_higher"]))
    full_delta_mean = float(np.mean(full_delta))
    trunc_delta_mean = float(np.mean(trunc_delta))
    return {
        "variant": variant_cfg["variant"],
        "window_definition": variant_cfg["window_definition"],
        "signal_type": variant_cfg["signal_type"],
        "targeted_policy": variant_cfg["targeted_policy"],
        "representation": variant_cfg["representation"],
        "d": variant_cfg["d"],
        "session_count": int(metrics_df["session_id"].nunique()),
        "slice_count": int(merged["split_id"].nunique()),
        "gaussian_mean_nll": float(metrics_df[metrics_df["model"] == "gaussian"]["heldout_nll"].mean()),
        "trunc_mean_nll": float(metrics_df[metrics_df["model"] == "truncated_vine"]["heldout_nll"].mean()),
        "full_mean_nll": float(metrics_df[metrics_df["model"] == "full_vine"]["heldout_nll"].mean()),
        "mean_full_delta_vs_gaussian": full_delta_mean,
        "prop_full_beats_gaussian": full_beats_gauss,
        "prop_full_beats_trunc": full_beats_trunc,
        "mean_tc_higher": tc_mean,
        "median_tc_higher": tc_median,
        "status": _status_label(full_delta_mean, full_beats_gauss, full_beats_trunc, tc_mean, trunc_delta_mean),
    }


def _pick_best_variant(summary_df: pd.DataFrame, variant_subset: Sequence[str]) -> str:
    frame = summary_df[summary_df["variant"].isin(list(variant_subset))].copy()
    if frame.empty:
        raise RuntimeError("No candidate variants to choose from.")
    frame = frame.sort_values(
        ["mean_full_delta_vs_gaussian", "prop_full_beats_gaussian", "mean_tc_higher"],
        ascending=[False, False, False],
    )
    return str(frame.iloc[0]["variant"])


def _run_dynamic_variant(
    variant_cfg: Dict[str, Any],
    neural_data: Dict[str, Any],
    data_root: Path,
    families: Sequence[str],
    seed: int,
    train_fraction: float,
    dynamic_min_block_trials: int,
    selection_mode: str,
    session_signal_cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    variant = str(variant_cfg["variant"])
    signal_type = str(variant_cfg["signal_type"])
    representation = str(variant_cfg["representation"])
    window_definition = str(variant_cfg["window_definition"])
    targeted_policy = str(variant_cfg["targeted_policy"])
    n_neurons_selected = int(variant_cfg["n_neurons_selected"])
    d_total = int(variant_cfg["d"])

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        cache_key = (session_id, signal_type)
        if cache_key not in session_signal_cache:
            session_signal_cache[cache_key] = _materialize_session_signal(session_id, payload, signal_type, data_root)
        signal_cache = session_signal_cache[cache_key]
        baseline_all, selection_response_all, feature_matrix_all, valid_mask, feature_labels = _prepare_representation_data(
            signal_cache=signal_cache,
            representation=representation,
            window_definition=window_definition,
        )
        session_df = payload["trial_table"].reset_index(drop=True).copy()
        session_df = session_df[valid_mask].copy().reset_index(drop=True)
        if session_df.empty:
            continue
        baseline = baseline_all[valid_mask]
        selection_response = selection_response_all[valid_mask]
        feature_matrix = feature_matrix_all[valid_mask]
        targeted_mask = _targeted_union_mask(payload, baseline.shape[1])

        dose_counts = session_df.groupby("dose").size().sort_index()
        eligible = dose_counts[dose_counts >= 3 * int(dynamic_min_block_trials)]
        if eligible.empty:
            continue
        dose = float(np.max(eligible.index.to_numpy(dtype=float)))
        dose_df = session_df[session_df["dose"] == dose].sort_values("trial_order_within_session").copy()
        block_frames = np.array_split(dose_df.index.to_numpy(dtype=int), 3)
        block_defs = {"early": block_frames[0], "middle": block_frames[1], "late": block_frames[2]}
        if min(len(block) for block in block_defs.values()) < int(dynamic_min_block_trials):
            continue

        raw_matrix = None
        for block_id, block_idx in block_defs.items():
            rng = np.random.default_rng(_session_seed(seed, session_id, extra=int(dose) + len(block_id) + d_total))
            train_pos, test_pos = _split_positions_random(len(block_idx), train_fraction, rng)
            train_idx = np.asarray(block_idx[train_pos], dtype=int)
            test_idx = np.asarray(block_idx[test_pos], dtype=int)
            if len(train_idx) < 5 or len(test_idx) < 3:
                continue
            selected, _meta = _select_indices(
                baseline_train=baseline[train_idx],
                response_train=selection_response[train_idx],
                targeted_mask=targeted_mask,
                d=n_neurons_selected,
                selection_mode=selection_mode,
                targeted_policy=targeted_policy,
                seed=_session_seed(seed, session_id, extra=int(dose) + len(block_id) + 101),
            )
            if selected.size != n_neurons_selected:
                continue
            if raw_matrix is None:
                raw_matrix = _build_feature_matrix_for_selected(
                    feature_matrix=feature_matrix,
                    selected=selected,
                    representation=representation,
                    n_neurons_selected=n_neurons_selected,
                    feature_labels=feature_labels,
                )
            train_x = raw_matrix[train_idx]
            test_x = raw_matrix[test_idx]
            try:
                gaussian_nll, trunc_nll, full_nll = _evaluate_raw_features(
                    train_x=train_x,
                    test_x=test_x,
                    families=families,
                    seed=_session_seed(seed, session_id, extra=int(dose) + len(block_id) + 303),
                )
            except Exception as exc:
                LOGGER.warning("Dynamic %s %s %s failed: %s", variant, session_id, block_id, exc)
                continue
            tc_higher = trunc_nll - full_nll
            common = {
                "variant": variant,
                "analysis_view": "within_session_dynamic",
                "session_id": session_id,
                "dose": dose,
                "block_id": block_id,
                "signal_type": signal_type,
                "window_definition": window_definition,
                "targeted_policy": targeted_policy,
                "representation": representation,
                "d": d_total,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "selection_mode": selection_mode,
                "tc_higher": float(tc_higher),
            }
            rows.extend(
                [
                    {**common, "model": "gaussian", "heldout_nll": gaussian_nll, "delta_vs_gaussian": 0.0},
                    {**common, "model": "truncated_vine", "heldout_nll": trunc_nll, "delta_vs_gaussian": gaussian_nll - trunc_nll},
                    {**common, "model": "full_vine", "heldout_nll": full_nll, "delta_vs_gaussian": gaussian_nll - full_nll},
                ]
            )
    return pd.DataFrame(rows)


def _session_quality_table(
    best_variant_cfg: Dict[str, Any],
    best_static_metrics: pd.DataFrame,
    neural_data: Dict[str, Any],
    data_root: Path,
    session_signal_cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> pd.DataFrame:
    signal_type = str(best_variant_cfg["signal_type"])
    representation = str(best_variant_cfg["representation"])
    window_definition = str(best_variant_cfg["window_definition"])
    rows: List[Dict[str, Any]] = []

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        cache_key = (session_id, signal_type)
        if cache_key not in session_signal_cache:
            session_signal_cache[cache_key] = _materialize_session_signal(session_id, payload, signal_type, data_root)
        signal_cache = session_signal_cache[cache_key]
        baseline_all, selection_response_all, _features, valid_mask, _labels = _prepare_representation_data(
            signal_cache=signal_cache,
            representation=representation,
            window_definition=window_definition,
        )
        session_df = payload["trial_table"].reset_index(drop=True).copy()
        session_df = session_df[valid_mask].copy().reset_index(drop=True)
        baseline = baseline_all[valid_mask]
        response = selection_response_all[valid_mask]
        delta = np.asarray(response - baseline, dtype=np.float64)
        trial_mean = np.nanmean(delta, axis=1) if delta.size else np.array([], dtype=float)
        mean_effect = np.nanmean(delta, axis=0) if delta.size else np.array([], dtype=float)
        rows.append(
            {
                "session_id": session_id,
                "n_trials": int(len(session_df)),
                "n_neurons_available": int(payload["roi_lookup"].shape[0]),
                "response_magnitude": float(np.nanmean(np.abs(delta))) if delta.size else np.nan,
                "response_variability": float(np.nanstd(trial_mean)) if trial_mean.size else np.nan,
                "responsive_fraction": float(np.nanmean(mean_effect > 0.0)) if mean_effect.size else np.nan,
                "mean_full_delta_vs_gaussian": float(
                    best_static_metrics[
                        (best_static_metrics["session_id"] == session_id) & (best_static_metrics["model"] == "full_vine")
                    ]["delta_vs_gaussian"].mean()
                ),
                "mean_tc_higher": float(
                    best_static_metrics[
                        (best_static_metrics["session_id"] == session_id) & (best_static_metrics["model"] == "full_vine")
                    ]["tc_higher"].mean()
                ),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    mag_q = df["response_magnitude"].quantile(0.25)
    resp_q = df["responsive_fraction"].quantile(0.25)
    trials_q = df["n_trials"].quantile(0.25)
    high_mag = df["response_magnitude"].quantile(0.6)
    high_resp = df["responsive_fraction"].quantile(0.6)
    flags: List[str] = []
    for row in df.itertuples(index=False):
        if row.response_magnitude <= mag_q or row.responsive_fraction <= resp_q or row.n_trials <= trials_q:
            flags.append("possible_low_quality")
        elif row.response_magnitude >= high_mag and row.responsive_fraction >= high_resp:
            flags.append("higher_quality")
        else:
            flags.append("typical")
    df["quality_flag"] = flags
    return df.sort_values("session_id").reset_index(drop=True)


def _write_csv_dual(df: pd.DataFrame, filename: str, out_root: Path, results_data_root: Path) -> None:
    df.to_csv(out_root / filename, index=False)
    df.to_csv(results_data_root / filename, index=False)


def _plot_group(summary_df: pd.DataFrame, variants: Sequence[str], value_cols: Sequence[str], titles: Sequence[str], out_path: Path) -> None:
    frame = summary_df[summary_df["variant"].isin(list(variants))].copy()
    fig, axes = plt.subplots(1, len(value_cols), figsize=(5.2 * len(value_cols), 4.1))
    if len(value_cols) == 1:
        axes = [axes]
    for ax, col, title in zip(axes, value_cols, titles):
        ax.bar(frame["variant"], frame[col], color="#4c78a8" if "delta" in col else "#2ca02c")
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_signal_comparison(summary_df: pd.DataFrame, variants: Sequence[str], out_path: Path) -> None:
    frame = summary_df[summary_df["variant"].isin(list(variants))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].bar(frame["signal_type"], frame["mean_full_delta_vs_gaussian"], color="#d62728")
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[0].set_title("Full vs Gaussian by signal")
    axes[1].bar(frame["signal_type"], frame["mean_tc_higher"], color="#2ca02c")
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_title("TC_higher by signal")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_session_quality(session_quality_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {
        "possible_low_quality": "#d62728",
        "typical": "#7f7f7f",
        "higher_quality": "#2ca02c",
    }
    for flag, group in session_quality_df.groupby("quality_flag"):
        axes[0].scatter(group["response_magnitude"], group["mean_full_delta_vs_gaussian"], color=colors.get(flag, "#7f7f7f"), label=flag, s=42)
        axes[1].scatter(group["responsive_fraction"], group["mean_tc_higher"], color=colors.get(flag, "#7f7f7f"), label=flag, s=42)
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Response magnitude")
    axes[0].set_ylabel("Mean full delta vs Gaussian")
    axes[0].set_title("Session quality vs full-vine gain")
    axes[1].set_xlabel("Responsive fraction")
    axes[1].set_ylabel("Mean TC_higher")
    axes[1].set_title("Session quality vs higher-order gain")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_variant(best_static_metrics: pd.DataFrame, best_dynamic_metrics: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    dose_frame = (
        best_static_metrics[best_static_metrics["model"] == "full_vine"]
        .groupby("dose")["delta_vs_gaussian"]
        .apply(lambda s: pd.Series({"mean": _mean_ci(s)[0], "ci": _mean_ci(s)[1]}))
        .unstack()
        .reset_index()
        .sort_values("dose")
    )
    axes[0].errorbar(dose_frame["dose"], dose_frame["mean"], yerr=dose_frame["ci"], marker="o", capsize=3, linewidth=2, color="#d62728")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Best last-chance variant: full vs Gaussian")
    axes[0].set_xlabel("Dose")
    axes[0].set_ylabel("Delta vs Gaussian")

    if best_dynamic_metrics.empty:
        axes[1].text(0.5, 0.5, "No dynamic blocks", ha="center", va="center")
        axes[1].axis("off")
    else:
        full_dyn = best_dynamic_metrics[best_dynamic_metrics["model"] == "full_vine"].copy()
        order = {"early": 0, "middle": 1, "late": 2}
        full_dyn["block_order"] = full_dyn["block_id"].map(order)
        summary = full_dyn.groupby(["block_id", "block_order"])["delta_vs_gaussian"].mean().reset_index().sort_values("block_order")
        for session_id, group in full_dyn.groupby("session_id"):
            group = group.sort_values("block_order")
            axes[1].plot(group["block_order"], group["delta_vs_gaussian"], color="#bdbdbd", alpha=0.6)
        axes[1].plot(summary["block_order"], summary["delta_vs_gaussian"], color="#31a354", linewidth=2.8, marker="o")
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set_xticks([0, 1, 2], ["early", "middle", "late"])
        axes[1].set_title("Dynamic check under best variant")
        axes[1].set_ylabel("Delta vs Gaussian")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root).resolve()
    results_data_root = results_root / "data"
    results_plots_root = results_root / "plots"
    results_runs_root = results_root / "runs"
    for path in [results_root, results_data_root, results_plots_root, results_runs_root]:
        path.mkdir(parents=True, exist_ok=True)

    families = list(FAMILY_VARIANTS[args.family_variant])
    data_root = Path(args.data_root).resolve()
    manifest = builder.build_manifest(data_root)
    trials_df, neural_data, trial_inference_rows = builder.build_trials(data_root, manifest)

    session_signal_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    variant_configs: List[Dict[str, Any]] = []
    metrics_frames: List[pd.DataFrame] = []
    benchmark_frames: List[pd.DataFrame] = []
    lookup_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    # Phase 1: window / artifact audit on the current best static construction base.
    window_variant_names: List[str] = []
    for window_name in WINDOW_DEFS:
        cfg = {
            "variant": f"window_{window_name}",
            "representation": "scalar",
            "window_definition": window_name,
            "signal_type": "spks",
            "targeted_policy": "exclude",
            "d": 4,
            "n_neurons_selected": 4,
        }
        variant_configs.append(cfg)
        window_variant_names.append(cfg["variant"])
        bdf, mdf, ldf = _evaluate_static_variant(
            variant_cfg=cfg,
            neural_data=neural_data,
            data_root=data_root,
            families=families,
            seed=int(args.seed),
            n_repeats=int(args.n_repeats),
            train_fraction=float(args.train_fraction),
            min_trials_per_slice=int(args.min_trials_per_slice),
            selection_mode=args.selection_mode,
            session_signal_cache=session_signal_cache,
        )
        if not bdf.empty:
            benchmark_frames.append(bdf)
        if not mdf.empty:
            metrics_frames.append(mdf)
        if not ldf.empty:
            ldf = ldf.copy()
            ldf["variant"] = cfg["variant"]
            ldf["signal_type"] = cfg["signal_type"]
            ldf["window_definition"] = cfg["window_definition"]
            lookup_frames.append(ldf)
        summary_rows.append(_summarize_variant(mdf, cfg))

    summary_df = pd.DataFrame(summary_rows)
    best_window_variant = _pick_best_variant(summary_df, window_variant_names)
    best_window_name = str(summary_df.loc[summary_df["variant"] == best_window_variant, "window_definition"].iloc[0])
    LOGGER.info("Best window variant: %s (%s)", best_window_variant, best_window_name)

    # Phase 3: signal audit using the best window.
    signal_variant_names: List[str] = []
    for signal_type in SIGNAL_TYPES:
        cfg = {
            "variant": f"signal_{signal_type}_window_{best_window_name}",
            "representation": "scalar",
            "window_definition": best_window_name,
            "signal_type": signal_type,
            "targeted_policy": "exclude",
            "d": 4,
            "n_neurons_selected": 4,
        }
        variant_configs.append(cfg)
        signal_variant_names.append(cfg["variant"])
        bdf, mdf, ldf = _evaluate_static_variant(
            variant_cfg=cfg,
            neural_data=neural_data,
            data_root=data_root,
            families=families,
            seed=int(args.seed),
            n_repeats=int(args.n_repeats),
            train_fraction=float(args.train_fraction),
            min_trials_per_slice=int(args.min_trials_per_slice),
            selection_mode=args.selection_mode,
            session_signal_cache=session_signal_cache,
        )
        if not bdf.empty:
            benchmark_frames.append(bdf)
        if not mdf.empty:
            metrics_frames.append(mdf)
        if not ldf.empty:
            ldf = ldf.copy()
            ldf["variant"] = cfg["variant"]
            ldf["signal_type"] = cfg["signal_type"]
            ldf["window_definition"] = cfg["window_definition"]
            lookup_frames.append(ldf)
        summary_rows.append(_summarize_variant(mdf, cfg))

    summary_df = pd.DataFrame(summary_rows)
    best_signal_variant = _pick_best_variant(summary_df, signal_variant_names)
    best_signal_type = str(summary_df.loc[summary_df["variant"] == best_signal_variant, "signal_type"].iloc[0])
    LOGGER.info("Best signal variant: %s (%s)", best_signal_variant, best_signal_type)

    # Phase 4: neuron policy audit using the best scalar signal/window.
    policy_variant_names: List[str] = []
    for targeted_policy in ["exclude", "include", "mixed"]:
        cfg = {
            "variant": f"policy_{targeted_policy}_{best_signal_type}_{best_window_name}",
            "representation": "scalar",
            "window_definition": best_window_name,
            "signal_type": best_signal_type,
            "targeted_policy": targeted_policy,
            "d": 4,
            "n_neurons_selected": 4,
        }
        variant_configs.append(cfg)
        policy_variant_names.append(cfg["variant"])
        bdf, mdf, ldf = _evaluate_static_variant(
            variant_cfg=cfg,
            neural_data=neural_data,
            data_root=data_root,
            families=families,
            seed=int(args.seed),
            n_repeats=int(args.n_repeats),
            train_fraction=float(args.train_fraction),
            min_trials_per_slice=int(args.min_trials_per_slice),
            selection_mode=args.selection_mode,
            session_signal_cache=session_signal_cache,
        )
        if not bdf.empty:
            benchmark_frames.append(bdf)
        if not mdf.empty:
            metrics_frames.append(mdf)
        if not ldf.empty:
            ldf = ldf.copy()
            ldf["variant"] = cfg["variant"]
            ldf["signal_type"] = cfg["signal_type"]
            ldf["window_definition"] = cfg["window_definition"]
            lookup_frames.append(ldf)
        summary_rows.append(_summarize_variant(mdf, cfg))

    # Phase 2: small time-binned representation using the best signal/window.
    binned_cfg = {
        "variant": f"binned_{best_signal_type}_{best_window_name}",
        "representation": "binned",
        "window_definition": "two_bin_stim_post",
        "signal_type": best_signal_type,
        "targeted_policy": "exclude",
        "d": 4,
        "n_neurons_selected": 2,
    }
    variant_configs.append(binned_cfg)
    bdf, mdf, ldf = _evaluate_static_variant(
        variant_cfg=binned_cfg,
        neural_data=neural_data,
        data_root=data_root,
        families=families,
        seed=int(args.seed),
        n_repeats=int(args.n_repeats),
        train_fraction=float(args.train_fraction),
        min_trials_per_slice=int(args.min_trials_per_slice),
        selection_mode=args.selection_mode,
        session_signal_cache=session_signal_cache,
    )
    if not bdf.empty:
        benchmark_frames.append(bdf)
    if not mdf.empty:
        metrics_frames.append(mdf)
    if not ldf.empty:
        ldf = ldf.copy()
        ldf["variant"] = binned_cfg["variant"]
        ldf["signal_type"] = binned_cfg["signal_type"]
        ldf["window_definition"] = binned_cfg["window_definition"]
        lookup_frames.append(ldf)
    summary_rows.append(_summarize_variant(mdf, binned_cfg))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_full_delta_vs_gaussian", "prop_full_beats_gaussian", "mean_tc_higher"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_overall_variant = str(summary_df.iloc[0]["variant"])
    best_variant_cfg = next(cfg for cfg in variant_configs if cfg["variant"] == best_overall_variant)
    LOGGER.info("Best overall representation variant: %s", best_overall_variant)

    metrics_audit_df = pd.concat(metrics_frames, axis=0, ignore_index=True) if metrics_frames else pd.DataFrame()
    benchmark_audit_df = pd.concat(benchmark_frames, axis=0, ignore_index=True) if benchmark_frames else pd.DataFrame()
    lookup_audit_df = pd.concat(lookup_frames, axis=0, ignore_index=True).drop_duplicates() if lookup_frames else pd.DataFrame()

    best_dynamic_df = _run_dynamic_variant(
        variant_cfg=best_variant_cfg,
        neural_data=neural_data,
        data_root=data_root,
        families=families,
        seed=int(args.seed),
        train_fraction=float(args.train_fraction),
        dynamic_min_block_trials=int(args.dynamic_min_block_trials),
        selection_mode=args.selection_mode,
        session_signal_cache=session_signal_cache,
    )
    if not best_dynamic_df.empty:
        metrics_audit_df = pd.concat([metrics_audit_df, best_dynamic_df], axis=0, ignore_index=True)

    best_static_metrics = metrics_audit_df[
        (metrics_audit_df["analysis_view"] == "dose_static") & (metrics_audit_df["variant"] == best_overall_variant)
    ].copy()
    session_quality_df = _session_quality_table(
        best_variant_cfg=best_variant_cfg,
        best_static_metrics=best_static_metrics,
        neural_data=neural_data,
        data_root=data_root,
        session_signal_cache=session_signal_cache,
    )

    if not session_quality_df.empty:
        good_sessions = session_quality_df[session_quality_df["quality_flag"] != "possible_low_quality"]["session_id"].tolist()
        if good_sessions:
            high_quality_metrics = best_static_metrics[best_static_metrics["session_id"].isin(good_sessions)]
            if not high_quality_metrics.empty:
                gaussian = high_quality_metrics[high_quality_metrics["model"] == "gaussian"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "gaussian_nll"})
                full = high_quality_metrics[high_quality_metrics["model"] == "full_vine"][["split_id", "heldout_nll", "tc_higher"]].rename(columns={"heldout_nll": "full_nll"})
                merged = gaussian.merge(full, on="split_id")
                summary_df["exploratory_high_quality_subset_full_delta"] = np.nan
                summary_df["exploratory_high_quality_subset_tc"] = np.nan
                summary_df.loc[summary_df["variant"] == best_overall_variant, "exploratory_high_quality_subset_full_delta"] = float(
                    np.mean(merged["gaussian_nll"] - merged["full_nll"])
                )
                summary_df.loc[summary_df["variant"] == best_overall_variant, "exploratory_high_quality_subset_tc"] = float(
                    np.mean(merged["tc_higher"])
                )

    temporal_axis_label = _dynamic_axis_label(best_dynamic_df)

    _write_csv_dual(summary_df, "representation_audit_summary.csv", out_root, results_data_root)
    _write_csv_dual(session_quality_df, "session_quality_audit.csv", out_root, results_data_root)
    _write_csv_dual(metrics_audit_df, "metrics_table_representation_audit.csv", out_root, results_data_root)
    if not lookup_audit_df.empty:
        lookup_audit_df.to_csv(out_root / "neuron_lookup_representation_audit.csv", index=False)
        lookup_audit_df.to_csv(results_data_root / "neuron_lookup_representation_audit.csv", index=False)
    if not benchmark_audit_df.empty:
        benchmark_audit_df.to_parquet(out_root / "benchmark_table_representation_audit.parquet", index=False)
        benchmark_audit_df.to_parquet(results_data_root / "benchmark_table_representation_audit.parquet", index=False)

    _plot_group(
        summary_df=summary_df,
        variants=window_variant_names,
        value_cols=["mean_full_delta_vs_gaussian", "mean_tc_higher"],
        titles=["Window audit: full vs Gaussian", "Window audit: TC_higher"],
        out_path=results_plots_root / "fig_repr_window_comparison.png",
    )
    _plot_signal_comparison(summary_df, signal_variant_names, results_plots_root / "fig_repr_signal_type_comparison.png")
    _plot_group(
        summary_df=summary_df,
        variants=policy_variant_names,
        value_cols=["mean_full_delta_vs_gaussian", "mean_tc_higher"],
        titles=["Targeted-policy audit: full vs Gaussian", "Targeted-policy audit: TC_higher"],
        out_path=results_plots_root / "fig_repr_targeted_policy_comparison.png",
    )
    _plot_session_quality(session_quality_df, results_plots_root / "fig_repr_session_quality.png")
    _plot_best_variant(best_static_metrics, best_dynamic_df, results_plots_root / "fig_repr_best_last_chance_variant.png")

    for fig_name in [
        "fig_repr_window_comparison.png",
        "fig_repr_signal_type_comparison.png",
        "fig_repr_targeted_policy_comparison.png",
        "fig_repr_session_quality.png",
        "fig_repr_best_last_chance_variant.png",
    ]:
        src = results_plots_root / fig_name
        if src.exists():
            (out_root / fig_name).write_bytes(src.read_bytes())

    metadata = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "results_root": str(results_root),
        "family_variant": args.family_variant,
        "families": families,
        "selection_mode": args.selection_mode,
        "n_repeats": int(args.n_repeats),
        "train_fraction": float(args.train_fraction),
        "min_trials_per_slice": int(args.min_trials_per_slice),
        "dynamic_min_block_trials": int(args.dynamic_min_block_trials),
        "window_definitions_seconds": WINDOW_DEFS,
        "baseline_window_seconds": BASELINE_WINDOW,
        "binned_windows_seconds": {name: window for name, window in BIN_DEFS},
        "signal_types": {
            "spks": "Suite2p spks matrix",
            "F": "raw fluorescence matrix F",
            "Fcorr": "neuropil-corrected fluorescence computed as F - 0.7 * Fneu",
        },
        "best_window_variant": best_window_variant,
        "best_signal_variant": best_signal_variant,
        "best_overall_variant": best_overall_variant,
        "best_overall_variant_config": best_variant_cfg,
        "temporal_axis_label": temporal_axis_label,
        "trial_inference_row_count": int(len(trial_inference_rows)),
    }
    _write_json(out_root / "representation_audit_metadata.json", metadata)
    _write_json(results_data_root / "representation_audit_metadata.json", metadata)

    LOGGER.info("Representation audit best overall variant: %s", best_overall_variant)
    LOGGER.info("Temporal axis under best audited variant: %s", temporal_axis_label)


if __name__ == "__main__":
    main()
