#!/usr/bin/env python3
"""Decision-oriented neuroscientific validity audit for the Dalgleish benchmark."""

from __future__ import annotations

import argparse
import itertools
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
    _split_positions_random,
    _write_json,
    configure_logging,
)
from scripts.debug_stimulation_exp.run_dalgleish_representation_audit import (
    _evaluate_raw_features,
    _load_signal_matrix,
    _compute_single_window,
    _mean_ci,
)
from scripts.debug_stimulation_exp.run_dalgleish_real_data_decision_rerun import _dynamic_axis_label


LOGGER = logging.getLogger("dalgleish_neurosci_validity")

BASELINE_WINDOW = (-1.0, -0.1)
WINDOW_STUDY: Dict[str, Tuple[float, float]] = {
    "stim_0p0_1p0": (0.0, 1.0),
    "stim_0p0_0p5": (0.0, 0.5),
    "delayed_0p2_0p7": (0.2, 0.7),
    "early_post_0p7_1p4": (0.7, 1.4),
    "long_0p2_1p2": (0.2, 1.2),
}
REP_DEPTH_BINS: Dict[str, List[Tuple[str, Tuple[float, float]]]] = {
    "scalar_1bin": [("resp_0p2_1p4", (0.2, 1.4))],
    "two_bin": [
        ("late_stim_0p2_0p7", (0.2, 0.7)),
        ("early_post_0p7_1p4", (0.7, 1.4)),
    ],
    "four_bin": [
        ("bin1_0p2_0p5", (0.2, 0.5)),
        ("bin2_0p5_0p8", (0.5, 0.8)),
        ("bin3_0p8_1p1", (0.8, 1.1)),
        ("bin4_1p1_1p4", (1.1, 1.4)),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dalgleish neuroscientific validity audit.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--results_root", default="results/stimulation_exp_benchmark")
    parser.add_argument("--family_variant", choices=sorted(FAMILY_VARIANTS.keys()), default="stable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--selection_mode", choices=["responsive_random", "topk_responsive"], default="topk_responsive")
    parser.add_argument("--min_trials_floor", type=int, default=18)
    parser.add_argument("--dynamic_min_block_trials", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _pairwise_jaccard(selections: List[Sequence[int]]) -> float:
    if len(selections) < 2:
        return 1.0
    vals: List[float] = []
    for a, b in itertools.combinations(selections, 2):
        aa = set(int(x) for x in a)
        bb = set(int(x) for x in b)
        denom = len(aa | bb)
        vals.append(1.0 if denom == 0 else len(aa & bb) / denom)
    return float(np.mean(vals)) if vals else 1.0


def _targeted_union_mask(payload: Dict[str, Any], n_neurons: int) -> np.ndarray:
    mask = np.zeros(n_neurons, dtype=bool)
    for roi_indices in payload.get("targeted_roi_by_program", {}).values():
        for idx in roi_indices:
            if 0 <= int(idx) < n_neurons:
                mask[int(idx)] = True
    return mask


def _select_indices(
    baseline_train: np.ndarray,
    response_train: np.ndarray,
    targeted_mask: np.ndarray,
    n_neurons_requested: int,
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

    if targeted_policy == "non_targeted":
        pool = np.flatnonzero(positive_mask & ~targeted_mask)
        if pool.size < n_neurons_requested:
            pool = np.flatnonzero(finite_mask & ~targeted_mask)
        picked = _pick(pool, n_neurons_requested)
        return picked, {"candidate_count": int(pool.size), "policy": targeted_policy}

    if targeted_policy == "targeted":
        pool = np.flatnonzero(positive_mask & targeted_mask)
        if pool.size < n_neurons_requested:
            pool = np.flatnonzero(finite_mask & targeted_mask)
        picked = _pick(pool, n_neurons_requested)
        return picked, {"candidate_count": int(pool.size), "policy": targeted_policy}

    if targeted_policy == "mixed":
        n_target = max(1, n_neurons_requested // 2)
        n_non = n_neurons_requested - n_target
        target_pool = np.flatnonzero(positive_mask & targeted_mask)
        non_pool = np.flatnonzero(positive_mask & ~targeted_mask)
        if target_pool.size < n_target:
            target_pool = np.flatnonzero(finite_mask & targeted_mask)
        if non_pool.size < n_non:
            non_pool = np.flatnonzero(finite_mask & ~targeted_mask)
        targ = _pick(target_pool, n_target)
        non = _pick(non_pool, n_non)
        if targ.size != n_target or non.size != n_non:
            return np.array([], dtype=int), {
                "policy": targeted_policy,
                "candidate_targeted": int(target_pool.size),
                "candidate_non_targeted": int(non_pool.size),
                "warning": f"Need {n_target} targeted and {n_non} non-targeted neurons.",
            }
        return np.concatenate([targ, non]).astype(int), {
            "policy": targeted_policy,
            "candidate_targeted": int(target_pool.size),
            "candidate_non_targeted": int(non_pool.size),
        }

    raise ValueError(f"Unknown targeted_policy={targeted_policy}")


def _prepare_windows_for_session(
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
    out: Dict[str, Any] = {
        "frame_rate_hz": frame_rate_hz,
        "baseline": baseline,
        "baseline_valid": baseline_valid,
        "windows": {},
    }
    for name, secs in WINDOW_STUDY.items():
        arr, valid = _compute_single_window(traces, stim_frames, frame_rate_hz, secs)
        out["windows"][name] = {
            "array": arr,
            "valid": baseline_valid & valid & session_df["is_valid"].to_numpy(dtype=bool) & session_df["dose"].notna().to_numpy(),
            "seconds": secs,
        }
    for rep_name, bins in REP_DEPTH_BINS.items():
        bin_arrays = []
        valid = baseline_valid.copy()
        for bin_label, secs in bins:
            arr, v = _compute_single_window(traces, stim_frames, frame_rate_hz, secs)
            bin_arrays.append((bin_label, secs, arr))
            valid &= v
        valid &= session_df["is_valid"].to_numpy(dtype=bool) & session_df["dose"].notna().to_numpy()
        out["windows"][rep_name] = {
            "bins": bin_arrays,
            "valid": valid,
        }
    return out


def _build_variant_specs() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for window_name in WINDOW_STUDY:
        specs.append(
            {
                "study": "window_validity",
                "variant": f"window_{window_name}_10n",
                "signal_type": "spks",
                "targeted_policy": "non_targeted",
                "feature_kind": "scalar_window",
                "window_name": window_name,
                "n_neurons": 10,
            }
        )
    for count in [4, 10, 25, 50]:
        specs.append(
            {
                "study": "neuron_count_scaling",
                "variant": f"scaling_early_post_{count:02d}n",
                "signal_type": "spks",
                "targeted_policy": "non_targeted",
                "feature_kind": "scalar_window",
                "window_name": "early_post_0p7_1p4",
                "n_neurons": count,
            }
        )
    specs.extend(
        [
            {
                "study": "representation_depth",
                "variant": "depth_scalar_10n",
                "signal_type": "spks",
                "targeted_policy": "non_targeted",
                "feature_kind": "depth_bins",
                "depth_name": "scalar_1bin",
                "n_neurons": 10,
            },
            {
                "study": "representation_depth",
                "variant": "depth_2bin_5n",
                "signal_type": "spks",
                "targeted_policy": "non_targeted",
                "feature_kind": "depth_bins",
                "depth_name": "two_bin",
                "n_neurons": 5,
            },
            {
                "study": "representation_depth",
                "variant": "depth_4bin_3n",
                "signal_type": "spks",
                "targeted_policy": "non_targeted",
                "feature_kind": "depth_bins",
                "depth_name": "four_bin",
                "n_neurons": 3,
            },
        ]
    )
    for signal_type in ["spks", "F", "Fcorr"]:
        specs.append(
            {
                "study": "signal_type",
                "variant": f"signal_{signal_type}_early_post_10n",
                "signal_type": signal_type,
                "targeted_policy": "non_targeted",
                "feature_kind": "scalar_window",
                "window_name": "early_post_0p7_1p4",
                "n_neurons": 10,
            }
        )
    for policy in ["non_targeted", "targeted", "mixed"]:
        specs.append(
            {
                "study": "neuron_policy",
                "variant": f"policy_{policy}_depth2bin",
                "signal_type": "spks",
                "targeted_policy": policy,
                "feature_kind": "depth_bins",
                "depth_name": "two_bin",
                "n_neurons": 5,
            }
        )
    return specs


def _variant_feature_definition(spec: Dict[str, Any]) -> Tuple[List[Tuple[str, Tuple[float, float]]], int]:
    if spec["feature_kind"] == "scalar_window":
        window_name = str(spec["window_name"])
        return [(window_name, WINDOW_STUDY[window_name])], int(spec["n_neurons"])
    depth_name = str(spec["depth_name"])
    return REP_DEPTH_BINS[depth_name], int(spec["n_neurons"])


def _feature_semantics_rows(spec: Dict[str, Any], frame_rates: Sequence[float]) -> List[Dict[str, Any]]:
    bins, n_neurons = _variant_feature_definition(spec)
    rates = np.asarray(list(frame_rates), dtype=float)
    rows: List[Dict[str, Any]] = []
    feature_idx = 1
    for bin_idx, (bin_label, secs) in enumerate(bins):
        start_s, end_s = secs
        frame_offsets = np.round(np.array([start_s, end_s])[:, None] * rates[None, :]).astype(int)
        frame_counts = frame_offsets[1] - frame_offsets[0]
        for neuron_slot in range(1, n_neurons + 1):
            rows.append(
                {
                    "variant": spec["variant"],
                    "study": spec["study"],
                    "feature_index": feature_idx,
                    "feature_name": f"x{feature_idx}",
                    "neuron_slot": neuron_slot,
                    "signal_type": spec["signal_type"],
                    "bin_label": bin_label,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "duration_s": float(end_s - start_s),
                    "median_n_frames": float(np.median(frame_counts)),
                    "min_n_frames": int(np.min(frame_counts)),
                    "max_n_frames": int(np.max(frame_counts)),
                    "semantics": f"neuron slot {neuron_slot} {spec['signal_type']} mean from {start_s:.1f} s to {end_s:.1f} s",
                }
            )
            feature_idx += 1
    return rows


def _evaluate_variant(
    spec: Dict[str, Any],
    neural_data: Dict[str, Any],
    data_root: Path,
    families: Sequence[str],
    seed: int,
    n_repeats: int,
    train_fraction: float,
    min_trials_floor: int,
    selection_mode: str,
    cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    selection_records: List[Dict[str, Any]] = []
    viability_rows: List[Dict[str, Any]] = []
    bins, n_neurons = _variant_feature_definition(spec)
    total_dim = len(bins) * n_neurons
    min_trials_required = max(int(min_trials_floor), 2 * total_dim + 4)

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        cache_key = (session_id, str(spec["signal_type"]))
        if cache_key not in cache:
            cache[cache_key] = _prepare_windows_for_session(session_id, payload, str(spec["signal_type"]), data_root)
        prep = cache[cache_key]
        session_df = payload["trial_table"].reset_index(drop=True).copy()

        if spec["feature_kind"] == "scalar_window":
            w = prep["windows"][str(spec["window_name"])]
            valid_mask = np.asarray(w["valid"], dtype=bool)
            baseline_all = np.asarray(prep["baseline"], dtype=np.float32)
            response_all = np.asarray(w["array"], dtype=np.float32)
            bin_arrays = [(str(spec["window_name"]), WINDOW_STUDY[str(spec["window_name"])], response_all)]
        else:
            depth = prep["windows"][str(spec["depth_name"])]
            valid_mask = np.asarray(depth["valid"], dtype=bool)
            baseline_all = np.asarray(prep["baseline"], dtype=np.float32)
            bin_arrays = depth["bins"]
            response_all = np.mean(np.stack([arr for _label, _secs, arr in bin_arrays], axis=0), axis=0).astype(np.float32)

        session_df = session_df[valid_mask].copy().reset_index(drop=True)
        if session_df.empty:
            continue
        baseline = baseline_all[valid_mask]
        response = response_all[valid_mask]
        bin_data = [(label, secs, np.asarray(arr[valid_mask], dtype=np.float32)) for label, secs, arr in bin_arrays]

        targeted_mask = _targeted_union_mask(payload, baseline.shape[1])
        dose_counts = session_df.groupby("dose").size()
        eligible_doses = sorted(float(dose) for dose, n in dose_counts.items() if int(n) >= min_trials_required)
        viability_rows.append(
            {
                "variant": spec["variant"],
                "study": spec["study"],
                "session_id": session_id,
                "signal_type": spec["signal_type"],
                "targeted_policy": spec["targeted_policy"],
                "n_total_rois": int(payload["roi_lookup"].shape[0]),
                "n_targeted_rois": int(np.sum(targeted_mask)),
                "n_non_targeted_rois": int(np.sum(~targeted_mask)),
                "n_neurons_requested": n_neurons,
                "n_feature_dims": total_dim,
                "min_trials_required": int(min_trials_required),
                "eligible_dose_count": int(len(eligible_doses)),
                "candidate_policy": spec["targeted_policy"],
            }
        )
        if not eligible_doses:
            continue

        for repeat_id in range(int(n_repeats)):
            rng = np.random.default_rng(seed + repeat_id + sum(ord(ch) for ch in session_id) + total_dim)
            dose_splits: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
            train_pool: List[int] = []
            for dose in eligible_doses:
                group_idx = session_df.index[session_df["dose"] == dose].to_numpy(dtype=int)
                train_pos, test_pos = _split_positions_random(len(group_idx), train_fraction, rng)
                train_idx = group_idx[train_pos]
                test_idx = group_idx[test_pos]
                if len(train_idx) < 5 or len(test_idx) < 3:
                    continue
                dose_splits[dose] = (train_idx, test_idx)
                train_pool.extend(train_idx.tolist())
            if not dose_splits:
                continue

            train_pool_arr = np.array(sorted(set(train_pool)), dtype=int)
            selected, selection_meta = _select_indices(
                baseline_train=baseline[train_pool_arr],
                response_train=response[train_pool_arr],
                targeted_mask=targeted_mask,
                n_neurons_requested=n_neurons,
                selection_mode=selection_mode,
                targeted_policy=str(spec["targeted_policy"]),
                seed=seed + repeat_id + 17 * total_dim,
            )
            selection_records.append(
                {
                    "variant": spec["variant"],
                    "study": spec["study"],
                    "session_id": session_id,
                    "repeat_id": repeat_id,
                    "selected_indices": selected.tolist(),
                    "selection_meta": selection_meta,
                }
            )
            if selected.size != n_neurons:
                continue

            raw_features = np.concatenate([arr[:, selected] for _label, _secs, arr in bin_data], axis=1).astype(np.float64)
            for dose, (train_idx, test_idx) in dose_splits.items():
                train_x = raw_features[train_idx]
                test_x = raw_features[test_idx]
                try:
                    gaussian_nll, trunc_nll, full_nll = _evaluate_raw_features(
                        train_x=train_x,
                        test_x=test_x,
                        families=families,
                        seed=seed + repeat_id + int(dose) + total_dim,
                    )
                except Exception as exc:
                    LOGGER.warning("%s %s dose=%s repeat=%s failed: %s", spec["variant"], session_id, dose, repeat_id, exc)
                    continue
                tc_higher = trunc_nll - full_nll
                common = {
                    "study": spec["study"],
                    "variant": spec["variant"],
                    "session_id": session_id,
                    "dose": float(dose),
                    "signal_type": spec["signal_type"],
                    "targeted_policy": spec["targeted_policy"],
                    "n_neurons_requested": n_neurons,
                    "n_feature_dims": total_dim,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "repeat_id": repeat_id,
                    "split_id": f"{spec['variant']}__{session_id}__dose_{int(dose):03d}__repeat_{repeat_id:02d}",
                    "tc_higher": float(tc_higher),
                }
                rows.extend(
                    [
                        {**common, "analysis_view": "dose_static", "model": "gaussian", "heldout_nll": gaussian_nll, "delta_vs_gaussian": 0.0},
                        {**common, "analysis_view": "dose_static", "model": "truncated_vine", "heldout_nll": trunc_nll, "delta_vs_gaussian": gaussian_nll - trunc_nll},
                        {**common, "analysis_view": "dose_static", "model": "full_vine", "heldout_nll": full_nll, "delta_vs_gaussian": gaussian_nll - full_nll},
                    ]
                )
    return pd.DataFrame(rows), selection_records, viability_rows


def _summarize_variant(spec: Dict[str, Any], metrics_df: pd.DataFrame) -> Dict[str, Any]:
    bins, n_neurons = _variant_feature_definition(spec)
    total_dim = len(bins) * n_neurons
    if metrics_df.empty:
        return {
            "study": spec["study"],
            "variant": spec["variant"],
            "window_definition": spec.get("window_name", spec.get("depth_name", "")),
            "signal_type": spec["signal_type"],
            "targeted_policy": spec["targeted_policy"],
            "n_neurons_requested": n_neurons,
            "n_feature_dims": total_dim,
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
    gaussian = metrics_df[metrics_df["model"] == "gaussian"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "g"})
    trunc = metrics_df[metrics_df["model"] == "truncated_vine"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "t"})
    full = metrics_df[metrics_df["model"] == "full_vine"][["split_id", "heldout_nll", "tc_higher", "session_id"]].rename(columns={"heldout_nll": "f"})
    merged = gaussian.merge(trunc, on="split_id").merge(full, on="split_id")
    full_delta = merged["g"] - merged["f"]
    trunc_delta = merged["g"] - merged["t"]
    full_beats_g = float(np.mean(merged["f"] < merged["g"]))
    full_beats_t = float(np.mean(merged["f"] < merged["t"]))
    tc_mean = float(np.mean(merged["tc_higher"]))
    tc_median = float(np.median(merged["tc_higher"]))
    full_delta_mean = float(np.mean(full_delta))
    trunc_delta_mean = float(np.mean(trunc_delta))
    session_count = int(metrics_df["session_id"].nunique())
    slice_count = int(merged["split_id"].nunique())
    # Decision-oriented status: require both coverage and some evidence that gains are
    # not purely pairwise before calling a variant promising for the paper story.
    if slice_count < 20 or session_count < 3:
        status = "undersampled_or_unviable"
    elif total_dim > 24 and slice_count < 40:
        status = "undersampled_or_unviable"
    elif full_delta_mean > 0.05 and tc_mean > 0.02 and full_beats_g >= 0.55 and full_beats_t >= 0.50:
        status = "promising_main_figure"
    elif full_delta_mean > 0.0 and (full_beats_g >= 0.50 or tc_median > 0.0):
        status = "usable_but_heterogeneous"
    elif trunc_delta_mean > 0.05 and full_delta_mean <= 0.0:
        status = "pairwise_dominant"
    else:
        status = "not_useful"
    return {
        "study": spec["study"],
        "variant": spec["variant"],
        "window_definition": spec.get("window_name", spec.get("depth_name", "")),
        "signal_type": spec["signal_type"],
        "targeted_policy": spec["targeted_policy"],
        "n_neurons_requested": n_neurons,
        "n_feature_dims": total_dim,
        "session_count": session_count,
        "slice_count": slice_count,
        "gaussian_mean_nll": float(metrics_df[metrics_df["model"] == "gaussian"]["heldout_nll"].mean()),
        "trunc_mean_nll": float(metrics_df[metrics_df["model"] == "truncated_vine"]["heldout_nll"].mean()),
        "full_mean_nll": float(metrics_df[metrics_df["model"] == "full_vine"]["heldout_nll"].mean()),
        "mean_full_delta_vs_gaussian": full_delta_mean,
        "prop_full_beats_gaussian": full_beats_g,
        "prop_full_beats_trunc": full_beats_t,
        "mean_tc_higher": tc_mean,
        "median_tc_higher": tc_median,
        "status": status,
    }


def _window_validity_table(neural_data: Dict[str, Any], data_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        traces = _load_signal_matrix(data_root / session_id, "spks")
        session_df = payload["trial_table"].reset_index(drop=True)
        frame_rate_hz = float(pd.to_numeric(session_df["frame_rate_hz"], errors="coerce").dropna().iloc[0])
        del traces
        for name, secs in WINDOW_STUDY.items():
            start_s, end_s = secs
            start_f = int(round(start_s * frame_rate_hz))
            end_f = int(round(end_s * frame_rate_hz))
            rows.append(
                {
                    "session_id": session_id,
                    "window_definition": name,
                    "frame_rate_hz": frame_rate_hz,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "duration_s": float(end_s - start_s),
                    "start_frame_offset": start_f,
                    "end_frame_offset": end_f,
                    "n_frames": int(end_f - start_f),
                    "sampling_note": "frame offsets computed at the effective imaging frame rate used by the builder",
                }
            )
    return pd.DataFrame(rows).sort_values(["window_definition", "session_id"]).reset_index(drop=True)


def _trial_spacing_table(trials_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    usable = trials_df[trials_df["is_valid"] & trials_df["dose"].notna()].copy()
    for session_id, group in usable.groupby("session_id"):
        times = np.sort(group["stimulation_time"].to_numpy(dtype=float))
        iti = np.diff(times) if len(times) >= 2 else np.array([], dtype=float)
        rows.append(
            {
                "session_id": session_id,
                "n_trials": int(len(group)),
                "min_iti_s": float(np.min(iti)) if iti.size else np.nan,
                "median_iti_s": float(np.median(iti)) if iti.size else np.nan,
                "frac_iti_lt_2s": float(np.mean(iti < 2.0)) if iti.size else np.nan,
                "frac_iti_lt_3s": float(np.mean(iti < 3.0)) if iti.size else np.nan,
                "carryover_risk_flag": "possible_carryover" if iti.size and (np.mean(iti < 3.0) > 0.1 or np.min(iti) < 2.0) else "low_carryover_risk",
            }
        )
    return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)


def _selection_stability_table(selection_records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not selection_records:
        return pd.DataFrame()
    frame = pd.DataFrame(selection_records)
    for (variant, session_id), group in frame.groupby(["variant", "session_id"]):
        rows.append(
            {
                "variant": variant,
                "session_id": session_id,
                "selection_stability": _pairwise_jaccard(group["selected_indices"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _session_quality_refined(
    neural_data: Dict[str, Any],
    best_variant: str,
    metrics_df: pd.DataFrame,
    window_table: pd.DataFrame,
    spacing_df: pd.DataFrame,
    selection_stability_df: pd.DataFrame,
) -> pd.DataFrame:
    best_full = metrics_df[(metrics_df["variant"] == best_variant) & (metrics_df["model"] == "full_vine") & (metrics_df["analysis_view"] == "dose_static")]
    rows: List[Dict[str, Any]] = []
    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        session_trials = payload["trial_table"].reset_index(drop=True)
        targeted_mask = _targeted_union_mask(payload, payload["roi_lookup"].shape[0])
        spacing_row = spacing_df[spacing_df["session_id"] == session_id]
        rows.append(
            {
                "session_id": session_id,
                "n_trials": int(session_trials["is_valid"].sum()),
                "n_neurons_available": int(payload["roi_lookup"].shape[0]),
                "n_targeted_mapped": int(np.sum(targeted_mask)),
                "n_non_targeted_available": int(np.sum(~targeted_mask)),
                "mean_full_delta_vs_gaussian": float(best_full[best_full["session_id"] == session_id]["delta_vs_gaussian"].mean()),
                "mean_tc_higher": float(best_full[best_full["session_id"] == session_id]["tc_higher"].mean()),
                "selection_stability": float(
                    selection_stability_df[
                        (selection_stability_df["variant"] == best_variant) & (selection_stability_df["session_id"] == session_id)
                    ]["selection_stability"].mean()
                ),
                "min_iti_s": float(spacing_row["min_iti_s"].iloc[0]) if not spacing_row.empty else np.nan,
                "median_iti_s": float(spacing_row["median_iti_s"].iloc[0]) if not spacing_row.empty else np.nan,
                "carryover_risk_flag": str(spacing_row["carryover_risk_flag"].iloc[0]) if not spacing_row.empty else "unknown",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    flags: List[str] = []
    for row in df.itertuples(index=False):
        if row.n_trials < 180 or row.n_non_targeted_available < 50 or row.selection_stability < 0.5:
            flags.append("possible_low_quality")
        elif row.mean_full_delta_vs_gaussian > 0.02 and row.mean_tc_higher > 0.02 and row.carryover_risk_flag == "low_carryover_risk":
            flags.append("higher_quality")
        else:
            flags.append("typical")
    df["quality_flag"] = flags
    return df.sort_values("session_id").reset_index(drop=True)


def _plot_bar(summary_df: pd.DataFrame, study: str, x: str, out_path: Path, title_left: str, title_right: str) -> None:
    frame = summary_df[summary_df["study"] == study].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].bar(frame[x], frame["mean_full_delta_vs_gaussian"], color="#d62728")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title(title_left)
    axes[0].tick_params(axis="x", rotation=28)
    axes[1].bar(frame[x], frame["mean_tc_higher"], color="#2ca02c")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title(title_right)
    axes[1].tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_session_quality(session_quality_df: pd.DataFrame, out_path: Path) -> None:
    if session_quality_df.empty:
        return
    colors = {"possible_low_quality": "#d62728", "typical": "#7f7f7f", "higher_quality": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for flag, group in session_quality_df.groupby("quality_flag"):
        axes[0].scatter(group["n_non_targeted_available"], group["mean_full_delta_vs_gaussian"], color=colors.get(flag, "#7f7f7f"), label=flag, s=40)
        axes[1].scatter(group["median_iti_s"], group["mean_tc_higher"], color=colors.get(flag, "#7f7f7f"), label=flag, s=40)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("Non-targeted neurons available")
    axes[0].set_ylabel("Mean full delta vs Gaussian")
    axes[0].set_title("Neuron availability vs full-vine gain")
    axes[1].set_xlabel("Median inter-trial interval (s)")
    axes[1].set_ylabel("Mean TC_higher")
    axes[1].set_title("Trial spacing vs higher-order gain")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic(best_dynamic_df: pd.DataFrame, out_path: Path) -> None:
    if best_dynamic_df.empty:
        return
    full = best_dynamic_df[best_dynamic_df["model"] == "full_vine"].copy()
    block_order = {"early": 0, "middle": 1, "late": 2}
    full["block_order"] = full["block_id"].map(block_order)
    summary = full.groupby(["block_id", "block_order"])["delta_vs_gaussian"].mean().reset_index().sort_values("block_order")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for session_id, group in full.groupby("session_id"):
        group = group.sort_values("block_order")
        ax.plot(group["block_order"], group["delta_vs_gaussian"], color="#bdbdbd", alpha=0.7)
    ax.plot(summary["block_order"], summary["delta_vs_gaussian"], color="#1f77b4", linewidth=2.5, marker="o")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks([0, 1, 2], ["early", "middle", "late"])
    ax.set_ylabel("Full-vine delta vs Gaussian")
    ax.set_title("Dynamic check under best neuro-validity variant")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    results_root = Path(args.results_root).resolve()
    data_dir = results_root / "data"
    plots_dir = results_root / "plots"
    runs_dir = results_root / "runs"
    for path in [out_root, results_root, data_dir, plots_dir, runs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest = builder.build_manifest(data_root)
    trials_df, neural_data, _trial_inference_rows = builder.build_trials(data_root, manifest)
    families = list(FAMILY_VARIANTS[args.family_variant])

    specs = _build_variant_specs()
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    metrics_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    selection_records: List[Dict[str, Any]] = []
    viability_rows: List[Dict[str, Any]] = []

    frame_rates = sorted(
        float(pd.to_numeric(payload["trial_table"]["frame_rate_hz"], errors="coerce").dropna().iloc[0])
        for payload in neural_data.values()
        if payload.get("used", False)
    )
    feature_semantics_rows: List[Dict[str, Any]] = []
    for spec in specs:
        feature_semantics_rows.extend(_feature_semantics_rows(spec, frame_rates))
        metrics_df, sel_records, viability = _evaluate_variant(
            spec=spec,
            neural_data=neural_data,
            data_root=data_root,
            families=families,
            seed=int(args.seed),
            n_repeats=int(args.n_repeats),
            train_fraction=float(args.train_fraction),
            min_trials_floor=int(args.min_trials_floor),
            selection_mode=args.selection_mode,
            cache=cache,
        )
        if not metrics_df.empty:
            metrics_frames.append(metrics_df)
        selection_records.extend(sel_records)
        viability_rows.extend(viability)
        summary_rows.append(_summarize_variant(spec, metrics_df))
        LOGGER.info("%s complete: %d metric rows", spec["variant"], len(metrics_df))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_full_delta_vs_gaussian", "prop_full_beats_gaussian", "mean_tc_higher"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    metrics_df = pd.concat(metrics_frames, axis=0, ignore_index=True) if metrics_frames else pd.DataFrame()
    selection_stability_df = _selection_stability_table(selection_records)
    neuron_count_df = pd.DataFrame(viability_rows)
    if not selection_stability_df.empty and not neuron_count_df.empty:
        neuron_count_df = neuron_count_df.merge(selection_stability_df, on=["variant", "session_id"], how="left")
    window_validity_df = _window_validity_table(neural_data, data_root)
    trial_spacing_df = _trial_spacing_table(trials_df)

    # Choose the next benchmark candidate from viable, interpretable variants.
    candidate_frame = summary_df[summary_df["slice_count"] >= 40].copy()
    if candidate_frame.empty:
        candidate_frame = summary_df.copy()
    best_variant = str(candidate_frame.iloc[0]["variant"])
    best_static_metrics = metrics_df[(metrics_df["variant"] == best_variant) & (metrics_df["analysis_view"] == "dose_static")].copy()
    session_quality_refined_df = _session_quality_refined(
        neural_data=neural_data,
        best_variant=best_variant,
        metrics_df=metrics_df,
        window_table=window_validity_df,
        spacing_df=trial_spacing_df,
        selection_stability_df=selection_stability_df,
    )

    # Secondary dynamic check only for the best variant.
    best_spec = next(spec for spec in specs if spec["variant"] == best_variant)
    best_dynamic_rows: List[Dict[str, Any]] = []
    bins, n_neurons = _variant_feature_definition(best_spec)
    total_dim = len(bins) * n_neurons
    try:
        for session_id, payload in neural_data.items():
            if not payload.get("used", False):
                continue
            cache_key = (session_id, str(best_spec["signal_type"]))
            if cache_key not in cache:
                cache[cache_key] = _prepare_windows_for_session(session_id, payload, str(best_spec["signal_type"]), data_root)
            prep = cache[cache_key]
            session_df = payload["trial_table"].reset_index(drop=True).copy()
            if best_spec["feature_kind"] == "scalar_window":
                w = prep["windows"][str(best_spec["window_name"])]
                valid_mask = np.asarray(w["valid"], dtype=bool)
                response_all = np.asarray(w["array"], dtype=np.float32)
                bin_arrays = [(str(best_spec["window_name"]), WINDOW_STUDY[str(best_spec["window_name"])], response_all)]
            else:
                depth = prep["windows"][str(best_spec["depth_name"])]
                valid_mask = np.asarray(depth["valid"], dtype=bool)
                bin_arrays = depth["bins"]
                response_all = np.mean(np.stack([arr for _label, _secs, arr in bin_arrays], axis=0), axis=0).astype(np.float32)
            baseline_all = np.asarray(prep["baseline"], dtype=np.float32)
            session_df = session_df[valid_mask].copy().reset_index(drop=True)
            if session_df.empty:
                continue
            baseline = baseline_all[valid_mask]
            response = response_all[valid_mask]
            bin_data = [(label, secs, np.asarray(arr[valid_mask], dtype=np.float32)) for label, secs, arr in bin_arrays]
            targeted_mask = _targeted_union_mask(payload, baseline.shape[1])
            dose_counts = session_df.groupby("dose").size().sort_index()
            eligible = dose_counts[dose_counts >= 3 * int(args.dynamic_min_block_trials)]
            if eligible.empty:
                continue
            dose = float(np.max(eligible.index.to_numpy(dtype=float)))
            dose_df = session_df[session_df["dose"] == dose].sort_values("trial_order_within_session").copy()
            blocks = np.array_split(dose_df.index.to_numpy(dtype=int), 3)
            block_defs = {"early": blocks[0], "middle": blocks[1], "late": blocks[2]}
            if min(len(v) for v in block_defs.values()) < int(args.dynamic_min_block_trials):
                continue
            for block_id, block_idx in block_defs.items():
                rng = np.random.default_rng(int(args.seed) + sum(ord(c) for c in session_id) + len(block_id) + total_dim)
                train_pos, test_pos = _split_positions_random(len(block_idx), args.train_fraction, rng)
                train_idx = np.asarray(block_idx[train_pos], dtype=int)
                test_idx = np.asarray(block_idx[test_pos], dtype=int)
                if len(train_idx) < 5 or len(test_idx) < 3:
                    continue
                selected, _meta = _select_indices(
                    baseline_train=baseline[train_idx],
                    response_train=response[train_idx],
                    targeted_mask=targeted_mask,
                    n_neurons_requested=n_neurons,
                    selection_mode=args.selection_mode,
                    targeted_policy=str(best_spec["targeted_policy"]),
                    seed=int(args.seed) + len(block_id) + total_dim,
                )
                if selected.size != n_neurons:
                    continue
                raw_features = np.concatenate([arr[:, selected] for _label, _secs, arr in bin_data], axis=1).astype(np.float64)
                try:
                    gaussian_nll, trunc_nll, full_nll = _evaluate_raw_features(
                        train_x=raw_features[train_idx],
                        test_x=raw_features[test_idx],
                        families=families,
                        seed=int(args.seed) + len(block_id) + int(dose),
                    )
                except Exception as exc:
                    LOGGER.warning("Dynamic best variant %s %s %s failed: %s", best_variant, session_id, block_id, exc)
                    continue
                tc_higher = trunc_nll - full_nll
                common = {
                    "study": "dynamic_followup",
                    "variant": best_variant,
                    "analysis_view": "within_session_dynamic",
                    "session_id": session_id,
                    "dose": dose,
                    "block_id": block_id,
                    "signal_type": best_spec["signal_type"],
                    "targeted_policy": best_spec["targeted_policy"],
                    "n_neurons_requested": n_neurons,
                    "n_feature_dims": total_dim,
                    "tc_higher": float(tc_higher),
                }
                best_dynamic_rows.extend(
                    [
                        {**common, "model": "gaussian", "heldout_nll": gaussian_nll, "delta_vs_gaussian": 0.0},
                        {**common, "model": "truncated_vine", "heldout_nll": trunc_nll, "delta_vs_gaussian": gaussian_nll - trunc_nll},
                        {**common, "model": "full_vine", "heldout_nll": full_nll, "delta_vs_gaussian": gaussian_nll - full_nll},
                    ]
                )
    except Exception as exc:
        LOGGER.warning("Dynamic follow-up failed for %s: %s", best_variant, exc)

    best_dynamic_df = pd.DataFrame(best_dynamic_rows)
    temporal_axis_label = _dynamic_axis_label(best_dynamic_df)

    # Write outputs.
    summary_path = data_dir / "representation_audit_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(out_root / "representation_audit_summary.csv", index=False)
    metrics_df.to_csv(data_dir / "metrics_table_representation_audit.csv", index=False)
    metrics_df.to_csv(out_root / "metrics_table_representation_audit.csv", index=False)
    neuron_count_df.to_csv(data_dir / "neuron_count_audit.csv", index=False)
    neuron_count_df.to_csv(out_root / "neuron_count_audit.csv", index=False)
    window_validity_df.to_csv(data_dir / "window_validity_audit.csv", index=False)
    window_validity_df.to_csv(out_root / "window_validity_audit.csv", index=False)
    trial_spacing_df.to_csv(data_dir / "trial_spacing_audit.csv", index=False)
    trial_spacing_df.to_csv(out_root / "trial_spacing_audit.csv", index=False)
    session_quality_refined_df.to_csv(data_dir / "session_quality_audit_refined.csv", index=False)
    session_quality_refined_df.to_csv(out_root / "session_quality_audit_refined.csv", index=False)
    selection_stability_df.to_csv(data_dir / "selection_stability_audit.csv", index=False)
    selection_stability_df.to_csv(out_root / "selection_stability_audit.csv", index=False)
    pd.DataFrame(feature_semantics_rows).to_csv(data_dir / "feature_semantics_audit.csv", index=False)
    pd.DataFrame(feature_semantics_rows).to_csv(out_root / "feature_semantics_audit.csv", index=False)
    summary_df[summary_df["study"] == "representation_depth"].to_csv(data_dir / "representation_depth_audit.csv", index=False)
    summary_df[summary_df["study"] == "representation_depth"].to_csv(out_root / "representation_depth_audit.csv", index=False)
    if not best_dynamic_df.empty:
        best_dynamic_df.to_csv(data_dir / "dynamic_neurosci_audit.csv", index=False)
        best_dynamic_df.to_csv(out_root / "dynamic_neurosci_audit.csv", index=False)

    # Figures.
    _plot_bar(summary_df, "window_validity", "variant", plots_dir / "fig_neuro_window_validity.png", "Window audit: full vs Gaussian", "Window audit: TC_higher")
    _plot_bar(summary_df, "neuron_count_scaling", "n_neurons_requested", plots_dir / "fig_neuro_count_scaling.png", "Neuron-count scaling: full vs Gaussian", "Neuron-count scaling: TC_higher")
    _plot_bar(summary_df, "representation_depth", "variant", plots_dir / "fig_neuro_representation_depth.png", "Representation depth: full vs Gaussian", "Representation depth: TC_higher")
    _plot_session_quality(session_quality_refined_df, plots_dir / "fig_neuro_session_quality.png")
    _plot_dynamic(best_dynamic_df, plots_dir / "fig_neuro_best_variant_dynamic.png")

    metadata = {
        "best_variant": best_variant,
        "best_variant_temporal_axis": temporal_axis_label,
        "families": families,
        "family_variant": args.family_variant,
        "n_repeats": int(args.n_repeats),
        "train_fraction": float(args.train_fraction),
        "baseline_window_seconds": BASELINE_WINDOW,
        "window_study_seconds": WINDOW_STUDY,
        "representation_bins_seconds": {
            key: {label: secs for label, secs in bins} for key, bins in REP_DEPTH_BINS.items()
        },
        "decision_questions": [
            "Is the current analysis too undersampled in neuron count?",
            "What neuron count range seems scientifically meaningful and still numerically viable?",
            "Is the main bottleneck neuron count, time window, temporal compression, or neuron-class choice?",
            "What exact next benchmark should we run after this audit?",
        ],
    }
    _write_json(data_dir / "neurosci_validity_audit_metadata.json", metadata)
    _write_json(out_root / "neurosci_validity_audit_metadata.json", metadata)

    LOGGER.info("Best neuro-validity variant: %s | temporal axis: %s", best_variant, temporal_axis_label)


if __name__ == "__main__":
    main()
