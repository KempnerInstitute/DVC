#!/usr/bin/env python3
"""Run the Dalgleish photostimulation real-data benchmark for DVC.

This script reuses:
- the local Dalgleish builder for session/trial extraction
- the repository's own Gaussian, 1-truncated vine, and full-vine methods

It produces:
- dvc_ready/benchmark_table.parquet
- dvc_ready/metrics_table.csv
- dvc_ready/neuron_lookup.csv
- dvc_ready/benchmark_metadata.json
- candidate figure PNGs at the project root
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.stimulation_exp_benchmark.build_dalgleish_dvc_dataset as dalgleish_builder
from dvc_package.core.param_copula import copulaccdf, copulapdf
from dvc_package.experiments.simulation_benchmarks import (
    _estimate_hub_by_correlation,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
    _gaussian_copula_nll_given_corr,
)
import torch


LOGGER = logging.getLogger("dalgleish_real_data")
FAMILY_VARIANTS: Dict[str, List[str]] = {
    "default": ["ind", "gaussian", "student", "clayton", "frank", "gumbel", "joe"],
    "stable": ["ind", "gaussian", "student", "clayton"],
}
DEFAULT_FAMILY_VARIANT = "stable"


@dataclass
class SliceResult:
    benchmark_rows: List[Dict[str, Any]]
    metrics_rows: List[Dict[str, Any]]
    neuron_lookup_rows: List[Dict[str, Any]]
    warning: Optional[str] = None


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dalgleish real-data DVC benchmark.")
    parser.add_argument("--data_root", default="dataset_stimulation", help="Dalgleish dataset root.")
    parser.add_argument("--out_root", default="dvc_ready", help="Benchmark output directory.")
    parser.add_argument("--d", type=int, default=6, help="Feature dimension.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--selection_mode",
        choices=["responsive_random", "topk_responsive"],
        default="topk_responsive",
        help="Neuron selection mode.",
    )
    parser.add_argument(
        "--include_targeted_neurons",
        action="store_true",
        help="Include directly targeted neurons when they can be mapped to ROIs.",
    )
    parser.add_argument("--n_repeats", type=int, default=3, help="Repeated random splits for the dose benchmark.")
    parser.add_argument("--train_fraction", type=float, default=0.7, help="Train fraction per static analysis slice.")
    parser.add_argument("--min_trials_per_slice", type=int, default=0, help="Override minimum trials per session x dose slice.")
    parser.add_argument("--dynamic_min_trials", type=int, default=36, help="Minimum trials for within-session early/late analysis.")
    parser.add_argument(
        "--family_variant",
        choices=sorted(FAMILY_VARIANTS.keys()),
        default=DEFAULT_FAMILY_VARIANT,
        help="Family set used for truncated and full-vine fits.",
    )
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=0,
        help="Optional limit on the number of sessions, keeping the sessions with the most usable trials.",
    )
    parser.add_argument(
        "--optimize_full_structure",
        action="store_true",
        help="Enable full-vine structure optimization. Disabled by default for faster real-data runs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def _min_trials_per_slice(d: int, override: int) -> int:
    if override > 0:
        return int(override)
    return max(18, 2 * int(d) + 4)


def _split_positions_random(
    n_rows: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_rows < 2:
        return np.arange(n_rows, dtype=int), np.array([], dtype=int)
    n_train = int(math.floor(train_fraction * n_rows))
    n_train = min(max(n_train, 5), n_rows - 1)
    perm = rng.permutation(n_rows)
    train_pos = np.sort(perm[:n_train])
    test_pos = np.sort(perm[n_train:])
    return train_pos.astype(int), test_pos.astype(int)


def _split_positions_chronological(
    n_rows: int,
    train_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_rows < 2:
        return np.arange(n_rows, dtype=int), np.array([], dtype=int)
    n_train = int(math.floor(train_fraction * n_rows))
    n_train = min(max(n_train, 5), n_rows - 1)
    train_pos = np.arange(n_train, dtype=int)
    test_pos = np.arange(n_train, n_rows, dtype=int)
    return train_pos, test_pos


def _session_seed(base_seed: int, session_id: str, extra: int = 0) -> int:
    session_term = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(session_id)))
    return int(base_seed + session_term + extra)


def _targeted_mask(payload: Dict[str, Any], n_neurons: int) -> np.ndarray:
    mask = np.zeros(n_neurons, dtype=bool)
    for roi_indices in payload.get("targeted_roi_by_program", {}).values():
        for idx in roi_indices:
            if 0 <= int(idx) < n_neurons:
                mask[int(idx)] = True
    return mask


def _select_neurons_for_train_pool(
    payload: Dict[str, Any],
    train_indices: np.ndarray,
    d: int,
    selection_mode: str,
    exclude_targeted_neurons: bool,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    arrays = payload["arrays"]
    baseline_train = np.asarray(arrays["baseline"][train_indices], dtype=np.float64)
    stim_train = np.asarray(arrays["stim"][train_indices], dtype=np.float64)
    scores = dalgleish_builder._compute_responsiveness_scores(baseline_train, stim_train)
    mean_effect = np.nanmean(stim_train - baseline_train, axis=0)

    candidate_mask = np.isfinite(scores) & np.isfinite(mean_effect)
    if exclude_targeted_neurons:
        candidate_mask &= ~_targeted_mask(payload, scores.shape[0])
    responsive_mask = candidate_mask & (mean_effect > 0.0)
    candidate_indices = np.flatnonzero(responsive_mask)
    if candidate_indices.size < d:
        candidate_indices = np.flatnonzero(candidate_mask)

    meta: Dict[str, Any] = {
        "candidate_count": int(candidate_indices.size),
        "excluded_targeted_neurons": bool(exclude_targeted_neurons),
    }
    if candidate_indices.size < d:
        meta["warning"] = f"Only {candidate_indices.size} eligible neurons for d={d}."
        return np.array([], dtype=int), meta

    if selection_mode == "responsive_random":
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(candidate_indices, size=d, replace=False))
    else:
        order = np.argsort(scores[candidate_indices])[::-1]
        selected = np.sort(candidate_indices[order[:d]])

    meta["selected_scores"] = scores[selected].astype(float).tolist()
    meta["selected_mean_effect"] = mean_effect[selected].astype(float).tolist()
    return selected.astype(int), meta


def _fit_train_only_ecdf(train_x: np.ndarray) -> List[Dict[str, np.ndarray]]:
    return dalgleish_builder._fit_empirical_cdf(train_x)


def _apply_train_only_ecdf(x: np.ndarray, mappings: List[Dict[str, np.ndarray]]) -> np.ndarray:
    return dalgleish_builder._apply_empirical_cdf(x, mappings)


def _winsorize_train_apply(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_clip, test_clip, _bounds = dalgleish_builder._winsorize_train_apply_test(train_x, test_x)
    return train_clip, test_clip


def _score_gaussian_from_pobs(u_train: np.ndarray, u_test: np.ndarray) -> float:
    z_train = norm.ppf(np.clip(np.asarray(u_train, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    z_test = norm.ppf(np.clip(np.asarray(u_test, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    corr = np.corrcoef(z_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr = corr + 1e-4 * np.eye(corr.shape[0])
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    dstd = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    corr = corr / np.outer(dstd, dstd)
    np.fill_diagonal(corr, 1.0)
    return float(_gaussian_copula_nll_given_corr(z_test, corr))


def _score_vine_on_uniforms(vine: Any, u_test: np.ndarray) -> float:
    u = torch.tensor(np.asarray(u_test, dtype=np.float32), dtype=torch.float32)
    n, d = u.shape
    if n == 0:
        return float("nan")
    u_work = torch.zeros((n, d, d), dtype=torch.float32, device=u.device)
    u_work[:, 0, :] = u
    log_cop = torch.zeros(n, dtype=torch.float32, device=u.device)

    for level in range(d - 1):
        edges = vine.ind_vine[level] if level < len(vine.ind_vine) else []
        cops = vine.copulas[level] if level < len(vine.copulas) else []
        for edge_idx, edge in enumerate(edges):
            if edge_idx >= len(cops):
                continue
            cop = cops[edge_idx]
            i, j = int(edge[0]), int(edge[1])
            uv = torch.stack([u_work[:, level, i], u_work[:, level, j]], dim=1)
            pdf = torch.nan_to_num(copulapdf(cop, uv), nan=1e-30, posinf=1e30, neginf=1e-30).clamp_min(1e-30)
            log_cop = log_cop + torch.log(pdf)
            h_val = torch.nan_to_num(
                copulaccdf(cop, uv),
                nan=0.5,
                posinf=1.0 - 1e-6,
                neginf=1e-6,
            ).clamp(1e-6, 1.0 - 1e-6)
            u_work[:, level + 1, j] = h_val
            u_work[:, level + 1, i] = u_work[:, level, i]
    return float((-log_cop).mean().detach().cpu())


def _with_quieter_repo_logging(fn: Any, *args: Any, **kwargs: Any) -> Any:
    root_logger = logging.getLogger()
    prior_level = root_logger.level
    try:
        root_logger.setLevel(logging.ERROR)
        return fn(*args, **kwargs)
    finally:
        root_logger.setLevel(prior_level)


def _restrict_to_top_sessions(
    trials_df: pd.DataFrame,
    neural_data: Dict[str, Any],
    max_sessions: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    if max_sessions <= 0:
        return trials_df, neural_data, sorted(k for k, v in neural_data.items() if v.get("used", False))
    usable = trials_df[trials_df["is_valid"] & trials_df["dose"].notna()].copy()
    session_order = (
        usable.groupby("session_id").size().sort_values(ascending=False).head(int(max_sessions)).index.tolist()
    )
    kept = {k: v for k, v in neural_data.items() if k in session_order}
    filtered_trials = trials_df[trials_df["session_id"].isin(session_order)].copy()
    return filtered_trials, kept, session_order


def _format_feature_rows(
    slice_df: pd.DataFrame,
    u_slice: np.ndarray,
    analysis_view: str,
    split_role: str,
    split_id: str,
    selection_mode: str,
    targeted_policy: str,
    window_id: Optional[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row_idx, (_, meta_row) in enumerate(slice_df.iterrows()):
        out = {
            "analysis_view": analysis_view,
            "session_id": meta_row["session_id"],
            "trial_id": meta_row["trial_id"],
            "dose": float(meta_row["dose"]) if pd.notna(meta_row["dose"]) else np.nan,
            "condition_label": meta_row["condition"],
            "trial_order": int(meta_row["trial_order_within_session"]),
            "split": split_role,
            "split_id": split_id,
            "selection_mode": selection_mode,
            "targeted_policy": targeted_policy,
            "window_id": window_id if window_id is not None else "",
        }
        for col_idx in range(u_slice.shape[1]):
            out[f"x{col_idx + 1}"] = float(u_slice[row_idx, col_idx])
        rows.append(out)
    return rows


def _feature_lookup_rows(
    payload: Dict[str, Any],
    session_id: str,
    selected: np.ndarray,
    analysis_view: str,
    split_id: str,
    selection_mode: str,
    targeted_policy: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for j, roi_idx in enumerate(selected):
        roi_row = payload["roi_lookup"].iloc[int(roi_idx)]
        rows.append(
            {
                "analysis_view": analysis_view,
                "session_id": session_id,
                "split_id": split_id,
                "selection_mode": selection_mode,
                "targeted_policy": targeted_policy,
                "x_col": f"x{j + 1}",
                "roi_filtered_index": int(roi_row["roi_filtered_index"]),
                "roi_original_index": int(roi_row["roi_original_index"]),
                "x_center": float(roi_row["x_center"]) if pd.notna(roi_row["x_center"]) else np.nan,
                "y_center": float(roi_row["y_center"]) if pd.notna(roi_row["y_center"]) else np.nan,
            }
        )
    return rows


def _evaluate_slice(
    payload: Dict[str, Any],
    session_id: str,
    group_df: pd.DataFrame,
    train_global_idx: np.ndarray,
    test_global_idx: np.ndarray,
    selected: np.ndarray,
    analysis_view: str,
    split_id: str,
    selection_mode: str,
    targeted_policy: str,
    dose: float,
    condition_label: str,
    repeat_id: int,
    window_id: Optional[str],
    optimize_full_structure: bool,
    families: List[str],
) -> SliceResult:
    arrays = payload["arrays"]
    raw = np.asarray(arrays["diff"][:, selected], dtype=np.float64)
    train_x = raw[train_global_idx]
    test_x = raw[test_global_idx]

    # De-mean within this session x dose slice using train only.
    train_mean = np.nanmean(train_x, axis=0)
    train_centered = train_x - train_mean
    test_centered = test_x - train_mean
    train_wins, test_wins = _winsorize_train_apply(train_centered, test_centered)
    mappings = _fit_train_only_ecdf(train_wins)
    u_train = _apply_train_only_ecdf(train_wins, mappings)
    u_test = _apply_train_only_ecdf(test_wins, mappings)

    group_train_df = group_df.loc[train_global_idx].copy()
    group_test_df = group_df.loc[test_global_idx].copy()
    benchmark_rows = _format_feature_rows(
        slice_df=group_train_df,
        u_slice=u_train,
        analysis_view=analysis_view,
        split_role="train",
        split_id=split_id,
        selection_mode=selection_mode,
        targeted_policy=targeted_policy,
        window_id=window_id,
    )
    benchmark_rows.extend(
        _format_feature_rows(
            slice_df=group_test_df,
            u_slice=u_test,
            analysis_view=analysis_view,
            split_role="test",
            split_id=split_id,
            selection_mode=selection_mode,
            targeted_policy=targeted_policy,
            window_id=window_id,
        )
    )

    hub = int(_estimate_hub_by_correlation(norm.ppf(np.clip(u_train, 1e-6, 1.0 - 1e-6))))
    order = [hub] + [idx for idx in range(len(selected)) if idx != hub]

    gaussian_nll = _score_gaussian_from_pobs(u_train, u_test)
    warning = None
    try:
        trunc_vine = _with_quieter_repo_logging(
            _fit_truncated_cvine_level0,
            x_train=u_train.astype(np.float32),
            families=families,
            order=order,
        )
        trunc_nll = _score_vine_on_uniforms(trunc_vine, u_test)
    except Exception as exc:
        trunc_nll = float("nan")
        warning = f"Truncated vine failed for {split_id}: {exc}"

    try:
        full_vine = _with_quieter_repo_logging(
            _fit_parametric_vine,
            x_train=u_train.astype(np.float32),
            families=families,
            optimize_structure=optimize_full_structure,
            seed=repeat_id,
        )
        full_nll = _score_vine_on_uniforms(full_vine, u_test)
    except Exception as exc:
        full_nll = float("nan")
        msg = f"Full vine failed for {split_id}: {exc}"
        warning = msg if warning is None else f"{warning} | {msg}"

    pairwise_gain = gaussian_nll - trunc_nll if np.isfinite(gaussian_nll) and np.isfinite(trunc_nll) else np.nan
    full_gain = gaussian_nll - full_nll if np.isfinite(gaussian_nll) and np.isfinite(full_nll) else np.nan
    tc_higher = trunc_nll - full_nll if np.isfinite(trunc_nll) and np.isfinite(full_nll) else np.nan

    common = {
        "analysis_view": analysis_view,
        "session_id": session_id,
        "dose": float(dose),
        "condition_label": condition_label,
        "window_id": window_id if window_id is not None else "",
        "repeat_id": int(repeat_id),
        "split_id": split_id,
        "n_train": int(len(train_global_idx)),
        "n_test": int(len(test_global_idx)),
        "selection_mode": selection_mode,
        "targeted_policy": targeted_policy,
        "tc_higher": float(tc_higher) if np.isfinite(tc_higher) else np.nan,
    }
    metrics_rows = [
        {
            **common,
            "model": "gaussian",
            "heldout_nll": float(gaussian_nll),
            "delta_vs_gaussian": 0.0,
        },
        {
            **common,
            "model": "truncated_vine",
            "heldout_nll": float(trunc_nll) if np.isfinite(trunc_nll) else np.nan,
            "delta_vs_gaussian": float(pairwise_gain) if np.isfinite(pairwise_gain) else np.nan,
        },
        {
            **common,
            "model": "full_vine",
            "heldout_nll": float(full_nll) if np.isfinite(full_nll) else np.nan,
            "delta_vs_gaussian": float(full_gain) if np.isfinite(full_gain) else np.nan,
        },
    ]

    lookup_rows = _feature_lookup_rows(
        payload=payload,
        session_id=session_id,
        selected=selected,
        analysis_view=analysis_view,
        split_id=split_id,
        selection_mode=selection_mode,
        targeted_policy=targeted_policy,
    )
    return SliceResult(
        benchmark_rows=benchmark_rows,
        metrics_rows=metrics_rows,
        neuron_lookup_rows=lookup_rows,
        warning=warning,
    )


def _run_dose_static_view(
    trials_df: pd.DataFrame,
    neural_data: Dict[str, Any],
    d: int,
    selection_mode: str,
    exclude_targeted_neurons: bool,
    n_repeats: int,
    train_fraction: float,
    seed: int,
    min_trials_per_slice: int,
    optimize_full_structure: bool,
    families: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    benchmark_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    lookup_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {"analysis_view": "dose_static", "sessions": {}}

    targeted_policy = "include_targeted" if not exclude_targeted_neurons else "exclude_direct_targets"

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        session_df = payload["trial_table"].reset_index(drop=True).copy()
        session_df = session_df[session_df["is_valid"] & session_df["dose"].notna()].copy()
        if session_df.empty:
            continue

        dose_counts = session_df.groupby("dose").size()
        eligible_doses = sorted(float(dose) for dose, n in dose_counts.items() if int(n) >= min_trials_per_slice)
        if not eligible_doses:
            warnings.append(f"{session_id}: no dose slices met min_trials_per_slice={min_trials_per_slice}.")
            continue

        session_summary = {"eligible_doses": eligible_doses, "repeats": []}
        for repeat_id in range(int(n_repeats)):
            rng = np.random.default_rng(_session_seed(seed, session_id, extra=1009 * repeat_id))
            dose_split_map: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
            train_pool: List[int] = []

            for dose in eligible_doses:
                group_idx = session_df.index[session_df["dose"] == dose].to_numpy(dtype=int)
                train_pos, test_pos = _split_positions_random(len(group_idx), train_fraction, rng)
                train_global_idx = group_idx[train_pos]
                test_global_idx = group_idx[test_pos]
                if len(train_global_idx) < 5 or len(test_global_idx) < 3:
                    continue
                dose_split_map[float(dose)] = (train_global_idx, test_global_idx)
                train_pool.extend(train_global_idx.tolist())

            if not dose_split_map:
                warnings.append(f"{session_id}: no usable train/test splits in repeat {repeat_id}.")
                continue

            selected, selection_meta = _select_neurons_for_train_pool(
                payload=payload,
                train_indices=np.array(sorted(set(train_pool)), dtype=int),
                d=d,
                selection_mode=selection_mode,
                exclude_targeted_neurons=exclude_targeted_neurons,
                seed=_session_seed(seed, session_id, extra=repeat_id),
            )
            if selected.size != d:
                warnings.append(f"{session_id}: neuron selection failed in repeat {repeat_id}: {selection_meta.get('warning', 'unknown')}")
                continue

            session_summary["repeats"].append(
                {
                    "repeat_id": repeat_id,
                    "selected_indices": selected.astype(int).tolist(),
                    "selection_meta": selection_meta,
                    "n_slices": len(dose_split_map),
                }
            )

            for dose, (train_global_idx, test_global_idx) in sorted(dose_split_map.items()):
                group_df = session_df.loc[session_df["dose"] == dose]
                condition_label = str(group_df["condition"].iloc[0])
                split_id = f"{session_id}__dose_{int(dose):03d}__repeat_{repeat_id:02d}"
                slice_result = _evaluate_slice(
                    payload=payload,
                    session_id=session_id,
                    group_df=session_df,
                    train_global_idx=train_global_idx,
                    test_global_idx=test_global_idx,
                    selected=selected,
                    analysis_view="dose_static",
                    split_id=split_id,
                    selection_mode=selection_mode,
                    targeted_policy=targeted_policy,
                    dose=float(dose),
                    condition_label=condition_label,
                    repeat_id=seed + repeat_id,
                    window_id=None,
                    optimize_full_structure=optimize_full_structure,
                    families=families,
                )
                benchmark_rows.extend(slice_result.benchmark_rows)
                metrics_rows.extend(slice_result.metrics_rows)
                lookup_rows.extend(slice_result.neuron_lookup_rows)
                if slice_result.warning:
                    warnings.append(slice_result.warning)

        summary["sessions"][session_id] = session_summary

    return (
        pd.DataFrame(benchmark_rows),
        pd.DataFrame(metrics_rows),
        pd.DataFrame(lookup_rows).drop_duplicates(),
        warnings,
        summary,
    )


def _run_within_session_dynamic_view(
    neural_data: Dict[str, Any],
    d: int,
    selection_mode: str,
    exclude_targeted_neurons: bool,
    train_fraction: float,
    seed: int,
    min_trials: int,
    optimize_full_structure: bool,
    families: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    benchmark_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    lookup_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {"analysis_view": "within_session_dynamic", "sessions": {}}

    targeted_policy = "include_targeted" if not exclude_targeted_neurons else "exclude_direct_targets"

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        session_df = payload["trial_table"].reset_index(drop=True).copy()
        session_df = session_df[session_df["is_valid"] & session_df["dose"].notna()].copy()
        if session_df.empty:
            continue

        dose_counts = session_df.groupby("dose").size().sort_index()
        eligible = dose_counts[dose_counts >= int(min_trials)]
        if eligible.empty:
            continue

        # Use the strongest-dose condition with enough trials as the compact dynamic check.
        dose = float(np.max(eligible.index.to_numpy(dtype=float)))
        dose_df = session_df[session_df["dose"] == dose].sort_values("trial_order_within_session").copy()
        if len(dose_df) < int(min_trials):
            continue

        mid = len(dose_df) // 2
        block_defs = {
            "early": dose_df.iloc[:mid].index.to_numpy(dtype=int),
            "late": dose_df.iloc[mid:].index.to_numpy(dtype=int),
        }
        session_summary = {"dose": dose, "blocks": {}}

        for block_name, block_idx in block_defs.items():
            if len(block_idx) < max(18, 2 * d + 2):
                continue
            train_pos, test_pos = _split_positions_chronological(len(block_idx), train_fraction)
            train_global_idx = block_idx[train_pos]
            test_global_idx = block_idx[test_pos]
            if len(train_global_idx) < 5 or len(test_global_idx) < 3:
                continue

            selected, selection_meta = _select_neurons_for_train_pool(
                payload=payload,
                train_indices=train_global_idx,
                d=d,
                selection_mode=selection_mode,
                exclude_targeted_neurons=exclude_targeted_neurons,
                seed=_session_seed(seed, session_id, extra=len(block_name)),
            )
            if selected.size != d:
                warnings.append(f"{session_id} {block_name}: neuron selection failed: {selection_meta.get('warning', 'unknown')}")
                continue

            condition_label = str(dose_df["condition"].iloc[0])
            split_id = f"{session_id}__dynamic_dose_{int(dose):03d}__{block_name}"
            slice_result = _evaluate_slice(
                payload=payload,
                session_id=session_id,
                group_df=session_df,
                train_global_idx=train_global_idx,
                test_global_idx=test_global_idx,
                selected=selected,
                analysis_view="within_session_dynamic",
                split_id=split_id,
                selection_mode=selection_mode,
                targeted_policy=targeted_policy,
                dose=dose,
                condition_label=condition_label,
                repeat_id=seed,
                window_id=block_name,
                optimize_full_structure=optimize_full_structure,
                families=families,
            )
            benchmark_rows.extend(slice_result.benchmark_rows)
            metrics_rows.extend(slice_result.metrics_rows)
            lookup_rows.extend(slice_result.neuron_lookup_rows)
            if slice_result.warning:
                warnings.append(slice_result.warning)
            session_summary["blocks"][block_name] = {
                "n_trials": int(len(block_idx)),
                "n_train": int(len(train_global_idx)),
                "n_test": int(len(test_global_idx)),
                "selected_indices": selected.astype(int).tolist(),
            }

        if session_summary["blocks"]:
            summary["sessions"][session_id] = session_summary

    return (
        pd.DataFrame(benchmark_rows),
        pd.DataFrame(metrics_rows),
        pd.DataFrame(lookup_rows).drop_duplicates(),
        warnings,
        summary,
    )


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(1.96 * np.std(vals, ddof=1) / math.sqrt(vals.size))


def _plot_candidate1(metrics_df: pd.DataFrame, out_path: Path) -> None:
    static_df = metrics_df[metrics_df["analysis_view"] == "dose_static"].copy()
    model_df = static_df[static_df["model"].isin(["truncated_vine", "full_vine"])].copy()
    agg_gain = (
        model_df.groupby(["dose", "model"])["delta_vs_gaussian"]
        .apply(lambda s: pd.Series({"mean": _mean_ci(s)[0], "ci": _mean_ci(s)[1]}))
        .unstack()
    )
    tc_df = (
        static_df.groupby(["dose", "split_id"])["tc_higher"].first().reset_index()
        .groupby("dose")["tc_higher"]
        .apply(lambda s: pd.Series({"mean": _mean_ci(s)[0], "ci": _mean_ci(s)[1]}))
        .unstack()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"truncated_vine": "#1f77b4", "full_vine": "#d62728"}
    for model in ["truncated_vine", "full_vine"]:
        frame = (
            model_df[model_df["model"] == model]
            .groupby("dose")["delta_vs_gaussian"]
            .apply(lambda s: pd.Series({"mean": _mean_ci(s)[0], "ci": _mean_ci(s)[1]}))
            .unstack()
            .reset_index()
            .sort_values("dose")
        )
        axes[0].errorbar(
            frame["dose"],
            frame["mean"],
            yerr=frame["ci"],
            marker="o",
            linewidth=2,
            capsize=3,
            color=colors[model],
            label=model.replace("_", " "),
        )
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[0].set_title("Model gain vs Gaussian")
    axes[0].set_xlabel("Dose")
    axes[0].set_ylabel("Gaussian NLL - model NLL")
    axes[0].legend(frameon=False)

    tc_frame = (
        static_df.groupby(["dose", "split_id"])["tc_higher"].first().reset_index()
        .groupby("dose")["tc_higher"]
        .apply(lambda s: pd.Series({"mean": _mean_ci(s)[0], "ci": _mean_ci(s)[1]}))
        .unstack()
        .reset_index()
        .sort_values("dose")
    )
    axes[1].errorbar(
        tc_frame["dose"],
        tc_frame["mean"],
        yerr=tc_frame["ci"],
        marker="o",
        linewidth=2,
        capsize=3,
        color="#2ca02c",
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_title("Higher-order contribution")
    axes[1].set_xlabel("Dose")
    axes[1].set_ylabel("TC_higher = NLL(1-trunc) - NLL(full)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate2(metrics_df: pd.DataFrame, out_path: Path) -> None:
    static_df = metrics_df[metrics_df["analysis_view"] == "dose_static"].copy()
    gaussian = static_df[static_df["model"] == "gaussian"][["split_id", "dose", "heldout_nll"]].rename(columns={"heldout_nll": "gaussian_nll"})
    trunc = static_df[static_df["model"] == "truncated_vine"][["split_id", "dose", "heldout_nll"]].rename(columns={"heldout_nll": "trunc_nll"})
    full = static_df[static_df["model"] == "full_vine"][["split_id", "dose", "heldout_nll"]].rename(columns={"heldout_nll": "full_nll"})
    merged = gaussian.merge(trunc, on=["split_id", "dose"]).merge(full, on=["split_id", "dose"])
    merged["pairwise_gain"] = merged["gaussian_nll"] - merged["trunc_nll"]
    merged["higher_order_gain"] = merged["trunc_nll"] - merged["full_nll"]
    dose_frame = merged.groupby("dose")[["pairwise_gain", "higher_order_gain"]].mean().reset_index().sort_values("dose")

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(dose_frame["dose"], dose_frame["pairwise_gain"], width=10, color="#4c78a8", label="Gaussian -> 1-trunc")
    ax.bar(
        dose_frame["dose"],
        dose_frame["higher_order_gain"],
        bottom=dose_frame["pairwise_gain"],
        width=10,
        color="#f58518",
        label="1-trunc -> full vine",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Dose")
    ax.set_ylabel("Held-out NLL gain")
    ax.set_title("Pairwise vs higher-order decomposition")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate3(metrics_df: pd.DataFrame, out_path: Path) -> None:
    static_df = metrics_df[(metrics_df["analysis_view"] == "dose_static") & (metrics_df["model"] == "full_vine")].copy()
    tc_df = static_df.groupby(["session_id", "dose", "split_id"])["tc_higher"].first().reset_index()
    session_mean = tc_df.groupby(["session_id", "dose"])["tc_higher"].mean().reset_index()
    overall = session_mean.groupby("dose")["tc_higher"].mean().reset_index().sort_values("dose")

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for session_id, group in session_mean.groupby("session_id"):
        group = group.sort_values("dose")
        ax.plot(group["dose"], group["tc_higher"], color="#9ecae1", linewidth=1.2, alpha=0.6)
        ax.scatter(group["dose"], group["tc_higher"], color="#9ecae1", s=16, alpha=0.6)
    ax.plot(overall["dose"], overall["tc_higher"], color="#08519c", linewidth=2.8, marker="o")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Dose")
    ax.set_ylabel("Mean TC_higher by session")
    ax.set_title("Session-level robustness of higher-order gain")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_candidate4(metrics_df: pd.DataFrame, out_path: Path) -> None:
    dyn = metrics_df[(metrics_df["analysis_view"] == "within_session_dynamic") & (metrics_df["model"] == "full_vine")].copy()
    if dyn.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No dynamic slices met inclusion criteria.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    order = {"early": 0, "late": 1}
    dyn["block_order"] = dyn["window_id"].map(order)
    tc_df = dyn.groupby(["session_id", "window_id", "block_order"])["tc_higher"].first().reset_index()
    summary = tc_df.groupby("window_id")["tc_higher"].mean().reset_index()
    summary["block_order"] = summary["window_id"].map(order)
    summary = summary.sort_values("block_order")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for session_id, group in tc_df.groupby("session_id"):
        group = group.sort_values("block_order")
        ax.plot(group["block_order"], group["tc_higher"], color="#bdbdbd", linewidth=1.2, alpha=0.7)
        ax.scatter(group["block_order"], group["tc_higher"], color="#969696", s=22, alpha=0.75)
    ax.plot(summary["block_order"], summary["tc_higher"], color="#31a354", linewidth=3, marker="o")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xticks([0, 1], ["early", "late"])
    ax.set_ylabel("TC_higher")
    ax.set_title("Within-session dynamic check at strongest eligible dose")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, tuple):
            return [_convert(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return [_convert(v) for v in obj.tolist()]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_convert(payload), handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    min_trials_per_slice = _min_trials_per_slice(args.d, args.min_trials_per_slice)
    exclude_targeted_neurons = not args.include_targeted_neurons
    families = list(FAMILY_VARIANTS[args.family_variant])

    LOGGER.info("Building manifest and trial summaries from %s", args.data_root)
    manifest = dalgleish_builder.build_manifest(args.data_root)
    dalgleish_builder.print_manifest_summary(manifest)
    trials_df, neural_data, trial_inference_rows = dalgleish_builder.build_trials(args.data_root, manifest)

    if trials_df.empty:
        raise RuntimeError("No usable trials were extracted; benchmark cannot proceed.")

    trials_df, neural_data, session_subset = _restrict_to_top_sessions(
        trials_df=trials_df,
        neural_data=neural_data,
        max_sessions=int(args.max_sessions),
    )
    if args.max_sessions > 0:
        LOGGER.info("Restricted benchmark to %d sessions: %s", len(session_subset), ", ".join(session_subset))

    LOGGER.info("Running dose-stratified benchmark with d=%d repeats=%d", args.d, args.n_repeats)
    static_benchmark_df, static_metrics_df, static_lookup_df, static_warnings, static_summary = _run_dose_static_view(
        trials_df=trials_df,
        neural_data=neural_data,
        d=args.d,
        selection_mode=args.selection_mode,
        exclude_targeted_neurons=exclude_targeted_neurons,
        n_repeats=args.n_repeats,
        train_fraction=args.train_fraction,
        seed=args.seed,
        min_trials_per_slice=min_trials_per_slice,
        optimize_full_structure=bool(args.optimize_full_structure),
        families=families,
    )

    LOGGER.info("Running compact within-session dynamic benchmark")
    dynamic_benchmark_df, dynamic_metrics_df, dynamic_lookup_df, dynamic_warnings, dynamic_summary = _run_within_session_dynamic_view(
        neural_data=neural_data,
        d=args.d,
        selection_mode=args.selection_mode,
        exclude_targeted_neurons=exclude_targeted_neurons,
        train_fraction=args.train_fraction,
        seed=args.seed,
        min_trials=args.dynamic_min_trials,
        optimize_full_structure=bool(args.optimize_full_structure),
        families=families,
    )

    benchmark_df = pd.concat([static_benchmark_df, dynamic_benchmark_df], axis=0, ignore_index=True)
    metrics_df = pd.concat([static_metrics_df, dynamic_metrics_df], axis=0, ignore_index=True)
    neuron_lookup_df = pd.concat([static_lookup_df, dynamic_lookup_df], axis=0, ignore_index=True).drop_duplicates()

    if benchmark_df.empty or metrics_df.empty:
        raise RuntimeError("Benchmark produced no usable rows.")

    benchmark_df = benchmark_df.sort_values(["analysis_view", "session_id", "dose", "split_id", "split", "trial_order"]).reset_index(drop=True)
    metrics_df = metrics_df.sort_values(["analysis_view", "session_id", "dose", "window_id", "repeat_id", "model"]).reset_index(drop=True)
    neuron_lookup_df = neuron_lookup_df.sort_values(["analysis_view", "session_id", "split_id", "x_col"]).reset_index(drop=True)

    LOGGER.info("Writing benchmark tables to %s", out_root)
    benchmark_df.to_parquet(out_root / "benchmark_table.parquet", index=False)
    metrics_df.to_csv(out_root / "metrics_table.csv", index=False)
    neuron_lookup_df.to_csv(out_root / "neuron_lookup.csv", index=False)
    trials_df.to_csv(out_root / "trial_table.csv", index=False)
    pd.DataFrame(trial_inference_rows).to_csv(out_root / "trial_inference.csv", index=False)
    _write_json(out_root / "dataset_manifest.json", manifest)

    metadata = {
        "data_root": str(Path(args.data_root).resolve()),
        "out_root": str(out_root),
        "d": int(args.d),
        "selection_mode": args.selection_mode,
        "family_variant": args.family_variant,
        "families": families,
        "targeted_policy": "include_targeted" if args.include_targeted_neurons else "exclude_direct_targets",
        "session_subset": session_subset,
        "n_repeats": int(args.n_repeats),
        "train_fraction": float(args.train_fraction),
        "min_trials_per_slice": int(min_trials_per_slice),
        "dynamic_min_trials": int(args.dynamic_min_trials),
        "optimize_full_structure": bool(args.optimize_full_structure),
        "leakage_policy": {
            "selection": "session-level neuron selection uses train trials only within the current repeat or block",
            "centering": "demeaning within session x dose uses train means only",
            "winsorization": "1%/99% clipping fit on train only and applied to test",
            "pseudo_observations": "empirical CDF mappings fit on train only and applied to test by interpolation",
            "model_scoring": "repo vine models are fit with repository functions; held-out scoring uses train-derived pseudo-observations directly to avoid test-rank leakage",
        },
        "analysis_views": {
            "dose_static": static_summary,
            "within_session_dynamic": dynamic_summary,
        },
        "warnings": sorted(set(static_warnings + dynamic_warnings)),
    }
    _write_json(out_root / "benchmark_metadata.json", metadata)

    LOGGER.info("Generating candidate figures")
    _plot_candidate1(metrics_df, PROJECT_ROOT / "fig_realdata_candidate1.png")
    _plot_candidate2(metrics_df, PROJECT_ROOT / "fig_realdata_candidate2.png")
    _plot_candidate3(metrics_df, PROJECT_ROOT / "fig_realdata_candidate3.png")
    _plot_candidate4(metrics_df, PROJECT_ROOT / "fig_realdata_candidate4.png")

    LOGGER.info("Benchmark rows: %d | metrics rows: %d | neuron lookup rows: %d", len(benchmark_df), len(metrics_df), len(neuron_lookup_df))
    LOGGER.info("Candidate figures written to project root.")


if __name__ == "__main__":
    main()
