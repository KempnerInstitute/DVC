#!/usr/bin/env python3
"""Maintained helper functions for the Dalgleish stimulation benchmark pipeline."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

import scripts.stimulation_exp_benchmark.build_dalgleish_dvc_dataset as builder
from dvc_package.core.param_copula import copulaccdf, copulapdf
from dvc_package.experiments.simulation_benchmarks import _gaussian_copula_nll_given_corr

FAMILY_VARIANTS: Dict[str, List[str]] = {
    "default": ["ind", "gaussian", "student", "clayton", "frank", "gumbel", "joe"],
    "stable": ["ind", "gaussian", "student", "clayton"],
}

BASELINE_WINDOW = (-1.0, -0.1)
DELAYED_WINDOW = (0.2, 0.7)
POST_WINDOW = (0.7, 1.4)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


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


def _session_seed(base_seed: int, session_id: str, extra: int = 0) -> int:
    session_term = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(session_id)))
    return int(base_seed + session_term + extra)


def _fit_train_only_ecdf(train_x: np.ndarray) -> List[Dict[str, np.ndarray]]:
    return builder._fit_empirical_cdf(train_x)


def _apply_train_only_ecdf(x: np.ndarray, mappings: List[Dict[str, np.ndarray]]) -> np.ndarray:
    return builder._apply_empirical_cdf(x, mappings)


def _winsorize_train_apply(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_clip, test_clip, _bounds = builder._winsorize_train_apply_test(train_x, test_x)
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
