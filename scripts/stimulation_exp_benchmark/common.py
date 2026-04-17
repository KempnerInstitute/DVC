from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from dvc_package.experiments.simulation_benchmarks import _mean_copula_nll

from . import build_dalgleish_dvc_dataset as builder


FAMILY_VARIANTS: Dict[str, List[str]] = {
    "stable": ["ind", "gaussian", "student", "clayton"],
    "extended": ["ind", "gaussian", "student", "clayton", "frank", "gumbel", "joe"],
}


def configure_logging(verbose: bool = False) -> None:
    builder.configure_logging(verbose=verbose)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    builder.write_json(path, payload)


def _winsorize_train_apply(
    train_x: np.ndarray,
    test_x: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    train_clip, test_clip, _ = builder._winsorize_train_apply_test(
        np.asarray(train_x, dtype=np.float64),
        np.asarray(test_x, dtype=np.float64),
        lower_q=lower_q,
        upper_q=upper_q,
    )
    return train_clip, test_clip


def _fit_train_only_ecdf(train_x: np.ndarray) -> List[Dict[str, np.ndarray]]:
    return builder._fit_empirical_cdf(np.asarray(train_x, dtype=np.float64))


def _apply_train_only_ecdf(x: np.ndarray, mappings: List[Dict[str, np.ndarray]]) -> np.ndarray:
    return builder._apply_empirical_cdf(np.asarray(x, dtype=np.float64), mappings)


def _split_positions_random(
    n_rows: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_rows < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.arange(int(n_rows), dtype=int)
    rng.shuffle(order)
    n_train = int(np.floor(float(train_fraction) * float(n_rows)))
    n_train = int(np.clip(n_train, 1, n_rows - 1))
    train_pos = np.sort(order[:n_train].astype(int))
    test_pos = np.sort(order[n_train:].astype(int))
    return train_pos, test_pos


def _score_gaussian_from_pobs(u_train: np.ndarray, u_test: np.ndarray, ridge: float = 1e-4) -> float:
    z_train = norm.ppf(np.clip(np.asarray(u_train, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    z_test = norm.ppf(np.clip(np.asarray(u_test, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    if z_train.shape[0] < 5 or z_test.shape[0] < 1:
        return float("nan")

    corr = np.corrcoef(z_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    corr += float(ridge) * np.eye(corr.shape[0], dtype=np.float64)
    dstd = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    corr = corr / np.outer(dstd, dstd)
    np.fill_diagonal(corr, 1.0)

    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0 or not np.isfinite(logdet):
        return float("nan")
    inv_corr = np.linalg.inv(corr)
    quad = np.einsum("ni,ij,nj->n", z_test, inv_corr - np.eye(corr.shape[0]), z_test)
    log_c = -0.5 * logdet - 0.5 * quad
    return float(-np.mean(log_c))


def _score_vine_on_uniforms(vine: Any, u_test: np.ndarray) -> float:
    return float(_mean_copula_nll(vine, np.asarray(u_test, dtype=np.float32)))


def _targeted_mask_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    n_neurons = int(payload.get("n_neurons", 0))
    mask = np.zeros(n_neurons, dtype=bool)
    targeted = payload.get("targeted_roi_by_program", {}) or {}
    for roi_indices in targeted.values():
        arr = np.asarray(list(roi_indices), dtype=int).reshape(-1)
        arr = arr[(arr >= 0) & (arr < n_neurons)]
        mask[arr] = True
    if not mask.any():
        catalog = payload.get("target_catalog")
        if isinstance(catalog, pd.DataFrame) and "targeted_roi_indices" in catalog.columns:
            for values in catalog["targeted_roi_indices"].dropna().tolist():
                arr = np.asarray(list(values), dtype=int).reshape(-1)
                arr = arr[(arr >= 0) & (arr < n_neurons)]
                mask[arr] = True
    return mask


def _prepare_session_cache(session_id: str, payload: Dict[str, Any], data_root: Path | str) -> Dict[str, Any]:
    arrays = payload.get("arrays", {})
    delayed = np.asarray(arrays.get("delayed", arrays.get("stim")), dtype=np.float64)
    post = np.asarray(arrays.get("post"), dtype=np.float64)
    if delayed.ndim != 2 or post.ndim != 2:
        raise ValueError(f"Session {session_id} is missing 2D delayed/post arrays")
    if delayed.shape != post.shape:
        raise ValueError(f"Session {session_id} delayed/post shapes do not match: {delayed.shape} vs {post.shape}")

    session_df = payload["trial_table"].copy().reset_index(drop=True)
    if "trial_order_within_session" not in session_df.columns:
        session_df["trial_order_within_session"] = np.arange(len(session_df), dtype=int)
    if "t" not in session_df.columns:
        session_df["t"] = session_df["trial_order_within_session"]

    return {
        "session_id": str(session_id),
        "data_root": str(Path(data_root)),
        "session_df": session_df,
        "delayed": delayed,
        "post": post,
        "targeted_mask": _targeted_mask_from_payload(payload),
        "warnings": list(payload.get("warnings", [])),
        "frame_rate_hz": payload.get("frame_rate_hz"),
        "roi_lookup": payload.get("roi_lookup"),
    }


def _build_split_plan(
    *,
    session_df: pd.DataFrame,
    feature_dim: int,
    seed: int,
    session_id: str,
    n_repeats: int,
    train_fraction: float,
    min_trials_floor: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    session_df = session_df.reset_index(drop=True).copy()
    min_train = max(int(feature_dim) + 1, 5)
    min_test = 3
    min_trials = max(int(min_trials_floor), min_train + min_test)

    for dose, dose_df in session_df.groupby("dose", sort=True):
        if not np.isfinite(float(dose)):
            continue
        idx = dose_df.index.to_numpy(dtype=int)
        if idx.size < min_trials:
            continue
        for repeat_id in range(int(n_repeats)):
            repeat_seed = int(seed) + 7919 * repeat_id + 101 * len(str(session_id)) + int(round(float(dose) * 10.0))
            rng = np.random.default_rng(repeat_seed)
            train_pos, test_pos = _split_positions_random(idx.size, float(train_fraction), rng)
            if len(train_pos) < min_train or len(test_pos) < min_test:
                continue
            train_idx = idx[train_pos]
            test_idx = idx[test_pos]
            out.append(
                {
                    "session_id": str(session_id),
                    "dose": float(dose),
                    "repeat_id": int(repeat_id),
                    "slice_key": f"{session_id}__dose_{float(dose):g}__repeat_{repeat_id}",
                    "train_idx": train_idx.astype(int),
                    "test_idx": test_idx.astype(int),
                }
            )
    return out


@contextlib.contextmanager
def _temporary_log_level(logger_names: Sequence[str], level: int):
    saved: List[Tuple[logging.Logger, int]] = []
    for name in logger_names:
        log = logging.getLogger(name)
        saved.append((log, log.level))
        log.setLevel(level)
    try:
        yield
    finally:
        for log, old_level in saved:
            log.setLevel(old_level)


def _with_quieter_repo_logging(func: Any, *args: Any, **kwargs: Any) -> Any:
    with _temporary_log_level(
        ["DVC.vine", "dvc_package", "dalgleish_dvc_dataset", "matplotlib"],
        logging.WARNING,
    ):
        return func(*args, **kwargs)
