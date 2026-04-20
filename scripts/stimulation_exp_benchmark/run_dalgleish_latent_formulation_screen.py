#!/usr/bin/env python3
"""Compact latent formulation screen for the Dalgleish dataset.

This reconstructs the latent representation screen described in the
project notes so the claimed C1/C2/C3 results can be rerun on the
current branch from committed code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    _winsorize_train_apply,
    _with_quieter_repo_logging,
    _write_json,
    configure_logging,
)
from dvc_package.experiments.simulation_benchmarks import (
    _estimate_hub_by_correlation,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
)
from scipy.stats import norm


LOGGER = logging.getLogger("dalgleish_latent_formulation")

LATENT_VARIANTS: List[Dict[str, Any]] = [
    {
        "variant": "C1_latent_post_pca4",
        "family": "C",
        "source_space": "non_targeted",
        "representation": "post_only",
        "n_components": 4,
    },
    {
        "variant": "C2_latent_2bin_pca4",
        "family": "C",
        "source_space": "non_targeted",
        "representation": "delayed_post",
        "n_components": 4,
    },
    {
        "variant": "C3_latent_2bin_pca6",
        "family": "C",
        "source_space": "non_targeted",
        "representation": "delayed_post",
        "n_components": 6,
    },
    {
        "variant": "targeted_2bin_pca4",
        "family": "source_space_screen",
        "source_space": "targeted",
        "representation": "delayed_post",
        "n_components": 4,
    },
    {
        "variant": "mixed_2bin_pca4",
        "family": "source_space_screen",
        "source_space": "mixed",
        "representation": "delayed_post",
        "n_components": 4,
    },
    {
        "variant": "non_targeted_2bin_pca4",
        "family": "source_space_screen",
        "source_space": "non_targeted",
        "representation": "delayed_post",
        "n_components": 4,
    },
    {
        "variant": "non_targeted_2bin_pca6",
        "family": "source_space_screen",
        "source_space": "non_targeted",
        "representation": "delayed_post",
        "n_components": 6,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dalgleish latent formulation screen.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--results_root", default="results/stimulation_exp_benchmark_formulation")
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
    parser.add_argument("--bootstrap_draws", type=int, default=4000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _source_indices(targeted_mask: np.ndarray, source_space: str) -> np.ndarray:
    if source_space == "non_targeted":
        return np.flatnonzero(~targeted_mask).astype(int)
    if source_space == "targeted":
        return np.flatnonzero(targeted_mask).astype(int)
    if source_space == "mixed":
        return np.arange(targeted_mask.size, dtype=int)
    raise ValueError(f"Unknown source_space={source_space}")


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
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "gaussian_copula_nll": np.nan,
        "truncated_vine_nll": np.nan,
        "full_vine_nll": np.nan,
        "gaussian_status": "not_run",
        "truncated_vine_status": "not_run",
        "full_vine_status": "not_run",
    }
    try:
        u_train, u_test = _fit_transform_uniforms(train_x, test_x)
        out["gaussian_copula_nll"] = float(_score_gaussian_from_pobs(u_train, u_test))
        out["gaussian_status"] = "success"
        hub = int(_estimate_hub_by_correlation(norm.ppf(np.clip(u_train, 1e-6, 1.0 - 1e-6))))
        order = [hub] + [idx for idx in range(u_train.shape[1]) if idx != hub]
        try:
            trunc_vine = _with_quieter_repo_logging(
                _fit_truncated_cvine_level0,
                x_train=u_train.astype(np.float32),
                families=list(families),
                order=order,
            )
            out["truncated_vine_nll"] = float(_score_vine_on_uniforms(trunc_vine, u_test))
            out["truncated_vine_status"] = "success"
        except Exception as exc:
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
        except Exception as exc:
            out["full_vine_status"] = f"failed:{exc}"
    except Exception as exc:
        out["gaussian_status"] = f"failed:{exc}"
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


def _build_matrix(
    cache: Dict[str, Any],
    source_space: str,
    representation: str,
) -> Tuple[np.ndarray, np.ndarray]:
    targeted_mask = np.asarray(cache["targeted_mask"], dtype=bool)
    source_idx = _source_indices(targeted_mask, source_space)
    delayed = np.asarray(cache["delayed"][:, source_idx], dtype=np.float64)
    post = np.asarray(cache["post"][:, source_idx], dtype=np.float64)
    if representation == "post_only":
        return post, source_idx
    if representation == "delayed_post":
        return np.concatenate([delayed, post], axis=1).astype(np.float64), source_idx
    raise ValueError(f"Unknown representation={representation}")


def _aggregate_summary(raw_df: pd.DataFrame, bootstrap_draws: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant, group in raw_df.groupby("variant", sort=False):
        vals_fg = group["full_vs_gaussian"].to_numpy(dtype=float)
        vals_tc = group["tc_higher"].to_numpy(dtype=float)
        mean_fg, ci_low_fg, ci_high_fg = _mean_bootstrap_ci(vals_fg, seed=seed + len(variant), draws=bootstrap_draws)
        mean_tc, ci_low_tc, ci_high_tc = _mean_bootstrap_ci(vals_tc, seed=seed + 1000 + len(variant), draws=bootstrap_draws)
        rows.append(
            {
                "variant": variant,
                "family": str(group["family"].iloc[0]),
                "source_space": str(group["source_space"].iloc[0]),
                "representation": str(group["representation"].iloc[0]),
                "n_components": int(group["n_components"].iloc[0]),
                "mean_full_vs_gaussian": mean_fg,
                "ci_low_full_vs_gaussian": ci_low_fg,
                "ci_high_full_vs_gaussian": ci_high_fg,
                "prop_full_lt_gaussian": float(np.mean(vals_fg > 0.0)),
                "mean_tc_higher": mean_tc,
                "ci_low_tc_higher": ci_low_tc,
                "ci_high_tc_higher": ci_high_tc,
                "prop_full_lt_1trunc": float(np.mean(vals_tc > 0.0)),
                "mean_gaussian_to_trunc": float(np.nanmean(group["gaussian_to_trunc"].to_numpy(dtype=float))),
                "retained_pca_variance": float(np.nanmean(group["pca_variance_retained"].to_numpy(dtype=float))),
                "n_slices": int(group["slice_key"].nunique()),
                "n_sessions": int(group["session_id"].nunique()),
                "n_attempted_slices": int(group["attempted_slice_count"].iloc[0]) if "attempted_slice_count" in group.columns else int(group["slice_key"].nunique()),
            }
        )
    return pd.DataFrame(rows)


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
    results_root = Path(args.results_root).resolve()
    data_dir = results_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    families = list(FAMILY_VARIANTS[args.family_variant])

    manifest = builder.build_manifest(data_root)
    _trials_df, neural_data, _ = builder.build_trials(data_root, manifest)
    session_cache: Dict[str, Dict[str, Any]] = {}
    for session_id, payload in neural_data.items():
        if payload.get("used", False):
            session_cache[str(session_id)] = _prepare_session_cache(str(session_id), payload, data_root)

    raw_rows: List[Dict[str, Any]] = []
    attempt_counts: Dict[str, int] = {}
    for cfg in LATENT_VARIANTS:
        variant = str(cfg["variant"])
        source_space = str(cfg["source_space"])
        representation = str(cfg["representation"])
        n_components = int(cfg["n_components"])
        family = str(cfg["family"])
        LOGGER.info("Running %s", variant)
        for session_id, cache in session_cache.items():
            try:
                source_matrix, source_idx = _build_matrix(cache, source_space, representation)
            except Exception as exc:
                LOGGER.debug("Skipping %s %s while building matrix: %s", variant, session_id, exc)
                continue
            if int(source_idx.size) < int(n_components):
                continue
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
                attempt_counts[variant] = attempt_counts.get(variant, 0) + 1
                train_idx = np.asarray(split["train_idx"], dtype=int)
                test_idx = np.asarray(split["test_idx"], dtype=int)
                try:
                    train_scores, test_scores, explained, _components = _fit_pca_with_components(
                        train_x=source_matrix[train_idx],
                        test_x=source_matrix[test_idx],
                        n_components=n_components,
                    )
                except Exception as exc:
                    LOGGER.debug("PCA failed for %s %s: %s", variant, split["slice_key"], exc)
                    continue
                scored = _fit_publication_models(
                    train_scores,
                    test_scores,
                    families=families,
                    seed=int(args.seed) + int(split["repeat_id"]) + n_components,
                )
                if not (
                    np.isfinite(scored["gaussian_copula_nll"])
                    and np.isfinite(scored["truncated_vine_nll"])
                    and np.isfinite(scored["full_vine_nll"])
                ):
                    continue
                raw_rows.append(
                    {
                        "variant": variant,
                        "family": family,
                        "source_space": source_space,
                        "representation": representation,
                        "n_components": n_components,
                        "session_id": session_id,
                        "dose": float(split["dose"]),
                        "repeat_id": int(split["repeat_id"]),
                        "slice_key": str(split["slice_key"]),
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "gaussian_copula_nll": float(scored["gaussian_copula_nll"]),
                        "truncated_vine_nll": float(scored["truncated_vine_nll"]),
                        "full_vine_nll": float(scored["full_vine_nll"]),
                        "full_vs_gaussian": float(scored["gaussian_copula_nll"] - scored["full_vine_nll"]),
                        "gaussian_to_trunc": float(scored["gaussian_copula_nll"] - scored["truncated_vine_nll"]),
                        "tc_higher": float(scored["truncated_vine_nll"] - scored["full_vine_nll"]),
                        "pca_variance_retained": float(np.sum(explained)),
                        "attempted_slice_count": np.nan,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)
    if not raw_df.empty:
        raw_df["attempted_slice_count"] = raw_df["variant"].map(attempt_counts).astype(float)
    summary_df = _aggregate_summary(raw_df, bootstrap_draws=int(args.bootstrap_draws), seed=int(args.seed))

    raw_df.to_csv(data_dir / "latent_formulation_raw_slices.csv", index=False)
    summary_df.to_csv(data_dir / "latent_formulation_summary.csv", index=False)

    metadata = {
        "family_variant": args.family_variant,
        "window_backbone": builder.get_window_backbone_name(),
        "window_backbone_metadata": builder.get_window_backbone_metadata(),
        "seed": int(args.seed),
        "n_repeats": int(args.n_repeats),
        "train_fraction": float(args.train_fraction),
        "min_trials_floor": int(args.min_trials_floor),
        "n_variants": int(len(LATENT_VARIANTS)),
        "n_sessions": int(raw_df["session_id"].nunique()) if not raw_df.empty else 0,
        "n_slices": int(raw_df["slice_key"].nunique()) if not raw_df.empty else 0,
    }
    _write_json(data_dir / "latent_formulation_metadata.json", metadata)
    LOGGER.info("Latent formulation screen complete: %d slice rows across %d variants", len(raw_df), len(LATENT_VARIANTS))


if __name__ == "__main__":
    main()
