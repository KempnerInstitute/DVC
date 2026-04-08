#!/usr/bin/env python3
"""Full stable rerun and paper-decision summary for the Dalgleish real-data benchmark."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.stimulation_exp_benchmark.build_dalgleish_dvc_dataset as builder
from scripts.debug_stimulation_exp.run_dalgleish_real_data_benchmark import (
    DEFAULT_FAMILY_VARIANT,
    FAMILY_VARIANTS,
    _apply_train_only_ecdf,
    _feature_lookup_rows,
    _fit_train_only_ecdf,
    _restrict_to_top_sessions,
    _score_gaussian_from_pobs,
    _score_vine_on_uniforms,
    _select_neurons_for_train_pool,
    _session_seed,
    _split_positions_random,
    _with_quieter_repo_logging,
    _winsorize_train_apply,
    _write_json,
    configure_logging,
)
from dvc_package.experiments.simulation_benchmarks import (
    _estimate_hub_by_correlation,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
)
from scipy.stats import norm


LOGGER_NAME = "dalgleish_real_data_decision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full stable Dalgleish real-data decision rerun.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--selection_mode", choices=["responsive_random", "topk_responsive"], default="topk_responsive")
    parser.add_argument("--family_variant", choices=sorted(FAMILY_VARIANTS.keys()), default=DEFAULT_FAMILY_VARIANT)
    parser.add_argument("--min_trials_per_slice", type=int, default=18)
    parser.add_argument("--dynamic_min_block_trials", type=int, default=12)
    parser.add_argument("--dynamic_max_doses_per_session", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _variant_configs() -> List[Dict[str, Any]]:
    return [
        {
            "variant": "variant_A_stim_mean_d4",
            "feature_mode": "stim_mean",
            "d": 4,
            "residualize": True,
            "include_targeted": False,
        },
        {
            "variant": "variant_B_stim_mean_d6",
            "feature_mode": "stim_mean",
            "d": 6,
            "residualize": True,
            "include_targeted": False,
        },
        {
            "variant": "variant_C_diff_d4",
            "feature_mode": "stim_minus_baseline",
            "d": 4,
            "residualize": True,
            "include_targeted": False,
        },
    ]


def _feature_matrix(payload: Dict[str, Any], selected: np.ndarray, feature_mode: str) -> np.ndarray:
    arrays = payload["arrays"]
    if feature_mode == "stim_mean":
        return np.asarray(arrays["stim"][:, selected], dtype=np.float64)
    if feature_mode == "stim_minus_baseline":
        return np.asarray(arrays["diff"][:, selected], dtype=np.float64)
    raise ValueError(f"Unknown feature_mode={feature_mode}")


def _format_rows(
    slice_df: pd.DataFrame,
    u_slice: np.ndarray,
    variant: str,
    analysis_view: str,
    split_role: str,
    split_id: str,
    selection_mode: str,
    targeted_policy: str,
    block_id: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row_idx, (_, meta_row) in enumerate(slice_df.iterrows()):
        out: Dict[str, Any] = {
            "variant": variant,
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
            "block_id": block_id,
        }
        for col_idx in range(u_slice.shape[1]):
            out[f"x{col_idx + 1}"] = float(u_slice[row_idx, col_idx])
        rows.append(out)
    return rows


def _score_slice(
    payload: Dict[str, Any],
    session_df: pd.DataFrame,
    selected: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_mode: str,
    residualize: bool,
    families: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    raw = _feature_matrix(payload, selected, feature_mode)
    train_x = raw[train_idx]
    test_x = raw[test_idx]

    if residualize:
        train_mean = np.nanmean(train_x, axis=0)
        train_x = train_x - train_mean
        test_x = test_x - train_mean

    train_x, test_x = _winsorize_train_apply(train_x, test_x)
    mappings = _fit_train_only_ecdf(train_x)
    u_train = _apply_train_only_ecdf(train_x, mappings)
    u_test = _apply_train_only_ecdf(test_x, mappings)

    hub = int(_estimate_hub_by_correlation(norm.ppf(np.clip(u_train, 1e-6, 1.0 - 1e-6))))
    order = [hub] + [idx for idx in range(selected.size) if idx != hub]

    gaussian_nll = _score_gaussian_from_pobs(u_train, u_test)
    trunc_vine = _with_quieter_repo_logging(
        _fit_truncated_cvine_level0,
        x_train=u_train.astype(np.float32),
        families=list(families),
        order=order,
    )
    trunc_nll = _score_vine_on_uniforms(trunc_vine, u_test)

    full_vine = _with_quieter_repo_logging(
        _fit_parametric_vine,
        x_train=u_train.astype(np.float32),
        families=list(families),
        optimize_structure=False,
        seed=seed,
    )
    full_nll = _score_vine_on_uniforms(full_vine, u_test)
    return u_train, u_test, float(gaussian_nll), float(trunc_nll), float(full_nll)


def _status_label(
    full_delta_mean: float,
    full_beats_gauss: float,
    full_beats_trunc: float,
    tc_mean: float,
    trunc_delta_mean: float,
) -> str:
    if np.isfinite(full_delta_mean) and full_delta_mean > 0.05 and full_beats_gauss >= 0.55 and (tc_mean > 0.0 or full_beats_trunc >= 0.45):
        return "promising_main_figure"
    if np.isfinite(full_delta_mean) and full_delta_mean > -0.05 and (full_beats_gauss >= 0.35 or full_beats_trunc >= 0.35):
        return "usable_but_heterogeneous"
    if np.isfinite(trunc_delta_mean) and trunc_delta_mean > 0.05 and (not np.isfinite(full_delta_mean) or full_delta_mean <= 0.0):
        return "pairwise_dominant"
    return "not_useful"


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(1.96 * np.std(vals, ddof=1) / math.sqrt(vals.size))


def _run_static_variant(
    trials_df: pd.DataFrame,
    neural_data: Dict[str, Any],
    variant_cfg: Dict[str, Any],
    families: Sequence[str],
    seed: int,
    n_repeats: int,
    train_fraction: float,
    min_trials_per_slice: int,
    selection_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    variant = str(variant_cfg["variant"])
    d = int(variant_cfg["d"])
    feature_mode = str(variant_cfg["feature_mode"])
    residualize = bool(variant_cfg["residualize"])
    include_targeted = bool(variant_cfg["include_targeted"])
    exclude_targeted = not include_targeted
    targeted_policy = "include_targeted" if include_targeted else "exclude_direct_targets"

    benchmark_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    lookup_rows: List[Dict[str, Any]] = []
    session_summary: Dict[str, Any] = {}

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        session_df = payload["trial_table"].reset_index(drop=True)
        session_df = session_df[session_df["is_valid"] & session_df["dose"].notna()].copy()
        if session_df.empty:
            continue
        dose_counts = session_df.groupby("dose").size()
        eligible_doses = sorted(float(dose) for dose, n in dose_counts.items() if int(n) >= int(min_trials_per_slice))
        if not eligible_doses:
            continue

        repeats_meta: List[Dict[str, Any]] = []
        for repeat_id in range(int(n_repeats)):
            rng = np.random.default_rng(_session_seed(seed, session_id, extra=9973 * repeat_id + d))
            dose_split_map: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
            train_pool: List[int] = []
            for dose in eligible_doses:
                group_idx = session_df.index[session_df["dose"] == dose].to_numpy(dtype=int)
                train_pos, test_pos = _split_positions_random(len(group_idx), train_fraction, rng)
                train_idx = group_idx[train_pos]
                test_idx = group_idx[test_pos]
                if len(train_idx) < 5 or len(test_idx) < 3:
                    continue
                dose_split_map[dose] = (train_idx, test_idx)
                train_pool.extend(train_idx.tolist())

            if not dose_split_map:
                continue

            selected, selection_meta = _select_neurons_for_train_pool(
                payload=payload,
                train_indices=np.array(sorted(set(train_pool)), dtype=int),
                d=d,
                selection_mode=selection_mode,
                exclude_targeted_neurons=exclude_targeted,
                seed=_session_seed(seed, session_id, extra=repeat_id + 17 * d),
            )
            if selected.size != d:
                continue

            repeats_meta.append(
                {
                    "repeat_id": repeat_id,
                    "selected_indices": selected.astype(int).tolist(),
                    "selection_meta": selection_meta,
                    "n_slices": len(dose_split_map),
                }
            )

            for dose, (train_idx, test_idx) in sorted(dose_split_map.items()):
                slice_seed = _session_seed(seed, session_id, extra=repeat_id + int(100 * dose) + d)
                try:
                    u_train, u_test, gaussian_nll, trunc_nll, full_nll = _score_slice(
                        payload=payload,
                        session_df=session_df,
                        selected=selected,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        feature_mode=feature_mode,
                        residualize=residualize,
                        families=families,
                        seed=slice_seed,
                    )
                except Exception as exc:
                    session_summary.setdefault(session_id, {}).setdefault("warnings", []).append(
                        f"{variant} repeat={repeat_id} dose={dose}: {exc}"
                    )
                    continue

                split_id = f"{variant}__{session_id}__dose_{int(dose):03d}__repeat_{repeat_id:02d}"
                group_train_df = session_df.loc[train_idx].copy()
                group_test_df = session_df.loc[test_idx].copy()
                benchmark_rows.extend(
                    _format_rows(
                        group_train_df,
                        u_train,
                        variant=variant,
                        analysis_view="dose_static",
                        split_role="train",
                        split_id=split_id,
                        selection_mode=selection_mode,
                        targeted_policy=targeted_policy,
                        block_id="",
                    )
                )
                benchmark_rows.extend(
                    _format_rows(
                        group_test_df,
                        u_test,
                        variant=variant,
                        analysis_view="dose_static",
                        split_role="test",
                        split_id=split_id,
                        selection_mode=selection_mode,
                        targeted_policy=targeted_policy,
                        block_id="",
                    )
                )
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
                condition_label = str(session_df.loc[session_df["dose"] == dose, "condition"].iloc[0])
                tc_higher = trunc_nll - full_nll
                common = {
                    "variant": variant,
                    "analysis_view": "dose_static",
                    "session_id": session_id,
                    "dose": float(dose),
                    "condition_label": condition_label,
                    "block_id": "",
                    "repeat_id": repeat_id,
                    "split_id": split_id,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "selection_mode": selection_mode,
                    "targeted_policy": targeted_policy,
                    "feature_mode": feature_mode,
                    "d": d,
                    "residualize": residualize,
                    "tc_higher": float(tc_higher),
                }
                metrics_rows.extend(
                    [
                        {**common, "model": "gaussian", "heldout_nll": gaussian_nll, "delta_vs_gaussian": 0.0},
                        {**common, "model": "truncated_vine", "heldout_nll": trunc_nll, "delta_vs_gaussian": gaussian_nll - trunc_nll},
                        {**common, "model": "full_vine", "heldout_nll": full_nll, "delta_vs_gaussian": gaussian_nll - full_nll},
                    ]
                )

        if repeats_meta:
            session_summary.setdefault(session_id, {})["eligible_doses"] = eligible_doses
            session_summary.setdefault(session_id, {})["repeats"] = repeats_meta

    return (
        pd.DataFrame(benchmark_rows),
        pd.DataFrame(metrics_rows),
        pd.DataFrame(lookup_rows).drop_duplicates(),
        session_summary,
    )


def _summarize_variants(metrics_df: pd.DataFrame) -> pd.DataFrame:
    static_df = metrics_df[metrics_df["analysis_view"] == "dose_static"].copy()
    if static_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for variant, group in static_df.groupby("variant"):
        gaussian = group[group["model"] == "gaussian"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "gaussian_nll"})
        trunc = group[group["model"] == "truncated_vine"][["split_id", "heldout_nll"]].rename(columns={"heldout_nll": "trunc_nll"})
        full = group[group["model"] == "full_vine"][["split_id", "heldout_nll", "tc_higher", "session_id", "dose"]].rename(columns={"heldout_nll": "full_nll"})
        merged = gaussian.merge(trunc, on="split_id").merge(full, on="split_id")
        full_delta = merged["gaussian_nll"] - merged["full_nll"]
        trunc_delta = merged["gaussian_nll"] - merged["trunc_nll"]
        full_beats_gauss = float(np.mean(merged["full_nll"] < merged["gaussian_nll"])) if len(merged) else np.nan
        full_beats_trunc = float(np.mean(merged["full_nll"] < merged["trunc_nll"])) if len(merged) else np.nan
        tc_mean = float(np.mean(merged["tc_higher"])) if len(merged) else np.nan
        tc_median = float(np.median(merged["tc_higher"])) if len(merged) else np.nan
        full_delta_mean = float(np.mean(full_delta)) if len(merged) else np.nan
        trunc_delta_mean = float(np.mean(trunc_delta)) if len(merged) else np.nan
        rows.append(
            {
                "variant": variant,
                "gaussian_mean_nll": float(group[group["model"] == "gaussian"]["heldout_nll"].mean()),
                "truncated_mean_nll": float(group[group["model"] == "truncated_vine"]["heldout_nll"].mean()),
                "full_mean_nll": float(group[group["model"] == "full_vine"]["heldout_nll"].mean()),
                "truncated_delta_vs_gaussian_mean": trunc_delta_mean,
                "full_delta_vs_gaussian_mean": full_delta_mean,
                "full_beats_gaussian_prop": full_beats_gauss,
                "full_beats_trunc_prop": full_beats_trunc,
                "tc_higher_mean": tc_mean,
                "tc_higher_median": tc_median,
                "usable_sessions": int(group["session_id"].nunique()),
                "usable_slices": int(merged["split_id"].nunique()),
                "status_label": _status_label(full_delta_mean, full_beats_gauss, full_beats_trunc, tc_mean, trunc_delta_mean),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(
        ["full_delta_vs_gaussian_mean", "full_beats_gaussian_prop", "tc_higher_mean"],
        ascending=[False, False, False],
    )
    return summary_df.reset_index(drop=True)


def _choose_best_variant(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        raise RuntimeError("No static variant results were produced.")
    return str(summary_df.iloc[0]["variant"])


def _run_dynamic_best_variant(
    neural_data: Dict[str, Any],
    variant_cfg: Dict[str, Any],
    families: Sequence[str],
    seed: int,
    train_fraction: float,
    dynamic_min_block_trials: int,
    max_doses_per_session: int,
    selection_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    variant = str(variant_cfg["variant"])
    d = int(variant_cfg["d"])
    feature_mode = str(variant_cfg["feature_mode"])
    residualize = bool(variant_cfg["residualize"])
    include_targeted = bool(variant_cfg["include_targeted"])
    exclude_targeted = not include_targeted
    targeted_policy = "include_targeted" if include_targeted else "exclude_direct_targets"
    min_total_trials = 3 * int(dynamic_min_block_trials)

    benchmark_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    lookup_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"variant": variant, "sessions": {}}

    for session_id, payload in neural_data.items():
        if not payload.get("used", False):
            continue
        session_df = payload["trial_table"].reset_index(drop=True)
        session_df = session_df[session_df["is_valid"] & session_df["dose"].notna()].copy()
        if session_df.empty:
            continue
        dose_counts = session_df.groupby("dose").size().sort_index()
        eligible_doses = [float(dose) for dose, n in dose_counts.items() if int(n) >= min_total_trials]
        if not eligible_doses:
            continue
        chosen_doses = sorted(eligible_doses, reverse=True)[: int(max_doses_per_session)]
        session_summary: Dict[str, Any] = {}

        for dose in chosen_doses:
            dose_df = session_df[session_df["dose"] == dose].sort_values("trial_order_within_session").copy()
            block_frames = np.array_split(dose_df.index.to_numpy(dtype=int), 3)
            block_defs = {"early": block_frames[0], "middle": block_frames[1], "late": block_frames[2]}
            if min(len(block) for block in block_defs.values()) < int(dynamic_min_block_trials):
                continue

            for block_id, block_idx in block_defs.items():
                rng = np.random.default_rng(_session_seed(seed, session_id, extra=int(100 * dose) + len(block_id) + d))
                train_pos, test_pos = _split_positions_random(len(block_idx), train_fraction, rng)
                train_idx = np.asarray(block_idx[train_pos], dtype=int)
                test_idx = np.asarray(block_idx[test_pos], dtype=int)
                if len(train_idx) < 5 or len(test_idx) < 3:
                    continue
                selected, selection_meta = _select_neurons_for_train_pool(
                    payload=payload,
                    train_indices=train_idx,
                    d=d,
                    selection_mode=selection_mode,
                    exclude_targeted_neurons=exclude_targeted,
                    seed=_session_seed(seed, session_id, extra=len(block_id) + int(dose)),
                )
                if selected.size != d:
                    continue
                slice_seed = _session_seed(seed, session_id, extra=int(100 * dose) + len(block_id) + 31 * d)
                try:
                    u_train, u_test, gaussian_nll, trunc_nll, full_nll = _score_slice(
                        payload=payload,
                        session_df=session_df,
                        selected=selected,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        feature_mode=feature_mode,
                        residualize=residualize,
                        families=families,
                        seed=slice_seed,
                    )
                except Exception as exc:
                    session_summary.setdefault(f"dose_{int(dose):03d}", {}).setdefault("warnings", []).append(str(exc))
                    continue

                split_id = f"{variant}__{session_id}__dynamic_dose_{int(dose):03d}__{block_id}"
                benchmark_rows.extend(
                    _format_rows(
                        session_df.loc[train_idx].copy(),
                        u_train,
                        variant=variant,
                        analysis_view="within_session_dynamic",
                        split_role="train",
                        split_id=split_id,
                        selection_mode=selection_mode,
                        targeted_policy=targeted_policy,
                        block_id=block_id,
                    )
                )
                benchmark_rows.extend(
                    _format_rows(
                        session_df.loc[test_idx].copy(),
                        u_test,
                        variant=variant,
                        analysis_view="within_session_dynamic",
                        split_role="test",
                        split_id=split_id,
                        selection_mode=selection_mode,
                        targeted_policy=targeted_policy,
                        block_id=block_id,
                    )
                )
                lookup_rows.extend(
                    _feature_lookup_rows(
                        payload=payload,
                        session_id=session_id,
                        selected=selected,
                        analysis_view="within_session_dynamic",
                        split_id=split_id,
                        selection_mode=selection_mode,
                        targeted_policy=targeted_policy,
                    )
                )
                condition_label = str(dose_df["condition"].iloc[0])
                tc_higher = trunc_nll - full_nll
                common = {
                    "variant": variant,
                    "analysis_view": "within_session_dynamic",
                    "session_id": session_id,
                    "dose": float(dose),
                    "condition_label": condition_label,
                    "block_id": block_id,
                    "repeat_id": 0,
                    "split_id": split_id,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "selection_mode": selection_mode,
                    "targeted_policy": targeted_policy,
                    "feature_mode": feature_mode,
                    "d": d,
                    "residualize": residualize,
                    "tc_higher": float(tc_higher),
                }
                metrics_rows.extend(
                    [
                        {**common, "model": "gaussian", "heldout_nll": gaussian_nll, "delta_vs_gaussian": 0.0},
                        {**common, "model": "truncated_vine", "heldout_nll": trunc_nll, "delta_vs_gaussian": gaussian_nll - trunc_nll},
                        {**common, "model": "full_vine", "heldout_nll": full_nll, "delta_vs_gaussian": gaussian_nll - full_nll},
                    ]
                )
                session_summary.setdefault(f"dose_{int(dose):03d}", {})[block_id] = {
                    "n_trials": int(len(block_idx)),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "selected_indices": selected.astype(int).tolist(),
                    "selection_meta": selection_meta,
                }
        if session_summary:
            summary["sessions"][session_id] = session_summary

    return (
        pd.DataFrame(benchmark_rows),
        pd.DataFrame(metrics_rows),
        pd.DataFrame(lookup_rows).drop_duplicates(),
        summary,
    )


def _dynamic_axis_label(dynamic_df: pd.DataFrame) -> str:
    if dynamic_df.empty:
        return "no_useful_temporal_axis"
    full_df = dynamic_df[dynamic_df["model"] == "full_vine"].copy()
    if full_df.empty:
        return "no_useful_temporal_axis"
    block_means = full_df.groupby("block_id")["delta_vs_gaussian"].mean()
    tc_means = full_df.groupby("block_id")["tc_higher"].mean()
    if {"early", "middle", "late"}.issubset(block_means.index):
        spread = float(block_means.max() - block_means.min())
        tc_spread = float(tc_means.max() - tc_means.min()) if len(tc_means) else 0.0
        if spread > 0.15 or tc_spread > 0.15:
            return "credible_temporal_axis"
        if spread > 0.05 or tc_spread > 0.05:
            return "weak_temporal_axis"
    return "no_useful_temporal_axis"


def _plot_variant_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    axes[0].bar(summary_df["variant"], summary_df["full_delta_vs_gaussian_mean"], color="#d62728")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Full vs Gaussian")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(summary_df["variant"], summary_df["full_beats_trunc_prop"], color="#4c78a8")
    axes[1].axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.6)
    axes[1].set_title("Prop. full beats 1-trunc")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(summary_df["variant"], summary_df["tc_higher_mean"], color="#2ca02c")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("Mean TC_higher")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_dose_summary(metrics_df: pd.DataFrame, best_variant: str, out_path: Path) -> None:
    df = metrics_df[(metrics_df["variant"] == best_variant) & (metrics_df["analysis_view"] == "dose_static")].copy()
    model_df = df[df["model"].isin(["truncated_vine", "full_vine"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
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
        axes[0].errorbar(frame["dose"], frame["mean"], yerr=frame["ci"], marker="o", capsize=3, linewidth=2, color=colors[model], label=model)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title(f"{best_variant}: gain vs Gaussian")
    axes[0].set_xlabel("Dose")
    axes[0].set_ylabel("Gaussian NLL - model NLL")
    axes[0].legend(frameon=False)

    tc_frame = (
        df.groupby(["dose", "split_id"])["tc_higher"]
        .first()
        .reset_index()
        .groupby("dose")["tc_higher"]
        .apply(lambda s: pd.Series({"mean": _mean_ci(s)[0], "ci": _mean_ci(s)[1]}))
        .unstack()
        .reset_index()
        .sort_values("dose")
    )
    axes[1].errorbar(tc_frame["dose"], tc_frame["mean"], yerr=tc_frame["ci"], marker="o", capsize=3, linewidth=2, color="#2ca02c")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Higher-order contribution")
    axes[1].set_xlabel("Dose")
    axes[1].set_ylabel("TC_higher")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_best_session_robustness(metrics_df: pd.DataFrame, best_variant: str, out_path: Path) -> None:
    df = metrics_df[(metrics_df["variant"] == best_variant) & (metrics_df["analysis_view"] == "dose_static")].copy()
    full_df = df[df["model"] == "full_vine"].copy()
    session_mean = full_df.groupby(["session_id", "dose"])["delta_vs_gaussian"].mean().reset_index()
    overall = session_mean.groupby("dose")["delta_vs_gaussian"].mean().reset_index().sort_values("dose")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for session_id, group in session_mean.groupby("session_id"):
        group = group.sort_values("dose")
        axes[0].plot(group["dose"], group["delta_vs_gaussian"], color="#bdbdbd", alpha=0.6)
        axes[0].scatter(group["dose"], group["delta_vs_gaussian"], color="#bdbdbd", s=16, alpha=0.7)
    axes[0].plot(overall["dose"], overall["delta_vs_gaussian"], color="#d62728", linewidth=2.5, marker="o")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_title("Session robustness: full vs Gaussian")
    axes[0].set_xlabel("Dose")
    axes[0].set_ylabel("Mean delta vs Gaussian")

    tc_df = full_df.groupby(["session_id", "dose", "split_id"])["tc_higher"].first().reset_index()
    tc_session = tc_df.groupby(["session_id", "dose"])["tc_higher"].mean().reset_index()
    tc_overall = tc_session.groupby("dose")["tc_higher"].mean().reset_index().sort_values("dose")
    for session_id, group in tc_session.groupby("session_id"):
        group = group.sort_values("dose")
        axes[1].plot(group["dose"], group["tc_higher"], color="#c7e9c0", alpha=0.6)
        axes[1].scatter(group["dose"], group["tc_higher"], color="#c7e9c0", s=16, alpha=0.7)
    axes[1].plot(tc_overall["dose"], tc_overall["tc_higher"], color="#238b45", linewidth=2.5, marker="o")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Session robustness: TC_higher")
    axes[1].set_xlabel("Dose")
    axes[1].set_ylabel("Mean TC_higher")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dynamic_check(dynamic_df: pd.DataFrame, best_variant: str, out_path: Path) -> None:
    full_df = dynamic_df[(dynamic_df["variant"] == best_variant) & (dynamic_df["model"] == "full_vine")].copy()
    if full_df.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.text(0.5, 0.5, "No dynamic blocks met inclusion criteria.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return
    block_order = {"early": 0, "middle": 1, "late": 2}
    full_df["block_order"] = full_df["block_id"].map(block_order)
    per_session = full_df.groupby(["session_id", "block_id", "block_order"])["delta_vs_gaussian"].mean().reset_index()
    summary = per_session.groupby(["block_id", "block_order"])["delta_vs_gaussian"].mean().reset_index().sort_values("block_order")
    tc_per_session = full_df.groupby(["session_id", "block_id", "block_order"])["tc_higher"].mean().reset_index()
    tc_summary = tc_per_session.groupby(["block_id", "block_order"])["tc_higher"].mean().reset_index().sort_values("block_order")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for session_id, group in per_session.groupby("session_id"):
        group = group.sort_values("block_order")
        axes[0].plot(group["block_order"], group["delta_vs_gaussian"], color="#bdbdbd", alpha=0.7)
        axes[0].scatter(group["block_order"], group["delta_vs_gaussian"], color="#969696", s=20, alpha=0.75)
    axes[0].plot(summary["block_order"], summary["delta_vs_gaussian"], color="#d62728", linewidth=2.8, marker="o")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xticks([0, 1, 2], ["early", "middle", "late"])
    axes[0].set_title("Dynamic check: full vs Gaussian")
    axes[0].set_ylabel("Delta vs Gaussian")

    for session_id, group in tc_per_session.groupby("session_id"):
        group = group.sort_values("block_order")
        axes[1].plot(group["block_order"], group["tc_higher"], color="#c7e9c0", alpha=0.7)
        axes[1].scatter(group["block_order"], group["tc_higher"], color="#74c476", s=20, alpha=0.75)
    axes[1].plot(tc_summary["block_order"], tc_summary["tc_higher"], color="#238b45", linewidth=2.8, marker="o")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks([0, 1, 2], ["early", "middle", "late"])
    axes[1].set_title("Dynamic check: TC_higher")
    axes[1].set_ylabel("TC_higher")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    logger = __import__("logging").getLogger(LOGGER_NAME)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    families = list(FAMILY_VARIANTS[args.family_variant])

    logger.info("Building manifest and trial summaries from %s", args.data_root)
    manifest = builder.build_manifest(args.data_root)
    builder.print_manifest_summary(manifest)
    trials_df, neural_data, trial_inference_rows = builder.build_trials(args.data_root, manifest)
    trials_df, neural_data, session_subset = _restrict_to_top_sessions(trials_df, neural_data, max_sessions=0)

    all_benchmark: List[pd.DataFrame] = []
    all_metrics: List[pd.DataFrame] = []
    all_lookup: List[pd.DataFrame] = []
    variant_run_summary: Dict[str, Any] = {}

    for variant_cfg in _variant_configs():
        logger.info("Running full static benchmark for %s", variant_cfg["variant"])
        benchmark_df, metrics_df, lookup_df, session_summary = _run_static_variant(
            trials_df=trials_df,
            neural_data=neural_data,
            variant_cfg=variant_cfg,
            families=families,
            seed=int(args.seed),
            n_repeats=int(args.n_repeats),
            train_fraction=float(args.train_fraction),
            min_trials_per_slice=int(args.min_trials_per_slice),
            selection_mode=args.selection_mode,
        )
        if not benchmark_df.empty:
            all_benchmark.append(benchmark_df)
        if not metrics_df.empty:
            all_metrics.append(metrics_df)
        if not lookup_df.empty:
            lookup_df = lookup_df.copy()
            lookup_df["variant"] = variant_cfg["variant"]
            lookup_df["feature_mode"] = variant_cfg["feature_mode"]
            lookup_df["d"] = variant_cfg["d"]
            all_lookup.append(lookup_df)
        variant_run_summary[variant_cfg["variant"]] = session_summary
        logger.info(
            "%s produced %d benchmark rows and %d metric rows",
            variant_cfg["variant"],
            len(benchmark_df),
            len(metrics_df),
        )

    if not all_metrics:
        raise RuntimeError("No variant produced usable benchmark rows.")

    benchmark_df = pd.concat(all_benchmark, axis=0, ignore_index=True)
    metrics_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    lookup_df = pd.concat(all_lookup, axis=0, ignore_index=True).drop_duplicates() if all_lookup else pd.DataFrame()
    variant_summary_df = _summarize_variants(metrics_df)
    best_variant = _choose_best_variant(variant_summary_df)
    best_cfg = next(cfg for cfg in _variant_configs() if cfg["variant"] == best_variant)
    logger.info("Selected best variant for dynamic follow-up: %s", best_variant)

    dyn_benchmark_df, dyn_metrics_df, dyn_lookup_df, dynamic_summary_meta = _run_dynamic_best_variant(
        neural_data=neural_data,
        variant_cfg=best_cfg,
        families=families,
        seed=int(args.seed),
        train_fraction=float(args.train_fraction),
        dynamic_min_block_trials=int(args.dynamic_min_block_trials),
        max_doses_per_session=int(args.dynamic_max_doses_per_session),
        selection_mode=args.selection_mode,
    )
    if not dyn_benchmark_df.empty:
        benchmark_df = pd.concat([benchmark_df, dyn_benchmark_df], axis=0, ignore_index=True)
    if not dyn_metrics_df.empty:
        metrics_df = pd.concat([metrics_df, dyn_metrics_df], axis=0, ignore_index=True)
    if not dyn_lookup_df.empty:
        dyn_lookup_df = dyn_lookup_df.copy()
        dyn_lookup_df["variant"] = best_variant
        dyn_lookup_df["feature_mode"] = best_cfg["feature_mode"]
        dyn_lookup_df["d"] = best_cfg["d"]
        lookup_df = pd.concat([lookup_df, dyn_lookup_df], axis=0, ignore_index=True).drop_duplicates()

    dynamic_summary_df = metrics_df[metrics_df["analysis_view"] == "within_session_dynamic"].copy()
    temporal_axis_label = _dynamic_axis_label(dynamic_summary_df)

    benchmark_df = benchmark_df.sort_values(["variant", "analysis_view", "session_id", "dose", "block_id", "split_id", "split", "trial_order"]).reset_index(drop=True)
    metrics_df = metrics_df.sort_values(["variant", "analysis_view", "session_id", "dose", "block_id", "repeat_id", "model"]).reset_index(drop=True)
    if not lookup_df.empty:
        lookup_df = lookup_df.sort_values(["variant", "analysis_view", "session_id", "split_id", "x_col"]).reset_index(drop=True)

    benchmark_df.to_parquet(out_root / "benchmark_table.parquet", index=False)
    metrics_df.to_csv(out_root / "metrics_table.csv", index=False)
    variant_summary_df.to_csv(out_root / "variant_summary.csv", index=False)
    dynamic_summary_df.to_csv(out_root / "dynamic_summary.csv", index=False)
    if not lookup_df.empty:
        lookup_df.to_csv(out_root / "neuron_lookup.csv", index=False)
    trials_df.to_csv(out_root / "trial_table.csv", index=False)
    pd.DataFrame(trial_inference_rows).to_csv(out_root / "trial_inference.csv", index=False)
    _write_json(out_root / "dataset_manifest.json", manifest)

    metadata = {
        "data_root": str(Path(args.data_root).resolve()),
        "out_root": str(out_root),
        "family_variant": args.family_variant,
        "families": families,
        "selection_mode": args.selection_mode,
        "n_repeats": int(args.n_repeats),
        "train_fraction": float(args.train_fraction),
        "min_trials_per_slice": int(args.min_trials_per_slice),
        "dynamic_min_block_trials": int(args.dynamic_min_block_trials),
        "dynamic_max_doses_per_session": int(args.dynamic_max_doses_per_session),
        "session_subset": session_subset,
        "variants": _variant_configs(),
        "best_variant": best_variant,
        "dynamic_axis_label": temporal_axis_label,
        "variant_run_summary": variant_run_summary,
        "dynamic_summary_meta": dynamic_summary_meta,
        "leakage_policy": {
            "selection": "neuron selection uses train trials only within each repeat or each dynamic block",
            "centering": "demeaning uses train means only when residualize=True",
            "winsorization": "winsorization bounds fit on train only and applied to test",
            "pseudo_observations": "empirical CDF fit on train only and applied to test",
            "model_scoring": "repository vine fit path with held-out scoring on train-derived pseudo-observations",
        },
    }
    _write_json(out_root / "benchmark_metadata.json", metadata)

    _plot_variant_comparison(variant_summary_df, PROJECT_ROOT / "fig_realdata_fullrerun_variant_comparison.png")
    _plot_best_dose_summary(metrics_df, best_variant, PROJECT_ROOT / "fig_realdata_best_variant_dose_summary.png")
    _plot_best_session_robustness(metrics_df, best_variant, PROJECT_ROOT / "fig_realdata_best_variant_session_robustness.png")
    _plot_dynamic_check(dynamic_summary_df, best_variant, PROJECT_ROOT / "fig_realdata_best_variant_dynamic_check.png")

    logger.info("Best variant: %s | temporal axis: %s", best_variant, temporal_axis_label)
    logger.info("Benchmark rows=%d metrics rows=%d", len(benchmark_df), len(metrics_df))


if __name__ == "__main__":
    main()
