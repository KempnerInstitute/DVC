#!/usr/bin/env python3
"""Debug and validate the Dalgleish real-data benchmark for DVC."""

from __future__ import annotations

import argparse
import itertools
import logging
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
from dvc_package.experiments.simulation_benchmarks import (
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
    _gaussian_copula_nll_fit_eval,
    _make_levelwise_cvine,
    _mean_copula_nll,
)
from scripts.debug_stimulation_exp.run_dalgleish_real_data_benchmark import (
    DEFAULT_FAMILY_VARIANT,
    FAMILY_VARIANTS,
    _apply_train_only_ecdf,
    _fit_train_only_ecdf,
    _restrict_to_top_sessions,
    _score_vine_on_uniforms,
    _select_neurons_for_train_pool,
    _session_seed,
    _split_positions_random,
    _with_quieter_repo_logging,
    _winsorize_train_apply,
)


LOGGER = logging.getLogger("dalgleish_debug")
OUT_ROOT = PROJECT_ROOT / "dvc_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug the Dalgleish real-data benchmark.")
    parser.add_argument("--data_root", default="dataset_stimulation")
    parser.add_argument("--out_root", default="dvc_ready")
    parser.add_argument("--max_sessions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.getLogger().setLevel(level)
    logging.getLogger("DVC.vine").setLevel(logging.ERROR)


def rank_uniform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    u = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        ranks = np.argsort(np.argsort(x[:, j], kind="mergesort"), kind="mergesort").astype(np.float64) + 1.0
        u[:, j] = ranks / (n + 1.0)
    return np.clip(u, 1e-6, 1.0 - 1e-6)


def evaluate_models_from_arrays(
    train_x: np.ndarray,
    test_x: np.ndarray,
    families: Sequence[str],
    scorer_mode: str,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    out: Dict[str, float] = {
        "gaussian": np.nan,
        "truncated_vine": np.nan,
        "full_vine": np.nan,
        "tc_higher": np.nan,
    }
    status = {"gaussian": "ok", "truncated_vine": "ok", "full_vine": "ok"}

    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    mappings = _fit_train_only_ecdf(train_x.astype(np.float64))
    u_test = _apply_train_only_ecdf(test_x.astype(np.float64), mappings)

    out["gaussian"] = float(_gaussian_copula_nll_fit_eval(train_x, test_x))

    try:
        trunc = _with_quieter_repo_logging(
            _fit_truncated_cvine_level0,
            x_train=train_x,
            families=list(families),
            order=list(range(train_x.shape[1])),
        )
        if scorer_mode == "repo":
            out["truncated_vine"] = float(_mean_copula_nll(trunc, test_x))
        else:
            out["truncated_vine"] = float(_score_vine_on_uniforms(trunc, u_test))
    except Exception as exc:
        status["truncated_vine"] = f"fail:{exc}"

    try:
        full = _with_quieter_repo_logging(
            _fit_parametric_vine,
            x_train=train_x,
            families=list(families),
            optimize_structure=False,
            seed=seed,
        )
        if scorer_mode == "repo":
            out["full_vine"] = float(_mean_copula_nll(full, test_x))
        else:
            out["full_vine"] = float(_score_vine_on_uniforms(full, u_test))
    except Exception as exc:
        status["full_vine"] = f"fail:{exc}"

    if np.isfinite(out["truncated_vine"]) and np.isfinite(out["full_vine"]):
        out["tc_higher"] = float(out["truncated_vine"] - out["full_vine"])
    return out, status


def status_from_expectation(dataset_name: str, family_variant: str, scores: Dict[str, float]) -> str:
    gauss = scores["gaussian"]
    trunc = scores["truncated_vine"]
    full = scores["full_vine"]
    if dataset_name == "gaussian_only":
        if family_variant == "default" and np.isfinite(full) and full > gauss + 10.0:
            return "fail"
        if np.isfinite(full) and np.isfinite(gauss) and abs(full - gauss) <= 2.0:
            return "pass"
        return "warning"
    if dataset_name == "pairwise_nongaussian":
        if family_variant == "default" and np.isfinite(full) and np.isfinite(trunc) and full > trunc + 10.0:
            return "fail"
        if np.isfinite(trunc) and np.isfinite(gauss) and trunc < gauss and np.isfinite(full) and full <= trunc + 1.0:
            return "pass"
        return "warning"
    if dataset_name == "higher_order":
        if np.isfinite(full) and np.isfinite(trunc) and full < trunc:
            return "pass"
        return "fail"
    return "warning"


def generate_synthetic_datasets(seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    sigma = 0.5 * np.ones((6, 6), dtype=np.float64) + 0.5 * np.eye(6, dtype=np.float64)
    out["gaussian_only"] = (
        rng.multivariate_normal(np.zeros(6), sigma, size=200).astype(np.float32),
        rng.multivariate_normal(np.zeros(6), sigma, size=100).astype(np.float32),
    )

    pair_vine = _make_levelwise_cvine(
        6,
        order=list(range(6)),
        level_families=["student", "ind", "ind", "ind", "ind"],
        level_thetas=[(0.6, 4.0), None, None, None, None],
    )
    out["pairwise_nongaussian"] = (
        pair_vine.sample(200).astype(np.float32),
        pair_vine.sample(100).astype(np.float32),
    )

    higher_vine = _make_levelwise_cvine(
        6,
        order=list(range(6)),
        level_families=["student", "clayton", "clayton", "clayton", "clayton"],
        level_thetas=[(0.6, 4.0), 2.0, 2.0, 2.0, 2.0],
    )
    out["higher_order"] = (
        higher_vine.sample(200).astype(np.float32),
        higher_vine.sample(100).astype(np.float32),
    )
    return out


def top_real_slices(
    trials_df: pd.DataFrame,
    max_sessions: int,
    min_trials: int,
) -> List[Tuple[str, float, int]]:
    usable = trials_df[trials_df["is_valid"] & trials_df["dose"].notna()].copy()
    usable, _neural_dummy, sessions = _restrict_to_top_sessions(usable, {sid: {"used": True} for sid in usable["session_id"].unique()}, max_sessions)
    counts = (
        usable.groupby(["session_id", "dose"]).size().reset_index(name="n").sort_values(["n", "session_id", "dose"], ascending=[False, True, True])
    )
    counts = counts[counts["n"] >= int(min_trials)]
    return [(str(r.session_id), float(r.dose), int(r.n)) for r in counts.itertuples(index=False)]


def build_real_slice_data(
    payload: Dict[str, Any],
    session_df: pd.DataFrame,
    dose: float,
    d: int,
    feature_mode: str,
    residualize: bool,
    include_targeted: bool,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    group_idx = session_df.index[session_df["dose"] == dose].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    train_pos, test_pos = _split_positions_random(len(group_idx), 0.7, rng)
    train_idx = group_idx[train_pos]
    test_idx = group_idx[test_pos]

    selected, meta = _select_neurons_for_train_pool(
        payload=payload,
        train_indices=train_idx,
        d=d,
        selection_mode="topk_responsive",
        exclude_targeted_neurons=not include_targeted,
        seed=seed,
    )
    if selected.size != d:
        raise RuntimeError(meta.get("warning", f"Could not select d={d} neurons"))

    arrays = payload["arrays"]
    if feature_mode == "stim_mean":
        raw = np.asarray(arrays["stim"][:, selected], dtype=np.float64)
    else:
        raw = np.asarray(arrays["diff"][:, selected], dtype=np.float64)
    train_x = raw[train_idx]
    test_x = raw[test_idx]
    if residualize:
        train_mean = np.nanmean(train_x, axis=0)
        train_x = train_x - train_mean
        test_x = test_x - train_mean
    train_x, test_x = _winsorize_train_apply(train_x, test_x)
    return train_x, test_x, {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "selected_indices": selected.astype(int).tolist(),
    }


def pairwise_jaccard(sets: List[Sequence[int]]) -> float:
    if len(sets) < 2:
        return 1.0
    vals: List[float] = []
    for a, b in itertools.combinations(sets, 2):
        aa = set(int(x) for x in a)
        bb = set(int(x) for x in b)
        denom = len(aa | bb)
        vals.append(1.0 if denom == 0 else len(aa & bb) / denom)
    return float(np.mean(vals)) if vals else 1.0


def compute_selection_stability(
    neural_data: Dict[str, Any],
    trials_df: pd.DataFrame,
    sessions: Sequence[str],
    d_values: Sequence[int],
    include_targeted: bool,
    seed: int,
) -> Dict[Tuple[str, int, bool], float]:
    out: Dict[Tuple[str, int, bool], float] = {}
    for session_id in sessions:
        payload = neural_data[session_id]
        session_df = payload["trial_table"].reset_index(drop=True)
        session_df = session_df[session_df["is_valid"] & session_df["dose"].notna()].copy()
        for d in d_values:
            picks: List[Sequence[int]] = []
            for rep in range(3):
                group_idx = session_df.index.to_numpy(dtype=int)
                rng = np.random.default_rng(_session_seed(seed, session_id, rep + 17 * d))
                train_pos, _ = _split_positions_random(len(group_idx), 0.7, rng)
                train_idx = group_idx[train_pos]
                selected, _meta = _select_neurons_for_train_pool(
                    payload=payload,
                    train_indices=train_idx,
                    d=d,
                    selection_mode="topk_responsive",
                    exclude_targeted_neurons=not include_targeted,
                    seed=_session_seed(seed, session_id, rep + d),
                )
                if selected.size == d:
                    picks.append(selected.tolist())
            out[(session_id, d, include_targeted)] = pairwise_jaccard(picks)
    return out


def run_debug_suite(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = builder.build_manifest(args.data_root)
    trials_df, neural_data, _trial_inference = builder.build_trials(args.data_root, manifest)
    trials_df, neural_data, session_subset = _restrict_to_top_sessions(trials_df, neural_data, int(args.max_sessions))
    sessions = [sid for sid in session_subset if sid in neural_data and neural_data[sid].get("used", False)]
    LOGGER.info("Using sessions for debug suite: %s", ", ".join(sessions))

    debug_rows: List[Dict[str, Any]] = []
    slice_diag_rows: List[Dict[str, Any]] = []

    # Phase 1: audit notes.
    audit_notes = [
        ("audit_transform_consistency", "same model objects, same held-out samples; current custom scorer differs only in held-out pseudo-observation mapping"),
        ("audit_sign_scale", "Gaussian and vine scorers all return mean held-out NLL with lower=better; no sum/mean mix was found in the checked code paths"),
        ("audit_boundary_handling", "All checked paths clip pseudo-observations into (1e-6, 1-1e-6)"),
    ]
    for check_name, notes in audit_notes:
        debug_rows.append(
            {
                "check_name": check_name,
                "dataset_type": "synthetic",
                "slice_id": "code_audit",
                "variant": "audit",
                "model": "all",
                "heldout_nll": np.nan,
                "delta_vs_gaussian": np.nan,
                "tc_higher": np.nan,
                "status": "pass",
                "notes": notes,
            }
        )

    # Phase 2A: synthetic sanity.
    synth = generate_synthetic_datasets(seed=args.seed)
    for dataset_name, (train_x, test_x) in synth.items():
        for family_variant, fams in FAMILY_VARIANTS.items():
            repo_scores, _status = evaluate_models_from_arrays(
                train_x=train_x,
                test_x=test_x,
                families=fams,
                scorer_mode="repo",
                seed=args.seed,
            )
            for model in ["gaussian", "truncated_vine", "full_vine"]:
                heldout = repo_scores[model]
                delta = repo_scores["gaussian"] - heldout if np.isfinite(heldout) and np.isfinite(repo_scores["gaussian"]) else np.nan
                row_status = status_from_expectation(dataset_name, family_variant, repo_scores)
                debug_rows.append(
                    {
                        "check_name": "synthetic_sanity",
                        "dataset_type": "synthetic",
                        "slice_id": dataset_name,
                        "variant": f"family={family_variant};score=repo",
                        "model": model,
                        "heldout_nll": float(heldout) if np.isfinite(heldout) else np.nan,
                        "delta_vs_gaussian": float(delta) if np.isfinite(delta) else np.nan,
                        "tc_higher": float(repo_scores["tc_higher"]) if np.isfinite(repo_scores["tc_higher"]) else np.nan,
                        "status": row_status,
                        "notes": "Synthetic sanity check under repository scorer.",
                    }
                )

    # Phase 2B: real-data scorer comparison on representative slices.
    representative = top_real_slices(trials_df, max_sessions=int(args.max_sessions), min_trials=20)[:6]
    for session_id, dose, n_trials in representative:
        payload = neural_data[session_id]
        session_df = payload["trial_table"].reset_index(drop=True)
        session_df = session_df[session_df["is_valid"] & session_df["dose"].notna()].copy()
        for family_variant, fams in FAMILY_VARIANTS.items():
            train_x, test_x, meta = build_real_slice_data(
                payload=payload,
                session_df=session_df,
                dose=dose,
                d=6,
                feature_mode="stim_minus_baseline",
                residualize=True,
                include_targeted=False,
                seed=_session_seed(args.seed, session_id, int(dose)),
            )
            repo_scores, fit_status = evaluate_models_from_arrays(train_x, test_x, fams, scorer_mode="repo", seed=args.seed)
            custom_scores, _ = evaluate_models_from_arrays(train_x, test_x, fams, scorer_mode="custom", seed=args.seed)
            for model in ["truncated_vine", "full_vine"]:
                repo_nll = repo_scores[model]
                custom_nll = custom_scores[model]
                diff = custom_nll - repo_nll if np.isfinite(repo_nll) and np.isfinite(custom_nll) else np.nan
                debug_rows.append(
                    {
                        "check_name": "real_scorer_comparison",
                        "dataset_type": "real",
                        "slice_id": f"{session_id}__dose_{int(dose):03d}",
                        "variant": f"family={family_variant};repo_vs_custom",
                        "model": model,
                        "heldout_nll": float(custom_nll) if np.isfinite(custom_nll) else np.nan,
                        "delta_vs_gaussian": float(diff) if np.isfinite(diff) else np.nan,
                        "tc_higher": float(custom_scores["tc_higher"]) if np.isfinite(custom_scores["tc_higher"]) else np.nan,
                        "status": "pass" if np.isfinite(diff) and abs(diff) < 10.0 else "warning",
                        "notes": f"repo_score={repo_nll:.4f}; custom_score={custom_nll:.4f}; n_train={meta['n_train']}; n_test={meta['n_test']}; fit={fit_status.get(model,'ok')}",
                    }
                )

    # Phase 3 + 4: compact real-data ablations and sample-size diagnostics.
    base_sessions = sessions
    stability_excl = compute_selection_stability(neural_data, trials_df, base_sessions, [4, 6, 8], include_targeted=False, seed=args.seed)
    stability_incl = compute_selection_stability(neural_data, trials_df, base_sessions, [4, 6, 8], include_targeted=True, seed=args.seed)
    ablation_specs = [
        ("base", {"residualize": True, "include_targeted": False, "feature_mode": "stim_minus_baseline", "family_variant": "stable", "d": 6}),
        ("no_residualization", {"residualize": False, "include_targeted": False, "feature_mode": "stim_minus_baseline", "family_variant": "stable", "d": 6}),
        ("include_targeted", {"residualize": True, "include_targeted": True, "feature_mode": "stim_minus_baseline", "family_variant": "stable", "d": 6}),
        ("stim_mean", {"residualize": True, "include_targeted": False, "feature_mode": "stim_mean", "family_variant": "stable", "d": 6}),
        ("family_default", {"residualize": True, "include_targeted": False, "feature_mode": "stim_minus_baseline", "family_variant": "default", "d": 6}),
        ("d4", {"residualize": True, "include_targeted": False, "feature_mode": "stim_minus_baseline", "family_variant": "stable", "d": 4}),
        ("d8", {"residualize": True, "include_targeted": False, "feature_mode": "stim_minus_baseline", "family_variant": "stable", "d": 8}),
    ]

    updated_metrics_rows: List[Dict[str, Any]] = []
    for variant_name, spec in ablation_specs:
        families = FAMILY_VARIANTS[spec["family_variant"]]
        include_targeted = bool(spec["include_targeted"])
        for session_id in base_sessions:
            payload = neural_data[session_id]
            session_df = payload["trial_table"].reset_index(drop=True)
            session_df = session_df[session_df["is_valid"] & session_df["dose"].notna()].copy()
            dose_counts = session_df.groupby("dose").size()
            eligible_doses = sorted(float(dose) for dose, n in dose_counts.items() if int(n) >= 20)
            for dose in eligible_doses:
                try:
                    train_x, test_x, meta = build_real_slice_data(
                        payload=payload,
                        session_df=session_df,
                        dose=dose,
                        d=int(spec["d"]),
                        feature_mode=str(spec["feature_mode"]),
                        residualize=bool(spec["residualize"]),
                        include_targeted=include_targeted,
                        seed=_session_seed(args.seed, session_id, int(100 * dose + spec["d"])),
                    )
                    scores, fit_status = evaluate_models_from_arrays(
                        train_x=train_x,
                        test_x=test_x,
                        families=families,
                        scorer_mode="custom",
                        seed=args.seed,
                    )
                    stability = (stability_incl if include_targeted else stability_excl).get((session_id, int(spec["d"]), include_targeted), np.nan)
                    full_delta = scores["gaussian"] - scores["full_vine"] if np.isfinite(scores["full_vine"]) else np.nan
                    trunc_delta = scores["gaussian"] - scores["truncated_vine"] if np.isfinite(scores["truncated_vine"]) else np.nan
                    for model in ["gaussian", "truncated_vine", "full_vine"]:
                        heldout = scores[model]
                        delta = scores["gaussian"] - heldout if np.isfinite(heldout) else np.nan
                        debug_rows.append(
                            {
                                "check_name": "real_ablation",
                                "dataset_type": "real",
                                "slice_id": f"{session_id}__dose_{int(dose):03d}",
                                "variant": variant_name,
                                "model": model,
                                "heldout_nll": float(heldout) if np.isfinite(heldout) else np.nan,
                                "delta_vs_gaussian": float(delta) if np.isfinite(delta) else np.nan,
                                "tc_higher": float(scores["tc_higher"]) if np.isfinite(scores["tc_higher"]) else np.nan,
                                "status": "pass" if np.isfinite(heldout) else "fail",
                                "notes": f"family={spec['family_variant']}; residualize={spec['residualize']}; targeted={include_targeted}; feature={spec['feature_mode']}; d={spec['d']}",
                            }
                        )
                        if variant_name == "base":
                            updated_metrics_rows.append(
                                {
                                    "analysis_view": "dose_static",
                                    "session_id": session_id,
                                    "dose": dose,
                                    "window_id": "",
                                    "repeat_id": 0,
                                    "split_id": f"{session_id}__dose_{int(dose):03d}__stable_debug",
                                    "n_train": meta["n_train"],
                                    "n_test": meta["n_test"],
                                    "selection_mode": "topk_responsive",
                                    "targeted_policy": "include_targeted" if include_targeted else "exclude_direct_targets",
                                    "model": model,
                                    "heldout_nll": float(heldout) if np.isfinite(heldout) else np.nan,
                                    "delta_vs_gaussian": float(delta) if np.isfinite(delta) else np.nan,
                                    "tc_higher": float(scores["tc_higher"]) if np.isfinite(scores["tc_higher"]) else np.nan,
                                    "condition_label": f"dose_{int(dose):03d}",
                                }
                            )

                    slice_diag_rows.append(
                        {
                            "session_id": session_id,
                            "dose": dose,
                            "analysis_view": "dose_static",
                            "variant": variant_name,
                            "n_train": meta["n_train"],
                            "n_test": meta["n_test"],
                            "d": int(spec["d"]),
                            "selection_mode": "topk_responsive",
                            "targeted_policy": "include_targeted" if include_targeted else "exclude_direct_targets",
                            "selection_stability": stability,
                            "fit_status": "ok" if all(v == "ok" for v in fit_status.values()) else str(fit_status),
                            "score_status": "ok" if np.isfinite(full_delta) and np.isfinite(trunc_delta) else "warning",
                        }
                    )
                except Exception as exc:
                    slice_diag_rows.append(
                        {
                            "session_id": session_id,
                            "dose": dose,
                            "analysis_view": "dose_static",
                            "variant": variant_name,
                            "n_train": np.nan,
                            "n_test": np.nan,
                            "d": int(spec["d"]),
                            "selection_mode": "topk_responsive",
                            "targeted_policy": "include_targeted" if include_targeted else "exclude_direct_targets",
                            "selection_stability": np.nan,
                            "fit_status": f"fail:{exc}",
                            "score_status": "fail",
                        }
                    )

    debug_df = pd.DataFrame(debug_rows)
    slice_diag_df = pd.DataFrame(slice_diag_rows)
    updated_metrics_df = pd.DataFrame(updated_metrics_rows)

    debug_df.to_csv(out_root / "debug_summary.csv", index=False)
    slice_diag_df.to_csv(out_root / "slice_diagnostics.csv", index=False)
    if not updated_metrics_df.empty:
        updated_metrics_df.to_csv(out_root / "metrics_table.csv", index=False)

    make_debug_figures(debug_df=debug_df, slice_diag_df=slice_diag_df, out_root=PROJECT_ROOT)


def make_debug_figures(debug_df: pd.DataFrame, slice_diag_df: pd.DataFrame, out_root: Path) -> None:
    scorer = debug_df[debug_df["check_name"] == "real_scorer_comparison"].copy()
    if not scorer.empty:
        parts = scorer["notes"].str.extract(r"repo_score=([-0-9.]+); custom_score=([-0-9.]+)")
        scorer["repo_score"] = pd.to_numeric(parts[0], errors="coerce")
        scorer["custom_score"] = pd.to_numeric(parts[1], errors="coerce")
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for model, group in scorer.groupby("model"):
            ax.scatter(group["repo_score"], group["custom_score"], label=model, alpha=0.8)
        lims = [
            np.nanmin([scorer["repo_score"].min(), scorer["custom_score"].min()]),
            np.nanmax([scorer["repo_score"].max(), scorer["custom_score"].max()]),
        ]
        ax.plot(lims, lims, color="black", linewidth=1)
        ax.set_xlabel("Repository score")
        ax.set_ylabel("Current custom score")
        ax.set_title("Real-data scorer comparison")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_root / "fig_debug_scorer_comparison.png", bbox_inches="tight")
        plt.close(fig)

    synth = debug_df[debug_df["check_name"] == "synthetic_sanity"].copy()
    if not synth.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
        for ax, dataset in zip(axes, ["gaussian_only", "pairwise_nongaussian", "higher_order"]):
            frame = synth[(synth["slice_id"] == dataset) & (synth["model"].isin(["gaussian", "truncated_vine", "full_vine"]))].copy()
            pivot = frame.pivot_table(index="model", columns="variant", values="heldout_nll", aggfunc="mean")
            pivot = pivot.reindex(["gaussian", "truncated_vine", "full_vine"])
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(dataset.replace("_", " "))
            ax.set_ylabel("Held-out NLL")
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_root / "fig_debug_synthetic_sanity.png", bbox_inches="tight")
        plt.close(fig)

    abl = debug_df[(debug_df["check_name"] == "real_ablation") & (debug_df["model"] == "full_vine")].copy()
    if not abl.empty:
        summary = abl.groupby("variant")[["delta_vs_gaussian", "tc_higher"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].bar(summary["variant"], summary["delta_vs_gaussian"], color="#4c78a8")
        axes[0].axhline(0.0, color="black", linewidth=1)
        axes[0].set_title("Full-vine gain vs Gaussian")
        axes[0].tick_params(axis="x", rotation=35)
        axes[1].bar(summary["variant"], summary["tc_higher"], color="#f58518")
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set_title("TC_higher by ablation")
        axes[1].tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(out_root / "fig_debug_ablations.png", bbox_inches="tight")
        plt.close(fig)

        sample = slice_diag_df[slice_diag_df["variant"].isin(["base", "family_default"])].copy()
        merged = sample.merge(
            abl[["slice_id", "variant", "delta_vs_gaussian"]],
            left_on=["variant"],
            right_on=["variant"],
            how="left",
        )
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        for variant, group in sample.groupby("variant"):
            vals = abl[abl["variant"] == variant]
            keyed = vals["slice_id"].str.extract(r"(.+)__dose_(\d+)").rename(columns={0: "session_id", 1: "dose_key"})
            vals = vals.reset_index(drop=True).join(keyed)
            vals["dose"] = pd.to_numeric(vals["dose_key"], errors="coerce")
            joined = group.merge(vals[["session_id", "dose", "delta_vs_gaussian"]], on=["session_id", "dose"], how="left")
            ax.scatter(joined["n_train"], joined["delta_vs_gaussian"], label=variant, alpha=0.8)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("n_train")
        ax.set_ylabel("Full-vine delta vs Gaussian")
        ax.set_title("Sample size vs full-vine performance")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_root / "fig_debug_sample_size.png", bbox_inches="tight")
        plt.close(fig)

        before = abl[abl["variant"] == "family_default"][["slice_id", "heldout_nll"]].rename(columns={"heldout_nll": "default_nll"})
        after = abl[abl["variant"] == "base"][["slice_id", "heldout_nll"]].rename(columns={"heldout_nll": "stable_nll"})
        paired = before.merge(after, on="slice_id", how="inner")
        if not paired.empty:
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            for row in paired.itertuples(index=False):
                ax.plot([0, 1], [row.default_nll, row.stable_nll], color="#969696", alpha=0.6)
            ax.scatter(np.zeros(len(paired)), paired["default_nll"], color="#d62728", label="default")
            ax.scatter(np.ones(len(paired)), paired["stable_nll"], color="#2ca02c", label="stable")
            ax.set_xticks([0, 1], ["default", "stable"])
            ax.set_ylabel("Full-vine held-out NLL")
            ax.set_title("Before/after family restriction")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out_root / "fig_debug_before_after_fix.png", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    run_debug_suite(args)


if __name__ == "__main__":
    main()
