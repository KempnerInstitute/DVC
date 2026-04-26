#!/usr/bin/env python3
"""Configurable four-phase showcase benchmark for the final standalone workflow.

This keeps the original benchmark logic available via `--preset current`, while
also supporting:
- `--preset fixed`: preserve the phase-2 star when adding the phase-3 triplet
- `--preset harder`: fixed phase-3 semantics plus stronger higher-order and tail
  non-Gaussianity to make the Gaussian baselines fail more clearly
- `--preset contrast`: preserve a moderate Gaussian star and add pairwise-matched
  XOR-style triplets that Gaussian SSM should miss
- `--preset contrast_harder`: use multiplicative triplets to create a much larger
  higher-order gap over Gaussian baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.time.nonparametric_dynamic_cvine import WindowedNonparametricCVine
from showcase_analysis_utils import (
    DEFAULT_RESULTS_ROOT,
    ShowcaseConfig,
    aggregate_seed_runs,
    enrich_phasewise_deltas,
    evaluate_regularized_dynamic_dvc,
    evaluate_static_baselines,
    generate_sequence,
    make_seed_list,
    phase_acceptance_flags,
    split_train_test,
)


DEFAULT_OUT = DEFAULT_RESULTS_ROOT / "run_showcase_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final standalone showcase benchmark.")
    parser.add_argument(
        "--preset",
        choices=["current", "fixed", "harder", "contrast", "contrast_harder"],
        default="contrast_harder",
        help="Benchmark preset. `contrast_harder` is the recommended default.",
    )
    parser.add_argument(
        "--variant",
        choices=[
            "current",
            "fixed_phase3",
            "triplet_only",
            "xor_triplets",
            "xor_only",
            "multiplicative_triplets",
            "multiplicative_only",
        ],
        default=None,
        help="Optional direct override of the phase-3 generator semantics.",
    )
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of benchmark seeds.")
    parser.add_argument("--base-seed", type=int, default=2026, help="Base seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory for summary.json.",
    )
    parser.add_argument(
        "--n-per-time",
        type=int,
        default=None,
        help="Override samples per window.",
    )
    parser.add_argument(
        "--triplet-rho",
        type=float,
        default=None,
        help="Override phase-3 Student-t root-edge strength.",
    )
    parser.add_argument(
        "--triplet-nu",
        type=float,
        default=None,
        help="Override phase-3 Student-t degrees of freedom.",
    )
    parser.add_argument(
        "--triplet-clayton-theta",
        type=float,
        default=None,
        help="Override phase-3 higher-tree Clayton strength.",
    )
    parser.add_argument(
        "--tail-theta",
        type=float,
        default=None,
        help="Override phase-4 Clayton tail strength.",
    )
    parser.add_argument(
        "--skip-nf",
        action="store_true",
        help="Skip the NF-copula baseline.",
    )
    parser.add_argument(
        "--skip-mine",
        action="store_true",
        help="Skip the MINE pairwise MI panel metrics.",
    )
    parser.add_argument(
        "--mine-epochs",
        type=int,
        default=60,
        help="MINE epochs per window when MINE is enabled.",
    )
    parser.add_argument(
        "--nf-epochs",
        type=int,
        default=40,
        help="NF-copula epochs per window when NF is enabled.",
    )
    parser.add_argument(
        "--include-regularized-dvc",
        action="store_true",
        help="Also evaluate the regularized dynamic DVC comparator.",
    )
    parser.add_argument(
        "--include-nonparametric-dvc",
        action="store_true",
        help="Also evaluate the repaired windowed nonparametric DVC comparator.",
    )
    parser.add_argument(
        "--np-vine-type",
        choices=["c-vine", "d-vine", "r-vine", "auto"],
        default="d-vine",
        help="Vine structure for the windowed nonparametric DVC comparator.",
    )
    parser.add_argument("--np-knots", type=int, default=7, help="Uniform grid knots for the nonparametric comparator.")
    parser.add_argument("--np-higher-tree-validation-margin", type=float, default=None)
    parser.add_argument("--np-higher-tree-boundary-frac-threshold", type=float, default=None)
    return parser.parse_args()


def build_benchmark_setup(args: argparse.Namespace) -> Tuple[ShowcaseConfig, str, str]:
    base = ShowcaseConfig()
    if args.preset == "current":
        config = base
        variant = "current"
        rationale = "Original benchmark semantics, including the phase-3 overwrite behavior."
    elif args.preset == "fixed":
        config = base
        variant = "fixed_phase3"
        rationale = "Preserve the phase-2 star while adding the phase-3 higher-order triplet."
    elif args.preset == "harder":
        config = replace(
            base,
            triplet_rho=0.75,
            triplet_nu=3.5,
            triplet_clayton_theta=3.0,
            tail_theta=3.0,
        )
        variant = "fixed_phase3"
        rationale = (
            "Preserve the phase-2 star and strengthen the phase-3 higher-order "
            "and phase-4 tail signals so Gaussian baselines fail more clearly."
        )
    elif args.preset == "contrast":
        config = replace(
            base,
            pair_leaves=(1, 2, 3),
            pair_rho=0.6,
            phase3_mode="xor_triplets",
            triplet_blocks=((4, 5, 6), (7, 8, 9)),
            tail_theta=3.0,
            xor_jitter_std=5e-4,
        )
        variant = "xor_triplets"
        rationale = (
            "Keep a moderate Gaussian star in phase 2, then add two disjoint "
            "pairwise-matched XOR-style triplets in phase 3 so the higher-order "
            "signal is much less visible to Gaussian correlation models."
        )
    else:
        config = replace(
            base,
            pair_leaves=(1, 2, 3),
            pair_rho=0.55,
            phase3_mode="multiplicative_triplets",
            triplet_blocks=((4, 5, 6), (7, 8, 9)),
            multiplicative_noise_std=0.10,
            tail_theta=3.5,
        )
        variant = "multiplicative_triplets"
        rationale = (
            "Keep a moderate Gaussian star in phase 2, then add two disjoint "
            "multiplicative triplets in phase 3 to create strong higher-order "
            "structure that should open a much larger DVC-vs-Gaussian gap."
        )

    if args.n_per_time is not None:
        config = replace(config, n_per_time=int(args.n_per_time))
    if args.triplet_rho is not None:
        config = replace(config, triplet_rho=float(args.triplet_rho))
    if args.triplet_nu is not None:
        config = replace(config, triplet_nu=float(args.triplet_nu))
    if args.triplet_clayton_theta is not None:
        config = replace(config, triplet_clayton_theta=float(args.triplet_clayton_theta))
    if args.tail_theta is not None:
        config = replace(config, tail_theta=float(args.tail_theta))
    if args.variant is not None:
        variant = str(args.variant)

    return config, variant, rationale


def _run_pairwise_mine(
    x_by_t: List[np.ndarray],
    pair: tuple[int, int],
    *,
    mine_epochs: int,
) -> List[float]:
    from dvc_package.baselines.mine import mine_mi_estimate

    out: List[float] = []
    for t_idx, x in enumerate(x_by_t):
        try:
            mi = mine_mi_estimate(
                x[:, pair[0]],
                x[:, pair[1]],
                n_epochs=mine_epochs,
                seed=2026 + t_idx,
            )
        except Exception:
            mi = float("nan")
        out.append(float(mi))
    return out


def _run_pairwise_gaussian_mi(x_by_t: List[np.ndarray], pair: tuple[int, int]) -> List[float]:
    import pandas as pd

    out: List[float] = []
    for x in x_by_t:
        tau = float(pd.Series(x[:, pair[0]]).corr(pd.Series(x[:, pair[1]]), method="kendall"))
        if not np.isfinite(tau):
            out.append(float("nan"))
            continue
        tau = float(np.clip(tau, -0.999, 0.999))
        rho = float(np.clip(np.sin(np.pi * tau / 2.0), -0.999, 0.999))
        out.append(float(-0.5 * np.log(max(1e-12, 1.0 - rho * rho))))
    return out


def _merge_dynamic_rows(rows: List[Dict[str, Any]], dynamic_eval: Dict[str, Any]) -> None:
    for t_idx, row in enumerate(rows):
        row["nll_reg_dvc"] = float(dynamic_eval["nll"][t_idx])
        row["tc_total_reg_dvc"] = float(dynamic_eval["tc_total_reg_dvc"][t_idx])


def _count_np_edge_models(vine: Any) -> Dict[str, int]:
    n_kernel = 0
    n_ind = 0
    for level in getattr(vine, "copulas", []):
        for cop in level:
            family = str(getattr(cop, "family", "kercop"))
            selected = str(getattr(cop, "validation", {}).get("selected_model", "kernel"))
            if family == "ind" or selected == "independence":
                n_ind += 1
            else:
                n_kernel += 1
    return {"kernel": int(n_kernel), "independence": int(n_ind)}


def _run_windowed_nonparametric(
    x_train_by_t: List[np.ndarray],
    x_test_by_t: List[np.ndarray],
    *,
    vine_type: str,
    knots: int,
    higher_tree_validation_margin: float | None,
    higher_tree_boundary_frac_threshold: float | None,
) -> Dict[str, Any]:
    npc_dict: Dict[str, Any] = {
        "opt_method": "LL1",
        "max_iter_phase1": 1,
        "max_iter_phase2": 1,
        "normal_iters_phase1": 5,
        "normal_iters_phase2": 5,
        "final_normalization_iters": 25,
        "batch_size": 1,
        "data_space": "x",
    }
    if higher_tree_validation_margin is not None:
        npc_dict["higher_tree_validation_margin"] = float(higher_tree_validation_margin)
    if higher_tree_boundary_frac_threshold is not None:
        npc_dict["higher_tree_boundary_frac_threshold"] = float(higher_tree_boundary_frac_threshold)

    model = WindowedNonparametricCVine(
        knots=knots,
        npc_dict=npc_dict,
        vine_type=str(vine_type),
    )
    result = model.fit(x_train_by_t)
    nll = model.evaluate(x_test_by_t)
    selected_depth = []
    n_kernel = []
    n_independence = []
    for vine in result.vines_by_time:
        summary = getattr(vine, "nonparametric_summary", {}) or {}
        selected_depth.append(float(summary.get("selected_depth", float("nan"))))
        counts = _count_np_edge_models(vine)
        n_kernel.append(float(counts["kernel"]))
        n_independence.append(float(counts["independence"]))
    return {
        "nll": [float(v) for v in nll],
        "selected_depth": selected_depth,
        "n_kernel_edges": n_kernel,
        "n_independence_edges": n_independence,
        "order": list(result.order),
        "vine_type": str(vine_type),
        "mean_nll_by_time": [float(v) for v in result.mean_nll_by_time],
        "npc_dict": npc_dict,
        "knots": int(knots),
    }


def _run_single_seed(
    seed: int,
    *,
    config: ShowcaseConfig,
    variant: str,
    skip_nf: bool,
    skip_mine: bool,
    mine_epochs: int,
    nf_epochs: int,
    include_regularized_dvc: bool,
    include_nonparametric_dvc: bool,
    np_vine_type: str,
    np_knots: int,
    np_higher_tree_validation_margin: float | None,
    np_higher_tree_boundary_frac_threshold: float | None,
) -> Dict[str, Any]:
    windows = generate_sequence(seed=seed, config=config, variant=variant)
    x_train_by_t, x_test_by_t = split_train_test(windows, config.train_frac)
    static_eval = evaluate_static_baselines(
        x_train_by_t,
        x_test_by_t,
        config=config,
        seed=seed,
        skip_nf=skip_nf,
        nf_epochs=nf_epochs,
    )
    rows: List[Dict[str, Any]] = static_eval["rows"]

    dynamic_summary: Dict[str, Any] | None = None
    if include_regularized_dvc:
        dynamic_summary = evaluate_regularized_dynamic_dvc(x_train_by_t, x_test_by_t)
        _merge_dynamic_rows(rows, dynamic_summary)

    nonparametric_summary: Dict[str, Any] | None = None
    if include_nonparametric_dvc:
        nonparametric_summary = _run_windowed_nonparametric(
            x_train_by_t,
            x_test_by_t,
            vine_type=np_vine_type,
            knots=np_knots,
            higher_tree_validation_margin=np_higher_tree_validation_margin,
            higher_tree_boundary_frac_threshold=np_higher_tree_boundary_frac_threshold,
        )
        for t_idx, row in enumerate(rows):
            row["nll_np_windowed"] = float(nonparametric_summary["nll"][t_idx])
            row["tc_total_np_windowed"] = float(-nonparametric_summary["nll"][t_idx])
            row["np_selected_depth"] = float(nonparametric_summary["selected_depth"][t_idx])
            row["np_n_kernel_edges"] = float(nonparametric_summary["n_kernel_edges"][t_idx])
            row["np_n_independence_edges"] = float(nonparametric_summary["n_independence_edges"][t_idx])

    if skip_mine:
        mine_01 = [float("nan")] * config.t
        mine_56 = [float("nan")] * config.t
    else:
        print(f"\nRunning MINE for pair (0, 1) [seed {seed}] ...")
        mine_01 = _run_pairwise_mine(windows, pair=(0, 1), mine_epochs=mine_epochs)
        print(f"Running MINE for pair (5, 6) [seed {seed}] ...")
        mine_56 = _run_pairwise_mine(windows, pair=(5, 6), mine_epochs=mine_epochs)
    dvc_mi_01 = _run_pairwise_gaussian_mi(windows, pair=(0, 1))
    dvc_mi_56 = _run_pairwise_gaussian_mi(windows, pair=(5, 6))

    for t_idx in range(config.t):
        rows[t_idx]["mine_mi_pair01"] = float(mine_01[t_idx])
        rows[t_idx]["mine_mi_pair56"] = float(mine_56[t_idx])
        rows[t_idx]["dvc_pair_mi01"] = float(dvc_mi_01[t_idx])
        rows[t_idx]["dvc_pair_mi56"] = float(dvc_mi_56[t_idx])

    for phase_name in config.phases:
        phase_rows = [row for row in rows if row["phase_name"] == phase_name]
        tc_mean = float(np.nanmean([row["tc_total_dvc"] for row in phase_rows]))
        higher_mean = float(np.nanmean([row["tc_higher_dvc"] for row in phase_rows]))
        ssm_mean = float(np.nanmean([row.get("tc_total_ssm", np.nan) for row in phase_rows]))
        print(
            f"seed={seed} phase={phase_name:<22} "
            f"TC_total={tc_mean:+.3f} "
            f"TC_higher={higher_mean:+.3f} "
            f"TC_ssm={ssm_mean:+.3f}"
        )

    out: Dict[str, Any] = {
        "seed": int(seed),
        "rows": rows,
        "ssm_process_variance": float(static_eval["ssm_process_variance"]),
    }
    if dynamic_summary is not None:
        out["reg_dynamic_root_sequence"] = list(dynamic_summary["root_sequence"])
        out["reg_dynamic_total_family_switches"] = int(dynamic_summary["total_family_switches"])
        out["reg_dynamic_total_parameter_drift"] = float(dynamic_summary["total_parameter_drift"])
    if nonparametric_summary is not None:
        out["nonparametric_windowed"] = nonparametric_summary
    return out


def _summarize_regularized_dvc(seed_runs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    with_dynamic = [run for run in seed_runs if "reg_dynamic_root_sequence" in run]
    if not with_dynamic:
        return None
    return {
        "n_runs": len(with_dynamic),
        "mean_total_family_switches": float(
            np.mean([run["reg_dynamic_total_family_switches"] for run in with_dynamic])
        ),
        "mean_total_parameter_drift": float(
            np.mean([run["reg_dynamic_total_parameter_drift"] for run in with_dynamic])
        ),
        "root_sequences": [run["reg_dynamic_root_sequence"] for run in with_dynamic],
    }


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    config, variant, rationale = build_benchmark_setup(args)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = make_seed_list(args.n_seeds, args.base_seed)
    seed_runs = [
        _run_single_seed(
            seed,
            config=config,
            variant=variant,
            skip_nf=bool(args.skip_nf),
            skip_mine=bool(args.skip_mine),
            mine_epochs=int(args.mine_epochs),
            nf_epochs=int(args.nf_epochs),
            include_regularized_dvc=bool(args.include_regularized_dvc),
            include_nonparametric_dvc=bool(args.include_nonparametric_dvc),
            np_vine_type=str(args.np_vine_type),
            np_knots=int(args.np_knots),
            np_higher_tree_validation_margin=(
                None if args.np_higher_tree_validation_margin is None else float(args.np_higher_tree_validation_margin)
            ),
            np_higher_tree_boundary_frac_threshold=(
                None if args.np_higher_tree_boundary_frac_threshold is None else float(args.np_higher_tree_boundary_frac_threshold)
            ),
        )
        for seed in seeds
    ]
    aggregated = aggregate_seed_runs(seed_runs, config=config)
    phasewise = enrich_phasewise_deltas(aggregated["rows"], config=config)
    summary: Dict[str, Any] = {
        **aggregated,
        "d": int(config.d),
        "T": int(config.t),
        "n_per_time": int(config.n_per_time),
        "phase_boundaries": list(config.phase_boundaries),
        "phase_names": list(config.phases),
        "preset": args.preset,
        "variant": variant,
        "benchmark_rationale": rationale,
        "skip_nf": bool(args.skip_nf),
        "skip_mine": bool(args.skip_mine),
        "include_regularized_dvc": bool(args.include_regularized_dvc),
        "include_nonparametric_dvc": bool(args.include_nonparametric_dvc),
        "np_vine_type": str(args.np_vine_type),
        "np_knots": int(args.np_knots),
        "np_higher_tree_validation_margin": (
            None if args.np_higher_tree_validation_margin is None else float(args.np_higher_tree_validation_margin)
        ),
        "np_higher_tree_boundary_frac_threshold": (
            None if args.np_higher_tree_boundary_frac_threshold is None else float(args.np_higher_tree_boundary_frac_threshold)
        ),
        "phasewise_summary": phasewise,
        "acceptance_flags": phase_acceptance_flags(phasewise),
        "recommendation": (
            "Use `fixed` for a semantically correct replacement of the original benchmark. "
            "Use `contrast` for a pairwise-matched higher-order phase, and "
            "use `contrast_harder` as the default when the goal is to open a much larger "
            "DVC-vs-Gaussian gap."
        ),
    }
    reg_dynamic_summary = _summarize_regularized_dvc(seed_runs)
    if reg_dynamic_summary is not None:
        summary["regularized_dynamic_summary"] = reg_dynamic_summary

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print("\nPhasewise DVC advantage summary:")
    for phase_name, metrics in phasewise.items():
        print(
            f"{phase_name:24s} "
            f"DVC-SSM={metrics['dvc_minus_ssm']:+.3f} "
            f"DVC-Gauss={metrics['dvc_minus_gauss']:+.3f} "
            f"TC_higher={metrics['tc_higher_dvc']:+.3f}"
        )
        if bool(args.include_nonparametric_dvc):
            print(
                f"{'':24s} "
                f"NP-SSM={metrics.get('np_minus_ssm', float('nan')):+.3f} "
                f"NP-Gauss={metrics.get('np_minus_gauss', float('nan')):+.3f} "
                f"NP-DVC={metrics.get('np_minus_dvc', float('nan')):+.3f}"
            )


if __name__ == "__main__":
    main()
