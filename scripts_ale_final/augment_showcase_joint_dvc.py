#!/usr/bin/env python3
"""Add joint switching-DVC fields to an existing showcase summary.

The original showcase benchmark stores the independent windowed full-vine
control under the historical ``*_dvc`` keys.  This augmentation fits the real
joint switching-state DVC on the same generated windows and adds explicit
``*_switching_dvc`` fields, leaving the windowed control untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.experiments.simulation_benchmarks import _fit_switching_dynamic_cvine_from_splits  # noqa: E402
from scripts_ale_final.showcase_analysis_utils import (  # noqa: E402
    FAMILIES,
    ShowcaseConfig,
    generate_sequence,
    split_train_test,
)


DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "showcase_ale_final"
    / "proper_sota_nf_mine_1seed"
    / "summary.json"
)


def _config_from_summary(summary: dict[str, Any]) -> ShowcaseConfig:
    payload = summary.get("config", {})
    return ShowcaseConfig(
        d=int(payload.get("d", summary.get("d", 10))),
        t=int(payload.get("t", summary.get("T", 60))),
        n_per_time=int(payload.get("n_per_time", summary.get("n_per_time", 300))),
        train_frac=float(payload.get("train_frac", summary.get("train_frac", 0.85))),
        phase_boundaries=tuple(payload.get("phase_boundaries", summary.get("phase_boundaries", [0, 15, 30, 45, 60]))),
        phases=tuple(payload.get("phases", summary.get("phase_names", []))),
        pair_root=int(payload.get("pair_root", 0)),
        pair_leaves=tuple(payload.get("pair_leaves", [1, 2, 3])),
        pair_rho=float(payload.get("pair_rho", 0.55)),
        phase3_mode=str(payload.get("phase3_mode", summary.get("variant", "multiplicative_triplets"))),
        triplet_blocks=tuple(tuple(block) for block in payload.get("triplet_blocks", [[4, 5, 6], [7, 8, 9]])),
        triplet_rho=float(payload.get("triplet_rho", 0.65)),
        triplet_nu=float(payload.get("triplet_nu", 4.5)),
        triplet_clayton_theta=float(payload.get("triplet_clayton_theta", 2.0)),
        multiplicative_noise_std=float(payload.get("multiplicative_noise_std", 0.10)),
        xor_jitter_std=float(payload.get("xor_jitter_std", 1e-3)),
        tail_block=tuple(payload.get("tail_block", [0, 1, 2, 3])),
        tail_theta=float(payload.get("tail_theta", 3.5)),
    )


def augment(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    config = _config_from_summary(summary)
    variant = str(summary.get("variant", config.phase3_mode))
    seeds = [int(seed) for seed in summary.get("seeds", [])]
    if not seeds:
        raise ValueError(f"No seeds found in {summary_path}")

    all_nll: list[np.ndarray] = []
    family_switches: list[int] = []
    parameter_drifts: list[float] = []
    for seed in seeds:
        print(f"Fitting joint switching DVC for showcase seed {seed} ...", flush=True)
        windows = generate_sequence(seed=seed, config=config, variant=variant)
        x_train_by_t, x_test_by_t = split_train_test(windows, config.train_frac)
        result, nll = _fit_switching_dynamic_cvine_from_splits(
            x_train_by_t,
            x_test_by_t,
            families=FAMILIES,
            order=list(range(config.d)),
            family_switch_penalty=0.08,
            parameter_drift_penalty=0.02,
            activation_penalty=0.0,
        )
        all_nll.append(np.asarray(nll, dtype=np.float64))
        family_switches.append(int(result.total_family_switches()))
        parameter_drifts.append(float(result.total_parameter_drift()))

    nll_stack = np.vstack(all_nll)
    nll_mean = np.nanmean(nll_stack, axis=0)
    nll_std = np.nanstd(nll_stack, axis=0)

    rows = summary.get("rows", [])
    if len(rows) != config.t:
        raise ValueError(f"Expected {config.t} rows, found {len(rows)}")
    for t_idx, row in enumerate(rows):
        nll = float(nll_mean[t_idx])
        std = float(nll_std[t_idx])
        tc_total = float(-nll)
        tc_pair = float(row.get("tc_pair_dvc", np.nan))
        row["nll_switching_dvc"] = nll
        row["nll_switching_dvc_std"] = std
        row["tc_total_switching_dvc"] = tc_total
        row["tc_total_switching_dvc_std"] = std
        row["tc_pair_switching_dvc"] = tc_pair
        row["tc_higher_switching_dvc"] = float(tc_total - tc_pair) if np.isfinite(tc_pair) else float("nan")
        row["nll_gap_windowed_vine_vs_switching_dvc"] = float(row.get("nll_dvc", np.nan) - nll)

    summary["include_switching_dvc"] = True
    summary["switching_dvc_summary"] = {
        "n_runs": len(seeds),
        "mean_total_family_switches": float(np.mean(family_switches)),
        "mean_total_parameter_drift": float(np.mean(parameter_drifts)),
        "seeds": seeds,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Updated {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    augment(args.summary)


if __name__ == "__main__":
    main()
