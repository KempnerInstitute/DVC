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
    gaussian_mi_from_tau,
    generate_sequence,
    showcase_truth_by_phase,
    split_train_test,
)


DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "results"
    / "showcase_ale_final"
    / "proper_sota_nf_mine_1seed"
    / "summary.json"
)


def _dvc_sample_pair_mi(
    vine: Any,
    pair: tuple[int, int],
    *,
    n_samples: int,
    seed: int,
) -> float:
    """Estimate marginal pairwise MI from samples drawn from a fitted DVC vine."""
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception:
        mutual_info_regression = None

    np_state = np.random.get_state()
    torch_state = None
    torch_mod = None
    try:
        try:
            import torch as torch_mod  # type: ignore[no-redef]

            torch_state = torch_mod.random.get_rng_state()
            torch_mod.manual_seed(int(seed))
        except Exception:
            torch_mod = None
        np.random.seed(int(seed))
        samples = vine.sample(int(n_samples)) if hasattr(vine, "sample") else None
    finally:
        np.random.set_state(np_state)
        if torch_mod is not None and torch_state is not None:
            try:
                torch_mod.random.set_rng_state(torch_state)
            except Exception:
                pass

    if samples is None:
        return float("nan")
    try:
        import torch

        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] <= max(pair) or x.shape[0] < 20:
        return float("nan")
    xi = x[:, int(pair[0])]
    xj = x[:, int(pair[1])]
    good = np.isfinite(xi) & np.isfinite(xj)
    xi = xi[good]
    xj = xj[good]
    if xi.size < 20 or np.std(xi) < 1e-12 or np.std(xj) < 1e-12:
        return 0.0

    if mutual_info_regression is not None:
        try:
            mi = mutual_info_regression(
                xi.reshape(-1, 1),
                xj,
                n_neighbors=5,
                random_state=int(seed),
            )[0]
            if np.isfinite(mi):
                return float(max(mi, 0.0))
        except Exception:
            pass

    try:
        import pandas as pd

        tau = float(pd.Series(xi).corr(pd.Series(xj), method="kendall"))
        return gaussian_mi_from_tau(tau)
    except Exception:
        return float("nan")


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
    all_trunc_nll: list[np.ndarray] = []
    all_pair_mi01: list[np.ndarray] = []
    all_pair_mi56: list[np.ndarray] = []
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
        all_trunc_nll.append(np.asarray(result.evaluate_truncated_level0(x_test_by_t), dtype=np.float64))
        mi01 = []
        mi56 = []
        for t_idx, vine in enumerate(result.vines_by_time):
            mi01.append(
                _dvc_sample_pair_mi(
                    vine,
                    (0, 1),
                    n_samples=1500,
                    seed=900_000 + 101 * int(seed) + int(t_idx),
                )
            )
            mi56.append(
                _dvc_sample_pair_mi(
                    vine,
                    (5, 6),
                    n_samples=1500,
                    seed=910_000 + 101 * int(seed) + int(t_idx),
                )
            )
        all_pair_mi01.append(np.asarray(mi01, dtype=np.float64))
        all_pair_mi56.append(np.asarray(mi56, dtype=np.float64))
        family_switches.append(int(result.total_family_switches()))
        parameter_drifts.append(float(result.total_parameter_drift()))

    nll_stack = np.vstack(all_nll)
    nll_mean = np.nanmean(nll_stack, axis=0)
    nll_std = np.nanstd(nll_stack, axis=0)
    trunc_stack = np.vstack(all_trunc_nll)
    trunc_mean = np.nanmean(trunc_stack, axis=0)
    trunc_std = np.nanstd(trunc_stack, axis=0)
    # tc_higher = NLL_trunc - NLL_full evaluated on the same fit/splits per seed.
    # Compute the per-seed difference *before* taking std to capture the
    # (typically positive) covariance between the two terms; combining marginal
    # stds via sqrt(var_a + var_b) would assume independence and overestimate.
    tc_higher_stack = trunc_stack - nll_stack
    tc_higher_std_per_t = np.nanstd(tc_higher_stack, axis=0)
    mi01_stack = np.vstack(all_pair_mi01)
    mi56_stack = np.vstack(all_pair_mi56)
    mi01_mean = np.nanmean(mi01_stack, axis=0)
    mi01_std = np.nanstd(mi01_stack, axis=0)
    mi56_mean = np.nanmean(mi56_stack, axis=0)
    mi56_std = np.nanstd(mi56_stack, axis=0)

    rows = summary.get("rows", [])
    if len(rows) != config.t:
        raise ValueError(f"Expected {config.t} rows, found {len(rows)}")
    truth_by_phase = showcase_truth_by_phase(config, variant)
    for t_idx, row in enumerate(rows):
        row.update(truth_by_phase.get(str(row.get("phase_name", "")), {}))
        nll = float(nll_mean[t_idx])
        std = float(nll_std[t_idx])
        trunc_nll = float(trunc_mean[t_idx])
        trunc_std_t = float(trunc_std[t_idx])
        tc_total = float(-nll)
        tc_pair = float(-trunc_nll)
        row["nll_switching_dvc"] = nll
        row["nll_switching_dvc_std"] = std
        row["nll_trunc_switching_dvc"] = trunc_nll
        row["nll_trunc_switching_dvc_std"] = trunc_std_t
        row["tc_total_switching_dvc"] = tc_total
        row["tc_total_switching_dvc_std"] = std
        row["tc_pair_switching_dvc"] = tc_pair
        row["tc_pair_switching_dvc_std"] = trunc_std_t
        row["tc_higher_switching_dvc"] = float(trunc_nll - nll) if np.isfinite(tc_pair) else float("nan")
        row["tc_higher_switching_dvc_std"] = float(tc_higher_std_per_t[t_idx])
        row["nll_gap_windowed_vine_vs_switching_dvc"] = float(row.get("nll_dvc", np.nan) - nll)
        if "tau_gauss_pair_mi01" not in row and "dvc_pair_mi01" in row:
            row["tau_gauss_pair_mi01"] = float(row.get("dvc_pair_mi01", np.nan))
        if "tau_gauss_pair_mi56" not in row and "dvc_pair_mi56" in row:
            row["tau_gauss_pair_mi56"] = float(row.get("dvc_pair_mi56", np.nan))
        row["dvc_switch_pair_mi01"] = float(mi01_mean[t_idx])
        row["dvc_switch_pair_mi01_std"] = float(mi01_std[t_idx])
        row["dvc_switch_pair_mi56"] = float(mi56_mean[t_idx])
        row["dvc_switch_pair_mi56_std"] = float(mi56_std[t_idx])

    phasewise = summary.get("phasewise_summary", {})
    for phase_name in config.phases:
        phase_rows = [row for row in rows if str(row.get("phase_name")) == str(phase_name)]
        if not phase_rows:
            continue
        payload = dict(phasewise.get(phase_name, {})) if isinstance(phasewise.get(phase_name, {}), dict) else {}

        def mean_field(field: str) -> float:
            vals = np.asarray([float(row.get(field, np.nan)) for row in phase_rows], dtype=np.float64)
            return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

        payload.update(
            {
                "tc_total_switching_dvc": mean_field("tc_total_switching_dvc"),
                "tc_pair_switching_dvc": mean_field("tc_pair_switching_dvc"),
                "tc_higher_switching_dvc": mean_field("tc_higher_switching_dvc"),
                "switching_dvc_minus_windowed": mean_field("tc_total_switching_dvc") - mean_field("tc_total_dvc"),
                "switching_dvc_minus_gauss": mean_field("tc_total_switching_dvc") - mean_field("tc_gauss"),
                "switching_dvc_minus_ssm": mean_field("tc_total_switching_dvc") - mean_field("tc_total_ssm"),
            }
        )
        phasewise[phase_name] = payload
    summary["phasewise_summary"] = phasewise

    summary["include_switching_dvc"] = True
    summary["switching_dvc_summary"] = {
        "n_runs": len(seeds),
        "mean_total_family_switches": float(np.mean(family_switches)),
        "mean_total_parameter_drift": float(np.mean(parameter_drifts)),
        "pair_mi_estimator": "sampled fitted DVC-switch vine + sklearn mutual_info_regression",
        "pair_mi_n_samples": 1500,
        "seeds": seeds,
        "truncation_note": "tc_pair_switching_dvc and tc_higher_switching_dvc are evaluated from the nested 1-truncated version of the fitted joint switching DVC, not from the windowed control.",
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
