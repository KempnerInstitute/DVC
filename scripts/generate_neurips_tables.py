#!/usr/bin/env python3
"""
Run DVC experiment configs and generate NeurIPS-ready result tables.

This script can:
1) execute one or more YAML configs via ExperimentRunner
2) parse generated JSON result files
3) emit CSV + LaTeX tables for paper writing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.experiments.experiment_framework import ExperimentRunner


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _result_json_path(config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    output_dir = Path(cfg["output_dir"])
    run_name = cfg["name"]
    return output_dir / f"{run_name}_results.json"


def _run_config(config_path: Path) -> Dict[str, Any]:
    runner = ExperimentRunner()
    return runner.run_from_config(str(config_path))


def _load_result_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _summarize_probability(result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_name = result.get("config", {}).get("name", "unknown")
    data_info = result.get("data_info", {})
    dep = result.get("probability_analysis", {}).get("dependence_analysis", {})
    vine_analysis = result.get("vine_analysis", {})

    rows = []
    for vine_type, payload in vine_analysis.items():
        corr_err = (
            payload.get("correlation_preservation", {}).get("error")
            if isinstance(payload, dict)
            else None
        )
        rows.append(
            {
                "experiment_name": run_name,
                "vine_type": vine_type,
                "fit_success": bool(payload.get("fit_success", False)),
                "n_edges": payload.get("n_edges"),
                "corr_preservation_error": _safe_float(corr_err),
                "n_samples": data_info.get("n_samples"),
                "n_variables": data_info.get("n_variables"),
                "mean_abs_correlation": _safe_float(dep.get("mean_abs_correlation")),
            }
        )

    detail_df = pd.DataFrame(rows)

    best_row = None
    valid = detail_df.dropna(subset=["corr_preservation_error"])
    if not valid.empty:
        best_row = valid.loc[valid["corr_preservation_error"].idxmin()]

    summary_df = pd.DataFrame(
        [
            {
                "experiment_name": run_name,
                "experiment_type": "probability_analysis",
                "n_samples": data_info.get("n_samples"),
                "n_variables": data_info.get("n_variables"),
                "best_vine_type": None if best_row is None else best_row["vine_type"],
                "best_corr_preservation_error": (
                    None if best_row is None else float(best_row["corr_preservation_error"])
                ),
                "mean_abs_correlation": _safe_float(dep.get("mean_abs_correlation")),
            }
        ]
    )
    return detail_df, summary_df


def _summarize_entropy(result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_name = result.get("config", {}).get("name", "unknown")
    methods = result.get("method_comparison", {})

    rows = []
    for method_name, datasets in methods.items():
        for dataset_name, value in datasets.items():
            rows.append(
                {
                    "experiment_name": run_name,
                    "method": method_name,
                    "dataset": dataset_name,
                    "entropy_estimate": _safe_float(value),
                }
            )

    detail_df = pd.DataFrame(rows)

    kernel_df = detail_df[detail_df["method"] == "kernel_entropy"]
    kernel_mean = None if kernel_df.empty else float(kernel_df["entropy_estimate"].mean())

    summary_df = pd.DataFrame(
        [
            {
                "experiment_name": run_name,
                "experiment_type": "entropy_analysis",
                "n_datasets": int(detail_df["dataset"].nunique()) if not detail_df.empty else 0,
                "n_methods": int(detail_df["method"].nunique()) if not detail_df.empty else 0,
                "kernel_entropy_mean": kernel_mean,
            }
        ]
    )
    return detail_df, summary_df


def _summarize_time_dependent(result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_name = result.get("config", {}).get("name", "unknown")
    data_info = result.get("data_info", {})
    model = result.get("model_results", {})
    corr_stats = result.get("correlation_analysis", {}).get("correlation_statistics", {})

    pair_rows = []
    corr_std_values = []
    for pair_name, pair_payload in corr_stats.items():
        corr_payload = pair_payload.get("correlation", {})
        row = {
            "experiment_name": run_name,
            "pair": pair_name,
            "corr_mean": _safe_float(corr_payload.get("mean")),
            "corr_std": _safe_float(corr_payload.get("std")),
            "corr_min": _safe_float(corr_payload.get("min")),
            "corr_max": _safe_float(corr_payload.get("max")),
            "corr_trend": _safe_float(corr_payload.get("trend")),
        }
        if row["corr_std"] is not None:
            corr_std_values.append(row["corr_std"])
        pair_rows.append(row)

    detail_df = pd.DataFrame(pair_rows)
    avg_corr_std = None if not corr_std_values else float(sum(corr_std_values) / len(corr_std_values))

    summary_df = pd.DataFrame(
        [
            {
                "experiment_name": run_name,
                "experiment_type": "time_dependent",
                "n_time_steps": data_info.get("n_time_steps"),
                "n_edges": model.get("n_edges"),
                "model_created": bool(model.get("model_created", False)),
                "bandwidth_mean": _safe_float(model.get("bandwidth_statistics", {}).get("mean")),
                "bandwidth_std": _safe_float(model.get("bandwidth_statistics", {}).get("std")),
                "avg_pair_corr_std": avg_corr_std,
            }
        ]
    )
    return detail_df, summary_df


def _summarize_neurips_simulations(result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_name = result.get("config", {}).get("name", "unknown")
    rows = result.get("summary_table", [])

    detail_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        detail_rows.append({"experiment_name": run_name, **row})
    detail_df = pd.DataFrame(detail_rows)

    # Minimal one-row summary.
    n_scenarios = int(detail_df["scenario"].nunique()) if not detail_df.empty and "scenario" in detail_df else 0
    mean_gap = None
    if not detail_df.empty and "nll_gap_mean" in detail_df:
        mean_gap = float(detail_df["nll_gap_mean"].mean())

    root_acc = None
    if not detail_df.empty and "root_recovery_accuracy" in detail_df:
        root_acc = float(detail_df["root_recovery_accuracy"].mean())

    summary_df = pd.DataFrame(
        [
            {
                "experiment_name": run_name,
                "experiment_type": "neurips_simulations",
                "n_scenarios": n_scenarios,
                "mean_nll_gap_mean": mean_gap,
                "mean_root_recovery_accuracy": root_acc,
            }
        ]
    )
    return detail_df, summary_df


def _as_1d_float_array(value: Any) -> Optional[np.ndarray]:
    """Convert scalar/list-like value to a finite 1D float array."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        arr = np.asarray([value], dtype=np.float64)
    else:
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            return None
    if arr.size == 0:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return arr


def _extract_effective_behavior_rows(
    result: Dict[str, Any],
    run_name: str,
) -> pd.DataFrame:
    """Build per-scenario/per-baseline effective-behavior metrics from time-series NLL gaps."""
    scenarios = result.get("scenarios", {})
    if not isinstance(scenarios, dict):
        return pd.DataFrame()

    gap_fields = {
        "nll_gap": "Gaussian copula",
        "nll_gap_truncated_level0": "1-truncated C-vine",
        "nll_gap_glasso": "Graphical Lasso",
        "nll_gap_tvgl": "TVGL (Frobenius)",
        "nll_gap_kde_flow": "KDE-flow (time BW)",
    }

    rows: List[Dict[str, Any]] = []
    for scenario, payload in scenarios.items():
        if not isinstance(payload, dict):
            continue
        cp = payload.get("change_point")
        cp_idx = None
        try:
            cp_idx = int(cp)
        except Exception:
            cp_idx = None

        for field, baseline in gap_fields.items():
            series = _as_1d_float_array(payload.get(field))
            if series is None:
                continue
            n = int(series.size)
            pos_frac = float(np.mean(series > 0.0))
            row = {
                "experiment_name": run_name,
                "scenario": str(scenario),
                "baseline": baseline,
                "gap_key": field,
                "n_timepoints_or_trials": n,
                "mean_gap": float(np.mean(series)),
                "std_gap": float(np.std(series)),
                "median_gap": float(np.median(series)),
                "min_gap": float(np.min(series)),
                "max_gap": float(np.max(series)),
                "positive_fraction": pos_frac,
            }
            if cp_idx is not None and n > 1:
                cp_clip = int(np.clip(cp_idx, 1, n - 1))
                pre = series[:cp_clip]
                post = series[cp_clip:]
                row["pre_change_mean_gap"] = float(np.mean(pre))
                row["post_change_mean_gap"] = float(np.mean(post))
                row["post_minus_pre_gap"] = float(np.mean(post) - np.mean(pre))
                row["change_point"] = cp_clip
            rows.append(row)

    return pd.DataFrame(rows)


def _plot_effective_behavior_heatmaps(df: pd.DataFrame, out_png: Path) -> None:
    """Plot scenario x baseline heatmaps for mean NLL gap and positive-time fraction."""
    if df.empty:
        return
    required = {"scenario", "baseline", "mean_gap", "positive_fraction"}
    if not required.issubset(set(df.columns)):
        return

    mean_piv = df.pivot_table(
        index="scenario",
        columns="baseline",
        values="mean_gap",
        aggfunc="mean",
    )
    pos_piv = df.pivot_table(
        index="scenario",
        columns="baseline",
        values="positive_fraction",
        aggfunc="mean",
    )

    if mean_piv.empty or pos_piv.empty:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    sns.heatmap(
        mean_piv,
        ax=axes[0],
        cmap="RdYlGn",
        center=0.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "NLL gap (baseline - DVC)"},
    )
    axes[0].set_title("Mean held-out copula NLL gap")
    axes[0].set_xlabel("Baseline")
    axes[0].set_ylabel("Scenario")

    sns.heatmap(
        pos_piv,
        ax=axes[1],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Fraction(gap > 0)"},
    )
    axes[1].set_title("Positive-gap fraction")
    axes[1].set_xlabel("Baseline")
    axes[1].set_ylabel("Scenario")

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _write_table(df: pd.DataFrame, csv_path: Path, tex_path: Path):
    df.to_csv(csv_path, index=False)
    tex = df.to_latex(
        index=False,
        escape=True,
        na_rep="",
        float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "",
    )
    tex_path.write_text(tex)


def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS result tables from DVC configs/results.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/probability_analysis.yaml",
            "configs/entropy_analysis.yaml",
            "configs/time_dependent.yaml",
            "configs/neurips_simulations.yaml",
        ],
        help="YAML configs to include.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run configs before parsing result JSON files.",
    )
    parser.add_argument(
        "--outdir",
        default="results/neurips_tables",
        help="Output directory for tables.",
    )
    args = parser.parse_args()

    config_paths = [Path(p) for p in args.configs]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result_jsons: List[Path] = []
    for config_path in config_paths:
        if args.run:
            _run_config(config_path)
        result_json = _result_json_path(config_path)
        if not result_json.exists():
            raise FileNotFoundError(f"Expected result file not found: {result_json}")
        result_jsons.append(result_json)

    prob_details = []
    entropy_details = []
    time_details = []
    sim_details = []
    sim_effective_details = []
    summary_rows = []

    for result_json in result_jsons:
        result = _load_result_json(result_json)
        experiment_type = result.get("experiment_type")

        if experiment_type == "probability_analysis":
            detail_df, summary_df = _summarize_probability(result)
            prob_details.append(detail_df)
            summary_rows.append(summary_df)
        elif experiment_type == "entropy_analysis":
            detail_df, summary_df = _summarize_entropy(result)
            entropy_details.append(detail_df)
            summary_rows.append(summary_df)
        elif experiment_type == "time_dependent":
            detail_df, summary_df = _summarize_time_dependent(result)
            time_details.append(detail_df)
            summary_rows.append(summary_df)
        elif experiment_type == "neurips_simulations":
            detail_df, summary_df = _summarize_neurips_simulations(result)
            sim_details.append(detail_df)
            summary_rows.append(summary_df)
            run_name = result.get("config", {}).get("name", "unknown")
            eff_df = _extract_effective_behavior_rows(result, run_name=run_name)
            if not eff_df.empty:
                sim_effective_details.append(eff_df)
                # Keep the figure with simulation plots so `prepare_draft_assets.py` auto-vendors it.
                plots_dir = result_json.parent / "plots"
                _plot_effective_behavior_heatmaps(
                    eff_df,
                    plots_dir / "effective_behavior_heatmap.png",
                )

    if prob_details:
        prob_df = pd.concat(prob_details, ignore_index=True)
        _write_table(
            prob_df,
            outdir / "probability_vine_detail.csv",
            outdir / "probability_vine_detail.tex",
        )

    if entropy_details:
        entropy_df = pd.concat(entropy_details, ignore_index=True)
        _write_table(
            entropy_df,
            outdir / "entropy_method_detail.csv",
            outdir / "entropy_method_detail.tex",
        )

    if time_details:
        time_df = pd.concat(time_details, ignore_index=True)
        _write_table(
            time_df,
            outdir / "time_pair_detail.csv",
            outdir / "time_pair_detail.tex",
        )

    if sim_details:
        sim_df = pd.concat(sim_details, ignore_index=True)
        _write_table(
            sim_df,
            outdir / "neurips_simulation_detail.csv",
            outdir / "neurips_simulation_detail.tex",
        )

    if sim_effective_details:
        eff_df = pd.concat(sim_effective_details, ignore_index=True)
        _write_table(
            eff_df,
            outdir / "neurips_effective_behavior_detail.csv",
            outdir / "neurips_effective_behavior_detail.tex",
        )
        eff_summary_df = (
            eff_df.groupby(["baseline"], as_index=False)
            .agg(
                mean_gap=("mean_gap", "mean"),
                mean_positive_fraction=("positive_fraction", "mean"),
                scenarios_covered=("scenario", "nunique"),
            )
            .sort_values("mean_gap", ascending=False)
            .reset_index(drop=True)
        )
        _write_table(
            eff_summary_df,
            outdir / "neurips_effective_behavior_summary.csv",
            outdir / "neurips_effective_behavior_summary.tex",
        )

    if summary_rows:
        summary_df = pd.concat(summary_rows, ignore_index=True)
        _write_table(
            summary_df,
            outdir / "master_summary.csv",
            outdir / "master_summary.tex",
        )

    print(f"Wrote NeurIPS tables to: {outdir}")


if __name__ == "__main__":
    main()
