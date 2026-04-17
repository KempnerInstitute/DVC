#!/usr/bin/env python3
"""Run the real-world finance crisis benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.experiments.experiment_framework import ExperimentRunner


def _print_summary(results: dict) -> None:
    outdir = Path(results.get("config", {}).get("output_dir", "results"))
    scenarios = results.get("scenarios", {})
    if not isinstance(scenarios, dict):
        return
    for scenario_name, payload in scenarios.items():
        if not isinstance(payload, dict):
            continue
        print(f"\nScenario: {scenario_name}")
        print(f"  assets: {payload.get('assets', [])}")
        print(f"  windows: {len(payload.get('window_end_dates', []))}")
        if "n_windows_skipped" in payload:
            print(f"  skipped windows: {int(payload.get('n_windows_skipped', 0))}")
        if "root_change_count" in payload:
            roots = payload.get("root_unique", [])
            print(f"  root changes: {int(payload.get('root_change_count', 0))} (unique roots: {roots})")
        print("  figures:")
        print(f"    - {outdir / 'plots' / f'{scenario_name}_nll_gap_panel.png'}")
        print(f"    - {outdir / 'plots' / f'{scenario_name}_tail_dependence_panel.png'}")
        print(f"    - {outdir / 'plots' / f'{scenario_name}_family_heatmap.png'}")
        flags = payload.get("outperformance_flags", {})
        if isinstance(flags, dict):
            print("  crisis outperformance flags (mean gap > 0):")
            for baseline, val in flags.items():
                print(f"    - {baseline}: {bool(val)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run finance crisis benchmark config.")
    parser.add_argument(
        "--config",
        default="configs/finance_crisis_benchmarks.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")

    runner = ExperimentRunner()
    results = runner.run_from_config(str(cfg))
    _print_summary(results)
    print(f"\nResults written to: {results.get('config', {}).get('output_dir', 'results')}")


if __name__ == "__main__":
    main()
