#!/usr/bin/env python3
"""Run the real-world finance crisis benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.experiments.real_world import run_finance_crisis_benchmark_suite


def _print_summary(results: dict, output_dir: Path) -> None:
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
        print(f"    - {output_dir / 'plots' / f'{scenario_name}_nll_gap_panel.png'}")
        print(f"    - {output_dir / 'plots' / f'{scenario_name}_tail_dependence_panel.png'}")
        print(f"    - {output_dir / 'plots' / f'{scenario_name}_family_heatmap.png'}")
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

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    output_dir = Path(cfg.get("output_dir", "results/finance_crisis_benchmarks"))
    seed = int(cfg.get("seed", 2026))
    scenarios = cfg.get("analysis_config", {}).get("scenarios", [])

    results = run_finance_crisis_benchmark_suite(
        output_dir=output_dir,
        seed=seed,
        scenarios=scenarios,
    )

    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    _print_summary(results, output_dir)
    print(f"\nResults written to: {output_dir}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
