#!/usr/bin/env python3
"""Run a matched-session Allen VBN cohort analysis with windowed DVC."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.real_data import load_allen_vbn_session
from dvc_package.real_data.windowed_analysis import (
    analyze_session_windows,
    cohort_summary_table,
    plot_cohort_summary,
    result_to_dict,
    summarize_cohort_results,
)


DEFAULT_DATA_ROOT = Path(
    "/n/netscratch/kempner_dev/Lab/hsafaai/results/kempner_project_b/datasets/allen_vbn"
)
DEFAULT_MANIFEST_STEM = "allen_vbn_cohort"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "analysis" / "cohort"
DEFAULT_REPO_RESULTS = PROJECT_ROOT / "results" / "allen_vbn_cohort"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Allen VBN matched-session cohort with windowed DVC.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest-stem", type=str, default=DEFAULT_MANIFEST_STEM)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-results-root", type=Path, default=DEFAULT_REPO_RESULTS)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--top-k-regions", type=int, default=5)
    parser.add_argument("--min-units-per-region", type=int, default=20)
    parser.add_argument("--min-presence-ratio", type=float, default=0.95)
    parser.add_argument("--min-firing-rate", type=float, default=0.1)
    return parser.parse_args()


def _session_paths(data_root: Path, manifest_stem: str) -> List[Path]:
    manifest_root = data_root / "manifests"
    manifest_path = manifest_root / f"{manifest_stem}_session_ids.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing session file: {manifest_path}")
    out: List[Path] = []
    for line in manifest_path.read_text().splitlines():
        token = line.strip()
        if not token:
            continue
        session_id = int(token)
        path = data_root / "sessions" / str(session_id) / f"ecephys_session_{session_id}.nwb"
        if not path.exists():
            raise FileNotFoundError(f"Missing session file: {path}")
        out.append(path)
    return out


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    repo_results = args.repo_results_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    repo_results.mkdir(parents=True, exist_ok=True)

    manifest_stem = str(args.manifest_stem).strip() or DEFAULT_MANIFEST_STEM
    manifest_json = data_root / "manifests" / f"{manifest_stem}_sessions.json"

    results = []
    for session_path in _session_paths(data_root, manifest_stem):
        session = load_allen_vbn_session(
            session_path,
            manifest_path=manifest_json,
            top_k_regions=int(args.top_k_regions),
            min_units_per_region=int(args.min_units_per_region),
            min_presence_ratio=float(args.min_presence_ratio),
            min_firing_rate=float(args.min_firing_rate),
        )
        result = analyze_session_windows(
            session,
            window_size=int(args.window_size),
            stride=int(args.stride),
        )
        results.append(result)

    results = sorted(results, key=lambda r: (str(r.mouse_id), str(r.experience_level), int(r.session_id)))
    session_payload = {"sessions": [result_to_dict(r) for r in results]}
    summary_payload = summarize_cohort_results(results)
    summary_table = cohort_summary_table(results)

    json_path = output_root / "allen_vbn_cohort_results.json"
    json_path.write_text(json.dumps(session_payload, indent=2))

    summary_json_path = output_root / "allen_vbn_cohort_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2))

    summary_csv_path = output_root / "allen_vbn_cohort_summary.csv"
    summary_table.to_csv(summary_csv_path, index=False)

    fig_path = output_root / "allen_vbn_cohort_summary.png"
    plot_cohort_summary(results, out_path=str(fig_path))

    for path in [json_path, summary_json_path, summary_csv_path, fig_path]:
        shutil.copy2(path, repo_results / path.name)

    print(f"Saved cohort session JSON: {json_path}")
    print(f"Saved cohort summary JSON: {summary_json_path}")
    print(f"Saved cohort summary CSV: {summary_csv_path}")
    print(f"Saved cohort figure: {fig_path}")
    print(f"Mirrored results into: {repo_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
