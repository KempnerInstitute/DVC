#!/usr/bin/env python3
"""
Generate and vendor paper assets into drafts/ for standalone Overleaf builds.

This script can:
1) run the benchmark configs (via generate_neurips_tables.py --run)
2) copy tables/figures/result-json assets under drafts/
3) optionally compile the LaTeX draft
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DRAFTS_ROOT = PROJECT_ROOT / "drafts"

TABLES_SRC = RESULTS_ROOT / "neurips_tables"
TABLES_DST = DRAFTS_ROOT / "tables" / "neurips_tables"

FIGURES_DST = DRAFTS_ROOT / "figures" / "neurips_results"
ARTIFACTS_DST = DRAFTS_ROOT / "artifacts" / "results"
MANIFEST_PATH = DRAFTS_ROOT / "assets_manifest.json"


def _run(cmd: List[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tables() -> List[str]:
    copied: List[str] = []
    TABLES_DST.mkdir(parents=True, exist_ok=True)
    for src in sorted(TABLES_SRC.glob("*")):
        if src.suffix.lower() not in {".tex", ".csv"}:
            continue
        dst = TABLES_DST / src.name
        _copy_file(src, dst)
        copied.append(str(dst.relative_to(DRAFTS_ROOT)))
    return copied


def _copy_figures() -> List[str]:
    copied: List[str] = []
    FIGURES_DST.mkdir(parents=True, exist_ok=True)

    # Copy all plot PNGs from experiment outputs and namespace by experiment.
    for src in sorted(RESULTS_ROOT.glob("*/plots/*.png")):
        experiment_name = src.parts[-3]
        dst_name = f"{experiment_name}_{src.name}"
        dst = FIGURES_DST / dst_name
        _copy_file(src, dst)
        copied.append(str(dst.relative_to(DRAFTS_ROOT)))
    return copied


def _copy_result_jsons() -> List[str]:
    copied: List[str] = []
    ARTIFACTS_DST.mkdir(parents=True, exist_ok=True)
    for src in sorted(RESULTS_ROOT.glob("*/*_results.json")):
        experiment_name = src.parts[-2]
        dst = ARTIFACTS_DST / f"{experiment_name}_{src.name}"
        _copy_file(src, dst)
        copied.append(str(dst.relative_to(DRAFTS_ROOT)))
    return copied


def _write_manifest(copied: Dict[str, List[str]]) -> None:
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "drafts_root": str(DRAFTS_ROOT),
        "assets": copied,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _compile_draft(tex: str) -> None:
    _run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex],
        cwd=DRAFTS_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare standalone drafts/ assets (tables, figures, json) for Overleaf."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run benchmark configs before syncing assets.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the draft TeX after syncing assets.",
    )
    parser.add_argument(
        "--tex",
        default="dvc_neurips_2026.tex",
        help="Draft TeX file inside drafts/ to compile.",
    )
    args = parser.parse_args()

    if args.run:
        _run(
            [sys.executable, "scripts/generate_neurips_tables.py", "--run"],
            cwd=PROJECT_ROOT,
        )

    if not TABLES_SRC.exists():
        raise FileNotFoundError(f"Missing tables source directory: {TABLES_SRC}")

    copied_assets = {
        "tables": _copy_tables(),
        "figures": _copy_figures(),
        "result_json": _copy_result_jsons(),
    }
    _write_manifest(copied_assets)

    if args.compile:
        _compile_draft(args.tex)

    print("Draft assets synced.")
    print(f"  tables: {len(copied_assets['tables'])}")
    print(f"  figures: {len(copied_assets['figures'])}")
    print(f"  result_json: {len(copied_assets['result_json'])}")
    print(f"  manifest: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
