#!/usr/bin/env python3
"""Select a small clean Allen VBN pilot cohort from downloaded metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_OUTPUT_ROOT = Path(
    "/n/netscratch/kempner_dev/Lab/hsafaai/results/kempner_project_b/datasets/allen_vbn"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose a clean Allen VBN pilot cohort.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Allen VBN root containing metadata/. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--n-mice",
        type=int,
        default=1,
        help="Number of mice to include in the pilot cohort.",
    )
    parser.add_argument(
        "--manifest-stem",
        type=str,
        default="allen_vbn_pilot",
        help="Prefix used when writing the cohort session-id and JSON manifests.",
    )
    return parser.parse_args()


def _clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("abnormal_histology", "abnormal_activity"):
        if col in out.columns:
            out[col] = out[col].replace({"[]": pd.NA, "": pd.NA})
            out = out[out[col].isna()]
    return out


def select_pairs(df: pd.DataFrame, n_mice: int) -> pd.DataFrame:
    clean = _clean_sessions(df)
    keep: List[pd.DataFrame] = []
    scored: List[Dict[str, object]] = []

    for mouse_id, group in clean.groupby("mouse_id"):
        levels = set(group["experience_level"].dropna().astype(str))
        if not {"Familiar", "Novel"}.issubset(levels):
            continue

        familiar = group[group["experience_level"] == "Familiar"].sort_values("unit_count", ascending=False).head(1)
        novel = group[group["experience_level"] == "Novel"].sort_values("unit_count", ascending=False).head(1)
        if familiar.empty or novel.empty:
            continue

        chosen = pd.concat([familiar, novel], ignore_index=True)
        total_units = float(chosen["unit_count"].fillna(0).sum())
        scored.append({"mouse_id": int(mouse_id), "total_units": total_units, "rows": chosen})

    scored = sorted(scored, key=lambda x: x["total_units"], reverse=True)[:n_mice]
    for item in scored:
        keep.append(item["rows"])

    if not keep:
        raise RuntimeError("No clean Familiar/Novel mouse pairs found in the metadata.")
    return pd.concat(keep, ignore_index=True)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    meta_path = output_root / "metadata" / "ecephys_sessions.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    df = pd.read_csv(meta_path)
    cohort = select_pairs(df, n_mice=args.n_mice)
    cohort = cohort.sort_values(["mouse_id", "session_number", "experience_level"]).reset_index(drop=True)

    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    stem = str(args.manifest_stem).strip() or "allen_vbn_pilot"
    text_path = manifest_root / f"{stem}_session_ids.txt"
    json_path = manifest_root / f"{stem}_sessions.json"

    session_ids = [int(x) for x in cohort["ecephys_session_id"].tolist()]
    text_path.write_text("\n".join(str(x) for x in session_ids) + "\n")
    json_path.write_text(
        json.dumps(
            {
                "session_ids": session_ids,
                "rows": cohort.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )

    print(f"Selected {len(session_ids)} sessions across {args.n_mice} mice")
    print(f"Session IDs file: {text_path}")
    print(f"Session manifest: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
