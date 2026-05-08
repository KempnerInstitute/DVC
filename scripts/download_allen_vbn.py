#!/usr/bin/env python3
"""Download Allen Visual Behavior Neuropixels metadata and selected sessions.

This uses the public Allen Institute S3 bucket directly, so the initial
download path does not require AllenSDK. Session file locations are discovered
from the public S3 listing rather than hard-coded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List


BUCKET_ROOT = "https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com"
PROJECT_PREFIX = "visual-behavior-neuropixels"
SESSION_PREFIX = f"{PROJECT_PREFIX}/behavior_ecephys_sessions"
METADATA_PREFIX = f"{PROJECT_PREFIX}/project_metadata"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("DVC_DATA_DIR", "data/allen_vbn")).expanduser()
METADATA_FILES = (
    "behavior_sessions.csv",
    "channels.csv",
    "ecephys_sessions.csv",
    "probes.csv",
    "units.csv",
)
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _download_file(url: str, dst: Path, overwrite: bool = False) -> Dict[str, object]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return {"path": str(dst), "url": url, "status": "skipped", "bytes": dst.stat().st_size}

    with urllib.request.urlopen(url) as resp:
        total = resp.headers.get("Content-Length")
        size = int(total) if total is not None else None
        with dst.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    return {"path": str(dst), "url": url, "status": "downloaded", "bytes": dst.stat().st_size}


def _list_s3_objects(prefix: str) -> List[Dict[str, object]]:
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    url = f"{BUCKET_ROOT}/?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url) as resp:
        root = ET.fromstring(resp.read())

    objects: List[Dict[str, object]] = []
    for node in root.findall("s3:Contents", XML_NS):
        key = node.findtext("s3:Key", default="", namespaces=XML_NS)
        size_text = node.findtext("s3:Size", default="0", namespaces=XML_NS)
        objects.append({"key": key, "size": int(size_text)})
    return objects


def download_metadata(output_root: Path, overwrite: bool = False) -> List[Dict[str, object]]:
    metadata_root = output_root / "metadata"
    records: List[Dict[str, object]] = []
    for name in METADATA_FILES:
        url = f"{BUCKET_ROOT}/{METADATA_PREFIX}/{name}"
        dst = metadata_root / name
        records.append(_download_file(url, dst, overwrite=overwrite))
    return records


def _resolve_session_ids(ids: Iterable[int] | None, session_file: Path | None) -> List[int]:
    out: List[int] = []
    if ids:
        out.extend(int(x) for x in ids)
    if session_file is not None:
        for line in session_file.read_text().splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            out.append(int(token))
    return sorted(dict.fromkeys(out))


def download_sessions(
    session_ids: Iterable[int],
    output_root: Path,
    overwrite: bool = False,
    include_lfp: bool = False,
) -> List[Dict[str, object]]:
    session_root = output_root / "sessions"
    records: List[Dict[str, object]] = []

    for session_id in session_ids:
        prefix = f"{SESSION_PREFIX}/{session_id}/"
        objects = _list_s3_objects(prefix)
        if not objects:
            raise FileNotFoundError(f"No S3 objects found for session {session_id} under prefix {prefix}")

        wanted = []
        for obj in objects:
            key = str(obj["key"])
            if key.endswith(f"ecephys_session_{session_id}.nwb"):
                wanted.append(obj)
            elif include_lfp and key.endswith(".nwb"):
                wanted.append(obj)

        if not wanted:
            raise FileNotFoundError(f"No session NWB found for session {session_id}")

        for obj in wanted:
            key = str(obj["key"])
            rel_name = Path(key).name
            url = f"{BUCKET_ROOT}/{urllib.parse.quote(key, safe='/')}"
            dst = session_root / str(session_id) / rel_name
            rec = _download_file(url, dst, overwrite=overwrite)
            rec.update({"session_id": int(session_id), "s3_key": key, "size_bytes": int(obj["size"])})
            records.append(rec)
    return records


def _write_manifest(output_root: Path, payload: Dict[str, object]) -> Path:
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    path = manifest_root / "download_manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Allen VBN metadata and selected sessions.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only the metadata CSV files.",
    )
    parser.add_argument(
        "--session-id",
        action="append",
        type=int,
        help="Session ID to download. May be repeated.",
    )
    parser.add_argument(
        "--session-file",
        type=Path,
        help="Plain-text file with one session ID per line.",
    )
    parser.add_argument(
        "--include-lfp",
        action="store_true",
        help="Also download per-probe LFP NWB files for each selected session.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_records = download_metadata(output_root, overwrite=args.overwrite)
    session_ids = _resolve_session_ids(args.session_id, args.session_file)
    session_records: List[Dict[str, object]] = []

    if not args.metadata_only and session_ids:
        session_records = download_sessions(
            session_ids=session_ids,
            output_root=output_root,
            overwrite=args.overwrite,
            include_lfp=args.include_lfp,
        )

    manifest = {
        "output_root": str(output_root),
        "metadata_files": metadata_records,
        "session_ids": session_ids,
        "session_files": session_records,
        "include_lfp": bool(args.include_lfp),
    }
    manifest_path = _write_manifest(output_root, manifest)

    print(f"Allen VBN output root: {output_root}")
    print(f"Metadata files handled: {len(metadata_records)}")
    print(f"Session files handled: {len(session_records)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
