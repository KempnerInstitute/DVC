#!/usr/bin/env python3
"""Download and extract the Dalgleish Figshare dataset."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen


LOGGER = logging.getLogger("dalgleish_figshare_download")
FIGSHARE_API_URL = "https://api.figshare.com/v2/articles/13128950/versions/2"


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def fetch_article_metadata() -> Dict[str, Any]:
    with urlopen(FIGSHARE_API_URL) as response:
        return json.load(response)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "-C", "-", "--fail", "--output", str(destination), url]
    LOGGER.info("Downloading %s", destination.name)
    subprocess.run(cmd, check=True)


def extract_zip(zip_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    cmd = ["unzip", "-o", "-q", str(zip_path), "-d", str(extract_root)]
    LOGGER.info("Extracting %s", zip_path.name)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Dalgleish Figshare dataset.")
    parser.add_argument(
        "--output_root",
        default="/n/netscratch/kempner_dev/Lab/hsafaai/results/kempner_project_b/datasets/dalgleish_figshare",
        help="Root directory for downloaded Dalgleish artifacts.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded session zip files into dataset_stimulation/.",
    )
    parser.add_argument(
        "--session_limit",
        type=int,
        default=None,
        help="Optional cap on the number of Figshare files to download.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    output_root = Path(args.output_root).expanduser().resolve()
    raw_root = output_root / "raw_zips"
    extract_root = output_root / "dataset_stimulation"
    metadata_path = output_root / "figshare_article_v2.json"

    article = fetch_article_metadata()
    write_json(metadata_path, article)
    files: List[Dict[str, Any]] = list(article.get("files", []))
    if args.session_limit is not None:
        files = files[: int(args.session_limit)]

    total_size = int(sum(int(item.get("size", 0)) for item in files))
    LOGGER.info("Figshare article: %s", article.get("title", "unknown"))
    LOGGER.info("Files selected: %d", len(files))
    LOGGER.info("Total bytes selected: %d", total_size)

    manifest_rows: List[Dict[str, Any]] = []
    for item in files:
        name = str(item["name"])
        url = str(item["download_url"])
        size = int(item.get("size", 0))
        destination = raw_root / name
        download_file(url, destination)
        if args.extract:
            extract_zip(destination, extract_root)
        manifest_rows.append(
            {
                "name": name,
                "size": size,
                "download_url": url,
                "downloaded_to": str(destination),
                "extracted_to": str(extract_root) if args.extract else None,
            }
        )

    write_json(
        output_root / "download_manifest.json",
        {
            "figshare_api_url": FIGSHARE_API_URL,
            "output_root": str(output_root),
            "raw_root": str(raw_root),
            "extract_root": str(extract_root),
            "extract_enabled": bool(args.extract),
            "files": manifest_rows,
        },
    )
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
