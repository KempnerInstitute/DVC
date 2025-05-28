"""Combine all PyTorch-based implementation files into one script.

Usage:
    python tools/combine_pytorch.py [-o OUTPUT]

This collects every ``.py`` file under ``src/DVC`` and concatenates them in
alphabetical order into a single file.  A heading with the file name is
inserted before each file's contents for clarity.
"""

import argparse
from pathlib import Path
from typing import Iterable


def gather_files(base: Path) -> Iterable[Path]:
    """Yield all ``.py`` files under ``base`` sorted alphabetically."""
    return sorted(base.rglob("*.py"))


def combine(paths: Iterable[Path], output: Path) -> None:
    """Write the contents of ``paths`` to ``output`` with headings."""
    with output.open("w") as out:
        for p in paths:
            rel = p.relative_to(Path.cwd()) if p.is_absolute() else p
            out.write(f"# File: {rel}\n")
            out.write(p.read_text())
            out.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine PyTorch source files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("all_pytorch.py"),
        help="Path of the combined output file",
    )
    args = parser.parse_args()

    base_dir = Path("src") / "DVC"
    files = gather_files(base_dir)
    combine(files, args.output)


if __name__ == "__main__":
    main()