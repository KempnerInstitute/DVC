#!/usr/bin/env python3
"""Thin wrapper around the standalone `scripts_ale_final` showcase benchmark.

This keeps the final benchmark workflow self-contained inside
`scripts_ale_final/` while exposing the recommended run modes:

- `parametric`: main benchmark with the repaired parametric DVC comparator
- `with_np`: same benchmark plus the repaired nonparametric DVC as a
  supplementary comparator
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT_ROOT / "scripts_ale_final" / "run_showcase_benchmark.py"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "showcase_ale_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["parametric", "with_np"],
        default="parametric",
        help=(
            "`parametric` is the final main benchmark. "
            "`with_np` adds the repaired nonparametric DVC as a supplementary comparator."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["fixed", "contrast_harder"],
        default="contrast_harder",
        help="Final recommended showcase preset. `contrast_harder` is the default.",
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--n-per-time", type=int, default=None)
    parser.add_argument(
        "--phase-duration",
        type=int,
        default=None,
        help="Windows per phase; combine with --cycle-repeats for recurrent showcase schedules.",
    )
    parser.add_argument(
        "--cycle-repeats",
        type=int,
        default=1,
        help="Repeat the four-phase showcase cycle this many times.",
    )
    parser.add_argument(
        "--multiplicative-noise-std",
        type=float,
        default=None,
        help="Override the multiplicative triplet noise used by contrast_harder.",
    )
    parser.add_argument("--mine-epochs", type=int, default=60)
    parser.add_argument("--nf-epochs", type=int, default=40)
    parser.add_argument("--skip-mine", action="store_true")
    parser.add_argument("--skip-nf", action="store_true")
    parser.add_argument("--include-regularized-dvc", action="store_true")
    parser.add_argument(
        "--np-temporal-smoothing",
        type=float,
        default=0.12,
        help="Coupled nonparametric smoothing bandwidth used only in `with_np` mode.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults under results/showcase_ale_final/.",
    )
    return parser.parse_args()


def _default_out_dir(args: argparse.Namespace) -> Path:
    suffix = "with_np" if args.mode == "with_np" else "parametric"
    return DEFAULT_RESULTS_ROOT / f"{args.preset}_{suffix}"


def build_command(args: argparse.Namespace) -> list[str]:
    out_dir = args.out if args.out is not None else _default_out_dir(args)
    cmd = [
        sys.executable,
        str(BASE_SCRIPT),
        "--preset",
        str(args.preset),
        "--n-seeds",
        str(args.n_seeds),
        "--base-seed",
        str(args.base_seed),
        "--mine-epochs",
        str(args.mine_epochs),
        "--nf-epochs",
        str(args.nf_epochs),
        "--out",
        str(out_dir),
    ]
    if args.n_per_time is not None:
        cmd.extend(["--n-per-time", str(args.n_per_time)])
    if args.phase_duration is not None:
        cmd.extend(["--phase-duration", str(args.phase_duration)])
    if int(args.cycle_repeats) != 1:
        cmd.extend(["--cycle-repeats", str(args.cycle_repeats)])
    if args.multiplicative_noise_std is not None:
        cmd.extend(["--multiplicative-noise-std", str(args.multiplicative_noise_std)])
    if args.skip_mine:
        cmd.append("--skip-mine")
    if args.skip_nf:
        cmd.append("--skip-nf")
    if args.include_regularized_dvc:
        cmd.append("--include-regularized-dvc")
    if args.mode == "with_np":
        cmd.extend(
            [
                "--include-nonparametric-dvc",
                "--np-vine-type",
                "d-vine",
                "--np-knots",
                "7",
                "--np-temporal-smoothing",
                str(args.np_temporal_smoothing),
                "--np-higher-tree-validation-margin",
                "0.05",
            ]
        )
    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    print("Running final showcase benchmark:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
