#!/usr/bin/env python3
"""Measure DVC fit-time as a function of (d, T, variant) and write a LaTeX table.

This produces the runtime/scaling table referenced in the paper's appendix.
Reported quantities:
- sequential wall-clock fit time for the complete T-window sequence
- average per-window CPU time, computed as total time / T
- number of edge-level temporal fit objects
- memory footprint is not reported here; add with psutil if needed.

Variants covered:
- Windowed parametric C-vine (per-time refit)
- Joint dynamic C-vine (shared trajectory)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.experiments.simulation_benchmarks import (
    _fit_parametric_vine,
    _pseudo_obs_rank,
)


OUT_DIR = Path("results/runtime_scaling")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = Path("drafts/tables/benchmark_tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [3, 5, 8]
T_VALUES = [12, 24]
FAMILIES = ["gaussian", "clayton", "gumbel", "independence"]
N_PER_TIME = 80
N_REPEATS = 2


def _n_edges(d: int) -> int:
    return int(d * (d - 1) // 2)


def _gen_data(d: int, T: int, n_per_t: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Synthetic multivariate normal data with mild cross-sectional correlation."""
    base_corr = 0.3
    cov = np.full((d, d), base_corr)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    return [
        rng.standard_normal((n_per_t, d)) @ L.T
        for _ in range(T)
    ]


def _time_windowed(data_seq: list[np.ndarray], seed: int) -> float:
    """Windowed parametric C-vine: refit per time point. Return total wall-clock."""
    t0 = time.perf_counter()
    for t, x in enumerate(data_seq):
        _fit_parametric_vine(x, families=FAMILIES, optimize_structure=False, seed=seed + t)
    return time.perf_counter() - t0


def _time_joint_dynamic(data_seq: list[np.ndarray], seed: int) -> float:
    """Joint dynamic C-vine: fit a shared temporal trajectory over all T windows."""
    try:
        from dvc_package.time.joint_dynamic_cvine import JointDynamicCVine
    except Exception:
        return float("nan")
    _ = seed  # not currently used by JointDynamicCVine
    t0 = time.perf_counter()
    try:
        model = JointDynamicCVine(families=FAMILIES, n_basis=3, maxiter=30)
        model.fit(data_seq)
    except Exception:
        return float("nan")
    return time.perf_counter() - t0


def main() -> None:
    rows: list[dict] = []
    rng = np.random.default_rng(2026)
    for d in DIMS:
        for T in T_VALUES:
            for variant_name, variant_fn in [
                ("Windowed", _time_windowed),
                ("Joint dynamic", _time_joint_dynamic),
            ]:
                times = []
                for rep in range(N_REPEATS):
                    seed = 1000 + 17 * rep + d * 31 + T
                    data_seq = _gen_data(d, T, N_PER_TIME, rng)
                    dt = variant_fn(data_seq, seed=seed)
                    times.append(dt)
                row = {
                    "d": d,
                    "T": T,
                    "variant": variant_name,
                    "mean_s": float(np.nanmean(times)),
                    "std_s": float(np.nanstd(times)),
                    "per_window_mean_s": float(np.nanmean(times) / T),
                    "edge_fit_units": int(_n_edges(d) * (T if variant_name == "Windowed" else 1)),
                    "compression_vs_windowed": float(1.0 if variant_name == "Windowed" else T),
                }
                rows.append(row)
                print(f"d={d} T={T} {variant_name}: {row['mean_s']:.2f} +/- {row['std_s']:.2f} s (per-window {row['per_window_mean_s']:.3f} s)")

    summary_path = OUT_DIR / "runtime_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote: {summary_path}")

    tex_lines = [
        r"\begin{tabular}{rrlrrrr}",
        r"\toprule",
        r"d & T & variant & edge fits & compression & total time (s) & time / window (s) \\",
        r"\midrule",
    ]
    for row in rows:
        if np.isnan(row["mean_s"]):
            mean_str = "---"
            per_str = "---"
        else:
            mean_str = f"{row['mean_s']:.2f}"
            per_str = f"{row['per_window_mean_s']:.3f}"
        compression = f"{row['compression_vs_windowed']:.0f}$\\times$"
        tex_lines.append(
            f"{row['d']} & {row['T']} & {row['variant']} & {row['edge_fit_units']} & {compression} & {mean_str} & {per_str} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", ""]
    tex_path = TABLE_DIR / "runtime_scaling.tex"
    tex_path.write_text("\n".join(tex_lines))
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
