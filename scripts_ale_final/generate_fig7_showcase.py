#!/usr/bin/env python3
"""Generate Figure 7 (four-phase detection showcase) using the shared paper style.

Four panels:
- (A) Ground-truth phase timeline.
- (B) Total correlation $\\TC(t)$ estimated by DVC, NF-copula, and the Gaussian
      state-space baseline, against the ground-truth phases.
- (C) DVC decomposition: $\\TC_{\\mathrm{pair}}(t)$ vs $\\TC_{\\mathrm{higher}}(t)$,
      showing the higher-order signal concentrating in the root-aligned triplet phase.
- (D) Pairwise MI: MINE vs DVC rank-tau pair MI for two representative pairs.

Styling is delegated entirely to ``dvc_package.visualization.paper_style`` so
this figure matches the look-and-feel of the other main-paper figures.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-showcase-final-"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.visualization.paper_style import (  # noqa: E402
    COLORS,
    add_panel_label,
    apply_style,
)

DEFAULT_RESULTS = PROJECT_ROOT / "results" / "showcase_ale_final" / "contrast_harder_parametric"
OUT_DIR = PROJECT_ROOT / "drafts" / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase band palette: muted, colorblind-safe, same hues as the rest of the paper
PHASE_BANDS = {
    "independent":           "#e8eaed",
    "pairwise-block":        "#cfe2f3",
    "pairwise+higher-order": "#f9d4d4",
    "tail-block":            "#fde9c9",
}
PHASE_LABELS = {
    "independent":           "independent",
    "pairwise-block":        "pairwise star block",
    "pairwise+higher-order": "pairwise + higher-order triplet",
    "tail-block":            "Clayton tail block",
}


def _load() -> dict:
    results_dir = Path(os.environ.get("DVC_SHOWCASE_RESULTS_DIR", str(DEFAULT_RESULTS)))
    with open(results_dir / "summary.json") as fh:
        return json.load(fh)


def _summary_meta(summary: dict) -> tuple[int, list[int], list[str]]:
    if "T" in summary and "phase_boundaries" in summary and "phase_names" in summary:
        return int(summary["T"]), list(summary["phase_boundaries"]), list(summary["phase_names"])
    config = summary.get("config", {})
    return (
        int(config["t"]),
        list(config["phase_boundaries"]),
        list(config["phases"]),
    )


def _phase_bands(ax: plt.Axes, boundaries, names) -> None:
    for i in range(len(boundaries) - 1):
        ax.axvspan(
            boundaries[i],
            boundaries[i + 1] - 0.5,
            color=PHASE_BANDS[names[i]],
            alpha=0.9,
            zorder=0,
            linewidth=0,
        )


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([row.get(key, np.nan) for row in rows], dtype=np.float64)


def _rolling_mean(arr: np.ndarray, window: int = 3) -> np.ndarray:
    if arr.size == 0 or window <= 1:
        return arr.copy()
    return (
        pd.Series(arr)
        .rolling(window, min_periods=1, center=True)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _plot_mean_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray | None,
    *,
    color: str,
    label: str,
    ls: object = "-",
    lw: float = 1.7,
    smooth_window: int = 3,
    band_alpha: float = 0.14,
) -> None:
    mean_s = _rolling_mean(mean, window=smooth_window)
    if std is not None and np.isfinite(std).any():
        std_s = _rolling_mean(std, window=smooth_window)
        lo = mean_s - std_s
        hi = mean_s + std_s
        ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, linewidth=0.0, zorder=1)
    ax.plot(x, mean_s, color=color, ls=ls, lw=lw, label=label, zorder=2)


def main() -> None:
    apply_style()

    summary = _load()
    rows = summary["rows"]
    T, boundaries, phase_names = _summary_meta(summary)
    t_axis = np.arange(T)

    tc_dvc = _series(rows, "tc_total_dvc")
    tc_dvc_std = _series(rows, "tc_total_dvc_std")
    tc_nf = _series(rows, "tc_total_nf")
    tc_nf_std = _series(rows, "tc_total_nf_std")
    tc_ssm = _series(rows, "tc_total_ssm")
    tc_ssm_std = _series(rows, "tc_total_ssm_std")
    tc_pair = _series(rows, "tc_pair_dvc")
    tc_pair_std = _series(rows, "tc_pair_dvc_std")
    tc_higher = _series(rows, "tc_higher_dvc")
    tc_higher_std = _series(rows, "tc_higher_dvc_std")
    mine_01 = _series(rows, "mine_mi_pair01")
    mine_01_std = _series(rows, "mine_mi_pair01_std")
    mine_56 = _series(rows, "mine_mi_pair56")
    mine_56_std = _series(rows, "mine_mi_pair56_std")
    dvc_mi_01 = _series(rows, "dvc_pair_mi01")
    dvc_mi_01_std = _series(rows, "dvc_pair_mi01_std")
    dvc_mi_56 = _series(rows, "dvc_pair_mi56")
    dvc_mi_56_std = _series(rows, "dvc_pair_mi56_std")

    # NeurIPS-width vertical 4-panel figure. Panel heights chosen so that the
    # phase-timeline strip A is compact and panels B-D get equal visible space.
    fig, axes = plt.subplots(
        4, 1,
        figsize=(7.0, 8.3),
        sharex=True,
        gridspec_kw={"height_ratios": [0.28, 1.0, 1.0, 1.0], "hspace": 0.32},
    )

    # -------- Panel A: ground-truth phase timeline --------
    ax = axes[0]
    _phase_bands(ax, boundaries, phase_names)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    mids = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
    for x_mid, nm in zip(mids, phase_names):
        ax.text(x_mid, 0.50, PHASE_LABELS[nm], ha="center", va="center",
                fontsize=7, color="0.15")
    for x in boundaries[1:-1]:
        ax.axvline(x - 0.5, color="0.6", lw=0.5, alpha=0.8)
    ax.set_xlim(-0.5, T - 0.5)
    ax.spines["left"].set_visible(False)
    add_panel_label(ax, "A")

    # -------- Panel B: total correlation trajectories --------
    ax = axes[1]
    _phase_bands(ax, boundaries, phase_names)
    _plot_mean_with_band(
        ax, t_axis, tc_dvc, tc_dvc_std,
        color=COLORS["black"], label="DVC (full vine)", lw=2.0, smooth_window=3, band_alpha=0.12
    )
    _plot_mean_with_band(
        ax, t_axis, tc_nf, tc_nf_std,
        color=COLORS["green"], label="NF-copula", ls=(0, (4, 2)), lw=1.5, smooth_window=3, band_alpha=0.10
    )
    _plot_mean_with_band(
        ax, t_axis, tc_ssm, tc_ssm_std,
        color=COLORS["blue"], label="Gaussian SSM", ls=(0, (2, 1.5)), lw=1.5, smooth_window=3, band_alpha=0.10
    )
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel(r"$\mathrm{TC}(t)$  (nats)")
    ax.legend(loc="upper left", ncol=3, handlelength=2.2, columnspacing=1.2,
              borderpad=0.3, frameon=True)
    add_panel_label(ax, "B")

    # -------- Panel C: DVC pair vs higher-order decomposition --------
    ax = axes[2]
    _phase_bands(ax, boundaries, phase_names)
    _plot_mean_with_band(
        ax, t_axis, tc_pair, tc_pair_std,
        color=COLORS["blue"], label=r"$\mathrm{TC}_\mathrm{pair}(t)$", lw=1.9, smooth_window=3, band_alpha=0.12
    )
    _plot_mean_with_band(
        ax, t_axis, tc_higher, tc_higher_std,
        color=COLORS["red"], label=r"$\mathrm{TC}_\mathrm{higher}(t)$", lw=1.9, smooth_window=3, band_alpha=0.12
    )
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel("nats")
    ax.legend(loc="upper left", ncol=2, handlelength=2.2, columnspacing=1.2,
              borderpad=0.3)
    add_panel_label(ax, "C")

    # -------- Panel D: pairwise MI -- MINE vs DVC rank-tau --------
    ax = axes[3]
    _phase_bands(ax, boundaries, phase_names)
    _plot_mean_with_band(
        ax, t_axis, mine_01, mine_01_std,
        color=COLORS["blue"], label=r"MINE $(X_0, X_1)$", lw=1.7, smooth_window=3, band_alpha=0.10
    )
    _plot_mean_with_band(
        ax, t_axis, dvc_mi_01, dvc_mi_01_std,
        color=COLORS["blue"], label=r"DVC pair $(X_0, X_1)$", ls=(0, (2, 1.5)), lw=1.3, smooth_window=3, band_alpha=0.08
    )
    _plot_mean_with_band(
        ax, t_axis, mine_56, mine_56_std,
        color=COLORS["red"], label=r"MINE $(X_5, X_6)$", lw=1.7, smooth_window=3, band_alpha=0.10
    )
    _plot_mean_with_band(
        ax, t_axis, dvc_mi_56, dvc_mi_56_std,
        color=COLORS["red"], label=r"DVC pair $(X_5, X_6)$", ls=(0, (2, 1.5)), lw=1.3, smooth_window=3, band_alpha=0.08
    )
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel("MI (nats)")
    ax.set_xlabel(r"time-window index $t$")
    ax.legend(loc="upper right", ncol=2, handlelength=2.0, columnspacing=1.0,
              borderpad=0.3, fontsize=6.5)
    add_panel_label(ax, "D")

    fig.align_ylabels(axes[1:])

    out_pdf = OUT_DIR / "fig7_showcase.pdf"
    out_png = OUT_DIR / "fig7_showcase.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote: {out_pdf}")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main()
