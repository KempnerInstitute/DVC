#!/usr/bin/env python3
"""Generate Figure 7 (four-phase detection showcase) using the shared paper style.

Four panels:
- (A) Ground-truth phase timeline.
- (B) Total correlation $\\TC(t)$ estimated by DVC, NF-copula, and the Gaussian
      state-space baseline, against the ground-truth phases.
- (C) DVC decomposition: $\\TC_{\\mathrm{pair}}(t)$ vs $\\TC_{\\mathrm{higher}}(t)$,
      showing the higher-order signal concentrating in the mixed phase.
- (D) Pairwise MI: MINE vs DVC rank-tau pair MI for two representative pairs.

Styling is delegated entirely to ``dvc_package.visualization.paper_style`` so
this figure matches the look-and-feel of the other main-paper figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.visualization.paper_style import (  # noqa: E402
    COLORS,
    add_panel_label,
    apply_style,
)

RESULTS = PROJECT_ROOT / "results" / "showcase"
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
    "pairwise-block":        "pairwise block",
    "pairwise+higher-order": "pairwise + XOR triplet",
    "tail-block":            "Clayton tail block",
}


def _load() -> dict:
    with open(RESULTS / "summary.json") as fh:
        return json.load(fh)


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


def _empirical_kendall_pair(x_a: np.ndarray, x_b: np.ndarray) -> float:
    import pandas as pd
    return float(pd.Series(x_a).corr(pd.Series(x_b), method="kendall"))


def main() -> None:
    apply_style()

    summary = _load()
    rows = summary["rows"]
    T = summary["T"]
    boundaries = summary["phase_boundaries"]
    phase_names = summary["phase_names"]
    t_axis = np.arange(T)

    tc_dvc    = np.array([r["tc_total_dvc"] for r in rows], dtype=np.float64)
    tc_nf     = np.array([r["tc_total_nf"] for r in rows], dtype=np.float64)
    tc_ssm    = np.array([r.get("tc_total_ssm", np.nan) for r in rows], dtype=np.float64)
    tc_pair   = np.array([r["tc_pair_dvc"] for r in rows], dtype=np.float64)
    tc_higher = np.array([r["tc_higher_dvc"] for r in rows], dtype=np.float64)
    mine_01   = np.array([r["mine_mi_pair01"] for r in rows], dtype=np.float64)
    mine_56   = np.array([r["mine_mi_pair56"] for r in rows], dtype=np.float64)

    # DVC-native pairwise MI estimate: Gaussian-copula MI under rank-Kendall tau.
    dvc_mi_01 = np.full(T, np.nan)
    dvc_mi_56 = np.full(T, np.nan)
    from scripts.run_showcase_benchmark import _generate_window  # type: ignore

    rng = np.random.default_rng(2026)
    for t in range(T):
        x = _generate_window(t, rng)
        for pair, storage in [((0, 1), dvc_mi_01), ((5, 6), dvc_mi_56)]:
            tau = _empirical_kendall_pair(x[:, pair[0]], x[:, pair[1]])
            if np.isfinite(tau):
                rho = np.clip(np.sin(np.pi * tau / 2.0), -0.999, 0.999)
                storage[t] = float(-0.5 * np.log(1.0 - rho ** 2))

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
    ax.plot(t_axis, tc_dvc, color=COLORS["black"],  lw=1.7, label="DVC (full vine)")
    ax.plot(t_axis, tc_nf,  color=COLORS["green"],  lw=1.3, ls=(0, (4, 2)),
            label="NF-copula", alpha=0.95)
    ax.plot(t_axis, tc_ssm, color=COLORS["blue"],   lw=1.3, ls=(0, (2, 1.5)),
            label="Gaussian SSM")
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel(r"$\mathrm{TC}(t)$  (nats)")
    ax.legend(loc="upper left", ncol=3, handlelength=2.2, columnspacing=1.2,
              borderpad=0.3, frameon=True)
    add_panel_label(ax, "B")

    # -------- Panel C: DVC pair vs higher-order decomposition --------
    ax = axes[2]
    _phase_bands(ax, boundaries, phase_names)
    ax.plot(t_axis, tc_pair,   color=COLORS["blue"], lw=1.8,
            label=r"$\mathrm{TC}_\mathrm{pair}(t)$")
    ax.plot(t_axis, tc_higher, color=COLORS["red"],  lw=1.8,
            label=r"$\mathrm{TC}_\mathrm{higher}(t)$")
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel("nats")
    ax.legend(loc="upper left", ncol=2, handlelength=2.2, columnspacing=1.2,
              borderpad=0.3)
    add_panel_label(ax, "C")

    # -------- Panel D: pairwise MI -- MINE vs DVC rank-tau --------
    ax = axes[3]
    _phase_bands(ax, boundaries, phase_names)
    ax.plot(t_axis, mine_01,   color=COLORS["blue"], lw=1.6,
            label=r"MINE $(X_0, X_1)$")
    ax.plot(t_axis, dvc_mi_01, color=COLORS["blue"], lw=1.2, ls=(0, (2, 1.5)),
            label=r"DVC pair $(X_0, X_1)$")
    ax.plot(t_axis, mine_56,   color=COLORS["red"],  lw=1.6,
            label=r"MINE $(X_5, X_6)$")
    ax.plot(t_axis, dvc_mi_56, color=COLORS["red"],  lw=1.2, ls=(0, (2, 1.5)),
            label=r"DVC pair $(X_5, X_6)$")
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
