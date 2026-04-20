#!/usr/bin/env python3
"""Generate a paper-ready Figure 6 composite for the finance FRED benchmark.

Panel A: NLL gap vs Gaussian SSM and 1-truncated vine over time, with crisis shading.
Panel B: Per-scope (calm, GFC, COVID) mean NLL gap heatmap across baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


RESULTS = Path("results/finance_crisis_fred")
OUT_DIR = Path("drafts/figures/paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRISES = [
    ("GFC", "2008-09-01", "2009-06-30"),
    ("COVID", "2020-02-15", "2020-06-30"),
]


def _load_scenario():
    with open(RESULTS / "summary.json") as fh:
        data = json.load(fh)
    scenarios = data["scenarios"]
    (name, payload), = scenarios.items()
    return name, payload


def _gap_series(payload, key):
    return np.asarray(payload.get(key, []), dtype=np.float64)


def _shade_crises(ax):
    ylo, yhi = ax.get_ylim()
    for name, start, end in CRISES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#fde0dd", alpha=0.5, zorder=0)
    ax.set_ylim(ylo, yhi)


def make_figure():
    name, payload = _load_scenario()
    dates = pd.to_datetime(payload["window_end_dates"])

    gaps = {
        "Gaussian SSM": _gap_series(payload, "nll_gap_state_space"),
        "1-truncated vine": _gap_series(payload, "nll_gap_truncated_level0"),
    }

    crisis_df = pd.read_csv(RESULTS / "data" / "fred_cross_asset_crises_crisis_summary.csv")
    pivot = crisis_df.pivot(index="baseline", columns="scope", values="mean_gap")
    scope_order = ["calm", "GFC", "COVID"]
    baseline_order = [
        "Gaussian copula",
        "1-truncated C-vine",
        "Graphical Lasso",
        "TVGL (Frobenius)",
        "Gaussian SSM",
    ]
    pivot = pivot.reindex(index=baseline_order, columns=scope_order)

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [2.3, 1.0]}
    )

    ax = axes[0]
    colors = {"Gaussian SSM": "#1f77b4", "1-truncated vine": "#d62728"}
    for label, arr in gaps.items():
        ax.plot(dates, arr, color=colors[label], linewidth=1.1, label=label, alpha=0.85)
    ax.axhline(0.0, color="k", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_ylabel("NLL gap (baseline \u2212 DVC; positive = DVC better)")
    ax.set_xlabel("Window end date")
    ax.set_title("(A) Cross-asset NLL gap vs representative baselines", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for name, start, end in CRISES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#fde0dd", alpha=0.5, zorder=0)
        ax.text(
            pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2,
            ax.get_ylim()[1] * 0.92,
            name,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#a94442",
        )

    ax = axes[1]
    data = pivot.values
    vmax = float(np.nanmax(np.abs(data)))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(scope_order)))
    ax.set_xticklabels(scope_order, fontsize=9)
    ax.set_yticks(range(len(baseline_order)))
    ax.set_yticklabels(baseline_order, fontsize=9)
    for i in range(len(baseline_order)):
        for j in range(len(scope_order)):
            val = data[i, j]
            ax.text(
                j,
                i,
                f"{val:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if abs(val) < vmax * 0.5 else "white",
            )
    ax.set_title("(B) Mean NLL gap by regime", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("DVC - baseline (nats)", fontsize=9)

    fig.tight_layout()
    out_pdf = OUT_DIR / "fig6_finance.pdf"
    out_png = OUT_DIR / "fig6_finance.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out_pdf}")
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    make_figure()
