#!/usr/bin/env python3
"""Generate Figure 7 (four-phase detection showcase) using the shared paper style.

Four panels:
- (A) Ground-truth phase timeline.
- (B) Total correlation $\\TC(t)$ estimated by joint switching DVC, the
      windowed full-vine control, NF-copula, and the Gaussian state-space
      baseline, against oracle/analytic ground truth.
- (C) Joint switching-DVC decomposition: $\\TC_{\\mathrm{pair}}(t)$ vs
      $\\TC_{\\mathrm{higher}}(t)$, with the windowed control and oracle targets.
- (D) Pairwise MI: MINE vs the full-vine rank-tau pair MI proxy for two
      representative pairs, with oracle pairwise MI targets.

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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.visualization.paper_style import (  # noqa: E402
    COLORS,
    add_panel_label,
    apply_style,
)
from scripts_ale_final.showcase_analysis_utils import (  # noqa: E402
    ShowcaseConfig,
    showcase_truth_by_phase,
)

DEFAULT_RESULTS = PROJECT_ROOT / "results" / "showcase_ale_final" / "contrast_harder_parametric"
FALLBACK_RESULTS = (
    PROJECT_ROOT / "results" / "showcase_ale_final" / "proper_sota_nf_mine_1seed",
    PROJECT_ROOT / "results" / "showcase_ale_final" / "proper_parametric_core_3seed",
    PROJECT_ROOT / "results" / "showcase_ale_final" / "truth_smoke_parametric",
)
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
    "pairwise-block":        "Gaussian star",
    "pairwise+higher-order": "star + triplets",
    "tail-block":            "Clayton tail",
}


def _load() -> dict:
    env_results = os.environ.get("DVC_SHOWCASE_RESULTS_DIR")
    if env_results:
        results_dir = Path(env_results)
    else:
        candidates = (DEFAULT_RESULTS, *FALLBACK_RESULTS)
        results_dir = next((path for path in candidates if (path / "summary.json").exists()), DEFAULT_RESULTS)
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Could not find {summary_path}. Set DVC_SHOWCASE_RESULTS_DIR to a showcase results directory."
        )
    with open(summary_path) as fh:
        summary = json.load(fh)
    return _ensure_truth_fields(summary)


def _ensure_truth_fields(summary: dict) -> dict:
    rows = summary.get("rows", [])
    if not rows or "truth_tc_total" in rows[0]:
        return summary
    config_payload = summary.get("config", {})
    if not config_payload:
        return summary
    config = ShowcaseConfig(
        d=int(config_payload.get("d", 10)),
        t=int(config_payload.get("t", summary.get("T", 60))),
        n_per_time=int(config_payload.get("n_per_time", summary.get("n_per_time", 300))),
        train_frac=float(config_payload.get("train_frac", 0.85)),
        phase_boundaries=tuple(config_payload.get("phase_boundaries", summary.get("phase_boundaries", [0, 15, 30, 45, 60]))),
        phases=tuple(config_payload.get("phases", summary.get("phase_names", []))),
        pair_root=int(config_payload.get("pair_root", 0)),
        pair_leaves=tuple(config_payload.get("pair_leaves", [1, 2, 3])),
        pair_rho=float(config_payload.get("pair_rho", 0.55)),
        phase3_mode=str(config_payload.get("phase3_mode", "multiplicative_triplets")),
        triplet_blocks=tuple(tuple(block) for block in config_payload.get("triplet_blocks", [[4, 5, 6], [7, 8, 9]])),
        triplet_rho=float(config_payload.get("triplet_rho", 0.65)),
        triplet_nu=float(config_payload.get("triplet_nu", 4.5)),
        triplet_clayton_theta=float(config_payload.get("triplet_clayton_theta", 2.0)),
        multiplicative_noise_std=float(config_payload.get("multiplicative_noise_std", 0.10)),
        xor_jitter_std=float(config_payload.get("xor_jitter_std", 1e-3)),
        tail_block=tuple(config_payload.get("tail_block", [0, 1, 2, 3])),
        tail_theta=float(config_payload.get("tail_theta", 3.5)),
    )
    truth_by_phase = showcase_truth_by_phase(config, str(summary.get("variant", config.phase3_mode)))
    for row in rows:
        row.update(truth_by_phase.get(row.get("phase_name"), {}))
    summary["ground_truth_by_phase"] = truth_by_phase
    return summary


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


def _plot_truth(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    ls: object = (0, (5, 2, 1.5, 2)),
    lw: float = 1.35,
) -> None:
    if np.isfinite(values).any():
        ax.plot(
            x,
            values,
            color=color,
            ls=ls,
            lw=lw,
            drawstyle="steps-post",
            label=label,
            zorder=3,
        )


def _ordered_legend(ax: plt.Axes, order: list[str], **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = {label: handle for handle, label in zip(handles, labels)}
    ordered_labels = [label for label in order if label in by_label]
    ordered_handles = [by_label[label] for label in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, **kwargs)


def main() -> None:
    apply_style()

    summary = _load()
    rows = summary["rows"]
    T, boundaries, phase_names = _summary_meta(summary)
    t_axis = np.arange(T)

    tc_windowed = _series(rows, "tc_total_dvc")
    tc_windowed_std = _series(rows, "tc_total_dvc_std")
    tc_switching = _series(rows, "tc_total_switching_dvc")
    tc_switching_std = _series(rows, "tc_total_switching_dvc_std")
    has_switching = bool(np.isfinite(tc_switching).any())
    tc_nf = _series(rows, "tc_total_nf")
    tc_nf_std = _series(rows, "tc_total_nf_std")
    tc_ssm = _series(rows, "tc_total_ssm")
    tc_ssm_std = _series(rows, "tc_total_ssm_std")
    tc_pair_windowed = _series(rows, "tc_pair_dvc")
    tc_pair_windowed_std = _series(rows, "tc_pair_dvc_std")
    tc_higher_windowed = _series(rows, "tc_higher_dvc")
    tc_higher_windowed_std = _series(rows, "tc_higher_dvc_std")
    tc_pair = _series(rows, "tc_pair_switching_dvc") if has_switching else tc_pair_windowed
    tc_pair_std = _series(rows, "tc_pair_switching_dvc_std") if has_switching else tc_pair_windowed_std
    tc_higher = _series(rows, "tc_higher_switching_dvc") if has_switching else tc_higher_windowed
    tc_higher_std = _series(rows, "tc_higher_switching_dvc_std") if has_switching else tc_higher_windowed_std
    mine_01 = _series(rows, "mine_mi_pair01")
    mine_01_std = _series(rows, "mine_mi_pair01_std")
    mine_56 = _series(rows, "mine_mi_pair56")
    mine_56_std = _series(rows, "mine_mi_pair56_std")
    dvc_mi_01 = _series(rows, "dvc_pair_mi01")
    dvc_mi_01_std = _series(rows, "dvc_pair_mi01_std")
    dvc_mi_56 = _series(rows, "dvc_pair_mi56")
    dvc_mi_56_std = _series(rows, "dvc_pair_mi56_std")
    truth_tc = _series(rows, "truth_tc_total")
    truth_pair = _series(rows, "truth_tc_pair_oracle")
    truth_higher = _series(rows, "truth_tc_higher_oracle")
    truth_mi_01 = _series(rows, "truth_pair_mi01")
    truth_mi_56 = _series(rows, "truth_pair_mi56")

    # NeurIPS-width compact figure: a thin phase timeline plus a 1x3 results
    # strip. This preserves temporal readability while taking much less
    # vertical space than a four-row layout.
    fig = plt.figure(figsize=(7.0, 3.45))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.22, 1.0],
        hspace=0.64,
        wspace=0.36,
    )
    axes = np.array(
        [
            fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ],
        dtype=object,
    )

    # -------- Panel A: ground-truth phase timeline --------
    ax = axes[0]
    _phase_bands(ax, boundaries, phase_names)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    mids = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]
    for x_mid, nm in zip(mids, phase_names):
        ax.text(x_mid, 0.50, PHASE_LABELS[nm], ha="center", va="center",
                fontsize=5.8, color="0.15")
    for x in boundaries[1:-1]:
        ax.axvline(x - 0.5, color="0.6", lw=0.5, alpha=0.8)
    ax.set_xlim(-0.5, T - 0.5)
    ax.spines["left"].set_visible(False)
    add_panel_label(ax, "A", fontsize=8)

    # -------- Panel B: total correlation trajectories --------
    ax = axes[1]
    _phase_bands(ax, boundaries, phase_names)
    if has_switching:
        _plot_mean_with_band(
            ax, t_axis, tc_switching, tc_switching_std,
            color=COLORS["black"], label="DVC-switch", lw=2.0, smooth_window=3, band_alpha=0.12
        )
        _plot_mean_with_band(
            ax, t_axis, tc_windowed, tc_windowed_std,
            color=COLORS["gray"], label="Win. vine", ls=(0, (3, 1.5)), lw=1.45,
            smooth_window=3, band_alpha=0.08
        )
    else:
        _plot_mean_with_band(
            ax, t_axis, tc_windowed, tc_windowed_std,
            color=COLORS["black"], label="Win. vine", lw=2.0, smooth_window=3, band_alpha=0.12
        )
    _plot_mean_with_band(
        ax, t_axis, tc_nf, tc_nf_std,
        color=COLORS["green"], label="NF-copula", ls=(0, (4, 2)), lw=1.5, smooth_window=3, band_alpha=0.10
    )
    _plot_mean_with_band(
        ax, t_axis, tc_ssm, tc_ssm_std,
        color=COLORS["blue"], label="Gaussian SSM", ls=(0, (2, 1.5)), lw=1.5, smooth_window=3, band_alpha=0.10
    )
    _plot_truth(
        ax, t_axis, truth_tc,
        color=COLORS["purple"], label="oracle TC", lw=1.45
    )
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel(r"$\mathrm{TC}(t)$  (nats)")
    _ordered_legend(
        ax,
        ["DVC-switch", "Win. vine", "Gaussian SSM", "oracle TC", "NF-copula"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        handlelength=1.4,
        columnspacing=0.7,
        borderpad=0.20,
        frameon=False,
        fontsize=4.8,
    )
    add_panel_label(ax, "B", fontsize=8)

    # -------- Panel C: full-vine pair vs higher-order decomposition --------
    ax = axes[2]
    _phase_bands(ax, boundaries, phase_names)
    _plot_mean_with_band(
        ax, t_axis, tc_pair, tc_pair_std,
        color=COLORS["blue"], label=r"DVC-switch pair", lw=1.9, smooth_window=3, band_alpha=0.12
    )
    _plot_mean_with_band(
        ax, t_axis, tc_higher, tc_higher_std,
        color=COLORS["red"], label=r"DVC-switch higher", lw=1.9, smooth_window=3, band_alpha=0.12
    )
    if has_switching:
        _plot_mean_with_band(
            ax, t_axis, tc_higher_windowed, tc_higher_windowed_std,
            color=COLORS["gray"], label=r"Win. vine higher", ls=(0, (3, 1.5)),
            lw=1.25, smooth_window=3, band_alpha=0.05
        )
    _plot_truth(
        ax, t_axis, truth_pair,
        color=COLORS["blue"], label=r"oracle pair", ls=(0, (4, 2)), lw=1.2
    )
    _plot_truth(
        ax, t_axis, truth_higher,
        color=COLORS["red"], label=r"oracle higher", ls=(0, (4, 2)), lw=1.2
    )
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel("nats")
    _ordered_legend(
        ax,
        [
            r"DVC-switch pair",
            r"DVC-switch higher",
            r"Win. vine higher",
            r"oracle pair",
            r"oracle higher",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        handlelength=1.25,
        columnspacing=0.65,
        borderpad=0.20,
        frameon=False,
        fontsize=4.8,
    )
    add_panel_label(ax, "C", fontsize=8)

    # -------- Panel D: pairwise MI -- MINE vs full-vine rank-tau proxy --------
    ax = axes[3]
    _phase_bands(ax, boundaries, phase_names)
    _plot_mean_with_band(
        ax, t_axis, mine_01, mine_01_std,
        color=COLORS["blue"], label=r"MINE star", lw=1.7, smooth_window=3, band_alpha=0.10
    )
    _plot_mean_with_band(
        ax, t_axis, dvc_mi_01, dvc_mi_01_std,
        color=COLORS["cyan"], label=r"Vine $\tau$ star", ls=(0, (5, 1.5, 1.2, 1.5)), lw=1.35, smooth_window=3, band_alpha=0.08
    )
    _plot_mean_with_band(
        ax, t_axis, mine_56, mine_56_std,
        color=COLORS["red"], label=r"MINE triplet", lw=1.7, smooth_window=3, band_alpha=0.10
    )
    _plot_mean_with_band(
        ax, t_axis, dvc_mi_56, dvc_mi_56_std,
        color=COLORS["orange"], label=r"Vine $\tau$ triplet", ls=(0, (5, 1.5, 1.2, 1.5)), lw=1.35, smooth_window=3, band_alpha=0.08
    )
    _plot_truth(
        ax, t_axis, truth_mi_01,
        color="#08306B", label=r"oracle star", ls=(0, (1, 1.2)), lw=1.25
    )
    _plot_truth(
        ax, t_axis, truth_mi_56,
        color="#67000D", label=r"oracle triplet", ls=(0, (1, 1.2)), lw=1.25
    )
    ax.axhline(0.0, color=COLORS["gray"], lw=0.5, ls="--", alpha=0.8, zorder=0.5)
    ax.set_ylabel("MI (nats)")
    ax.set_xlabel(r"time-window index $t$")
    _ordered_legend(
        ax,
        [
            r"MINE star",
            r"Vine $\tau$ star",
            r"oracle star",
            r"MINE triplet",
            r"Vine $\tau$ triplet",
            r"oracle triplet",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        handlelength=1.15,
        columnspacing=0.45,
        borderpad=0.15,
        frameon=False,
        fontsize=4.25,
    )
    add_panel_label(ax, "D", fontsize=8)

    for ax in axes[1:]:
        ax.set_xlim(-0.5, T - 0.5)
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
