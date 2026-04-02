#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_ORDER = ["graphical_lasso", "gaussian_ssm", "gaussian_copula", "truncated_vine"]
BASELINE_LABELS = {
    "graphical_lasso": "Graphical\nLasso",
    "gaussian_ssm": "Gaussian\nSSM",
    "gaussian_copula": "Gaussian\ncopula",
    "truncated_vine": "1-trunc\nvine",
}
SOURCE_ORDER = ["targeted_2bin_pca4", "mixed_2bin_pca6", "non_targeted_2bin_pca6"]
SOURCE_LABELS = {
    "targeted_2bin_pca4": "Targeted",
    "mixed_2bin_pca6": "Mixed",
    "non_targeted_2bin_pca6": "Non-targeted",
}
FAMILY_GROUP_ORDER = [
    "independence",
    "gaussian_like_elliptical",
    "heavy_tailed_elliptical",
    "lower_tail_asymmetric",
]
FAMILY_GROUP_LABELS = {
    "independence": "Independence",
    "gaussian_like_elliptical": "Gaussian-like\nelliptical",
    "heavy_tailed_elliptical": "Heavy-tailed\nelliptical",
    "lower_tail_asymmetric": "Lower-tail\nasymmetric",
}
FAMILY_RAW_ORDER = ["ind", "gaussian", "student", "clayton"]
FAMILY_RAW_LABELS = {
    "ind": "ind",
    "gaussian": "gaussian",
    "student": "student",
    "clayton": "clayton",
}
PANEL_COLORS = {
    "blue": "#2B6CB0",
    "red": "#C53030",
    "green": "#2F855A",
    "purple": "#6B46C1",
    "orange": "#C05621",
    "gray": "#B8B8B8",
    "dark": "#2D3748",
}
FAMILY_COLORS = {
    "independence": "#CBD5E0",
    "gaussian_like_elliptical": "#90CDF4",
    "heavy_tailed_elliptical": "#F6AD55",
    "lower_tail_asymmetric": "#FC8181",
    "ind": "#CBD5E0",
    "gaussian": "#90CDF4",
    "student": "#F6AD55",
    "clayton": "#FC8181",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Dalgleish latent publication figures from existing outputs.")
    parser.add_argument("--results_root", type=Path, default=Path("results/stimulation_exp_benchmark"))
    parser.add_argument("--out_root", type=Path, default=Path("dvc_ready"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
        }
    )


def _bootstrap_ci(values: Iterable[float], seed: int, draws: int = 4000) -> tuple[float, float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean_v = float(np.mean(arr))
    if arr.size == 1:
        return mean_v, mean_v, mean_v
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(draws, arr.size), replace=True)
    means = np.mean(samples, axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean_v, float(low), float(high)


def _jitter(n: int, width: float = 0.10) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n)


def _draw_interval(ax: plt.Axes, x: float, mean_v: float, low: float, high: float, color: str) -> None:
    ax.vlines(x, low, high, color=color, linewidth=2.2, zorder=4)
    ax.scatter([x], [mean_v], color=color, edgecolor="white", linewidth=0.8, s=48, zorder=5)


def _plot_panel_a(ax: plt.Axes, static_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    panel = static_df[static_df["row_type"] == "baseline_session"].copy()
    x = np.arange(len(BASELINE_ORDER))
    for xpos, baseline in zip(x, BASELINE_ORDER):
        vals = panel.loc[panel["baseline"] == baseline, ["session_id", "delta_vs_full"]].sort_values("session_id")
        jit = _jitter(len(vals), width=0.11)
        ax.scatter(
            np.full(len(vals), xpos) + jit,
            vals["delta_vs_full"].to_numpy(dtype=float),
            color=PANEL_COLORS["gray"],
            s=16,
            alpha=0.85,
            zorder=2,
        )
        stat = stats_df[(stats_df["analysis_scope"] == "panel_a_baseline") & (stats_df["comparison"] == f"full vine vs {baseline}")]
        if not stat.empty and np.isfinite(stat["estimate"].iloc[0]):
            _draw_interval(
                ax,
                xpos,
                float(stat["estimate"].iloc[0]),
                float(stat["ci_low"].iloc[0]),
                float(stat["ci_high"].iloc[0]),
                PANEL_COLORS["blue"],
            )
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.grid(axis="y", alpha=0.5)
    ax.set_xticks(x, [BASELINE_LABELS[b] for b in BASELINE_ORDER])
    ax.set_ylabel("Baseline NLL - Full-vine NLL")
    ax.set_title("A. Full vine outperforms all usable baselines", loc="left")
    ax.text(
        0.99,
        0.02,
        "TVGL attempted but no usable\nsession-level latent-static output",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )


def _plot_panel_b(ax: plt.Axes, static_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    decomp = static_df[static_df["row_type"] == "decomposition_session"].copy()
    order = ["low_level_pairwise_gain", "higher_order_gain"]
    labels = ["Pairwise non-\nGaussian gain", "Higher-order\ngain"]
    x = np.arange(len(order))
    for xpos, component in zip(x, order):
        vals = decomp.loc[decomp["component"] == component, ["session_id", "value"]].sort_values("session_id")
        jit = _jitter(len(vals), width=0.08)
        ax.scatter(
            np.full(len(vals), xpos) + jit,
            vals["value"].to_numpy(dtype=float),
            color=PANEL_COLORS["gray"],
            s=18,
            alpha=0.9,
            zorder=2,
        )
        stat = stats_df[(stats_df["analysis_scope"] == "panel_b_decomposition") & (stats_df["comparison"] == component)]
        if not stat.empty:
            _draw_interval(
                ax,
                xpos,
                float(stat["estimate"].iloc[0]),
                float(stat["ci_low"].iloc[0]),
                float(stat["ci_high"].iloc[0]),
                PANEL_COLORS["red"],
            )
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.grid(axis="y", alpha=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out NLL gain")
    ax.set_title("B. Gain includes a higher-order component", loc="left")


def _plot_source_metric(ax: plt.Axes, source_df: pd.DataFrame, metric: str, title: str, color: str) -> None:
    x = np.arange(len(SOURCE_ORDER))
    for _, group in source_df.groupby("session_id"):
        group = group.set_index("variant").reindex(SOURCE_ORDER).reset_index()
        ax.plot(
            x,
            group[metric].to_numpy(dtype=float),
            color="#D0D0D0",
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )
        ax.scatter(
            x,
            group[metric].to_numpy(dtype=float),
            color="#B9B9B9",
            s=12,
            alpha=0.9,
            zorder=2,
        )
    for xpos, variant in zip(x, SOURCE_ORDER):
        vals = source_df.loc[source_df["variant"] == variant, metric].to_numpy(dtype=float)
        mean_v, low, high = _bootstrap_ci(vals, seed=123 + xpos + len(metric))
        _draw_interval(ax, xpos, mean_v, low, high, color)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.grid(axis="y", alpha=0.5)
    ax.set_xticks(x, [SOURCE_LABELS[v] for v in SOURCE_ORDER])
    ax.set_title(title, fontsize=11)


def _plot_panel_d(ax: plt.Axes, family_df: pd.DataFrame) -> None:
    fam = family_df[
        (family_df["analysis_scope"] == "static_source_space")
        & (family_df["summary_level"] == "grouped_by_source")
        & (family_df["family_group"].isin(FAMILY_GROUP_ORDER))
    ].copy()
    pivot = (
        fam.pivot_table(index="source_space", columns="family_group", values="edge_fraction", aggfunc="first")
        .reindex(["targeted", "mixed", "non_targeted"])
        .fillna(0.0)
    )
    y = np.arange(len(pivot.index))
    left = np.zeros(len(pivot.index), dtype=float)
    for fam_name in FAMILY_GROUP_ORDER:
        vals = pivot[fam_name].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=0.62,
            color=FAMILY_COLORS[fam_name],
            edgecolor="white",
            linewidth=0.8,
            label=FAMILY_GROUP_LABELS[fam_name],
        )
        left += vals
    ax.set_yticks(y, ["Targeted", "Mixed", "Non-targeted"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of fitted pair-copula edges")
    ax.set_title("D. Dependence is heavy-tailed and asymmetric", loc="left")
    ax.grid(axis="x", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)


def _plot_main_figure(static_df: pd.DataFrame, stats_df: pd.DataFrame, family_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(14.2, 8.8))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=0.58)
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c1 = fig.add_subplot(gs[1, 0:2])
    ax_c2 = fig.add_subplot(gs[1, 2:4])
    ax_d = fig.add_subplot(gs[1, 4:6])

    _plot_panel_a(ax_a, static_df, stats_df)
    _plot_panel_b(ax_b, static_df, stats_df)

    source_df = static_df[static_df["row_type"] == "source_space_session"].copy()
    _plot_source_metric(
        ax_c1,
        source_df,
        "full_vs_gaussian",
        "C. Strongest signal in recruited/non-targeted space",
        PANEL_COLORS["green"],
    )
    _plot_source_metric(
        ax_c2,
        source_df,
        "tc_higher",
        "Higher-order gain by source space",
        PANEL_COLORS["orange"],
    )
    ax_c1.set_ylabel("Full-vs-Gaussian")
    ax_c2.set_ylabel("TC_higher")
    _plot_panel_d(ax_d, family_df)

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.95, wspace=0.78, hspace=0.55)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_dose_supplement(dose_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    session_df = dose_df[dose_df["scope"] == "session"].copy()
    pooled_df = dose_df[dose_df["scope"] == "pooled"].sort_values("dose").copy()
    metrics = [
        ("full_delta_vs_gaussian", "mean_full_delta_vs_gaussian", "ci_low_full_delta_vs_gaussian", "ci_high_full_delta_vs_gaussian", "Full-vs-Gaussian", PANEL_COLORS["purple"]),
        ("tc_higher", "mean_tc_higher", "ci_low_tc_higher", "ci_high_tc_higher", "TC_higher", PANEL_COLORS["orange"]),
    ]
    for ax, (session_key, mean_key, low_key, high_key, ylabel, color) in zip(axes, metrics):
        for _, group in session_df.groupby("session_id"):
            group = group.sort_values("dose")
            ax.plot(group["dose"], group[session_key], color="#CFCFCF", linewidth=0.9, alpha=0.75)
            ax.scatter(group["dose"], group[session_key], color="#BDBDBD", s=10, alpha=0.8)
        ax.plot(pooled_df["dose"], pooled_df[mean_key], color=color, linewidth=2.4, marker="o", markersize=4.5)
        ax.fill_between(pooled_df["dose"], pooled_df[low_key], pooled_df[high_key], color=color, alpha=0.16)
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.grid(axis="y", alpha=0.5)
        ax.set_xlabel("Dose")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Dose robustness of overall latent gain", loc="left")
    axes[1].set_title("Dose robustness of higher-order gain", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_family_supplement(family_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # Left: grouped family classes by dose for main variant.
    grouped = family_df[
        (family_df["analysis_scope"] == "static_main")
        & (family_df["summary_level"] == "grouped_by_dose")
        & (family_df["family_group"].isin(FAMILY_GROUP_ORDER))
    ].copy()
    grouped["dose"] = grouped["dose"].astype(float)
    doses = sorted(grouped["dose"].dropna().unique().tolist())
    x = np.arange(len(doses))
    width = 0.18
    for idx, fam_name in enumerate(FAMILY_GROUP_ORDER):
        vals = []
        for dose in doses:
            row = grouped[(grouped["dose"] == dose) & (grouped["family_group"] == fam_name)]
            vals.append(float(row["edge_fraction"].iloc[0]) if not row.empty else np.nan)
        axes[0].bar(
            x + (idx - 1.5) * width,
            vals,
            width=width,
            color=FAMILY_COLORS[fam_name],
            label=FAMILY_GROUP_LABELS[fam_name],
        )
    axes[0].set_xticks(x, [str(int(d)) for d in doses])
    axes[0].set_xlabel("Dose")
    axes[0].set_ylabel("Edge fraction")
    axes[0].set_title("Grouped dependence classes by dose", loc="left")
    axes[0].grid(axis="y", alpha=0.4)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")

    # Right: raw stable families by source space.
    raw = family_df[
        (family_df["analysis_scope"] == "static_source_space")
        & (family_df["summary_level"] == "grouped_by_source")
        & (family_df["family_raw"].isin(FAMILY_RAW_ORDER))
    ].copy()
    pivot = (
        raw.pivot_table(index="source_space", columns="family_raw", values="edge_fraction", aggfunc="first")
        .reindex(["targeted", "mixed", "non_targeted"])
        .fillna(0.0)
    )
    y = np.arange(len(pivot.index))
    left = np.zeros(len(pivot.index), dtype=float)
    for fam_name in FAMILY_RAW_ORDER:
        vals = pivot[fam_name].to_numpy(dtype=float)
        axes[1].barh(
            y,
            vals,
            left=left,
            height=0.62,
            color=FAMILY_COLORS[fam_name],
            edgecolor="white",
            linewidth=0.8,
            label=FAMILY_RAW_LABELS[fam_name],
        )
        left += vals
    axes[1].set_yticks(y, ["Targeted", "Mixed", "Non-targeted"])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Edge fraction")
    axes[1].set_title("Raw stable families by source space", loc="left")
    axes[1].grid(axis="x", alpha=0.4)
    axes[1].legend(fontsize=8, ncol=2, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _set_style()

    results_root = args.results_root.resolve()
    data_dir = results_root / "data"
    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    static_df = pd.read_csv(data_dir / "latent_publication_static_summary.csv")
    stats_df = pd.read_csv(data_dir / "latent_publication_stats_summary.csv")
    family_df = pd.read_csv(data_dir / "latent_publication_family_summary.csv")
    dose_df = pd.read_csv(data_dir / "latent_followup_dose_summary.csv")

    main_fig = plots_dir / "fig_latent_publication_final.png"
    dose_fig = plots_dir / "fig_latent_publication_dose_supplement.png"
    family_fig = plots_dir / "fig_latent_publication_family_supplement.png"

    _plot_main_figure(static_df, stats_df, family_df, main_fig)
    _plot_dose_supplement(dose_df, dose_fig)
    _plot_family_supplement(family_df, family_fig)

    panel_map = {
        "main_figure": {
            "file": str(main_fig.relative_to(results_root.parent)),
            "panels": {
                "A": {
                    "message": "Full vine outperforms all usable baselines",
                    "source_tables": [
                        "latent_publication_static_summary.csv",
                        "latent_publication_stats_summary.csv",
                        "latent_publication_baseline_feasibility.csv",
                    ],
                },
                "B": {
                    "message": "Gain includes a higher-order component",
                    "source_tables": [
                        "latent_publication_static_summary.csv",
                        "latent_publication_stats_summary.csv",
                    ],
                },
                "C": {
                    "message": "Strongest signal in recruited/non-targeted space",
                    "source_tables": [
                        "latent_publication_static_summary.csv",
                        "latent_publication_stats_summary.csv",
                    ],
                },
                "D": {
                    "message": "Dependence is heavy-tailed and asymmetric",
                    "source_tables": [
                        "latent_publication_family_summary.csv",
                    ],
                },
            },
        },
        "supplement_figures": {
            "dose": str(dose_fig.relative_to(results_root.parent)),
            "family": str(family_fig.relative_to(results_root.parent)),
        },
        "dynamic_note": "Dynamic/time-history panels were intentionally left unchanged in this refresh pass.",
    }

    panel_map_path = data_dir / "latent_publication_figure_panel_map.json"
    panel_map_path.write_text(json.dumps(panel_map, indent=2))
    (out_root / "latent_publication_figure_panel_map.json").write_text(json.dumps(panel_map, indent=2))


if __name__ == "__main__":
    main()
