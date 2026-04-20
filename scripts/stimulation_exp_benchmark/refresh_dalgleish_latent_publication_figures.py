#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.visualization.paper_style import COLORS, TEXTWIDTH, apply_style


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
    "blue": COLORS["blue"],
    "red": COLORS["red"],
    "green": COLORS["green"],
    "purple": COLORS["purple"],
    "orange": COLORS["orange"],
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
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--draft_figures_dir", type=Path, default=Path("drafts/figures/dalgleish"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _set_style() -> None:
    apply_style()
    plt.rcParams.update(
        {
            "legend.frameon": False,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.5,
            "axes.titleweight": "bold",
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
    ax.set_title("A. Full vine outperforms usable baselines", loc="left")
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
    ax.set_title("B. Higher-order gain is positive on average", loc="left")


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
    ax.set_title(title, fontsize=10)


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
    ax.set_title("D. Heavy-tailed and asymmetric dependence", loc="left")
    ax.grid(axis="x", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)


def _plot_main_figure(static_df: pd.DataFrame, stats_df: pd.DataFrame, family_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(TEXTWIDTH * 2.2, TEXTWIDTH * 1.4))
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
        "C. Strongest signal in recruited /\nnon-targeted space",
        PANEL_COLORS["green"],
    )
    _plot_source_metric(
        ax_c2,
        source_df,
        "tc_higher",
        "Higher-order gain\nby source space",
        PANEL_COLORS["orange"],
    )
    ax_c1.set_ylabel("Full-vs-Gaussian")
    ax_c2.set_ylabel("TC_higher")
    _plot_panel_d(ax_d, family_df)

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.95, wspace=0.78, hspace=0.55)
    fig.savefig(out_path, dpi=600)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_dynamic_supplement(dynamic_df: pd.DataFrame, out_path: Path) -> None:
    if dynamic_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    block_order = ["early", "middle", "late"]
    bx = np.arange(len(block_order))
    metrics = [
        ("full_vs_gaussian", "Dynamic gain", PANEL_COLORS["purple"]),
        ("tc_higher", "Dynamic higher-order gain", PANEL_COLORS["orange"]),
    ]
    for ax, (metric, ylabel, color) in zip(axes, metrics):
        for basis_mode, line_color in [("window_train_basis", color), ("common_basis", PANEL_COLORS["red"])]:
            sub = dynamic_df[dynamic_df["basis_mode"] == basis_mode].copy()
            if sub.empty:
                continue
            for _, group in sub.groupby("session_id"):
                group = group.set_index("block_id").reindex(block_order).reset_index()
                ax.plot(bx, group[metric], color=line_color, linewidth=0.9, alpha=0.16)
            agg = sub.groupby("block_id")[[metric]].mean().reindex(block_order)
            ax.plot(
                bx,
                agg[metric].to_numpy(dtype=float),
                color=line_color,
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                label="Window basis" if basis_mode == "window_train_basis" else "Common basis",
            )
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.grid(axis="y", alpha=0.5)
        ax.set_xticks(bx, ["Early", "Middle", "Late"])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_title("Dynamic gain by block", loc="left")
    axes[1].set_title("Dynamic higher-order gain by block", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_family_supplement(family_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH * 2.0, TEXTWIDTH * 0.95))

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
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _mirror_to_drafts(files: Iterable[Path], draft_figures_dir: Path) -> None:
    draft_figures_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        if src.exists():
            shutil.copy2(src, draft_figures_dir / src.name)


def _mirror_with_suffix(files: Iterable[Path], draft_figures_dir: Path, suffix: str) -> None:
    draft_figures_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        if src.exists():
            target = draft_figures_dir / f"{src.stem}{suffix}{src.suffix}"
            shutil.copy2(src, target)


def main() -> None:
    args = parse_args()
    _set_style()

    results_root = args.results_root.resolve()
    data_dir = results_root / "data"
    plots_dir = results_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_root = (args.out_root.resolve() if args.out_root is not None else data_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    draft_figures_dir = args.draft_figures_dir.resolve()

    static_df = pd.read_csv(data_dir / "latent_publication_static_summary.csv")
    stats_df = pd.read_csv(data_dir / "latent_publication_stats_summary.csv")
    family_df = pd.read_csv(data_dir / "latent_publication_family_summary.csv")
    dynamic_df = pd.read_csv(data_dir / "latent_publication_dynamic_summary.csv")

    main_fig = plots_dir / "fig_latent_publication_final.png"
    dynamic_fig = plots_dir / "fig_latent_publication_dynamic_supplement.png"
    family_fig = plots_dir / "fig_latent_publication_family_supplement.png"

    _plot_main_figure(static_df, stats_df, family_df, main_fig)
    _plot_dynamic_supplement(dynamic_df, dynamic_fig)
    _plot_family_supplement(family_df, family_fig)
    generated = [
        main_fig,
        main_fig.with_suffix(".pdf"),
        dynamic_fig,
        dynamic_fig.with_suffix(".pdf"),
        family_fig,
        family_fig.with_suffix(".pdf"),
    ]
    _mirror_to_drafts(generated, draft_figures_dir)
    if "stim_post" in results_root.name:
        _mirror_with_suffix(generated, draft_figures_dir, "_stim_post")

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
                    "message": "Estimated higher-order gain is positive on average",
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
            "dynamic": str(dynamic_fig.relative_to(results_root.parent)),
            "family": str(family_fig.relative_to(results_root.parent)),
        },
        "draft_figure_dir": str(draft_figures_dir),
    }

    panel_map_path = data_dir / "latent_publication_figure_panel_map.json"
    panel_map_path.write_text(json.dumps(panel_map, indent=2))
    (out_root / "latent_publication_figure_panel_map.json").write_text(json.dumps(panel_map, indent=2))


if __name__ == "__main__":
    main()
