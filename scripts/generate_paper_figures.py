#!/usr/bin/env python3
"""Generate all publication-quality figures for the DVC paper.

Reads experiment results from JSON and produces NeurIPS-ready figures with
consistent styling (serif fonts, colorblind-safe palette, panel labels).

Usage:
    PYTHONPATH=src:$PYTHONPATH python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dvc_package.visualization.paper_style import (
    COLORS,
    COLOR_CYCLE,
    EPISODE_COLORS,
    EPISODE_NAMES,
    FAMILY_COLORS,
    GAP_KEY_TO_NAME,
    METHOD_STYLES,
    RCPARAMS,
    SHORT_METHOD_NAMES,
    SHORT_SCENARIO_NAMES,
    TEXTWIDTH,
    add_episode_shading,
    add_panel_label,
    apply_style,
    plot_nll_gaps,
)

RESULTS_JSON = PROJECT_ROOT / "results" / "simulation_benchmarks" / "simulation_benchmarks_results.json"
ALLEN_VBN_PILOT_FIGURE = PROJECT_ROOT / "results" / "allen_vbn_pilot" / "allen_vbn_pilot_summary.png"
FIGURES_DIR = PROJECT_ROOT / "drafts" / "figures" / "paper"


def _load_results() -> Dict[str, Any]:
    with RESULTS_JSON.open() as f:
        return json.load(f)


def _arr(x: Any) -> np.ndarray:
    """Convert JSON list/scalar to numpy array."""
    return np.asarray(x, dtype=np.float64)


def _collect_nll_gaps(scenario: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract available NLL gap arrays from a scenario dict."""
    gaps = {}
    for key in GAP_KEY_TO_NAME:
        if key in scenario:
            gaps[key] = _arr(scenario[key])
    return gaps


# ==========================================================================
# Figure 2: Higher-Order Structure & Structural Recovery
# ==========================================================================

def _short_legend(ax: plt.Axes, **kwargs: Any) -> None:
    """Replace legend labels with short method names."""
    handles, labels = ax.get_legend_handles_labels()
    short = [SHORT_METHOD_NAMES.get(l, l) for l in labels]
    defaults = dict(fontsize=5.5, handlelength=1.2, handletextpad=0.4,
                    borderpad=0.3, labelspacing=0.3)
    defaults.update(kwargs)
    ax.legend(handles, short, **defaults)


def generate_figure2(results: Dict[str, Any], out_dir: Path) -> Path:
    """Multiplicative triplet (row 1) + hub switch (row 2)."""
    from dvc_package.experiments.simulation_benchmarks import generate_multiplicative_triplet

    mult = results["scenarios"]["multiplicative_triplet"]
    hub = results["scenarios"]["hub_switch"]

    # --- Regenerate scatter data (lightweight, deterministic) ---
    mult_data = generate_multiplicative_triplet(
        n_samples=int(mult["n_samples"]),
        noise_std=float(mult["noise_std"]),
        seed=int(results["seed"]),
    )
    x_all = _arr(mult_data["data"])  # (N, 3): X, Y, Z

    fig = plt.figure(figsize=(TEXTWIDTH, 2.8))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.55, wspace=0.38,
    )

    # --- Row 1: Multiplicative triplet scatterplots ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    X, Y, Z = x_all[:, 0], x_all[:, 1], x_all[:, 2]
    scatter_kw = dict(s=2, alpha=0.2, rasterized=True, edgecolors="none")

    ax_a.scatter(Y, Z, c=COLORS["blue"], **scatter_kw)
    ax_a.set_title("All samples", fontsize=7)
    ax_a.set_xlabel("$Y$")
    ax_a.set_ylabel("$Z$")
    gap_val = float(mult["nll_gap"])
    ax_a.text(
        0.03, 0.97, f"$\\Delta$NLL={gap_val:+.0f}",
        transform=ax_a.transAxes, fontsize=6,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85),
    )

    mask_pos = X > 0
    ax_b.scatter(Y[mask_pos], Z[mask_pos], c=COLORS["orange"], **scatter_kw)
    ax_b.set_title("$X > 0$", fontsize=7)
    ax_b.set_xlabel("$Y$")
    ax_b.set_ylabel("$Z$")

    ax_c.scatter(Y[~mask_pos], Z[~mask_pos], c=COLORS["green"], **scatter_kw)
    ax_c.set_title("$X < 0$", fontsize=7)
    ax_c.set_xlabel("$Y$")
    ax_c.set_ylabel("$Z$")

    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")

    # --- Row 2: Hub switch ---
    ax_d = fig.add_subplot(gs[1, :2])  # hub identity — span 2 cols
    ax_e = fig.add_subplot(gs[1, 2])   # NLL gap

    T = int(hub["n_time_steps"])
    time = np.arange(T)
    cp = int(hub["change_point"])

    hub_data = {
        "True":    (hub["true_hubs"],  COLORS["black"], "-",  "s", 1.8),
        "DVC":     (hub["estimated_hubs"], COLORS["blue"], "-", "o", 1.4),
        "Reg. DVC": (hub.get("regularized_estimated_hubs"), "#8C510A", "-.", "P", 1.2),
        "Corr":    (hub.get("corr_hub_estimated_hubs"), COLORS["orange"], "--", "^", 1.0),
        "GLasso":  (hub.get("glasso_hub_estimated_hubs"), COLORS["green"], ":", "v", 1.0),
        "TVGL":    (hub.get("tvgl_hub_estimated_hubs"), COLORS["purple"], "-.", "D", 1.0),
    }
    for label, (vals, color, ls, marker, lw) in hub_data.items():
        if vals is None:
            continue
        arr = _arr(vals)
        ax_d.plot(time[:len(arr)], arr, color=color, ls=ls, lw=lw,
                  marker=marker, ms=2.5, markevery=3, label=label)

    ax_d.axvline(time[cp], color=COLORS["gray"], lw=0.8, ls="--", alpha=0.7)
    ax_d.set_xlabel("Time step")
    ax_d.set_ylabel("Hub index")
    ax_d.set_title("Hub identity over time", fontsize=7)
    ax_d.legend(fontsize=5, ncol=3, loc="upper center",
                handlelength=1.0, columnspacing=0.6, handletextpad=0.3,
                borderpad=0.2, bbox_to_anchor=(0.5, 1.0))

    # Accuracy annotation — bottom-left, below the transition
    acc_dvc = hub.get("root_recovery_accuracy", 0)
    acc_reg = hub.get("regularized_root_recovery_accuracy", 0)
    acc_corr = hub.get("corr_hub_recovery_accuracy", 0)
    acc_gl = hub.get("glasso_hub_recovery_accuracy", 0)
    acc_tvgl = hub.get("tvgl_hub_recovery_accuracy", 0)
    acc_text = (
        f"Acc: DVC {acc_dvc:.0%}  Reg {acc_reg:.0%}  "
        f"Corr {acc_corr:.0%}  GL {acc_gl:.0%}  TVGL {acc_tvgl:.0%}"
    )
    ax_d.text(
        0.02, 0.03, acc_text, transform=ax_d.transAxes, fontsize=5,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="wheat", ec="0.7", alpha=0.85),
    )
    add_panel_label(ax_d, "D")

    # NLL gap panel
    hub_gaps = _collect_nll_gaps(hub)
    plot_nll_gaps(ax_e, time, hub_gaps, change_point=cp, legend=False, markevery=3)
    ax_e.set_title("NLL gap", fontsize=7)
    _short_legend(ax_e, loc="upper right", ncol=1)
    add_panel_label(ax_e, "E")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.93)
    out = out_dir / "fig2_structure.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Figure 2 saved: {out}")
    return out


# ==========================================================================
# Figure 3: Tail Dynamics
# ==========================================================================

def generate_figure3(results: Dict[str, Any], out_dir: Path) -> Path:
    """Dynamic tail-df (row 1) + tail switch (row 2)."""
    tail_df = results["scenarios"]["dynamic_tail_df"]
    tail_sw = results["scenarios"]["tail_switch"]

    fig = plt.figure(figsize=(TEXTWIDTH, 3.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38)

    # --- Row 1: Dynamic tail-df ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    T1 = int(tail_df["n_time_steps"])
    time1 = np.arange(T1)
    cp1 = int(tail_df["change_point"])

    tail_emp = _arr(tail_df["tail_emp_upper_q95"])
    tail_fit = _arr(tail_df["tail_fit_upper"])
    ax_a.plot(time1, tail_emp, color=COLORS["green"], lw=1.0, label="Emp. $q_{95}$")
    ax_a.plot(time1, tail_fit, color=COLORS["orange"], lw=1.0, label="Fitted $\\lambda$")
    ax_a.axvline(time1[cp1], color=COLORS["gray"], lw=0.8, ls="--", alpha=0.7)
    ax_a.set_xlabel("Time step")
    ax_a.set_ylabel("Tail dep. $\\lambda$")
    ax_a.set_title("Tail dependence trajectory", fontsize=7)
    ax_a.legend(fontsize=5.5, loc="upper right", handlelength=1.2)
    add_panel_label(ax_a, "A")

    gaps1 = _collect_nll_gaps(tail_df)
    plot_nll_gaps(ax_b, time1, gaps1, change_point=cp1, legend=False, markevery=3)
    ax_b.set_title("NLL gap (dyn. tail-DF)", fontsize=7)
    _short_legend(ax_b, loc="upper left", ncol=1)
    add_panel_label(ax_b, "B")

    # --- Row 2: Tail switch ---
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    T2 = int(tail_sw["n_time_steps"])
    time2 = np.arange(T2)
    cp2 = int(tail_sw["change_point"])

    family_codes = np.array(tail_sw["level0_family_codes"])
    codebook = tail_sw["family_codebook"]

    n_fam = len(codebook)
    fam_cmap_colors = []
    for i in range(n_fam):
        fname = codebook[i] if i < len(codebook) else "ind"
        fam_cmap_colors.append(FAMILY_COLORS.get(fname, COLORS["gray"]))
    cmap_fam = mcolors.ListedColormap(fam_cmap_colors)
    bounds = np.arange(-0.5, n_fam + 0.5, 1)
    norm_fam = mcolors.BoundaryNorm(bounds, cmap_fam.N)

    im = ax_c.imshow(
        family_codes.T, aspect="auto", cmap=cmap_fam, norm=norm_fam,
        origin="lower", interpolation="nearest",
    )
    ax_c.axvline(cp2, color=COLORS["black"], lw=0.8, ls="--", alpha=0.7)
    ax_c.set_xlabel("Time step")
    ax_c.set_ylabel("Edge")
    ax_c.set_title("Fitted copula family", fontsize=7)
    cbar = fig.colorbar(im, ax=ax_c, ticks=np.arange(n_fam), shrink=0.75, pad=0.03,
                        aspect=15)
    cbar.ax.set_yticklabels([c[:4] for c in codebook], fontsize=4.5)
    add_panel_label(ax_c, "C")

    gaps2 = _collect_nll_gaps(tail_sw)
    plot_nll_gaps(ax_d, time2, gaps2, change_point=cp2, legend=False, markevery=3)
    ax_d.set_title("NLL gap (tail switch)", fontsize=7)
    _short_legend(ax_d, loc="upper left", ncol=1)
    add_panel_label(ax_d, "D")

    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.94)
    out = out_dir / "fig3_tail_dynamics.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Figure 3 saved: {out}")
    return out


# ==========================================================================
# Figure 4: Agent Interaction Episodes
# ==========================================================================

def generate_figure4(results: Dict[str, Any], out_dir: Path) -> Path:
    """Agent interaction episodes — 5-panel showpiece."""
    ep = results["scenarios"]["agent_interaction_episodes"]

    T = int(ep["n_time_steps"])
    time = np.arange(T)
    episode_labels = _arr(ep["episode_labels"]).astype(int)
    tc_pair = _arr(ep["tc_pairwise"])
    tc_higher = _arr(ep["tc_higher_order"])
    schedule = ep["episode_schedule"]

    fig = plt.figure(figsize=(TEXTWIDTH, 4.0))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.15, 1.0, 1.0],
        hspace=0.55, wspace=0.38,
    )

    # --- Row 1 (A): Ground truth timeline — minimal height ---
    ax_a = fig.add_subplot(gs[0, :])
    etype_map = {"independence": 0, "pairwise": 1, "higher_order": 2, "mixed": 3}
    for entry in schedule:
        t0 = int(entry["t_start"])
        t1 = int(entry["t_end"])
        ecode = etype_map.get(entry["type"], 0)
        ax_a.axvspan(t0 - 0.5, t1 + 0.5, color=EPISODE_COLORS[ecode], alpha=0.5)
        mid_t = (t0 + t1) / 2
        ax_a.text(mid_t, 0.5, EPISODE_NAMES.get(ecode, ""), ha="center", va="center",
                  fontsize=5.5, transform=ax_a.get_xaxis_transform())

    ax_a.set_xlim(-0.5, T - 0.5)
    ax_a.set_yticks([])
    ax_a.spines["left"].set_visible(False)
    ax_a.set_xlabel("Time step", fontsize=7)
    add_panel_label(ax_a, "A", y=1.45)

    # --- Row 2 (B): TC decomposition ---
    ax_b = fig.add_subplot(gs[1, 0])
    tc_pair_plot = np.maximum(tc_pair, 0)
    tc_higher_plot = np.maximum(tc_higher, 0)

    ax_b.fill_between(time, 0, tc_pair_plot, color=COLORS["blue"], alpha=0.45,
                       label="$\\mathrm{TC}_{\\mathrm{pair}}$")
    ax_b.fill_between(time, tc_pair_plot, tc_pair_plot + tc_higher_plot,
                       color=COLORS["red"], alpha=0.45,
                       label="$\\mathrm{TC}_{\\mathrm{higher}}$")
    total = tc_pair_plot + tc_higher_plot
    ax_b.plot(time, total, color=COLORS["black"], lw=0.8, alpha=0.6)
    add_episode_shading(ax_b, time, episode_labels, alpha=0.08)
    ax_b.set_xlabel("Time step")
    ax_b.set_ylabel("TC (nats)")
    ax_b.set_title("TC decomposition", fontsize=7)
    ax_b.legend(fontsize=5.5, loc="upper left", handlelength=1.2,
                borderpad=0.3, labelspacing=0.25)
    add_panel_label(ax_b, "B")

    # --- Row 2 (C): NLL gap vs baselines ---
    ax_c = fig.add_subplot(gs[1, 1])
    ep_gaps = _collect_nll_gaps(ep)
    plot_nll_gaps(ax_c, time, ep_gaps, legend=False, markevery=4)
    add_episode_shading(ax_c, time, episode_labels, alpha=0.08)
    ax_c.set_title("NLL gap vs. baselines", fontsize=7)
    _short_legend(ax_c, loc="upper left", ncol=2, columnspacing=0.5)
    add_panel_label(ax_c, "C")

    # --- Row 3 (D): Binary detection vs order classification ---
    ax_d = fig.add_subplot(gs[2, 0])
    det_metrics = ep.get("method_detection_metrics", {})
    compare_methods = ["DVC", "Regularized DVC"]
    if det_metrics and all(m in det_metrics for m in compare_methods):
        x = np.arange(len(compare_methods))
        width = 0.32
        f1_vals = [det_metrics[m].get("f1", 0.0) for m in compare_methods]
        order_vals = [
            float(ep.get("order_classification_accuracy", 0.0)),
            float(ep.get("regularized_order_classification_accuracy", 0.0)),
        ]

        ax_d.bar(
            x - width / 2,
            f1_vals,
            width,
            color=COLORS["green"],
            alpha=0.75,
            label="Binary F1",
        )
        ax_d.bar(
            x + width / 2,
            order_vals,
            width,
            color=COLORS["red"],
            alpha=0.75,
            label="Order acc.",
        )
        for xpos, val in zip(x - width / 2, f1_vals):
            ax_d.text(xpos, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)
        for xpos, val in zip(x + width / 2, order_vals):
            ax_d.text(xpos, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)

        ax_d.set_xticks(x)
        ax_d.set_xticklabels([SHORT_METHOD_NAMES.get(m, m) for m in compare_methods], fontsize=5.5)
        ax_d.set_ylim(0, 1.08)
        ax_d.set_ylabel("Score")
        ax_d.set_title("Detection vs. order", fontsize=7)
        ax_d.legend(fontsize=5.5, loc="lower left", handlelength=1.2, borderpad=0.3)
    add_panel_label(ax_d, "D")

    # --- Row 3 (E): Detection timeline raster ---
    ax_e = fig.add_subplot(gs[2, 1])
    if det_metrics:
        method_names = list(det_metrics.keys())
        det_matrix = np.zeros((len(method_names), T))
        for i, m in enumerate(method_names):
            detected = det_metrics[m].get("detected", [])
            det_matrix[i, :len(detected)] = detected

        det_cmap = mcolors.ListedColormap(["#f5f5f5", COLORS["green"]])
        ax_e.imshow(det_matrix, aspect="auto", cmap=det_cmap, vmin=0, vmax=1,
                    interpolation="nearest")
        short_m = [SHORT_METHOD_NAMES.get(m, m[:12]) for m in method_names]
        ax_e.set_yticks(range(len(method_names)))
        ax_e.set_yticklabels(short_m, fontsize=5.5)
        ax_e.set_xlabel("Time step")
        ax_e.set_title("Detection timeline", fontsize=7)

        for entry in schedule:
            t0, t1 = int(entry["t_start"]), int(entry["t_end"])
            ecode = etype_map.get(entry["type"], 0)
            if ecode > 0:
                ax_e.axvspan(t0 - 0.5, t1 + 0.5, color=EPISODE_COLORS[ecode],
                             alpha=0.15, zorder=2)
    add_panel_label(ax_e, "E")

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.08, top=0.96)
    out = out_dir / "fig4_agent_episodes.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Figure 4 saved: {out}")
    return out


# ==========================================================================
# Figure 5: Summary Heatmap
# ==========================================================================

def generate_figure5(results: Dict[str, Any], out_dir: Path) -> Path:
    """Scenario x baseline summary heatmap."""
    scenarios = results["scenarios"]

    scenario_names = list(scenarios.keys())
    baseline_keys = list(GAP_KEY_TO_NAME.keys())
    baseline_names = [GAP_KEY_TO_NAME[k] for k in baseline_keys]

    # Short labels for axes
    short_bl = [SHORT_METHOD_NAMES.get(n, n) for n in baseline_names]
    short_sc = [SHORT_SCENARIO_NAMES.get(s, s) for s in scenario_names]

    gap_matrix = np.full((len(scenario_names), len(baseline_keys)), np.nan)
    pos_matrix = np.full((len(scenario_names), len(baseline_keys)), np.nan)

    for i, sname in enumerate(scenario_names):
        sc = scenarios[sname]
        for j, bkey in enumerate(baseline_keys):
            if bkey in sc:
                arr = _arr(sc[bkey])
                gap_matrix[i, j] = float(np.mean(arr))
                pos_matrix[i, j] = float(np.mean(arr > 0))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2))

    # Panel A: Mean NLL gap
    clip_val = 2.0
    gap_display_clipped = np.clip(gap_matrix, -0.5, clip_val)

    im_a = ax_a.imshow(gap_display_clipped, cmap="YlOrBr", vmin=-0.2, vmax=clip_val,
                       aspect="auto", interpolation="nearest")
    for i in range(gap_matrix.shape[0]):
        for j in range(gap_matrix.shape[1]):
            val = gap_matrix[i, j]
            if np.isnan(val):
                continue
            text = f"{val:.2f}" if abs(val) < 10 else f"{val:.0f}"
            color = "white" if gap_display_clipped[i, j] > clip_val * 0.6 else "black"
            ax_a.text(j, i, text, ha="center", va="center", fontsize=5.5, color=color)

    ax_a.set_xticks(range(len(short_bl)))
    ax_a.set_xticklabels(short_bl, fontsize=5, rotation=40, ha="right")
    ax_a.set_yticks(range(len(short_sc)))
    ax_a.set_yticklabels(short_sc, fontsize=6)
    ax_a.set_title("Mean $\\Delta$NLL (baseline$-$DVC)", fontsize=7)
    cbar_a = fig.colorbar(im_a, ax=ax_a, shrink=0.75, pad=0.02, aspect=15)
    cbar_a.ax.tick_params(labelsize=5)
    add_panel_label(ax_a, "A")

    # Panel B: Positive-gap fraction
    im_b = ax_b.imshow(pos_matrix, cmap="Blues", vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
    for i in range(pos_matrix.shape[0]):
        for j in range(pos_matrix.shape[1]):
            val = pos_matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.65 else "black"
            ax_b.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=color)

    ax_b.set_xticks(range(len(short_bl)))
    ax_b.set_xticklabels(short_bl, fontsize=5, rotation=40, ha="right")
    ax_b.set_yticks(range(len(short_sc)))
    ax_b.set_yticklabels(short_sc, fontsize=6)
    ax_b.set_title("Positive-gap fraction", fontsize=7)
    cbar_b = fig.colorbar(im_b, ax=ax_b, shrink=0.75, pad=0.02, aspect=15)
    cbar_b.ax.tick_params(labelsize=5)
    add_panel_label(ax_b, "B")

    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.20, top=0.92, wspace=0.45)
    out = out_dir / "fig5_summary_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Figure 5 saved: {out}")
    return out


# ==========================================================================
# Appendix figures
# ==========================================================================

AUX_RESULTS = {
    "probability": PROJECT_ROOT / "results" / "probability_analysis" / "probability_analysis_multivariate_results.json",
    "entropy": PROJECT_ROOT / "results" / "entropy_analysis" / "entropy_information_estimation_results.json",
    "time_dependent": PROJECT_ROOT / "results" / "time_dependent" / "time_dependent_vine_analysis_results.json",
}


def _load_aux_results(name: str) -> Optional[Dict[str, Any]]:
    """Try loading auxiliary experiment results."""
    p = AUX_RESULTS.get(name)
    if p is None or not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def generate_figS1_probability(out_dir: Path) -> Optional[Path]:
    """Figure S1: correlation heatmap + mutual-information bar chart."""
    prob = _load_aux_results("probability")
    if prob is None:
        print("  Figure S1 skipped (no probability_analysis results found)")
        return None

    dep = prob["probability_analysis"]["dependence_analysis"]
    corr = np.array(dep["correlation_matrix"])
    tau = np.array(dep["kendall_tau_matrix"])
    info = prob["probability_analysis"]["information_measures"]

    d = corr.shape[0]
    var_labels = [f"$X_{{{i}}}$" for i in range(d)]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))

    # Panel A: Correlation + Kendall-tau heatmap (lower tri = corr, upper tri = tau)
    mask_display = np.zeros_like(corr)
    combined = np.tril(corr, -1) + np.triu(tau, 1)
    np.fill_diagonal(combined, 1.0)
    im_a = ax_a.imshow(combined, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for i in range(d):
        for j in range(d):
            val = combined[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax_a.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    ax_a.set_xticks(range(d))
    ax_a.set_xticklabels(var_labels, fontsize=7)
    ax_a.set_yticks(range(d))
    ax_a.set_yticklabels(var_labels, fontsize=7)
    ax_a.set_title("Pearson (lower) / Kendall (upper)", fontsize=8)
    cbar_a = fig.colorbar(im_a, ax=ax_a, shrink=0.8, pad=0.02)
    cbar_a.ax.tick_params(labelsize=5.5)
    add_panel_label(ax_a, "A")

    # Panel B: Mutual information bar chart
    mi_pairs = []
    mi_values = []
    for i in range(d):
        for j in range(i + 1, d):
            key = f"mutual_info_{i}_{j}"
            if key in info:
                mi_pairs.append(f"$X_{{{i}}}$-$X_{{{j}}}$")
                mi_values.append(info[key])
    y_pos = np.arange(len(mi_pairs))
    bars = ax_b.barh(y_pos, mi_values, height=0.5, color=COLORS["blue"], alpha=0.8)
    for i, v in enumerate(mi_values):
        ax_b.text(v + 0.005, y_pos[i], f"{v:.3f}", va="center", fontsize=5.5)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(mi_pairs, fontsize=6.5)
    ax_b.set_xlabel("Mutual information (nats)")
    ax_b.set_title("Pairwise mutual information", fontsize=8)
    add_panel_label(ax_b, "B")

    out = out_dir / "figS1_probability.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Figure S1 saved: {out}")
    return out


def generate_figS2_entropy(out_dir: Path) -> Optional[Path]:
    """Figure S2: entropy method comparison + decomposition."""
    ent = _load_aux_results("entropy")
    if ent is None:
        print("  Figure S2 skipped (no entropy results found)")
        return None

    mc = ent["method_comparison"]
    ea = ent["entropy_analysis"]
    datasets = ["low_entropy", "medium_entropy", "high_entropy", "mixture"]
    ds_labels = ["Low", "Medium", "High", "Mixture"]
    methods = ["gaussian_entropy", "kernel_entropy", "knn_entropy"]
    method_labels = ["Gaussian", "Kernel", "kNN"]
    method_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))

    # Panel A: Grouped bar chart — entropy by method and dataset
    x = np.arange(len(datasets))
    bar_w = 0.22
    for mi, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        vals = [mc[method][ds] for ds in datasets]
        ax_a.bar(x + (mi - 1) * bar_w, vals, bar_w, color=color, alpha=0.8, label=label)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ds_labels, fontsize=7)
    ax_a.set_ylabel("Entropy (nats)")
    ax_a.set_title("Entropy estimates by method", fontsize=8)
    ax_a.legend(fontsize=6, loc="upper left", handlelength=1.2)
    add_panel_label(ax_a, "A")

    # Panel B: Entropy decomposition — marginal sum vs total (interaction info)
    total = [ea[ds]["entropy_decomposition"]["total_entropy"] for ds in datasets]
    marginal_sum = [ea[ds]["entropy_decomposition"]["marginal_sum"] for ds in datasets]
    interaction = [ea[ds]["entropy_decomposition"]["interaction_information"] for ds in datasets]

    x2 = np.arange(len(datasets))
    ax_b.bar(x2 - 0.15, marginal_sum, 0.28, color=COLORS["cyan"], alpha=0.8, label="Marginal sum")
    ax_b.bar(x2 + 0.15, total, 0.28, color=COLORS["purple"], alpha=0.8, label="Joint entropy")
    # Annotate interaction information
    for i, ii in enumerate(interaction):
        y_max = max(marginal_sum[i], total[i])
        ax_b.text(x2[i], y_max + 0.05, f"II={ii:+.2f}", ha="center", fontsize=5.5,
                  color=COLORS["red"])
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(ds_labels, fontsize=7)
    ax_b.set_ylabel("Entropy (nats)")
    ax_b.set_title("Entropy decomposition", fontsize=8)
    ax_b.legend(fontsize=6, loc="upper left", handlelength=1.2)
    add_panel_label(ax_b, "B")

    out = out_dir / "figS2_entropy.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Figure S2 saved: {out}")
    return out


def generate_figS3_time_dependent(out_dir: Path) -> Optional[Path]:
    """Figure S3: correlation evolution + bandwidth evolution."""
    tdep = _load_aux_results("time_dependent")
    if tdep is None:
        print("  Figure S3 skipped (no time_dependent results found)")
        return None

    ca = tdep["correlation_analysis"]
    corr_over_time = np.array(ca["correlations_over_time"])  # (T, d, d)
    T = corr_over_time.shape[0]
    time = np.arange(T)

    # Extract unique pairs from upper triangle
    d = corr_over_time.shape[1]
    pair_labels = []
    pair_series = []
    for i in range(d):
        for j in range(i + 1, d):
            pair_labels.append(f"$\\rho_{{{i},{j}}}$")
            pair_series.append(corr_over_time[:, i, j])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))

    # Panel A: Pairwise correlations over time
    pair_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for k, (label, series) in enumerate(zip(pair_labels, pair_series)):
        color = pair_colors[k % len(pair_colors)]
        ax_a.plot(time, series, color=color, lw=1.2, label=label)
    ax_a.axhline(0, color=COLORS["gray"], lw=0.5, ls="--")
    ax_a.set_xlabel("Time step")
    ax_a.set_ylabel("Pearson correlation")
    ax_a.set_title("Correlation evolution", fontsize=8)
    ax_a.legend(fontsize=6, loc="best", handlelength=1.5)
    add_panel_label(ax_a, "A")

    # Panel B: Bandwidth statistics over time (from model_results)
    mr = tdep["model_results"]
    bw_metrics = mr["bandwidth_metrics"]
    # Extract per-edge mean bandwidths
    edge_labels = []
    edge_bw_means = []
    for key in sorted(bw_metrics.keys()):
        if "mean_bw" in key:
            edge_labels.append(key.replace("_mean_bw", "").replace("tree_", "T").replace("_edge_", "E"))
            vals = bw_metrics[key]
            if isinstance(vals, list):
                edge_bw_means.append(vals)
            else:
                edge_bw_means.append([vals])

    bw_colors = [COLORS["blue"], COLORS["red"], COLORS["green"]]
    if edge_bw_means:
        x_bw = np.arange(len(edge_labels))
        for dim_idx in range(len(edge_bw_means[0])):
            dim_vals = [ebw[dim_idx] if dim_idx < len(ebw) else 0 for ebw in edge_bw_means]
            ax_b.bar(x_bw + dim_idx * 0.3 - 0.15, dim_vals, 0.28,
                     color=bw_colors[dim_idx % len(bw_colors)], alpha=0.8,
                     label=f"Dim {dim_idx}")
        ax_b.set_xticks(x_bw)
        ax_b.set_xticklabels(edge_labels, fontsize=6, rotation=30, ha="right")

    ax_b.set_ylabel("Mean bandwidth")
    ax_b.set_title("KDE-flow learned bandwidths", fontsize=8)
    ax_b.legend(fontsize=6, loc="best", handlelength=1.2)
    add_panel_label(ax_b, "B")

    out = out_dir / "figS3_time_dependent.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Figure S3 saved: {out}")
    return out


def generate_figS4_allen_vbn_pilot(out_dir: Path) -> Optional[Path]:
    """Copy the Allen VBN pilot summary figure into the paper figure set."""
    if not ALLEN_VBN_PILOT_FIGURE.exists():
        print(f"  Skipping Figure S4; missing Allen pilot figure: {ALLEN_VBN_PILOT_FIGURE}")
        return None

    out = out_dir / "figS4_allen_vbn_pilot.png"
    shutil.copy2(ALLEN_VBN_PILOT_FIGURE, out)
    print(f"  Figure S4 saved: {out}")
    return out


def generate_appendix_figures(out_dir: Path) -> List[Path]:
    """Generate appendix figures S1-S4."""
    paths: List[Path] = []
    for gen_func in [
        generate_figS1_probability,
        generate_figS2_entropy,
        generate_figS3_time_dependent,
        generate_figS4_allen_vbn_pilot,
    ]:
        result = gen_func(out_dir)
        if result is not None:
            paths.append(result)
    return paths


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = _load_results()

    print("Generating main-text figures:")
    generate_figure2(results, FIGURES_DIR)
    generate_figure3(results, FIGURES_DIR)
    generate_figure4(results, FIGURES_DIR)
    generate_figure5(results, FIGURES_DIR)

    print("\nGenerating appendix figures:")
    generate_appendix_figures(FIGURES_DIR)

    print(f"\nAll figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
