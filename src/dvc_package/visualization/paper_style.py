"""Shared publication style for DVC paper figures.

Provides a consistent NeurIPS-quality visual identity:
- Serif fonts (Times/DejaVu) at sizes that read well at column width
- Wong (2011) colorblind-safe palette
- Standardized method styling across all benchmark figures
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# NeurIPS layout constants
# ---------------------------------------------------------------------------
TEXTWIDTH = 5.5      # inches (NeurIPS single-column text width)
FULL_PAGE_H = 9.0    # inches (usable height)

# ---------------------------------------------------------------------------
# rcParams — apply once at script start via apply_style()
# ---------------------------------------------------------------------------
RCPARAMS: Dict[str, object] = {
    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 8,
    "mathtext.fontset": "dejavuserif",
    # Axes
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Ticks
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    # Lines / markers
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": False,
}

# ---------------------------------------------------------------------------
# Wong (2011) colorblind-safe palette
# ---------------------------------------------------------------------------
COLORS = {
    "black":  "#000000",
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "cyan":   "#56B4E9",
    "yellow": "#F0E442",
    "gray":   "#999999",
}

# Ordered list for cycling through plot elements
COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["cyan"],
    COLORS["yellow"],
]

# ---------------------------------------------------------------------------
# Method styling — consistent across ALL benchmark figures
# ---------------------------------------------------------------------------
METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "Gaussian copula":    {"color": COLORS["blue"],   "ls": "-",  "lw": 1.2, "marker": "o", "ms": 2.5},
    "1-truncated C-vine": {"color": COLORS["cyan"],   "ls": "--", "lw": 1.2, "marker": "s", "ms": 2.5},
    "Graphical Lasso":    {"color": COLORS["green"],  "ls": ":",  "lw": 1.3, "marker": "^", "ms": 2.5},
    "TVGL (Frobenius)":   {"color": COLORS["red"],    "ls": "-.", "lw": 1.2, "marker": "v", "ms": 2.5},
    "Gaussian SSM":       {"color": COLORS["purple"], "ls": "--", "lw": 1.2, "marker": "D", "ms": 2.5},
    "KDE-flow (time BW)": {"color": COLORS["orange"], "ls": "-",  "lw": 1.2, "marker": "x", "ms": 2.5},
}

# Short display names for compact legends and axis labels
SHORT_METHOD_NAMES: Dict[str, str] = {
    "Gaussian copula":    "Gauss. cop.",
    "1-truncated C-vine": "1-trunc.",
    "Graphical Lasso":    "GLasso",
    "TVGL (Frobenius)":   "TVGL",
    "Gaussian SSM":       "Gauss. SSM",
    "KDE-flow (time BW)": "KDE-flow",
}

SHORT_SCENARIO_NAMES: Dict[str, str] = {
    "multiplicative_triplet":       "Mult. triplet",
    "dynamic_tail_df":              "Dyn. tail-DF",
    "tail_switch":                  "Tail switch",
    "hub_switch":                   "Hub switch",
    "agent_interaction_episodes":   "Agent epis.",
}

# Mapping from results-JSON gap keys to display names
GAP_KEY_TO_NAME = {
    "nll_gap":                 "Gaussian copula",
    "nll_gap_truncated_level0": "1-truncated C-vine",
    "nll_gap_glasso":          "Graphical Lasso",
    "nll_gap_tvgl":            "TVGL (Frobenius)",
    "nll_gap_state_space":     "Gaussian SSM",
    "nll_gap_kde_flow":        "KDE-flow (time BW)",
}

# ---------------------------------------------------------------------------
# Episode colors (agent interaction scenario)
# ---------------------------------------------------------------------------
EPISODE_COLORS = {
    0: "#d9d9d9",          # independence (light gray)
    1: COLORS["blue"],     # pairwise
    2: COLORS["red"],      # higher-order
    3: COLORS["purple"],   # mixed
}

EPISODE_NAMES = {
    0: "Independence",
    1: "Pairwise",
    2: "Higher-order",
    3: "Mixed",
}

# ---------------------------------------------------------------------------
# Copula family discrete colormap (for family heatmaps)
# ---------------------------------------------------------------------------
FAMILY_COLORS = {
    "ind":      "#d9d9d9",
    "gaussian": COLORS["blue"],
    "student":  COLORS["cyan"],
    "clayton":  COLORS["orange"],
    "gumbel":   COLORS["green"],
    "frank":    COLORS["purple"],
    "joe":      COLORS["red"],
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def apply_style() -> None:
    """Apply NeurIPS publication rcParams globally."""
    plt.rcParams.update(RCPARAMS)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.06,
    y: float = 1.10,
    fontsize: int = 10,
) -> None:
    """Add bold panel label (A, B, C, ...) relative to an axes."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="right",
    )


def plot_nll_gaps(
    ax: plt.Axes,
    time: np.ndarray,
    gaps: Dict[str, np.ndarray],
    change_point: Optional[int] = None,
    ylabel: str = r"$\Delta$NLL (nats)",
    legend: bool = True,
    markevery: int = 4,
) -> None:
    """Plot NLL gap trajectories for multiple baselines on a single axes.

    Parameters
    ----------
    gaps : dict mapping gap-key (from JSON) to 1-D array of per-timestep gaps
    """
    for key, series in gaps.items():
        name = GAP_KEY_TO_NAME.get(key, key)
        style = METHOD_STYLES.get(name, {})
        ax.plot(
            time[:len(series)],
            series,
            color=style.get("color", COLORS["gray"]),
            ls=style.get("ls", "-"),
            lw=style.get("lw", 1.2),
            marker=style.get("marker", None),
            ms=style.get("ms", 3),
            markevery=markevery,
            label=name,
        )
    ax.axhline(0, color=COLORS["gray"], lw=0.6, ls="--", zorder=0)
    if change_point is not None:
        ax.axvline(time[change_point], color=COLORS["gray"], lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best", ncol=1, handlelength=1.8)


def add_episode_shading(
    ax: plt.Axes,
    time: np.ndarray,
    episode_labels: np.ndarray,
    alpha: float = 0.12,
) -> None:
    """Add light background shading to mark episode types."""
    for t_idx in range(len(episode_labels)):
        ep = int(episode_labels[t_idx])
        if ep == 0:
            continue
        t_lo = time[t_idx] - 0.5 if t_idx > 0 else time[t_idx]
        t_hi = time[t_idx] + 0.5 if t_idx < len(time) - 1 else time[t_idx]
        ax.axvspan(t_lo, t_hi, color=EPISODE_COLORS[ep], alpha=alpha, zorder=0)
