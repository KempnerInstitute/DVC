"""Publication-quality plot defaults for DVC figures.

Provides:
- a global rcParams preset (`apply_style`),
- a colorblind-safe palette (Wong, 2011) and a cycling color list,
- a default copula-family color map for heatmaps,
- a small `add_panel_label` helper for panel labels.

Project-specific styling (paper method registries, episode shading,
NLL-gap helpers) lives outside the public package alongside the figure
scripts that use it.
"""

from __future__ import annotations

import logging
from typing import Dict

import matplotlib.pyplot as plt


# Single-column / full-page sizing defaults that work well for two-column
# paper layouts and most slide decks.
TEXTWIDTH = 5.5     # inches
FULL_PAGE_H = 9.0   # inches

RCPARAMS: Dict[str, object] = {
    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 8.4,
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Axes
    "axes.labelsize": 8,
    "axes.titlesize": 8.8,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 4.0,
    "axes.labelpad": 2.0,
    # Ticks
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.98,
    "legend.edgecolor": "0.75",
    "legend.fancybox": False,
    "legend.borderpad": 0.35,
    "legend.borderaxespad": 0.35,
    # Lines / markers
    "lines.linewidth": 2.0,
    "lines.markersize": 4,
    "patch.linewidth": 0.6,
    # Grid
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
    # Figure
    "figure.dpi": 220,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "savefig.transparent": False,
    "figure.constrained_layout.use": False,
}


# Wong (2011) colorblind-safe palette.
COLORS: Dict[str, str] = {
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

COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["cyan"],
    COLORS["yellow"],
]

FAMILY_COLORS: Dict[str, str] = {
    "ind":      "#d9d9d9",
    "gaussian": COLORS["blue"],
    "student":  COLORS["cyan"],
    "clayton":  COLORS["orange"],
    "gumbel":   COLORS["green"],
    "frank":    COLORS["purple"],
    "joe":      COLORS["red"],
}


def apply_style() -> None:
    """Apply the publication rcParams globally."""
    plt.rcParams.update(RCPARAMS)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.06,
    y: float = 1.10,
    fontsize: int = 10,
) -> None:
    """Add a bold panel label (e.g. 'A', 'B', 'C') anchored to an axes."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="right",
    )
