#!/usr/bin/env python
"""
Generate Figure 1: Dynamic Vine Copula (DVC) Overview
=====================================================

Multi-panel schematic for NeurIPS 2026 submission showing:
  A) C-vine tree structure (4 variables, 3 tree levels)
  B) Bivariate copula family gallery (6 density contour plots)
  C) DVC pipeline flow diagram

Usage:
    cd /n/holylabs/kempner_dev/Users/hsafaai/Code/DVC
    PYTHONPATH=src:$PYTHONPATH .venv/bin/python scripts/generate_figure1_overview.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats


# ---------------------------------------------------------------------------
# Global style settings (NeurIPS publication quality)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Computer Modern Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": True,
    "axes.spines.right": True,
})

# Color palette (professional, colorblind-friendly)
COLORS = {
    "node": "#4C72B0",       # steel blue
    "hub": "#C44E52",        # muted red
    "edge_t1": "#4C72B0",    # blue
    "edge_t2": "#DD8452",    # orange
    "edge_t3": "#55A868",    # green
    "box_data": "#E8EAF6",   # light indigo
    "box_proc": "#E3F2FD",   # light blue
    "box_out": "#E8F5E9",    # light green
    "arrow": "#37474F",      # dark grey
    "text": "#212121",       # near black
}


# ===========================================================================
# Panel A: C-vine tree structure
# ===========================================================================
def draw_panel_a(ax):
    """Draw a 4-variable C-vine with 3 tree levels."""
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("C-vine structure (4 variables)", fontsize=10, fontweight="bold",
                 pad=8)

    # ---- Node positions ----
    # Tree 1: hub X1 at center, X2/X3/X4 around it
    t1_hub = (0.5, 0.92)
    t1_nodes = {
        "X1": t1_hub,
        "X2": (0.05, 0.60),
        "X3": (0.50, 0.55),
        "X4": (0.95, 0.60),
    }

    # Tree 2: conditional edges become nodes
    t2_nodes = {
        "12": (0.18, 0.32),
        "13": (0.50, 0.30),
        "14": (0.82, 0.32),
    }

    # Tree 3: single node
    t3_nodes = {
        "13|2": (0.35, 0.07),
        "14|2": (0.65, 0.07),
    }

    node_radius = 0.055

    def draw_node(pos, label, color, fontsize=9, bold=False):
        circle = plt.Circle(pos, node_radius, fc=color, ec="white",
                            linewidth=1.5, zorder=5, alpha=0.92)
        ax.add_patch(circle)
        weight = "bold" if bold else "normal"
        ax.text(pos[0], pos[1], label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight=weight, zorder=6)

    def draw_edge(p1, p2, style="-", color="grey", lw=1.5, label=None,
                  label_offset=(0, 0.04), label_fontsize=7.5,
                  label_color="#333333", label_bg=True):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=style,
                color=color, linewidth=lw, zorder=2, solid_capstyle="round")
        if label:
            mx = (p1[0] + p2[0]) / 2 + label_offset[0]
            my = (p1[1] + p2[1]) / 2 + label_offset[1]
            bbox_props = None
            if label_bg:
                bbox_props = dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85)
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=label_fontsize, color=label_color,
                    fontstyle="italic", bbox=bbox_props, zorder=4)

    # ---- Tree level labels ----
    tree_label_x = -0.12
    ax.text(tree_label_x, 0.75, r"$T_1$", fontsize=11, fontweight="bold",
            color=COLORS["edge_t1"], ha="center", va="center")
    ax.text(tree_label_x, 0.31, r"$T_2$", fontsize=11, fontweight="bold",
            color=COLORS["edge_t2"], ha="center", va="center")
    ax.text(tree_label_x, 0.07, r"$T_3$", fontsize=11, fontweight="bold",
            color=COLORS["edge_t3"], ha="center", va="center")

    # Light horizontal separators
    ax.axhline(y=0.44, xmin=0.0, xmax=1.0, color="#BDBDBD",
               linewidth=0.5, linestyle=":", zorder=1)
    ax.axhline(y=0.19, xmin=0.0, xmax=1.0, color="#BDBDBD",
               linewidth=0.5, linestyle=":", zorder=1)

    # ---- Tree 1 edges ----
    copulas_t1 = {"X2": "Gaussian", "X3": "Clayton", "X4": "Student-t"}
    offsets_t1 = {"X2": (-0.04, 0.03), "X3": (0.06, 0.04), "X4": (0.04, 0.03)}
    for name, pos in t1_nodes.items():
        if name == "X1":
            continue
        draw_edge(t1_hub, pos, style="-", color=COLORS["edge_t1"], lw=2.0,
                  label=copulas_t1[name], label_offset=offsets_t1[name],
                  label_fontsize=7, label_color=COLORS["edge_t1"])

    # ---- Tree 2 edges ----
    edge_labels_t2 = [
        ("12", "13", r"$X_{2},X_{3}|X_{1}$", "Frank"),
        ("13", "14", r"$X_{3},X_{4}|X_{1}$", "Gumbel"),
    ]
    for n1, n2, edge_lbl, cop_lbl in edge_labels_t2:
        p1 = t2_nodes[n1]
        p2 = t2_nodes[n2]
        draw_edge(p1, p2, style="--", color=COLORS["edge_t2"], lw=1.8,
                  label=cop_lbl, label_offset=(0, 0.04),
                  label_fontsize=7, label_color=COLORS["edge_t2"])

    # ---- Tree 3 edge ----
    draw_edge(t3_nodes["13|2"], t3_nodes["14|2"], style=":",
              color=COLORS["edge_t3"], lw=1.8,
              label="Joe", label_offset=(0, 0.04),
              label_fontsize=7, label_color=COLORS["edge_t3"])

    # ---- Draw nodes (on top of edges) ----
    # T1 nodes
    draw_node(t1_hub, r"$X_1$", COLORS["hub"], fontsize=10, bold=True)
    for name in ["X2", "X3", "X4"]:
        subscript = name[1]
        draw_node(t1_nodes[name], f"$X_{subscript}$", COLORS["node"], fontsize=9)

    # T2 nodes
    t2_labels = {
        "12": r"$X_{1}X_{2}$",
        "13": r"$X_{1}X_{3}$",
        "14": r"$X_{1}X_{4}$",
    }
    for name, pos in t2_nodes.items():
        draw_node(pos, t2_labels[name], COLORS["edge_t2"],
                  fontsize=6.5)

    # T3 nodes
    t3_labels = {
        "13|2": r"$23|1$",
        "14|2": r"$34|1$",
    }
    for name, pos in t3_nodes.items():
        draw_node(pos, t3_labels[name], COLORS["edge_t3"], fontsize=6.5)


# ===========================================================================
# Panel B: Copula density gallery
# ===========================================================================

def gaussian_copula_pdf(u, v, rho=0.7):
    """Gaussian copula density."""
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    rho2 = rho ** 2
    return (1.0 / np.sqrt(1 - rho2)) * np.exp(
        -(rho2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho2))
    )


def student_copula_pdf(u, v, rho=0.7, nu=3):
    """Student-t copula density via joint/marginal ratio."""
    x = stats.t.ppf(u, df=nu)
    y = stats.t.ppf(v, df=nu)
    # Bivariate t joint density
    f_joint = stats.multivariate_t.pdf(
        np.column_stack([x.ravel(), y.ravel()]),
        loc=[0, 0],
        shape=[[1, rho], [rho, 1]],
        df=nu,
    ).reshape(x.shape)
    # Marginal densities
    f_x = stats.t.pdf(x, df=nu)
    f_y = stats.t.pdf(y, df=nu)
    # Copula density = joint / product of marginals
    c = f_joint / (f_x * f_y)
    return c


def clayton_copula_pdf(u, v, alpha=2.0):
    """Clayton copula density, alpha > 0."""
    c = (1 + alpha) * (u * v) ** (-(1 + alpha)) * (
        u ** (-alpha) + v ** (-alpha) - 1
    ) ** (-(2 + 1.0 / alpha))
    return c


def frank_copula_pdf(u, v, theta=8.0):
    """Frank copula density."""
    et = np.exp(-theta)
    etu = np.exp(-theta * u)
    etv = np.exp(-theta * v)
    num = -theta * (et - 1) * np.exp(-theta * (u + v))
    den = ((et - 1) + (etu - 1) * (etv - 1)) ** 2
    return num / den


def gumbel_copula_pdf(u, v, theta=2.0):
    """Gumbel copula density via the standard formula."""
    lu = -np.log(u)
    lv = -np.log(v)
    lu_t = lu ** theta
    lv_t = lv ** theta
    s = lu_t + lv_t
    s_inv = s ** (1.0 / theta)
    C = np.exp(-s_inv)
    # c(u,v) = C(u,v) / (u*v) * (lu*lv)^{theta-1} / s^{2-1/theta}
    #          * (s_inv + theta - 1)
    c = C / (u * v) * (lu * lv) ** (theta - 1) / s ** (2 - 1.0 / theta) * (
        s_inv + theta - 1
    )
    return c


def joe_copula_pdf(u, v, theta=3.0):
    """Joe copula density."""
    ubar = 1 - u
    vbar = 1 - v
    ubar_t = ubar ** theta
    vbar_t = vbar ** theta
    S = ubar_t + vbar_t - ubar_t * vbar_t
    c = (
        ubar ** (theta - 1)
        * vbar ** (theta - 1)
        * S ** (1.0 / theta - 2)
        * (theta - 1 + S)
    )
    return c


def draw_panel_b(fig, gs_inner):
    """Draw 2x3 grid of copula density contour plots."""
    copula_specs = [
        ("Gaussian", gaussian_copula_pdf, {"rho": 0.7}, r"$\rho=0.7$"),
        ("Student-t", student_copula_pdf, {"rho": 0.7, "nu": 3}, r"$\rho=0.7,\,\nu=3$"),
        ("Clayton", clayton_copula_pdf, {"alpha": 2.0}, r"$\alpha=2$"),
        ("Frank", frank_copula_pdf, {"theta": 8.0}, r"$\theta=8$"),
        ("Gumbel", gumbel_copula_pdf, {"theta": 2.0}, r"$\theta=2$"),
        ("Joe", joe_copula_pdf, {"theta": 3.0}, r"$\theta=3$"),
    ]

    n_grid = 100
    eps = 0.01
    u_grid = np.linspace(eps, 1 - eps, n_grid)
    v_grid = np.linspace(eps, 1 - eps, n_grid)
    U, V = np.meshgrid(u_grid, v_grid)

    # Determine global color limits for consistent colormap
    all_densities = []
    for name, func, kwargs, _ in copula_specs:
        Z = func(U, V, **kwargs)
        Z = np.clip(Z, 0, 15)  # clip extreme values
        all_densities.append(Z)

    vmin = 0
    vmax = 5.0  # cap for visual clarity -- lower bound shows tail structure

    cmap = plt.cm.viridis

    axes_b = []
    for idx, (name, func, kwargs, param_str) in enumerate(copula_specs):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs_inner[row, col])
        axes_b.append(ax)

        Z = all_densities[idx]
        Z_plot = np.clip(Z, vmin, vmax)

        cf = ax.contourf(U, V, Z_plot, levels=np.linspace(vmin, vmax, 20),
                         cmap=cmap, extend="max")
        ax.contour(U, V, Z_plot, levels=np.linspace(vmin, vmax, 8),
                   colors="k", linewidths=0.3, alpha=0.3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        # Title with family name
        ax.set_title(f"{name}\n{param_str}", fontsize=8, pad=3)

        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

        # Only show x-axis labels on the bottom row
        if row == 1:
            ax.set_xlabel("$u$", fontsize=8)
            ax.set_xticklabels(["0", "0.5", "1"], fontsize=7)
        else:
            ax.set_xticklabels([])

        # Only show y-axis labels on the left column
        if col == 0:
            ax.set_ylabel("$v$", fontsize=8)
            ax.set_yticklabels(["0", "0.5", "1"], fontsize=7)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis="both", which="both", length=2, pad=2)

    # Add a shared colorbar - compute position from the rightmost axes
    fig.canvas.draw()  # force layout so positions are finalized
    # Use the last axis in top-right and bottom-right to span the colorbar
    pos_tr = axes_b[2].get_position()   # top-right subplot
    pos_br = axes_b[5].get_position()   # bottom-right subplot
    cbar_ax = fig.add_axes([
        pos_tr.x1 + 0.005,
        pos_br.y0 + 0.02,
        0.008,
        pos_tr.y1 - pos_br.y0 - 0.04,
    ])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("$c(u,v)$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return axes_b


# ===========================================================================
# Panel C: Pipeline flow diagram
# ===========================================================================
def draw_panel_c(ax):
    """Draw the DVC pipeline flow diagram."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("DVC pipeline", fontsize=10, fontweight="bold", pad=8)

    # Box specifications: (x_center, y_center, width, height, label, color, detail)
    # detail is shown as a smaller line inside the box
    boxes = [
        (0.50, 0.93, 0.75, 0.10,
         "Multivariate time series  $\\mathbf{X}_t$",
         COLORS["box_data"], None),
        (0.50, 0.76, 0.75, 0.10,
         "Rank transform (pseudo-observations)",
         COLORS["box_proc"], r"$\hat{U}_{ij} = R_{ij}/(n{+}1)$"),
        (0.50, 0.57, 0.75, 0.10,
         "Vine structure selection",
         COLORS["box_proc"], "C-vine / D-vine / R-vine"),
        (0.50, 0.38, 0.75, 0.10,
         "Edge-wise copula family fitting",
         COLORS["box_proc"], "AIC / BIC model selection"),
        (0.50, 0.19, 0.75, 0.10,
         "Dynamic vine sequence  $\\{V_t\\}$",
         COLORS["box_proc"], "sliding windows"),
        (0.50, 0.03, 0.75, 0.10,
         "Information readouts",
         COLORS["box_out"], "TC, MI, tail dependence"),
    ]

    for x, y, w, h, label, color, detail in boxes:
        fancy = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.012",
            facecolor=color, edgecolor="#546E7A", linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(fancy)
        if detail:
            # Two lines: main label on top, detail smaller below
            ax.text(x, y + 0.018, label, ha="center", va="center",
                    fontsize=7, color=COLORS["text"], fontweight="medium",
                    zorder=4)
            ax.text(x, y - 0.018, detail, ha="center", va="center",
                    fontsize=6, color="#616161", fontstyle="italic",
                    zorder=4)
        else:
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=7.5, color=COLORS["text"], fontweight="medium",
                    zorder=4)

    # Arrows between boxes
    arrow_style = "Simple,tail_width=2.5,head_width=7,head_length=5"
    arrow_color = COLORS["arrow"]
    box_ys = [b[1] for b in boxes]
    box_h = boxes[0][3]

    for i in range(len(box_ys) - 1):
        y_start = box_ys[i] - box_h / 2
        y_end = box_ys[i + 1] + box_h / 2
        mid_y = (y_start + y_end) / 2
        ax.annotate(
            "",
            xy=(0.50, y_end + 0.003),
            xytext=(0.50, y_start - 0.003),
            arrowprops=dict(
                arrowstyle="-|>",
                color=arrow_color,
                lw=1.2,
                mutation_scale=12,
            ),
            zorder=5,
        )


# ===========================================================================
# Main figure assembly
# ===========================================================================
def main():
    fig = plt.figure(figsize=(14, 4.5))

    # Top-level grid: 3 panels with width ratios
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.5, 0.85],
                  left=0.03, right=0.97, top=0.90, bottom=0.05,
                  wspace=0.30)

    # ---- Panel A: C-vine structure ----
    ax_a = fig.add_subplot(gs[0, 0])
    draw_panel_a(ax_a)

    # ---- Panel B: Copula densities (2x3 sub-grid) ----
    gs_b = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 1],
                                   hspace=0.45, wspace=0.22)
    axes_b = draw_panel_b(fig, gs_b)

    # ---- Panel C: Pipeline ----
    ax_c = fig.add_subplot(gs[0, 2])
    draw_panel_c(ax_c)

    # ---- Panel labels (A, B, C) ----
    label_kwargs = dict(fontsize=16, fontweight="bold", color="black",
                        ha="left", va="top")

    # Panel A label
    ax_a_pos = ax_a.get_position()
    fig.text(ax_a_pos.x0 - 0.01, ax_a_pos.y1 + 0.06, "A", **label_kwargs)

    # Panel B label - use top-left subplot of panel B grid
    ax_b0_pos = axes_b[0].get_position()
    fig.text(ax_b0_pos.x0 - 0.01, ax_b0_pos.y1 + 0.06, "B", **label_kwargs)

    # Panel C label
    ax_c_pos = ax_c.get_position()
    fig.text(ax_c_pos.x0 - 0.01, ax_c_pos.y1 + 0.06, "C", **label_kwargs)

    # ---- Save ----
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "drafts", "figures", "neurips_results",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "figure1_vine_overview.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Figure saved to: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
