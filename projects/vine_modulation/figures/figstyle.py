"""The shared style of the manuscript's seven figures.

Text.  Helvetica, a PDF core font, at 8 to 12 pt at print size (panel letters 12 pt bold, panel titles 9 pt bold, most
labels 8 pt), as PLOS Computational Biology asks; nothing is embedded, and the TIFFs for submission are rasterised
by prepare_figures.py, which checks the font and the sizes.  Formulas are drawn as vector outlines in a sans-serif math
face (smath), so they add no font to the file.  Minus signs are en dashes, which the core font has.

Colour.  Two series colours, checked with a palette validator for a white print surface (lightness band, chroma floor,
separation under simulated protanopia and deuteranopia and for normal vision, contrast): blue #1c5cab and orange
#d95926 pass every check.  Each has a dark shade, a mid tone and a light tint for fills.  Greys carry structure,
models and references.  Colour never identifies a series alone: every series also has its own marker shape or line
style and a legend or a direct label.

Helpers.  pro() sets the style; letter_title() writes a panel letter and title in figure coordinates; canvas() gives
an axis with equal aspect for drawing; card(), arrow(), pyramidal(), rodent() and parity_cube() draw schematic
elements; gaussian_density() and density_panel() draw copula densities; save_pro() writes a PDF without time stamps.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE = "#1c5cab", "#d95926"                           # the two validated series colours
BLUE_DARK, BLUE_MID, BLUE_TINT = "#123f78", "#7ea6dc", "#dde8f6"
ORANGE_DARK, ORANGE_MID, ORANGE_TINT = "#9a3a12", "#f0a27c", "#fbe3d6"
INK, INK2, MUTED = "#1a1a1a", "#4a4a4a", "#8c8c8c"             # text, secondary lines, recessive marks
RULE, GRID, GREY_LIGHT = "#bdbdbd", "#e4e4e4", "#efeeea"       # rules, grid lines, light fills
PANEL = "#f4f4f2"                                             # background of schematic cards


def pro():
    """The print style of the redrawn figures."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica"], "pdf.use14corefonts": True, "axes.unicode_minus": False,
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4, "axes.labelpad": 2,
        "lines.linewidth": 1.3, "lines.markersize": 4.5, "legend.frameon": False, "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.2, "legend.handletextpad": 0.4, "legend.columnspacing": 1.2, "hatch.linewidth": 0.6,
        "savefig.dpi": 300, "text.color": INK, "mathtext.fontset": "dejavusans",
    })


def minus(text):
    """Typographic minus (an en dash, which the core font has) in a number string."""
    return text.replace("-", "–")


def ticks_minus(ax, axis="both"):
    """Tick labels with en-dash minus signs."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    for which in (("x",) if axis == "x" else ("y",) if axis == "y" else ("x", "y")):
        formatter = ScalarFormatter(useOffset=False)
        getattr(ax, f"{which}axis").set_major_formatter(FuncFormatter(lambda v, p, f=formatter: minus(f"{v:g}")))


def letter_title(fig, x, y, letter_text, title_text=None, gap=0.024, size=9):
    """Panel letter (12 pt bold) and an optional title (9 pt bold) on one baseline, in figure coordinates."""
    fig.text(x, y, letter_text, fontsize=12, fontweight="bold", ha="left", va="baseline", color=INK)
    if title_text:
        fig.text(x + gap, y, title_text, fontsize=size, fontweight="bold", ha="left", va="baseline", color=INK)


def smath(ax, x, y, tex, size=9, ha="left", va="baseline", color=INK, transform=None, zorder=6, rotation=0):
    """Mathtext as vector outlines in the sans-serif math face (DejaVu Sans), to sit beside Helvetica text."""
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    from matplotlib import transforms
    import matplotlib as mpl
    with mpl.rc_context({"mathtext.fontset": "dejavusans", "mathtext.default": "it"}):
        path = TextPath((0, 0), tex, size=size, prop=FontProperties(family="DejaVu Sans"))
    box = path.get_extents()
    dx = {"left": -box.x0, "center": -(box.x0 + box.x1) / 2, "right": -box.x1}[ha]
    dy = {"baseline": 0.0, "bottom": -box.y0, "center": -(box.y0 + box.y1) / 2, "top": -box.y1}[va]
    anchor = transforms.ScaledTranslation(x, y, transform if transform is not None else ax.transAxes)
    place = transforms.Affine2D().translate(dx, dy).rotate_deg(rotation).scale(1 / 72) + ax.figure.dpi_scale_trans + anchor
    patch = PathPatch(path, facecolor=color, edgecolor="none", lw=0, transform=place, zorder=zorder, clip_on=False)
    ax.add_artist(patch)
    return patch


def canvas(fig, rect, xlim=(0, 1), ylim=None):
    """An axis for drawing in physical proportions: one data unit is the same length on both axes."""
    ax = fig.add_axes(rect)
    width = rect[2] * fig.get_figwidth()
    height = rect[3] * fig.get_figheight()
    if ylim is None:
        ylim = (0, (xlim[1] - xlim[0]) * height / width)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return ax


def blank(ax, xlim=(0, 1), ylim=(0, 1)):
    """An axis for a schematic: no frame, fixed limits."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def spines(ax, keep=("left", "bottom")):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(side in keep)


def card(ax, x, y, w, h, face=PANEL, edge="none", radius=0.06, lw=0.7, zorder=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=face, edgecolor=edge, linewidth=lw,
                                zorder=zorder))


def arrow(ax, start, end, color=INK2, lw=0.8, style="-|>", scale=7, connection="arc3,rad=0", zorder=3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale, color=color, linewidth=lw, connectionstyle=connection,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def pyramidal(ax, x, y, s, face="white", edge=INK2, lw=0.9, zorder=3):
    """A pyramidal cell: triangular soma, apical dendrite with a fork, two basal dendrites (axis with equal aspect)."""
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon([(x - 0.55 * s, y - 0.42 * s), (x + 0.55 * s, y - 0.42 * s), (x, y + 0.55 * s)], closed=True, facecolor=face,
                         edgecolor=edge, lw=lw, joinstyle="round", zorder=zorder))
    kw = dict(color=edge, lw=lw, solid_capstyle="round", zorder=zorder - 1)
    ax.plot([x, x], [y + 0.55 * s, y + 1.25 * s], **kw)
    ax.plot([x, x - 0.28 * s], [y + 1.0 * s, y + 1.32 * s], **dict(kw, lw=lw * 0.8))
    ax.plot([x, x + 0.26 * s], [y + 1.08 * s, y + 1.38 * s], **dict(kw, lw=lw * 0.8))
    for sx in (-1, 1):
        ax.plot([x + sx * 0.42 * s, x + sx * 0.78 * s], [y - 0.42 * s, y - 0.75 * s], **dict(kw, lw=lw * 0.8))


def rodent(ax, x, y, s, color=MUTED, facing=1, zorder=4):
    """A rat seen from the side (axis with equal aspect): body, head, ear, eye and tail."""
    from matplotlib.patches import Polygon
    f = facing
    ax.add_patch(Ellipse((x, y), 1.7 * s, 0.82 * s, color=color, zorder=zorder))
    ax.add_patch(Polygon([(x + f * 0.55 * s, y + 0.30 * s), (x + f * 1.45 * s, y - 0.05 * s), (x + f * 0.55 * s, y - 0.32 * s)],
                         closed=True, color=color, zorder=zorder, joinstyle="round"))
    ax.add_patch(Ellipse((x + f * 0.70 * s, y + 0.36 * s), 0.30 * s, 0.38 * s, color=color, zorder=zorder))
    ax.add_patch(Circle((x + f * 1.05 * s, y + 0.06 * s), 0.055 * s, color="white", zorder=zorder + 1))
    t = np.linspace(0, 1, 40)
    ax.plot(x - f * (0.80 * s + 1.15 * s * t), y - 0.12 * s + 0.22 * s * np.sin(np.pi * 1.3 * t), color=color, lw=1.1, zorder=zorder - 1,
            solid_capstyle="round")
    for dx in (-0.45, 0.35):
        ax.plot([x + f * dx * s, x + f * (dx + 0.08) * s], [y - 0.38 * s, y - 0.55 * s], color=color, lw=1.1, zorder=zorder - 1,
                solid_capstyle="round")


def parity_cube(ax, x0, y0, s, labels=None, values=None, value_color=INK, size=8, depth=(0.48, 0.34)):
    """The eight patterns (a, b, c) of three binary cells as the corners of a cube; odd patterns orange (+), even blue (-).

    labels: {pattern: text} written beside each corner; values: {pattern: text} written inside the corner disc."""
    dx, dy = depth[0] * s, depth[1] * s

    def at(a, b, c):
        return np.array([x0 + a * s + c * dx, y0 + b * s + c * dy])
    hidden = (0, 0, 1)                                  # with depth drawn up and to the right, the back-bottom-left corner is hidden
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for d in range(3):
                    p = [a, b, c]
                    if p[d] == 0:
                        q = p.copy()
                        q[d] = 1
                        P, Q = at(*p), at(*q)
                        dashed = hidden in (tuple(p), tuple(q))
                        ax.plot([P[0], Q[0]], [P[1], Q[1]], color=RULE, lw=0.9, zorder=1, ls=(0, (2.2, 1.6)) if dashed else "-")
    r = 0.15 * s
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                odd = (a + b + c) % 2 == 1
                P = at(a, b, c)
                ax.add_patch(Circle(P, r, facecolor=ORANGE if odd else BLUE, edgecolor="white", lw=1.0, zorder=3))
                key = f"{a}{b}{c}"
                if values and key in values:
                    ax.text(P[0], P[1], values[key], ha="center", va="center", color="white", fontsize=size, fontweight="bold", zorder=4)
                if labels and key in labels:
                    off = np.array([(-1 if a == 0 else 1) * 0.30 * s, (-1 if b == 0 else 1) * 0.10 * s])
                    ax.text(*(P + off), labels[key], ha="right" if a == 0 else "left", va="center", fontsize=size, color=value_color, zorder=4)
    return at


def gaussian_density(rho_of_z, z_values, grid=np.linspace(-3.2, 3.2, 161)):
    """Density of (X, Y) on a grid for a mixture over z of bivariate normals with standard margins and correlation rho(z)."""
    X, Y = np.meshgrid(grid, grid)
    total = np.zeros_like(X)
    for z in z_values:
        r = float(np.clip(rho_of_z(z), -0.98, 0.98))
        total += np.exp(-(X ** 2 - 2 * r * X * Y + Y ** 2) / (2 * (1 - r ** 2))) / (2 * np.pi * np.sqrt(1 - r ** 2))
    return X, Y, total / len(z_values)


def density_panel(ax, X, Y, D, colour, levels=6):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("tint", ["#ffffff", colour])
    top = D.max()
    lv = np.linspace(top * 0.08, top, levels + 1)
    ax.contourf(X, Y, D, levels=lv, cmap=cmap, extend="neither", zorder=1)
    ax.contour(X, Y, D, levels=lv, colors="white", linewidths=0.5, zorder=2)
    ax.set(xlim=(X.min(), X.max()), ylim=(Y.min(), Y.max()), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color(RULE)
        side.set_linewidth(0.6)


def save_pro(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
