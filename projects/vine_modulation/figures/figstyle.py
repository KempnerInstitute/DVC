"""The shared style of the manuscript's figures: one palette, one set of sizes and fonts, and small helpers.

Colours.  Two series colours, checked with the data-visualisation validator for a white print surface (OKLCH lightness
band, chroma floor, separation under simulated protanopia and deuteranopia, separation for normal vision, contrast):
blue #1c5cab and orange #d95926 pass every check (worst colour-blind separation 22.9, normal vision 33.1 in OKLab
units x 100).  Colour never carries identity alone: every series also has its own marker shape or line style, and a
legend or a direct label names it.  Grey is reserved for models, references and schematic structure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE = "#1c5cab", "#d95926"
BLUE_LIGHT, ORANGE_LIGHT = "#cde2fb", "#f9d9cb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GREY, GREY_LIGHT, GRID = "#6b6a65", "#efeeea", "#e1e0d9"
FULL, COLUMN = 7.5, 3.6                     # inches; PLOS Computational Biology allows at most 7.5 in


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.3, "lines.markersize": 4.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.handlelength": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "axes.titlepad": 4, "text.color": INK,
        "mathtext.default": "regular",
    })


def letter(ax, text, x=-0.10, y=1.04, fig=None):
    """Panel letter, bold, at the top left of the panel."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)


def title(ax, text, x=0.0):
    ax.set_title(text, loc="left", fontsize=8, color=INK, x=x)


def zero(ax, axis="y", value=0.0):
    """A solid hairline at zero (no dashes: a dashed rule reads as a threshold)."""
    (ax.axhline if axis == "y" else ax.axvline)(value, color=MUTED, lw=0.6, zorder=0)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def box(ax, x, y, w, h, text=None, face=GREY_LIGHT, edge=INK2, size=7, weight="normal", radius=0.02, color=INK, lw=0.6, zorder=2):
    """A rounded box in data coordinates of a schematic axis, with optional centred text."""
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)
    ax.add_patch(patch)
    if text is not None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, color=INK2, lw=0.8, style="-|>", scale=7, connection="arc3,rad=0", zorder=3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale, color=color, linewidth=lw, connectionstyle=connection,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def blank(ax, xlim=(0, 1), ylim=(0, 1)):
    """An axis for a schematic: no frame, fixed limits."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def journal():
    """Use the standard PDF Times family at final print size."""
    apply()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times"], "pdf.use14corefonts": True,
        "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "lines.markersize": 5, "axes.titlepad": 8,
    })


def panel(ax, letter, title):
    ax.set_title(f"{letter}    {title}", loc="left", fontweight="bold", fontsize=11)


def save_journal(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)



# ---------------------------------------------------------------------------------------------------------------------
# Print-size helpers for the redrawn figures (25 November 2026).  Text stays in the Times core fonts at 8 to 12 pt, as
# PLOS requires and prepare_figures.py checks; formulas are drawn as vector outlines (STIX, a Times-compatible face),
# so they add no font to the file.

TEXT, SMALL, LABEL, TITLE, LETTER = 9, 8, 9, 10, 12          # points at print size
WIDTH = 7.25                                                  # inches, the PLOS full-page width used for every figure


def print_style():
    journal()
    plt.rcParams.update({"font.size": TEXT, "axes.labelsize": LABEL, "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
                         "legend.fontsize": SMALL, "axes.titlesize": TITLE, "axes.titlepad": 4, "lines.markersize": 4.5,
                         "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "xtick.major.size": 3, "ytick.major.size": 3, "legend.handlelength": 2.0, "hatch.linewidth": 0.6})


def heading(fig, x, y, letter_text, title_text=None, gap=0.022):
    """Panel letter (bold) and title (bold) in figure coordinates, so that letters line up across a figure."""
    fig.text(x, y, letter_text, fontsize=LETTER, fontweight="bold", ha="left", va="baseline", color=INK)
    if title_text:
        fig.text(x + gap, y, title_text, fontsize=TITLE, fontweight="bold", ha="left", va="baseline", color=INK)


def math(ax, x, y, tex, size=TEXT, ha="left", va="baseline", color=INK, transform=None, zorder=5):
    """Write mathtext as filled outlines at (x, y) in `transform` (axes coordinates by default)."""
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    from matplotlib import transforms
    import matplotlib as mpl
    with mpl.rc_context({"mathtext.fontset": "stix", "mathtext.default": "it"}):
        path = TextPath((0, 0), tex, size=size, prop=FontProperties(family="STIXGeneral"))
    box = path.get_extents()
    dx = {"left": -box.x0, "center": -(box.x0 + box.x1) / 2, "right": -box.x1}[ha]
    dy = {"baseline": 0.0, "bottom": -box.y0, "center": -(box.y0 + box.y1) / 2, "top": -box.y1}[va]
    anchor = transforms.ScaledTranslation(x, y, transform if transform is not None else ax.transAxes)
    place = transforms.Affine2D().translate(dx, dy).scale(1 / 72) + ax.figure.dpi_scale_trans + anchor
    patch = PathPatch(path, facecolor=color, edgecolor="none", lw=0, transform=place, zorder=zorder, clip_on=False)
    ax.add_artist(patch)
    return patch


def neuron(ax, x, y, radius, label=None, face="white", edge=INK2, lw=0.9, size=TEXT, color=INK, zorder=4, italic=True):
    """A cell drawn as a circle with an optional label (in axes data coordinates with equal aspect)."""
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder))
    if label is not None:
        ax.text(x, y, label, ha="center", va="center", fontsize=size, color=color, zorder=zorder + 1,
                fontstyle="italic" if italic else "normal")


def spines(ax, keep=("left", "bottom")):
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(side in keep)
