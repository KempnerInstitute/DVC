"""Fig 1: why three cells fire together more often, the vine edge that separates the explanations, its two
trajectories, and a simulation of what a reversed modulation does to the data.

    draw_fig1_concept.py OUTPUT_PDF SOURCE_CSV

A  four explanations of more frequent three-cell coactivity (schematic).
B  a vine on three variables: two pair copulas with the root, then the pair given the root (schematic).
C  the conditional pair's correlation and its two coefficient paths: true values (lines) and fits (circles) of the
   simulated 'moving baseline' law of evidence/worked.json, in which the baseline changes and the modulation persists.
D  simulated observations of one window before and one after a reversal of the modulation (b = 0, g = +0.58 then
   -0.58, the 'reversal' law of evidence/worked.json): normal scores of X and Y for the lowest and highest thirds of Z
   and for all Z.  Z is standard normal; given Z = z, (X, Y) is bivariate normal with standard margins and correlation
   tanh(b + g z).  2,000 draws per window, seed 20261125; Kendall's tau of each cloud is printed.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
HEIGHT = 6.55
SEED, DRAWS = 20261125, 2000


def inch(value):
    return value / HEIGHT


def tile(ax, kind):
    """One explanation of more frequent three-cell coactivity, as a small network."""
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    X, Y, Z = (0.35, 0.30), (1.25, 0.30), (0.80, 0.98)
    thin = dict(color=fs.MUTED, lw=0.9, zorder=1)
    for a, b in ((X, Y), (X, Z), (Y, Z)):
        ax.plot([a[0], b[0]], [a[1], b[1]], **thin)
    r = 0.15
    face = {k: "white" for k in "XYZ"}
    if kind == "rates":
        face = {k: fs.ORANGE_LIGHT for k in "XYZ"}
        r = 0.19
    if kind in ("pair", "modulation"):
        ax.plot([X[0], Y[0]], [X[1], Y[1]], color=fs.INK, lw=3.2, solid_capstyle="round", zorder=2)
    if kind == "modulation":
        mid = ((X[0] + Y[0]) / 2, X[1] + 0.02)
        fs.arrow(ax, (Z[0], Z[1] - r - 0.02), (mid[0], mid[1] + 0.09), color=fs.ORANGE, lw=1.6, scale=10)
        face["Z"] = fs.ORANGE_LIGHT
    if kind == "input":
        source = (0.80, 1.30)
        ax.add_patch(plt.Rectangle((source[0] - 0.24, source[1] - 0.075), 0.48, 0.15, facecolor=fs.BLUE_LIGHT, edgecolor=fs.BLUE, lw=0.9, zorder=3))
        ax.text(source[0], source[1], "input", ha="center", va="center", fontsize=fs.SMALL, zorder=4)
        ax.set_ylim(0, 1.42)
        for target in (X, Y, Z):
            start = (source[0] + (target[0] - source[0]) * 0.12, source[1] - 0.08)
            fs.arrow(ax, start, (target[0] + (source[0] - target[0]) * 0.18, target[1] + r + 0.01), color=fs.BLUE, lw=1.1, scale=8)
    for name, (x, y) in zip("XYZ", (X, Y, Z)):
        fs.neuron(ax, x, y, r, name, face=face[name], size=fs.TEXT)


def vine(ax):
    """B: tree 1 pairs X and Y with the root Z; tree 2 pairs X with Y given Z."""
    fs.blank(ax, (0, 1), (0, 1))
    ax.text(0.0, 0.83, "tree 1", fontsize=fs.SMALL, color=fs.INK2, va="center")
    ax.text(0.0, 0.43, "tree 2", fontsize=fs.SMALL, color=fs.INK2, va="center")
    xs = {"X": 0.30, "Z": 0.60, "Y": 0.90}
    ax.plot([xs["X"], xs["Y"]], [0.83, 0.83], color=fs.INK2, lw=1.1, zorder=1)
    from matplotlib.patches import Ellipse
    bbox = ax.get_position()
    stretch = (bbox.width * ax.figure.get_figwidth()) / (bbox.height * ax.figure.get_figheight())   # keeps circles round
    for name, x in xs.items():
        ax.add_patch(Ellipse((x, 0.83), 0.075, 0.075 * stretch, facecolor=fs.GREY_LIGHT if name == "Z" else "white",
                             edgecolor=fs.INK2, lw=0.9, zorder=3))
        ax.text(x, 0.83, name, ha="center", va="center", fontsize=fs.TEXT, fontstyle="italic", zorder=4)
    fs.math(ax, 0.45, 0.90, r"$c_{XZ}$", size=fs.TEXT, ha="center", va="bottom", transform=ax.transData)
    fs.math(ax, 0.75, 0.90, r"$c_{YZ}$", size=fs.TEXT, ha="center", va="bottom", transform=ax.transData)
    left, right = (0.38, 0.43), (0.82, 0.43)
    ax.plot([left[0], right[0]], [0.43, 0.43], color=fs.ORANGE, lw=2.6, zorder=1, solid_capstyle="butt")
    for (x, y), text in ((left, r"$XZ$"), (right, r"$YZ$")):
        fs.box(ax, x - 0.075, y - 0.06, 0.15, 0.12, None, face="white", edge=fs.INK2, radius=0.02, zorder=3)
        fs.math(ax, x, y, text, size=fs.TEXT, ha="center", va="center", transform=ax.transData)
    fs.math(ax, 0.60, 0.50, r"$c_{XY\,|\,Z}\,(\,\cdot\,;\,z)$", size=fs.TEXT, ha="center", va="bottom", transform=ax.transData, color=fs.INK)
    for x in (0.45, 0.75):
        fs.arrow(ax, (x, 0.78), ((0.38 if x < 0.6 else 0.82), 0.50), color=fs.MUTED, lw=0.7, scale=6, style="-")
    fs.math(ax, 0.50, 0.17, r"$c(u_X,u_Y,u_Z)=c_{XZ}\;c_{YZ}\;c_{XY|Z}(u_{X|Z},u_{Y|Z};z)$", size=fs.TEXT, ha="center",
            va="center", transform=ax.transData)
    ax.text(0.50, 0.02, "the pair X, Y given Z is one edge of the vine", ha="center", va="bottom", fontsize=fs.SMALL, color=fs.INK2)


def trajectories(fig, rect, worked, records):
    """C: the edge's correlation and the two coefficient paths of the 'moving baseline' law."""
    x0, y0, w, h = rect
    head = fig.add_axes([x0, y0 + h * 0.78, w, h * 0.22])
    fs.blank(head)
    fs.math(head, 0.0, 0.70, r"$\rho_t(z)=\tanh\,(\,b_t+g_t\,z\,)$", size=11, va="center")
    head.text(0.0, 0.18, "correlation of the pair in window t at conditioning value z", fontsize=fs.SMALL, color=fs.INK2, va="center")
    data = worked["examples"]["constant"]
    true, fitted = np.asarray(data["true"]), np.asarray(data["fitted"])
    windows = np.arange(1, len(true) + 1)
    names = (("baseline", 0, "the baseline moves"), ("modulation", 1, "the modulation persists"))
    for k, (name, column, note) in enumerate(names):
        ax = fig.add_axes([x0 + 0.12 * w, y0 + h * (0.42 - 0.40 * k), 0.86 * w, h * 0.30])
        ax.axvspan(4.5, 8.5, color=fs.GREY_LIGHT, lw=0, zorder=0)
        fs.zero(ax)
        ax.step(windows, true[:, column], where="mid", color=fs.INK2, lw=1.4, zorder=2)
        ax.plot(windows, fitted[:, column], ls="none", marker="o", ms=4.2, mfc=fs.BLUE, mec=fs.BLUE, zorder=3)
        ax.set(xlim=(0.5, 12.5), ylim=(-0.35, 0.75), yticks=[0, 0.5], xticks=[1, 4, 8, 12])
        ax.set_yticklabels(["0", "0.5"])
        fs.math(ax, -0.10, 0.5, r"$b_t$" if column == 0 else r"$g_t$", size=fs.TEXT + 1, ha="right", va="center")
        ax.text(0.02, 0.95, note, transform=ax.transAxes, ha="left", va="top", fontsize=fs.SMALL, color=fs.INK)
        if k == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Window", labelpad=1)
        for t in windows:
            records.append(("C", name, int(t), float(true[t - 1, column]), float(fitted[t - 1, column])))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=fs.INK2, lw=1.4, label="true"),
               Line2D([], [], ls="none", marker="o", ms=4.2, mfc=fs.BLUE, mec=fs.BLUE, label="fitted")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(x0 + w, y0 + h * 0.885), ncol=2, fontsize=fs.SMALL,
               handlelength=1.6, columnspacing=1.0, borderaxespad=0)


def simulate(g, rng):
    z = rng.standard_normal(DRAWS)
    rho = np.tanh(g * z)
    e1, e2 = rng.standard_normal((2, DRAWS))
    return z, e1, rho * e1 + np.sqrt(1 - rho ** 2) * e2


def clouds(fig, rect, records):
    """D: a reversed modulation exchanges the clouds at low and high Z and leaves the X, Y cloud as it was."""
    x0, y0, w, h = rect
    rng = np.random.default_rng(SEED)
    rows = (("window 3", 0.58), ("window 6", -0.58))
    columns = (("lowest third of Z", fs.BLUE), ("highest third of Z", fs.ORANGE), ("all values of Z", fs.MUTED))
    size = h * 0.40
    for i, (label, g) in enumerate(rows):
        z, x, y = simulate(g, rng)
        low, high = np.quantile(z, [1 / 3, 2 / 3])
        masks = (z <= low, z >= high, np.ones_like(z, dtype=bool))
        for j, ((name, colour), mask) in enumerate(zip(columns, masks)):
            ax = fig.add_axes([x0 + 0.13 * w + j * 0.215 * w, y0 + h * (0.50 - 0.47 * i), 0.165 * w, size])
            show = np.flatnonzero(mask)[:400]
            ax.plot(x[show], y[show], ls="none", marker="o", ms=1.6, mfc=colour, mec="none", alpha=0.75, rasterized=False)
            tau = kendalltau(x[mask], y[mask])[0]
            ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
            ax.set_aspect("equal")
            fs.spines(ax, ("left", "bottom", "right", "top"))
            for side in ax.spines.values():
                side.set_color(fs.MUTED)
                side.set_linewidth(0.6)
            fs.box(ax, 0.03, 0.80, 0.42, 0.16, None, face="white", edge="none", radius=0.03, zorder=4)
            ax.patches[-1].set_transform(ax.transAxes)
            value = "0.00" if abs(tau) < 0.005 else f"{tau:+.2f}"
            fs.math(ax, 0.06, 0.835, rf"$\tau={value}$", size=fs.SMALL, va="bottom", zorder=6)
            if i == 0:
                ax.set_title(name, fontsize=fs.SMALL, pad=3)
            if i == 1:
                ax.set_xlabel("X", fontstyle="italic", labelpad=1)
            if j == 0:
                ax.set_ylabel("Y", fontstyle="italic", labelpad=1, rotation=0, va="center")
            records.append(("D", label, name, float(g), float(tau), int(mask.sum())))
        lab = fig.add_axes([x0, y0 + h * (0.50 - 0.47 * i), 0.10 * w, size])
        fs.blank(lab)
        lab.text(0.0, 0.62, label, fontsize=fs.TEXT, va="center")
        fs.math(lab, 0.0, 0.36, rf"$g={g:+.2f}$", size=fs.TEXT, va="center")
    note = fig.add_axes([x0 + 0.13 * w + 3 * 0.215 * w - 0.01, y0, 0.24 * w, h])
    fs.blank(note)
    note.text(0.0, 0.78, "Reversing the modulation\nexchanges the clouds at\nlow and high Z ...", fontsize=fs.TEXT, va="center", linespacing=1.3)
    note.text(0.0, 0.33, "... while the cloud of X and\nY over all Z, and the X, Z\nand Y, Z clouds, stay the same", fontsize=fs.TEXT, va="center", linespacing=1.3)


def draw(output, source):
    fs.print_style()
    worked = json.loads((HERE / "evidence" / "worked.json").read_text())
    fig = plt.figure(figsize=(fs.WIDTH, HEIGHT))
    records = []
    fs.heading(fig, 0.0, 1 - inch(0.17), "A", "Three cells fire together more often: four explanations")
    labels = (("rates", "Each cell fires more", "first order"), ("pair", "Stronger association\nin one pair", "second order"),
              ("modulation", "Pair association rises\nwhen the third fires", "third order"), ("input", "Shared input\nrecruits all three", "common input, e.g. replay"))
    for i, (kind, text, order) in enumerate(labels):
        ax = fig.add_axes([0.02 + i * 0.245, inch(4.86), 0.20, inch(1.18)])
        tile(ax, kind)
        cap = fig.add_axes([0.02 + i * 0.245, inch(4.42), 0.22, inch(0.42)])
        fs.blank(cap)
        cap.text(0.42, 0.98, text, ha="center", va="top", fontsize=fs.TEXT, linespacing=1.15)
        cap.text(0.42, 0.0 if "\n" in text else 0.30, order, ha="center", va="bottom", fontsize=fs.SMALL, color=fs.INK2, fontstyle="italic")
    fs.heading(fig, 0.0, inch(4.08), "B", "A vine of conditional pairs")
    vine(fig.add_axes([0.0, inch(2.30), 0.47, inch(1.70)]))
    fs.heading(fig, 0.52, inch(4.08), "C", "One edge, two trajectories")
    trajectories(fig, [0.52, inch(2.36), 0.46, inch(1.66)], worked, records)
    fs.heading(fig, 0.0, inch(2.02), "D", "What a reversed modulation does to the data (simulation)")
    clouds(fig, [0.0, inch(0.02), 1.0, inch(1.92)], records)
    fs.save_journal(fig, output)
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "quantity", "window_or_cloud", "value", "value_2", "count"))
        for r in records:
            writer.writerow(r + ("",) * (6 - len(r)))
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
