"""Fig 1: why three cells fire together more often, the vine edge that separates the explanations, its two
coefficient paths, and what a reversed modulation does to the distribution of the pair.

    draw_fig1_concept.py OUTPUT_PDF SOURCE_CSV

A  four explanations of more frequent three-cell coactivity (schematic).
B  a vine on three variables (schematic).
C  the edge's correlation and its two coefficient paths: true values (lines) and joint fits (circles) of the simulated
   'moving baseline' law of evidence/worked.json, in which the baseline changes in windows 5-8 and the modulation persists.
D  the law of the pair (X, Y) one window before and one window during a reversal of the modulation (b = 0; g = +0.58,
   then -0.58, the 'reversal' law of evidence/worked.json), within the lowest and the highest third of Z and over all Z.
   Z is standard normal; given Z = z, (X, Y) is bivariate normal with standard margins and correlation tanh(b + g z).
   Densities are exact mixtures over z (shaded contours, each panel scaled to its own maximum); Kendall's tau of each is
   estimated from 20,000 simulated draws of that subset (seed 20261126).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 7.25, 6.75
SEED, DRAWS = 20261126, 100_000


def inch(v):
    return v / HEIGHT


def triad(ax, cx, cy, kind):
    """Three pyramidal cells X (left), Y (right) and Z (top) and their links, with the emphasis of one explanation."""
    X, Y, Z = (cx - 0.40, cy - 0.16), (cx + 0.40, cy - 0.16), (cx, cy + 0.32)
    for a, b in ((X, Y), (X, Z), (Y, Z)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=fs.RULE, lw=0.9, zorder=1)
    faces = {"X": "white", "Y": "white", "Z": "white"}
    if kind == "rates":
        faces = {k: fs.ORANGE_TINT for k in faces}
        for (x, y) in (X, Y, Z):
            for j in range(3):
                ax.plot([x + 0.17 + 0.045 * j] * 2, [y + 0.02, y + 0.15], color=fs.ORANGE, lw=1.0, solid_capstyle="round", zorder=4)
    if kind in ("pair", "modulated"):
        ax.plot([X[0], Y[0]], [X[1], Y[1]], color=fs.ORANGE, lw=3.0, solid_capstyle="round", zorder=2)
        faces["X"] = faces["Y"] = fs.ORANGE_TINT
    if kind == "modulated":
        faces["Z"] = fs.ORANGE_TINT
        fs.arrow(ax, (cx, cy + 0.14), (cx, cy - 0.11), color=fs.ORANGE_DARK, lw=1.4, scale=9, zorder=5)
    if kind == "input":
        hub = (cx, cy + 0.0)
        ax.add_patch(plt.Circle(hub, 0.085, facecolor=fs.BLUE, edgecolor="white", lw=0.8, zorder=6))
        for (x, y) in (X, Y, Z):
            v = np.array([x - hub[0], y - hub[1]])
            v = v / np.linalg.norm(v)
            fs.arrow(ax, (hub[0] + 0.10 * v[0], hub[1] + 0.10 * v[1]), (x - 0.13 * v[0], y - 0.13 * v[1]), color=fs.BLUE, lw=1.2, scale=7, zorder=5)
        faces = {k: fs.BLUE_TINT for k in faces}
    for name, (x, y) in zip("XYZ", (X, Y, Z)):
        fs.pyramidal(ax, x, y, 0.13, face=faces[name], edge=fs.INK2, lw=0.9)
        offset = {"X": (-0.19, -0.16), "Y": (0.19, -0.16), "Z": (-0.19, 0.06)}[name]
        ax.text(x + offset[0], y + offset[1], name, ha="center", va="center", fontsize=8, fontstyle="italic", color=fs.INK)


def explanations(fig, records):
    cards = (("rates", "Rates", "each cell fires more", "first order"),
             ("pair", "Pair coupling", "one pair is more\nstrongly associated", "second order"),
             ("modulated", "Modulated coupling", "a pair is more associated\nwhen the third fires", "third order"),
             ("input", "Shared input", "a common input\nrecruits all three", "can mimic third order"))
    for i, (kind, title, text, order) in enumerate(cards):
        x0 = 0.005 + i * 0.25
        ax = fs.canvas(fig, [x0, inch(4.80), 0.235, inch(1.62)], xlim=(0, 1.7))
        top = ax.get_ylim()[1]
        fs.card(ax, 0.0, 0.0, 1.7, top, face=fs.PANEL, radius=0.08)
        triad(ax, 0.85, 0.86, kind)
        ax.text(0.85, top - 0.08, title, ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(0.85, 0.30, text, ha="center", va="center", fontsize=8, color=fs.INK2, linespacing=1.1)
        ax.text(0.85, 0.07, order, ha="center", va="bottom", fontsize=8, color=fs.ORANGE_DARK if kind != "input" else fs.BLUE_DARK,
                fontstyle="italic")
        records.append(("A", kind, title, "", "", ""))


def vine(fig):
    ax = fs.canvas(fig, [0.0, inch(2.55), 0.47, inch(1.95)], xlim=(0, 3.4))
    top = ax.get_ylim()[1]
    band1, band2 = (top * 0.56, top * 0.97), (top * 0.20, top * 0.52)
    for (lo, hi), label in ((band1, "Tree 1"), (band2, "Tree 2")):
        fs.card(ax, 0.0, lo, 3.4, hi - lo, face=fs.PANEL, radius=0.06)
        ax.text(0.10, hi - 0.08, label, ha="left", va="top", fontsize=8, color=fs.INK2, fontweight="bold")
    y1 = (band1[0] + band1[1]) / 2 - 0.03
    xs = {"X": 0.95, "Z": 1.85, "Y": 2.75}
    ax.plot([xs["X"], xs["Y"]], [y1, y1], color=fs.INK2, lw=1.1, zorder=1)
    for name, x in xs.items():
        ax.add_patch(plt.Circle((x, y1), 0.17, facecolor=fs.GREY_LIGHT if name == "Z" else "white", edgecolor=fs.INK2, lw=0.9, zorder=3))
        ax.text(x, y1, name, ha="center", va="center", fontsize=9, fontstyle="italic", zorder=4)
    fs.smath(ax, 1.40, y1 + 0.10, r"$c_{XZ}$", size=9, ha="center", va="bottom", transform=ax.transData)
    fs.smath(ax, 2.30, y1 + 0.10, r"$c_{YZ}$", size=9, ha="center", va="bottom", transform=ax.transData)
    ax.text(1.85, y1 - 0.24, "root", ha="center", va="top", fontsize=8, color=fs.MUTED)
    y2 = (band2[0] + band2[1]) / 2 - 0.02
    left, right = 1.10, 2.60
    ax.plot([left, right], [y2, y2], color=fs.ORANGE, lw=2.6, solid_capstyle="butt", zorder=1)
    for x, text in ((left, r"$X,Z$"), (right, r"$Y,Z$")):
        fs.card(ax, x - 0.30, y2 - 0.14, 0.60, 0.28, face="white", edge=fs.INK2, radius=0.06, lw=0.8, zorder=3)
        fs.smath(ax, x, y2, text, size=9, ha="center", va="center", transform=ax.transData)
    for xa, xb in ((1.40, left), (2.30, right)):
        ax.plot([xa, xb], [y1 - 0.03, y2 + 0.16], color=fs.RULE, lw=0.8, ls=(0, (2, 1.5)), zorder=0)
    fs.smath(ax, 1.85, y2 + 0.10, r"$c_{XY|Z}(\cdot\,;z)$", size=9, ha="center", va="bottom", transform=ax.transData, color=fs.ORANGE_DARK)
    fs.smath(ax, 1.70, top * 0.07, r"$c(u_X,u_Y,u_Z)=c_{XZ}\,c_{YZ}\,c_{XY|Z}$", size=9, ha="center", va="center", transform=ax.transData)


def trajectories(fig, worked, records):
    x0, w = 0.545, 0.445
    head = fig.add_axes([x0, inch(4.05), w, inch(0.42)])
    fs.blank(head)
    fs.smath(head, 0.0, 0.62, r"$\rho_t(z)=\tanh\,(\,b_t+g_t\,z\,)$", size=11, va="center")
    head.text(0.0, 0.05, "correlation of the pair in window t at conditioning value z", fontsize=8, color=fs.INK2, va="center")
    data = worked["examples"]["constant"]
    true, fitted = np.asarray(data["true"]), np.asarray(data["fitted"])
    windows = np.arange(1, len(true) + 1)
    paths = (("baseline", fs.BLUE, fs.BLUE_TINT), ("modulation", fs.ORANGE, fs.ORANGE_TINT))
    for column, (name, colour, tint) in enumerate(paths):
        ax = fig.add_axes([x0 + 0.07, inch(3.22 - 0.68 * column), w - 0.08, inch(0.52)])
        ax.axvspan(4.5, 8.5, color="#f1f1ef", lw=0, zorder=0)
        steps_x = np.repeat(np.arange(0.5, 13.0, 1.0), 2)[1:-1]
        steps_y = np.repeat(true[:, column], 2)
        ax.fill_between(steps_x, 0, steps_y, color=tint, lw=0, zorder=1)
        ax.plot(steps_x, steps_y, color=colour, lw=1.4, zorder=2)
        ax.plot(windows, fitted[:, column], ls="none", marker="o", ms=3.8, mfc="white", mec=colour, mew=1.0, zorder=3)
        ax.axhline(0, color=fs.MUTED, lw=0.5, zorder=1)
        ax.set(xlim=(0.5, 12.5), ylim=(-0.3, 0.72), yticks=[0, 0.5], xticks=[1, 4, 8, 12])
        fs.ticks_minus(ax, "y")
        fs.smath(ax, -0.075, 0.5, r"$b_t$" if column == 0 else r"$g_t$", size=10, ha="right", va="center")
        ax.text(0.015, 0.96, name.capitalize(), transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color=fs.INK2)
        if column == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Window")
        for t in windows:
            records.append(("C", name, int(t), float(true[t - 1, column]), float(fitted[t - 1, column]), ""))
    handles = [Line2D([], [], color=fs.INK2, lw=1.4, label="true"), Line2D([], [], ls="none", marker="o", ms=3.8, mfc="white", mec=fs.INK2, label="fitted")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(x0 + w, inch(3.78)), ncol=2, fontsize=8, borderaxespad=0, handlelength=1.6)


def densities(fig, records):
    rng = np.random.default_rng(SEED)
    thirds = ((0.0, 1 / 3), (2 / 3, 1.0), (0.0, 1.0))
    columns = (("Z in its lowest third", fs.BLUE), ("Z in its highest third", fs.ORANGE), ("all values of Z", fs.MUTED))
    rows = (("window 3", 0.58), ("window 6", -0.58))
    size = 1.02
    x_left = 0.115
    for i, (label, g) in enumerate(rows):
        z_all = rng.standard_normal(DRAWS)
        rho = np.tanh(g * z_all)
        e1, e2 = rng.standard_normal((2, DRAWS))
        xs, ys = e1, rho * e1 + np.sqrt(1 - rho ** 2) * e2
        cut = np.quantile(z_all, [1 / 3, 2 / 3])
        masks = (z_all <= cut[0], z_all >= cut[1], np.ones(DRAWS, dtype=bool))
        for j, ((name, colour), (lo, hi), mask) in enumerate(zip(columns, thirds, masks)):
            q = np.linspace(lo, hi, 62)[1:-1]
            X, Y, D = fs.gaussian_density(lambda z, g=g: np.tanh(g * z), norm.ppf(q))
            ax = fig.add_axes([x_left + j * 0.205, inch(1.18 - 1.12 * i), size / WIDTH, size / HEIGHT])
            fs.density_panel(ax, X, Y, D, colour)
            tau = kendalltau(xs[mask][:20000], ys[mask][:20000])[0]
            value = "0.00" if abs(tau) < 0.005 else f"{tau:+.2f}"
            fs.smath(ax, 0.05, 0.93, rf"$\tau = {value}$", size=8, ha="left", va="top", zorder=8)
            if i == 0:
                ax.set_title(name, fontsize=8, pad=4, color=fs.INK)
            records.append(("D", label, name, float(g), float(tau), int(mask.sum())))
        lab = fig.add_axes([0.0, inch(1.18 - 1.12 * i), 0.10, size / HEIGHT])
        fs.blank(lab)
        lab.text(0.0, 0.62, label, fontsize=8, fontweight="bold", va="center")
        fs.smath(lab, 0.0, 0.38, rf"$g = {g:+.2f}$", size=8, va="center")
    note = fig.add_axes([x_left + 3 * 0.205 - 0.01, inch(0.06), 0.29, inch(2.18)])
    fs.blank(note)
    fs.arrow(note, (0.05, 0.62), (0.05, 0.38), color=fs.INK2, lw=1.0, style="<|-|>", scale=8)
    note.text(0.14, 0.50, "low and high Z\nexchange their\nassociation", fontsize=8, va="center", linespacing=1.15)
    note.text(0.14, 0.13, "over all Z the pair's\nlaw is the same,\nand so are the X, Z\nand Y, Z laws", fontsize=8, va="center", linespacing=1.15,
              color=fs.INK2)


def draw(output, source):
    fs.pro()
    worked = json.loads((HERE / "evidence" / "worked.json").read_text())
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    records = []
    fs.letter_title(fig, 0.0, inch(6.55), "A", "More frequent coactivity of three cells: four explanations")
    explanations(fig, records)
    fs.letter_title(fig, 0.0, inch(4.52), "B", "A vine of conditional pairs")
    vine(fig)
    fs.letter_title(fig, 0.52, inch(4.52), "C", "One edge, two coefficient paths")
    trajectories(fig, worked, records)
    fs.letter_title(fig, 0.0, inch(2.42), "D", "Reversing modulation preserves all three pair distributions")
    densities(fig, records)
    fs.save_pro(fig, output)
    with Path(source).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "quantity", "window_or_set", "value", "value_2", "count"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
