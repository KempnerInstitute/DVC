"""Fig 2: four changes seen from one conditional pair, what the pairwise laws show of a reversal, the binary trio at
fixed pair margins, and the nested comparisons that separate a three-cell term from its change.

    draw_fig2_worked.py OUTPUT_PDF SOURCE_CSV

A-D  conditional Kendall's tau at z = -1 and z = +1 in four simulated laws (evidence/worked.json): true values
     (lines) and fits on 300 observations per window (symbols); windows 5-8 shaded.
E    a fresh simulation of the reversal law (b = 0; g = +0.58, -0.58 in windows 5-8, then +0.58): Z standard normal,
     (X, Y) given Z = z bivariate normal with standard margins and correlation tanh(b + g z); 4,000 draws per
     window, seed 20261126.  Kendall's tau of the three pairs over all draws, and of X and Y within the lowest and the
     highest third of Z.
F    eight spike patterns of three binary cells: 1/8 each before; 1/16 moved from every even to every odd pattern.
G    the nested comparisons (schematic).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
HEIGHT = 6.05
SEED, DRAWS = 20261126, 4000
CASES = (("pair", "A", "Pair changes,\nno modulation"), ("constant", "B", "Baseline moves,\nmodulation persists"),
         ("reversal", "C", "Modulation reverses"), ("quartet", "D", "Fourth-order episode"))


def inch(value):
    return value / HEIGHT


def conditional(theta, feature):
    theta = np.asarray(theta)
    rho = np.clip(np.tanh(theta[:, 0] + feature * theta[:, 1]), -0.98, 0.98)
    return 2 / np.pi * np.arcsin(rho)


def worked_panels(fig, worked, records):
    windows = np.arange(1, 13)
    series = ((-1, fs.BLUE, (0, (3.5, 1.8)), "s", False, "z = -1"), (1, fs.ORANGE, "-", "o", True, "z = +1"))
    for i, (case, letter, title) in enumerate(CASES):
        x0 = 0.075 + i * 0.235
        ax = fig.add_axes([x0, inch(3.92), 0.19, inch(1.30)])
        ax.axvspan(4.5, 8.5, color=fs.GREY_LIGHT, lw=0, zorder=0)
        fs.zero(ax)
        data = worked["examples"][case]
        for value, colour, style, marker, filled, label in series:
            true, fitted = conditional(data["true"], value), conditional(data["fitted"], value)
            ax.step(windows, true, where="mid", color=colour, ls=style, lw=1.4, zorder=2)
            ax.plot(windows + 0.1 * value, fitted, ls="none", marker=marker, ms=3.8, mfc=colour if filled else "white", mec=colour,
                    mew=0.9, zorder=3)
            for t in windows:
                records.append(("A-D", case, label, int(t), float(true[t - 1]), float(fitted[t - 1])))
        ax.set(ylim=(-0.62, 0.62), xlim=(0.5, 12.5), xticks=[1, 4, 8, 12], yticks=[-0.5, 0, 0.5])
        ax.set_xlabel("Window", labelpad=1)
        if i == 0:
            ax.set_ylabel("Conditional Kendall tau")
        else:
            ax.set_yticklabels([])
        fs.heading(fig, x0 - (0.07 if i == 0 else 0.03), inch(5.62), letter)
        fig.text(x0 + 0.095, inch(5.66), title, ha="center", va="top", fontsize=fs.TEXT, fontweight="bold", linespacing=1.1)
    handles = [Line2D([], [], color=c, ls=st, marker=m, ms=3.8, mfc=c if f else "white", mec=c, lw=1.4, label=lab)
               for _, c, st, m, f, lab in series]
    handles += [Line2D([], [], color=fs.INK2, lw=1.4, label="true"), Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.INK2, ms=3.8, label="fitted"),
                Patch(facecolor=fs.GREY_LIGHT, edgecolor="none", label="windows 5-8")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, inch(3.52)), ncol=5, fontsize=fs.SMALL, handlelength=2.2,
               columnspacing=1.4, borderaxespad=0)


def invariance(fig, rect, records):
    """E: a reversal changes the conditional association and none of the three pairwise laws."""
    x0, y0, w, h = rect
    ax = fig.add_axes(rect)
    rng = np.random.default_rng(SEED)
    windows = np.arange(1, 13)
    g = np.where((windows >= 5) & (windows <= 8), -0.58, 0.58)
    out = {k: [] for k in ("xy", "xz", "yz", "low", "high")}
    for gw in g:
        z = rng.standard_normal(DRAWS)
        rho = np.tanh(gw * z)
        e1, e2 = rng.standard_normal((2, DRAWS))
        x, y = e1, rho * e1 + np.sqrt(1 - rho ** 2) * e2
        low, high = np.quantile(z, [1 / 3, 2 / 3])
        out["xy"].append(kendalltau(x, y)[0])
        out["xz"].append(kendalltau(x, z)[0])
        out["yz"].append(kendalltau(y, z)[0])
        out["low"].append(kendalltau(x[z <= low], y[z <= low])[0])
        out["high"].append(kendalltau(x[z >= high], y[z >= high])[0])
    ax.axvspan(4.5, 8.5, color=fs.GREY_LIGHT, lw=0, zorder=0)
    fs.zero(ax)
    styles = (("high", "X, Y in the highest third of Z", fs.ORANGE, "-", "o", True), ("low", "X, Y in the lowest third of Z", fs.BLUE, (0, (3.5, 1.8)), "s", False),
              ("xy", "X, Y over all Z", fs.INK, "-", "^", True), ("xz", "X, Z", fs.INK2, (0, (1, 1.5)), "D", False), ("yz", "Y, Z", fs.MUTED, (0, (5, 1.5, 1, 1.5)), "v", False))
    for key, label, colour, style, marker, filled in styles:
        values = np.asarray(out[key])
        ax.plot(windows, values, color=colour, ls=style, marker=marker, ms=3.6, mfc=colour if filled else "white", mec=colour, lw=1.2, label=label)
        for t in windows:
            records.append(("E", key, label, int(t), float(values[t - 1]), ""))
    ax.set(xlim=(0.5, 12.5), ylim=(-0.5, 0.5), xticks=[1, 4, 8, 12], yticks=[-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xlabel("Window", labelpad=1)
    ax.set_ylabel("Kendall tau")
    last = {k: float(np.mean(np.asarray(v)[-3:])) for k, v in out.items()}
    for key, text, y in (("high", "X, Y given Z high", last["high"]), ("low", "X, Y given Z low", last["low"]),
                         ("xy", "X, Y; X, Z; Y, Z\nover all Z", 0.0)):
        ax.text(12.8, y, text, ha="left", va="center", fontsize=fs.SMALL, color=fs.INK, clip_on=False, linespacing=1.1)


def binary(fig, rect, records):
    """F: moving 1/16 from even to odd patterns keeps every pair and changes the trio."""
    x0, y0, w, h = rect
    ax = fig.add_axes([x0 + 0.11 * w, y0 + 0.34 * h, 0.50 * w, 0.62 * h])
    order = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    xs = np.array([0, 1.3, 2.3, 3.3, 4.6, 5.6, 6.6, 7.9])
    odd = np.array([sum(p) % 2 == 1 for p in order])
    after = np.where(odd, 3 / 16, 1 / 16)
    for x, value, o, pattern in zip(xs, after, odd, order):
        ax.bar(x, value, width=0.74, color=fs.ORANGE if o else fs.BLUE, edgecolor="white", linewidth=0, hatch=None if o else "////", zorder=2)
        records.append(("F", "".join(map(str, pattern)), "after", "", float(value), 0.125))
    for (a, b), value in (((0, 0), 1 / 16), ((1, 3), 3 / 16), ((4, 6), 1 / 16), ((7, 7), 3 / 16)):
        ax.text((xs[a] + xs[b]) / 2, value + 0.005, "3/16" if value > 0.1 else "1/16", ha="center", va="bottom", fontsize=fs.SMALL)
    ax.axhline(1 / 8, color=fs.INK2, lw=1.0, ls=(0, (3, 2)), zorder=3)
    ax.text(-0.5, 1 / 8 + 0.004, "before", ha="left", va="bottom", fontsize=fs.SMALL, color=fs.INK2)
    ax.set(xlim=(-0.6, 8.5), ylim=(0, 0.235), yticks=[0, 1 / 16, 1 / 8, 3 / 16], xticks=[])
    ax.set_yticklabels(["0", "1/16", "1/8", "3/16"])
    ax.set_ylabel("Probability after")
    blend = ax.get_xaxis_transform()
    for row in range(3):
        level = -0.11 - 0.115 * row
        ax.text(-0.9, level, f"cell {row + 1}", transform=blend, ha="right", va="center", fontsize=fs.SMALL, color=fs.INK2, clip_on=False)
        for x, pattern in zip(xs, order):
            ax.plot(x, level, transform=blend, marker="o", ms=4.2, mfc=fs.INK if pattern[row] else "white", mec=fs.INK if pattern[row] else fs.INK2,
                    mew=0.8, clip_on=False)
    note = fig.add_axes([x0 + 0.645 * w, y0 + 0.05 * h, 0.355 * w, 0.92 * h])
    fs.blank(note)
    note.text(0.0, 0.97, "each pair fires together\nwith probability 1/4,\nbefore and after", va="top", fontsize=fs.SMALL, linespacing=1.2)
    fs.math(note, 0.0, 0.55, r"$\mathrm{OR}_{12\,|\,3\,\mathrm{fires}}=9$", size=fs.SMALL, va="center")
    fs.math(note, 0.0, 0.42, r"$\mathrm{OR}_{12\,|\,3\,\mathrm{silent}}=1/9$", size=fs.SMALL, va="center")
    fs.math(note, 0.0, 0.22, r"$\eta=\log 9-\log\,\frac{1}{9}=4.39$", size=fs.TEXT, va="center")


def nested(fig, rect):
    """G: three explanations of the same held-out observations and what the gain of each says."""
    ax = fig.add_axes(rect)
    fs.blank(ax)
    boxes = ((0.00, "1", "Rates and pairs may change;", "no three-cell term"), (0.36, "2", "+ a persistent", "three-cell term"),
             (0.72, "3", "+ a change of the term", "in a proposed interval"))
    for x, number, first, second in boxes:
        fs.box(ax, x, 0.30, 0.27, 0.62, None, face=fs.GREY_LIGHT, edge=fs.INK2, radius=0.02)
        ax.text(x + 0.025, 0.61, number, ha="center", va="center", fontsize=fs.TEXT, fontweight="bold")
        ax.text(x + 0.145, 0.74, first, ha="center", va="center", fontsize=fs.TEXT)
        ax.text(x + 0.145, 0.48, second, ha="center", va="center", fontsize=fs.TEXT)
    for x0, x1, text in ((0.27, 0.36, "gain: a three-cell\nterm exists"), (0.63, 0.72, "gain: the term\nchanges")):
        fs.arrow(ax, (x0 + 0.005, 0.61), (x1 - 0.005, 0.61), color=fs.INK2, lw=1.0, scale=9)
        ax.text((x0 + x1) / 2, 0.14, text, ha="center", va="center", fontsize=fs.SMALL, color=fs.INK, linespacing=1.15)


def draw(output, source):
    fs.print_style()
    worked = json.loads((HERE / "evidence" / "worked.json").read_text())
    fig = plt.figure(figsize=(fs.WIDTH, HEIGHT))
    records = []
    worked_panels(fig, worked, records)
    fs.heading(fig, 0.0, inch(3.10), "E", "A reversal leaves every pairwise law unchanged")
    invariance(fig, [0.075, inch(1.48), 0.235, inch(1.42)], records)
    fs.heading(fig, 0.50, inch(3.10), "F", "Same pairs, different trio")
    binary(fig, [0.50, inch(1.28), 0.50, inch(1.70)], records)
    fs.heading(fig, 0.0, inch(0.98), "G", "Three nested explanations of the same observations")
    nested(fig, [0.03, inch(0.08), 0.94, inch(0.80)])
    fs.save_journal(fig, output)
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "law_or_pair", "series", "window_or_pattern", "value", "value_2"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
