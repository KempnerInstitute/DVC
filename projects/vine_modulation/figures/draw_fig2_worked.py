"""Fig 2: four changes seen from one conditional pair, what the pairwise laws show of a reversal, the binary trio at
fixed pair margins, and the nested comparisons that separate a three-cell term from its change.

    draw_fig2_worked.py OUTPUT_PDF SOURCE_CSV

A-D  conditional Kendall's tau at z = -1 and z = +1 in four simulated laws (evidence/worked.json): true values
     (lines) and joint fits on 300 observations per window (symbols); windows 5-8 shaded.
E    a fresh simulation of the reversal law (b = 0; g = +0.58, -0.58 in windows 5-8, then +0.58): Z standard normal,
     (X, Y) given Z = z bivariate normal with standard margins and correlation tanh(b + g z); 4,000 draws per
     window, seed 20261126.  Kendall's tau of the three pairs over all draws, and of X and Y within the lowest and the
     highest third of Z.
F    eight spike patterns of three binary cells as the corners of a cube: 1/8 each before; 1/16 moved from every
     pattern with an even number of active cells to every pattern with an odd number.
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
WIDTH, HEIGHT = 7.25, 6.45
SEED, DRAWS = 20261126, 4000
CASES = (("pair", "A", "Pair change"), ("constant", "B", "Moving baseline"), ("reversal", "C", "Reversed modulation"),
         ("quartet", "D", "Fourth-order episode"))
BAND = "#f1f1ef"


def inch(value):
    return value / HEIGHT


def conditional(theta, feature):
    theta = np.asarray(theta)
    rho = np.clip(np.tanh(theta[:, 0] + feature * theta[:, 1]), -0.98, 0.98)
    return 2 / np.pi * np.arcsin(rho)


def worked_panels(fig, worked, records):
    windows = np.arange(1, 13)
    series = ((-1, fs.BLUE, (0, (3.2, 1.6)), "s", False, "z = –1"), (1, fs.ORANGE, "-", "o", True, "z = +1"))
    for i, (case, letter, title) in enumerate(CASES):
        x0 = 0.085 + i * 0.232
        ax = fig.add_axes([x0, inch(4.30), 0.185, inch(1.42)])
        ax.axvspan(4.5, 8.5, color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=fs.MUTED, lw=0.5, zorder=1)
        data = worked["examples"][case]
        for value, colour, style, marker, filled, label in series:
            true, fitted = conditional(data["true"], value), conditional(data["fitted"], value)
            ax.step(windows, true, where="mid", color=colour, ls=style, lw=1.5, zorder=2)
            ax.plot(windows + 0.1 * value, fitted, ls="none", marker=marker, ms=3.9, mfc=colour if filled else "white", mec=colour, mew=0.9, zorder=3)
            for t in windows:
                records.append(("A-D", case, label.replace("–", "-"), int(t), float(true[t - 1]), float(fitted[t - 1])))
        ax.set(ylim=(-0.62, 0.62), xlim=(0.5, 12.5), xticks=[1, 4, 8, 12], yticks=[-0.5, 0, 0.5])
        fs.ticks_minus(ax, "y")
        ax.set_xlabel("Window")
        if i:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left", pad=5)
        fs.letter_title(fig, x0 - (0.075 if i == 0 else 0.03), inch(5.98), letter)
    fs.smath(fig.axes[0], -0.30, 0.5, r"Conditional Kendall $\tau$", size=8, ha="center", va="center", rotation=90)
    handles = [Line2D([], [], color=c, ls=st, marker=m, ms=3.9, mfc=c if f else "white", mec=c, lw=1.5, label=lab) for _, c, st, m, f, lab in series]
    handles += [Line2D([], [], color=fs.INK2, lw=1.5, label="true"), Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.INK2, ms=3.9, label="fitted"),
                Patch(facecolor=BAND, edgecolor="none", label="windows 5–8")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, inch(3.80)), ncol=5, fontsize=8, handlelength=2.2, columnspacing=1.6,
               borderaxespad=0)


def invariance(fig, records):
    ax = fig.add_axes([0.085, inch(1.70), 0.30, inch(1.48)])
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
    ax.axvspan(4.5, 8.5, color=BAND, lw=0, zorder=0)
    ax.axhline(0, color=fs.MUTED, lw=0.5, zorder=1)
    styles = (("high", fs.ORANGE, "-", "o", True), ("low", fs.BLUE, (0, (3.2, 1.6)), "s", False),
              ("xy", fs.INK2, "-", "^", True), ("xz", fs.MUTED, "-", "D", False), ("yz", fs.MUTED, "-", "v", False))
    for key, colour, style, marker, filled in styles:
        values = np.asarray(out[key])
        ax.plot(windows, values, color=colour, ls=style, marker=marker, ms=3.6, mfc=colour if filled else "white", mec=colour, mew=0.8, lw=1.2, zorder=3)
        for t in windows:
            records.append(("E", key, "", int(t), float(values[t - 1]), ""))
    ax.set(xlim=(0.5, 12.5), ylim=(-0.5, 0.5), xticks=[1, 4, 8, 12], yticks=[-0.4, 0, 0.4])
    fs.ticks_minus(ax, "y")
    ax.set_xlabel("Window")
    fs.smath(ax, -0.16, 0.5, r"Kendall $\tau$", size=8, ha="center", va="center", rotation=90)
    last = {k: float(np.mean(np.asarray(v)[-3:])) for k, v in out.items()}
    for key, text, y in (("high", "X, Y with Z high", last["high"]), ("low", "X, Y with Z low", last["low"]), ("xy", "each pair over all Z", 0.0)):
        ax.text(12.9, y, text, ha="left", va="center", fontsize=8, color=fs.INK, clip_on=False)


def cube_panel(fig, records):
    ax = fs.canvas(fig, [0.555, inch(1.32), 0.44, inch(2.02)], xlim=(0, 3.19))
    top = ax.get_ylim()[1]
    s, x0, y0 = 0.86, 0.38, 0.30
    fs.parity_cube(ax, x0, y0, s, size=8)
    at = lambda a, b, c: np.array([x0 + a * s + c * 0.48 * s, y0 + b * s + c * 0.34 * s])
    patterns = ("000", "100", "010", "001", "110", "101", "011", "111")
    for p in patterns:
        a, b, c = (int(ch) for ch in p)
        P = at(a, b, c)
        odd = p.count("1") % 2 == 1
        ax.text(P[0], P[1], "3" if odd else "1", ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=5)
        records.append(("F", p, "after", "", 3 / 16 if odd else 1 / 16, 0.125))
    o, e1, e2, e3 = at(0, 0, 0), at(1, 0, 0), at(0, 1, 0), at(0, 0, 1)
    ax.text((o[0] + e1[0]) / 2, o[1] - 0.19, "cell 1", ha="center", va="top", fontsize=8, color=fs.INK2)
    ax.text(o[0] - 0.19, (o[1] + e2[1]) / 2, "cell 2", ha="center", va="bottom", fontsize=8, color=fs.INK2, rotation=90, rotation_mode="anchor")
    top_left, top_back = at(0, 1, 0), at(0, 1, 1)
    angle = np.degrees(np.arctan2(top_back[1] - top_left[1], top_back[0] - top_left[0]))
    mid = (top_left + top_back) / 2
    ax.text(mid[0] - 0.06, mid[1] + 0.07, "cell 3", ha="center", va="bottom", fontsize=8, color=fs.INK2, rotation=angle, rotation_mode="anchor")
    x_text = 2.05
    handles = [Line2D([], [], ls="none", marker="o", ms=8, mfc=fs.ORANGE, mec="white", label="odd: 3/16"),
               Line2D([], [], ls="none", marker="o", ms=8, mfc=fs.BLUE, mec="white", label="even: 1/16")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(x_text / 3.19, 1.0), fontsize=8, handletextpad=0.3, borderaxespad=0,
              title="after (before: 2/16)", title_fontsize=8, alignment="left")
    ax.text(x_text, top * 0.50, "each pair fires\ntogether with\nprobability 1/4", ha="left", va="center", fontsize=8, linespacing=1.15)
    fs.smath(ax, x_text, top * 0.25, r"$\mathrm{OR}_{12\,|\,3=1}=9$", size=8, transform=ax.transData)
    fs.smath(ax, x_text, top * 0.14, r"$\mathrm{OR}_{12\,|\,3=0}=1/9$", size=8, transform=ax.transData)
    fs.smath(ax, x_text, top * 0.02, r"$\eta=4.39$", size=9, transform=ax.transData)


def nested(fig):
    ax = fs.canvas(fig, [0.03, inch(0.04), 0.94, inch(0.98)], xlim=(0, 6.8))
    top = ax.get_ylim()[1]
    cards = ((0.0, "1", "Rates and pairs", "may change; no three-cell term", "zero"),
             (2.35, "2", "+ a persistent", "three-cell term", "flat"),
             (4.70, "3", "+ a change of that term", "in a proposed interval", "step"))
    for x, number, first, second, shape in cards:
        fs.card(ax, x, 0.08, 2.1, top - 0.16, face=fs.PANEL, radius=0.08)
        ax.add_patch(plt.Circle((x + 0.22, top - 0.26), 0.13, facecolor=fs.INK2, edgecolor="none", zorder=3))
        ax.text(x + 0.22, top - 0.26, number, ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=4)
        ax.text(x + 0.45, top - 0.20, first, ha="left", va="center", fontsize=8, fontweight="bold")
        ax.text(x + 0.45, top - 0.40, second, ha="left", va="center", fontsize=8, color=fs.INK2)
        t = np.linspace(0, 1, 100)
        base = 0.28
        level = {"zero": np.zeros_like(t), "flat": np.full_like(t, 0.16), "step": np.where((t > 0.4) & (t < 0.7), 0.30, 0.16)}[shape]
        ax.add_patch(plt.Rectangle((x + 0.35 + 0.4 * 1.4, base - 0.02), 0.3 * 1.4, 0.36, facecolor=BAND, edgecolor="none", zorder=1))
        ax.plot([x + 0.35, x + 1.75], [base, base], color=fs.MUTED, lw=0.6, zorder=2)
        ax.plot(x + 0.35 + 1.4 * t, base + level, color=fs.ORANGE, lw=1.6, zorder=3)
        fs.smath(ax, x + 1.85, base + 0.02, r"$\eta_t$", size=8, transform=ax.transData)
    for x0, x1, text in ((2.1, 2.35, "gain: a three-cell\nterm exists"), (4.45, 4.70, "gain: the term\nchanges")):
        fs.arrow(ax, (x0 + 0.02, top * 0.55), (x1 - 0.02, top * 0.55), color=fs.INK2, lw=1.0, scale=8, zorder=5)


def draw(output, source):
    fs.pro()
    worked = json.loads((HERE / "evidence" / "worked.json").read_text())
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    records = []
    worked_panels(fig, worked, records)
    fs.letter_title(fig, 0.0, inch(3.42), "E", "A reversal leaves every pairwise law unchanged")
    invariance(fig, records)
    fs.letter_title(fig, 0.555, inch(3.42), "F", "Same pairs, different trio")
    cube_panel(fig, records)
    fs.letter_title(fig, 0.0, inch(1.12), "G", "Three nested explanations of the same observations")
    nested(fig)
    fs.save_pro(fig, output)
    with Path(source).open("w", newline="") as handle:
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
