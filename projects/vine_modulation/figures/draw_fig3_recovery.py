"""Fig 3: timing episodes on a known edge, and finding participants, changes and active windows among 50 variables.

    draw_fig3_recovery.py OUTPUT_PDF SOURCE_CSV

A, B  the fitted modulation (slope against the conditioning score) of the four estimators on the first simulated
      dataset of the abrupt and the recurring design at strength 0.6 and 100 training observations per window
      (evidence/figures/known_edge_examples.tsv; true episodes shaded).  A window counts as active when its absolute
      slope reaches half the path's peak; filled bars are active windows.
C     exact recovery of every active and inactive window by design and training size: 80 datasets per point (two
      strengths, 40 seeds), 95% Wilson intervals (evidence/known_edge_timing.csv).
D     the search among 50 variables (schematic).
E     planted episodes whose subset, declared change and active windows were all recovered, by simulated law and
      training size (evidence/worked.json, challenge records).
F     what the search did where nothing was planted (evidence/worked.json and evidence/challenge_terms.tsv).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
HEIGHT = 7.35
METHODS = (("joint", "Joint trajectory", fs.BLUE, "o", "-"), ("gamml", "Additive copula (GAM)", fs.INK2, "D", (0, (1, 1.5))),
           ("switching", "Windowwise switching", fs.ORANGE, "s", (0, (4, 2))), ("smooth_x", "Smooth trajectory", fs.MUTED, "^", (0, (5, 1.5, 1, 1.5))))
DESIGNS = (("abrupt", "One episode,\nchanging background"), ("abrupt_static", "One episode,\nstatic background"),
           ("recurring", "Three recurring episodes,\nchanging background"))


def inch(value):
    return value / HEIGHT


def active(path):
    path = np.abs(np.asarray(path, dtype=float))
    peak = path.max()
    return np.zeros(len(path), dtype=bool) if peak == 0 else path >= 0.5 * peak


def example(fig, rect, rows, records, names=True):
    x0, y0, w, h = rect
    outcomes = [r for r in csv.DictReader((HERE / "evidence" / "known_edge_timing.csv").open())
                if r["design"] == rows[0]["design"] and int(r["n"]) == int(rows[0]["n"])]
    windows = np.array([int(r["window"]) for r in rows])
    truth = np.array([float(r["true_modulation"]) for r in rows]) != 0
    strip = h / (len(METHODS) + 0.55)
    for i, (key, name, colour, marker, _) in enumerate(METHODS):
        ax = fig.add_axes([x0, y0 + h - (i + 1) * strip, w, strip * 0.78])
        path = np.array([float(r[key]) for r in rows])
        on = active(path)
        for t in windows[truth]:
            ax.axvspan(t - 0.5, t + 0.5, color=fs.GREY_LIGHT, lw=0, zorder=0)
        ax.bar(windows, path, width=0.72, color=np.where(on, colour, "white"), edgecolor=colour, linewidth=0.7, zorder=2)
        peak = np.abs(path).max()
        ax.axhline(0.5 * peak, color=fs.INK2, lw=0.6, ls=(0, (1, 1.2)), zorder=3)
        fs.zero(ax)
        ax.set(xlim=(0.4, 24.6), ylim=(-0.25, 0.8), yticks=[], xticks=[1, 8, 16, 24] if i == len(METHODS) - 1 else [])
        fs.spines(ax, ("bottom",) if i == len(METHODS) - 1 else ())
        if i == len(METHODS) - 1:
            ax.set_xlabel("Window", labelpad=1)
        if names:
            ax.text(-0.03, 0.45, name, transform=ax.transAxes, ha="right", va="center", fontsize=fs.SMALL)
        wrong = int((on != truth).sum())
        share = 100 * np.mean([int(r["half_" + key]) for r in outcomes])
        ax.text(1.03, 0.62, "exact" if wrong == 0 else f"{wrong} wrong", transform=ax.transAxes, ha="left",
                va="center", fontsize=fs.SMALL, color=fs.INK, fontweight="bold" if wrong == 0 else "normal")
        ax.text(1.03, 0.10, f"all: {share:.0f}%", transform=ax.transAxes, ha="left", va="center", fontsize=fs.SMALL, color=fs.INK2)
        records.append(("A" if rows[0]["design"] == "abrupt" else "B", rows[0]["design"], key, int(rows[0]["n"]), "", "", "", int(wrong), ";".join(str(int(t)) for t in windows[on])))
    key_ax = fig.add_axes([x0, y0 + h - 0.50 * strip, w, strip * 0.42])
    fs.blank(key_ax, (0.4, 24.6), (0, 1))
    for t in windows[truth]:
        key_ax.add_patch(plt.Rectangle((t - 0.5, 0.15), 1, 0.7, facecolor=fs.GREY_LIGHT, edgecolor="none"))
    if names:
        key_ax.text(-0.03, 0.5, "true episodes", transform=key_ax.transAxes, ha="right", va="center", fontsize=fs.SMALL, color=fs.INK2)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def designs(fig, rect, records):
    x0, y0, w, h = rect
    rows = list(csv.DictReader((HERE / "evidence" / "known_edge_timing.csv").open()))
    sizes = (100, 250, 1000)
    for j, (design, title) in enumerate(DESIGNS):
        ax = fig.add_axes([x0 + j * 0.335 * w + 0.06 * w, y0, 0.25 * w, h * 0.78])
        for m, (key, name, colour, marker, style) in enumerate(METHODS):
            ys, lows, highs = [], [], []
            for n in sizes:
                cell = [int(r["half_" + key]) for r in rows if r["design"] == design and int(r["n"]) == n]
                k, total = sum(cell), len(cell)
                lo, hi = wilson(k, total)
                ys.append(100 * k / total)
                lows.append(100 * lo)
                highs.append(100 * hi)
                records.append(("C", design, key, n, "", k, total, "", ""))
            xs = np.arange(3) + (m - 1.5) * 0.07
            ax.errorbar(xs, ys, yerr=[np.array(ys) - lows, np.array(highs) - ys], color=colour, ls=style, marker=marker, ms=4.2,
                        mfc=colour if key == "joint" else "white", mec=colour, lw=1.2, elinewidth=0.8, capsize=0, zorder=3 - 0.1 * m)
        ax.set(xlim=(-0.3, 2.3), ylim=(0, 104), xticks=range(3), yticks=[0, 25, 50, 75, 100])
        ax.set_xticklabels(["100", "250", "1,000"])
        ax.grid(axis="y", color=fs.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=fs.SMALL + 0.5, pad=3, linespacing=1.05)
        if j == 0:
            ax.set_ylabel("Exact recovery (%)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Training observations per window", labelpad=1, fontsize=fs.SMALL)
    handles = [Line2D([], [], color=c, ls=st, marker=mk, ms=4.2, mfc=c if k == "joint" else "white", mec=c, lw=1.2, label=nm)
               for k, nm, c, mk, st in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y0 + h * 0.97), ncol=4, fontsize=fs.SMALL, handlelength=2.6,
               columnspacing=1.6, borderaxespad=0)


def pipeline(ax):
    fs.blank(ax)
    steps = (("Scan", "candidate features\nof earlier variables"), ("Validate", "representation and\npenalties"),
             ("Confirm", "the assembled vine\non fresh data"), ("Test change", "piecewise against\nconstant, half-samples"),
             ("Score", "held-out density"))
    for i, (name, text) in enumerate(steps):
        y = 0.90 - i * 0.200
        fs.box(ax, 0.02, y - 0.075, 0.30, 0.15, name, face=fs.BLUE_LIGHT if i < 3 else fs.GREY_LIGHT, edge=fs.INK2, size=fs.SMALL, weight="bold", radius=0.02)
        ax.text(0.36, y, text, ha="left", va="center", fontsize=fs.SMALL, linespacing=1.1)
        if i < len(steps) - 1:
            fs.arrow(ax, (0.17, y - 0.08), (0.17, y - 0.125), color=fs.INK2, lw=0.9, scale=7)


def search(ax, worked, records):
    by = defaultdict(lambda: [0, 0])
    for d in worked["challenge"]:
        for t in d["terms"]:
            if t["shape"] in ("a", "b"):
                key = (d["law"], d["n"], t["order"])
                by[key][1] += 1
                by[key][0] += int(t["installed"] and t["changed"] and t["exact"])
    laws = sorted({k[0] for k in by if k[2] == 3})
    ys = {law: i for i, law in enumerate(laws[::-1])}
    for (law, n, order), (k, total) in sorted(by.items()):
        if order != 3:
            continue
        y = ys[law] + (0.17 if n == 1000 else -0.17)
        ax.plot([0, 100 * k / total], [y, y], color=fs.BLUE if n == 1000 else fs.ORANGE, lw=1.0, zorder=2)
        ax.plot(100 * k / total, y, marker="o" if n == 1000 else "s", ms=4.2, mfc=fs.BLUE if n == 1000 else "white",
                mec=fs.BLUE if n == 1000 else fs.ORANGE, zorder=3)
        records.append(("E", law, "", n, order, k, total, "", ""))
    for law, y in ys.items():
        small, large = by[(law, 300, 3)], by[(law, 1000, 3)]
        ax.text(106, y, f"{small[0] + large[0]} of {small[1] + large[1]}", va="center", fontsize=fs.SMALL, color=fs.INK2)
    quartets = [(n, k, total) for (law, n, order), (k, total) in by.items() if order == 4]
    y4 = -1.3
    for n, k, total in sorted(quartets):
        y = y4 + (0.17 if n == 1000 else -0.17)
        ax.plot([0, 100 * k / total], [y, y], color=fs.BLUE if n == 1000 else fs.ORANGE, lw=1.0)
        ax.plot(100 * k / total, y, marker="o" if n == 1000 else "s", ms=4.2, mfc=fs.BLUE if n == 1000 else "white", mec=fs.BLUE if n == 1000 else fs.ORANGE)
        records.append(("E", "quartets", "", n, 4, k, total, "", ""))
    q = dict((n, (k, total)) for n, k, total in quartets)
    ax.text(106, y4, f"{q[300][0] + q[1000][0]} of {q[300][1] + q[1000][1]}", va="center", fontsize=fs.SMALL, color=fs.INK2)
    ax.axhline(-0.65, color=fs.GRID, lw=0.8)
    ax.set_yticks(list(ys.values()) + [y4])
    ax.set_yticklabels([f"{law}" for law in ys] + ["fourth order"])
    ax.tick_params(axis="y", length=0)
    ax.set(xlim=(0, 100), ylim=(y4 - 0.55, len(laws) - 0.45), xticks=[0, 50, 100])
    ax.set_xlabel("Episodes fully recovered (%)", labelpad=1)
    fs.spines(ax, ("bottom",))
    handles = [Line2D([], [], color=fs.BLUE, marker="o", ms=4.2, mfc=fs.BLUE, mec=fs.BLUE, lw=1.0, label="1,000 per window"),
               Line2D([], [], color=fs.ORANGE, marker="s", ms=4.2, mfc="white", mec=fs.ORANGE, lw=1.0, label="300 per window")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, -0.30), ncol=2, fontsize=fs.SMALL, handlelength=1.6,
              columnspacing=0.8, handletextpad=0.4, borderaxespad=0)


def scorecard(ax, worked, records):
    rows = list(csv.DictReader((HERE / "evidence" / "challenge_terms.tsv").open(), delimiter="\t"))
    constant = [t for d in worked["challenge"] for t in d["terms"] if t["shape"] == "constant" and t["order"] == 3]
    controls = [r for r in rows if float(r["strength"]) == 0]
    planted = [r for r in rows if float(r["strength"]) > 0]
    items = (("Persistent third-order terms\ndeclared changing", sum(bool(t["changed"]) for t in constant), len(constant), False),
             ("Control datasets with a\nretained modulated term", sum(int(r["extra_installed"]) > 0 for r in controls), len(controls), False),
             ("Extra terms in planted\ndatasets declared changing", sum(int(r["extra_changing"]) for r in planted), sum(int(r["extra_installed"]) for r in planted), False),
             ("Planted datasets with a\nheld-out density gain", sum(float(r["test_gain"]) > 0 for r in planted), len(planted), True))
    for i, (label, k, total, good) in enumerate(items):
        y = len(items) - 1 - i
        ax.barh(y, 100, height=0.42, color=fs.GREY_LIGHT, edgecolor="none", zorder=1)
        ax.barh(y, 100 * k / max(total, 1), height=0.42, color=fs.BLUE if good else fs.ORANGE, edgecolor="none", zorder=2)
        ax.text(-4, y, label, ha="right", va="center", fontsize=fs.SMALL, linespacing=1.05)
        ax.text(103, y, f"{k} of {total}", ha="left", va="center", fontsize=fs.SMALL, fontweight="bold")
        records.append(("F", label.replace("\n", " "), "", "", "", k, total, "", ""))
    ax.set(xlim=(0, 100), ylim=(-0.6, len(items) - 0.4), xticks=[0, 50, 100], yticks=[])
    ax.set_xlabel("Share (%)", labelpad=1)
    fs.spines(ax, ("bottom",))


def draw(output, source):
    fs.print_style()
    worked = json.loads((HERE / "evidence" / "worked.json").read_text())
    examples = defaultdict(list)
    for r in csv.DictReader((HERE / "evidence" / "figures" / "known_edge_examples.tsv").open(), delimiter="\t"):
        examples[r["design"]].append(r)
    fig = plt.figure(figsize=(fs.WIDTH, HEIGHT))
    records = []
    fs.heading(fig, 0.0, 1 - inch(0.17), "A", "Fitted modulation, one abrupt episode")
    example(fig, [0.175, inch(5.05), 0.30, inch(2.0)], examples["abrupt"], records)
    fs.heading(fig, 0.555, 1 - inch(0.17), "B", "Fitted modulation, recurring episodes")
    example(fig, [0.60, inch(5.05), 0.30, inch(2.0)], examples["recurring"], records, names=False)
    fs.heading(fig, 0.0, inch(4.72), "C", "Exact recovery of every window on a known edge")
    designs(fig, [0.0, inch(2.62), 1.0, inch(1.70)], records)
    fs.heading(fig, 0.0, inch(2.10), "D", "Search among 50 variables")
    pipeline(fig.add_axes([0.0, inch(0.15), 0.31, inch(1.80)]))
    fs.heading(fig, 0.33, inch(2.10), "E", "Planted episodes recovered")
    search(fig.add_axes([0.42, inch(0.62), 0.14, inch(1.38)]), worked, records)
    fs.heading(fig, 0.64, inch(2.10), "F", "Controls and prediction")
    scorecard(fig.add_axes([0.80, inch(0.42), 0.13, inch(1.50)]), worked, records)
    fs.save_journal(fig, output)
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "design_or_check", "method", "training_n", "interaction_order", "successes", "denominator", "wrong_windows", "called_windows"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
