"""Fig 3: timing episodes on a known edge, and finding participants, changes and active windows among 50 variables.

    draw_fig3_recovery.py OUTPUT_PDF SOURCE_CSV

A, B  the fitted modulation (slope of the fitted dependence against the known conditioning score) of the four estimators on
      the first simulated dataset of the single-episode and the recurring design at strength 0.6 and 100 training
      observations per window (evidence/figures/known_edge_examples.tsv), window by window; outlined cells are the true
      episodes and dots the windows an estimator calls active (absolute slope at least half the path's peak).
C     exact recovery of every active and inactive window by design and training size: 80 datasets per point (two
      strengths, 40 seeds), 95% Wilson intervals (evidence/known_edge_timing.csv).
D     the search among 50 variables (schematic).
E     planted episodes whose subset, declared change and active windows were all recovered, by simulated law and
      training size (evidence/worked.json, challenge records).
F     what the search did where nothing was planted, and held-out prediction (evidence/worked.json and
      evidence/challenge_terms.tsv).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 7.25, 7.30
METHODS = (("joint", "Joint trajectory", fs.BLUE, "o", "-"), ("gamml", "Additive copula", fs.INK2, "D", (0, (1, 1.4))),
           ("switching", "Windowwise switching", fs.ORANGE, "s", (0, (4, 2))), ("smooth_x", "Smooth trajectory", fs.MUTED, "^", (0, (5, 1.5, 1, 1.5))))
DESIGNS = (("abrupt", "One episode"), ("abrupt_static", "One episode, static background"), ("recurring", "Three recurring episodes"))
SLOPES = LinearSegmentedColormap.from_list("slope", [(0.0, fs.BLUE_MID), (0.5, "#ffffff"), (0.68, fs.ORANGE_TINT), (0.86, fs.ORANGE), (1.0, fs.ORANGE_DARK)])


def inch(value):
    return value / HEIGHT


def active(path):
    path = np.abs(np.asarray(path, dtype=float))
    peak = path.max()
    return np.zeros(len(path), dtype=bool) if peak == 0 else path >= 0.5 * peak


def example(fig, rect, rows, records, names=True, letter="A"):
    x0, y0, w, h = rect
    outcomes = [r for r in csv.DictReader((HERE / "evidence" / "known_edge_timing.csv").open())
                if r["design"] == rows[0]["design"] and int(r["n"]) == int(rows[0]["n"])]
    windows = np.array([int(r["window"]) for r in rows])
    truth = np.array([float(r["true_modulation"]) for r in rows])
    on_true = truth != 0
    matrix = [truth] + [np.array([float(r[key]) for r in rows]) for key, *_ in METHODS]
    ax = fig.add_axes([x0, y0, w, h])
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.3, vmax=0.75)
    ax.imshow(np.vstack(matrix), aspect="auto", cmap=SLOPES, norm=norm, interpolation="nearest", extent=(0.5, len(windows) + 0.5, len(matrix) - 0.5, -0.5))
    for i in range(1, len(matrix)):
        on = active(matrix[i])
        ax.plot(windows[on], np.full(on.sum(), i), ls="none", marker="o", ms=2.6, mfc=fs.INK, mec="none")
    for t in windows[on_true]:
        for i in range(len(matrix)):
            ax.add_patch(plt.Rectangle((t - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=fs.INK, lw=0.5, zorder=3))
    ax.axhline(0.5, color="white", lw=2.2)
    ax.set(xticks=[1, 8, 16, 24], yticks=range(len(matrix)), xlim=(0.5, len(windows) + 0.5))
    ax.set_yticklabels(["truth"] + [m[1] for m in METHODS] if names else [""] * len(matrix), fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Window")
    for side in ax.spines.values():
        side.set_visible(False)
    for i, (key, name, colour, marker, _) in enumerate(METHODS, 1):
        on = active(matrix[i])
        wrong = int((on != on_true).sum())
        share = 100 * np.mean([int(r["half_" + key]) for r in outcomes])
        ax.text(1.02, 1 - (i + 0.5) / len(matrix), "exact" if wrong == 0 else f"{wrong} wrong", transform=ax.transAxes, ha="left", va="center",
                fontsize=8, fontweight="bold" if wrong == 0 else "normal")
        ax.text(1.42, 1 - (i + 0.5) / len(matrix), f"{share:.0f}%", transform=ax.transAxes, ha="right", va="center", fontsize=8, color=fs.INK2)
        records.append((letter, rows[0]["design"], key, int(rows[0]["n"]), "", "", "", wrong, ";".join(str(int(t)) for t in windows[on])))
    ax.text(1.02, 1 - 0.5 / len(matrix), "here", transform=ax.transAxes, ha="left", va="center", fontsize=8, color=fs.INK2)
    ax.text(1.42, 1 - 0.5 / len(matrix), "all 80", transform=ax.transAxes, ha="right", va="center", fontsize=8, color=fs.INK2)
    return ax


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def designs(fig, records):
    rows = list(csv.DictReader((HERE / "evidence" / "known_edge_timing.csv").open()))
    sizes = (100, 250, 1000)
    for j, (design, title) in enumerate(DESIGNS):
        ax = fig.add_axes([0.085 + j * 0.31, inch(2.88), 0.25, inch(1.30)])
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
            xs = np.arange(3) + (m - 1.5) * 0.08
            ax.errorbar(xs, ys, yerr=[np.array(ys) - lows, np.array(highs) - ys], color=colour, ls=style, marker=marker, ms=4.4,
                        mfc=colour if key == "joint" else "white", mec=colour, mew=1.0, lw=1.3, elinewidth=0.8, capsize=0, zorder=3 - 0.1 * m)
        ax.set(xlim=(-0.3, 2.3), ylim=(0, 105), xticks=range(3), yticks=[0, 50, 100])
        ax.set_xticklabels(["100", "250", "1,000"])
        ax.grid(axis="y", color=fs.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left", pad=4)
        if j == 0:
            ax.set_ylabel("Exact recovery (%)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Observations per window")
    handles = [Line2D([], [], color=c, ls=st, marker=mk, ms=4.4, mfc=c if k == "joint" else "white", mec=c, lw=1.3, label=nm)
               for k, nm, c, mk, st in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, inch(4.38)), ncol=4, fontsize=8, handlelength=2.6, columnspacing=1.8,
               borderaxespad=0)


def pipeline(fig):
    ax = fs.canvas(fig, [0.0, inch(0.10), 0.30, inch(1.95)], xlim=(0, 2.175))
    top = ax.get_ylim()[1]
    steps = (("Scan", "candidate features"), ("Validate", "form and penalties"), ("Confirm", "assembled vine"), ("Test change", "against a constant"),
             ("Score", "held-out density"))
    gap = (top - 0.10) / len(steps)
    for i, (name, text) in enumerate(steps):
        y = top - 0.05 - (i + 0.5) * gap
        fs.card(ax, 0.05, y - 0.15, 0.95, 0.30, face=fs.BLUE_TINT if i < 3 else fs.PANEL, edge=fs.BLUE if i < 3 else fs.MUTED, radius=0.06, lw=0.7)
        ax.text(0.525, y, name, ha="center", va="center", fontsize=8, fontweight="bold", color=fs.BLUE_DARK if i < 3 else fs.INK)
        ax.text(1.10, y, text, ha="left", va="center", fontsize=8, color=fs.INK2)
        if i < len(steps) - 1:
            fs.arrow(ax, (0.525, y - 0.155), (0.525, y - gap + 0.155), color=fs.INK2, lw=0.8, scale=6)


def search(fig, worked, records):
    ax = fig.add_axes([0.42, inch(0.62), 0.16, inch(1.42)])
    by = defaultdict(lambda: [0, 0])
    for d in worked["challenge"]:
        for t in d["terms"]:
            if t["shape"] in ("a", "b"):
                key = (d["law"], d["n"], t["order"])
                by[key][1] += 1
                by[key][0] += int(t["installed"] and t["changed"] and t["exact"])
    laws = sorted({k[0] for k in by if k[2] == 3})
    ys = {law: i for i, law in enumerate(laws[::-1])}
    y4 = -1.3
    for (law, n, order), (k, total) in sorted(by.items()):
        y = (ys[law] if order == 3 else y4) + (0.17 if n == 1000 else -0.17)
        colour = fs.BLUE if n == 1000 else fs.ORANGE
        ax.plot([0, 100 * k / total], [y, y], color=colour, lw=1.0, zorder=2)
        ax.plot(100 * k / total, y, marker="o" if n == 1000 else "s", ms=4.2, mfc=colour if n == 1000 else "white", mec=colour, mew=0.9, zorder=3)
        records.append(("E", law if order == 3 else "quartets", "", n, order, k, total, "", ""))
    for law, y in list(ys.items()) + [("fourth order", y4)]:
        key = (law, 3) if law != "fourth order" else None
        small = by[(law, 300, 3)] if key else [sum(v[0] for (l, n, o), v in by.items() if o == 4 and n == 300), 0]
        large = by[(law, 1000, 3)] if key else [sum(v[0] for (l, n, o), v in by.items() if o == 4 and n == 1000), 0]
        total = (small[1] + large[1]) if key else sum(v[1] for (l, n, o), v in by.items() if o == 4)
        ax.text(104, y, f"{small[0] + large[0]}/{total}", va="center", fontsize=8, color=fs.INK2)
    ax.axhline(-0.65, color=fs.GRID, lw=0.8)
    ax.set_yticks(list(ys.values()) + [y4])
    ax.set_yticklabels(list(ys) + ["fourth order"])
    ax.tick_params(axis="y", length=0)
    ax.set(xlim=(0, 100), ylim=(y4 - 0.55, len(laws) - 0.45), xticks=[0, 50, 100])
    ax.set_xlabel("Episodes recovered (%)")
    fs.spines(ax, ("bottom",))
    handles = [Line2D([], [], color=fs.BLUE, marker="o", ms=4.2, mfc=fs.BLUE, mec=fs.BLUE, lw=1.0, label="1,000"),
               Line2D([], [], color=fs.ORANGE, marker="s", ms=4.2, mfc="white", mec=fs.ORANGE, lw=1.0, label="300 per window")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2, fontsize=8, handlelength=1.6, columnspacing=0.9,
              borderaxespad=0)


def scorecard(fig, worked, records):
    ax = fig.add_axes([0.83, inch(0.62), 0.12, inch(1.42)])
    rows = list(csv.DictReader((HERE / "evidence" / "challenge_terms.tsv").open(), delimiter="\t"))
    constant = [t for d in worked["challenge"] for t in d["terms"] if t["shape"] == "constant" and t["order"] == 3]
    controls = [r for r in rows if float(r["strength"]) == 0]
    planted = [r for r in rows if float(r["strength"]) > 0]
    items = (("Persistent terms\ndeclared changing", sum(bool(t["changed"]) for t in constant), len(constant), False),
             ("Controls with a\nmodulated term", sum(int(r["extra_installed"]) > 0 for r in controls), len(controls), False),
             ("Extra terms\ndeclared changing", sum(int(r["extra_changing"]) for r in planted), sum(int(r["extra_installed"]) for r in planted), False),
             ("Planted datasets with\na held-out gain", sum(float(r["test_gain"]) > 0 for r in planted), len(planted), True))
    for i, (label, k, total, good) in enumerate(items):
        y = len(items) - 1 - i
        ax.barh(y, 100, height=0.46, color=fs.GRID, edgecolor="none", zorder=1)
        ax.barh(y, 100 * k / max(total, 1), height=0.46, color=fs.BLUE if good else fs.ORANGE, edgecolor="none", zorder=2)
        ax.text(-5, y, label, ha="right", va="center", fontsize=8, linespacing=1.05)
        ax.text(104, y, f"{k}/{total}", ha="left", va="center", fontsize=8, fontweight="bold")
        records.append(("F", label.replace("\n", " "), "", "", "", k, total, "", ""))
    ax.set(xlim=(0, 100), ylim=(-0.6, len(items) - 0.4), xticks=[0, 100], yticks=[])
    ax.set_xlabel("Share (%)")
    fs.spines(ax, ("bottom",))


def draw(output, source):
    fs.pro()
    worked = json.loads((HERE / "evidence" / "worked.json").read_text())
    examples = defaultdict(list)
    for r in csv.DictReader((HERE / "evidence" / "figures" / "known_edge_examples.tsv").open(), delimiter="\t"):
        examples[r["design"]].append(r)
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    records = []
    fs.letter_title(fig, 0.0, inch(7.08), "A", "Fitted modulation: one episode")
    example(fig, [0.155, inch(5.35), 0.235, inch(1.45)], examples["abrupt"], records, letter="A")
    fs.letter_title(fig, 0.55, inch(7.08), "B", "Fitted modulation: recurring episodes")
    example(fig, [0.60, inch(5.35), 0.235, inch(1.45)], examples["recurring"], records, names=False, letter="B")
    key = fig.add_axes([0.155, inch(4.95), 0.20, inch(0.07)])
    gradient = np.linspace(-0.3, 0.75, 200)[None, :]
    key.imshow(gradient, aspect="auto", cmap=SLOPES, norm=TwoSlopeNorm(vcenter=0.0, vmin=-0.3, vmax=0.75), extent=(-0.3, 0.75, 0, 1))
    key.set(yticks=[], xticks=[0, 0.6])
    key.set_xticklabels(["0", "0.6"])
    key.tick_params(axis="x", length=2, pad=1)
    for side in key.spines.values():
        side.set_visible(False)
    fig.text(0.37, inch(4.97), "fitted slope; dots: windows called active; outlines: true episodes", fontsize=8, color=fs.INK2, va="center")
    fs.letter_title(fig, 0.0, inch(4.62), "C", "Exact recovery of every window on a known edge")
    designs(fig, records)
    fs.letter_title(fig, 0.0, inch(2.18), "D", "Search among 50 variables")
    pipeline(fig)
    fs.letter_title(fig, 0.31, inch(2.18), "E", "Planted episodes recovered")
    search(fig, worked, records)
    fs.letter_title(fig, 0.62, inch(2.18), "F", "Controls and prediction")
    scorecard(fig, worked, records)
    fs.save_pro(fig, output)
    with Path(source).open("w", newline="") as handle:
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
