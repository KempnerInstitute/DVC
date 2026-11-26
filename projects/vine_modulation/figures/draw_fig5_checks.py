"""Fig 5: what the changed pairs, a change of state and the timing say about the three-way contrast.

    draw_fig5_checks.py OUTPUT_PDF SOURCE_CSV

A  the check (schematic), and for each session of the second cohort the recorded change of the pair and three-way
   contrasts against 200 cohorts simulated from a pairwise model fitted to that session's own ripples (empirical 2.5th
   to 97.5th percentiles of the usable estimates and their mean, evidence/figures/e65_intervals.tsv and
   e65_summary.json); open symbols mark the two sessions whose model failed its adequacy criteria; "7 sessions" pools
   the sessions whose model passed.  Rat 1 had two usable simulated three-way estimates, so its range is omitted.
B  the three-way contrast in subsets of ripples (evidence/figures/e64_posthoc_states.tsv): by the laboratory's sleep
   scoring in the first cohort (Achilles) and by a delta-power rule, matched delta power and matched time since the last
   movement in the second; the last match was added after the main results.  95% intervals.
C  the three steps (schematic) and the two contrasts at each step (evidence/hippocampus/steps.tsv), 95% intervals.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 7.25, 5.95
SESSIONS = (("RAT1or4_SESS", "1"), ("RAT2_SESS1", "2a"), ("RAT2_SESS2", "2b"), ("RAT3_SESS1", "3a"), ("RAT3_SESS2", "3b"),
            ("RAT1_SESS1", "4a"), ("RAT4_SESS2", "4b"), ("RAT5_SESS1", "5a"), ("RAT5_SESS2", "5b"))
RANGE = "#d9d9d6"


def inch(value):
    return value / HEIGHT


def mini_network(ax, x, y, r, shared=True):
    """Three cells with pair couplings and, optionally, a shared-activity hub (axis with equal aspect)."""
    angles = np.deg2rad([90, 210, 330])
    pts = [(x + r * np.cos(a), y + r * np.sin(a)) for a in angles]
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=fs.INK2, lw=1.4, solid_capstyle="round", zorder=2)
    if shared:
        ax.add_patch(Circle((x, y), 0.22 * r, facecolor=fs.MUTED, edgecolor="white", lw=0.8, zorder=3))
        for p in pts:
            ax.plot([x, p[0]], [y, p[1]], color=fs.MUTED, lw=0.7, ls=(0, (1.5, 1.2)), zorder=1)
    for p in pts:
        ax.add_patch(Circle(p, 0.30 * r, facecolor="white", edgecolor=fs.INK2, lw=0.9, zorder=4))


def scheme(fig, rect):
    ax = fs.canvas(fig, rect, xlim=(0, 1))
    top = ax.get_ylim()[1]
    fs.card(ax, 0.0, 0.0, 1.0, top, face=fs.PANEL, radius=0.05)
    # 1. the session's ripples
    y1 = top - 0.10
    ax.text(0.5, y1, "Each session", ha="center", va="center", fontsize=8, fontweight="bold")
    rng = np.random.default_rng(5)
    for i in range(4):
        for t in np.sort(rng.uniform(0.18, 0.82, 6)):
            ax.plot([t, t], [y1 - 0.13 - 0.055 * i, y1 - 0.10 - 0.055 * i], color=fs.INK2, lw=0.8)
    # 2. the fitted pairwise model
    y2 = top * 0.50
    mini_network(ax, 0.5, y2 + 0.07, 0.12)
    ax.text(0.5, y2 - 0.09, "pairwise model:", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.5, y2 - 0.185, "pair couplings and", ha="center", va="center", fontsize=8)
    ax.text(0.5, y2 - 0.275, "shared activity", ha="center", va="center", fontsize=8)
    # 3. simulated cohorts
    y3 = 0.17
    for k in range(3):
        ax.add_patch(FancyBboxPatch((0.30 + 0.04 * k, y3 + 0.03 - 0.03 * k), 0.32, 0.09, boxstyle="round,pad=0,rounding_size=0.02",
                                    facecolor="white", edgecolor=fs.INK2, lw=0.6, zorder=2 + k))
    ax.text(0.5, y3 - 0.075, "200 simulated cohorts", ha="center", va="center", fontsize=8)
    for a, b in ((y1 - 0.33, y2 + 0.22), (y2 - 0.33, y3 + 0.16)):
        ax.annotate("", xy=(0.5, b), xytext=(0.5, a), arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=0.8, mutation_scale=7,
                                                                       shrinkA=0, shrinkB=0))


def pairwise(fig, rect, records):
    x0, y0, w, h = rect
    summary = json.loads((HERE / "evidence" / "figures" / "e65_summary.json").read_text())
    with (HERE / "evidence/figures/e65_intervals.tsv").open() as handle:
        ranges = {(r["session"], r["term"]): r for r in csv.DictReader(handle, delimiter="\t")}
    matched = set(summary["matched_sessions"]["sessions"])
    for k, (term, title, limits) in enumerate((("pairs", "Pair contrast", (-0.4, 1.6)), ("trios", "Three-way contrast", (-1.1, 1.4)))):
        ax = fig.add_axes([x0 + k * 0.53 * w, y0, 0.44 * w, h])
        ax.axhline(0, color=fs.INK2, lw=0.6, zorder=1)
        xs = list(range(len(SESSIONS))) + [len(SESSIONS) + 0.9]
        entries = [(summary["sessions"][s][term], s in matched, label, s) for s, label in SESSIONS]
        entries.append((summary["matched_sessions"][term], True, "7 sessions", "matched"))
        ax.axvspan(len(SESSIONS) + 0.3, len(SESSIONS) + 1.5, color=fs.PANEL, lw=0, zorder=0)
        for x, (e, ok, label, session) in zip(xs, entries):
            bounds = ranges[session, term]
            assert int(bounds["usable"]) == e["usable"]
            if e.get("usable", 0) >= 20:
                low, high = float(bounds["lower"]), float(bounds["upper"])
                ax.add_patch(FancyBboxPatch((x - 0.27, max(low, limits[0])), 0.54, min(high, limits[1]) - max(low, limits[0]),
                                            boxstyle="round,pad=0,rounding_size=0.08", facecolor=RANGE, edgecolor="none", zorder=1,
                                            mutation_aspect=(limits[1] - limits[0]) / 10))
                ax.plot([x - 0.27, x + 0.27], [e["mean"]] * 2, color=fs.INK2, lw=1.0, zorder=2, solid_capstyle="butt")
            marker = "D" if label == "7 sessions" else "^"
            value = e["recorded"]
            ax.plot(x, np.clip(value, *limits), marker=marker, ms=5.6 if marker == "^" else 5.0, mfc=fs.ORANGE if ok else "white",
                    mec=fs.ORANGE, mew=1.1, zorder=3, ls="none")
            records.append(("A", term, label, e.get("usable", 0), float(value), float(e["mean"]), float(e["sd"]), ok, bounds["lower"], bounds["upper"]))
        ax.set(xlim=(-0.6, len(SESSIONS) + 1.5), ylim=limits, xticks=xs)
        ax.set_xticklabels([label for _, label in SESSIONS] + ["7 sess."])
        ax.tick_params(axis="x", length=0, pad=3)
        fs.ticks_minus(ax, "y")
        ax.text(0.0, 1.03, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xlabel("Rat and session (second cohort)", labelpad=2)
        if k == 0:
            ax.set_ylabel("Change, after minus before", labelpad=2)
    handles = [Line2D([], [], ls="none", marker="^", ms=5.6, mfc=fs.ORANGE, mec=fs.ORANGE, label="recorded"),
               Line2D([], [], ls="none", marker="^", ms=5.6, mfc="white", mec=fs.ORANGE, label="recorded, model failed checks"),
               plt.Rectangle((0, 0), 1, 1, facecolor=RANGE, edgecolor="none", label="simulated, central 95%"),
               Line2D([], [], color=fs.INK2, lw=1.0, label="simulated mean")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x0 - 0.01, y0 + h + inch(0.25)), ncol=4, fontsize=8, borderaxespad=0,
               columnspacing=1.0, handlelength=1.2, handletextpad=0.4, frameon=False)


def states(fig, rect, records):
    rows = {(r["cohort"], r["group"], r["contrast"], r["term"]): r
            for r in csv.DictReader((HERE / "evidence" / "figures" / "e64_posthoc_states.tsv").open(), delimiter="\t")}
    items = (("first", "achilles", "maze_sleep", "Non-REM ripples (main)", True), ("first", "achilles", "maze_other", "Drowsy or waking ripples", False),
             ("second", "cohort", "maze", "All ripples (main)", True), ("second", "cohort", "maze_sleep", "Sleep-like, by delta power", False),
             ("second", "cohort", "maze_other", "Other ripples", False), ("second", "cohort", "maze_matched", "Matched delta power", False),
             ("second", "cohort", "maze_still", "Matched time since moving*", False))
    x0, y0, w, h = rect
    ax = fig.add_axes([x0 + 0.47 * w, y0, 0.40 * w, h])
    ax.axvline(0, color=fs.INK2, lw=0.6, zorder=1)
    ys = []
    y = 0.0
    for i, (cohort, group, contrast, label, main) in enumerate(items):
        if i == 2:
            y -= 0.9
        ys.append(y)
        r = rows[(cohort, group, contrast, "trios")]
        value, se = float(r["difference"]), float(r["se_rule"])
        colour, marker = (fs.BLUE, "o") if cohort == "first" else (fs.ORANGE, "^")
        if main:
            ax.axhspan(y - 0.5, y + 0.5, color=fs.PANEL, lw=0, zorder=0)
        ax.plot([value - 1.96 * se, value + 1.96 * se], [y, y], color=colour, lw=1.2, zorder=2, solid_capstyle="butt")
        ax.plot(value, y, marker=marker, ms=5.6, mfc=colour if main else "white", mec=colour, mew=1.0, zorder=3)
        ax.text(-0.04, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                fontweight="bold" if main else "normal")
        ax.text(1.04, y, fs.minus(f"{value:.2f}"), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8)
        y -= 1.0
        records.append(("B", cohort, label, contrast, value, se, "", "", "", ""))
    ax.set(xlim=(-0.1, 0.65), ylim=(ys[-1] - 0.7, 1.35), yticks=[], xticks=[0, 0.2, 0.4, 0.6])
    fs.ticks_minus(ax, "x")
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    for label, yy, colour in (("First cohort: Achilles", ys[0] + 0.95, fs.BLUE), ("Second cohort: five rats", ys[2] + 0.95, fs.ORANGE_DARK)):
        ax.text(-0.04, yy, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontweight="bold", color=colour)
    ax.set_xlabel("Three-way contrast,\nafter minus before", labelpad=2)


def timeline(fig, rect):
    ax = fs.canvas(fig, rect, xlim=(0, 1))
    top = ax.get_ylim()[1]
    yb, hb = top - 0.120, 0.075
    blocks = ((0.00, 0.16, "1st half", fs.PANEL), (0.16, 0.32, "2nd half", fs.PANEL), (0.355, 0.625, "maze or tracks", fs.BLUE_TINT),
              (0.66, 0.82, "1st half", fs.PANEL), (0.82, 0.98, "2nd half", fs.PANEL))
    for a, b, label, face in blocks:
        ax.add_patch(FancyBboxPatch((a + 0.004, yb), b - a - 0.008, hb, boxstyle="round,pad=0,rounding_size=0.012", facecolor=face,
                                    edgecolor=fs.RULE, lw=0.6))
        ax.text((a + b) / 2, yb + hb / 2, label, ha="center", va="center", fontsize=8)
    ax.text(0.16, yb + hb + 0.012, "Rest before", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(0.82, yb + hb + 0.012, "Rest after", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for (a, b), label, yy in ((((0.08, 0.24)), "within before", yb - 0.026), (((0.24, 0.74)), "across", yb - 0.074),
                              (((0.74, 0.90)), "within after", yb - 0.026)):
        ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=0.8, mutation_scale=6,
                                                                    shrinkA=0, shrinkB=0))
        for end in (a, b):
            ax.plot([end, end], [yy, yb], color=fs.RULE, lw=0.6, zorder=0)
        ax.text((a + b) / 2, yy - 0.008, label, ha="center", va="top", fontsize=8, color=fs.INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))


def steps(fig, rect, records):
    x0, y0, w, h = rect
    rows = {(r["cohort"], r["group"], r["step"], r["term"]): r
            for r in csv.DictReader((HERE / "evidence" / "hippocampus" / "steps.tsv").open(), delimiter="\t")}
    for k, (term, title) in enumerate((("pairs", "Pair contrast"), ("trios", "Three-way contrast"))):
        ax = fig.add_axes([x0 + k * 0.53 * w + 0.11 * w, y0, 0.36 * w, h])
        ax.axhline(0, color=fs.INK2, lw=0.6, zorder=1)
        for j, (cohort, group, colour, marker, style, offset, label) in enumerate((("first", "achilles", fs.BLUE, "o", "-", -0.09, "Achilles"),
                                                                                  ("second", "cohort", fs.ORANGE, "^", (0, (4, 2)), 0.09, "Five rats"))):
            values, errors = [], []
            for step in ("before", "across", "after"):
                r = rows[(cohort, group, step, term)]
                values.append(float(r["difference"]))
                errors.append(1.96 * float(r["se_rule"]))
                records.append(("C", term, label, step, float(r["difference"]), float(r["se_rule"]), "", "", "", ""))
            ax.errorbar(np.arange(3) + offset, values, yerr=errors, color=colour, marker=marker, ls=style, ms=5.0, lw=1.2, capsize=0,
                        mfc=colour, mec=colour, label=label, zorder=3)
        ax.set(xlim=(-0.45, 2.45), xticks=range(3), ylim=(-0.32, 0.80), yticks=[-0.2, 0, 0.2, 0.4, 0.6])
        ax.set_xticklabels(["within\nbefore", "across", "within\nafter"])
        ax.tick_params(axis="x", length=0, pad=3)
        fs.ticks_minus(ax, "y")
        ax.text(0.0, 1.04, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
        if k == 0:
            ax.set_ylabel("Change at the step", labelpad=2)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([h_[0] for h_ in handles], labels, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=8, handlelength=2.2,
                      frameon=False, borderaxespad=0.2, ncol=2, columnspacing=1.0)


def draw(output, source):
    fs.pro()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    records = []
    fs.letter_title(fig, 0.0, inch(5.78), "A", "Recorded changes against each session’s pairwise model")
    scheme(fig, [0.0, inch(3.36), 0.175, inch(2.04)])
    pairwise(fig, [0.275, inch(3.70), 0.725, inch(1.48)], records)
    fs.letter_title(fig, 0.0, inch(2.88), "B", "Sleep state and stillness")
    states(fig, [0.0, inch(0.47), 0.46, inch(2.10)], records)
    fs.letter_title(fig, 0.50, inch(2.88), "C", "When the contrasts changed")
    timeline(fig, [0.515, inch(1.90), 0.485, inch(0.84)])
    steps(fig, [0.47, inch(0.42), 0.53, inch(1.30)], records)
    fs.save_pro(fig, output)
    with Path(source).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "term_or_cohort", "row", "count_or_subset", "recorded_or_estimate", "simulated_mean_or_se", "simulated_sd",
                         "model_adequate", "simulation_percentile_2.5", "simulation_percentile_97.5"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
