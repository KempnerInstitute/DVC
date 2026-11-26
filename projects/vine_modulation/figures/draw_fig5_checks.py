"""Fig 5: what the changed pairs, a change of state and the timing say about the three-way contrast.

    draw_fig5_checks.py OUTPUT_PDF SOURCE_CSV

A  for each session of the second cohort, the recorded change of the pair and three-way contrasts against 200
   cohorts simulated from a pairwise model fitted to that session's own ripples (mean and central 95%,
   evidence/figures/e65_summary.json); open symbols mark the two sessions whose model failed its adequacy criteria;
   "7 sessions" pools the sessions whose model passed.
B  the three-way contrast in subsets of ripples (evidence/figures/e64_posthoc_states.tsv): by the laboratory's sleep
   scoring in the first cohort (Achilles) and by a delta-power rule, matched delta power and matched time since the last
   movement in the second; the last match was added after the main results.  95% intervals.
C  the two contrasts at three consecutive steps (evidence/hippocampus/steps.tsv), 95% intervals.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
HEIGHT = 5.35
SESSIONS = (("RAT1or4_SESS", "1"), ("RAT2_SESS1", "2a"), ("RAT2_SESS2", "2b"), ("RAT3_SESS1", "3a"), ("RAT3_SESS2", "3b"),
            ("RAT1_SESS1", "4a"), ("RAT4_SESS2", "4b"), ("RAT5_SESS1", "5a"), ("RAT5_SESS2", "5b"))


def inch(value):
    return value / HEIGHT


def pairwise(fig, rect, records):
    x0, y0, w, h = rect
    summary = json.loads((HERE / "evidence" / "figures" / "e65_summary.json").read_text())
    with (HERE / "evidence/figures/e65_intervals.tsv").open() as handle:
        ranges = {(r["session"], r["term"]): r for r in csv.DictReader(handle, delimiter="\t")}
    matched = set(summary["matched_sessions"]["sessions"])
    for k, (term, title, limits) in enumerate((("pairs", "Pair contrast", (-0.4, 1.6)), ("trios", "Three-way contrast", (-1.1, 1.4)))):
        ax = fig.add_axes([x0 + 0.10 * w + k * 0.50 * w, y0, 0.39 * w, h])
        fs.zero(ax)
        xs = list(range(len(SESSIONS))) + [len(SESSIONS) + 0.8]
        entries = [(summary["sessions"][s][term], s in matched, label, s) for s, label in SESSIONS]
        entries.append((summary["matched_sessions"][term], True, "7 sessions", "matched"))
        for x, (e, ok, label, session) in zip(xs, entries):
            bounds = ranges[session, term]
            assert int(bounds["usable"]) == e["usable"]
            if e.get("usable", 0) >= 20:
                low, high = float(bounds["lower"]), float(bounds["upper"])
                ax.add_patch(plt.Rectangle((x - 0.28, max(low, limits[0])), 0.56, min(high, limits[1]) - max(low, limits[0]),
                                           facecolor=fs.GREY_LIGHT, edgecolor=fs.MUTED, lw=0.6, zorder=1))
                ax.plot([x - 0.28, x + 0.28], [e["mean"]] * 2, color=fs.INK2, lw=1.0, zorder=2)
            marker = "D" if label == "7 sessions" else "^"
            value = e["recorded"]
            ax.plot(x, np.clip(value, *limits), marker=marker, ms=5.2, mfc=fs.ORANGE if ok else "white", mec=fs.ORANGE, mew=1.1, zorder=3, ls="none")
            if value < limits[0]:
                ax.annotate(f"{value:.2f}", (x, limits[0]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=fs.SMALL, color=fs.INK2)
            records.append(("A", term, label, e.get("usable", 0), float(value), float(e["mean"]), float(e["sd"]), ok, bounds["lower"], bounds["upper"]))
        ax.axvline(len(SESSIONS) + 0.15, color=fs.GRID, lw=0.8)
        ax.set(xlim=(-0.6, len(SESSIONS) + 1.4), ylim=limits, xticks=xs)
        ax.set_xticklabels([label for _, label in SESSIONS] + ["7\nsessions"], fontsize=fs.SMALL)
        ax.set_title(title, loc="left", fontsize=fs.TEXT, pad=3)
        ax.set_xlabel("Rat and session", labelpad=1)
        if k == 0:
            ax.set_ylabel("Change after minus before")
    handles = [Line2D([], [], ls="none", marker="^", ms=5.2, mfc=fs.ORANGE, mec=fs.ORANGE, label="recorded"),
               plt.Rectangle((0, 0), 1, 1, facecolor=fs.GREY_LIGHT, edgecolor=fs.MUTED, lw=0.6, label="pairwise model: mean, central 95%"),
               Line2D([], [], ls="none", marker="^", ms=5.2, mfc="white", mec=fs.ORANGE, label="model failed its adequacy criteria")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(x0 + 0.5 * w, y0 + h + inch(0.08)), ncol=3, fontsize=fs.SMALL,
               borderaxespad=0, columnspacing=1.4, handlelength=1.6)


def states(ax, records):
    rows = {(r["cohort"], r["group"], r["contrast"], r["term"]): r for r in csv.DictReader((HERE / "evidence" / "figures" / "e64_posthoc_states.tsv").open(), delimiter="\t")}
    items = (("first", "achilles", "maze_sleep", "Achilles: non-REM ripples (main)"), ("first", "achilles", "maze_other", "Achilles: drowsy or waking ripples"),
             ("second", "cohort", "maze", "Five rats: all ripples (main)"), ("second", "cohort", "maze_sleep", "Five rats: sleep-like by delta power"),
             ("second", "cohort", "maze_other", "Five rats: other ripples"), ("second", "cohort", "maze_matched", "Five rats: matched delta power"),
             ("second", "cohort", "maze_still", "Five rats: matched time since moving*"))
    fs.zero(ax, "x")
    for i, (cohort, group, contrast, label) in enumerate(items):
        r = rows[(cohort, group, contrast, "trios")]
        value, se = float(r["difference"]), float(r["se_rule"])
        y = -i - (0.5 if cohort == "second" else 0.0)
        colour, marker = (fs.BLUE, "o") if cohort == "first" else (fs.ORANGE, "^")
        ax.plot([value - 1.96 * se, value + 1.96 * se], [y, y], color=colour, lw=1.3)
        ax.plot(value, y, marker=marker, ms=5.2, mfc=colour if "main" in label else "white", mec=colour, mew=1.0)
        ax.text(-0.05, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=fs.SMALL)
        records.append(("B", cohort, label, contrast, value, se, "", "", "", ""))
    ax.set(xlim=(-0.1, 0.65), ylim=(-len(items) - 0.2, 0.6), yticks=[], xticks=[0, 0.2, 0.4, 0.6])
    fs.spines(ax, ("bottom",))
    ax.set_xlabel("Three-way contrast, after minus before", labelpad=1)


def steps(fig, rect, records):
    x0, y0, w, h = rect
    rows = {(r["cohort"], r["group"], r["step"], r["term"]): r for r in csv.DictReader((HERE / "evidence" / "hippocampus" / "steps.tsv").open(), delimiter="\t")}
    for k, (term, title) in enumerate((("pairs", "Pair contrast"), ("trios", "Three-way contrast"))):
        ax = fig.add_axes([x0 + k * 0.50 * w + 0.12 * w, y0, 0.36 * w, h])
        fs.zero(ax)
        for j, (cohort, group, colour, marker, style, offset, label) in enumerate((("first", "achilles", fs.BLUE, "o", "-", -0.08, "Achilles"),
                                                                                  ("second", "cohort", fs.ORANGE, "^", (0, (4, 2)), 0.08, "Five rats"))):
            values, errors = [], []
            for step in ("before", "across", "after"):
                r = rows[(cohort, group, step, term)]
                values.append(float(r["difference"]))
                errors.append(1.96 * float(r["se_rule"]))
                records.append(("C", term, label, step, float(r["difference"]), float(r["se_rule"]), "", "", "", ""))
            ax.errorbar(np.arange(3) + offset, values, yerr=errors, color=colour, marker=marker, ls=style, ms=4.8, lw=1.3, capsize=0,
                        mfc=colour, mec=colour, label=label)
        ax.set(xlim=(-0.4, 2.4), xticks=range(3), ylim=(-0.32, 0.62))
        ax.set_xticklabels(["within\nbefore", "across", "within\nafter"], fontsize=fs.SMALL)
        ax.set_title(title, loc="left", fontsize=fs.TEXT, pad=3)
        if k == 0:
            ax.set_ylabel("Change at the step")
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(x0 + 0.56 * w, 0.0), ncol=2, fontsize=fs.SMALL,
                       handlelength=2.4, columnspacing=1.4, borderaxespad=0)


def draw(output, source):
    fs.print_style()
    fig = plt.figure(figsize=(fs.WIDTH, HEIGHT))
    records = []
    fs.heading(fig, 0.0, 1 - inch(0.17), "A", "Recorded changes and predictions from each session’s pairwise model")
    pairwise(fig, [0.0, inch(3.02), 1.0, inch(1.62)], records)
    fs.heading(fig, 0.0, inch(2.42), "B", "Sleep state and stillness controls")
    states(fig.add_axes([0.30, inch(0.42), 0.18, inch(1.86)]), records)
    fs.heading(fig, 0.52, inch(2.42), "C", "When the contrasts changed")
    steps(fig, [0.52, inch(0.62), 0.48, inch(1.60)], records)
    fs.save_journal(fig, output)
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "term_or_cohort", "row", "count_or_subset", "recorded_or_estimate", "simulated_mean_or_se", "simulated_sd", "model_adequate", "simulation_percentile_2.5", "simulation_percentile_97.5"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
