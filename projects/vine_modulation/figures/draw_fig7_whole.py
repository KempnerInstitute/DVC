"""Fig 7: whole held-out ripples beyond replay, the calibration of the change comparisons, and timing without the
boundary between the rests.

    draw_fig7_whole.py OUTPUT_PDF SOURCE_CSV

A  the sequential model of a whole ripple (schematic).
B  gain in held-out log probability of each addition, per recording and for each group (95% interval over blocks of
   consecutive ripples), in nats per 1,000 ripples (evidence/hippocampus/fixed_fold_*.tsv, training-fold scales).
C  simulated cohorts matched to each recording design (24 per scenario): share with a change found (z >= 1.96) on the
   primary scale and on the common-count scale of S1 Text, with 95% Clopper-Pearson intervals
   (evidence/hippocampus/fold_calibration.tsv, common_count_scenarios.tsv).
D  for every recording, the fit of two replay models split at each candidate point minus one model, relative to its
   best split, against the distance of the split from the experience (evidence/figures/timing_profiles.tsv).
E  held-out comparisons of where to split the replay model in the second cohort, per session and in total (95%
   intervals over blocks): the split at one third of the ripples (the rest schedule) against no split, a split learned
   from the training folds against the schedule, and the true boundary against the schedule
   (evidence/hippocampus/schedule_predictions.tsv).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
HEIGHT = 8.55
FIRST = ("Achilles-10252013_left", "Achilles-10252013_right", "Achilles-11012013_left", "Achilles-11012013_right")


def inch(value):
    return value / HEIGHT


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def schematic(ax):
    fs.blank(ax)
    xs = np.linspace(0.06, 0.62, 6)
    y = 0.58
    for k, x in enumerate(xs):
        face = fs.ORANGE_LIGHT if k == 5 else fs.GREY_LIGHT
        ax.plot(x, y, marker="o", ms=15, mfc=face, mec=fs.ORANGE if k in (2, 4) else fs.INK2, mew=1.3 if k in (2, 4) else 0.8, zorder=3)
        ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=fs.SMALL, zorder=4)
    ax.text(0.03, y - 0.15, "place cells in field order", ha="left", va="center", fontsize=fs.SMALL, color=fs.INK2, fontstyle="italic")
    fs.box(ax, 0.30, 0.88, 0.32, 0.11, "replay model's prediction", face=fs.BLUE_LIGHT, size=fs.SMALL, radius=0.02)
    fs.arrow(ax, (0.58, 0.88), (xs[5] - 0.005, y + 0.07), lw=0.9, scale=7)
    for k in (1, 3):
        fs.arrow(ax, (xs[k] + 0.02, y + 0.05), (xs[5] - 0.035, y + 0.035), lw=0.8, scale=6, color=fs.INK2, connection="arc3,rad=-0.35")
    ax.text(0.40, 0.80, "pair edges", ha="center", va="bottom", fontsize=fs.SMALL)
    ax.plot([xs[2], xs[2], xs[4], xs[4]], [y - 0.06, y - 0.10, y - 0.10, y - 0.06], color=fs.ORANGE, lw=1.3)
    fs.arrow(ax, (xs[3], y - 0.10), (xs[5] - 0.02, y - 0.06), lw=0.9, scale=7, color=fs.ORANGE, connection="arc3,rad=0.4")
    ax.text(0.34, 0.26, "three-cell modulation: an active pair that closes a trio of a class", ha="center", va="center", fontsize=fs.SMALL)
    steps = ("replay", "+ pair edges", "+ persistent\nmodulation", "+ shared\nchange", "+ change of the\noverlapping class")
    for i, text in enumerate(steps):
        x = 0.70 + (i % 1) * 0.0
        yy = 0.92 - i * 0.19
        fs.box(ax, 0.70, yy - 0.075, 0.29, 0.15, None, face=fs.GREY_LIGHT if i else fs.BLUE_LIGHT, edge=fs.INK2, radius=0.02)
        ax.text(0.845, yy, text, ha="center", va="center", fontsize=fs.SMALL, linespacing=1.0)
        if i < len(steps) - 1:
            fs.arrow(ax, (0.845, yy - 0.08), (0.845, yy - 0.11), color=fs.INK2, lw=0.8, scale=6)
    ax.text(0.34, 0.08, "each model scores the same held-out blocks of consecutive ripples", ha="center", va="center", fontsize=fs.SMALL,
            color=fs.INK2, fontstyle="italic")


def gains(fig, rect, records):
    x0, y0, w, h = rect
    per = defaultdict(dict)
    for r in read(HERE / "evidence" / "hippocampus" / "fixed_fold_recordings.tsv"):
        scale = 1000 / int(r["ripples"])
        per[r["recording"]][r["comparison"]] = (float(r["training_scaled_nats"]) * scale, float(r["training_scaled_se"]) * scale, int(r["ripples"]))
    summary = {(r["group"], r["comparison"]): r for r in read(HERE / "evidence" / "hippocampus" / "fixed_fold_summary.tsv")}
    totals = {(r["group"], r["comparison"]): int(r["ripples"]) for r in read(HERE / "evidence" / "hippocampus" / "whole_ripples.tsv")}
    comparisons = (("pairs", "pair edges"), ("modulation", "persistent\nmodulation"), ("time_varying", "shared\nchange"),
                   ("class_change", "class-specific\nchange"))
    for k, (comparison, label) in enumerate(comparisons):
        big = comparison == "pairs"
        ax = fig.add_axes([x0 + (0.10 if big else 0.32 + (k - 1) * 0.235) * w, y0, (0.15 if big else 0.17) * w, h])
        fs.zero(ax)
        for j, (group, colour, marker, members) in enumerate((("achilles", fs.BLUE, "o", FIRST),
                                                               ("cohort", fs.ORANGE, "^", [r for r in per if r.startswith("RAT")]))):
            values = [per[r][comparison][0] for r in members if comparison in per[r]]
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.plot(j + jitter, values, ls="none", marker=marker, ms=3.6, mfc="white", mec=colour, mew=0.8, zorder=2)
            s = summary[(group, comparison)]
            scale = 1000 / totals[(group, comparison)]
            value, se = float(s["training_scaled_nats"]) * scale, float(s["training_scaled_se"]) * scale
            ax.errorbar(j + 0.30, value, yerr=1.96 * se, marker="D", ms=4.6, color=colour, mfc=colour, capsize=0, lw=1.3, zorder=3)
            records.append(("B", group, comparison, f"{value:.4f}", f"{se:.4f}", f"{s['recordings_positive']} of {s['recordings']} recordings positive"))
        ax.set(xlim=(-0.45, 1.6), xticks=[0.1, 1.1])
        ax.set_xticklabels(["Ach.", "5 rats"], fontsize=fs.SMALL)
        ax.set_title(label, fontsize=fs.SMALL + 0.5, pad=3, linespacing=1.0)
        if k == 0:
            ax.set_ylabel("Held-out gain (nats per\n1,000 ripples)")
        if k == 1:
            ax.set_ylim(-4, 8)
        if k > 1:
            ax.set_ylim(-4, 8)
            ax.set_yticklabels([])


def calibration(fig, rect, records):
    x0, y0, w, h = rect
    primary = {(r["group"], r["scenario"], r["comparison"]): r for r in read(HERE / "evidence" / "hippocampus" / "fold_calibration.tsv")}
    common = {(r["group"], r["scenario"], r["comparison"]): r for r in read(HERE / "evidence" / "hippocampus" / "common_count_scenarios.tsv")}
    from scipy.stats import beta

    def interval(k, n):
        return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)

    scenarios = (("replay", "replay changes only"), ("apart", "classes shift together"), ("class", "overlapping class changes"))
    for g, (group, title) in enumerate((("cohort", "matched to the five rats"), ("achilles", "matched to Achilles"))):
        ax = fig.add_axes([x0 + 0.21 * w + g * 0.31 * w, y0, 0.25 * w, h])
        for i, (scenario, label) in enumerate(scenarios):
            for c, (comparison, colour, marker) in enumerate((("time_varying", fs.INK2, "o"), ("class_change", fs.ORANGE, "^"))):
                y = -i - 0.17 + 0.34 * c
                for source, filled, dy in ((primary, True, 0.0), (common, False, 0.0)):
                    r = source[(group, scenario, comparison)]
                    k = int(r["training_scaled_exceedances"] if source is primary else r["crossings"])
                    n = int(r["cohorts"])
                    lo, hi = interval(k, n)
                    offset = -0.07 if filled else 0.07
                    ax.plot([lo, hi], [y + offset] * 2, color=colour, lw=1.0, alpha=1 if filled else 0.7)
                    ax.plot(k / n, y + offset, marker=marker, ms=4.4, mfc=colour if filled else "white", mec=colour, mew=0.9)
                    records.append(("C", group, f"{scenario}; {comparison}", "primary" if filled else "common count", k, n))
            if g == 0:
                ax.text(-0.06, -i, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=fs.SMALL)
        ax.set(xlim=(-0.03, 1.03), ylim=(-2.6, 0.55), yticks=[], xticks=[0, 0.5, 1])
        fs.spines(ax, ("bottom",))
        ax.set_title(title, fontsize=fs.SMALL + 0.5, pad=3)
        ax.set_xlabel("Share of 24 cohorts\nwith a change found", labelpad=1, fontsize=fs.SMALL)
    handles = [Line2D([], [], ls="none", marker="o", mfc=fs.INK2, mec=fs.INK2, ms=4.4, label="shared change"),
               Line2D([], [], ls="none", marker="^", mfc=fs.ORANGE, mec=fs.ORANGE, ms=4.4, label="class-specific change"),
               Line2D([], [], ls="none", marker="o", mfc=fs.INK2, mec=fs.INK2, ms=4.4, label="filled: primary scale"),
               Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.INK2, ms=4.4, label="open: common count")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(x0 + 0.80 * w, y0 + 0.5 * h), ncol=1, fontsize=fs.SMALL, borderaxespad=0,
               handletextpad=0.3, labelspacing=0.6)


def profiles(ax, records):
    rows = read(HERE / "evidence" / "figures" / "timing_profiles.tsv")
    by = defaultdict(list)
    boundary = {}
    for r in rows:
        if r["location"] == "boundary":
            continue
        by[r["recording"]].append((float(r["location"]), float(r["split_minus_one_law"])))
        boundary[r["recording"]] = float(r["boundary"])
    for name, points in sorted(by.items()):
        if name.startswith("Cicero") or name.startswith("Gatsby"):
            continue
        points = np.array(sorted(points))
        rel = points[:, 1] - points[:, 1].max()
        x = points[:, 0] - boundary[name]
        first = name.startswith("Achilles")
        colour = fs.BLUE if first else fs.ORANGE
        ax.plot(x, rel, color=colour, lw=0.9, ls="-" if first else (0, (3, 1.5)), alpha=0.9)
        best = int(np.argmax(points[:, 1]))
        ax.plot(x[best], 0, marker="o" if first else "^", ms=4.2, mfc=colour, mec=colour, zorder=3)
        records.append(("D", name, f"{boundary[name]:.4f}", f"{points[best, 0]:.2f}", float(points[best, 1]), ""))
    ax.axvline(0, color=fs.INK, lw=0.9)
    ax.axvspan(-0.10, 0.10, color=fs.GREY_LIGHT, lw=0, zorder=0)
    ax.set(xlim=(-0.65, 0.92), ylim=(-260, 20))
    ax.set_xlabel("Split point minus the experience\n(fraction of the session's ripples)", labelpad=1)
    ax.set_ylabel("Fit relative to the\nbest split (nats)")
    handles = [Line2D([], [], color=fs.BLUE, lw=0.9, marker="o", ms=4.2, label="Achilles hemispheres"),
               Line2D([], [], color=fs.ORANGE, lw=0.9, ls=(0, (3, 1.5)), marker="^", ms=4.2, label="second-cohort sessions")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=fs.SMALL, borderaxespad=0, handlelength=2.2)


def schedule(ax, records):
    rows = read(HERE / "evidence" / "hippocampus" / "schedule_predictions.tsv")
    contrasts = (("schedule_minus_stationary", "schedule split\nvs no split"), ("learned_minus_schedule", "learned split\nvs schedule"),
                 ("given_minus_schedule", "true boundary\nvs schedule"))
    fs.zero(ax)
    for i, (contrast, label) in enumerate(contrasts):
        chosen = [r for r in rows if r["contrast"] == contrast]
        sessions = [r for r in chosen if r["recording"].startswith("RAT")]
        values = np.array([float(r["total_nats"]) for r in sessions])
        ax.plot(i + np.linspace(-0.15, 0.15, len(values)), values, ls="none", marker="^", ms=3.6, mfc="white", mec=fs.ORANGE, mew=0.8)
        total = sum(values)
        se = np.sqrt(sum(float(r["block_se"]) ** 2 for r in sessions))
        ax.errorbar(i + 0.33, total / len(values), yerr=1.96 * se / len(values), marker="D", ms=4.6, color=fs.ORANGE, capsize=0, lw=1.3)
        records.append(("E", contrast, "nine sessions", f"{total:.2f}", f"{se:.2f}", f"{int((values > 0).sum())} of {len(values)} positive"))
    ax.set(xlim=(-0.5, 2.6), xticks=range(3))
    ax.set_xticklabels([c[1] for c in contrasts], fontsize=fs.SMALL)
    ax.set_ylabel("Held-out gain per\nsession (nats)")


def draw(output, source):
    fs.print_style()
    fig = plt.figure(figsize=(fs.WIDTH, HEIGHT))
    records = []
    fs.heading(fig, 0.0, inch(8.38), "A", "Whole ripples, one cell at a time, beyond the replay model")
    schematic(fig.add_axes([0.0, inch(6.40), 1.0, inch(1.85)]))
    fs.heading(fig, 0.0, inch(6.18), "B", "Gain on held-out ripples from each addition")
    gains(fig, [0.0, inch(4.55), 1.0, inch(1.28)], records)
    fs.heading(fig, 0.0, inch(4.05), "C", "Calibration in simulated cohorts")
    calibration(fig, [0.0, inch(2.98), 1.0, inch(0.78)], records)
    fs.heading(fig, 0.0, inch(2.28), "D", "Replay change without the boundary")
    profiles(fig.add_axes([0.10, inch(0.62), 0.26, inch(1.32)]), records)
    fs.heading(fig, 0.60, inch(2.28), "E", "Where to split the replay model")
    schedule(fig.add_axes([0.70, inch(0.62), 0.29, inch(1.32)]), records)
    fs.save_journal(fig, output)
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "group_or_recording", "comparison_or_boundary", "value_or_best_split", "se_or_gain", "note"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
