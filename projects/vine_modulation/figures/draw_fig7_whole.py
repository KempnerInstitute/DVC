"""Fig 7: whole held-out ripples beyond replay, the calibration of the change comparisons, and timing without the
boundary between the rests.

    draw_fig7_whole.py OUTPUT_PDF SOURCE_CSV

A  the sequential model of a whole ripple and the nested comparisons (schematic).
B  gain in held-out log probability of each addition, per recording and for each group (95% interval over blocks of
   consecutive ripples), in nats per 1,000 ripples (evidence/hippocampus/fixed_fold_*.tsv, training-fold scales).
C  simulated cohorts matched to each recording design (24 per scenario): how many found a shared or a class-specific
   change (z >= 1.96) on the primary scale and on the common-count scale of S1 Text
   (evidence/hippocampus/fold_calibration.tsv, common_count_scenarios.tsv).
D  for every recording, the fit of two replay models split at each candidate point minus one model, relative to its
   best split, against the distance of the split from the experience (evidence/figures/timing_profiles.tsv).
E  held-out comparisons of where to split the replay model in the second cohort, per session and on average (95%
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
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 7.25, 8.40
FIRST = ("Achilles-10252013_left", "Achilles-10252013_right", "Achilles-11012013_left", "Achilles-11012013_right")
MODULATION = fs.ORANGE_DARK


def inch(value):
    return value / HEIGHT


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def schematic(fig, rect):
    ax = fs.canvas(fig, rect, xlim=(0, 1))
    top = ax.get_ylim()[1]
    # cells in field order: two earlier cells active, the next cell predicted
    xs = np.linspace(0.045, 0.325, 5)
    target = 0.415
    yc = 0.135
    r = 0.0135
    fired = (False, True, False, True, False)
    base = 0.010
    x = np.linspace(0.01, 0.45, 500)
    for c in list(xs) + [target]:
        ax.plot(x, base + 0.024 * np.exp(-0.5 * ((x - c) / 0.017) ** 2), color=fs.MUTED, lw=0.8)
    ax.plot([0.01, 0.45], [base, base], color=fs.INK2, lw=0.7)
    ax.text(0.455, base + 0.004, "fields", ha="left", va="center", fontsize=8, color=fs.INK2)
    for k, (c, on) in enumerate(zip(xs, fired)):
        ax.add_patch(Circle((c, yc), r, facecolor=fs.INK if on else "white", edgecolor=fs.INK2, lw=0.9, zorder=4))
        ax.text(c, yc + r + 0.005, str(k + 1), ha="center", va="bottom", fontsize=8, color=fs.INK2)
    ax.add_patch(Circle((target, yc), r * 1.3, facecolor="white", edgecolor=fs.INK, lw=1.3, zorder=4))
    ax.text(target, yc, "?", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=5)
    ax.text(target, yc + 1.3 * r + 0.005, "6", ha="center", va="bottom", fontsize=8, color=fs.INK2)
    # pair edges from the active cells, above
    for k, rad in ((1, -0.38), (3, -0.55)):
        ax.annotate("", xy=(target - 0.006, yc + 1.3 * r), xytext=(xs[k] + 0.004, yc + r),
                    arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=0.8, mutation_scale=6, shrinkA=1, shrinkB=1,
                                    connectionstyle=f"arc3,rad={rad}"))
    ax.text((xs[1] + target) / 2 + 0.02, yc + 0.088, "pair edges", ha="center", va="bottom", fontsize=8, color=fs.INK2)
    # the active pair that closes a trio with the predicted cell, below
    yb = yc - 0.040
    ax.plot([xs[1], xs[1], xs[3], xs[3]], [yc - r - 0.006, yb, yb, yc - r - 0.006], color=MODULATION, lw=1.3, solid_capstyle="round")
    ax.annotate("", xy=(target - 0.004, yc - 1.3 * r), xytext=(xs[3] + 0.002, yb),
                arrowprops=dict(arrowstyle="-|>", color=MODULATION, lw=1.2, mutation_scale=7, shrinkA=0, shrinkB=1,
                                connectionstyle="arc3,rad=0.35"))
    ax.text((xs[1] + xs[3]) / 2, yb - 0.006, "active pair closing a trio", ha="center", va="top", fontsize=8, color=MODULATION)
    # the replay model's prediction and the whole-ripple probability
    fs.card(ax, 0.505, yc + 0.045, 0.12, 0.034, face=fs.BLUE_TINT, edge=fs.BLUE, radius=0.008, lw=0.7)
    ax.text(0.565, yc + 0.062, "replay model", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(target + 1.3 * r * 0.75, yc + 1.3 * r * 0.75), xytext=(0.51, yc + 0.050),
                arrowprops=dict(arrowstyle="-|>", color=fs.BLUE, lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0))
    fs.smath(ax, 0.505, yc - 0.004, r"$P(\mathrm{ripple})=\prod_i\,P(x_i\mid x_{<i})$", size=8.5, va="center", transform=ax.transData)
    ax.text(0.505, yc - 0.042, "cells predicted in field order;", ha="left", va="center", fontsize=8, color=fs.INK2)
    ax.text(0.505, yc - 0.064, "a normalized whole-ripple law", ha="left", va="center", fontsize=8, color=fs.INK2)
    # nested models
    lx, lw_ = 0.795, 0.200
    steps = (("Replay model", fs.BLUE_TINT), ("+ pair edges", fs.PANEL), ("+ persistent modulation", fs.PANEL), ("+ shared change", fs.PANEL),
             ("+ class-specific change", fs.PANEL))
    ax.text(lx + lw_ / 2, top - 0.008, "Nested models", ha="center", va="center", fontsize=8, fontweight="bold")
    for k, (label, face) in enumerate(steps):
        yy = top - 0.042 - 0.040 * k
        fs.card(ax, lx, yy - 0.014, lw_, 0.028, face=face, edge=fs.RULE, radius=0.007, lw=0.6)
        ax.text(lx + lw_ / 2, yy, label, ha="center", va="center", fontsize=8)
        if k:
            ax.annotate("", xy=(lx + lw_ / 2, yy + 0.014), xytext=(lx + lw_ / 2, yy + 0.026),
                        arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(lx + lw_ / 2, top - 0.042 - 0.040 * 4 - 0.030, "scored on the same held-out", ha="center", va="center", fontsize=8, color=fs.INK2)
    ax.text(lx + lw_ / 2, top - 0.042 - 0.040 * 4 - 0.050, "blocks of consecutive ripples", ha="center", va="center", fontsize=8, color=fs.INK2)


def gains(fig, rect, records):
    x0, y0, w, h = rect
    per = defaultdict(dict)
    for r in read(HERE / "evidence" / "hippocampus" / "fixed_fold_recordings.tsv"):
        scale = 1000 / int(r["ripples"])
        per[r["recording"]][r["comparison"]] = (float(r["training_scaled_nats"]) * scale, float(r["training_scaled_se"]) * scale, int(r["ripples"]))
    summary = {(r["group"], r["comparison"]): r for r in read(HERE / "evidence" / "hippocampus" / "fixed_fold_summary.tsv")}
    totals = {(r["group"], r["comparison"]): int(r["ripples"]) for r in read(HERE / "evidence" / "hippocampus" / "whole_ripples.tsv")}
    comparisons = (("pairs", "+ pair edges"), ("modulation", "+ persistent\nmodulation"), ("time_varying", "+ shared\nchange"),
                   ("class_change", "+ class-specific\nchange"))
    for k, (comparison, label) in enumerate(comparisons):
        big = comparison == "pairs"
        ax = fig.add_axes([x0 + (0.085 if big else 0.33 + (k - 1) * 0.235) * w, y0, (0.15 if big else 0.17) * w, h])
        ax.axhline(0, color=fs.INK2, lw=0.6, zorder=1)
        for j, (group, colour, marker, members) in enumerate((("achilles", fs.BLUE, "o", FIRST),
                                                               ("cohort", fs.ORANGE, "^", [r for r in per if r.startswith("RAT")]))):
            values = [per[r][comparison][0] for r in members if comparison in per[r]]
            jitter = np.linspace(-0.13, 0.13, len(values))
            ax.plot(j + jitter, values, ls="none", marker=marker, ms=3.8, mfc="white", mec=colour, mew=0.8, zorder=2)
            s = summary[(group, comparison)]
            scale = 1000 / totals[(group, comparison)]
            value, se = float(s["training_scaled_nats"]) * scale, float(s["training_scaled_se"]) * scale
            ax.errorbar(j + 0.33, value, yerr=1.96 * se, marker="D", ms=4.8, color=colour, mfc=colour, capsize=0, lw=1.3, zorder=3)
            records.append(("B", group, comparison, f"{value:.4f}", f"{se:.4f}", f"{s['recordings_positive']} of {s['recordings']} recordings positive"))
        ax.set(xlim=(-0.45, 1.6), xticks=[0.1, 1.1])
        ax.set_xticklabels(["Achilles", "5 rats"])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.text(0.5, 1.03, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold", linespacing=1.05)
        fs.ticks_minus(ax, "y")
        if k == 0:
            ax.set_ylabel("Held-out gain\n(nats per 1,000 ripples)", labelpad=2)
        if k >= 1:
            ax.set_ylim(-4, 8)
        if k > 1:
            ax.set_yticklabels([])
    fig.text(x0 + 0.33 * w - 0.045, y0 + h * 1.0, "", fontsize=8)


def calibration(fig, rect, records):
    x0, y0, w, h = rect
    primary = {(r["group"], r["scenario"], r["comparison"]): r for r in read(HERE / "evidence" / "hippocampus" / "fold_calibration.tsv")}
    common = {(r["group"], r["scenario"], r["comparison"]): r for r in read(HERE / "evidence" / "hippocampus" / "common_count_scenarios.tsv")}
    scenarios = (("replay", "Replay changes only"), ("apart", "Both classes shift together"), ("class", "Overlapping class changes"))
    columns = (("time_varying", primary, "primary"), ("time_varying", common, "common count"),
               ("class_change", primary, "primary"), ("class_change", common, "common count"))
    for g, (group, title) in enumerate((("cohort", "Matched to the five rats"), ("achilles", "Matched to Achilles"))):
        ax = fig.add_axes([x0 + 0.285 * w + g * 0.365 * w, y0, 0.33 * w, h])
        ax.set(xlim=(0, 4), ylim=(3, 0))
        ax.axis("off")
        for i, (scenario, label) in enumerate(scenarios):
            for j, (comparison, source, name) in enumerate(columns):
                r = source[(group, scenario, comparison)]
                k = int(r["training_scaled_exceedances"] if source is primary else r["crossings"])
                n = int(r["cohorts"])
                share = k / n
                shade = plt.matplotlib.colors.to_hex(np.array([1, 1, 1]) * (1 - share) + np.array(plt.matplotlib.colors.to_rgb(fs.INK2)) * share)
                ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, facecolor=shade, edgecolor=fs.RULE if share == 0 else "none", lw=0.5))
                ax.text(j + 0.5, i + 0.5, f"{k}", ha="center", va="center", fontsize=8, color="white" if share > 0.55 else fs.INK,
                        fontweight="bold" if share > 0 else "normal")
                records.append(("C", group, f"{scenario}; {comparison}", name, k, n))
            if g == 0:
                ax.text(-0.12, i + 0.5, label, ha="right", va="center", fontsize=8)
        for j, (comparison, source, name) in enumerate(columns):
            ax.text(j + 0.5, -0.10, "primary" if name == "primary" else "common", ha="center", va="bottom", fontsize=8, color=fs.INK2)
        for j0, name in ((0, "shared change"), (2, "class-specific")):
            ax.plot([j0 + 0.08, j0 + 1.92], [-0.62, -0.62], color=fs.INK2, lw=0.6)
            ax.text(j0 + 1.0, -0.68, name, ha="center", va="bottom", fontsize=8)
        ax.text(2.0, -1.25, title, ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.text(x0 + 0.285 * w, y0 - inch(0.16), "Numbers: cohorts out of 24 in which the change was found (shading in proportion).", ha="left",
             va="top", fontsize=8, color=fs.INK2)


def profiles(ax, records):
    rows = read(HERE / "evidence" / "figures" / "timing_profiles.tsv")
    by = defaultdict(list)
    boundary = {}
    for r in rows:
        if r["location"] == "boundary":
            continue
        by[r["recording"]].append((float(r["location"]), float(r["split_minus_one_law"])))
        boundary[r["recording"]] = float(r["boundary"])
    ax.axvspan(-0.10, 0.10, color=fs.PANEL, lw=0, zorder=0)
    ax.axvline(0, color=fs.INK, lw=0.8, zorder=1)
    near = {"first": [0, 0], "second": [0, 0]}
    for name, points in sorted(by.items()):
        if name.startswith("Cicero") or name.startswith("Gatsby"):
            continue
        points = np.array(sorted(points))
        rel = points[:, 1] - points[:, 1].max()
        x = points[:, 0] - boundary[name]
        first = name.startswith("Achilles")
        colour = fs.BLUE if first else fs.ORANGE
        ax.plot(x, rel, color=colour, lw=0.9, ls="-" if first else (0, (3, 1.5)), alpha=0.9, zorder=2)
        best = int(np.argmax(points[:, 1]))
        ax.plot(x[best], 0, marker="o" if first else "^", ms=4.6, mfc=colour, mec="white", mew=0.5, zorder=3, clip_on=False)
        key = "first" if first else "second"
        near[key][0] += abs(x[best]) <= 0.10
        near[key][1] += 1
        records.append(("D", name, f"{boundary[name]:.4f}", f"{points[best, 0]:.2f}", float(points[best, 1]), ""))
    ax.set(xlim=(-0.65, 0.92), ylim=(-260, 12))
    fs.ticks_minus(ax)
    ax.set_xlabel("Split minus the experience (fraction of ripples)", labelpad=2)
    ax.set_ylabel("Fit minus best split (nats)", labelpad=2)
    ax.text(0.0, 18, "experience", ha="center", va="bottom", fontsize=8)
    handles = [Line2D([], [], color=fs.BLUE, lw=0.9, marker="o", ms=4.4, label=f"Achilles: {near['first'][0]} of {near['first'][1]} best splits in band"),
               Line2D([], [], color=fs.ORANGE, lw=0.9, ls=(0, (3, 1.5)), marker="^", ms=4.4,
                      label=f"Five rats: {near['second'][0]} of {near['second'][1]}")]
    ax.figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.0), fontsize=8, borderaxespad=0.2, handlelength=2.2,
                     frameon=False, ncol=2, columnspacing=1.2)


def schedule(ax, records):
    rows = read(HERE / "evidence" / "hippocampus" / "schedule_predictions.tsv")
    contrasts = (("schedule_minus_stationary", "schedule\nvs no split"), ("learned_minus_schedule", "learned split\nvs schedule"),
                 ("given_minus_schedule", "true boundary\nvs schedule"))
    ax.axhline(0, color=fs.INK2, lw=0.6, zorder=1)
    for i, (contrast, label) in enumerate(contrasts):
        chosen = [r for r in rows if r["contrast"] == contrast]
        sessions = [r for r in chosen if r["recording"].startswith("RAT")]
        values = np.array([float(r["total_nats"]) for r in sessions])
        ax.plot(i + np.linspace(-0.16, 0.16, len(values)), values, ls="none", marker="^", ms=3.8, mfc="white", mec=fs.ORANGE, mew=0.8, zorder=2)
        total = sum(values)
        se = np.sqrt(sum(float(r["block_se"]) ** 2 for r in sessions))
        ax.errorbar(i + 0.36, total / len(values), yerr=1.96 * se / len(values), marker="D", ms=4.8, color=fs.ORANGE, capsize=0, lw=1.3, zorder=3)
        records.append(("E", contrast, "nine sessions", f"{total:.2f}", f"{se:.2f}", f"{int((values > 0).sum())} of {len(values)} positive"))
    ax.set(xlim=(-0.5, 2.65), xticks=range(3))
    ax.set_xticklabels([c[1] for c in contrasts])
    ax.tick_params(axis="x", length=0, pad=3)
    fs.ticks_minus(ax, "y")
    ax.set_ylabel("Gain per session (nats)", labelpad=2)


def draw(output, source):
    fs.pro()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    records = []
    fs.letter_title(fig, 0.0, inch(8.25), "A", "Whole ripples, one cell at a time, beyond the replay model")
    schematic(fig, [0.0, inch(6.28), 1.0, inch(1.80)])
    fs.letter_title(fig, 0.0, inch(5.98), "B", "Gain on held-out ripples from each addition")
    gains(fig, [0.0, inch(4.45), 1.0, inch(1.12)], records)
    fs.letter_title(fig, 0.0, inch(3.90), "C", "Calibration in simulated cohorts")
    calibration(fig, [0.0, inch(2.62), 1.0, inch(0.72)], records)
    fs.letter_title(fig, 0.0, inch(2.08), "D", "Replay change without the boundary")
    profiles(fig.add_axes([0.085, inch(0.66), 0.40, inch(1.20)]), records)
    fs.letter_title(fig, 0.56, inch(2.08), "E", "Where to split the replay model (five rats)")
    schedule(fig.add_axes([0.655, inch(0.66), 0.335, inch(1.20)]), records)
    fs.save_pro(fig, output)
    with Path(source).open("w", newline="") as handle:
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
