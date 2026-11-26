"""Fig 6: a model of replayed track stretches, what it decodes from real ripples, how replay changed, how much of the
three-way change it reproduces, and a check of the replay-relative estimate where the truth is known.

    draw_fig6_replay.py OUTPUT_PDF SOURCE_CSV

A  the replay model (schematic: the fields, cells and the ripple pattern are drawings).
B  three ripples of the rest after the tracks (rat 3, second session; evidence/figures/hippocampus_examples.json): the
   fitted model's posterior over the centre of the replayed stretch on the running direction it decodes (summed over
   stretch widths and strengths), and below it the running fields (scaled to a peak of 1) of the cells that fired in the
   ripple and have a field on that direction (peak at least half the cell's largest), ordered by peak, with the most
   probable stretch outlined.  The three are the ripples with the most active cells among those whose most probable
   state is a stretch of about 40 cm or 1 m with probability at least 0.8.
C  fitted share of ripples on a stretch of at most about 1 m, in each rest (evidence/hippocampus/content_laws_*.tsv).
D  change after minus before, overlapping minus separate trios: recorded (95% interval), mean and central 95% of 200
   cohorts simulated from the fitted models, the same with one rate factor per cell shared by both rests (E78;
   evidence/hippocampus/shared_factor_content.tsv), and relative to the models' prediction from the other cells (95%
   interval); evidence/hippocampus/content_*.tsv.
E  simulated populations with the second cohort's cells and ripples (32 cohorts per scenario): the main estimate and
   the replay-relative estimate of the three-way change, mean and central 95% (evidence/figures/e66_check.tsv).  The
   scenarios of experiments/e66_content_law_check.py: nothing changes (share of ripples with a place 0.3 in both rests);
   rare replay triples (0.02 to 0.06); replay doubles (0.25 to 0.5); replay becomes stronger (share 0.3, weak to strong
   gains); and a three-way change planted in the overlapping trios while replay doubles.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 7.25, 7.70
FIELDS = LinearSegmentedColormap.from_list("fields", ["#ffffff", "#d6d6d6", "#8a8a8a", "#3a3a3a", "#111111"])
STRETCH = fs.BLUE


def inch(value):
    return value / HEIGHT


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def schematic(fig, rect):
    ax = fs.canvas(fig, rect, xlim=(0, 1))
    top = ax.get_ylim()[1]
    # running fields along the track, a replayed stretch over three of them
    x = np.linspace(0.14, 0.64, 400)
    centres = np.linspace(0.18, 0.60, 7)
    base, height = top - 0.165, 0.085
    left, right = centres[2] - 0.036, centres[4] + 0.036
    ax.add_patch(Rectangle((left, base - 0.165), right - left, 0.165 + height + 0.02, facecolor=fs.BLUE_TINT, edgecolor="none", zorder=0))
    ax.text((left + right) / 2, base + height + 0.030, "replayed stretch", ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=fs.BLUE_DARK)
    for k, c in enumerate(centres):
        y = base + height * np.exp(-0.5 * ((x - c) / 0.026) ** 2)
        ax.plot(x, y, color=fs.INK if 2 <= k <= 4 else fs.MUTED, lw=1.0, zorder=2)
    ax.plot([0.14, 0.64], [base, base], color=fs.INK2, lw=0.8, zorder=2)
    ax.text(0.115, base + 0.040, "running\nfields", ha="right", va="center", fontsize=8, color=fs.INK2, linespacing=1.1)
    # the cells under their fields: likely participants inside the stretch
    yc = base - 0.075
    fired = (False, False, True, True, True, False, True)
    for k, c in enumerate(centres):
        inside = 2 <= k <= 4
        fs.pyramidal(ax, c, yc - 0.012, 0.026, face=fs.INK2 if inside else "white", edge=fs.INK2, lw=0.8, zorder=3)
    ax.text(0.115, yc, "cells", ha="right", va="center", fontsize=8, color=fs.INK2)
    # one generated ripple: independent participation
    yr = yc - 0.085
    for k, c in enumerate(centres):
        ax.add_patch(Rectangle((c - 0.016, yr - 0.016), 0.032, 0.032, facecolor=fs.INK if fired[k] else "white", edgecolor=fs.INK2,
                               lw=0.7, zorder=3))
        ax.text(c, yr - 0.028, "1" if fired[k] else "0", ha="center", va="top", fontsize=8)
    ax.text(0.115, yr, "ripple", ha="right", va="center", fontsize=8, color=fs.INK2)
    for k, line in enumerate(("Given state and strength, cells take part", "independently, each as its field predicts")):
        ax.text(0.0, yr - 0.085 - 0.040 * k, line, ha="left", va="top", fontsize=8)
    # the states and strengths
    sx = 0.715
    ax.text(sx, top - 0.030, "States", ha="left", va="center", fontsize=8, fontweight="bold")
    states = (("4-cm bin", 0.015), ("about 40 cm", 0.10), ("about 1 m", 0.22), ("whole direction", 1.0), ("no location", 0.0))
    for k, (label, width) in enumerate(states):
        yy = top - 0.080 - 0.052 * k
        ax.add_patch(FancyBboxPatch((sx, yy - 0.009), 0.10, 0.018, boxstyle="round,pad=0,rounding_size=0.008", facecolor="white",
                                    edgecolor=fs.INK2, lw=0.6, ls="-" if width else (0, (2, 1.5)), zorder=2))
        if width:
            w = max(0.10 * width, 0.004)
            ax.add_patch(Rectangle((sx + 0.05 - w / 2, yy - 0.009), w, 0.018, facecolor=STRETCH, edgecolor="none", zorder=3))
        ax.text(sx + 0.115, yy, label, ha="left", va="center", fontsize=8)
    ys = top - 0.080 - 0.052 * 5 - 0.020
    ax.text(sx, ys, "Five strengths", ha="left", va="center", fontsize=8, fontweight="bold")
    for k, g in enumerate((0.25, 0.5, 1.0, 2.0, 4.0)):
        h = 0.010 + 0.011 * np.log2(g / 0.25)
        ax.add_patch(Rectangle((sx + 0.004 + 0.024 * k, ys - 0.100), 0.016, h, facecolor=fs.INK2, edgecolor="none"))
    ax.text(sx + 0.135, ys - 0.075, "gain 0.25 to 4", ha="left", va="center", fontsize=8, color=fs.INK2)
    ax.text(sx, ys - 0.150, "One model per rest", ha="left", va="center", fontsize=8, color=fs.INK2)


def decoded(fig, rect, examples, records):
    """B: posterior over the replayed stretch and the fields of the cells that fired, on the decoded direction."""
    x0, y0, w, h = rect
    edges = np.arange(51) * 0.04
    positions = edges[:-1] + 0.02
    for i, ripple in enumerate(examples["ripples"]):
        left = x0 + i * 0.345 * w
        width = 0.29 * w
        m = ripple["best_map"]
        post = np.asarray(ripple["position_posterior"][str(m)])
        centre, half = ripple["best_centre_bin"], ripple["best_half_width_bins"]
        lo, hi = 0.04 * max(centre - half, 0), 0.04 * min(centre + half + 1, 50)
        top_ax = fig.add_axes([left, y0 + 0.74 * h, width, 0.22 * h])
        top_ax.fill_between(positions, 0, post, step="mid", color=STRETCH, lw=0, alpha=0.85)
        top_ax.set(xlim=(0, 2.0), ylim=(0, 1.0), xticks=[], yticks=[0, 1] if i == 0 else [])
        for side in ("left", "right", "top"):
            top_ax.spines[side].set_visible(i == 0 and side == "left")
        if i == 0:
            top_ax.set_ylabel("posterior", labelpad=1)
        title_ax = top_ax
        maps = np.asarray(ripple["fired_maps_on_best_map"])
        here = maps[maps.max(1) >= 0.5]
        order = np.argsort(np.argmax(here, axis=1), kind="stable")
        here = here[order] / here[order].max(1, keepdims=True)
        ax = fig.add_axes([left, y0, width, 0.68 * h])
        ax.imshow(here, aspect="auto", cmap=FIELDS, vmin=0, vmax=1, extent=(0, 2.0, len(here) + 0.5, 0.5), interpolation="nearest")
        ax.add_patch(Rectangle((lo, 0.5), hi - lo, len(here), facecolor="none", edgecolor=STRETCH, lw=1.3, zorder=3))
        peaks = positions[np.argmax(here, axis=1)]
        inside = int(((peaks >= lo) & (peaks <= hi)).sum())
        ax.set(xlim=(0, 2.0), ylim=(len(here) + 0.5, 0.5), xticks=[0, 1, 2], yticks=[1, len(here)] if i == 0 else [])
        ax.set_xlabel("Position (m)", labelpad=1)
        if i == 0:
            ax.set_ylabel("Fired cell with a field here", labelpad=1)
        title_ax.text(0.0, 1.30, f"{ripple['active_cells']} cells fired", transform=title_ax.transAxes, ha="left", va="bottom", fontsize=8)
        title_ax.text(0.0, 1.04, f"{inside} of {len(here)} inside", transform=title_ax.transAxes, ha="left", va="bottom", fontsize=8,
                      color=fs.INK2)
        records.append(("B", int(ripple["index"]), int(ripple["active_cells"]), f"direction {m + 1}; {lo:.2f}-{hi:.2f} m",
                        float(ripple["best_state_probability"]), f"{len(here)} fields; {inside} inside"))


def shares(ax, records):
    folder = HERE / "evidence" / "hippocampus"
    stretch = lambda r: sum(float(r[f"weight_{k}"]) for k in ("bin", "within_20cm", "within_48cm"))
    summary = []
    for position, (cohort, label, colour, marker) in enumerate((("first", "Achilles", fs.BLUE, "o"), ("second", "Five rats", fs.ORANGE, "^"))):
        value = {}
        for r in read(folder / f"content_laws_{cohort}.tsv"):
            if cohort == "first" and not r["session"].startswith("Achilles"):
                continue
            value.setdefault(r["session"], {})[r["rest"]] = stretch(r)
        left = 2.4 * position
        rises = 0
        for unit, v in sorted(value.items()):
            ax.plot([left, left + 1], [v["PRE"], v["POST"]], color=colour, lw=0.9, marker=marker, ms=4.6, mfc="white", mec=colour,
                    mew=1.0, alpha=0.95)
            rises += v["POST"] > v["PRE"]
            records.append(("C", label, unit, "", v["PRE"], v["POST"]))
        ax.text(left + 0.5, 1.03, label, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(left + 0.5, -0.20, f"{rises} of {len(value)} rise", ha="center", va="top", fontsize=8, color=fs.INK2)
        summary.append((label, rises, len(value)))
    ax.set(xlim=(-0.4, 3.8), ylim=(0, 1.0), xticks=[0, 1, 2.4, 3.4], yticks=[0, 0.5, 1.0])
    ax.set_xticklabels(["before", "after", "before", "after"])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("Share of ripples on a\nstretch of at most 1 m", labelpad=2)
    return summary


def changes(fig, rect, records):
    x0, y0, w, h = rect
    folder = HERE / "evidence" / "hippocampus"
    tables = {c: {(r["group"], r["estimate"], r["term"]): r for r in read(folder / f"content_{c}.tsv")} for c in ("first", "second")}
    shared = {(r["cohort"], r["group"], r["estimate"], r["term"]): r for r in read(folder / "shared_factor_content.tsv")}   # E78
    rows = (("recorded", "recorded", "o", True), ("law", "simulated from the model", "s", False),
            ("shared", "simulated, one factor per cell", "D", False), ("adjusted", "relative to replay", "^", True))
    layout = []
    y = 0.0
    for cohort, group, label, colour in (("first", "achilles", "Achilles", fs.BLUE), ("second", "cohort", "Five rats", fs.ORANGE)):
        layout.append(("head", label, colour, y))
        y -= 1.0
        for estimate, name, marker, filled in rows:
            layout.append((estimate, name, (cohort, group, label, colour, marker, filled), y))
            y -= 1.0
        y -= 0.35
    bottom = y + 0.35 - 0.6
    for k, (term, title, limits) in enumerate((("pairs", "Pair contrast", (-0.12, 0.6)), ("trios", "Three-way contrast", (-0.2, 0.45)))):
        ax = fig.add_axes([x0 + (0.30 + 0.37 * k) * w, y0, 0.24 * w, h])
        ax.axvline(0, color=fs.INK2, lw=0.6, zorder=1)
        for estimate, name, info, yy in layout:
            if estimate == "head":
                if k == 0:
                    ax.text(-0.06, yy, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8, fontweight="bold",
                            color=fs.BLUE if name == "Achilles" else fs.ORANGE_DARK)
                continue
            cohort, group, label, colour, marker, filled = info
            r = tables[cohort][(group, "adjusted" if estimate == "adjusted" else "plain", term)]
            if estimate == "law":
                value, low, high = float(r["their_mean"]), float(r["their_2.5th_percentile"]), float(r["their_97.5th_percentile"])
            elif estimate == "shared":
                q = shared[(cohort, group, "plain", term)]
                value, low, high = float(q["shared_mean"]), float(q["shared_low"]), float(q["shared_high"])
            else:
                value, se = float(r["recorded_difference"]), float(r["se"])
                low, high = value - 1.96 * se, value + 1.96 * se
            if estimate == "recorded":
                ax.axhspan(yy - 0.5, yy + 0.5, color=fs.PANEL, lw=0, zorder=0)
            ax.plot([low, high], [yy, yy], color=colour, lw=1.2, zorder=2, solid_capstyle="butt")
            ax.plot(value, yy, marker=marker, ms=5.0 if marker != "D" else 4.4, mfc=colour if filled else "white", mec=colour, mew=1.0, zorder=3)
            ax.text(1.04, yy, fs.minus(f"{value:.2f}"), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8)
            if k == 0:
                ax.text(-0.06, yy, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8)
            records.append(("D", term, label, name, value, f"{low:.4f};{high:.4f}"))
        ax.set(xlim=limits, ylim=(bottom, 0.6), yticks=[])
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        fs.ticks_minus(ax, "x")
        ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xlabel("After minus before", labelpad=2)


def check(fig, rect, records):
    rows = {(r["scenario"], r["quantity"]): r for r in read(HERE / "evidence" / "figures" / "e66_check.tsv")}
    scenarios = (("same", "Nothing changes"), ("rare", "Rare replay, tripled"), ("replay", "Replay doubled"), ("stronger", "Replay stronger"),
                 ("planted", "Three-way change beyond replay"))
    x0, y0, w, h = rect
    ax = fig.add_axes([x0 + 0.36 * w, y0, 0.50 * w, h])
    ax.axvline(0, color=fs.INK2, lw=0.6, zorder=1)
    for i, (scenario, label) in enumerate(scenarios):
        y = -i * 1.0 - (0.45 if scenario == "planted" else 0)
        if scenario == "planted":
            ax.axhspan(y - 0.5, y + 0.5, color=fs.PANEL, lw=0, zorder=0)
        for quantity, colour, marker, filled, dy in (("plain_trios", fs.INK2, "o", False, 0.17), ("adjusted_trios", fs.BLUE, "s", True, -0.17)):
            r = rows[(scenario, quantity)]
            mean, low, high = float(r["mean"]), float(r["2.5th_percentile"]), float(r["97.5th_percentile"])
            ax.plot([low, high], [y + dy, y + dy], color=colour, lw=1.2, zorder=2, solid_capstyle="butt")
            ax.plot(mean, y + dy, marker=marker, ms=4.8, mfc=colour if filled else "white", mec=colour, mew=1.0, zorder=3)
            records.append(("E", scenario, quantity, "", mean, f"{low:.4f};{high:.4f}"))
        ax.text(-0.04, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8,
                fontweight="bold" if scenario == "planted" else "normal")
    ax.set(xlim=(-0.3, 0.7), ylim=(-4.95, 0.6), yticks=[], xticks=[-0.2, 0, 0.2, 0.4, 0.6])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    fs.ticks_minus(ax, "x")
    ax.set_xlabel("Estimated three-way change, after minus before", labelpad=2)
    handles = [Line2D([], [], color=fs.INK2, marker="o", mfc="white", mec=fs.INK2, ms=4.8, lw=1.2, label="main estimate"),
               Line2D([], [], color=fs.BLUE, marker="s", mfc=fs.BLUE, mec=fs.BLUE, ms=4.8, lw=1.2, label="relative to replay")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, handlelength=1.8, borderaxespad=0.1, ncol=2,
              frameon=False)
    stronger = float(rows[("stronger", "plain_trios")]["mean"])
    kept = float(rows[("planted", "adjusted_trios")]["mean"]) / float(rows[("planted", "plain_trios")]["mean"])
    return stronger, kept


def draw(output, source):
    fs.pro()
    examples = json.loads((HERE / "evidence" / "figures" / "hippocampus_examples.json").read_text())
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    records = []
    fs.letter_title(fig, 0.0, inch(7.53), "A", "A model of replayed stretches")
    schematic(fig, [0.0, inch(4.95), 0.505, inch(2.40)])
    fs.letter_title(fig, 0.535, inch(7.53), "B", "Three ripples decoded (rat 3, after the tracks)")
    decoded(fig, [0.590, inch(5.30), 0.410, inch(1.84)], examples, records)
    fs.letter_title(fig, 0.0, inch(4.60), "C", "Replay changed")
    summary = shares(fig.add_axes([0.085, inch(2.92), 0.22, inch(1.38)]), records)
    fs.letter_title(fig, 0.36, inch(4.60), "D", "How much of the change replay reproduces")
    changes(fig, [0.36, inch(2.62), 0.64, inch(1.72)], records)
    fs.letter_title(fig, 0.0, inch(2.02), "E", "The replay-relative estimate where the truth is known (simulation)")
    stronger, kept = check(fig, [0.0, inch(0.42), 0.78, inch(1.30)], records)
    note = fig.add_axes([0.80, inch(0.42), 0.20, inch(1.30)])
    fs.blank(note)
    lines = ("Changes of replay alone", "move the main estimate", fs.minus(f"(stronger replay: {stronger:.2f})"), "but not the replay-",
             "relative one, which keeps", f"about {100 * kept:.0f}% of a change", "planted beyond replay.")
    for k, line in enumerate(lines):
        note.text(0.0, 0.98 - 0.135 * k, line, ha="left", va="top", fontsize=8, color=fs.INK2)
    fs.save_pro(fig, output)
    with Path(source).open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "quantity_or_term", "row", "series", "value_or_before", "after_or_interval"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
