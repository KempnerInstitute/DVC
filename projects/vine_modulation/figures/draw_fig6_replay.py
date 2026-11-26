"""Fig 6: a model of replayed track stretches, what it decodes from real ripples, how replay changed, how much of the
three-way change it reproduces, and a check of the replay-relative estimate where the truth is known.

    draw_fig6_replay.py OUTPUT_PDF SOURCE_CSV

A  the replay model (schematic).
B  three ripples of the rest after the tracks (rat 3, second session; evidence/figures/hippocampus_examples.json):
   spikes of the 96 place cells in ten 10-ms steps, cells ordered by running direction and field peak, and beside each
   row the posterior probability of the fitted model that the ripple replays a stretch within 12 cm of that cell's
   field peak.
C  fitted share of ripples on a stretch of at most about 1 m, in each rest (evidence/hippocampus/content_laws_*.tsv).
D  change after minus before, overlapping minus separate trios: recorded (95% interval), mean and central 95% of 200
   cohorts simulated from the fitted models, and relative to the models' prediction from the other cells (95%
   interval); evidence/hippocampus/content_*.tsv.
E  simulated populations with the second cohort's cells and ripples (32 cohorts per scenario): the main estimate and
   the replay-relative estimate of the three-way change, mean and central 95% (evidence/figures/e66_check.tsv).
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
HEIGHT = 7.45
RATS = {"RAT1or4_SESS": 1, "RAT2_SESS1": 2, "RAT2_SESS2": 2, "RAT3_SESS1": 3, "RAT3_SESS2": 3, "RAT1_SESS1": 4, "RAT4_SESS2": 4,
        "RAT5_SESS1": 5, "RAT5_SESS2": 5}
RAT_MARKERS = {1: "s", 2: "^", 3: "D", 4: "v", 5: "P"}


def inch(value):
    return value / HEIGHT


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def schematic(ax):
    fs.blank(ax)
    x = np.linspace(0.02, 0.98, 400)
    centres = np.linspace(0.10, 0.90, 7)
    base, top = 0.58, 0.24
    on = (2, 3, 4)
    for k, c in enumerate(centres):
        y = base + top * np.exp(-0.5 * ((x - c) / 0.045) ** 2)
        if k in on:
            ax.fill_between(x, base, y, color=fs.BLUE_LIGHT, lw=0, zorder=1)
        ax.plot(x, y, color=fs.INK if k in on else fs.MUTED, lw=1.0, zorder=2)
        ax.text(c, base + top + 0.03, str(k + 1), ha="center", va="bottom", fontsize=fs.SMALL, color=fs.INK2)
    ax.plot([0.02, 0.98], [base, base], color=fs.INK2, lw=1.4)
    left, right = centres[2] - 0.07, centres[4] + 0.07
    ax.plot([left, left, right, right], [base - 0.025, base - 0.06, base - 0.06, base - 0.025], color=fs.BLUE, lw=1.4)
    ax.text((left + right) / 2, base - 0.08, "a replayed stretch, at one of five strengths", ha="center", va="top", fontsize=fs.SMALL)
    ax.text(0.02, 0.36, "Given the stretch and its strength, each place cell takes part\nindependently, with a probability set by its running field.",
            ha="left", va="top", fontsize=fs.SMALL, linespacing=1.2)
    ax.text(0.02, 0.13, "One model per rest; a stretch can be a 4-cm bin, a piece of\nabout 40 cm or 1 m, a whole direction, or no location.",
            ha="left", va="top", fontsize=fs.SMALL, linespacing=1.2, color=fs.INK2)


def decoded(fig, rect, examples, records):
    """B: the running fields of the cells that fired in each ripple, on the running direction the model decodes."""
    x0, y0, w, h = rect
    positions = np.arange(50) * 0.04 + 0.02
    for i, ripple in enumerate(examples["ripples"]):
        ax = fig.add_axes([x0 + 0.06 * w + i * 0.325 * w, y0, 0.27 * w, h])
        centre, half = ripple["best_centre_bin"], ripple["best_half_width_bins"]
        lo, hi = 0.04 * max(centre - half, 0), 0.04 * min(centre + half + 1, 50)
        ax.axvspan(lo, hi, color=fs.BLUE_LIGHT, lw=0, zorder=0)
        maps = np.asarray(ripple["fired_maps_on_best_map"])
        here = maps[maps.max(1) >= 0.5]
        for rate in here:
            ax.plot(positions, rate, color=fs.INK2, lw=0.6, alpha=0.8, zorder=2)
        peaks = positions[np.argmax(here, axis=1)]
        ax.plot(peaks, np.full(len(peaks), 1.07), ls="none", marker="v", ms=3.2, color=fs.INK, clip_on=False, zorder=3)
        ax.set(xlim=(0, 2.0), ylim=(0, 1.12), xticks=[0, 1, 2], yticks=[0, 1] if i == 0 else [])
        if i == 0:
            ax.set_ylabel("Field (peak = 1)", labelpad=1)
        ax.set_xlabel("Position (m)", labelpad=1)
        inside = int(((peaks >= lo) & (peaks <= hi)).sum())
        ax.set_title(f"{ripple['active_cells']} fired; {len(here)} fields here,\n{inside} peak in the stretch", fontsize=fs.SMALL, pad=9, linespacing=1.05)
        records.append(("B", int(ripple["index"]), int(ripple["active_cells"]), f"direction {ripple['best_map'] + 1}; {lo:.2f}-{hi:.2f} m",
                        float(ripple["best_state_probability"]), f"{len(here)} fields; {inside} inside"))
    kinds = examples["decoded_kinds_after"]
    stretch = kinds["bin"] + kinds["within_20cm"] + kinds["within_48cm"]
    fig.text(x0 + 0.06 * w, y0 - inch(0.38), f"Shaded: the decoded stretch. In this rest, {100 * kinds['none']:.0f}% of ripples were most\n"
             f"probably no location and {100 * stretch:.0f}% a stretch of at most about 1 m.",
             fontsize=fs.SMALL, color=fs.INK2, ha="left", va="top", linespacing=1.2)


def shares(ax, records):
    folder = HERE / "evidence" / "hippocampus"
    stretch = lambda r: sum(float(r[f"weight_{k}"]) for k in ("bin", "within_20cm", "within_48cm"))
    for position, (cohort, label, colour) in enumerate((("first", "Achilles", fs.BLUE), ("second", "Five rats", fs.ORANGE))):
        value = {}
        for r in read(folder / f"content_laws_{cohort}.tsv"):
            if cohort == "first" and not r["session"].startswith("Achilles"):
                continue
            value.setdefault(r["session"], {})[r["rest"]] = stretch(r)
        left = 2.3 * position
        for unit, v in sorted(value.items()):
            marker = "o" if cohort == "first" else RAT_MARKERS[RATS[unit]]
            ax.plot([left, left + 1], [v["PRE"], v["POST"]], color=colour, lw=0.9, marker=marker, ms=4.4, mfc="white", mec=colour, mew=1.0)
            records.append(("C", label, unit, "", v["PRE"], v["POST"]))
        ax.text(left + 0.5, 1.04, label, ha="center", va="bottom", fontsize=fs.SMALL)
    ax.set(xlim=(-0.4, 3.7), ylim=(0, 1.02), xticks=[0, 1, 2.3, 3.3], yticks=[0, 0.5, 1.0])
    ax.set_xticklabels(["before", "after", "before", "after"], fontsize=fs.SMALL)
    ax.set_ylabel("Share of ripples on a\nstretch of at most 1 m")
    handles = [Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.BLUE, mew=1.0, ms=4.4, label="hemisphere")]
    handles += [Line2D([], [], ls="none", marker=m, mfc="white", mec=fs.ORANGE, mew=1.0, ms=4.4, label=f"rat {r}") for r, m in RAT_MARKERS.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=3, fontsize=fs.SMALL, handletextpad=0.1, columnspacing=0.5,
              borderaxespad=0)


def changes(fig, rect, records):
    x0, y0, w, h = rect
    folder = HERE / "evidence" / "hippocampus"
    tables = {c: {(r["group"], r["estimate"], r["term"]): r for r in read(folder / f"content_{c}.tsv")} for c in ("first", "second")}
    rows = (("recorded", "recorded", "o", True), ("law", "simulated", "s", False), ("adjusted", "relative", "^", True))
    for k, (term, title, limits) in enumerate((("pairs", "Pair contrast", (-0.12, 0.6)), ("trios", "Three-way contrast", (-0.2, 0.45)))):
        ax = fig.add_axes([x0 + (0.20 + 0.40 * k) * w, y0, 0.27 * w, h])
        fs.zero(ax, "x")
        y = 0.0
        ticks, names = [], []
        for cohort, group, label, colour in (("first", "achilles", "Achilles", fs.BLUE), ("second", "cohort", "Five rats", fs.ORANGE)):
            ticks.append(y)
            names.append(label)
            y -= 0.9
            for estimate, name, marker, filled in rows:
                r = tables[cohort][(group, "adjusted" if estimate == "adjusted" else "plain", term)]
                if estimate == "law":
                    value, low, high = float(r["their_mean"]), float(r["their_2.5th_percentile"]), float(r["their_97.5th_percentile"])
                else:
                    value, se = float(r["recorded_difference"]), float(r["se"])
                    low, high = value - 1.96 * se, value + 1.96 * se
                ax.plot([low, high], [y, y], color=colour, lw=1.3)
                ax.plot(value, y, marker=marker, ms=5.0, mfc=colour if filled else "white", mec=colour, mew=1.0)
                ax.text(1.02, y, f"{value:+.2f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=fs.SMALL, color=fs.INK2)
                ticks.append(y)
                names.append(name)
                records.append(("D", term, label, name.replace("\n", " "), value, f"{low:.4f};{high:.4f}"))
                y -= 0.9
            y -= 0.3
        ax.set(xlim=limits, ylim=(y + 0.6, 0.5))
        ax.set_yticks(ticks)
        ax.set_yticklabels(names if k == 0 else [""] * len(names), fontsize=fs.SMALL)
        for tick, name in zip(ax.get_yticklabels(), names):
            if name in ("Achilles", "Five rats"):
                tick.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        fs.spines(ax, ("bottom",))
        ax.set_title(title, fontsize=fs.TEXT, pad=3)
        ax.set_xlabel("After minus before", labelpad=1)


def check(ax, records):
    rows = {(r["scenario"], r["quantity"]): r for r in read(HERE / "evidence" / "figures" / "e66_check.tsv")}
    scenarios = (("same", "nothing changes"), ("rare", "replay rarer"), ("replay", "replay more frequent"), ("stronger", "replay stronger"),
                 ("planted", "a three-way change\nbeyond replay"))
    fs.zero(ax, "x")
    for i, (scenario, label) in enumerate(scenarios):
        y = -i * 1.0 - (0.4 if scenario == "planted" else 0)
        for quantity, colour, marker, filled, dy in (("plain_trios", fs.INK2, "o", False, 0.17), ("adjusted_trios", fs.ORANGE, "^", True, -0.17)):
            r = rows[(scenario, quantity)]
            mean, low, high = float(r["mean"]), float(r["2.5th_percentile"]), float(r["97.5th_percentile"])
            ax.plot([low, high], [y + dy, y + dy], color=colour, lw=1.2)
            ax.plot(mean, y + dy, marker=marker, ms=4.8, mfc=colour if filled else "white", mec=colour, mew=1.0)
            records.append(("E", scenario, quantity, "", mean, f"{low:.4f};{high:.4f}"))
        ax.text(-0.04, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=fs.SMALL, linespacing=1.05)
    ax.axhline(-3.7, color=fs.GRID, lw=0.8)
    ax.set(xlim=(-0.3, 0.7), ylim=(-4.8, 0.6), yticks=[], xticks=[-0.2, 0, 0.2, 0.4, 0.6])
    fs.spines(ax, ("bottom",))
    ax.set_xlabel("Estimated three-way change", labelpad=1)
    handles = [Line2D([], [], color=fs.INK2, marker="o", mfc="white", mec=fs.INK2, ms=4.8, lw=1.2, label="main estimate"),
               Line2D([], [], color=fs.ORANGE, marker="^", mfc=fs.ORANGE, mec=fs.ORANGE, ms=4.8, lw=1.2, label="relative to replay")]
    ax.legend(handles=handles, loc="upper right", fontsize=fs.SMALL, handlelength=1.8, borderaxespad=0.1)


def draw(output, source):
    fs.print_style()
    examples = json.loads((HERE / "evidence" / "figures" / "hippocampus_examples.json").read_text())
    fig = plt.figure(figsize=(fs.WIDTH, HEIGHT))
    records = []
    fs.heading(fig, 0.0, 1 - inch(0.17), "A", "A model of replayed stretches")
    schematic(fig.add_axes([0.0, inch(5.55), 0.40, inch(1.62)]))
    fs.heading(fig, 0.43, 1 - inch(0.17), "B", "Fields of the cells that fired in three ripples (rat 3)")
    decoded(fig, [0.43, inch(5.78), 0.57, inch(1.0)], examples, records)
    fs.heading(fig, 0.0, inch(4.88), "C", "Replay changed")
    shares(fig.add_axes([0.085, inch(2.98), 0.22, inch(1.62)]), records)
    fs.heading(fig, 0.34, inch(4.88), "D", "How much of the change replay reproduces")
    changes(fig, [0.34, inch(3.05), 0.66, inch(1.62)], records)
    fs.heading(fig, 0.0, inch(2.05), "E", "The replay-relative estimate where the truth is known (simulation)")
    check(fig.add_axes([0.27, inch(0.42), 0.42, inch(1.45)]), records)
    note = fig.add_axes([0.72, inch(0.42), 0.27, inch(1.45)])
    fs.blank(note)
    note.text(0.0, 0.95, "Changes of replay alone move the\nmain estimate (stronger replay:\n+0.45) but not the replay-relative\none, which keeps most of a change\nplanted beyond replay.",
              ha="left", va="top", fontsize=fs.SMALL, linespacing=1.25)
    fs.save_journal(fig, output)
    with source.open("w", newline="") as handle:
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
