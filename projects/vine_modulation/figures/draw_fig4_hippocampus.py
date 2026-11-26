"""Fig 4: the hippocampal design, the classes from running, what a ripple contributes, the estimator at fixed pair
margins, and the change of the pair and three-way terms in every animal.

    draw_fig4_hippocampus.py OUTPUT_PDF SOURCE_CSV

A  the design of both cohorts (schematic; the ripple trace is a drawing, not a recording).
B  4-cm rate maps during running of one overlapping and one separate trio of rat 3, second session
   (evidence/figures/hippocampus_examples.json, chosen by the rules recorded there).
C  one ripple of the rest after the tracks in the same session: spikes of its 96 place cells in ten 10-ms steps, cells
   ordered by running direction and field peak; and the counts of the eight patterns of the overlapping trio of B over
   the ripples of each rest.
D  the estimator (schematic).
E, F  change from the rest before to the rest after, overlapping minus separate classes, with 95% intervals
   (estimate +/- 1.96 standard errors including the cohort adjustments of S1 Text): pair term (E) and three-way term
   (F); evidence/hippocampus/first_cohort_*.tsv and second_cohort.tsv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
HEIGHT = 8.15
ROWS = (("Achilles (2 sessions)", "achilles", "first_cohort_reserved", "se_common_factor", "first"),
        ("Cicero (3)", "cicero", "first_cohort_reserved", "se_common_factor", "first"),
        ("Buddy (1)", "buddy", "first_cohort_reserved", "se_common_factor", "first"),
        ("Achilles and Cicero", "achilles_cicero", "first_cohort_reserved", "se_common_factor", "pooled_first"),
        ("Gatsby (2; development)", "gatsby", "first_cohort_development", "se_calibrated", "development"),
        *[(f"Rat {i} ({1 if i == 1 else 2})", f"rat{i}", "second_cohort", "se_rule", "second") for i in range(1, 6)],
        ("Five rats (9)", "cohort", "second_cohort", "se_rule", "pooled_second"))
STYLE = {"first": (fs.BLUE, "o", True), "pooled_first": (fs.BLUE, "D", True), "development": (fs.BLUE, "o", False),
         "second": (fs.ORANGE, "^", True), "pooled_second": (fs.ORANGE, "D", True)}


def inch(value):
    return value / HEIGHT


def design(ax):
    fs.blank(ax)
    for x, w, text, face in ((0.00, 0.28, "Rest before", fs.GREY_LIGHT), (0.36, 0.28, "Novel maze\nor tracks", fs.BLUE_LIGHT),
                             (0.72, 0.28, "Rest after", fs.GREY_LIGHT)):
        fs.box(ax, x, 0.70, w, 0.22, None, face=face, edge=fs.INK2, radius=0.02)
        ax.text(x + w / 2, 0.81, text, ha="center", va="center", fontsize=fs.TEXT, linespacing=1.05)
    for a, b in ((0.28, 0.36), (0.64, 0.72)):
        fs.arrow(ax, (a + 0.005, 0.81), (b - 0.005, 0.81), color=fs.INK2, lw=0.9, scale=8)
    rng = np.random.default_rng(3)
    for x0 in (0.0, 0.72):
        for t in np.sort(rng.uniform(x0 + 0.02, x0 + 0.26, 9)):
            ax.plot([t, t], [0.60, 0.66], color=fs.INK, lw=0.8)
        ax.text(x0 + 0.14, 0.55, "ripples", ha="center", va="top", fontsize=fs.SMALL, color=fs.INK2)
    ax.text(0.50, 0.62, "fields measured\nduring running", ha="center", va="top", fontsize=fs.SMALL, color=fs.INK2, linespacing=1.05)
    t = np.linspace(-1, 1, 400)
    wave = np.exp(-(t / 0.28) ** 2) * np.sin(2 * np.pi * 9 * t)
    ax.add_patch(plt.Rectangle((0.08, 0.10), 0.36, 0.30, facecolor=fs.ORANGE_LIGHT, edgecolor="none", zorder=0))
    ax.plot(0.02 + (t + 1) / 2 * 0.48, 0.25 + 0.10 * wave, color=fs.INK2, lw=0.8)
    ax.text(0.26, 0.06, "100 ms around the ripple peak", ha="center", va="top", fontsize=fs.SMALL)
    fs.arrow(ax, (0.80, 0.585), (0.62, 0.36), color=fs.MUTED, lw=0.7, scale=6, connection="arc3,rad=-0.25")
    ax.text(0.56, 0.36, "each place cell:\nfired (1) or not (0)", ha="left", va="center", fontsize=fs.SMALL, linespacing=1.1)
    for i, bit in enumerate((1, 0, 1, 1, 0, 0, 1)):
        ax.add_patch(plt.Rectangle((0.57 + i * 0.055, 0.14), 0.045, 0.08, facecolor=fs.INK if bit else "white", edgecolor=fs.INK2, lw=0.6))


def fields(fig, rect, examples, records):
    x0, y0, w, h = rect
    positions = np.arange(50) * 0.04 + 0.02
    for k, (key, title) in enumerate((("overlapping", "Overlapping trio"), ("separate", "Separate trio"))):
        ax = fig.add_axes([x0 + 0.14 * w, y0 + h * (0.56 - 0.50 * k), 0.84 * w, h * 0.34])
        trio = examples[key]
        for rate, style in zip(trio["rates"], ("-", (0, (4, 1.6)), (0, (1, 1.2)))):
            ax.plot(positions, rate, color=fs.BLUE if key == "overlapping" else fs.ORANGE, ls=style, lw=1.3)
        low, high = min(trio["correlations"]), max(trio["correlations"])
        ax.set(xlim=(0, 2.0), ylim=(0, None), xticks=[0, 0.5, 1.0, 1.5, 2.0])
        ax.set_title(title, loc="left", fontsize=fs.TEXT, pad=2)
        ax.text(1.0, 1.02, f"map correlations {low:+.2f} to {high:+.2f}".replace("+0", "0").replace("-0", "-0"), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fs.SMALL, color=fs.INK2)
        ax.set_ylabel("Rate (Hz)", labelpad=1)
        if k == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Position on one running direction (m)", labelpad=1)
        for c, rate in zip(trio["cells"], trio["rates"]):
            records.append(("B", key, f"cell {c}", float(np.max(rate)), float(positions[int(np.argmax(rate))]), ""))


def raster(fig, rect, examples, records):
    x0, y0, w, h = rect
    ripple = examples["ripples"][0]
    steps = np.asarray(ripple["steps"])                                     # (cells in field order, 10)
    ax = fig.add_axes([x0 + 0.10 * w, y0 + 0.10 * h, 0.30 * w, 0.80 * h])
    cells, t = np.nonzero(steps)
    ax.plot(t * 10 + 5, cells, ls="none", marker="|", ms=3.5, mew=0.9, color=fs.INK)
    peak_map = np.asarray(examples["peak_map"])
    for m in range(1, int(peak_map.max()) + 1):
        edge = np.flatnonzero(peak_map >= m)[0] - 0.5
        ax.axhline(edge, color=fs.GRID, lw=0.6, zorder=0)
    ax.set(xlim=(0, 100), ylim=(steps.shape[0] - 0.5, -0.5), xticks=[0, 50, 100], yticks=[0, steps.shape[0] - 1])
    ax.set_yticklabels(["1", str(steps.shape[0])])
    ax.set_xlabel("Time (ms)", labelpad=1)
    ax.set_ylabel("Place cell, by field", labelpad=1)
    fired = steps.max(1)
    col = fig.add_axes([x0 + 0.42 * w, y0 + 0.10 * h, 0.03 * w, 0.80 * h])
    col.imshow(fired[:, None], aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["white", fs.INK]), interpolation="nearest")
    col.set(xticks=[], yticks=[])
    for side in col.spines.values():
        side.set_color(fs.INK2)
        side.set_linewidth(0.6)
    col.set_title("fired", fontsize=fs.SMALL, pad=2)
    records.append(("C", "ripple", int(ripple["index"]), int(fired.sum()), int(steps.sum()), ""))
    trio = examples["overlapping"]
    table = fig.add_axes([x0 + 0.56 * w, y0 + 0.18 * h, 0.42 * w, 0.70 * h])
    fs.blank(table)
    table.text(0.0, 1.0, "Patterns of the overlapping\ntrio of B, ripples per rest", ha="left", va="top", fontsize=fs.SMALL, linespacing=1.1)
    for j, pattern in enumerate(trio["patterns"]):
        y = 0.66 - j * 0.083
        for r, bit in enumerate(pattern):
            table.plot(0.05 + r * 0.08, y, marker="o", ms=3.4, mfc=fs.INK if bit == "1" else "white", mec=fs.INK if bit == "1" else fs.INK2, mew=0.7)
        for c, (value, colour) in enumerate(((trio["counts_before"][j], fs.INK), (trio["counts_after"][j], fs.INK))):
            table.text(0.50 + c * 0.32, y, f"{value:,}", ha="right", va="center", fontsize=fs.SMALL, color=fs.ORANGE if value == 0 else colour,
                       fontweight="bold" if value == 0 else "normal")
        records.append(("C", "overlapping trio", pattern, trio["counts_before"][j], trio["counts_after"][j], ""))
    table.text(0.45, 0.74, "before", ha="center", va="bottom", fontsize=fs.SMALL, color=fs.INK2)
    table.text(0.77, 0.74, "after", ha="center", va="bottom", fontsize=fs.SMALL, color=fs.INK2)
    table.text(0.0, -0.06, "zero counts in bold: one trio\ncannot resolve its own change", ha="left", va="top", fontsize=fs.SMALL,
               color=fs.INK2, linespacing=1.1)


def estimator(ax):
    fs.blank(ax)
    patterns = ("000", "100", "010", "001", "110", "101", "011", "111")
    sign = [(-1) ** (p.count("1") + 1) for p in patterns]                  # +1 for an odd number of active cells
    ax.text(0.0, 0.97, "Shift one trio's table by a multiple of", ha="left", va="top", fontsize=fs.SMALL)
    fs.math(ax, 0.66, 0.955, r"$s(x)$", size=fs.SMALL + 1, va="top")
    for j, (p, s) in enumerate(zip(patterns, sign)):
        x = 0.06 + j * 0.115
        for r, bit in enumerate(p):
            ax.plot(x, 0.74 - r * 0.065, marker="o", ms=3.4, mfc=fs.INK if bit == "1" else "white", mec=fs.INK if bit == "1" else fs.INK2, mew=0.7)
        ax.text(x, 0.50, "+" if s > 0 else "–", ha="center", va="center", fontsize=fs.TEXT + 1, color=fs.ORANGE if s > 0 else fs.BLUE,
                fontweight="bold")
    ax.text(0.0, 0.40, "every pair margin stays fixed; the shift", ha="left", va="top", fontsize=fs.SMALL)
    ax.text(0.0, 0.32, "moves only the three-way log odds", ha="left", va="top", fontsize=fs.SMALL)
    fs.math(ax, 0.0, 0.135, r"$\eta=\sum_x\,s(x)\,\log p(x)$", size=fs.TEXT, va="center")
    fs.math(ax, 0.47, 0.135, r"$\delta=\eta_{\mathrm{after}}-\eta_{\mathrm{before}}$", size=fs.TEXT, va="center")
    ax.text(0.0, 0.02, "One shared change per class, from the exact\nconditional likelihood of all its trios; the contrast\n"
            "is overlapping minus separate trios.", ha="left", va="top", fontsize=fs.SMALL, linespacing=1.15)


def estimates(folder):
    tables = {}
    for file in {r[2] for r in ROWS}:
        with (folder / (file + ".tsv")).open() as handle:
            tables[file] = {(r["group"], r["contrast"], r["quantity"]): r for r in csv.DictReader(handle, delimiter="\t")}
    out = {}
    for term in ("pairs", "trios"):
        for label, group, table, se_key, kind in ROWS:
            if group == "buddy" and term == "trios":
                continue
            row = tables[table][(group, "maze", term + "_difference")]
            out[(term, label)] = (float(row["delta"]), float(row[se_key]), kind, int(row["sessions"]))
    return out


def forest(ax, values, term, limits, records, labels):
    fs.zero(ax, "x")
    positions, y = [], 0.0
    for label, group, _, _, kind in ROWS:
        if kind == "second" and label.startswith("Rat 1"):
            y -= 0.6
        positions.append((label, y))
        y -= 1.0
    for label, y in positions:
        if (term, label) not in values:
            ax.text(limits[0] + 0.02 * (limits[1] - limits[0]), y, "too few trios", ha="left", va="center", fontsize=fs.SMALL, color=fs.MUTED,
                    fontstyle="italic")
            continue
        estimate, se, kind, sessions = values[(term, label)]
        colour, marker, filled = STYLE[kind]
        low, high = estimate - 1.96 * se, estimate + 1.96 * se
        ax.plot([max(low, limits[0]), min(high, limits[1])], [y, y], color=colour, lw=1.3, solid_capstyle="butt")
        if low < limits[0]:
            ax.plot(limits[0], y, marker="<", color=colour, ms=4, clip_on=False)
        if high > limits[1]:
            ax.plot(limits[1], y, marker=">", color=colour, ms=4, clip_on=False)
        ax.plot(estimate, y, marker=marker, ms=5.2 if marker != "D" else 4.8, mfc=colour if filled else "white", mec=colour, mew=1.0, zorder=3)
        ax.text(1.02, y, f"{estimate:+.2f}".replace("-", "-"), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=fs.SMALL, color=fs.INK2)
        records.append(("E" if term == "pairs" else "F", term, label, sessions, estimate, se))
    ax.set_yticks([y for _, y in positions])
    ax.set_yticklabels([label for label, _ in positions] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set(xlim=limits, ylim=(positions[-1][1] - 0.7, 0.7))
    fs.spines(ax, ("bottom",))
    ax.axhline(-4.8, color=fs.GRID, lw=0.8)


def draw(output, source):
    fs.print_style()
    examples = json.loads((HERE / "evidence" / "figures" / "hippocampus_examples.json").read_text())
    values = estimates(HERE / "evidence" / "hippocampus")
    fig = plt.figure(figsize=(fs.WIDTH, HEIGHT))
    records = []
    fs.heading(fig, 0.0, 1 - inch(0.17), "A", "Two cohorts, one design")
    design(fig.add_axes([0.0, inch(6.12), 0.50, inch(1.70)]))
    fig.text(0.0, inch(5.98), "First cohort: 4 rats, sleep before and after a novel maze\nSecond cohort: 5 rats, rest before and after two novel tracks",
             ha="left", va="top", fontsize=fs.SMALL, color=fs.INK2, linespacing=1.2)
    fs.heading(fig, 0.54, 1 - inch(0.17), "B", "Running defines the classes (rat 3)")
    fields(fig, [0.54, inch(5.70), 0.46, inch(2.15)], examples, records)
    fs.heading(fig, 0.0, inch(5.30), "C", "Each ripple gives one pattern per trio (rat 3)")
    raster(fig, [0.0, inch(3.30), 0.55, inch(1.90)], examples, records)
    fs.heading(fig, 0.59, inch(5.30), "D", "A shared change at fixed pair margins")
    estimator(fig.add_axes([0.59, inch(3.55), 0.40, inch(1.60)]))
    fs.heading(fig, 0.0, inch(2.98), "E", "Pair term")
    fs.heading(fig, 0.60, inch(2.98), "F", "Three-way term")
    forest(fig.add_axes([0.255, inch(0.66), 0.25, inch(2.16)]), values, "pairs", (-0.7, 1.05), records, labels=True)
    forest(fig.add_axes([0.64, inch(0.66), 0.27, inch(2.16)]), values, "trios", (-0.9, 1.05), records, labels=False)
    for x in (0.255 + 0.125, 0.64 + 0.135):
        fig.text(x, inch(0.06), "Change after minus before,\noverlapping minus separate", ha="center", va="bottom", fontsize=fs.SMALL, linespacing=1.1)
    fig.text(0.255 + 0.125, inch(2.86), "log odds ratio", ha="center", va="bottom", fontsize=fs.SMALL, color=fs.INK2)
    fig.text(0.64 + 0.135, inch(2.86), "three-way log odds", ha="center", va="bottom", fontsize=fs.SMALL, color=fs.INK2)
    fs.save_journal(fig, output)
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("panel", "quantity", "row", "count_or_sessions", "value_or_estimate", "value_2_or_standard_error"))
        writer.writerows(records)
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source_data", type=Path)
    a = parser.parse_args()
    print(draw(a.output, a.source_data), "rows")
