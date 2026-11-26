"""Fig 4: the hippocampal design, the classes from running, what a ripple contributes, the estimator at fixed pair
margins, and the change of the pair and three-way terms in every animal.

    draw_fig4_hippocampus.py OUTPUT_PDF SOURCE_CSV

A  the design of both cohorts (illustration; the rats, rest boxes and ripple traces are drawings).  The place fields
   drawn over the track are the running rate maps of panel B.
B  running rate maps, in 4-cm bins, of the 24 place cells of rat 3, second session, whose fields peak on the running
   direction of the overlapping trio (each scaled to its peak and ordered by the peak), and the maps in Hz of one
   overlapping and one separate trio (evidence/figures/hippocampus_examples.json, chosen by the rules recorded there).
C  one ripple of the rest after the tracks in the same session (the first ripple of the recorded rule): spikes of its 96
   place cells in ten 10-ms steps, cells ordered by running direction and field peak, and the pattern of the overlapping
   trio of B in this ripple.
D  the estimator, with the counts of the eight patterns of the overlapping trio of B over the ripples of each rest
   (source rows: pattern, ripples before, ripples after, s(x)).  The tables A - u s and B + u s keep every pair margin;
   the relative probabilities are the shift likelihood of Methods.
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 7.25, 8.45
ROWS = (("Achilles (2 sessions)", "achilles", "first_cohort_reserved", "se_common_factor", "first"),
        ("Cicero (3)", "cicero", "first_cohort_reserved", "se_common_factor", "first"),
        ("Buddy (1)", "buddy", "first_cohort_reserved", "se_common_factor", "first"),
        ("Achilles and Cicero", "achilles_cicero", "first_cohort_reserved", "se_common_factor", "pooled_first"),
        ("Gatsby (2; development)", "gatsby", "first_cohort_development", "se_calibrated", "development"),
        *[(f"Rat {i} ({1 if i == 1 else 2})", f"rat{i}", "second_cohort", "se_rule", "second") for i in range(1, 6)],
        ("Five rats (9)", "cohort", "second_cohort", "se_rule", "pooled_second"))
STYLE = {"first": (fs.BLUE, "o", True), "pooled_first": (fs.BLUE, "D", True), "development": (fs.BLUE, "o", False),
         "second": (fs.ORANGE, "^", True), "pooled_second": (fs.ORANGE, "D", True)}
FIELDS = LinearSegmentedColormap.from_list("fields", ["#ffffff", "#d6d6d6", "#8a8a8a", "#3a3a3a", "#111111"])
PATTERNS = ("000", "100", "010", "001", "110", "101", "011", "111")
TRIO_STYLES = ("-", (0, (3.2, 1.4)), (0, (1, 1.1)))


def inch(value):
    return value / HEIGHT


def rest_box(ax, x, y, w, h, before, ripple_count):
    """A rest box with a sleeping rat, and a schematic ripple trace beneath it."""
    fs.card(ax, x, y, w, h, face=fs.PANEL, radius=0.012)
    ax.text(x + w / 2, y + h - 0.018, "Rest before" if before else "Rest after", ha="center", va="top", fontsize=8, fontweight="bold")
    bx, by, bw, bh = x + 0.040, y + 0.150, w - 0.080, 0.150
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=0.010", facecolor="white", edgecolor=fs.INK2,
                                lw=0.7, zorder=1))
    fs.rodent(ax, bx + bw * 0.46, by + 0.045, 0.040, color=fs.MUTED, facing=1, zorder=3)
    for dx, dy, size in ((0.0, 0.0, 8.0), (0.022, 0.020, 9.0), (0.048, 0.042, 10.5)):
        ax.text(bx + bw * 0.62 + dx, by + 0.085 + dy, "z", fontsize=size, fontstyle="italic", fontweight="bold", color=fs.INK2,
                ha="left", va="baseline", zorder=4)
    t = np.linspace(0, 1, 900)
    centres = np.linspace(0.18, 0.82, ripple_count)
    trace = 0.10 * np.sin(2 * np.pi * 3.1 * t) + 0.05 * np.sin(2 * np.pi * 7.3 * t + 1.0)
    for c in centres:
        trace += np.exp(-((t - c) / 0.022) ** 2) * np.sin(2 * np.pi * 95 * t)
    ax.plot(x + 0.02 + (w - 0.04) * t, y + 0.080 + 0.022 * trace, color=fs.INK2, lw=0.55, zorder=2)
    for c in centres:
        ax.plot(x + 0.02 + (w - 0.04) * c, y + 0.118, marker="v", ms=2.8, color=fs.INK, mec="none", zorder=3)
    ax.text(x + w / 2, y + 0.026, "ripples", ha="center", va="center", fontsize=8, color=fs.INK2)


def track(ax, x, y, w, h, examples):
    """A linear track with a running rat and the running fields of panel B drawn along it."""
    fs.card(ax, x, y, w, h, face=fs.PANEL, radius=0.012)
    ax.text(x + w / 2, y + h - 0.018, "Novel maze or tracks", ha="center", va="top", fontsize=8, fontweight="bold")
    tx, ty, tw, th = x + 0.025, y + 0.085, w - 0.05, 0.030
    ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=0.012", facecolor="white", edgecolor=fs.INK2,
                                lw=0.7, zorder=1))
    fs.rodent(ax, tx + tw * 0.40, ty + th + 0.018, 0.034, color=fs.MUTED, facing=1, zorder=4)
    d = examples["direction_maps"]
    rates = np.asarray(d["rates"])
    scaled = rates / rates.max(1, keepdims=True)
    pos = np.linspace(0, 1, rates.shape[1])
    base = ty + th + 0.062
    for i, row in enumerate(scaled):
        shade = ("#3a3a3a", "#8a8a8a", "#bdbdbd")[i % 3]
        ax.fill_between(tx + tw * pos, base, base + 0.075 * row, color=shade, alpha=0.30, lw=0, zorder=2)
        ax.plot(tx + tw * pos, base + 0.075 * row, color=shade, lw=0.5, zorder=2)
    ax.text(tx, base + 0.095, "place fields of B", ha="left", va="bottom", fontsize=8, color=fs.INK2)
    ax.annotate("", xy=(tx + tw - 0.01, ty - 0.022), xytext=(tx + 0.01, ty - 0.022),
                arrowprops=dict(arrowstyle="<|-|>", color=fs.INK2, lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(x + w / 2, ty - 0.036, "running in both directions", ha="center", va="top", fontsize=8, color=fs.INK2)


def design(fig, rect, examples):
    ax = fs.canvas(fig, rect, xlim=(0, 1))
    top = ax.get_ylim()[1]
    y, h = top - 0.385, 0.380
    rest_box(ax, 0.0, y, 0.255, h, True, 3)
    track(ax, 0.315, y, 0.370, h, examples)
    rest_box(ax, 0.745, y, 0.255, h, False, 4)
    for a, b in ((0.258, 0.312), (0.688, 0.742)):
        ax.annotate("", xy=(b, y + h / 2), xytext=(a, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0))
    rows = (("o", fs.BLUE, "First cohort, 4 rats: sleep before and after a novel maze"),
            ("^", fs.ORANGE, "Second cohort, 5 rats: rest before and after two novel tracks"))
    for k, (marker, colour, text) in enumerate(rows):
        yy = y - 0.040 - 0.044 * k
        ax.plot(0.012, yy, marker=marker, ms=4.6, color=colour, mec=colour, clip_on=False)
        ax.text(0.032, yy, text, ha="left", va="center", fontsize=8)


def place_map(fig, rect, examples, records):
    d = examples["direction_maps"]
    rates = np.asarray(d["rates"])
    scaled = rates / rates.max(1, keepdims=True)
    n, bins = scaled.shape
    ax = fig.add_axes(rect)
    ax.imshow(scaled, aspect="auto", cmap=FIELDS, vmin=0, vmax=1, extent=(0, bins * 0.04, 0.5, n + 0.5), interpolation="nearest",
              origin="lower")
    trio = [d["cells"].index(c) for c in examples["overlapping"]["cells"]]
    lo, hi = min(trio) + 0.5, max(trio) + 1.5
    ax.add_patch(Rectangle((0, lo), bins * 0.04, hi - lo, facecolor="none", edgecolor=fs.ORANGE, lw=1.1, zorder=4, clip_on=False))
    ax.set(xlim=(0, bins * 0.04), ylim=(0.5, n + 0.5), xticks=[0, 1, 2], yticks=[1, n])
    ax.text(bins * 0.04 + 0.06, (lo + hi) / 2, "trio", ha="left", va="center", fontsize=8, fontweight="bold", color=fs.ORANGE_DARK)
    ax.set_xlabel("Position (m)", labelpad=1)
    ax.set_ylabel("Place cell, by peak", labelpad=1)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(fs.INK2)
    key = fig.add_axes([rect[0], rect[1] + rect[3] + inch(0.07), rect[2] * 0.45, inch(0.06)])
    key.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=FIELDS, extent=(0, 1, 0, 1))
    key.set(xticks=[0, 1], yticks=[])
    key.set_xticklabels(["0", "peak"])
    key.tick_params(axis="x", length=0, pad=1, labelsize=8)
    for side in key.spines.values():
        side.set_visible(False)
    key.xaxis.set_ticks_position("top")
    for i, (cell, row) in enumerate(zip(d["cells"], rates)):
        records.append(("B", "rate map, direction of the trio", f"cell {cell}", i + 1, float(row.max()), float(np.argmax(row) * 0.04 + 0.02)))
    return trio, (lo, hi)


def trio_fields(fig, rect, trio, title, records, panel_key, xlabel):
    ax = fig.add_axes(rect)
    positions = np.arange(50) * 0.04 + 0.02
    for rate, style in zip(trio["rates"], TRIO_STYLES):
        ax.plot(positions, rate, color=fs.ORANGE_DARK if panel_key.startswith("overlapping") else fs.INK, ls=style, lw=1.1)
    ax.set(xlim=(0, 2.0), ylim=(0, None), xticks=[0, 1, 2])
    low, high = min(trio["correlations"]), max(trio["correlations"])
    ax.text(0.0, 1.30, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    ax.text(0.0, 1.05, fs.minus(f"map r {low:.2f} to {high:.2f}"), transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color=fs.INK2)
    ax.set_ylabel("Hz", labelpad=1)
    if xlabel:
        ax.set_xlabel("Position (m)", labelpad=1)
    else:
        ax.set_xticklabels([])
    for c, rate in zip(trio["cells"], trio["rates"]):
        records.append(("B", panel_key, f"cell {c}", int(np.argmax(rate)), float(np.max(rate)), float(positions[int(np.argmax(rate))])))
    return ax


def ripple_raster(fig, rect, examples, records):
    ripple = examples["ripples"][0]
    steps = np.asarray(ripple["steps"]) > 0                                # (cells in field order, 10)
    n = steps.shape[0]
    order = list(examples["order"])
    rows = [order.index(c) for c in examples["overlapping"]["cells"]]
    x0, y0, w, h = rect
    ax = fig.add_axes([x0 + 0.115 * w, y0, 0.50 * w, h])
    for r in rows:
        ax.add_patch(Rectangle((-50, r - 0.5), 100, 1.0, facecolor=fs.ORANGE_TINT, edgecolor="none", zorder=0))
    cells, t = np.nonzero(steps)
    for c, s in zip(cells, t):
        ax.add_patch(Rectangle((s * 10 - 49.0, c - 0.46), 8.0, 0.92, facecolor=fs.ORANGE_DARK if c in rows else fs.INK, edgecolor="none", zorder=2))
    peak_map = np.asarray(examples["peak_map"])
    starts = [int(np.flatnonzero(peak_map == m)[0]) for m in range(int(peak_map.max()) + 1)]
    ends = starts[1:] + [n]
    for m, (a, b) in enumerate(zip(starts, ends)):
        if m:
            ax.axhline(a - 0.5, color=fs.RULE, lw=0.6, zorder=1)
        ax.text(-54, (a + b - 1) / 2, f"{m + 1}", ha="right", va="center", fontsize=8, color=fs.INK2)
    ax.text(-67, n / 2, "Running direction", ha="center", va="center", rotation=90, fontsize=8)
    ax.set(xlim=(-50, 50), ylim=(n - 0.5, -0.5), xticks=[-50, 0, 50], yticks=[])
    fs.ticks_minus(ax, "x")
    ax.set_xlabel("Time from ripple peak (ms)", labelpad=1)
    ax.spines["left"].set_visible(False)
    fired = steps.max(1)
    col = fig.add_axes([x0 + 0.635 * w, y0, 0.045 * w, h])
    col.imshow(fired[:, None], aspect="auto", cmap=ListedColormap(["white", fs.INK]), interpolation="nearest", extent=(0, 1, n - 0.5, -0.5))
    for r in rows:
        col.add_patch(Rectangle((0, r - 0.5), 1, 1, facecolor=fs.ORANGE_DARK if fired[r] else fs.ORANGE_TINT, edgecolor="none", zorder=2))
    col.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(n - 0.5, -0.5))
    for side in col.spines.values():
        side.set_visible(True)
        side.set_color(fs.INK2)
        side.set_linewidth(0.6)
    col.text(0.5, 1.012, "fired", transform=col.transAxes, ha="center", va="bottom", fontsize=8)
    # the trio's pattern in this ripple
    word = "".join(str(int(fired[r])) for r in rows)
    pat = fig.add_axes([x0 + 0.72 * w, y0, 0.28 * w, h])
    fs.blank(pat)
    yc = 1 - (np.mean(rows) + 0.5) / n
    pat.annotate("", xy=(0.10, yc), xytext=(-0.13, yc), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=fs.INK2, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))
    pat.text(0.16, yc + 0.17, "Trio of B", ha="left", va="bottom", fontsize=8, fontweight="bold")
    for k, r in enumerate(rows):
        on = bool(fired[r])
        pat.plot(0.24 + 0.22 * k, yc + 0.075, marker="o", ms=7.0, mfc=fs.ORANGE_DARK if on else "white", mec=fs.ORANGE_DARK, mew=0.9)
        pat.text(0.24 + 0.22 * k, yc - 0.005, str(int(on)), ha="center", va="top", fontsize=8)
    pat.text(0.16, yc - 0.10, f"pattern {word}, one", ha="left", va="top", fontsize=8, color=fs.INK2)
    pat.text(0.16, yc - 0.17, "corner of the cubes", ha="left", va="top", fontsize=8, color=fs.INK2)
    records.append(("C", "ripple: place cells fired, spike steps", f"ripple {int(ripple['index'])}", int(fired.sum()), int(steps.sum()), ""))
    records.append(("C", "pattern of the overlapping trio in this ripple", word, "", "", ""))
    return word


def admissible(before, after):
    sign = np.array([(-1) ** (p.count("1") + 1) for p in PATTERNS])
    a, b = np.array(before), np.array(after)
    top = int(max(a.max(), b.max()))
    return [u for u in range(-top, top + 1) if np.all(a - u * sign >= 0) and np.all(b + u * sign >= 0)]


def estimator(fig, rect, trio, word, records):
    ax = fs.canvas(fig, rect, xlim=(0, 1))
    top = ax.get_ylim()[1]
    s = 0.150
    signs = {p: ("+" if p.count("1") % 2 else "–") for p in PATTERNS}
    for k, (key, title) in enumerate((("counts_before", "Rest before"), ("counts_after", "Rest after"))):
        x0 = 0.115 + 0.50 * k
        y0 = top - 0.330
        at = fs.parity_cube(ax, x0, y0, s, values=signs, size=8, depth=(0.50, 0.36))
        ax.text(x0 + 0.5 * s + 0.25 * s - 0.04, top - 0.030, title, ha="right", va="center", fontsize=8, fontweight="bold")
        fs.smath(ax, x0 + 0.5 * s + 0.25 * s - 0.025, top - 0.030, "$A(x)$" if k == 0 else "$B(x)$", size=8.5, va="center",
                 transform=ax.transData)
        counts = dict(zip(PATTERNS, trio[key]))
        for p in PATTERNS:
            a, b, c = (int(ch) for ch in p)
            P = at(a, b, c)
            dx = (-1 if a == 0 else 1) * 0.034
            dy = 0.024 if (b == 1 and c == 1) else (-0.024 if (b == 0 and c == 0) else 0.0)
            value = counts[p]
            ax.text(P[0] + dx, P[1] + dy, f"{value:,}", ha="right" if a == 0 else "left", va="center", fontsize=8,
                    fontweight="bold" if value == 0 else "normal", color=fs.INK, zorder=5)
            if p == word and key == "counts_after":
                ax.add_patch(plt.Circle(P, 0.026, facecolor="none", edgecolor=fs.INK, lw=0.9, zorder=5))
    for p, before, after in zip(PATTERNS, trio["counts_before"], trio["counts_after"]):
        # one row per pattern: ripples of the rest before, of the rest after, and s(x) (+1 odd, -1 even)
        records.append(("D", "overlapping trio", p, before, after, 1 if p.count("1") % 2 else -1))
    us = admissible(trio["counts_before"], trio["counts_after"])
    yf = top - 0.425
    fs.smath(ax, 0.0, yf, r"$P_\delta(u)\ \propto\ e^{\,\delta u}\ /\ \Pi_x\,[A(x)-u\,s(x)]!\;[B(x)+u\,s(x)]!$", size=8.5,
             va="center", transform=ax.transData)
    lines = ("Moving u counts along s keeps every pair margin: each edge",
             "joins a + and a – pattern. Here only u = 0 keeps every count",
             "nonnegative, so alone this trio carries no information about",
             "the change. One change per class is fitted to all its trios.")
    if len(us) != 1 or us[0] != 0:
        raise ValueError(f"the note assumes one admissible table, found u in {us}")
    for k, line in enumerate(lines):
        ax.text(0.0, yf - 0.060 - 0.036 * k, line, ha="left", va="top", fontsize=8, color=fs.INK2)
    records.append(("D", "admissible shifts u", "overlapping trio of B", len(us), us[0] if us else "", us[-1] if us else ""))


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


def positions():
    out, y = [], 0.0
    for label, group, _, _, kind in ROWS:
        if label.startswith("Rat 1"):
            y -= 1.15
        out.append((label, kind, y))
        y -= 1.0
    return out


def forest(ax, values, term, limits, records):
    rows = positions()
    for label, kind, y in rows:
        if kind.startswith("pooled"):
            ax.axhspan(y - 0.5, y + 0.5, color=fs.PANEL, zorder=0, lw=0)
    ax.axvline(0, color=fs.INK2, lw=0.6, zorder=1)
    for label, kind, y in rows:
        if (term, label) not in values:
            ax.text(limits[0] + 0.03 * (limits[1] - limits[0]), y, "too few overlapping trios", ha="left", va="center", fontsize=8,
                    color=fs.MUTED, fontstyle="italic", zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
            continue
        estimate, se, kind, sessions = values[(term, label)]
        colour, marker, filled = STYLE[kind]
        low, high = estimate - 1.96 * se, estimate + 1.96 * se
        ax.plot([max(low, limits[0]), min(high, limits[1])], [y, y], color=colour, lw=1.4 if kind.startswith("pooled") else 1.1,
                solid_capstyle="butt", zorder=2)
        if low < limits[0]:
            ax.plot(limits[0], y, marker="<", color=colour, ms=3.6, clip_on=False, zorder=3)
        if high > limits[1]:
            ax.plot(limits[1], y, marker=">", color=colour, ms=3.6, clip_on=False, zorder=3)
        ax.plot(estimate, y, marker=marker, ms=5.6 if marker != "D" else 5.0, mfc=colour if filled else "white", mec=colour, mew=1.0, zorder=3)
        ax.text(1.07, y, fs.minus(f"{estimate:.2f}"), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8,
                fontweight="bold" if kind.startswith("pooled") else "normal")
        records.append(("E" if term == "pairs" else "F", term, label, sessions, estimate, se))
    ax.set_yticks([y for _, _, y in rows])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set(xlim=limits, ylim=(rows[-1][2] - 0.75, 0.75))
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    fs.ticks_minus(ax, "x")
    return rows


def draw(output, source):
    fs.pro()
    examples = json.loads((HERE / "evidence" / "figures" / "hippocampus_examples.json").read_text())
    values = estimates(HERE / "evidence" / "hippocampus")
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    records = []
    # row 1: design and fields
    fs.letter_title(fig, 0.0, inch(8.30), "A", "One design in two cohorts")
    design(fig, [0.0, inch(6.32), 0.525, inch(1.86)], examples)
    fs.letter_title(fig, 0.565, inch(8.30), "B", "Running fields define the classes (rat 3)")
    place_map(fig, [0.625, inch(6.40), 0.150, inch(1.56)], examples, records)
    trio_fields(fig, [0.843, inch(7.32), 0.130, inch(0.52)], examples["overlapping"], "Overlapping trio", records, "overlapping trio", False)
    trio_fields(fig, [0.843, inch(6.40), 0.130, inch(0.52)], examples["separate"], "Separate trio", records, "separate trio", True)
    # row 2: one ripple and the estimator
    fs.letter_title(fig, 0.0, inch(5.86), "C", "Each ripple gives one pattern per trio")
    word = ripple_raster(fig, [0.0, inch(3.52), 0.50, inch(2.08)], examples, records)
    fs.letter_title(fig, 0.545, inch(5.86), "D", "A shared change at fixed pair margins")
    estimator(fig, [0.555, inch(3.20), 0.445, inch(2.52)], examples["overlapping"], word, records)
    # row 3: forests
    fs.letter_title(fig, 0.0, inch(2.86), "E", "Pair term")
    fs.letter_title(fig, 0.585, inch(2.86), "F", "Three-way term")
    rect_e = [0.255, inch(0.50), 0.255, inch(2.18)]
    rect_f = [0.665, inch(0.50), 0.255, inch(2.18)]
    rows = forest(fig.add_axes(rect_e), values, "pairs", (-0.7, 1.05), records)
    forest(fig.add_axes(rect_f), values, "trios", (-0.9, 1.05), records)
    span = rows[0][2] + 0.75 - (rows[-1][2] - 0.75)
    for label, kind, y in rows:
        yy = inch(0.50) + inch(2.18) * (y - (rows[-1][2] - 0.75)) / span
        fig.text(0.245, yy, label, ha="right", va="center", fontsize=8, fontweight="bold" if kind.startswith("pooled") else "normal")
    for name, first in (("First cohort", True), ("Second cohort", False)):
        y = rows[0][2] if first else next(y for label, _, y in rows if label.startswith("Rat 1"))
        yy = inch(0.50) + inch(2.18) * (y + 0.78 - (rows[-1][2] - 0.75)) / span
        fig.text(0.0, yy, name, ha="left", va="center", fontsize=8, fontweight="bold", color=fs.BLUE if first else fs.ORANGE_DARK)
    for rect, unit in ((rect_e, "log odds ratio"), (rect_f, "three-way log odds")):
        fig.text(rect[0] + rect[2] / 2, inch(0.06), f"After minus before, overlapping minus separate\n({unit})", ha="center",
                 va="bottom", fontsize=8, linespacing=1.15)
    fs.save_pro(fig, output)
    with Path(source).open("w", newline="") as handle:
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
