"""Place fields on the maze and the classes of pairs and trios they define (DANDI 000044, one session file).

The classes of the hippocampal study come from the maze epoch alone: which pyramidal cells have place fields that
overlap. No spike of the sleep epochs is read here, so the classes cannot be shaped by what is tested in sleep.

THE RULE (written and committed on 23 November 2026, before any position value was read):

1. Running. Position is the lab's linearized position, which the lab defined only while the rat ran along the
   track. Velocity is the difference of positions 10 samples ahead and 10 behind (0.51 s; around the circle on the
   circular tracks). A sample is running when both are defined and the speed exceeds 5 cm/s. On linear tracks the
   two running directions are kept apart (place fields in CA1 depend on direction there); on circular tracks only
   the direction with more running samples is used.
2. Maps. 4-cm bins. For every pyramidal cell and direction: spikes at running samples of that direction (a spike
   belongs to the nearest position sample) over the time spent, both smoothed with a Gaussian of 1.5 bins (around
   the circle on circular tracks) before the division; bins with less than 0.1 s are not used.
3. A cell has a place field in a direction when it fired at least 50 spikes there, its peak rate is at least 1
   spike/s and its spatial information is at least 0.25 bits per spike (Skaggs); a place cell has one in at least
   one direction.
4. Overlap of two cells: the correlation of their maps over the used bins of all directions. A pair of place cells
   of one hemisphere is `overlapping` when it is at least 0.5 and `separate` when it is at most 0.
5. A trio of place cells of one hemisphere is `overlapping` when its three pairs are, and `separate` when its
   three pairs are.

    python -I hippocampal_place_fields.py FILE.nwb OUTPUT_FOLDER      (OUTPUT_FOLDER must not exist)
    python -I hippocampal_place_fields.py --selftest
    python -I hippocampal_place_fields.py --summary FIELDS_FOLDER UNITS.tsv OUT.tsv
`--summary` lists the classes of every hemisphere whose ripple channel passed its check, with what a shared
change of each class's term could resolve. That figure uses single-cell participations only (the count of section
56: `1 / (1/(n_A prod r_A(1 - r_A)) + 1/(n_B prod r_B(1 - r_B)))` per pair or trio, summed), a design effect of
1.5 and 80% power; no pair or trio statistic of the sleep epochs enters it.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

BIN, AHEAD, SPEED, SMOOTH = 0.04, 10, 0.05, 1.5
MIN_SPIKES, MIN_PEAK, MIN_INFORMATION, MIN_TIME = 50, 1.0, 0.25, 0.1
OVERLAP, SEPARATE = 0.5, 0.0


def strings(dataset):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]])


def velocity(position, rate, length=None):
    """Metres per second from positions AHEAD samples ahead and behind; around the circle when length is given."""
    out = np.full(len(position), np.nan)
    step = position[2 * AHEAD:] - position[:-2 * AHEAD]
    if length is not None:
        step = (step + length / 2) % length - length / 2
    out[AHEAD:-AHEAD] = step * rate / (2 * AHEAD)
    return out


def smooth(values, circular):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(values, dtype=float), SMOOTH, mode="wrap" if circular else "nearest")


def rate_map(spike_bins, sample_bins, bins, rate, circular):
    """Smoothed spikes over smoothed time; bins with too little time are NaN. Also the information per spike."""
    time = smooth(np.bincount(sample_bins, minlength=bins) / rate, circular)
    spikes = smooth(np.bincount(spike_bins, minlength=bins), circular)
    used = time >= MIN_TIME
    out = np.full(bins, np.nan)
    out[used] = spikes[used] / time[used]
    information = 0.0
    if used.any() and np.nansum(out[used] * time[used]) > 0:
        p = time[used] / time[used].sum()
        mean = (p * out[used]).sum()
        ratio = out[used] / mean
        positive = ratio > 0
        information = float((p[positive] * ratio[positive] * np.log2(ratio[positive])).sum())
    return out, information


def classes(correlation, place, side):
    """Pairs and trios of each class within one hemisphere, as arrays of indices into the cells."""
    cells = np.flatnonzero(place & side)
    pairs = {"overlapping": [], "separate": []}
    for i, j in itertools.combinations(cells, 2):
        if np.isfinite(correlation[i, j]):
            if correlation[i, j] >= OVERLAP:
                pairs["overlapping"].append((i, j))
            elif correlation[i, j] <= SEPARATE:
                pairs["separate"].append((i, j))
    trios = {}
    for name, members in pairs.items():
        linked = set(members)
        trios[name] = [(i, j, k) for i, j, k in itertools.combinations(cells, 3)
                       if (i, j) in linked and (i, k) in linked and (j, k) in linked]
    return ({k: np.array(v, dtype=np.int64).reshape(-1, 2) for k, v in pairs.items()},
            {k: np.array(v, dtype=np.int64).reshape(-1, 3) for k, v in trios.items()})


def fields(path, output):
    import h5py
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, "r") as nwb:
        series = nwb["processing/behavior/position/linearized_position"]
        position = series["data"][:].astype(float).ravel()
        rate, start = float(series["starting_time"].attrs["rate"]), float(series["starting_time"][()])
        description = series.attrs["description"]
        description = description.decode() if isinstance(description, bytes) else str(description)
        epochs = nwb["intervals/epochs"]
        label = strings(epochs["label"])
        maze = (float(epochs["start_time"][:][label == "MAZE"][0]), float(epochs["stop_time"][:][label == "MAZE"][0]))
        units = nwb["units"]
        unit_id, unit_type, unit_side = units["id"][:], strings(units["cell_type"]), strings(units["hemisphere"])
        ends = units["spike_times_index"][:].astype(np.int64)
        starts = np.concatenate(([0], ends[:-1]))
        times = units["spike_times"][:]
        pyramidal = np.flatnonzero(unit_type == "pyramidal cell")
        spikes = []
        for k in pyramidal:
            t = times[starts[k]:ends[k]]
            spikes.append(np.sort(t[(t >= maze[0]) & (t < maze[1])]))  # the maze epoch only: nothing else is kept
        del times
    circular = "circular" in description
    finite = np.isfinite(position)
    length = float(np.ceil(np.nanmax(position) / BIN) * BIN)
    bins = int(round(length / BIN))
    v = velocity(position, rate, length if circular else None)
    sample_bin = np.clip((np.where(finite, position, 0.0) / BIN).astype(int), 0, bins - 1)
    running = {"up": finite & (v > SPEED), "down": finite & (v < -SPEED)}
    if circular:
        keep = max(running, key=lambda d: running[d].sum())
        running = {keep: running[keep]}
    maps, has_field, information, peak, count = {}, {}, {}, {}, {}
    for direction, mask in running.items():
        maps[direction] = np.full((len(pyramidal), bins), np.nan)
        has_field[direction] = np.zeros(len(pyramidal), dtype=bool)
        information[direction], peak[direction], count[direction] = (np.zeros(len(pyramidal)) for _ in range(3))
        for c, t in enumerate(spikes):
            sample = np.rint((t - start) * rate).astype(int)
            sample = sample[(sample >= 0) & (sample < len(position))]
            sample = sample[mask[sample]]
            m, info = rate_map(sample_bin[sample], sample_bin[mask], bins, rate, circular)
            maps[direction][c], information[direction][c], count[direction][c] = m, info, len(sample)
            peak[direction][c] = np.nanmax(m) if np.isfinite(m).any() else 0.0
            has_field[direction][c] = len(sample) >= MIN_SPIKES and peak[direction][c] >= MIN_PEAK and info >= MIN_INFORMATION
    place = np.any([has_field[d] for d in running], axis=0)
    stacked = np.concatenate([maps[d] for d in running], axis=1)
    used = np.isfinite(stacked).all(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(stacked[:, used]) if used.sum() > 2 else np.full((len(pyramidal),) * 2, np.nan)
    report = dict(file=path.name, track="circular" if circular else "linear", length_m=length, bins=bins,
                  directions={d: dict(seconds=round(float(m.sum() / rate), 1)) for d, m in running.items()},
                  used_bins=int(used.sum()), pyramidal=len(pyramidal), place_cells=int(place.sum()),
                  rule=dict(bin=BIN, ahead=AHEAD, speed=SPEED, smooth=SMOOTH, min_spikes=MIN_SPIKES, min_peak=MIN_PEAK,
                            min_information=MIN_INFORMATION, overlap=OVERLAP, separate=SEPARATE),
                  scope="Maze epoch only: no spike of the sleep epochs was read.", hemispheres={})
    saved = dict(unit=unit_id[pyramidal], correlation=correlation, place=place, hemisphere=unit_side[pyramidal].astype("U5"))
    for direction in running:
        saved[f"map_{direction}"] = maps[direction]
    side_of = unit_side[pyramidal]
    for side in ("left", "right"):
        mask = side_of == side
        pairs, trios = classes(correlation, place, mask)
        report["hemispheres"][side] = dict(
            pyramidal=int(mask.sum()), place_cells=int((place & mask).sum()),
            pairs={k: len(v) for k, v in pairs.items()}, trios={k: len(v) for k, v in trios.items()},
            cells_in_overlapping_trios=int(len(np.unique(trios["overlapping"]))),
            possible_pairs=int((place & mask).sum() * ((place & mask).sum() - 1) // 2))
        for name in pairs:
            saved[f"{side}_pairs_{name}"] = unit_id[pyramidal][pairs[name]] if len(pairs[name]) else np.zeros((0, 2), dtype=np.int64)
            saved[f"{side}_trios_{name}"] = unit_id[pyramidal][trios[name]] if len(trios[name]) else np.zeros((0, 3), dtype=np.int64)
    lines = ["unit\themisphere\tplace_cell\t" + "\t".join(f"{q}_{d}" for d in running for q in ("field", "spikes", "peak_rate", "information", "peak_position"))]
    for c in range(len(pyramidal)):
        row = [str(int(unit_id[pyramidal][c])), side_of[c], str(int(place[c]))]
        for d in running:
            where = float(np.nanargmax(maps[d][c]) * BIN + BIN / 2) if np.isfinite(maps[d][c]).any() else float("nan")
            row += [str(int(has_field[d][c])), str(int(count[d][c])), f"{peak[d][c]:.3f}", f"{information[d][c]:.3f}", f"{where:.2f}"]
        lines.append("\t".join(row))
    (output / "cells.tsv").write_text("\n".join(lines) + "\n")
    np.savez_compressed(output / "fields.npz", **saved)
    report["complete"] = True
    (output / "fields.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def selftest():
    rng = np.random.default_rng(0)
    rate = 39.0
    t = np.arange(0, 1200, 1 / rate)
    position = 0.8 + 0.8 * np.sin(2 * np.pi * t / 20.0)          # back and forth on 1.6 m every 20 s
    v = velocity(position, rate)
    assert np.nanmax(np.abs(v[AHEAD:-AHEAD] - 0.8 * 2 * np.pi / 20 * np.cos(2 * np.pi * t[AHEAD:-AHEAD] / 20))) < 0.01
    wrapped = velocity(np.array([2.8, 2.85, 0.0, 0.05] * 10)[:2 * AHEAD + 1], 1.0, 2.9)
    assert np.isfinite(wrapped[AHEAD]) and abs(wrapped[AHEAD]) < 0.2
    bins = 40
    sample_bin = np.clip((position / BIN).astype(int), 0, bins - 1)
    up = v > SPEED
    centre, width, peak_rate = 1.0, 0.08, 12.0
    intensity = peak_rate * np.exp(-0.5 * ((position - centre) / width) ** 2) / rate
    spikes = np.flatnonzero((rng.random(len(t)) < intensity) & up)
    m, information = rate_map(sample_bin[spikes], sample_bin[up], bins, rate, False)
    assert abs(np.nanargmax(m) * BIN + BIN / 2 - centre) < 0.06 and 6 < np.nanmax(m) < 16 and information > 1.0
    flat = np.flatnonzero((rng.random(len(t)) < 2.0 / rate) & up)
    m2, information2 = rate_map(sample_bin[flat], sample_bin[up], bins, rate, False)
    assert information2 < 0.1
    correlation = np.array([[1, 0.8, 0.7, -0.2], [0.8, 1, 0.6, -0.1], [0.7, 0.6, 1, -0.3], [-0.2, -0.1, -0.3, 1.0]])
    pairs, trios = classes(correlation, np.array([True, True, True, True]), np.array([True, True, True, True]))
    assert len(pairs["overlapping"]) == 3 and len(pairs["separate"]) == 3
    assert trios["overlapping"].tolist() == [[0, 1, 2]] and len(trios["separate"]) == 0
    print("self-test passed")


def summary(folder, units, out, design_effect=1.5, power=1.959964 + 0.841621):
    import csv
    import math
    rows = list(csv.DictReader(open(units), delimiter="\t"))
    part = {(r["session"], int(r["unit"])): r for r in rows}
    checked = {(r["session"], r["hemisphere"]) for r in rows if r["check"] == "passed"}
    lines = ["session\themisphere\ttrack\tpyramidal\tplace_cells\tclass\tsets\tmedian_participation\tinformation\tse\tsmallest_fold_change"]
    totals = {}
    for path in sorted(Path(folder).glob("*/fields.npz")):
        session = path.parent.name
        data, report = np.load(path), json.loads((path.parent / "fields.json").read_text())
        for side in ("left", "right"):
            if (session, side) not in checked:
                continue
            for kind in ("pairs", "trios"):
                for name in ("overlapping", "separate"):
                    sets = data[f"{side}_{kind}_{name}"]
                    information, median = 0.0, float("nan")
                    if len(sets):
                        get = lambda key: np.array([[float(part[(session, int(u))][key]) for u in s] for s in sets])
                        r_a, r_b = get("participation_PRE_fixed100ms"), get("participation_POST_fixed100ms")
                        first = part[(session, int(sets[0][0]))]
                        a = np.prod(r_a * (1 - r_a), axis=1) * int(first["nrem_ripples_PRE_fixed100ms"])
                        b = np.prod(r_b * (1 - r_b), axis=1) * int(first["nrem_ripples_POST_fixed100ms"])
                        with np.errstate(divide="ignore"):
                            information = float(np.where((a > 0) & (b > 0), 1 / (1 / a + 1 / b), 0.0).sum())
                        median = float(np.median(np.r_[r_a.ravel(), r_b.ravel()]))
                    for group in (session.split("-")[0], "all animals"):
                        total = totals.setdefault((group, kind, name), [0, 0.0])
                        total[0] += len(sets)
                        total[1] += information
                    se = math.sqrt(design_effect / information) if information > 0 else float("nan")
                    lines.append("\t".join(map(str, (
                        session, side, report["track"], report["hemispheres"][side]["pyramidal"],
                        report["hemispheres"][side]["place_cells"], f"{kind} {name}", len(sets), f"{median:.3f}",
                        f"{information:.1f}", f"{se:.4f}", f"{math.exp(power * se):.3f}" if se == se and se < 5 else "nan"))))
    for (group, kind, name), (count, information) in sorted(totals.items()):
        se = math.sqrt(design_effect / information) if information > 0 else float("nan")
        lines.append("\t".join(map(str, (group, "all", "", "", "", f"{kind} {name}", count, "", f"{information:.1f}",
                                          f"{se:.4f}", f"{math.exp(power * se):.3f}" if se == se and se < 5 else "nan"))))
    Path(out).write_text("\n".join(lines) + "\n")
    return len(lines) - 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif sys.argv[1] == "--summary":
        print(summary(sys.argv[2], sys.argv[3], sys.argv[4]), "rows")
    else:
        result = fields(Path(sys.argv[1]), Path(sys.argv[2]))
        print(json.dumps(dict(file=result["file"], track=result["track"], place_cells=result["place_cells"],
                              hemispheres={s: dict(place=h["place_cells"], pairs=h["pairs"], trios=h["trios"])
                                           for s, h in result["hemispheres"].items()})))
