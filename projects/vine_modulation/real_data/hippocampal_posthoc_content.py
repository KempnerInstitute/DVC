"""Post hoc, first cohort (E62): is the change of the three-neuron term after the maze a change of what is replayed?

Written on 24 November 2026, after E66 had been read on the second cohort (a law of
replayed place and gain, with place cells independent given both, reproduced the second cohort's change of the
pair term and of the three-way term) and before anything of this file was run on the first cohort.  The first
cohort is where the change was claimed (Achilles) and where it is aligned in time with the maze (section 62).

The law, its fit and the three estimates (`plain`, `loglinear`, `adjusted`) are those of
`dryad_posthoc_content.py`, applied hemisphere by hemisphere to the events of E62: the 100-ms windows around the
non-REM ripples of the sleep before (PRE) and after (POST) the maze, all pyramidal cells of the hemisphere, E62's
classes.  The states of a hemisphere are pieces of the rate maps of its place cells on the maze (`fields.npz` of
`hippocampal_place_fields.py`: 4-cm bins, smoothed, one map per running direction; on a circular track only the
direction run more), a piece being a bin, the bins within 20 cm, within 48 cm or the whole map; on a circular
track the pieces wrap around.  Bins that some place cell's map does not cover are left out.  `plain` pools the
hemispheres as E62 did: coinciding ripples of the two hemispheres of a session are one unit, and the standard
error of the reserved animals' pair difference carries E62's factor of 1.79 (none for trios).  The other two
treat the ripples of a hemisphere as independent of those of the other hemisphere.

Groups: achilles (two sessions, four hemispheres), cicero (three sessions), achilles_cicero (the group of E62's
claim), gatsby (the development animal).  Buddy has no trios to speak of and is left out.

    hippocampal_posthoc_content.py fit DATA_FOLDER CENSUS_ROOT FIELDS_ROOT OUT_FOLDER SESSION
    hippocampal_posthoc_content.py study OUT_FOLDER [WORKERS]
    hippocampal_posthoc_content.py simulate OUT_FOLDER REPLICATES SEED [WORKERS]
    hippocampal_posthoc_content.py summary OUT_FOLDER
    hippocampal_posthoc_content.py clustered OUT_FOLDER

`clustered` (post hoc, 24 November): the `loglinear` and `adjusted`
estimates of every group again, with the variance summed over units in which coinciding ripples of the two
hemispheres of a session are one unit (as `plain` and E62 do), beside the variance with every ripple a unit.

What decides (as in E66, fixed here before the run): for achilles, and for achilles_cicero, the recorded
`adjusted` three-way difference is *beyond the content-only law* if it lies above the 97.5th percentile of the
same quantity in 200 cohorts simulated from the fitted laws (laws fitted again to every simulated cohort), and
*explained by content* if it lies within their central 95%.  Reported beside it: the share of the recorded
`plain` three-way difference that those cohorts give.  The law is a fair stand-in for replay only if its cohorts
give at least half of the recorded `plain` change of the pair term.  The expectation recorded before the run was that the
same as in the second cohort (explained by content); a replayed maze is as aligned with the maze in time as a
change of how the cells combine would be, so the timing in Achilles does not decide between them.
What the made-up cohorts of E66 showed about these estimates was measured at the second cohort's sizes; the
first cohort's hemispheres have more ripples (651 to 3,385 per rest) and fewer place cells (9 to 41), so the
power to see a change beyond replay is not the same and is not measured here.
"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dvc_package.modulation.binary_pooled import combine, contrast  # noqa: E402

GROUPS = {"achilles": ("Achilles-10252013", "Achilles-11012013"), "cicero": ("Cicero-09012014", "Cicero-09102014", "Cicero-09172014"),
          "achilles_cicero": ("Achilles-10252013", "Achilles-11012013", "Cicero-09012014", "Cicero-09102014", "Cicero-09172014"),
          "gatsby": ("Gatsby-08022013", "Gatsby-08282013")}
SESSIONS = sorted({s for members in GROUPS.values() for s in members})
RESERVED_PAIR_FACTOR, MINIMUM_PLACE_CELLS = 1.79, 3
FILES = "sub-{animal}_ses-{session}_behavior+ecephys.nwb"


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


content = module("dryad_posthoc_content")


def map_states(maps, circular):
    """Mean rate of every cell over every state, as `dryad_posthoc_content.content_states`, with pieces that wrap
    around on a circular track.  maps: [(cells, bins)] one per running direction.  The last state is none."""
    columns, kinds = [], []
    for block in maps:
        bins = block.shape[1]
        for k, half in enumerate(content.HALF_WIDTHS):
            if circular:
                offsets = np.arange(-half, half + 1)
                if len(offsets) > bins:
                    offsets = np.arange(bins)
                columns.append(np.stack([block[:, (np.arange(bins) + o) % bins] for o in offsets]).mean(0))
            else:
                running = np.concatenate([np.zeros((len(block), 1)), np.cumsum(block, axis=1)], axis=1)
                low, high = np.clip(np.arange(bins) - half, 0, bins), np.clip(np.arange(bins) + half + 1, 0, bins)
                columns.append((running[:, high] - running[:, low]) / (high - low))
            kinds.append(np.full(bins, k))
        columns.append(block.mean(1, keepdims=True))
        kinds.append(np.array([3]))
    every = np.concatenate(maps, axis=1)
    columns.append(every.mean(1, keepdims=True))
    kinds.append(np.array([4]))
    return np.concatenate(columns, axis=1), np.concatenate(kinds)


def hemispheres(data, census_root, fields_root, session):
    """The hemispheres of a session as sessions of `dryad_posthoc_content`, and the ripple peaks of each."""
    e62 = module("hippocampal_study")
    animal = session.split("-")[0]
    name, found, _ = e62.load(Path(data) / FILES.format(animal=animal, session=session), Path(census_root) / session, Path(fields_root) / session)
    stored = np.load(Path(fields_root) / session / "fields.npz")
    track = json.loads((Path(fields_root) / session / "fields.json").read_text())["track"]
    row_of = {int(u): i for i, u in enumerate(stored["unit"])}
    map_keys = [k for k in ("map_up", "map_down") if k in stored.files]
    out = {}
    for side, h in found.items():
        x = e62.participation(h["spikes"], h["begin"], h["end"])
        rows = np.array([row_of[int(u)] for u in h["units"]])
        place = stored["place"][rows]
        cells = np.flatnonzero(place)
        if len(cells) < MINIMUM_PLACE_CELLS:
            continue
        maps = [stored[k][rows[cells]] for k in map_keys]
        maps = [m[:, np.isfinite(m).all(0)] for m in maps]
        rates, kinds = map_states([m for m in maps if m.shape[1] > 0], track == "circular")
        sets = {(kind, cls): h["sets"][f"{kind}_{cls}"] for kind in content.KINDS for cls in content.CLASSES}
        out[side] = (dict(session=f"{session}_{side}", x=x, epoch=h["epoch"], cells=cells, rates=rates, kinds=kinds, window=0.1, sets=sets,
                          sampled=content.sample_of(sets)), h["peak"])
    return out


def fit(data, census_root, fields_root, out, session):
    """Fit the law of each rest of every hemisphere of a session; store them and the ripple peaks."""
    described = {}
    for side, (d, peak) in hemispheres(data, census_root, fields_root, session).items():
        described[side] = content.store(out, d, content.laws_of(d))
        np.save(Path(out) / "sessions" / f"{d['session']}_peak.npy", peak)
    return described


def stored(out):
    """Every stored hemisphere: name -> (session, laws, peaks)."""
    names = [n for n in content.sessions_of(out)]
    return {name: (*content.stored(out, name), np.load(Path(out) / "sessions" / f"{name}_peak.npy")) for name in names}


def clustered(entries, peaks):
    """Unit labels of every hemisphere's ripples before and after: coinciding ripples of one session are one unit."""
    e62 = module("hippocampal_study")
    labels, offset = {}, np.zeros(2, dtype=np.int64)
    for session in SESSIONS:
        names = [n for n in entries if n.rsplit("_", 1)[0] == session]
        if not names:
            continue
        for k, rest in enumerate(content.RESTS):
            parts, counted = e62.clusters_of([peaks[n][rest] for n in names])
            for n, part in zip(names, parts):
                labels.setdefault(n, [None, None])[k] = part + offset[k]
            offset[k] += counted
    return labels


def pooled(entries, labels, kind, factor):
    fits = {}
    for cls in content.CLASSES:
        parts, labels_a, labels_b = [], [], []
        for name, entry in entries.items():
            part = entry["plain"].get((kind, cls))
            if part is not None:
                parts.append(part)
                labels_a.append(labels[name][0])
                labels_b.append(labels[name][1])
        if not parts:
            return None
        fits[cls] = combine(parts, labels_a, labels_b)
    d = contrast(fits["overlapping"], fits["separate"])
    return dict(difference=float(d["delta"]), se=float(d["se"] * factor), overlapping=float(fits["overlapping"].delta), overlapping_se=float(fits["overlapping"].se),
                separate=float(fits["separate"].delta), separate_se=float(fits["separate"].se))


def results(entries, peaks):
    """The three estimates for every group, from the per-hemisphere entries (names `<session>_<side>`)."""
    out = {}
    for group, members in GROUPS.items():
        chosen = {n: e for n, e in entries.items() if n.rsplit("_", 1)[0] in members}
        if not chosen:
            continue
        labels = clustered(chosen, peaks)
        factor = RESERVED_PAIR_FACTOR if group != "gatsby" else 1.0
        out[group] = {"plain": {kind: pooled(chosen, labels, kind, factor if kind == "pairs" else 1.0) for kind in content.KINDS},
                      "loglinear": {kind: content.term_change(list(chosen.values()), kind, False) for kind in content.KINDS},
                      "adjusted": {kind: content.term_change(list(chosen.values()), kind, True) for kind in content.KINDS}}
    return out


def clustered_estimates(out):
    """`loglinear` and `adjusted` of every group with coinciding bilateral ripples as one unit (post hoc)."""
    out = Path(out)
    entries, peaks = {}, {}
    for name in content.sessions_of(out):
        d, laws = content.stored(out, name)
        entries[name] = content.session_parts(d, laws)
        peaks[name] = peaks_by_rest(d, np.load(out / "sessions" / f"{name}_peak.npy"))
    report = {}
    for group, members in GROUPS.items():
        chosen = {n: e for n, e in entries.items() if n.rsplit("_", 1)[0] in members}
        if not chosen:
            continue
        labels = clustered(chosen, peaks)
        names = list(chosen)
        units = [tuple(labels[n]) for n in names]
        report[group] = {which: {kind: dict(ripples_as_units=content.term_change([chosen[n] for n in names], kind, which == "adjusted"),
                                            bilateral_units=content.term_change([chosen[n] for n in names], kind, which == "adjusted", units))
                                 for kind in content.KINDS} for which in ("loglinear", "adjusted")}
        report[group]["units"] = {rest: int(sum(len(peaks[n][rest]) for n in names)) for rest in content.RESTS}
        report[group]["clusters"] = {rest: int(max(int(labels[n][k].max()) + 1 for n in names)) for k, rest in enumerate(content.RESTS)}
    (out / "clustered.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def peaks_by_rest(d, peak):
    return {rest: peak[d["epoch"] == rest] for rest in content.RESTS}


def study_one(arguments):
    out, name = arguments
    d, laws = content.stored(out, name)
    return name, content.session_parts(d, laws), peaks_by_rest(d, np.load(Path(out) / "sessions" / f"{name}_peak.npy"))


def study(out, workers=1):
    out = Path(out)
    entries, peaks = {}, {}
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for name, entry, peak in (map if workers <= 1 else pool.map)(study_one, [(str(out), n) for n in content.sessions_of(out)]):
            entries[name], peaks[name] = entry, peak
            print("done", name, flush=True)
    report = dict(groups=results(entries, peaks), hemispheres={n: json.loads((out / "sessions" / f"{n}.json").read_text()) for n in entries})
    (out / "study.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


CACHE = {}


def simulate_cohort(arguments):
    out, seed, replicate = arguments
    started = time.time()
    if out not in CACHE:
        CACHE[out] = stored(out)
    entries, peaks, cycles = {}, {}, []
    for index, (name, (d, laws, peak)) in enumerate(CACHE[out].items()):
        x = content.simulated_session(d, laws, np.random.default_rng([seed, index, replicate]))
        again = content.laws_of(d, x)
        cycles += [law.cycles for law in again.values()]
        entries[name], peaks[name] = content.session_parts(d, again, x), peaks_by_rest(d, peak)
    return dict(replicate=replicate, seed=seed, groups=results(entries, peaks), largest_cycles=int(max(cycles)), seconds=round(time.time() - started, 1))


def simulate(out, replicates, seed, workers):
    path = Path(out) / "cohorts.jsonl"
    have = {json.loads(line)["replicate"] for line in path.read_text().splitlines()} if path.exists() else set()
    tasks = [(str(out), seed, r) for r in range(replicates) if r not in have]
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for row in (map if workers <= 1 else pool.map)(simulate_cohort, tasks):
            with open(path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row["replicate"], row["seconds"], "s", flush=True)


def summarise(out):
    """The simulated cohorts beside the recordings, group by group, and the reading by the rule of the header."""
    out = Path(out)
    rows = [json.loads(line) for line in (out / "cohorts.jsonl").read_text().splitlines()]
    recorded = json.loads((out / "study.json").read_text())["groups"]
    report = dict(cohorts=len(rows), groups={}, reading={})
    for group in recorded:
        entry = {}
        for which in content.ESTIMATES:
            for kind in content.KINDS:
                values = [r["groups"][group][which][kind] for r in rows]
                d = np.array([v["difference"] for v in values if v is not None and np.isfinite(v["difference"])])
                here = recorded[group][which][kind]
                if len(d) < 2 or here is None:
                    continue
                entry[f"{which}_{kind}"] = dict(recorded=here["difference"], recorded_se=here["se"], usable=int(len(d)), mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                                low=float(np.quantile(d, 0.025)), high=float(np.quantile(d, 0.975)), largest=float(d.max()),
                                                cohorts_at_or_above_recorded=int((d >= here["difference"]).sum()),
                                                share_given_by_the_law=float(d.mean() / here["difference"]) if here["difference"] else None)
        report["groups"][group] = entry
        if group in ("achilles", "achilles_cicero") and {"adjusted_trios", "plain_trios", "plain_pairs"} <= set(entry):
            a, p = entry["adjusted_trios"], entry["plain_pairs"]
            report["reading"][group] = dict(
                fair_stand_in=bool(p["mean"] >= 0.5 * p["recorded"]),
                adjusted_three_way="beyond the content-only law" if a["recorded"] > a["high"] else "explained by content" if a["recorded"] >= a["low"] else "below the content-only law",
                plain_three_way_share_given_by_the_law=entry["plain_trios"]["share_given_by_the_law"])
    (out / "content_only_summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        print(json.dumps(fit(*sys.argv[2:7]), indent=1))
    elif mode == "study":
        r = study(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
        content.show(r["groups"])
    elif mode == "simulate":
        simulate(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif mode == "clustered":
        r = clustered_estimates(sys.argv[2])
        for group, entry in r.items():
            for which in ("loglinear", "adjusted"):
                for kind in content.KINDS:
                    a, b = entry[which][kind]["ripples_as_units"], entry[which][kind]["bilateral_units"]
                    if a is None:
                        continue
                    print(f"{group:16s} {which:9s} {kind:5s} {a['difference']:+.3f}  SE ripples {a['se']:.3f}  SE bilateral units {b['se']:.3f}  z {b['difference'] / b['se']:+.2f}")
    elif mode == "summary":
        r = summarise(sys.argv[2])
        for group, entry in r["groups"].items():
            for name, v in entry.items():
                print(f"{group:16s} {name:16s} recorded {v['recorded']:+.3f} | law's cohorts {v['mean']:+.3f} (sd {v['sd']:.3f}; 95% {v['low']:+.3f} to {v['high']:+.3f}); "
                      f"{v['cohorts_at_or_above_recorded']} of {v['usable']} at or above the recorded value")
        print(json.dumps(r["reading"], indent=1))
    else:
        raise SystemExit(__doc__)
