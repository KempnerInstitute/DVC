"""E62.  Pair and three-neuron terms of place cells in sleep ripples before and after a novel maze.

    hippocampal_study.py session FILE.nwb CENSUS_FOLDER FIELDS_FOLDER OUTPUT_FOLDER     one session
    hippocampal_study.py combine OUTPUT_ROOT NAME SESSION [SESSION ...]                 an animal, or several

Inputs per session: the ripples of the count (`hippocampal_census.py`, second version: NREM ripples, windows of
100 ms around the peak, one checked channel per hemisphere) and the classes from the maze
(`hippocampal_place_fields.py`: overlapping and separate pairs and trios of place cells of one hemisphere).

What is estimated.  An event is a ripple of a hemisphere; a cell is 1 when it fires in the 100-ms window.  For a
class and two sets of events A and B, `dvc_package.modulation.binary_pooled` gives the change, from A to B, of the pair term (the
log odds ratio of two cells at fixed single-cell margins) or of the three-way term (at fixed pair margins) shared
by the class, within four strata of the number of other active pyramidal cells of the hemisphere (quartiles over
A and B together).  Every pair or trio keeps its own lower-order terms in A and in B and its own term.

Sets of events (blocks).  PRE and POST are the NREM ripples before and after the maze.  PRE1, PRE2 and POST1,
POST2 are their halves, cut at the median ripple time of the session (both hemispheres together).

Contrasts (B minus A):
  null    PRE2 - PRE1    nothing happened between them: a built-in check of the estimator on the same recording
  maze    POST - PRE     the question
  after   POST2 - POST1  does what changed at the maze fade (opposite sign to `maze`) or stay

Quantities, for each contrast:
  the shift of the overlapping class and of the separate class, for pairs and for trios, and their difference
  `overlapping - separate` with the covariance of the two (they share cells and events).  The difference is the
  quantity of interest: a shift common to all place cells (a different depth of sleep before and after) cancels.

Combination.  Hemispheres of a session and sessions of an animal share one shift per class: their tables enter one
conditional likelihood.

Variance.  The unit is the ripple; ripples of the two hemispheres whose peaks are within 100 ms of each other
are one unit, so that coinciding ripples are not counted as independent.  Beside it, `se_bouts` takes the bout of
NREM sleep (an interval of the lab's scoring) as the unit: the scores of all ripples of a bout are summed before
the variance is taken.  In the development animal the two agreed on average (ratio 0.68 to 1.16 over the
seventeen combined quantities, mean 0.97), with no sign that ripples of a bout resemble each other, and the bout
version is the noisier one (16 to 52 bouts per set of events); the ripple is therefore the unit of the claims.

Further outputs for the `maze` contrast (secondary):
  loglinear    the pooled log-linear comparator on the overlapping trios and on the separate trios
  single       exact conditional p-values trio by trio (the vine fitted edge by edge)
  held_out     ripples alternate between two folds; the shift is estimated on one fold and the conditional log
               probability of the other fold's split of the joint counts is compared with that under no shift (the
               model with changing pairs and a persistent term), for the pooled shift and for trio-by-trio shifts
  outside      for pairs: the same contrast in 100-ms windows centred 400 ms before each ripple that touch no
               ripple (is the change of the pair term specific to ripples)

ANIMALS (fixed on 23 November 2026, before any pair or trio statistic of any animal was computed).  Gatsby (two
sessions, three hemispheres with a checked channel) is the development animal: the pipeline is run on it first,
its results may be looked at while the pipeline is still being changed, and they are reported as development.
Achilles, Buddy and Cicero are reserved: the pipeline is run on them once, after it is frozen as the tag
`hippocampus-e62`, and their results are the study's.  By section 56 Achilles and Cicero can carry both
questions and Buddy neither (4 overlapping trios, 8 overlapping pairs).

CLAIMS (written on the same day, before any such statistic was computed; z is the estimate over its standard
error, and "differs from zero" is |z| of at least 1.96):
 (c0) Built-in check.  The `null` contrast (PRE2 - PRE1) of each difference does not differ from zero in the
      reserved animals together.  If it does for a quantity, the `maze` contrast of that quantity is reported and
      not counted.  (Added after the development animal, before the reserved ones: the z of the `null`
      contrast of each difference is also taken hemisphere by hemisphere; if their root mean square over the
      reserved hemispheres exceeds 1.5 for a quantity, every standard error of that quantity is multiplied by
      it.  In the development animal's three hemispheres it was 1.7 for the pair difference.)
 (c1) Pair term, the known effect.  The `maze` difference `overlapping - separate` of the pair term is positive
      and differs from zero in the reserved animals together, and is positive in Achilles and in Cicero.
 (c2) Three-way term, the question; no sign is predicted.  A change is claimed when the `maze` difference
      `overlapping - separate` of the three-way term differs from zero in the reserved animals together, is at
      least 0.05 in size, and has the same sign in Achilles and in Cicero.  It is called replicated when it
      differs from zero in each of the two by itself.  Otherwise its 95% interval is reported as a bound on the
      change.
 (c3) Trio by trio, nothing: the false discovery rule at 0.05 declares no overlapping trio in the reserved
      animals (section 56: no single trio can resolve a twofold change).
 (c4) Held out: the pooled shift of the trios gains over no shift if and only if (c2) claims a change; trio-by-
      trio shifts lose (a negative gain).
 (c5) `after` and `outside` are reported without a prediction.

AMENDED AFTER THE GATE (E61 simulations, before any pair or trio statistic of the recordings):
 - the size of 0.05 in (c2): in simulated hemispheres without a change of the triplet term the measured term
   drifted by up to 0.037 when pair couplings or rates of other cells changed; a shift below 0.05 is not claimed
   as a three-neuron change;
 - the shifts of the two classes are reported one by one beside their difference: when a changing class makes up
   most of the population's activity the other class moves too, because the strata are built on that activity;
 - Buddy does not enter the three-way question (4 overlapping trios; the gate's failing cell was its template).
   For (c2) the reserved animals together are Achilles and Cicero.

DEVELOPMENT (Gatsby; `hippocampus/study_dev`, `study_dev2`; looked at before the pipeline was frozen): the pipeline
ran; the `null` differences were +0.001 (se 0.042) for pairs and -0.041 (0.167) for trios; the `maze` difference of
the pair term was +0.109 (0.029) and that of the three-way term -0.044 (0.094); in `after` the pair difference was
-0.099 (0.038) and the three-way difference +0.289 (0.121).  What changed in the pipeline because of it: the
variance by bouts was added and kept as a second figure, and the rule on the hemisphere-by-hemisphere `null` z.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dvc_package.modulation.binary_pooled import SplitFamily, combine, contrast, prepare, shared_shift, single_trio_pvalues, trio_tables  # noqa: E402
from dvc_package.modulation.loglinear_pooled import pooled_loglinear_change  # noqa: E402

WINDOW, STRATA, CLUSTER = "fixed100ms", 4, 0.1
CONTRASTS = (("null", "PRE1", "PRE2"), ("maze", "PRE", "POST"), ("after", "POST1", "POST2"))
KINDS, CLASSES = ("pairs", "trios"), ("overlapping", "separate")
OUTSIDE_LAG, OUTSIDE_HALF, OUTSIDE_MARGIN = 0.4, 0.05, 0.05


def strings(dataset):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]])


def count_in(spikes, begin, end):
    return np.searchsorted(spikes, end, side="left") - np.searchsorted(spikes, begin, side="left")


def edges_of(size_a, size_b, strata=STRATA):
    return np.unique(np.quantile(np.r_[size_a, size_b], np.arange(1, strata) / strata).round())


def describe(fit):
    out = dict(delta=fit.delta, se=fit.se, z=fit.z, p=fit.p, sets=fit.trios, informative_rows=fit.informative_rows,
               information=fit.information, design_effect=fit.design_effect, delta_asymptotic=fit.delta_asymptotic)
    if fit.other and "bouts" in fit.other:
        out["se_bouts"] = fit.other["bouts"][0]
    return out


def difference(first, second):
    out = contrast(first, second)
    if first.other and second.other and "bouts" in first.other and "bouts" in second.other:
        out["se_bouts"] = contrast(first, second, "bouts")["se"]
    return out


def bout_of(times, nrem):
    """Index of the NREM bout (an interval of the lab's scoring) that holds each time."""
    return np.clip(np.searchsorted(nrem[0], times, side="right") - 1, 0, max(len(nrem[0]) - 1, 0))


def clusters_of(times_by_part):
    """Labels of the events of several parts: events whose times are within CLUSTER of the previous one join it."""
    times = np.concatenate(times_by_part)
    order = np.argsort(times, kind="stable")
    label = np.zeros(len(times), dtype=np.int64)
    if len(times):
        label[order] = np.cumsum(np.r_[0, (np.diff(times[order]) >= CLUSTER).astype(np.int64)])
    cuts = np.cumsum([len(t) for t in times_by_part])[:-1]
    return np.split(label, cuts), (int(label.max()) + 1 if len(label) else 0)


def load(path, census, fields):
    import h5py
    session = Path(fields).name
    ripples, classes = np.load(Path(census) / "ripples.npz"), np.load(Path(fields) / "fields.npz")
    checks = json.loads((Path(census) / "census.json").read_text())["hemispheres"]
    with h5py.File(path, "r") as nwb:
        units = nwb["units"]
        unit_id, unit_type, unit_side = units["id"][:], strings(units["cell_type"]), strings(units["hemisphere"])
        ends = units["spike_times_index"][:].astype(np.int64)
        starts = np.concatenate(([0], ends[:-1]))
        times = units["spike_times"][:]
        states = nwb["processing/behavior/sleep_states"]
        s_label, s_begin, s_end = strings(states["label"]), states["start_time"][:], states["stop_time"][:]
    nrem = (np.sort(s_begin[s_label == "NREM"]), np.sort(s_end[s_label == "NREM"]))
    out = {}
    for side in ("left", "right"):
        if side not in checks or checks[side]["check"] != "passed":
            continue
        cells = np.flatnonzero((unit_side == side) & (unit_type == "pyramidal cell"))
        spikes = [np.sort(times[starts[k]:ends[k]]) for k in cells]
        index = {int(u): i for i, u in enumerate(unit_id[cells])}
        sets = {f"{kind}_{name}": np.array([[index[int(u)] for u in row] for row in classes[f"{side}_{kind}_{name}"]],
                                           dtype=np.int64).reshape(-1, 2 if kind == "pairs" else 3)
                for kind in KINDS for name in CLASSES}
        events = ripples[f"{side}_{WINDOW}"]
        keep = (ripples[f"{side}_{WINDOW}_state"] == "NREM") & np.isin(ripples[f"{side}_{WINDOW}_epoch"], ("PRE", "POST"))
        every = ripples[f"{side}_max500ms"]  # every detected ripple of the hemisphere, for the windows outside ripples
        out[side] = dict(units=unit_id[cells], spikes=spikes, sets=sets, begin=events[keep, 0], end=events[keep, 1],
                         peak=events[keep, 2], epoch=ripples[f"{side}_{WINDOW}_epoch"][keep],
                         all_begin=every[:, 0], all_end=every[:, 1], all_fixed_peak=events[:, 2])
    return session, out, nrem


def participation(spikes, begin, end):
    return np.column_stack([count_in(s, begin, end) > 0 for s in spikes]).astype(np.int8) if len(spikes) else np.zeros((len(begin), 0), np.int8)


def outside_windows(h, nrem):
    """Windows of 100 ms centred OUTSIDE_LAG before each ripple, in NREM sleep, that touch no ripple."""
    centre = h["peak"] - OUTSIDE_LAG
    begin, end = centre - OUTSIDE_HALF, centre + OUTSIDE_HALF
    k = np.searchsorted(nrem[0], begin, side="right") - 1
    inside = (k >= 0) & (end <= nrem[1][np.clip(k, 0, len(nrem[1]) - 1)])
    order = np.argsort(h["all_begin"])
    rb, re = h["all_begin"][order] - OUTSIDE_MARGIN, h["all_end"][order] + OUTSIDE_MARGIN
    j = np.clip(np.searchsorted(rb, end, side="left") - 1, 0, max(len(rb) - 1, 0))
    longest = np.maximum.accumulate(re) if len(re) else re
    touches = (rb[j] < end) & (longest[j] > begin) if len(rb) else np.zeros(len(begin), bool)
    peaks = np.sort(h["all_fixed_peak"])
    near = np.searchsorted(peaks, end + OUTSIDE_HALF, side="left") - np.searchsorted(peaks, begin - OUTSIDE_HALF, side="right")
    ok = inside & ~touches & (near == 0)
    return begin[ok], end[ok], centre[ok], h["epoch"][ok]


def blocks_of(peak, epoch, cut_pre, cut_post):
    pre, post = epoch == "PRE", epoch == "POST"
    return {"PRE": pre, "POST": post, "PRE1": pre & (peak < cut_pre), "PRE2": pre & (peak >= cut_pre),
            "POST1": post & (peak < cut_post), "POST2": post & (peak >= cut_post)}


def held_out(x_a, x_b, sets, size_a, size_b):
    """Gain in conditional log probability of the held-out fold, pooled shift and set-by-set shifts, two folds."""
    fold_a, fold_b = np.arange(len(x_a)) % 2, np.arange(len(x_b)) % 2
    pooled, single, information = 0.0, 0.0, 0.0
    for train in (0, 1):
        a_train, b_train, a_test, b_test = fold_a == train, fold_b == train, fold_a != train, fold_b != train
        edges = edges_of(size_a[a_train], size_b[b_train])
        patterns = 2 ** sets.shape[1]
        ta = trio_tables(x_a[a_train], sets, size_a[a_train], edges).reshape(-1, patterns)
        tb = trio_tables(x_b[b_train], sets, size_b[b_train], edges).reshape(-1, patterns)
        family = SplitFamily(ta, tb)
        delta = shared_shift(family)[0]
        mean, variance = family.moments(0.0)
        strata = len(edges) + 1
        score = -mean.reshape(len(sets), strata).sum(1)
        info = variance.reshape(len(sets), strata).sum(1)
        each = np.clip(np.where(info > 0, score / np.where(info > 0, info, 1.0), 0.0), -2.0, 2.0)  # one step per set
        test = SplitFamily(trio_tables(x_a[a_test], sets, size_a[a_test], edges).reshape(-1, patterns),
                           trio_tables(x_b[b_test], sets, size_b[b_test], edges).reshape(-1, patterns))
        zero = test.log_probability(0.0)
        if np.isfinite(delta):
            pooled += float((test.log_probability(delta) - zero).sum())
        single += float((test.log_probability(np.repeat(each, strata)) - zero).sum())
        information += float(test.moments(0.0)[1].sum())
    return dict(pooled_gain=pooled, single_gain=single, information=information)


def session(path, census, fields, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    name, hemispheres, nrem = load(path, census, fields)
    report = dict(session=name, window=WINDOW, strata=STRATA, hemispheres={}, combined={},
                  scope="Pair and trio statistics of the sleep epochs: computed here for the first time.")
    saved, parts, peaks, bouts = {}, {}, {}, {}
    report["nrem_bouts"] = int(len(nrem[0]))
    every = np.concatenate([h["peak"] for h in hemispheres.values()]) if hemispheres else np.zeros(0)
    every_epoch = np.concatenate([h["epoch"] for h in hemispheres.values()]) if hemispheres else np.zeros(0, "U4")
    cut_pre = float(np.median(every[every_epoch == "PRE"])) if (every_epoch == "PRE").any() else 0.0
    cut_post = float(np.median(every[every_epoch == "POST"])) if (every_epoch == "POST").any() else 0.0
    report["cuts"] = dict(PRE=cut_pre, POST=cut_post)
    for side, h in hemispheres.items():
        x = participation(h["spikes"], h["begin"], h["end"])
        size = x.sum(1)
        masks = blocks_of(h["peak"], h["epoch"], cut_pre, cut_post)
        bout = bout_of(h["peak"], nrem)
        side_report = dict(cells=len(h["units"]), events={k: int(v.sum()) for k, v in masks.items()},
                           sets={k: len(v) for k, v in h["sets"].items()}, contrasts={})
        for contrast_name, a_name, b_name in CONTRASTS:
            a, b = masks[a_name], masks[b_name]
            edges = edges_of(size[a], size[b])
            entry = {}
            for kind in KINDS:
                fits = {}
                for cls in CLASSES:
                    sets = h["sets"][f"{kind}_{cls}"]
                    if len(sets) == 0:
                        continue
                    part = prepare(x[a], x[b], sets, size[a], size[b], edges)
                    parts[(contrast_name, kind, cls, side)] = part
                    fits[cls] = combine([part], also={"bouts": ([bout[a]], [bout[b]])})
                    entry[f"{kind}_{cls}"] = describe(fits[cls])
                if len(fits) == 2:
                    entry[f"{kind}_difference"] = difference(fits["overlapping"], fits["separate"])
            peaks[(contrast_name, side)] = (h["peak"][a], h["peak"][b])
            bouts[(contrast_name, side)] = (bout[a], bout[b])
            side_report["contrasts"][contrast_name] = entry
        # secondary outputs, for the maze contrast
        a, b = masks["PRE"], masks["POST"]
        edges = edges_of(size[a], size[b])
        secondary = {}
        for cls in CLASSES:
            trios = h["sets"][f"trios_{cls}"]
            if len(trios):
                fit = pooled_loglinear_change(x[a], x[b], trios)
                secondary[f"loglinear_{cls}"] = dict(delta=fit.delta, se=fit.se, z=fit.z, cells=fit.cells)
        for kind in KINDS:
            sets = h["sets"][f"{kind}_overlapping"]
            if len(sets):
                p = single_trio_pvalues(x[a], x[b], sets, size[a], size[b], edges)
                saved[f"{side}|single|{kind}"] = p
                secondary[f"single_{kind}_smallest_p"] = float(p.min())
                secondary[f"held_out_{kind}"] = held_out(x[a], x[b], sets, size[a], size[b])
        ob, oe, oc, oepoch = outside_windows(h, nrem)
        xo = participation(h["spikes"], ob, oe)
        so = xo.sum(1)
        pre, post = oepoch == "PRE", oepoch == "POST"
        secondary["outside_events"] = dict(PRE=int(pre.sum()), POST=int(post.sum()))
        if pre.sum() > 50 and post.sum() > 50:
            oedges = edges_of(so[pre], so[post])
            fits = {}
            obout = bout_of(oc, nrem)
            for cls in CLASSES:
                sets = h["sets"][f"pairs_{cls}"]
                if len(sets):
                    part = prepare(xo[pre], xo[post], sets, so[pre], so[post], oedges)
                    parts[("outside", "pairs", cls, side)] = part
                    fits[cls] = combine([part], also={"bouts": ([obout[pre]], [obout[post]])})
                    secondary[f"outside_pairs_{cls}"] = describe(fits[cls])
            if len(fits) == 2:
                secondary["outside_pairs_difference"] = difference(fits["overlapping"], fits["separate"])
            peaks[("outside", side)] = (oc[pre], oc[post])
            bouts[("outside", side)] = (obout[pre], obout[post])
        side_report["maze_secondary"] = secondary
        report["hemispheres"][side] = side_report

    # the hemispheres of the session together: coinciding ripples are one unit; beside it, bouts as units
    for contrast_name in [c[0] for c in CONTRASTS] + ["outside"]:
        sides = [s for s in hemispheres if (contrast_name, s) in peaks]
        if not sides:
            continue
        labels_a, count_a = clusters_of([peaks[(contrast_name, s)][0] for s in sides])
        labels_b, count_b = clusters_of([peaks[(contrast_name, s)][1] for s in sides])
        for i, s in enumerate(sides):
            saved[f"{contrast_name}|{s}|cluster_a"], saved[f"{contrast_name}|{s}|cluster_b"] = labels_a[i], labels_b[i]
            saved[f"{contrast_name}|{s}|bout_a"], saved[f"{contrast_name}|{s}|bout_b"] = bouts[(contrast_name, s)]
        saved[f"{contrast_name}|clusters"] = np.array([count_a, count_b])
        entry = {}
        for kind in KINDS:
            fits = {}
            for cls in CLASSES:
                use = [i for i, s in enumerate(sides) if (contrast_name, kind, cls, s) in parts]
                if not use:
                    continue
                group = [parts[(contrast_name, kind, cls, sides[i])] for i in use]
                # clusters keep their session-wide numbering, so that the two classes are aligned for the difference
                fits[cls] = combine(group, [labels_a[i] for i in use], [labels_b[i] for i in use],
                                    also={"bouts": ([bouts[(contrast_name, sides[i])][0] for i in use],
                                                    [bouts[(contrast_name, sides[i])][1] for i in use])})
                entry[f"{kind}_{cls}"] = describe(fits[cls])
            if len(fits) == 2:
                entry[f"{kind}_difference"] = difference(fits["overlapping"], fits["separate"])
        report["combined"][contrast_name] = entry
    for (contrast_name, kind, cls, side), part in parts.items():
        key = f"{contrast_name}|{kind}|{cls}|{side}"
        saved[key + "|a"], saved[key + "|b"], saved[key + "|g_a"], saved[key + "|g_b"] = part.a, part.b, part.g_a, part.g_b
        saved[key + "|meta"] = np.array([part.information_asymptotic, part.score_asymptotic, part.sets, part.strata])
    saved["nrem_bouts"] = np.array([len(nrem[0])])
    np.savez_compressed(output / "parts.npz", **saved)
    report["complete"] = True
    (output / "study.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def combine_sessions(root, name, sessions):
    """One shift per class over several sessions (an animal, or all reserved animals)."""
    from dvc_package.modulation.binary_pooled import Part
    root = Path(root)
    report = dict(name=name, sessions=list(sessions), contrasts={})
    data = {s: np.load(root / s / "parts.npz") for s in sessions}
    for contrast_name in [c[0] for c in CONTRASTS] + ["outside"]:
        entry, offsets, total, bout_offset, bout_total = {}, {}, np.zeros(2, dtype=np.int64), {}, 0
        for s in sessions:
            bout_offset[s] = bout_total
            bout_total += int(data[s]["nrem_bouts"][0])
            if f"{contrast_name}|clusters" in data[s]:
                offsets[s] = total.copy()
                total = total + data[s][f"{contrast_name}|clusters"]
        for kind in KINDS:
            fits = {}
            for cls in CLASSES:
                group, labels_a, labels_b, bouts_a, bouts_b = [], [], [], [], []
                for s in sessions:
                    for side in ("left", "right"):
                        key = f"{contrast_name}|{kind}|{cls}|{side}"
                        if key + "|a" not in data[s]:
                            continue
                        meta = data[s][key + "|meta"]
                        group.append(Part(a=data[s][key + "|a"], b=data[s][key + "|b"], g_a=data[s][key + "|g_a"],
                                          g_b=data[s][key + "|g_b"], information_asymptotic=float(meta[0]),
                                          score_asymptotic=float(meta[1]), sets=int(meta[2]), strata=int(meta[3])))
                        labels_a.append(data[s][f"{contrast_name}|{side}|cluster_a"] + offsets[s][0])
                        labels_b.append(data[s][f"{contrast_name}|{side}|cluster_b"] + offsets[s][1])
                        bouts_a.append(data[s][f"{contrast_name}|{side}|bout_a"] + bout_offset[s])
                        bouts_b.append(data[s][f"{contrast_name}|{side}|bout_b"] + bout_offset[s])
                if not group:
                    continue
                fits[cls] = combine(group, labels_a, labels_b, also={"bouts": (bouts_a, bouts_b)})
                entry[f"{kind}_{cls}"] = describe(fits[cls])
                entry[f"{kind}_{cls}"]["bouts"] = [int(len(np.unique(np.concatenate(bouts_a)))), int(len(np.unique(np.concatenate(bouts_b))))]
            if len(fits) == 2:
                entry[f"{kind}_difference"] = difference(fits["overlapping"], fits["separate"])
        report["contrasts"][contrast_name] = entry
    # trio by trio and pair by pair over the sessions: the false discovery rule at 0.05
    for kind in KINDS:
        p = np.concatenate([data[s][f"{side}|single|{kind}"] for s in sessions for side in ("left", "right")
                            if f"{side}|single|{kind}" in data[s]] or [np.zeros(0)])
        if len(p):
            order = np.sort(p)
            passed = np.flatnonzero(order <= 0.05 * np.arange(1, len(p) + 1) / len(p))
            report[f"single_{kind}"] = dict(sets=int(len(p)), declared=int(passed[-1] + 1) if len(passed) else 0,
                                            smallest_p=float(order[0]))
    studies = [json.loads((root / s / "study.json").read_text()) for s in sessions]
    # the built-in check hemisphere by hemisphere: z of the `null` contrast of each difference (claim c0)
    for kind in KINDS:
        zs = [h["contrasts"]["null"][f"{kind}_difference"]["z"] for st in studies for h in st["hemispheres"].values()
              if f"{kind}_difference" in h["contrasts"]["null"]]
        zs = [float(z) for z in zs if z is not None and np.isfinite(z)]
        rms = float(np.sqrt(np.mean(np.square(zs)))) if zs else float("nan")
        factor = rms if rms > 1.5 else 1.0
        report[f"null_z_{kind}"] = dict(z=zs, root_mean_square=rms, factor=factor)
        for entry in report["contrasts"].values():
            if f"{kind}_difference" in entry:
                d = entry[f"{kind}_difference"]
                d["se_calibrated"] = d["se"] * factor
                d["z_calibrated"] = d["delta"] / d["se_calibrated"] if d["se_calibrated"] > 0 else float("nan")
    for kind in KINDS:
        gains = [h["maze_secondary"][f"held_out_{kind}"] for st in studies for h in st["hemispheres"].values()
                 if f"held_out_{kind}" in h["maze_secondary"]]
        if gains:
            report[f"held_out_{kind}"] = {k: float(sum(g[k] for g in gains)) for k in ("pooled_gain", "single_gain", "information")}
    (root / f"combined_{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    if sys.argv[1] == "session":
        result = session(Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
        print(json.dumps(dict(session=result["session"], hemispheres=list(result["hemispheres"]),
                              maze={k: {q: round(v[q], 4) for q in ("delta", "se", "z")}
                                    for k, v in result["combined"].get("maze", {}).items()})))
    elif sys.argv[1] == "combine":
        result = combine_sessions(sys.argv[2], sys.argv[3], sys.argv[4:])
        print(json.dumps(result, indent=1))
    else:
        raise SystemExit(__doc__)
