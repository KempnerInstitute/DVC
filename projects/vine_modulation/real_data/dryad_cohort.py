"""E64.  A second cohort for E62: rats resting before and after two novel tracks (Tirole, Huelin Gorriz and
others 2022; doi:10.5061/dryad.ksn02v76h).  Count, classes and study, by the rules of E62 wherever the files allow.

    dryad_cohort.py census DATA_FOLDER SESSION OUTPUT_FOLDER      ripples and participation (marginal only)
    dryad_cohort.py fields DATA_FOLDER SESSION OUTPUT_FOLDER      place fields on the tracks and the classes
    dryad_cohort.py study  DATA_FOLDER SESSION CENSUS FIELDS OUTPUT_FOLDER     pair and three-way terms
    dryad_cohort.py claims STUDY_ROOT             the claims below, from the outputs of `hippocampal_study.py combine`
                                                  for the groups `cohort`, `rat1`, `rat1+`, `rat2`, `rat3`, `rat4`,
                                                  `rat4+`, `rat5` (`+`: with the doubly labelled session)

THE RULE.  Written on 24 November 2026 after the audit of times, counts and labels and
before any spike, field potential or position of these recordings was used for anything else.  Where it departs
from E62 the reason is what the files hold.

Sessions.  Nine, each with spikes, position and field potentials; the field-potential file of a session is the
one whose first and last times agree with its spikes (SESSIONS below; the labels of three files are wrong).  The
session whose spikes are released under two labels is called `RAT1or4_SESS`; it is rat 1's or rat 4's.
Epochs.  The stretches in the rest pot are the runs of samples the lab marked as in the pot, runs less than five
minutes apart being one.  PRE is the first stretch; POST is the first stretch of at least 30 minutes that begins
after the second listed track.  (One session returns to the tracks after that stretch; nothing later is used.)
State.  The files have no sleep scoring.  The state is stillness in the rest pot: the lab's speed below 4 cm/s.
Ripples.  The detector of E62 at the same settings (130 to 200 Hz, edges at 2 SD, peak above 5 SD, runs closer
than 30 ms merged, 20 to 500 ms) on the one channel the lab chose for ripples, normalised over the still samples
of PRE and POST together; the event is the 100 ms around the peak, peaks closer than 100 ms thinned to the higher.
A ripple counts when its peak is in PRE or POST at a still sample.  One set of ripples for the session.
Units.  There are no cell types: a unit is taken as a pyramidal cell when its rate over the session is below
5 spikes/s and it fired at least 100 spikes.  Two rats have tetrodes in visual cortex that the files do not mark:
a tetrode is kept when its pyramidal units together fire at least 2 times more inside the ripples than in the
rest of still PRE and POST; units of other tetrodes are left out.  The ripple channel is accepted when the kept
pyramidal units fire at least 3 times more inside the ripples.
Classes.  As in E62 (`hippocampal_place_fields.py`), from the first two listed tracks: running samples have a
speed above 5 cm/s on the lab's linearized position (difference over about half a second); one map per track and
running direction, 4-cm bins smoothed over 1.5 bins; a place cell fired at least 50 spikes in a map with a peak
of at least 1 spike/s and at least 0.25 bits per spike; two place cells are overlapping when their maps (all
maps side by side) correlate at 0.5 or more and separate at 0 or less; a trio is overlapping, or separate, when
its three pairs are.  One population per session: both hippocampi together, because there is one ripple channel.
Study.  The contrasts, quantities, strata (four, of the number of other active pyramidal cells) and estimator of
E62.  The second figure for the standard error takes five-minute blocks of time as units.  Secondary, as in E62:
the comparator, set by set, held out; and each class split into sets on different tetrodes and the others.

CLAIMS, written with the rule (the sign is now predicted from E62):
 (d1) Pair term: the `maze` difference `overlapping - separate` is positive in the cohort.
 (d2) Three-way term, the replication.  It is replicated in this cohort if the `maze` difference
      `overlapping - separate` is positive, differs from zero and is at least 0.05 over the nine sessions, and is
      positive in at least four of the five rats, whichever rat the doubly labelled session belongs to.  If the
      cohort's 95% interval lies below 0.10 (half of Achilles's 0.20), the cohort speaks against an effect of
      Achilles's size.  Anything else is reported as undecided.
 (d3) The built-in check and its factor, trio by trio, and held out: as (c0), (c3), (c4) of E62.

THE SIMULATION CHECK (E63, `experiments/e63_cohort2_gate.py`; run once with its reserved seed on 24 November 2026,
before any pair or trio statistic of these recordings was computed; 200 simulated cohorts for each
of four scenarios).  Five of its six expectations hold; (h1) does not, and by the rule written there the check is
not passed:
  - without a change the cohort's test of the three-way difference rejected in 7.5% of the cohorts when only the
    participations differ, in 10.0% when the pair coupling of the overlapping class changes and in 6.0% when the
    spread of the population's activity changes, against a bound of 8%;
  - the reported standard error is 5 to 11% smaller than the spread of the estimate, and with the changing pair
    coupling the estimate drifts by +0.028 (within its bound of 0.03);
  - the rule (d2) declared a replication in 2%, 4% and 2% of the cohorts without a change (bound 5%) and in 73.5%
    of the cohorts with a change that the estimate measures as 0.20; the cohort's test found that change in
    93.5%; no cohort with the change "spoke against" it, and 24 to 49% of the cohorts without one did.
Protocol amendment after the simulation check and before computing any pair or trio statistic from these recordings: every standard error of the three-way difference of this cohort is multiplied by 1.11, the largest
ratio of spread to reported standard error among the scenarios without a change (CHECK_FACTOR; `claims` applies
it on top of the factor of the built-in check).  Applied to the check's own cohorts, the test then rejects in
3.0%, 6.5% and 5.5% without a change and in 91% with it, and the rule (d2) declares in 1%, 3% and 2% and in 71%.
A drift of up to 0.03 under changing pairs remains; the size of 0.05 in (d2) is there for it.
"""
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SESSIONS = {  # name: (label of the spikes and position, label of the field potentials, rat)
    "RAT1_SESS1": ("RAT1_SESS1", "RAT4_SESS3", "rat1"), "RAT1or4_SESS": ("RAT1_SESS2", "RAT1_SESS1", "rat1or4"),
    "RAT2_SESS1": ("RAT2_SESS1", "RAT2_SESS1", "rat2"), "RAT2_SESS2": ("RAT2_SESS2", "RAT2_SESS2", "rat2"),
    "RAT3_SESS1": ("RAT3_SESS1", "RAT3_SESS1", "rat3"), "RAT3_SESS2": ("RAT3_SESS2", "RAT3_SESS2", "rat3"),
    "RAT4_SESS2": ("RAT4_SESS2", "RAT4_SESS2", "rat4"), "RAT5_SESS1": ("RAT5_SESS1", "RAT5_SESS1", "rat5"),
    "RAT5_SESS2": ("RAT5_SESS2", "RAT5_SESS2", "rat5"),
}
STILL, GAP, POST_MIN, RATE_MAX, SPIKES_MIN, TETRODE_GAIN, CHANNEL_GAIN = 4.0, 300.0, 1800.0, 5.0, 100, 2.0, 3.0
WINDOW, BLOCK = 0.1, 300.0
ASSIGNMENTS = (("rat1+", "rat2", "rat3", "rat4", "rat5"), ("rat1", "rat2", "rat3", "rat4+", "rat5"))
SIZE, AGAINST = 0.05, 0.10      # of the rule (d2): the smallest change claimed; half of Achilles's
CHECK_FACTOR = 1.11             # on every standard error of the three-way difference (amendment after E63)
GROUPS = ("cohort", "rat1", "rat1+", "rat1or4", "rat2", "rat3", "rat4", "rat4+", "rat5")


def groups_of(sessions):
    """The sessions of each rat; the doubly labelled session once with rat 1 (`rat1+`) and once with rat 4."""
    double = [s for s in sessions if SESSIONS[s][2] == "rat1or4"]
    groups = {rat: [s for s in sessions if SESSIONS[s][2] == rat] for rat in ("rat1", "rat2", "rat3", "rat4", "rat5")}
    groups["rat1+"], groups["rat4+"] = groups["rat1"] + double, groups["rat4"] + double
    return {name: members for name, members in groups.items() if members}


def rule_d2(estimate, se, by_group):
    """The rule (d2) for one cohort: replicated; speaks against an effect of Achilles's size; rats positive.

    estimate, se: the `maze` difference `overlapping - separate` of the three-way term over all sessions and its
    standard error.  by_group: the same difference for each rat (names as in ASSIGNMENTS).  The number of rats is
    the smaller of the two assignments of the doubly labelled session; a rat without an estimate is not positive.
    """
    rats = min(sum(bool(by_group.get(name, float("nan")) > 0) for name in names) for names in ASSIGNMENTS)
    usable = bool(np.isfinite(estimate) and np.isfinite(se) and se > 0)
    replicated = bool(usable and estimate > 0 and estimate / se >= 1.96 and estimate >= SIZE and rats >= 4)
    against = bool(usable and estimate + 1.96 * se < AGAINST)
    return replicated, against, int(rats)


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def count_in(spikes, begin, end):
    return np.searchsorted(spikes, end, side="left") - np.searchsorted(spikes, begin, side="left")


def load_behaviour(folder, label):
    from scipy.io import loadmat
    p = loadmat(Path(folder) / f"{label}_extracted_position.mat", squeeze_me=True, struct_as_record=False)["position"]
    t = np.asarray(p.t, dtype=float).ravel()
    in_box = np.isfinite(np.asarray(p.sleepbox, dtype=float).ravel())[:len(t)]
    speed = np.asarray(p.v_cm, dtype=float).ravel()[:len(t)]
    tracks = []
    for track in np.atleast_1d(p.linear)[:2]:
        linear = np.asarray(track.linear, dtype=float).ravel()[:len(t)] / 100.0
        on = np.flatnonzero(np.isfinite(linear))
        tracks.append(dict(linear=linear, start=float(t[on[0]]), stop=float(t[on[-1]]), length=float(track.length)))
    index = np.flatnonzero(in_box)
    cut = np.flatnonzero(np.diff(t[index]) > GAP)
    blocks = list(zip(t[np.r_[index[0], index[cut + 1]]], t[np.r_[index[cut], index[-1]]]))
    pre = blocks[0]
    later = [b for b in blocks if b[0] > tracks[1]["stop"] and b[1] - b[0] >= POST_MIN]
    return dict(t=t, in_box=in_box, speed=speed, tracks=tracks, epochs={"PRE": pre, "POST": later[0]},
                blocks=[(float(a), float(b)) for a, b in blocks])


def load_spikes(folder, label):
    from scipy.io import loadmat
    c = loadmat(Path(folder) / f"{label}_extracted_clusters.mat", squeeze_me=True, struct_as_record=False)["clusters"]
    ids, times = np.asarray(c.spike_id).ravel().astype(int), np.asarray(c.spike_times, dtype=float).ravel()
    conversion = np.atleast_2d(np.asarray(c.id_conversion)).astype(int)
    order = np.argsort(ids, kind="stable")
    ids, times = ids[order], times[order]
    spikes = {int(u): np.sort(times[np.searchsorted(ids, u, "left"):np.searchsorted(ids, u, "right")]) for u in conversion[:, 0]}
    return spikes, {int(u): int(code) // 1000 for u, code in conversion}, float(times.min()), float(times.max())


class Trace:
    """The lab's ripple channel: read in pieces, so that the whole trace is never in memory twice."""

    def __init__(self, folder, label):
        path = Path(folder) / f"{label}_extracted_CSC.mat"
        with open(path, "rb") as handle:
            old = handle.read(10) == b"MATLAB 5.0"
        self.file = None
        if old:
            from scipy.io import loadmat
            for channel in np.atleast_1d(loadmat(path, squeeze_me=True, struct_as_record=False)["CSC"]):
                if str(channel.channel_label) == "best_ripple":
                    raw = np.asarray(channel.CSCraw, dtype=np.float32).ravel()
                    time = np.asarray(channel.CSCtime, dtype=float).ravel()
                    self.n, self.fs, self.first, self.last = len(raw), float(channel.SR), float(time[0]), float(time[-1])
                    self.read = lambda i, j: raw[i:j].astype(np.float64)
                    return
            raise ValueError("no ripple channel")
        import h5py
        self.file = h5py.File(path, "r")
        group = self.file["CSC"]
        for k in range(max(group["CSCtime"].shape)):
            get = lambda name: self.file[group[name][k, 0] if group[name].shape[1] == 1 else group[name][0, k]]
            if "".join(chr(int(c)) for c in np.asarray(get("channel_label")).ravel()) == "best_ripple":
                raw, time = get("CSCraw"), get("CSCtime")
                along = raw.shape[0] == 1
                self.n, self.fs = max(raw.shape), float(np.asarray(get("SR")).ravel()[0])
                self.first = float(time[0, 0])
                self.last = float(time[0, self.n - 1] if time.shape[0] == 1 else time[self.n - 1, 0])
                self.read = (lambda i, j: np.asarray(raw[0, i:j], dtype=np.float64)) if along else (lambda i, j: np.asarray(raw[i:j, 0], dtype=np.float64))
                return
        raise ValueError("no ripple channel")

    def close(self):
        if self.file is not None:
            self.file.close()


def ripple_power(trace, detector, piece=2_000_000, pad=4000):
    """Smoothed ripple-band power of the whole trace, computed in overlapping pieces and kept as float32."""
    out = np.empty(trace.n, dtype=np.float32)
    for start in range(0, trace.n, piece):
        a, b = max(0, start - pad), min(trace.n, start + piece + pad)
        stop = min(trace.n, start + piece)
        out[start:stop] = detector.smoothed_power(trace.read(a, b), trace.fs)[start - a:stop - a]
    return out


def census(folder, session, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    detector = module("hippocampal_census")
    label, field_label, rat = SESSIONS[session]
    behaviour = load_behaviour(folder, label)
    spikes, tetrode, first, last = load_spikes(folder, label)
    trace = Trace(folder, field_label)
    fs, n = trace.fs, trace.n
    if abs((trace.last - trace.first) - (n - 1) / fs) > 0.05:
        raise ValueError("the field potential is not evenly sampled")
    t, epochs = behaviour["t"], behaviour["epochs"]
    clock = lambda index: trace.first + np.asarray(index) / fs      # time of a sample of the field potential

    def state_of(times):
        """Epoch name of each time, and whether the animal is still in the pot then."""
        index = np.clip(np.rint((times - t[0]) / (t[1] - t[0])).astype(int), 0, len(t) - 1)
        still = behaviour["in_box"][index] & (behaviour["speed"][index] < STILL)
        epoch = np.full(len(times), "none", dtype="U4")
        for name, (a, b) in epochs.items():
            epoch[(times >= a) & (times <= b)] = name
        return epoch, still

    use = np.zeros(n, dtype=bool)                 # still, in the pot, in PRE or POST
    span = {}
    for name, (a, b) in epochs.items():
        i, j = max(0, int(np.ceil((a - trace.first) * fs))), min(n, int(np.floor((b - trace.first) * fs)) + 1)
        span[name] = (i, j)
        for start in range(i, j, 2_000_000):
            index = np.arange(start, min(j, start + 2_000_000))
            at = np.clip(np.rint((clock(index) - t[0]) / (t[1] - t[0])).astype(np.int64), 0, len(t) - 1)
            use[index] = behaviour["in_box"][at] & (behaviour["speed"][at] < STILL)
    z = ripple_power(trace, detector)
    trace.close()
    chosen = z[use].astype(np.float64)
    mean, sd = float(chosen.mean()), float(chosen.std())
    del chosen
    z -= np.float32(mean)
    z /= np.float32(sd)
    starts, stops, peaks, value = detector.find_ripples(z, fs, 500.0)
    del z
    peak_time = clock(peaks)
    keep = detector.thin(peak_time, value, WINDOW)
    peak_time, value = peak_time[keep], value[keep]
    epoch, still = state_of(peak_time)
    ok = still & (epoch != "none")
    peak_time, value, epoch = peak_time[ok], value[ok], epoch[ok]
    begin, end = peak_time - WINDOW / 2, peak_time + WINDOW / 2
    seconds = {name: float(use[i:j].sum() / fs) for name, (i, j) in span.items()}
    duration = last - first
    units = sorted(spikes)
    rate = {u: len(spikes[u]) / duration for u in units}
    pyramidal = [u for u in units if rate[u] < RATE_MAX and len(spikes[u]) >= SPIKES_MIN]
    change = np.diff(np.concatenate(([0], use.view(np.int8), [0])))
    still_begin, still_end = clock(np.flatnonzero(change == 1)), clock(np.flatnonzero(change == -1) - 1)
    del change

    def gain(members):
        inside = sum(int(count_in(spikes[u], begin, end).sum()) for u in members)
        total = sum(int(count_in(spikes[u], still_begin, still_end).sum()) for u in members)
        in_seconds, all_seconds = len(begin) * WINDOW, sum(seconds.values())
        outside = (total - inside) / max(all_seconds - in_seconds, 1e-9)
        return (inside / in_seconds) / outside if outside > 0 and in_seconds > 0 else float("nan")

    by_tetrode = {tt: [u for u in pyramidal if tetrode[u] == tt] for tt in sorted({tetrode[u] for u in pyramidal})}
    tetrode_gain = {tt: gain(members) for tt, members in by_tetrode.items()}
    kept_tetrodes = [tt for tt, g in tetrode_gain.items() if g >= TETRODE_GAIN]
    kept = [u for u in pyramidal if tetrode[u] in kept_tetrodes]
    x = np.column_stack([count_in(spikes[u], begin, end) > 0 for u in kept]) if kept else np.zeros((len(begin), 0), bool)
    report = dict(session=session, rat=rat, spikes_label=label, field_label=field_label, rate=fs, units=len(units),
                  pyramidal=len(pyramidal), kept=len(kept), tetrodes=len({tetrode[u] for u in units}),
                  kept_tetrodes=kept_tetrodes, tetrode_gain={str(k): round(v, 2) for k, v in tetrode_gain.items()},
                  channel_gain=round(gain(kept), 2) if kept else None,
                  channel_check="passed" if kept and gain(kept) >= CHANNEL_GAIN else "FAILED",
                  epochs={k: [float(a), float(b)] for k, (a, b) in epochs.items()}, still_seconds=seconds,
                  scope="Marginal quantities only: no joint count of two or more named cells was computed.")
    for name in epochs:
        pick = epoch == name
        r = x[pick].mean(0) if pick.any() and len(kept) else np.zeros(len(kept))
        size = x[pick].sum(1) if len(kept) else np.zeros(0)
        report[name] = dict(ripples=int(pick.sum()), rate_per_s=round(float(pick.sum() / seconds[name]), 4) if seconds[name] else None,
                            participation_quantiles=[round(float(q), 4) for q in np.quantile(r, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(r) else [],
                            active_mean=round(float(size.mean()), 3) if len(size) else None,
                            active_sd_over_independent=round(float(size.std() / np.sqrt((r * (1 - r)).sum())), 3) if len(size) and r.sum() else None)
    lines = ["unit\ttetrode\trate\tpyramidal\tkept\tparticipation_PRE\tparticipation_POST"]
    for u in units:
        k = kept.index(u) if u in kept else -1
        values = [x[epoch == name, k].mean() if k >= 0 and (epoch == name).any() else float("nan") for name in ("PRE", "POST")]
        lines.append(f"{u}\t{tetrode[u]}\t{rate[u]:.4f}\t{int(u in pyramidal)}\t{int(u in kept)}\t{values[0]:.5f}\t{values[1]:.5f}")
    (output / "units.tsv").write_text("\n".join(lines) + "\n")
    np.savez_compressed(output / "ripples.npz", peak=peak_time, value=value, epoch=epoch, kept=np.array(kept, dtype=np.int64))
    (output / "census.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def fields(folder, session, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    place = module("hippocampal_place_fields")
    label, _, rat = SESSIONS[session]
    behaviour = load_behaviour(folder, label)
    spikes, tetrode, first, last = load_spikes(folder, label)
    t = behaviour["t"]
    step = float(t[1] - t[0])
    ahead = max(1, int(round(0.256 / step)))          # the half-second difference of E62 at this camera's rate
    units = sorted(spikes)
    maps, has_field, names, running = [], [], [], {}
    for k, track in enumerate(behaviour["tracks"]):
        linear = track["linear"]
        bins = int(round(track["length"] / place.BIN))
        v = np.full(len(linear), np.nan)
        v[ahead:-ahead] = (linear[2 * ahead:] - linear[:-2 * ahead]) / (2 * ahead * step)
        finite = np.isfinite(linear)
        sample_bin = np.clip((np.where(finite, linear, 0.0) / place.BIN).astype(int), 0, bins - 1)
        for direction, mask in (("up", finite & (v > place.SPEED)), ("down", finite & (v < -place.SPEED))):
            block = np.full((len(units), bins), np.nan)
            field = np.zeros(len(units), dtype=bool)
            for c, u in enumerate(units):
                s = spikes[u]
                sample = np.rint((s[(s >= t[0]) & (s <= t[-1])] - t[0]) / step).astype(int)
                sample = sample[(sample >= 0) & (sample < len(t))]
                sample = sample[mask[sample]]
                m, information = place.rate_map(sample_bin[sample], sample_bin[mask], bins, 1 / step, False)
                block[c] = m
                peak = np.nanmax(m) if np.isfinite(m).any() else 0.0
                field[c] = len(sample) >= place.MIN_SPIKES and peak >= place.MIN_PEAK and information >= place.MIN_INFORMATION
            maps.append(block), has_field.append(field), names.append(f"track{k + 1}_{direction}")
            running[names[-1]] = round(float(mask.sum() * step), 1)
    is_place = np.any(has_field, axis=0)
    stacked = np.concatenate(maps, axis=1)
    used = np.isfinite(stacked).all(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(stacked[:, used])
    report = dict(session=session, rat=rat, units=len(units), maps=names, used_bins=int(used.sum()),
                  running_seconds=running, place_cells_all_units=int(is_place.sum()),
                  scope="The tracks only: no spike of the rest epochs was read.")
    np.savez_compressed(output / "fields.npz", unit=np.array(units, dtype=np.int64), correlation=correlation, place=is_place,
                        tetrode=np.array([tetrode[u] for u in units], dtype=np.int64))
    (output / "fields.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def class_sets(fields_folder, kept):
    """Overlapping and separate pairs and trios among the kept pyramidal units, as indices into `kept`.

    Post hoc sensitivity runs only: the environment variables E64_OVERLAP and E64_SEPARATE replace the two
    thresholds of the classes (the study used those of `hippocampal_place_fields`, 0.5 and 0).
    """
    place = module("hippocampal_place_fields")
    place.OVERLAP = float(os.environ.get("E64_OVERLAP", place.OVERLAP))
    place.SEPARATE = float(os.environ.get("E64_SEPARATE", place.SEPARATE))
    data = np.load(Path(fields_folder) / "fields.npz")
    position = {int(u): i for i, u in enumerate(data["unit"])}
    rows = np.array([position[int(u)] for u in kept], dtype=np.int64)
    correlation = data["correlation"][np.ix_(rows, rows)]
    is_place = data["place"][rows]
    pairs, trios = place.classes(correlation, is_place, np.ones(len(kept), dtype=bool))
    return {f"pairs_{k}": v for k, v in pairs.items()} | {f"trios_{k}": v for k, v in trios.items()}, data["tetrode"][rows], is_place


def study(folder, session, census_folder, fields_folder, output):
    from dvc_package.modulation.binary_pooled import combine, prepare, single_trio_pvalues
    from dvc_package.modulation.loglinear_pooled import pooled_loglinear_change
    e62 = module("hippocampal_study")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    label, _, rat = SESSIONS[session]
    ripples = np.load(Path(census_folder) / "ripples.npz")
    kept = [int(u) for u in ripples["kept"]]
    spikes, tetrode, first, last = load_spikes(folder, label)
    peak, epoch = ripples["peak"], ripples["epoch"]
    x = np.column_stack([count_in(spikes[u], peak - WINDOW / 2, peak + WINDOW / 2) > 0 for u in kept]).astype(np.int8)
    size = x.sum(1)
    sets, cell_tetrode, is_place = class_sets(fields_folder, kept)
    cut_pre = float(np.median(peak[epoch == "PRE"]))
    cut_post = float(np.median(peak[epoch == "POST"]))
    masks = e62.blocks_of(peak, epoch, cut_pre, cut_post)
    bout = np.floor(peak / BLOCK).astype(np.int64)
    bout -= bout.min()
    side = dict(cells=len(kept), place_cells=int(is_place.sum()), events={k: int(v.sum()) for k, v in masks.items()},
                sets={k: len(v) for k, v in sets.items()}, contrasts={})
    saved, parts = {"nrem_bouts": np.array([int(bout.max()) + 1])}, {}
    for name, a_name, b_name in e62.CONTRASTS:
        a, b = masks[a_name], masks[b_name]
        edges = e62.edges_of(size[a], size[b])
        entry = {}
        for kind in e62.KINDS:
            fits = {}
            for cls in e62.CLASSES:
                chosen = sets[f"{kind}_{cls}"]
                if len(chosen) == 0:
                    continue
                part = prepare(x[a], x[b], chosen, size[a], size[b], edges)
                parts[(name, kind, cls)] = part
                fits[cls] = combine([part], also={"bouts": ([bout[a]], [bout[b]])})
                entry[f"{kind}_{cls}"] = e62.describe(fits[cls])
            if len(fits) == 2:
                entry[f"{kind}_difference"] = e62.difference(fits["overlapping"], fits["separate"])
        side["contrasts"][name] = entry
        saved[f"{name}|left|cluster_a"], saved[f"{name}|left|cluster_b"] = np.arange(a.sum()), np.arange(b.sum())
        saved[f"{name}|left|bout_a"], saved[f"{name}|left|bout_b"] = bout[a], bout[b]
        saved[f"{name}|clusters"] = np.array([int(a.sum()), int(b.sum())])
    a, b = masks["PRE"], masks["POST"]
    edges = e62.edges_of(size[a], size[b])
    secondary = {}
    for cls in e62.CLASSES:
        trios = sets[f"trios_{cls}"]
        if len(trios):
            fit = pooled_loglinear_change(x[a], x[b], trios)
            secondary[f"loglinear_{cls}"] = dict(delta=fit.delta, se=fit.se, z=fit.z, cells=fit.cells)
    for kind in e62.KINDS:
        chosen = sets[f"{kind}_overlapping"]
        if len(chosen):
            p = single_trio_pvalues(x[a], x[b], chosen, size[a], size[b], edges)
            saved[f"left|single|{kind}"] = p
            secondary[f"single_{kind}_smallest_p"] = float(p.min())
            secondary[f"held_out_{kind}"] = e62.held_out(x[a], x[b], chosen, size[a], size[b])
        for cls in e62.CLASSES:
            chosen = sets[f"{kind}_{cls}"]
            if len(chosen) == 0:
                continue
            apart = np.array([len(set(cell_tetrode[s])) == chosen.shape[1] for s in chosen])
            for tag, mask in (("different_tetrodes", apart), ("shared_tetrode", ~apart)):
                if mask.any():
                    fit = combine([prepare(x[a], x[b], chosen[mask], size[a], size[b], edges)])
                    secondary[f"{kind}_{cls}_{tag}"] = dict(sets=int(mask.sum()), delta=fit.delta, se=fit.se, z=fit.z)
    side["maze_secondary"] = secondary
    for (name, kind, cls), part in parts.items():
        key = f"{name}|{kind}|{cls}|left"
        saved[key + "|a"], saved[key + "|b"], saved[key + "|g_a"], saved[key + "|g_b"] = part.a, part.b, part.g_a, part.g_b
        saved[key + "|meta"] = np.array([part.information_asymptotic, part.score_asymptotic, part.sets, part.strata])
    np.savez_compressed(output / "parts.npz", **saved)
    report = dict(session=session, rat=rat, hemispheres={"left": side}, combined=side["contrasts"], complete=True,
                  note="one population per session; it is stored under the name `left` so that the tables of E62 read it")
    (output / "study.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def claims(root):
    """The claims (d1) to (d3) from the combined outputs, and a table of every group with the rule's standard error.

    The standard error of a difference `overlapping - separate` carries the factor of the built-in check over the
    cohort's sessions and, for the three-way term, CHECK_FACTOR.  Writes `claims.json` and `claims_table.tsv`.
    """
    root = Path(root)
    read = lambda name: json.loads((root / f"combined_{name}.json").read_text())
    groups = {g: read(g) for g in GROUPS if (root / f"combined_{g}.json").exists()}
    cohort = groups["cohort"]
    factor = {kind: float(cohort[f"null_z_{kind}"]["factor"]) * (CHECK_FACTOR if kind == "trios" else 1.0) for kind in ("pairs", "trios")}
    out = dict(sessions=cohort["sessions"], groups={g: v["sessions"] for g, v in groups.items()},
               factor_of_the_built_in_check={kind: float(cohort[f"null_z_{kind}"]["factor"]) for kind in factor},
               check_factor_trios=CHECK_FACTOR)
    lines = ["group\tsessions\tcontrast\tquantity\tdelta\tse\tse_rule\tse_blocks\tsets\tinformation\tdesign_effect"]
    text = lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.5g}" if isinstance(v, float) else str(v))
    for g, report in groups.items():
        for name, entry in report["contrasts"].items():
            for quantity, values in entry.items():
                kind = quantity.split("_")[0]
                rule = values["se"] * factor[kind] if quantity.endswith("_difference") and values.get("se") is not None else None
                lines.append("\t".join([g, str(len(report["sessions"])), name, quantity] + [text(v) for v in (
                    values.get("delta"), values.get("se"), rule, values.get("se_bouts"), values.get("sets"), values.get("information"),
                    values.get("design_effect"))]))
    (root / "claims_table.tsv").write_text("\n".join(lines) + "\n")
    for kind in ("pairs", "trios"):
        entry = dict(factor=factor[kind], null_z_root_mean_square=cohort[f"null_z_{kind}"]["root_mean_square"],
                     null_z=cohort[f"null_z_{kind}"]["z"])
        for name in ("null", "maze", "after"):
            d = cohort["contrasts"][name][f"{kind}_difference"]
            se = d["se"] * factor[kind]
            entry[name] = dict(estimate=d["delta"], se=se, se_ripples=d["se"], se_blocks=d.get("se_bouts"), z=d["delta"] / se,
                               interval=[d["delta"] - 1.96 * se, d["delta"] + 1.96 * se],
                               overlapping=cohort["contrasts"][name][f"{kind}_overlapping"]["delta"],
                               separate=cohort["contrasts"][name][f"{kind}_separate"]["delta"])
        entry["by_group"] = {g: dict(estimate=v["contrasts"]["maze"][f"{kind}_difference"]["delta"],
                                     se=v["contrasts"]["maze"][f"{kind}_difference"]["se"] * factor[kind])
                             for g, v in groups.items() if f"{kind}_difference" in v["contrasts"]["maze"]}
        entry["single"] = cohort.get(f"single_{kind}")
        entry["held_out"] = cohort.get(f"held_out_{kind}")
        out[kind] = entry
    pairs, trios = out["pairs"], out["trios"]
    replicated, against, rats = rule_d2(trios["maze"]["estimate"], trios["maze"]["se"],
                                        {g: v["estimate"] for g, v in trios["by_group"].items()})
    check = {kind: bool(abs(out[kind]["null"]["z"]) < 1.96) for kind in ("pairs", "trios")}
    out["verdicts"] = dict(
        d1_pair_term_positive=bool(pairs["maze"]["estimate"] > 0),
        d1_pair_term_differs_from_zero=bool(abs(pairs["maze"]["z"]) >= 1.96),
        d2_replicated=bool(replicated and check["trios"]), d2_by_the_rule_before_the_built_in_check=replicated,
        d2_speaks_against_achilles_size=against, d2_rats_positive_of_five=rats,
        d3_built_in_check_holds=check,
        d3_no_single_trio_declared=bool(trios["single"] is not None and trios["single"]["declared"] == 0),
        d3_held_out_pooled_gain_agrees=bool(trios["held_out"] is not None and (trios["held_out"]["pooled_gain"] > 0) == bool(replicated)),
        d3_held_out_single_shifts_lose=bool(trios["held_out"] is not None and trios["held_out"]["single_gain"] < 0))
    (root / "claims.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "claims":
        print(json.dumps(claims(sys.argv[2]), indent=1))
    elif mode == "census":
        r = census(sys.argv[2], sys.argv[3], sys.argv[4])
        print(json.dumps({k: r[k] for k in ("session", "units", "pyramidal", "kept", "kept_tetrodes", "tetrode_gain", "channel_gain",
                                             "channel_check", "still_seconds")} | {e: r[e] for e in ("PRE", "POST")}))
    elif mode == "fields":
        r = fields(sys.argv[2], sys.argv[3], sys.argv[4])
        print(json.dumps({k: r[k] for k in ("session", "units", "maps", "used_bins", "place_cells_all_units")}))
    elif mode == "study":
        r = study(*sys.argv[2:7])
        print(json.dumps(dict(session=r["session"], cells=r["hemispheres"]["left"]["cells"], sets=r["hemispheres"]["left"]["sets"],
                              maze={k: {q: round(v[q], 4) for q in ("delta", "se", "z")} for k, v in r["combined"]["maze"].items()})))
    else:
        raise SystemExit(__doc__)
