"""Count of ripples and of each cell's participation in the hippocampal sleep-maze-sleep recordings.

A feasibility count for one session file of DANDI 000044 (version 0.261106.0236). It measures how many ripples
there are and how often each cell takes part, which decide whether a change of a three-neuron term between the
sleep before (PRE) and after (POST) the maze could be measured. It computes marginal quantities only.

THE RULE (written and committed on 23 November 2026, before any spike time or field potential of these recordings
was read; it is the count's rule, not yet the frozen rule of a study):

1. States are the lab's five-state scoring (`processing/behavior/sleep_states`); epochs are PRE and POST.
2. Channel. For each hemisphere, among the channels located in CA1 that the lab did not mark bad, the channel with
   the largest median power in the ripple band over 40 windows of 20 s spread evenly over the NREM sleep of the
   session. (Changed in the second version, see below.)
3. Ripples are found with the lab's detector (FindRipples of FMAToolbox, bz_FindRipples of buzcode) at its default
   settings: band 130 to 200 Hz (here a zero-phase Butterworth filter of order 4); the squared signal, smoothed by
   a centred moving average of 11 samples and normalised by its mean and standard deviation over all NREM samples
   of the session (one normalisation for PRE and POST); a ripple is a run above 2 SD whose peak exceeds 5 SD; runs
   closer than 30 ms are merged; duration at least 20 ms and at most 100 ms. A second count allows up to 500 ms.
   The published study's own detection settings were not found; these are the lab toolbox's.
4. A ripple belongs to the state and the epoch of its peak. NREM is the count of interest; the other states are
   reported as a check of the detector (ripples should be rare in REM and in active waking).
5. A unit takes part in a ripple if it fires at least one spike between the ripple's start and its end. Each unit
   is counted against the ripples of its own hemisphere, so that ripples are defined by the field potential and
   not by the spikes of the cells counted.
6. Computed: numbers, rates, durations and intervals of ripples; each unit's participation in each epoch; the
   number of pyramidal cells active in each ripple (for the dependence between successive ripples only); and, from
   the participations alone, the standard error a pair or trio term would have if the cells were independent
   (`1 / sqrt(n x product of r(1 - r))`, per epoch, and for the difference between epochs).
   NOT computed, in any epoch: any joint count of two or more named cells (co-participation, correlation, odds
   ratio) and any contrast of such a quantity between epochs.

SECOND VERSION (23 November 2026, after the first count, `census_v1`; both changes follow from marginal quantities
of that count and were written here and committed before the second count was run):

(a) Channel. In one hemisphere (Gatsby, 28 August 2013, left) rule 2 chose a channel whose events were not
    ripples: their rate in REM sleep was half their rate in NREM sleep, pyramidal cells fired only twice as much
    inside them as outside, a quarter of them had no pyramidal cell active, and 16% coincided with a ripple of the
    other hemisphere. In the other hemispheres with at least five pyramidal cells, pyramidal cells fired 4.2 to
    8.2 times more inside ripples and the rate in REM was at most 0.30 of the rate in NREM. Channels are now tried
    in the order of the strongest channel of each shank (strongest shank first), and a channel is accepted when
    the pyramidal cells of the hemisphere fire at least 3 times more inside its NREM ripples than in the rest of
    NREM sleep and its ripple rate in REM is at most 0.4 of its rate in NREM. A hemisphere with fewer than five
    pyramidal cells is not checked. If no shank passes, the hemisphere is marked FAILED. The check uses the summed
    firing of all pyramidal cells of the hemisphere, once, to choose a channel; it does not select ripples.
(b) Windows. With the detector's own bounds the median ripple lasted 31 to 38 ms and the median pyramidal cell
    took part in 7 to 16% of the ripples. Two further counts use a window of fixed length centred on the ripple's
    peak, 100 ms and 200 ms, for the ripples of the 500-ms set thinned so that windows do not overlap (of two
    peaks closer than the window, the higher is kept).

    python -I hippocampal_census.py FILE.nwb OUTPUT_FOLDER        (OUTPUT_FOLDER must not exist)
    python -I hippocampal_census.py --selftest
"""
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

VERSION = 2
BAND = (130.0, 200.0)
LOW, HIGH = 2.0, 5.0
GAP_MS, MIN_MS, MAX_MS = 30.0, 20.0, (100.0, 500.0)
FIXED_MS = (100.0, 200.0)
SMOOTH = 11
WINDOWS, WINDOW_S = 40, 20.0
GAIN_MIN, REM_RATIO_MAX, CHECK_MIN_CELLS = 3.0, 0.4, 5
STATES = ("Wake", "Drowsy", "NREM", "Intermediate", "REM")
# standard error of a log-odds term below which a change of this size is found with 80% power (two-sided 5%)
POWER = 1.959964 + 0.841621
FOLDS = (1.2, 1.5, 2.0, 4.0)


def strings(dataset):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]])


def band_filter(x, fs):
    from scipy import signal
    sos = signal.butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0)


def smoothed_power(x, fs):
    y = band_filter(x, fs)
    y *= y
    c = np.cumsum(np.concatenate(([0.0], y)))
    s = (c[SMOOTH:] - c[:-SMOOTH]) / SMOOTH
    half = SMOOTH // 2
    return np.concatenate((np.full(half, s[0]), s, np.full(half, s[-1])))


def find_ripples(z, fs, max_ms):
    """Start, stop (exclusive), peak sample and peak value of the ripples in the normalised power z."""
    above = z > LOW
    edges = np.diff(above.astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1) + 1, np.flatnonzero(edges == -1) + 1
    if len(starts) and len(stops) and stops[0] <= starts[0]:
        stops = stops[1:]  # a run that began before the recording is incomplete
    if len(starts) and (len(stops) == 0 or starts[-1] >= stops[-1]):
        starts = starts[:-1]  # a run that does not end is incomplete
    if len(starts) == 0:
        return (np.zeros(0, dtype=np.int64),) * 3 + (np.zeros(0),)
    keep = (starts[1:] - stops[:-1]) >= GAP_MS / 1000.0 * fs
    starts, stops = starts[np.concatenate(([True], keep))], stops[np.concatenate((keep, [True]))]
    duration = (stops - starts) / fs * 1000.0
    good = (duration >= MIN_MS) & (duration <= max_ms)
    starts, stops = starts[good], stops[good]
    peaks = np.array([a + int(np.argmax(z[a:b])) for a, b in zip(starts, stops)], dtype=np.int64)
    peak_value = z[peaks] if len(peaks) else np.zeros(0)
    good = peak_value >= HIGH
    return starts[good], stops[good], peaks[good], peak_value[good]


def thin(peak_time, value, width):
    """Keep ripples whose peaks are at least `width` apart; of two closer peaks the higher is kept."""
    keep, removed = np.zeros(len(peak_time), dtype=bool), np.zeros(len(peak_time), dtype=bool)
    for i in np.argsort(-value, kind="stable"):
        if removed[i]:
            continue
        keep[i] = True
        a = np.searchsorted(peak_time, peak_time[i] - width, side="right")
        b = np.searchsorted(peak_time, peak_time[i] + width, side="left")
        removed[a:b] = True
    return keep


def interval_index(times, begin, end):
    """Index of the interval [begin, end) that holds each time, or -1."""
    order = np.argsort(begin)
    k = np.searchsorted(begin[order], times, side="right") - 1
    inside = (k >= 0) & (times < end[order][np.clip(k, 0, len(order) - 1)])
    return np.where(inside, order[np.clip(k, 0, len(order) - 1)], -1)


def count_in(spikes, begin, end):
    return np.searchsorted(spikes, end, side="left") - np.searchsorted(spikes, begin, side="left")


def firing_gain(spike_list, begin, end, sleep_begin, sleep_end):
    """Rate of the cells inside the events, the rate in the rest of the given sleep, and their ratio."""
    seconds, inside_seconds = float((sleep_end - sleep_begin).sum()), float((end - begin).sum())
    if len(spike_list) == 0 or len(begin) == 0 or inside_seconds <= 0 or seconds <= inside_seconds:
        return None
    inside = sum(int(count_in(s, begin, end).sum()) for s in spike_list)
    total = sum(int(count_in(s, sleep_begin, sleep_end).sum()) for s in spike_list)
    rate_in = inside / inside_seconds / len(spike_list)
    rate_out = (total - inside) / (seconds - inside_seconds) / len(spike_list)
    return dict(rate_in_ripples=round(rate_in, 3), rate_outside=round(rate_out, 3),
                gain=round(rate_in / rate_out, 2) if rate_out > 0 else None)


def quantiles(x, q=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, dtype=float)
    return [round(float(v), 4) for v in np.quantile(x, q)] if len(x) else []


def autocorrelation(x, lags=(1, 2, 3)):
    x = np.asarray(x, dtype=float)
    if len(x) < 50 or x.std() == 0:
        return []
    x = x - x.mean()
    return [round(float((x[lag:] * x[:-lag]).mean() / (x * x).mean()), 4) for lag in lags]


def term_census(r_pre, r_post, n_pre, n_post, order):
    """Standard errors that terms of the given order would have under independence, from participations alone."""
    cells = len(r_pre)
    out = dict(order=order, cells=cells, terms=math.comb(cells, order) if cells >= order else 0)
    if cells < order or n_pre == 0 or n_post == 0:
        return out
    a, b = r_pre * (1 - r_pre), r_post * (1 - r_post)
    index = np.array(list(itertools.combinations(range(cells), order)), dtype=np.int32)
    info_pre, info_post = n_pre * np.prod(a[index], axis=1), n_post * np.prod(b[index], axis=1)
    with np.errstate(divide="ignore"):
        variance = 1.0 / info_pre + 1.0 / info_post
    se = np.sqrt(variance)
    finite = np.isfinite(se)
    out["estimable"] = int(finite.sum())
    out["se_quantiles_10_25_50_75_90"] = quantiles(se[finite])
    out["terms_resolving_fold_change"] = {str(f): int((se[finite] <= math.log(f) / POWER).sum()) for f in FOLDS}
    # pooled bounds: every term independent (too generous: terms share cells), and terms that share no cell
    out["information_all_terms"] = float((1.0 / variance[finite]).sum()) if finite.any() else 0.0
    rank = np.argsort(-(a * b))
    groups = [rank[i:i + order] for i in range(0, cells - cells % order, order)]
    v = np.array([1.0 / (n_pre * np.prod(a[g])) + 1.0 / (n_post * np.prod(b[g])) if np.all(a[g] * b[g] > 0) else np.inf
                  for g in groups])
    out["disjoint_terms"] = int(np.isfinite(v).sum())
    out["information_disjoint_terms"] = float((1.0 / v[np.isfinite(v)]).sum()) if np.isfinite(v).any() else 0.0
    for name in ("all_terms", "disjoint_terms"):
        information = out[f"information_{name}"]
        out[f"pooled_se_{name}"] = round(1.0 / math.sqrt(information), 5) if information > 0 else None
        out[f"smallest_fold_change_{name}"] = round(math.exp(POWER / math.sqrt(information)), 3) if information > 0 else None
    return out


def census(path, output):
    import h5py
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = dict(file=path.name, version=VERSION,
                  rule=dict(band=BAND, low=LOW, high=HIGH, gap_ms=GAP_MS, min_ms=MIN_MS, max_ms=MAX_MS,
                            fixed_ms=FIXED_MS, smooth=SMOOTH, windows=WINDOWS, window_s=WINDOW_S,
                            gain_min=GAIN_MIN, rem_ratio_max=REM_RATIO_MAX, check_min_cells=CHECK_MIN_CELLS),
                  scope="Marginal quantities only: no joint count of two or more named cells was computed.")
    with h5py.File(path, "r") as nwb:
        series = nwb["processing/ecephys/LFP/lfp"]
        lfp, fs, t0 = series["data"], float(series["starting_time"].attrs["rate"]), float(series["starting_time"][()])
        samples = lfp.shape[0]
        region = series["electrodes"][:]
        table = nwb["general/extracellular_ephys/electrodes"]
        bad = table["bad_electrode"][:].astype(bool)[region]
        channel_side = strings(table["hemisphere"])[region]
        channel_shank = table["spike_group"][:][region]
        in_ca1 = strings(table["location"])[region] == "CA1 field of hippocampus"
        states = nwb["processing/behavior/sleep_states"]
        s_label, s_begin, s_end = strings(states["label"]), states["start_time"][:], states["stop_time"][:]
        epochs = nwb["intervals/epochs"]
        e_label, e_begin, e_end = strings(epochs["label"]), epochs["start_time"][:], epochs["stop_time"][:]
        units = nwb["units"]
        unit_id, unit_type, unit_side = units["id"][:], strings(units["cell_type"]), strings(units["hemisphere"])
        unit_shank = units["spike_group"][:]
        ends = units["spike_times_index"][:].astype(np.int64)
        all_spikes = units["spike_times"][:]
        spikes = [np.sort(all_spikes[a:b]) for a, b in zip(np.concatenate(([0], ends[:-1])), ends)]
        del all_spikes
        duration = samples / fs
        pyramidal = unit_type == "pyramidal cell"
        report.update(fs=fs, samples=samples, hours=round(duration / 3600, 3), units=len(unit_id),
                      epochs={l: [float(a), float(b)] for l, a, b in zip(e_label, e_begin, e_end)})

        # each state within each sleep epoch: seconds and intervals
        state_seconds = {e: {s: 0.0 for s in STATES} for e in ("PRE", "POST")}
        pieces = {e: {s: [] for s in STATES} for e in ("PRE", "POST")}
        for label, a, b in zip(s_label, s_begin, s_end):
            for e, ea, eb in zip(e_label, e_begin, e_end):
                if e in state_seconds and label in STATES and min(b, eb) > max(a, ea):
                    state_seconds[e][label] += min(b, eb) - max(a, ea)
                    pieces[e][label].append((max(a, ea), min(b, eb)))
        pieces = {e: {s: (np.array([p[0] for p in v]), np.array([p[1] for p in v])) for s, v in d.items()}
                  for e, d in pieces.items()}
        report["state_seconds"] = {e: {s: round(float(v), 1) for s, v in d.items()} for e, d in state_seconds.items()}
        sleep_nrem = tuple(np.concatenate([pieces[e]["NREM"][k] for e in ("PRE", "POST")]) for k in (0, 1))
        nrem_total = sum(state_seconds[e]["NREM"] for e in ("PRE", "POST"))
        rem_total = sum(state_seconds[e]["REM"] for e in ("PRE", "POST"))

        def labels(times):
            state = interval_index(times, s_begin, s_end)
            epoch = interval_index(times, e_begin, e_end)
            return (np.where(state >= 0, s_label[np.clip(state, 0, len(s_label) - 1)], "none"),
                    np.where(epoch >= 0, e_label[np.clip(epoch, 0, len(e_label) - 1)], "none"))

        # NREM samples, and the windows used to rank the channels by ripple-band power
        nrem = np.zeros(samples, dtype=bool)
        candidates = []
        width = int(WINDOW_S * fs)
        for label, a, b in zip(s_label, s_begin, s_end):
            if label != "NREM":
                continue
            i, j = max(0, int(math.ceil((a - t0) * fs))), min(samples, int(math.floor((b - t0) * fs)))
            nrem[i:j] = True
            candidates += list(range(i, j - width + 1, width))
        chosen = [candidates[int(k)] for k in np.linspace(0, len(candidates) - 1, min(WINDOWS, len(candidates)))]
        power = np.empty((len(chosen), lfp.shape[1]))
        for k, i in enumerate(chosen):
            block = band_filter(lfp[i:i + width, :], fs)
            power[k] = (block * block).mean(axis=0)
        median_power = np.median(power, axis=0)
        report["channel_selection"] = dict(windows=len(chosen), nrem_window_candidates=len(candidates))

        hemispheres = {}
        for side in ("left", "right"):
            eligible = np.flatnonzero((channel_side == side) & ~bad & in_ca1)
            if len(eligible) == 0:
                continue
            best = {}
            for c in eligible:
                g = int(channel_shank[c])
                if g not in best or median_power[c] > median_power[best[g]]:
                    best[g] = int(c)
            order = sorted(best.values(), key=lambda c: -median_power[c])
            side_pyramidal = [spikes[k] for k in np.flatnonzero((unit_side == side) & pyramidal)]
            tried, accepted, fallback = [], None, None
            for channel in order:
                z = smoothed_power(lfp[:, channel], fs)
                mean, sd = float(z[nrem].mean()), float(z[nrem].std())
                z -= mean
                z /= sd
                starts, stops, peaks, value = find_ripples(z, fs, MAX_MS[0])
                state_name, epoch_name = labels(t0 + peaks / fs)
                asleep = np.isin(epoch_name, ("PRE", "POST"))
                n_nrem, n_rem = int((asleep & (state_name == "NREM")).sum()), int((asleep & (state_name == "REM")).sum())
                ratio = (n_rem / rem_total) / (n_nrem / nrem_total) if rem_total > 60 and n_nrem > 0 else None
                pick = asleep & (state_name == "NREM")
                gain = firing_gain(side_pyramidal, t0 + starts[pick] / fs, t0 + stops[pick] / fs, *sleep_nrem)
                gain = gain["gain"] if gain else None
                entry = dict(channel=int(channel), shank=int(channel_shank[channel]), nrem_ripples=n_nrem,
                             rem_to_nrem_rate=round(ratio, 4) if ratio is not None else None, pyramidal_gain=gain,
                             power_mean=mean, power_sd=sd)
                tried.append(entry)
                if len(side_pyramidal) < CHECK_MIN_CELLS:
                    accepted = (entry, z, "not checked: fewer than five pyramidal cells")
                    break
                if gain is not None and gain >= GAIN_MIN and ratio is not None and ratio <= REM_RATIO_MAX:
                    accepted = (entry, z, "passed")
                    break
                if fallback is None or (gain or 0.0) > (fallback[0]["pyramidal_gain"] or 0.0):
                    fallback = (entry, z, "FAILED: no shank passed the check; the channel with the largest gain is used")
                del z
            entry, z, status = accepted if accepted is not None else fallback
            hemispheres[side] = dict(z=z, channel=entry["channel"], check=status, tried=tried,
                                     eligible_channels=len(eligible), shanks=len(order))
            del accepted, fallback

    session_rate = np.array([len(s) / duration for s in spikes])
    tertile = np.full(len(unit_id), -1)
    if pyramidal.sum() >= 3:
        cuts = np.quantile(session_rate[pyramidal], [1 / 3, 2 / 3])
        tertile[pyramidal] = np.searchsorted(cuts, session_rate[pyramidal], side="right")

    unit_rows = {int(u): dict(unit=int(u), cell_type=unit_type[k], hemisphere=unit_side[k], shank=int(unit_shank[k]),
                              spikes=len(spikes[k]), session_rate=round(float(session_rate[k]), 4),
                              rate_tertile=int(tertile[k])) for k, u in enumerate(unit_id)}
    events_out = {}
    report["hemispheres"] = {}
    for side, h in hemispheres.items():
        z = h.pop("z")
        members = np.flatnonzero(unit_side == side)
        is_pyr = pyramidal[members]
        is_int = unit_type[members] == "interneuron"
        side_report = dict(h, units=len(members), pyramidal=int(is_pyr.sum()), interneurons=int(is_int.sum()))
        sets = {}
        for max_ms in MAX_MS:
            starts, stops, peaks, value = find_ripples(z, fs, max_ms)
            sets[f"max{int(max_ms)}ms"] = (t0 + starts / fs, t0 + stops / fs, t0 + peaks / fs, value)
        _, _, peak_all, value_all = sets[f"max{int(MAX_MS[-1])}ms"]
        for fixed in FIXED_MS:
            keep = thin(peak_all, value_all, fixed / 1000.0)
            sets[f"fixed{int(fixed)}ms"] = (peak_all[keep] - fixed / 2000.0, peak_all[keep] + fixed / 2000.0,
                                            peak_all[keep], value_all[keep])
        for key, (begin, end, peak_time, peak_value) in sets.items():
            state_name, epoch_name = labels(peak_time)
            events_out[f"{side}_{key}"] = np.column_stack((begin, end, peak_time, peak_value))
            events_out[f"{side}_{key}_state"] = state_name.astype("U12")
            events_out[f"{side}_{key}_epoch"] = epoch_name.astype("U4")
            if len(members):
                n_spikes = np.column_stack([count_in(spikes[k], begin, end) for k in members])
            else:
                n_spikes = np.zeros((len(begin), 0), dtype=int)
            took_part = n_spikes > 0
            size = took_part[:, is_pyr].sum(axis=1)
            result = dict(ripples_all_states=len(begin))
            result["rate_per_s_by_state"] = {
                e: {s: (round(float(((epoch_name == e) & (state_name == s)).sum() / state_seconds[e][s]), 4)
                        if state_seconds[e][s] > 60 else None) for s in STATES} for e in ("PRE", "POST")}
            result["count_by_state"] = {e: {s: int(((epoch_name == e) & (state_name == s)).sum()) for s in STATES}
                                        for e in ("PRE", "POST")}
            participation = {}
            for e in ("PRE", "POST"):
                pick = (epoch_name == e) & (state_name == "NREM")
                n = int(pick.sum())
                gaps = np.diff(begin[pick])
                r = took_part[pick].mean(axis=0) if n else np.zeros(len(members))
                participation[e] = (n, r)
                nrem_e = float(state_seconds[e]["NREM"])
                firing = {}
                for name, mask in (("pyramidal", is_pyr), ("interneuron", is_int)):
                    g = firing_gain([spikes[k] for k in members[mask]], begin[pick], end[pick], *pieces[e]["NREM"])
                    if g:
                        firing[name] = g
                result[e] = dict(
                    nrem_ripples=n, nrem_seconds=round(nrem_e, 1), rate_per_s=round(n / nrem_e, 4) if nrem_e else None,
                    duration_ms_quantiles=quantiles((end[pick] - begin[pick]) * 1000.0),
                    interval_s_quantiles=quantiles(gaps),
                    within_200ms_of_previous=round(float((gaps < 0.2).mean()), 4) if n > 1 else None,
                    firing=firing,
                    pyramidal_active_per_ripple=dict(mean=round(float(size[pick].mean()), 3) if n else None,
                                                     quantiles=quantiles(size[pick]),
                                                     share_with_none=round(float((size[pick] == 0).mean()), 4) if n else None,
                                                     autocorrelation_lags_1_2_3=autocorrelation(size[pick])),
                    pyramidal_participation_quantiles=quantiles(r[is_pyr]),
                    pyramidal_participation_by_rate_tertile={
                        name: quantiles(r[is_pyr & (tertile[members] == t)], (0.25, 0.5, 0.75))
                        for t, name in enumerate(("slow", "middle", "fast"))},
                    pyramidal_cells_at_least={str(c): int((r[is_pyr] >= c).sum()) for c in (0.02, 0.05, 0.1, 0.2, 0.3)})
                for k, u in enumerate(unit_id[members]):
                    unit_rows[int(u)][f"participation_{e}_{key}"] = round(float(r[k]), 5)
                    unit_rows[int(u)][f"nrem_ripples_{e}_{key}"] = n
            (n_pre, r_pre), (n_post, r_post) = participation["PRE"], participation["POST"]
            result["pyramidal_cells_at_least_in_both_epochs"] = {
                str(c): int(((r_pre[is_pyr] >= c) & (r_post[is_pyr] >= c)).sum()) for c in (0.02, 0.05, 0.1, 0.2, 0.3)}
            result["pairs"] = term_census(r_pre[is_pyr], r_post[is_pyr], n_pre, n_post, 2)
            result["trios"] = term_census(r_pre[is_pyr], r_post[is_pyr], n_pre, n_post, 3)
            side_report[key] = result
        report["hemispheres"][side] = side_report
        del z

    # ripples of the left hemisphere that overlap a ripple of the right (how far ripples are shared)
    if len(hemispheres) == 2:
        for key in (f"max{int(m)}ms" for m in MAX_MS):
            a, b = events_out[f"left_{key}"], events_out[f"right_{key}"]
            if len(a) and len(b):
                k = np.clip(np.searchsorted(b[:, 0], a[:, 1], side="left") - 1, 0, len(b) - 1)
                overlap = (b[k, 0] < a[:, 1]) & (b[k, 1] > a[:, 0])
                report.setdefault("left_ripples_overlapping_a_right_ripple", {})[key] = round(float(overlap.mean()), 4)

    columns = sorted({c for row in unit_rows.values() for c in row},
                     key=lambda c: (c.startswith(("participation", "nrem_ripples")), c))
    lines = ["\t".join(columns)] + ["\t".join(str(row.get(c, "")) for c in columns) for row in unit_rows.values()]
    (output / "units.tsv").write_text("\n".join(lines) + "\n")
    np.savez_compressed(output / "ripples.npz", **events_out)
    report.update(seconds=round(time.time() - started, 1), complete=True)
    (output / "census.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def selftest():
    """The detector on a synthetic trace: 150-Hz bursts of known number, position and length in noise."""
    rng = np.random.default_rng(1)
    fs, seconds = 1250.0, 400
    x = rng.normal(size=int(fs * seconds))
    t = np.arange(len(x)) / fs
    centres = np.arange(2.0, seconds - 2.0, 4.0)
    lengths = np.where(np.arange(len(centres)) % 4 == 0, 0.2, 0.05)  # every fourth burst lasts 200 ms
    for c, length in zip(centres, lengths):
        m = np.abs(t - c) < length / 2
        x[m] += 4.0 * np.sin(2 * np.pi * 150.0 * t[m])
    z = smoothed_power(x, fs)
    z = (z - z.mean()) / z.std()
    for max_ms, expected in ((100.0, int((lengths < 0.1).sum())), (500.0, len(centres))):
        starts, stops, peaks, value = find_ripples(z, fs, max_ms)
        found = t[peaks]
        hit = np.array([np.abs(centres - f).min() < 0.11 for f in found])
        print(f"max {max_ms:.0f} ms: {len(found)} found, {expected} expected, {int(hit.sum())} at a burst, "
              f"median duration {np.median((stops - starts) / fs * 1000):.0f} ms")
        assert hit.all() and abs(len(found) - expected) <= 0.02 * expected, "detector self-test failed"
    # merging: two runs 10 ms apart are one ripple
    z2 = np.zeros(2000)
    z2[100:130] = 6.0
    z2[142:170] = 3.0
    starts, stops, peaks, value = find_ripples(z2, fs, 100.0)
    assert list(starts) == [100] and list(stops) == [170] and 100 <= peaks[0] < 130, "merging failed"
    # a run that reaches the end of the trace does not disturb the last complete ripple
    z3 = np.zeros(2000)
    z3[100:140] = 6.0
    z3[1900:] = 50.0
    starts, stops, peaks, value = find_ripples(z3, fs, 100.0)
    assert list(starts) == [100] and list(stops) == [140] and value[0] == 6.0, "incomplete run mishandled"
    # thinning: of peaks closer than the window the higher is kept
    keep = thin(np.array([1.00, 1.05, 1.20, 1.26, 2.00]), np.array([5.0, 9.0, 7.0, 6.0, 5.5]), 0.1)
    assert list(keep) == [False, True, True, False, True], keep
    # intervals, counts and gains
    assert list(interval_index(np.array([0.5, 1.5, 2.5]), np.array([1.0, 0.0]), np.array([2.0, 1.0]))) == [1, 0, -1]
    assert list(count_in(np.array([0.1, 0.2, 0.9]), np.array([0.0, 0.5]), np.array([0.3, 1.0]))) == [2, 1]
    g = firing_gain([np.array([0.05, 0.15, 5.0, 9.0])], np.array([0.0]), np.array([0.2]), np.array([0.0]), np.array([10.0]))
    assert abs(g["gain"] - (2 / 0.2) / (2 / 9.8)) < 0.01, g
    # the standard error of a trio term at participation 0.2 and 3,000 ripples per epoch is 0.403
    out = term_census(np.full(3, 0.2), np.full(3, 0.2), 3000, 3000, 3)
    assert abs(out["se_quantiles_10_25_50_75_90"][2] - 0.4034) < 2e-3, out
    assert abs(out["pooled_se_disjoint_terms"] - 0.4034) < 2e-3 and out["disjoint_terms"] == 1, out
    print("self-test passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        result = census(Path(sys.argv[1]), Path(sys.argv[2]))
        print(json.dumps(dict(file=result["file"], seconds=result["seconds"],
                              channels={s: [h["channel"], h["check"]] for s, h in result["hemispheres"].items()},
                              nrem_ripples={s: {e: h["max100ms"][e]["nrem_ripples"] for e in ("PRE", "POST")}
                                            for s, h in result["hemispheres"].items()})))
