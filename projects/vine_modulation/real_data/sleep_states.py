"""A sleep-state rule from the theta and delta bands of the pyramidal-layer field potential, for the second cohort.

Why.  The second hippocampal cohort (E64, `dryad_cohort.py`) has no sleep scoring; its events are the ripples of
still periods in the rest pot, before and after the tracks.  Its three-way difference rises over the session
 and one reading is that the animals pass from quiet waking into sleep.  The
first cohort (DANDI 000044) carries the lab's five-state scoring, which the lab made from the pyramidal-layer
spectrum, "in particular the theta (5-10 Hz) to delta (1-4 Hz) ratio", and from movement and EMG.  A rule built
from the same two bands is validated here against those labels at the times of ripples, and frozen before it is
applied to the second cohort.

Features, the same in both cohorts.  The channel used for ripples (first cohort: the checked channel of each
hemisphere; second cohort: the lab's `best_ripple` channel) is averaged in blocks down to 125 samples a second.
In windows of 4 s stepped by 1 s (Hann taper, mean removed) the periodogram gives the mean power at 1 to 4 Hz
(delta) and at 5 to 10 Hz (theta).  Two features, each passed through a running median of nine windows:
  ratio   log(theta / delta)
  level   log(delta), minus its mean and divided by its standard deviation over the windows of the two rest epochs
          of the recording (PRE and POST; all their windows, whatever the animal does)
An event (the 100 ms around a ripple's peak) takes the values of the window whose centre is nearest.

    sleep_states.py features_nwb FILE.nwb CENSUS_FOLDER OUT.npz        first cohort: features, the lab's states, events
    sleep_states.py features_dryad DATA SESSION CENSUS_FOLDER OUT.npz  second cohort: features, stillness, events
    sleep_states.py validate OUT.json DEVELOPMENT FEATURES.npz [...]   first cohort: candidate rules against the lab

Candidate rules for "NREM-like" (the lab's NREM and its Intermediate sub-type against its Drowsy, Wake and REM),
judged at the ripple events of the two sleep epochs.  The constant of each is chosen on the development animal
(Gatsby) to maximise the balanced accuracy and then applied to the three other animals:
  ratio   ratio below a constant
  level   level above a constant

FIRST VALIDATION (24 November 2026, `hippocampus/sleep_nwb_v1`, the ratio only).  The ratio does not tell the
lab's NREM from its Drowsy state at ripples: both have a low ratio.  With its constant from Gatsby (-1.1) the
balanced accuracy was 0.56 (Achilles), 0.63 (Buddy) and 0.73 (Cicero); a threshold set by each session's own
split called 82 to 95% of the Drowsy ripples NREM-like.  The lab's description of Drowsy is "low overall spectral
power", which is why the level was added; looked at on the saved features before this file was changed, its
constant from Gatsby (+0.60) gave 0.71, 0.82 and 0.83.  Two other ways of scaling the delta power were worse
(relative to its median on the maze: 0.64, 0.79, 0.82; the session's own split: 0.60, 0.64, 0.71).

THE RULE (fixed with `sleep_nwb_v2`, before any state of the second cohort was computed):
  a ripple is NREM-like when its level is above LEVEL_NREM = 0.60.
Validation on the three held-out animals showed that this threshold classifies sleep state imperfectly.  The analyses of the second cohort therefore also compare ripples matched on the
level itself, which needs no threshold.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TARGET_RATE, WINDOW_S, STEP_S, MEDIAN = 125.0, 4.0, 1.0, 9
DELTA, THETA = (1.0, 4.0), (5.0, 10.0)
LEVEL_NREM = 0.60       # the rule: a ripple is NREM-like when its level is above this (chosen on Gatsby)
NREM_LIKE = ("NREM", "Intermediate")
STATES = ("NREM", "Intermediate", "Drowsy", "Wake", "REM")


def band_feature(read, n, fs, first_time, batch=20000):
    """Centres of the windows; log(theta / delta) and log(delta), each after the running median of nine windows."""
    from scipy.signal import medfilt
    q = max(1, int(round(fs / TARGET_RATE)))
    rate = fs / q
    usable = n - n % q
    piece = 2_000_000 - 2_000_000 % q
    low = np.concatenate([read(i, min(usable, i + piece)).astype(np.float64).reshape(-1, q).mean(1) for i in range(0, usable, piece)])
    window, step = int(round(WINDOW_S * rate)), int(round(STEP_S * rate))
    count = (len(low) - window) // step + 1
    taper = np.hanning(window)
    frequency = np.fft.rfftfreq(window, 1 / rate)
    in_delta, in_theta = (frequency >= DELTA[0]) & (frequency <= DELTA[1]), (frequency >= THETA[0]) & (frequency <= THETA[1])
    delta, theta = np.empty(count), np.empty(count)
    for start in range(0, count, batch):
        index = np.arange(window)[None, :] + step * np.arange(start, min(count, start + batch))[:, None]
        segment = low[index]
        segment = (segment - segment.mean(1, keepdims=True)) * taper
        power = np.abs(np.fft.rfft(segment, axis=1)) ** 2
        delta[start:start + len(index)] = power[:, in_delta].mean(1)
        theta[start:start + len(index)] = power[:, in_theta].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(theta / delta)
    ratio = np.where(np.isfinite(ratio), ratio, np.nanmedian(ratio[np.isfinite(ratio)]))
    centre = first_time + (step * np.arange(count) + window / 2) / rate
    return centre, medfilt(ratio, MEDIAN), medfilt(np.log(np.maximum(delta, 1e-300)), MEDIAN)


def level_of(delta, rest):
    """The smoothed log delta power in units of its spread over the windows of the rest epochs."""
    return (delta - delta[rest].mean()) / delta[rest].std()


def nearest(centre, times):
    index = np.clip(np.searchsorted(centre, times), 1, len(centre) - 1)
    return np.where(np.abs(centre[index - 1] - times) <= np.abs(centre[index] - times), index - 1, index)


def otsu(values, bins=128):
    """The threshold that maximises the between-class variance of the values (Otsu), over their central 99%."""
    low, high = np.quantile(values, [0.005, 0.995])
    counts, edges = np.histogram(np.clip(values, low, high), bins=bins, range=(low, high))
    mid = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(counts)
    total = weight[-1]
    mean = np.cumsum(counts * mid)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean[-1] * weight / total - mean) ** 2 / (weight * (total - weight))
    between[~np.isfinite(between)] = -1
    return float(edges[int(np.argmax(between)) + 1])


def strings(dataset):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in dataset[:]])


def features_nwb(path, census, out):
    import h5py
    report = json.loads((Path(census) / "census.json").read_text())
    ripples = np.load(Path(census) / "ripples.npz")
    saved = {}
    with h5py.File(path, "r") as nwb:
        series = nwb["processing/ecephys/LFP/lfp"]
        data, fs = series["data"], float(series["starting_time"].attrs["rate"])
        first = float(series["starting_time"][()])
        states = nwb["processing/behavior/sleep_states"]
        label, begin, end = strings(states["label"]), states["start_time"][:], states["stop_time"][:]
        epochs = nwb["intervals/epochs"]
        epoch_label, epoch_begin, epoch_end = strings(epochs["label"]), epochs["start_time"][:], epochs["stop_time"][:]
        for side, h in report["hemispheres"].items():
            if h.get("check") != "passed":
                continue
            channel = int(h["channel"])
            centre, ratio, delta = band_feature(lambda i, j: data[i:j, channel], data.shape[0], fs, first)
            order = np.argsort(begin)
            k = np.clip(np.searchsorted(begin[order], centre, side="right") - 1, 0, len(order) - 1)
            state = np.where((centre >= begin[order][k]) & (centre < end[order][k]), label[order][k], "none")
            epoch = np.full(len(centre), "none", dtype="U8")
            for name, a, b in zip(epoch_label, epoch_begin, epoch_end):
                epoch[(centre >= a) & (centre < b)] = name
            events = ripples[f"{side}_fixed100ms"]
            at = nearest(centre, events[:, 2])
            level = level_of(delta, np.isin(epoch, ("PRE", "POST")))
            saved.update({f"{side}_centre": centre, f"{side}_ratio": ratio, f"{side}_level": level, f"{side}_state": state, f"{side}_epoch": epoch,
                          f"{side}_event_peak": events[:, 2], f"{side}_event_state": ripples[f"{side}_fixed100ms_state"],
                          f"{side}_event_epoch": ripples[f"{side}_fixed100ms_epoch"], f"{side}_event_ratio": ratio[at], f"{side}_event_level": level[at],
                          f"{side}_channel": np.array([channel])})
    saved["epoch_names"] = epoch_label
    np.savez_compressed(out, **saved)
    return sorted({k.split("_")[0] for k in saved if k.endswith("_centre")})


def features_dryad(data, session, census, out):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dryad_cohort", HERE / "dryad_cohort.py")
    cohort = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cohort)
    label, field_label, _ = cohort.SESSIONS[session]
    behaviour = cohort.load_behaviour(data, label)
    trace = cohort.Trace(data, field_label)
    centre, ratio, delta = band_feature(trace.read, trace.n, trace.fs, trace.first)
    trace.close()
    t = behaviour["t"]
    still = behaviour["in_box"] & (behaviour["speed"] < cohort.STILL)
    step = float(t[1] - t[0])
    cumulative = np.concatenate(([0], np.cumsum(still)))
    a = np.clip(np.searchsorted(t, centre - WINDOW_S / 2), 0, len(t))
    b = np.clip(np.searchsorted(t, centre + WINDOW_S / 2), 0, len(t))
    still_share = np.where(b > a, (cumulative[b] - cumulative[a]) / np.maximum(b - a, 1), 0.0)
    moving = np.flatnonzero(~still)                      # camera samples at which the animal is not still in the pot
    epoch = np.full(len(centre), "none", dtype="U8")
    for name, (start, stop) in behaviour["epochs"].items():
        epoch[(centre >= start) & (centre <= stop)] = name
    ripples = np.load(Path(census) / "ripples.npz")
    peak = ripples["peak"]
    at = nearest(centre, peak)
    sample = np.clip(np.searchsorted(t, peak), 0, len(t) - 1)
    last = np.searchsorted(moving, sample) - 1           # the last moving sample before each ripple
    since = np.where(last >= 0, (sample - moving[np.clip(last, 0, max(len(moving) - 1, 0))]) * step, sample * step) if len(moving) else sample * step
    level = level_of(delta, np.isin(epoch, ("PRE", "POST")))
    np.savez_compressed(out, centre=centre, ratio=ratio, level=level, still_share=still_share, epoch=epoch, event_peak=peak,
                        event_epoch=ripples["epoch"], event_ratio=ratio[at], event_level=level[at], event_still_seconds=since,
                        rate=np.array([trace.fs]))
    return session


def validate(out, development, files):
    sessions = {}
    for path in files:
        data = np.load(path)
        name = Path(path).stem
        for side in sorted({k.split("_")[0] for k in data.files if k.endswith("_centre")}):
            keep = np.isin(data[f"{side}_event_epoch"], ("PRE", "POST")) & np.isin(data[f"{side}_event_state"], STATES)
            sessions[f"{name}|{side}"] = dict(animal=name.split("-")[0], state=data[f"{side}_event_state"][keep], epoch=data[f"{side}_event_epoch"][keep],
                                              ratio=-data[f"{side}_event_ratio"][keep], level=data[f"{side}_event_level"][keep])

    def score(called, state):
        truth = np.isin(state, NREM_LIKE)
        sensitivity = float(called[truth].mean()) if truth.any() else float("nan")
        specificity = float((~called[~truth]).mean()) if (~truth).any() else float("nan")
        return dict(events=int(len(state)), sensitivity=sensitivity, specificity=specificity, balanced=0.5 * (sensitivity + specificity),
                    called_nrem_like={s: (float(called[state == s].mean()) if (state == s).any() else None) for s in STATES},
                    events_by_state={s: int((state == s).sum()) for s in STATES})

    def area(positive, negative):
        values = np.r_[positive, negative]
        rank = np.argsort(np.argsort(values)) + 1.0
        return float((rank[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))

    report = dict(development=development, rule=dict(feature="level", constant=LEVEL_NREM), constants={}, animals={}, sessions={})
    grid = np.linspace(-3.0, 3.0, 241)
    dev = [v for v in sessions.values() if v["animal"] == development]
    for feature in ("ratio", "level"):        # the ratio enters negated, so that both features call NREM-like above a constant
        balanced = [np.mean([score(v[feature] > c, v["state"])["balanced"] for v in dev]) for c in grid]
        report["constants"][feature] = float(grid[int(np.argmax(balanced))])
    for key, v in sessions.items():
        truth = np.isin(v["state"], NREM_LIKE)
        drowsy = v["state"] == "Drowsy"
        report["sessions"][key] = dict(animal=v["animal"], rule=score(v["level"] > LEVEL_NREM, v["state"]),
                                       area_nrem_like_against_drowsy={f: (area(v[f][truth], v[f][drowsy]) if drowsy.sum() > 20 else None) for f in ("ratio", "level")},
                                       events_by_epoch={e: int((v["epoch"] == e).sum()) for e in ("PRE", "POST")})
    for animal in sorted({v["animal"] for v in sessions.values()}):
        group = [v for v in sessions.values() if v["animal"] == animal]
        state = np.concatenate([v["state"] for v in group])
        entry = dict(hemispheres=len(group), rule=score(np.concatenate([v["level"] > LEVEL_NREM for v in group]), state))
        for feature in ("ratio", "level"):
            entry[feature] = score(np.concatenate([v[feature] > report["constants"][feature] for v in group]), state)
        report["animals"][animal] = entry
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "features_nwb":
        print(Path(sys.argv[2]).name, features_nwb(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif mode == "features_dryad":
        print(features_dryad(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
    elif mode == "validate":
        r = validate(sys.argv[2], sys.argv[3], sys.argv[4:])
        print("constants chosen on", r["development"], ":", {k: round(v, 3) for k, v in r["constants"].items()}, "(the ratio's is of minus the ratio); the rule:", r["rule"])
        for animal, v in r["animals"].items():
            for rule in ("ratio", "level", "rule"):
                s = v[rule]
                print(f"{animal:9s} {rule:6s} events {s['events']:6d} sensitivity {s['sensitivity']:.3f} specificity {s['specificity']:.3f} balanced {s['balanced']:.3f} | called NREM-like: "
                      + " ".join(f"{k} {('%.2f' % x) if x is not None else '-'} ({s['events_by_state'][k]})" for k, x in s["called_nrem_like"].items()))
    else:
        raise SystemExit(__doc__)
