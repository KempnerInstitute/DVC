"""E70: a replay model that uses the order of spikes within a ripple, judged first on held-out prediction.

In Achilles the original track-stretch model accounted for only part of the trio change. This 24 November protocol evaluates whether a trajectory model predicts cell participation better on held-out observations. Both cohorts use the same specification and information from cells outside each tested trio.

The sequence law.  A ripple (the 100 ms around its peak) replays a trajectory or nothing, at a strength g in
{1/4, 1/2, 1, 2, 4}.  A trajectory runs along one running direction of one track at constant speed across the
window, from 8-cm bin s0 to bin s0 + shift, shift in {0, +-2, +-4, +-8, +-12} bins (0 to 0.96 m in 100 ms; on a
circular track it wraps around).  Place cell i spikes as an inhomogeneous Poisson process with rate
g * c_i * r_i(position at time t), r_i its rate map from running (the 4-cm maps of `fields`, joined in pairs), c_i
a factor of the cell; for no location, r_i is its mean rate over all maps, constant in time.  The window is cut
into five steps of 20 ms.  What the law describes of a ripple is, for every place cell, whether it fired and in
which step its first spike fell: the probability that the first spike falls in step k is taken as
rate(step k) * exp(-integrated rate up to the middle of step k), and that of no spike exp(-integrated rate over
the window).  Weights of the trajectories and strengths (one set for the trajectories, one for no location) and
the factors are fitted by EM accelerated as in `dryad_posthoc_content.py`, one law per rest.  Its probability that
a cell takes part, 1 - exp(-g c_i * integrated r_i), is that of the stretch law with the stretch covered by the
trajectory; what the order adds is a sharper estimate, from the other cells, of where the ripple is.

    replay_sequences.py extract COHORT DATA CENSUS_OR_FIELDS CONTENT_FOLDER OUT_FOLDER NAME
    replay_sequences.py predict CONTENT_FOLDER FIRST_FOLDER OUT_FOLDER NAME [CENSUS_ROOT]
    replay_sequences.py summary OUT_FOLDER
    replay_sequences.py diagnose CONTENT_FOLDER FIRST_FOLDER PREDICT_FOLDER OUT_FOLDER NAME [CENSUS_ROOT|-] [FIELDS_ROOT]

`diagnose` (post hoc, added on 24 November after `predict` had shown the sequence law predicting worse than the
stretch law in every group): the same trajectories and strengths, but each ripple described only by which cells
fired (a stretch law whose stretches are the trajectories' mean rates), fitted and scored on the same folds and
ripples.  Read beside the scores of `predict`: stretch minus this law is what the trajectory states cost or gain,
this law minus the sequence law what the first-spike likelihood costs or gains.

`extract` (COHORT = second or first): the step of the first spike of every place cell in every ripple of a
recording stored by `dryad_posthoc_content.py fit` or `hippocampal_posthoc_content.py fit` (-1 when the cell did
not fire), checked against the stored participation.

`predict`: held-out prediction.  Ripples of the two rests in time order, the blocks and folds of E67 and E69.  For
each fold left out, a sequence law and a stretch law (section 66) are fitted to the other folds' ripples of each
rest; then, for every ripple of the fold left out and every place cell, each law gives the probability that the
cell fired, from the OTHER place cells of that ripple: the sequence law from their first-spike steps, the stretch
law from whether they fired.  Score of a ripple: the sum over its place cells of the log probability of what the
cell did; the difference sequence minus stretch, summed over ripples, with blocks as units within each rest.

Expectations and the decision, written before anything was run.
  (p1) Second cohort, nine sessions: the sequence law predicts better than the stretch law (difference positive
       with z of at least 1.96).
  (p2) Achilles, four hemispheres: the same.
  The sequence law is used for the three-way change measured beyond replay (the `adjusted` estimate of sections 66
  and 68, the trio's patterns predicted from the other place cells, now from their first-spike steps) only in a
  cohort where it predicts better; where it does not, the stretch law's result stands and the sequence law is
  reported as not better.  The same specification is used in both cohorts.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

COARSE, STEPS, SHIFTS, WINDOW = 2, 5, (0, 2, -2, 4, -4, 8, -8, 12, -12), 0.1
BLOCKS, FOLDS = 10, 5
GROUPS = {"cohort": ("RAT1_SESS1", "RAT1or4_SESS", "RAT2_SESS1", "RAT2_SESS2", "RAT3_SESS1", "RAT3_SESS2", "RAT4_SESS2", "RAT5_SESS1", "RAT5_SESS2"),
          "rat2": ("RAT2_SESS1", "RAT2_SESS2"), "rat3": ("RAT3_SESS1", "RAT3_SESS2"), "rat5": ("RAT5_SESS1", "RAT5_SESS2"),
          "achilles": ("Achilles-10252013_left", "Achilles-10252013_right", "Achilles-11012013_left", "Achilles-11012013_right"),
          "cicero": ("Cicero-09012014_left", "Cicero-09012014_right", "Cicero-09102014_left", "Cicero-09102014_right", "Cicero-09172014_left",
                     "Cicero-09172014_right"),
          "gatsby": ("Gatsby-08022013_left", "Gatsby-08022013_right", "Gatsby-08282013_left")}


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


content = module("dryad_posthoc_content")
GAINS = content.GAINS


def maps_of(rates, kinds):
    """The 4-cm maps of a stored recording (the runs of single-bin states, map by map), from its stretch states."""
    kinds = np.asarray(kinds)
    maps, run = [], []
    for column, kind in enumerate(kinds):
        if kind == 0:
            run.append(column)
        elif run:
            maps.append(np.asarray(rates)[:, run])
            run = []
    return maps


def coarse(block):
    """A map at 8 cm: consecutive pairs of 4-cm bins averaged (a last odd bin alone)."""
    bins = block.shape[1]
    pairs = [block[:, k:k + COARSE].mean(1) for k in range(0, bins, COARSE)]
    return np.stack(pairs, axis=1)


def trajectory_rates(maps, circular):
    """Rate of every cell at the middle of every step of every trajectory: (trajectories, cells, STEPS), the
    state without a place last; and for each trajectory its map and shift (-1, 0 for no location)."""
    out, labels = [], []
    middle = (np.arange(STEPS) + 0.5) / STEPS
    for m, block in enumerate(maps):
        fine = coarse(np.asarray(block, dtype=np.float64))
        bins = fine.shape[1]
        for start in range(bins):
            for shift in SHIFTS:
                end = start + shift
                if not circular and not 0 <= end < bins:
                    continue
                position = start + shift * middle
                low = np.floor(position).astype(int)
                frac = position - low
                if circular:
                    a, b = fine[:, low % bins], fine[:, (low + 1) % bins]
                else:
                    a, b = fine[:, np.clip(low, 0, bins - 1)], fine[:, np.clip(low + 1, 0, bins - 1)]
                out.append(a * (1 - frac) + b * frac)                                  # (cells, STEPS)
                labels.append((m, shift))
    every = np.concatenate([np.asarray(b, dtype=np.float64) for b in maps], axis=1).mean(1)
    out.append(np.repeat(every[:, None], STEPS, axis=1))
    labels.append((-1, 0))
    return np.stack(out), np.array(labels)


def first_steps(spikes, begin, end):
    """Step of the first spike of each cell in each window (-1 if none): (windows, cells), and its offset in seconds."""
    step = np.full((len(begin), len(spikes)), -1, dtype=np.int8)
    offset = np.full((len(begin), len(spikes)), np.nan, dtype=np.float32)
    for c, s in enumerate(spikes):
        s = np.asarray(s)
        index = np.searchsorted(s, begin, side="left")
        inside = index < len(s)
        first = np.full(len(begin), np.inf)
        first[inside] = s[index[inside]]
        fired = first < end
        offset[fired, c] = first[fired] - begin[fired]
        step[fired, c] = np.clip(((first[fired] - begin[fired]) / (end[fired] - begin[fired]) * STEPS).astype(int), 0, STEPS - 1)
    return step, offset


class Sequence:
    """A sequence law fitted to the first-spike steps given (see the header)."""

    def __init__(self, step, rates, cycles=content.CYCLES):
        step = np.asarray(step)
        self.rates = np.asarray(rates, dtype=np.float64)                                  # (T, cells, STEPS)
        dt = WINDOW / STEPS
        self.full = self.rates.sum(2) * dt                                                # (T, cells)
        self.until = np.cumsum(self.rates, axis=2) * dt - 0.5 * self.rates * dt          # (T, cells, STEPS)
        self.log_rate = np.log(np.maximum(self.rates, 1e-6))
        self.step = step
        self.active = [(step == k).astype(np.float64) for k in range(STEPS)]
        self.silent = (step < 0).astype(np.float64)
        self.fired = (step >= 0).astype(np.float64)
        self.fixed = sum(a @ self.log_rate[:, :, k].T for k, a in enumerate(self.active))  # (events, T)
        self.n_active = self.fired.sum(1)
        observed = max(float(self.fired.mean()), 1e-4)
        low, high = 1e-4, 1e4
        for _ in range(60):
            middle = np.sqrt(low * high)
            low, high = (middle, high) if (-np.expm1(-middle * self.full[-1])).mean() < observed else (low, middle)
        self.factor = np.full(self.rates.shape[1], np.sqrt(low * high))
        self.weight = np.full((len(GAINS), self.rates.shape[0]), 1.0 / (len(GAINS) * self.rates.shape[0]))
        self.cycles, previous = 0, -np.inf
        while self.cycles < cycles:
            first = self.parameters()
            value = self.round()
            second = self.parameters()
            after_one = self.round()
            third = self.parameters()
            step_, bend = second - first, third - 2 * second + first
            length = -max(1.0, float(np.linalg.norm(step_) / max(np.linalg.norm(bend), 1e-300)))
            self.place(first - 2 * length * step_ + length ** 2 * bend)
            reached = self.round()
            if not np.isfinite(reached) or reached < after_one:
                self.place(third)
            self.cycles += 1
            if value - previous < content.TOLERANCE:
                break
            previous = value

    def parameters(self):
        place = self.weight[:, :-1]
        return np.concatenate([np.log(self.factor), place.sum(1), place.sum(0), self.weight[:, -1]])

    def place(self, parameters):
        cells, gains = len(self.factor), len(GAINS)
        self.factor = np.exp(np.clip(parameters[:cells], np.log(1e-6), np.log(1e6)))
        rest = np.maximum(parameters[cells:], content.TINY)
        by_gain, by_state, none = rest[:gains], rest[gains:-gains], rest[-gains:]
        weight = np.empty_like(self.weight)
        weight[:, :-1] = np.outer(by_gain, by_state) / by_state.sum()
        weight[:, -1] = none
        self.weight = weight / weight.sum()

    def intensity(self, active=None, silent=None):
        """Integrated intensity at factor c and strength 1, up to each cell's first spike or over the window: (events, T)."""
        active = self.active if active is None else active
        silent = self.silent if silent is None else silent
        total = silent @ (self.factor[None, :] * self.full).T
        for k, a in enumerate(active):
            total += a @ (self.factor[None, :] * self.until[:, :, k]).T
        return total

    def likelihood(self, step=None):
        """Log probability of every event at every strength and trajectory: (gains, events, T)."""
        if step is None:
            fixed, n_active, fired, intensity = self.fixed, self.n_active, self.fired, self.intensity()
        else:
            step = np.asarray(step)
            active = [(step == k).astype(np.float64) for k in range(STEPS)]
            fixed = sum(a @ self.log_rate[:, :, k].T for k, a in enumerate(active))
            fired = (step >= 0).astype(np.float64)
            n_active = fired.sum(1)
            intensity = self.intensity(active, (step < 0).astype(np.float64))
        log_factor = fired @ np.log(self.factor)
        return np.stack([fixed + (n_active * np.log(g) + log_factor)[:, None] - g * intensity for g in GAINS])

    def joint(self, log_likelihood):
        return log_likelihood + np.log(np.maximum(self.weight, 1e-300))[:, None, :]

    def round(self):
        joint = self.joint(self.likelihood())
        top = joint.max(axis=(0, 2), keepdims=True)
        post = np.exp(joint - top)
        norm = post.sum(axis=(0, 2), keepdims=True)
        post /= norm
        total = post.sum(1)
        weighted = sum(g * post[k] for k, g in enumerate(GAINS))                       # (events, T)
        expected = (self.silent.T @ weighted * self.full.T).sum(1)
        for k, a in enumerate(self.active):
            expected += (a.T @ weighted * self.until[:, :, k].T).sum(1)
        self.weight = content.Content.weights_of(total)
        self.factor = np.clip(self.fired.sum(0) / np.maximum(expected, 1e-12), 1e-6, 1e6)
        return float((top + np.log(norm)).sum())

    def evidence(self, step=None):
        joint = self.joint(self.likelihood(step))
        top = joint.max(axis=(0, 2))
        return top + np.log(np.exp(joint - top[None, :, None]).sum(axis=(0, 2)))

    def participation(self):
        """Probability that each cell takes part, at each strength and trajectory: (gains, T, cells)."""
        return np.stack([-np.expm1(-g * self.factor[None, :] * self.full) for g in GAINS])

    def predicted(self, step):
        """Probability that each cell fired, from the other cells' first-spike steps: (events, cells)."""
        step = np.asarray(step)
        total = self.joint(self.likelihood(step))                                       # (gains, events, T)
        p = np.clip(self.participation(), 1e-9, 1 - 1e-9)
        out = np.empty(step.shape)
        for i in range(step.shape[1]):
            fired = step[:, i] >= 0
            k = np.clip(step[:, i], 0, STEPS - 1)
            own = np.empty_like(total)
            for gi, g in enumerate(GAINS):
                spike = np.log(g * self.factor[i]) + self.log_rate[:, i, :].T[k] - g * self.factor[i] * self.until[:, i, :].T[k]   # (events, T)
                none = -g * self.factor[i] * self.full[:, i][None, :]
                own[gi] = np.where(fired[:, None], spike, none)
            rest = total - own
            weight = np.exp(rest - rest.max(axis=(0, 2), keepdims=True))
            weight /= weight.sum(axis=(0, 2), keepdims=True)
            out[:, i] = np.einsum("get,gt->e", weight, p[:, :, i])
        return out


def stretch_predicted(law, x):
    """Probability that each cell fired, from whether the other cells fired, under a stretch law: (events, cells)."""
    members = np.arange(x.shape[1])[:, None]
    return law.predicted(x, members)[:, :, 1]


def blocks_of(epoch):
    block = np.full(len(epoch), -1)
    for k, rest in enumerate(("PRE", "POST")):
        rows = np.flatnonzero(epoch == rest)
        block[rows] = k * BLOCKS + (np.arange(len(rows)) * BLOCKS) // max(len(rows), 1)
    return block, np.where(block >= 0, block % FOLDS, -1)


def circular_of(name, fields_root):
    session = name.rsplit("_", 1)[0]
    path = Path(fields_root) / session / "fields.json"
    return path.exists() and json.loads(path.read_text())["track"] == "circular"


def extract(cohort, data, census_or_fields, content_folder, out, name):
    """First-spike steps of the place cells of a stored recording, in the order of its stored ripples."""
    d, _ = content.stored(content_folder, name)
    if cohort == "second":
        dryad = module("dryad_cohort")
        ripples = np.load(Path(census_or_fields) / name / "ripples.npz")
        kept = [int(u) for u in ripples["kept"]]
        spikes = dryad.load_spikes(data, dryad.SESSIONS[name][0])[0]
        peak = ripples["peak"]
        begin, end = peak - dryad.WINDOW / 2, peak + dryad.WINDOW / 2
        trains = [spikes[kept[c]] for c in d["cells"]]
    else:
        e62 = module("hippocampal_study")
        session, side = name.rsplit("_", 1)
        census_root, fields_root = census_or_fields.split(",") if "," in census_or_fields else (census_or_fields, None)
        animal = session.split("-")[0]
        _, found, _ = e62.load(Path(data) / f"sub-{animal}_ses-{session}_behavior+ecephys.nwb", Path(census_root) / session, Path(fields_root) / session)
        h = found[side]
        begin, end = h["begin"], h["end"]
        trains = [h["spikes"][c] for c in d["cells"]]
    step, offset = first_steps(trains, np.asarray(begin, dtype=float), np.asarray(end, dtype=float))
    assert step.shape[0] == d["x"].shape[0] and np.array_equal(step >= 0, d["x"][:, d["cells"]] > 0), "first spikes disagree with the stored participation"
    folder = Path(out) / "first"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}.npz", step=step, offset=offset)
    return dict(name=name, ripples=int(step.shape[0]), place_cells=int(step.shape[1]), fired=float((step >= 0).mean()),
                steps=[int((step == k).sum()) for k in range(STEPS)])


def recording(content_folder, first_folder, name, census_root=None, fields_root=None):
    d, _ = content.stored(content_folder, name)
    peak_file = Path(content_folder) / "sessions" / f"{name}_peak.npy"
    peak = np.load(peak_file) if peak_file.exists() else np.load(Path(census_root) / name / "ripples.npz")["peak"]
    keep = np.flatnonzero((d["epoch"] == "PRE") | (d["epoch"] == "POST"))
    keep = keep[np.argsort(peak[keep], kind="stable")]
    step = np.load(Path(first_folder) / "first" / f"{name}.npz")["step"][keep]
    circular = circular_of(name, fields_root) if fields_root else False
    rates, labels = trajectory_rates(maps_of(d["rates"], d["kinds"]), circular)
    return dict(name=name, step=step, x=(step >= 0).astype(np.int8), epoch=d["epoch"][keep], rates=rates, labels=labels,
                stretch_rates=d["rates"], window=d["window"], circular=circular)


def predict(content_folder, first_folder, out, name, census_root=None, fields_root=None):
    started = time.time()
    rec = recording(content_folder, first_folder, name, census_root, fields_root)
    step, x, epoch = rec["step"], rec["x"], rec["epoch"]
    block, fold = blocks_of(epoch)
    score = {"sequence": np.zeros(len(x)), "stretch": np.zeros(len(x))}
    laws = []
    for left in range(FOLDS):
        test = np.flatnonzero(fold == left)
        for rest in ("PRE", "POST"):
            train = np.flatnonzero((fold != left) & (epoch == rest))
            rows = test[epoch[test] == rest]
            sequence = Sequence(step[train], rec["rates"])
            stretch = content.Content(x[train], rec["stretch_rates"], rec["window"])
            for model, p in (("sequence", sequence.predicted(step[rows])), ("stretch", stretch_predicted(stretch, x[rows]))):
                p = np.clip(p, 1e-9, 1 - 1e-9)
                score[model][rows] = (x[rows] * np.log(p) + (1 - x[rows]) * np.log1p(-p)).sum(1)
            share = sequence.weight.sum(0)
            laws.append(dict(fold=left, rest=rest, cycles=[sequence.cycles, stretch.cycles],
                             sequence_moving=float(share[:-1][rec["labels"][:-1, 1] != 0].sum()), sequence_still=float(share[:-1][rec["labels"][:-1, 1] == 0].sum()),
                             sequence_none=float(share[-1]), stretch_none=float(stretch.weight[:, -1].sum())))
    difference = score["sequence"] - score["stretch"]
    sums = np.array([difference[block == b].sum() for b in range(2 * BLOCKS)])
    variance = sum(BLOCKS / (BLOCKS - 1) * float(((part - part.mean()) ** 2).sum()) for part in (sums[:BLOCKS], sums[BLOCKS:]))
    report = dict(name=name, place_cells=int(x.shape[1]), ripples=[int((epoch == r).sum()) for r in ("PRE", "POST")], trajectories=int(len(rec["rates"])),
                  circular=rec["circular"], difference=float(difference.sum()), variance=variance, before=float(sums[:BLOCKS].sum()), after=float(sums[BLOCKS:].sum()),
                  per_ripple=dict(sequence=float(score["sequence"].mean()), stretch=float(score["stretch"].mean())), laws=laws, seconds=round(time.time() - started, 1))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", block=block, **score)
    (out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def diagnose(content_folder, first_folder, predict_folder, out, name, census_root=None, fields_root=None):
    """The trajectory states without spike timing, on the folds and ripples of `predict` (post hoc; see the header)."""
    started = time.time()
    rec = recording(content_folder, first_folder, name, census_root, fields_root)
    x, epoch = rec["x"], rec["epoch"]
    block, fold = blocks_of(epoch)
    mean_rates = rec["rates"].mean(2).T                                                  # (cells, trajectories)
    score = np.zeros(len(x))
    for left in range(FOLDS):
        test = np.flatnonzero(fold == left)
        for rest in ("PRE", "POST"):
            train = np.flatnonzero((fold != left) & (epoch == rest))
            rows = test[epoch[test] == rest]
            law = content.Content(x[train], mean_rates, WINDOW)
            p = np.clip(stretch_predicted(law, x[rows]), 1e-9, 1 - 1e-9)
            score[rows] = (x[rows] * np.log(p) + (1 - x[rows]) * np.log1p(-p)).sum(1)
    stored = np.load(Path(predict_folder) / f"{name}.npz")
    assert np.array_equal(stored["block"], block)

    def compared(first, second):
        difference = first - second
        sums = np.array([difference[block == b].sum() for b in range(2 * BLOCKS)])
        variance = sum(BLOCKS / (BLOCKS - 1) * float(((part - part.mean()) ** 2).sum()) for part in (sums[:BLOCKS], sums[BLOCKS:]))
        return dict(difference=float(difference.sum()), variance=variance)

    report = dict(name=name, trajectory_states_minus_stretch=compared(score, stored["stretch"]), sequence_minus_trajectory_states=compared(stored["sequence"], score),
                  per_ripple=dict(trajectory_states=float(score.mean()), sequence=float(stored["sequence"].mean()), stretch=float(stored["stretch"].mean())),
                  seconds=round(time.time() - started, 1))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def summarise_diagnosis(out):
    out = Path(out)
    recordings = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.stem != "summary"}
    groups = {}
    for group, members in GROUPS.items():
        chosen = [recordings[n] for n in members if n in recordings]
        if chosen:
            groups[group] = {key: dict(difference=sum(r[key]["difference"] for r in chosen), se=float(np.sqrt(sum(r[key]["variance"] for r in chosen))),
                                       recordings_positive=int(sum(r[key]["difference"] > 0 for r in chosen)))
                             for key in ("trajectory_states_minus_stretch", "sequence_minus_trajectory_states")}
            for value in groups[group].values():
                value["z"] = value["difference"] / value["se"] if value["se"] > 0 else None
    (out / "summary.json").write_text(json.dumps(dict(groups=groups), indent=1) + "\n")
    return groups


def summarise(out):
    out = Path(out)
    recordings = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.stem != "summary"}
    groups = {}
    for group, members in GROUPS.items():
        chosen = [recordings[n] for n in members if n in recordings]
        if not chosen:
            continue
        total, variance = sum(r["difference"] for r in chosen), sum(r["variance"] for r in chosen)
        ripples = sum(sum(r["ripples"]) for r in chosen)
        groups[group] = dict(recordings=len(chosen), difference=total, se=float(np.sqrt(variance)), z=float(total / np.sqrt(variance)) if variance > 0 else None,
                             per_ripple=total / ripples, recordings_positive=int(sum(r["difference"] > 0 for r in chosen)))
    report = dict(groups=groups, recordings={n: {k: r[k] for k in ("difference", "variance", "place_cells", "ripples", "per_ripple")} for n, r in recordings.items()},
                  decision={g: bool(groups.get(g) and groups[g]["z"] is not None and groups[g]["z"] >= 1.96) for g in ("cohort", "achilles")})
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "extract":
        print(json.dumps(extract(*sys.argv[2:8])))
    elif mode == "predict":
        r = predict(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None,
                    sys.argv[7] if len(sys.argv) > 7 else None)
        print(json.dumps({k: v for k, v in r.items() if k != "laws"}, indent=1))
    elif mode == "summary":
        print(json.dumps({k: v for k, v in summarise(sys.argv[2]).items() if k != "recordings"}, indent=1))
    elif mode == "diagnose":
        print(json.dumps(diagnose(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6],
                                  sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != "-" else None, sys.argv[8] if len(sys.argv) > 8 else None)))
    elif mode == "diagnosis_summary":
        print(json.dumps(summarise_diagnosis(sys.argv[2]), indent=1))
    else:
        raise SystemExit(__doc__)
