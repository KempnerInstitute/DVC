"""E72: a state-space decoder of replay content, judged first on held-out prediction.

The constant-speed sequence model gave lower held-out scores than the track-stretch model. This 24 November protocol compares a state-space model admitting stationary, continuous and fragmented trajectories (Denovellis et al. 2021, eLife 10:e64505). We use sorted spikes and the same held-out prediction protocol as for the sequence model.

The law.  A ripple (the 100 ms around its peak) is cut into ten steps of 10 ms.  A hidden state per step is a position
(an 8-cm bin of one running direction of one track; on a circular track the bins wrap) or "no location".  From one
step to the next a position moves by a mixture of three kernels with weights fitted to the ripples:
  continuous   a step of a random walk within the same direction, standard deviation one bin (8 cm per 10 ms)
  fragmented   a jump to any position of any direction, uniformly
  stationary   no move
and a position goes to "no location", or "no location" to a position, with fitted probabilities.  The first step's
state has fitted weights.  A strength g in {1/4, 1/2, 1, 2, 4} holds for the whole ripple (fitted weights).  Given
state and strength, place cell i spikes in a step with probability 1 - exp(-g c_i r_i(state) 0.01 s), r_i its running
map (the 4-cm maps of `fields`, joined in pairs) at the position or its mean over all maps for no location, c_i a
factor of the cell.  What the law describes of a ripple is, for every place cell and step, whether the cell spiked.
Fitted per rest by EM (forward-backward for every strength), speeded up by extrapolating over two rounds as in
`dryad_posthoc_content.py` (SQUAREM), until a cycle of three rounds raises the log probability by less than 0.01 or
150 cycles have run (plain EM needed 250 to 400 rounds on made-up ripples).

    replay_states.py extract COHORT DATA CENSUS_OR_FIELDS CONTENT_FOLDER OUT_FOLDER NAME
    replay_states.py predict CONTENT_FOLDER STEPS_FOLDER OUT_FOLDER NAME [CENSUS_ROOT|-] [FIELDS_ROOT]
    replay_states.py summary OUT_FOLDER

`extract`: whether each place cell spiked in each 10-ms step of every stored ripple (as `replay_sequences.py`).
`predict`: E70's comparison on the same blocks and folds: for every ripple of the fold left out and every place cell,
the probability that the cell fired in the ripple, from the other place cells, by this law (from their spikes step by
step) and by the stretch law (from whether they fired); difference of the summed log probabilities, blocks as units.

Expectations and the decision, written before anything was run.
  (s1) Second cohort: the state-space law predicts better than the stretch law (z of at least 1.96).
  (s2) Achilles: the same.
  Only in a cohort where it predicts better would it replace the stretch law for the three-way change measured beyond
  replay; otherwise the stretch law's results stand.  I do not know which way this goes: the law uses more of each
  ripple than the stretch law, but E70 showed that a poor description of timing costs more than timing gives.

Corrected 25 November 2026, after the first run (e72/predict_v1, not to be used).  Two faults of the fit, neither of
the law: (1) where a cell's map rate was zero at a position, the factor update divided 0 by 0; the bisection then
always moved down, every factor fell round after round and the strengths piled up at 4, so that EM lowered the log
probability (by about 30,000 over ten rounds on 1,500 ripples of Achilles) and the stopping rule ended most fits after
two cycles; (2) the first-step weights also give where a ripple enters a position from no location, and their update
counted only the first steps.  Both are fixed (a zero-rate state adds nothing to the factor's derivative; the exact
update counts first steps and entries together), a test now checks that EM rounds never lower the log probability, and
the comparison was run again unchanged (e72/predict_v2).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

STEPS, DT, SPREAD, CYCLES, TOLERANCE, TINY = 10, 0.01, 1.0, 150, 1e-2, 1e-12
BLOCKS, FOLDS = 10, 5


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


content = module("dryad_posthoc_content")
sequences = module("replay_sequences")
GAINS, GROUPS = content.GAINS, sequences.GROUPS


def positions(maps, circular):
    """Rates of every cell at every position (8-cm bins of every map) and at no location (last): (states, cells);
    the map of each position (-1 for no location) and the random-walk kernel between positions (rows sum to one)."""
    blocks = [sequences.coarse(np.asarray(m, dtype=np.float64)) for m in maps]
    rates = np.concatenate([b.T for b in blocks] + [np.concatenate(blocks, axis=1).mean(1)[None, :]])
    which = np.concatenate([np.full(b.shape[1], k) for k, b in enumerate(blocks)] + [np.array([-1])])
    places = len(which) - 1
    walk = np.zeros((places, places))
    start = 0
    for b in blocks:
        bins = b.shape[1]
        index = np.arange(bins)
        distance = np.abs(index[:, None] - index[None, :]).astype(float)
        if circular:
            distance = np.minimum(distance, bins - distance)
        kernel = np.exp(-0.5 * (distance / SPREAD) ** 2)
        walk[start:start + bins, start:start + bins] = kernel / kernel.sum(1, keepdims=True)
        start += bins
    return rates, which, walk


def step_activity(spikes, begin, end):
    """Whether each cell spiked in each step of each window: (windows, cells, STEPS) of 0 and 1."""
    out = np.zeros((len(begin), len(spikes), STEPS), dtype=np.int8)
    for c, s in enumerate(spikes):
        s = np.asarray(s)
        for k in range(STEPS):
            low = begin + (end - begin) * k / STEPS
            high = begin + (end - begin) * (k + 1) / STEPS
            out[:, c, k] = (np.searchsorted(s, high, side="left") - np.searchsorted(s, low, side="left")) > 0
    return out


class States:
    """The state-space law fitted to step activity y (events, cells, STEPS) (see the header)."""

    def __init__(self, y, rates, walk, cycles=CYCLES):
        y = np.asarray(y, dtype=np.float64)
        self.rates, self.walk = np.asarray(rates, dtype=np.float64), np.asarray(walk, dtype=np.float64)
        states, places = self.rates.shape[0], self.walk.shape[0]
        self.places = places
        fired = y.max(2)
        observed = max(float(fired.mean()), 1e-4)
        low, high = 1e-4, 1e4
        for _ in range(60):          # start: one factor, so that no location at strength 1 gives the recorded participation
            middle = np.sqrt(low * high)
            low, high = (middle, high) if (-np.expm1(-middle * self.rates[-1] * DT * STEPS)).mean() < observed else (low, middle)
        self.factor = np.full(self.rates.shape[1], np.sqrt(low * high))
        self.gain_weight = np.full(len(GAINS), 1.0 / len(GAINS))
        self.first = np.full(states, 1.0 / states)
        self.mixture = np.array([0.6, 0.1, 0.3])                                # continuous, fragmented, stationary
        self.leave, self.enter = 0.05, 0.05                                      # position -> none, none -> position
        self.cycles, previous, value = 0, -np.inf, None
        while self.cycles < cycles:
            first = self.parameters()
            value = self.round(y)
            second = self.parameters()
            after_one = self.round(y)
            third = self.parameters()
            step, bend = second - first, third - 2 * second + first
            length = -max(1.0, float(np.linalg.norm(step) / max(np.linalg.norm(bend), 1e-300)))
            self.place(first - 2 * length * step + length ** 2 * bend)
            reached = self.round(y)
            if not np.isfinite(reached) or reached < after_one:
                self.place(third)
            self.cycles += 1
            if value - previous < TOLERANCE:
                break
            previous = value
        self.log_likelihood = value

    def parameters(self):
        return np.concatenate([np.log(self.factor), self.gain_weight, self.first, self.mixture, [self.leave, self.enter]])

    def place(self, theta):
        cells, gains, states = len(self.factor), len(GAINS), len(self.first)
        self.factor = np.exp(np.clip(theta[:cells], np.log(1e-6), np.log(1e6)))
        parts, start = [], cells
        for size in (gains, states, 3):
            part = np.maximum(theta[start:start + size], TINY)
            parts.append(part / part.sum())
            start += size
        self.gain_weight, self.first, self.mixture = parts
        self.leave, self.enter = (float(np.clip(v, 1e-6, 1 - 1e-6)) for v in theta[start:start + 2])

    def transition(self):
        """Transition matrix between states (positions, then no location): (states, states)."""
        places = self.places
        moves = self.mixture[0] * self.walk + self.mixture[1] / places + self.mixture[2] * np.eye(places)
        out = np.zeros((places + 1, places + 1))
        out[:places, :places] = (1 - self.leave) * moves
        out[:places, places] = self.leave
        start = self.first[:places] / max(self.first[:places].sum(), 1e-300)
        out[places, :places] = self.enter * start
        out[places, places] = 1 - self.enter
        return out

    def emission(self, y, leave_out=None, silent=None):
        """Log probability of each event's step activity in each state and step, at each strength: (gains, events, STEPS, states).

        leave_out: a cell whose activity is not counted; silent: with it, the cell is counted as silent in every step."""
        out = []
        for g in GAINS:
            p = np.clip(-np.expm1(-g * self.factor[None, :] * self.rates * DT), 1e-9, 1 - 1e-9)     # (states, cells)
            log_p, log_q = np.log(p), np.log1p(-p)
            keep = np.ones(p.shape[1], dtype=bool)
            if leave_out is not None:
                keep[leave_out] = False
            e = np.einsum("ect,sc->ets", y[:, keep, :], (log_p - log_q)[:, keep]) + log_q[:, keep].sum(1)[None, None, :]
            if leave_out is not None and silent:
                e = e + log_q[:, leave_out][None, None, :]
            out.append(e)
        return np.stack(out)

    def forward(self, emission, backward=False):
        """Forward (and backward) passes at every strength; log probability of every event at every strength."""
        transition = self.transition()
        gains, events, steps, states = emission.shape
        top = emission.max(axis=3, keepdims=True)
        scaled = np.exp(emission - top)
        alpha = np.empty_like(scaled)
        norm = np.empty((gains, events, steps))
        a = self.first[None, None, :] * scaled[:, :, 0, :]
        for t in range(steps):
            if t:
                a = (a @ transition) * scaled[:, :, t, :]
            norm[:, :, t] = a.sum(2)
            a = a / norm[:, :, t, None]
            alpha[:, :, t, :] = a
        log_z = np.log(norm).sum(2) + top[..., 0].sum(2)                                       # (gains, events)
        if not backward:
            return log_z, None, None, transition, scaled, norm
        beta = np.ones_like(scaled)
        for t in range(steps - 2, -1, -1):
            beta[:, :, t, :] = ((beta[:, :, t + 1, :] * scaled[:, :, t + 1, :]) @ transition.T) / norm[:, :, t + 1, None]
        return log_z, alpha, beta, transition, scaled, norm

    def evidence(self, y, leave_out=None, silent=None):
        log_z = self.forward(self.emission(y, leave_out, silent))[0] + np.log(self.gain_weight)[:, None]
        top = log_z.max(0)
        return top + np.log(np.exp(log_z - top[None, :]).sum(0))

    def round(self, y):
        log_z, alpha, beta, transition, scaled, norm = self.forward(self.emission(y), backward=True)
        joint = log_z + np.log(self.gain_weight)[:, None]
        top = joint.max(0)
        total = top + np.log(np.exp(joint - top[None, :]).sum(0))
        responsibility = np.exp(joint - total[None, :])                                       # (gains, events)
        occupancy = alpha * beta * responsibility[:, :, None, None]                            # (gains, events, steps, states)
        self.gain_weight = np.maximum(responsibility.sum(1), 1e-12)
        self.gain_weight /= self.gain_weight.sum()
        initial = occupancy[:, :, 0, :].sum((0, 1))                                          # expected first states
        flow = np.zeros_like(transition)
        states = transition.shape[0]
        for t in range(1, alpha.shape[2]):                 # expected transitions: T * (A' B), without the (events, states, states) array
            before = (alpha[:, :, t - 1, :] * responsibility[:, :, None]).reshape(-1, states)
            after = (beta[:, :, t, :] * scaled[:, :, t, :] / norm[:, :, t, None]).reshape(-1, states)
            flow += transition * (before.T @ after)
        places = self.places
        inside = flow[:places, :places]
        moves = self.mixture[0] * self.walk + self.mixture[1] / places + self.mixture[2] * np.eye(places)
        parts = np.array([(inside * self.mixture[0] * self.walk / np.maximum(moves, 1e-300)).sum(),
                          (inside * (self.mixture[1] / places) / np.maximum(moves, 1e-300)).sum(),
                          (inside * self.mixture[2] * np.eye(places) / np.maximum(moves, 1e-300)).sum()])
        self.mixture = np.maximum(parts, 1e-9) / np.maximum(parts, 1e-9).sum()
        # first-step weights: they also give where a ripple enters a position from no location (transition()), so their
        # exact update counts the expected first steps and the expected entries together (corrected 25 November 2026)
        entering = initial[:places] + flow[places, :places]
        located = initial[:places].sum() / max(initial.sum(), 1e-300)
        first = np.append(located * entering / max(entering.sum(), 1e-300), 1 - located)
        self.first = np.maximum(first, 1e-12) / np.maximum(first, 1e-12).sum()
        self.leave = float(np.clip(flow[:places, places].sum() / max(flow[:places].sum(), 1e-300), 1e-6, 1 - 1e-6))
        self.enter = float(np.clip(flow[places, :places].sum() / max(flow[places].sum(), 1e-300), 1e-6, 1 - 1e-6))
        # factors: the expected log probability of every cell's steps, maximised by bisection on its derivative
        active = np.einsum("gets,ect->gcs", occupancy, y)                                      # (gains, cells, states)
        every = occupancy.sum((1, 2))                                                           # (gains, states)
        silent = every[:, None, :] - active
        exposure = np.array(GAINS)[:, None, None] * self.rates.T[None, :, :] * DT               # (gains, cells, states)
        low, high = self.factor / 4, self.factor * 4
        for _ in range(24):
            middle = np.sqrt(low * high)
            with np.errstate(over="ignore"):                  # a state where the cell's rate is zero adds nothing (it gave 0/0
                ratio = np.divide(active * exposure, np.expm1(middle[None, :, None] * exposure),   # before 25 November 2026)
                                  out=np.zeros_like(exposure), where=exposure > 0)
            pull = ratio.sum((0, 2))
            push = (silent * exposure).sum((0, 2))
            up = pull > push
            low, high = np.where(up, middle, low), np.where(up, high, middle)
        self.factor = np.clip(np.sqrt(low * high), 1e-6, 1e6)
        return float(total.sum())

    def predicted(self, y):
        """Probability that each cell fired in each event, from the other cells' step activity: (events, cells)."""
        out = np.empty((y.shape[0], y.shape[1]))
        for i in range(y.shape[1]):
            without = self.evidence(y, i, silent=False)
            quiet = self.evidence(y, i, silent=True)
            out[:, i] = -np.expm1(np.minimum(quiet - without, 0.0))
        return out


def extract(cohort, data, census_or_fields, content_folder, out, name):
    """Step activity of the place cells of a stored recording, in the order of its stored ripples."""
    d, _ = content.stored(content_folder, name)
    if cohort == "second":
        dryad = sequences.module("dryad_cohort")
        ripples = np.load(Path(census_or_fields) / name / "ripples.npz")
        kept = [int(u) for u in ripples["kept"]]
        spikes = dryad.load_spikes(data, dryad.SESSIONS[name][0])[0]
        begin, end = ripples["peak"] - dryad.WINDOW / 2, ripples["peak"] + dryad.WINDOW / 2
        trains = [spikes[kept[c]] for c in d["cells"]]
    else:
        e62 = sequences.module("hippocampal_study")
        session, side = name.rsplit("_", 1)
        census_root, fields_root = census_or_fields.split(",")
        animal = session.split("-")[0]
        _, found, _ = e62.load(Path(data) / f"sub-{animal}_ses-{session}_behavior+ecephys.nwb", Path(census_root) / session, Path(fields_root) / session)
        begin, end = found[side]["begin"], found[side]["end"]
        trains = [found[side]["spikes"][c] for c in d["cells"]]
    y = step_activity(trains, np.asarray(begin, dtype=float), np.asarray(end, dtype=float))
    assert np.array_equal(y.max(2) > 0, d["x"][:, d["cells"]] > 0), "step activity disagrees with the stored participation"
    folder = Path(out) / "steps"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}.npz", y=y)
    return dict(name=name, ripples=int(y.shape[0]), place_cells=int(y.shape[1]), active_steps=float(y.mean()), fired=float(y.max(2).mean()))


def predict(content_folder, steps_folder, out, name, census_root=None, fields_root=None):
    started = time.time()
    d, _ = content.stored(content_folder, name)
    peak_file = Path(content_folder) / "sessions" / f"{name}_peak.npy"
    peak = np.load(peak_file) if peak_file.exists() else np.load(Path(census_root) / name / "ripples.npz")["peak"]
    keep = np.flatnonzero((d["epoch"] == "PRE") | (d["epoch"] == "POST"))
    keep = keep[np.argsort(peak[keep], kind="stable")]
    y = np.load(Path(steps_folder) / "steps" / f"{name}.npz")["y"][keep].astype(np.float64)
    x, epoch = (y.max(2) > 0).astype(np.int8), d["epoch"][keep]
    circular = sequences.circular_of(name, fields_root) if fields_root else False
    rates, which, walk = positions(sequences.maps_of(d["rates"], d["kinds"]), circular)
    block, fold = sequences.blocks_of(epoch)
    score = {"states": np.zeros(len(x)), "stretch": np.zeros(len(x))}
    laws = []
    for left in range(FOLDS):
        test = np.flatnonzero(fold == left)
        for rest in ("PRE", "POST"):
            train = np.flatnonzero((fold != left) & (epoch == rest))
            rows = test[epoch[test] == rest]
            law = States(y[train], rates, walk)
            stretch = content.Content(x[train], d["rates"], d["window"])
            for model, p in (("states", law.predicted(y[rows])), ("stretch", sequences.stretch_predicted(stretch, x[rows]))):
                p = np.clip(p, 1e-9, 1 - 1e-9)
                score[model][rows] = (x[rows] * np.log(p) + (1 - x[rows]) * np.log1p(-p)).sum(1)
            laws.append(dict(fold=left, rest=rest, cycles=law.cycles, mixture=law.mixture.tolist(), leave=law.leave, enter=law.enter,
                             first_none=float(law.first[-1]), gains=law.gain_weight.tolist(), stretch_cycles=stretch.cycles))
    difference = score["states"] - score["stretch"]
    sums = np.array([difference[block == b].sum() for b in range(2 * BLOCKS)])
    variance = sum(BLOCKS / (BLOCKS - 1) * float(((part - part.mean()) ** 2).sum()) for part in (sums[:BLOCKS], sums[BLOCKS:]))
    report = dict(name=name, place_cells=int(x.shape[1]), ripples=[int((epoch == r).sum()) for r in ("PRE", "POST")], positions=int(len(which) - 1),
                  circular=circular, difference=float(difference.sum()), variance=variance, per_ripple=dict(states=float(score["states"].mean()), stretch=float(score["stretch"].mean())),
                  laws=laws, seconds=round(time.time() - started, 1))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", block=block, **score)
    (out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def summarise(out):
    out = Path(out)
    recordings = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.stem != "summary"}
    groups = {}
    for group, members in GROUPS.items():
        chosen = [recordings[n] for n in members if n in recordings]
        if not chosen:
            continue
        total, variance = sum(r["difference"] for r in chosen), sum(r["variance"] for r in chosen)
        groups[group] = dict(recordings=len(chosen), difference=total, se=float(np.sqrt(variance)), z=float(total / np.sqrt(variance)) if variance > 0 else None,
                             recordings_positive=int(sum(r["difference"] > 0 for r in chosen)))
    report = dict(groups=groups, decision={g: bool(groups.get(g) and groups[g]["z"] is not None and groups[g]["z"] >= 1.96) for g in ("cohort", "achilles")},
                  recordings={n: {k: r[k] for k in ("difference", "variance", "place_cells", "ripples", "per_ripple", "positions")} for n, r in recordings.items()})
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
    else:
        raise SystemExit(__doc__)
