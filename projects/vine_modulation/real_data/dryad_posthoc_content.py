"""Post hoc, second cohort (E64): is the higher three-way term after the tracks a change of what is replayed?

This post hoc analysis was specified on 24 November 2026 after the second-cohort class comparison. Cells with neighboring place fields can fire together when their shared field is replayed. Mixing over represented locations and replay strengths can generate a three-way association even when cells fire independently within each state. Changing the mixture may then change the marginal three-way association.

The content-only law.  A ripple has a state and a gain, and each place cell takes part independently given both:

    P(cell i active | state, gain) = 1 - exp(-gain * c_i * r_i(state) * 0.1 s)

  state  a piece of one running direction of one track: a 4-cm bin, or the bins within 20 cm or within 48 cm of
         it (a replayed stretch crosses several fields inside the 0.1 s of an event), or the whole direction; or
         none.  r_i(state) is the mean of the cell's rate map over the piece (the maps of E64's classes: both
         tracks, both running directions, 4-cm bins, smoothed), and for none its mean over all maps.
  gain   one of 1/4, 1/2, 1, 2, 4.
  c_i    one factor per cell (ripples do not drive every cell as running does).
One law is fitted to each rest of each session by maximum likelihood: the weights of the states, of the gains (one
set for the states with a place, one for none) and the factors c_i; EM from equal weights, speeded up by
extrapolating over two rounds (SQUAREM, Varadhan and Roland 2008), until a cycle of three rounds raises the log
likelihood of the events by less than 0.01 or 300 cycles have run.

    dryad_posthoc_content.py fit DATA CENSUS_ROOT FIELDS_ROOT OUT_FOLDER SESSION
    dryad_posthoc_content.py study OUT_FOLDER [WORKERS]
    dryad_posthoc_content.py simulate OUT_FOLDER REPLICATES SEED [WORKERS]
    dryad_posthoc_content.py summary OUT_FOLDER

`fit` stores, for one session, its ripples, states and sets and the law of each rest (OUT_FOLDER/sessions/); the
other modes read only what it stored.

Three estimates of the `maze` contrast (rest after minus rest before) of `overlapping - separate`, for the pair
term and for the three-way term:
  plain      the estimator of E64 on E64's sets (exact conditional, four strata of the number of other active cells).
  loglinear  for each rest and class, one log-linear law for the patterns of all its sets (one coefficient for a
             cell taking part, one for two cells together and, for trios, one for all three), fitted by maximum
             likelihood; the term is the highest coefficient.
  adjusted   the same log-linear law as a tilt of what the content-only law predicts: for every ripple and set,
             the probability of each pattern of the set under the law of the ripple's rest GIVEN THE RIPPLE'S
             OTHER PLACE CELLS (the posterior over state and gain is taken without the set's own cells).  If the
             cells of a set are independent given the replayed place and its gain, the tilt is zero whatever is
             replayed and however often; its highest coefficient is the term beyond the content-only law.
`loglinear` and `adjusted` use every overlapping set and at most 600 separate pairs and 600 separate trios per
session (seed 1); their standard errors treat ripples as the independent units.  Pooling and rats (as in the
article) as E64; the factor 1.11 on the three-way standard error of `plain` only.

`simulate`: cohorts from the content-only laws.  For every recorded ripple a state and a gain are drawn from its
posterior under the law of its rest, and every place cell then takes part independently with the law's
probability; the other cells keep their recorded activity.  The three estimates are computed on each simulated
cohort exactly as on the recordings, with the laws fitted again to the simulated ripples.

What decides (fixed on 24 November before anything was run on the recordings; the first draft of this file
stratified the estimator of E64 by fifths of a decoded score, and a check on made-up populations showed that such
strata do not remove a change that is only more replay; that estimate was dropped for
`adjusted`, which has the right null by construction):
  beyond the content-only law   the recorded `adjusted` three-way difference lies above the 97.5th percentile of
                                the simulated cohorts' `adjusted` three-way differences;
  explained by content          it lies within their central 95%.
Reported beside it: the share of the recorded `plain` three-way difference that the cohorts of the law give (their
mean over the recorded value) and where the recorded value lies among them.  The law is a fair stand-in for
replay only if its cohorts give at least half of the recorded `plain` change of the PAIR term; otherwise the
comparison says nothing about content and is recorded as such.  The made-up cohorts of
`experiments/e66_content_law_check.py` say how these readings behave when the answer is known.
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

from dvc_package.modulation.binary_pooled import combine, contrast, prepare  # noqa: E402

GAINS, HALF_WIDTHS, FLOOR = (0.25, 0.5, 1.0, 2.0, 4.0), (0, 5, 12), 1e-4
CYCLES, TOLERANCE, TINY = 300, 1e-2, 1e-12
SAMPLE, CHECK_FACTOR = 600, 1.11
KINDS, CLASSES, RESTS = ("pairs", "trios"), ("overlapping", "separate"), ("PRE", "POST")
ESTIMATES = ("plain", "loglinear", "adjusted")
STATE_KINDS = ("bin", "within_20cm", "within_48cm", "whole_direction", "none")
RATS = {"cohort": None, "rat1": ("RAT1or4_SESS",), "rat2": ("RAT2_SESS1", "RAT2_SESS2"), "rat3": ("RAT3_SESS1", "RAT3_SESS2"),
        "rat4": ("RAT1_SESS1", "RAT4_SESS2"), "rat5": ("RAT5_SESS1", "RAT5_SESS2")}


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def rate_maps(cohort, place, behaviour, spikes, units):
    """Rate maps of the given units on the first two tracks, both directions, side by side (as `dryad_cohort.fields`).

    Returns the maps over the bins every map covers, (units, bins), and the map each bin belongs to.
    """
    t = behaviour["t"]
    step = float(t[1] - t[0])
    ahead = max(1, int(round(0.256 / step)))
    maps = []
    for track in behaviour["tracks"]:
        linear = track["linear"]
        bins = int(round(track["length"] / place.BIN))
        v = np.full(len(linear), np.nan)
        v[ahead:-ahead] = (linear[2 * ahead:] - linear[:-2 * ahead]) / (2 * ahead * step)
        finite = np.isfinite(linear)
        sample_bin = np.clip((np.where(finite, linear, 0.0) / place.BIN).astype(int), 0, bins - 1)
        for mask in (finite & (v > place.SPEED), finite & (v < -place.SPEED)):
            block = np.full((len(units), bins), np.nan)
            for c, u in enumerate(units):
                s = spikes[u]
                sample = np.rint((s[(s >= t[0]) & (s <= t[-1])] - t[0]) / step).astype(int)
                sample = sample[(sample >= 0) & (sample < len(t))]
                block[c] = place.rate_map(sample_bin[sample[mask[sample]]], sample_bin[mask], bins, 1 / step, False)[0]
            maps.append(block)
    stacked = np.concatenate(maps, axis=1)
    which = np.concatenate([np.full(block.shape[1], m) for m, block in enumerate(maps)])
    covered = np.isfinite(stacked).all(0)
    return stacked[:, covered], which[covered]


def content_states(fields, which):
    """Mean rate of every cell over every state: pieces of each map, each whole map, and none (the last column).

    fields: (cells, bins); which: the map of each bin.  Returns the rates, (cells, states), and the kind of each
    state as an index into STATE_KINDS.
    """
    fields = np.asarray(fields, dtype=np.float64)
    columns, kinds = [], []
    for m in np.unique(which):
        block = fields[:, which == m]
        bins = block.shape[1]
        running = np.concatenate([np.zeros((len(block), 1)), np.cumsum(block, axis=1)], axis=1)
        for k, half in enumerate(HALF_WIDTHS):
            low, high = np.clip(np.arange(bins) - half, 0, bins), np.clip(np.arange(bins) + half + 1, 0, bins)
            columns.append((running[:, high] - running[:, low]) / (high - low))
            kinds.append(np.full(bins, k))
        columns.append(block.mean(1, keepdims=True))
        kinds.append(np.array([3]))
    columns.append(fields.mean(1, keepdims=True))
    kinds.append(np.array([4]))
    return np.concatenate(columns, axis=1), np.concatenate(kinds)


class Content:
    """A content-only law fitted to the events given (see the header): weights of states and gains, cell factors."""

    def __init__(self, x_place, rates, window, cycles=CYCLES):
        x = np.asarray(x_place, dtype=np.float64)
        self.rates, self.window = np.asarray(rates, dtype=np.float64), window
        self.exposure = np.array(GAINS)[:, None, None] * self.rates[None, :, :] * window        # (gains, cells, states): expected spikes at factor 1
        observed = max(float(x.mean()), 1e-4)
        low, high = 1e-4, 1e4
        for _ in range(60):          # start: one factor for all cells, at which no place and gain 1 give the recorded mean participation
            middle = np.sqrt(low * high)
            low, high = (middle, high) if (-np.expm1(-middle * self.exposure[2, :, -1])).mean() < observed else (low, middle)
        self.factor = np.full(len(self.rates), np.sqrt(low * high))
        self.weight = np.full(self.exposure[:, 0, :].shape, 1.0 / self.exposure[:, 0, :].size)
        self.tables()
        self.cycles, self.log_likelihood, previous = 0, None, -np.inf
        while self.cycles < cycles:
            first = self.parameters()
            value = self.round(x)                                   # the log likelihood at `first`; the law moves one round on
            second = self.parameters()
            after_one = self.round(x)
            third = self.parameters()
            step, bend = second - first, third - 2 * second + first
            length = -max(1.0, float(np.linalg.norm(step) / max(np.linalg.norm(bend), 1e-300)))
            self.place(first - 2 * length * step + length ** 2 * bend)
            reached = self.round(x)                                 # the log likelihood at the extrapolated law; one round on from it
            if not np.isfinite(reached) or reached < after_one:     # the extrapolation did not help: keep the two plain rounds
                self.place(third)
            self.cycles += 1
            self.log_likelihood = value
            if value - previous < TOLERANCE:
                break
            previous = value

    @classmethod
    def restored(cls, rates, window, factor, weight):
        """A law with the factors and weights given (nothing is fitted)."""
        law = cls(np.zeros((1, len(rates)), dtype=np.int8), rates, window, cycles=0)
        law.factor, law.weight = np.asarray(factor, dtype=np.float64), np.asarray(weight, dtype=np.float64)
        law.tables()
        return law

    def parameters(self):
        """Log factors, then the weights as their margins: gains and states among the states with a place, gains for none."""
        place = self.weight[:, :-1]
        return np.concatenate([np.log(self.factor), place.sum(1), place.sum(0), self.weight[:, -1]])

    def place(self, parameters):
        cells, gains = len(self.factor), len(GAINS)
        self.factor = np.exp(np.clip(parameters[:cells], np.log(1e-6), np.log(1e6)))
        rest = np.maximum(parameters[cells:], TINY)
        by_gain, by_state, none = rest[:gains], rest[gains:-gains], rest[-gains:]
        weight = np.empty_like(self.weight)
        weight[:, :-1] = np.outer(by_gain, by_state) / by_state.sum()
        weight[:, -1] = none
        self.weight = weight / weight.sum()
        self.tables()

    def round(self, x):
        """One round of EM.  Returns the log likelihood of the events under the law as it was before the round."""
        joint = self.joint(self.likelihood(x))
        top = joint.max(axis=(0, 2), keepdims=True)
        post = np.exp(joint - top)
        norm = post.sum(axis=(0, 2), keepdims=True)
        post /= norm                                                                         # (gains, events, states)
        total = post.sum(1)
        active = np.stack([x.T @ post[g] for g in range(len(GAINS))])                        # (gains, cells, states)
        self.weight = self.weights_of(total)
        self.factor = self.factors_of(active, total)
        self.tables()
        return float((top + np.log(norm)).sum())

    def tables(self):
        self.p = np.clip(-np.expm1(-self.factor[None, :, None] * self.exposure), FLOOR, 1 - FLOOR)   # (gains, cells, states)
        self.log_p, self.log_q = np.log(self.p), np.log1p(-self.p)
        self.table = self.p.transpose(0, 2, 1).reshape(-1, self.p.shape[1])                      # (gains * states, cells)

    @staticmethod
    def weights_of(total):
        """Weights from the summed posteriors: gain and state independent among the states with a place; none apart."""
        place = total[:, :-1]
        weight = np.empty_like(total)
        weight[:, :-1] = np.outer(place.sum(1), place.sum(0)) / max(place.sum(), 1e-300)
        weight[:, -1] = total[:, -1]
        return weight / weight.sum()

    def factors_of(self, active, total):
        """The factor of each cell that maximises the expected log likelihood (bisection on its derivative)."""
        positive = self.exposure > 0
        exposure = np.where(positive, self.exposure, 1.0)
        silent = ((total[:, None, :] - active) * self.exposure).sum((0, 2))
        low, high = self.factor / 4, self.factor * 4
        for _ in range(24):
            middle = np.sqrt(low * high)
            with np.errstate(over="ignore"):
                pull = np.where(positive, active * exposure / np.expm1(middle[None, :, None] * exposure), 0.0).sum((0, 2))
            up = pull > silent
            low, high = np.where(up, middle, low), np.where(up, high, middle)
        return np.clip(np.sqrt(low * high), 1e-6, 1e6)

    def likelihood(self, x_place):
        """Log likelihood of every event at every gain and state: (gains, events, states)."""
        x = np.asarray(x_place, dtype=np.float64)
        return np.stack([x @ (self.log_p[g] - self.log_q[g]) + self.log_q[g].sum(0)[None, :] for g in range(len(GAINS))])

    def joint(self, log_likelihood):
        return log_likelihood + np.log(np.maximum(self.weight, 1e-300))[:, None, :]

    def posterior(self, log_likelihood):
        joint = self.joint(log_likelihood)
        weight = np.exp(joint - joint.max(axis=(0, 2), keepdims=True))
        return weight / weight.sum(axis=(0, 2), keepdims=True)

    def evidence(self, log_likelihood):
        """Log probability of every event under the law."""
        joint = self.joint(log_likelihood)
        top = joint.max(axis=(0, 2))
        return top + np.log(np.exp(joint - top[None, :, None]).sum(axis=(0, 2)))

    def decoded(self, x_place):
        """Posterior over gain and state of every event, as (events, gains * states) in the order of `self.table`."""
        post = self.posterior(self.likelihood(x_place))
        return post.transpose(1, 0, 2).reshape(post.shape[1], -1)

    def draw(self, x_place, rng):
        """New events from the law: a state and gain from the posterior of each event given, then independent cells."""
        running = np.cumsum(self.decoded(x_place), axis=1)
        pick = (rng.random(len(running))[:, None] > running).sum(1).clip(0, running.shape[1] - 1)
        p = self.table[pick]
        return (rng.random(p.shape) < p).astype(np.int8)

    def predicted(self, x_place, members, decoded=None):
        """Probability of each pattern of each set in each event given the event's other cells: (events, sets, patterns).

        members: (sets, 2 or 3) indices into the place cells.  The posterior over gain and state is taken without
        the set's own cells, and the cells of the set are independent given both.  Pattern p has cell k of the set
        active when bit k of p is set (the coding of `dvc_package.modulation.binary_pooled`).
        """
        x, members = np.asarray(x_place, dtype=np.int64), np.asarray(members, dtype=np.int64)
        decoded = self.decoded(x) if decoded is None else decoded
        size = members.shape[1]
        bits = (np.arange(2 ** size)[:, None] >> np.arange(size)[None, :]) & 1                  # (patterns, cells of a set)
        out = np.empty((len(x), len(members), 2 ** size), dtype=np.float32)
        for k, cells in enumerate(members):
            p = self.table[:, cells]                                                              # (gains * states, cells of the set)
            own = np.where(bits[:, None, :] == 1, p[None], 1 - p[None]).prod(2)                   # the set's own likelihood of each pattern
            code = x[:, cells] @ (1 << np.arange(size))
            joint = decoded @ (own / own[0]).T                                                    # right for the events in which the set is silent
            for c in np.unique(code[code > 0]):
                rows = np.flatnonzero(code == c)
                joint[rows] = decoded[rows] @ (own / own[c]).T
            out[:, k] = joint / joint.sum(1, keepdims=True)
        return out

    def describe(self, x_place, kinds):
        """What the fitted law looks like, and how much better than independent cells it describes the events."""
        x = np.asarray(x_place, dtype=np.float64)
        mean = np.clip(x.mean(0), 1e-6, 1 - 1e-6)
        independent = float((x @ np.log(mean) + (1 - x) @ np.log1p(-mean)).mean())
        by_state = self.weight.sum(0)
        implied = (self.weight[:, None, :] * self.p).sum((0, 2))
        return dict(events=int(len(x)), weight={name: float(by_state[kinds == k].sum()) for k, name in enumerate(STATE_KINDS)},
                    gain={str(g): float(w) for g, w in zip(GAINS, self.weight.sum(1))},
                    log_likelihood_per_event=float(self.evidence(self.likelihood(x)).mean()), independent_cells_per_event=independent, cycles=self.cycles,
                    factor_median=float(np.median(self.factor)), largest_participation_error=float(np.abs(implied - x.mean(0)).max()))


def statistics(size):
    """What a log-linear law of a pair or trio counts in each pattern: cells active, pairs of them, and all three."""
    active = ((np.arange(2 ** size)[:, None] >> np.arange(size)[None, :]) & 1).sum(1)
    columns = [active, active * (active - 1) // 2] + ([active == 3] if size == 3 else [])
    return np.column_stack(columns).astype(np.float64)


def tilted(blocks, size, rounds=80):
    """One log-linear tilt of the patterns of many sets away from the probabilities predicted for them.

    blocks: [(predicted, code)], one per session; code (events, sets) holds the observed patterns and predicted
    (events, sets, patterns) their probabilities, or None for equal probabilities (a plain log-linear law).  The
    coefficients theta maximise the sum over events and sets of
        log[ predicted(q) exp(theta . T(q)) / sum_q' predicted(q') exp(theta . T(q')) ]   at the observed q,
    with T = `statistics`.  Returns theta and, for each block, every event's part in theta (events, coefficients):
    the squares of the parts of a coefficient add up to its variance with events as the independent units.
    Returns None when the coefficients run away (a pattern that never occurs).
    """
    counted = statistics(size)
    theta = np.zeros(counted.shape[1])

    def moments():
        gradient, curvature, scores = np.zeros(len(theta)), np.zeros((len(theta), len(theta))), []
        base = np.exp(counted @ theta)
        for predicted, code in blocks:
            if predicted is None:
                w = base / base.sum()
                mean = w @ counted
                score = counted[code].sum(1) - code.shape[1] * mean[None, :]
                curvature += code.size * ((counted * w[:, None]).T @ counted - np.outer(mean, mean))
            else:
                w = predicted * base[None, None, :]
                w /= w.sum(2, keepdims=True)
                mean = w @ counted                                                       # (events, sets, coefficients)
                score = (counted[code] - mean).sum(1)
                flat = mean.reshape(-1, len(theta))
                curvature += (counted * w.sum((0, 1))[:, None]).T @ counted - flat.T @ flat
            scores.append(score)
            gradient += score.sum(0)
        return gradient, curvature, scores

    for _ in range(rounds):
        gradient, curvature, scores = moments()
        try:
            step = np.linalg.solve(curvature + 1e-9 * np.eye(len(theta)), gradient)
        except np.linalg.LinAlgError:
            return None
        step *= min(1.0, 2.0 / max(float(np.abs(step).max()), 1e-300))
        theta = theta + step
        if not np.isfinite(theta).all() or np.abs(theta).max() > 30:
            return None
        if np.abs(step).max() < 1e-8:
            break
    else:
        return None
    gradient, curvature, scores = moments()
    inverse = np.linalg.inv(curvature)
    return theta, [score @ inverse for score in scores]


def load(data, census_root, fields_root, session):
    cohort, place = module("dryad_cohort"), module("hippocampal_place_fields")
    label = cohort.SESSIONS[session][0]
    ripples = np.load(Path(census_root) / session / "ripples.npz")
    kept = [int(u) for u in ripples["kept"]]
    spikes = cohort.load_spikes(data, label)[0]
    peak, epoch = ripples["peak"], ripples["epoch"]
    x = np.column_stack([cohort.count_in(spikes[u], peak - cohort.WINDOW / 2, peak + cohort.WINDOW / 2) > 0 for u in kept]).astype(np.int8)
    sets, _, is_place = cohort.class_sets(Path(fields_root) / session, kept)
    cells = np.flatnonzero(is_place)                    # place cells, as indices into the kept units
    fields, which = rate_maps(cohort, place, cohort.load_behaviour(data, label), spikes, [kept[c] for c in cells])
    rates, kinds = content_states(fields, which)
    sets = {(kind, cls): sets[f"{kind}_{cls}"] for kind in KINDS for cls in CLASSES}
    return dict(session=session, x=x, epoch=epoch, cells=cells, rates=rates, kinds=kinds, window=cohort.WINDOW, sets=sets, sampled=sample_of(sets))


def sample_of(sets, seed=1):
    """The sets of `loglinear` and `adjusted`: every overlapping one, and at most SAMPLE separate ones of each size."""
    rng = np.random.default_rng(seed)
    chosen = {}
    for kind in KINDS:
        for cls in CLASSES:
            members = sets[(kind, cls)]
            if cls == "separate" and len(members) > SAMPLE:
                members = members[np.sort(rng.choice(len(members), SAMPLE, replace=False))]
            chosen[(kind, cls)] = members
    return chosen


def laws_of(d, x=None):
    """The content-only law of each rest of a session (of the events x in place of the recorded ones, if given)."""
    x = d["x"] if x is None else x
    return {rest: Content(x[d["epoch"] == rest][:, d["cells"]], d["rates"], d["window"]) for rest in RESTS}


def plain_parts(d, x=None):
    """Tables of the `maze` contrast for E64's estimator."""
    e62 = module("hippocampal_study")
    x = d["x"] if x is None else x
    a, b = d["epoch"] == "PRE", d["epoch"] == "POST"
    size = x.sum(1)
    edges = e62.edges_of(size[a], size[b])
    return {key: prepare(x[a], x[b], members, size[a], size[b], edges) for key, members in d["sets"].items() if len(members)}


def session_parts(d, laws, x=None):
    """What a session gives to the three estimates.

    plain: the tables of E64's estimator.  patterns[(rest, kind, class)]: the probabilities of the patterns of the
    sampled sets in the ripples of the rest under the law of the rest, given the other place cells (events, sets,
    patterns), and the observed patterns (events, sets).  x: events in place of the recorded ones.
    """
    x = d["x"] if x is None else x
    cells = d["cells"]
    position = {int(c): i for i, c in enumerate(cells)}
    patterns = {}
    for rest, law in laws.items():
        events = x[d["epoch"] == rest]
        decoded = law.decoded(events[:, cells])
        for (kind, cls), members in d["sampled"].items():
            if len(members):
                inside = np.array([[position[int(c)] for c in row] for row in members])
                code = events[:, members].astype(np.int64) @ (1 << np.arange(members.shape[1]))
                patterns[(rest, kind, cls)] = (law.predicted(events[:, cells], inside, decoded), code)
    return dict(plain=plain_parts(d, x), patterns=patterns, n_a=int((d["epoch"] == "PRE").sum()), n_b=int((d["epoch"] == "POST").sum()))


def store(out, d, laws):
    """Write a session's ripples, states and sets and the laws of its rests to OUT/sessions/."""
    folder, session = Path(out) / "sessions", d["session"]
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(x=d["x"], epoch=d["epoch"], cells=d["cells"], rates=d["rates"], kinds=d["kinds"], window=np.array(d["window"]))
    for (kind, cls), members in d["sets"].items():
        arrays[f"sets_{kind}_{cls}"] = members
        arrays[f"sampled_{kind}_{cls}"] = d["sampled"][(kind, cls)]
    for rest, law in laws.items():
        arrays[f"factor_{rest}"], arrays[f"weight_{rest}"] = law.factor, law.weight
    np.savez_compressed(folder / f"{session}.npz", **arrays)
    described = dict(session=session, place_cells=int(len(d["cells"])), kept=int(d["x"].shape[1]), states=int(d["rates"].shape[1]),
                     sets={f"{kind}_{cls}": [int(len(d["sets"][(kind, cls)])), int(len(d["sampled"][(kind, cls)]))] for kind in KINDS for cls in CLASSES},
                     laws={rest: law.describe(d["x"][d["epoch"] == rest][:, d["cells"]], d["kinds"]) for rest, law in laws.items()})
    (folder / f"{session}.json").write_text(json.dumps(described, indent=1) + "\n")
    return described


def fit(data, census_root, fields_root, out, session):
    """Fit the law of each rest of a session and store what the other modes need of it."""
    d = load(data, census_root, fields_root, session)
    return store(out, d, laws_of(d))


def stored(out, session):
    """A session as `store` wrote it, and its laws."""
    data = np.load(Path(out) / "sessions" / f"{session}.npz")
    d = dict(session=session, x=data["x"], epoch=data["epoch"], cells=data["cells"], rates=data["rates"], kinds=data["kinds"], window=float(data["window"]),
             sets={(kind, cls): data[f"sets_{kind}_{cls}"] for kind in KINDS for cls in CLASSES},
             sampled={(kind, cls): data[f"sampled_{kind}_{cls}"] for kind in KINDS for cls in CLASSES})
    laws = {rest: Content.restored(d["rates"], d["window"], data[f"factor_{rest}"], data[f"weight_{rest}"]) for rest in RESTS}
    return d, laws


def sessions_of(out):
    return sorted(path.stem for path in (Path(out) / "sessions").glob("*.npz"))


def simulated_session(d, laws, rng):
    """One session from the content-only laws: the place cells of every ripple drawn again from the law of its rest."""
    out = d["x"].copy()
    for rest, law in laws.items():
        rows = np.flatnonzero(d["epoch"] == rest)
        out[np.ix_(rows, d["cells"])] = law.draw(d["x"][rows][:, d["cells"]], rng)
    return out


def pooled(entries, kind, factor=1.0):
    """E64's estimate of the change of a term, overlapping minus separate, shared over the sessions given."""
    fits = {}
    for cls in CLASSES:
        parts, labels_a, labels_b, offset = [], [], [], np.zeros(2, dtype=np.int64)
        for entry in entries:
            part = entry["plain"].get((kind, cls))
            if part is not None:
                parts.append(part)
                labels_a.append(np.arange(entry["n_a"]) + offset[0])
                labels_b.append(np.arange(entry["n_b"]) + offset[1])
            offset = offset + np.array([entry["n_a"], entry["n_b"]])
        if not parts:
            return None
        fits[cls] = combine(parts, labels_a, labels_b)
    d = contrast(fits["overlapping"], fits["separate"])
    return dict(difference=float(d["delta"]), se=float(d["se"] * factor), overlapping=float(fits["overlapping"].delta), overlapping_se=float(fits["overlapping"].se),
                separate=float(fits["separate"].delta), separate_se=float(fits["separate"].se))


def term_change(entries, kind, adjusted, units=None):
    """The term of each class in each rest as a log-linear coefficient shared over the sessions given, and its change.

    adjusted: tilt the content-only law's predictions (True) or equal probabilities (False, a plain log-linear law).
    units: optional, one pair (labels before, labels after) per entry, integers that name the unit of every ripple
    and are unique across entries except where ripples of different entries form one unit (coinciding ripples of
    the two hemispheres of a session).  The variance then sums each unit's contributions before squaring.  Without
    it every ripple is its own unit (the estimates of sections 66 and 68).
    """
    size = 2 if kind == "pairs" else 3
    out, change, variance = {}, 0.0, 0.0
    for k, (sign, rest) in enumerate(((-1.0, "PRE"), (1.0, "POST"))):
        top, parts = {}, {}
        for cls in CLASSES:
            blocks, owners = [], []
            for index, entry in enumerate(entries):
                block = entry["patterns"].get((rest, kind, cls))
                if block is not None and block[1].size:
                    blocks.append((block[0] if adjusted else None, block[1]))
                    owners.append(index)
            fitted = tilted(blocks, size) if blocks else None
            if fitted is None:
                return None
            top[cls] = float(fitted[0][-1])
            parts[cls] = {index: part[:, -1] for index, part in zip(owners, fitted[1])}
        spread = 0.0
        if units is None:
            for index in set(parts["overlapping"]) | set(parts["separate"]):
                spread += float(np.sum((parts["overlapping"].get(index, 0.0) - parts["separate"].get(index, 0.0)) ** 2))
        else:
            labels = {index: np.asarray(units[index][k], dtype=np.int64) for index in set(parts["overlapping"]) | set(parts["separate"])}
            summed = np.zeros(max(int(v.max()) + 1 if len(v) else 0 for v in labels.values()))
            for index, label in labels.items():
                np.add.at(summed, label, parts["overlapping"].get(index, 0.0) - parts["separate"].get(index, 0.0))
            spread = float((summed ** 2).sum())
        out[rest.lower()] = dict(overlapping=top["overlapping"], separate=top["separate"], difference=top["overlapping"] - top["separate"], se=float(np.sqrt(spread)))
        change += sign * out[rest.lower()]["difference"]
        variance += spread
    return dict(difference=float(change), se=float(np.sqrt(variance)), **out)


def estimates(entries):
    """The three estimates for the sessions given, for pairs and trios."""
    return {"plain": {kind: pooled(entries, kind, CHECK_FACTOR if kind == "trios" else 1.0) for kind in KINDS},
            "loglinear": {kind: term_change(entries, kind, False) for kind in KINDS},
            "adjusted": {kind: term_change(entries, kind, True) for kind in KINDS}}


def results(entries):
    """The estimates for the cohort and for each rat, from the per-session entries."""
    return {group: estimates([entries[s] for s in (sorted(entries) if members is None else members) if s in entries]) for group, members in RATS.items()}


def study_session(arguments):
    out, session = arguments
    d, laws = stored(out, session)
    return session, session_parts(d, laws)


def study(out, workers=1):
    out = Path(out)
    entries = {}
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for session, entry in (map if workers <= 1 else pool.map)(study_session, [(str(out), session) for session in sessions_of(out)]):
            entries[session] = entry
            print("done", session, flush=True)
    report = dict(groups=results(entries), sessions={s: json.loads((out / "sessions" / f"{s}.json").read_text()) for s in entries},
                  plain_by_session={s: {kind: pooled([e], kind, CHECK_FACTOR if kind == "trios" else 1.0) for kind in KINDS} for s, e in entries.items()})
    (out / "study.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


CACHE = {}


def setting(out):
    if out not in CACHE:
        CACHE[out] = {session: stored(out, session) for session in sessions_of(out)}
    return CACHE[out]


def simulate_cohort(arguments):
    out, seed, replicate = arguments
    started = time.time()
    entries, cycles = {}, []
    for index, (session, (d, laws)) in enumerate(setting(out).items()):
        x = simulated_session(d, laws, np.random.default_rng([seed, index, replicate]))
        again = laws_of(d, x)                                        # as on the recordings: laws fitted to the ripples at hand
        cycles += [law.cycles for law in again.values()]
        entries[session] = session_parts(d, again, x)
    return dict(replicate=replicate, seed=seed, groups=results(entries), largest_cycles=int(max(cycles)), seconds=round(time.time() - started, 1))


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
    """The simulated cohorts beside the recordings: where each recorded estimate lies among the cohorts of the law."""
    out = Path(out)
    rows = [json.loads(line) for line in (out / "cohorts.jsonl").read_text().splitlines()]
    recorded = json.loads((out / "study.json").read_text())["groups"]
    report = dict(cohorts=len(rows), groups={})
    for group in RATS:
        entry = {}
        for which in ESTIMATES:
            for kind in KINDS:
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
    cohort = report["groups"]["cohort"]
    if {"adjusted_trios", "plain_trios", "plain_pairs"} <= set(cohort):
        a, p = cohort["adjusted_trios"], cohort["plain_pairs"]
        report["reading"] = dict(
            fair_stand_in=bool(p["mean"] >= 0.5 * p["recorded"]),
            adjusted_three_way="beyond the content-only law" if a["recorded"] > a["high"] else "explained by content" if a["recorded"] >= a["low"] else "below the content-only law",
            plain_three_way_share_given_by_the_law=cohort["plain_trios"]["share_given_by_the_law"])
    (out / "content_only_summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def show(groups):
    for group, entry in groups.items():
        for which in ESTIMATES:
            p, t = entry[which]["pairs"], entry[which]["trios"]
            if p is None or t is None:
                print(f"{group:7s} {which:9s} not estimable")
                continue
            print(f"{group:7s} {which:9s} pairs {p['difference']:+.3f} ({p['se']:.3f}) | trios {t['difference']:+.3f} ({t['se']:.3f}) z {t['difference'] / t['se']:+.2f}"
                  + ("" if which == "plain" else f"  [three-way term, overlapping minus separate: before {t['pre']['difference']:+.3f}, after {t['post']['difference']:+.3f}]"))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        print(json.dumps(fit(*sys.argv[2:7]), indent=1))
    elif mode == "study":
        show(study(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)["groups"])
    elif mode == "simulate":
        simulate(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif mode == "summary":
        r = summarise(sys.argv[2])
        for group, entry in r["groups"].items():
            for name, v in entry.items():
                print(f"{group:7s} {name:16s} recorded {v['recorded']:+.3f} | law's cohorts {v['mean']:+.3f} (sd {v['sd']:.3f}; 95% {v['low']:+.3f} to {v['high']:+.3f}); "
                      f"{v['cohorts_at_or_above_recorded']} of {v['usable']} at or above the recorded value")
        print(json.dumps(r.get("reading"), indent=1))
    else:
        raise SystemExit(__doc__)
