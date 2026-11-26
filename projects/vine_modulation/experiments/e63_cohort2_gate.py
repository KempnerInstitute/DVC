"""E63.  Simulation check of the contrast and of the rule that E64 uses on the second cohort.

The 23 November protocol required a simulation of the primary contrast before opening new biological outcomes. The simulation takes the retained pyramidal units, pre- and post-track participation, ripple counts, variability of population activity (1.4–2.2 times the independent-cell standard deviation), and place-field-defined pairs and trios from each session. Pair and trio rest statistics were reserved for evaluation.

Law of a simulated session (events independent; Gibbs sweeps given the state s of the event):

    log P(x | s) = sum_i (h_i + c_i sigma s) x_i + sum_{i<j} J_ij x_i x_j + K g(x) + constant

  s      the state of the event, standard normal, shared by all cells;  c_i  the cell's loading (log-normal with
         mean 1, sd of the logarithm 0.3, drawn once per session);  sigma  set for each epoch so that the number
         of active cells varies as much as recorded;
  J      a background (normal, sd 0.1) plus j = 0.2 for the overlapping pairs;
  g(x)   the number of overlapping trios whose three cells are all active;  K = 0.02, a small persistent term;
  h      set so that every cell has its recorded participation in A and in B.

Scenarios (what differs between A and B besides participations, numbers of ripples and the recorded spread):
  rates   nothing else
  pairs   j is 0.3 in B
  gain    sigma in B is 1.4 times the value that gives the recorded spread
  shift   as `pairs`; then the three-way term of every overlapping trio is raised in B by D, by moves of its
          table (below)

The planted change (`plant`).  For three cells the tables with given pair margins differ by multiples of one
direction, and a step along it adds one to the count of (1,1,1).  A move for a trio switches one of its cells on
in an event where the other two are both active and in one where both are silent, and off in an event where only
the one and in an event where only the other is active: the trio's eight counts take that step, its three pair
margins and every cell's participation stay as they were.  For each event of B in which a trio is fully active a
move is made with probability D w (w = (1/n111) / sum over the eight patterns of 1/n, one unless (1,1,1) is not
the rarest pattern), so that the expected change of the trio's term is D; the four events are taken, where the
stratum has them, from the stratum of that event (the number of other active cells).  Moves for one trio also
touch the tables of the trios that share the switched cell; what the cohort's estimate measures on average is
therefore read from the development run, and D is chosen so that it is about 0.2 (Achilles: 0.20).

Why not a larger K in B.  A cell belongs to up to 46 overlapping trios and the groups of mutually overlapping
cells have up to 8 cells.  Three development runs with a term in the law failed for that reason: with K + D = 0.17
a dense group switches on as a whole and the recorded participations could not be held in three sessions
(`e63/dev1`, errors of 0.11 to 0.32); a cap of 3 on each cell's own count left the two densest sessions unstable
and planted changes that differed fourfold between sessions (`e63/dev2_*`); a bounded term K min(6, g(x)) was
stable, but in the dense sessions the bound is reached in most events and the measured change was negative
(`e63/dev3_*`: -0.07 to -0.11 in rat 3 against +0.15 to +0.33 in the sessions with few trios).

A simulated cohort is one simulated session per recorded session.  Its quantities are those of E64, computed with
the same functions: four strata of the number of other active cells; one shift per class from the conditional
likelihood of all sessions together; the difference `overlapping - separate` with the covariance of the two
classes; the same for each rat (the doubly labelled session once with rat 1, `rat1+`, and once with rat 4,
`rat4+`) and for each session.  The rule (d2) of E64 is applied to every simulated cohort by the study's own
function (`dryad_cohort.rule_d2`).
Not simulated: the built-in check on the halves of PRE and its factor, which can only widen a standard error.

    e63_cohort2_gate.py run CENSUS_ROOT FIELDS_ROOT OUT REPLICATES SEED [WORKERS] [SESSIONS] [SCENARIOS]
    e63_cohort2_gate.py summary OUT
    e63_cohort2_gate.py factor OUT FACTOR      after the check: the same cohorts with every standard error of the
                                               three-way difference multiplied by FACTOR (post hoc)
    e63_cohort2_gate.py comparator CENSUS_ROOT FIELDS_ROOT OUT REPLICATES WORKERS SESSION+SESSION
                                               post hoc: the log-linear comparator on sessions simulated without a
                                               change (the laws and settings of OUT, a folder of a finished run)

`run` first sets h and sigma of every session and law (kept in OUT/calibration), then simulates the cohorts; it
can be started again and goes on where it stopped.  For trials only: SESSIONS and SCENARIOS restrict the run
(names joined by `+`, or `all`), and the environment variable E63_SHIFT replaces D.  The check uses neither.

Seeds: development 970630 (used to fix D); check 990630, run once after this file is committed with D fixed.

DEVELOPMENT (seed 970630; looked at while this file was still being changed).  `e63/dev1`, 24 cohorts per
scenario, the laws without a change as they are now: the cohort's estimate of the three-way difference averaged
+0.015 (rates), +0.024 (pairs) and +0.006 (gain) with a standard deviation of 0.045 to 0.052 against a reported
standard error of 0.054 to 0.057; its test rejected in 1 of 24 cohorts of each scenario, and the rule (d2) declared
0, 1 and 1 of 24.  `e63/dev4_d15` and `dev4_d20`, 48 cohorts of `shift` each: with D = 0.15 the cohort's estimate
averaged 0.211 (0.11 to 0.30 over the nine sessions; pair difference 0.12 as in `pairs`) and with D = 0.20, 0.262.
D = 0.14 is therefore taken (0.20 by the line through the two).  With D = 0.15 the cohort's test rejected in 47
of 48 cohorts and the rule (d2) declared a replication in 34.

EXPECTATIONS (written before the check run; 200 simulated cohorts per scenario):
 (h0) The simulation is what it says: in every session and law the simulated participations are within 0.02 of
      the recorded ones and the spread within 0.1 of the recorded one.
 (h1) Without a change of the triplet term (rates, pairs, gain) the cohort's test of the three-way difference
      rejects in at most 8% of the cohorts of each scenario (a true 5% gives at most 16 of 200 in 98 of 100 runs).
 (h2) Without a change the mean of the cohort's estimate is within 0.03 of zero in each scenario.
 (h3) With the shift the cohort's test rejects in at least 80% of the cohorts.
 (h4) The rule (d2) declares a replication in at most 5% of the cohorts of each scenario without a change.
 (h5) The pair term's difference is found by the cohort's test in at least 90% of the cohorts of `pairs` and of
      `shift`, and its test rejects in at most 11% of those of `rates` and of `gain`.
The check is passed if (h0) to (h4) hold.  Reported without a criterion: how often the rule (d2) declares a
replication with the shift (the rule asks for the sign in four of five rats, and three rats have little
information); how often a cohort "speaks against" an effect of Achilles's size; and the same quantities for each
rat and each session.

POST HOC (designed on 24 November 2026 after E64 had been run and read; not part of the check).  In the recordings
the pair term's difference `overlapping - separate` changed by +0.45 (overlapping pairs +0.35, separate pairs
-0.10), three times the +0.15 that the scenario `pairs` of the check produces.  The scenario `pairs` is therefore
run again with a pair coupling of the overlapping class in B (E63_PAIR_AFTER) and a coupling added to the separate
pairs in B (E63_SEPARATE_AFTER) chosen so that the simulated pair terms move as the recorded ones did, to see how
far the measured three-way difference moves when only pairs change by that much, and how often the rule (d2) then
declares a replication.  Seed 990632; outputs `e63/posthoc_pairs_*`.  With couplings of 0.45 and 0.52 the
simulated pair difference was +0.56 and +0.67, but the law could not hold the recorded participations in the
three densest sessions (errors of 0.09 to 0.57): a pair coupling of that size switches their groups on as a whole.
The pair change is therefore planted as the trio change is, by moves of the pair tables (`plant_pairs`): scenario
`moved` has the law of `rates` and then, in B, the pair terms of the overlapping and of the separate pairs moved
by E63_MOVE_OVERLAPPING and E63_MOVE_SEPARATE; `moved_shift` then plants D as `shift` does.  Development seed 970632 to set the two amounts so
that the simulated pair terms move as the recorded ones; then seed 990633; outputs `e63/posthoc_moved_*`.
That stand-in was dropped after its development runs: a move of a pair's table puts the same number of events
into both states of a third cell and so lowers the three-way term by construction (-0.08 to -0.10 for a pair
difference of +0.20 to +0.23), which a change of couplings does not.  The matched run is therefore made with the
couplings, session by session (E63_BY_SESSION=1 reads `e63_posthoc_matched.json`: each session's pair coupling in
B is the one at which its simulated pair term moved about as the recorded one, kept below the value at which its
law no longer held the recorded participations); scenarios `pairs` and `shift`, seed 990633, 200 cohorts, outputs
`e63/posthoc_matched`.  Outcome of the post hoc simulations: the simulated pair difference is +0.44 (recorded
+0.45); the three-way difference then averages +0.024 over the cohort (spread 0.064, largest of 200 cohorts 0.178,
recorded 0.296) and +0.029 over the seven sessions whose law held; the rule (d2) declares in 6%; with D planted as
well the estimate averages 0.205.  The laws of the two sessions of rat 3 did not hold the recorded participations
at their matched couplings (errors 0.22 and 0.77).
THE CHECK'S OUTCOME is in `real_data/dryad_cohort.py` (module docstring): (h1) failed, the others held.
"""
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dvc_package.modulation.binary_pooled import combine, contrast, prepare  # noqa: E402

BACKGROUND_SD, PAIR, TRIPLET, LOADING_SD, GAIN_FACTOR, SWEEPS = 0.1, 0.2, 0.02, 0.3, 1.4, 30
# Post hoc runs only (after E64; see POST HOC in the header): a larger pair coupling of the overlapping class in B
# and a coupling added to the separate pairs in B.  The check used the defaults.
PAIR_AFTER = float(os.environ.get("E63_PAIR_AFTER", "0.3"))
SEPARATE_AFTER = float(os.environ.get("E63_SEPARATE_AFTER", "0"))
# E63_BY_SESSION=1 takes the two from `e63_posthoc_matched.json`, session by session (post hoc, matched run).
BY_SESSION = json.loads((HERE / "e63_posthoc_matched.json").read_text()) if os.environ.get("E63_BY_SESSION") else None
# Post hoc scenarios `moved` and `moved_shift`: the law of `rates`, then the pair terms of the two classes are
# moved in B by moves of the pair tables (`plant_pairs`), by these amounts; `moved_shift` then plants D as `shift`.
MOVE_OVERLAPPING = float(os.environ.get("E63_MOVE_OVERLAPPING", "0.3"))
MOVE_SEPARATE = float(os.environ.get("E63_MOVE_SEPARATE", "-0.1"))
POSTHOC = ("moved", "moved_shift")
SHIFT = float(os.environ.get("E63_SHIFT", "0.14"))       # D, fixed on the development seed (header)
SCENARIOS = ("rates", "pairs", "gain", "shift")
LAWS = {"A": ("rates", "A"), "B_rates": ("rates", "B"), "B_pairs": ("pairs", "B"), "B_gain": ("gain", "B")}
# name: scenario and block; the law of A is the same in every scenario, and `shift` uses the law of `pairs`
LAW_OF = {"rates": "B_rates", "pairs": "B_pairs", "gain": "B_gain", "shift": "B_pairs", "moved": "B_rates", "moved_shift": "B_rates"}
RATE_TOLERANCE, SPREAD_TOLERANCE = 0.02, 0.1
KINDS, CLASSES = ("pairs", "trios"), ("overlapping", "separate")


def cohort_module():
    spec = importlib.util.spec_from_file_location("dryad_cohort", HERE.parent / "real_data" / "dryad_cohort.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def template(census_root, fields_root, session):
    cohort = cohort_module()
    folder = Path(census_root) / session
    ripples = np.load(folder / "ripples.npz")
    kept = [int(u) for u in ripples["kept"]]
    rows = {int(r.split("\t")[0]): r.split("\t") for r in (folder / "units.tsv").read_text().splitlines()[1:]}
    census = json.loads((folder / "census.json").read_text())
    sets, _, _ = cohort.class_sets(Path(fields_root) / session, kept)
    return dict(rate_a=np.clip([float(rows[u][5]) for u in kept], 0.005, 0.95),
                rate_b=np.clip([float(rows[u][6]) for u in kept], 0.005, 0.95),
                n_a=int((ripples["epoch"] == "PRE").sum()), n_b=int((ripples["epoch"] == "POST").sum()),
                spread_a=float(census["PRE"]["active_sd_over_independent"]),
                spread_b=float(census["POST"]["active_sd_over_independent"]), sets=sets,
                index=sorted(cohort.SESSIONS).index(session), rat=cohort.SESSIONS[session][2])


def law(cells, sets, scenario, block, seed, index):
    """Couplings, loadings and the triplet term of a session; the same in A and B but for what the scenario changes."""
    rng = np.random.default_rng([seed, index, 7])
    background = np.triu(rng.normal(0.0, BACKGROUND_SD, (cells, cells)), 1)
    coupling = background + background.T
    loading = np.exp(rng.normal(-0.5 * LOADING_SD ** 2, LOADING_SD, cells))
    later = block == "B"
    pair_after, separate_after = PAIR_AFTER, SEPARATE_AFTER
    if BY_SESSION is not None:
        pair_after, separate_after = BY_SESSION[sorted(cohort_module().SESSIONS)[index]]
    pair = pair_after if later and scenario in ("pairs", "shift") else PAIR
    for i, j in sets["pairs_overlapping"]:
        coupling[i, j] += pair
        coupling[j, i] += pair
    if later and scenario in ("pairs", "shift") and separate_after != 0:
        for i, j in sets["pairs_separate"]:
            coupling[i, j] += separate_after
            coupling[j, i] += separate_after
    partners = [([], []) for _ in range(cells)]
    for i, j, k in sets["trios_overlapping"]:
        for a, b, c in ((i, j, k), (j, i, k), (k, i, j)):
            partners[a][0].append(b)
            partners[a][1].append(c)
    partners = [(np.array(p[0], dtype=np.int64), np.array(p[1], dtype=np.int64)) for p in partners]
    return dict(coupling=coupling, loading=loading, partners=partners, sigma=0.5, triplet=TRIPLET)


def simulate(rng, n, h, model, sweeps=SWEEPS):
    cells = len(h)
    base = h[None, :] + (model["sigma"] * model["loading"])[None, :] * rng.normal(0.0, 1.0, n)[:, None]
    x = (rng.random((n, cells)) < expit(base)).astype(np.float64)
    for _ in range(sweeps):
        for i in rng.permutation(cells):
            field = base[:, i] + x @ model["coupling"][:, i]
            first, second = model["partners"][i]
            if len(first) and model["triplet"] != 0:
                field = field + model["triplet"] * (x[:, first] * x[:, second]).sum(1)
            x[:, i] = rng.random(n) < expit(field)
    return x.astype(np.int8)


def plant(rng, x, trios, target, edges):
    """Raise the three-way term of the given trios in the events x by about `target` (see the header).

    Returns the changed events and the number of moves.  Each move leaves the three pair margins of its trio and
    the participation of every cell as they were, and adds one to the trio's count of (1,1,1).
    """
    x = np.array(x, dtype=np.int8)
    n, made = len(x), 0
    for t in rng.permutation(len(trios)):
        cells = np.asarray(trios[t], dtype=np.int64)
        code = x[:, cells[0]] + 2 * x[:, cells[1]] + 4 * x[:, cells[2]]
        counts = np.bincount(code, minlength=8).astype(float)
        if counts[7] == 0:
            continue
        weight = (1.0 / counts[7]) / (1.0 / counts).sum() if counts.min() > 0 else 1.0
        spawning = np.flatnonzero((code == 7) & (rng.random(n) < target * weight))
        if len(spawning) == 0:
            continue
        level = np.searchsorted(edges, x.sum(1) - x[:, cells].sum(1), side="right")     # not changed by the moves
        for event in spawning:
            turn = int(rng.integers(3))
            k, a, b = cells[turn], cells[(turn + 1) % 3], cells[(turn + 2) % 3]
            own, first, second = x[:, k], x[:, a], x[:, b]
            wanted = ((own == 0) & (first == 1) & (second == 1), (own == 0) & (first == 0) & (second == 0),
                      (own == 1) & (first == 1) & (second == 0), (own == 1) & (first == 0) & (second == 1))
            chosen = []
            for mask in wanted:
                inside = np.flatnonzero(mask & (level == level[event]))
                pool = inside if len(inside) else np.flatnonzero(mask)
                if len(pool) == 0:
                    break
                chosen.append(int(pool[rng.integers(len(pool))]))
            if len(chosen) < 4:
                continue
            x[chosen[0], k], x[chosen[1], k], x[chosen[2], k], x[chosen[3], k] = 1, 1, 0, 0
            made += 1
    return x, made


def plant_pairs(rng, x, classes, edges):
    """Move the pair term of classes of pairs in the events x; `classes` is a list of (pairs, target).

    A move for a pair switches one of its two cells on in an event where only the other is active and off in an
    event where only it is active (the pair's count of (1,1) rises by one), or the reverse for a negative target
    (off where both are active, on where both are silent).  Both participations stay as they were.  For each
    event in which the pair is fully active a move is made with probability |target| w (w = (1/n11) / sum over the
    four patterns of 1/n), so that the expected change of the pair's log odds ratio is `target`; the two events
    are taken, where the stratum has them, from the stratum of that event (the number of other active cells).
    """
    x = np.array(x, dtype=np.int8)
    n, made = len(x), 0
    todo = [(pair, target) for pairs, target in classes if target != 0 for pair in np.asarray(pairs, dtype=np.int64)]
    for index in rng.permutation(len(todo)):
        cells, target = todo[index]
        code = x[:, cells[0]] + 2 * x[:, cells[1]]
        counts = np.bincount(code, minlength=4).astype(float)
        if counts[3] == 0:
            continue
        weight = (1.0 / counts[3]) / (1.0 / counts).sum() if counts.min() > 0 else 1.0
        spawning = np.flatnonzero((code == 3) & (rng.random(n) < abs(target) * weight))
        if len(spawning) == 0:
            continue
        level = np.searchsorted(edges, x.sum(1) - x[:, cells].sum(1), side="right")
        for event in spawning:
            turn = int(rng.integers(2))
            k, a = cells[turn], cells[1 - turn]
            own, other = x[:, k], x[:, a]
            if target > 0:
                wanted = ((own == 0) & (other == 1), (own == 1) & (other == 0))          # switched on, switched off
            else:
                wanted = ((own == 0) & (other == 0), (own == 1) & (other == 1))
            chosen = []
            for mask in wanted:
                inside = np.flatnonzero(mask & (level == level[event]))
                pool = inside if len(inside) else np.flatnonzero(mask)
                if len(pool) == 0:
                    break
                chosen.append(int(pool[rng.integers(len(pool))]))
            if len(chosen) < 2:
                continue
            x[chosen[0], k], x[chosen[1], k] = 1, 0
            made += 1
    return x, made


def spread_of(x):
    """Standard deviation of the number of active cells over that of independent cells with the same rates."""
    rate = x.mean(0)
    return float(x.sum(1).std() / np.sqrt((rate * (1 - rate)).sum()))


def calibrate(rng, target, spread, model, n=12000, rounds=12):
    """h, and sigma when a spread is given, so that the simulated participations and spread are the recorded ones."""
    h, model = logit(target), dict(model)
    for _ in range(rounds):
        x = simulate(rng, n, h, model, sweeps=15)
        rate = np.clip(x.mean(0), 1 / n, 1 - 1 / n)
        h = h + 0.8 * (logit(target) - logit(rate))
        if spread is not None:
            excess, wanted = max(spread_of(x) ** 2 - 1, 1e-3), max(spread ** 2 - 1, 1e-3)
            model["sigma"] = float(np.clip(model["sigma"] * (wanted / excess) ** 0.4, 0.02, 4.0))
    for _ in range(3):      # sigma is now fixed: the participations once more, on more events
        rate = np.clip(simulate(rng, 2 * n, h, model, sweeps=15).mean(0), 0.5 / n, 1 - 0.5 / n)
        h = h + 0.8 * (logit(target) - logit(rate))
    x = simulate(rng, 3 * n, h, model)
    return h, model["sigma"], float(np.abs(x.mean(0) - target).max()), spread_of(x)


def calibrate_law(arguments):
    census_root, fields_root, session, name, seed, out = arguments
    started = time.time()
    folder = Path(out) / "calibration"
    t = template(census_root, fields_root, session)
    scenario, block = LAWS[name]
    cells, index = len(t["rate_a"]), t["index"]
    model = law(cells, t["sets"], scenario, block, seed, index)
    target, spread = (t["rate_a"], t["spread_a"]) if block == "A" else (t["rate_b"], t["spread_b"])
    if name == "B_gain":        # no spread is asked for: sigma is a multiple of the one that `rates` needed
        model["sigma"] = GAIN_FACTOR * json.loads((folder / f"{session}__B_rates.json").read_text())["sigma"]
        spread = None
    h, sigma, error, got = calibrate(np.random.default_rng([seed, index, 50 + list(LAWS).index(name)]), target, spread, model)
    report = dict(session=session, law=name, cells=cells, shift=SHIFT,
                  pair_after=PAIR_AFTER if BY_SESSION is None else BY_SESSION[session][0],
                  separate_after=SEPARATE_AFTER if BY_SESSION is None else BY_SESSION[session][1],
                  seed=seed, sigma=sigma, rate_error=error,
                  spread=got, spread_wanted=spread, seconds=round(time.time() - started, 1))
    np.save(folder / f"{session}__{name}.npy", h)
    (folder / f"{session}__{name}.json").write_text(json.dumps(report) + "\n")
    return report


def pooled(entries, kind):
    """One shift per class over the given sessions, as `hippocampal_study.combine_sessions`; and the difference.

    Returns [difference, its se, overlapping shift, its se, separate shift, its se], or None without both classes.
    """
    fits = {}
    for cls in CLASSES:
        parts, labels_a, labels_b, offset = [], [], [], np.zeros(2, dtype=np.int64)
        for entry in entries:  # events keep one numbering over the sessions, so that the two classes are aligned
            part = entry["parts"].get((kind, cls))
            if part is not None:
                parts.append(part)
                labels_a.append(np.arange(entry["n_a"]) + offset[0])
                labels_b.append(np.arange(entry["n_b"]) + offset[1])
            offset = offset + np.array([entry["n_a"], entry["n_b"]])
        if parts:
            fits[cls] = combine(parts, labels_a, labels_b)
    if len(fits) < 2:
        return None
    d = contrast(fits["overlapping"], fits["separate"])
    return [d["delta"], d["se"], fits["overlapping"].delta, fits["overlapping"].se, fits["separate"].delta, fits["separate"].se]


CACHE = {}


def setting(census_root, fields_root, out, sessions):
    key = (str(census_root), str(fields_root), str(out), tuple(sessions))
    if key not in CACHE:
        entry, folder = {}, Path(out) / "calibration"
        for session in sessions:
            c = {name: (np.load(folder / f"{session}__{name}.npy"),
                        float(json.loads((folder / f"{session}__{name}.json").read_text())["sigma"])) for name in LAWS}
            entry[session] = (template(census_root, fields_root, session), c)
        e62 = cohort_module().module("hippocampal_study")
        CACHE[key] = (entry, e62.edges_of)
    return CACHE[key]


def run_cohort(arguments):
    census_root, fields_root, out, seed, sessions, scenario, replicate = arguments
    started = time.time()
    info, edges_of = setting(census_root, fields_root, out, sessions)
    entries, moves, pair_moves = {}, {}, {}
    for session in sessions:
        t, c = info[session]
        cells, sets, index = len(t["rate_a"]), t["sets"], t["index"]
        model_a, model_b = law(cells, sets, scenario, "A", seed, index), law(cells, sets, scenario, "B", seed, index)
        (h_a, model_a["sigma"]), (h_b, model_b["sigma"]) = c["A"], c[LAW_OF[scenario]]
        rng = np.random.default_rng([seed, index, 9, (SCENARIOS + POSTHOC).index(scenario), replicate])
        x_a, x_b = simulate(rng, t["n_a"], h_a, model_a), simulate(rng, t["n_b"], h_b, model_b)
        if scenario in POSTHOC:
            x_b, pair_moves[session] = plant_pairs(rng, x_b, [(sets["pairs_overlapping"], MOVE_OVERLAPPING),
                                                              (sets["pairs_separate"], MOVE_SEPARATE)], edges_of(x_a.sum(1), x_b.sum(1)))
        if scenario in ("shift", "moved_shift"):
            x_b, moves[session] = plant(rng, x_b, sets["trios_overlapping"], SHIFT, edges_of(x_a.sum(1), x_b.sum(1)))
        size_a, size_b = x_a.sum(1), x_b.sum(1)
        edges = edges_of(size_a, size_b)
        parts = {(kind, cls): prepare(x_a, x_b, sets[f"{kind}_{cls}"], size_a, size_b, edges)
                 for kind in KINDS for cls in CLASSES if len(sets[f"{kind}_{cls}"])}
        entries[session] = dict(parts=parts, n_a=t["n_a"], n_b=t["n_b"])
    groups = cohort_module().groups_of(sessions)
    row = dict(scenario=scenario, replicate=replicate, shift=SHIFT, seed=seed, moves=moves)
    if scenario in POSTHOC:
        row.update(pair_moves=pair_moves, move_overlapping=MOVE_OVERLAPPING, move_separate=MOVE_SEPARATE)
    for kind in KINDS:
        done = {}

        def fit(members):
            if tuple(members) not in done:
                done[tuple(members)] = pooled([entries[s] for s in members], kind)
            return done[tuple(members)]

        row[kind] = dict(cohort=fit(sessions), groups={name: fit(members) for name, members in groups.items()},
                         sessions={s: fit([s]) for s in sessions})
    row["seconds"] = round(time.time() - started, 1)
    return row


def run(census_root, fields_root, out, replicates, seed, workers, only=None, scenarios=SCENARIOS):
    out = Path(out)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    recorded = sorted(p.name for p in Path(census_root).iterdir() if (p / "ripples.npz").exists())
    sessions = [s for s in recorded if only is None or s in only]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        mapper = map if workers <= 1 else pool.map
        for phase in ([name for name in LAWS if name != "B_gain"], ["B_gain"]):      # B_gain needs B_rates
            missing = [(census_root, fields_root, s, name, seed, str(out)) for s in sessions for name in phase
                       if not (out / "calibration" / f"{s}__{name}.json").exists()]
            for r in mapper(calibrate_law, missing):
                print("calibrated", r["session"], r["law"], r["seconds"], "s; sigma", round(r["sigma"], 3), "rate error",
                      round(r["rate_error"], 4), "spread", round(r["spread"], 3), "wanted", r["spread_wanted"], flush=True)
        have = set()
        for scenario in scenarios:
            path = out / f"cohorts_{scenario}.jsonl"
            if path.exists():
                have |= {(scenario, json.loads(line)["replicate"]) for line in path.read_text().splitlines()}
        tasks = [(census_root, fields_root, str(out), seed, sessions, scenario, r) for r in range(replicates) for scenario in scenarios
                 if (scenario, r) not in have]
        for row in mapper(run_cohort, tasks):
            with open(out / f"cohorts_{row['scenario']}.jsonl", "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row["scenario"], row["replicate"], row["seconds"], "s", flush=True)


def rates_of(values):
    """Summary of a list of [difference, se, ...] (None where a fit was not possible)."""
    usable = [v for v in values if v is not None and np.isfinite(v[0]) and np.isfinite(v[1]) and v[1] > 0]
    if not usable:
        return dict(usable=0, of=len(values))
    d, se = np.array([v[0] for v in usable]), np.array([v[1] for v in usable])
    return dict(usable=len(usable), of=len(values), mean=float(d.mean()), sd=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                se=float(se.mean()), reject=float((np.abs(d / se) >= 1.96).mean()), positive=float((d > 0).mean()),
                z_sd=float((d / se).std(ddof=1)) if len(d) > 1 else float("nan"))


def summarise(out):
    out = Path(out)
    rule_d2 = cohort_module().rule_d2          # the study's own function
    laws = [json.loads(path.read_text()) for path in sorted((out / "calibration").glob("*.json"))]
    report = dict(scenarios={}, calibration={}, shift=sorted({c["shift"] for c in laws}), seed=sorted({c["seed"] for c in laws}),
                  pair_after=sorted({c.get("pair_after", 0.3) for c in laws}), separate_after=sorted({c.get("separate_after", 0.0) for c in laws}))
    for c in laws:
        report["calibration"].setdefault(c["session"], {})[c["law"]] = [round(c["sigma"], 3), round(c["rate_error"], 4), round(c["spread"], 3),
                                                                        c["spread_wanted"]]
    report["worst_rate_error"] = max(c["rate_error"] for c in laws)
    report["worst_spread_error"] = max(abs(c["spread"] - c["spread_wanted"]) for c in laws if c["spread_wanted"] is not None)
    for scenario in SCENARIOS + POSTHOC:
        path = out / f"cohorts_{scenario}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        entry = dict(cohorts=len(rows), seconds=float(np.mean([r["seconds"] for r in rows])))
        for kind in KINDS:
            entry[kind] = dict(cohort=rates_of([r[kind]["cohort"] for r in rows]),
                               groups={g: rates_of([r[kind]["groups"][g] for r in rows]) for g in rows[0][kind]["groups"]},
                               sessions={s: rates_of([r[kind]["sessions"][s] for r in rows]) for s in rows[0][kind]["sessions"]},
                               overlapping=rates_of([r[kind]["cohort"][2:4] if r[kind]["cohort"] else None for r in rows]),
                               separate=rates_of([r[kind]["cohort"][4:6] if r[kind]["cohort"] else None for r in rows]))
        verdicts = []
        for r in rows:
            c = r["trios"]["cohort"]
            by_group = {g: (v[0] if v is not None else float("nan")) for g, v in r["trios"]["groups"].items()}
            verdicts.append(rule_d2(c[0], c[1], by_group) if c is not None else (False, False, 0))
        entry["rule"] = dict(replicated=float(np.mean([v[0] for v in verdicts])), against=float(np.mean([v[1] for v in verdicts])),
                             rats_positive={str(k): int(sum(v[2] == k for v in verdicts)) for k in range(6)})
        report["scenarios"][scenario] = entry
    got = report["scenarios"]
    nulls = [s for s in ("rates", "pairs", "gain") if s in got]
    if len(nulls) == 3 and "shift" in got:
        h0 = report["worst_rate_error"] <= RATE_TOLERANCE and report["worst_spread_error"] <= SPREAD_TOLERANCE
        h1 = all(got[s]["trios"]["cohort"]["reject"] <= 0.08 for s in nulls)
        h2 = all(abs(got[s]["trios"]["cohort"]["mean"]) <= 0.03 for s in nulls)
        h3 = got["shift"]["trios"]["cohort"]["reject"] >= 0.80
        h4 = all(got[s]["rule"]["replicated"] <= 0.05 for s in nulls)
        h5 = (all(got[s]["pairs"]["cohort"]["reject"] >= 0.90 for s in ("pairs", "shift"))
              and all(got[s]["pairs"]["cohort"]["reject"] <= 0.11 for s in ("rates", "gain")))
        report["expectations"] = dict(h0=bool(h0), h1=bool(h1), h2=bool(h2), h3=bool(h3), h4=bool(h4), h5=bool(h5),
                                      passed=bool(h0 and h1 and h2 and h3 and h4))
    (out / "e63_summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def comparator_session(arguments):
    census_root, fields_root, out, session, replicate = arguments
    from dvc_package.modulation.loglinear_pooled import pooled_loglinear_change
    started = time.time()
    info, _ = setting(census_root, fields_root, out, (session,))
    t, c = info[session]
    seed = int(json.loads((Path(out) / "calibration" / f"{session}__A.json").read_text())["seed"])
    cells, sets, index = len(t["rate_a"]), t["sets"], t["index"]
    model_a, model_b = law(cells, sets, "rates", "A", seed, index), law(cells, sets, "rates", "B", seed, index)
    (h_a, model_a["sigma"]), (h_b, model_b["sigma"]) = c["A"], c["B_rates"]
    rng = np.random.default_rng([seed, index, 11, replicate])
    x_a, x_b = simulate(rng, t["n_a"], h_a, model_a), simulate(rng, t["n_b"], h_b, model_b)
    row = dict(session=session, replicate=replicate, n_a=t["n_a"], n_b=t["n_b"], cells=cells)
    for cls in CLASSES:
        fit = pooled_loglinear_change(x_a, x_b, sets[f"trios_{cls}"])
        row[cls] = [fit.delta, fit.se, fit.cells]
    row["seconds"] = round(time.time() - started, 1)
    return row


def comparator(census_root, fields_root, out, replicates, workers, sessions):
    """Post hoc: what the log-linear comparator gives on sessions simulated without a change (scenario `rates`)."""
    out = Path(out)
    tasks = [(census_root, fields_root, str(out), s, r) for s in sessions for r in range(replicates)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list((map if workers <= 1 else pool.map)(comparator_session, tasks))
    report = {}
    for s in sessions:
        entry = dict(n_a=next(r["n_a"] for r in rows if r["session"] == s), n_b=next(r["n_b"] for r in rows if r["session"] == s),
                     cells=next(r["cells"] for r in rows if r["session"] == s))
        for cls in CLASSES:
            got = np.array([r[cls][:2] for r in rows if r["session"] == s and np.isfinite(r[cls][0]) and np.isfinite(r[cls][1]) and r[cls][1] > 0])
            if len(got):
                entry[cls] = dict(usable=len(got), mean=float(got[:, 0].mean()), sd=float(got[:, 0].std(ddof=1)), se=float(got[:, 1].mean()),
                                  reject=float((np.abs(got[:, 0] / got[:, 1]) >= 1.96).mean()),
                                  cells=float(np.mean([r[cls][2] for r in rows if r["session"] == s])))
        report[s] = entry
    (out / "e63_comparator_without_change.json").write_text(json.dumps(dict(replicates=replicates, sessions=report), indent=1) + "\n")
    return report


def with_factor(out, factor):
    """Post hoc: the cohort's test and the rule (d2) on the saved cohorts, the standard error times `factor`."""
    out = Path(out)
    rule_d2 = cohort_module().rule_d2
    report = dict(factor=factor, scenarios={})
    for scenario in SCENARIOS + POSTHOC:
        path = out / f"cohorts_{scenario}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        d = np.array([r["trios"]["cohort"][0] for r in rows])
        se = np.array([r["trios"]["cohort"][1] for r in rows])
        verdicts = [rule_d2(r["trios"]["cohort"][0], r["trios"]["cohort"][1] * factor,
                            {g: (v[0] if v is not None else float("nan")) for g, v in r["trios"]["groups"].items()}) for r in rows]
        report["scenarios"][scenario] = dict(
            cohorts=len(rows), mean=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se.mean()), sd_over_se=float(d.std(ddof=1) / se.mean()),
            reject=float((np.abs(d / (se * factor)) >= 1.96).mean()), reject_positive=float((d / (se * factor) >= 1.96).mean()),
            coverage_of_own_mean=float((np.abs(d - d.mean()) <= 1.96 * se * factor).mean()),
            rule_replicated=float(np.mean([v[0] for v in verdicts])), rule_against=float(np.mean([v[1] for v in verdicts])))
    (out / f"e63_factor_{factor:g}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    if sys.argv[1] == "factor":
        print(json.dumps(with_factor(sys.argv[2], float(sys.argv[3])), indent=1))
    elif sys.argv[1] == "comparator":
        print(json.dumps(comparator(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7].split("+")), indent=1))
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]) if len(sys.argv) > 7 else 1,
            set(sys.argv[8].split("+")) if len(sys.argv) > 8 and sys.argv[8] != "all" else None,
            tuple(sys.argv[9].split("+")) if len(sys.argv) > 9 and sys.argv[9] != "all" else SCENARIOS)
    elif sys.argv[1] == "summary":
        r = summarise(sys.argv[2])
        short = {s: dict(cohorts=e["cohorts"], seconds=round(e["seconds"], 1), rule=e["rule"],
                         **{k: {q: (round(v, 4) if isinstance(v, float) else v) for q, v in e[k]["cohort"].items()} for k in KINDS})
                 for s, e in r["scenarios"].items()}
        print(json.dumps(dict(shift=r["shift"], seed=r["seed"], worst_rate_error=r["worst_rate_error"],
                              worst_spread_error=r["worst_spread_error"], scenarios=short, expectations=r.get("expectations")), indent=1))
    else:
        raise SystemExit(__doc__)
