"""E69: does a vine-edge modulation that changes predict whole held-out ripples beyond a replay model?

This 24 November protocol compares persistent and changing modulations of the same replay and pairwise background. Both classes can have distinct baseline associations. Successive fits add a shared change and then a class-specific difference. Penalties are selected on training data and the models are scored on the same held-out ripples. Simulations test whether the last comparison confuses a common shift with a difference between classes. The original comparison and its development are recorded separately in the study ledger.

Population: the place cells of a recording (a session of the second cohort, a hemisphere of the first), in the
order of the peaks of their maps; the ripples of the rest before (A) and after (B) the experience; the classes of
E62 and E64 (all overlapping and separate pairs and trios).

The law of a ripple is the product of each place cell's probability given the place cells before it:

    logit P(x_i = 1 | x_1..x_{i-1}) = o_i + h_i + sum_{j<i} (J_ij + u (a O_ij + b S_ij)) x_j
                                       + (d_O + u (c + Delta)) N_O(i) + (d_S + u c) N_S(i)

  o_i    the replay model's own logit of x_i given x_1..x_{i-1}: the law of `real_data/dryad_posthoc_content.py`
         for the ripple's rest, fitted to the training ripples of that rest, with its posterior over stretch and
         strength updated cell by cell.  With nothing else, the product is the replay law's probability of the
         ripple.  The replay model is the same in every variant.
  u      0 before, 1 after.
  h      a small correction of each cell (weak penalty, standard deviation 10).
  J      pair edges beyond replay, the same in both rests (ridge, standard deviation TAU);  a, b  the change of the
         edges of overlapping and of separate pairs (weak penalty).
  N_O(i) the number of pairs of active cells before i that close an overlapping trio with i, in units of its
         spread over the RECORDED ripples of the recording (a constant of the recording, the same for simulated
         versions of it, so that a change that is the same for both classes in these units is the null of Delta
         in the check as in the study); N_S the same for separate trios.  d_O, d_S: each class's modulation of its
         edges by a third cell (its starting strength); c: their shared change; Delta: how much more the
         overlapping class changes than the separate one.  Weak penalty on all of them.

Variants, nested, each fitted by penalised maximum likelihood to the training ripples:
  replay       o and h only
  pairs        + J, a, b
  persistent   + d_O, d_S                     (classes with their own starting strengths, no change)
  shared       + c                            (both classes change by the same amount)
  class        + Delta                        (one more parameter than `shared`)
Comparisons (held-out log probability of whole ripples of the first variant minus that of the second):
  class_change  class - shared         THE TEST: do the classes change differently, with presence and a shared
                                       change in both models (the counterpart of E64's contrast)
  time_varying  shared - persistent    does a modulation that changes predict beyond one that persists
  modulation    persistent - pairs;    pairs  pairs - replay
Held-out: the blocks and folds of E67 (ten consecutive blocks per rest with equal numbers of ripples, block m in
fold m mod 5).  For each fold left out: the two replay laws are fitted to the other folds' ripples of each rest;
TAU is chosen from (0.1, 0.2, 0.4, 0.8) by `pairs` fitted to three of the four training folds and scored on the
fourth (the fold after the one left out), so that nothing of the scored fold enters the choice; all variants are
then fitted to the four training folds and score the fold left out.  Standard errors: blocks as units within each
rest; recordings add.

    e69_replay_vine.py study CONTENT_FOLDER OUT_FOLDER NAME [CENSUS_ROOT|-] [class|common]
    e69_replay_vine.py check CONTENT_FOLDER CENSUS_ROOT OUT_FOLDER SCENARIO REPLICATES SEED [WORKERS] [GROUP] [class|common]
    e69_replay_vine.py summary OUT_FOLDER

CONTENT_FOLDER holds the recordings stored by `dryad_posthoc_content.py fit` (second cohort, NAME a session,
CENSUS_ROOT for the times of its ripples) or by `hippocampal_posthoc_content.py fit` (first cohort, NAME
`<session>_<hemisphere>`).

`check`: cohorts of the nine sessions of the second cohort in which the place cells of every ripple are drawn
again: a stretch and strength from the ripple's posterior under the stored law of its rest, then each place cell
in the order of the fields with logit p_i(stretch, strength) + (d_O + u c_O) N_O(i) + (d_S + u c_S) N_S(i), with
the counts in units of their spread in the recording.  Scenarios:
  replay   d_O = d_S = c_O = c_S = 0          (replay only)
  apart    d_O = 0.20, d_S = 0, c_O = c_S = 0.15   (the classes start apart and move together)
  class    d_O = d_S = 0, c_O = 0.15, c_S = 0       (only the overlapping class changes)

`check` with GROUP (a key of GROUPS; default `cohort`, the nine second-cohort sessions) draws the cohorts from the
stored laws of that group's recordings instead.  The 24 November extension tests the same three scenarios in Achilles's four hemispheres (`achilles`, CONTENT_FOLDER the first
cohort's stored hemispheres, CENSUS_ROOT unused), the power check at Achilles's sizes that section 69 lacked.
Expectations for it, written before it was run: (e1) and (e2) as below; (e3) is reported, not required, since the
point is to learn the power at these sizes.

Expectations, written before anything was run.
 Check (24 cohorts per scenario)
  (e1) `replay`: `class_change` and `time_varying` each reach z 1.96 in at most 8% of cohorts.
  (e2) `apart`: `class_change` reaches z 1.96 in at most 8% of cohorts; `time_varying` in at least
       half.
  (e3) `class`: `class_change` reaches z 1.96 in at least half.  If it does not, the study cannot show an absence.
 Study (second cohort, nine sessions; first cohort, Achilles's four hemispheres, and the other first-cohort
 hemispheres reported beside them)
  (w1) the second cohort's `class_change` is positive with z at least 1.96.  The replay-only simulations predicted no positive residual here.
  (w2) Achilles's `class_change` is positive with z at least 1.96.  The replay-only simulations left open a positive residual here.
  (w3) `time_varying` in each, reported.
 A positive (w1) or (w2) means that a changing vine-edge modulation of the overlapping trios predicts whole held-out
 ripples beyond the replay model and beyond a change shared with the separate trios.
"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

TAUS, WEAK, BLOCKS, FOLDS = (0.1, 0.2, 0.4, 0.8), 10.0, 10, 5
VARIANTS = {"replay": (), "pairs": ("a", "b"), "persistent": ("a", "b", "d_o", "d_s"),
            "shared": ("a", "b", "d_o", "d_s", "c"), "class": ("a", "b", "d_o", "d_s", "c", "delta")}
COMPARISONS = {"class_change": ("class", "shared"), "time_varying": ("shared", "persistent"), "modulation": ("persistent", "pairs"),
               "pairs": ("pairs", "replay")}
SCENARIOS = {"replay": dict(d_o=0.0, d_s=0.0, c_o=0.0, c_s=0.0), "apart": dict(d_o=0.20, d_s=0.0, c_o=0.15, c_s=0.15),
             "class": dict(d_o=0.0, d_s=0.0, c_o=0.15, c_s=0.0)}
SECOND = ("RAT1_SESS1", "RAT1or4_SESS", "RAT2_SESS1", "RAT2_SESS2", "RAT3_SESS1", "RAT3_SESS2", "RAT4_SESS2", "RAT5_SESS1", "RAT5_SESS2")
GROUPS = {"cohort": SECOND, "rat1": ("RAT1or4_SESS",), "rat2": ("RAT2_SESS1", "RAT2_SESS2"), "rat3": ("RAT3_SESS1", "RAT3_SESS2"),
          "rat4": ("RAT1_SESS1", "RAT4_SESS2"), "rat5": ("RAT5_SESS1", "RAT5_SESS2"),
          "achilles": ("Achilles-10252013_left", "Achilles-10252013_right", "Achilles-11012013_left", "Achilles-11012013_right"),
          "cicero": ("Cicero-09012014_left", "Cicero-09012014_right", "Cicero-09102014_left", "Cicero-09102014_right", "Cicero-09172014_left",
                     "Cicero-09172014_right"),
          "gatsby": ("Gatsby-08022013_left", "Gatsby-08022013_right", "Gatsby-08282013_left")}


def module(name, folder="real_data"):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / folder / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


content = module("dryad_posthoc_content")


def recording(content_folder, name, census_root=None):
    """The place cells of a stored recording: ripples of the two rests in time order, classes, field order."""
    d, laws = content.stored(content_folder, name)
    peak_file = Path(content_folder) / "sessions" / f"{name}_peak.npy"
    if peak_file.exists():
        peak = np.load(peak_file)
    elif census_root is not None:
        peak = np.load(Path(census_root) / name / "ripples.npz")["peak"]
    else:
        raise ValueError("no ripple times for " + name)
    assert len(peak) == len(d["x"])
    keep = np.flatnonzero((d["epoch"] == "PRE") | (d["epoch"] == "POST"))
    keep = keep[np.argsort(peak[keep], kind="stable")]
    cells = np.asarray(d["cells"])
    position = np.full(d["x"].shape[1], -1)
    position[cells] = np.arange(len(cells))
    sets = {key: position[np.asarray(v, dtype=np.int64)] for key, v in d["sets"].items()}
    assert all((v >= 0).all() for v in sets.values())
    order = np.argsort(np.asarray(d["rates"])[:, np.asarray(d["kinds"]) == 0].argmax(1), kind="stable")
    return dict(name=name, x=d["x"][keep][:, cells].astype(np.int8), epoch=d["epoch"][keep], sets=sets, order=order,
                rates=d["rates"], kinds=d["kinds"], window=d["window"], laws=laws)


def features(x, sets, order):
    """In the cells' order: the cells, active earlier partners in overlapping and separate pairs, and active earlier
    pairs closing an overlapping or separate trio with the cell (raw counts)."""
    n = x.shape[1]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    ordered = np.asarray(x)[:, order].astype(np.float64)
    out = dict(x=ordered)
    for cls, short in (("overlapping", "o"), ("separate", "s")):
        later = np.zeros((n, n))
        members = rank[np.asarray(sets[("pairs", cls)], dtype=np.int64).reshape(-1, 2)]
        later[members.max(1), members.min(1)] = 1.0
        out["x" + short] = ordered @ later.T
        trios = np.sort(rank[np.asarray(sets[("trios", cls)], dtype=np.int64).reshape(-1, 3)], axis=1)
        total = np.zeros_like(ordered)
        for start in range(0, len(trios), 2000):
            part = trios[start:start + 2000]
            both = ordered[:, part[:, 0]] * ordered[:, part[:, 1]]
            last = np.zeros((len(part), n))
            last[np.arange(len(part)), part[:, 2]] = 1.0
            total += both @ last
        out["n" + short] = total
    return out


def scale_trio_features(raw, training, unit_rule="class"):
    """Express active-pair counts in training-fold units for both training and test ripples."""
    scales = {key: float(raw[key][training].std()) or 1.0 for key in ("no", "ns")}
    if unit_rule == "common":
        scales = dict.fromkeys(scales, max(scales.values()))
    elif unit_rule != "class":
        raise ValueError("unit_rule must be class or common")
    scaled = dict(raw)
    for key, scale in scales.items():
        scaled[key] = raw[key] / scale
    return scaled, scales


def replay_logits(law, x_ordered, order):
    """The replay law's logit of every cell given the cells before it, in the cells' order: (events, cells)."""
    table = law.table[:, order]                                                         # (stretch and strength, cells)
    log_p, log_q = np.log(table), np.log1p(-table)
    log_weight = np.log(np.maximum(law.weight.ravel(), 1e-300))
    running = np.repeat(log_weight[None, :], len(x_ordered), axis=0)
    out = np.empty(x_ordered.shape)
    for i in range(x_ordered.shape[1]):
        top = running.max(1, keepdims=True)
        weight = np.exp(running - top)
        q = np.clip((weight @ table[:, i]) / weight.sum(1), 1e-9, 1 - 1e-9)
        out[:, i] = np.log(q) - np.log1p(-q)
        running += np.where(x_ordered[:, i:i + 1] > 0, log_p[None, :, i], log_q[None, :, i])
    return out


def problem(f, rows, o, u, variant, tau):
    """Penalised negative log likelihood of a variant on the ripples `rows` and its gradient."""
    x = f["x"][rows]
    n = x.shape[1]
    lower = np.tril_indices(n, -1)
    pairs = len(lower[0]) if variant != "replay" else 0
    names = VARIANTS[variant]
    uu = u[rows][:, None]
    columns = {"a": uu * f["xo"][rows], "b": uu * f["xs"][rows], "d_o": f["no"][rows], "d_s": f["ns"][rows],
               "c": uu * (f["no"][rows] + f["ns"][rows]), "delta": uu * f["no"][rows]}
    precision = np.r_[np.full(n, 1 / WEAK ** 2), np.full(pairs, 1 / tau ** 2), np.full(len(names), 1 / WEAK ** 2)]
    base = o[rows]

    def logits(theta):
        z = base + theta[:n][None, :]
        if pairs:
            coupling = np.zeros((n, n))
            coupling[lower] = theta[n:n + pairs]
            z = z + x @ coupling.T
        for k, name in enumerate(names):
            z = z + theta[n + pairs + k] * columns[name]
        return z

    def objective(theta):
        z = logits(theta)
        value = float((np.logaddexp(0.0, z) - x * z).sum() + 0.5 * (precision * theta ** 2).sum())
        r = expit(z) - x
        gradient = np.empty_like(theta)
        gradient[:n] = r.sum(0)
        if pairs:
            gradient[n:n + pairs] = (r.T @ x)[lower]
        for k, name in enumerate(names):
            gradient[n + pairs + k] = float((r * columns[name]).sum())
        return value, gradient + precision * theta

    return objective, logits, n, pairs


def fit(f, rows, o, u, variant, tau, start=None):
    objective, logits, n, pairs = problem(f, rows, o, u, variant, tau)
    theta = np.zeros(n + pairs + len(VARIANTS[variant])) if start is None else np.asarray(start, dtype=np.float64)
    found = minimize(objective, theta, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000, ftol=1e-11, gtol=1e-6))
    return found.x


def widen(theta, n, source, target):
    """The parameters of a fitted variant as a start for a variant that contains it."""
    pairs_source = 0 if source == "replay" else n * (n - 1) // 2
    pairs_target = 0 if target == "replay" else n * (n - 1) // 2
    out = np.zeros(n + pairs_target + len(VARIANTS[target]))
    out[:n + pairs_source] = theta[:n + pairs_source]
    for k, name in enumerate(VARIANTS[source]):
        out[n + pairs_target + VARIANTS[target].index(name)] = theta[n + pairs_source + k]
    return out


def log_probability(f, rows, o, u, variant, tau, theta):
    _, logits, _, _ = problem(f, rows, o, u, variant, tau)
    z = logits(theta)
    x = f["x"][rows]
    return (x * z - np.logaddexp(0.0, z)).sum(1)


def blocks_of(epoch):
    """Block of every ripple (ripples in time order): BLOCKS consecutive blocks per rest, numbered rest by rest; and its fold."""
    block = np.full(len(epoch), -1)
    for k, rest in enumerate(("PRE", "POST")):
        rows = np.flatnonzero(epoch == rest)
        block[rows] = k * BLOCKS + (np.arange(len(rows)) * BLOCKS) // max(len(rows), 1)
    return block, np.where(block >= 0, block % FOLDS, -1)


def held_out(rec, x=None, unit_rule="class"):
    """Held-out log probability of every ripple under each variant, and what was chosen along the way.

    x: place cells to use in place of the recorded ones (a simulated recording), in the recording's cell order.
    """
    epoch, order = rec["epoch"], rec["order"]
    x = rec["x"] if x is None else x
    raw = features(x, rec["sets"], order)
    u = (epoch == "POST").astype(np.float64)
    block, fold = blocks_of(epoch)
    n = x.shape[1]
    scores = {v: np.zeros(len(x)) for v in VARIANTS}
    chosen = []
    for left in range(FOLDS):
        train, test = np.flatnonzero(fold != left), np.flatnonzero(fold == left)
        f, units = scale_trio_features(raw, train, unit_rule)
        inner_val = np.flatnonzero(fold == (left + 1) % FOLDS)
        inner_train = np.flatnonzero((fold != left) & (fold != (left + 1) % FOLDS))
        o, cycles = np.empty(x.shape), {}
        for rest in ("PRE", "POST"):
            rows = epoch == rest
            law = content.Content(x[np.flatnonzero(rows & (fold != left))], rec["rates"], rec["window"])
            o[rows] = replay_logits(law, f["x"][rows], order)
            cycles[rest] = law.cycles
        replay = fit(f, inner_train, o, u, "replay", 1.0)
        by_tau = {}
        for tau in TAUS:
            theta = fit(f, inner_train, o, u, "pairs", tau, widen(replay, n, "replay", "pairs"))
            by_tau[tau] = float(log_probability(f, inner_val, o, u, "pairs", tau, theta).sum())
        tau = max(by_tau, key=by_tau.get)
        theta = {"replay": fit(f, train, o, u, "replay", tau)}
        for name, source in (("pairs", "replay"), ("persistent", "pairs"), ("shared", "persistent"), ("class", "shared")):
            theta[name] = fit(f, train, o, u, name, tau, widen(theta[source], n, source, name))
        for name in VARIANTS:
            scores[name][test] = log_probability(f, test, o, u, name, tau, theta[name])
        pairs = n * (n - 1) // 2
        chosen.append(dict(fold=left, units=units, tau=tau, inner_scores={str(k): v for k, v in by_tau.items()}, cycles=cycles,
                           coefficients={name: {k: float(theta[name][n + pairs + i]) for i, k in enumerate(VARIANTS[name])}
                                         for name in VARIANTS if VARIANTS[name]}))
    label = "training fold" if unit_rule == "class" else "common count, training fold"
    return scores, block, dict(scaling=label, folds=chosen)


def compared(scores, block):
    """Each comparison: total held-out difference, its variance with blocks as units within each rest, and by rest."""
    out = {}
    for name, (first, second) in COMPARISONS.items():
        difference = scores[first] - scores[second]
        sums = np.array([difference[block == b].sum() for b in range(2 * BLOCKS)])
        variance = sum(BLOCKS / (BLOCKS - 1) * float(((part - part.mean()) ** 2).sum()) for part in (sums[:BLOCKS], sums[BLOCKS:]))
        out[name] = dict(total=float(difference.sum()), variance=variance, before=float(sums[:BLOCKS].sum()), after=float(sums[BLOCKS:].sum()))
    out["ripples"] = int(len(block))
    out["replay_per_ripple"] = float(scores["replay"].mean())
    return out


def gathered(entries):
    """Recordings added: total score of each comparison, its standard error and z, and per ripple."""
    entries = [e for e in entries if e is not None]
    if not entries:
        return None
    ripples = sum(e["ripples"] for e in entries)
    out = dict(recordings=len(entries), ripples=ripples)
    for name in COMPARISONS:
        total, variance = sum(e[name]["total"] for e in entries), sum(e[name]["variance"] for e in entries)
        out[name] = dict(total=total, se=float(np.sqrt(variance)), z=float(total / np.sqrt(variance)) if variance > 0 else None,
                         per_ripple=total / ripples, recordings_positive=int(sum(e[name]["total"] > 0 for e in entries)))
    return out


def study(content_folder, out, name, census_root=None, unit_rule="class"):
    started = time.time()
    rec = recording(content_folder, name, census_root)
    scores, block, chosen = held_out(rec, unit_rule=unit_rule)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", block=block, **scores)
    report = dict(name=name, place_cells=int(rec["x"].shape[1]), ripples=[int((rec["epoch"] == r).sum()) for r in ("PRE", "POST")],
                  chosen=chosen, compared=compared(scores, block), seconds=round(time.time() - started, 1))
    (out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def generate(rec, coefficients, rng, equal_per_pair=False):
    """Place cells drawn again: a stretch and strength from each ripple's posterior under the stored law of its rest,
    then each cell in the order of the fields with the scenario's modulation (see the header)."""
    x, epoch, order = rec["x"], rec["epoch"], rec["order"]
    n = x.shape[1]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    recorded = features(x, rec["sets"], order)
    units = {"o": float(recorded["no"].std()) or 1.0, "s": float(recorded["ns"].std()) or 1.0}
    if equal_per_pair:
        # Preserve persistent baselines and express both changes per additional active pair.
        if coefficients["c_o"] != coefficients["c_s"]:
            raise ValueError("An equal-per-pair plant needs one shared change")
        common = max(units.values())
        coefficients = dict(coefficients, c_o=coefficients["c_o"] * units["o"] / common,
                            c_s=coefficients["c_s"] * units["s"] / common)
    p = np.empty(x.shape)
    for rest in ("PRE", "POST"):
        rows = np.flatnonzero(epoch == rest)
        law = rec["laws"][rest]
        running = np.cumsum(law.decoded(x[rows]), axis=1)
        pick = (rng.random(len(rows))[:, None] > running).sum(1).clip(0, running.shape[1] - 1)
        p[rows] = law.table[pick]
    closing = {}
    for cls, short in (("overlapping", "o"), ("separate", "s")):
        trios = np.asarray(rec["sets"][("trios", cls)], dtype=np.int64).reshape(-1, 3)
        by_last = {}
        for trio in trios:
            ranked = sorted(trio, key=lambda c: rank[c])
            by_last.setdefault(int(ranked[2]), []).append((int(ranked[0]), int(ranked[1])))
        closing[short] = {cell: np.array(v) for cell, v in by_last.items()}
    u = (epoch == "POST").astype(np.float64)
    out = np.zeros(x.shape, dtype=np.int8)
    logit_p = np.log(np.clip(p, 1e-9, 1 - 1e-9)) - np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
    for cell in order:
        z = logit_p[:, cell].copy()
        for short in ("o", "s"):
            partners = closing[short].get(int(cell))
            if partners is None:
                continue
            count = (out[:, partners[:, 0]].astype(np.float64) * out[:, partners[:, 1]]).sum(1) / units[short]
            z += (coefficients[f"d_{short}"] + u * coefficients[f"c_{short}"]) * count
        out[:, cell] = rng.random(len(z)) < expit(z)
    return out


CACHE = {}


def check_cohort(arguments):
    content_folder, census_root, scenario, seed, replicate, group, unit_rule = arguments
    started = time.time()
    if (content_folder, group) not in CACHE:
        CACHE[(content_folder, group)] = {name: recording(content_folder, name, None if census_root == "-" else census_root) for name in GROUPS[group]}
    entries = {}
    for index, (name, rec) in enumerate(CACHE[(content_folder, group)].items()):
        rng = np.random.default_rng([seed, index, replicate])
        x = generate(rec, SCENARIOS[scenario], rng, equal_per_pair=(unit_rule == "common" and scenario == "apart"))
        scores, block, _ = held_out(rec, x, unit_rule=unit_rule)
        entries[name] = compared(scores, block)
    return dict(scenario=scenario, replicate=replicate, seed=seed, group=group, unit_rule=unit_rule, cohort=gathered(list(entries.values())), recordings=entries,
                seconds=round(time.time() - started, 1))


def check(content_folder, census_root, out, scenario, replicates, seed, workers, group="cohort", unit_rule="class"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{scenario}.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any((r["scenario"], r["seed"], r["group"], r.get("unit_rule","class")) !=
           (scenario, seed, group, unit_rule) for r in existing):
        raise ValueError("The existing output uses a different calibration design.")
    have = {r["replicate"] for r in existing}
    tasks = [(str(content_folder), str(census_root), scenario, seed, r, group, unit_rule) for r in range(replicates) if r not in have]
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for row in (map if workers <= 1 else pool.map)(check_cohort, tasks):
            with open(path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row["replicate"], row["seconds"], "s", {k: (round(v["total"], 2), None if v["z"] is None else round(v["z"], 2))
                                                         for k, v in row["cohort"].items() if isinstance(v, dict)}, flush=True)


def summarise(out):
    out = Path(out)
    report = {}
    runs = sorted(out.glob("*.jsonl"))
    if runs:                                                              # a check folder
        for path in runs:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            entry = dict(cohorts=len(rows))
            for name in COMPARISONS:
                totals = np.array([r["cohort"][name]["total"] for r in rows])
                z = np.array([r["cohort"][name]["z"] for r in rows], dtype=np.float64)
                entry[name] = dict(mean_total=float(totals.mean()), sd_total=float(totals.std(ddof=1)) if len(rows) > 1 else None,
                                   mean_z=float(np.nanmean(z)), sd_z=float(np.nanstd(z, ddof=1)) if len(rows) > 1 else None,
                                   share_z_above_1_96=float(np.mean(z >= 1.96)), share_positive=float(np.mean(totals > 0)))
            report[path.stem] = entry
        if {"replay", "apart", "class"} <= set(report):
            report["expectations"] = dict(
                e1_replay_quiet=bool(report["replay"]["class_change"]["share_z_above_1_96"] <= 0.08 and report["replay"]["time_varying"]["share_z_above_1_96"] <= 0.08),
                e2_apart_together_no_class_change=bool(report["apart"]["class_change"]["share_z_above_1_96"] <= 0.08),
                e2_apart_together_time_varying_found=bool(report["apart"]["time_varying"]["share_z_above_1_96"] >= 0.5),
                e3_class_change_found=bool(report["class"]["class_change"]["share_z_above_1_96"] >= 0.5))
    else:                                                                 # a study folder
        recordings = {path.stem: json.loads(path.read_text()) for path in sorted(out.glob("*.json")) if path.stem != "summary"}
        report["groups"] = {group: gathered([recordings[n]["compared"] for n in members if n in recordings]) for group, members in GROUPS.items()}
        report["recordings"] = {n: dict(compared=r["compared"], place_cells=r["place_cells"], ripples=r["ripples"],
                                        taus=[c["tau"] for c in r["chosen"]["folds"]]) for n, r in recordings.items()}
        groups = report["groups"]
        report["expectations"] = {g: dict(class_change_z=None if groups.get(g) is None else groups[g]["class_change"]["z"],
                                          positive_with_z_1_96=bool(groups.get(g) is not None and groups[g]["class_change"]["z"] is not None
                                                                    and groups[g]["class_change"]["z"] >= 1.96))
                                  for g in ("cohort", "achilles")}
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "study":
        census = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
        r = study(sys.argv[2], sys.argv[3], sys.argv[4], census, sys.argv[6] if len(sys.argv) > 6 else "class")
        print(json.dumps({k: v for k, v in r.items() if k != "chosen"}, indent=1))
    elif mode == "check":
        check(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]) if len(sys.argv) > 8 else 1,
              sys.argv[9] if len(sys.argv) > 9 else "cohort", sys.argv[10] if len(sys.argv) > 10 else "class")
    elif mode == "summary":
        print(json.dumps({k: v for k, v in summarise(sys.argv[2]).items() if k != "recordings"}, indent=1))
    else:
        raise SystemExit(__doc__)
