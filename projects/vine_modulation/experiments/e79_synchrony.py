"""E79: is the persistent whole-ripple three-cell gain class-independent population synchrony?

Written on 26 November 2026, before anything of it was run on the recordings or on simulations.  The design, the
expectations and the decision rule are in `experiments/E79_SYNCHRONY.md`; they are summarised here.

The model is E69's (`e69_replay_vine.py`, imported and not changed): each place cell's logit, given the cells before
it in field order, is the replay law's prediction, cell intercepts, pair edges J with their changes a, b, a
persistent modulation d_O N_O + d_S N_S by the active earlier pairs that close an overlapping or a separate trio
with the cell, a shared change c and a class-specific change Delta.  E79 adds class-independent covariates of cell i
from the cells before it (k = the number of active earlier cells):
    N_all = k (k - 1) / 2     all active earlier pairs, whatever their class (primary synchrony covariate)
    k, k^2, k^3               sensitivity: a flexible function of the number of active earlier cells
each divided by its training-fold standard deviation, as N_O and N_S are (section 73).

Variants (nested; penalised maximum likelihood on the four training folds; weak penalty, SD 10, on every added
coefficient; TAU chosen exactly as in E69):
    replay, pairs                                   as E69
    persistent0   pairs + d_O N_O + d_S N_S        E69's persistent model, recomputed (reproduction)
    sync          pairs + e N_all
    common        sync + d (N_O + N_S)             one trio coefficient for both classes
    persistent    sync + d_O N_O + d_S N_S
    shared        persistent + c u (N_O + N_S)
    class         shared + Delta u N_O
    sync_k        pairs + g1 k + g2 k^2 + g3 k^3
    persistent_k  sync_k + d_O N_O + d_S N_S
Held-out comparisons (first minus second; whole ripples; blocks as units within each rest; recordings add):
    synchrony = sync - pairs; modulation = persistent - sync; common_modulation = common - sync;
    class_difference = persistent - common; time_varying = shared - persistent; class_change = class - shared;
    modulation_e69 = persistent0 - pairs; synchrony_k = sync_k - pairs; modulation_k = persistent_k - sync_k.
Coefficients: `persistent0`, `sync`, `persistent` and `persistent_k` are also fitted to all ripples of a recording
(TAU the most frequent of the folds' choices); blocked sandwich standard errors (penalised Hessian with intercepts and
pair edges profiled out, efficient scores summed within the 20 blocks and centred within each rest, factor
B/(B-1)); recordings pooled by inverse-variance weights.

Decision for the independent cohort (written before the runs): the persistent gain is synchrony if `modulation` has
blocked z < 1.96 and is below half of `modulation_e69`; a three-cell term beyond synchrony if `modulation` has
z >= 1.96; class-specific only if, in addition, `class_difference` has z >= 1.96 or the pooled d_O - d_S is positive
with z >= 1.96.  Otherwise unresolved.  The same rule is applied to `modulation_k` as a sensitivity.

Check: cohorts of the nine second-cohort sessions drawn as in E69's `check` (a stretch and strength from each
ripple's posterior under the stored law of its rest, then each cell in field order), with
    replay              nothing planted
    synchrony           0.15 N_all (per recorded SD of N_all), no class modulation
    persistent_class    0.15 N_O (per recorded SD of N_O), no synchrony, no change
scored with replay, pairs, persistent0, sync, common and persistent.  Expectations in E79_SYNCHRONY.md.

    e79_synchrony.py study CONTENT_FOLDER OUT_FOLDER NAME [CENSUS_ROOT|-]
    e79_synchrony.py study_all FIRST_FOLDER SECOND_FOLDER SECOND_CENSUS OUT_FOLDER WORKERS
    e79_synchrony.py check CONTENT_FOLDER CENSUS_ROOT OUT_FOLDER SCENARIO REPLICATES SEED [WORKERS]
    e79_synchrony.py summary OUT_FOLDER [REFERENCE_E69_STUDY_FOLDER]
"""
import importlib.util
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e69_replay_vine", HERE / "e69_replay_vine.py")
e69 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e69)
content = e69.content

TAUS, WEAK, BLOCKS, FOLDS = e69.TAUS, e69.WEAK, e69.BLOCKS, e69.FOLDS
COUNTS = ("no", "ns", "na", "k1", "k2", "k3")
VARIANTS = {"replay": (), "pairs": ("a", "b"), "persistent0": ("a", "b", "d_o", "d_s"), "sync": ("a", "b", "e"),
            "common": ("a", "b", "e", "d"), "persistent": ("a", "b", "e", "d_o", "d_s"),
            "shared": ("a", "b", "e", "d_o", "d_s", "c"), "class": ("a", "b", "e", "d_o", "d_s", "c", "delta"),
            "sync_k": ("a", "b", "g1", "g2", "g3"), "persistent_k": ("a", "b", "g1", "g2", "g3", "d_o", "d_s")}
SOURCES = (("pairs", "replay"), ("persistent0", "pairs"), ("sync", "pairs"), ("common", "sync"), ("persistent", "sync"),
           ("shared", "persistent"), ("class", "shared"), ("sync_k", "pairs"), ("persistent_k", "sync_k"))
COMPARISONS = {"pairs": ("pairs", "replay"), "synchrony": ("sync", "pairs"), "modulation_e69": ("persistent0", "pairs"),
               "modulation": ("persistent", "sync"), "common_modulation": ("common", "sync"),
               "class_difference": ("persistent", "common"), "time_varying": ("shared", "persistent"),
               "class_change": ("class", "shared"), "synchrony_k": ("sync_k", "pairs"), "modulation_k": ("persistent_k", "sync_k")}
CHECK_VARIANTS = ("replay", "pairs", "persistent0", "sync", "common", "persistent")
SCENARIOS = {"replay": dict(d_o=0.0, d_s=0.0, c_o=0.0, c_s=0.0, e=0.0),
             "synchrony": dict(d_o=0.0, d_s=0.0, c_o=0.0, c_s=0.0, e=0.15),
             "persistent_class": dict(d_o=0.15, d_s=0.0, c_o=0.0, c_s=0.0, e=0.0)}
FULL_FITS = (("pairs", "replay"), ("persistent0", "pairs"), ("sync", "pairs"), ("persistent", "sync"), ("sync_k", "pairs"),
             ("persistent_k", "sync_k"))
SANDWICH = ("persistent0", "sync", "persistent", "persistent_k")
GROUPS = e69.GROUPS


def features79(x, sets, order):
    """E69's features (cells in field order, earlier partners, trio-closing counts) and the synchrony covariates."""
    out = e69.features(x, sets, order)
    ordered = out["x"]
    k = np.cumsum(ordered, axis=1) - ordered                           # active cells before each cell, in field order
    out["na"] = k * (k - 1) / 2
    out["k1"], out["k2"], out["k3"] = k, k ** 2, k ** 3
    return out


def scale79(raw, training):
    """Every count in units of its spread over the training ripples (a constant count has unit one), as in E73."""
    scales = {key: float(raw[key][training].std()) or 1.0 for key in COUNTS}
    scaled = dict(raw)
    for key, scale in scales.items():
        scaled[key] = raw[key] / scale
    return scaled, scales


def columns79(f, rows, u):
    uu = u[rows][:, None]
    return {"a": uu * f["xo"][rows], "b": uu * f["xs"][rows], "d_o": f["no"][rows], "d_s": f["ns"][rows],
            "c": uu * (f["no"][rows] + f["ns"][rows]), "delta": uu * f["no"][rows], "e": f["na"][rows],
            "d": f["no"][rows] + f["ns"][rows], "g1": f["k1"][rows], "g2": f["k2"][rows], "g3": f["k3"][rows]}


def problem79(f, rows, o, u, variant, tau):
    """E69's penalised negative log likelihood, with E79's columns (identical for E69's variants)."""
    x = f["x"][rows]
    n = x.shape[1]
    lower = np.tril_indices(n, -1)
    pairs = len(lower[0]) if variant != "replay" else 0
    names = VARIANTS[variant]
    every = columns79(f, rows, u)
    columns = {name: every[name] for name in names}
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

    return objective, logits, n, pairs, columns


def fit79(f, rows, o, u, variant, tau, start=None):
    objective, _, n, pairs, _ = problem79(f, rows, o, u, variant, tau)
    theta = np.zeros(n + pairs + len(VARIANTS[variant])) if start is None else np.asarray(start, dtype=np.float64)
    found = minimize(objective, theta, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000, ftol=1e-11, gtol=1e-6))
    return found.x


def widen79(theta, n, source, target):
    pairs_source = 0 if source == "replay" else n * (n - 1) // 2
    pairs_target = 0 if target == "replay" else n * (n - 1) // 2
    out = np.zeros(n + pairs_target + len(VARIANTS[target]))
    out[:n + pairs_source] = theta[:n + pairs_source]
    for k, name in enumerate(VARIANTS[source]):
        out[n + pairs_target + VARIANTS[target].index(name)] = theta[n + pairs_source + k]
    return out


def logprob79(f, rows, o, u, variant, tau, theta):
    _, logits, _, _, _ = problem79(f, rows, o, u, variant, tau)
    z = logits(theta)
    x = f["x"][rows]
    return (x * z - np.logaddexp(0.0, z)).sum(1)


def replay_part(rec, x, f, epoch, train_mask):
    """The replay law of each rest fitted to the given ripples, and its logits for every ripple."""
    o, cycles = np.empty(x.shape), {}
    for rest in ("PRE", "POST"):
        rows = epoch == rest
        law = content.Content(x[np.flatnonzero(rows & train_mask)], rec["rates"], rec["window"])
        o[rows] = e69.replay_logits(law, f["x"][rows], rec["order"])
        cycles[rest] = law.cycles
    return o, cycles


def held_out79(rec, x=None, variants=None):
    """Held-out log probability of every ripple under each variant (E69's folds, replay laws and TAU choice)."""
    epoch, order = rec["epoch"], rec["order"]
    x = rec["x"] if x is None else x
    raw = features79(x, rec["sets"], order)
    u = (epoch == "POST").astype(np.float64)
    block, fold = e69.blocks_of(epoch)
    n = x.shape[1]
    wanted = [v for v in VARIANTS if variants is None or v in variants]
    scores = {v: np.zeros(len(x)) for v in wanted}
    chosen = []
    for left in range(FOLDS):
        train, test = np.flatnonzero(fold != left), np.flatnonzero(fold == left)
        f, units = scale79(raw, train)
        inner_val = np.flatnonzero(fold == (left + 1) % FOLDS)
        inner_train = np.flatnonzero((fold != left) & (fold != (left + 1) % FOLDS))
        o, cycles = replay_part(rec, x, f, epoch, fold != left)
        replay = fit79(f, inner_train, o, u, "replay", 1.0)
        by_tau = {}
        for tau in TAUS:
            theta = fit79(f, inner_train, o, u, "pairs", tau, widen79(replay, n, "replay", "pairs"))
            by_tau[tau] = float(logprob79(f, inner_val, o, u, "pairs", tau, theta).sum())
        tau = max(by_tau, key=by_tau.get)
        theta = {"replay": fit79(f, train, o, u, "replay", tau)}
        for name, source in SOURCES:
            if name in wanted:
                theta[name] = fit79(f, train, o, u, name, tau, widen79(theta[source], n, source, name))
        for name in wanted:
            scores[name][test] = logprob79(f, test, o, u, name, tau, theta[name])
        pairs = n * (n - 1) // 2
        chosen.append(dict(fold=left, units=units, tau=tau, inner_scores={str(k): v for k, v in by_tau.items()}, cycles=cycles,
                           coefficients={name: {k: float(theta[name][n + pairs + i]) for i, k in enumerate(VARIANTS[name])}
                                         for name in wanted if VARIANTS[name]}))
    return scores, block, dict(scaling="training fold", folds=chosen)


def compared79(scores, block):
    """Each comparison whose variants were scored: total, blocked variance within each rest, totals by rest."""
    out = {}
    for name, (first, second) in COMPARISONS.items():
        if first not in scores or second not in scores:
            continue
        difference = scores[first] - scores[second]
        sums = np.array([difference[block == b].sum() for b in range(2 * BLOCKS)])
        variance = sum(BLOCKS / (BLOCKS - 1) * float(((part - part.mean()) ** 2).sum()) for part in (sums[:BLOCKS], sums[BLOCKS:]))
        out[name] = dict(total=float(difference.sum()), variance=variance, before=float(sums[:BLOCKS].sum()), after=float(sums[BLOCKS:].sum()))
    out["ripples"] = int(len(block))
    return out


def gathered79(entries):
    entries = [e for e in entries if e is not None]
    if not entries:
        return None
    ripples = sum(e["ripples"] for e in entries)
    out = dict(recordings=len(entries), ripples=ripples)
    for name in COMPARISONS:
        if not all(name in e for e in entries):
            continue
        total, variance = sum(e[name]["total"] for e in entries), sum(e[name]["variance"] for e in entries)
        out[name] = dict(total=total, se=float(np.sqrt(variance)), z=float(total / np.sqrt(variance)) if variance > 0 else None,
                         per_1000_ripples=1000 * total / ripples, recordings_positive=int(sum(e[name]["total"] > 0 for e in entries)))
    return out


def sandwich(f, rows, o, u, variant, tau, theta, block):
    """Blocked sandwich covariance of the variant's added coefficients, with cell intercepts and pair edges profiled out."""
    names = VARIANTS[variant]
    _, logits, n, pairs, columns = problem79(f, rows, o, u, variant, tau)
    x = f["x"][rows]
    z = logits(theta)
    p = expit(z)
    w, residual = p * (1 - p), p - x
    F = np.stack([columns[name] for name in names], axis=2)                      # (ripples, cells, coefficients)
    g = len(names)
    blk = block[rows]
    indicator = np.zeros((2 * BLOCKS, len(rows)))
    indicator[blk, np.arange(len(rows))] = 1.0
    scores_global = indicator @ (residual[:, :, None] * F).sum(1)               # (blocks, coefficients)
    hessian = np.diag(np.full(g, 1 / WEAK ** 2))
    profile_hessian, profile_scores = np.zeros((g, g)), np.zeros((2 * BLOCKS, g))
    for i in range(n):
        v = np.concatenate([np.ones((len(rows), 1)), x[:, :i]], axis=1)      # the cell's intercept and earlier cells
        weighted = v * w[:, i:i + 1]
        local = weighted.T @ v + np.diag(np.r_[1 / WEAK ** 2, np.full(i, 1 / tau ** 2)])
        cross = weighted.T @ F[:, i, :]
        hessian += (F[:, i, :] * w[:, i:i + 1]).T @ F[:, i, :]
        solved = np.linalg.solve(local, cross)
        profile_hessian += cross.T @ solved
        profile_scores += (indicator @ (v * residual[:, i:i + 1])) @ solved
    information = hessian - profile_hessian
    efficient = scores_global - profile_scores
    meat = np.zeros((g, g))
    for rest in range(2):
        part = efficient[rest * BLOCKS:(rest + 1) * BLOCKS]
        centred = part - part.mean(0)
        meat += BLOCKS / (BLOCKS - 1) * centred.T @ centred
    bread = np.linalg.inv(information)
    covariance = bread @ meat @ bread
    return dict(names=list(names), estimate=[float(v) for v in theta[n + pairs:]], covariance=covariance.tolist())


def full_fit(rec, tau):
    """The variants of FULL_FITS on all ripples of the recording, and blocked sandwich errors of their coefficients."""
    epoch, order, x = rec["epoch"], rec["order"], rec["x"]
    raw = features79(x, rec["sets"], order)
    rows = np.arange(len(x))
    f, units = scale79(raw, rows)
    u = (epoch == "POST").astype(np.float64)
    o, cycles = replay_part(rec, x, f, epoch, np.ones(len(x), dtype=bool))
    block, _ = e69.blocks_of(epoch)
    n = x.shape[1]
    theta = {"replay": fit79(f, rows, o, u, "replay", tau)}
    for name, source in FULL_FITS:
        theta[name] = fit79(f, rows, o, u, name, tau, widen79(theta[source], n, source, name))
    return dict(tau=tau, units=units, cycles=cycles, coefficients={name: sandwich(f, rows, o, u, name, tau, theta[name], block)
                                                                  for name in SANDWICH})


def study(content_folder, out, name, census_root=None):
    started = time.time()
    rec = e69.recording(content_folder, name, census_root)
    scores, block, chosen = held_out79(rec)
    taus = [c["tau"] for c in chosen["folds"]]
    tau = Counter(taus).most_common(1)[0][0]
    coefficients = full_fit(rec, tau)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", block=block, **scores)
    report = dict(name=name, place_cells=int(rec["x"].shape[1]), ripples=[int((rec["epoch"] == r).sum()) for r in ("PRE", "POST")],
                  chosen=chosen, compared=compared79(scores, block), full_fit=coefficients, seconds=round(time.time() - started, 1))
    (out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def _study_task(arguments):
    folder, out, name, census = arguments
    try:
        report = study(folder, out, name, census)
        return name, report["seconds"], None
    except Exception as error:                                                   # recorded and reported, never hidden
        return name, None, repr(error)


def study_all(first, second, second_census, out, workers):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(second, str(out), s, second_census) for s in e69.SECOND]
    tasks += [(first, str(out), p.stem, None) for p in sorted(Path(first, "sessions").glob("*.npz"))
              if not p.stem.endswith("_peak") and p.stem in {n for g in ("achilles", "cicero", "gatsby") for n in GROUPS[g]}]
    tasks = [t for t in tasks if not (out / f"{t[2]}.json").exists()]
    failures = []
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for future in as_completed([pool.submit(_study_task, t) for t in tasks]):
            name, seconds, error = future.result()
            print(name, seconds, error or "", flush=True)
            if error:
                failures.append((name, error))
    return failures


def generate79(rec, coefficients, rng):
    """E69's generator (stretch and strength from each ripple's posterior, cells in field order) with a planted
    class-independent term e N_all as well as the class terms (counts in units of their recorded spread)."""
    x, epoch, order = rec["x"], rec["epoch"], rec["order"]
    n = x.shape[1]
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    recorded = features79(x, rec["sets"], order)
    units = {"o": float(recorded["no"].std()) or 1.0, "s": float(recorded["ns"].std()) or 1.0, "a": float(recorded["na"].std()) or 1.0}
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
    active = np.zeros(len(x))
    for cell in order:
        z = logit_p[:, cell].copy()
        for short in ("o", "s"):
            partners = closing[short].get(int(cell))
            if partners is None:
                continue
            count = (out[:, partners[:, 0]].astype(np.float64) * out[:, partners[:, 1]]).sum(1) / units[short]
            z += (coefficients[f"d_{short}"] + u * coefficients[f"c_{short}"]) * count
        z += coefficients["e"] * active * (active - 1) / 2 / units["a"]
        out[:, cell] = rng.random(len(z)) < expit(z)
        active += out[:, cell]
    return out


CACHE = {}


def _check_task(arguments):
    content_folder, census_root, scenario, seed, replicate, index, name = arguments
    started = time.time()
    if name not in CACHE:
        CACHE[name] = e69.recording(content_folder, name, None if census_root == "-" else census_root)
    rec = CACHE[name]
    rng = np.random.default_rng([seed, index, replicate])
    x = generate79(rec, SCENARIOS[scenario], rng)
    scores, block, _ = held_out79(rec, x, CHECK_VARIANTS)
    return dict(scenario=scenario, replicate=replicate, seed=seed, name=name, compared=compared79(scores, block),
                seconds=round(time.time() - started, 1))


def check(content_folder, census_root, out, scenario, replicates, seed, workers):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{scenario}.jsonl"
    have = set()
    if path.exists():
        have = {(r["replicate"], r["name"]) for r in map(json.loads, path.read_text().splitlines())}
    tasks = [(str(content_folder), str(census_root), scenario, seed, r, i, name)
             for r in range(replicates) for i, name in enumerate(GROUPS["cohort"]) if (r, name) not in have]
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for future in as_completed([pool.submit(_check_task, t) for t in tasks]):
            row = future.result()
            with open(path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row["scenario"], row["replicate"], row["name"], row["seconds"], flush=True)


def pooled(values):
    """Inverse-variance pooling of (estimate, variance) pairs."""
    values = [(m, v) for m, v in values if v is not None and v > 0 and np.isfinite(m)]
    if not values:
        return None
    weights = np.array([1 / v for _, v in values])
    estimate = float(np.sum(weights * np.array([m for m, _ in values])) / weights.sum())
    se = float(np.sqrt(1 / weights.sum()))
    return dict(estimate=estimate, se=se, z=estimate / se, recordings=len(values))


def coefficient_rows(record, variant):
    """Estimate and blocked variance of each coefficient; a coefficient whose count is constant in the recording (no
    trio of that class) is not identified and gets no variance, so it does not enter a pooled estimate."""
    fitted = record["full_fit"]["coefficients"][variant]
    names, estimate, covariance = fitted["names"], np.array(fitted["estimate"]), np.array(fitted["covariance"])
    identified = {name: covariance[k, k] > 1e-12 for k, name in enumerate(names)}
    out = {}
    for name in names:
        k = names.index(name)
        out[name] = (float(estimate[k]), float(covariance[k, k]) if identified[name] else None)
    if "d_o" in names and "d_s" in names:
        i, j = names.index("d_o"), names.index("d_s")
        both = identified["d_o"] and identified["d_s"]
        out["d_o_minus_d_s"] = (float(estimate[i] - estimate[j]),
                                float(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j]) if both else None)
    return out


def summarise(out, reference=None):
    out = Path(out)
    runs = sorted(out.glob("*.jsonl"))
    report = {}
    if runs:                                                                      # a check folder
        for path in runs:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            by_replicate = {}
            for row in rows:
                by_replicate.setdefault(row["replicate"], []).append(row["compared"])
            complete = {r: gathered79(v) for r, v in by_replicate.items() if len(v) == len(GROUPS["cohort"])}
            entry = dict(cohorts=len(complete))
            names = [c for c in COMPARISONS if complete and all(c in g for g in complete.values())]
            for name in names:
                totals = np.array([g[name]["total"] for g in complete.values()])
                z = np.array([g[name]["z"] for g in complete.values()], dtype=np.float64)
                entry[name] = dict(mean_total=float(totals.mean()), sd_total=float(totals.std(ddof=1)) if len(totals) > 1 else None,
                                   mean_z=float(np.nanmean(z)), crossings=int(np.sum(z >= 1.96)), share_positive=float(np.mean(totals > 0)))
            report[path.stem] = entry
        expectations = {}
        if "replay" in report and report["replay"]["cohorts"]:
            expectations["c1_replay_quiet"] = bool(report["replay"]["synchrony"]["crossings"] <= 1 and report["replay"]["modulation"]["crossings"] <= 1)
        if "synchrony" in report and report["synchrony"]["cohorts"]:
            s = report["synchrony"]
            expectations["c2_synchrony_found"] = bool(s["synchrony"]["crossings"] >= s["cohorts"] / 2)
            expectations["c2_synchrony_not_passed_for_class_modulation"] = bool(s["modulation"]["crossings"] <= 1)
            expectations["c2_reported_e69_modulation_crossings"] = s["modulation_e69"]["crossings"]
        if "persistent_class" in report and report["persistent_class"]["cohorts"]:
            s = report["persistent_class"]
            expectations["c3_planted_class_term_kept"] = bool(s["modulation"]["crossings"] >= s["cohorts"] / 2)
            expectations["c3_reported_class_difference_crossings"] = s["class_difference"]["crossings"]
        report["expectations"] = expectations
        (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
        return report
    recordings = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.stem != "summary"}
    report["groups"] = {g: gathered79([recordings[n]["compared"] for n in members if n in recordings]) for g, members in GROUPS.items()}
    report["coefficients"] = {}
    for g, members in GROUPS.items():
        present = [recordings[n] for n in members if n in recordings]
        if not present:
            continue
        entry = {}
        for variant in SANDWICH:
            rows = [coefficient_rows(r, variant) for r in present]
            entry[variant] = {name: dict(pooled([row[name] for row in rows]) or {},
                                         recordings_positive=int(sum(row[name][0] > 0 for row in rows)))
                              for name in rows[0]}
        report["coefficients"][g] = entry
    report["recordings"] = {n: dict(compared=r["compared"], place_cells=r["place_cells"], ripples=r["ripples"],
                                    taus=[c["tau"] for c in r["chosen"]["folds"]], seconds=r["seconds"],
                                    coefficients={v: {k: dict(estimate=m, se=float(np.sqrt(var)) if var else None)
                                                      for k, (m, var) in coefficient_rows(r, v).items()} for v in SANDWICH})
                            for n, r in recordings.items()}
    expectations = {}
    if reference is not None:
        reference = Path(reference)
        worst = {}
        for n in recordings:
            with np.load(out / f"{n}.npz") as mine, np.load(reference / f"{n}.npz") as theirs:
                np.testing.assert_array_equal(mine["block"], theirs["block"])
                for ours, original in (("replay", "replay"), ("pairs", "pairs"), ("persistent0", "persistent")):
                    worst[(n, ours)] = float(np.abs(mine[ours] - theirs[original]).max())
        expectations["s0_largest_per_ripple_difference"] = max(worst.values())
        expectations["s0_reproduces_e73"] = bool(max(worst.values()) <= 1e-6)
    groups = report["groups"]
    for g in ("cohort", "achilles"):
        if groups.get(g) is None:
            continue
        c = groups[g]
        expectations[f"s1_{g}_synchrony"] = c["synchrony"]
        diff = report["coefficients"][g]["persistent"]["d_o_minus_d_s"]
        for label, mod, e69_mod in (("s2", "modulation", "modulation_e69"), ("s4", "modulation_k", "modulation_e69")):
            m = c[mod]
            if m["z"] is not None and m["z"] >= 1.96:
                specific = (c["class_difference"]["z"] or 0) >= 1.96 or (diff.get("z", 0) >= 1.96 and diff.get("estimate", 0) > 0)
                decision = "class-specific three-cell term beyond synchrony" if specific else "three-cell term beyond synchrony, shared by both classes"
            elif m["total"] < 0.5 * c[e69_mod]["total"]:
                decision = "persistent gain is synchrony"
            else:
                decision = "unresolved"
            expectations[f"{label}_{g}_decision"] = decision
        expectations[f"s3_{g}_no_change_gain"] = bool(all(c[k]["z"] is None or c[k]["z"] < 1.96 for k in ("time_varying", "class_change")))
    report["expectations"] = expectations
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "study":
        r = study(sys.argv[2], sys.argv[3], sys.argv[4], None if len(sys.argv) <= 5 or sys.argv[5] == "-" else sys.argv[5])
        print(json.dumps(r["compared"], indent=1))
    elif mode == "study_all":
        failed = study_all(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
        print("failures", failed)
        sys.exit(1 if failed else 0)
    elif mode == "check":
        check(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]) if len(sys.argv) > 8 else 1)
    elif mode == "summary":
        r = summarise(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        print(json.dumps({k: v for k, v in r.items() if k != "recordings"}, indent=1))
    else:
        raise SystemExit(__doc__)
