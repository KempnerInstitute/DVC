"""E71: where do the replay law and the vine modulation change when the boundary between the rests is not given?

This 24 November protocol estimates the time of a change without supplying the experimental boundary as the location. The replay fit and the vine's residual-modulation fit select their candidate splits using training data. Their held-out scores measure the contribution of an estimated boundary.

Recordings: the place cells of the nine second-cohort sessions and of the 13 first-cohort hemispheres, the ripples of
both rests in time order, rank r = (index + 1/2) / number of ripples (no gap for the experience).  The boundary b is
the rank of the first ripple after the experience; it is used only to report where things were found.

Part 1, the replay law.  For a candidate change point k (0.05 to 0.95 in steps of 0.05), the ripples with rank below
k get one stretch law (section 66) and the others another; its fit is the sum of the two laws' log probabilities.
  profile   all ripples, every k, beside one law for all ripples: where the split fits best, by how much, and how
            much worse the true boundary b fits than the best k (in-sample, the same number of parameters for every k)
  held out  the blocks and folds of E67 and E69: within the four training folds the best k is chosen; the laws split
            there score the fold left out ("blind"), beside laws split at b fitted to the same folds ("given") and one
            law for all training ripples ("none").  Comparisons blind - none, given - blind; blocks as units.
Part 2, the vine modulation.  E69's sequential law with the replay law split at the blind k of the fold as its base
(no boundary given), pair edges J the same throughout, class changes of the edges (a, b) at that k, the classes'
three-cell modulations d_O, d_S, and a change of the modulation (shared c, and Delta for the overlapping class) at its
own location m.  TAU chosen as in E69 within the training folds; m chosen within the training folds by the fit of
`shared` with J held at the `persistent` fit, then everything refitted.  Scores on the fold left out:
  learned_change  shared(m) - persistent       does a modulation change, placed blind, predict
  learned_class   class(m) - shared(m)         does the overlapping class change differently
and the m chosen in each fold beside b.

    e71_blind_timing.py run CONTENT_FOLDER OUT_FOLDER NAME [CENSUS_ROOT|-]
    e71_blind_timing.py summary OUT_FOLDER

Expectations, written before anything was run.
  (t1) Part 1: the best k of the profile lies within 0.10 of b in at least six of the nine second-cohort sessions and
       in at least three of Achilles's four hemispheres.
  (t2) Part 1 held out: blind - none is positive with z of at least 1.96 in the second cohort (the replay law changes,
       and its change is found without the boundary).
  (t3) Part 1 held out: given - blind is not positive with z of at least 1.96 (knowing the boundary adds nothing).
  (m1) Part 2: learned_change and learned_class are not positive with z of at least 1.96 in either cohort (after E69).
  (m2) Part 2: the modulation's m lies within 0.10 of b in at most half of the folds of the second cohort.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LOCATIONS = tuple(float(k) for k in np.round(np.arange(0.05, 0.951, 0.05), 3))
MINIMUM = 30


def module(name, folder):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / folder / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


e69 = module("e69_replay_vine", "experiments")
content = e69.content
GROUPS, FOLDS, BLOCKS, TAUS, WEAK, VARIANTS = e69.GROUPS, e69.FOLDS, e69.BLOCKS, e69.TAUS, e69.WEAK, e69.VARIANTS


def evidence(law, x):
    return law.evidence(law.likelihood(x))


def split_laws(x, rates, window, rows, rank, locations, choose_from=LOCATIONS):
    """For each candidate k: the summed log probability of `rows` under two stretch laws split at k; and, among the
    candidates in `choose_from`, the best one with its two laws."""
    values, best = {}, None
    for k in locations:
        before, after = rows[rank[rows] < k], rows[rank[rows] >= k]
        if len(before) < MINIMUM or len(after) < MINIMUM:
            continue
        a, b = content.Content(x[before], rates, window), content.Content(x[after], rates, window)
        values[k] = float(evidence(a, x[before]).sum() + evidence(b, x[after]).sum())
        if k in choose_from and (best is None or values[k] > best[0]):
            best = (values[k], k, a, b)
    return values, best


def problem(f, rows, o, u_pair, u_mod, variant, tau, hold=None):
    """E69's penalised negative log likelihood with the pair change at u_pair and the modulation change at u_mod."""
    x = f["x"][rows]
    n = x.shape[1]
    lower = np.tril_indices(n, -1)
    pairs = len(lower[0]) if variant != "replay" else 0
    names = VARIANTS[variant]
    up, um = u_pair[rows][:, None], u_mod[rows][:, None]
    columns = {"a": up * f["xo"][rows], "b": up * f["xs"][rows], "d_o": f["no"][rows], "d_s": f["ns"][rows],
               "c": um * (f["no"][rows] + f["ns"][rows]), "delta": um * f["no"][rows]}
    precision = np.r_[np.full(n, 1 / WEAK ** 2), np.full(pairs, 1 / tau ** 2), np.full(len(names), 1 / WEAK ** 2)]
    base = o[rows]
    held = None
    if hold is not None and pairs:
        coupling = np.zeros((n, n))
        coupling[lower] = hold
        held = x @ coupling.T

    def logits(theta):
        z = base + theta[:n][None, :]
        if pairs:
            if held is not None:
                z = z + held
            else:
                coupling = np.zeros((n, n))
                coupling[lower] = theta[n:n + pairs]
                z = z + x @ coupling.T
        for k, name in enumerate(names):
            z = z + theta[n + pairs + k] * columns[name]
        return z

    def objective(theta):
        if held is not None:
            theta = theta.copy()
            theta[n:n + pairs] = hold
        z = logits(theta)
        value = float((np.logaddexp(0.0, z) - x * z).sum() + 0.5 * (precision * theta ** 2).sum())
        r = expit(z) - x
        gradient = np.empty_like(theta)
        gradient[:n] = r.sum(0)
        if pairs:
            gradient[n:n + pairs] = 0.0 if held is not None else (r.T @ x)[lower]
        for k, name in enumerate(names):
            gradient[n + pairs + k] = float((r * columns[name]).sum())
        gradient += precision * theta
        if held is not None:
            gradient[n:n + pairs] = 0.0
        return value, gradient

    return objective, logits, n, pairs


def fit(f, rows, o, u_pair, u_mod, variant, tau, start=None, hold=None):
    objective, _, n, pairs = problem(f, rows, o, u_pair, u_mod, variant, tau, hold)
    theta = np.zeros(n + pairs + len(VARIANTS[variant])) if start is None else np.array(start, dtype=np.float64)
    if hold is not None and pairs:
        theta[n:n + pairs] = hold
    found = minimize(objective, theta, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=6000, ftol=1e-9 if hold is not None else 1e-11, gtol=1e-6))
    theta = found.x
    if hold is not None and pairs:
        theta[n:n + pairs] = hold
    return theta, float(found.fun)


def log_probability(f, rows, o, u_pair, u_mod, variant, tau, theta):
    _, logits, _, _ = problem(f, rows, o, u_pair, u_mod, variant, tau)
    z = logits(theta)
    x = f["x"][rows]
    return (x * z - np.logaddexp(0.0, z)).sum(1)


def compared(difference, block):
    sums = np.array([difference[block == b].sum() for b in range(2 * BLOCKS)])
    variance = sum(BLOCKS / (BLOCKS - 1) * float(((part - part.mean()) ** 2).sum()) for part in (sums[:BLOCKS], sums[BLOCKS:]))
    return dict(total=float(difference.sum()), variance=variance)


def run(content_folder, out, name, census_root=None):
    started = time.time()
    rec = e69.recording(content_folder, name, census_root)
    report, score, block, rank = analyse(rec)
    report["seconds"] = round(time.time() - started, 1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", block=block, rank=rank, **score)
    (out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def analyse(rec):
    """Parts 1 and 2 for one recording (see the header): the report, the held-out scores, the blocks and the ranks."""
    name = rec["name"]
    x, epoch, order = rec["x"], rec["epoch"], rec["order"]
    n = len(x)
    rank = (np.arange(n) + 0.5) / n
    boundary = float((epoch == "PRE").sum() / n)
    rates, window = rec["rates"], rec["window"]
    every = np.arange(n)
    one = content.Content(x, rates, window)
    profile, best = split_laws(x, rates, window, every, rank, LOCATIONS + (boundary,))
    report = dict(name=name, ripples=int(n), boundary=boundary, place_cells=int(x.shape[1]),
                  profile={str(k): v for k, v in profile.items()}, one_law=float(evidence(one, x).sum()))
    report["profile_best"] = dict(location=best[1], gain_over_one_law=best[0] - report["one_law"],
                                  boundary_minus_best=profile[boundary] - best[0] if boundary in profile else None)
    del best

    raw = e69.features(x, rec["sets"], order)
    block, fold = e69.blocks_of(epoch)
    nparams = x.shape[1]
    score = {key: np.zeros(n) for key in ("none", "blind", "given", "persistent", "shared", "class")}
    folds = []
    for left in range(FOLDS):
        train, test = np.flatnonzero(fold != left), np.flatnonzero(fold == left)
        f, units = e69.scale_trio_features(raw, train)
        _, (_, k, before_law, after_law) = split_laws(x, rates, window, train, rank, LOCATIONS)
        early, late = test[rank[test] < k], test[rank[test] >= k]
        score["blind"][early], score["blind"][late] = evidence(before_law, x[early]), evidence(after_law, x[late])
        given = {rest: content.Content(x[train[epoch[train] == rest]], rates, window) for rest in ("PRE", "POST")}
        for rest, law in given.items():
            rows = test[epoch[test] == rest]
            score["given"][rows] = evidence(law, x[rows])
        none = content.Content(x[train], rates, window)
        score["none"][test] = evidence(none, x[test])
        # Part 2: the vine on top of the blind replay split
        o = np.empty(x.shape)
        o[rank < k] = e69.replay_logits(before_law, f["x"][rank < k], order)
        o[rank >= k] = e69.replay_logits(after_law, f["x"][rank >= k], order)
        u_pair = (rank >= k).astype(np.float64)
        zero = np.zeros(n)
        inner_val = np.flatnonzero(fold == (left + 1) % FOLDS)
        inner_train = np.flatnonzero((fold != left) & (fold != (left + 1) % FOLDS))
        replay_theta, _ = fit(f, inner_train, o, u_pair, zero, "replay", 1.0)
        by_tau = {}
        for tau in TAUS:
            theta, _ = fit(f, inner_train, o, u_pair, zero, "pairs", tau, e69.widen(replay_theta, nparams, "replay", "pairs"))
            by_tau[tau] = float(log_probability(f, inner_val, o, u_pair, zero, "pairs", tau, theta).sum())
        tau = max(by_tau, key=by_tau.get)
        theta_replay, _ = fit(f, train, o, u_pair, zero, "replay", tau)
        theta_pairs, _ = fit(f, train, o, u_pair, zero, "pairs", tau, e69.widen(theta_replay, nparams, "replay", "pairs"))
        theta_persistent, _ = fit(f, train, o, u_pair, zero, "persistent", tau, e69.widen(theta_pairs, nparams, "pairs", "persistent"))
        pairs = nparams * (nparams - 1) // 2
        hold = theta_persistent[nparams:nparams + pairs]
        values = {}
        for m in LOCATIONS:
            u_mod = (rank >= m).astype(np.float64)
            if u_mod[train].min() == u_mod[train].max():
                continue
            _, values[m] = fit(f, train, o, u_pair, u_mod, "shared", tau, e69.widen(theta_persistent, nparams, "persistent", "shared"), hold)
        m = min(values, key=values.get)
        u_mod = (rank >= m).astype(np.float64)
        theta_shared, _ = fit(f, train, o, u_pair, u_mod, "shared", tau, e69.widen(theta_persistent, nparams, "persistent", "shared"))
        theta_class, _ = fit(f, train, o, u_pair, u_mod, "class", tau, e69.widen(theta_shared, nparams, "shared", "class"))
        for key, variant, theta, um in (("persistent", "persistent", theta_persistent, zero), ("shared", "shared", theta_shared, u_mod), ("class", "class", theta_class, u_mod)):
            score[key][test] = log_probability(f, test, o, u_pair, um, variant, tau, theta)
        folds.append(dict(fold=left, units=units, replay_location=k, modulation_location=m, tau=tau,
                          modulation_profile={str(key): value for key, value in values.items()}))
    report.update(folds=folds, blind_minus_none=compared(score["blind"] - score["none"], block), given_minus_blind=compared(score["given"] - score["blind"], block),
                  learned_change=compared(score["shared"] - score["persistent"], block), learned_class=compared(score["class"] - score["shared"], block))
    return report, score, block, rank


def summarise(out):
    out = Path(out)
    recordings = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json")) if p.stem != "summary"}
    groups = {}
    for group, members in GROUPS.items():
        chosen = [recordings[n] for n in members if n in recordings]
        if not chosen:
            continue
        entry = dict(recordings=len(chosen))
        for key in ("blind_minus_none", "given_minus_blind", "learned_change", "learned_class"):
            total, variance = sum(r[key]["total"] for r in chosen), sum(r[key]["variance"] for r in chosen)
            entry[key] = dict(total=total, se=float(np.sqrt(variance)), z=float(total / np.sqrt(variance)) if variance > 0 else None,
                              recordings_positive=int(sum(r[key]["total"] > 0 for r in chosen)))
        entry["profile_within_0_10"] = int(sum(abs(r["profile_best"]["location"] - r["boundary"]) <= 0.10 for r in chosen))
        entry["replay_folds_within_0_10"] = int(sum(abs(f["replay_location"] - r["boundary"]) <= 0.10 for r in chosen for f in r["folds"]))
        entry["modulation_folds_within_0_10"] = int(sum(abs(f["modulation_location"] - r["boundary"]) <= 0.10 for r in chosen for f in r["folds"]))
        entry["folds"] = int(sum(len(r["folds"]) for r in chosen))
        groups[group] = entry
    cohort, achilles = groups.get("cohort"), groups.get("achilles")
    expectations = {}
    if cohort:
        expectations.update(
            t1_second_cohort=bool(cohort["profile_within_0_10"] >= 6),
            t2_replay_change_found_blind=bool(cohort["blind_minus_none"]["z"] is not None and cohort["blind_minus_none"]["z"] >= 1.96),
            t3_boundary_adds_nothing=bool(not (cohort["given_minus_blind"]["z"] is not None and cohort["given_minus_blind"]["z"] >= 1.96)),
            m1_second_cohort=bool(not any(cohort[k]["z"] is not None and cohort[k]["z"] >= 1.96 for k in ("learned_change", "learned_class"))),
            m2_modulation_not_at_the_tracks=bool(cohort["modulation_folds_within_0_10"] <= cohort["folds"] / 2))
    if achilles:
        expectations.update(t1_achilles=bool(achilles["profile_within_0_10"] >= 3),
                            m1_achilles=bool(not any(achilles[k]["z"] is not None and achilles[k]["z"] >= 1.96 for k in ("learned_change", "learned_class"))))
    report = dict(groups=groups, expectations=expectations,
                  recordings={n: dict(boundary=r["boundary"], profile_best=r["profile_best"], replay_locations=[f["replay_location"] for f in r["folds"]],
                                      modulation_locations=[f["modulation_location"] for f in r["folds"]]) for n, r in recordings.items()})
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        r = run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None)
        print(json.dumps({k: v for k, v in r.items() if k not in ("folds", "profile")}, indent=1))
    elif mode == "summary":
        print(json.dumps({k: v for k, v in summarise(sys.argv[2]).items() if k != "recordings"}, indent=1))
    else:
        raise SystemExit(__doc__)
