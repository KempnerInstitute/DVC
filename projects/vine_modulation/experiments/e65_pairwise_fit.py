"""E65 (post hoc).  What a pairwise law fitted to each session's own ripples gives for the three-way difference.

The previous control imposed one coupling on every overlapping pair. It failed to reproduce participation in two recordings of rat 3, which contributed about two thirds of the information. This post hoc analysis fits a pairwise model to each session and checks its participation and pair changes before interpreting its three-way predictions.

Law of a session (cells: the kept units of E64; events: its ripples before, A, and after, B, the tracks):

    log P_e(x | s) = sum_i (h_i^e + sigma_e s) x_i + sum_{i<j} (J_ij + [e = B] (a O_ij + b S_ij)) x_i x_j + constant

  J     one coupling per pair of cells, the same in A and B;  O, S  the overlapping and the separate pairs of E64;
  a, b  the change of the coupling of the two classes from A to B;  h^e  every cell's own term in each epoch;
  s     the state of the event, standard normal, shared by all cells with the same loading;  sigma_e  set so that
        the number of active cells varies from ripple to ripple as much as recorded in that epoch.
There is no term of three or more cells and no trio-specific or class-specific parameter beyond the two pair
changes: what the law knows about three cells of a class it knows through pairs and through a state that all
cells share.  (The shared state is in every scenario of E63 too; without it a pairwise law gave a spread of 1.1
to 1.2 where 1.6 was recorded.)

Fit (`fit`).  1. J, h and (a, b) by penalised pseudo-likelihood on the recorded events of both epochs (every
cell's logistic regression on all other cells; a and b enter every regression through the pairs of the two
classes; a ridge penalty on J, and a weak one, standard deviation 10, on the intercepts, for cells that never
fire in one epoch).  2. h and sigma of each epoch are then set by simulation until every simulated participation
and the spread are the recorded ones.  3. a and b are corrected until the pair terms of the two classes, measured
on simulated sessions with the estimator of E64, move from A to B as the recorded ones did.  A session counts as
matched when the largest participation error is at most 0.02, the spread is within 0.1 of the recorded one in
both epochs and both pair changes are within 0.05 of the recorded ones.
The penalty.  A first fit with a prior standard deviation of 1 and no shared state gave couplings of -1.0 to
+1.1 and a law that, used as a generator, missed the recorded participations by up to 0.5 (`e65/fit1_cancelled_*`).
The prior standard deviations 0.3, 0.2 and 0.1 are therefore tried in that order and the first law that is
matched is kept (the least penalised one that reproduces the recordings); if none is matched the last is kept and
the session is reported as not matched.  Every attempt is in the session's report.

Run (`run`).  Simulated cohorts, one simulated session per recorded session, analysed as E64 was (strata,
conditional pooling, the difference with the covariance of the classes; rats as in the article, section 62).
Read: the three-way difference `overlapping - separate` per session, per rat, for the cohort and for the sessions
whose law is matched taken together, against the recorded values; and how often the rule (d2), with the article's
rats, would declare a replication.

    e65_pairwise_fit.py fit DATA CENSUS_ROOT FIELDS_ROOT STUDY_ROOT OUT SESSION [SEED]
    e65_pairwise_fit.py run CENSUS_ROOT FIELDS_ROOT OUT REPLICATES SEED [WORKERS]
    e65_pairwise_fit.py summary STUDY_ROOT OUT
    e65_pairwise_fit.py table OUT TABLE.tsv        one row per session, per rat and per pooled group, from the summary

Seeds: fit 970650 (the simulations inside the fit); run 990650.  Post hoc: designed after E64 was read.
"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dvc_package.modulation.binary_pooled import combine, contrast, prepare  # noqa: E402

PRIOR_SDS, OUTER, ROUNDS, PROBES = (0.3, 0.2, 0.1), 15, 4, 12
RATE_TOLERANCE, PAIR_TOLERANCE, SPREAD_TOLERANCE = 0.02, 0.05, 0.1
ARTICLE = {"rat1": ["RAT1or4_SESS"], "rat2": ["RAT2_SESS1", "RAT2_SESS2"], "rat3": ["RAT3_SESS1", "RAT3_SESS2"],
           "rat4": ["RAT1_SESS1", "RAT4_SESS2"], "rat5": ["RAT5_SESS1", "RAT5_SESS2"]}
KINDS, CLASSES = ("pairs", "trios"), ("overlapping", "separate")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort_module():
    return load("dryad_cohort", HERE.parent / "real_data" / "dryad_cohort.py")


def gate_module():
    return load("e63_cohort2_gate", HERE / "e63_cohort2_gate.py")


def recorded(data, census_root, fields_root, session):
    """The recorded events of a session (kept cells by ripples) before and after the tracks, and the classes."""
    cohort = cohort_module()
    label = cohort.SESSIONS[session][0]
    ripples = np.load(Path(census_root) / session / "ripples.npz")
    kept = [int(u) for u in ripples["kept"]]
    spikes = cohort.load_spikes(data, label)[0]
    peak, epoch = ripples["peak"], ripples["epoch"]
    x = np.column_stack([cohort.count_in(spikes[u], peak - cohort.WINDOW / 2, peak + cohort.WINDOW / 2) > 0 for u in kept]).astype(np.int8)
    sets = cohort.class_sets(Path(fields_root) / session, kept)[0]
    return x[epoch == "PRE"], x[epoch == "POST"], sets


def class_matrices(cells, sets):
    overlapping, separate = np.zeros((cells, cells)), np.zeros((cells, cells))
    for matrix, pairs in ((overlapping, sets["pairs_overlapping"]), (separate, sets["pairs_separate"])):
        for i, j in pairs:
            matrix[i, j] = matrix[j, i] = 1.0
    return overlapping, separate


def logistic_offset(z, y, offset, precision, beta, iterations=40, tolerance=1e-7):
    """Penalised logistic regression with an offset, by Newton steps from `beta`."""
    for _ in range(iterations):
        p = expit(z @ beta + offset)
        hessian = z.T @ (z * (p * (1 - p))[:, None]) + np.diag(precision)
        step = np.linalg.solve(hessian, z.T @ (y - p) - precision * beta)
        largest = np.abs(step).max()
        if largest > 5:
            step *= 5 / largest
        beta = beta + step
        if largest < tolerance:
            break
    return beta, expit(z @ beta + offset)


def pseudo_likelihood(x_a, x_b, sets, prior_sd=PRIOR_SDS[0], outer=OUTER):
    """J of epoch A (symmetrised), h per epoch and the class changes (a, b), by penalised pseudo-likelihood.

    Fitted in a centred form, in which a pair of a class has the coupling C - w_B a in A and C + w_A a in B (w the
    epochs' shares of the events): C and a are then nearly unrelated and the alternation converges in a few rounds.
    The coupling returned is that of A.
    """
    cells = x_a.shape[1]
    x = np.vstack([x_a, x_b]).astype(float)
    later = np.r_[np.zeros(len(x_a)), np.ones(len(x_b))]
    share_b = len(x_b) / len(x)
    centred = later - share_b                                   # -w_B in A, +w_A in B
    overlapping, separate = class_matrices(cells, sets)
    s_overlapping, s_separate = (x @ overlapping) * centred[:, None], (x @ separate) * centred[:, None]
    a = b = 0.0
    weights = np.zeros((cells, cells))
    field = np.zeros((cells, 2))
    betas = [np.zeros(cells + 1) for _ in range(cells)]
    # a weak penalty on the two intercepts as well: a cell that never fires in one epoch has no finite intercept
    precision = np.r_[0.01, 0.01, np.full(cells - 1, 1.0 / prior_sd ** 2)]
    history = []
    for _ in range(outer):
        gradient, hessian = np.zeros(2), np.zeros((2, 2))
        for i in range(cells):
            others = np.delete(np.arange(cells), i)
            z = np.column_stack((1 - later, later, x[:, others]))
            betas[i], p = logistic_offset(z, x[:, i], a * s_overlapping[:, i] + b * s_separate[:, i], precision, betas[i])
            field[i], weights[i, others] = betas[i][:2], betas[i][2:]
            g = np.column_stack((s_overlapping[:, i], s_separate[:, i]))
            gradient += g.T @ (x[:, i] - p)
            hessian += g.T @ (g * (p * (1 - p))[:, None])
        step = np.clip(np.linalg.solve(hessian + 1e-6 * np.eye(2), gradient), -0.5, 0.5)
        a, b = a + float(step[0]), b + float(step[1])
        history.append([a, b])
        if np.abs(step).max() < 1e-3:
            break
    return field, 0.5 * (weights + weights.T) - share_b * (a * overlapping + b * separate), a, b, history


def model_of(coupling, overlapping, separate, a, b, later, sigma=0.0):
    """The law in the form `e63_cohort2_gate.simulate` takes: pair couplings and a shared state, no triplet term."""
    cells = len(coupling)
    matrix = coupling + (a * overlapping + b * separate if later else 0.0)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    return dict(coupling=matrix, loading=np.ones(cells), partners=[empty] * cells, sigma=float(sigma), triplet=0.0)


def pair_changes(gate, rng, law, n_a, n_b, sets, edges_of, probes=PROBES):
    """Changes of the two classes' pair terms from A to B on simulated sessions (mean over `probes` sessions)."""
    got = {cls: [] for cls in CLASSES}
    for _ in range(probes):
        x_a, x_b = gate.simulate(rng, n_a, law["h_a"], law["model_a"]), gate.simulate(rng, n_b, law["h_b"], law["model_b"])
        size_a, size_b = x_a.sum(1), x_b.sum(1)
        edges = edges_of(size_a, size_b)
        for cls in CLASSES:
            got[cls].append(combine([prepare(x_a, x_b, sets[f"pairs_{cls}"], size_a, size_b, edges)]).delta)
    return {cls: float(np.mean(v)) for cls, v in got.items()}


def fit(data, census_root, fields_root, study_root, out, session, seed=970650):
    started = time.time()
    out = Path(out)
    (out / "laws").mkdir(parents=True, exist_ok=True)
    gate, cohort = gate_module(), cohort_module()
    edges_of = cohort.module("hippocampal_study").edges_of
    x_a, x_b, sets = recorded(data, census_root, fields_root, session)
    cells = x_a.shape[1]
    maze = json.loads((Path(study_root) / session / "study.json").read_text())["combined"]["maze"]
    wanted = {cls: maze[f"pairs_{cls}"]["delta"] for cls in CLASSES}
    census = json.loads((Path(census_root) / session / "census.json").read_text())
    spread = [float(census["PRE"]["active_sd_over_independent"]), float(census["POST"]["active_sd_over_independent"])]
    overlapping, separate = class_matrices(cells, sets)
    target_a, target_b = np.clip(x_a.mean(0), 0.5 / len(x_a), 0.95), np.clip(x_b.mean(0), 0.5 / len(x_b), 0.95)
    index = sorted(cohort.SESSIONS).index(session)
    rng = np.random.default_rng([seed, index])
    attempts, kept = [], None
    for prior_sd in PRIOR_SDS:
        field, coupling, a, b, history = pseudo_likelihood(x_a, x_b, sets, prior_sd)
        rounds, sigma = [], [0.5, 0.5]
        for _ in range(ROUNDS):
            model_a = model_of(coupling, overlapping, separate, a, b, False, sigma[0])
            model_b = model_of(coupling, overlapping, separate, a, b, True, sigma[1])
            h_a, sigma[0], error_a, spread_a = gate.calibrate(rng, target_a, spread[0], model_a)
            h_b, sigma[1], error_b, spread_b = gate.calibrate(rng, target_b, spread[1], model_b)
            law = dict(h_a=h_a, h_b=h_b, model_a=dict(model_a, sigma=sigma[0]), model_b=dict(model_b, sigma=sigma[1]))
            got = pair_changes(gate, rng, law, len(x_a), len(x_b), sets, edges_of)
            rounds.append(dict(a=a, b=b, sigma=list(sigma), rate_error=max(error_a, error_b), spread=[spread_a, spread_b], pair_changes=got))
            gaps = {cls: wanted[cls] - got[cls] for cls in CLASSES}
            holds = max(error_a, error_b) <= RATE_TOLERANCE and max(abs(spread_a - spread[0]), abs(spread_b - spread[1])) <= SPREAD_TOLERANCE
            if not holds and max(error_a, error_b) > 5 * RATE_TOLERANCE:
                break                                      # this penalty gives a law that cannot hold the participations
            if holds and max(abs(v) for v in gaps.values()) <= 0.6 * PAIR_TOLERANCE:
                break
            a, b = a + 0.8 * gaps["overlapping"], b + 0.8 * gaps["separate"]
        last = rounds[-1]
        matched = bool(last["rate_error"] <= RATE_TOLERANCE and max(abs(last["spread"][0] - spread[0]), abs(last["spread"][1] - spread[1])) <= SPREAD_TOLERANCE
                       and all(abs(wanted[cls] - last["pair_changes"][cls]) <= PAIR_TOLERANCE for cls in CLASSES))
        attempts.append(dict(prior_sd=prior_sd, pseudo_likelihood=dict(a=history[-1][0], b=history[-1][1], steps=len(history)), rounds=rounds, matched=matched,
                             coupling_quantiles=[float(q) for q in np.quantile(coupling[np.triu_indices(cells, 1)], [0.01, 0.25, 0.5, 0.75, 0.99])]))
        kept = dict(coupling=coupling, law=law, last=last, prior_sd=prior_sd, matched=matched)      # the matched one, or else the most penalised
        if matched:
            break
    np.savez_compressed(out / "laws" / f"{session}.npz", coupling=kept["coupling"], h_a=kept["law"]["h_a"], h_b=kept["law"]["h_b"], a=np.array(kept["last"]["a"]),
                        b=np.array(kept["last"]["b"]), sigma=np.array(kept["last"]["sigma"]), n=np.array([len(x_a), len(x_b)]))
    report = dict(session=session, cells=cells, events=[len(x_a), len(x_b)], prior_sd=kept["prior_sd"], rounds=attempts[-1]["rounds"],
                  attempts=attempts, recorded_pair_changes=wanted, recorded_spread=spread, matched=kept["matched"], seconds=round(time.time() - started, 1))
    (out / "laws" / f"{session}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


CACHE = {}


def setting(census_root, fields_root, out):
    key = (str(census_root), str(fields_root), str(out))
    if key not in CACHE:
        gate, cohort = gate_module(), cohort_module()
        laws = {}
        for session in sorted(cohort.SESSIONS):
            stored = np.load(Path(out) / "laws" / f"{session}.npz")
            ripples = np.load(Path(census_root) / session / "ripples.npz")
            kept = [int(u) for u in ripples["kept"]]
            sets = cohort.class_sets(Path(fields_root) / session, kept)[0]
            overlapping, separate = class_matrices(len(kept), sets)
            a, b = float(stored["a"]), float(stored["b"])
            matched = bool(json.loads((Path(out) / "laws" / f"{session}.json").read_text())["matched"])
            laws[session] = dict(sets=sets, matched=matched, n_a=int(stored["n"][0]), n_b=int(stored["n"][1]), h_a=stored["h_a"], h_b=stored["h_b"],
                                 model_a=model_of(stored["coupling"], overlapping, separate, a, b, False, stored["sigma"][0]),
                                 model_b=model_of(stored["coupling"], overlapping, separate, a, b, True, stored["sigma"][1]))
        CACHE[key] = (gate, cohort.module("hippocampal_study").edges_of, laws)
    return CACHE[key]


def run_cohort(arguments):
    census_root, fields_root, out, seed, replicate = arguments
    started = time.time()
    gate, edges_of, laws = setting(census_root, fields_root, out)
    sessions = sorted(laws)
    entries = {}
    for index, session in enumerate(sessions):
        law = laws[session]
        rng = np.random.default_rng([seed, index, replicate])
        x_a, x_b = gate.simulate(rng, law["n_a"], law["h_a"], law["model_a"]), gate.simulate(rng, law["n_b"], law["h_b"], law["model_b"])
        size_a, size_b = x_a.sum(1), x_b.sum(1)
        edges = edges_of(size_a, size_b)
        parts = {(kind, cls): prepare(x_a, x_b, law["sets"][f"{kind}_{cls}"], size_a, size_b, edges)
                 for kind in KINDS for cls in CLASSES if len(law["sets"][f"{kind}_{cls}"])}
        entries[session] = dict(parts=parts, n_a=law["n_a"], n_b=law["n_b"])
    row = dict(replicate=replicate, seed=seed)
    matched = [s for s in sessions if laws[s]["matched"]]
    for kind in KINDS:
        row[kind] = dict(cohort=gate.pooled([entries[s] for s in sessions], kind), matched=gate.pooled([entries[s] for s in matched], kind),
                         rats={rat: gate.pooled([entries[s] for s in members], kind) for rat, members in ARTICLE.items()},
                         sessions={s: gate.pooled([entries[s]], kind) for s in sessions})
    row["seconds"] = round(time.time() - started, 1)
    return row


def run(census_root, fields_root, out, replicates, seed, workers):
    out = Path(out)
    path = out / "cohorts.jsonl"
    have = {json.loads(line)["replicate"] for line in path.read_text().splitlines()} if path.exists() else set()
    tasks = [(census_root, fields_root, str(out), seed, r) for r in range(replicates) if r not in have]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in (map if workers <= 1 else pool.map)(run_cohort, tasks):
            with open(path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row["replicate"], row["seconds"], "s", flush=True)


def summarise(study_root, out):
    out = Path(out)
    cohort = cohort_module()
    rows = [json.loads(line) for line in (out / "cohorts.jsonl").read_text().splitlines()]
    laws = {p.stem: json.loads(p.read_text()) for p in sorted((out / "laws").glob("*.json"))}
    claims = json.loads((Path(study_root) / "claims.json").read_text())
    factor = claims["check_factor_trios"]

    def describe(values, recorded_value=None, se_factor=1.0):
        usable = [v for v in values if v is not None and np.isfinite(v[0]) and np.isfinite(v[1]) and v[1] > 0]
        d, se = np.array([v[0] for v in usable]), np.array([v[1] for v in usable]) * se_factor
        entry = dict(usable=len(usable), mean=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se.mean()), reject=float((np.abs(d / se) >= 1.96).mean()),
                     largest=float(d.max()))
        if recorded_value is not None:
            entry.update(recorded=recorded_value, share_at_or_above_recorded=float((d >= recorded_value).mean()))
        return entry

    report = dict(cohorts=len(rows), sessions={}, rats={}, laws={})
    for session, law in laws.items():
        maze = json.loads((Path(study_root) / session / "study.json").read_text())["combined"]["maze"]
        last = law["rounds"][-1]
        report["laws"][session] = dict(matched=law["matched"], prior_sd=law["prior_sd"], rate_error=last["rate_error"], a=last["a"], b=last["b"], sigma=last["sigma"],
                                       simulated_pair_changes=last["pair_changes"], recorded_pair_changes=law["recorded_pair_changes"],
                                       simulated_spread=last["spread"], recorded_spread=law["recorded_spread"])
        report["sessions"][session] = dict(trios=describe([r["trios"]["sessions"][session] for r in rows], maze["trios_difference"]["delta"], factor),
                                           pairs=describe([r["pairs"]["sessions"][session] for r in rows], maze["pairs_difference"]["delta"]))
    recorded_rats = json.loads((Path(study_root).parent / "dryad_rats_v1" / "posthoc_rats.json").read_text())["rats_as_in_the_article"]["three_way_difference"]
    for rat in ARTICLE:
        report["rats"][rat] = describe([r["trios"]["rats"][rat] for r in rows], recorded_rats[rat], factor)
    report["cohort"] = dict(trios=describe([r["trios"]["cohort"] for r in rows], claims["trios"]["maze"]["estimate"], factor),
                            pairs=describe([r["pairs"]["cohort"] for r in rows], claims["pairs"]["maze"]["estimate"]))
    # the sessions whose law is matched, pooled: recorded with the frozen pooling of E64 (written beside the links of `dryad_rats_v1`)
    matched = sorted(s for s, law in laws.items() if law["matched"])
    maze = cohort.module("hippocampal_study").combine_sessions(Path(study_root).parent / "dryad_rats_v1", "e65_matched", matched)["contrasts"]["maze"]
    report["matched_sessions"] = dict(sessions=matched,
                                      trios=describe([r["trios"]["matched"] for r in rows], maze["trios_difference"]["delta"], factor),
                                      pairs=describe([r["pairs"]["matched"] for r in rows], maze["pairs_difference"]["delta"]),
                                      recorded_trios_se=maze["trios_difference"]["se"] * factor)
    declared = []
    for r in rows:
        c = r["trios"]["cohort"]
        positive = sum(bool(v is not None and v[0] > 0) for v in r["trios"]["rats"].values())
        declared.append(bool(c is not None and c[0] > 0 and c[0] / (c[1] * factor) >= 1.96 and c[0] >= cohort.SIZE and positive >= 4))
    report["rule_d2_with_the_article_rats"] = float(np.mean(declared))
    (out / "e65_summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        r = fit(*sys.argv[2:8], *(int(v) for v in sys.argv[8:9]))
        last = r["rounds"][-1]
        print(json.dumps(dict(session=r["session"], matched=r["matched"], prior_sd=r["prior_sd"], attempts=[[x["prior_sd"], x["matched"], len(x["rounds"])] for x in r["attempts"]],
                              rate_error=round(last["rate_error"], 4), a=round(last["a"], 3), b=round(last["b"], 3), sigma=last["sigma"], simulated=last["pair_changes"],
                              recorded=r["recorded_pair_changes"], spread=last["spread"], recorded_spread=r["recorded_spread"], seconds=r["seconds"])))
    elif mode == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]) if len(sys.argv) > 7 else 1)
    elif mode == "summary":
        r = summarise(sys.argv[2], sys.argv[3])
        print(json.dumps(dict(cohorts=r["cohorts"], cohort=r["cohort"], matched_sessions=r["matched_sessions"], rats=r["rats"],
                              rule=r["rule_d2_with_the_article_rats"]), indent=1))
        for s, v in r["sessions"].items():
            print(s, "matched" if r["laws"][s]["matched"] else "NOT MATCHED", "three-way: simulated %+.3f (sd %.3f) recorded %+.3f; pairs: simulated %+.3f recorded %+.3f"
                  % (v["trios"]["mean"], v["trios"]["sd"], v["trios"]["recorded"], v["pairs"]["mean"], v["pairs"]["recorded"]))
    elif mode == "table":
        r = json.loads((Path(sys.argv[2]) / "e65_summary.json").read_text())
        text = lambda v: "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
        lines = ["unit\tmatched\tprior_sd\tlargest_participation_error\tcoupling_change_overlapping\tcoupling_change_separate\tsigma_before\tsigma_after"
                 "\tspread_simulated_before\tspread_simulated_after\tspread_recorded_before\tspread_recorded_after"
                 "\tpair_change_overlapping_simulated\tpair_change_overlapping_recorded\tpair_change_separate_simulated\tpair_change_separate_recorded"
                 "\tpair_difference_simulated\tpair_difference_recorded\tthree_way_simulated_mean\tthree_way_simulated_sd\tthree_way_simulated_largest"
                 "\tthree_way_recorded\tshare_of_cohorts_at_or_above_recorded\tcohorts"]
        for session, v in r["sessions"].items():
            law, t = r["laws"][session], v["trios"]
            lines.append("\t".join(text(x) for x in (
                session, law["matched"], law["prior_sd"], law["rate_error"], law["a"], law["b"], law["sigma"][0], law["sigma"][1], law["simulated_spread"][0],
                law["simulated_spread"][1], law["recorded_spread"][0], law["recorded_spread"][1], law["simulated_pair_changes"]["overlapping"],
                law["recorded_pair_changes"]["overlapping"], law["simulated_pair_changes"]["separate"], law["recorded_pair_changes"]["separate"],
                v["pairs"]["mean"], v["pairs"]["recorded"], t["mean"], t["sd"], t["largest"], t["recorded"], t["share_at_or_above_recorded"], t["usable"])))
        groups = [(rat, None, v) for rat, v in r["rats"].items()] + [("cohort, nine sessions", r["cohort"]["pairs"], r["cohort"]["trios"]),
                                                                    ("the sessions whose law is matched", r["matched_sessions"]["pairs"], r["matched_sessions"]["trios"])]
        for name, pairs, t in groups:
            lines.append("\t".join(text(x) for x in (name, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", pairs and pairs["mean"], pairs and pairs["recorded"],
                                                     t["mean"], t["sd"], t["largest"], t["recorded"], t["share_at_or_above_recorded"], t["usable"])))
        Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
        print(len(lines) - 1, "rows")
    else:
        raise SystemExit(__doc__)
