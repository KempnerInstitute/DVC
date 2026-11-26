"""E78 (post hoc): the content-only replay law with one set of cell factors shared by the two rests.

Written on 26 November 2026, before anything of this file was run on the recordings; the design, expectations and
decision rule are in `experiments/E78_SHARED_FACTORS.md` (summarised here).  The law of
`dryad_posthoc_content.py` is fitted to each rest separately, factors `c_i` included:

    P(cell i active | state, gain) = 1 - exp(-gain * c_i * r_i(state) * 0.1 s).

Here one law per session (per hemisphere in the first cohort) keeps the separate weights of states and gains of
each rest but one factor per cell for both rests.  EM as in `Content`: posteriors in each rest under its own
weights; the weights of a rest from its own posteriors; each factor maximises the expected log likelihood summed
over the events of both rests (`Content.factors_of` on the pooled expected activity and exposure); rounds
extrapolated in pairs (SQUAREM) over the joint parameters (log factors, then each rest's weight margins); the
same start (one factor matching the pooled mean participation at no place and gain 1, equal weights); stop when a
cycle raises the summed log likelihood of both rests by less than 0.02 (the separate fits: 0.01 per rest) or
after 300 cycles.  The estimates (`plain`, `loglinear`, `adjusted`) and the simulated cohorts are those of
`dryad_posthoc_content.py` (second cohort) and `hippocampal_posthoc_content.py` (first cohort), with the
shared-factor law in place of the separate laws, fitted again to every simulated cohort.

    shared_factor_content.py fit COHORT SOURCE_FOLDER OUT_FOLDER [WORKERS]
    shared_factor_content.py study COHORT OUT_FOLDER [WORKERS]
    shared_factor_content.py simulate COHORT OUT_FOLDER REPLICATES SEED [WORKERS]
    shared_factor_content.py summary COHORT OUT_FOLDER

COHORT is `second` (E66's nine sessions, SOURCE_FOLDER hippocampus/dryad_content_v1) or `first` (E68's 13
hemispheres, SOURCE_FOLDER hippocampus/posthoc_content_v1).  `fit` reads the sessions E66/E68 stored (ripples,
states, sets and their separate laws), fits the shared-factor law, stores it in the same format under
OUT_FOLDER/sessions, and writes OUT_FOLDER/fit_comparison.json: in-sample log likelihood of the shared and the
separate laws on the recorded ripples, AIC and BIC differences (the separate law has one more parameter per place
cell), the movement of the separate fits' factors between rests, and the fitted share of ripples on a stretch of
at most about 1 m before and after under both laws.

What decides (fixed before the run; `experiments/E78_SHARED_FACTORS.md`), for the second cohort and for Achilles:
the law is a fair stand-in only if its cohorts give at least half of the recorded `plain` pair change; "a replay
model reproduces the rise in the independent cohort" stands with factors fixed across rests if the shared-factor
cohorts give at least half of the recorded `plain` three-way change and the recorded `adjusted` three-way
difference lies within their central 95%, and is weakened (the reproduction needs a change of excitability) if
they give less than half or the recorded `adjusted` value lies above their 97.5th percentile.  For Achilles the
share given by the cohorts replaces the "about half" of E68.
"""
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def module(name):
    """Load a module of this folder and register it, so that worker processes can unpickle its functions."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


content = module("dryad_posthoc_content")
first = module("hippocampal_posthoc_content")
TOLERANCE = 2 * content.TOLERANCE
STRETCH = ("bin", "within_20cm", "within_48cm")


def fit_shared(d, x=None, cycles=content.CYCLES, tolerance=TOLERANCE):
    """The shared-factor law of a session: {rest: Content} with one factor vector and each rest's own weights."""
    x = d["x"] if x is None else x
    cells = np.asarray(d["cells"])
    events = {rest: np.asarray(x[d["epoch"] == rest][:, cells], dtype=np.float64) for rest in content.RESTS}
    states = np.asarray(d["rates"]).shape[1]
    uniform = np.full((len(content.GAINS), states), 1.0 / (len(content.GAINS) * states))
    laws = {rest: content.Content.restored(d["rates"], d["window"], np.ones(len(cells)), uniform.copy()) for rest in content.RESTS}
    # start as Content does, on the events of both rests together
    exposure = laws["PRE"].exposure
    pooled = np.concatenate([events[rest] for rest in content.RESTS])
    observed = max(float(pooled.mean()), 1e-4)
    low, high = 1e-4, 1e4
    for _ in range(60):
        middle = np.sqrt(low * high)
        low, high = (middle, high) if (-np.expm1(-middle * exposure[2, :, -1])).mean() < observed else (low, middle)
    start = np.full(len(cells), np.sqrt(low * high))
    for law in laws.values():
        law.factor = start.copy()
        law.tables()
    n = len(cells)

    def parameters():
        return np.concatenate([laws["PRE"].parameters()] + [laws["POST"].parameters()[n:]])

    def place(vector):
        k = (len(vector) - n) // 2
        laws["PRE"].place(np.concatenate([vector[:n], vector[n:n + k]]))
        laws["POST"].place(np.concatenate([vector[:n], vector[n + k:]]))

    def one_round():
        totals, actives, value = {}, {}, 0.0
        for rest in content.RESTS:
            law, x_rest = laws[rest], events[rest]
            joint = law.joint(law.likelihood(x_rest))
            top = joint.max(axis=(0, 2), keepdims=True)
            post = np.exp(joint - top)
            norm = post.sum(axis=(0, 2), keepdims=True)
            post /= norm
            totals[rest] = post.sum(1)
            actives[rest] = np.stack([x_rest.T @ post[g] for g in range(len(content.GAINS))])
            value += float((top + np.log(norm)).sum())
        factor = laws["PRE"].factors_of(actives["PRE"] + actives["POST"], totals["PRE"] + totals["POST"])
        for rest in content.RESTS:
            laws[rest].weight = content.Content.weights_of(totals[rest])
            laws[rest].factor = factor
            laws[rest].tables()
        return value

    done, previous, value = 0, -np.inf, None
    while done < cycles:
        first_point = parameters()
        value = one_round()
        second_point = parameters()
        after_one = one_round()
        third_point = parameters()
        step, bend = second_point - first_point, third_point - 2 * second_point + first_point
        length = -max(1.0, float(np.linalg.norm(step) / max(np.linalg.norm(bend), 1e-300)))
        place(first_point - 2 * length * step + length ** 2 * bend)
        reached = one_round()
        if not np.isfinite(reached) or reached < after_one:
            place(third_point)
        done += 1
        if value - previous < tolerance:
            break
        previous = value
    for law in laws.values():
        law.cycles, law.log_likelihood = done, value
    return laws


def log_likelihood(law, x_rest):
    return float(law.evidence(law.likelihood(np.asarray(x_rest, dtype=np.float64))).sum())


def stretch_share(law, kinds):
    by_state = law.weight.sum(0)
    return float(sum(by_state[kinds == k].sum() for k, name in enumerate(content.STATE_KINDS) if name in STRETCH))


def compare(d, separate, shared):
    """In-sample fit of the separate and the shared-factor laws, and how far the separate factors move between rests."""
    cells = np.asarray(d["cells"])
    rows = {}
    total_separate = total_shared = 0.0
    events = 0
    for rest in content.RESTS:
        x_rest = d["x"][d["epoch"] == rest][:, cells]
        a, b = log_likelihood(separate[rest], x_rest), log_likelihood(shared[rest], x_rest)
        rows[rest] = dict(events=int(len(x_rest)), separate=a, shared=b, separate_per_event=a / max(len(x_rest), 1), shared_per_event=b / max(len(x_rest), 1),
                          stretch_share_separate=stretch_share(separate[rest], d["kinds"]), stretch_share_shared=stretch_share(shared[rest], d["kinds"]))
        total_separate, total_shared, events = total_separate + a, total_shared + b, events + len(x_rest)
    move = np.log(separate["POST"].factor) - np.log(separate["PRE"].factor)
    gap = total_separate - total_shared
    n = len(cells)
    return dict(place_cells=int(n), events=int(events), rests=rows, separate=total_separate, shared=total_shared, gap=gap, gap_per_event=gap / max(events, 1),
                aic_difference=2 * gap - 2 * n, bic_difference=2 * gap - n * np.log(events), cycles=int(shared["PRE"].cycles),
                factor_move=dict(median_abs=float(np.median(np.abs(move))), mean=float(move.mean()), sd=float(move.std(ddof=1)) if n > 1 else 0.0,
                                 q25=float(np.quantile(move, 0.25)), q75=float(np.quantile(move, 0.75)), share_up=float((move > 0).mean())),
                shared_vs_separate=dict(median_abs_pre=float(np.median(np.abs(np.log(shared["PRE"].factor) - np.log(separate["PRE"].factor)))),
                                        median_abs_post=float(np.median(np.abs(np.log(shared["POST"].factor) - np.log(separate["POST"].factor))))))


def fit_one(arguments):
    cohort, source, out, name = arguments
    started = time.time()
    d, separate = content.stored(source, name)
    shared = fit_shared(d)
    content.store(out, d, shared)
    if cohort == "first":
        shutil.copyfile(Path(source) / "sessions" / f"{name}_peak.npy", Path(out) / "sessions" / f"{name}_peak.npy")
    return name, dict(compare(d, separate, shared), seconds=round(time.time() - started, 1))


def fit(cohort, source, out, workers=1):
    out = Path(out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    names = content.sessions_of(source)
    report = {}
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for name, row in (map if workers <= 1 else pool.map)(fit_one, [(cohort, str(source), str(out), n) for n in names]):
            report[name] = row
            print(f"{name:28s} cells {row['place_cells']:3d} events {row['events']:5d} gap {row['gap']:9.2f} ({row['gap_per_event']:.4f} per event) "
                  f"AIC diff {row['aic_difference']:9.2f} BIC diff {row['bic_difference']:9.2f} cycles {row['cycles']:3d} "
                  f"factor move median |.| {row['factor_move']['median_abs']:.3f} mean {row['factor_move']['mean']:+.3f}", flush=True)
    names = sorted(report)
    gap = sum(report[n]["gap"] for n in names)
    events = sum(report[n]["events"] for n in names)
    cells = sum(report[n]["place_cells"] for n in names)
    summary = dict(units=len(names), gap=gap, gap_per_event=gap / events, extra_parameters=cells, aic_difference=2 * gap - 2 * cells,
                   bic_difference=sum(report[n]["bic_difference"] for n in names),
                   separate_preferred_by_aic=int(sum(report[n]["aic_difference"] > 0 for n in names)),
                   separate_preferred_by_bic=int(sum(report[n]["bic_difference"] > 0 for n in names)),
                   factor_move_median_abs_of_units=float(np.median([report[n]["factor_move"]["median_abs"] for n in names])),
                   stretch_share_rises_shared=int(sum(report[n]["rests"]["POST"]["stretch_share_shared"] > report[n]["rests"]["PRE"]["stretch_share_shared"] for n in names)),
                   stretch_share_rises_separate=int(sum(report[n]["rests"]["POST"]["stretch_share_separate"] > report[n]["rests"]["PRE"]["stretch_share_separate"] for n in names)))
    (out / "fit_comparison.json").write_text(json.dumps(dict(summary=summary, units=report), indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return summary


CACHE = {}


def setting(cohort, out):
    if out not in CACHE:
        if cohort == "second":
            CACHE[out] = {name: content.stored(out, name) for name in content.sessions_of(out)}
        else:
            CACHE[out] = first.stored(out)
    return CACHE[out]


def simulate_cohort(arguments):
    """One cohort from the shared-factor laws, the shared-factor law fitted again to it, and the three estimates."""
    cohort, out, seed, replicate = arguments
    started = time.time()
    entries, peaks, cycles = {}, {}, []
    for index, (name, stored) in enumerate(setting(cohort, out).items()):
        d, laws = stored[0], stored[1]
        x = content.simulated_session(d, laws, np.random.default_rng([seed, index, replicate]))
        again = fit_shared(d, x)
        cycles.append(again["PRE"].cycles)
        entries[name] = content.session_parts(d, again, x)
        if cohort == "first":
            peaks[name] = first.peaks_by_rest(d, stored[2])
    groups = content.results(entries) if cohort == "second" else first.results(entries, peaks)
    return dict(replicate=replicate, seed=seed, groups=groups, largest_cycles=int(max(cycles)), seconds=round(time.time() - started, 1))


def simulate(cohort, out, replicates, seed, workers):
    path = Path(out) / "cohorts.jsonl"
    have = {json.loads(line)["replicate"] for line in path.read_text().splitlines()} if path.exists() else set()
    tasks = [(cohort, str(out), seed, r) for r in range(replicates) if r not in have]
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        for row in (map if workers <= 1 else pool.map)(simulate_cohort, tasks):
            with open(path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(row["replicate"], row["seconds"], "s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[2] not in ("second", "first"):
        raise SystemExit(__doc__)
    mode, cohort = sys.argv[1], sys.argv[2]
    driver = content if cohort == "second" else first
    if mode == "fit":
        fit(cohort, sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    elif mode == "study":
        report = driver.study(sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
        content.show(report["groups"])
    elif mode == "simulate":
        simulate(cohort, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 1)
    elif mode == "summary":
        report = driver.summarise(sys.argv[3])
        for group, entry in report["groups"].items():
            for name, v in entry.items():
                print(f"{group:16s} {name:16s} recorded {v['recorded']:+.3f} | law's cohorts {v['mean']:+.3f} (sd {v['sd']:.3f}; 95% {v['low']:+.3f} to {v['high']:+.3f}); "
                      f"{v['cohorts_at_or_above_recorded']} of {v['usable']} at or above the recorded value")
        print(json.dumps(report.get("reading"), indent=1))
    else:
        raise SystemExit(__doc__)
