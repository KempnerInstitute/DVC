"""E66: made-up cohorts for the content-only law of real_data/dryad_posthoc_content.py, at the second cohort's sizes.

Written on 24 November 2026, before the content-only law was fitted to any recording.  It asks three things of
what `dryad_posthoc_content.py` does with the recordings (fit a law to each rest; the `adjusted` estimate;
cohorts drawn from the laws):
  1. when the made-up cohort IS a content-only law, is the `adjusted` three-way difference near zero whatever the
     change of content, and how wide is it (scenarios without a planted change)?
  2. do cohorts drawn from the fitted laws repeat the `plain` change of the pair term and of the three-way term?
  3. when a three-way change is planted beyond the place, with the pair tables left as they were (the moves of
     E63), how much of it does `adjusted` keep, and how much do the cohorts of the fitted laws give back
     (scenario `planted`)?

    e66_content_law_check.py truth OUT.json EVENTS COHORTS SEED WORKERS
    e66_content_law_check.py run OUT.json SCENARIO REPLICATES SEED WORKERS
    e66_content_law_check.py summary OUT_FOLDER

A made-up session: place cells with one field (sd 12 cm, peak about 8 Hz, on one or both running directions of
one of two tracks, 45 bins of 4 cm each), cell factors with a log-normal spread of 0.4, and the states of the
content-only law.  Classes as in the study (correlation of the maps at least 0.5, or at most 0).  A true law
gives a share of the ripples a place (bins 20%, pieces within 20 cm 40%, within 48 cm 30%, whole directions 10%,
all places alike) with gains weighted as the scenario says, and the others none, at low gain.  Nine sessions with
the place cells and the numbers of ripples of the second cohort.

`truth`: the `plain` and `loglinear` changes of each scenario's true laws with many ripples (no fitting), to see
what each scenario is.
`run`: for every replicate a cohort is drawn from the true laws ("recorded") and laws are fitted to its rests;
the three estimates are computed for it as for the recordings; one cohort is drawn from the fitted laws ("law")
and the `plain` estimate computed for it.
`summary`: the scenarios side by side, and how often `adjusted` puts a replicate of each scenario above the 97.5th
percentile of the replicates of the scenarios without a planted change.
"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


content = module(PROJECT / "real_data" / "dryad_posthoc_content.py", "dryad_posthoc_content")
place = module(PROJECT / "real_data" / "hippocampal_place_fields.py", "hippocampal_place_fields")
e63 = module(PROJECT / "experiments" / "e63_cohort2_gate.py", "e63_cohort2_gate")
e62 = module(PROJECT / "real_data" / "hippocampal_study.py", "hippocampal_study")

SIZES = [(29, 403, 1116), (32, 128, 2357), (71, 625, 1643), (41, 414, 1016), (84, 684, 1650), (96, 154, 1489), (43, 345, 2086), (67, 531, 1042), (76, 725, 1028)]
BINS, MAPS, SEPARATE_SAMPLE, SCALE = 45, 4, 2000, 2.0
MIDDLE, STRONG, WEAK = (0.1, 0.2, 0.4, 0.2, 0.1), (0.0, 0.05, 0.25, 0.35, 0.35), (0.3, 0.4, 0.2, 0.1, 0.0)
SCENARIOS = {  # share of the ripples with a place before and after; weights of the gains before and after; planted three-way change
    "same": dict(share=(0.3, 0.3), gains=(MIDDLE, MIDDLE), plant=0.0),
    "replay": dict(share=(0.25, 0.5), gains=(MIDDLE, MIDDLE), plant=0.0),
    "rare": dict(share=(0.02, 0.06), gains=(MIDDLE, MIDDLE), plant=0.0),
    "rare_strong": dict(share=(0.01, 0.03), gains=(STRONG, STRONG), plant=0.0),
    "stronger": dict(share=(0.3, 0.3), gains=(WEAK, STRONG), plant=0.0),
    "planted": dict(share=(0.25, 0.5), gains=(MIDDLE, MIDDLE), plant=0.4),
}


def session_truth(rng, cells):
    """Fields, states and classes of a made-up session, and its cells' factors."""
    which = np.repeat(np.arange(MAPS), BINS)
    position = np.tile(np.arange(BINS), MAPS)
    track = rng.integers(0, 2, cells)
    both_directions = rng.random(cells) < 0.5
    direction = rng.integers(0, 2, cells)
    centre, peak = rng.uniform(2, BINS - 2, cells), np.exp(rng.normal(np.log(8.0), 0.5, cells))
    fields = np.full((cells, MAPS * BINS), 0.2)
    for c in range(cells):
        for d in range(2):
            if both_directions[c] or d == direction[c]:
                on = which == 2 * track[c] + d
                fields[c, on] += peak[c] * np.exp(-0.5 * ((position[on] - centre[c]) / 3.0) ** 2)
    rates, kinds = content.content_states(fields, which)
    pairs, trios = place.classes(np.corrcoef(fields), np.ones(cells, dtype=bool), np.ones(cells, dtype=bool))
    sets = {("pairs", k): v for k, v in pairs.items()} | {("trios", k): v for k, v in trios.items()}
    for key, members in sets.items():
        if key[1] == "separate" and len(members) > SEPARATE_SAMPLE:
            sets[key] = members[np.sort(rng.choice(len(members), SEPARATE_SAMPLE, replace=False))]
    return dict(rates=rates, kinds=kinds, sets=sets, factor=SCALE * np.exp(rng.normal(0.0, 0.4, cells)), cells=cells)


def true_law(truth, share, gains):
    """A law of the family: a share of ripples with a place (mixed pieces, the gains given), the others without."""
    law = content.Content(np.zeros((1, truth["cells"]), dtype=np.int8), truth["rates"], 0.1, cycles=0)
    kinds = truth["kinds"]
    state = np.zeros(len(kinds))
    for k, part in enumerate((0.2, 0.4, 0.3, 0.1)):
        state[kinds == k] = part / (kinds == k).sum()
    weight = np.zeros((len(content.GAINS), len(kinds)))
    weight[:, :-1] = share * np.outer(gains, state[:-1])
    weight[:, -1] = (1 - share) * np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    law.weight, law.factor = weight / weight.sum(), truth["factor"]
    law.tables()
    return law


def sample(law, events, rng):
    pick = rng.choice(law.weight.size, events, p=law.weight.ravel())
    p = law.table[pick]
    return (rng.random(p.shape) < p).astype(np.int8)


def made_up(scenario, rng, events=None):
    """A cohort from the true laws of a scenario, as sessions of `dryad_posthoc_content`, and the number of planted moves."""
    s, moves, sessions = SCENARIOS[scenario], 0, []
    for index, (cells, n_pre, n_post) in enumerate(SIZES):
        truth = session_truth(rng, cells)
        n_pre, n_post = (n_pre, n_post) if events is None else (events, events)
        x_pre = sample(true_law(truth, s["share"][0], s["gains"][0]), n_pre, rng)
        x_post = sample(true_law(truth, s["share"][1], s["gains"][1]), n_post, rng)
        if s["plant"]:
            x_post, made = e63.plant(rng, x_post, truth["sets"][("trios", "overlapping")], s["plant"], e62.edges_of(x_pre.sum(1), x_post.sum(1)))
            moves += made
        sessions.append(dict(session=f"s{index}", x=np.concatenate([x_pre, x_post]), epoch=np.repeat(["PRE", "POST"], [n_pre, n_post]), cells=np.arange(cells),
                             rates=truth["rates"], kinds=truth["kinds"], window=0.1, sets=truth["sets"], sampled=content.sample_of(truth["sets"])))
    return sessions, moves


def plain_entry(d, x=None):
    return dict(plain=content.plain_parts(d, x), n_a=int((d["epoch"] == "PRE").sum()), n_b=int((d["epoch"] == "POST").sum()))


def truth_of(arguments):
    scenario, events, seed, index = arguments
    sessions, moves = made_up(scenario, np.random.default_rng([seed, index]), events)
    entries = []
    for d in sessions:
        entry = plain_entry(d)
        entry["patterns"] = {(rest, kind, cls): (None, d["x"][d["epoch"] == rest][:, members].astype(np.int64) @ (1 << np.arange(members.shape[1])))
                             for rest in content.RESTS for (kind, cls), members in d["sampled"].items() if len(members)}
        entries.append(entry)
    return dict(scenario=scenario, cohort=index, moves=moves, participation=[float(np.mean([d["x"][d["epoch"] == rest].mean() for d in sessions])) for rest in content.RESTS],
                plain={kind: content.pooled(entries, kind) for kind in content.KINDS},
                loglinear={kind: content.term_change(entries, kind, False) for kind in content.KINDS})


def replicate(arguments):
    scenario, seed, index = arguments
    started = time.time()
    rng = np.random.default_rng([seed, index])
    sessions, moves = made_up(scenario, rng)
    entries, drawn, errors, cycles = [], [], [], []
    for d in sessions:
        laws = content.laws_of(d)
        entries.append(content.session_parts(d, laws))
        again = content.simulated_session(d, laws, rng)
        errors.append(float(max(np.abs(again[d["epoch"] == rest].mean(0) - d["x"][d["epoch"] == rest].mean(0)).max() for rest in content.RESTS)))
        cycles += [law.cycles for law in laws.values()]
        drawn.append(plain_entry(d, again))
    return dict(scenario=scenario, replicate=index, recorded=content.estimates(entries), law={kind: content.pooled(drawn, kind, content.CHECK_FACTOR if kind == "trios" else 1.0) for kind in content.KINDS},
                moves=moves, participation=float(np.mean([d["x"].mean() for d in sessions])), largest_participation_error=max(errors),
                cycles=dict(largest=int(max(cycles)), mean=float(np.mean(cycles))), seconds=round(time.time() - started, 1))


def truth(out, events, cohorts, seed, workers):
    tasks = [(scenario, events, seed, index) for scenario in SCENARIOS for index in range(cohorts)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(truth_of, tasks))
    summary = {}
    for scenario in SCENARIOS:
        mine = [r for r in rows if r["scenario"] == scenario]
        summary[scenario] = {f"{which}_{kind}": dict(mean=float(np.mean([r[which][kind]["difference"] for r in mine])),
                                                     sd_between_cohorts=float(np.std([r[which][kind]["difference"] for r in mine], ddof=1)) if len(mine) > 1 else None)
                             for which in ("plain", "loglinear") for kind in content.KINDS}
        summary[scenario]["participation"] = [float(np.mean([r["participation"][k] for r in mine])) for k in (0, 1)]
        summary[scenario]["moves"] = float(np.mean([r["moves"] for r in mine]))
        print(scenario, json.dumps(summary[scenario]), flush=True)
    Path(out).write_text(json.dumps(dict(events=events, cohorts=cohorts, seed=seed, summary=summary, rows=rows), indent=1) + "\n")


def described(rows):
    """Mean, spread and mean standard error of every estimate over the replicates of a scenario."""
    out = {}
    for which in content.ESTIMATES + ("law",):
        for kind in content.KINDS:
            values = [(row["law"] if which == "law" else row["recorded"][which])[kind] for row in rows]
            values = [v for v in values if v is not None and np.isfinite(v["difference"])]
            d = np.array([v["difference"] for v in values])
            if len(d) > 1:
                out[f"{which}_{kind}"] = dict(usable=int(len(d)), mean=float(d.mean()), sd=float(d.std(ddof=1)), se=float(np.mean([v["se"] for v in values])),
                                              low=float(np.quantile(d, 0.025)), high=float(np.quantile(d, 0.975)))
    for kind in content.KINDS:           # the cohort drawn from the fitted laws against the cohort they were fitted to
        both = [(row["law"][kind]["difference"], row["recorded"]["plain"][kind]["difference"]) for row in rows]
        gap = np.array([a - b for a, b in both])
        out[f"law_minus_recorded_{kind}"] = dict(mean=float(gap.mean()), se=float(gap.std(ddof=1) / np.sqrt(len(gap))))
    return out


def run(out, scenario, replicates, seed, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(replicate, [(scenario, seed, r) for r in range(replicates)]):
            rows.append(row)
            trios = {which: None if row["recorded"][which]["trios"] is None else round(row["recorded"][which]["trios"]["difference"], 3) for which in content.ESTIMATES}
            print(row["replicate"], row["seconds"], "s", trios, "law", round(row["law"]["trios"]["difference"], 3), row["cycles"], flush=True)
    summary = described(rows)
    summary["participation"] = float(np.mean([row["participation"] for row in rows]))
    summary["largest_participation_error"] = float(np.max([row["largest_participation_error"] for row in rows]))
    summary["moves"] = float(np.mean([row["moves"] for row in rows]))
    summary["largest_cycles"] = int(max(row["cycles"]["largest"] for row in rows))
    Path(out).write_text(json.dumps(dict(scenario=scenario, replicates=replicates, seed=seed, summary=summary, rows=rows), indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def summarise(folder):
    folder = Path(folder)
    runs = {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json")) if path.stem in SCENARIOS}
    report = dict(scenarios={name: r["summary"] for name, r in runs.items()}, above={})
    for kind in content.KINDS:
        null = np.array([row["recorded"]["adjusted"][kind]["difference"] for name, r in runs.items() if not SCENARIOS[name]["plant"]
                         for row in r["rows"] if row["recorded"]["adjusted"][kind] is not None])
        if len(null) < 20:
            continue
        cut = float(np.quantile(null, 0.975))
        report["above"][kind] = dict(replicates_without_a_planted_change=int(len(null)), their_mean=float(null.mean()), their_sd=float(null.std(ddof=1)), percentile_97_5=cut,
                                     share_above={name: float(np.mean([row["recorded"]["adjusted"][kind]["difference"] > cut for row in r["rows"]
                                                                       if row["recorded"]["adjusted"][kind] is not None])) for name, r in runs.items()})
    (folder / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    if sys.argv[1] == "truth":
        truth(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    elif sys.argv[1] == "summary":
        r = summarise(sys.argv[2])
        for name, entry in r["scenarios"].items():
            print(name)
            for key, v in entry.items():
                if isinstance(v, dict) and "mean" in v:
                    print(f"   {key:28s} {v['mean']:+.3f}" + (f" (sd {v['sd']:.3f}; mean SE {v['se']:.3f})" if "sd" in v else f" (SE {v['se']:.3f})"))
        print(json.dumps(r["above"], indent=1))
    else:
        raise SystemExit(__doc__)
