"""E61.  The gate of the hippocampal study: simulations matched to the recordings' counts.

Question: with the numbers of ripples and the participations measured in the recordings (section 56), does a
change of the three-way term shared by a class of trios come out of the pooled estimator
(`dvc_package.modulation.binary_pooled.pooled_change`) when it is there, and stay out when only single-cell rates, pair terms or
the population's gain change?  No pair or trio statistic of the recordings is used: a simulated hemisphere takes
from a recorded one the number of pyramidal cells, the number of NREM ripples before (A) and after (B) the maze,
and each cell's participation in both (100-ms windows; `verification/hippocampal_census_v2_units.tsv`).

Law of a simulated hemisphere (events are independent; sampled by Gibbs sweeps):

    log odds that cell i is active = h_i + b s + c z_g(i) + sum_j J_ij x_j + sum over class trios {i,j,k} K x_j x_k

  s       a gain shared by all cells, normal with standard deviation sigma (b = 0.25: the number of active cells
          per ripple then varies about 1.2 times more than for independent cells, as in the recordings)
  J       pair couplings: a background drawn once (normal, sd 0.1) plus j for pairs of the same group
  class   the cells of a class are in G groups of g cells; the class trios are the trios within a group
  K       the triplet term of every class trio
  z       optionally, an assembly variable per group: with probability q all its cells get c added
  h       set so that every cell has its recorded participation in A and in B, whatever else is in the law

A uniform triplet term over all trios of a group is bounded by the group's size: the pattern "all active" gains
K * C(g, 3) + j * C(g, 2), so j and K are scaled to the group size (classes `6x5`, `4x8`, `3x12`).

Scenarios (what differs between A and B besides the recorded participations and numbers of ripples):
  rates       nothing else                              (no change of any term)
  pairs       the within-group pair coupling j is 1.5 j in B
  gain        sigma is 1 in A and 1.4 in B
  both        pairs and gain
  hetero      as `pairs`, and each class trio has its own triplet term (K0 plus a normal deviation of sd K0)
  shift1      as `pairs`, and K is K0 + d1 in B         (d1, d2: per class, see CLASSES)
  shift2      as `pairs`, and K is K0 + d2 in B
              The estimator measures the three-way term of a trio's own table (the other cells summed out), which
              moves two to five times more than K does when every trio of a group moves together.  d2 was set, on
              development seeds, so that the measured term moves by about 0.2 (a 20% change of its odds ratio) in
              every class, and d1 so that it moves by 0.06 to 0.15.
  assembly1   q is 0 in A and 0.05 in B with c = 1      (K = K0 in both)
  assembly2   q is 0 in A and 0.05 in B with c = 2

Estimators, per simulated hemisphere: `strata4` (the pooled conditional estimate with four strata of the number of
other active cells; the one the study would use), `strata1` and `strata8`, the unconditional one-step estimate of
the same fit (`asymptotic`), the pooled log-linear comparator (`loglinear`), `strata4` on a control class (as many
trios, each with its three cells in three different groups), and trio by trio (exact conditional p-values with
the false discovery rule at 0.05 over the class).  A test rejects when |z| > 1.96.

    e61_hippocampal_gate.py cell OUT TEMPLATE CLASS SCENARIO REPLICATES SEED     one cell of the design
    e61_hippocampal_gate.py all OUT REPLICATES SEED [WORKERS [PART OF]]          every cell (process pool), or
                                                                                 every OF-th cell from PART
    e61_hippocampal_gate.py summary OUT                                          table and expectations

Seeds: development 970610 (looked at while this file was written: are the laws what they say, does it run, how
far does the measured term move for a planted shift, does the estimate drift without a change for one, four,
eight, sixteen strata or exact matching on the number of other active cells; `e61/smoke`, `e61/dev1` and
`e61/dev2` of the results folder, 3, 16 and 120 replicates of some cells: four strata had no drift and the
smallest spread, exact matching lost 27 to 46% of the information); gate 990610, run once after this file was
committed.

EXPECTATIONS (written on 23 November 2026, after the development runs of seed 970610 and before the gate run;
`summary` evaluates them).  200 replicates per cell; with a true rate of 5% the observed rate is below 9% in 99
of 100 cells.
 (g1) No cell fails, and in every cell the calibrated participations are within 0.015 of the recorded ones.
 (g2) Level.  In each of the five scenarios without a change of the triplet term (rates, pairs, gain, both,
      hetero), `strata4` rejects in at most 9% of the hemispheres in every cell, and in at most 6.5% over all
      cells of the scenario.
 (g3) No drift.  In those cells the mean of the `strata4` estimate is within 0.03 of zero (a sixth of the 20%
      change the study looks for; the largest in development was 0.016).  The mean over its standard deviation
      over replicates is reported.
 (g4) Standard error.  In every cell the mean standard error of `strata4` is between 0.85 and 1.2 times the
      standard deviation of the estimate over replicates.
 (g5) Design effect.  In the cells without a change its median is between 1 and 2.2 (development: 1.4 to 1.8):
      trios that share cells add information not far from independent trios.  With a change it is larger and is
      reported.
 (g6) A shared change is found.  shift2 on the strongest template (Achilles 25 October, right): `strata4` rejects
      in at least 60% of the hemispheres for the class of 60 trios and in at least 85% for the classes of 224
      and 660 trios.  shift1 and the other templates: reported (shift1 expected between 25% and 75% on the
      strongest template).  For every template and class the smallest shift of the measured term resolved with
      80% power is reported as 2.8 times the standard deviation of the estimate in the `pairs` cell.
 (g7) It is found in the class that has it: in shift1 and shift2 the control class rejects in at most 9%.
 (g8) Trio by trio nothing is found: in shift2 the median number of trios declared by the false discovery rule
      is 0 in every cell, and without a change at most 5% of the hemispheres have any declared trio.
 (g9) The unconditional estimate (`asymptotic`) drifts where tables are sparse: reported, no prediction for the
      size; expected to fail (g2) or (g3) in at least one cell of the template with the fewest ripples.
 (g10) The log-linear comparator: reported without a prediction for its level; in shift2 its rejection rate is
      expected to be within 15 points of `strata4`.
 (g11) Assembly scenarios: reported without a prediction (the triplet term that an assembly variable induces is
      small and not monotone in q).
The gate is passed if (g1) to (g7) hold.
"""
import csv
import itertools
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dvc_package.modulation.binary_pooled import pooled_change, single_trio_pvalues  # noqa: E402
from dvc_package.modulation.loglinear_pooled import pooled_loglinear_change  # noqa: E402

UNITS = HERE.parent / "verification" / "hippocampal_census_v2_units.tsv"
TEMPLATES = {  # recorded hemispheres: the strongest, two middle ones, the one with the fewest ripples
    "achilles": ("Achilles-10252013", "right"),
    "gatsby": ("Gatsby-08022013", "right"),
    "cicero": ("Cicero-09172014", "right"),
    "buddy": ("Buddy-06272013", "left"),
}
CLASSES = {  # groups, cells per group, within-group pair coupling j, triplet term K0, shifts d1 and d2
    "6x5": dict(groups=6, size=5, j=0.20, k0=0.05, shifts=(0.06, 0.10)),
    "4x8": dict(groups=4, size=8, j=0.10, k0=0.02, shifts=(0.04, 0.07)),
    "3x12": dict(groups=3, size=12, j=0.05, k0=0.01, shifts=(0.02, 0.035)),
}
SCENARIOS = ("rates", "pairs", "gain", "both", "hetero", "shift1", "shift2", "assembly1", "assembly2")
NULLS = ("rates", "pairs", "gain", "both", "hetero")
GAIN, BACKGROUND_SD, SWEEPS = 0.25, 0.1, 30


def template(name):
    session, side = TEMPLATES[name]
    rows = [r for r in csv.DictReader(open(UNITS), delimiter="\t")
            if r["session"] == session and r["hemisphere"] == side and r["cell_type"] == "pyramidal cell"]
    rate_a = np.array([float(r["participation_PRE_fixed100ms"]) for r in rows])
    rate_b = np.array([float(r["participation_POST_fixed100ms"]) for r in rows])
    return (np.clip(rate_a, 0.005, 0.95), np.clip(rate_b, 0.005, 0.95),
            int(rows[0]["nrem_ripples_PRE_fixed100ms"]), int(rows[0]["nrem_ripples_POST_fixed100ms"]))


def layout(cells, spec, seed):
    """Groups of the class (fixed for a template and class), class trios, control trios, background couplings."""
    rng = np.random.default_rng([seed, cells, spec["groups"], spec["size"]])
    order = rng.permutation(cells)
    groups = [np.sort(order[g * spec["size"]:(g + 1) * spec["size"]]) for g in range(spec["groups"])]
    trios = np.array([t for g in groups for t in itertools.combinations(g, 3)])
    control = set()
    while len(control) < len(trios):
        three = rng.choice(len(groups), 3, replace=False)
        control.add(tuple(sorted(int(rng.choice(groups[g])) for g in three)))
    background = rng.normal(0.0, BACKGROUND_SD, (cells, cells))
    background = np.triu(background, 1)
    return groups, trios, np.array(sorted(control)), background + background.T


def law(cells, groups, background, spec, scenario, block, seed):
    """Everything of the law of one block (A or B) except h."""
    j, k, sigma, q, c = spec["j"], spec["k0"], 1.0, 0.0, 0.0
    if block == "B":
        if scenario in ("pairs", "both", "hetero", "shift1", "shift2"):
            j = 1.5 * j
        if scenario in ("gain", "both"):
            sigma = 1.4
        if scenario == "shift1":
            k += spec["shifts"][0]
        if scenario == "shift2":
            k += spec["shifts"][1]
        if scenario == "assembly1":
            q, c = 0.05, 1.0
        if scenario == "assembly2":
            q, c = 0.05, 2.0
    coupling = background.copy()
    group = np.full(cells, -1)
    for g, members in enumerate(groups):
        group[members] = g
        coupling[np.ix_(members, members)] += j * (1 - np.eye(len(members)))
    tensors = None
    if scenario == "hetero":  # the same trio-specific terms in A and B
        rng = np.random.default_rng([seed, 77, cells, len(groups)])
        tensors = []
        for members in groups:
            size = len(members)
            t = np.zeros((size, size, size))
            for a, b_, c_ in itertools.combinations(range(size), 3):
                value = spec["k0"] * (1 + rng.normal())
                for p in itertools.permutations((a, b_, c_)):
                    t[p] = value
            tensors.append(t)
    return dict(coupling=coupling, group=group, groups=groups, k=k, tensors=tensors, sigma=sigma, q=q, c=c)


def simulate(rng, n, h, model, sweeps=SWEEPS):
    cells = len(h)
    base = h[None, :] + GAIN * rng.normal(0.0, model["sigma"], n)[:, None]
    if model["q"] > 0:
        on = rng.random((n, len(model["groups"]))) < model["q"]
        for g, members in enumerate(model["groups"]):
            base[:, members] += model["c"] * on[:, [g]]
    x = (rng.random((n, cells)) < expit(base)).astype(np.float64)
    position = {int(i): (g, int(np.flatnonzero(members == i)[0]))
                for g, members in enumerate(model["groups"]) for i in members}
    for _ in range(sweeps):
        for i in rng.permutation(cells):
            field = base[:, i] + x @ model["coupling"][:, i]
            if i in position:
                g, where = position[int(i)]
                inside = x[:, model["groups"][g]]
                if model["tensors"] is not None:
                    field = field + 0.5 * np.einsum("ej,jk,ek->e", inside, model["tensors"][g][where], inside)
                elif model["k"] != 0:
                    others = inside.sum(1) - x[:, i]
                    field = field + model["k"] * others * (others - 1) / 2
            x[:, i] = rng.random(n) < expit(field)
    return x.astype(np.int8)


def calibrate(rng, target, model, n=20000, rounds=12):
    """h for which every cell has its target participation under the model (checked on a larger sample)."""
    h = logit(target)
    for _ in range(rounds):
        rate = np.clip(simulate(rng, n, h, model, sweeps=20).mean(0), 1 / n, 1 - 1 / n)
        h = h + 0.8 * (logit(target) - logit(rate))
    x = simulate(rng, 3 * n, h, model)
    size = x.sum(1)
    together = np.mean([x[:, members].all(1).mean() for members in model["groups"]])
    return h, dict(worst_rate_error=float(np.abs(x.mean(0) - target).max()), size_mean=float(size.mean()),
                   size_sd=float(size.std()), independent_sd=float(np.sqrt((target * (1 - target)).sum())),
                   whole_group_active=float(together))


def strata_edges(size_a, size_b, strata):
    if strata == 1:
        return None
    return np.unique(np.quantile(np.r_[size_a, size_b], np.arange(1, strata) / strata).round())


def declared(p, level=0.05):
    """Number of trios declared by the false discovery rule (Benjamini and Hochberg)."""
    order = np.sort(p)
    passed = np.flatnonzero(order <= level * np.arange(1, len(p) + 1) / len(p))
    return int(passed[-1] + 1) if len(passed) else 0


def run_cell(name, class_name, scenario, replicates, seed, spec=None):
    spec = CLASSES[class_name] if spec is None else spec
    rate_a, rate_b, n_a, n_b = template(name)
    cells = len(rate_a)
    if spec["groups"] * spec["size"] > cells:
        raise ValueError("the class does not fit in the template")
    groups, trios, control, background = layout(cells, spec, 4242)
    model_a = law(cells, groups, background, spec, scenario, "A", 4242)
    model_b = law(cells, groups, background, spec, scenario, "B", 4242)
    tag = [seed, list(TEMPLATES).index(name), list(CLASSES).index(class_name), SCENARIOS.index(scenario)]
    h_a, check_a = calibrate(np.random.default_rng(tag + [1]), rate_a, model_a)
    h_b, check_b = calibrate(np.random.default_rng(tag + [2]), rate_b, model_b)
    rows, started = [], time.time()
    for r in range(replicates):
        rng = np.random.default_rng(tag + [100 + r])
        x_a, x_b = simulate(rng, n_a, h_a, model_a), simulate(rng, n_b, h_b, model_b)
        size_a, size_b = x_a.sum(1), x_b.sum(1)
        row = dict(replicate=r)
        for strata in (4, 1, 8):
            edges = strata_edges(size_a, size_b, strata)
            fit = (pooled_change(x_a, x_b, trios) if edges is None
                   else pooled_change(x_a, x_b, trios, size_a, size_b, edges))
            row.update({f"strata{strata}_delta": fit.delta, f"strata{strata}_se": fit.se, f"strata{strata}_z": fit.z,
                        f"strata{strata}_design": fit.design_effect, f"strata{strata}_information": fit.information})
            if strata == 4:
                row.update(asymptotic_delta=fit.delta_asymptotic, asymptotic_z=fit.z_asymptotic,
                           informative_rows=fit.informative_rows)
                edges4 = edges
        fit = pooled_change(x_a, x_b, control, size_a, size_b, edges4)
        row.update(control_delta=fit.delta, control_se=fit.se, control_z=fit.z)
        fit = pooled_loglinear_change(x_a, x_b, trios)
        row.update(loglinear_delta=fit.delta, loglinear_se=fit.se, loglinear_z=fit.z, loglinear_cells=fit.cells,
                   loglinear_design=fit.design_effect)
        row["single_declared"] = declared(single_trio_pvalues(x_a, x_b, trios, size_a, size_b, edges4))
        rows.append(row)
    return dict(template=name, klass=class_name, scenario=scenario, replicates=replicates, seed=seed, cells=cells,
                n_a=n_a, n_b=n_b, trios=len(trios), calibration=dict(A=check_a, B=check_b), rows=rows,
                seconds=round(time.time() - started, 1))


def cell_file(out, name, class_name, scenario):
    return Path(out) / f"{name}_{class_name}_{scenario}.json"


def work(arguments):
    out, name, class_name, scenario, replicates, seed = arguments
    target = cell_file(out, name, class_name, scenario)
    if target.exists():
        return str(target), "kept"
    try:
        result = run_cell(name, class_name, scenario, replicates, seed)
        target.write_text(json.dumps(result) + "\n")
        return str(target), "done"
    except Exception:
        Path(str(target) + ".failed").write_text(traceback.format_exc())
        return str(target), "FAILED"


def design():
    return [(name, class_name, scenario) for name in TEMPLATES for class_name in CLASSES for scenario in SCENARIOS
            if CLASSES[class_name]["groups"] * CLASSES[class_name]["size"] <= len(template(name)[0])]


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def summarise(out):
    out = Path(out)
    table, failed = [], sorted(p.name for p in out.glob("*.failed"))
    for name, class_name, scenario in design():
        path = cell_file(out, name, class_name, scenario)
        if not path.exists():
            failed.append(path.name + " (missing)")
            continue
        cell = json.loads(path.read_text())
        get = lambda key: finite([row[key] for row in cell["rows"]])
        line = dict(template=name, klass=class_name, scenario=scenario, replicates=cell["replicates"],
                    trios=cell["trios"], cells=cell["cells"], n_a=cell["n_a"], n_b=cell["n_b"],
                    rate_error=max(cell["calibration"][b]["worst_rate_error"] for b in "AB"),
                    size_sd_ratio=cell["calibration"]["A"]["size_sd"] / cell["calibration"]["A"]["independent_sd"],
                    whole_group_a=cell["calibration"]["A"]["whole_group_active"],
                    whole_group_b=cell["calibration"]["B"]["whole_group_active"])
        for key in ("strata4", "strata1", "strata8", "control", "loglinear", "asymptotic"):
            z, delta = get(f"{key}_z"), get(f"{key}_delta")
            line[f"{key}_reject"] = float((np.abs(z) > 1.96).mean()) if len(z) else float("nan")
            line[f"{key}_mean"] = float(delta.mean()) if len(delta) else float("nan")
            line[f"{key}_sd"] = float(delta.std(ddof=1)) if len(delta) > 1 else float("nan")
            if key != "asymptotic":
                se = get(f"{key}_se")
                line[f"{key}_se"] = float(se.mean()) if len(se) else float("nan")
            line[f"{key}_valid"] = int(len(z))
        line["design_median"] = float(np.median(get("strata4_design")))
        declared_counts = np.array([row["single_declared"] for row in cell["rows"]])
        line["single_median"] = float(np.median(declared_counts))
        line["single_any"] = float((declared_counts > 0).mean())
        table.append(line)
    columns = list(table[0]) if table else []
    text = ["\t".join(columns)] + ["\t".join(f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c])
                                             for c in columns) for row in table]
    (out / "e61_summary.tsv").write_text("\n".join(text) + "\n")

    nulls = [row for row in table if row["scenario"] in NULLS]
    verdicts = {}
    verdicts["g1"] = (not failed and all(row["rate_error"] <= 0.015 for row in table),
                      f"{len(failed)} failed or missing cells; worst participation error "
                      f"{max((row['rate_error'] for row in table), default=float('nan')):.4f}")
    worst = max(nulls, key=lambda row: row["strata4_reject"])
    by_scenario = {s: float(np.mean([row["strata4_reject"] for row in nulls if row["scenario"] == s])) for s in NULLS}
    verdicts["g2"] = (worst["strata4_reject"] <= 0.09 and max(by_scenario.values()) <= 0.065,
                      f"largest cell {worst['strata4_reject']:.3f} ({worst['template']} {worst['klass']} "
                      f"{worst['scenario']}); by scenario " + ", ".join(f"{s} {v:.3f}" for s, v in by_scenario.items()))
    drift = max(nulls, key=lambda row: abs(row["strata4_mean"]))
    relative = max(nulls, key=lambda row: abs(row["strata4_mean"]) / row["strata4_sd"])
    verdicts["g3"] = (abs(drift["strata4_mean"]) <= 0.03,
                      f"largest |mean| {abs(drift['strata4_mean']):.4f} ({drift['template']} {drift['klass']} "
                      f"{drift['scenario']}); largest |mean| / sd {abs(relative['strata4_mean']) / relative['strata4_sd']:.3f} "
                      f"({relative['template']} {relative['klass']} {relative['scenario']})")
    ratios = [row["strata4_se"] / row["strata4_sd"] for row in table]
    verdicts["g4"] = (min(ratios) >= 0.85 and max(ratios) <= 1.2, f"se / sd from {min(ratios):.3f} to {max(ratios):.3f}")
    designs = [row["design_median"] for row in nulls]
    changed = [row["design_median"] for row in table if row["scenario"] not in NULLS]
    verdicts["g5"] = (min(designs) >= 1.0 and max(designs) <= 2.2,
                      f"median design effect without a change from {min(designs):.3f} to {max(designs):.3f}; "
                      f"with a change from {min(changed):.3f} to {max(changed):.3f}")
    strong = [row for row in table if row["template"] == "achilles" and row["scenario"] == "shift2"]
    need = {"6x5": 0.60, "4x8": 0.85, "3x12": 0.85}
    resolved = "; ".join(f"{row['template']} {row['klass']} {2.8 * row['strata4_sd']:.3f}"
                         for row in table if row["scenario"] == "pairs")
    verdicts["g6"] = (bool(strong) and all(row["strata4_reject"] >= need[row["klass"]] for row in strong),
                      "shift2 on the strongest template: " + ", ".join(f"{row['klass']} {row['strata4_reject']:.2f}" for row in strong)
                      + "; shift2 elsewhere: " + ", ".join(f"{row['template']} {row['klass']} {row['strata4_reject']:.2f} (moved {row['strata4_mean']:.3f})"
                                                           for row in table if row["scenario"] == "shift2" and row["template"] != "achilles")
                      + "; shift1: " + ", ".join(f"{row['template']} {row['klass']} {row['strata4_reject']:.2f} (moved {row['strata4_mean']:.3f})"
                                                 for row in table if row["scenario"] == "shift1")
                      + "; smallest shift resolved with 80% power: " + resolved)
    shifted = [row for row in table if row["scenario"] in ("shift1", "shift2")]
    verdicts["g7"] = (all(row["control_reject"] <= 0.09 for row in shifted),
                      f"largest control rejection {max(row['control_reject'] for row in shifted):.3f}")
    verdicts["g8"] = (all(row["single_median"] == 0 for row in table if row["scenario"] == "shift2")
                      and all(row["single_any"] <= 0.05 for row in nulls),
                      f"shift2 medians {sorted({row['single_median'] for row in table if row['scenario'] == 'shift2'})}; "
                      f"largest share of null hemispheres with a declared trio {max(row['single_any'] for row in nulls):.3f}")
    sparse = [row for row in nulls if row["template"] == "buddy"]
    verdicts["g9"] = (any(row["asymptotic_reject"] > 0.09 or abs(row["asymptotic_mean"]) / row["asymptotic_sd"] > 0.3
                          for row in sparse),
                      "asymptotic, fewest ripples: " + ", ".join(
                          f"{row['klass']} {row['scenario']} reject {row['asymptotic_reject']:.2f} mean/sd "
                          f"{row['asymptotic_mean'] / row['asymptotic_sd']:.2f}" for row in sparse))
    gap = [abs(row["loglinear_reject"] - row["strata4_reject"]) for row in table if row["scenario"] == "shift2"]
    level = [row["loglinear_reject"] for row in nulls]
    verdicts["g10"] = (max(gap) <= 0.15, f"largest gap in shift2 {max(gap):.2f}; its rejection without a change from "
                                         f"{min(level):.3f} to {max(level):.3f}")
    assembly = [row for row in table if row["scenario"].startswith("assembly")]
    verdicts["g11"] = (None, "; ".join(f"{row['template']} {row['klass']} {row['scenario']}: strata4 "
                                       f"{row['strata4_reject']:.2f} (mean {row['strata4_mean']:.3f}), log-linear "
                                       f"{row['loglinear_reject']:.2f}" for row in assembly))
    gate = all(verdicts[k][0] for k in ("g1", "g2", "g3", "g4", "g5", "g6", "g7"))
    report = dict(cells=len(table), failed=failed, gate_passed=bool(gate),
                  expectations={k: dict(holds=(None if v[0] is None else bool(v[0])), detail=v[1]) for k, v in verdicts.items()})
    (out / "e61_expectations.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "cell":
        out, name, class_name, scenario, replicates, seed = sys.argv[2:8]
        Path(out).mkdir(parents=True, exist_ok=True)
        print(work((out, name, class_name, scenario, int(replicates), int(seed))))
    elif mode == "all":
        out, replicates, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        Path(out).mkdir(parents=True, exist_ok=True)
        tasks = [(out, *cell, replicates, seed) for cell in design()]
        if len(sys.argv) > 7:  # one of several jobs that share the design; a cell's result does not depend on it
            tasks = tasks[int(sys.argv[6])::int(sys.argv[7])]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for path, status in pool.map(work, tasks):
                print(status, path, flush=True)
    elif mode == "summary":
        result = summarise(sys.argv[2])
        print(json.dumps(result, indent=1))
    else:
        raise SystemExit(__doc__)
