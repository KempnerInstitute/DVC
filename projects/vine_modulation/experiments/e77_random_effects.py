"""E77: the class contrast across animals as a random effect.

    e77_random_effects.py EVIDENCE_FOLDER OUTPUT_JSON OUTPUT_TSV

The pooled contrasts of the paper treat the animals of a cohort as fixed: every trio of every animal enters one
conditional likelihood with one shared shift per class.  A reader who asks how large the contrast is in the population
of animals needs the between-animal variance.  This script takes the per-animal estimates and their standard errors
(the same standard errors as Table S2, including the cohort adjustments) and fits the normal-normal random-effects
model

    estimate_a ~ N(mu, se_a^2 + tau^2)

by restricted maximum likelihood (REML), with the DerSimonian-Laird moment estimate of tau^2 as a check.  Intervals for
mu use the Hartung-Knapp-Sidik-Jonkman variance with a t distribution on k - 1 degrees of freedom, which keeps its
level with few animals; the 95% prediction interval for a new animal uses t on k - 2 degrees of freedom.

Written before the computation (26 November 2026): this is a description of the recorded estimates, not a test with
a decision.  We expected the random-effects interval of the independent cohort's three-way contrast to be much wider
than the fixed-effect one, because rat 3 carries most of the information and the rats disagree (rat 1 is negative with
a large standard error).

Inputs, from v3/paper/evidence/hippocampus: second_cohort.tsv (rats 1-5; standard error 'se_rule'),
first_cohort_reserved.tsv (Achilles and Cicero; 'se_common_factor', which equals 'se' for the three-way term) and
first_cohort_development.tsv (Gatsby; 'se_calibrated'; reported as a sensitivity only).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats


def read(path):
    with Path(path).open() as handle:
        return {(r["group"], r["contrast"], r["quantity"]): r for r in csv.DictReader(handle, delimiter="\t")}


def tau2_dl(y, v):
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    return max(0.0, (q - (len(y) - 1)) / c), q


def tau2_reml(y, v):
    def negative(log_tau2):
        t2 = np.exp(log_tau2)
        w = 1 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))
    grid = np.linspace(-14, 3, 400)
    values = [negative(g) for g in grid]
    start = grid[int(np.argmin(values))]
    best = optimize.minimize_scalar(negative, bounds=(start - 0.5, start + 0.5), method="bounded")
    t2 = float(np.exp(best.x))
    zero = negative(-30.0)
    return (0.0 if zero <= best.fun + 1e-9 else t2)


def random_effects(names, y, se):
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    k = len(y)
    w_fixed = 1 / v
    fixed = float(np.sum(w_fixed * y) / np.sum(w_fixed))
    fixed_se = float(np.sqrt(1 / np.sum(w_fixed)))
    dl, q = tau2_dl(y, v)
    t2 = tau2_reml(y, v)
    w = 1 / (v + t2)
    mu = float(np.sum(w * y) / np.sum(w))
    hk_var = float(np.sum(w * (y - mu) ** 2) / ((k - 1) * np.sum(w)))
    hk_se = float(np.sqrt(max(hk_var, 1 / np.sum(w))))                # the usual floor at the plain random-effects variance
    t_ci, t_pi = stats.t.ppf(0.975, k - 1), stats.t.ppf(0.975, k - 2) if k > 2 else np.nan
    i2 = float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0
    share = w_fixed / np.sum(w_fixed)
    return dict(animals=list(names), estimates=y.tolist(), standard_errors=np.sqrt(v).tolist(),
                fixed=dict(estimate=fixed, se=fixed_se, lower=fixed - 1.96 * fixed_se, upper=fixed + 1.96 * fixed_se),
                information_share=dict(zip(names, share.round(4).tolist())),
                tau2_reml=t2, tau_reml=float(np.sqrt(t2)), tau2_dersimonian_laird=dl, q=float(q), i2=i2,
                random=dict(estimate=mu, se_hksj=hk_se, lower=mu - t_ci * hk_se, upper=mu + t_ci * hk_se,
                            prediction_lower=float(mu - t_pi * np.sqrt(t2 + hk_se ** 2)) if k > 2 else None,
                            prediction_upper=float(mu + t_pi * np.sqrt(t2 + hk_se ** 2)) if k > 2 else None))


def main(folder, out_json, out_tsv):
    folder = Path(folder)
    second = read(folder / "second_cohort.tsv")
    first = read(folder / "first_cohort_reserved.tsv")
    development = read(folder / "first_cohort_development.tsv")
    report, rows = {}, []
    for term in ("trios", "pairs"):
        quantity = term + "_difference"
        rats = [f"rat{i}" for i in range(1, 6)]
        sets = {
            "independent cohort (five rats)": ([f"Rat {i}" for i in range(1, 6)],
                                               [float(second[(r, "maze", quantity)]["delta"]) for r in rats],
                                               [float(second[(r, "maze", quantity)]["se_rule"]) for r in rats]),
            "first cohort (Achilles, Cicero)": (["Achilles", "Cicero"],
                                                [float(first[(g, "maze", quantity)]["delta"]) for g in ("achilles", "cicero")],
                                                [float(first[(g, "maze", quantity)]["se_common_factor"]) for g in ("achilles", "cicero")]),
            "all seven animals": ([f"Rat {i}" for i in range(1, 6)] + ["Achilles", "Cicero"],
                                  [float(second[(r, "maze", quantity)]["delta"]) for r in rats]
                                  + [float(first[(g, "maze", quantity)]["delta"]) for g in ("achilles", "cicero")],
                                  [float(second[(r, "maze", quantity)]["se_rule"]) for r in rats]
                                  + [float(first[(g, "maze", quantity)]["se_common_factor"]) for g in ("achilles", "cicero")]),
            "all eight with Gatsby (development)": ([f"Rat {i}" for i in range(1, 6)] + ["Achilles", "Cicero", "Gatsby"],
                                                    [float(second[(r, "maze", quantity)]["delta"]) for r in rats]
                                                    + [float(first[(g, "maze", quantity)]["delta"]) for g in ("achilles", "cicero")]
                                                    + [float(development[("gatsby", "maze", quantity)]["delta"])],
                                                    [float(second[(r, "maze", quantity)]["se_rule"]) for r in rats]
                                                    + [float(first[(g, "maze", quantity)]["se_common_factor"]) for g in ("achilles", "cicero")]
                                                    + [float(development[("gatsby", "maze", quantity)]["se_calibrated"])]),
        }
        for name, (animals, y, se) in sets.items():
            r = random_effects(animals, y, se)
            report[f"{term}: {name}"] = r
            rows.append((term, name, len(animals), r["fixed"]["estimate"], r["fixed"]["lower"], r["fixed"]["upper"],
                         r["random"]["estimate"], r["random"]["lower"], r["random"]["upper"], r["tau_reml"],
                         r["random"]["prediction_lower"], r["random"]["prediction_upper"], r["i2"]))
    Path(out_json).write_text(json.dumps(report, indent=2) + "\n")
    with Path(out_tsv).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("term", "animals", "k", "fixed", "fixed_lower", "fixed_upper", "random", "random_lower_hksj",
                         "random_upper_hksj", "tau_reml", "prediction_lower", "prediction_upper", "i2"))
        for row in rows:
            writer.writerow([f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v) for v in row])
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_tsv", type=Path)
    a = parser.parse_args()
    for row in main(a.evidence, a.output_json, a.output_tsv):
        print("\t".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row))
