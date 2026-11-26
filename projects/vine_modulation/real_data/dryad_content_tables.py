"""Tables of the replay-content analysis (E66) from its outputs; nothing is estimated here.

    dryad_content_tables.py CONTENT_FOLDER OUT_PREFIX [CHECK_FOLDER ...]

Reads `sessions/<session>.json`, `study.json` and `content_only_summary.json` of
`hippocampus/dryad_content_<label>` and writes
  OUT_PREFIX_laws.tsv      the content-only law of each rest of each session
  OUT_PREFIX_content.tsv   the three estimates for the cohort and the rats: the recordings beside the cohorts of the law
  OUT_PREFIX_terms.tsv     the log-linear and the adjusted term of each class in each rest (cohort and rats)
and, from the folders of the made-up cohorts (`e66/check_<label>`; their scenarios are put together),
  OUT_PREFIX_check_truth.tsv   what each scenario is (true laws, many ripples)
  OUT_PREFIX_check.tsv         every estimate over the replicates of each scenario
  OUT_PREFIX_check_above.tsv   how often `adjusted` puts a replicate above the 97.5th percentile of the replicates
                               without a planted change
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

STATE_KINDS = ("bin", "within_20cm", "within_48cm", "whole_direction", "none")
ESTIMATES, KINDS, RESTS = ("plain", "loglinear", "adjusted"), ("pairs", "trios"), ("PRE", "POST")


def write(path, header, rows):
    Path(path).write_text("\n".join(["\t".join(header)] + ["\t".join(str(v) for v in row) for row in rows]) + "\n")
    return len(rows)


def number(value, digits=4):
    return "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.{digits}g}"


def content_tables(root, prefix):
    rows = []
    for path in sorted((root / "sessions").glob("*.json")):
        d = json.loads(path.read_text())
        for rest in RESTS:
            law = d["laws"][rest]
            rows.append([d["session"], d["place_cells"], d["kept"], d["states"], rest, law["events"], law["cycles"]]
                        + [number(law["weight"][k]) for k in STATE_KINDS] + [number(v) for v in law["gain"].values()]
                        + [number(law["log_likelihood_per_event"], 6), number(law["independent_cells_per_event"], 6),
                           number(law["log_likelihood_per_event"] - law["independent_cells_per_event"]), number(law["factor_median"]),
                           number(law["largest_participation_error"])])
    gains = list(json.loads(next(iter(sorted((root / "sessions").glob("*.json")))).read_text())["laws"]["PRE"]["gain"])
    print(write(prefix + "_laws.tsv", ["session", "place_cells", "kept_cells", "states", "rest", "ripples", "cycles"] + [f"weight_{k}" for k in STATE_KINDS]
                + [f"weight_gain_{g}" for g in gains] + ["log_likelihood_per_ripple", "independent_cells_per_ripple", "gain_over_independent_cells",
                                                         "median_cell_factor", "largest_participation_error"], rows), "law rows")
    study = json.loads((root / "study.json").read_text())
    summary = json.loads((root / "content_only_summary.json").read_text()) if (root / "content_only_summary.json").exists() else None
    rows, terms = [], []
    for group, entry in study["groups"].items():
        for which in ESTIMATES:
            for kind in KINDS:
                v = entry[which][kind]
                s = None if summary is None else summary["groups"][group].get(f"{which}_{kind}")
                rows.append([group, which, kind] + (["", "", ""] if v is None else [number(v["difference"]), number(v["se"]), number(v["difference"] / v["se"], 3)])
                            + ([""] * 7 if s is None else [s["usable"], number(s["mean"]), number(s["sd"]), number(s["low"]), number(s["high"]),
                                                           s["cohorts_at_or_above_recorded"], number(s["share_given_by_the_law"])]))
                if which != "plain" and v is not None:
                    for rest in ("pre", "post"):
                        terms.append([group, which, kind, rest, number(v[rest]["overlapping"]), number(v[rest]["separate"]), number(v[rest]["difference"]), number(v[rest]["se"])])
    print(write(prefix + "_content.tsv", ["group", "estimate", "term", "recorded_difference", "se", "z", "cohorts_of_the_law", "their_mean", "their_sd",
                                          "their_2.5th_percentile", "their_97.5th_percentile", "cohorts_at_or_above_recorded", "share_given_by_the_law"], rows), "content rows")
    print(write(prefix + "_terms.tsv", ["group", "estimate", "term", "rest", "overlapping", "separate", "overlapping_minus_separate", "se"], terms), "term rows")
    if summary is not None:
        print("reading:", json.dumps(summary.get("reading")))


def check_tables(folders, prefix):
    truth, runs = {}, {}
    for folder in folders:
        if (folder / "truth.json").exists():
            truth = json.loads((folder / "truth.json").read_text())
        for path in sorted(folder.glob("*.json")):
            d = json.loads(path.read_text())
            if isinstance(d, dict) and "rows" in d and "scenario" in d:
                runs[d["scenario"]] = d
    if truth:
        rows = [[name, truth["events"], truth["cohorts"], number(e["participation"][0]), number(e["participation"][1]), number(e["moves"])]
                + [number(e[f"{which}_{kind}"]["mean"]) for which in ("plain", "loglinear") for kind in KINDS] for name, e in truth["summary"].items()]
        print(write(prefix + "_check_truth.tsv", ["scenario", "ripples_per_rest_and_session", "cohorts", "participation_before", "participation_after", "planted_moves"]
                    + [f"{which}_{kind}" for which in ("plain", "loglinear") for kind in KINDS], rows), "truth rows")
    rows = []
    for name, run in runs.items():
        for key, v in run["summary"].items():
            if isinstance(v, dict) and "mean" in v:
                rows.append([name, run["replicates"], run["seed"], key, v.get("usable", ""), number(v["mean"]), number(v.get("sd")), number(v.get("se")),
                             number(v.get("low")), number(v.get("high"))])
    print(write(prefix + "_check.tsv", ["scenario", "replicates", "seed", "quantity", "usable", "mean", "sd_over_replicates", "mean_se_or_se_of_mean",
                                        "2.5th_percentile", "97.5th_percentile"], rows), "check rows")
    planted = {name for name, run in runs.items() if run["summary"].get("moves", 0) > 0}
    rows = []
    for which in ESTIMATES:
        for kind in KINDS:
            def values(name):
                return np.array([r["recorded"][which][kind]["difference"] for r in runs[name]["rows"] if r["recorded"][which][kind] is not None])
            null = np.concatenate([values(name) for name in runs if name not in planted])
            cut = float(np.quantile(null, 0.975))
            for name in runs:
                v = values(name)
                rows.append([which, kind, name, "yes" if name in planted else "no", len(v), number(float(v.mean())), number(float(v.std(ddof=1))),
                             len(null), number(float(null.mean())), number(float(null.std(ddof=1))), number(cut), number(float((v > cut).mean()))])
    print(write(prefix + "_check_above.tsv", ["estimate", "term", "scenario", "planted_change", "replicates", "mean", "sd", "replicates_without_planted_change",
                                              "their_mean", "their_sd", "their_97.5th_percentile", "share_of_this_scenario_above_it"], rows), "above rows")


if __name__ == "__main__":
    root, prefix = Path(sys.argv[1]), sys.argv[2]
    if (root / "study.json").exists():
        content_tables(root, prefix)
    if len(sys.argv) > 3:
        check_tables([Path(p) for p in sys.argv[3:]], prefix)
