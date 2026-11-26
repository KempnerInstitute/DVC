"""Compare standardized and per-active-pair changes on the matched simulations and recorded ripples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

import e69_replay_vine as ripple


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(path, rows):
    with path.open("w") as out:
        writer = csv.DictWriter(out, list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_replicates(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 24 and sorted(r["replicate"] for r in rows) == list(range(24)), path
    return sorted(rows, key=lambda r: r["replicate"])


def summarize(primary_recordings, primary_calibration, current, output):
    result, totals, scales, receipts = [], [], [], {}
    output.mkdir(parents=True, exist_ok=True)
    with (current/"scale_differences.tsv").open() as handle:
        scale_rows = list(csv.DictReader(handle, delimiter="\t"))
    for group in ("cohort", "achilles"):
        ids = set(ripple.GROUPS[group])
        rows = [r for r in scale_rows if r["group"] == group]
        assert {(r["recording"],int(r["replicate"]),int(r["fold"])) for r in rows} == {
            (name, rep, fold) for name in ids for rep in range(24) for fold in range(5)}
        delta = np.abs([float(r["difference"]) for r in rows])
        scales.append(dict(group=group, recordings=len(ids), simulations=24, folds=5,
                           original_shared_change=0.15, median_absolute_difference=np.median(delta),
                           p90_absolute_difference=np.quantile(delta,.9), maximum_absolute_difference=delta.max()))
        for scenario in ("replay","apart","class"):
            old = primary_calibration/group/scenario/(scenario+".jsonl")
            new = current/group/scenario/(scenario+".jsonl")
            a, b = load_replicates(old), load_replicates(new)
            for left, right in zip(a,b):
                for field in ("group","scenario","seed","replicate"):
                    assert left[field] == right[field], (group,scenario,field)
                assert right["unit_rule"] == "common" and set(right["recordings"]) == ids
                for name, record in right["recordings"].items():
                    earlier = left["recordings"][name]
                    assert earlier["ripples"] == record["ripples"]
                    if scenario != "apart":  # Both scoring rules saw exactly the same planted activity.
                        np.testing.assert_allclose(earlier["pairs"]["total"], record["pairs"]["total"],
                                                   atol=2e-5,rtol=1e-10)
            receipts[group+"/"+scenario] = {"primary":sha(old), "per_pair":sha(new)}
            for contrast in ("time_varying","class_change"):
                hits = sum(row["cohort"][contrast]["z"] >= 1.96 for row in b)
                result.append(dict(group=group,scenario=scenario,comparison=contrast,cohorts=len(b),
                                   crossings=hits,proportion=hits/24,lower=beta.ppf(.025,hits,25-hits) if hits else 0.,
                                   upper=beta.ppf(.975,hits+1,24-hits) if hits<24 else 1.,
                                   same_generated_ripples_as_primary=int(scenario!="apart")))
        events = []
        for name in sorted(ids):
            path = current/"recordings"/(name+".json")
            row = json.loads(path.read_bytes())
            ref = json.loads((primary_recordings/path.name).read_bytes())
            assert row["ripples"] == ref["ripples"] and row["place_cells"] == ref["place_cells"]
            folds = row["chosen"]["folds"]
            assert len(folds)==5 and all(f["units"]["no"]==f["units"]["ns"]>0 for f in folds)
            with np.load(path.with_suffix(".npz")) as new, np.load(primary_recordings/(name+".npz")) as old:
                for key in ("block","replay","pairs"):
                    np.testing.assert_allclose(new[key],old[key],rtol=0,atol=2e-8)
                for key,(top,bottom) in ripple.COMPARISONS.items():
                    np.testing.assert_allclose((new[top]-new[bottom]).sum(),row["compared"][key]["total"],rtol=1e-10,atol=1e-7)
            events.append(row["compared"])
            receipts[name] = {"primary":sha(primary_recordings/(name+".npz")), "per_pair":sha(path.with_suffix(".npz"))}
        group_results = ripple.gathered(events)
        for contrast in ripple.COMPARISONS:
            score = group_results[contrast]
            totals.append(dict(group=group,comparison=contrast,recordings=len(events),ripples=group_results["ripples"],
                               total_nats=score["total"],blocked_se=score["se"],positive_recordings=score["recordings_positive"]))
    write_table(output/"common_count_scenarios.tsv",result)
    write_table(output/"common_count_populations.tsv",totals)
    write_table(output/"common_count_scale_reference.tsv",scales)
    (output/"common_count_checksums.json").write_text(json.dumps(receipts,indent=2,sort_keys=True)+"\n")
    for g in ("cohort","achilles"):
        print(g,{r["scenario"]:r["crossings"] for r in result if r["group"]==g and r["comparison"]=="class_change"})
        print([r for r in totals if r["group"]==g and r["comparison"]=="class_change"][0])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("primary_recordings","primary_calibration","current","output"):
        cli.add_argument(key,type=Path)
    summarize(**vars(cli.parse_args()))
