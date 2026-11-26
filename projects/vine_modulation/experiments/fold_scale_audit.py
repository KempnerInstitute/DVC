"""Verify per-ripple held-out scores and export fixed and searched boundary controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

FIXED = {"pairs": ("pairs","replay"), "modulation": ("persistent","pairs"),
         "time_varying": ("shared","persistent"), "class_change": ("class","shared")}
BLIND = {"blind_minus_none": ("blind","none"), "given_minus_blind": ("given","blind"),
         "learned_change": ("shared","persistent"), "learned_class": ("class","shared")}


def tsv(path, rows):
    with path.open("w") as out:
        writer = csv.DictWriter(out, rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(mode, previous, current, output):
    pairs = FIXED if mode == "fixed" else BLIND
    original, updated = (json.loads((f/"summary.json").read_text()) for f in (previous,current))
    expected = set(original["recordings"])
    def names(folder):
        return {p.stem for p in folder.glob("*.npz")}
    if len(expected) != 22 or names(previous) != expected or names(current) != expected or set(updated["recordings"]) != expected:
        raise ValueError("Incomplete set of recordings")
    checksums, records, timing = {}, [], []
    for name in sorted(expected):
        record = json.loads((current/f"{name}.json").read_text())
        folds = record["chosen"]["folds"] if mode == "fixed" else record["folds"]
        assert sorted(f["fold"] for f in folds) == list(range(5))
        assert all(set(f["units"]) == {"no","ns"} and all(v>0 and np.isfinite(v) for v in f["units"].values()) for f in folds)
        with np.load(previous/f"{name}.npz") as old, np.load(current/f"{name}.npz") as new:
            assert set(old.files) == set(new.files)
            assert all(np.isfinite(new[key]).all() for key in new.files)
            unmodified = ("block","replay","pairs") if mode == "fixed" else ("block","rank","none","blind","given")
            for key in unmodified:
                np.testing.assert_allclose(new[key],old[key],atol=2e-8,rtol=0)
            for contrast,(high,low) in pairs.items():
                d = (new[high]-new[low])
                event_count = sum(record["ripples"]) if mode == "fixed" else record["ripples"]
                assert len(d) == event_count and d.shape == new["block"].shape
                r = record["compared"][contrast] if mode == "fixed" else record[contrast]
                total = float(d.sum())
                np.testing.assert_allclose(total,r["total"],atol=1e-7,rtol=1e-10)
                records.append(dict(recording=name,comparison=contrast,place_cells=record["place_cells"],ripples=len(d),
                                    original_nats=float((old[high]-old[low]).sum()),training_scaled_nats=total,
                                    training_scaled_se=float(np.sqrt(r["variance"]))))
        if mode == "blind":
            boundary = record["boundary"]
            fitted = record["profile_best"]["location"]
            timing.append(dict(recording=name,boundary=boundary,fitted=fitted,one_third=1/3,
                               fitted_error=abs(fitted-boundary),one_third_error=abs(1/3-boundary)))
        checksums[name] = {ext:hashlib.sha256((current/f"{name}.{ext}").read_bytes()).hexdigest() for ext in ("json","npz")}
    tsv(output/f"{mode}_fold_recordings.tsv",records)
    if timing:
        tsv(output/"timing_reference.tsv",timing)
    comparisons = []
    for group in ("cohort","achilles"):
        for key in pairs:
            old, new = original["groups"][group][key], updated["groups"][group][key]
            comparisons.append(dict(group=group,comparison=key,original_nats=old["total"],original_se=old["se"],
                                    training_scaled_nats=new["total"],training_scaled_se=new["se"],
                                    recordings=updated["groups"][group]["recordings"],recordings_positive=new["recordings_positive"]))
    tsv(output/f"{mode}_fold_summary.tsv",comparisons)
    if mode == "fixed":
        rows = []
        for group in ("cohort","achilles"):
            g = updated["groups"][group]
            for key in pairs:
                e = g[key]
                rows.append(dict(group=group,comparison=key,total=e["total"],se=e["se"],z=e["z"],
                                 recordings=g["recordings"],ripples=g["ripples"],recordings_positive=e["recordings_positive"]))
        tsv(output/"whole_ripples.tsv",rows)
    receipt = dict(mode=mode,recordings=len(expected),folds_per_recording=5,
                   original_expectations=original["expectations"],training_scaled_expectations=updated["expectations"],
                   files_sha256=checksums)
    (output/f"{mode}_fold_receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(f"{mode}: checked individual predictions in all {len(expected)} recordings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixed","blind"))
    parser.add_argument("previous",type=Path)
    parser.add_argument("current",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    audit(args.mode,args.previous,args.current,args.output)
