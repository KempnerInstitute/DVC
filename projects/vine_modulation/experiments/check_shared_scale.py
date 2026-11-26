"""Measure the planted change in the units of each training fold of the shared-change control."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import e69_replay_vine as ripple


def recording_units(item):
    group, index, name, content, census, seed = item
    rec = ripple.recording(content, name, census)
    original = ripple.features(rec["x"], rec["sets"], rec["order"])
    reference = {k: float(original[k].std()) or 1.0 for k in ("no","ns")}
    _, folds = ripple.blocks_of(rec["epoch"])
    rows = []
    for replicate in range(24):
        rng = np.random.default_rng([seed,index,replicate])
        x = ripple.generate(rec,ripple.SCENARIOS["apart"],rng)
        observed = ripple.features(x, rec["sets"], rec["order"])
        for fold in range(ripple.FOLDS):
            fitted = {k: float(observed[k][folds != fold].std()) or 1.0 for k in reference}
            shifts = {k: 0.15 * fitted[k] / reference[k] for k in reference}
            rows.append(dict(group=group, recording=name, replicate=replicate, fold=fold,
                             overlap_shift=shifts["no"], separate_shift=shifts["ns"],
                             difference=shifts["no"]-shifts["ns"]))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("second_content",type=Path)
    parser.add_argument("second_census",type=Path)
    parser.add_argument("first_content",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    tasks=[]
    for g,src,census,seed in (("cohort",args.second_content,args.second_census,990690),
                              ("achilles",args.first_content,None,990693)):
        tasks.extend((g,i,name,src,census,seed) for i,name in enumerate(ripple.GROUPS[g]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        data=[row for record in pool.map(recording_units,tasks) for row in record]
    assert len(data)==13*24*5
    with args.output.open("w") as out:
        writer=csv.DictWriter(out, list(data[0]), delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    for g in ("cohort","achilles"):
        diff=[abs(x["difference"]) for x in data if x["group"]==g]
        print(g, "absolute scale discrepancy, median and 90th percentile:", np.quantile(diff,[.5,.9]))


if __name__=="__main__":
    main()
