"""Compare a schedule-based split with the learned replay split on the same held-out ripples."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import e71_blind_timing as timing


def score_session(name, content, census, reference):
    rec = timing.e69.recording(content,name,census)
    x, rates, window = rec["x"],rec["rates"],rec["window"]
    count = len(x)
    rank = (np.arange(count)+.5)/count
    blocks,folds = timing.e69.blocks_of(rec["epoch"])
    estimates = np.full(count,np.nan)
    with np.load(reference/f"{name}.npz") as stored:
        np.testing.assert_array_equal(stored["block"],blocks)
        np.testing.assert_array_equal(stored["rank"],rank)
        for fold in range(timing.FOLDS):
            train,test = np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            # Refitting the stationary baseline verifies the data, folds and scoring code.
            one = timing.content.Content(x[train],rates,window)
            np.testing.assert_allclose(timing.evidence(one,x[test]),stored["none"][test],rtol=0,atol=2e-8)
            for early in (True,False):
                side = (rank<1/3)==early
                fit,held = train[side[train]], test[side[test]]
                if len(fit)<timing.MINIMUM:
                    raise ValueError(f"Insufficient training ripples in {name}, fold {fold}")
                two = timing.content.Content(x[fit],rates,window)
                if len(held):
                    estimates[held] = timing.evidence(two,x[held])
        assert np.isfinite(estimates).all()
        result = {}
        for label,delta in (("schedule_minus_stationary",estimates-stored["none"]),
                            ("learned_minus_schedule",stored["blind"]-estimates),
                            ("given_minus_schedule",stored["given"]-estimates)):
            result[label] = timing.compared(delta,blocks)
    return dict(recording=name,ripples=count,comparisons=result)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("content","census","reference","output"):
        cli.add_argument(name,type=Path)
    args=cli.parse_args()
    results=[]
    for name in timing.GROUPS["cohort"]:
        r=score_session(name,args.content,args.census,args.reference)
        results.append(r)
        print(name,r["ripples"],flush=True)
    assert len(results)==9
    rows=[]
    for recording in results:
        for key,r in recording["comparisons"].items():
            rows.append(dict(recording=recording["recording"],contrast=key,total_nats=r["total"],
                             block_se=np.sqrt(r["variance"]),ripples=recording["ripples"]))
    for key in results[0]["comparisons"]:
        subset=[r for r in rows if r["contrast"]==key]
        rows.append(dict(recording="cohort",contrast=key,total_nats=sum(r["total_nats"] for r in subset),
                         block_se=np.sqrt(sum(r["block_se"]**2 for r in subset)),ripples=sum(r["ripples"] for r in subset)))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"schedule_predictions.tsv").open("w") as out:
        w=csv.DictWriter(out,rows[0],delimiter="\t",lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (args.output/"schedule_checks.json").write_text(json.dumps(dict(scheduled_fraction=1/3,
        recordings=len(results),folds_per_recording=timing.FOLDS,verified_stationary_tolerance=2e-8),indent=2)+"\n")


if __name__ == "__main__":
    main()
