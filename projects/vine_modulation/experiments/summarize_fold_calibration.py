"""Check every simulated cohort against its earlier replicate and report calibration by scenario."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


def trials(path):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==24 and sorted(row["replicate"] for row in rows)==list(range(24)),path
    return {r["replicate"]:r for r in rows}


def write_tsv(path,rows):
    with path.open("w") as out:
        writer=csv.DictWriter(out,list(rows[0]),delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(previous,current,output):
    all_rows,summary,receipts=[],[],{}
    for group,old_dir,seed,units in (("cohort","check_v1",990690,9),("achilles","check_achilles_v1",990693,4)):
        for scenario in ("replay","apart","class"):
            a_path=previous/old_dir/(scenario+".jsonl")
            b_path=current/group/scenario/(scenario+".jsonl")
            a,b=trials(a_path),trials(b_path)
            receipts[group+"/"+scenario] = dict(original=hashlib.sha256(a_path.read_bytes()).hexdigest(),
                                                training_scaled=hashlib.sha256(b_path.read_bytes()).hexdigest())
            for i in range(24):
                old,new=a[i],b[i]
                assert old["seed"]==new["seed"]==seed and old["scenario"]==new["scenario"]==scenario
                assert set(old["recordings"])==set(new["recordings"]) and len(new["recordings"])==units
                for name,record in new["recordings"].items():
                    orig=old["recordings"][name]
                    assert record["ripples"]==orig["ripples"]
                    np.testing.assert_allclose(record["pairs"]["total"],orig["pairs"]["total"],atol=2e-5,rtol=1e-10)
                for k in ("time_varying","class_change"):
                    old_z,new_z=old["cohort"][k]["z"],new["cohort"][k]["z"]
                    assert np.isfinite([old_z,new_z]).all()
                    all_rows.append(dict(group=group,scenario=scenario,replicate=i,comparison=k,
                                         old_z=old_z,training_scaled_z=new_z,training_scaled_nats=new["cohort"][k]["total"]))
            for k in ("time_varying","class_change"):
                rows=[r for r in all_rows if (r["group"],r["scenario"],r["comparison"])==(group,scenario,k)]
                yes=np.array([r["training_scaled_z"]>=1.96 for r in rows])
                was=np.array([r["old_z"]>=1.96 for r in rows])
                n,hits=len(rows),int(sum(yes))
                summary.append(dict(group=group,scenario=scenario,comparison=k,cohorts=n,original_exceedances=int(sum(was)),
                                    training_scaled_exceedances=hits,decisions_differ=int(sum(yes!=was)),
                                    mean_training_scaled_nats=np.mean([r["training_scaled_nats"] for r in rows]),
                                    proportion_lower=beta.ppf(.025,hits,n-hits+1) if hits else 0.,
                                    proportion_upper=beta.ppf(.975,hits+1,n-hits) if hits<n else 1.))
    output.mkdir(exist_ok=True,parents=True)
    write_tsv(output/"fold_calibration.tsv",summary)
    write_tsv(output/"fold_calibration_replicates.tsv",all_rows)
    (output/"fold_calibration_receipt.json").write_text(json.dumps(receipts,indent=2,sort_keys=True)+"\n")
    for r in summary:
        print(r["group"],r["scenario"],r["comparison"],r["training_scaled_exceedances"],"/",r["cohorts"])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("previous","current","output"):
        cli.add_argument(key,type=Path)
    summarize(**vars(cli.parse_args()))
