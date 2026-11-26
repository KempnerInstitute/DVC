"""Count proposed, installed and changing terms on the frozen 50-variable study receipts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit(folder, worked, output):
    expected={(r["law"],r["n"],r["seed"]):r["sha256"] for r in json.loads(worked.read_text())["challenge"]}
    rows=[]
    for path in sorted(folder.glob("e31_a2_*_n*_g*_s*.json")):
        raw=path.read_bytes()
        r=json.loads(raw)["summary"]
        key=(r["law"],r["n"],r["seed"])
        assert hashlib.sha256(raw).hexdigest()==expected.pop(key),key
        terms=r["unplanted"]
        assert r["d"]==50 and (r["g"]>0)==bool(r["planted"])
        rows.append(dict(law=r["law"],n=r["n"],seed=r["seed"],strength=r["g"],
                         planted_terms=len(r["planted"]),extra_installed=sum(bool(t["installed"]) for t in terms),
                         extra_changing=sum(bool(t["change"]) for t in terms),
                         test_gain=r["scores"]["assembled"]-r["scores"]["simplified"]))
    assert len(rows)==64 and not expected
    with output.open("w") as out:
        writer=csv.DictWriter(out,rows[0],delimiter="\t",lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for kind,subset in (("null",[r for r in rows if not r["strength"]]),("planted",[r for r in rows if r["strength"]])):
        print(kind,len(subset),sum(r["extra_installed"] for r in subset),
              sum(r["extra_changing"] for r in subset),sum(r["extra_changing"]>0 for r in subset),
              sum(r["test_gain"]>0 for r in subset))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("folder","worked","output"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    audit(args.folder,args.worked,args.output)
