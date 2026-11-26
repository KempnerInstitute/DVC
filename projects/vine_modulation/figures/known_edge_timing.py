"""Read the saved E17/E17b fits and compare the active windows on matched simulations."""
import argparse
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path


DESIGNS = ("abrupt", "abrupt_static", "recurring")
SIZES = (100, 250, 1000)
PARTS = ("n100", "n250", "n1000a", "n1000b")
MODELS = ("joint", "switching", "smooth_x", "gamml")
FIELDS = ("design", "g", "n", "seed")
TARGETS = set(product(DESIGNS, (.3, .6), SIZES, range(980000, 980040)))


def paired_intervals(csv_path, draws=20000, seed=20261125):
    """Percentile Monte Carlo intervals, resampling matched datasets within each design and strength."""
    import numpy as np
    rng = np.random.default_rng(seed)
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    out=[]
    for n in SIZES:
        estimates=[]
        for design, strength in product(DESIGNS,(.3,.6)):
            cell = sorted((r for r in rows if int(r["n"])==n and r["design"]==design and float(r["g"])==strength),
                          key=lambda r:int(r["seed"]))
            assert len(cell)==40 and len({r["seed"] for r in cell})==40
            values=np.array([[int(r["half_joint"]),int(r["half_gamml"])] for r in cell])
            samples = rng.integers(0,len(cell),size=(draws,len(cell)))
            estimates.append(values[samples].mean(axis=1))
        both=np.mean(estimates,axis=0)
        for key,values in (("joint",both[:,0]),("gamml",both[:,1]),("joint_minus_gamml",both[:,0]-both[:,1])):
            lo,hi=np.quantile(values,(.025,.975))
            out.append(dict(n=n,quantity=key,lower=float(lo),upper=float(hi),draws=draws,seed=seed))
    return out


def read_fits(root, prefix, hashes):
    found = {}
    for part in PARTS:
        path = root / f"{prefix}_{part}.json"
        content = path.read_bytes()
        hashes[path.name] = hashlib.sha256(content).hexdigest()
        for item in json.loads(content):
            key = tuple(item[k] for k in FIELDS)
            if key not in TARGETS:
                continue
            if key in found or "error" in item:
                raise ValueError(f"Duplicate or failed fit in {path.name}: {key}")
            found[key] = item
    if found.keys() != TARGETS:
        raise ValueError(f"Missing {len(TARGETS - found.keys())} fits for {prefix}")
    return found


def half_peak(values):
    if len(values) != 24 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected 24 finite slopes")
    largest = max(map(abs, values))
    return {t for t, x in enumerate(values) if largest and abs(x) >= largest / 2}


def summarize(root, csv_path, summary_path):
    hashes = {}
    original = read_fits(root, "e17_joint_trajectories", hashes)
    addendum = read_fits(root, "e17b_addendum", hashes)
    rows = []
    for key in sorted(TARGETS):
        a, b = original[key], addendum[key]
        # Both runs regenerated the same realizations. Check two shared arms on all 24 windows.
        if any(a["arms"][m] != b["arms"][m] for m in ("oracle", "windows")):
            raise ValueError(f"The two sets of fits have different test observations: {key}")
        truth = half_peak(a["c"])
        active = {t for t, v in enumerate(a["c"]) if v != 0}
        if truth != active:
            raise ValueError(f"Half-peak and planted support differ on these rectangular episodes: {key}")
        fit = a["structure"]
        joint = {t for start, stop in fit["joint"]["pieces"][1] for t in range(start, stop)}
        switching = set(fit["switching"]["term_windows"][0])
        row = dict(zip(FIELDS, key))
        row["state_joint"] = int(active == joint)
        row["state_switching"] = int(active == switching)
        for model in MODELS:
            source = a if model in ("joint", "switching") else b
            row["half_" + model] = int(half_peak(source["arms"][model]["slope"]) == truth)
        rows.append(row)
    outcomes = list(rows[0])[len(FIELDS):]
    cells = []
    for n in SIZES:
        sample = [row for row in rows if row["n"] == n]
        cells.append(dict(n=n, datasets=len(sample),
                          **{k: sum(row[k] for row in sample) for k in outcomes}))
    totals = {k: sum(cell[k] for cell in cells) for k in ["datasets", *outcomes]}
    if (totals["datasets"], totals["state_joint"], totals["state_switching"]) != (720, 575, 66):
        raise ValueError("Native support counts differ from the study ledger")
    with csv_path.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        out.writeheader()
        out.writerows(rows)
    summary = dict(cells=cells, totals=totals, input_sha256=hashes,
                   csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                   original_generation_pin="180ba61", addendum_generation_pin="67b4172",
                   time_grid=24, exact="Every active window and no inactive window selected",
                   half_peak="Absolute slope >= 0.5 times its own peak. An all-zero path selects no windows.")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(cells))
    print(json.dumps(totals))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "csv", "summary"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    summarize(args.source, args.csv, args.summary)
