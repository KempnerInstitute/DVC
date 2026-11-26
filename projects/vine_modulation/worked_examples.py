"""Generate the conditional-dependence examples and count E31 recovery from individual receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from dvc_package.modulation import trajectory, vinetraj

LAWS = ("step", "ushape", "clayton", "sum", "hub", "mixed", "exposed", "nested")
CASES = ("pair", "constant", "reversal", "quartet")


def profiles(case):
    t = np.arange(12)
    episode = (t >= 4) & (t < 8)
    baseline = np.where(episode, 0.55, -0.15) if case in ("pair", "constant") else np.zeros(12)
    slopes = {"pair": np.zeros(12), "constant": np.full(12, 0.42),
              "reversal": np.where(episode, -0.58, 0.58), "quartet": np.where(episode, 0.6, 0.0)}
    return np.column_stack((baseline, slopes[case]))


def sample(rng, beta, case, n):
    z1, z2, x, noise = rng.normal(size=(4, len(beta), n))
    feature = z1 * z2 if case == "quartet" else z1
    eta = beta[:, :1] + beta[:, 1:] * feature
    rho = np.clip(np.tanh(eta), -0.98, 0.98)
    y = rho * x + np.sqrt(1 - rho ** 2) * noise
    return trajectory.edge_data(ndtr(x), ndtr(y), feature[:, :, None], centre=False)


def examples():
    rows = {}
    for i, case in enumerate(CASES):
        seed = 118301 + i
        rng = np.random.default_rng(seed)
        truth = profiles(case)
        train, validation, test = [sample(rng, truth, case, n) for n in (300, 300, 1000)]
        fit, _ = vinetraj.tune(train, validation, 1, 300)
        rows[case] = {"seed": seed, "true": truth.tolist(), "fitted": fit.theta.tolist(),
                      "n_train": 300, "n_validation": 300, "n_test": 1000,
                      "test_log_density": trajectory.score(fit.theta, test)}
    return rows


def challenge(folder):
    rows = []
    ids = set()
    for path in sorted(folder.glob("e31_a2_*_n*_g*_s*.json")):
        raw = path.read_bytes()
        s = json.loads(raw)["summary"]
        key = (s["law"], s["n"], s["seed"])
        if key in ids or s["d"] != 50 or s["replicates"] != 399:
            raise ValueError(f"Unexpected challenge result {path.name}")
        ids.add(key)
        terms = []
        for p in s["planted"]:
            if p["shape"] not in ("a", "b", "constant", "ramp"):
                raise ValueError(f"Unknown trajectory in {path.name}")
            terms.append(dict(order=p["order"], shape=p["shape"], installed=bool(p["installed"]),
                              changed=bool(p["change"]),
                              exact=bool(p["installed"] and p["windows_on"] == p["windows_true"])))
        rows.append(dict(law=key[0], n=key[1], seed=key[2], g=s["g"], terms=terms,
                         sha256=hashlib.sha256(raw).hexdigest()))
    expected = {(law, n, seed) for law in LAWS for n in (300, 1000) for seed in range(990000, 990004)}
    if ids != expected:
        raise ValueError(f"Incomplete or unexpected cohort: {ids ^ expected}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("e31", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    report = {"git": os.environ["DVC_PAPER_PIN"], "examples": examples(), "challenge": challenge(args.e31)}
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
