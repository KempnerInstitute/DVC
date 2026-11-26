"""Fit a continuous conditional vine edge or a binary trio with fixed pair margins."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np


def binary_example():
    from dvc_package.modulation.binary_pooled import pooled_change

    patterns = np.array(list(product((0, 1), repeat=3)), dtype=int)
    odd = patterns.sum(axis=1) % 2 == 1
    before = np.repeat(patterns, 40, axis=0)
    after = np.repeat(patterns, np.where(odd, 60, 20), axis=0)
    pair_tables = {}
    for i, j in combinations(range(3), 2):
        def counts(events):
            return np.bincount(2 * events[:, i] + events[:, j], minlength=4)
        a, b = counts(before), counts(after)
        np.testing.assert_array_equal(a, b)
        pair_tables[f"{i},{j}"] = a.tolist()
    estimate = pooled_change(before, after, [[0, 1, 2]])
    return dict(events_per_period=len(before), pair_counts_in_either_period=pair_tables,
                before_counts=[40]*8, after_counts=np.where(odd,60,20).tolist(),
                pattern_order=["".join(map(str,w)) for w in patterns],
                population_log_odds_change=float(4*np.log(3)),
                conditional_estimate=float(estimate.delta), units="natural log odds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=("continuous", "spikes"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.example == "continuous":
        from worked_examples import examples
        result = examples()
    else:
        result = binary_example()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
