# Showcase Changes

The original four-phase showcase mixed a Gaussian star in phase 2 with a
root-aligned triplet in phase 3. The phase-3 helper used by that version could
overwrite the original star root, so phase 3 was not always a clean
"phase-2 star plus higher-order structure" contrast.

The final workflow keeps the original semantics available through
`--preset current` and `--preset fixed`, but recommends `contrast_harder` for
the paper-facing comparison:

- phase 1: independent variables
- phase 2: moderate Gaussian star, rooted at `X_0`
- phase 3: the same star plus two disjoint multiplicative triplet blocks
  with stress-test noise `0.10`
- phase 4: strong Clayton lower-tail dependence

This makes the benchmark less Gaussian-friendly. The Gaussian SSM remains a
strong comparator in the pairwise star phase, while DVC should separate itself
when the signal is higher-order or tail-asymmetric. The phase-3 oracle TC is
large because the multiplicative triplets are close to deterministic, so it is
best interpreted as a difficult detection/decomposition stress test rather
than an easy absolute-calibration benchmark.

The regenerated workflow also writes explicit oracle targets into
`summary.json`: analytic TC for the Gaussian star, Monte Carlo/oracle TC for
the multiplicative triplets, Monte Carlo MI plus exact lower-tail coefficient
for the Clayton phase, and the corresponding pairwise/higher-order breakdown.
These fields should be used to judge correctness, not just separation from a
baseline. For calibration sweeps, use `--multiplicative-noise-std` to make
the multiplicative phase less deterministic and compare absolute TC errors
against the oracle fields.
