# E78. The replay model with one set of cell factors for both rests

Written on 26 November 2026, before the shared-factor law was fitted to any recording (E66 and E68 had been read;
`RESULTS.md` sections 66 and 68).

## Why

The content-only law of `real_data/dryad_posthoc_content.py` is fitted separately to each rest: the weights of the
states, the weights of the gains **and the cell factors** `c_i` in

    P(cell i active | state, gain) = 1 - exp(-gain * c_i * r_i(state) * 0.1 s).

A referee can say that part of the reproduced three-way rise comes from cells becoming more excitable in the rest
after the experience (larger `c_i`), not from a change of what is replayed. A higher factor makes a cell follow
the replayed state more steeply, which by itself raises the pair and three-way terms that a mixture of states
induces among the cells of one place.

## Design

`real_data/shared_factor_content.py` (new file; nothing existing changes). One law per session (per hemisphere in
the first cohort) with **one factor per cell shared by the two rests** and, as before, separate weights of the
states and of the gains in each rest. Fitted by the same EM: in each round the posteriors over state and gain are
computed in each rest under that rest's weights; the weights of a rest are updated from its own posteriors; each
cell's factor maximises the expected log likelihood summed over the events of both rests (pooled expected
activity and exposure); the rounds are extrapolated in pairs (SQUAREM) over the joint parameter vector (log
factors, then the weight margins of each rest); the same start (one factor for all cells matching the pooled mean
participation at no place and gain 1, equal weights); stop when a cycle raises the summed log likelihood of both
rests by less than 0.02 (the separate fits stop at 0.01 per rest) or after 300 cycles.

Then, unchanged, the estimates and the simulation of E66 (second cohort, nine sessions,
`dryad_posthoc_content.py`) and E68 (first cohort, 13 hemispheres, `hippocampal_posthoc_content.py`): `plain`
(E64's estimator), `loglinear`, `adjusted` (log-linear tilt of the law's prediction from the ripple's other
place cells), 200 cohorts drawn from the fitted shared-factor laws, the shared-factor law fitted again to every
cohort, seeds 990782 (second cohort) and 990784 (first cohort). Inputs: the stored sessions of E66
(`hippocampus/dryad_content_v1/sessions`) and E68 (`hippocampus/posthoc_content_v1/sessions`); outputs
`e78/dryad_shared_v1` and `e78/posthoc_shared_v1`.

Also reported: the in-sample log likelihood of the shared-factor and the separate laws on the recorded ripples
(the shared law is the separate law with `c_i` equal in the two rests, so it can only lose; the separate law has
one more parameter per place cell, which gives AIC and BIC differences), how much the factors of the separate
fits move between rests (`log c_POST - log c_PRE`, median absolute value and mean, by session), and the fitted
share of ripples on a stretch of at most about 1 m before and after under the shared-factor law.

## Expectations (mine, before the run)

1. The separate factors fit better in every session, by more than the AIC penalty in most of them: excitability
   relative to the running fields differs between sleep or rest periods.
2. Second cohort: the cohorts of the shared-factor laws still give at least half of the recorded `plain`
   three-way change (+0.296), because the share of ripples on a stretch and the length of the stretches rise after
   the tracks in eight of nine sessions under the separate laws, and that does not need new factors.
3. Second cohort: the recorded `adjusted` three-way difference under the shared-factor laws lies within the central
   95% of its cohorts (explained by content).
4. Achilles: as in E68, the cohorts give about half of the recorded `plain` three-way change and the `adjusted`
   three-way difference lies above the 97.5th percentile of the cohorts.

## What decides

As in E66 and E68, for the second cohort (all five rats) and for Achilles:

- The shared-factor law is a fair stand-in for replay only if its cohorts give at least half of the recorded
  `plain` change of the **pair** term; otherwise the comparison says nothing about content and is recorded as such.
- **The statement "a replay model reproduces the rise in the independent cohort" stands with factors fixed across
  rests** if the shared-factor cohorts give at least half of the recorded `plain` three-way change and the recorded
  `adjusted` three-way difference lies within their central 95%.
- **It is weakened** (the reproduction then needs cell factors that differ between rests, a change of
  excitability, and the paper must say so) if the cohorts give less than half of the recorded `plain` three-way
  change, or the recorded `adjusted` three-way difference lies above their 97.5th percentile.
- For Achilles the share of the recorded `plain` three-way change given by the shared-factor cohorts replaces the
  "about half" of E68 in whatever the paper says, and the `adjusted` reading is reported by the same rule.

## What it cannot say

The comparison is between two members of the same family. A shared factor removes a per-cell change of gain that
is the same for every state; it does not remove changes of a cell's field between running and the rests, or
changes of excitability that depend on the replayed place.
