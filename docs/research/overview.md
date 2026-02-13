# Research Overview

DVC is aimed at modeling multivariate dependency structure beyond pairwise correlation, with an emphasis on:

- pair-copula decomposition (vine models),
- entropy and mutual-information analysis,
- and early time-dependent extensions via neural bandwidth flows.

## Current Research Questions This Code Supports

1. How much non-Gaussian dependence exists in a high-dimensional system?
2. Which vine structures (C/D/R) best capture observed dependency patterns?
3. How stable are estimated dependency structures across time windows?
4. How much information is missed by pairwise-only analyses?

## Practical Positioning

- The `core/` and `optimization/` modules are the most mature for reproducible studies.
- The `time/` modules are usable for synthetic experiments and method prototyping.
- `archive/` preserves TensorFlow-era implementations for historical comparison.

## Suggested Workflow

1. Start with static vine fitting and structure comparisons.
2. Quantify entropy/MI gains over baseline Gaussian or pairwise metrics.
3. Introduce temporal segmentation and then time-dependent flows.
4. Report robustness across seeds, sample sizes, and vine types.

See `docs/research/analysis_protocol.md` for a concrete protocol and reporting template.

