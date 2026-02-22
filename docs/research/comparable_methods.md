# Comparable Methods and Experiment Extensions

This page maps DVC to strong comparison families for high-dimensional, time-varying dependency analysis.

## Baselines Already Implemented in This Repository

- Gaussian copula
- 1-truncated C-vine (pairwise-only vine baseline)
- Graphical Lasso Gaussian copula
- TVGL-style Gaussian precision dynamics (`tvgl_frobenius`)
- Gaussian copula state-space model (`gaussian_copula_state_space_nll_fit_eval`)
- KDE-flow time-conditioned bandwidth baseline

## Recommended Additional Comparables

Use these to strengthen the NeurIPS positioning beyond copula-specific comparisons.

1. Dynamic conditional correlation (DCC)
   - Why: canonical dynamic covariance baseline with explicit temporal parameters.
   - Reference: Engle (2002), Journal of Business and Economic Statistics.
   - Link: https://www.nber.org/papers/w8554
2. Latent dynamical models for neural populations
   - Why: strong neural-data baselines for time-varying shared structure.
   - References: GPFA (Yu et al., 2009), PLDS (Macke et al., 2011), LFADS (Pandarinath et al., 2018).
   - Links:
     - GPFA: https://pmc.ncbi.nlm.nih.gov/articles/PMC2712272/
     - LFADS: https://www.nature.com/articles/s41592-018-0109-9
3. Time-series causal discovery
   - Why: complementary structure-recovery baseline under lagged nonlinear dependencies.
   - Reference: PCMCI (Runge et al., 2019).
   - Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC6881151/
4. Kernel and neural dependence measures
   - Why: nonparametric dependence detection without explicit joint density models.
   - References: HSIC (Gretton et al., 2005), MINE (Belghazi et al., 2018), distance correlation (Szekely et al., 2007).
   - Links:
     - MINE: https://proceedings.mlr.press/v80/belghazi18a.html
     - Distance correlation: https://arxiv.org/abs/0803.4101
5. High-order information diagnostics
   - Why: direct synergy/redundancy summaries complementary to likelihood metrics.
   - Reference: O-information (Rosas et al., 2019).
   - Link: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.032305

## Suggested New Simulation Scenarios

1. Matched pairwise and matched tails, changing only conditional dependence
   - Construct two regimes with identical pairwise copulas but different higher-tree terms.
   - Goal: isolate the benefit of full-vine conditioning beyond level-0 edges.
2. Regime-switching latent factor model with nonlinear observation map
   - Example: latent AR(1) factors, observed through monotone and non-monotone transforms.
   - Goal: test robustness to realistic nonstationary neural-like dynamics.
3. Mixed discrete-continuous dependencies
   - Spike-count-like discrete channels plus continuous LFP-like channels.
   - Goal: evaluate practical relevance for multimodal neuroscience data.
4. Scaling stress test
   - Sweep variables (`d`), samples per window (`N`), and time slices (`T`).
   - Goal: report runtime, memory, and NLL tradeoff curves against baselines.
5. Controlled misspecification
   - Add missingness, outliers, and marginal distribution drift with fixed copula.
   - Goal: verify that dependence conclusions remain stable under preprocessing noise.

## Recommended Evaluation Outputs

- Held-out copula NLL and NLL gaps per time slice
- Change-point detection error for known switch times
- Structure recovery metrics:
  - edge precision/recall/F1
  - family confusion matrix
  - hub/root recovery accuracy
- Tail-dependence recovery error (`lambda_U`, `lambda_L`)
- Runtime and memory scaling curves
- Seed and window-size sensitivity intervals
