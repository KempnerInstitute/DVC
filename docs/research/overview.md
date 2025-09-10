# Research overview

This codebase enables time-dependent inference with vine copulas. Pair-copula bandwidths are parameterized as functions of time and optimized by maximum likelihood, allowing nonparametric dependence to evolve.

Key elements:
- Time-conditioned pair copulas via per-edge flows (MLP or B-spline flows).
- Optional structure policies (static or periodic re-optimization).
- Information-theoretic readouts over time (entropy, MI, conditional MI).
- Scenario generators and benchmark runner with CSV logging.

Positioning:
- Prior work uses static neural spline flows to define copulas and marginals. Here, temporal dynamics are explicit in the pair-copula mechanism and structure policy. Compare to mixed vine copula flows in [arXiv:2207.04832](https://arxiv.org/abs/2207.04832).

Roadmap:
- Richer per-edge flows (rational-quadratic splines) and covariate-conditioned dynamics.
- Time-conditional sampling and predictive evaluation.
- Change-point detection and edge-persistence metrics for interaction dynamics.
