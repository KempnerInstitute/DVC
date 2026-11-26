# Model comparisons

A comparison is most informative when the models receive the same training observations, marginal transformation and held-out outcomes.

| Comparator | Question |
| --- | --- |
| Gaussian copula | Is a Gaussian dependence model sufficient for the joint density? |
| Vine truncated after the first tree | How much predictive fit comes from conditional edges in a particular factorization? |
| Gaussian precision or state-space dynamics | Can evolving covariance account for the change? |
| Generalized additive pair copula | On a known edge, how well does smooth variation with time and a conditioning covariate recover an episode? |
| Persistent modulation | Once the same changing pairwise background is fitted, does allowing the modulation itself to vary improve prediction? |
| Replay model for hippocampal ripples | How much of the recorded three-cell contrast is reproduced when cells act independently given the represented track stretch and its strength? |

The [vine modulation study](../user-guide/vine-modulation.md) gives the simulation designs, R comparator and held-out neural comparisons. Other reference models are available in `dvc_package.baselines`. A gain over first-tree truncation concerns that factorization and can arise from pairwise population structure.
