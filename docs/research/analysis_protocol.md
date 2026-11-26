# Evaluating temporal dependence

1. Define the observation, time window and prediction target. A ripple is one binary population event. In the continuous simulations, each window contains repeated observations of the same variables.
2. Fit marginal transformations on training observations. Keep validation and final scoring data separate. For temporally correlated recordings, choose blocks appropriate to the dependence timescale.
3. Fit a pairwise background and compare persistent and changing modulation on that same background. State the pair, conditioning feature, vine order and coefficient scale.
4. Evaluate prediction and recovery separately. A model may increase held-out density while missing the exact set of participants or the start and end of an episode. Evaluate timing against known intervals when available.
5. Match uncertainty to the replication unit. Windows, ripples, sessions and animals answer different questions. A simulation reference distribution and a confidence interval for a recorded effect serve different purposes.

The [study guide](../user-guide/vine-modulation.md) links the recording protocols, matched controls, examples and figure data used in the hippocampal application.
