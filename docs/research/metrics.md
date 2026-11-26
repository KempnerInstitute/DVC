# Interpreting the outputs

| Quantity | Interpretation |
| --- | --- |
| Held-out log density | Predictive fit on new observations. Score competing models on the same observations and marginal transformation. |
| Edge family and parameter | The shape and strength of association of one pair in its conditional coordinates. |
| Continuous modulation coefficient | Change in the conditional pair parameter as a function of a conditioning feature. The selected vine order, feature and parameter scale are part of its definition. |
| Binary three-cell contrast | Difference between the pair's conditional log odds ratios when the third cell is active and when it is silent. The change can be estimated at fixed pair margins. |
| Exact interval recovery | In a simulation with known truth, every active and inactive window is identified. |
| Whole-event log score | The log probability of a complete held-out population pattern, including its sequential conditional predictions. |

Entropy and mutual information summarize distributions or groups of variables. An edge parameter or a gain from adding a vine tree has a different interpretation. [Density and information estimates](../user-guide/evaluation.md) · [Neural study](../user-guide/vine-modulation.md)
