"""Estimate temporal changes in the dependence of a pair on other variables.

Use `fit(train, validation, confirm, test)` for a continuous population.
Each split is a sequence of observations-by-variables arrays, one per
window, with consistent variable identities. A fitted coefficient refers
to a conditioning feature on a particular edge of the selected vine.
Use the held-out scores and the change decisions for their respective
prediction and timing questions.

For three binary indicators, `pooled_change(before, after, trios)`
estimates a shared change in the conditional log odds ratio at fixed pair
margins. Rows are events and each trio identifies three column indices.
"""
from .binary_pooled import pooled_change
from .vinemodel2 import fit

__all__ = ["fit", "pooled_change"]
