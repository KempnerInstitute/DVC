"""Test of a change in the strength of a detected modulation (presence is tested by the scan; this tests change).

The fitted coefficient of a modulation depends slightly on the pairwise layer (rank-based margins attenuate it,
differently for different pair families), so comparing coefficients across windows directly would report changes
when only the pairwise dependence changes.  The test therefore calibrates the statistic by a parametric bootstrap
from the fitted model in which the modulation coefficient is constant over the run while everything else, including
the changing pairwise layer, is kept as fitted.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from . import cvine
from . import families as fam


def carrier(order: Sequence[int], subset: Sequence[int]):
    """The edge of a C-vine with this root order that carries the interaction ``subset``: (level, edge index,
    positions of the modulating roots).  The edge joins the two members that come last in the order."""
    positions = sorted(list(order).index(int(v)) for v in subset)
    level, last = positions[-2], positions[-1]
    return level, last - level - 1, tuple(positions[:-2])


def coefficients(model, windows, level, edge, components, ridge=1e-3, eps=fam.EPS, which=None):
    """Per-window coefficients (T, 1 + number of components) of a Gaussian modulated copula for one edge, given the
    model's transforms below that edge.  ``which``: windows to compute (others stay nan)."""
    out = np.full((len(windows), 1 + len(components)), np.nan)
    for t, u in enumerate(windows):
        if which is not None and t not in which:
            continue
        f = cvine.forward(model, t, u, eps, keep=True)
        v = f['pits'][level]
        out[t] = cvine.fit_modulated_gaussian(v[:, level + 1 + edge], v[:, level],
                                              cvine.component_features(components, f['zeta'][:, :level]), ridge, eps)[0]
    return out


def cusum(values):
    """Largest absolute partial sum of the centred values, divided by sqrt(length)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(np.max(np.abs(np.cumsum(values - values.mean())[:-1])) / np.sqrt(values.size))


def change_test(model, windows: Sequence[np.ndarray], subset, run, replicates=99, rng=None, refit=None):
    """Bootstrap test of a constant modulation coefficient of the interaction ``subset`` over the windows ``run``.

    The coefficient is that of the linear shape for a triple and of the product for a quadruple, estimated in every
    window of the run on the edge that carries the subset in the model's vine.  refit(list of windows) must return a
    model fitted exactly as ``model`` was (same root order, options and allowed interactions).
    Returns a dict with the coefficient path, the statistic, the bootstrap p-value and the mean bootstrap statistic,
    or None when the run has fewer than four windows.
    """
    rng = np.random.default_rng() if rng is None else rng
    run = [int(t) for t in run]
    if len(run) < 4:
        return None
    level, edge, members = carrier(model.order, subset)
    component = ('linear', members) if len(members) == 1 else ('product', members)
    beta = coefficients(model, windows, level, edge, (component,), which=set(run))
    statistic = cusum(beta[run, 1])
    null = copy.deepcopy(model)
    slope = float(beta[run, 1].mean())
    for t in run:
        null.edges[level][edge][t] = cvine.EdgeState('gaussian', components=(component,), beta=np.array([beta[t, 0], slope]))
    draws = np.empty(replicates)
    for b in range(replicates):
        sim = [cvine.pseudo_observations(cvine.simulate(null, t, u.shape[0], rng)) for t, u in enumerate(windows)]
        again = refit(sim)
        draws[b] = cusum(coefficients(again, sim, level, edge, (component,), which=set(run))[run, 1])
    return dict(level=level, edge=edge, windows=run, coefficients=beta[:, 1].tolist(), statistic=statistic,
                p=float((1 + np.sum(draws >= statistic)) / (replicates + 1)), null_mean=float(draws.mean()))
