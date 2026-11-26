"""Margins of the windows (11 November 2026).

Until now every window had its own margins: ranks within the window.  The rank transform of n observations is a
noisy estimate of the distribution function, and that noise attenuates a strong modulation.
Where the margin of a variable does not change with time, its distribution function can be estimated from all
windows together, with T times as many observations.

  window    ranks within every window; held-out observations through the window's training distribution
  pooled    ranks over all training windows; held-out observations through the pooled training distribution
  aligned   one shape per variable for all windows and the location and scale of every window: the values of a
            window are centred at its median and divided by its scale, then ranked over all windows
  choose    pooled for a variable unless a change of its location or spread over the windows is detected
            (`stationary`), window margins otherwise; with ``align`` a variable whose location or spread changes
            gets aligned margins unless its shape changes too (`same_shape`)
  ranks     the same choice applied to windows that have no held-out part (confirmation windows, simulated
            datasets of a bootstrap), with the flags of `choose`

A flag is True or 'pooled', 'aligned', or False or 'window'.

Windows are given as a list of arrays (n_t, d), or an array (T, n, d).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2_contingency, kruskal, rankdata

CLIP = 1e-6
BINS = 8                                                            # of the test of a common shape


def _held_out(ordered: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Training distribution function at held-out values, as `training_transform` of the experiments."""
    out = np.empty(values.shape)
    for j in range(values.shape[1]):
        out[:, j] = (np.searchsorted(ordered[:, j], values[:, j], side='right') + 0.5) / (ordered.shape[0] + 1.0)
    return np.clip(out, CLIP, 1.0 - CLIP)


def window(train: Sequence[np.ndarray], others: Optional[Dict[str, Sequence[np.ndarray]]] = None) -> Dict[str, List[np.ndarray]]:
    """Pseudo-observations with the margins of every window on its own."""
    train = [np.asarray(x, dtype=np.float64) for x in train]
    out = dict(train=[rankdata(x, axis=0, method='average') / (x.shape[0] + 1.0) for x in train])
    for role, windows in (others or {}).items():
        out[role] = [_held_out(np.sort(x, axis=0), np.asarray(y, dtype=np.float64)) for x, y in zip(train, windows)]
    return out


def pooled(train: Sequence[np.ndarray], others: Optional[Dict[str, Sequence[np.ndarray]]] = None) -> Dict[str, List[np.ndarray]]:
    """Pseudo-observations with one margin per variable for all windows."""
    train = [np.asarray(x, dtype=np.float64) for x in train]
    sizes = np.cumsum([0] + [x.shape[0] for x in train])
    everything = np.concatenate(train, axis=0)
    ranks = rankdata(everything, axis=0, method='average') / (everything.shape[0] + 1.0)
    out = dict(train=[ranks[a:b] for a, b in zip(sizes[:-1], sizes[1:])])
    ordered = np.sort(everything, axis=0)
    for role, windows in (others or {}).items():
        out[role] = [_held_out(ordered, np.asarray(y, dtype=np.float64)) for y in windows]
    return out


def _location_scale(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median of every column and half the distance between its 0.841 and 0.159 quantiles (one standard deviation
    for a normal margin)."""
    low, centre, high = np.quantile(x, (0.158655, 0.5, 0.841345), axis=0)
    return centre, np.maximum(0.5 * (high - low), 1e-300)


def _standardised(train: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    constants = [_location_scale(x) for x in train]
    return [(x - m) / s for x, (m, s) in zip(train, constants)], constants


def aligned(train: Sequence[np.ndarray], others: Optional[Dict[str, Sequence[np.ndarray]]] = None) -> Dict[str, List[np.ndarray]]:
    """Pseudo-observations with one shape per variable and the location and scale of every window.  The values of
    a window are centred at the window's median and divided by its scale; the standardised values of all training
    windows are ranked together.  Held-out observations of a window are standardised with the training values of
    that window."""
    train = [np.asarray(x, dtype=np.float64) for x in train]
    values, constants = _standardised(train)
    out = pooled(values)
    ordered = np.sort(np.concatenate(values, axis=0), axis=0)
    for role, windows in (others or {}).items():
        out[role] = [_held_out(ordered, (np.asarray(y, dtype=np.float64) - m) / s) for y, (m, s) in zip(windows, constants)]
    return out


def _thinned(train: Sequence[np.ndarray], block: int) -> List[np.ndarray]:
    """Every ``block``-th observation of every window, from the middle of the first block (for tests that assume
    independent observations when the observations of a window are serially dependent)."""
    if block <= 1:
        return list(train)
    return [x[min(block // 2, len(x) - 1)::block] for x in train]


def same_shape(train: Sequence[np.ndarray], level=0.05, block=1) -> List[bool]:
    """Per variable: True unless the standardised values (`aligned`) are found to have another distribution in
    some window: chi-square test of homogeneity of the windows over `BINS` classes of equal pooled frequency, at
    ``level`` / d.  Location and scale are fitted in every window, so the test is conservative.
    ``block`` > 1: the standardised values are thinned to one per block before the test."""
    train = [np.asarray(x, dtype=np.float64) for x in train]
    values, _ = _standardised(train)
    values = _thinned(values, block)
    everything = np.concatenate(values, axis=0)
    d = everything.shape[1]
    keep = []
    for j in range(d):
        edges = np.quantile(everything[:, j], np.arange(1, BINS) / BINS)
        table = np.array([np.bincount(np.searchsorted(edges, x[:, j], side='right'), minlength=BINS) for x in values])
        keep.append(bool(chi2_contingency(table, correction=False)[1] >= level / d))
    return keep


def stationary(train: Sequence[np.ndarray], level=0.05, block=1) -> List[bool]:
    """Per variable: True unless a change over the windows is detected in its location (Kruskal-Wallis test on the
    values) or in its spread (the same test on the absolute deviations from the overall median); each test at
    ``level`` / (2 d).  The tests assume independent observations; for serially dependent ones (``block`` > 1)
    they are run on one observation per block, otherwise they reject for every variable and the rule falls back
    to margins within windows, which a window of dependent observations estimates badly."""
    train = _thinned([np.asarray(x, dtype=np.float64) for x in train], block)
    d = train[0].shape[1]
    keep = []
    for j in range(d):
        values = [x[:, j] for x in train]
        centre = np.median(np.concatenate(values))
        p = min(kruskal(*values).pvalue, kruskal(*[np.abs(v - centre) for v in values]).pvalue)
        keep.append(bool(p >= level / (2 * d)))
    return keep


def choose(train: Sequence[np.ndarray], others: Optional[Dict[str, Sequence[np.ndarray]]] = None,
           level=0.05, align=False, block=1) -> Tuple[Dict[str, List[np.ndarray]], list]:
    """Pooled margins for the variables whose margin is not found to change, window margins for the others.
    Returns (pseudo-observations by role, per variable whether it is pooled).
    With ``align`` a variable whose location or spread changes gets aligned margins unless its shape is found to
    change too, and the second value holds one of 'pooled', 'aligned', 'window' per variable.
    ``block``: observations per block for the tests when the observations are serially dependent."""
    keep = stationary(train, level, block)
    if not align:
        if all(keep):
            return pooled(train, others), keep
        if not any(keep):
            return window(train, others), keep
        a, b = pooled(train, others), window(train, others)
        mask = np.array(keep)[None, :]
        return {role: [np.where(mask, x, y) for x, y in zip(a[role], b[role])] for role in a}, keep
    shape = same_shape(train, level, block) if not all(keep) else keep
    kinds = ['pooled' if k else 'aligned' if s else 'window' for k, s in zip(keep, shape)]
    return _mixture(kinds, train, others), kinds


def _kind(flag) -> str:
    if isinstance(flag, (bool, np.bool_)):
        return 'pooled' if flag else 'window'
    if flag in ('pooled', 'aligned', 'window'):
        return flag
    raise ValueError(f'unknown kind of margin {flag!r}')


def _mixture(flags, train, others=None) -> Dict[str, List[np.ndarray]]:
    """Every variable with the margins its flag names."""
    kinds = [_kind(flag) for flag in flags]
    makers = dict(window=window, pooled=pooled, aligned=aligned)
    parts = {kind: makers[kind](train, others) for kind in sorted(set(kinds))}
    if len(parts) == 1:
        return next(iter(parts.values()))
    out = {}
    for role in next(iter(parts.values())):
        out[role] = []
        for t in range(len(train)):
            columns = [parts[kind][role][t][:, j] for j, kind in enumerate(kinds)]
            out[role].append(np.stack(columns, axis=1))
    return out


def ranks(windows: Sequence[np.ndarray], flags: Optional[Sequence] = None) -> List[np.ndarray]:
    """Pseudo-observations of the windows themselves, every variable with the margins its flag names: ranks over
    all windows together ('pooled' or True), the same after centring and scaling every window ('aligned'), ranks
    within every window ('window' or False; for all variables when ``flags`` is None)."""
    windows = [np.asarray(x, dtype=np.float64) for x in windows]
    if flags is None:
        return window(windows)['train']
    if len(flags) != windows[0].shape[1]:
        raise ValueError('one flag per variable is needed')
    return _mixture(flags, windows)['train']
