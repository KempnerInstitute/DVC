"""Estimate uncertainty in edge modulation by refitting the margins and lower trees on random half-samples.

Refitting propagates uncertainty from estimated transforms and allows blocks of consecutive observations for serial data. A parametric bootstrap can be miscalibrated when the fitted edge omits modulation or the lower trees approximate the data poorly.

Null hypothesis.  The modulation coefficients of the edge are the same in every window; the baseline may change.
The coefficients meant are those the working model of the edge has in the limit of many observations, whatever the
law (for a law inside the model class, its coefficients).

The test.
1. Estimates.  The whole estimation is run on the data: the vine below the edge, the calibration of the edge's two
   transforms, the fit of the edge under the null hypothesis (one coefficient per term for all windows, a baseline
   per window) and, from its scores, a one-step estimate of every coefficient in every window
   (`score_estimates`; `window_estimates`, the maximum-likelihood coefficients of single windows, are the
   alternative, less precise and with a variance that halves overstate).
2. Half-samples.  The same estimation is run on random halves of the data: in every window half of the
   observations, drawn without replacement (whole blocks of consecutive observations with ``block`` > 1), with the
   margins estimated again from the half.  For an average, the variance over halves of size n / 2 is the sampling
   variance at size n.  A half carries the estimation of the margins and of the trees below the edge, which
   redrawing the edge alone does not (17.11), and it carries the law as it is, including serial dependence that is
   shorter than a block.  Nothing is simulated from a model.
3. Statistic (`gain`).  A tested coefficient is a sequence over the windows with one variance per window, from the
   halves.  Under that quadratic model the statistic is the cost of the best constant sequence minus the cost of
   the best piecewise sequence (pieces that are constants or straight lines, a price for every jump), found by the
   dynamic programme of `trajectory`; summed over the tested terms.
4. Reference distribution.  The same statistic of every half's deviations from the mean of the halves, with the
   variances computed without that half.  Under the null hypothesis the estimates deviate from their common value
   as the halves deviate from their mean.

What it does not do.  It tests the coefficients of the working model: if the model of the edge is incomplete and
the background changes, the coefficient it converges to can change although the modulation of the law does not.
The halves have half the observations, so second-order effects (few observations per coefficient) make the
reference distribution too wide and the test conservative.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtri

from . import cvine, margins, trajectory as tr
from . import families as fam


# Defaults were selected on independent development simulations.
KEEP = 0.5                                                          # share of a window in a part (`change_test`)
ORDER, JUMP = 2, 6.0                                                # pieces of the statistic: straight lines, price of a jump


def window_estimates(windows: Sequence[np.ndarray], lower_fit, components: Sequence[cvine.Component],
                     calibrate: Optional[int] = None, columns: Optional[Sequence[int]] = None, ridge=1e-3,
                     eps=fam.EPS) -> np.ndarray:
    """Modulation coefficients (T, K) of the top edge of a C-vine, every window on its own.

    windows: list of (n_t, d) pseudo-observations, columns in vine order; the edge is the one of the last two
    columns given the others.  ``lower_fit(windows)`` returns the simplified vine below that edge.  With
    ``calibrate`` (a degree) the two transforms of the edge are calibrated against the conditioning coordinates
    (all of them, or those of ``columns``).  ``ridge`` is that of the trajectory model."""
    data = [np.asarray(w, dtype=np.float64) for w in windows]
    level = data[0].shape[1] - 2
    lower = lower_fit(data)
    if calibrate is None:
        inputs = tr.edge_inputs(lower, data, level, 0, components, eps)
    else:
        inputs, _ = tr.edge_inputs_calibrated(lower, data, level, 0, components, degree=calibrate, columns=columns, eps=eps)
    fitted = tr.fit(tr.edge_data(*inputs), tr.penalty(len(components), smooth=0.0, jump=0.0, ridge=ridge), search=False)
    return fitted.theta[:, 1:].copy()


def score_estimates(windows: Sequence[np.ndarray], lower_fit, components: Sequence[cvine.Component],
                    calibrate: Optional[int] = None, columns: Optional[Sequence[int]] = None, ridge=1e-3,
                    eps=fam.EPS) -> np.ndarray:
    """One-step estimates (T, K) of the modulation coefficients of every window, from the fit under the null
    hypothesis.  Arguments as `window_estimates`.

    The model is fitted with every modulation coefficient constant over the windows and the baseline free in every
    window.  For window t, with g the derivative of the window's negative log-likelihood with respect to the
    modulation coefficients at that fit and I their expected information given the window's baseline, the
    estimates are the common coefficients minus the solution of I x = g: one scoring step of the window's
    coefficients from the common ones, all terms together, so that a change of one term is not read as a change
    of another whose feature is correlated with it.  Under the null hypothesis its deviation from the common value
    is an average over the observations of the window, so the variance over halves is its sampling variance; the
    maximum-likelihood coefficients of single windows (`window_estimates`) have a large part that is not an
    average when a window has few observations per coefficient, and halves overstate their variance."""
    data = [np.asarray(w, dtype=np.float64) for w in windows]
    level = data[0].shape[1] - 2
    lower = lower_fit(data)
    if calibrate is None:
        inputs = tr.edge_inputs(lower, data, level, 0, components, eps)
    else:
        inputs, _ = tr.edge_inputs_calibrated(lower, data, level, 0, components, degree=calibrate, columns=columns, eps=eps)
    edge = tr.edge_data(*inputs)
    width = len(components)
    null = tr.Penalty((0.0,) + (np.inf,) * width, (0.0,) * (width + 1), (1,) * (width + 1), (False,) + (True,) * width, 0.0, ridge)
    theta = tr.fit(edge, null, search=False).theta
    gradient, information, _ = tr._derivatives(theta, edge)
    cross = information[:, 1:, 0]
    given = information[:, 1:, 1:] - cross[:, :, None] * cross[:, None, :] / np.maximum(information[:, 0, 0], 1e-12)[:, None, None]
    index = np.arange(width)
    given[:, index, index] += 1e-9 + ridge * edge.counts[:, None]
    return theta[:, 1:] - np.linalg.solve(given, gradient[:, 1:, None])[:, :, 0]


def profile_estimates(windows: Sequence[np.ndarray], lower_fit, components: Sequence[cvine.Component],
                      calibrate: Optional[int] = None, columns: Optional[Sequence[int]] = None, ridge=1e-3,
                      eps=fam.EPS) -> np.ndarray:
    """One-step estimates (T, K), every term from the fit under its own null hypothesis (16 November 2026).
    Arguments as `window_estimates`.

    `score_estimates` starts all terms from the fit in which every modulation coefficient is constant.  When
    another term of the edge changes strongly, that fit is far from the truth of the windows in the other term,
    and the error of one step, which is of second order in that distance, lands in this term: a constant term
    beside one that switches on and off is then found changing.  Here term k starts from the
    fit in which only its own coefficient is constant over the windows; the other modulation coefficients, like
    the baseline, are free in every window.  Under the null hypothesis of term k that fit is consistent for every
    coefficient of every window, whatever the other terms do.  The estimate is the k-th component of one scoring
    step of all coefficients of the window from that fit.  To first order it has the variance of the estimate of
    `score_estimates` (both solve for all terms of the window together), and its deviation under the null
    hypothesis is again an average over the observations of the window.  With one term it is `score_estimates`."""
    width = len(components)
    if width <= 1:
        return score_estimates(windows, lower_fit, components, calibrate, columns, ridge, eps)
    data = [np.asarray(w, dtype=np.float64) for w in windows]
    level = data[0].shape[1] - 2
    lower = lower_fit(data)
    if calibrate is None:
        inputs = tr.edge_inputs(lower, data, level, 0, components, eps)
    else:
        inputs, _ = tr.edge_inputs_calibrated(lower, data, level, 0, components, degree=calibrate, columns=columns, eps=eps)
    edge = tr.edge_data(*inputs)
    out = np.empty((edge.steps, width))
    index = np.arange(1, width + 1)
    for k in range(width):
        null = tr.Penalty((0.0,) + tuple(np.inf if j == k else 0.0 for j in range(width)), (0.0,) * (width + 1), (1,) * (width + 1),
                          (False,) + tuple(j == k for j in range(width)), 0.0, ridge)
        theta = tr.fit(edge, null, search=False).theta
        gradient, information, _ = tr._derivatives(theta, edge)
        information[:, index, index] += 1e-9 + ridge * edge.counts[:, None]
        information[:, 0, 0] = np.maximum(information[:, 0, 0], 1e-12)
        out[:, k] = theta[:, 1 + k] - np.linalg.solve(information, gradient[:, :, None])[:, 1 + k, 0]
    return out


ESTIMATORS = dict(window=window_estimates, score=score_estimates, profile=profile_estimates)


def autocorrelation(windows: Sequence[np.ndarray]) -> float:
    """The largest absolute lag-one autocorrelation of a variable's normal scores (pooled over the windows, every
    window centred).  windows: pseudo-observations, rows in the order of observation."""
    above, below = 0.0, 0.0
    for w in windows:
        x = ndtri(np.clip(np.asarray(w, dtype=np.float64), 1e-6, 1.0 - 1e-6))
        x = x - x.mean(axis=0)
        above = above + (x[1:] * x[:-1]).sum(axis=0)
        below = below + (x * x).sum(axis=0)
    return float(np.max(np.abs(above / np.maximum(below, 1e-300))))


def autocorrelations(windows: Sequence[np.ndarray]) -> np.ndarray:
    """The lag-one autocorrelation of every variable's normal scores (d,), computed as in `autocorrelation`."""
    above, below = 0.0, 0.0
    for w in windows:
        x = ndtri(np.clip(np.asarray(w, dtype=np.float64), 1e-6, 1.0 - 1e-6))
        x = x - x.mean(axis=0)
        above = above + (x[1:] * x[:-1]).sum(axis=0)
        below = below + (x * x).sum(axis=0)
    return np.asarray(above / np.maximum(below, 1e-300), dtype=np.float64)


def block_of(r: float, smallest: int, factor=3.0, floor=0.1, least_blocks=8) -> int:
    """The block length of `block_length` for an autocorrelation r and the size of the smallest window."""
    if r < floor:
        return 1
    longest = max(1, int(smallest) // least_blocks)
    return int(min(longest, np.ceil(factor * (1.0 + r) / (1.0 - min(r, 0.999)))))


def pair_serial(ri: float, rj: float, smallest: int, cap=0.98) -> Tuple[float, int]:
    """Serial dependence of the statistics of a pair of variables with lag-one autocorrelations ri and rj.
    Returns (inflation, block).  A product of two autoregressive series has the autocorrelation q = ri rj, and
    the variance of its average is inflated by (1 + q) / (1 - q): the factor for the selection costs of the pair's
    copula.  The block is that of `block_length` at the geometric mean sqrt(q).  With ri = rj both are what
    `vinemodel.fit` uses for every pair, from the largest autocorrelation of a variable; here every pair has its
    own, so that one slow variable does not decide for pairs of fast ones."""
    q = max(float(ri), 0.0) * max(float(rj), 0.0)
    return (1.0 + min(q, cap)) / (1.0 - min(q, cap)), block_of(round(float(np.sqrt(q)), 12), smallest)


def block_length(windows: Sequence[np.ndarray], factor=3.0, floor=0.1, least_blocks=8) -> int:
    """A block length for serially dependent observations, from the windows themselves (pseudo-observations, rows
    in the order of observation).  r: `autocorrelation`.  One (single observations) if r is below ``floor``;
    otherwise ``factor`` times the correlation time (1 + r) / (1 - r) of an autoregression with that coefficient,
    rounded up, and at most a ``least_blocks``-th of the smallest window.  The statistics of an edge are products
    of several such series and forget faster than one of them, so this errs towards long blocks."""
    r = autocorrelation(windows)
    if r < floor:
        return 1
    longest = max(1, min(len(w) for w in windows) // least_blocks)
    return int(min(longest, np.ceil(factor * (1.0 + r) / (1.0 - min(r, 0.999)))))


def half(size: int, keep: float, block: int, rng) -> np.ndarray:
    """Sorted row indices of a random part of a window of ``size`` observations: a share ``keep`` of them, or of
    its blocks of ``block`` consecutive observations."""
    if block <= 1:
        return np.sort(rng.permutation(size)[:max(1, int(keep * size))])
    starts = np.arange(0, size, block)
    chosen = np.sort(rng.permutation(starts.size)[:max(1, int(keep * starts.size))])
    return np.concatenate([np.arange(starts[i], min(starts[i] + block, size)) for i in chosen])


def scale(size: int, kept: int) -> float:
    """Factor that turns the deviation of a part of ``kept`` of ``size`` observations, drawn without replacement,
    into a deviation with the sampling variance of all of them: for an average, the variance over such parts is
    s^2 (size - kept) / (kept size), and the target is s^2 / size.  One for a half."""
    return float(np.sqrt(kept / max(size - kept, 1)))


def _segment(cost: np.ndarray, jump: float) -> float:
    """Smallest total cost of a division of the windows into consecutive pieces; every piece after the first costs
    ``jump``.  cost[l, r]: cost of the piece of windows l..r."""
    steps = cost.shape[0]
    best = np.empty(steps)
    for r in range(steps):
        value = cost[0, r]
        if r:
            value = min(value, float(np.min(best[:r] + jump + cost[1:r + 1, r])))
        best[r] = value
    return float(best[-1])


def gain(estimates: np.ndarray, precision: np.ndarray, order=ORDER, jump=JUMP):
    """For one coefficient sequence (T,) with precisions (T,): (gain, q).  Under the quadratic model
    sum_t precision_t (x_t - estimate_t)^2 / 2, gain is the cost of the best constant minus the cost of the best
    piecewise sequence with pieces of ``order`` 1 (constants) or 2 (straight lines) and the price ``jump`` for every
    piece after the first; q is twice the cost of the best constant, the weighted sum of squares about the
    weighted mean (Cochran's statistic)."""
    constant = tr._piece_costs(precision, estimates, np.inf, 1)
    cost = constant if order == 1 else tr._piece_costs(precision, estimates, np.inf, order)
    whole = float(constant[0, -1])
    value = whole - _segment(cost, float(jump))
    return (value if value > 1e-9 * (1.0 + whole) else 0.0), 2.0 * whole              # rounding is not a gain


def within(estimates: np.ndarray, precision: np.ndarray, tolerance: float, relative=False) -> np.ndarray:
    """The estimates (T,) pulled towards their precision-weighted mean by ``tolerance``: a deviation of at most
    that size becomes zero, a larger one is reduced by it.  ``relative``: the tolerance is that share of the
    absolute value of the weighted mean."""
    if tolerance <= 0.0:
        return estimates
    centre = float(np.sum(precision * estimates) / np.sum(precision))
    band = tolerance * abs(centre) if relative else tolerance
    deviation = estimates - centre
    return centre + np.sign(deviation) * np.maximum(np.abs(deviation) - band, 0.0)


def reference(estimates: np.ndarray, deviations: np.ndarray, terms: Sequence[int], order=ORDER, jump=JUMP, tolerance=0.0,
              relative=False) -> dict:
    """Statistic and p-values from the estimates (T, K) and the scaled deviations of the parts from their mean
    (B, T, K).  Returns dict(statistic, p, q, p_q, terms={k: dict(gain, p, q, p_q)}, null (B,), variance (T, K)).
    ``statistic`` and ``p`` use the piecewise gain summed over ``terms``; ``q`` and ``p_q`` Cochran's statistic.
    tolerance: shrink the observed estimates towards their estimated precision-weighted mean, using either an
    absolute tolerance or a fraction of the mean's absolute value. The half-sample deviations are unshrunk.
    A band about some other common center can give a different decision. Error control after choosing edges
    and orientations is evaluated empirically."""
    draws, steps, width = deviations.shape
    if draws < 3:
        raise ValueError('at least three parts are needed')
    squares = (deviations * deviations).sum(axis=0)
    variance = np.maximum(squares / (draws - 1.0), 1e-12)
    observed = np.array([gain(within(estimates[:, k], 1.0 / variance[:, k], tolerance, relative), 1.0 / variance[:, k], order, jump)
                         for k in range(width)])                                                                # (K, 2)
    null = np.empty((draws, width, 2))
    inflate = draws / (draws - 1.0)
    for b in range(draws):
        left = np.maximum((squares - inflate * deviations[b] ** 2) / (draws - 2.0), 1e-12)                     # variance without this part
        point = np.sqrt(inflate) * deviations[b]                                                                # its deviation from the mean of the others, variance restored
        null[b] = [gain(point[:, k], 1.0 / left[:, k], order, jump) for k in range(width)]
    chosen = [int(k) for k in terms]
    level = lambda value, values: float((1 + np.sum(values >= value - 1e-12)) / (draws + 1)) if value > 0.0 else 1.0
    out = dict(variance=variance, null=null[:, chosen, 0].sum(axis=1))
    out['statistic'], out['q'] = float(observed[chosen, 0].sum()), float(observed[chosen, 1].sum())
    out['p'], out['p_q'] = level(out['statistic'], out['null']), level(out['q'], null[:, chosen, 1].sum(axis=1))
    out['terms'] = {k: dict(gain=float(observed[k, 0]), p=level(observed[k, 0], null[:, k, 0]), q=float(observed[k, 1]),
                            p_q=level(observed[k, 1], null[:, k, 1])) for k in chosen}
    return out


def change_test(windows: Sequence[np.ndarray], lower_fit, components: Sequence[cvine.Component], replicates=99, rng=None,
                calibrate: Optional[int] = None, columns: Optional[Sequence[int]] = None, pooled: Optional[Sequence] = None,
                terms: Optional[Sequence[int]] = None, keep=KEEP, block=1, order=ORDER, jump=JUMP, ridge=1e-3, eps=fam.EPS,
                raw: Optional[Sequence[np.ndarray]] = None, estimate='score', tolerance=0.0, relative=False) -> dict:
    """Test that the modulation coefficients of the top edge of a C-vine are the same in every window, with the
    reference distribution from random halves of the data (see the module).

    windows, lower_fit, components, calibrate, columns, pooled: as `trajectory.change_test`.  The halves get their
    pseudo-observations the way ``pooled`` says the data got theirs (`margins.ranks`; by default ranks within every
    window).
    raw: the values the windows were ranked from, same shapes; the halves are ranked from these.  Needed only when
    a flag of ``pooled`` is 'aligned' (the alignment uses the location and scale of the values); otherwise the
    windows themselves serve, since ranks do not change under an increasing transform.
    terms: indices (from 0) of the modulation components whose constancy is tested (default: all).  Every tested
    term also gets its own p-value, from the same halves.
    keep: share of the observations (or blocks) of a window in a part; 0.5 needs no scaling.
    block: observations per block, for serially dependent observations; 1 = single observations.
    order, jump: the pieces of the statistic (`gain`).
    estimate: 'score' (`score_estimates`, the default), 'profile' (`profile_estimates`: every term from the fit in
    which only that term is constant; the same as 'score' for one term) or 'window' (`window_estimates`).
    tolerance, relative: test that no coefficient differs from a common value by more than this, or by more than
    this share of the common value (`reference`); 0 = exact constancy.
    Returns dict(statistic, p, q, p_q, terms, null, estimates (T, K), variance (T, K), deviations (B, T, K): the
    scaled deviations of the halves from their mean, sizes).  `reference` recomputes the statistic from estimates
    and deviations, for other pieces or terms."""
    rng = np.random.default_rng() if rng is None else rng
    data = [np.asarray(w, dtype=np.float64) for w in windows]
    width = len(components)
    chosen = list(range(width)) if terms is None else sorted({int(k) for k in terms})
    if not chosen or not set(chosen) <= set(range(width)):
        raise ValueError('terms must index the modulation components')
    if not 0.0 < keep < 1.0:
        raise ValueError('keep must lie between 0 and 1')
    if pooled is not None and len(pooled) != data[0].shape[1]:
        raise ValueError('pooled needs one flag per column')
    source = data if raw is None else [np.asarray(w, dtype=np.float64) for w in raw]
    if [w.shape for w in source] != [w.shape for w in data]:
        raise ValueError('raw must have the shapes of the windows')
    if estimate not in ESTIMATORS:
        raise ValueError("estimate must be 'window', 'score' or 'profile'")
    estimator = ESTIMATORS[estimate]
    estimate = lambda sample: estimator(sample, lower_fit, components, calibrate, columns, ridge, eps)
    estimates = estimate(data)
    units = lambda size: int(np.ceil(size / block)) if block > 1 else size
    factor = np.array([scale(units(len(w)), max(1, int(keep * units(len(w))))) for w in data])
    parts = np.empty((replicates, len(data), width))
    for b in range(replicates):
        rows = [half(len(w), keep, block, rng) for w in data]
        parts[b] = estimate(margins.ranks([w[r] for w, r in zip(source, rows)], pooled))
    deviations = (parts - parts.mean(axis=0)) * factor[None, :, None]
    out = reference(estimates, deviations, chosen, order, jump, tolerance, relative)
    out.update(estimates=estimates, deviations=deviations, sizes=[len(w) for w in data])
    return out
