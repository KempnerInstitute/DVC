"""Statistics of the second detector version (P2, 10 November 2026).

P1's statistic is the studentised mean of (a b - mean(a b)) g with a, b the normal scores of the pair's conditional
transforms.  The run on fresh laws (E10), the pair-layer ablation (E12) and the comparison with the published test
(E11) showed three weaknesses, each addressed here.

1. Pair-layer misfit.  If the fitted pair copulas leave a conditional mean or variance of a transform given the
   conditioning variable, the product reads it as modulation.  `smooth_standardize` removes both: a local quadratic
   fit over the nearest neighbours in the rank of the conditioning score for the mean, a local mean of squared
   residuals for the variance.  Each residual is divided by the square root of its own variance factor, which
   reproduces a constant coupling of the pair exactly at every point.  (Without that, and in the polynomial version
   `scan.conditional_standardize`, the leverage grows with the conditioning value and biases the statistic when
   the pair has a constant coupling.  A running mean, and a local linear fit, left a bias at the two ends of the
   conditioning range on a U-shape, where the quadratic test function is largest; both were tried first.)
   Smoothing makes neighbouring residuals dependent and the statistic slightly over-dispersed at small samples;
   `null_scale` gives its null standard deviation by simulation through the same steps.
2. Serial dependence.  For two independent stationary series p and g, n Var(mean(p g)) is the sum over lags of
   gamma_p(l) gamma_g(l) (Bartlett).  With ``serial`` the variance of the statistic is the plain one plus twice the
   tapered sum over lags >= 1 of the products of the sample autocovariances of the pair series and of the modulator.
   Each factor is estimated from its own series, so many lags can be used without the noise of a long-run variance
   of the product series.  It assumes what circular shifts assume: no dependence between the two series across
   lags under the null.
3. Power when the pair has a constant coupling.  With ``score='correlation'`` the series is
   a b - (r / 2) (a^2 + b^2), r the pair's sample correlation: the influence function of a correlation coefficient.
   Its variance is (1 - r^2)^2 for a Gaussian pair against 1 + r^2 for the product, and it is insensitive to first
   order to a common error in the two conditional variances.  After the variances have been standardised locally
   the product behaves like a local correlation, so its plain variance estimate is too large by the same factor
   (statistic too conservative); the correlation series has the right variance in both cases.  Under the null (the
   pair independent of the conditioning variables) any function of (a, b) gives a valid test.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import ndtri


DEGREE = 2


def default_half(n):
    """Neighbours on each side in the rank of the conditioning score."""
    return int(min(max(10, np.ceil(2.0 * np.sqrt(n))), (n - 1) // 2))


@lru_cache(maxsize=16)
def _smoother(n, half, degree=DEGREE):
    """Local polynomial smoother over rank neighbours, in the order of the conditioning score.

    The regressor is the normal score of the rank, so the sorted regressor values depend on n only and the smoother
    is computed once per sample size.  Tricube weights in rank distance; windows are cut at the two ends.
    Returns (neighbour index (n, m), mean weights (n, m), variance factor (n,), variance weights (n, m)): the
    fitted mean at point i is the sum over m of weights[i, m] * x[index[i, m]], and for independent noise of unit
    variance the residual has variance factor[i] = 1 - 2 s_ii + sum_j s_ij^2.
    """
    score = ndtri((np.arange(n) + 1.0) / (n + 1.0))
    offsets = np.arange(-half, half + 1)
    index = np.arange(n)[:, None] + offsets[None]
    valid = (index >= 0) & (index < n)
    index = np.clip(index, 0, n - 1)
    kernel = np.where(valid, ((1.0 - (np.abs(offsets) / (half + 1.0)) ** 3) ** 3)[None], 0.0)
    step = score[index] - score[:, None]
    design = np.stack([step ** power for power in range(degree + 1)], axis=-1)        # (n, m, degree + 1)
    weighted = design * kernel[:, :, None]
    gram = np.einsum('nmp,nmq->npq', weighted, design)
    weights = np.linalg.solve(gram, weighted.transpose(0, 2, 1))[:, 0, :]             # fitted value at the point
    factor = 1.0 - 2.0 * weights[:, half] + (weights * weights).sum(axis=1)
    return index, weights, np.maximum(factor, 0.05), kernel / kernel.sum(axis=1, keepdims=True)


def _apply(weights, index, x):
    out = np.zeros_like(x)
    for m in range(weights.shape[1]):
        out += weights[:, m, None] * x[index[:, m]]
    return out


def smooth_standardize(a, z, half=None):
    """Remove from every column of a its mean and variance as functions of the conditioning score z.

    a: (n, p) or (n,), z: (n,); only the ranks of z are used.  Mean: local quadratic fit over the ``half`` nearest
    neighbours on each side in the rank of z (default 2 sqrt(n)), tricube weights; every residual is divided by
    the square root of its variance factor, which reproduces a constant variance and a constant coupling of two
    columns exactly at every point.  Variance: local mean of the squared residuals with the same weights, clipped
    to [1/4, 4] times the overall value.  Returns (n, p) with overall mean 0 and variance 1 per column.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    n = a.shape[0]
    half = default_half(n) if half is None else int(half)
    index, weights, factor, flat = _smoother(n, half)
    order = np.argsort(np.asarray(z, dtype=np.float64), kind='stable')
    x = a[order]
    residual = (x - _apply(weights, index, x)) / np.sqrt(factor)[:, None]
    square = residual * residual
    overall = square.mean(axis=0)
    out = residual / np.sqrt(np.clip(_apply(flat, index, square), 0.25 * overall, 4.0 * overall))
    out = (out - out.mean(axis=0)) / out.std(axis=0)
    result = np.empty_like(out)
    result[order] = out
    return result


def standardize_on(a, conditioning, half=None):
    """`smooth_standardize` on each column of ``conditioning`` in turn (additive in the conditioning scores)."""
    conditioning = np.asarray(conditioning, dtype=np.float64)
    if conditioning.ndim == 1:
        conditioning = conditioning[:, None]
    for column in range(conditioning.shape[1]):
        a = smooth_standardize(a, conditioning[:, column], half)
    return a


def autocovariance(x, max_lag):
    """Sample autocovariances (divisor n) of series centred along axis 0, lags 0..max_lag: (max_lag + 1, ...)."""
    n = x.shape[0]
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(x, n=size, axis=0)
    return np.fft.irfft(f.real ** 2 + f.imag ** 2, n=size, axis=0)[:max_lag + 1] / n


def default_max_lag(n):
    return int(min(n - 1, max(20, n // 5)))


def modulation_scan(a, b, g, score='correlation', serial=False, max_lag=None):
    """Studentised modulation statistics for every pair (column of a, column of b) and every column of g.

    a: (n, p), b: (n, q), g: (n, m).  Returns z of shape (m, p, q).  ``score``: 'product' or 'correlation' (see the
    module header).  ``serial``: Bartlett variance for serially dependent observations.
    """
    a, b, g = (np.asarray(x, dtype=np.float64) for x in (a, b, g))
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if g.ndim == 1:
        g = g[:, None]
    n = a.shape[0]
    core = a[:, :, None] * b[:, None, :]
    if score == 'correlation':
        r = np.clip(core.mean(axis=0), -0.95, 0.95)
        core = core - 0.5 * r[None] * ((a * a)[:, :, None] + (b * b)[:, None, :])
    elif score != 'product':
        raise ValueError(score)
    core -= core.mean(axis=0)
    square = core * core
    if serial:
        lags = default_max_lag(n) if max_lag is None else int(max_lag)
        weights = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
        pair_cov = autocovariance(core, lags)[1:]                       # (lags, p, q)
    out = np.empty((g.shape[1], a.shape[1], b.shape[1]))
    for index in range(g.shape[1]):
        gi = g[:, index] - g[:, index].mean()
        mean = np.einsum('npq,n->pq', core, gi) / n
        variance = np.einsum('npq,n->pq', square, gi * gi) / n - mean * mean
        if serial:
            extra = 2.0 * np.tensordot(weights * autocovariance(gi, lags)[1:], pair_cov, axes=(0, 0))
            variance = np.maximum(variance + extra, 0.25 * variance)
        variance = np.maximum(variance, 1e-12) * n / (n - 1.0)
        out[index] = np.sqrt(n) * mean / np.sqrt(variance)
    return out


@lru_cache(maxsize=64)
def null_scale(n, conditioning, kind, draws=2000, seed=20261110):
    """Null standard deviation of the statistic after `smooth_standardize` on ``conditioning`` scores.

    Independent standard normal a, b and conditioning scores, ``draws`` pairs, the same steps as the detector:
    kind 'linear' or 'quadratic' uses that function of the last conditioning score, 'product' the product of the
    last two.  Fixed seed, so the value depends only on the arguments.
    """
    rng = np.random.default_rng([seed, n, conditioning, len(kind)])
    scores = ndtri((np.arange(n) + 1.0) / (n + 1.0))
    z = np.stack([scores[rng.permutation(n)] for _ in range(conditioning)], axis=1)
    a = rng.normal(size=(n, 2 * draws))
    a = standardize_on(a, z)
    a, b = a[:, :draws], a[:, draws:]
    if kind == 'product':
        g = z[:, -1] * z[:, -2]
    elif kind == 'quadratic':
        g = (z[:, -1] ** 2 - 1.0) / np.sqrt(2.0)
    else:
        g = z[:, -1]
    g = g - g.mean()
    r = np.clip((a * b).mean(axis=0), -0.95, 0.95)
    core = a * b - 0.5 * r * (a * a + b * b)
    core -= core.mean(axis=0)
    mean = core.T @ g / n
    variance = (core * core).T @ (g * g) / n - mean * mean
    return float(np.std(np.sqrt(n) * mean / np.sqrt(variance * n / (n - 1.0))))
