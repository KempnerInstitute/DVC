"""Score statistics for modulation of a pair's dependence by other variables.

Setting.  ``a`` and ``b`` are normal scores of two variables after their dependence on a conditioning set D has been
removed (conditional probability transforms from a fitted vine), and ``g`` is a centred function of the conditioning
variables.  If the conditional copula of the pair does not change with the conditioning variables (the simplifying
assumption), then (a, b) is independent of D and E[(a b - E[a b]) g] = 0.  A conditioning variable that modulates the
dependence of the pair makes it nonzero.  The statistic is the studentised sample mean of (a b - mean(a b)) g; it is
the score statistic for the coefficient of g in a Gaussian pair copula whose Fisher-z correlation is linear in g,
evaluated at correlation zero, and it is a valid moment test whatever the conditional copula family.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2, norm, rankdata

SHAPES = ('linear', 'quadratic')


def normal_scores(w, rerank=True, eps=1e-10):
    """Normal scores of probability transforms (columns).  With ``rerank`` the transforms are replaced by their
    ranks within the sample first, so every column has exactly the same bounded set of scores."""
    w = np.asarray(w, dtype=np.float64)
    if rerank:
        w = rankdata(w, axis=0) / (w.shape[0] + 1.0)
    return ndtri(np.clip(w, eps, 1.0 - eps))


def shape_functions(z, shapes: Sequence[str] = SHAPES):
    """Centred test functions of one modulating normal score z (n,): columns in the order of ``shapes``."""
    out = []
    for shape in shapes:
        if shape == 'linear':
            g = z
        elif shape == 'quadratic':
            g = (z * z - 1.0) / np.sqrt(2.0)
        else:
            raise ValueError(shape)
        out.append(g - g.mean())
    return np.stack(out, axis=1)


def hermite_basis(z, degree=4):
    """Orthonormal Hermite polynomials of degrees 1..degree of every column of z: (n, columns * degree)."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    parts = [z, (z ** 2 - 1.0) / np.sqrt(2.0), (z ** 3 - 3.0 * z) / np.sqrt(6.0), (z ** 4 - 6.0 * z ** 2 + 3.0) / np.sqrt(24.0)]
    return np.concatenate(parts[:degree], axis=1)


def conditional_standardize(a, z, degree=4):
    """Remove from every column of a the part of its mean and variance that still depends on the conditioning
    scores z (least squares on Hermite polynomials of each conditioning coordinate, additive).

    A correct pair layer leaves a independent of z and this changes nothing beyond noise.  A wrong one leaves a
    conditional mean or variance, which the product statistic would read as modulation.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    design = np.column_stack((np.ones(a.shape[0]), hermite_basis(z, degree)))
    q, _ = np.linalg.qr(design)
    residual = a - q @ (q.T @ a)
    square = residual * residual
    mean_square = square.mean(axis=0)
    variance = np.clip(q @ (q.T @ square), 0.25 * mean_square, 4.0 * mean_square)
    out = residual / np.sqrt(variance)
    out = out - q @ (q.T @ out)                      # rescaling can reintroduce a small conditional mean
    return out / out.std(axis=0)


def moment_scan(a, b, g, hac_lags=0):
    """Studentised modulation statistics for every pair (column of a, column of b) and every column of g.

    a: (n, p), b: (n, q), g: (n, m) centred.  Returns z with shape (m, p, q).
    ``hac_lags`` > 0 replaces the variance by a Bartlett-kernel long-run variance (serially dependent samples),
    floored at a quarter of the plain variance.
    """
    a, b, g = (np.asarray(x, dtype=np.float64) for x in (a, b, g))
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if g.ndim == 1:
        g = g[:, None]
    n = a.shape[0]
    mean_ab = a.T @ b / n
    a2, b2 = a * a, b * b
    out = np.empty((g.shape[1], a.shape[1], b.shape[1]))
    for index in range(g.shape[1]):
        gi = g[:, index] - g[:, index].mean()
        g2 = gi * gi
        mean = a.T @ (b * gi[:, None]) / n
        second = a2.T @ (b2 * g2[:, None]) / n - 2.0 * mean_ab * (a.T @ (b * g2[:, None]) / n) + mean_ab ** 2 * g2.mean()
        variance = second - mean * mean
        plain = variance.copy()
        for lag in range(1, int(hac_lags) + 1):
            w = gi[:-lag] * gi[lag:]
            a0, a1, b0, b1 = a[:-lag], a[lag:], b[:-lag], b[lag:]
            cross = ((a0 * a1).T @ ((b0 * b1) * w[:, None])
                     - mean_ab * (a0.T @ (b0 * w[:, None]) + a1.T @ (b1 * w[:, None])) + mean_ab ** 2 * w.sum()) / n
            variance = variance + 2.0 * (1.0 - lag / (hac_lags + 1.0)) * (cross - mean * mean * (n - lag) / n)
        if hac_lags:
            variance = np.maximum(variance, 0.25 * plain)          # a long-run variance estimate can be negative
        variance = np.maximum(variance, 1e-12) * n / (n - 1.0)
        out[index] = np.sqrt(n) * mean / np.sqrt(variance)
    return out


def moment_slopes(a, b, g):
    """Effect size on the correlation scale: least-squares slope of the pair product on each modulator,
    mean((a b - mean(a b)) g) / mean(g^2), shape (m, p, q).  For the linear shape this is the change of the pair's
    normal-score correlation per standard deviation of the modulator; unlike the studentised statistic it does not
    depend on the number of observations or on the variance of the product."""
    a, b, g = (np.asarray(x, dtype=np.float64) for x in (a, b, g))
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if g.ndim == 1:
        g = g[:, None]
    n = a.shape[0]
    out = np.empty((g.shape[1], a.shape[1], b.shape[1]))
    for index in range(g.shape[1]):
        gi = g[:, index] - g[:, index].mean()
        out[index] = (a.T @ (b * gi[:, None]) / n) / (gi @ gi / n)
    return out


def third_order_scan(scores, z, shapes: Sequence[str] = SHAPES, hac_lags=0, standardize=0, slopes=False):
    """Exhaustive scan of all (i, j | k).  ``standardize`` > 0 applies ``conditional_standardize`` of that degree.
    With ``slopes`` the unstudentised effect sizes of ``moment_slopes`` are returned instead of z statistics.

    scores: (n, d, d) with scores[:, i, k] the normal score of P(U_i <= u_i | U_k = u_k) (diagonal ignored).
    z: (n, d) normal scores of the variables.  Returns stat (m, d, d, d) indexed [shape, i, j, k], symmetric in
    (i, j), nan where two indices coincide.
    """
    n, d, _ = scores.shape
    stat = np.full((len(shapes), d, d, d), np.nan)
    for k in range(d):
        others = np.array([v for v in range(d) if v != k])
        a = scores[:, others, k]
        if standardize:
            a = conditional_standardize(a, z[:, k], standardize)
        g = shape_functions(z[:, k], shapes)
        zz = moment_slopes(a, a, g) if slopes else moment_scan(a, a, g, hac_lags=hac_lags)
        for s in range(len(shapes)):
            block = zz[s].copy()
            np.fill_diagonal(block, np.nan)
            stat[s][np.ix_(others, others, [k])] = block[:, :, None]
    return stat


def triple_table(stat):
    """Per unordered triple: chi-square statistics (sum of squares over shapes) of its three orientations.

    Returns triples (m, 3) with i < j < k and q (m, 3): columns are the orientations conditioning on k, j, i.
    """
    d = stat.shape[1]
    triples = np.array(list(combinations(range(d), 3)))
    i, j, k = triples.T
    sq = np.nansum(stat ** 2, axis=0)
    q = np.stack((sq[i, j, k], sq[i, k, j], sq[j, k, i]), axis=1)
    return triples, q


def orientation_free_pvalues(stat):
    """Intersection-union p-value per triple: the largest of the three orientation p-values (chi-square with as many
    degrees of freedom as shapes).  A triple is declared modulated only if every orientation rejects."""
    triples, q = triple_table(stat)
    p = chi2.sf(q, df=stat.shape[0])
    return triples, p.max(axis=1), p


def two_sided_p(z):
    return 2.0 * norm.sf(np.abs(z))


def bonferroni_threshold(n_tests, level=0.05):
    return float(norm.isf(level / (2.0 * n_tests)))


def holm(p, level=0.05):
    """Holm step-down rejections for a flat array of p-values."""
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, index in enumerate(order):
        if p[index] <= level / (m - rank):
            reject[index] = True
        else:
            break
    return reject


def benjamini_hochberg(p, level=0.1):
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    passed = p[order] <= level * (np.arange(1, m + 1) / m)
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        reject[order[:np.max(np.nonzero(passed)[0]) + 1]] = True
    return reject


def interval_scan(z):
    """Largest standardised partial sum over contiguous windows.  z: (T, ...) statistics that are independent
    N(0, 1) across windows under the null.  Returns (max |sum|/sqrt(length), start, stop) arrays over the trailing
    dimensions; stop is exclusive."""
    z = np.asarray(z, dtype=np.float64)
    steps = z.shape[0]
    cumulative = np.concatenate((np.zeros((1,) + z.shape[1:]), np.cumsum(z, axis=0)))
    best = np.zeros(z.shape[1:])
    start = np.zeros(z.shape[1:], dtype=int)
    stop = np.ones(z.shape[1:], dtype=int)
    for s in range(steps):
        for e in range(s + 1, steps + 1):
            value = np.abs(cumulative[e] - cumulative[s]) / np.sqrt(e - s)
            better = value > best
            best = np.where(better, value, best)
            start = np.where(better, s, start)
            stop = np.where(better, e, stop)
    return best, start, stop


def significant_intervals(z, threshold):
    """Disjoint runs of windows whose standardised partial sum reaches the threshold, strongest first, then the
    same search on the windows left and right of each run.  z: (T,).  Returns sorted (start, stop, value)."""
    z = np.asarray(z, dtype=np.float64)
    found, segments = [], [(0, z.size)]
    while segments:
        low, high = segments.pop()
        if high <= low:
            continue
        value, start, stop = interval_scan(z[low:high])
        if float(value) >= threshold:
            start, stop = low + int(start), low + int(stop)
            found.append((start, stop, float(value)))
            segments.extend(((low, start), (stop, high)))
    return sorted(found)


def interval_threshold(steps, n_tests, level=0.05):
    """Bonferroni threshold for interval-scan maxima: all contiguous intervals of all tests, two-sided."""
    return float(norm.isf(level / (2.0 * n_tests * steps * (steps + 1) / 2.0)))


def interval_scan_null(steps, draws=20000, seed=0):
    """Monte Carlo null distribution of the interval-scan maximum for independent N(0, 1) window statistics."""
    rng = np.random.default_rng(seed)
    return np.sort(interval_scan(rng.normal(size=(steps, draws)))[0])


def permutation_max(a, b, g, permutations=200, rng=None, circular=False, min_shift=1):
    """Null distribution of max |z| over all pairs and shapes when the modulators are permuted against the pairs
    (or circularly shifted, which keeps serial dependence).  Returns an array (permutations,)."""
    rng = np.random.default_rng() if rng is None else rng
    n = a.shape[0]
    out = np.empty(permutations)
    for index in range(permutations):
        if circular:
            shifted = np.roll(g, int(rng.integers(min_shift, n - min_shift)), axis=0)
        else:
            shifted = g[rng.permutation(n)]
        out[index] = np.nanmax(np.abs(moment_scan(a, b, shifted)))
    return out
