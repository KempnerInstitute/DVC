"""A change of the three-way term shared by many binary trios, by exact conditional inference.

For three binary variables the three-way log-linear term is the modulation of one pair's log odds ratio by the
third variable, and it is the same number whichever variable is called the third. Once all two-way margins are
fixed, a table has one free number: the tables with given two-way margins are `table + t * PARITY`.

Take two sets of events, A and B (two epochs, or two blocks of time). Given the two-way margins of a trio in A and
in B and the total count of the pattern (1,1,1) in both, the split of that total between A and B has a law that
contains no lower-order term of either set and not the trio's own three-way term. It depends only on the
difference `delta` of the three-way term between B and A:

    P(u)  proportional to  exp(delta * u) / prod_x (B(x) + u s(x))! (A(x) - u s(x))!

where u is how many more (1,1,1) events B has than observed and s = PARITY. This is the three-factor analogue of
the conditional analysis of a common odds ratio in sparse 2 x 2 tables. Each trio keeps its own single-cell and
pair terms in each set and its own three-way term, and sparse tables bias nothing: under delta = 0 the conditional
mean of u is exact. Strata (levels of the activity of the other cells, for example) are further pairs of tables
with their own margins and their own three-way term.

A delta shared by many trios is estimated from the sum of the conditional scores (the conditional maximum
likelihood estimate). Trios that share cells are not independent. The variance of the pooled score is the sum of
the conditional variances times a design effect, estimated from the per-event efficient scores summed over trios
(for independent cells these scores are uncorrelated between trios, whatever cells they share, and the design
effect is one).

The same holds one order lower. For two binary variables the tables with given single-cell margins are
`table + t * (1, -1, -1, 1)`, the free number is the count of (1,1), and the split of that count between A and B
depends only on the change of the pair's log odds ratio: the classical conditional analysis of 2 x 2 tables.
Every function here takes sets of two or of three cells (`order` is read from the shape) and uses the parity
vector of that order: +1 for the pattern with every cell active and alternating with each cell switched off.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, ndtr

from .binary_tables import PARITY

ODD, EVEN = PARITY > 0, PARITY < 0
ONES = np.array([bin(k).count("1") for k in range(8)])
PARITY_PAIR = np.array([1.0, -1.0, -1.0, 1.0])


def parity_of(patterns):
    """The parity vector of the tables with this many patterns (4 for pairs, 8 for trios)."""
    if patterns == 8:
        return PARITY
    if patterns == 4:
        return PARITY_PAIR
    raise ValueError("Expected tables of pairs (4 patterns) or of trios (8 patterns)")


def pattern_codes(x, sets):
    """Pattern index of every event for every set of cells: x0 + 2 x1 (+ 4 x2)."""
    code = x[:, sets[:, 0]] + 2 * x[:, sets[:, 1]]
    if sets.shape[1] == 3:
        code = code + 4 * x[:, sets[:, 2]]
    return code


def trio_tables(x, trios, size=None, edges=None, batch=4000):
    """Counts of the patterns of each set of cells, by stratum: an array (sets, strata, 4 or 8).

    x: events by cells, 0 or 1. trios: (T, 3) cell indices, or (T, 2) for pairs. The pattern index is
    x0 + 2 x1 + 4 x2, as in `binary_tables`. size: per event, the number of active cells of the reference
    population (which must contain the cells of every set); the stratum of an event for a set is where `size`
    minus the set's own active cells falls among `edges` (so the activity of the other cells never contains the
    set's own outcome). Without `size` there is one stratum.
    """
    x = np.asarray(x, dtype=np.int64)
    trios = np.asarray(trios, dtype=np.int64)
    if trios.ndim != 2 or trios.shape[1] not in (2, 3):
        raise ValueError("Expected sets of two or three cells")
    patterns = 2 ** trios.shape[1]
    strata = 1 if size is None else len(edges) + 1
    out = np.zeros((len(trios), strata, patterns), dtype=np.int64)
    for start in range(0, len(trios), batch):
        part = trios[start:start + batch]
        code = pattern_codes(x, part)
        if size is not None:
            level = np.searchsorted(edges, np.asarray(size)[:, None] - ONES[code], side="right")
            code = code + patterns * level
        code += (patterns * strata) * np.arange(len(part))[None, :]
        out[start:start + len(part)] = np.bincount(code.ravel(), minlength=patterns * strata * len(part)).reshape(-1, strata, patterns)
    return out


class SplitFamily:
    """The conditional law of the split of the (1,1,1) count between two tables, for many pairs of tables.

    a, b: arrays (rows, 8) of counts, or (rows, 4) for pairs. For each row, u runs over the integers for which
    `b + u s` and `a - u s` are tables; the observed tables are u = 0.
    """

    def __init__(self, a, b, batch_cells=4_000_000):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.ndim != 2 or a.shape != b.shape or np.any(a < 0) or np.any(b < 0):
            raise ValueError("Expected two aligned arrays of non-negative counts, one row per table")
        parity = parity_of(a.shape[1])
        up, down = parity > 0, parity < 0
        self.rows = len(a)
        self.lower = np.maximum(-b[:, up].min(1), -a[:, down].min(1))
        self.upper = np.minimum(b[:, down].min(1), a[:, up].min(1))
        width = (self.upper - self.lower + 1).astype(np.int64)
        self.informative = width > 1
        self.batches = []
        order = np.flatnonzero(self.informative)
        order = order[np.argsort(width[order], kind="stable")]
        start = 0
        while start < len(order):
            stop, widest = start, int(width[order[start]])
            while stop < len(order) and (stop - start + 1) * int(width[order[stop]]) <= max(batch_cells, widest):
                stop += 1
            stop = max(stop, start + 1)
            rows = order[start:stop]
            u = self.lower[rows, None] + np.arange(int(width[rows].max()))[None, :]
            valid = u <= self.upper[rows, None]
            use = np.where(valid, u, 0.0)
            logw = np.zeros(u.shape)
            for k in range(len(parity)):
                logw -= gammaln(b[rows, k, None] + use * parity[k] + 1.0)
                logw -= gammaln(a[rows, k, None] - use * parity[k] + 1.0)
            logw = np.where(valid, logw, -np.inf)
            logw -= logw.max(1, keepdims=True)
            self.batches.append((rows, u, logw))
            start = stop

    def moments(self, delta=0.0):
        """Conditional mean and variance of u in every row, at a shift delta of the three-way term."""
        mean, variance = np.zeros(self.rows), np.zeros(self.rows)
        for rows, u, logw in self.batches:
            tilted = logw + delta * u
            tilted -= tilted.max(1, keepdims=True)
            p = np.exp(tilted)
            p /= p.sum(1, keepdims=True)
            m = (p * u).sum(1)
            mean[rows] = m
            variance[rows] = (p * (u - m[:, None]) ** 2).sum(1)
        return mean, variance

    def log_probability(self, delta=0.0):
        """Log conditional probability of the observed split (u = 0) in every row, at a shift delta.

        delta may be one number or one per row. Rows without information have probability one.
        """
        out = np.zeros(self.rows)
        delta = np.broadcast_to(np.asarray(delta, dtype=float), (self.rows,))
        for rows, u, logw in self.batches:
            tilted = logw + delta[rows, None] * u
            peak = tilted.max(1, keepdims=True)
            total = peak[:, 0] + np.log(np.exp(tilted - peak).sum(1))
            at_zero = np.take_along_axis(logw, (-self.lower[rows]).astype(np.int64)[:, None], axis=1)[:, 0]
            out[rows] = at_zero - total
        return out

    def pmf(self, row):
        """Support and probabilities of u for one row at delta = 0 (for exact tests of single rows)."""
        for rows, u, logw in self.batches:
            hit = np.flatnonzero(rows == row)
            if len(hit):
                p = np.exp(logw[hit[0]])
                keep = p > 0
                return u[hit[0]][keep], p[keep] / p[keep].sum()
        return np.zeros(1), np.ones(1)


def shared_shift(family, groups=None, tolerance=1e-9):
    """Conditional maximum likelihood estimate of the shift shared by all rows.

    Returns the estimate, the conditional information at the estimate, and the score and information at zero.
    The estimate is not finite when every informative row sits at an end of its range.
    """
    mean0, variance0 = family.moments(0.0)
    score0, information0 = -mean0.sum(), variance0.sum()
    if information0 <= 0:
        return np.nan, 0.0, 0.0, 0.0
    delta = 0.0
    for _ in range(100):
        mean, variance = family.moments(delta)
        score, information = -mean.sum(), variance.sum()
        if information <= 1e-12:
            return (np.inf if score > 0 else -np.inf), 0.0, score0, information0
        step = float(np.clip(score / information, -1.0, 1.0))
        delta += step
        if abs(step) < tolerance:
            break
        if abs(delta) > 30:
            return (np.inf if delta > 0 else -np.inf), 0.0, score0, information0
    mean, variance = family.moments(delta)
    return delta, variance.sum(), score0, information0


def fitted_split(a, b, steps=70):
    """The two tables with the observed margins, the observed total of (1,1,1) and one common three-way term.

    This is the unconditional fit of "no change". Returns the real number u (the fitted b is `b + u s`, the fitted
    a is `a - u s`) and the two fitted tables. Rows without information are returned unchanged with u = 0.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    parity = parity_of(a.shape[1])
    up, down = parity > 0, parity < 0
    lower = np.maximum(-b[:, up].min(1), -a[:, down].min(1))
    upper = np.minimum(b[:, down].min(1), a[:, up].min(1))
    open_range = upper > lower
    left, right = lower.copy(), upper.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(steps):
            middle = 0.5 * (left + right)
            gap = np.log(b + middle[:, None] * parity) @ parity - np.log(a - middle[:, None] * parity) @ parity
            below = gap < 0
            left, right = np.where(below, middle, left), np.where(below, right, middle)
    u = np.where(open_range, 0.5 * (left + right), 0.0)
    return u, a - u[:, None] * parity, b + u[:, None] * parity


def efficient_weights(a, b):
    """Per-pattern efficient scores of the shift for events of each table, and its asymptotic information.

    For an event of B with pattern x the score is `w_b[row, x]`, for an event of A, `w_a[row, x]`. They sum over
    events to the asymptotic score of the shift (minus the fitted u), and their variance is the information
    `1 / (sum 1/fitted_a + sum 1/fitted_b)`.
    """
    u, fa, fb = fitted_split(a, b)
    parity = parity_of(fa.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_a, inverse_b = 1.0 / fa, 1.0 / fb
        total_a, total_b = inverse_a.sum(1), inverse_b.sum(1)
        ok = np.isfinite(total_a) & np.isfinite(total_b) & (total_a > 0) & (total_b > 0)
        information = np.where(ok, 1.0 / (total_a + total_b), 0.0)
        # weight of B's events: I_a / (I_a + I_b) = total_b / (total_a + total_b); efficient score s / (p H)
        w_b = np.where(ok[:, None], parity * inverse_b / (total_a + total_b)[:, None], 0.0)
        w_a = np.where(ok[:, None], -parity * inverse_a / (total_a + total_b)[:, None], 0.0)
    return w_a, w_b, information, np.where(ok, -u, 0.0)


@dataclass
class PooledChange:
    delta: float            # conditional maximum likelihood estimate of the shared shift (B minus A), log odds
    se: float               # its standard error, with the design effect
    z: float                # score test of no shift, with the design effect
    p: float                # two-sided
    score: float            # pooled conditional score at zero
    information: float      # sum of the conditional variances at zero
    design_effect: float    # variance of the summed per-event scores over the sum of the informations
    trios: int
    informative_rows: int
    delta_asymptotic: float  # one-step estimate from the unconditional fit (biased when tables are sparse)
    z_asymptotic: float
    row_score: np.ndarray    # per trio and stratum: conditional score at zero
    row_information: np.ndarray
    influence_a: np.ndarray = None  # per event (or per cluster of events) of A: its share of the estimate's error
    influence_b: np.ndarray = None  # the squares of both sum to se ** 2
    other: dict = None              # name -> (se, influence_a, influence_b) for further ways of clustering the events


@dataclass
class Part:
    """The tables of one class in one population (a hemisphere of a session) and its per-event scores."""
    a: np.ndarray              # (sets * strata, patterns) counts in A
    b: np.ndarray
    g_a: np.ndarray            # per event of A: the efficient scores of the shift, summed over the sets
    g_b: np.ndarray
    information_asymptotic: float
    score_asymptotic: float
    sets: int
    strata: int


def prepare(x_a, x_b, trios, size_a=None, size_b=None, edges=None, batch=4000):
    """Tables and per-event scores of the given pairs or trios for the events x_a and x_b (see `pooled_change`)."""
    trios = np.asarray(trios, dtype=np.int64)
    x_a, x_b = np.asarray(x_a, dtype=np.int64), np.asarray(x_b, dtype=np.int64)
    a = trio_tables(x_a, trios, size_a, edges, batch)
    b = trio_tables(x_b, trios, size_b, edges, batch)
    strata, patterns = a.shape[1], a.shape[2]
    w_a, w_b, information_asym, score_asym = efficient_weights(a.reshape(-1, patterns), b.reshape(-1, patterns))
    w_a, w_b = w_a.reshape(len(trios), strata, patterns), w_b.reshape(len(trios), strata, patterns)

    def event_sums(x, size, w):
        total = np.zeros(len(x))
        for start in range(0, len(trios), batch):
            part = trios[start:start + batch]
            code = pattern_codes(x, part)
            if size is None:
                level = np.zeros_like(code)
            else:
                level = np.searchsorted(edges, np.asarray(size)[:, None] - ONES[code], side="right")
            total += w[start:start + len(part)][np.arange(len(part))[None, :], level, code].sum(1)
        return total

    return Part(a=a.reshape(-1, patterns), b=b.reshape(-1, patterns), g_a=event_sums(x_a, size_a, w_a),
                g_b=event_sums(x_b, size_b, w_b), information_asymptotic=float(information_asym.sum()),
                score_asymptotic=float(score_asym.sum()), sets=len(trios), strata=strata)


def combine(parts, clusters_a=None, clusters_b=None, also=None):
    """One shift shared by the classes of several populations (hemispheres, sessions).

    parts: a list of `Part`. clusters_a, clusters_b: optionally, for each part, an integer label for each of its
    events in A and in B. Events with the same label (ripples of the two hemispheres that coincide; ripples of
    one bout of sleep) are summed before the variance is taken, so that their dependence is in the standard
    error; labels must not be shared between events that are independent. Without labels every event stands
    alone. With labels, the variance carries the factor G / (G - 1) for the G clusters of each set of events.
    also: further clusterings, a dict name -> (clusters_a, clusters_b) (or (None, None) for every event alone);
    their standard errors and influence values are returned in `other`.
    """
    parts = [part for part in parts if part.sets > 0]
    if not parts:
        return PooledChange(np.nan, np.nan, np.nan, np.nan, 0.0, 0.0, np.nan, 0, 0, np.nan, np.nan,
                            np.zeros((0, 1)), np.zeros((0, 1)))
    family = SplitFamily(np.vstack([part.a for part in parts]), np.vstack([part.b for part in parts]))
    delta, information_at, score0, information0 = shared_shift(family)
    mean0, variance0 = family.moments(0.0)
    total_asym = sum(part.information_asymptotic for part in parts)
    score_asym = sum(part.score_asymptotic for part in parts)

    def summed(values, labels):
        centred = [v - v.mean() if len(v) else v for v in values]
        if labels is None:
            return np.concatenate(centred) if centred else np.zeros(0)
        labels = [np.asarray(l, dtype=np.int64) for l in labels]
        size = max((int(l.max()) + 1 for l in labels if len(l)), default=0)
        return np.bincount(np.concatenate(labels), weights=np.concatenate(centred), minlength=size)

    def clustered(labels_a, labels_b):
        """Design effect, standard error and influence values for one way of clustering the events."""
        sums = [summed([part.g_a for part in parts], labels_a), summed([part.g_b for part in parts], labels_b)]
        factors = []
        for values, labels in zip(sums, (labels_a, labels_b)):
            groups = int(np.count_nonzero(values)) if labels is not None else 0
            factors.append(np.sqrt(groups / (groups - 1)) if groups > 1 else 1.0)
        sums = [values * factor for values, factor in zip(sums, factors)]
        meat_ = (sums[0] ** 2).sum() + (sums[1] ** 2).sum()
        design_ = meat_ / total_asym if total_asym > 0 else np.nan
        if not np.isfinite(design_) or design_ <= 0:
            design_ = np.nan
        se_ = np.sqrt(design_ / information_at) if information_at > 0 and np.isfinite(design_) else np.nan
        # shares of the error: summed scores over the asymptotic information, rescaled to the conditional one
        scale_ = (se_ / np.sqrt(meat_)) if np.isfinite(se_) and meat_ > 0 else np.nan
        return design_, se_, sums[0] * scale_, sums[1] * scale_

    design, se, influence_a, influence_b = clustered(clusters_a, clusters_b)
    z = score0 / np.sqrt(information0 * design) if information0 > 0 and np.isfinite(design) else np.nan
    other = None
    if also:
        other = {}
        for name, (labels_a, labels_b) in also.items():
            _, se_other, ia, ib = clustered(labels_a, labels_b)
            other[name] = (float(se_other), ia, ib)
    return PooledChange(
        delta=float(delta), se=float(se), z=float(z), p=float(2 * ndtr(-abs(z))) if np.isfinite(z) else np.nan,
        score=float(score0), information=float(information0), design_effect=float(design),
        trios=int(sum(part.sets for part in parts)), informative_rows=int(family.informative.sum()),
        delta_asymptotic=float(score_asym / total_asym) if total_asym > 0 else np.nan,
        z_asymptotic=float(score_asym / np.sqrt(total_asym * design)) if total_asym > 0 and np.isfinite(design) else np.nan,
        row_score=-mean0, row_information=variance0, influence_a=influence_a, influence_b=influence_b, other=other)


def pooled_change(x_a, x_b, trios, size_a=None, size_b=None, edges=None, batch=4000):
    """Shift of the three-way term from the events x_a to the events x_b, shared by the given trios.

    x_a, x_b: events by cells (0 or 1). trios: (T, 3) cell indices; with (T, 2) the shift is that of the pairs'
    log odds ratio. size_a, size_b, edges: optional strata, see `trio_tables`. Returns a `PooledChange`.
    """
    part = prepare(x_a, x_b, trios, size_a, size_b, edges, batch)
    fit = combine([part])
    fit.row_score = fit.row_score.reshape(part.sets, part.strata)
    fit.row_information = fit.row_information.reshape(part.sets, part.strata)
    return fit


def contrast(first, second, clustering=None):
    """Difference of two pooled shifts estimated on the same events (two classes), with its standard error.

    The two results must come from `combine` with the same events or the same cluster labels, so that their
    influence values are aligned; the covariance of the two estimates is then in the standard error.
    clustering: the name of one of the further clusterings given to `combine` as `also`.
    """
    if clustering is None:
        se_1, a_1, b_1 = first.se, first.influence_a, first.influence_b
        se_2, a_2, b_2 = second.se, second.influence_a, second.influence_b
    else:
        (se_1, a_1, b_1), (se_2, a_2, b_2) = first.other[clustering], second.other[clustering]
    if a_1 is None or a_2 is None:
        raise ValueError("The two estimates are not on the same events")
    size_a, size_b = max(len(a_1), len(a_2)), max(len(b_1), len(b_2))
    a_1, a_2 = np.pad(a_1, (0, size_a - len(a_1))), np.pad(a_2, (0, size_a - len(a_2)))
    b_1, b_2 = np.pad(b_1, (0, size_b - len(b_1))), np.pad(b_2, (0, size_b - len(b_2)))
    difference = first.delta - second.delta
    variance = ((a_1 - a_2) ** 2).sum() + ((b_1 - b_2) ** 2).sum()
    se = float(np.sqrt(variance))
    z = difference / se if se > 0 and np.isfinite(difference) else np.nan
    return dict(delta=float(difference), se=se, z=float(z), p=float(2 * ndtr(-abs(z))) if np.isfinite(z) else np.nan,
                correlation=float(1 - variance / (se_1 ** 2 + se_2 ** 2)) if se_1 > 0 and se_2 > 0 else np.nan)


def single_trio_pvalues(x_a, x_b, trios, size_a=None, size_b=None, edges=None):
    """Exact two-sided conditional p-value of no change for each trio by itself (strata convolved)."""
    trios = np.asarray(trios, dtype=np.int64)
    a = trio_tables(np.asarray(x_a, dtype=np.int64), trios, size_a, edges)
    b = trio_tables(np.asarray(x_b, dtype=np.int64), trios, size_b, edges)
    strata, patterns = a.shape[1], a.shape[2]
    family = SplitFamily(a.reshape(-1, patterns), b.reshape(-1, patterns))
    out = np.ones(len(trios))
    for t in range(len(trios)):
        support, p = np.zeros(1), np.ones(1)
        for s in range(strata):
            u, q = family.pmf(t * strata + s)
            first = support[0] + u[0]
            p = np.convolve(p, q)
            support = first + np.arange(len(p))
        observed = p[np.argmin(np.abs(support))]
        out[t] = min(1.0, float(p[p <= observed * (1 + 1e-9)].sum()))
    return out


# ---------------------------------------------------------------------------------------------------------------
# A step at an unknown boundary between blocks of time
# ---------------------------------------------------------------------------------------------------------------

def _block_weights(table, parity):
    """Support (its first value) and normalised weights of t for one table: table + t * parity is a table."""
    lower, upper = -int(table[parity > 0].min()), int(table[parity < 0].min())
    u = np.arange(lower, upper + 1)
    logw = np.zeros(len(u))
    for k in range(len(parity)):
        logw -= gammaln(table[k] + u * parity[k] + 1.0)
    return lower, np.exp(logw - logw.max())


def scan_rows(tables):
    """Exact conditional scores of a step of the shared term at every boundary between consecutive blocks.

    tables: (rows, blocks, patterns) counts; a row is one pair or trio in one stratum, a block a stretch of time
    with its own lower-order terms. Under one term common to all blocks, and given every block's margins and the
    row's total of the top pattern, the allocation of that total to the blocks has a known law. For the boundary
    after block c the statistic is the part allocated to the later blocks. Returns, summed over rows, the score
    and the conditional variance for each of the blocks - 1 boundaries, and per row and block the conditional
    variance of the block's own share (from which the information of each block is recovered).
    """
    tables = np.asarray(tables, dtype=float)
    rows, blocks, patterns = tables.shape
    parity = parity_of(patterns)
    score, variance, single = np.zeros(blocks - 1), np.zeros(blocks - 1), np.zeros((rows, blocks))

    def moments(values, weights):
        total = weights.sum()
        if total <= 0:
            return 0.0, 0.0
        mean = (weights * values).sum() / total
        return mean, (weights * (values - mean) ** 2).sum() / total

    for r in range(rows):
        parts = [_block_weights(tables[r, b], parity) for b in range(blocks)]
        if all(len(w) == 1 for _, w in parts):
            continue
        prefix = [(0, np.ones(1))]
        for low, w in parts:
            offset, a = prefix[-1]
            a = np.convolve(a, w)
            prefix.append((offset + low, a / a.max()))
        suffix = [(0, np.ones(1))]
        for low, w in reversed(parts):
            offset, a = suffix[-1]
            a = np.convolve(a, w)
            suffix.append((offset + low, a / a.max()))
        suffix = suffix[::-1]                      # suffix[c]: blocks c .. end
        for c in range(1, blocks):
            (before_start, before), (after_start, after) = prefix[c], suffix[c]
            later = after_start + np.arange(len(after))          # what the later blocks hold beyond the observed
            index = -later - before_start
            ok = (index >= 0) & (index < len(before))
            mean, var = moments(later[ok], after[ok] * before[index[ok]])
            score[c - 1] -= mean
            variance[c - 1] += var
        for b in range(blocks):
            (start_1, a_1), (start_2, a_2) = prefix[b], suffix[b + 1]
            others = np.convolve(a_1, a_2)
            low, w = parts[b]
            own = low + np.arange(len(w))
            index = -own - (start_1 + start_2)
            ok = (index >= 0) & (index < len(others))
            single[r, b] = moments(own[ok], w[ok] * others[index[ok]])[1]
    return score, variance, single


def block_information(single):
    """Information of each block, summed over rows, from the conditional variances of the blocks' own shares.

    For independent blocks with informations v_b the share of block b, given the total, has variance
    v_b (1 - v_b / sum v): solved row by row for v.
    """
    single = np.asarray(single, dtype=float)
    v = single.copy()
    for _ in range(200):
        total = v.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            new = np.where(total > 0, single / np.clip(1 - v / np.where(total > 0, total, 1), 1e-6, None), 0.0)
        if np.abs(new - v).max() < 1e-10:
            v = new
            break
        v = new
    return v.sum(0)


def bridge_p(z_max, fractions, draws=200_000, seed=0):
    """P(max over the boundaries of |Z| >= z_max) when Z is a standardised Brownian bridge at the fractions."""
    fractions = np.asarray(fractions, dtype=float)
    keep = (fractions > 1e-6) & (fractions < 1 - 1e-6)
    if not keep.any() or not np.isfinite(z_max):
        return np.nan
    t = fractions[keep]
    rng = np.random.default_rng(seed)
    steps = np.diff(np.concatenate(([0.0], t, [1.0])))
    walk = np.cumsum(rng.normal(size=(draws, len(steps))) * np.sqrt(steps), axis=1)
    bridge = walk[:, :-1] - t * walk[:, [-1]]
    return float((np.abs(bridge / np.sqrt(t * (1 - t))).max(1) >= z_max).mean())


def scan_change(tables, design_effect=1.0):
    """Where, between consecutive blocks of time, does the term shared by a class step?

    tables: (rows, blocks, patterns), rows of any number of populations stacked. design_effect: the factor for
    sets that share cells (from `pooled_change` on the same class). Returns, for each boundary, the score, its
    conditional variance, z and the one-step size of the step; the boundary with the largest |z|; and the
    probability of a largest |z| of that size under no step anywhere (Brownian bridge at the blocks' shares of
    the information).
    """
    score, variance, single = scan_rows(tables)
    information = block_information(single)
    fractions = np.cumsum(information)[:-1] / information.sum() if information.sum() > 0 else np.full(len(score), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(variance > 0, score / np.sqrt(variance * design_effect), np.nan)
        step = np.where(variance > 0, score / variance, np.nan)
    best = int(np.nanargmax(np.abs(z))) if np.isfinite(z).any() else -1
    return dict(score=score.tolist(), variance=variance.tolist(), z=z.tolist(), step=step.tolist(),
                fractions=fractions.tolist(), block_information=information.tolist(), boundary=best + 1,
                z_max=float(np.abs(z[best])) if best >= 0 else float("nan"),
                p_max=bridge_p(float(np.abs(z[best])), fractions) if best >= 0 else float("nan"))
