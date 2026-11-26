"""The pooled log-linear comparator: a change of the triplet coefficient in each cell's full conditional law.

In a log-linear model of the whole population, the log odds that cell i is active given all the other cells is

    intercept_i + sum_j coupling_ij x_j + sum over the class trios {i, j, k} of K_ijk x_j x_k.

This comparator fits, for every cell of the class and separately for the two sets of events A and B, a logistic
regression of the cell on all the other cells of the population and on T_i(x), the number of the cell's class
trios whose two other cells are both active. The coefficient of T_i is the triplet term of cell i's class trios
(taken as common within the cell), given the whole population rather than given the trio alone. The shared change
is the precision-weighted mean over cells of (coefficient in B) minus (coefficient in A); its variance comes from
the per-event influence values summed over cells, so the dependence between the regressions of different cells is
in it. Every cell has its own intercept, couplings and triplet coefficient in each set, which is the same freedom
that the pooled trio estimator of `binary_pooled` gives each trio.

It differs from that estimator in what is held fixed: here all the other cells (a pseudo-likelihood of a pairwise
model of the population plus class triplet terms), there the trio's own pair margins within strata of the
population's activity. It is not exact in sparse data: logistic regression coefficients carry a bias of the order
of (number of regressors) / (number of events).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, ndtr


def logistic(z, y, precision, iterations=60, tolerance=1e-9):
    """Penalised logistic regression by Newton steps. Returns coefficients, fitted probabilities and the Hessian."""
    beta = np.zeros(z.shape[1])
    hessian = None
    for _ in range(iterations):
        p = expit(z @ beta)
        w = p * (1 - p)
        hessian = z.T @ (z * w[:, None]) + np.diag(precision)
        step = np.linalg.solve(hessian, z.T @ (y - p) - precision * beta)
        largest = np.abs(step).max()
        if largest > 5:
            step *= 5 / largest
        beta += step
        if largest < tolerance:
            break
    p = expit(z @ beta)
    return beta, p, z.T @ (z * (p * (1 - p))[:, None]) + np.diag(precision)


def class_regressors(x, trios, cells):
    """T_i(x) for each listed cell: the number of its class trios whose two other cells are both active."""
    x = np.asarray(x, dtype=float)
    trios = np.asarray(trios, dtype=np.int64).reshape(-1, 3)
    out = np.zeros((len(x), len(cells)))
    for c, cell in enumerate(cells):
        for position in range(3):
            rows = trios[trios[:, position] == cell]
            if len(rows):
                others = np.delete(rows, position, axis=1)
                out[:, c] += (x[:, others[:, 0]] * x[:, others[:, 1]]).sum(1)
    return out


@dataclass
class LoglinearChange:
    delta: float          # shared change of the triplet coefficient (B minus A), log odds
    se: float
    z: float
    p: float
    cells: int            # cells that entered
    design_effect: float  # variance from the summed influence values over the variance if cells were independent
    cell_delta: np.ndarray
    cell_variance: np.ndarray


def pooled_loglinear_change(x_a, x_b, trios, coupling_sd=2.0, minimum=5):
    """Shared change of the class triplet coefficient from the events x_a to the events x_b.

    x_a, x_b: events by cells (the whole population, 0 or 1). trios: (T, 3) cell indices. A cell enters when its
    regressor T_i is positive in at least `minimum` events with the cell active and `minimum` with it silent, in
    both sets.
    """
    x_a, x_b = np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)
    trios = np.asarray(trios, dtype=np.int64).reshape(-1, 3)
    cells = np.unique(trios)
    t_a, t_b = class_regressors(x_a, trios, cells), class_regressors(x_b, trios, cells)
    columns = x_a.shape[1]
    deltas, variances, influence_a, influence_b = [], [], [], []
    for c, cell in enumerate(cells):
        others = np.delete(np.arange(columns), cell)
        fits = []
        for x, t in ((x_a, t_a[:, c]), (x_b, t_b[:, c])):
            y = x[:, cell]
            if min(((t > 0) & (y > 0)).sum(), ((t > 0) & (y == 0)).sum()) < minimum:
                fits = None
                break
            z = np.column_stack((np.ones(len(x)), x[:, others], t))
            precision = np.r_[0.0, np.full(len(others), 1 / coupling_sd ** 2), 1e-8]
            beta, p, hessian = logistic(z, y, precision)
            direction = np.linalg.solve(hessian, np.eye(z.shape[1])[:, -1])
            fits.append((beta[-1], direction[-1], (z @ direction) * (y - p)))
        if fits is None:
            continue
        deltas.append(fits[1][0] - fits[0][0])
        variances.append(fits[0][1] + fits[1][1])
        influence_a.append(-fits[0][2])
        influence_b.append(fits[1][2])
    if not deltas:
        return LoglinearChange(np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.zeros(0), np.zeros(0))
    deltas, variances = np.array(deltas), np.array(variances)
    weight = (1 / variances) / (1 / variances).sum()
    g_a, g_b = weight @ np.array(influence_a), weight @ np.array(influence_b)
    variance = ((g_a - g_a.mean()) ** 2).sum() + ((g_b - g_b.mean()) ** 2).sum()
    delta = float(weight @ deltas)
    se = float(np.sqrt(variance))
    z = delta / se if se > 0 else np.nan
    return LoglinearChange(delta, se, float(z), float(2 * ndtr(-abs(z))) if np.isfinite(z) else np.nan, len(deltas),
                           float(variance * (1 / variances).sum()), deltas, variances)
