"""Joint coefficient trajectories for one edge of a dynamic vine.

Model.  In window t the edge is a Gaussian pair copula of the normal scores (x, y) of its two conditional transforms
with

    atanh rho = theta_0(t) + sum_k theta_k(t) f_k(zeta)        (clipped at +-ETA_MAX, as the modulated states of `cvine`)

theta_0 is the baseline trajectory: dependence that changes with time.  The f_k are functions of the conditioning
coordinates (`cvine.component_features`) and theta_k(t), k >= 1, are the modulation trajectories: how the conditioning
variables change the dependence, and how that changes with time.  For every value of the conditioning variables this
is a copula, so a vine with such edges remains a normalised density.

Estimator.  All windows are fitted together; no window has its own fit.  Every trajectory consists of pieces: inside
a piece it is smooth, between pieces it may jump, and a modulation trajectory may be switched off (exactly zero) in a
piece.

    objective = - log-likelihood of all windows
                + sum_p smooth_p * (sum of the squared differences of order `order_p` of theta_p inside its pieces)
                + sum_p jump_p   * (number of pieces of theta_p that are switched on; the first piece of the baseline is free)
                + include        * (number of modulation trajectories that are switched on anywhere)
                + ridge          * sum_t n_t * (squared modulation coefficients of window t)

smooth = 0 leaves the windows of a piece untied; smooth = inf makes a piece a polynomial of degree order - 1 in time
(a constant for order 1, a straight line for order 2).  Special cases: every window on its own (smooth 0, no
search), one smooth trajectory per coefficient (no search; a penalised spline with a knot at every window),
piecewise-constant switching (order 1, smooth inf).

Search (`fit`).  For a given set of pieces the coefficients are found by Fisher scoring with step halving on the
likelihood of all windows.  This is a local method: the objective is not convex (the observed information of an
observation near the origin is negative) and it is not smooth where the predictor is clipped, so the iteration can
stop at a stationary point that is not the minimum.  `stationarity` reports the gradient and the smallest observed
curvature at a fit, and `fit(..., polish=True)` leaves a point of negative curvature.  The pieces are found one
trajectory at a time: with the other trajectories held fixed and the likelihood replaced by its local quadratic
model, the best pieces of one trajectory are found by dynamic programming over all segmentations; this step, and
only this step, is exact.  The proposal is kept only if the objective of the refitted coefficients decreases.  The
result is a local minimum over pieces (`tests/test_trajectory.py` compares it with exhaustive search on small
problems).

Features.  A modulating function may be a function of a conditioning coordinate (shapes `linear`, `quadratic`,
`product` of `cvine.component_features`: the coordinates are independent, and a coordinate of a root is the part of
it that the earlier roots do not explain) or of the root's own normal score (shapes `own_linear`, `own_quadratic`,
`own_product`).  The own scores do not depend on the order of the conditioning set; they are dependent on each
other.  `cvine` states cannot carry own-score terms; `simulate_null` and the change test can.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import block_diag
from scipy.special import ndtr, ndtri

from . import cvine, margins, paths
from . import families as fam
from .cvine import ETA_MAX


# ----------------------------------------------------------------------------------------------------------------
# Data of one edge.
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Edge:
    """Observations of one edge in T windows.  x, y: normal scores (T, n); f: features (T, n, K) with the window
    means ``means`` (T, K) subtracted; w: (T, n), 1 for an observation and 0 for padding (windows of unequal size)."""
    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    w: np.ndarray
    means: np.ndarray

    def __post_init__(self):
        self.design = np.concatenate((self.w[:, :, None], self.f), axis=2)          # (T, n, 1 + K)
        self.xy = self.x * self.y
        self.sq = self.x * self.x + self.y * self.y
        self.counts = self.w.sum(axis=1)

    @property
    def steps(self):
        return self.x.shape[0]

    @property
    def width(self):
        return self.design.shape[2]


def edge_data(u: Sequence[np.ndarray], v: Sequence[np.ndarray], features: Sequence[np.ndarray], centre=True,
              eps=fam.EPS) -> Edge:
    """u[t], v[t]: the two conditional transforms of the edge in window t (in (0, 1)); features[t]: (n_t, K).

    centre=True subtracts each window's own feature means, so that the baseline is the mean Fisher-z correlation of
    the window; an array (T, K) subtracts those means instead (use the training means for held-out data);
    False subtracts nothing."""
    steps, size, k = len(u), max(len(a) for a in u), np.asarray(features[0]).shape[1]
    x, y, w = np.zeros((steps, size)), np.zeros((steps, size)), np.zeros((steps, size))
    f = np.zeros((steps, size, k))
    for t in range(steps):
        m = len(u[t])
        x[t, :m], y[t, :m] = ndtri(fam.clip01(np.asarray(u[t], float), eps)), ndtri(fam.clip01(np.asarray(v[t], float), eps))
        f[t, :m], w[t, :m] = features[t], 1.0
    counts = w.sum(axis=1)
    if centre is True:
        means = (f * w[:, :, None]).sum(axis=1) / counts[:, None]
    elif centre is False or centre is None:
        means = np.zeros((steps, k))
    else:
        means = np.asarray(centre, dtype=np.float64)
    return Edge(x, y, (f - means[:, None, :]) * w[:, :, None], w, means)


OWN = 'own_'


def uses_own_scores(components: Sequence[cvine.Component]) -> bool:
    return any(shape.startswith(OWN) for shape, _ in components)


def own_scores(model: cvine.DynamicCVine, data, level: int, eps=fam.EPS) -> np.ndarray:
    """Normal scores of the first ``level`` roots themselves (n, level): of their margins, not of their conditional
    transforms.  data: (n, d) pseudo-observations in the original column order."""
    return ndtri(fam.clip01(np.asarray(data, dtype=np.float64)[:, list(model.order[:level])], eps))


def component_features(components: Sequence[cvine.Component], zeta: np.ndarray, own: Optional[np.ndarray] = None) -> np.ndarray:
    """Features (n, K) of the components.  Shapes `linear`, `quadratic`, `product` act on the conditioning
    coordinates ``zeta``; the same shapes with the prefix `own_` act on the roots' own normal scores ``own``."""
    if not uses_own_scores(components):
        return cvine.component_features(components, zeta)
    if own is None:
        raise ValueError('components in the variables\' own scores need those scores')
    columns = [cvine.component_features([(shape[len(OWN):], members)], own) if shape.startswith(OWN)
               else cvine.component_features([(shape, members)], zeta) for shape, members in components]
    return np.concatenate(columns, axis=1) if columns else np.zeros((zeta.shape[0], 0))


def edge_inputs(model: cvine.DynamicCVine, windows: Sequence[np.ndarray], level: int, edge: int,
                components: Sequence[cvine.Component], eps=fam.EPS):
    """The inputs of an edge under a fitted vine: per window the transform of the edge's later variable, of the root
    of its level, and the features of ``components``.  ``model`` must have at least ``level`` fitted levels."""
    u, v, features = [], [], []
    own = uses_own_scores(components)
    for t, data in enumerate(windows):
        out = cvine.forward(model, t, data, eps, keep=True)
        pits = out['pits'][level]
        u.append(pits[:, level + 1 + edge].copy())
        v.append(pits[:, level].copy())
        features.append(component_features(components, out['zeta'][:, :level], own_scores(model, data, level, eps) if own else None))
    return u, v, features


# ----------------------------------------------------------------------------------------------------------------
# Calibration of the transforms of an edge against the conditioning coordinates.
#
# A vine computes the conditional transform of a variable given the conditioning set through the trees below the
# edge.  If these are misspecified the computed transform is not uniform given the conditioning variables, and the
# edge above inherits the error.  The computed transform is still increasing in the variable, so
# the true transform is its conditional distribution function given the conditioning coordinates.  Here that
# function is estimated by a Gaussian regression on the normal-score scale: per window, the mean of the normal
# score of the computed transform is a polynomial of the coordinates and the residual has one standard deviation.
# For an adequate vine the fitted mean is zero and the standard deviation one, up to estimation error.
# ----------------------------------------------------------------------------------------------------------------
def edge_transforms(model: cvine.DynamicCVine, windows: Sequence[np.ndarray], level: int, edge: int, eps=fam.EPS):
    """Per window: the two transforms of an edge under a fitted vine (as `edge_inputs`) and the conditioning
    coordinates zeta[:, :level]."""
    u, v, zeta = [], [], []
    for t, data in enumerate(windows):
        out = cvine.forward(model, t, data, eps, keep=True)
        pits = out['pits'][level]
        u.append(pits[:, level + 1 + edge].copy())
        v.append(pits[:, level].copy())
        zeta.append(out['zeta'][:, :level].copy())
    return u, v, zeta


def calibration_basis(zeta, degree=2, products=True) -> np.ndarray:
    """Columns: 1, the coordinates and, for degree 2, (zeta_k^2 - 1) / sqrt 2 and (with ``products``) zeta_a zeta_b
    for a < b."""
    zeta = np.asarray(zeta, dtype=np.float64)
    k = zeta.shape[1]
    columns = [np.ones(zeta.shape[0])] + [zeta[:, j] for j in range(k)]
    if degree >= 2:
        columns += [(zeta[:, j] ** 2 - 1.0) / np.sqrt(2.0) for j in range(k)]
        if products:
            columns += [zeta[:, a] * zeta[:, b] for a in range(k) for b in range(a + 1, k)]
    return np.column_stack(columns)


@dataclass
class Calibration:
    """Fitted calibration of the two transforms of an edge.  beta: (T, 2, P) coefficients of the mean of the normal
    score (first the edge's later variable, then the root of its level); scale: (T, 2) residual standard deviations;
    columns: the conditioning coordinates used (None = all)."""
    degree: int
    products: bool
    beta: np.ndarray
    scale: np.ndarray
    columns: Optional[Tuple[int, ...]] = None

    def _basis(self, zeta):
        zeta = np.asarray(zeta, dtype=np.float64)
        return calibration_basis(zeta if self.columns is None else zeta[:, list(self.columns)], self.degree, self.products)

    def scores(self, t, u, v, zeta, eps=fam.EPS):
        """Normal scores of the computed transforms (n, 2) and of the calibrated ones (n, 2) in window t."""
        raw = np.column_stack([ndtri(fam.clip01(np.asarray(u, dtype=np.float64), eps)),
                               ndtri(fam.clip01(np.asarray(v, dtype=np.float64), eps))])
        return raw, (raw - self._basis(zeta) @ self.beta[t].T) / self.scale[t][None, :]

    def apply(self, t, u, v, zeta, eps=fam.EPS):
        """The calibrated transforms of window t."""
        _, new = self.scores(t, u, v, zeta, eps)
        return ndtr(new[:, 0]), ndtr(new[:, 1])

    def log_jacobian(self, t, u, v, zeta, eps=fam.EPS):
        """Per observation: log density of the two computed transforms given the coordinates under the calibration
        (sum of the two; zero for the identity).  Added to the log density of the vine, with the edge evaluated at
        the calibrated transforms, it gives the log density of the calibrated model."""
        raw, new = self.scores(t, u, v, zeta, eps)
        return (0.5 * raw * raw - 0.5 * new * new - np.log(self.scale[t])[None, :]).sum(axis=1)

    def gated(self, u: Sequence[np.ndarray], v: Sequence[np.ndarray], zeta: Sequence[np.ndarray], eps=fam.EPS):
        """Keep the calibration of a transform only if it raises the log density of that transform on held-out
        observations (lists over windows, usually the validation windows); otherwise that transform keeps the
        identity.  The calibration has 2 P + 2 coefficients per window, so for an adequate vine it lowers the
        held-out density by about that number over twice the window size; the gate removes this cost.
        Returns (calibration, gain): gain (2,) is the mean held-out log density per observation and window."""
        gain = np.zeros(2)
        for t in range(len(u)):
            raw, new = self.scores(t, u[t], v[t], zeta[t], eps)
            gain += (0.5 * raw * raw - 0.5 * new * new - np.log(self.scale[t])[None, :]).mean(axis=0)
        gain /= max(len(u), 1)
        beta, scale = self.beta.copy(), self.scale.copy()
        for k in range(2):
            if not gain[k] > 0.0:
                beta[:, k, :], scale[:, k] = 0.0, 1.0
        return Calibration(self.degree, self.products, beta, scale, self.columns), gain


def fit_calibration(u: Sequence[np.ndarray], v: Sequence[np.ndarray], zeta: Sequence[np.ndarray], degree=2, products=True,
                    columns: Optional[Sequence[int]] = None, eps=fam.EPS) -> Calibration:
    """Least-squares fit, window by window, of the normal scores of the two transforms on the calibration basis of
    the conditioning coordinates.  A window with fewer than three observations per basis function keeps the
    identity."""
    columns = None if columns is None else tuple(int(c) for c in columns)
    k = np.asarray(zeta[0]).shape[1] if columns is None else len(columns)
    width = calibration_basis(np.zeros((1, k)), degree, products).shape[1]
    steps = len(u)
    beta, scale = np.zeros((steps, 2, width)), np.ones((steps, 2))
    for t in range(steps):
        z = np.asarray(zeta[t], dtype=np.float64)
        design = calibration_basis(z if columns is None else z[:, list(columns)], degree, products)
        if design.shape[0] < 3 * width:
            continue
        raw = np.column_stack([ndtri(fam.clip01(np.asarray(u[t], dtype=np.float64), eps)),
                               ndtri(fam.clip01(np.asarray(v[t], dtype=np.float64), eps))])
        coefficients = np.linalg.lstsq(design, raw, rcond=None)[0]                 # (P, 2)
        residual = raw - design @ coefficients
        beta[t] = coefficients.T
        scale[t] = np.sqrt(np.maximum((residual * residual).mean(axis=0), 1e-12))
    return Calibration(int(degree), bool(products), beta, scale, columns)


def edge_inputs_calibrated(model: cvine.DynamicCVine, windows: Sequence[np.ndarray], level: int, edge: int,
                           components: Sequence[cvine.Component], calibration: Optional[Calibration] = None, degree=2,
                           products=True, columns: Optional[Sequence[int]] = None, eps=fam.EPS):
    """`edge_inputs` with calibrated transforms.  With ``calibration=None`` the calibration is fitted on these
    windows (use the training windows); pass the returned object for validation and test windows.
    Returns ((u, v, features), calibration)."""
    u, v, zeta = edge_transforms(model, windows, level, edge, eps)
    if calibration is None:
        calibration = fit_calibration(u, v, zeta, degree, products, columns, eps)
    new_u, new_v, features = [], [], []
    own = uses_own_scores(components)
    for t in range(len(u)):
        a, b = calibration.apply(t, u[t], v[t], zeta[t], eps)
        new_u.append(a)
        new_v.append(b)
        features.append(component_features(components, zeta[t], own_scores(model, windows[t], level, eps) if own else None))
    return (new_u, new_v, features), calibration


def _terms(theta, edge: Edge):
    """Per observation: log density, its derivative with respect to the predictor (zero where the predictor is
    clipped) and the expected information 1 + rho^2 (zero where clipped); all zero on padding."""
    eta = np.einsum('tnk,tk->tn', edge.design, theta)
    inside = np.abs(eta) < ETA_MAX
    rho = np.tanh(np.clip(eta, -ETA_MAX, ETA_MAX))
    s = 1.0 - rho * rho
    values = edge.w * (-0.5 * np.log(s) - (rho * rho * edge.sq - 2.0 * rho * edge.xy) / (2.0 * s))
    slope = edge.w * np.where(inside, rho + ((1.0 + rho * rho) * edge.xy - rho * edge.sq) / s, 0.0)
    weight = edge.w * np.where(inside, 1.0 + rho * rho, 0.0)
    return values, slope, weight


def _derivatives(theta, edge: Edge, ridge=0.0):
    """Gradient (T, K+1) and expected Hessian (T, K+1, K+1) of the negative log-likelihood of every window, plus
    the ridge term on the modulation coefficients; and the total log-likelihood."""
    values, slope, weight = _terms(theta, edge)
    gradient = -np.einsum('tn,tnk->tk', slope, edge.design)
    hessian = np.matmul((edge.design * weight[:, :, None]).transpose(0, 2, 1), edge.design)
    if ridge > 0.0 and edge.width > 1:
        r = 2.0 * ridge * edge.counts
        gradient[:, 1:] += r[:, None] * theta[:, 1:]
        index = np.arange(1, edge.width)
        hessian[:, index, index] += r[:, None]
    return gradient, hessian, float(values.sum())


def _observed(theta, edge: Edge):
    """Observed Hessian (T, K+1, K+1) of the negative log-likelihood of every window (no ridge).  Per observation
    the second derivative of the log density with respect to the predictor is
    s + ((2 rho xy - x^2 - y^2) s + 2 rho ((1 + rho^2) xy - rho (x^2 + y^2))) / s with s = 1 - rho^2; its
    expectation under the model is -(1 + rho^2).  Rows at the clipping bound contribute nothing."""
    eta = np.einsum('tnk,tk->tn', edge.design, theta)
    inside = np.abs(eta) < ETA_MAX
    rho = np.tanh(np.clip(eta, -ETA_MAX, ETA_MAX))
    s = 1.0 - rho * rho
    second = s + ((2.0 * rho * edge.xy - edge.sq) * s + 2.0 * rho * ((1.0 + rho * rho) * edge.xy - rho * edge.sq)) / s
    weight = -edge.w * np.where(inside, second, 0.0)
    return np.matmul((edge.design * weight[:, :, None]).transpose(0, 2, 1), edge.design)


def window_scores(theta, edge: Edge):
    """Mean log density per observation of every window."""
    return _terms(theta, edge)[0].sum(axis=1) / edge.counts


def log_density(theta, edge: Edge) -> np.ndarray:
    """Log density of every observation, (T, n); zero on padding."""
    return _terms(theta, edge)[0]


def score(theta, edge: Edge) -> float:
    """Mean log density per observation (windows weighted by their size)."""
    return float(_terms(theta, edge)[0].sum() / edge.counts.sum())


# ----------------------------------------------------------------------------------------------------------------
# Penalty and pieces.
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Penalty:
    """One entry per coefficient (baseline first).  ``whole``: the trajectory is a single piece over all windows
    or, for a modulation coefficient, off everywhere."""
    smooth: Tuple[float, ...]
    jump: Tuple[float, ...]
    order: Tuple[int, ...]
    whole: Tuple[bool, ...]
    include: float = 0.0
    ridge: float = 1e-3


def _spread(value, k, kind):
    """Scalar -> every coefficient; pair -> (baseline, every modulation coefficient); otherwise one per coefficient."""
    if np.isscalar(value):
        return tuple(kind(value) for _ in range(k + 1))
    value = list(value)
    if len(value) == 2 and k != 1:
        return (kind(value[0]),) + tuple(kind(value[1]) for _ in range(k))
    if len(value) != k + 1:
        raise ValueError('one value, a (baseline, modulation) pair or one value per coefficient')
    return tuple(kind(x) for x in value)


def penalty(k: int, smooth=np.inf, jump=0.0, order=2, whole=False, include=0.0, ridge=1e-3) -> Penalty:
    """Penalty for an edge with k modulation components."""
    return Penalty(_spread(smooth, k, float), _spread(jump, k, float), _spread(order, k, int), _spread(whole, k, bool),
                   float(include), float(ridge))


def constant_modulation(pen: Penalty, terms: Optional[Sequence[int]] = None) -> Penalty:
    """The same penalty with every modulation trajectory restricted to a constant over all windows, or off; with
    ``terms`` (indices of modulation components, from 0) only those are restricted."""
    k = len(pen.smooth) - 1
    chosen = set(range(k)) if terms is None else {int(j) for j in terms}
    if not chosen <= set(range(k)):
        raise ValueError('terms must index the modulation components')
    smooth, order, whole = list(pen.smooth), list(pen.order), list(pen.whole)
    for j in chosen:
        smooth[1 + j], order[1 + j], whole[1 + j] = np.inf, 1, True
    return Penalty(tuple(smooth), pen.jump, tuple(order), tuple(whole), pen.include, pen.ridge)


def runs(piece, on) -> List[Tuple[int, int]]:
    """Pieces of one trajectory that are switched on: maximal runs (start, stop) of consecutive windows that are
    on and carry the same piece label."""
    out, start, steps = [], None, len(on)
    for t in range(steps + 1):
        if start is not None and (t == steps or not on[t] or piece[t] != piece[start]):
            out.append((start, t))
            start = None
        if t < steps and on[t] and start is None:
            start = t
    return out


def _difference_penalty(length, order):
    if length <= order:
        return np.zeros((length, length))
    d = np.diff(np.eye(length), n=order, axis=0)
    return d.T @ d


def _piece_basis(length, smooth, order):
    """Free parameters of a piece of ``length`` windows: basis (length, q) and penalty matrix (q, q)."""
    if np.isinf(smooth):
        q = min(order, length)
        time = np.arange(length) - (length - 1) / 2.0
        basis = np.stack([time ** j / max(1.0, (length - 1) / 2.0) ** j for j in range(q)], axis=1)
        return basis, np.zeros((q, q))
    return np.eye(length), smooth * _difference_penalty(length, order)


def _assemble(piece, on, pen: Penalty, counts):
    """Linear map from the free parameters to theta (T (K+1), m), the quadratic penalty on them (smoothness and
    ridge) and the price of the pieces."""
    steps, width = piece.shape
    columns, blocks, price = [], [], 0.0
    for p in range(width):
        found = runs(piece[:, p], on[:, p])
        priced = len(found) - (1 if p == 0 else 0)
        if priced > 0:
            price += pen.jump[p] * priced
        if p > 0 and found:
            price += pen.include
        for start, stop in found:
            basis, quadratic = _piece_basis(stop - start, pen.smooth[p], pen.order[p])
            block = np.zeros((steps * width, basis.shape[1]))
            block[np.arange(start, stop) * width + p] = basis
            columns.append(block)
            blocks.append(quadratic)
    mapping = np.concatenate(columns, axis=1)
    ridge = np.zeros((steps, width))
    ridge[:, 1:] = pen.ridge * counts[:, None]
    return mapping, block_diag(*blocks) + mapping.T @ (ridge.reshape(-1, 1) * mapping), price


def _solve(edge: Edge, mapping, quadratic, start, iterations=80, tolerance=1e-10):
    """Minimise - log-likelihood + phi' Q phi over the free parameters phi of fixed pieces (Fisher scoring with
    step halving).  Returns theta, the minimum and the log-likelihood."""
    steps, width = edge.steps, edge.width
    phi = np.linalg.lstsq(mapping, start.reshape(-1), rcond=None)[0]
    stacked = mapping.reshape(steps, width, -1)

    def evaluate(parameters):
        theta = (mapping @ parameters).reshape(steps, width)
        loglik = float(_terms(theta, edge)[0].sum())
        return -loglik + float(parameters @ quadratic @ parameters), theta, loglik

    value, theta, loglik = evaluate(phi)
    for _ in range(iterations):
        gradient, hessian, _ = _derivatives(theta, edge)
        slope = mapping.T @ gradient.reshape(-1) + 2.0 * quadratic @ phi
        curvature = np.einsum('tkm,tkj->mj', stacked, np.matmul(hessian, stacked)) + 2.0 * quadratic
        curvature[np.diag_indices_from(curvature)] += 1e-9
        step = np.linalg.solve(curvature, slope)
        decrease = float(slope @ step)
        if decrease < tolerance * (1.0 + abs(value)):
            break
        scale = 1.0
        while True:
            trial = evaluate(phi - scale * step)
            if trial[0] <= value - 1e-4 * scale * decrease or scale < 1e-8:
                break
            scale *= 0.5
        if trial[0] >= value:
            break
        phi = phi - scale * step
        value, theta, loglik = trial
    return theta, value, loglik


def _piece_costs(h, m, smooth, order):
    """Cost of every piece (l..r) of one trajectory under the quadratic model sum_t h_t (x_t - m_t)^2 / 2 plus the
    smoothness penalty: (T, T) with entry [l, r] for l <= r."""
    steps = h.size
    cost = np.full((steps, steps), np.inf)
    cumulative = np.r_[0.0, np.cumsum(0.5 * h * m * m)]
    for length in range(1, steps + 1):
        first = np.arange(steps - length + 1)
        index = first[:, None] + np.arange(length)[None, :]
        hs, ms = h[index], m[index]
        if smooth == 0.0:
            values = np.zeros(first.size)
        else:
            basis, quadratic = _piece_basis(length, smooth, order)
            if np.isinf(smooth):
                a = np.einsum('lq,sl,lr->sqr', basis, hs, basis)
                b = np.einsum('lq,sl->sq', basis, hs * ms)
            else:
                a = np.broadcast_to(2.0 * quadratic, (first.size, length, length)).copy()
                a[:, np.arange(length), np.arange(length)] += hs
                b = hs * ms
            solution = np.linalg.solve(a, b[:, :, None])[:, :, 0]
            values = cumulative[first + length] - cumulative[first] - 0.5 * np.einsum('sq,sq->s', b, solution)
        cost[first, first + length - 1] = np.maximum(values, 0.0)
    return cost


def best_pieces(h, m, smooth, order, jump, can_switch_off, first_free, whole=False, include=0.0):
    """Exact minimiser over all pieces of one trajectory of the quadratic model: sum_t h_t (x_t - m_t)^2 / 2, the
    smoothness penalty inside pieces, ``jump`` for every piece that is on (the first one free with ``first_free``)
    and ``include`` if any piece is on.  Returns (piece labels (T,), on (T,))."""
    steps = h.size
    cost = _piece_costs(h, m, smooth, order)
    off = 0.5 * h * m * m
    everything_off = (np.zeros(steps, dtype=int), np.zeros(steps, dtype=bool))
    if whole:
        on_cost = cost[0, steps - 1] + (0.0 if first_free else jump) + include
        if can_switch_off and off.sum() <= on_cost:
            return everything_off
        return np.zeros(steps, dtype=int), np.ones(steps, dtype=bool)
    best_on, best_off = np.full(steps, np.inf), np.full(steps, np.inf)      # windows 0..r with window r on / off
    begin = np.zeros(steps, dtype=int)
    for r in range(steps):
        before = np.zeros(r + 1)
        if r:
            before[1:] = np.minimum(best_on[:r], best_off[:r])
        price = np.full(r + 1, jump, dtype=float)
        if first_free:
            price[0] = 0.0
        candidates = before + price + cost[:r + 1, r]
        begin[r] = int(np.argmin(candidates))
        best_on[r] = candidates[begin[r]]
        if can_switch_off:
            best_off[r] = (min(best_on[r - 1], best_off[r - 1]) if r else 0.0) + off[r]
    piece, on = np.zeros(steps, dtype=int), np.zeros(steps, dtype=bool)
    r, label = steps - 1, 0
    while r >= 0:
        if best_on[r] < best_off[r]:
            piece[begin[r]:r + 1], on[begin[r]:r + 1] = label, True
            r = begin[r] - 1
        else:
            piece[r] = label
            r -= 1
        label += 1
    if can_switch_off and on.any() and min(best_on[-1], best_off[-1]) + include >= off.sum():
        return everything_off
    return piece.max() - piece, on


def quadratic_value(h, m, piece, on, smooth, order, jump, first_free, include=0.0):
    """Value of given pieces under the quadratic model of `best_pieces` (used to check its optimality)."""
    cost = _piece_costs(h, m, smooth, order)
    found = runs(piece, on)
    priced = len(found) - (1 if first_free and found and found[0][0] == 0 else 0)
    return float(sum(cost[a, b - 1] for a, b in found) + (0.5 * h * m * m)[~np.asarray(on, dtype=bool)].sum()
                 + (jump * priced if priced else 0.0) + (include if found else 0.0))


# ----------------------------------------------------------------------------------------------------------------
# Fit.
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Trajectory:
    theta: np.ndarray                      # (T, K+1): baseline, modulation coefficients (0 where switched off)
    piece: np.ndarray                      # (T, K+1) piece labels
    on: np.ndarray                         # (T, K+1)
    objective: float
    loglik: float                          # total training log-likelihood
    penalty: Penalty
    means: np.ndarray                      # (T, K) feature means subtracted in training
    sweeps: int = 0

    def pieces(self, p: int) -> List[Tuple[int, int]]:
        """Pieces of coefficient p that are switched on, as (start, stop) windows."""
        return runs(self.piece[:, p], self.on[:, p])

    def eta(self, t: int, features: np.ndarray) -> np.ndarray:
        """Fisher-z predictor of window t at rows of (uncentred) features (n, K); not clipped."""
        return self.theta[t, 0] + (np.asarray(features) - self.means[t]) @ self.theta[t, 1:]

    def coefficients(self) -> np.ndarray:
        """(T, K+1) in the convention of `cvine.EdgeState.beta`: the intercept refers to uncentred features."""
        out = self.theta.copy()
        out[:, 0] -= (self.means * self.theta[:, 1:]).sum(axis=1)
        return out

    def modulation_changes(self) -> bool:
        """True if some modulation trajectory is neither one piece over all windows nor off everywhere."""
        steps = self.theta.shape[0]
        return any(self.pieces(p) not in ([], [(0, steps)]) for p in range(1, self.theta.shape[1]))


def _structure_fit(edge, pen, piece, on, start):
    mapping, quadratic, price = _assemble(piece, on, pen, edge.counts)
    theta, value, loglik = _solve(edge, mapping, quadratic, start * on)
    return theta, value + price, loglik


def _curvatures(edge: Edge, mapping, quadratic, theta):
    """Gradient of the objective in the free parameters, and its expected and observed Hessians there."""
    steps, width = edge.steps, edge.width
    phi = np.linalg.lstsq(mapping, theta.reshape(-1), rcond=None)[0]
    stacked = mapping.reshape(steps, width, -1)
    gradient, expected, _ = _derivatives(theta, edge)
    slope = mapping.T @ gradient.reshape(-1) + 2.0 * quadratic @ phi
    fisher = np.einsum('tkm,tkj->mj', stacked, np.matmul(expected, stacked)) + 2.0 * quadratic
    observed = np.einsum('tkm,tkj->mj', stacked, np.matmul(_observed(theta, edge), stacked)) + 2.0 * quadratic
    return phi, slope, fisher, observed


def stationarity(edge: Edge, trajectory: Trajectory) -> dict:
    """Diagnostics of the coefficient fit for the pieces of ``trajectory``.  The coefficients are found by Fisher
    scoring, which stops where its step vanishes; that is the minimum over the coefficients only if the objective
    is convex there.

    decrement   the Newton decrement of the last step with the expected information (gradient' H^-1 gradient);
                near zero at a stationary point
    gradient    the largest absolute derivative of the objective with respect to a free parameter
    curvature   the smallest eigenvalue of the observed Hessian of the objective in the free parameters; negative
                means the point is not a local minimum
    clipped     observations whose predictor is at the clipping bound (the objective is not differentiable there)
    parameters  number of free parameters
    """
    mapping, quadratic, _ = _assemble(trajectory.piece, trajectory.on, trajectory.penalty, edge.counts)
    phi, slope, fisher, observed = _curvatures(edge, mapping, quadratic, trajectory.theta)
    fisher[np.diag_indices_from(fisher)] += 1e-9
    eta = np.einsum('tnk,tk->tn', edge.design, trajectory.theta)
    return dict(decrement=float(slope @ np.linalg.solve(fisher, slope)), gradient=float(np.max(np.abs(slope))) if slope.size else 0.0,
                curvature=float(np.linalg.eigvalsh(0.5 * (observed + observed.T))[0]) if slope.size else 0.0,
                clipped=int(((np.abs(eta) >= ETA_MAX) & (edge.w > 0)).sum()), parameters=int(slope.size))


def _polish(edge, pen, piece, on, theta, total, loglik, rounds=5, tolerance=1e-7):
    """Leave a stationary point of negative observed curvature: search along the direction of most negative
    curvature (both signs), then repeat the Fisher scoring.  Accepted only if the objective decreases."""
    mapping, quadratic, price = _assemble(piece, on, pen, edge.counts)
    steps, width = edge.steps, edge.width

    def value_at(parameters):
        candidate = (mapping @ parameters).reshape(steps, width)
        return -float(_terms(candidate, edge)[0].sum()) + float(parameters @ quadratic @ parameters) + price, candidate

    for _ in range(rounds):
        phi, slope, fisher, observed = _curvatures(edge, mapping, quadratic, theta)
        values, vectors = np.linalg.eigh(0.5 * (observed + observed.T))
        if values[0] >= -1e-8 * (1.0 + abs(values[-1])):
            break
        direction = vectors[:, 0]
        trials = [value_at(phi + sign * length * direction) for length in (2.0, 1.0, 0.5, 0.25, 0.1) for sign in (1.0, -1.0)]
        best = min(trials, key=lambda item: item[0])
        if best[0] >= total - tolerance:
            break
        new_theta, value, new_loglik = _solve(edge, mapping, quadratic, best[1])
        if value + price >= total - tolerance:
            break
        theta, total, loglik = new_theta, value + price, new_loglik
    return theta, total, loglik


def fit(edge: Edge, pen: Penalty, search=True, start: Optional[Trajectory] = None, sweeps=12, tolerance=1e-7,
        polish=False) -> Trajectory:
    """Fit the trajectories of one edge.  search=False keeps the starting pieces (one piece per coefficient over
    all windows unless ``start`` is given) and only solves for the coefficients.  ``polish``: after the search,
    check the observed curvature of the coefficient fit and leave a point of negative curvature (`stationarity`)."""
    steps, width = edge.steps, edge.width
    if start is None:
        piece, on = np.zeros((steps, width), dtype=int), np.ones((steps, width), dtype=bool)
        theta = np.zeros((steps, width))
        theta[:, 0] = np.arctanh(np.clip((edge.w * edge.xy).sum() / edge.counts.sum(), -0.95, 0.95))
    else:
        piece, on, theta = start.piece.copy(), start.on.copy(), start.theta.copy()
    theta, total, loglik = _structure_fit(edge, pen, piece, on, theta)
    done = 0
    if search:
        for done in range(1, sweeps + 1):
            changed = False
            for p in range(width):
                gradient, hessian, _ = _derivatives(theta, edge, pen.ridge)
                h = np.maximum(hessian[:, p, p], 1e-10)
                new_piece, new_on = best_pieces(h, theta[:, p] - gradient[:, p] / h, pen.smooth[p], pen.order[p], pen.jump[p],
                                                can_switch_off=p > 0, first_free=p == 0, whole=pen.whole[p],
                                                include=pen.include if p > 0 else 0.0)
                if runs(new_piece, new_on) == runs(piece[:, p], on[:, p]):
                    continue
                trial_piece, trial_on = piece.copy(), on.copy()
                trial_piece[:, p], trial_on[:, p] = new_piece, new_on
                trial = _structure_fit(edge, pen, trial_piece, trial_on, theta)
                if trial[1] < total - tolerance:
                    piece, on, (theta, total, loglik), changed = trial_piece, trial_on, trial, True
            if not changed:
                break
    if polish:
        theta, total, loglik = _polish(edge, pen, piece, on, theta, total, loglik)
    return Trajectory(theta * on, piece, on, float(total), float(loglik), pen, edge.means.copy(), done)


def tune(train: Edge, validation: Edge, penalties: Sequence[Penalty], search=True):
    """Fit every penalty on the training windows and keep the one with the best validation score.
    Returns (trajectory, validation score, index)."""
    best = None
    for index, pen in enumerate(penalties):
        trajectory = fit(train, pen, search=search)
        value = score(trajectory.theta, validation)
        if best is None or value > best[1] + 1e-12:
            best = (trajectory, value, index)
    return best


def identifiability(edge: Edge):
    """Two checks that baseline and modulation trajectories can be told apart.

    Returns (smallest eigenvalue of the correlation matrix of the features in every window, (T,): 1 = orthogonal,
    0 = collinear; share of each feature's variance that lies inside windows, (K,): near 0 means the feature hardly
    varies inside a window, so its effect cannot be told from a change of the baseline with time)."""
    k = edge.width - 1
    weight = edge.w[:, :, None]
    gram = np.matmul((edge.f * weight).transpose(0, 2, 1), edge.f) / edge.counts[:, None, None]       # features are centred
    scale = np.sqrt(np.maximum(np.diagonal(gram, axis1=1, axis2=2), 1e-300))
    smallest = np.linalg.eigvalsh(gram / (scale[:, :, None] * scale[:, None, :]))[:, 0] if k else np.ones(edge.steps)
    inside = np.einsum('tkk->k', gram * edge.counts[:, None, None]) / edge.counts.sum()
    grand = (edge.means * edge.counts[:, None]).sum(axis=0) / edge.counts.sum()
    between = ((edge.means - grand) ** 2 * edge.counts[:, None]).sum(axis=0) / edge.counts.sum()
    return smallest, inside / np.maximum(inside + between, 1e-300)


# ----------------------------------------------------------------------------------------------------------------
# The trajectory as an edge of a vine, and the test of a change of the modulation.
# ----------------------------------------------------------------------------------------------------------------
def as_states(trajectory: Trajectory, components: Sequence[cvine.Component]) -> List[cvine.EdgeState]:
    """The fitted edge as one `cvine.EdgeState` per window, so that the vine it belongs to can be evaluated and
    sampled by `cvine.forward` and `cvine.simulate`."""
    beta = trajectory.coefficients()
    out = []
    for t in range(beta.shape[0]):
        if trajectory.on[t, 1:].any():
            out.append(cvine.EdgeState('gaussian', components=tuple(components), beta=beta[t].copy()))
        else:
            out.append(cvine.EdgeState('gaussian', par=float(np.tanh(np.clip(beta[t, 0], -ETA_MAX, ETA_MAX)))))
    return out


def simulate_null(lower: cvine.DynamicCVine, trajectory: Trajectory, components: Sequence[cvine.Component],
                  calibration: Optional[Calibration], t: int, n: int, rng, eps=fam.EPS) -> np.ndarray:
    """n observations of window t from a fitted model of the top edge of a C-vine: the vine ``lower`` below the
    edge, the calibration of the edge's two transforms (None: the transforms are uniform given the conditioning
    variables) and the edge with the coefficients of ``trajectory``.  Columns in vine order.

    The conditioning variables come from ``lower``.  Given them, the two calibrated normal scores are bivariate
    normal with the correlation of the edge; the normal scores of the computed transforms are their means under
    the calibration plus their scales times the calibrated scores; the two variables follow by inverting the vine
    below the edge.  This is the model whose density is the vine density at the calibrated transforms plus
    `Calibration.log_jacobian`."""
    level = lower.depth
    d = level + 2
    steps = len(lower.edges[0][0])
    base = cvine.DynamicCVine(list(range(d)), list(lower.edges) + [[[cvine.EdgeState('ind', 0.0) for _ in range(steps)]]])
    w = rng.uniform(size=(n, d))
    zeta = ndtri(fam.clip01(w[:, :level], eps))
    own = None
    if uses_own_scores(components):
        own = ndtri(fam.clip01(cvine.simulate(base, t, n, rng, innovations=w)[:, :level], eps))
    rho = np.tanh(np.clip(trajectory.eta(t, component_features(components, zeta, own)), -ETA_MAX, ETA_MAX))
    second = rng.standard_normal(n)                                              # the root of the level
    first = rho * second + np.sqrt(1.0 - rho * rho) * rng.standard_normal(n)     # the later variable
    scores = np.column_stack([first, second])
    if calibration is not None:
        scores = calibration._basis(zeta) @ calibration.beta[t].T + scores * calibration.scale[t][None, :]
    w[:, level + 1], w[:, level] = ndtr(scores[:, 0]), ndtr(scores[:, 1])
    return cvine.simulate(base, t, n, rng, innovations=w)


def change_test(windows: Sequence[np.ndarray], lower_fit, components: Sequence[cvine.Component], pen: Penalty,
                replicates=99, rng=None, eps=fam.EPS, calibrate: Optional[int] = None,
                columns: Optional[Sequence[int]] = None, pooled: Optional[Sequence] = None, null='vine',
                terms: Optional[Sequence[int]] = None):
    """Test that the modulation of the top edge of a C-vine does not change with time.

    windows: list of (n_t, d) pseudo-observations, columns in vine order; the edge is the one of the last two
    columns given the others.  lower_fit(windows) must return the simplified vine below that edge (d - 2 levels).
    Statistic: objective of the fit whose modulation trajectories are constant over all windows (or off) minus the
    objective of the unrestricted fit, both with the penalty ``pen``.  The unrestricted search starts from the
    restricted fit, so the statistic is never negative and is exactly zero when freeing the modulation finds no
    better pieces.  Its null distribution is simulated from the restricted fit, with the vine below the edge as
    fitted, and everything is refitted on every simulated dataset (the lower vine too: a change of the pairwise
    layer alone changes fitted modulation coefficients slightly).  With ``calibrate`` (a degree, 1 or 2) the
    transforms of the edge are calibrated against the conditioning coordinates (`fit_calibration`; all of them, or
    those of ``columns``) in the observed data and in every simulated dataset.
    ``null``: what the simulation is from.  'vine' (as in E17 to E18): the fitted vine with the edge's
    coefficients, without the calibration, although the coefficients were estimated on calibrated transforms.
    'calibrated': the model that was fitted, calibration included (`simulate_null`); without ``calibrate`` it is
    the same model as 'vine'.  Terms in the variables' own scores need 'calibrated'.
    ``terms``: indices (from 0) of the modulation components whose constancy is tested; the others are free under
    the null as well.  Default: all of them, which tests the edge and says nothing about a single term.
    ``pooled``: per column, the flag of `margins.choose` that says how the margin of that variable was estimated
    when ``windows`` were made (over all windows, aligned, or within every window).  The simulated datasets then
    get their pseudo-observations the same way (`margins.ranks`); by default from ranks within every window.
    Returns dict(statistic, p, free, constant, null).
    """
    rng = np.random.default_rng() if rng is None else rng
    d = windows[0].shape[1]
    level = d - 2
    if pooled is not None and len(pooled) != d:
        raise ValueError('pooled needs one flag per column')
    if null not in ('vine', 'calibrated'):
        raise ValueError("null must be 'vine' or 'calibrated'")
    if null == 'vine' and uses_own_scores(components):
        raise ValueError("terms in the variables' own scores need null='calibrated'")
    restricted = constant_modulation(pen, terms)

    def both(data):
        lower = lower_fit(data)
        calibration = None
        if calibrate is None:
            edge = edge_data(*edge_inputs(lower, data, level, 0, components, eps))
        else:
            inputs, calibration = edge_inputs_calibrated(lower, data, level, 0, components, degree=calibrate, columns=columns, eps=eps)
            edge = edge_data(*inputs)
        constant = fit(edge, restricted)
        free = fit(edge, pen, start=constant)
        gain = constant.objective - free.objective
        return lower, free, constant, gain if gain > 1e-6 else 0.0, calibration

    lower, free, constant, statistic, calibration = both(list(windows))
    draws = np.zeros(replicates)
    if statistic > 0.0:
        if null == 'vine':
            model = cvine.DynamicCVine(list(range(d)), lower.edges + [[as_states(constant, components)]])
        for b in range(replicates):
            if null == 'vine':
                raw = [cvine.simulate(model, t, u.shape[0], rng) for t, u in enumerate(windows)]
            else:
                raw = [simulate_null(lower, constant, components, calibration, t, u.shape[0], rng, eps) for t, u in enumerate(windows)]
            draws[b] = both(margins.ranks(raw, pooled))[3]
    return dict(statistic=float(statistic), p=float((1 + np.sum(draws >= statistic)) / (replicates + 1)) if statistic > 0.0 else 1.0,
                free=free, constant=constant, null=draws)


def local_change_test(edge: Edge, pen: Penalty, replicates=99, rng=None):
    """Test that the modulation of an edge does not change with time, given the edge's transforms and features.

    Statistic as `change_test`.  Null distribution: the two normal scores are drawn, observation by observation,
    from the Gaussian pair with the correlation of the restricted fit (modulation constant over the windows or off,
    the baseline as fitted); the features are kept; both fits are repeated on every draw.  Nothing below the edge
    is refitted, so the test costs two fits of one edge per draw and can be used in a vine with many variables.
    It ignores that the transforms were estimated, and E17i shows that this matters: with a constant modulation of
    strength 0.6 it rejects in 17 of 40 datasets at n = 1000 where `change_test` rejects in 2.
    Kept as a fast screen and for the record; claims of a change need `change_test`.
    Returns dict(statistic, p, free, constant, null)."""
    rng = np.random.default_rng() if rng is None else rng

    def both(data):
        constant = fit(data, constant_modulation(pen))
        free = fit(data, pen, start=constant)
        gain = constant.objective - free.objective
        return free, constant, gain if gain > 1e-6 else 0.0

    free, constant, statistic = both(edge)
    draws = np.zeros(replicates)
    if statistic > 0.0:
        rho = np.tanh(np.clip(np.einsum('tnk,tk->tn', edge.design, constant.theta), -ETA_MAX, ETA_MAX))
        for b in range(replicates):
            x = rng.normal(size=edge.x.shape)
            y = rho * x + np.sqrt(1.0 - rho * rho) * rng.normal(size=edge.x.shape)
            draws[b] = both(Edge(x * edge.w, y * edge.w, edge.f, edge.w, edge.means))[2]
    return dict(statistic=float(statistic), p=float((1 + np.sum(draws >= statistic)) / (replicates + 1)) if statistic > 0.0 else 1.0,
                free=free, constant=constant, null=draws)


# ----------------------------------------------------------------------------------------------------------------
# Reference: the gated switching edge of `cvine.fit` on given transforms (window-by-window coefficients).
# ----------------------------------------------------------------------------------------------------------------
def switching_candidates(u, v, features, candidates: Sequence[Tuple[int, ...]], states: Sequence[str], ridge=1e-3,
                         eps=fam.EPS, nu_grid=fam.NU_GRID):
    """Window-by-window fits of the unmodulated states and of every modulated candidate (a tuple of feature
    columns), as `cvine.fit` makes them.  Returns a dict used by `switching_path`."""
    steps = len(u)
    fits = [fam.fit_states(np.asarray(u[t])[:, None], np.asarray(v[t])[:, None], states, nu_grid=nu_grid, eps=eps) for t in range(steps)]
    par, nu, loglik = (np.stack([f[i] for f in fits]) for i in range(3))
    modulated = []
    for columns in candidates:
        rows = [cvine.fit_modulated_gaussian(u[t], v[t], np.asarray(features[t])[:, list(columns)], ridge, eps) for t in range(steps)]
        modulated.append((np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])))
    return dict(states=list(states), par=par, nu=nu, loglik=loglik, modulated=modulated, candidates=[tuple(c) for c in candidates],
                counts=np.array([len(a) for a in u], dtype=float), k=np.asarray(features[0]).shape[1])


def switching_path(fits, switch=0.05, drift=0.0, activation=0.0):
    """State path of the switching objective over the candidates of `switching_candidates`.

    Returns (coefficients (T, K+1) in the convention of `cvine.EdgeState.beta`: Fisher-z intercept, then one
    coefficient per feature, zero where not modelled; chosen state index per window; names)."""
    states, counts = fits['states'], fits['counts']
    cost = paths.local_costs(fits['loglik'], counts, states, activation)[:, :, 0]
    distance = paths.parameter_distance(fits['par'], fits['nu'])[:, :, 0]
    extra_cost = [(1.0 + len(columns)) / counts - loglik + activation for columns, (_, loglik) in zip(fits['candidates'], fits['modulated'])]
    extra_distance = [np.r_[0.0, np.nan_to_num(np.abs(np.diff(beta, axis=0)).mean(axis=1))] for beta, _ in fits['modulated']]
    local = np.concatenate((cost, np.stack(extra_cost, axis=1)), axis=1)[:, :, None] if extra_cost else cost[:, :, None]
    dist = np.concatenate((distance, np.stack(extra_distance, axis=1)), axis=1)[:, :, None] if extra_cost else distance[:, :, None]
    choice = paths.viterbi(local, dist, switch=switch, drift=drift)[0][:, 0]
    theta = np.zeros((len(counts), fits['k'] + 1))
    names = []
    for t, s in enumerate(choice):
        if s >= len(states):
            columns = fits['candidates'][s - len(states)]
            beta = fits['modulated'][s - len(states)][0][t]
            theta[t, 0] = beta[0]
            theta[t, 1 + np.array(columns, dtype=int)] = beta[1:]
            names.append('modulated' + str(tuple(columns)))
        else:
            names.append(states[s])
            if states[s] != 'ind':
                if fam.split_state(states[s])[0] != 'gaussian':
                    raise ValueError('coefficients are defined for independence and Gaussian states only')
                theta[t, 0] = np.arctanh(fits['par'][t, s, 0])
    return theta, choice, names
