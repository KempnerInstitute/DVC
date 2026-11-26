"""Dynamic C-vine whose pair copulas switch state over time windows and may be modulated by conditioning variables.

Vine coordinates.  With root order r_0, r_1, ..., the edge at level l joins r_l and a later variable j given
D = {r_0, ..., r_{l-1}}.  zeta_m is the normal score of P(U_{r_m} <= u | U_{r_0}, ..., U_{r_{m-1}}); the zeta are
independent standard normal under any law.  A modulation component of an edge is a function of some of zeta_0 ..
zeta_{l-1}: one coordinate gives a third-order interaction among {r_m, r_l, j}, a product of two a fourth-order one.
Every subset of three or more variables corresponds to exactly one (edge, set of modulating roots).

A modulated edge is a Gaussian pair copula whose Fisher-z correlation is beta_0 + sum_c beta_c f_c(zeta); for each
value of the conditioning variables it is a copula, so the vine remains a normalised density.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import kendalltau, rankdata

from . import families as fam
from . import paths, scan

Component = Tuple[str, Tuple[int, ...]]          # (shape, positions of the modulating roots)
ETA_MAX = float(np.arctanh(0.98))                 # modulated correlations stay within +-0.98, the library's bound


@dataclass
class EdgeState:
    state: str = 'ind'
    par: float = float('nan')
    nu: float = float('nan')
    components: Tuple[Component, ...] = ()
    beta: Optional[np.ndarray] = None
    loglik: float = 0.0
    converged: bool = True

    @property
    def order(self) -> int:
        """Largest number of variables in one term of the fitted Fisher-z predictor.  This labels the working model;
        the order of the fitted modulation on the correlation scale is given by ``coupling_decomposition``."""
        return 2 + max((len(members) for _, members in self.components), default=0)


@dataclass
class DynamicCVine:
    order: List[int]
    edges: List[List[List[EdgeState]]]            # [level][edge][window]
    config: dict = field(default_factory=dict)

    @property
    def d(self):
        return len(self.order)

    @property
    def depth(self):
        return len(self.edges)

    @property
    def steps(self):
        return len(self.edges[0][0])

    def subset(self, level: int, edge: int, members: Sequence[int] = ()):
        """Variables of the interaction carried by a component of an edge."""
        return tuple(sorted([self.order[m] for m in members] + [self.order[level], self.order[level + 1 + edge]]))


def pseudo_observations(x):
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x, axis=0, method='average') / (x.shape[0] + 1.0)


def infer_order(windows: Sequence[np.ndarray]) -> List[int]:
    """Root order of DVC-switch: mean over windows of |Kendall tau|, root with the largest row sum, then greedy."""
    d = windows[0].shape[1]
    dep = np.zeros((d, d))
    for u in windows:
        for i, j in combinations(range(d), 2):
            tau = kendalltau(u[:, i], u[:, j])[0]
            value = abs(float(tau)) if np.isfinite(tau) else 0.0
            dep[i, j] += value / len(windows)
            dep[j, i] += value / len(windows)
    remaining = list(range(d))
    order = [remaining.pop(int(np.argmax(dep.sum(axis=1))))]
    while remaining:
        scores = [float(dep[v, remaining].sum() - dep[v, v]) for v in remaining]
        order.append(remaining.pop(int(np.argmax(scores))))
    return order


def enumerate_components(level: int, max_order: int, shapes: Sequence[str] = scan.SHAPES) -> List[Component]:
    out: List[Component] = []
    if max_order >= 3:
        out.extend((shape, (m,)) for m in range(level) for shape in shapes)
    for size in range(2, max_order - 1):
        out.extend(('product', members) for members in combinations(range(level), size))
    return out


def component_features(components: Sequence[Component], zeta: np.ndarray) -> np.ndarray:
    columns = []
    for shape, members in components:
        if shape == 'linear':
            columns.append(zeta[:, members[0]])
        elif shape == 'quadratic':
            columns.append((zeta[:, members[0]] ** 2 - 1.0) / np.sqrt(2.0))
        elif shape == 'product':
            columns.append(np.prod(zeta[:, list(members)], axis=1))
        else:
            raise ValueError(shape)
    return np.stack(columns, axis=1) if columns else np.zeros((zeta.shape[0], 0))


def edge_parameter(edge: EdgeState, zeta: np.ndarray):
    if not edge.components:
        return edge.par
    eta = edge.beta[0] + component_features(edge.components, zeta) @ edge.beta[1:]
    return np.tanh(np.clip(eta, -ETA_MAX, ETA_MAX))


def modulated_loglik(beta, design, xy, sq):
    """Per-observation log density of the Gaussian copula with atanh(rho) = design @ beta (clipped at +-ETA_MAX),
    its derivative with respect to the predictor (zero where the predictor is clipped), and rho."""
    eta = design @ beta
    rho = np.tanh(np.clip(eta, -ETA_MAX, ETA_MAX))
    s = 1.0 - rho * rho
    values = -0.5 * np.log(s) - (rho * rho * sq - 2.0 * rho * xy) / (2.0 * s)
    slope = np.where(np.abs(eta) < ETA_MAX, rho + ((1.0 + rho * rho) * xy - rho * sq) / s, 0.0)
    return values, slope, rho


def modulated_objective(beta, design, xy, sq, penalty):
    """Penalised negative mean log-likelihood and its gradient."""
    values, slope, _ = modulated_loglik(beta, design, xy, sq)
    return -values.mean() + penalty @ (beta * beta), -(design.T @ slope) / xy.size + 2.0 * penalty * beta


def fit_modulated_gaussian(u, v, features, ridge=1e-3, eps=fam.EPS):
    """Maximum likelihood for a Gaussian pair copula with atanh(rho) = beta_0 + features @ beta, the predictor being
    clipped at +-ETA_MAX.  Returns (beta, mean log-likelihood, converged).

    The gradient accounts for the clipping (rows at the bound contribute nothing).  If the quasi-Newton search does
    not report success, a derivative-free search continues from its end point; the result is never worse than the
    starting value (constant correlation), and ``converged`` records whether either search reported success.
    """
    x, y = ndtri(fam.clip01(u, eps)), ndtri(fam.clip01(v, eps))
    design = np.column_stack((np.ones(x.size), features))
    xy, sq = x * y, x * x + y * y
    start = np.zeros(design.shape[1])
    start[0] = np.arctanh(np.clip(xy.mean(), -0.95, 0.95))
    penalty = np.r_[0.0, np.full(design.shape[1] - 1, ridge)]
    args = (design, xy, sq, penalty)
    result = minimize(modulated_objective, start, args=args, jac=True, method='BFGS', options={'maxiter': 500, 'gtol': 1e-6})
    best, value, converged = result.x, float(result.fun), bool(result.success)
    if not converged or not np.isfinite(value):
        begin = best if np.all(np.isfinite(best)) and np.isfinite(value) else start
        polish = minimize(lambda beta: modulated_objective(beta, *args)[0], begin, method='Nelder-Mead',
                          options={'maxiter': 2000 * design.shape[1], 'xatol': 1e-6, 'fatol': 1e-10})
        if np.isfinite(polish.fun) and (not np.isfinite(value) or polish.fun <= value):
            best, value = polish.x, float(polish.fun)
        converged = bool(polish.success)
    if not np.isfinite(value) or value > modulated_objective(start, *args)[0]:
        best, converged = start, False
    return best, float(modulated_loglik(best, design, xy, sq)[0].mean()), converged


# ----------------------------------------------------------------------------------------------------------------
# Order on the correlation scale: the conditional normal-score correlation r(zeta) of the edge is decomposed by functional ANOVA over
# the independent coordinates it depends on.  A fitted state is additive on the Fisher-z scale, which is not the
# same thing: tanh(b0 + b1 z1 + b2 z2) has a component that depends jointly on z1 and z2.  These functions compute
# the decomposition of the fitted function itself, whatever its parameterisation.
# ----------------------------------------------------------------------------------------------------------------
def coupling_members(edge: EdgeState) -> Tuple[int, ...]:
    """Positions of the conditioning coordinates the fitted correlation of the edge depends on."""
    return tuple(sorted({m for _, members in edge.components for m in members}))


def _gauss_hermite(nodes):
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / w.sum()


def _projection(edge: EdgeState, kept: Tuple[int, ...], zeta: Optional[np.ndarray], nodes: int):
    """E[r | coordinates in ``kept``] with the other coordinates integrated against N(0, 1).  With ``kept`` empty
    the result is a number; otherwise one value per row of zeta."""
    members = coupling_members(edge)
    free = [m for m in members if m not in kept]
    x, w = _gauss_hermite(nodes)
    width = max(members) + 1
    grid = np.stack(np.meshgrid(*([x] * len(free)), indexing='ij'), axis=-1).reshape(-1, len(free)) if free else np.zeros((1, 0))
    weight = np.ones(1)
    for _ in free:
        weight = np.multiply.outer(weight, w).reshape(-1)
    rows = 1 if not kept else zeta.shape[0]
    points = np.zeros((rows, grid.shape[0], width))
    if kept:
        points[:, :, list(kept)] = zeta[:, None, list(kept)]
    if free:
        points[:, :, free] = grid[None, :, :]
    values = edge_parameter(edge, points.reshape(-1, width)).reshape(rows, grid.shape[0])
    out = values @ weight
    return float(out[0]) if not kept else out


def truncated_coupling(edge: EdgeState, zeta: np.ndarray, max_order: int, nodes=32, bound=0.98):
    """The fitted correlation with every component of order above ``max_order`` removed, at the rows of zeta
    (order of a component = 2 + number of coordinates it depends on).  A sum of lower-order components of a
    correlation need not be a correlation, so the result is clipped at +-``bound`` (``bound=None``: not clipped)."""
    members = coupling_members(edge)
    keep = max_order - 2
    if keep >= len(members):
        return edge_parameter(edge, zeta)
    out = np.zeros(zeta.shape[0])
    for size in range(0, max(keep, 0) + 1):
        coefficient = sum((-1) ** j * _binomial(len(members) - size, j) for j in range(0, keep - size + 1))
        if coefficient == 0:
            continue
        for kept in combinations(members, size):
            out = out + coefficient * _projection(edge, kept, zeta, nodes)
    return out if bound is None else np.clip(out, -bound, bound)


def _binomial(n, k):
    from math import comb
    return comb(n, k)


def coupling_decomposition(edge: EdgeState, nodes=40):
    """Variance of the fitted correlation by component: {coordinates (positions): variance}, with the mean under
    the key ``'mean'``.  The variances add up to the variance of the correlation over the conditioning variables."""
    members = coupling_members(edge)
    if not members:
        return {'mean': float(edge.par) if np.isfinite(edge.par) else 0.0}
    x, w = _gauss_hermite(nodes)
    width = max(members) + 1
    grid = np.stack(np.meshgrid(*([x] * len(members)), indexing='ij'), axis=-1).reshape(-1, len(members))
    points = np.zeros((grid.shape[0], width))
    points[:, list(members)] = grid
    values = edge_parameter(edge, points).reshape((nodes,) * len(members))
    axes = list(range(len(members)))

    def project(kept_axes):
        out = values
        for axis in reversed([a for a in axes if a not in kept_axes]):
            out = np.tensordot(out, w, axes=([axis], [0]))
        return out                                                    # array over the kept axes, in order

    def expand(array, kept_axes, target_axes):
        shape = [nodes if a in kept_axes else 1 for a in target_axes]
        return array.reshape(shape) if kept_axes else array

    result = {'mean': float(project(()))}
    for size in range(1, len(members) + 1):
        for kept_axes in combinations(axes, size):
            component = np.zeros((nodes,) * size)
            for inner in range(0, size + 1):
                for sub in combinations(kept_axes, inner):
                    component = component + (-1) ** (size - inner) * expand(project(sub), sub, kept_axes)
            weight = np.ones(1)
            for _ in kept_axes:
                weight = np.multiply.outer(weight, w)
            result[tuple(members[a] for a in kept_axes)] = float((component ** 2 * weight.reshape((nodes,) * size)).sum())
    return result


def forward(model: DynamicCVine, t: int, u, eps=fam.EPS, keep=False, max_order=None):
    """One pass through the vine at window t.  u: (n, d) pseudo-observations in the original column order.
    Returns log densities per level (n, depth), the zeta coordinates and, with ``keep``, the probability transforms
    at the start of every level (columns in vine order) and the per-edge log densities.  With ``max_order`` every
    modulated edge uses its fitted correlation truncated to components of at most that order (no refit)."""
    v = fam.clip01(np.asarray(u, dtype=np.float64)[:, model.order], eps)
    n, d = v.shape
    zeta = np.empty((n, model.depth))
    logc = np.zeros((n, model.depth))
    pits, edge_logc = [], []
    for level in range(model.depth):
        if keep:
            pits.append(v.copy())
        root = v[:, level].copy()
        zeta[:, level] = ndtri(root)
        row = []
        for e, pos in enumerate(range(level + 1, d)):
            edge = model.edges[level][e][t]
            if edge.state == 'ind' and not edge.components:
                row.append(np.zeros(n))
                continue
            if edge.components and max_order is not None:
                par = truncated_coupling(edge, zeta[:, :level], max_order)
            else:
                par = edge_parameter(edge, zeta[:, :level])
            lc, h = fam.evaluate(edge.state, v[:, pos], root, par, edge.nu, eps)
            logc[:, level] += lc
            v[:, pos] = h
            row.append(lc)
        edge_logc.append(row)
    if keep:
        pits.append(v.copy())
    return dict(logc=logc, zeta=zeta, pits=pits, edge_logc=edge_logc)


def simulate(model: DynamicCVine, t: int, n: int, rng, innovations=None):
    """Draw n observations from the vine at window t (inverse Rosenblatt transform).  ``innovations`` (n, d) are
    the uniform conditional probability transforms in vine order; by default independent uniforms.  Passing
    serially dependent innovations gives serially dependent observations with the same law at every time point."""
    d = model.d
    w = rng.uniform(size=(n, d)) if innovations is None else np.asarray(innovations, dtype=np.float64)
    zeta = ndtri(fam.clip01(w))
    out = np.empty((n, d))
    for pos in range(d):
        x = w[:, pos]
        for level in range(min(pos, model.depth) - 1, -1, -1):
            edge = model.edges[level][pos - level - 1][t]
            if edge.state == 'ind' and not edge.components:
                continue
            x = fam.h_inverse(edge.state, x, w[:, level], edge_parameter(edge, zeta[:, :level]), edge.nu)
        out[:, model.order[pos]] = x
    return out


def log_density(model: DynamicCVine, windows: Sequence[np.ndarray], eps=fam.EPS, max_order=None) -> List[np.ndarray]:
    return [forward(model, t, u, eps, max_order=max_order)['logc'].sum(axis=1) for t, u in enumerate(windows)]


def score(model: DynamicCVine, windows: Sequence[np.ndarray], eps=fam.EPS, max_order=None) -> float:
    """Mean held-out log copula density per observation (windows weighted by their size)."""
    values = np.concatenate(log_density(model, windows, eps, max_order))
    return float(values.mean())


def _edge_statistics(root, others, zeta, components, rerank, hac_lags=0, standardize=0):
    """z statistics (C, P) of the components for the edges (root, each column of others) in one window."""
    features = component_features(components, zeta)
    features = features - features.mean(axis=0)
    a = scan.normal_scores(root[:, None], rerank)
    b = scan.normal_scores(others, rerank)
    if standardize:
        a, b = scan.conditional_standardize(a, zeta, standardize), scan.conditional_standardize(b, zeta, standardize)
    return scan.moment_scan(a, b, features, hac_lags=hac_lags)[:, 0, :]


def anchored_scan(model: DynamicCVine, windows: Sequence[np.ndarray], max_order=4, shapes=scan.SHAPES, rerank=True,
                  eps=fam.EPS, hac_lags=0, standardize=0):
    """Modulation statistics of every edge of the fitted vine: each subset of 3..max_order variables once.

    Returns a list of records with keys level, edge, component, subset, order and z (one value per window).
    """
    records = {}
    for t, u in enumerate(windows):
        out = forward(model, t, u, eps, keep=True)
        for level in range(1, model.depth):
            components = enumerate_components(level, max_order, shapes)
            if not components:
                continue
            v = out['pits'][level]
            z = _edge_statistics(v[:, level], v[:, level + 1:], out['zeta'][:, :level], components, rerank, hac_lags,
                                 standardize)
            for c, component in enumerate(components):
                for e in range(z.shape[1]):
                    key = (level, e, component)
                    if key not in records:
                        records[key] = dict(level=level, edge=e, component=component,
                                            subset=model.subset(level, e, component[1]),
                                            order=2 + len(component[1]), z=np.full(len(windows), np.nan))
                    records[key]['z'][t] = z[c, e]
    return list(records.values())


def shift_null(model: DynamicCVine, windows: Sequence[np.ndarray], replicates=199, rng=None, min_shift=20, max_order=4,
               shapes=scan.SHAPES, rerank=True, eps=fam.EPS):
    """Null distributions for serially dependent observations.

    In every window the modulating coordinates are circularly shifted in time against the pair (a different random
    shift per window and replicate).  This keeps the serial dependence of both and removes any modulation.  Returns
    (largest |z| over all components and windows, largest interval-scan value over all components), one value per
    replicate, computed with the plain variance exactly as ``anchored_scan`` does; their 0.95 quantiles are familywise
    thresholds for the observed window statistics and interval-scan values.
    """
    rng = np.random.default_rng() if rng is None else rng
    inputs = []
    for t, u in enumerate(windows):
        out = forward(model, t, u, eps, keep=True)
        rows = []
        for level in range(1, model.depth):
            components = enumerate_components(level, max_order, shapes)
            if not components:
                continue
            v = out['pits'][level]
            features = component_features(components, out['zeta'][:, :level])
            rows.append((scan.normal_scores(v[:, [level]], rerank), scan.normal_scores(v[:, level + 1:], rerank),
                         features - features.mean(axis=0)))
        inputs.append(rows)
    largest, largest_interval = np.empty(replicates), np.empty(replicates)
    for b in range(replicates):
        z = []
        for t, rows in enumerate(inputs):
            n = windows[t].shape[0]
            shift = int(rng.integers(min_shift, max(n - min_shift, min_shift + 1)))
            z.append(np.concatenate([scan.moment_scan(a, others, np.roll(features, shift, axis=0))[:, 0, :].ravel()
                                     for a, others, features in rows]))
        z = np.array(z)
        largest[b] = np.abs(z).max()
        largest_interval[b] = scan.interval_scan(z)[0].max()
    return largest, largest_interval


def _fit_states_job(job):
    u, w, states, nu_grid, eps = job
    return fam.fit_states(u, w, states, nu_grid=nu_grid, eps=eps)


def fit(train: Sequence[np.ndarray], order: Optional[Sequence[int]] = None, states: Optional[Sequence[str]] = None,
        switch=0.05, drift=0.0, activation=0.0, max_order=2, shapes=scan.SHAPES, screen=None, max_depth=None,
        rerank=True, ridge=1e-3, eps=fam.EPS, nu_grid=fam.NU_GRID, max_joint=3, allowed=None, pool=None,
        serial: Optional[Sequence[float]] = None) -> DynamicCVine:
    """Fit the dynamic C-vine level by level.

    train: list of (n_t, d) pseudo-observations.  ``max_order = 2`` gives the simplified switching vine (the
    DVC-switch objective).

    Which modulations may be modelled is decided in one of two ways.
    - ``allowed`` (the pipeline's way): a dict {subset of variables: boolean mask over windows} of interactions
      confirmed by the detector (`pipeline.detect`).  The edge that carries a subset in this vine receives modulated
      candidate states in the windows of the mask (linear and quadratic shapes for a triple, the product for a
      quadruple); ``max_order`` is taken from the largest subset.
    - Otherwise, with ``max_order >= 3``, the vine's own scan: components whose statistics reach ``screen`` over a
      run of windows (``screen=None`` = familywise 0.05 over all components of this vine).  This depends on the
      root order and is kept for diagnostics.
    In both cases the switching path objective chooses between modulated and unmodulated states in every window.
    ``pool``: an executor; the family fits of the windows of a level are then run through it (same results, for
    vines with many variables).
    ``serial``: for serially dependent observations, the lag-one autocorrelation of every variable (in the column
    order of ``train``).  The costs of an edge are then scaled by the inflation of the variance of an average of
    products of its two series, (1 + q) / (1 - q) with q the product of their autocorrelations (at most 0.98):
    ``switch`` is multiplied by it and (inflation - 1) / mean window size is added to ``activation``, per edge.
    With one autocorrelation r for all variables this is ``switch * inflation`` and ``activation + (inflation - 1)
    / size`` with q = r^2, which `vinemodel.fit` passes for all edges.
    """
    states = list(states) if states is not None else fam.default_states()
    order = list(order) if order is not None else infer_order(train)
    if allowed is not None:
        allowed = {tuple(sorted(int(v) for v in subset)): np.asarray(mask, dtype=bool) for subset, mask in allowed.items()}
        max_order = max([2] + [len(subset) for subset in allowed])
    d = len(order)
    depth = d - 1 if max_depth is None else min(int(max_depth), d - 1)
    steps = len(train)
    counts = np.array([u.shape[0] for u in train], dtype=float)
    v = [fam.clip01(np.asarray(u, dtype=np.float64)[:, order], eps) for u in train]
    zeta = [np.empty((u.shape[0], depth)) for u in train]
    edges: List[List[List[EdgeState]]] = []
    screened_log, failures = [], 0
    if screen is None:
        total = sum((d - level - 1) * len(enumerate_components(level, max_order, shapes)) for level in range(1, depth))
        screen = scan.interval_threshold(steps, max(total, 1))
    if serial is not None:
        serial = np.maximum(np.asarray(serial, dtype=np.float64), 0.0)
        if serial.shape != (d,):
            raise ValueError('serial needs one autocorrelation per variable')
        serial = serial[order]                                       # in vine order
    base_switch, base_activation = switch, activation
    for level in range(depth):
        n_edges = d - level - 1
        if serial is not None:                                       # costs per edge of this level
            q = np.minimum(serial[level] * serial[level + 1:], 0.98)
            inflation = (1.0 + q) / (1.0 - q)
            switch, activation = base_switch * inflation, base_activation + (inflation - 1.0) / counts.mean()
        for t in range(steps):
            zeta[t][:, level] = ndtri(v[t][:, level])
        jobs = [(v[t][:, level + 1:], np.repeat(v[t][:, level:level + 1], n_edges, axis=1), states, nu_grid, eps)
                for t in range(steps)]
        fits = list(pool.map(_fit_states_job, jobs)) if pool is not None else [_fit_states_job(job) for job in jobs]
        par, nu, loglik = (np.stack([f[i] for f in fits]) for i in range(3))
        cost = paths.local_costs(loglik, counts, states, activation)
        distance = paths.parameter_distance(par, nu)
        choice, _ = paths.viterbi(cost, distance, switch=switch, drift=drift)
        level_edges = [[EdgeState(states[choice[t, e]], float(par[t, choice[t, e], e]), float(nu[t, choice[t, e], e]),
                                  loglik=float(loglik[t, choice[t, e], e])) for t in range(steps)]
                       for e in range(n_edges)]
        components = enumerate_components(level, max_order, shapes) if level >= 1 else []
        if components:
            if allowed is None:
                z = np.stack([_edge_statistics(v[t][:, level], v[t][:, level + 1:], zeta[t][:, :level], components, rerank)
                              for t in range(steps)])                       # (T, C, P)
            for e in range(n_edges):
                if allowed is None:
                    runs = {c: scan.significant_intervals(z[:, c, e], screen) for c in range(len(components))}
                    runs = {c: r for c, r in runs.items() if r}
                    picked = sorted(runs, key=lambda c: -max(v for _, _, v in runs[c]))[:max_joint]
                    active = {c: np.zeros(steps, dtype=bool) for c in picked}   # windows inside significant runs
                    for c in picked:
                        for start, stop, _ in runs[c]:
                            active[c][start:stop] = True
                    record = [[list(r) for r in runs[c]] for c in picked]
                else:
                    carried = {c: tuple(sorted([order[m] for m in components[c][1]] + [order[level], order[level + 1 + e]]))
                               for c in range(len(components))}
                    picked = [c for c in range(len(components)) if carried[c] in allowed and allowed[carried[c]].any()][:max_joint]
                    active = {c: allowed[carried[c]] for c in picked}
                    record = [list(carried[c]) for c in picked]
                if not picked:
                    continue
                screened_log.append(dict(level=level, edge=e, components=[components[c] for c in picked], runs=record))
                candidates = [((components[c],), active[c]) for c in picked]
                if len(picked) > 1:
                    candidates.append((tuple(components[c] for c in picked), np.any([active[c] for c in picked], axis=0)))
                betas, extra_cost, extra_distance = [], [], []
                for candidate, mask in candidates:
                    rows = [fit_modulated_gaussian(v[t][:, level + 1 + e], v[t][:, level],
                                                   component_features(candidate, zeta[t][:, :level]), ridge, eps)
                            if mask[t] else (np.full(1 + len(candidate), np.nan), -np.inf, True) for t in range(steps)]
                    beta = np.stack([r[0] for r in rows])
                    ll = np.array([r[1] for r in rows])
                    betas.append((beta, ll, np.array([r[2] for r in rows])))
                    extra_cost.append(np.where(mask, (1.0 + len(candidate)) / counts - ll + (activation if serial is None else activation[e]),
                                               np.inf))
                    extra_distance.append(np.r_[0.0, np.nan_to_num(np.abs(np.diff(beta, axis=0)).mean(axis=1))])
                local = np.concatenate((cost[:, :, e], np.stack(extra_cost, axis=1)), axis=1)[:, :, None]
                dist = np.concatenate((distance[:, :, e], np.stack(extra_distance, axis=1)), axis=1)[:, :, None]
                extended, _ = paths.viterbi(local, dist, switch=switch if serial is None else switch[e], drift=drift)
                for t in range(steps):
                    s = int(extended[t, 0]) - len(states)
                    if s >= 0:
                        level_edges[e][t] = EdgeState('gaussian', components=candidates[s][0], beta=betas[s][0][t],
                                                      loglik=float(betas[s][1][t]), converged=bool(betas[s][2][t]))
                        failures += int(not betas[s][2][t])
        edges.append(level_edges)
        for t in range(steps):
            root = v[t][:, level].copy()
            for e in range(n_edges):
                edge = level_edges[e][t]
                if edge.state == 'ind' and not edge.components:
                    continue
                v[t][:, level + 1 + e] = fam.evaluate(edge.state, v[t][:, level + 1 + e], root,
                                                      edge_parameter(edge, zeta[t][:, :level]), edge.nu, eps)[1]
    return DynamicCVine(order, edges, dict(states=states, switch=base_switch, drift=drift, activation=base_activation,
                                           serial=None if serial is None else serial.tolist(),
                                           max_order=max_order, shapes=list(shapes), screen=screen, depth=depth,
                                           rerank=rerank, ridge=ridge, screened=screened_log,
                                           gated=allowed is not None, not_converged=failures))


def modulated_edges(model: DynamicCVine):
    """Summary of the modulated states selected in the fit: subset, order and the windows in which each is active."""
    found = {}
    for level, level_edges in enumerate(model.edges):
        for e, path in enumerate(level_edges):
            for t, edge in enumerate(path):
                for component in edge.components:
                    key = (model.subset(level, e, component[1]), component[0])
                    found.setdefault(key, []).append(t)
    return [dict(subset=k[0], shape=k[1], order=len(k[0]), windows=w) for k, w in sorted(found.items())]


LIBRARY_STATES = {'ind': 'ind', 'gaussian': 'gaussian', 'student': 'student', 'clayton': 'clayton0',
                  'gumbel': 'gumbel0', 'frank': 'frank', 'joe': 'joe0', 'claytonrot90': 'clayton90'}


def from_library(path, order) -> DynamicCVine:
    """Build the model from a fitted DVC-switch path saved by the benchmark (``model_switch_XX.json['path']``):
    one entry per window with ``copulas[level][edge] = {family, theta}``."""
    depth, steps = len(path[0]['copulas']), len(path)
    edges = []
    for level in range(depth):
        level_edges = []
        for e in range(len(path[0]['copulas'][level])):
            row = []
            for t in range(steps):
                cop = path[t]['copulas'][level][e]
                state, theta = LIBRARY_STATES[cop['family']], cop['theta']
                if state == 'ind' or theta is None:
                    row.append(EdgeState('ind'))
                elif state == 'student':
                    row.append(EdgeState(state, float(theta[0]), float(theta[1])))
                else:
                    row.append(EdgeState(state, float(theta[0] if isinstance(theta, (list, tuple)) else theta)))
            level_edges.append(row)
        edges.append(level_edges)
    return DynamicCVine(list(order), edges, dict(source='dvc_package SwitchingDynamicCVine'))
