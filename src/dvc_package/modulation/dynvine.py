"""Assemble and evaluate one dynamic C-vine in a fixed root order.

`vinetraj.fit` fits a simplified vine, computes its conditional transforms and fits trajectory models of single
edges on them.  It returns the simplified vine and descriptions of those edges: not one model.  The sum of the
held-out gains of the edges is then not the held-out gain of a density, because an edge that changes also changes
the conditional transforms that every higher tree is built on.

`DynamicVine` is one model: a C-vine in a fixed root order whose edges are either the simplified states of a fitted
`cvine.DynamicCVine` or trajectory edges (a Gaussian pair copula whose Fisher-z correlation is a baseline plus
modulation terms, with one coefficient vector per window), and whose conditional transforms may be calibrated.
`forward` evaluates its log density and passes every changed transform on to the next tree; `simulate` is the
inverse map, so the density and the sampler belong to the same model.

Conditional-transform calibration.  A calibration belongs to one variable at one level: the transform of that
variable given the earlier roots, as the trees below compute it, is replaced by its conditional distribution
function given the coordinates under a Gaussian regression of its normal score (mean: a polynomial of selected
coordinates; one residual standard deviation per window).  Its log Jacobian is part of the density.  The root of a
level has one calibration for the whole level; a later variable has one for its edge at that level.  This is what
makes several calibrated edges one density: `trajectory.Calibration` calibrates the root's transform edge by edge,
which two edges of the same level would do differently.

`assemble` builds the model from a fitted simplified vine and the edges that `vinetraj.fit` kept.  It walks through
the levels once; when it reaches a kept edge it fits that edge on the transforms of the model assembled so far,
installs it and moves on, so every kept edge is fitted on the inputs it has in the final model.  The other edges
keep their simplified states and are evaluated on the propagated transforms; they are not refitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr, ndtri

from . import cvine, trajectory as tr
from . import families as fam
from .cvine import ETA_MAX


@dataclass
class TrajectoryEdge:
    """A Gaussian edge with Fisher-z correlation beta[t, 0] + features(components) @ beta[t, 1:] in window t
    (the convention of `cvine.EdgeState.beta`; features are not centred)."""
    components: Tuple[cvine.Component, ...]
    beta: np.ndarray


@dataclass
class TransformCalibration:
    """Calibration of one variable's transform at one level: per window the mean coefficients (T, P) of its normal
    score on `trajectory.calibration_basis` of the coordinates ``columns``, and the residual scale (T,)."""
    columns: Tuple[int, ...]
    beta: np.ndarray
    scale: np.ndarray
    degree: int = 2
    products: bool = True

    def basis(self, zeta):
        return tr.calibration_basis(np.asarray(zeta)[:, list(self.columns)], self.degree, self.products)

    def apply(self, t, x, zeta, eps=fam.EPS):
        """The calibrated transform and the log Jacobian, per observation."""
        raw = ndtri(fam.clip01(np.asarray(x, dtype=np.float64), eps))
        new = (raw - self.basis(zeta) @ self.beta[t]) / self.scale[t]
        return ndtr(new), 0.5 * raw * raw - 0.5 * new * new - np.log(self.scale[t])

    def invert(self, t, x, zeta, eps=fam.EPS):
        """The computed transform that has the calibrated transform x."""
        new = ndtri(fam.clip01(np.asarray(x, dtype=np.float64), eps))
        return ndtr(self.basis(zeta) @ self.beta[t] + self.scale[t] * new)


def fit_transform_calibration(scores: Sequence[np.ndarray], zeta: Sequence[np.ndarray], columns: Sequence[int], degree=2,
                              products=True) -> TransformCalibration:
    """Least squares, window by window, of the normal score of a transform on the calibration basis.  A window with
    fewer than three observations per basis function keeps the identity."""
    columns = tuple(int(c) for c in columns)
    width = tr.calibration_basis(np.zeros((1, len(columns))), degree, products).shape[1]
    beta, scale = np.zeros((len(scores), width)), np.ones(len(scores))
    for t, raw in enumerate(scores):
        design = tr.calibration_basis(np.asarray(zeta[t])[:, list(columns)], degree, products)
        if design.shape[0] < 3 * width:
            continue
        coefficients = np.linalg.lstsq(design, raw, rcond=None)[0]
        residual = raw - design @ coefficients
        beta[t], scale[t] = coefficients, np.sqrt(max(float((residual * residual).mean()), 1e-12))
    return TransformCalibration(columns, beta, scale, int(degree), bool(products))


class DynamicVine:
    """See the module header.  ``base``: a fitted `cvine.DynamicCVine` whose states are used wherever no trajectory
    edge is installed.  ``trajectories``: {(level, position): TrajectoryEdge}, position > level in vine order.
    ``calibrations``: {(level, position): TransformCalibration}, position >= level and level >= 1."""

    def __init__(self, base: cvine.DynamicCVine, trajectories: Optional[Dict[Tuple[int, int], TrajectoryEdge]] = None,
                 calibrations: Optional[Dict[Tuple[int, int], TransformCalibration]] = None):
        self.base = base
        self.order = list(base.order)
        self.trajectories = dict(trajectories or {})
        self.calibrations = dict(calibrations or {})
        for (level, position), cal in self.calibrations.items():
            if level < 1 or position < level or any(c >= level for c in cal.columns):
                raise ValueError('a calibration needs level >= 1, position >= level and coordinates of earlier roots')
        for level, position in self.trajectories:
            if not 0 <= level < position < self.d or level >= self.depth:
                raise ValueError('a trajectory edge joins the root of a level with a later position')

    @property
    def d(self):
        return self.base.d

    @property
    def depth(self):
        return self.base.depth

    @property
    def steps(self):
        return self.base.steps

    def level(self, t: int, level: int, v: np.ndarray, zeta: np.ndarray, own: np.ndarray, eps=fam.EPS) -> np.ndarray:
        """Apply one level in place: v (n, d) holds the transforms at the start of the level and receives those
        at the start of the next; zeta[:, level] receives the level's coordinate.  Returns the log density of the
        level per observation (pair copulas and calibrations)."""
        logc = np.zeros(v.shape[0])
        root = v[:, level]
        cal = self.calibrations.get((level, level))
        if cal is not None:
            root, jacobian = cal.apply(t, root, zeta, eps)
            logc += jacobian
            v[:, level] = root
        zeta[:, level] = ndtri(fam.clip01(root, eps))
        for position in range(level + 1, self.d):
            x = v[:, position]
            cal = self.calibrations.get((level, position))
            if cal is not None:
                x, jacobian = cal.apply(t, x, zeta, eps)
                logc += jacobian
            edge = self.trajectories.get((level, position))
            if edge is not None:
                eta = edge.beta[t, 0] + tr.component_features(edge.components, zeta[:, :level], own[:, :level]) @ edge.beta[t, 1:]
                lc, h = fam.evaluate('gaussian', x, root, np.tanh(np.clip(eta, -ETA_MAX, ETA_MAX)), None, eps)
            else:
                state = self.base.edges[level][position - level - 1][t]
                if state.state == 'ind' and not state.components:
                    v[:, position] = x
                    continue
                lc, h = fam.evaluate(state.state, x, root, cvine.edge_parameter(state, zeta[:, :level]), state.nu, eps)
            logc += lc
            v[:, position] = h
        return logc

    def start(self, u, eps=fam.EPS):
        """Transforms at the start of level 0 (the pseudo-observations in vine order), empty coordinates and the
        variables' own normal scores."""
        v = fam.clip01(np.asarray(u, dtype=np.float64)[:, self.order], eps)
        return v, np.empty((v.shape[0], self.depth)), ndtri(v)

    def forward(self, t: int, u, eps=fam.EPS) -> dict:
        """Log density per level (n, depth) and the coordinates (n, depth) of the observations u (n, d), given in
        the original column order, in window t."""
        v, zeta, own = self.start(u, eps)
        logc = np.zeros((v.shape[0], self.depth))
        for level in range(self.depth):
            logc[:, level] = self.level(t, level, v, zeta, own, eps)
        return dict(logc=logc, zeta=zeta)

    def log_density(self, windows: Sequence[np.ndarray], eps=fam.EPS) -> List[np.ndarray]:
        return [self.forward(t, u, eps)['logc'].sum(axis=1) for t, u in enumerate(windows)]

    def score(self, windows: Sequence[np.ndarray], eps=fam.EPS) -> float:
        """Mean log density per observation (windows weighted by their size)."""
        return float(np.concatenate(self.log_density(windows, eps)).mean())

    def simulate(self, t: int, n: int, rng, innovations=None, eps=fam.EPS) -> np.ndarray:
        """n observations of window t: the inverse of `forward`.  ``innovations`` (n, d): the uniform coordinates
        in vine order (by default independent uniforms); `forward` returns their normal scores as ``zeta``."""
        d = self.d
        w = rng.uniform(size=(n, d)) if innovations is None else np.asarray(innovations, dtype=np.float64)
        zeta = ndtri(fam.clip01(w, eps))
        out, own = np.empty((n, d)), np.empty((n, d))
        for position in range(d):
            x = w[:, position]
            if position < self.depth:
                cal = self.calibrations.get((position, position))
                if cal is not None:
                    x = cal.invert(t, x, zeta, eps)
            for level in range(min(position, self.depth) - 1, -1, -1):
                root = w[:, level]
                edge = self.trajectories.get((level, position))
                if edge is not None:
                    eta = edge.beta[t, 0] + tr.component_features(edge.components, zeta[:, :level], own[:, :level]) @ edge.beta[t, 1:]
                    x = fam.h_inverse('gaussian', x, root, np.tanh(np.clip(eta, -ETA_MAX, ETA_MAX)))
                else:
                    state = self.base.edges[level][position - level - 1][t]
                    if not (state.state == 'ind' and not state.components):
                        x = fam.h_inverse(state.state, x, root, cvine.edge_parameter(state, zeta[:, :level]), state.nu)
                cal = self.calibrations.get((level, position))
                if cal is not None:
                    x = cal.invert(t, x, zeta, eps)
            out[:, self.order[position]] = x
            own[:, position] = ndtri(fam.clip01(x, eps))
        return out


# ----------------------------------------------------------------------------------------------------------------
# Assembly from a fitted simplified vine and the edges kept by `vinetraj.fit`.
# ----------------------------------------------------------------------------------------------------------------
def _gain(cal: TransformCalibration, scores: Sequence[np.ndarray], zeta: Sequence[np.ndarray]) -> float:
    """Mean log density per observation that the calibration adds to a transform on these windows."""
    total = 0.0
    for t, raw in enumerate(scores):
        new = (raw - cal.basis(zeta[t]) @ cal.beta[t]) / cal.scale[t]
        total += float((0.5 * raw * raw - 0.5 * new * new - np.log(cal.scale[t])).mean())
    return total / max(len(scores), 1)


def standard_score(values: Sequence[np.ndarray], block=1) -> float:
    """Mean of the values over its standard error.  values: one array per window, in the order of observation; with
    ``block`` > 1 the units are the means of consecutive blocks of that many observations of a window (for
    serially dependent observations), and a remainder shorter than a block is left out."""
    if block > 1:
        units = [np.asarray(v, dtype=np.float64)[:(len(v) // block) * block].reshape(-1, block).mean(axis=1) for v in values]
    else:
        units = [np.asarray(v, dtype=np.float64) for v in values]
    units = np.concatenate(units) if units else np.zeros(0)
    if units.size < 2:
        return 0.0
    return float(units.mean() / max(units.std(ddof=1) / np.sqrt(units.size), 1e-12))


def assemble(base: cvine.DynamicCVine, train: Sequence[np.ndarray], validation: Sequence[np.ndarray], kept: Sequence[dict],
             root_columns: Optional[Dict[int, Sequence[int]]] = None, tune=None, install='always', eps=fam.EPS,
             margin=0.0, block=1):
    """Build one `DynamicVine` from the simplified vine ``base`` and the kept edges.

    kept: one dict per edge with ``level``, ``edge`` (index of the later variable at that level, as in
    `vinetraj.fit`), ``components``, ``penalty`` (a `trajectory.Penalty`, used as it is unless ``tune`` is given)
    and optionally ``columns`` (coordinates for the calibration of the later variable's transform).
    root_columns: {level: coordinates} for levels whose root transform may be calibrated.
    tune: a function (train edge, validation edge, number of components) -> trajectory, to choose the penalties
    again on the assembled transforms (for example `lambda a, b, k: vinetraj.tune(a, b, k, n)[0]`).
    A calibration is kept only if it raises the log density of its transform on the validation windows.
    install: 'always' installs every kept edge; 'gain' installs a kept edge only if, on the validation windows, the
    trajectory edge (with the Jacobian of its calibration) has a higher mean log density than the state that the
    simplified vine has for that edge, on the same inputs.  The pipeline keeps an edge when its modulated Gaussian
    predicts better than a Gaussian with the baseline alone; where the simplified vine uses another family for the
    pair, that is not the comparison that decides whether the density improves.
    margin, block: with install='gain' and ``margin`` > 0 the edge is installed only if that gain is at least
    ``margin`` standard errors (`standard_score` of the differences per observation, in blocks of ``block``
    observations, or of the edge's own ``block`` if its dict has one); the default 0 asks only for a positive
    mean gain.
    Returns (vine, report); report has one entry per kept edge: its key, which terms are on, pieces, whether the
    calibration was kept, the validation gain of the edge over its baseline-only fit on the same transforms
    (``validation_gain``), the validation gain over the simplified vine's state (``install_gain``) and its
    standard score (``install_z``), the most frequent state of the simplified vine on that edge (``state``) and
    whether the edge was installed.
    """
    if install not in ('always', 'gain'):
        raise ValueError("install must be 'always' or 'gain'")
    steps = len(train)
    wanted = {(int(k['level']), int(k['level']) + 1 + int(k['edge'])): k for k in kept}
    root_columns = {int(level): tuple(columns) for level, columns in (root_columns or {}).items()}
    vine = DynamicVine(base)
    roles = dict(train=[vine.start(u, eps) for u in train], validation=[vine.start(u, eps) for u in validation])
    report = []
    for level in range(vine.depth):
        if level in root_columns and level >= 1 and root_columns[level]:
            scores = {role: [ndtri(fam.clip01(v[:, level], eps)) for v, _, _ in data] for role, data in roles.items()}
            zetas = {role: [z for _, z, _ in data] for role, data in roles.items()}
            cal = fit_transform_calibration(scores['train'], zetas['train'], root_columns[level])
            gain = _gain(cal, scores['validation'], zetas['validation'])
            if gain > 0.0:
                vine.calibrations[(level, level)] = cal
            report.append(dict(key=(level, level), root_calibration=bool(gain > 0.0), calibration_gain=gain))
        for (lvl, position), spec in sorted(wanted.items()):
            if lvl != level:
                continue
            components = tuple((shape, tuple(int(m) for m in members)) for shape, members in spec['components'])
            # inputs of the edge in the model assembled so far: the root (after its calibration), the later variable
            inputs = {}
            for role, data in roles.items():
                rows = []
                for t, (v, zeta, own) in enumerate(data):
                    root = v[:, level]
                    cal = vine.calibrations.get((level, level))
                    if cal is not None:
                        root = cal.apply(t, root, zeta, eps)[0]
                    rows.append((v[:, position].copy(), root.copy(), zeta, own))
                inputs[role] = rows
            entry = dict(key=(level, position), calibration=False, calibration_gain=0.0)
            if spec.get('columns'):
                scores = {role: [ndtri(fam.clip01(x, eps)) for x, _, _, _ in rows] for role, rows in inputs.items()}
                zetas = {role: [z for _, _, z, _ in rows] for role, rows in inputs.items()}
                cal = fit_transform_calibration(scores['train'], zetas['train'], spec['columns'])
                entry['calibration_gain'] = _gain(cal, scores['validation'], zetas['validation'])
                if entry['calibration_gain'] > 0.0:
                    vine.calibrations[(level, position)] = cal
                    entry['calibration'] = True
            edges = {}
            for role, rows in inputs.items():
                later, roots, features = [], [], []
                for t, (x, root, zeta, own) in enumerate(rows):
                    cal = vine.calibrations.get((level, position))
                    later.append(cal.apply(t, x, zeta, eps)[0] if cal is not None else x)
                    roots.append(root)
                    features.append(tr.component_features(components, zeta[:, :level], own[:, :level]))
                edges[role] = (later, roots, features)
            fit_edge = tr.edge_data(*edges['train'])
            check = tr.edge_data(*edges['validation'], centre=fit_edge.means)
            if tune is not None:
                fitted = tune(fit_edge, check, len(components))
            else:
                fitted = tr.fit(fit_edge, spec['penalty'])
            plain = tr.fit(tr.edge_data(edges['train'][0], edges['train'][1], [np.zeros((len(a), 0)) for a in edges['train'][0]]),
                           tr.penalty(0, smooth=fitted.penalty.smooth[0], jump=fitted.penalty.jump[0], order=fitted.penalty.order[0]))
            plain_check = tr.edge_data(edges['validation'][0], edges['validation'][1], [np.zeros((len(a), 0)) for a in edges['validation'][0]])
            candidate = TrajectoryEdge(components, fitted.coefficients())
            # the same observations under the trajectory edge and under the state of the simplified vine
            states = base.edges[level][position - level - 1]
            gain, count, differences = 0.0, 0, []
            for t, (x, root, zeta, own) in enumerate(inputs['validation']):
                cal = vine.calibrations.get((level, position))
                later, jacobian = cal.apply(t, x, zeta, eps) if cal is not None else (x, 0.0)
                eta = candidate.beta[t, 0] + tr.component_features(components, zeta[:, :level], own[:, :level]) @ candidate.beta[t, 1:]
                new_value = fam.evaluate('gaussian', later, root, np.tanh(np.clip(eta, -ETA_MAX, ETA_MAX)), None, eps)[0] + jacobian
                state = states[t]
                old_value = 0.0 if state.state == 'ind' and not state.components else \
                    fam.evaluate(state.state, x, root, cvine.edge_parameter(state, zeta[:, :level]), state.nu, eps)[0]
                gain += float(np.sum(new_value - old_value))
                count += len(x)
                differences.append(np.asarray(new_value - old_value, dtype=np.float64) * np.ones(len(x)))
            names = [state.state for state in states]
            entry.update(on=[bool(fitted.pieces(1 + k)) for k in range(len(components))],
                         pieces=[fitted.pieces(p) for p in range(1 + len(components))],
                         validation_gain=float(tr.score(fitted.theta, check) - tr.score(plain.theta, plain_check)),
                         install_gain=gain / max(count, 1), install_z=standard_score(differences, int(spec.get('block', block))),
                         state=max(set(names), key=names.count), stationarity=tr.stationarity(fit_edge, fitted),
                         coefficients=candidate.beta.copy())
            entry['installed'] = bool(install == 'always' or (entry['install_gain'] > 0.0 and entry['install_z'] >= margin))
            if entry['installed']:
                vine.trajectories[(level, position)] = candidate
            else:
                vine.calibrations.pop((level, position), None)
                entry['calibration'] = False
            report.append(entry)
        for data in roles.values():
            for t, (v, zeta, own) in enumerate(data):
                vine.level(t, level, v, zeta, own, eps)
    return vine, report
