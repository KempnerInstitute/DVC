"""Coefficient trajectories on the edges of one fixed vine with many variables.

The model proposes candidate interactions in the following steps.

1. One simplified switching vine in one root order (`cvine.fit` with ``max_order = 2``).
2. Screening.  The scan of all its edges (`cvine.anchored_scan`).  A component of an edge is a candidate term if its
   interval statistic reaches a lenient threshold (a per-component level, not a familywise one: the screening should
   keep true terms; what is kept is decided later).  At most ``max_terms`` candidates per edge, the strongest.
3. For every edge with candidates:
   - its two transforms are calibrated against the coordinates of the candidate modulators and of the variables on
     which its two variables depend in the vine (`trajectory.fit_calibration` with products; at most
     ``max_columns`` coordinates), with the gate on the validation windows (`Calibration.gated`);
   - joint coefficient trajectories are fitted with the candidate terms, penalties chosen on the validation windows
     (`tune`), and for comparison a baseline-only trajectory;
   - the terms are kept if at least one is switched on somewhere and the edge with them predicts the validation
     observations better than the baseline-only edge by ``keep_z`` standard errors of the mean difference of the
     log densities (a noise allowance: with a lenient screening most candidates are false).
4. For every kept edge the test of a change of its modulation, on confirmation windows that were not used for
   screening, fitting or tuning.  The test is `trajectory.change_test` on the sub-vine of the variables the edge
   depends on (`relevant`): the two variables of the edge, the modulators, the calibration coordinates and their
   parents in the vine.  Everything in that sub-vine is refitted on the confirmation windows and in every
   bootstrap draw.  The trees of the sub-vine are working trees with independence and Gaussian pairs only
   (``test_states``): refitting other families costs seconds per draw, and the calibration of the edge's
   transforms is what corrects for trees that are not right (Proposition 8; the test keeps its level with
   misspecified trees in E17h).  Redrawing only the edge omits uncertainty in the lower-tree transforms.

What a kept term means: in this root order, the dependence of the two variables of the edge given the earlier roots
varies with the coordinate of the modulator.  The variables of the term (pair and modulator) are its subset; whether
the subset is an interaction in every order is a separate question (the strict detector, `p2.detect`).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cvine, margins, scan, trajectory as tr
from . import families as fam

SMOOTH = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, np.inf)             # times the window size, as in E17
JUMP = (3.0, 6.0, 12.0, 24.0)


def screen(model: cvine.DynamicCVine, windows: Sequence[np.ndarray], max_order=3, level=1e-3, max_terms=3,
           shapes=scan.SHAPES):
    """Candidate terms from the scan of the fitted vine.  Returns (candidates, threshold, number of components);
    candidates: {(level, edge): [dict(component, strength, start, stop, subset)]}, strongest first."""
    records = cvine.anchored_scan(model, windows, max_order=max_order, shapes=shapes)
    z = np.array([r['z'] for r in records])
    strength, start, stop = scan.interval_scan(z.T)
    threshold = float(np.quantile(scan.interval_scan_null(len(windows)), 1.0 - level))
    out: Dict[Tuple[int, int], List[dict]] = {}
    for i, r in enumerate(records):
        if strength[i] >= threshold:
            out.setdefault((int(r['level']), int(r['edge'])), []).append(
                dict(component=r['component'], strength=float(strength[i]), start=int(start[i]), stop=int(stop[i]),
                     subset=tuple(int(v) for v in r['subset'])))
    return ({key: sorted(items, key=lambda c: -c['strength'])[:max_terms] for key, items in out.items()}, threshold,
            len(records))


def parents(model: cvine.DynamicCVine, position: int) -> List[Tuple[float, int]]:
    """(strength, m) for every earlier position m whose edge with the variable at ``position`` is not independence
    in some window; strength is the mean log-likelihood of that edge per window."""
    out = []
    for m in range(min(position, model.depth)):
        states = model.edges[m][position - m - 1]
        if any(s.state != 'ind' or s.components for s in states):
            out.append((float(np.mean([s.loglik for s in states])), m))
    return out


def calibration_columns(model: cvine.DynamicCVine, level: int, edge: int, components, max_columns=4) -> List[int]:
    """Coordinates for the calibration of an edge: those of its candidate modulators, then the strongest parents of
    its two variables among the conditioning coordinates."""
    columns: List[int] = []
    for _, members in components:
        for m in members:
            if m not in columns:
                columns.append(int(m))
    strength: Dict[int, float] = {}
    for position in (level, level + 1 + edge):
        for value, m in parents(model, position):
            if m < level:
                strength[m] = max(strength.get(m, -np.inf), value)
    for m in sorted(strength, key=lambda k: -strength[k]):
        if len(columns) >= max_columns:
            break
        if m not in columns:
            columns.append(m)
    return sorted(columns[:max(max_columns, len({m for _, members in components for m in members}))])


def relevant(model: cvine.DynamicCVine, level: int, edge: int, components, columns: Sequence[int], max_size=8) -> List[int]:
    """Positions of the earlier roots that the transforms and the terms of an edge depend on: the modulators, the
    calibration coordinates and, recursively, the parents of these and of the two variables of the edge.  At most
    ``max_size - 2`` positions; modulators and calibration coordinates first, then the strongest parents."""
    first = []
    for m in [m for _, members in components for m in members] + list(columns):
        if m not in first:
            first.append(int(m))
    strength: Dict[int, float] = {}
    queue = [level, level + 1 + edge] + first
    seen = set(first)
    while queue:
        position = queue.pop(0)
        for value, m in parents(model, position):
            if m < level and m not in seen:
                seen.add(m)
                strength[m] = value
                queue.append(m)
    rest = sorted(strength, key=lambda m: -strength[m])
    return sorted((first + rest)[:max(max_size - 2, len(first))])


def states_among(model: cvine.DynamicCVine, positions: Sequence[int]) -> List[str]:
    """The pair-copula states the fitted vine uses on the edges among these positions, with independence and the
    Gaussian family first."""
    used = []
    positions = sorted(positions)
    for i, a in enumerate(positions):
        for b in positions[i + 1:]:
            if a < model.depth:
                for state in {s.state for s in model.edges[a][b - a - 1] if not s.components}:
                    if state not in used and state not in ('ind', 'gaussian'):
                        used.append(state)
    return ['ind', 'gaussian'] + sorted(used)


def collect(model: cvine.DynamicCVine, windows: Sequence[np.ndarray], edges: Sequence[Tuple[int, int]], eps=fam.EPS):
    """The transforms of several edges in one pass per window.
    Returns ({(level, edge): (u list, v list)}, list of zeta (n, depth))."""
    out = {key: ([], []) for key in edges}
    zeta = []
    for t, data in enumerate(windows):
        result = cvine.forward(model, t, data, eps, keep=True)
        zeta.append(result['zeta'].copy())
        for level, edge in edges:
            pits = result['pits'][level]
            out[(level, edge)][0].append(pits[:, level + 1 + edge].copy())
            out[(level, edge)][1].append(pits[:, level].copy())
    return out, zeta


def tune(train: tr.Edge, validation: tr.Edge, k: int, n: float, smooth=SMOOTH, jump=JUMP):
    """Validation choice of the penalties of one edge: smoothness and jump price tied over the coefficients first,
    then for the modulation, then for the baseline.  Returns (trajectory, validation score)."""
    grid = [n * s for s in smooth]
    make = lambda sb, jb, sm, jm: tr.penalty(k, smooth=(sb, sm), jump=(jb, jm))
    best = tr.tune(train, validation, [make(s, j, s, j) for s in grid for j in jump])
    if k:
        sb, jb = best[0].penalty.smooth[0], best[0].penalty.jump[0]
        again = tr.tune(train, validation, [make(sb, jb, s, j) for s in grid for j in jump])
        best = again if again[1] > best[1] else best
        sm, jm = best[0].penalty.smooth[1], best[0].penalty.jump[1]
        again = tr.tune(train, validation, [make(s, j, sm, jm) for s in grid for j in jump])
        best = again if again[1] > best[1] else best
    return best[0], float(best[1])


def _fit_edge(payload):
    """Worker: calibration, tuning, decision and test for one edge.  All coordinates are those of ``needed``."""
    start = time.time()
    key, components, columns, u, v, zeta, n, replicates, seed = (payload[name] for name in (
        'key', 'components', 'columns', 'u', 'v', 'zeta', 'n', 'replicates', 'seed'))
    keep_z, confirm = payload['keep_z'], payload.get('confirm')
    roles = list(u)
    calibration = tr.fit_calibration(u['train'], v['train'], zeta['train'], degree=2, products=True, columns=columns)
    gate, gain = calibration.gated(u['validation'], v['validation'], zeta['validation'])
    inputs = {}
    for role in roles:
        pairs = [gate.apply(t, u[role][t], v[role][t], zeta[role][t]) for t in range(len(u[role]))]
        inputs[role] = ([a for a, _ in pairs], [b for _, b in pairs],
                        [cvine.component_features(components, z) for z in zeta[role]])
    without = lambda role: (inputs[role][0], inputs[role][1], [np.zeros((len(a), 0)) for a in inputs[role][0]])
    train = tr.edge_data(*inputs['train'])
    validation = tr.edge_data(*inputs['validation'], centre=train.means)
    plain = tr.edge_data(*without('validation'))
    base, base_score = tune(tr.edge_data(*without('train')), plain, 0, n)
    fitted, score = tune(train, validation, len(components), n)
    on = [bool(fitted.pieces(1 + k)) for k in range(len(components))]
    difference = (tr.log_density(fitted.theta, validation) - tr.log_density(base.theta, plain))[validation.w > 0]
    z = float(difference.mean() / max(difference.std(ddof=1) / np.sqrt(difference.size), 1e-12))
    kept = bool(any(on) and z >= keep_z)
    out = dict(key=key, kept=kept, on=on, score=score, base_score=base_score, z=z, calibration_kept=[bool(g > 0.0) for g in gain],
               calibration_gain=[float(g) for g in gain], pieces=[fitted.pieces(p) for p in range(1 + len(components))],
               coefficients=np.round(fitted.coefficients(), 5).tolist(), changes=bool(fitted.modulation_changes()),
               smooth=[float(s) if np.isfinite(s) else 'inf' for s in fitted.penalty.smooth], jump=list(fitted.penalty.jump))
    if 'test' in roles:
        test = tr.edge_data(*inputs['test'], centre=train.means)
        out['test_gain'] = float(tr.score(fitted.theta, test) - tr.score(base.theta, tr.edge_data(*without('test'))))
    if kept and replicates and confirm is not None:
        size = confirm['windows'][0].shape[1]

        def lower_fit(windows):
            return cvine.fit(list(windows), order=list(range(size)), states=confirm['states'], max_order=2, max_depth=size - 2,
                             switch=confirm['switch'])
        result = tr.change_test(confirm['windows'], lower_fit, confirm['components'], fitted.penalty, replicates=replicates,
                                rng=np.random.default_rng(seed), calibrate=2, columns=confirm['columns'], pooled=confirm.get('pooled'))
        out['test'] = dict(statistic=result['statistic'], p=result['p'], variables=confirm['variables'])
    out['seconds'] = round(time.time() - start, 2)
    return out


def fit(train: Sequence[np.ndarray], validation: Sequence[np.ndarray], test: Optional[Sequence[np.ndarray]] = None,
        order: Optional[Sequence[int]] = None, states: Optional[Sequence[str]] = None, model: Optional[cvine.DynamicCVine] = None,
        workers=1, max_order=3, screen_level=1e-3, max_terms=3, max_columns=4, replicates=99, seed=0, switch=0.05,
        pool=None, confirm: Optional[Sequence[np.ndarray]] = None, keep_z=2.0, max_size=8, test_states=('ind', 'gaussian'),
        pooled: Optional[Sequence[bool]] = None, modulators_first=False):
    """The pipeline of this module.  Returns dict(order, model, threshold, components, edges, seconds); every entry
    of ``edges`` describes one screened edge: its pair, candidate terms with their subsets, whether the terms are
    kept, pieces, coefficients, validation and test gains per observation, and the local change test.
    ``pool``: an executor used for the windows of the vine fit and for the edges (otherwise ``workers`` processes
    are started for the edges and the vine is fitted in this process).
    ``confirm``: windows that are used for nothing but the change test of the kept edges (same variables and
    number of windows; any monotone transform of the margins).  Without them no change test is run.
    ``test_states``: pair-copula states of the working trees of the test's sub-vine; None = the states the fitted
    vine uses among the sub-vine's variables (`states_among`; slow).
    ``pooled``: per variable (column of the data), the flag of `margins.choose` that says how its margin was
    estimated when ``train``, ``validation`` and ``test`` were made (over all windows, aligned, or within every
    window).  The confirmation windows and the simulated datasets of the change test then get their
    pseudo-observations the same way (`margins.ranks`); by default from ranks within every window.
    ``modulators_first``: in the sub-vine of the change test the modulators of the edge come first (strongest
    term first), before the other variables the edge depends on.  The coordinate of the first one is then its own
    normal score, so a modulation by that variable is one term of the tested model and no part of it is left out
    (omitting modulation can inflate rejection of constant coefficients). Default: the
    order of the vine, as in E18."""
    clock = {}
    start = time.time()
    if model is None:
        order = cvine.infer_order(list(train)) if order is None else list(order)
        clock['order'] = time.time() - start
        start = time.time()
        model = cvine.fit(list(train), order=order, states=states, max_order=2, switch=switch, pool=pool)
        clock['vine'] = time.time() - start
    order = list(model.order)
    start = time.time()
    candidates, threshold, count = screen(model, list(train), max_order, screen_level, max_terms)
    clock['screen'] = time.time() - start
    start = time.time()
    keys = sorted(candidates)
    roles = dict(train=list(train), validation=list(validation))
    if test is not None:
        roles['test'] = list(test)
    collected = {role: collect(model, windows, keys) for role, windows in roles.items()}
    if pooled is not None and len(pooled) != len(order):
        raise ValueError('pooled needs one flag per variable')
    confirmation = margins.ranks(confirm, pooled) if confirm is not None and replicates else None
    payloads = []
    for index, (level, edge) in enumerate(keys):
        components = [c['component'] for c in candidates[(level, edge)]]
        columns = calibration_columns(model, level, edge, components, max_columns)
        needed = sorted(set(columns) | {m for _, members in components for m in members})
        place = {m: i for i, m in enumerate(needed)}
        extra = None
        if confirm is not None and replicates:
            roots = relevant(model, level, edge, components, columns, max_size)
            if modulators_first:
                lead = []
                for _, members in components:
                    lead += [int(m) for m in members if m not in lead]
                roots = lead + [m for m in roots if m not in lead]
            inside = {m: i for i, m in enumerate(roots)}
            variables = [order[m] for m in roots] + [order[level], order[level + 1 + edge]]
            extra = dict(windows=[w[:, variables] for w in confirmation], variables=variables,
                         pooled=None if pooled is None else [pooled[v] for v in variables],
                         components=[(shape, tuple(inside[m] for m in members)) for shape, members in components],
                         columns=[inside[m] for m in columns if m in inside],
                         states=list(test_states) if test_states is not None else states_among(model, roots + [level, level + 1 + edge]),
                         switch=switch)
        payloads.append(dict(
            key=(level, edge), components=[(shape, tuple(place[m] for m in members)) for shape, members in components],
            columns=[place[m] for m in columns], n=float(np.mean([len(w) for w in roles['train']])), replicates=replicates,
            seed=[int(seed), level, edge], keep_z=keep_z, confirm=extra,
            u={role: collected[role][0][(level, edge)][0] for role in roles}, v={role: collected[role][0][(level, edge)][1] for role in roles},
            zeta={role: [z[:, needed] for z in collected[role][1]] for role in roles}))
    clock['collect'] = time.time() - start
    start = time.time()
    if pool is not None:
        fitted = list(pool.map(_fit_edge, payloads, chunksize=1))
    elif workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as own:
            fitted = list(own.map(_fit_edge, payloads, chunksize=1))
    else:
        fitted = [_fit_edge(p) for p in payloads]
    clock['edges'] = time.time() - start
    edges = []
    for result in fitted:
        level, edge = result['key']
        items = candidates[(level, edge)]
        edges.append(dict(result, level=level, edge=edge, pair=(order[level], order[level + 1 + edge]),
                          components=[[c['component'][0], [int(m) for m in c['component'][1]]] for c in items],
                          modulators=[[order[m] for m in c['component'][1]] for c in items],
                          subsets=[list(c['subset']) for c in items], strength=[c['strength'] for c in items],
                          interval=[[c['start'], c['stop']] for c in items],
                          calibration_columns=[order[m] for m in calibration_columns(model, level, edge, [c['component'] for c in items], max_columns)]))
    return dict(order=order, model=model, threshold=threshold, components=count, edges=edges,
                seconds={k: round(v, 1) for k, v in clock.items()})
