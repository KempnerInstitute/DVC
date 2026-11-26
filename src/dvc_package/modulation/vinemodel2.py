"""Fit and test modulated edges in a dynamic C-vine.

Each edge starts with the state of a simplified switching C-vine. Screening proposes conditioning features.
The fit chooses trajectories, penalties and the feature representation using validation observations. Edge
updates propagate their conditional transforms to higher trees. The assembled density retains copula margins.
Gaussian working sub-vines used in the change tests can calibrate their transforms separately.

An edge must beat its simplified state by `install_z` standard errors. Where confirmation windows are
supplied, the assembly repeats this decision on those windows and checks the whole vine against the
simplified background. Without confirmation windows, the whole-model check uses the validation observations.

For each retained term the half-sample procedure profiles its coefficient under constancy while allowing
other terms to vary. It tests departures beyond `TOLERANCE` times the common coefficient. Terms on
overlapping pairs can enter joint models so that changes in a four-variable term need not be attributed
to a three-variable term on the same pair. Serial correction uses the two variables' autocorrelations for
each pair and the variable's own block length for its marginal fit.

The vine's fitted orientation always enters the test. An additional orientation enters when its fitted
trajectory activates the term and improves validation prediction by `SHOW_Z` standard errors. A change
requires agreement across the supported orientations and the relevant joint models. A retained coefficient
describes modulation of the conditioned pair in the selected root order.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import ndtri

from . import cvine, dynvine, halfsample, margins, scan, trajectory as tr, vinemodel, vinetraj
from . import families as fam

FLOOR = 0.1                                                          # autocorrelations below this count as none (`halfsample.block_length`)
TOLERANCE = 0.05                                                     # the tests look for a coefficient that leaves a band of this share of its common value on each side
SHOW_Z = 2.0                                                         # an orientation shows a term if the term adds this many standard errors to its validation score


def serial_profile(train: Sequence[np.ndarray], serial='auto'):
    """Lag-one autocorrelation of every variable (zero below `FLOOR`, at most 0.99) and its block length.
    serial: 'auto' (from the training windows, rows in the order of observation), None (independent observations)
    or one value per variable."""
    d = np.asarray(train[0]).shape[1]
    smallest = min(len(w) for w in train)
    if serial is None or serial is False:
        r = np.zeros(d)
    elif isinstance(serial, str):
        if serial != 'auto':
            raise ValueError("serial must be 'auto', None or one autocorrelation per variable")
        r = halfsample.autocorrelations(margins.window(list(train))['train'])
    else:
        r = np.asarray(serial, dtype=np.float64)
        if r.shape != (d,):
            raise ValueError('one autocorrelation per variable is needed')
    r = np.where(r >= FLOOR, np.minimum(r, 0.99), 0.0)
    return r, [halfsample.block_of(float(v), smallest) for v in r]


def choose_margins(train, others, level, align, blocks):
    """`margins.choose` with every variable's tests thinned by its own block.  Returns (data by role, kinds)."""
    if len(set(blocks)) == 1:
        data, flags = margins.choose(list(train), others, level, align=align, block=blocks[0])
        return data, [margins._kind(flag) for flag in flags]
    d = len(blocks)
    kinds: List[Optional[str]] = [None] * d
    for block in sorted(set(blocks)):
        columns = [j for j in range(d) if blocks[j] == block]
        part = [np.asarray(x, dtype=np.float64)[:, columns] for x in train]
        share = level * len(columns) / d                             # the tests divide their level by the number of columns they get
        keep = margins.stationary(part, share, block)
        shape = margins.same_shape(part, share, block) if align and not all(keep) else keep
        for j, k, s in zip(columns, keep, shape):
            kinds[j] = 'pooled' if k else ('aligned' if align and s else 'window')
    return margins._mixture(kinds, list(train), others), kinds


def screen(model: cvine.DynamicCVine, windows: Sequence[np.ndarray], max_order=3, level=1e-3, max_terms=3, shapes=scan.SHAPES,
           eps=fam.EPS, lags=None):
    """`vinemodel.screen` with the lags of the long-run variance per edge: ``lags(level, edge)`` -> int (None: no
    lags anywhere).  Edges of one level with the same lags are computed together; with one value for all edges
    this is `vinemodel.screen`."""
    if lags is None:
        return vinemodel.screen(model, windows, max_order, level, max_terms, shapes, eps, 0)
    steps = len(windows)
    components = {lvl: cvine.enumerate_components(lvl, max_order, shapes) for lvl in range(1, model.depth)}
    width = model.d if hasattr(model, 'd') else len(model.order)
    edge_lags = {lvl: np.array([int(lags(lvl, e)) for e in range(width - lvl - 1)]) for lvl in components}
    z = {}
    for t, u in enumerate(windows):
        out = cvine.forward(model, t, u, eps, keep=True)
        for lvl, parts in components.items():
            if not parts:
                continue
            v = out['pits'][lvl]
            later = v[:, lvl + 1:]
            values = np.empty((len(parts), later.shape[1]))
            for value in np.unique(edge_lags[lvl]):
                columns = np.nonzero(edge_lags[lvl] == value)[0]
                values[:, columns] = cvine._edge_statistics(v[:, lvl], later[:, columns], out['zeta'][:, :lvl], parts, True, int(value), 0)
            if lvl not in z:
                z[lvl] = np.full((steps,) + values.shape, np.nan)
            z[lvl][t] = values
    threshold = float(np.quantile(scan.interval_scan_null(steps), 1.0 - level))
    found: Dict = {}
    count = 0
    for lvl, values in z.items():
        many, edges = values.shape[1:]
        count += many * edges
        strength, start, stop = scan.interval_scan(values.reshape(steps, many * edges))
        for index in np.nonzero(strength >= threshold)[0]:
            c, e = divmod(int(index), edges)
            component = components[lvl][c]
            found.setdefault((lvl, e), []).append(dict(component=component, strength=float(strength[index]), start=int(start[index]),
                                                       stop=int(stop[index]), subset=tuple(int(x) for x in model.subset(lvl, e, component[1]))))
    return {key: sorted(items, key=lambda item: -item['strength'])[:max_terms] for key, items in found.items()}, threshold, count


def _edge(payload):
    """Worker for one edge: `vinemodel._edge` without a calibration of the two transforms."""
    start = time.time()
    key, u, v, zeta, own, n = (payload[name] for name in ('key', 'u', 'v', 'zeta', 'own', 'n'))
    without = lambda role: (u[role], v[role], [np.zeros((len(a), 0)) for a in u[role]])
    plain = tr.edge_data(*without('validation'))
    base, base_score = vinetraj.tune(tr.edge_data(*without('train')), plain, 0, n)
    best, tried = None, {}
    for name, components in payload['variants']:
        features = {role: [tr.component_features(components, z, o) for z, o in zip(zeta[role], own[role])] for role in u}
        train = tr.edge_data(u['train'], v['train'], features['train'])
        validation = tr.edge_data(u['validation'], v['validation'], features['validation'], centre=train.means)
        fitted, score = vinetraj.tune(train, validation, len(components), n)
        tried[name] = score
        if best is None or score > best['score'] + 1e-12:
            best = dict(name=name, components=components, features=features, train=train, validation=validation, fitted=fitted, score=score)
    fitted, train, validation, components = best['fitted'], best['train'], best['validation'], best['components']
    on = [bool(fitted.pieces(1 + k)) for k in range(len(components))]
    difference = tr.log_density(fitted.theta, validation) - tr.log_density(base.theta, plain)
    z = dynvine.standard_score([row[w > 0] for row, w in zip(difference, validation.w)], payload['block'])
    out = dict(key=key, kept=bool(any(on) and z >= payload['keep_z']), on=on, score=best['score'], base_score=base_score, z=z,
               features=best['name'], variant_scores=tried, pieces=[fitted.pieces(p) for p in range(1 + len(components))],
               penalty=fitted.penalty, stationarity=tr.stationarity(train, fitted))
    if 'test' in u:
        test = tr.edge_data(u['test'], v['test'], best['features']['test'], centre=train.means)
        out['test_gain'] = float(tr.score(fitted.theta, test) - tr.score(base.theta, tr.edge_data(*without('test'))))
    out['seconds'] = round(time.time() - start, 2)
    return out


def _orientation(payload):
    """Worker for one test: `vinemodel._orientation` with the estimate of the test as an option (``estimate``), and
    with what the terms add to the validation score in this description: ``on`` (the fitted trajectory has a term
    switched on in some window) and the standard score ``z`` of the gain.  With every term off the gain is that
    of two fits of the same baseline, of the size of the solver's tolerance, and its standard score means nothing."""
    start = time.time()
    size = len(payload['variables'])

    def lower_fit(windows):
        return cvine.fit(list(windows), order=list(range(size)), states=payload['states'], max_order=2, max_depth=size - 2,
                         switch=payload['switch'])
    train, validation = payload['train'], payload['validation']
    lower = lower_fit(train)
    n = float(np.mean([len(w) for w in train]))
    best, base = None, None
    for name, components in payload['variants']:
        inputs, calibration = {}, None
        for role, windows in (('train', train), ('validation', validation)):
            inputs[role], calibration = tr.edge_inputs_calibrated(lower, windows, size - 2, 0, components, calibration, degree=2,
                                                                   columns=payload['columns'])
        edge = tr.edge_data(*inputs['train'])
        held = tr.edge_data(*inputs['validation'], centre=edge.means)
        fitted, score = vinetraj.tune(edge, held, len(components), n)
        if base is None:
            empty = lambda role: tr.edge_data(inputs[role][0], inputs[role][1], [np.zeros((len(a), 0)) for a in inputs[role][0]])
            plain = empty('validation')
            without, base = vinetraj.tune(empty('train'), plain, 0, n)
            reference = tr.log_density(without.theta, plain)
        if best is None or score > best[1] + 1e-12:
            best = (name, score, components, tr.log_density(fitted.theta, held) - reference, held.w,
                    any(bool(fitted.pieces(1 + k)) for k in range(len(components))))
    name, score, components, difference, weights, on = best
    z = dynvine.standard_score([row[w > 0] for row, w in zip(difference, weights)], payload['block'])
    result = halfsample.change_test(payload['confirm'], lower_fit, components, replicates=payload['replicates'],
                                    rng=np.random.default_rng(payload['seed']), calibrate=2, columns=payload['columns'],
                                    pooled=payload['pooled'], block=payload['block'], raw=payload['raw'], keep=payload['keep'],
                                    estimate=payload['estimate'], tolerance=payload['tolerance'], relative=True)
    return dict(key=payload['key'], group=payload.get('group'), pair=list(payload['pair']), modulators=payload['modulators'],
                vine=payload['vine'], features=name, gain=float(score - base), z=float(z), on=bool(on),
                p={k: result['terms'][i]['p'] for i, k in enumerate(payload['terms'])},
                edge_p=result['p'], estimates=np.round(result['estimates'], 4).tolist(), variables=payload['variables'],
                block=payload['block'], seconds=round(time.time() - start, 2))


def groups_of(terms: Sequence[dict], max_group=4) -> List[dict]:
    """The joint models of step 3.  terms: dicts with ``name``, ``subset`` (variables), ``shape`` and ``strength``
    (larger first when a group has to be cut).  For every pair of variables that at least two terms contain:
    dict(pair, members: [(name, others, shape)], modulators: the other variables of the members in order).
    A group holds at most ``max_group`` modulating variables; members are added by strength while they fit."""
    by_pair: Dict = {}
    for term in terms:
        for x, y in combinations(sorted(int(v) for v in term['subset']), 2):
            by_pair.setdefault((x, y), []).append(term)
    out = []
    for (x, y), members in sorted(by_pair.items()):
        if len(members) < 2:
            continue
        chosen, modulators, seen = [], [], set()
        for term in sorted(members, key=lambda item: -item['strength']):
            others = tuple(sorted(int(v) for v in term['subset'] if v not in (x, y)))
            base = term['shape'][len(tr.OWN):] if term['shape'].startswith(tr.OWN) else term['shape']
            shape = tr.OWN + ('product' if len(others) == 2 else 'quadratic' if base == 'quadratic' else 'linear')
            if len(set(modulators) | set(others)) > max_group:
                continue
            modulators += [v for v in others if v not in modulators]
            if (shape, others) not in seen:
                seen.add((shape, others))
            chosen.append((term['name'], others, shape))
        if len({(shape, others) for _, others, shape in chosen}) >= 2:
            out.append(dict(pair=(x, y), members=chosen, modulators=modulators))
    return out


def fit(train: Sequence[np.ndarray], validation: Sequence[np.ndarray], confirm: Optional[Sequence[np.ndarray]] = None,
        test: Optional[Sequence[np.ndarray]] = None, order: Optional[Sequence[int]] = None, states: Optional[Sequence[str]] = None,
        align=True, margin_level=0.05, switch=0.05, max_order=3, screen_level=1e-3, max_terms=3, max_columns=4, keep_z=2.0,
        install_z=2.0, features='best', serial='auto', replicates=399, level=0.05, max_size=8, test_states=('ind', 'gaussian'),
        seed=0, workers=1, pool=None, margin='rule', orientations=True, estimate='profile', joint=True, max_group=4,
        confirm_install=True, tolerance=TOLERANCE, show_z=SHOW_Z, keep=halfsample.KEEP) -> dict:
    """The pipeline, second version (see the module).  Arguments as `vinemodel.fit`, except:
    serial: 'auto' (lag-one autocorrelation of every variable from the training windows), None (independent
    observations) or one autocorrelation per variable; it replaces ``block``.
    estimate: the estimate of the tests ('profile'; 'score' is that of the first version).
    joint: test terms that share two variables together (step 3 of the module); max_group: the largest number of
    modulating variables in such a model.
    confirm_install: with confirmation windows, keep an installed edge only if it also beats the simplified state
    on them, and compare the whole model with the simplified vine there (step 1 of the module).
    tolerance: the tests shrink observed per-window estimates toward their precision-weighted mean by this
    fraction of its absolute value. Zero tests exact constancy. This is the band used by the frozen pipeline.
    show_z: an orientation other than the vine's counts in the rule for a change only if the trajectory fitted
    there switches the term on in some window and raises the validation score by at least this many standard
    errors (step 5 of the module); None: every orientation counts, as in the first version.
    Returns the dict of `vinemodel.fit` with, in addition, ``serial`` (autocorrelations and blocks per variable),
    ``whole`` (gain of the assembled vine over the simplified vine on the confirmation or validation windows, its
    standard score, and whether the simplified vine was returned instead) and ``groups`` (the joint models).  An
    edge has ``confirm_gain`` and ``confirm_z`` (None if it was not installed on the validation windows).  For a tested term: ``p`` is the
    largest p-value over the orientations that show the term and the joint models, ``declared`` the false
    discovery rule on these; ``p_orientations`` and ``declared_orientations`` are the rule of the first version
    (every orientation, no joint model); ``joint`` lists the joint models of the term with its p-value in each; an
    orientation has ``z`` and ``shown``.
    """
    if features not in ('own', 'coordinates', 'best'):
        raise ValueError("features must be 'coordinates', 'own' or 'best'")
    if margin not in ('rule', 'window', 'pooled'):
        raise ValueError("margin must be 'rule', 'window' or 'pooled'")
    clock: Dict[str, float] = {}
    start = time.time()
    r, blocks = serial_profile(train, serial)
    smallest = min(len(w) for w in train)
    dependent = bool(np.any(r > 0.0))
    pair = lambda i, j: halfsample.pair_serial(float(r[i]), float(r[j]), smallest)       # (inflation, block) of two variables
    others = dict(validation=list(validation))
    if test is not None:
        others['test'] = list(test)
    second_pass = bool(confirm is not None and confirm_install)
    if second_pass:
        others['confirm'] = list(confirm)                            # with the margins of the training windows, as the other roles
    if margin == 'rule':
        data, flags = choose_margins(list(train), others, margin_level, align, blocks)
    else:
        data = (margins.window if margin == 'window' else margins.pooled)(list(train), others)
        flags = [margin] * np.asarray(train[0]).shape[1]
    clock['margins'] = time.time() - start

    own_pool = None
    if pool is None and workers > 1:
        own_pool = pool = ProcessPoolExecutor(max_workers=workers)
    try:
        start = time.time()
        order = cvine.infer_order(list(data['train'])) if order is None else list(order)
        size = float(np.mean([len(w) for w in data['train']]))
        model = cvine.fit(list(data['train']), order=order, states=states, max_order=2, switch=switch,
                          serial=r if dependent else None, pool=pool)
        clock['vine'] = time.time() - start
        start = time.time()
        edge_block = lambda level_, edge: pair(order[level_], order[level_ + 1 + edge])[1]
        candidates, threshold, count = screen(model, list(data['train']), max_order, screen_level, max_terms,
                                              lags=(lambda level_, edge: edge_block(level_, edge) if edge_block(level_, edge) > 1 else 0)
                                              if dependent else None)
        clock['screen'] = time.time() - start
        start = time.time()
        keys = sorted(candidates)
        roles = [role for role in data if role != 'confirm']
        collected = {role: vinetraj.collect(model, data[role], keys) for role in roles}
        scores = {role: [ndtri(fam.clip01(np.asarray(w, dtype=np.float64)[:, order], fam.EPS)) for w in data[role]] for role in roles}
        names = ('coordinates', 'own') if features == 'best' else (features,)
        payloads, described = [], {}
        for level_, edge in keys:
            raw_components = [c['component'] for c in candidates[(level_, edge)]]
            sets = {name: [(('own_' if name == 'own' else '') + shape, tuple(int(m) for m in members)) for shape, members in raw_components]
                    for name in names}
            columns = vinetraj.calibration_columns(model, level_, edge, raw_components, max_columns)      # for the tests only
            needed = sorted({m for _, members in raw_components for m in members})
            place = {m: i for i, m in enumerate(needed)}
            described[(level_, edge)] = dict(sets=sets, columns=columns)
            payloads.append(dict(
                key=(level_, edge), variants=[(name, [(shape, tuple(place[m] for m in members)) for shape, members in sets[name]]) for name in names],
                n=size, keep_z=keep_z, block=edge_block(level_, edge),
                u={role: collected[role][0][(level_, edge)][0] for role in roles}, v={role: collected[role][0][(level_, edge)][1] for role in roles},
                zeta={role: [z[:, needed] for z in collected[role][1]] for role in roles},
                own={role: [s[:, needed] for s in scores[role]] for role in roles}))
        clock['collect'] = time.time() - start
        start = time.time()
        fitted = list(pool.map(_edge, payloads, chunksize=1)) if pool is not None else [_edge(p) for p in payloads]
        for result in fitted:
            described[tuple(result['key'])]['components'] = described[tuple(result['key'])]['sets'][result['features']]
        clock['edges'] = time.time() - start

        # one model, without calibrations; then the whole model against the simplified vine
        start = time.time()
        specs = []
        for result in fitted:
            if not result['kept']:
                continue
            level_, edge = result['key']
            specs.append(dict(level=level_, edge=edge, components=described[(level_, edge)]['components'], penalty=result['penalty'],
                              columns=None, block=edge_block(level_, edge)))
        vine, report = dynvine.assemble(model, list(data['train']), list(data['validation']), specs, root_columns={}, install='gain',
                                        margin=install_z, block=1)
        assembled = {tuple(entry['key']): entry for entry in report if 'on' in entry}
        if second_pass and vine.trajectories:                        # the installed edges once more, judged on the confirmation windows
            again = [spec for spec in specs if assembled[(spec['level'], spec['level'] + 1 + spec['edge'])]['installed']]
            vine, report = dynvine.assemble(model, list(data['train']), list(data['confirm']), again, root_columns={}, install='gain',
                                            margin=install_z, block=1)
            for entry in report:
                if 'on' not in entry:
                    continue
                first = assembled[tuple(entry['key'])]
                first.update(confirm_gain=entry['install_gain'], confirm_z=entry['install_z'], installed=entry['installed'])
                if entry['installed']:
                    first.update(on=entry['on'], pieces=entry['pieces'], coefficients=entry['coefficients'], stationarity=entry['stationarity'])
        whole = dict(gain=0.0, z=0.0, fallback=False, windows='confirmation' if second_pass else 'validation')
        if vine.trajectories:
            windows = list(data['confirm' if second_pass else 'validation'])
            difference = [a - cvine.forward(model, t, u)['logc'].sum(axis=1) for t, (a, u) in enumerate(zip(vine.log_density(windows), windows))]
            longest = max(spec['block'] for spec in specs if assembled[(spec['level'], spec['level'] + 1 + spec['edge'])]['installed'])
            whole.update(gain=float(np.concatenate(difference).mean()), z=dynvine.standard_score(difference, longest))
            if whole['gain'] <= 0.0:
                vine, whole['fallback'] = dynvine.DynamicVine(model), True
                for entry in assembled.values():
                    entry['installed'] = False
        clock['assemble'] = time.time() - start

        # tests of the terms of the installed edges: every orientation, and the joint models of shared pairs
        start = time.time()
        tests, groups = {}, []
        if confirm is not None and replicates:
            raw_confirm = [np.asarray(w, dtype=np.float64) for w in confirm]
            ranked = margins.ranks(raw_confirm, flags)
            jobs = []

            def job(key, variables, variants, columns, terms, pair_, modulators, own_orientation, tag, group=None):
                return dict(key=key, group=group, variables=variables, train=[np.asarray(w)[:, variables] for w in data['train']],
                            validation=[np.asarray(w)[:, variables] for w in data['validation']], confirm=[w[:, variables] for w in ranked],
                            raw=[w[:, variables] for w in raw_confirm], pooled=[flags[v] for v in variables], variants=variants, columns=columns,
                            terms=terms, pair=pair_, modulators=modulators, vine=own_orientation, states=list(test_states), switch=switch,
                            replicates=replicates, block=pair(pair_[0], pair_[1])[1], keep=keep, estimate=estimate, tolerance=float(tolerance),
                            seed=[int(seed), key[0], key[1], tag])
            sets = lambda parts: [(name, [(('own_' if name == 'own' else '') + shape, members) for shape, members in parts])
                                  for name in (('coordinates', 'own') if features == 'best' else (features,))]
            listed = []                                              # the installed terms, for the joint models
            for result in fitted:
                level_, edge = result['key']
                entry = assembled.get((level_, level_ + 1 + edge))
                if entry is None or not entry['installed']:
                    continue
                info = described[(level_, edge)]
                on_terms = [k for k, on in enumerate(entry['on']) if on]
                if not on_terms:
                    continue
                plain = [candidates[(level_, edge)][k]['component'] for k in on_terms]
                roots = vinetraj.relevant(model, level_, edge, plain, info['columns'], max_size)
                inside = {m: i for i, m in enumerate(roots)}
                a, b = order[level_], order[level_ + 1 + edge]
                jobs.append(job((level_, edge), [order[m] for m in roots] + [a, b],
                                sets([(shape, tuple(inside[m] for m in members)) for shape, members in plain]),
                                [inside[m] for m in info['columns'] if m in inside], on_terms, (a, b),
                                [[order[m] for m in members] for _, members in plain], True, 0))
                calibrated = [order[m] for m in info['columns']]
                for k, (shape, members) in zip(on_terms, plain):
                    subset = [order[m] for m in members] + [a, b]
                    rest = [order[m] for m in roots if order[m] not in subset]
                    listed.append(dict(name=((level_, edge), k), subset=subset, shape=shape, strength=float(entry['install_z']), rest=rest,
                                       calibrated=calibrated))
                    if not orientations:
                        continue
                    tag = 100 * (k + 1)
                    for first in range(len(subset)):
                        for second in range(first + 1, len(subset)):
                            x, y = subset[first], subset[second]
                            if {x, y} == {a, b}:
                                continue
                            tag += 1
                            others_ = [v for v in subset if v not in (x, y)]
                            variables = rest + others_ + [x, y]
                            places = tuple(range(len(rest), len(rest) + len(others_)))
                            columns = sorted({variables.index(v) for v in calibrated if v in rest} | set(places))
                            jobs.append(job((level_, edge), variables, sets([('linear' if len(others_) == 1 else 'product', places)]), columns,
                                            [k], (x, y), [others_], False, tag))
            if joint:
                by_name = {term['name']: term for term in listed}
                for number, group in enumerate(groups_of(listed, max_group)):
                    x, y = group['pair']
                    members = [by_name[name] for name, _, _ in group['members']]
                    modulators = list(group['modulators'])
                    room = max(max_size - 2 - len(modulators), 0)
                    rest = []
                    for term in members:
                        rest += [v for v in term['rest'] if v not in rest and v not in modulators and v not in (x, y)]
                    rest = rest[:room]
                    variables = rest + modulators + [x, y]
                    calibrated = {v for term in members for v in term['calibrated']}
                    columns = sorted({variables.index(v) for v in calibrated if v in rest} | {variables.index(v) for v in modulators})
                    parts, index = [], {}
                    for name, others_, shape in group['members']:
                        component = (shape, tuple(variables.index(v) for v in others_))
                        if component not in index:
                            index[component] = len(parts)
                            parts.append(component)
                        group.setdefault('term_of', {})[name] = index[component]
                    first = group['members'][0][0]
                    jobs.append(job(first[0], variables, [('own', parts)], columns, list(range(len(parts))), (x, y),
                                    [[variables[i] for i in members_] for _, members_ in parts], False, 10000 + number, group=number))
                    groups.append(group)
            results = list(pool.map(_orientation, jobs, chunksize=1)) if pool is not None else [_orientation(j) for j in jobs]
            units, together = {}, {}
            for result in results:
                if result['group'] is not None:
                    group = groups[result['group']]
                    group.update(p=[result['p'][i] for i in sorted(result['p'])], variables=result['variables'], seconds=result['seconds'],
                                 estimates=result['estimates'], block=result['block'])
                    for name, position in group['term_of'].items():
                        together.setdefault(name, []).append(dict(pair=list(group['pair']), modulators=list(group['modulators']),
                                                                   p=float(result['p'][position]), terms=len(result['p'])))
                    continue
                for k, value in result['p'].items():
                    index = list(result['p']).index(k)                   # the vine's orientation tests all terms of the edge at once
                    units.setdefault((tuple(result['key']), k), []).append(dict(
                        pair=result['pair'], modulators=result['modulators'][index], features=result['features'], gain=result['gain'],
                        z=result['z'], on=result['on'], shown=bool(result['vine'] or show_z is None or (result['on'] and result['z'] >= show_z)),
                        p=value, vine=result['vine'], variables=result['variables'], block=result['block'], seconds=result['seconds']))
            names_ = sorted(units)
            alone = [max(o['p'] for o in units[name]) for name in names_]                     # every orientation: the first version
            shown = [max(o['p'] for o in units[name] if o['shown']) for name in names_]       # the vine's orientation is always among them
            largest = [max([p_value] + [g['p'] for g in together.get(name, [])]) for name, p_value in zip(names_, shown)]
            tops = [max(units[name], key=lambda o: o['gain']) for name in names_]
            second = vinemodel.discoveries([top['p'] for top in tops], level)
            first_version = vinemodel.discoveries(alone, level)
            for name, p_value, flag, top, other, p_alone, old in zip(names_, largest, vinemodel.discoveries(largest, level), tops, second, alone,
                                                                     first_version):
                key, k = name
                item = candidates[key][k]
                record = tests.setdefault(key, dict(terms={}))
                record['terms'][k] = dict(subset=sorted(int(v) for v in item['subset']), p=float(p_value), declared=bool(flag),
                                          p_orientations=float(p_alone), declared_orientations=bool(old), joint=together.get(name, []),
                                          orientations=units[name],
                                          directed=dict(pair=top['pair'], modulators=top['modulators'], features=top['features'], gain=top['gain'],
                                                        p=top['p'], declared=bool(other)))
            for record in tests.values():
                record['p'] = float(min(term['p'] for term in record['terms'].values()))
                record['declared'] = bool(any(term['declared'] for term in record['terms'].values()))
        clock['tests'] = time.time() - start
    finally:
        if own_pool is not None:
            own_pool.shutdown()

    edges = []
    for result in fitted:
        level_, edge = result['key']
        items = candidates[(level_, edge)]
        info = described[(level_, edge)]
        entry = assembled.get((level_, level_ + 1 + edge), {})
        pair_ = (order[level_], order[level_ + 1 + edge])
        record = dict(level=level_, edge=edge, pair=pair_, components=[[shape, [int(m) for m in members]] for shape, members in info['components']],
                      modulators=[[order[m] for m in members] for _, members in info['components']],
                      subsets=[sorted(set(pair_) | {order[m] for m in members}) for _, members in info['components']],
                      strength=[c['strength'] for c in items], interval=[[c['start'], c['stop']] for c in items],
                      calibration_columns=[order[m] for m in info['columns']], kept=result['kept'], z=result['z'],
                      features=result['features'], variant_scores=result['variant_scores'],
                      score=result['score'], base_score=result['base_score'], test_gain=result.get('test_gain'),
                      screen_on=result['on'], screen_pieces=result['pieces'], seconds=result['seconds'], block=edge_block(level_, edge),
                      smooth=[float(s) if np.isfinite(s) else 'inf' for s in result['penalty'].smooth], jump=list(result['penalty'].jump),
                      installed=bool(entry.get('installed', False)))
        if entry:
            record.update(on=entry['on'], pieces=entry['pieces'], coefficients=np.round(entry['coefficients'], 5).tolist(),
                          install_gain=entry['install_gain'], install_z=entry['install_z'], state=entry['state'],
                          confirm_gain=entry.get('confirm_gain'), confirm_z=entry.get('confirm_z'),
                          validation_gain=entry['validation_gain'], calibration=False, stationarity=entry['stationarity'])
        if (level_, edge) in tests:
            record['test'] = tests[(level_, edge)]
        edges.append(record)
    inflations = [pair(i, j)[0] for i, j in combinations(range(len(r)), 2)] if dependent else [1.0]
    out = dict(order=order, flags=flags, block=int(max(blocks)), inflation=float(max(inflations)),
               serial=dict(r=[float(v) for v in r], blocks=[int(b) for b in blocks]), simplified=model, vine=vine, edges=edges,
               threshold=threshold, components=count, root_calibrations=[], whole=whole,
               groups=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in group.items() if k != 'term_of'} for group in groups],
               seconds={k: round(v, 1) for k, v in clock.items()})
    if test is not None:
        windows = list(data['test'])
        out['scores'] = dict(simplified=float(np.concatenate([cvine.forward(model, t, u)['logc'].sum(axis=1) for t, u in enumerate(windows)]).mean()),
                             assembled=vine.score(windows))
    return out
