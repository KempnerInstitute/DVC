"""Fit a dynamic C-vine to observation windows and test its edge trajectories.

The original assembly routine uses the following sequence. The public study entry point is vinemodel2.fit.

1. Margins (`margins.choose` with alignment): per variable, over all windows where its margin does not change,
   aligned where its location or spread changes, within every window where its shape changes.
2. One simplified switching vine in one root order (`cvine.fit`).
3. Candidate terms of every edge from the scan of that vine at a lenient level (`screen`, the scan of
   `vinetraj.screen`; third order by default, fourth with ``max_order = 4``).
4. Every edge with candidates (`_edge`): calibration of its two transforms with the gate on validation data; joint
   coefficient trajectories with the candidate terms, penalties chosen on validation data.  The modulating
   features are functions of the modulators' coordinates in the vine or of their own normal scores.  Neither is
   right for every edge: a variable that modulates a pair by its own value is one own-score term in every root
   order, and a variable whose part that the earlier roots do not explain is what couples with the pair (in the
   law of E18, a leaf of a planted pair seen as the modulator of the edge of the other two) is one coordinate
   term.  With ``features='best'`` both are fitted and the edge uses the set with the higher validation score.
   The terms are kept if one is switched on and the edge predicts the validation observations better than the
   baseline-only edge by ``keep_z`` standard errors.
5. Assembly (`dynvine.assemble`): the kept edges become edges of one vine, each fitted on the transforms of the
   model assembled so far, and an edge is installed only if it predicts the validation observations better than
   the state the simplified vine has for that pair, by ``install_z`` standard errors.  The result is one density
   with its sampler.
6. For every term of an installed edge, on confirmation windows that were used for nothing else: the test that
   its modulation coefficient is the same in every window, with the reference distribution from random halves of
   the data (`halfsample.change_test`), in a sub-vine of the variables the edge depends on (`vinetraj.relevant`).
   The test is run in every orientation of the term's subset: every choice of the two variables that form the
   pair, the others modulating, each with the features (coordinates or own scores) that predict the validation
   observations better.  A change of the term is declared only if every orientation rejects (the largest of
   their p-values, then the false discovery rule over the installed terms).  Reason: when the pairwise
   dependence of a variable of the subset changes, the coefficient changes in the orientations in which that
   dependence enters it, and stays the same in an orientation that describes the interaction by quantities the
   change does not touch; a change of the interaction itself shows in all of them.  The orientation with the
   largest validation gain is reported as the directed reading ("k modulates the pair (i, j)"), with its own
   p-value and its own declaration by the false discovery rule: a second, less conservative rule that depends on
   that orientation being the right one.

Serial dependence: with ``block`` > 1 the tests of the margin rule (step 1) use one observation per block, the
simplified vine of step 2 chooses and switches its states with costs scaled by the variance inflation of an
average of products of two autocorrelated series (otherwise it fits the noise of every window), the scan of step
3 uses a long-run variance with that many lags, the standard errors of steps 4 and 5 block means, and the halves
of step 6 halves of blocks.  The penalties of step 4 are not adjusted (they are chosen on validation
data).

What a kept term means is unchanged: in this root order, the dependence of the two variables of the edge given the
earlier roots varies with the modulator.  Whether the subset is an interaction in other orders is the question of
the strict detector (`p2.detect`), which is not part of this module.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import ndtri

from . import cvine, dynvine, halfsample, margins, scan, trajectory as tr, vinetraj
from . import families as fam


def screen(model: cvine.DynamicCVine, windows: Sequence[np.ndarray], max_order=3, level=1e-3, max_terms=3, shapes=scan.SHAPES,
           eps=fam.EPS, lags=0):
    """The candidate terms of `vinetraj.screen`, computed level by level without a record for every component, so
    that fourth order can be scanned with a hundred variables (some four million components).
    ``lags``: for serially dependent observations, the number of lags of the long-run variance of the statistics
    (`scan.moment_scan`); with the plain variance their size grows with the dependence and nearly every edge is
    screened.  Returns (candidates, threshold, number of components) as `vinetraj.screen`."""
    steps = len(windows)
    components = {lvl: cvine.enumerate_components(lvl, max_order, shapes) for lvl in range(1, model.depth)}
    z = {}
    for t, u in enumerate(windows):
        out = cvine.forward(model, t, u, eps, keep=True)
        for lvl, parts in components.items():
            if not parts:
                continue
            v = out['pits'][lvl]
            values = cvine._edge_statistics(v[:, lvl], v[:, lvl + 1:], out['zeta'][:, :lvl], parts, True, int(lags), 0)   # (components, edges)
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
    """Worker of step 4 for one edge.  Coordinates and own scores are those of the positions ``needed``.
    ``variants``: (name, components) pairs, the feature sets to try; the first with the best validation score is
    used."""
    start = time.time()
    key, columns, u, v, zeta, own, n = (payload[name] for name in ('key', 'columns', 'u', 'v', 'zeta', 'own', 'n'))
    calibration = tr.fit_calibration(u['train'], v['train'], zeta['train'], degree=2, products=True, columns=columns)
    gate, gain = calibration.gated(u['validation'], v['validation'], zeta['validation'])
    pairs = {role: [gate.apply(t, u[role][t], v[role][t], zeta[role][t]) for t in range(len(u[role]))] for role in u}
    later = {role: [a for a, _ in rows] for role, rows in pairs.items()}
    root = {role: [b for _, b in rows] for role, rows in pairs.items()}
    without = lambda role: (later[role], root[role], [np.zeros((len(a), 0)) for a in later[role]])
    plain = tr.edge_data(*without('validation'))
    base, base_score = vinetraj.tune(tr.edge_data(*without('train')), plain, 0, n)
    best, tried = None, {}
    for name, components in payload['variants']:
        features = {role: [tr.component_features(components, z, o) for z, o in zip(zeta[role], own[role])] for role in u}
        train = tr.edge_data(later['train'], root['train'], features['train'])
        validation = tr.edge_data(later['validation'], root['validation'], features['validation'], centre=train.means)
        fitted, score = vinetraj.tune(train, validation, len(components), n)
        tried[name] = score
        if best is None or score > best['score'] + 1e-12:
            best = dict(name=name, components=components, features=features, train=train, validation=validation, fitted=fitted, score=score)
    fitted, train, validation, components = best['fitted'], best['train'], best['validation'], best['components']
    on = [bool(fitted.pieces(1 + k)) for k in range(len(components))]
    difference = tr.log_density(fitted.theta, validation) - tr.log_density(base.theta, plain)
    z = dynvine.standard_score([row[w > 0] for row, w in zip(difference, validation.w)], payload['block'])
    out = dict(key=key, kept=bool(any(on) and z >= payload['keep_z']), on=on, score=best['score'], base_score=base_score, z=z,
               features=best['name'], variant_scores=tried, calibration_kept=[bool(g > 0.0) for g in gain],
               calibration_gain=[float(g) for g in gain], pieces=[fitted.pieces(p) for p in range(1 + len(components))],
               penalty=fitted.penalty, stationarity=tr.stationarity(train, fitted))
    if 'test' in u:
        test = tr.edge_data(later['test'], root['test'], best['features']['test'], centre=train.means)
        out['test_gain'] = float(tr.score(fitted.theta, test) - tr.score(base.theta, tr.edge_data(*without('test'))))
    out['seconds'] = round(time.time() - start, 2)
    return out


def _orientation(payload):
    """Worker of step 6: the test of the terms of one edge in one orientation.  The sub-vine has the columns
    ``variables``: conditioning variables, then the modulators, then the pair.  ``variants``: (name, components)
    feature sets; the one with the best validation score of the tuned edge is tested.  ``terms``: for every
    component, the index of the candidate term of the vine's edge it stands for."""
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
        _, score = vinetraj.tune(edge, tr.edge_data(*inputs['validation'], centre=edge.means), len(components), n)
        if base is None:
            empty = lambda role: tr.edge_data(inputs[role][0], inputs[role][1], [np.zeros((len(a), 0)) for a in inputs[role][0]])
            base = vinetraj.tune(empty('train'), empty('validation'), 0, n)[1]
        if best is None or score > best[1] + 1e-12:
            best = (name, score, components)
    name, score, components = best
    result = halfsample.change_test(payload['confirm'], lower_fit, components, replicates=payload['replicates'],
                                    rng=np.random.default_rng(payload['seed']), calibrate=2, columns=payload['columns'],
                                    pooled=payload['pooled'], block=payload['block'], raw=payload['raw'], keep=payload['keep'])
    return dict(key=payload['key'], pair=list(payload['pair']), modulators=payload['modulators'], vine=payload['vine'], features=name,
                gain=float(score - base), p={k: result['terms'][i]['p'] for i, k in enumerate(payload['terms'])}, edge_p=result['p'],
                estimates=np.round(result['estimates'], 4).tolist(), variables=payload['variables'], seconds=round(time.time() - start, 2))


def discoveries(p: Sequence[float], level=0.05) -> List[bool]:
    """The false discovery rule of Benjamini and Hochberg."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros(p.size, dtype=bool)
    if p.size:
        rank = np.argsort(p)
        passed = np.nonzero(p[rank] <= level * np.arange(1, p.size + 1) / p.size)[0]
        if passed.size:
            out[rank[:passed[-1] + 1]] = True
    return out.tolist()


def fit(train: Sequence[np.ndarray], validation: Sequence[np.ndarray], confirm: Optional[Sequence[np.ndarray]] = None,
        test: Optional[Sequence[np.ndarray]] = None, order: Optional[Sequence[int]] = None, states: Optional[Sequence[str]] = None,
        align=True, margin_level=0.05, switch=0.05, max_order=3, screen_level=1e-3, max_terms=3, max_columns=4, keep_z=2.0,
        install_z=2.0, features='best', block=1, replicates=399, level=0.05, max_size=8, test_states=('ind', 'gaussian'), seed=0,
        workers=1, pool=None, margin='rule', orientations=True, keep=halfsample.KEEP) -> dict:
    """The pipeline of this module.

    train, validation, confirm, test: lists of windows (n_t, d) of observations on any scale (the same variables
    and the same number of windows in every role).  ``confirm`` is used only for the tests of step 6 (without it
    no test is run); ``test`` only for the held-out log density of the result.
    features: 'coordinates', 'own', or 'best' (both are fitted for every edge and the one with the higher
    validation score is used).  margin: 'rule' (step 1), or 'window' or 'pooled' for every variable.
    orientations: False tests every term only in the orientation of the vine.  block: observations per block for
    serially dependent observations, or 'auto' (`halfsample.block_length` of the training windows).
    keep: share of a window in a part of the half-sample test.
    Returns dict(order, flags, block, simplified, vine, edges, scores, seconds).  ``vine`` is the assembled
    `dynvine.DynamicVine`; ``edges`` has one entry per screened edge: its pair, terms and their modulators and
    subsets (variables), whether it was kept and installed, its pieces and coefficients in the assembled model, and
    for an installed edge ``test``: per term its subset, the p-value of a change (the largest over the
    orientations), whether it is declared, the orientations with their features, validation gains and p-values,
    and the directed reading; for the edge the smallest p-value of its terms and whether any is declared.  ``scores``:
    mean held-out log density per observation of the simplified and of the assembled vine on ``test``.
    """
    if features not in ('own', 'coordinates', 'best'):
        raise ValueError("features must be 'coordinates', 'own' or 'best'")
    if margin not in ('rule', 'window', 'pooled'):
        raise ValueError("margin must be 'rule', 'window' or 'pooled'")
    clock: Dict[str, float] = {}
    start = time.time()
    ranked = margins.window(list(train))['train']
    if block == 'auto':
        block = halfsample.block_length(ranked)
    # Serial dependence inflates the variance of an average of products of two series by (1 + r^2) / (1 - r^2); the
    # costs with which the simplified vine chooses and switches states are per observation and are scaled with it.
    inflation = 1.0
    if block > 1:
        r = min(halfsample.autocorrelation(ranked), 0.99)
        inflation = (1.0 + r * r) / (1.0 - r * r)
    others = dict(validation=list(validation))
    if test is not None:
        others['test'] = list(test)
    if margin == 'rule':
        data, flags = margins.choose(list(train), others, margin_level, align=align, block=block)
        flags = [margins._kind(flag) for flag in flags]
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
        model = cvine.fit(list(data['train']), order=order, states=states, max_order=2, switch=switch * inflation,
                          activation=(inflation - 1.0) / size, pool=pool)
        clock['vine'] = time.time() - start
        start = time.time()
        candidates, threshold, count = screen(model, list(data['train']), max_order, screen_level, max_terms, lags=block if block > 1 else 0)
        clock['screen'] = time.time() - start
        start = time.time()
        keys = sorted(candidates)
        collected = {role: vinetraj.collect(model, windows, keys) for role, windows in data.items()}
        scores = {role: [ndtri(fam.clip01(np.asarray(w, dtype=np.float64)[:, order], fam.EPS)) for w in windows] for role, windows in data.items()}
        names = ('coordinates', 'own') if features == 'best' else (features,)
        payloads, described = [], {}
        for level_, edge in keys:
            raw_components = [c['component'] for c in candidates[(level_, edge)]]
            sets = {name: [(('own_' if name == 'own' else '') + shape, tuple(int(m) for m in members)) for shape, members in raw_components]
                    for name in names}
            columns = vinetraj.calibration_columns(model, level_, edge, raw_components, max_columns)
            needed = sorted(set(columns) | {m for _, members in raw_components for m in members})
            place = {m: i for i, m in enumerate(needed)}
            described[(level_, edge)] = dict(sets=sets, columns=columns)
            payloads.append(dict(
                key=(level_, edge), variants=[(name, [(shape, tuple(place[m] for m in members)) for shape, members in sets[name]]) for name in names],
                columns=[place[m] for m in columns], n=float(np.mean([len(w) for w in data['train']])), keep_z=keep_z, block=block,
                u={role: collected[role][0][(level_, edge)][0] for role in data}, v={role: collected[role][0][(level_, edge)][1] for role in data},
                zeta={role: [z[:, needed] for z in collected[role][1]] for role in data},
                own={role: [s[:, needed] for s in scores[role]] for role in data}))
        clock['collect'] = time.time() - start
        start = time.time()
        fitted = list(pool.map(_edge, payloads, chunksize=1)) if pool is not None else [_edge(p) for p in payloads]
        for result in fitted:
            described[tuple(result['key'])]['components'] = described[tuple(result['key'])]['sets'][result['features']]
        clock['edges'] = time.time() - start

        # step 5: one model
        start = time.time()
        specs, root_columns = [], {}
        for result in fitted:
            if not result['kept']:
                continue
            level_, edge = result['key']
            info = described[(level_, edge)]
            specs.append(dict(level=level_, edge=edge, components=info['components'], penalty=result['penalty'], columns=info['columns']))
            if result['calibration_kept'][1] and level_ not in root_columns:
                root_columns[level_] = info['columns']
        vine, report = dynvine.assemble(model, list(data['train']), list(data['validation']), specs, root_columns=root_columns,
                                        install='gain', margin=install_z, block=block)
        assembled = {tuple(r['key']): r for r in report if 'on' in r}
        clock['assemble'] = time.time() - start

        # step 6: tests of the terms of the installed edges, in every orientation of their subsets
        start = time.time()
        tests = {}
        if confirm is not None and replicates:
            raw_confirm = [np.asarray(w, dtype=np.float64) for w in confirm]
            ranked = margins.ranks(raw_confirm, flags)
            jobs = []

            def job(key, variables, variants, columns, terms, pair, modulators, own_orientation, tag):
                return dict(key=key, variables=variables, train=[np.asarray(w)[:, variables] for w in data['train']],
                            validation=[np.asarray(w)[:, variables] for w in data['validation']], confirm=[w[:, variables] for w in ranked],
                            raw=[w[:, variables] for w in raw_confirm], pooled=[flags[v] for v in variables], variants=variants, columns=columns,
                            terms=terms, pair=pair, modulators=modulators, vine=own_orientation, states=list(test_states), switch=switch,
                            replicates=replicates, block=block, keep=keep, seed=[int(seed), key[0], key[1], tag])
            sets = lambda parts: [(name, [(('own_' if name == 'own' else '') + shape, members) for shape, members in parts])
                                  for name in (('coordinates', 'own') if features == 'best' else (features,))]
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
                if not orientations:
                    continue
                calibrated = [order[m] for m in info['columns']]
                for k, (shape, members) in zip(on_terms, plain):
                    subset = [order[m] for m in members] + [a, b]
                    rest = [order[m] for m in roots if order[m] not in subset]
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
            results = list(pool.map(_orientation, jobs, chunksize=1)) if pool is not None else [_orientation(j) for j in jobs]
            units = {}
            for result in results:
                for k, value in result['p'].items():
                    index = list(result['p']).index(k)                   # the vine's orientation tests all terms of the edge at once
                    units.setdefault((tuple(result['key']), k), []).append(dict(
                        pair=result['pair'], modulators=result['modulators'][index], features=result['features'], gain=result['gain'],
                        p=value, vine=result['vine'], variables=result['variables'], seconds=result['seconds']))
            names = sorted(units)
            largest = [max(o['p'] for o in units[name]) for name in names]
            tops = [max(units[name], key=lambda o: o['gain']) for name in names]
            second = discoveries([top['p'] for top in tops], level)
            for name, p_value, flag, top, other in zip(names, largest, discoveries(largest, level), tops, second):
                key, k = name
                item = candidates[key][k]
                record = tests.setdefault(key, dict(terms={}))
                record['terms'][k] = dict(subset=sorted(int(v) for v in item['subset']), p=float(p_value), declared=bool(flag),
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
        pair = (order[level_], order[level_ + 1 + edge])
        record = dict(level=level_, edge=edge, pair=pair, components=[[shape, [int(m) for m in members]] for shape, members in info['components']],
                      modulators=[[order[m] for m in members] for _, members in info['components']],
                      subsets=[sorted(set(pair) | {order[m] for m in members}) for _, members in info['components']],
                      strength=[c['strength'] for c in items], interval=[[c['start'], c['stop']] for c in items],
                      calibration_columns=[order[m] for m in info['columns']], kept=result['kept'], z=result['z'],
                      features=result['features'], variant_scores=result['variant_scores'],
                      score=result['score'], base_score=result['base_score'], test_gain=result.get('test_gain'),
                      screen_on=result['on'], screen_pieces=result['pieces'], seconds=result['seconds'],
                      smooth=[float(s) if np.isfinite(s) else 'inf' for s in result['penalty'].smooth], jump=list(result['penalty'].jump),
                      installed=bool(entry.get('installed', False)))
        if entry:
            record.update(on=entry['on'], pieces=entry['pieces'], coefficients=np.round(entry['coefficients'], 5).tolist(),
                          install_gain=entry['install_gain'], install_z=entry['install_z'], state=entry['state'],
                          validation_gain=entry['validation_gain'], calibration=bool(entry.get('calibration')),
                          stationarity=entry['stationarity'])
        if (level_, edge) in tests:
            record['test'] = tests[(level_, edge)]
        edges.append(record)
    out = dict(order=order, flags=flags, block=int(block), inflation=float(inflation), simplified=model, vine=vine, edges=edges,
               threshold=threshold, components=count,
               root_calibrations=sorted(int(r['key'][0]) for r in report if r.get('root_calibration')),
               seconds={k: round(v, 1) for k, v in clock.items()})
    if test is not None:
        windows = list(data['test'])
        out['scores'] = dict(simplified=float(np.concatenate([cvine.forward(model, t, u)['logc'].sum(axis=1) for t, u in enumerate(windows)]).mean()),
                             assembled=vine.score(windows))
    return out
