"""Scan subsets for supported modulation across orientations of the conditioned pair.

The scan uses a switching copula for each pair. A subset advances when all orientations support it, then stronger overlapping subsets enter the confirmation model.

1. Misfit of the pair layer: every conditional transform is standardised against each of its conditioning scores
   before the test (`robust.smooth_standardize`), and the statistic is divided by its simulated null scale.
2. Statistic: the influence function of a correlation in place of the product (`robust.modulation_scan`).
3. Serial dependence (option ``serial``): Bartlett variance.
4. Run of windows: the run of a triple is the intersection of the significant runs of its three conditioning
   choices (P1 reported the run of the weakest choice); for a quadruple, of the six intervals.
5. Sequential confirmation: a candidate that shares two variables with stronger confirmed triples is tested again
   in a vine rooted at their third variable in which those triples are modelled (gated transforms).  P1 conditioned
   on that variable with simplified transforms, which cannot explain a quadruple made by one variable modulating
   two pairs, nor a triple made by one variable modulating two pairs that share a member.

Detection output has the format of P1 (`pipeline.Detection`), so modelling, readouts and the change test follow
unchanged.

Changed after the development run E13 (before any confirmation run): the standardisation is applied after every
conditioning stage, not only before the test (`given`, `restandardize`), and the variables a candidate is
conditioned on are chosen with a dependence measure that also sees dependence without correlation (`dependence`).
Reason: on U-shaped stars P2 confirmed triples of leaves in 12 of 20 datasets at n = 250 and 20 of 20 at n = 1000.
Those leaves do interact when the hub is ignored, and the hub explains them, but the check that conditions on the
hub used pair copulas that cannot represent a U-shape.

Frozen on 10 November 2026 for the confirmation run E14 with the configuration below (serial variance on: in
E13 it changed no result on independent observations).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import List, Sequence

import numpy as np
from scipy.stats import norm, rankdata

from . import cvine, pairscan, paths, pipeline, robust, scan

CONFIG = dict(pipeline.CONFIG, version='P2', standardize=True, score='correlation', serial=True,
              intersect_runs=True, sequential=True)


def statistic(a, b, g, conditioning, kinds, config):
    """z statistics (len(kinds), p, q).  a, b: normal scores (n, p), (n, q), or b None for all pairs of columns of
    a; conditioning: (n, c) scores the transforms were conditioned on; g: (n, len(kinds)) test functions."""
    if config['standardize']:
        a = robust.standardize_on(a, conditioning)
        b = a if b is None else robust.standardize_on(b, conditioning)
    elif b is None:
        b = a
    z = robust.modulation_scan(a, b, g, score=config['score'], serial=config['serial'])
    if config['standardize']:
        count = 1 if np.ndim(conditioning) == 1 else np.shape(conditioning)[1]
        z = z / np.array([robust.null_scale(a.shape[0], count, kind) for kind in kinds])[:, None, None]
    return z


def given(layer, t, u, k, config):
    """Pseudo-observations of every variable given variable k (column k holds u_k itself): the pair layer's
    conditional transforms and, with ``standardize``, each standardised against the score of k and mapped back to
    (0, 1) by its ranks.  Without this step a misfit of the pair copula with k survives into the next
    conditioning stage, where the pair copulas among the conditioned variables absorb it in a way that the final
    standardisation cannot undo (E13: triples of leaves of a U-shaped star were not explained by their hub)."""
    w = pairscan.given(layer, t, u, k)
    if not config['standardize']:
        return w
    n, d = w.shape
    others = [v for v in range(d) if v != k]
    scores = robust.smooth_standardize(scan.normal_scores(w[:, others]), scan.normal_scores(np.asarray(u)[:, [k]])[:, 0])
    out = w.copy()
    out[:, others] = rankdata(scores, axis=0) / (n + 1.0)
    return out


def restandardize(h, conditioning, config):
    """Standardise probability transforms h (n, p) against conditioning scores and return pseudo-observations."""
    if not config['standardize']:
        return h
    scores = robust.standardize_on(scan.normal_scores(h), conditioning)
    return rankdata(scores, axis=0) / (h.shape[0] + 1.0)


def dependence(windows):
    """Mean over windows of a dependence measure that also sees dependence without correlation: the largest
    absolute correlation among (z_i, z_j), (z_i^2, z_j), (z_i, z_j^2), (z_i^2, z_j^2), z the normal scores.
    Used only to choose the variables a candidate is conditioned on."""
    total = 0.0
    for u in windows:
        z = scan.normal_scores(u)
        both = np.column_stack((z, z * z))
        c = np.abs(np.corrcoef(both, rowvar=False))
        d = z.shape[1]
        total = total + np.maximum.reduce([c[:d, :d], c[:d, d:], c[d:, :d], c[d:, d:]])
    return total / len(windows)


def _scan_window(args):
    layer, t, u, config = args
    shapes = tuple(config['shapes'])
    w = layer.transforms(t, u)
    n, d, _ = w.shape
    scores = scan.normal_scores(w.reshape(n, d * d)).reshape(n, d, d)
    z = scan.normal_scores(u)
    stat = np.full((len(shapes), d, d, d), np.nan)
    for k in range(d):
        others = np.array([v for v in range(d) if v != k])
        zz = statistic(scores[:, others, k], None, scan.shape_functions(z[:, k], shapes), z[:, [k]], shapes, config)
        for s in range(len(shapes)):
            block = zz[s].copy()
            np.fill_diagonal(block, np.nan)
            stat[s][np.ix_(others, others, [k])] = block[:, :, None]
    return stat.astype(np.float32)


def scan_all(layer, windows, config=CONFIG, workers=1):
    """Statistics (T, shapes, d, d, d) indexed [t, shape, i, j, k]."""
    tasks = [(layer, t, u, config) for t, u in enumerate(windows)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return np.stack(list(pool.map(_scan_window, tasks)))
    return np.stack([_scan_window(task) for task in tasks])


def _runs(mask):
    """Maximal runs of True in a boolean vector: list of (start, stop)."""
    out, start = [], None
    for t, value in enumerate(list(mask) + [False]):
        if value and start is None:
            start = t
        if not value and start is not None:
            out.append((start, t))
            start = None
    return out


def candidates(stat, config=CONFIG):
    """Triples whose three conditioning choices all exceed the familywise interval threshold.

    With ``intersect_runs`` the significant runs of the three choices (any shape) are intersected; a triple with an
    empty intersection is dropped, and the reported run is the common run on which the weakest choice is strongest.
    Returns (list of dicts, threshold, number of triples that passed before the intersection)."""
    steps, n_shapes, d = stat.shape[:3]
    stat = np.nan_to_num(stat.astype(np.float64))
    found, threshold = pairscan.candidates(stat, config['level'])
    if not config['intersect_runs']:
        return found, threshold, len(found)
    out = []
    for item in found:
        i, j, k = item['triple']
        common = np.ones(steps, dtype=bool)
        for a, b, c in ((i, j, k), (i, k, j), (j, k, i)):
            mask = np.zeros(steps, dtype=bool)
            for s in range(n_shapes):
                for start, stop, _ in scan.significant_intervals(stat[:, s, a, b, c], threshold):
                    mask[start:stop] = True
            common &= mask
        best = None
        for start, stop in _runs(common):
            value = min(np.abs(stat[start:stop, :, a, b, c].sum(axis=0)).max() / np.sqrt(stop - start)
                        for a, b, c in ((i, j, k), (i, k, j), (j, k, i)))
            if best is None or value > best[0]:
                best = (float(value), (start, stop))
        if best is not None:
            out.append(dict(item, value=best[0], interval=best[1], separate_value=item['value'],
                            separate_interval=item['interval']))
    return sorted(out, key=lambda r: -r['value']), threshold, len(found)


def hub_check(layer, windows, triple, interval, hubs: Sequence[int], config=CONFIG):
    """As `pairscan.hub_check` with the P2 statistic: smallest over the three conditioning choices of the largest
    |sum of window statistics| / sqrt(number of windows), after conditioning the triple on each hub."""
    states, shapes = layer.paths.states, tuple(config['shapes'])
    members = list(triple)
    span = range(interval[0], interval[1])
    out = {}
    for hub in hubs:
        w = [given(layer, t, windows[t], hub, config)[:, members] for t in span]
        zh = [scan.normal_scores(windows[t][:, [hub]]) for t in span]
        orient = []
        for c, (a, b) in ((2, (0, 1)), (1, (0, 2)), (0, (1, 2))):
            first = paths.fit_edge_paths([x[:, [a, b]] for x in w], [x[:, [c, c]] for x in w], states,
                                         switch=config['pair_switch'], nu_grid=config['pair_nu_grid'])
            total = np.zeros(len(shapes))
            for s, x in enumerate(w):
                h = paths.apply_states(states, first.choice[s], first.par[s], first.nu[s], x[:, [a, b]], x[:, [c, c]])[1]
                scores = scan.normal_scores(h)
                zc = scan.normal_scores(x[:, [c]])
                total += statistic(scores[:, [0]], scores[:, [1]], scan.shape_functions(zc[:, 0], shapes),
                                   np.column_stack((zh[s], zc)), shapes, config)[:, 0, 0]
            orient.append(float(np.max(np.abs(total)) / np.sqrt(len(w))))
        out[int(hub)] = min(orient)
    return out


def confirm(layer, windows, found: List[dict], config=CONFIG):
    """Explain-away check of P1 (`pairscan.confirm`) with the P2 statistic.

    One difference: P1's list of variables to condition on (`pairscan.conditioning_candidates`) can contain a
    member of the candidate itself when the system has fewer other variables than it asks for (five variables:
    always).  In P1 that adds a redundant test.  With standardised transforms it would be degenerate (a variable
    standardised against itself), so members are removed here and in `confirm_fourth`."""
    if not found:
        return [], float('nan')
    dep = dependence(windows)
    found = sorted(found, key=lambda r: -r['value'])
    plans = []
    for rank, item in enumerate(found):
        members = set(item['triple'])
        hubs = pairscan.conditioning_candidates(dep, item['triple'], config['explain_away_per_member'])
        for stronger in found[:rank]:
            if len(members & set(stronger['triple'])) == 2:
                hubs.append(int((set(stronger['triple']) - members).pop()))
        plans.append([h for h in dict.fromkeys(hubs) if h not in members])
    threshold = float(norm.isf(config['level'] / (2.0 * max(sum(len(p) for p in plans), 1) * 3 * len(config['shapes']))))
    out = []
    for item, hubs in zip(found, plans):
        values = hub_check(layer, windows, item['triple'], item['interval'], hubs, config)
        weakest = min(values, key=values.get) if values else None
        out.append(dict(item, hubs=values, weakest_hub=None if weakest is None else int(weakest),
                        confirmed=bool(weakest is None or values[weakest] > threshold)))
    return out, threshold


def _fourth_task(args):
    """All quadruples whose first conditioning variable is k (second l > k): entries above the screen."""
    layer, windows, k, config, screen = args
    states = layer.paths.states
    d, steps = layer.d, len(windows)
    others = np.array([v for v in range(d) if v != k])
    w = [given(layer, t, u, k, config)[:, others] for t, u in enumerate(windows)]
    zk = [scan.normal_scores(u[:, [k]]) for u in windows]
    pa, pb = np.triu_indices(d - 1, 1)
    fit = paths.fit_edge_paths([x[:, pa] for x in w], [x[:, pb] for x in w], states, switch=config['pair_switch'],
                               nu_grid=(5.0,))
    found = []
    for li in range(d - 1):
        l = int(others[li])
        if l < k:
            continue
        forward, reverse = np.nonzero(pb == li)[0], np.nonzero(pa == li)[0]
        members = np.concatenate((others[pa[forward]], others[pb[reverse]]))
        stat = np.empty((steps, members.size, members.size))
        for t in range(steps):
            x = w[t]
            parts = []
            if forward.size:
                parts.append(paths.apply_states(states, fit.choice[t, forward], fit.par[t, forward], fit.nu[t, forward],
                                                x[:, pa[forward]], x[:, pb[forward]])[1])
            if reverse.size:
                parts.append(paths.apply_states(states, fit.choice[t, reverse], fit.par[t, reverse], fit.nu[t, reverse],
                                                x[:, pa[reverse]], x[:, pb[reverse]], reverse=True)[1])
            zl = scan.normal_scores(x[:, [li]])
            g = zk[t][:, 0] * zl[:, 0]
            stat[t] = statistic(scan.normal_scores(np.column_stack(parts)), None, g - g.mean(),
                                np.column_stack((zk[t], zl)), ('product',), config)[0]
        strength, start, stop = scan.interval_scan(np.nan_to_num(stat))
        ii, jj = np.nonzero(np.triu(strength > screen, 1))
        for a_, b_ in zip(ii, jj):
            found.append((int(members[a_]), int(members[b_]), int(k), l, float(strength[a_, b_]), int(start[a_, b_]),
                          int(stop[a_, b_])))
    return found


def fourth_order_scan(layer, windows, config=CONFIG, workers=1):
    """Exhaustive fourth-order scan (as `pairscan.fourth_order_scan`) with the P2 statistic."""
    d, steps = layer.d, len(windows)
    threshold = scan.interval_threshold(steps, d * (d - 1) * (d - 2) * (d - 3) // 24, config['level'])
    tasks = [(layer, windows, k, config, threshold) for k in range(d - 1)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_fourth_task, tasks, chunksize=1))
    else:
        parts = [_fourth_task(task) for task in tasks]
    return [entry for part in parts for entry in part], threshold


def quadruples(entries, config=CONFIG):
    """Quadruples with all six choices of the pair above the threshold: {quadruple: dict(weakest, strongest entry,
    interval)}.  With ``intersect_runs`` the interval is the intersection of the six and must not be empty."""
    table = {}
    for i, j, k, l, value, start, stop in entries:
        table.setdefault(tuple(sorted((i, j, k, l))), []).append((value, i, j, k, l, start, stop))
    out = {}
    for q, rows in table.items():
        if len(rows) != 6:
            continue
        strongest = max(rows)
        interval = (strongest[5], strongest[6])
        if config['intersect_runs']:
            interval = (max(r[5] for r in rows), min(r[6] for r in rows))
            if interval[1] <= interval[0]:
                continue
        out[q] = dict(weakest=min(rows)[0], strongest=strongest, interval=interval)
    return out, sum(len(rows) == 6 for rows in table.values()), len(table)


def conditional_fourth(layer, windows, entry, interval, hub, config=CONFIG):
    """Statistic of one choice (i, j | k, l) over the interval after all four variables are conditioned on hub."""
    states = layer.paths.states
    _, i, j, k, l, _, _ = entry
    span = range(interval[0], interval[1])
    w = [given(layer, t, windows[t], hub, config)[:, [i, j, k, l]] for t in span]
    zh = [scan.normal_scores(windows[t][:, [hub]]) for t in span]
    first = paths.fit_edge_paths([x[:, [0, 1, 3]] for x in w], [x[:, [2, 2, 2]] for x in w], states,
                                 switch=config['pair_switch'], nu_grid=(5.0,))
    second_inputs = [restandardize(paths.apply_states(states, first.choice[s], first.par[s], first.nu[s], x[:, [0, 1, 3]],
                                                      x[:, [2, 2, 2]])[1],
                                   np.column_stack((zh[s], scan.normal_scores(x[:, [2]]))), config)
                     for s, x in enumerate(w)]                                                  # i, j, l given (hub, k)
    second = paths.fit_edge_paths([h[:, [0, 1]] for h in second_inputs], [h[:, [2, 2]] for h in second_inputs], states,
                                  switch=config['pair_switch'], nu_grid=(5.0,))
    total = 0.0
    for s, (x, h) in enumerate(zip(w, second_inputs)):
        a = scan.normal_scores(paths.apply_states(states, second.choice[s], second.par[s], second.nu[s], h[:, [0, 1]],
                                                  h[:, [2, 2]])[1])
        cond = np.column_stack((zh[s], scan.normal_scores(x[:, [2]]), scan.normal_scores(h[:, [2]])))
        g = cond[:, 1] * cond[:, 2]
        total += statistic(a[:, [0]], a[:, [1]], g - g.mean(), cond, ('product',), config)[0, 0, 0]
    return float(abs(total) / np.sqrt(len(w)))


def confirm_fourth(layer, windows, table, config=CONFIG):
    """Explain-away check for quadruples (as `pairscan.confirm_fourth`) with the P2 statistic."""
    passed = sorted(table.items(), key=lambda item: -item[1]['weakest'])
    if not passed:
        return [], float('nan')
    dep = dependence(windows)
    plans = []
    for rank, (q, v) in enumerate(passed):
        hubs = []
        for member in q:
            strength = dep[:, member].copy()
            strength[list(q)] = -1.0
            hubs.extend(int(h) for h in np.argsort(-strength)[:config['explain_away_per_member_fourth']])
        for other, _ in passed[:rank]:
            if len(set(q) & set(other)) == 3:
                hubs.append(int((set(other) - set(q)).pop()))
        plans.append([h for h in dict.fromkeys(hubs) if h not in q])
    threshold = float(norm.isf(config['level'] / (2.0 * max(sum(len(p) for p in plans), 1))))
    out = []
    for (q, v), hubs in zip(passed, plans):
        values = {h: conditional_fourth(layer, windows, v['strongest'], v['interval'], h, config) for h in hubs}
        weakest = min(values, key=values.get) if values else None
        out.append(dict(quadruple=q, value=v['weakest'], interval=v['interval'], hubs=values,
                        weakest_hub=None if weakest is None else int(weakest),
                        confirmed=bool(weakest is None or values[weakest] > threshold)))
    return out, threshold


# ----------------------------------------------------------------------------------------------------------------
# Sequential confirmation with gated transforms
# ----------------------------------------------------------------------------------------------------------------
def explainers(subset, triples):
    """{variable m outside subset: confirmed triples that contain m and two members of subset}."""
    out = {}
    for item in triples:
        common = set(item['subset']) & set(subset)
        if len(common) == 2 and len(item['subset']) == 3:
            out.setdefault(int((set(item['subset']) - common).pop()), []).append(tuple(item['subset']))
    return out


def gated_value(windows, subset, run, explainer, triples, config=CONFIG):
    """Statistic of ``subset`` over ``run`` in a vine rooted at ``explainer`` that models ``triples``.

    For every choice of the conditioned pair (i, j) of the subset: root order (explainer, the other members, i, j);
    a dynamic C-vine in which the given triples may be modelled (`cvine.fit` with ``allowed``); the statistic of
    the last edge against the other members' coordinate (linear and quadratic for a triple, the product for a
    quadruple), summed over the run.  Returns the smallest over the choices of |sum| / sqrt(number of windows).
    """
    members = sorted(subset)
    span = range(run[0], run[1])
    shapes = tuple(config['shapes'])
    values = []
    for pair in combinations(members, 2):
        columns = [explainer] + [v for v in members if v not in pair] + list(pair)
        position = {v: i for i, v in enumerate(columns)}
        sub = [np.asarray(windows[t])[:, columns] for t in span]
        allowed = {tuple(sorted(position[v] for v in triple)): np.ones(len(sub), dtype=bool) for triple in triples}
        model = cvine.fit(sub, order=list(range(len(columns))), states=pipeline.states(config),
                          switch=config['pair_switch'], nu_grid=config['pair_nu_grid'], allowed=allowed)
        level = len(columns) - 2
        total = 0.0
        for s, u in enumerate(sub):
            pits = cvine.forward(model, s, u, keep=True)['pits']
            top = scan.normal_scores(pits[level][:, [level, level + 1]])
            cond = scan.normal_scores(np.column_stack([pits[m][:, m] for m in range(level)]))
            if len(members) == 3:
                g, kinds = scan.shape_functions(cond[:, 1], shapes), shapes
            else:
                g, kinds = cond[:, 1] * cond[:, 2], ('product',)
                g = (g - g.mean())[:, None]
            total = total + statistic(top[:, [0]], top[:, [1]], g, cond, kinds, config)[:, 0, 0]
        values.append(float(np.max(np.abs(total)) / np.sqrt(len(sub))))
    return min(values)


def sequential(windows, triples, quadruples_, config=CONFIG):
    """Strongest first: a candidate is kept unless, for some variable m that is the third member of already kept
    triples sharing two variables with it, its statistic in the vine that models those triples falls below the
    threshold (Bonferroni over the planned tests).  Returns kept triples, kept quadruples and a log."""
    triples = sorted(triples, key=lambda r: -r['value'])
    quadruples_ = sorted(quadruples_, key=lambda r: -r['value'])
    planned = (sum(len(explainers(c['subset'], triples[:rank])) for rank, c in enumerate(triples))
               + sum(len(explainers(q['subset'], triples)) for q in quadruples_))
    threshold = float(norm.isf(config['level'] / (2.0 * max(planned, 1) * 6 * len(config['shapes']))))
    kept, log = [], []
    for item in triples:
        values = {m: gated_value(windows, item['subset'], item['run'], m, found, config)
                  for m, found in explainers(item['subset'], kept).items()}
        ok = all(v > threshold for v in values.values())
        log.append(dict(subset=list(item['subset']), values={int(m): v for m, v in values.items()}, kept=bool(ok)))
        if ok:
            kept.append(item)
    kept4 = []
    for item in quadruples_:
        values = {m: gated_value(windows, item['subset'], item['run'], m, found, config)
                  for m, found in explainers(item['subset'], kept).items()}
        ok = all(v > threshold for v in values.values())
        log.append(dict(subset=list(item['subset']), values={int(m): v for m, v in values.items()}, kept=bool(ok)))
        if ok:
            kept4.append(item)
    return kept, kept4, dict(threshold=threshold, tests=log)


def detect(windows: Sequence[np.ndarray], config=CONFIG, workers=1) -> pipeline.Detection:
    """Detection step of P2.  windows: list of (n_t, d) pseudo-observations (within-window ranks)."""
    d = windows[0].shape[1]
    layer = pairscan.fit_pair_layer(windows, states=pipeline.states(config), switch=config['pair_switch'],
                                    nu_grid=config['pair_nu_grid'], workers=workers)
    stat = scan_all(layer, windows, config, workers)
    loose, _ = pairscan.any_orientation(stat, config['level'])
    found, _, before = candidates(stat, config)
    checked, _ = confirm(layer, windows, found, config)
    triples = [dict(subset=tuple(c['triple']), run=tuple(c['interval']), value=c['value']) for c in checked if c['confirmed']]
    counts = dict(triples_any_choice=len(loose), triples_all_choices=before, triples_common_run=len(found),
                  triples_explain_away=len(triples))
    quads = []
    if config['fourth_order'] and d >= 4:
        entries, _ = fourth_order_scan(layer, windows, config, workers)
        table, all_six, any_choice = quadruples(entries, config)
        checked4, _ = confirm_fourth(layer, windows, table, config)
        quads = [dict(subset=tuple(c['quadruple']), run=tuple(c['interval']), value=c['value'])
                 for c in checked4 if c['confirmed']]
        counts.update(quadruples_any_choice=any_choice, quadruples_all_choices=all_six, quadruples_common_run=len(table),
                      quadruples_explain_away=len(quads))
    log = None
    if config['sequential']:
        triples, quads, log = sequential(windows, triples, quads, config)
    counts.update(triples_confirmed=len(triples), quadruples_confirmed=len(quads))
    detection = pipeline.Detection(triples, quads, counts, len(windows), layer)
    detection.sequential_log = log
    return detection
