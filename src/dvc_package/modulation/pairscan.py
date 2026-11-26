"""Exhaustive third-order scan for many variables.

1. Pair layer: a switching pair copula for every pair of variables (the DVC-switch edge objective, all pairs).
2. Scan: for every ordered triple (i, j | k) and window, the modulation statistic of the pair (i, j) after each has
   been transformed by its pair copula with k; aggregated over contiguous windows.
3. Confirmation of candidate triples: every one of the three conditioning choices must reject, and must still
   reject after the three variables are first conditioned on each of the variables most strongly dependent on
   them.  The last step removes triples that are explained by a common observed variable (leaves of a hub).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import norm

from . import families as fam
from . import paths, scan


@dataclass
class PairLayer:
    pairs: np.ndarray           # (P, 2) with i < k
    paths: paths.EdgePaths
    d: int

    def transforms(self, t, u, eps=fam.EPS):
        """w[:, i, k] = P(U_i <= u_i | U_k = u_k) at window t (diagonal = u)."""
        u = np.asarray(u, dtype=np.float64)
        i, k = self.pairs.T
        e = self.paths
        w = np.repeat(u[:, :, None], self.d, axis=2)
        w[:, i, k] = paths.apply_states(e.states, e.choice[t], e.par[t], e.nu[t], u[:, i], u[:, k], eps=eps)[1]
        w[:, k, i] = paths.apply_states(e.states, e.choice[t], e.par[t], e.nu[t], u[:, i], u[:, k], reverse=True, eps=eps)[1]
        return w


def given(layer, t, u, k, eps=fam.EPS):
    """Transforms of every variable given variable k at window t: (n, d); column k holds u_k itself."""
    u = np.asarray(u, dtype=np.float64)
    e, out = layer.paths, u.copy()
    first = np.nonzero(layer.pairs[:, 1] == k)[0]                 # pairs (i, k), i < k: forward direction
    second = np.nonzero(layer.pairs[:, 0] == k)[0]                # pairs (k, i), i > k: reverse direction
    if first.size:
        i = layer.pairs[first, 0]
        out[:, i] = paths.apply_states(e.states, e.choice[t, first], e.par[t, first], e.nu[t, first], u[:, i],
                                       np.repeat(u[:, [k]], first.size, axis=1), eps=eps)[1]
    if second.size:
        i = layer.pairs[second, 1]
        out[:, i] = paths.apply_states(e.states, e.choice[t, second], e.par[t, second], e.nu[t, second],
                                       np.repeat(u[:, [k]], second.size, axis=1), u[:, i], reverse=True, eps=eps)[1]
    return out


def _fit_window(args):
    u, pairs, states, nu_grid = args
    return fam.fit_states(u[:, pairs[:, 0]], u[:, pairs[:, 1]], states, nu_grid=nu_grid)


def fit_pair_layer(windows: Sequence[np.ndarray], states=None, switch=0.05, drift=0.0, activation=0.0,
                   nu_grid=(3.0, 5.0, 10.0), workers=1) -> PairLayer:
    states = list(states) if states is not None else fam.default_states()
    d = windows[0].shape[1]
    pairs = np.array([(i, k) for i in range(d) for k in range(i + 1, d)])
    tasks = [(np.asarray(u, dtype=np.float64), pairs, states, nu_grid) for u in windows]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fits = list(pool.map(_fit_window, tasks))
    else:
        fits = [_fit_window(task) for task in tasks]
    par, nu, loglik = (np.stack([f[i] for f in fits]) for i in range(3))
    cost = paths.local_costs(loglik, [u.shape[0] for u in windows], states, activation)
    choice, objective = paths.viterbi(cost, paths.parameter_distance(par, nu), switch=switch, drift=drift)
    take = lambda a: np.take_along_axis(a, choice[:, None, :], axis=1)[:, 0, :]
    return PairLayer(pairs, paths.EdgePaths(states, choice, take(par), take(nu), take(loglik), objective), d)


def _scan_window(args):
    layer, t, u, shapes, rerank, hac_lags, standardize, slopes = args
    w = layer.transforms(t, u)
    n, d, _ = w.shape
    scores = scan.normal_scores(w.reshape(n, d * d), rerank).reshape(n, d, d)
    z = scan.normal_scores(u, rerank)
    return scan.third_order_scan(scores, z, shapes, hac_lags=hac_lags, standardize=standardize,
                                 slopes=slopes).astype(np.float32)


def scan_all(layer: PairLayer, windows, shapes=scan.SHAPES, rerank=True, hac_lags=0, workers=1, standardize=0,
             slopes=False):
    """Statistics (T, shapes, d, d, d) indexed [t, shape, i, j, k]; with ``slopes`` the effect sizes instead."""
    tasks = [(layer, t, u, shapes, rerank, hac_lags, standardize, slopes) for t, u in enumerate(windows)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return np.stack(list(pool.map(_scan_window, tasks)))
    return np.stack([_scan_window(task) for task in tasks])


interval_threshold = scan.interval_threshold


def candidates(stat, level=0.05, eligible=None):
    """Triples in which all three conditioning choices exceed the familywise interval threshold.

    Returns a list of dicts (triple, value = smallest of the three orientation maxima, interval of the weakest
    orientation) and the threshold.  The triple-level p-value is the largest of its three orientation p-values
    (intersection-union test), so the multiplicity is C(d, 3) triples times shapes times intervals.
    ``eligible(i, j, k)`` restricts the triples that are examined (and counted in the multiplicity).
    """
    steps, n_shapes, d = stat.shape[:3]
    strength, start, stop = scan.interval_scan(np.nan_to_num(stat.astype(np.float64)))
    best_shape = strength.argmax(axis=0)                                  # (d, d, d)
    best = np.take_along_axis(strength, best_shape[None], axis=0)[0]
    start = np.take_along_axis(start, best_shape[None], axis=0)[0]
    stop = np.take_along_axis(stop, best_shape[None], axis=0)[0]
    n_triples = d * (d - 1) * (d - 2) // 6
    if eligible is not None:
        n_triples = sum(bool(eligible(i, j, k)) for i in range(d) for j in range(i + 1, d) for k in range(j + 1, d))
    threshold = interval_threshold(steps, n_shapes * max(n_triples, 1), level)
    out = []
    for i, j, k in zip(*np.nonzero(np.triu(np.ones((d, d)), 1)[:, :, None] * (best > threshold))):
        if not (i < j < k) or (eligible is not None and not eligible(int(i), int(j), int(k))):
            continue
        values = (best[i, j, k], best[i, k, j], best[j, k, i])
        if min(values) > threshold:
            weakest = int(np.argmin(values))
            a, b, c = ((i, j, k), (i, k, j), (j, k, i))[weakest]
            out.append(dict(triple=(int(i), int(j), int(k)), value=float(min(values)),
                            values=[float(v) for v in values], interval=(int(start[a, b, c]), int(stop[a, b, c]))))
    return sorted(out, key=lambda r: -r['value']), threshold


def any_orientation(stat, level=0.05):
    """Triples with at least one conditioning choice above the threshold (no orientation or hub rule)."""
    steps, n_shapes, d = stat.shape[:3]
    best = scan.interval_scan(np.nan_to_num(stat.astype(np.float64)))[0].max(axis=0)
    threshold = interval_threshold(steps, n_shapes * d * (d - 1) * (d - 2) // 2, level)
    found = set()
    for i, j, k in zip(*np.nonzero(best > threshold)):
        found.add(tuple(sorted((int(i), int(j), int(k)))))
    return found, threshold


def dependence(windows):
    """Mean over windows of the absolute correlation of normal scores."""
    return np.mean([np.abs(np.corrcoef(ndtri(fam.clip01(u)), rowvar=False)) for u in windows], axis=0)


def hub_check(layer: PairLayer, windows, triple, interval, hubs: Sequence[int], states=None, switch=0.05,
              shapes=scan.SHAPES, rerank=True, hac_lags=0):
    """Smallest orientation statistic of the triple, over the detected interval, after conditioning on each hub.

    Returns {hub: value}; value is min over the three conditioning choices of max over shapes of
    |sum of window statistics| / sqrt(number of windows).
    """
    states = list(states) if states is not None else layer.paths.states
    members = list(triple)
    span = range(interval[0], interval[1])
    out = {}
    for hub in hubs:
        w = [given(layer, t, windows[t], hub)[:, members] for t in span]              # (n, 3) given the hub
        zh = None
        orient = []
        for c, (a, b) in ((2, (0, 1)), (1, (0, 2)), (0, (1, 2))):
            first = paths.fit_edge_paths([x[:, [a, b]] for x in w], [x[:, [c, c]] for x in w], states, switch=switch,
                                         nu_grid=(3.0, 5.0, 10.0))
            total = np.zeros(len(shapes))
            for s, x in enumerate(w):
                h = paths.apply_states(states, first.choice[s], first.par[s], first.nu[s], x[:, [a, b]], x[:, [c, c]])[1]
                scores = scan.normal_scores(h, rerank)
                g = scan.shape_functions(scan.normal_scores(x[:, [c]], rerank)[:, 0], shapes)
                total += scan.moment_scan(scores[:, 0], scores[:, 1], g, hac_lags=hac_lags)[:, 0, 0]
            orient.append(float(np.max(np.abs(total)) / np.sqrt(len(w))))
        out[int(hub)] = min(orient)
    return out


def conditioning_candidates(dep, triple, per_member=3):
    """Variables to condition on when checking a triple: for each member the per_member most dependent other
    variables, and the variables most dependent on at least two members."""
    members = list(triple)
    chosen = []
    for member in members:
        strength = dep[:, member].copy()
        strength[members] = -1.0
        chosen.extend(int(h) for h in np.argsort(-strength)[:per_member])
    common = np.sort(dep[:, members], axis=1)[:, 1]
    common[members] = -1.0
    chosen.extend(int(h) for h in np.argsort(-common)[:per_member])
    return list(dict.fromkeys(chosen))


def confirm(layer: PairLayer, windows, found: List[dict], per_member=3, level=0.05, **kwargs):
    """Explain-away check of every candidate triple.

    A candidate is confirmed when all three conditioning choices still reject after the three variables are first
    conditioned on each of: (a) the variables most dependent on any of its members (a common driver, or a variable
    through which one member is only indirectly involved), and (b) the third variable of every stronger candidate
    that shares two variables with it (a triple produced by a correlate of a true member).  The threshold is
    Bonferroni over all these confirmation tests.
    """
    if not found:
        return [], float('nan')
    dep = dependence(windows)
    found = sorted(found, key=lambda r: -r['value'])
    plans = []
    for rank, item in enumerate(found):
        members = set(item['triple'])
        hubs = conditioning_candidates(dep, item['triple'], per_member)
        for stronger in found[:rank]:
            if len(members & set(stronger['triple'])) == 2:
                hubs.append(int((set(stronger['triple']) - members).pop()))
        plans.append(list(dict.fromkeys(hubs)))
    threshold = float(norm.isf(level / (2.0 * sum(len(p) for p in plans) * 3 * 2)))
    out = []
    for item, hubs in zip(found, plans):
        values = hub_check(layer, windows, item['triple'], item['interval'], hubs, **kwargs)
        weakest = min(values, key=values.get)
        out.append(dict(item, hubs=values, weakest_hub=int(weakest), confirmed=bool(values[weakest] > threshold)))
    return out, threshold


# ----------------------------------------------------------------------------------------------------------------
# Fourth order: all (i, j | k, l) with the product of the two conditioning coordinates as modulator.
# ----------------------------------------------------------------------------------------------------------------
def _fourth_task(args):
    """All quadruples whose first conditioning variable is k (second l > k).  Returns arrays of entries above screen."""
    layer, windows, k, states, switch, nu_grid, screen, rerank = args
    d, steps = layer.d, len(windows)
    others = np.array([v for v in range(d) if v != k])
    w = [given(layer, t, u, k)[:, others] for t, u in enumerate(windows)]             # (n, d - 1) given k
    zk = [scan.normal_scores(u[:, [k]], rerank)[:, 0] for u in windows]
    pa, pb = np.triu_indices(d - 1, 1)
    fit = paths.fit_edge_paths([x[:, pa] for x in w], [x[:, pb] for x in w], states, switch=switch, nu_grid=nu_grid)
    found = []
    for li in range(d - 1):
        l = int(others[li])
        if l < k:
            continue
        forward, reverse = np.nonzero(pb == li)[0], np.nonzero(pa == li)[0]
        members = np.concatenate((others[pa[forward]], others[pb[reverse]]))            # variables other than k, l
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
            a = scan.normal_scores(np.column_stack(parts), rerank)
            g = zk[t] * scan.normal_scores(x[:, [li]], rerank)[:, 0]
            stat[t] = scan.moment_scan(a, a, g - g.mean())[0]
        strength, start, stop = scan.interval_scan(stat)
        ii, jj = np.nonzero(np.triu(strength > screen, 1))
        for a_, b_ in zip(ii, jj):
            found.append((int(members[a_]), int(members[b_]), int(k), l, float(strength[a_, b_]), int(start[a_, b_]),
                          int(stop[a_, b_])))
    return found


def fourth_order_scan(layer: PairLayer, windows, states=None, switch=0.05, nu_grid=(5.0,), level=0.05, rerank=True,
                      workers=1):
    """Exhaustive fourth-order scan.  Every quadruple is examined in its six choices of the pair (i, j); the other
    two variables are conditioned on in increasing index order and their product is the modulator.

    Returns (entries, threshold): entries are tuples (i, j, k, l, interval-scan value, start, stop) above the
    familywise threshold (Bonferroni over C(d, 4) quadruples and all intervals; the quadruple-level p-value is the
    largest of its six, so a quadruple is declared only when all six entries are present)."""
    states = list(states) if states is not None else layer.paths.states
    d, steps = layer.d, len(windows)
    threshold = interval_threshold(steps, d * (d - 1) * (d - 2) * (d - 3) // 24, level)
    tasks = [(layer, windows, k, states, switch, nu_grid, threshold, rerank) for k in range(d - 1)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_fourth_task, tasks, chunksize=1))
    else:
        parts = [_fourth_task(task) for task in tasks]
    return [entry for part in parts for entry in part], threshold


def quadruples(entries):
    """Group entries by quadruple.  Returns {quadruple: dict(count, weakest value, strongest entry)}."""
    table = {}
    for i, j, k, l, value, start, stop in entries:
        table.setdefault(tuple(sorted((i, j, k, l))), []).append((value, i, j, k, l, start, stop))
    return {q: dict(count=len(v), weakest=min(v)[0], strongest=max(v)) for q, v in table.items()}


def conditional_fourth(layer: PairLayer, windows, entry, hub, states=None, switch=0.05, nu_grid=(5.0,), rerank=True):
    """Statistic of one orientation (i, j | k, l) over its interval after all four variables are first conditioned
    on hub: |sum of window statistics| / sqrt(number of windows)."""
    states = list(states) if states is not None else layer.paths.states
    value, i, j, k, l, start, stop = entry
    total = 0.0
    span = range(start, stop)
    w = [given(layer, t, windows[t], hub)[:, [i, j, k, l]] for t in span]              # given hub
    first = paths.fit_edge_paths([x[:, [0, 1, 3]] for x in w], [x[:, [2, 2, 2]] for x in w], states, switch=switch, nu_grid=nu_grid)
    second_inputs = []
    for s, x in enumerate(w):
        h = paths.apply_states(states, first.choice[s], first.par[s], first.nu[s], x[:, [0, 1, 3]], x[:, [2, 2, 2]])[1]
        second_inputs.append(h)                                                         # i, j, l given (hub, k)
    second = paths.fit_edge_paths([h[:, [0, 1]] for h in second_inputs], [h[:, [2, 2]] for h in second_inputs], states,
                                  switch=switch, nu_grid=nu_grid)
    for s, (x, h) in enumerate(zip(w, second_inputs)):
        a = scan.normal_scores(paths.apply_states(states, second.choice[s], second.par[s], second.nu[s], h[:, [0, 1]],
                                                  h[:, [2, 2]])[1], rerank)
        g = scan.normal_scores(x[:, [2]], rerank)[:, 0] * scan.normal_scores(h[:, [2]], rerank)[:, 0]
        total += scan.moment_scan(a[:, 0], a[:, 1], g - g.mean())[0, 0, 0]
    return float(abs(total) / np.sqrt(len(w)))


def confirm_fourth(layer: PairLayer, windows, table, per_member=2, level=0.05, **kwargs):
    """Explain-away check for quadruples that passed all six orientations: the strongest orientation must still
    reject after conditioning on (a) the variables most dependent on each member and (b) the missing member of
    every stronger candidate that shares three variables."""
    passed = sorted(((q, v) for q, v in table.items() if v['count'] == 6), key=lambda item: -item[1]['weakest'])
    if not passed:
        return [], float('nan')
    dep = dependence(windows)
    plans = []
    for rank, (q, v) in enumerate(passed):
        hubs = []
        for member in q:
            strength = dep[:, member].copy()
            strength[list(q)] = -1.0
            hubs.extend(int(h) for h in np.argsort(-strength)[:per_member])
        for other, _ in passed[:rank]:
            if len(set(q) & set(other)) == 3:
                hubs.append(int((set(other) - set(q)).pop()))
        plans.append(list(dict.fromkeys(hubs)))
    threshold = float(norm.isf(level / (2.0 * sum(len(p) for p in plans))))
    out = []
    for (q, v), hubs in zip(passed, plans):
        values = {h: conditional_fourth(layer, windows, v['strongest'], h, **kwargs) for h in hubs}
        weakest = min(values, key=values.get)
        out.append(dict(quadruple=q, value=v['weakest'], interval=(v['strongest'][5], v['strongest'][6]), hubs=values,
                        weakest_hub=int(weakest), confirmed=bool(values[weakest] > threshold)))
    return out, threshold
