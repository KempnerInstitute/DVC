"""E4.  Many variables with unknown interacting triples.

Background (pairwise only, not Gaussian): blocks of ten variables, each a star around a hub with a pair-copula family
that differs by block (Clayton, Gumbel, Student-t, Gaussian, Frank; Kendall tau 0.25-0.45 drawn per edge); hubs are
linked in a Gaussian chain (rho 0.4).  Leaves of one hub are a pairwise law whose three-variable margins are not
simplified, so the background contains thousands of triples that a scan without the hub rule would flag.
Planted (windows 3-6 of 9): K triples (i, j, k): leaves i, j of different blocks whose conditional correlation given
everything else is tanh(g * z_k), k a third variable.  In the null version nothing is planted.
With FOURTH = 1, three quadruples (i, j, k, l) are planted as well (correlation tanh(g * z_k * z_l), all four
variables in different blocks, disjoint from the triples) and the exhaustive fourth-order scan is run.
Usage: e4_large.py OUT.json D N G SEEDS WORKERS [FOURTH]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvc_package.modulation import families as fam, pairscan  # noqa: E402

T, EPISODE, BLOCK, PLANTED = 9, range(3, 7), 10, 5
FAMILIES = ('clayton0', 'gumbel0', 'student', 'gaussian', 'frank')


def tau_parameter(state, tau):
    if state in ('gaussian', 'student'):
        return np.sin(np.pi*tau/2)
    if state == 'clayton0':
        return 2*tau/(1-tau)
    if state == 'gumbel0':
        return 1/(1-tau)
    return {0.25: 2.37, 0.35: 3.51, 0.45: 4.87}[min((0.25, 0.35, 0.45), key=lambda v: abs(v-tau))]   # Frank


def design(d, seed, g, fourth=0):
    rng = np.random.default_rng([seed, d, 77])
    blocks = d//BLOCK
    hubs = [b*BLOCK for b in range(blocks)]
    edges = {}
    for b in range(blocks):
        state = FAMILIES[b % len(FAMILIES)]
        for leaf in range(b*BLOCK+1, (b+1)*BLOCK):
            edges[leaf] = (state, tau_parameter(state, rng.choice((0.25, 0.35, 0.45))))
    planted = []
    used = set(hubs)
    while g > 0 and len(planted) < PLANTED:
        i, j, k = (int(v) for v in rng.choice(d, 3, replace=False))
        if {i, j, k} & used or i//BLOCK == j//BLOCK:
            continue
        planted.append((i, j, k))
        used |= {i, j, k}
    quadruples = []
    while g > 0 and len(quadruples) < 3 * fourth:
        i, j, k, l = (int(v) for v in rng.choice(d, 4, replace=False))
        if {i, j, k, l} & used or len({v//BLOCK for v in (i, j, k, l)}) < 4:
            continue
        quadruples.append((i, j, k, l))
        used |= {i, j, k, l}
    return dict(d=d, hubs=hubs, edges=edges, planted=planted, quadruples=quadruples)


def sample(spec, n, t, g, rng):
    d = spec['d']
    u = np.empty((n, d))
    q = rng.uniform(size=(n, d))
    previous = None
    for hub in spec['hubs']:
        u[:, hub] = q[:, hub] if previous is None else fam.h_inverse('gaussian', q[:, hub], u[:, previous], 0.4)
        previous = hub
    modulated = {}
    if t in EPISODE:
        modulated = {j: (i, (k,)) for i, j, k in spec['planted']}
        modulated.update({j: (i, (k, l)) for i, j, k, l in spec['quadruples']})
    done = set(spec['hubs'])
    pending = [v for v in range(d) if v not in done]
    for _ in range(3):                                             # modulators must be drawn before the pairs they drive
        later = []
        for v in pending:
            if v in modulated and not (modulated[v][0] in done and all(m in done for m in modulated[v][1])):
                later.append(v)
                continue
            if v in modulated:
                i, drivers = modulated[v]
                rho = np.tanh(g*np.prod(ndtri(np.clip(u[:, list(drivers)], 1e-10, 1-1e-10)), axis=1))
                q[:, v] = ndtr(rho*ndtri(q[:, i])+np.sqrt(1-rho**2)*ndtri(q[:, v]))
            state, par = spec['edges'][v]
            u[:, v] = fam.h_inverse(state, q[:, v], u[:, (v//BLOCK)*BLOCK], par, 4.0 if state == 'student' else None)
            done.add(v)
        pending = later
    assert not pending
    return u


def run(d, n, g, seed, workers, fourth=0):
    start = time.time()
    spec = design(d, seed, g, fourth)
    rng = np.random.default_rng([seed, d, n, 5])
    raw = [sample(spec, n, t, g, rng) for t in range(T)]
    windows = [np.argsort(np.argsort(x, axis=0), axis=0)/(n+1.0)+1.0/(n+1.0) for x in raw]
    layer = pairscan.fit_pair_layer(windows, workers=workers)
    t_fit = time.time()-start
    stat = pairscan.scan_all(layer, windows, workers=workers)
    t_scan = time.time()-start-t_fit
    found, threshold = pairscan.candidates(stat)
    loose, _ = pairscan.any_orientation(stat)
    checked, confirm_threshold = pairscan.confirm(layer, windows, found)
    truth = {tuple(sorted(p)) for p in spec['planted']}
    same_block = lambda tr: len({v//BLOCK for v in tr}) == 1
    confirmed = [c for c in checked if c['confirmed']]
    out = dict(d=d, n=n, g=g, seed=seed, planted=sorted(truth), threshold=threshold, confirm_threshold=confirm_threshold,
               any_orientation=len(loose), any_orientation_true=len(loose & truth),
               any_orientation_same_block=sum(same_block(tr) for tr in loose),
               all_orientations=len(found), all_orientations_true=sum(c['triple'] in truth for c in found),
               all_orientations_same_block=sum(same_block(c['triple']) for c in found),
               confirmed=len(confirmed), confirmed_true=sum(c['triple'] in truth for c in confirmed),
               confirmed_false=[c for c in confirmed if c['triple'] not in truth],
               timing_error=[[c['interval'][0]-EPISODE[0], c['interval'][1]-EPISODE[-1]-1] for c in confirmed if c['triple'] in truth],
               missed=[list(tr) for tr in truth-{c['triple'] for c in confirmed}],
               seconds=dict(pair_layer=t_fit, scan=t_scan, total=time.time()-start))
    for c in out['confirmed_false']:
        c['explained'] = [list(p) for p in truth if len(set(p) & set(c['triple'])) == 2]
    out['rejected_candidates'] = [dict(triple=c['triple'], weakest_hub=c['weakest_hub'], value=min(c['hubs'].values()),
                                       true=c['triple'] in truth) for c in checked if not c['confirmed']]
    if fourth:
        t0 = time.time()
        entries, threshold4 = pairscan.fourth_order_scan(layer, windows, workers=workers)
        table = pairscan.quadruples(entries)
        checked4, confirm4 = pairscan.confirm_fourth(layer, windows, table)
        truth4 = {tuple(sorted(q)) for q in spec['quadruples']}
        confirmed4 = [c for c in checked4 if c['confirmed']]
        contains_triple = lambda q: any(set(p) < set(q) for p in truth)
        out['fourth'] = dict(planted=sorted(truth4), threshold=threshold4, confirm_threshold=confirm4,
                             any_orientation=len(table), any_orientation_true=sum(q in truth4 for q in table),
                             any_orientation_with_planted_triple=sum(contains_triple(q) for q in table),
                             all_orientations=len(checked4), all_orientations_true=sum(c['quadruple'] in truth4 for c in checked4),
                             confirmed=len(confirmed4), confirmed_true=sum(c['quadruple'] in truth4 for c in confirmed4),
                             confirmed_false=[dict(c, with_planted_triple=contains_triple(c['quadruple'])) for c in confirmed4
                                              if c['quadruple'] not in truth4],
                             timing_error=[[c['interval'][0]-EPISODE[0], c['interval'][1]-EPISODE[-1]-1] for c in confirmed4
                                           if c['quadruple'] in truth4],
                             seconds=time.time()-t0)
    return out


if __name__ == '__main__':
    target, d, n, g, seeds, workers = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    fourth = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    rows = []
    for s in range(seeds):
        rows.append(run(d, n, g, 930000+s, workers, fourth))
        r = rows[-1]
        print(f"d={d} n={n} g={g} seed={r['seed']}: one orientation {r['any_orientation']} (true {r['any_orientation_true']}, same block "
              f"{r['any_orientation_same_block']}); all orientations {r['all_orientations']} (true {r['all_orientations_true']}, same block "
              f"{r['all_orientations_same_block']}); confirmed {r['confirmed']} (true {r['confirmed_true']} of {len(r['planted'])}); "
              f"timing {r['timing_error']}; seconds {r['seconds']['total']:.0f}", flush=True)
        if fourth:
            f = r['fourth']
            print(f"   fourth order: one orientation {f['any_orientation']} (true {f['any_orientation_true']}, containing a planted "
                  f"triple {f['any_orientation_with_planted_triple']}); all six {f['all_orientations']} (true {f['all_orientations_true']}); "
                  f"confirmed {f['confirmed']} (true {f['confirmed_true']} of {len(f['planted'])}); timing {f['timing_error']}; "
                  f"seconds {f['seconds']:.0f}", flush=True)
        target.write_text(json.dumps(rows))
