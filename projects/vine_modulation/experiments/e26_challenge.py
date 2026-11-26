"""E26. Validate the first assembled vine on six generator families.

Development used the earlier planted-population simulations and one dataset per law with seed 970300. The run uses new seeds (983000 + s): for every law and n three datasets with planted modulations of
strength 0.6 and one with nothing planted (`fourth`, at 100 variables: two and one).  46 datasets.

Laws (`e26_laws.py`; 12 windows; n training, n validation and n confirmation observations per window and 1,000
test observations; planted modulations of strength 0.6, or none):
  dense     background that is not a tree (a simplified regular vine with three trees and non-Gaussian pairs, cycles
            everywhere); the eight planted modulations of E18
  overlap   two triples on one pair, two pairs with one modulator, a triple with a fourth-order term on its pair, a
            fourth-order term alone, two triples that share a leaf (third and fourth order; tree background)
  moving    constant modulations (and episodes) while the pairwise dependence of planted variables on their hubs
            changes: the leaves of every second block get stronger at window 6, the chain of hubs at window 3
  serial    the law of E18 with observations that follow an autoregression (coefficient 0.8) inside every window
  margins   the law of E18 with margins that change: a third of the variables drift in location and scale, a third
            change the shape of their margin at window 6
  fourth    four fourth-order episodes and four triples among 100 variables

Pipeline: `vinemodel.fit` with its defaults (margins by the rule, features chosen per edge, install margin of two
standard errors, half-sample tests in every orientation, 399 halves, blocks chosen from the data), with
``max_order = 4`` for the laws with fourth-order terms.  Confirmation layer for presence: the frozen strict
detector `p2.detect` on the training windows (ranks within windows), with fourth order for the law `overlap`; at
100 variables (`fourth`) its exhaustive fourth-order scan is not run and it is asked for triples only.  The strict
detector takes 26 to 49 minutes per dataset at n = 1000 (E18), several times the pipeline; it is run on every
dataset at n = 300 and, at n = 1000, on the first planted dataset and on the dataset with nothing planted of
every law (decided before the run, for time).

Reported for every planted term (a subset of three or four variables): screened, kept, installed; the windows in
which it is on in the assembled model against the planted windows; the test of a change (largest p-value over the
orientations; declared by the false discovery rule at 0.05 over the installed terms of the dataset); whether the
directed reading (the orientation with the largest validation gain) is the planted pair with the planted
modulators; whether the strict detector confirms the subset.  For every installed term that is not planted:
whether it sits on the edge of a planted term, shares two variables with a planted subset, or neither; whether a
change is declared for it.  Per dataset: the margins chosen, the block length, the held-out log density of the
simplified and of the assembled vine, the subsets that the strict detector confirms and that are not planted.

Expectations written before the run.  "Installed": the term is on in the assembled vine.  Changes are declared by
the false discovery rule at 0.05 over the installed terms of a dataset; the primary rule asks every orientation
to reject, the secondary rule only the directed reading.
 (a) Planted third-order terms are installed in at least 90% of the cases at n = 1000 and 80% at n = 300 in
     `overlap`, `moving`, `margins` and `fourth`; 85% and 70% in `dense`; 80% and 60% in `serial`.
 (b) Planted fourth-order terms (`overlap`, `fourth`) are installed in at least 75% of the cases at n = 1000; no
     prediction at n = 300.
 (c) For at least 75% of the installed third-order episodes the term is on in exactly the planted windows, in the
     laws without serial dependence.
 (d) Primary rule: a change is declared for at least 90% of the installed third-order terms that change at
     n = 1000 and 75% at n = 300 (`serial`: 75% and 50%), and for at most 4 of the installed terms that do not
     change over all laws (70 are planted) and at most 2 in any law at one n.  Secondary rule: at most 10% of the
     installed terms that do not change.  Fourth-order terms: no prediction for the primary rule (only
     all-coordinate and all-own-score feature sets are tried for two modulators); secondary rule at least 75% at
     n = 1000.
 (e) `moving`: of its 24 constant terms the primary rule declares at most 2, while at n = 1000 the test in the
     vine's own orientation alone has p <= 0.05 for at least 3 of the 12 (the case of 20.3 inside the pipeline).
 (f) Installed terms that are not planted and share fewer than two variables with a planted subset: at most 2 per
     dataset on average in every law with a tree background; no bound is predicted for `dense`, whose background
     is not a simplified C-vine (they are counted).  With nothing planted: at most 2 installed terms per dataset
     in the laws with a tree background, and the assembled vine is not below the simplified vine by more than
     0.005 nats per observation in any dataset of any law.
 (g) The assembled vine has a higher held-out log density than the simplified vine in every dataset with planted
     modulations.
 (h) `margins`: at least 90% of the variables whose margin does not change get margins over all windows, at least
     75% of the drifting ones aligned margins, and at n = 1000 at least 75% of those whose shape changes margins
     within windows.
 (i) `serial`: a block length above one in every dataset, and margins over all windows for at least 90% of the
     variables.
 (j) The strict detector confirms at most one subset that is not planted over all datasets on which it is run
     in the laws with a tree background (no prediction for `dense`), and at least 85% of the planted third-order
     terms among leaves at n = 1000 in `overlap`, `moving` and `margins` (the datasets on which it is run).
 (k) The directed reading is the planted pair with the planted modulators for at least 85% of the installed
     third-order terms at n = 1000 in the laws without serial dependence.
Usage: e26_challenge.py OUT.json LAW N G SEED WORKERS [REPLICATES=399] [STRICT=1]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common  # noqa: F401                                          (sets the import path of the package)
import e26_laws as laws
from dvc_package.modulation import margins, p2, vinemodel  # noqa: E402

T, TEST = laws.T, 1000
LAWS = dict(
    dense=dict(d=50, plan='e18', background='dense'),
    overlap=dict(d=50, plan='overlap', background='tree', max_order=4),
    moving=dict(d=50, plan='constant', background='tree', moving=True),
    serial=dict(d=50, plan='e18', background='tree', serial=0.8),
    margins=dict(d=50, plan='e18', background='tree', margins='moving'),
    fourth=dict(d=100, plan='fourth', background='tree', max_order=4, strict_fourth=False))


def generate(law, n, g, seed):
    options = LAWS[law]
    spec = laws.design(options['d'], seed, g, plan=options['plan'], background=options['background'], moving=options.get('moving', False))
    rng = np.random.default_rng([seed, options['d'], n, 263])
    raw = {role: [laws.sample(spec, m, t, g, rng, serial=options.get('serial', 0.0), margins=options.get('margins')) for t in range(T)]
           for role, m in (('train', n), ('validation', n), ('test', TEST), ('confirm', n))}
    return spec, raw


def evaluate(spec, result, strict, g):
    order = result['order']
    position = {v: k for k, v in enumerate(order)}
    edges = {(e['level'], e['edge']): e for e in result['edges']}
    truth = laws.subsets(spec, g)
    planted_subsets = {term['subset'] for term in truth}
    windows = lambda pieces: sorted({t for a, b in pieces for t in range(a, b)})
    confirmed = {tuple(sorted(x['subset'])) for x in strict} if strict is not None else None
    planted = []
    for term in truth:
        home = sorted(term['subset'], key=lambda v: position[v])
        key = (position[home[-2]], position[home[-1]]-position[home[-2]]-1)
        row = dict(kind=term['kind'], order=term['order'], subset=list(term['subset']), pair=list(term['pair']), modulators=list(term['modulators']),
                   shape=term['shape'], windows_true=term['windows'], level=key[0], screened=False, kept=False, installed=False,
                   windows_on=[], p=None, vine_p=None, change=False, directed=None, strict=None if confirmed is None else term['subset'] in confirmed)
        e = edges.get(key)
        if e is not None:
            for k, s in enumerate(e['subsets']):
                if tuple(sorted(s)) != term['subset']:
                    continue
                row['screened'] = True
                row['kept'] = row['kept'] or bool(e['kept'] and e['screen_on'][k])
                if e['installed'] and e['on'][k]:
                    row['installed'] = True
                    row['windows_on'] = sorted(set(row['windows_on']) | set(windows(e['pieces'][1+k])))
                    row['features'], row['install_z'] = e['features'], e['install_z']
                    tested = e.get('test', {}).get('terms', {}).get(k)
                    if tested is not None:
                        row['p'], row['change'] = tested['p'], bool(tested['declared'])
                        row['vine_p'] = next(o['p'] for o in tested['orientations'] if o['vine'])
                        row['orientations'] = [dict(pair=o['pair'], modulators=o['modulators'], features=o['features'], gain=o['gain'], p=o['p'])
                                               for o in tested['orientations']]
                        directed = tested['directed']
                        row['directed'] = bool(sorted(directed['pair']) == sorted(term['pair']) and sorted(directed['modulators']) == sorted(term['modulators']))
                        row['directed_p'], row['change_directed'] = directed['p'], bool(directed['declared'])
        planted.append(row)
    carriers = set()
    for term in truth:
        home = sorted(term['subset'], key=lambda v: position[v])
        carriers.add((position[home[-2]], position[home[-1]]-position[home[-2]]-1))
    unplanted = []
    for e in result['edges']:
        if not e['kept']:
            continue
        on = e['on'] if e['installed'] else e['screen_on']
        for k, s in enumerate(e['subsets']):
            if on[k] and tuple(sorted(s)) not in planted_subsets:
                shared = max((len(set(s) & set(t)) for t in planted_subsets), default=0)
                kind = 'same edge' if (e['level'], e['edge']) in carriers else 'shares two' if shared >= 2 else 'neither'
                tested = e.get('test', {}).get('terms', {}).get(k, {})
                unplanted.append(dict(subset=list(s), order=len(s), shape=e['components'][k][0], kind=kind, hub=bool(set(s) & set(spec['hubs'])),
                                      installed=bool(e['installed']), z=e['z'], install_z=e.get('install_z'), state=e.get('state'),
                                      p=tested.get('p'), change=bool(tested.get('declared', False)),
                                      change_directed=bool(tested.get('directed', {}).get('declared', False)),
                                      strict=None if confirmed is None else tuple(sorted(s)) in confirmed))
    return planted, unplanted


def run(law, n, g, seed, workers, replicates=399, with_strict=True):
    start = time.time()
    options = LAWS[law]
    spec, raw = generate(law, n, g, seed)
    clock = dict(data=round(time.time()-start, 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result = vinemodel.fit(raw['train'], raw['validation'], raw['confirm'], raw['test'], replicates=replicates, seed=seed, pool=pool,
                               max_order=options.get('max_order', 3), block='auto')
    clock.update(result['seconds'])
    strict = None
    if with_strict:
        begin = time.time()
        ranked = margins.window(raw['train'])['train']
        detection = p2.detect(ranked, dict(p2.CONFIG, fourth_order=options.get('strict_fourth', options.get('max_order', 3) >= 4)), workers=workers)
        strict = [dict(subset=list(x['subset']), run=list(x['run'])) for x in detection.triples+detection.quadruples]
        clock['strict'] = round(time.time()-begin, 1)
    planted, unplanted = evaluate(spec, result, strict, g)
    truth = {term['subset'] for term in laws.subsets(spec, g)}
    summary = dict(law=law, d=options['d'], n=n, g=g, seed=seed, replicates=replicates, order=result['order'][:12], block=result['block'],
                   flags={kind: result['flags'].count(kind) for kind in ('pooled', 'aligned', 'window')},
                   margin_kinds={kind: spec['margin_kind'].count(kind) for kind in ('drift', 'shape', 'fixed')} if options.get('margins') else None,
                   margin_table={kind: {flag: sum(k == kind and f == flag for k, f in zip(spec['margin_kind'], result['flags']))
                                        for flag in ('pooled', 'aligned', 'window')} for kind in ('drift', 'shape', 'fixed')} if options.get('margins') else None,
                   inflation=result['inflation'],
                   components=result['components'], screened_edges=len(result['edges']), kept_edges=int(sum(e['kept'] for e in result['edges'])),
                   installed_edges=int(sum(e['installed'] for e in result['edges'])), scores=result['scores'], planted=planted,
                   unplanted=unplanted, strict_other=[x for x in (strict or []) if tuple(sorted(x['subset'])) not in truth], seconds=clock)
    edges = [{k: v for k, v in e.items() if k != 'coefficients'} for e in result['edges']]
    return summary, edges


def describe(s):
    gain = s['scores']['assembled']-s['scores']['simplified']
    print(f"{s['law']} d {s['d']} n {s['n']} g {s['g']} seed {s['seed']}: margins {s['flags']}, block {s['block']}; screened edges {s['screened_edges']}, "
          f"kept {s['kept_edges']}, installed {s['installed_edges']}; held-out log density per observation: simplified {s['scores']['simplified']:.4f}, "
          f"assembled {s['scores']['assembled']:.4f} (gain {gain:+.4f}); seconds {s['seconds']}", flush=True)
    for kind in sorted({(r['kind'], r['shape'], r['order']) for r in s['planted']}):
        rows = [r for r in s['planted'] if (r['kind'], r['shape'], r['order']) == kind]
        changing = kind[1] != 'constant'
        print(f"  {kind[0]:16s} order {kind[2]} {kind[1]:8s}: {len(rows)} planted; screened {sum(r['screened'] for r in rows)}, kept {sum(r['kept'] for r in rows)}, "
              f"installed {sum(r['installed'] for r in rows)}, exact windows {sum(r['installed'] and r['windows_on'] == r['windows_true'] for r in rows)}, "
              f"change declared {sum(r['change'] for r in rows)} ({'changes' if changing else 'constant'}), p {[r['p'] for r in rows]}, "
              f"in the vine's orientation {[r['vine_p'] for r in rows]}, directed reading planted {sum(bool(r['directed']) for r in rows)}, "
              f"change declared in it {sum(bool(r.get('change_directed')) for r in rows)}, "
              f"strict {sum(bool(r['strict']) for r in rows)}", flush=True)
    u = s['unplanted']
    print('  unplanted terms: ' + '; '.join(
        f"{kind}: kept {sum(x['kind'] == kind for x in u)}, installed {sum(x['installed'] for x in u if x['kind'] == kind)}, "
        f"change declared {sum(x['change'] for x in u if x['kind'] == kind)}" for kind in ('same edge', 'shares two', 'neither'))
          + f"; strict detector, subsets not planted: {len(s['strict_other'])}", flush=True)


if __name__ == '__main__':
    target, law = Path(sys.argv[1]), sys.argv[2]
    n, g, seed, workers = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    replicates = int(sys.argv[7]) if len(sys.argv) > 7 else 399
    with_strict = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    summary, edges = run(law, n, g, seed, workers, replicates, with_strict)
    target.write_text(json.dumps(dict(summary=summary, edges=edges), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    describe(summary)
