"""E31.  The second version of the pipeline (`vinemodel2.fit`) on the laws of `e31_laws.py`.

The development seed was 970500. The frozen challenge uses eight laws, n = 300 and 1000, three planted datasets (990000 to 990002) and one control (990003) per law and sample size, for 64 datasets. Counts pool the three planted datasets within each design. A change requires agreement among supported orientations and joint models, followed by the false discovery rule at 0.05 over installed terms in the dataset. Unless explicitly restricted to installed terms, recovery counts include every planted term.

Expectations (written on 16 November 2026 after the development datasets of seed 970500 and before any dataset
of the challenge was generated):
 general
  (g1) No dataset fails.
  (g2) In every dataset with planted items of the six laws other than `ushape` and `clayton` (36) the assembled
       vine has a higher held-out log density than the simplified vine, and the simplified vine is not returned.
  (g3) Nothing planted (16 datasets): no change is declared; over the 14 datasets of the laws other than
       `clayton` at most 2 terms are installed.
  (g4) Terms that are not planted and do not share two variables with a planted term that changes: a change is
       declared for at most 2 over all datasets.  (Those that do share two variables with one are counted and
       reported without a prediction.)
  (g5) The strict detector, run on the first dataset of every law and n (16), confirms at most one subset that is
       not planted.
 step (per n: 9 constant terms, 9 episodes)
  (s1) Installed: at least 16 of 18 at n = 1000 and 14 of 18 at n = 300.
  (s2) Episodes declared: at least 7 of 9 at n = 1000 and 4 of 9 at n = 300.  Constant terms declared: at most 1
       of 9 at each n.
 ushape (per n: 9 constant terms, 9 episodes)
  (u1) A term is installed where its modulator is the first of its three variables in the root order (at least
       4 of 5 of those at n = 1000 and 2 of 3 at n = 300) and not elsewhere (at most 1 at each n): an even
       modulation leaves no correlation to modulate in the two other orientations, and a vine holds a term on
       the edge of its two later variables.
  (u2) Constant terms declared: at most 1 of 9 at each n.
  (u3) Of the installed episodes, a change is declared for at least half at each n.
 clayton (per n: 9 constant terms, 9 episodes, on pairs with a Clayton dependence)
  (c1) Installed: at least 9 of 18 at n = 1000; no prediction at n = 300.
  (c2) Constant terms declared: at most 1 of 9 at each n.  At n = 1000 the assembled vine is above the simplified
       vine on held-out data in all three datasets.  Nothing planted (the pairs keep their dependence): at most 2
       installed terms over the two datasets.
 sum (per n: 9 pairs with two terms at once, 6 of them with an episode of both terms and 3 with both constant,
      and 3 ordinary episodes and 3 ordinary constant terms: 15 terms that change, 9 constant)
  (m1) Installed: at least 21 of 24 at n = 1000 and 19 of 24 at n = 300.
  (m2) Changes declared: at least 12 of 15 at n = 1000 and 9 of 15 at n = 300; constant terms at most 1 of 9.
  (m3) At n = 1000 both changes are declared in at least 4 of the 6 pairs with an episode of two terms.
 hub (per n: 12 constant terms, 6 episodes; the modulator is a hub whose leaves get stronger by 0.30)
  (h1) Installed: at least 16 of 18 at n = 1000 and 14 of 18 at n = 300.
  (h2) Episodes declared: at least 5 of 6 at n = 1000 and 4 of 6 at n = 300.  Constant terms declared: at most 1
       of 12 at each n.
  (h3) Installed terms that are not planted and join a planted pair with a leaf of its modulating hub: a change
       is declared for at most 1 at each n.
 mixed (per n: 18 terms that change, 6 constant; autoregression 0.95, 0.95, 0.6, 0.3, 0 by block)
  (x1) Every dataset has at least three different block lengths among its variables.
  (x2) Installed: at least 18 of 24 at n = 1000 and 14 of 24 at n = 300.
  (x3) Changes declared: at least 12 of 18 at n = 1000 and 9 of 18 at n = 300; constant terms at most 1 of 6.
  (x4) Nothing planted: at most 1 installed term over the two datasets.
 exposed (per n: 18 constant terms, 6 episodes; the modulators' dependence on their hubs rises by 0.30)
  (e1) Installed: at least 22 of 24 at each n.
  (e2) Episodes declared: at least 5 of 6 at each n.
  (e3) Constant planted terms with a change declared for the term or for its companion (the pair with the hub of
       the modulator): at most 1 of 18 at each n (first version on the datasets of E29: 6 of 36 and 3 of 36).
 nested (per n: 12 constant triples with a fourth-order episode on their pair, 12 ordinary constant triples)
  (n1) Third order installed: at least 22 of 24 at each n.  Fourth order installed: at least 9 of 12 at n = 1000;
       no prediction at n = 300.
  (n2) Constant triples under an episode declared: at most 1 of 12 at each n.  Ordinary constant triples: at most
       1 of 12 at each n.
  (n3) Fourth-order episodes declared: at least 6 of 12 at n = 1000; no prediction at n = 300.
For every dataset: `vinemodel2.fit` on the training, validation and confirmation windows, the held-out log density
on the test windows, the evaluation of E26 (`e26_challenge.evaluate`: planted terms installed, windows, changes
declared; terms that are not planted) and, with STRICT=1, the strict detector (`p2.detect`) on the training
windows.
Usage: e31_challenge.py OUT.json LAW N G SEED WORKERS [REPLICATES=399] [STRICT=0]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common  # noqa: F401                                          (sets the import path of the package)
import e26_challenge as e26
import e26_laws as laws
import e30b_second_version as e30b
import e31_laws as e31
from dvc_package.modulation import margins, p2, vinemodel2  # noqa: E402


def run(law, n, g, seed, workers, replicates=399, with_strict=False):
    start = time.time()
    spec, raw, settings = e31.generate(law, n, g, seed)
    clock = dict(data=round(time.time()-start, 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result = vinemodel2.fit(raw['train'], raw['validation'], raw['confirm'], raw['test'], replicates=replicates, seed=seed, pool=pool,
                                max_order=settings.get('max_order', 3))
    clock.update(result['seconds'])
    strict = None
    if with_strict:
        begin = time.time()
        detection = p2.detect(margins.window(raw['train'])['train'], dict(p2.CONFIG, fourth_order=settings.get('max_order', 3) >= 4), workers=workers)
        strict = [dict(subset=list(x['subset']), run=list(x['run'])) for x in detection.triples+detection.quadruples]
        clock['strict'] = round(time.time()-begin, 1)
    planted, unplanted = e26.evaluate(spec, result, strict, g)
    for row in planted:                                             # how the term enters, for the tables
        item = next(i for i in spec['planted'] if tuple(i['pair']) == tuple(row['pair']) and any(tuple(t['modulators']) == tuple(row['modulators']) for t in i['terms']))
        row['link'] = next(t.get('link', 'linear') for t in item['terms'] if tuple(t['modulators']) == tuple(row['modulators']))
        row['copula'] = item.get('copula', 'gaussian')
    truth = {term['subset'] for term in laws.subsets(spec, g)}
    terms = []
    for e in result['edges']:
        for k, t in e.get('test', {}).get('terms', {}).items():
            terms.append(dict(subset=t['subset'], shape=e['components'][k][0], edge_pair=list(e['pair']), features=e['features'], install_z=e['install_z'],
                              p=t['p'], declared=t['declared'], p_orientations=t['p_orientations'], declared_orientations=t['declared_orientations'],
                              joint=t['joint'], preferred=t['directed'], orientations=[dict(pair=o['pair'], modulators=o['modulators'], features=o['features'],
                                                                                           gain=o['gain'], z=o['z'], shown=o['shown'], p=o['p'], vine=o['vine'], block=o['block'])
                                                                                      for o in t['orientations']]))
    r = np.array(result['serial']['r'])
    summary = dict(law=law, d=settings['d'], n=n, g=g, seed=seed, replicates=replicates, options={}, order=result['order'], block=result['block'],
                   inflation=result['inflation'], serial=dict(largest=float(r.max()), median=float(np.median(r)), blocks=sorted(set(result['serial']['blocks'])),
                                                              r=[round(float(v), 3) for v in r]),
                   flags={kind: result['flags'].count(kind) for kind in ('pooled', 'aligned', 'window')},
                   components=result['components'], screened_edges=len(result['edges']), kept_edges=int(sum(e['kept'] for e in result['edges'])),
                   installed_edges=int(sum(e['installed'] for e in result['edges'])), scores=result['scores'], whole=result['whole'],
                   first_pass_edges=int(sum(e.get('confirm_z') is not None for e in result['edges'])),
                   planted=planted, unplanted=unplanted, terms=terms, rising=spec['rising'], pairs=spec.get('pairs'),
                   groups=[dict(pair=g_['pair'], modulators=g_['modulators'], members=len(g_['members']), p=g_.get('p'), block=g_.get('block')) for g_ in result['groups']],
                   strict_other=None if strict is None else [x for x in strict if tuple(sorted(x['subset'])) not in truth], seconds=clock)
    edges = [{k: v for k, v in e.items() if k not in ('coefficients', 'test')} for e in result['edges']]
    return summary, edges


if __name__ == '__main__':
    target, law, n, g, seed, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    replicates = int(sys.argv[7]) if len(sys.argv) > 7 else 399
    with_strict = bool(int(sys.argv[8])) if len(sys.argv) > 8 else False
    summary, edges = run(law, n, g, seed, workers, replicates, with_strict)
    target.write_text(json.dumps(dict(summary=summary, edges=edges), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    e30b.describe(summary)
    if summary['strict_other'] is not None:
        print(f"  strict detector: subsets confirmed that are not planted {len(summary['strict_other'])}; planted terms confirmed "
              f"{sum(bool(r['strict']) for r in summary['planted'])} of {len(summary['planted'])}", flush=True)
