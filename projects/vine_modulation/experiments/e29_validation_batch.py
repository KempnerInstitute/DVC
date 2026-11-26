"""E29.  Targeted validation batch for the frozen pipeline `vinemodel-a1`.

Three objections to the challenge of E26 and one law for each (`e29_laws.py`):
  exposed, exposed_strong
                   E26's `moving` law hardly exposed its constant terms to the hazard it was built for.  Here every
                   planted term has its modulator in a block whose dependence on its hub rises at window 6: by
                   Kendall tau 0.15 as in E26 (`exposed`) or by 0.30 (`exposed_strong`).
  continuous       E26's `serial` law restarts its autoregression in every window and draws the roles as separate
                   series.  Here one series (autoregression 0.8) runs through all windows, and every window is cut
                   in time order into training, validation, confirmation and test stretches, 25 observations left
                   out between stretches.
  gaussian, pairwise, pairwise_moving
                   E26's dense background is a simplified regular vine.  Here the log density has pairwise terms
                   only on a dense graph with cycles: a Gaussian law (`gaussian`), a law that is not Gaussian
                   (`pairwise`: every pair term is twice as strong where both variables are inside (-1, 1) as
                   where one is outside), and the latter with pairwise terms that change at window 6
                   (`pairwise_moving`).  Nothing is planted.  Largest eigenvalue of the quadratic form of the pair
                   terms where they are strongest: 0.9 in `gaussian` and `pairwise`; 0.6 in `pairwise_moving`,
                   where the terms at every second ten of variables are multiplied by 1.5 from window 6.
The pipeline, its options, the strict detector and the evaluation are those of E26 (the steps of
`e26_challenge.run`, with this file's generator): n training, validation and confirmation observations per window
and 1,000 test observations, 12 windows, 399 halves.

Added to the pipeline's own tests, as a diagnostic that does not change them: every orientation test of the
pipeline is repeated with each of its two feature sets alone (coordinates, own scores), with the same halves.
The pipeline tests an orientation with the set that predicts the validation windows better; the diagnostic gives
the p-value of both, so that for every installed term all descriptions (three orientations by two feature sets
for a triple) are on record, and with them a second rule: a change only if every description rejects (false
discovery rule at 0.05 over the installed terms).  This rule is not part of the frozen pipeline.
Reason (development seed 970400, `exposed_strong`, n = 1000, 99 halves, after the expectations below were written):
one of six constant terms was declared by the primary rule.  There the vine's own orientation is the planted
one and the edge carries the modulator's hub as a second term; with both terms the coordinate description is
complete, predicts the validation windows as well as the own-score description (0.14474 against 0.14452 nats),
was chosen, and its coefficients change when the modulator's dependence on its hub changes.  The other five
terms chose own scores by 0.0005 to 0.004 nats.  So the choice between two descriptions that fit equally well
decides whether the invariant one is tested, and expectation (c) is likely to fail in `exposed_strong`.  It is
left as written.

Expectations written before the run.
`exposed` and `exposed_strong` (seeds 985000 to 985005, strength 0.6; in each law 36 constant terms and 12
episodes at each n):
 (a) at least 90% of the constant terms are installed at n = 1000 and 80% at n = 300;
 (b) the hazard is there: for at least 75% of the installed constant terms the test rejects (p <= 0.05) in at
     least one orientation, in `exposed` at n = 1000 (no prediction at n = 300, where the change of the coefficient
     is about two standard errors) and in `exposed_strong` at both n (90% at n = 1000);
 (c) the primary rule (every orientation rejects; false discovery rule at 0.05) declares a change for at most 2 of
     the installed constant terms of a law at each n;
 (d) a change is declared for at least 90% of the installed episodes at n = 1000 and 75% at n = 300;
 (e) the preferred orientation (largest validation gain) is the planted one for at least 85% of the installed terms
     at n = 1000; the number of constant terms that its own test declares is reported without a prediction.
`continuous` (seeds 987000 to 987002 with strength 0.6, 987003 with nothing planted):
 (f) a block length above one in every dataset;
 (g) planted terms installed: at least 80% at n = 1000 and 60% at n = 300;
 (h) a change is declared for at least 75% (n = 1000) and 50% (n = 300) of the installed terms that change, and for
     at most 1 of the installed constant terms at each n;
 (i) with nothing planted at most 2 installed terms, and the assembled vine not more than 0.005 nats per observation
     below the simplified vine; with planted modulations the assembled vine is above the simplified vine in every
     dataset.
Pairwise laws (seeds 986000 to 986002):
 (j) `gaussian`: at most 2 installed terms per dataset and no declared change;
 (k) `pairwise`: no bound is predicted for the number of installed terms (under this law the dependence of a pair
     given other variables can vary with them, although the log density has pairwise terms only); they are
     counted, with the held-out gain of the assembled vine.  Nothing changes over the windows: a change is declared
     for at most 5% of the installed terms and at most 2 per dataset;
 (l) `pairwise_moving`: no prediction.  Installed terms and declared changes are counted (when pairwise terms of the
     log density change, the modulation coefficients of a vine change with them);
 (m) the strict detector confirms at most one subset over the `gaussian` datasets on which it is run; no prediction
     for the other two laws.
Usage: e29_validation_batch.py OUT.json LAW N G SEED WORKERS [REPLICATES=399] [STRICT=1]
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
import e29_laws
from dvc_package.modulation import margins, p2, vinemodel  # noqa: E402

T, TEST = laws.T, e26.TEST
LAWS = dict(exposed=dict(d=50), exposed_strong=dict(d=50), continuous=dict(d=50), gaussian=dict(d=50), pairwise=dict(d=50), pairwise_moving=dict(d=50))
RISE = dict(exposed=0.15, exposed_strong=0.30)
RADIUS = dict(gaussian=0.9, pairwise=0.45, pairwise_moving=0.3)      # of the matrix b; the pair terms inside (-1, 1) are 2 b
GAP = 25


def generate(law, n, g, seed):
    d = LAWS[law]['d']
    rng = np.random.default_rng([seed, d, n, 293])
    roles = (('train', n), ('validation', n), ('test', TEST), ('confirm', n))
    if law in RISE:
        spec = e29_laws.exposed_design(d, seed, g, rise=RISE[law])
        raw = {role: [laws.sample(spec, m, t, g, rng) for t in range(T)] for role, m in roles}
    elif law == 'continuous':
        spec = laws.design(d, seed, g, plan='e18', background='tree')
        raw = e29_laws.continuous_sample(spec, dict(train=n, validation=n, confirm=n, test=TEST), g, 0.8, rng, gap=GAP)
    elif law in ('gaussian', 'pairwise', 'pairwise_moving'):
        spec = e29_laws.pairwise_design(d, seed, radius=RADIUS[law], saturating=law != 'gaussian')
        raw = {role: [e29_laws.pairwise_sample(spec, m, t, rng, moving=law == 'pairwise_moving') for t in range(T)] for role, m in roles}
    else:
        raise ValueError(law)
    return spec, raw


e26.LAWS.update(LAWS)                                               # the evaluation of E26 with this file's laws


class Recorder:
    """A pool that keeps the tasks of the pipeline's orientation tests; every task runs as in the pipeline."""

    def __init__(self, pool):
        self.pool, self.jobs = pool, []

    def map(self, function, iterable, **options):
        items = list(iterable)
        if function is vinemodel._orientation:
            self.jobs = items
        return self.pool.map(function, items, **options)


def run(law, n, g, seed, workers, replicates=399, with_strict=True):
    """The steps of `e26_challenge.run`, and the orientation tests once more with each feature set alone."""
    start = time.time()
    options = LAWS[law]
    spec, raw = generate(law, n, g, seed)
    clock = dict(data=round(time.time()-start, 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        recorder = Recorder(pool)
        result = vinemodel.fit(raw['train'], raw['validation'], raw['confirm'], raw['test'], replicates=replicates, seed=seed, pool=recorder,
                               max_order=options.get('max_order', 3), block='auto')
        clock.update(result['seconds'])
        begin = time.time()
        single = [dict(job, variants=[variant]) for job in recorder.jobs for variant in job['variants']]
        alone = list(pool.map(vinemodel._orientation, single, chunksize=1))
        clock['descriptions'] = round(time.time()-begin, 1)
    strict = None
    if with_strict:
        begin = time.time()
        ranked = margins.window(raw['train'])['train']
        detection = p2.detect(ranked, dict(p2.CONFIG, fourth_order=options.get('max_order', 3) >= 4), workers=workers)
        strict = [dict(subset=list(x['subset']), run=list(x['run'])) for x in detection.triples+detection.quadruples]
        clock['strict'] = round(time.time()-begin, 1)
    planted, unplanted = e26.evaluate(spec, result, strict, g)
    # every description of every installed term: {(level, edge, term): [dict(pair, modulators, features, p, vine)]}
    table = {}
    for out in alone:
        for index, (k, p) in enumerate(out['p'].items()):
            table.setdefault((tuple(out['key']), int(k)), []).append(dict(pair=out['pair'], modulators=out['modulators'][index], features=out['features'],
                                                                         p=float(p), vine=bool(out['vine'])))
    names = sorted(table)
    largest = [max(d['p'] for d in table[name]) for name in names]
    declared = dict(zip(names, vinemodel.discoveries(largest, 0.05)))
    edges = {(e['level'], e['edge']): e for e in result['edges']}
    position = {v: k for k, v in enumerate(result['order'])}
    mismatch = 0
    for (key, k), rows in table.items():                            # the set the pipeline chose must give the pipeline's p-value
        for chosen in edges[key]['test']['terms'][k]['orientations']:
            same = [d for d in rows if d['pair'] == chosen['pair'] and d['modulators'] == chosen['modulators'] and d['features'] == chosen['features']]
            mismatch += int(not same or abs(same[0]['p']-chosen['p']) > 1e-12)
    for row in planted:
        home = sorted(row['subset'], key=lambda v: position[v])
        key = (position[home[-2]], position[home[-1]]-position[home[-2]]-1)
        e = edges.get(key)
        if e is None or not row['installed']:
            continue
        for k, s in enumerate(e['subsets']):
            if sorted(s) == sorted(row['subset']) and (key, k) in table:
                row['descriptions'] = table[(key, k)]
                row['p_every'], row['change_every'] = float(max(d['p'] for d in table[(key, k)])), bool(declared[(key, k)])
    known = {tuple(sorted(row['subset'])) for row in planted}
    other = [dict(subset=sorted(edges[key]['subsets'][k]), p_every=float(max(d['p'] for d in rows)), change_every=bool(declared[(key, k)]),
                  change=bool(edges[key]['test']['terms'][k]['declared']), descriptions=rows)
             for (key, k), rows in table.items() if tuple(sorted(edges[key]['subsets'][k])) not in known]
    truth = {term['subset'] for term in laws.subsets(spec, g)}
    summary = dict(law=law, d=options['d'], n=n, g=g, seed=seed, replicates=replicates, order=result['order'][:12], block=result['block'],
                   flags={kind: result['flags'].count(kind) for kind in ('pooled', 'aligned', 'window')}, margin_kinds=None, margin_table=None,
                   inflation=result['inflation'], components=result['components'], screened_edges=len(result['edges']),
                   kept_edges=int(sum(e['kept'] for e in result['edges'])), installed_edges=int(sum(e['installed'] for e in result['edges'])),
                   scores=result['scores'], planted=planted, unplanted=unplanted, other_descriptions=other, description_mismatch=mismatch,
                   strict_other=[x for x in (strict or []) if tuple(sorted(x['subset'])) not in truth], seconds=clock)
    return summary, [{k: v for k, v in e.items() if k != 'coefficients'} for e in result['edges']]


def describe(s):
    e26.describe(s)
    constant = [r for r in s['planted'] if r['shape'] == 'constant' and r['installed'] and r.get('orientations')]
    if constant:
        print(f"  installed constant terms {len(constant)}: rejected in at least one orientation {sum(min(o['p'] for o in r['orientations']) <= 0.05 for r in constant)}, "
              f"in the vine's own orientation {sum(r['vine_p'] <= 0.05 for r in constant)}, in the preferred orientation {sum(r['directed_p'] <= 0.05 for r in constant)}, "
              f"in every orientation {sum(r['p'] <= 0.05 for r in constant)}, in every description {sum(r.get('p_every', 1.0) <= 0.05 for r in constant)}; "
              f"declared by the primary rule {sum(r['change'] for r in constant)}, by the secondary rule {sum(bool(r.get('change_directed')) for r in constant)}, "
              f"by the rule over every description {sum(bool(r.get('change_every')) for r in constant)}", flush=True)
    moving = [r for r in s['planted'] if r['shape'] != 'constant' and r['installed'] and r.get('orientations')]
    if moving:
        print(f"  installed changing terms {len(moving)}: declared by the primary rule {sum(r['change'] for r in moving)}, by the rule over every description "
              f"{sum(bool(r.get('change_every')) for r in moving)}", flush=True)
    installed = [u for u in s['unplanted'] if u['installed']]
    if installed and not s['planted']:
        print(f"  nothing planted: installed terms {len(installed)} (order 3: {sum(u['order'] == 3 for u in installed)}; shapes "
              f"{ {shape: sum(u['shape'].endswith(shape) for u in installed) for shape in ('linear', 'quadratic', 'product')} }), "
              f"changes declared {sum(u['change'] for u in installed)}, by the secondary rule {sum(u['change_directed'] for u in installed)}, "
              f"by the rule over every description {sum(o['change_every'] for o in s['other_descriptions'])}, "
              f"p <= 0.05 {sum(u['p'] is not None and u['p'] <= 0.05 for u in installed)}", flush=True)
    if s['description_mismatch']:
        print(f"  WARNING: {s['description_mismatch']} tests with the chosen feature set alone did not reproduce the pipeline's p-value", flush=True)


if __name__ == '__main__':
    target, law = Path(sys.argv[1]), sys.argv[2]
    n, g, seed, workers = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    replicates = int(sys.argv[7]) if len(sys.argv) > 7 else 399
    with_strict = bool(int(sys.argv[8])) if len(sys.argv) > 8 else True
    summary, edges = run(law, n, g, seed, workers, replicates, with_strict)
    target.write_text(json.dumps(dict(summary=summary, edges=edges), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    describe(summary)
