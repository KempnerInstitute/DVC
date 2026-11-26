"""E8 (pilot).  Directed, time-lagged interactions with the same machinery.

Law: six variables, Gaussian VAR(1): X(t) = A X(t-1) + noise, A = 0.5 on the diagonal, 0.3 for 0 -> 1 and 2 -> 3.
In windows 3-6 of 9 the target 5 receives c * X_3(t-1) * X_4(t-1): a directed third-order interaction (two lagged
sources, one target).  Each window holds n consecutive time points.
Analysis: stack [X(t-1), X(t)] (12 columns: lagged copies 0-5, present 6-11), ranks within the window, all-pairs
layer, exhaustive scan with a long-run variance (10 lags), then the orientation and explain-away rules.  A triple is
called directed when it joins lagged and present variables.  Truth: (3 lagged, 4 lagged, 5 present) = (3, 4, 11).
Only triples with at least one present variable are examined: a triple of lagged variables is the previous time
step's present triple, and the variables that could explain it (one step further back) are not in the stack.
(The first run, without this rule, confirmed the lagged copy of the simultaneous triple (3, 4, 5) in up to all
datasets at n = 1000; kept as `e8_directed_v1.json`.)
Usage: e8_directed.py OUT.json WORKERS REPLICATES
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dvc_package.modulation import cvine, families as fam, pairscan  # noqa: E402

T, D, EPISODE, TRUTH = 9, 6, range(3, 7), (3, 4, 11)
STATES = fam.default_states(('gaussian', 'student', 'clayton', 'gumbel', 'frank'))


def series(n, c, rng):
    a = 0.5*np.eye(D)
    a[1, 0] = a[3, 2] = 0.3
    x = np.zeros((T*n+200, D))
    for t in range(1, x.shape[0]):
        x[t] = a @ x[t-1]+rng.normal(size=D)
        window = (t-200)//n
        if t >= 200 and window in EPISODE:
            x[t, 5] += c*x[t-1, 3]*x[t-1, 4]
    return x[200:].reshape(T, n, D)


def run(task):
    c, n, seed = task
    rng = np.random.default_rng([seed, n, int(100*c)])
    x = series(n, c, rng)
    windows = [cvine.pseudo_observations(np.column_stack((w[:-1], w[1:]))) for w in x]
    layer = pairscan.fit_pair_layer(windows, states=STATES, switch=0.08)
    stat = pairscan.scan_all(layer, windows, hac_lags=10)
    found, threshold = pairscan.candidates(stat, eligible=lambda i, j, k: max(i, j, k) >= D)
    checked, _ = pairscan.confirm(layer, windows, found, hac_lags=10)
    kind = lambda tr: 'lagged' if max(tr) < D else ('present' if min(tr) >= D else 'directed')
    return dict(c=c, n=n, seed=seed, candidates=[dict(triple=list(r['triple']), value=r['value'], kind=kind(r['triple'])) for r in found],
                confirmed=[dict(triple=list(r['triple']), kind=kind(r['triple']), interval=list(r['interval']),
                                weakest=min(r['hubs'].values())) for r in checked if r['confirmed']])


if __name__ == '__main__':
    target, workers, replicates = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    tasks = [(c, n, 960000+s) for c in (0.0, 0.3, 0.6) for n in (300, 1000) for s in range(replicates)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    target.write_text(json.dumps(rows))
    print('c n | datasets: true directed triple confirmed, any other triple confirmed (directed, present-only, lagged-only), '
          'exact interval | mean number of candidates before the explain-away check')
    for c in (0.0, 0.3, 0.6):
        for n in (300, 1000):
            group = [r for r in rows if r['c'] == c and r['n'] == n]
            hit = np.mean([any(tuple(x['triple']) == TRUTH for x in r['confirmed']) for r in group])
            other = {k: np.mean([any(tuple(x['triple']) != TRUTH and x['kind'] == k for x in r['confirmed']) for r in group])
                     for k in ('directed', 'present', 'lagged')}
            exact = np.mean([any(tuple(x['triple']) == TRUTH and x['interval'] == [3, 7] for x in r['confirmed']) for r in group])
            print(f"{c:.1f} {n:4d} | {hit:.2f} | {other['directed']:.2f} {other['present']:.2f} {other['lagged']:.2f} | {exact:.2f} | "
                  f"{np.mean([len(r['candidates']) for r in group]):.1f}")
