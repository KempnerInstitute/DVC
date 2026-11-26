"""E29b.  A law whose log density has pairwise terms only and whose pairs do change their dependence with a third
variable (added to E29 after its pairwise laws were run; this file leaves the files of E29 untouched).

The pairwise law of E29 (`pairwise`: every pair term twice as strong where both variables are inside (-1, 1)) has
no term with more than two variables, and the pipeline installs nothing on it.  In 400,000 draws this law showed little third-order structure in the
sense of this project either: the correlation of a pair given a neighbour varies by at most 0.02 with the
neighbour (a standard score of 0.6 at the size of a dataset).  So that control does not
separate the two notions of "interaction":
  in the log density    a term with three variables;
  in this project       the dependence of a pair, given other variables, varies with one of them.
This law separates them.  50 variables on a ring:
    log p(x) = - sum_i x_i^2 / 2 + sum_{i<j} b_ij x_i x_j - (c / 2) sum_{(i, k) in S} x_i^2 x_k^2 + constant,
b = 0.36 between neighbours on the ring and 0.04 between second neighbours (largest eigenvalue 0.8); S: ten pairs
(i, i + 3), i every fifth variable from a random start ("variance partners"); c = 1.  Every term has two
variables.  Given the others, x_i is normal with precision 1 + c x_k^2 for a variable with a partner k (1 for the
others) and mean sum_j b_ij x_j over that precision: an exact Gibbs sampler, as in E29.  So the correlation of a
variable that has a partner with its neighbours on the ring falls as the partner's square grows: by the factor
1 / sqrt(1 + c x_k^2), 0.71 at x_k = 1 and 0.45 at x_k = 2.  In the sense of this project that is a third-order
interaction of the quadratic shape (a pair on the ring, modulated by the partner of one of its variables); in the
log density there is none.  Nothing changes over the windows.
A first design had these terms on the dense graph of E29 (one partner per variable among its neighbours, c = 0.5,
b with largest eigenvalue 0.9).  It was dropped after its development dataset (seed 970400, n = 1000): nothing was
installed (54 edges screened, 1 kept), because on a dense graph the pairs are weakly dependent (held-out log
density of the simplified vine 1.43 nats per observation) and a modulation of a weak dependence by 40% is below
what 12,000 observations show.  A law has to have pairs that are clearly dependent for the question to be asked.

Run: the frozen pipeline as in E29 (`e29_validation_batch.run`: tests in every orientation, both feature sets as
a diagnostic, strict detector on the first seed at each n).
Expectations written before the development dataset of this design and before the run:
 (a) the pipeline installs at least 10 terms in every dataset at n = 1000 (they are counted, with their shapes);
 (b) at least 75% of the installed terms at n = 1000 join a pair with the variance partner of one of its variables
     (they are the law's structure);
 (c) nothing changes over the windows: a change is declared, by the primary rule, for at most 5% of the installed
     terms and at most 2 per dataset;
 (d) no prediction for the strict detector (it may confirm subsets: they are real in its own sense) nor for
     n = 300.
Development dataset of this design (seed 970400, n = 1000, 99 halves; after the expectations were written): three
terms installed, all of the quadratic shape, none declared changing.  They are not the terms that were predicted:
each joins the two variables of a pair of variance partners (10 and 13, 5 and 8, 25 and 28), modulated by a ring
variable between them.  So (a) and (b) are expected to fail as written; they are left as written, and the terms
are also counted by whether their variables include both variables of a pair of variance partners.
Usage: e29b_scale_control.py OUT.json N SEED WORKERS [REPLICATES=399] [STRICT=1]
"""
import json
import sys
from pathlib import Path

import numpy as np

import common  # noqa: F401                                          (sets the import path of the package)
import e26_challenge as e26
import e26_laws as laws
import e29_laws
import e29_validation_batch as e29

T, TEST = laws.T, e26.TEST
C, NEAR, FAR, EVERY, OFFSET = 1.0, 0.36, 0.04, 5, 3


def scale_design(d, seed, c=C):
    rng = np.random.default_rng([seed, d, 296])
    b, edges = np.zeros((d, d)), set()
    for i in range(d):
        for step, value in ((1, NEAR), (2, FAR)):
            j = (i+step) % d
            b[i, j] = b[j, i] = value
            edges.add(tuple(sorted((i, j))))
    first = int(rng.integers(0, EVERY))
    partner = {i: (i+OFFSET) % d for i in range(first, d, EVERY)}
    coupling = np.zeros((d, d))
    for i, k in partner.items():
        coupling[i, k] += c
        coupling[k, i] += c
    assert np.abs(np.linalg.eigvalsh(b)).max() < 1.0
    return dict(d=d, edges=sorted(edges), b=b, c=coupling, partner=partner, planted=[], hubs=[], margin_kind=['fixed']*d, rising=[])


def scale_sample(spec, n, rng, sweeps=150):
    """n independent observations (the law does not change over the windows)."""
    d, b, c = spec['d'], spec['b'], spec['c']
    x = rng.normal(size=(n, d))
    for _ in range(sweeps):
        for i in range(d):
            precision = 1.0+(x*x)@c[i]
            x[:, i] = (x@b[i])/precision+rng.normal(size=n)/np.sqrt(precision)
    return x


def generate(law, n, g, seed):
    if law != 'pairwise_scale':
        raise ValueError(law)
    spec = scale_design(50, seed)
    rng = np.random.default_rng([seed, 50, n, 297])
    raw = {role: [scale_sample(spec, m, rng) for t in range(T)] for role, m in (('train', n), ('validation', n), ('test', TEST), ('confirm', n))}
    return spec, raw


e29.LAWS['pairwise_scale'] = dict(d=50)
e26.LAWS['pairwise_scale'] = dict(d=50)
e29.generate = generate


if __name__ == '__main__':
    target, n, seed, workers = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    replicates = int(sys.argv[5]) if len(sys.argv) > 5 else 399
    with_strict = bool(int(sys.argv[6])) if len(sys.argv) > 6 else True
    summary, edges = e29.run('pairwise_scale', n, 0.0, seed, workers, replicates, with_strict)
    spec = scale_design(50, seed)
    partner = spec['partner']
    related = lambda pair, modulators: any(partner.get(v) == m or partner.get(m) == v for v in pair for m in modulators)
    both = lambda variables: any(i in variables and k in variables for i, k in partner.items())
    rows = []
    for e in edges:
        if not e['installed']:
            continue
        for k, on in enumerate(e['on']):
            if on:
                tested = e.get('test', {}).get('terms', {}).get(k, {})
                rows.append(dict(pair=list(e['pair']), modulators=e['modulators'][k], shape=e['components'][k][0], features=e['features'],
                                 partner=bool(related(e['pair'], e['modulators'][k])), partner_pair=bool(both(set(e['pair']) | set(e['modulators'][k]))),
                                 p=tested.get('p'), declared=bool(tested.get('declared', False))))
    summary['terms'] = rows
    summary['partners'] = {str(i): k for i, k in partner.items()}
    target.write_text(json.dumps(dict(summary=summary, edges=edges), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    e29.describe(summary)
    print(f"  installed terms {len(rows)}: with the variance partner as modulator {sum(r['partner'] for r in rows)}, with a pair of variance partners among "
          f"their variables {sum(r['partner_pair'] for r in rows)}; shapes "
          f"{ {shape: sum(r['shape'].endswith(shape) for r in rows) for shape in ('linear', 'quadratic', 'product')} }; changes declared {sum(r['declared'] for r in rows)}; "
          f"strict detector: subsets confirmed {len(summary['strict_other'])}", flush=True)
