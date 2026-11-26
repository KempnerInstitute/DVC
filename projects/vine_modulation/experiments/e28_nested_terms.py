"""E28.  A constant third-order term under a fourth-order episode on its pair: what "the third-order interaction
changed" means.

E26, law `overlap`: a pair (i, j) whose Fisher-z correlation is c z_k + c'(t) z_k z_m, with c = 0.6 in every window
and c'(t) = 0.6 in windows 4 to 7, zero otherwise.  The protocol of E26 called the triple {i, j, k} a term that does
not change.  The frozen pipeline declared a change for 3 of the 6 (all three at n = 1000).  These are failures
against that protocol.  The reason found there: the pipeline tests the
term of a triple in every orientation in a model that has only that term, and while the product term is on such a
model fits 0.374 and not 0.6 (the population projection without the product term, calculated by Gauss-Hermite quadrature).  So two different quantities carry the name "the third-order coefficient":
  in the complete model   the coefficient of z_k when the product term is in the model too.  Planted: constant.
  in the lower-order model  the coefficient of z_k in the model of the three variables alone, the fourth averaged
                          out.  Planted: 0.6 outside the episode and 0.374 inside; it changes.
This experiment measures both on fresh data, in every orientation of the triple.

Law: `e26_laws` with the tree background of E18, 50 variables, 12 windows, strength 0.6; per dataset four pairs with
a constant triple and a fourth-order episode on the same pair (the item `nested_fourth` of E26) and four pairs with
an ordinary constant triple.  Seeds 984000 + s; n training, validation and confirmation observations per window.
For every item and every orientation of its triple (which two of the three variables are the pair; the third
modulates): a sub-vine with the hubs of the blocks of the variables involved, then the fourth variable m (nested
items), then the modulator o, then the pair; calibration on the coordinates of the pair's hubs, of o and of m.
  alone   the edge has the term of o only (the pipeline's test of the triple)
  joint   the edge has the term of o and the product term of o and m (nested items)
Each with coordinates and with own scores as features; the set that predicts the validation windows better is
tested, as in the pipeline.  The test is the frozen worker `vinemodel._orientation` (half-sample change test, 399
halves; for `joint` the two terms are tested in one model, each with its own p-value).  For the planted orientation
with own scores (the planted parametrisation) the tuned coefficient trajectories of both models are saved.

Expectations written before the run.  "Found changing": the largest p-value over the three orientations is at most
0.05 (the primary rule of the pipeline for one term).
 (a) Lower-order model, nested items: found changing for at least 75% of the items at n = 1000 (E26: 3 of 3); no
     prediction at n = 300 (E26: 0 of 3).  Ordinary constant items: at most 4 of 40 at each n.
 (b) Complete model, nested items: the triple term is found changing for at most 4 of 40 at each n; the product
     term (largest p-value over the three orientations in which it is fitted here) for at least 75% at n = 1000.
 (c) Planted orientation, own scores, n = 1000: the mean over the nested items of the fitted coefficient of the
     triple term in the episode windows is nearer to 0.374 than to 0.6 in the lower-order model and nearer to 0.6
     than to 0.374 in the complete model; outside the episode both are within 0.1 of 0.6.
Usage: e28_nested_terms.py OUT.json N SEED WORKERS [REPLICATES=399]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common  # noqa: F401                                          (sets the import path of the package)
import e26_laws as laws
from dvc_package.modulation import cvine, halfsample, margins, trajectory as tr, vinemodel, vinetraj  # noqa: E402

T, D, G = laws.T, 50, 0.6
laws.PLANS['nested'] = (('nested_fourth', 4), ('constant_leaf', 4))  # a plan of this experiment; the plans of E26 are unchanged
EPISODE = sorted(laws.EPISODES['a'])
INDUCED = 0.374                                                     # THEORY 6e


def generate(n, seed):
    spec = laws.design(D, seed, G, plan='nested', background='tree')
    rng = np.random.default_rng([seed, D, n, 281])
    raw = {role: [laws.sample(spec, n, t, G, rng) for t in range(T)] for role in ('train', 'validation', 'confirm')}
    return spec, raw


def trajectory(task):
    """Tuned coefficient trajectories of one edge model in one sub-vine: (key, coefficients (T, K), windows on)."""
    key, components, columns, train, validation = task
    size = train[0].shape[1]
    lower = cvine.fit(list(train), order=list(range(size)), states=['ind', 'gaussian'], max_order=2, max_depth=size-2, switch=0.05)
    inputs, calibration = {}, None
    for role, windows in (('train', train), ('validation', validation)):
        inputs[role], calibration = tr.edge_inputs_calibrated(lower, windows, size-2, 0, components, calibration, degree=2, columns=columns)
    edge = tr.edge_data(*inputs['train'])
    fitted, _ = vinetraj.tune(edge, tr.edge_data(*inputs['validation'], centre=edge.means), len(components), float(np.mean([len(w) for w in train])))
    return key, fitted.coefficients()[:, 1:].tolist(), [sorted({t for a, b in fitted.pieces(1+c) for t in range(a, b)}) for c in range(len(components))]


def run(n, seed, workers, replicates):
    start = time.time()
    spec, raw = generate(n, seed)
    data, flags = margins.choose(raw['train'], dict(validation=raw['validation']), 0.05, align=True, block=1)
    flags = [margins._kind(flag) for flag in flags]
    confirm = [np.asarray(w, dtype=np.float64) for w in raw['confirm']]
    ranked = margins.ranks(confirm, flags)
    hub = lambda v: (v//laws.BLOCK)*laws.BLOCK
    jobs, tasks, items = [], [], []
    for index, item in enumerate(spec['planted']):
        i, j = item['pair']
        k = item['terms'][0]['modulators'][0]
        m = item['terms'][1]['modulators'][1] if len(item['terms']) > 1 else None
        assert m is None or item['terms'][1]['modulators'][0] == k
        members = [i, j, k]
        hubs = sorted({hub(v) for v in members+([m] if m is not None else [])})
        lead = hubs+([m] if m is not None else [])
        orientations = []
        for number, (x, y) in enumerate(((i, j), (i, k), (j, k))):
            o = next(v for v in members if v not in (x, y))
            variables = lead+[o, x, y]
            po, pm = len(lead), (len(hubs) if m is not None else None)
            columns = sorted({variables.index(hub(x)), variables.index(hub(y)), po} | ({pm} if m is not None else set()))
            models = dict(alone=[('coordinates', [('linear', (po,))]), ('own', [('own_linear', (po,))])])
            if m is not None:
                models['joint'] = [('coordinates', [('linear', (po,)), ('product', (pm, po))]), ('own', [('own_linear', (po,)), ('own_product', (pm, po))])]
            for which, (name, variants) in enumerate(models.items()):
                jobs.append(dict(key=(index, number, name), variables=variables, train=[np.asarray(w)[:, variables] for w in data['train']],
                                 validation=[np.asarray(w)[:, variables] for w in data['validation']], confirm=[w[:, variables] for w in ranked],
                                 raw=[w[:, variables] for w in confirm], pooled=[flags[v] for v in variables], variants=variants, columns=columns,
                                 terms=list(range(len(variants[0][1]))), pair=(x, y), modulators=[[o]]+([[o, m]] if name == 'joint' else []),
                                 vine=False, states=['ind', 'gaussian'], switch=0.05, replicates=replicates, block=1, keep=halfsample.KEEP,
                                 seed=[seed, index, number, which]))
                if number == 0:                                      # the planted orientation, in the planted parametrisation
                    tasks.append(((index, name), variants[1][1], columns, jobs[-1]['train'], jobs[-1]['validation']))
            orientations.append(dict(pair=[x, y], modulator=o, variables=variables))
        items.append(dict(kind=item['kind'], pair=[i, j], k=k, m=m, orientations=orientations, planted={}))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tested = list(pool.map(vinemodel._orientation, jobs, chunksize=1))
        fitted = list(pool.map(trajectory, tasks, chunksize=1))
    for result in tested:
        index, number, name = result['key']
        items[index]['orientations'][number][name] = dict(features=result['features'], gain=result['gain'], p=[result['p'][c] for c in sorted(result['p'])],
                                                          estimates=result['estimates'], seconds=result['seconds'])
    for (index, name), coefficients, on in fitted:
        items[index]['planted'][name] = dict(coefficients=np.round(coefficients, 4).tolist(), on=on)
    return dict(n=n, seed=seed, replicates=replicates, flags={kind: flags.count(kind) for kind in ('pooled', 'aligned', 'window')},
                items=items, seconds=round(time.time()-start, 1))


def describe(s):
    largest = lambda item, name, term: max(o[name]['p'][term] for o in item['orientations'])
    nested = [item for item in s['items'] if item['m'] is not None]
    plain = [item for item in s['items'] if item['m'] is None]
    inside = np.zeros(T, dtype=bool)
    inside[EPISODE] = True
    print(f"n {s['n']} seed {s['seed']}: margins {s['flags']}; {s['seconds']} s", flush=True)
    print(f"  ordinary constant triples: found changing {sum(largest(item, 'alone', 0) <= 0.05 for item in plain)} of {len(plain)}; "
          f"largest p {[round(largest(item, 'alone', 0), 3) for item in plain]}", flush=True)
    print(f"  nested, lower-order model: found changing {sum(largest(item, 'alone', 0) <= 0.05 for item in nested)} of {len(nested)}; "
          f"largest p {[round(largest(item, 'alone', 0), 3) for item in nested]}", flush=True)
    print(f"  nested, complete model: triple term found changing {sum(largest(item, 'joint', 0) <= 0.05 for item in nested)} of {len(nested)} "
          f"(largest p {[round(largest(item, 'joint', 0), 3) for item in nested]}); product term {sum(largest(item, 'joint', 1) <= 0.05 for item in nested)} "
          f"(largest p {[round(largest(item, 'joint', 1), 3) for item in nested]})", flush=True)
    for name in ('alone', 'joint'):
        c = np.array([np.array(item['planted'][name]['coefficients'])[:, 0] for item in nested])
        print(f"  planted orientation, own scores, {name}: triple coefficient in the episode {c[:, inside].mean():.3f}, outside {c[:, ~inside].mean():.3f}"
              + (f"; product coefficient in the episode {np.mean([np.array(item['planted'][name]['coefficients'])[inside, 1].mean() for item in nested]):.3f}, "
                 f"outside {np.mean([np.array(item['planted'][name]['coefficients'])[~inside, 1].mean() for item in nested]):.3f}" if name == 'joint' else ''), flush=True)


if __name__ == '__main__':
    target, n, seed, workers = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    replicates = int(sys.argv[5]) if len(sys.argv) > 5 else 399
    summary = run(n, seed, workers, replicates)
    target.write_text(json.dumps(summary, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    describe(summary)
