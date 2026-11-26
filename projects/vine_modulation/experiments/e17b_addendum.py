"""E17b.  Addendum to E17 on the same datasets, written after the results of E17 were seen (post hoc; E17 is unchanged).

Two properties of the frozen comparison could favour the joint estimator for reasons that have nothing to do with it.
Both are removed here, for the comparison arms only; the joint estimator is not retuned.

1. Grid of the `smooth` arm.  In E17 the smallest smoothness on its grid (1 x n) was selected for the modulation in
   every `abrupt`, `recurring` and `sawtooth` dataset at n = 1000 and in most at n = 250, and for the baseline in nearly
   every dataset with a changing background.  A smooth trajectory with too large a smallest penalty cannot follow a
   jump, so that arm was not properly tuned there.
     smooth_x    the `smooth` arm with the grid extended downward by three decades:
                 (0.001, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, inf) x n for the baseline and for the modulation
                 (100 pairs).  As the smoothness goes to zero this arm becomes the window-by-window fit.
     smooth1_x   the same with the true term only.

2. Fitting of the comparator.  The scoring iteration of gamCopula 0.0.8 clips the working residual at its 2.5% and 97.5%
   quantiles in every iteration (`gamCopula:::wz.update`), so it does not converge to the penalised maximum-likelihood
   estimate: on identical data its slope is about 6-8% above the direct maximum-likelihood fit, at every n (E17d).
     gamml_vc4, gamml_vc1, gamml_te   the three forms of E17 with the same formulas, bases (4-24), smoothing criterion
                 (mgcv, inside every iteration) and validation choice, fitted by the same iteration without the
                 clipping and with step halving (`e17b_gam_ml_fit.R`).
     gamml       the form with the best validation score.

Same datasets as E17: same generator, seeds 980000 + s and order of random draws.  The arms `oracle` and `windows` are
recomputed and stored so that the summary can verify that every dataset is the one of E17.
Measures as in E17.  Expectations written before the run: (a) `smooth_x` is at least as good as `smooth` and as
`windows` in every cell, and remains clearly behind `joint` where the modulation is abrupt; (b) `gamml` improves on
`gam` mostly at n = 1000 and in the designs without jumps, where it should be level with or better than `joint`;
(c) where the modulation is abrupt `joint` stays ahead of `gamml`.
Usage: e17b_addendum.py OUT.json WORKERS RLIBRARY SEED0 SEEDS [designs=.. sizes=.. strengths=..]
"""
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from common import cvine, training_transform
from dvc_package.modulation import trajectory as tr
import e17_joint_trajectories as e17

T, TEST = e17.T, e17.TEST
SMOOTH_X = (0.001, 0.01, 0.1)+tuple(e17.SMOOTH)                 # times the number of observations per window
RUNNER = Path(__file__).resolve().parent/'e17b_gam_ml.R'
FITTER = Path(__file__).resolve().parent/'e17b_gam_ml_fit.R'


def gam_ml_arm(inputs, folder, library):
    """As `e17.gam_arm`, with the runner that fits by penalised maximum likelihood."""
    folder.mkdir(parents=True, exist_ok=True)
    for role, (u, v, features) in inputs.items():
        rows = np.concatenate([np.column_stack([v[t], u[t], np.full(len(u[t]), t/(T-1.0)), features[t]]) for t in range(T)])
        with gzip.open(folder/f'{role}.csv.gz', 'wt') as handle:
            np.savetxt(handle, rows, delimiter=',', header='u1,u2,time,z0,q0,z1,z01', comments='', fmt='%.8f')
    start = time.time()
    process = subprocess.run(['Rscript', str(RUNNER), str(folder.resolve()), str(Path(library).resolve()), ','.join(map(str, e17.BASES)),
                              str(FITTER)], capture_output=True, text=True)
    record = dict(seconds=time.time()-start)
    out = {}
    if process.returncode != 0:
        record['failed'] = process.stderr[-600:]
    else:
        result = json.loads((folder/'gam_results.json').read_text())
        keep = ('validation', 'converged', 'iterations', 'halvings', 'edf', 'seconds', 'error')
        record.update(selected=result['selected'], versions=result['versions'],
                      configurations={k: {name: v[name] for name in keep} for k, v in result['records'].items()})
        table = np.genfromtxt(gzip.open(folder/'test_par.csv.gz', 'rt'), delimiter=',', names=True)
        for form in result['selected']:
            out[f'gamml_{form}'] = np.arctanh(np.clip(np.atleast_1d(table[form]), -0.999, 0.999)).reshape(T, TEST)
        if result['selected']:
            chosen = max(result['selected'], key=lambda form: result['records'][result['selected'][form]]['validation'])
            out['gamml'] = out[f'gamml_{chosen}']
            record['best_form'] = chosen
    shutil.rmtree(folder, ignore_errors=True)
    return out, record


def run(task):
    design, g, n, seed, library, scratch = task
    try:
        return _run(design, g, n, seed, library, scratch)
    except Exception:                                                # reported, never dropped
        return dict(design=design, g=g, n=n, seed=seed, error=traceback.format_exc()[-1500:])


def _run(design, g, n, seed, library, scratch):
    clock = {}
    start = time.time()
    # the next lines are those of `e17._run`, so that the datasets are identical
    rng = np.random.default_rng([seed, n, int(round(100*g)), e17.DESIGNS.index(design)])
    truth, base, c = e17.law(design, g)
    raw = {role: np.stack([cvine.simulate(truth, t, m, rng) for t in range(T)]) for role, m in (('train', n), ('validation', n), ('test', TEST))}
    data = dict(train=np.stack([cvine.pseudo_observations(x) for x in raw['train']]),
                validation=training_transform(raw['train'], raw['validation']), test=training_transform(raw['train'], raw['test']))
    lower = e17.lower_fit(data['train'])
    inputs = {role: tr.edge_inputs(lower, list(data[role]), 2, 0, e17.COMPONENTS) for role in data}
    train = tr.edge_data(*inputs['train'])
    validation = tr.edge_data(*inputs['validation'], centre=train.means)
    test = tr.edge_data(*inputs['test'], centre=False)
    features = np.stack(inputs['test'][2])
    zeta0 = ndtri(raw['test'][:, :, 0])
    true_eta = base[:, None]+c[:, None]*zeta0
    out = dict(design=design, g=g, n=n, seed=seed, arms={}, selected={})
    clock['data'] = time.time()-start

    def add(name, eta):
        out['arms'][name] = e17.measures(eta, test.x, test.y, true_eta, zeta0)

    add('oracle', true_eta)
    u, v, f = inputs['train']
    add('windows', e17.predictor(np.array([cvine.fit_modulated_gaussian(u[t], v[t], f[t])[0] for t in range(T)]), features))

    start = time.time()
    smooth = [n*s for s in SMOOTH_X]
    grid = lambda k: [tr.penalty(k, smooth=(sb, sm), jump=0.0) for sb in smooth for sm in smooth]
    best = tr.tune(train, validation, grid(4), search=False)
    add('smooth_x', e17.predictor(best[0].coefficients(), features))
    out['selected']['smooth_x'] = dict(smooth=best[0].penalty.smooth)
    first = lambda role: tr.edge_data(inputs[role][0], inputs[role][1], [a[:, :1] for a in inputs[role][2]],
                                      centre=True if role == 'train' else train.means[:, :1])
    best = tr.tune(first('train'), first('validation'), grid(1), search=False)
    add('smooth1_x', e17.predictor(best[0].coefficients(), features))
    out['selected']['smooth1_x'] = dict(smooth=best[0].penalty.smooth)
    clock['smooth'] = time.time()-start

    if library:
        folder = Path(tempfile.mkdtemp(prefix=f'{design}_g{g}_n{n}_s{seed}_', dir=scratch))
        predictions, record = gam_ml_arm(inputs, folder, library)
        for name, eta in predictions.items():
            add(name, eta)
        out['gamml'] = record
        clock['gamml'] = record['seconds']
    out['seconds'] = {k: round(v, 2) for k, v in clock.items()}
    return e17.clean(out)


if __name__ == '__main__':
    target, workers, library = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    seed0, seeds = int(sys.argv[4]), int(sys.argv[5])
    options = dict(item.split('=') for item in sys.argv[6:])
    designs = options.get('designs', ','.join(e17.DESIGNS)).split(',')
    sizes = [int(v) for v in options.get('sizes', '100,250,1000').split(',')]
    strengths = [float(v) for v in options.get('strengths', '0.3,0.6').split(',')]
    library = '' if library in ('none', '-') else library
    scratch = target.parent/(target.stem+'_gam')
    scratch.mkdir(parents=True, exist_ok=True)
    tasks = [(design, g, n, seed0+s, library, str(scratch))
             for design in designs for g in ((0.0,) if design == 'null' else strengths) for n in sizes for s in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    target.write_text(json.dumps(rows))
    shutil.rmtree(scratch, ignore_errors=True)
    failed = [r for r in rows if 'error' in r]
    missing = [r for r in rows if 'error' not in r and library and 'gamml' not in r['arms']]
    print(f'{len(rows)} datasets, {len(failed)} failed, comparator missing in {len(missing)}')
    for r in failed[:5]:
        print(r['design'], r['g'], r['n'], r['seed'], r['error'][-400:])
    for r in missing[:5]:
        print(r['design'], r['g'], r['n'], r['seed'], r.get('gamml', {}).get('failed', '')[-400:])
    seconds = {}
    for r in rows:
        for k, v in r.get('seconds', {}).items():
            seconds.setdefault((r['n'], k), []).append(v)
    for key in sorted(seconds):
        print(f'median seconds, n = {key[0]}, {key[1]}: {np.median(seconds[key]):.1f}')
    sys.exit(1 if failed or missing else 0)
