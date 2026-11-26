"""E17.  Joint coefficient trajectories against window-by-window modulation and a tuned GAM copula.

Compare joint estimation over the whole sequence with separate window fits and a GAM on the same families, transforms, structures and validation data. Development seeds start at 970000. The designs and expectations were fixed before evaluation on seeds starting at 980000.

Law.  Four variables, C-vine with root order (0, 1, 2, 3), 24 windows.  The edge under study is (2, 3 | 0, 1), a
Gaussian conditional copula with Fisher-z correlation b(t) + c(t) zeta_0 (zeta_0: normal score of variable 0).
Background that changes with time (every design but `abrupt_static`):
    (0, 1) Gaussian 0.5;  (0, 2) Clayton (theta 1) in windows 0-13, then Gumbel (theta 1.5): same Kendall tau;
    (0, 3) Gaussian 0.3 in windows 0-5, then 0.6;  (1, 2 | 0) Gaussian 0.3;  (1, 3 | 0) Gaussian, 0.1 rising to 0.4;
    b(t) = 0.2 + 0.3 in windows 4-18 + 0.1 sin(2 pi t / 24): two jumps and a drift, none aligned with c(t).
Modulation c(t), strength g = 0.3 and 0.6:
    null       0                                     constant   g
    abrupt     g in windows 8-15                     recurring  g in windows 2-5, 10-13, 18-21
    ramp       g t / 23                              sawtooth   rises linearly from 0 (window 4) to g (window 15), then 0
    abrupt_static, ramp_static   as abrupt and ramp with a background that does not change (b = 0.35): jumps in the
               modulation only, and no jump anywhere (the comparator's own ground).
n = 100, 250, 1000 training observations per window, as many validation observations, 1,000 test observations.

Shared by every arm: within-window ranks; the vine below the edge (`cvine.fit`, simplified, switch 0.08, states
independence, Gaussian, Clayton, Gumbel, Frank); hence the same two conditional transforms and the same conditioning
coordinates.  Every arm models the edge as a Gaussian copula.  Candidate terms (one true, three decoys): zeta_0;
(zeta_0^2 - 1)/sqrt 2; zeta_1; zeta_0 zeta_1.  Tuning uses the validation data only.

Arms (all built by `vinemod`, except the comparator)
    windows     every window on its own, all four terms (`cvine.fit_modulated_gaussian`).
    switching   the present gated switching edge (`trajectory.switching_path`): window-by-window coefficients,
                states chosen by the switching objective; candidates: each term alone and all four; unmodulated
                states independence and Gaussian; the 12 penalties of `common.GRID`.
    smooth      joint, one smooth trajectory per coefficient, no jumps, nothing switched off; smoothness of the
                baseline and of the modulation from n x (1, 10, ..., 1e5, inf), all 49 pairs.
    joint       joint, piecewise smooth with jumps, modulation may be off (`trajectory.fit`, second differences);
                smoothness as above, jump price 3, 6, 12, 24; tuned with baseline and modulation tied (28 fits),
                then the modulation's pair, then the baseline's.
    joint1      the same with the true term only.
    gam_vc4, gam_vc1, gam_te, gam   the gamCopula engine (`e17_gam_edge.R`): time-varying coefficients of the four
                terms, of the true term only, a tensor smooth of time and zeta_0; time basis 4, 8, 12, 16, 20 or 24
                (24 windows: the largest possible) chosen on validation data; `gam` is the form with the best
                validation score.
    simplified  joint baseline only (reference for the scores).      oracle  the true b(t), c(t).

Measures, kept apart (all on the 1,000 test observations per window, at the true value of zeta_0)
    prediction      mean log density of the edge; reported as nats per observation below the oracle.
    function error  on the correlation scale: squared error of the modulation part (the fitted correlation minus
                    its mean over the window) and of the baseline part (that mean); root mean square over windows.
    coefficient     slope of the fitted Fisher-z predictor on zeta_0 in every window, against c(t).
    onset, offset   windows in which |slope| is at least half its largest value, against the same set for c(t)
                    (overlap of the two sets; exact agreement); for `joint` and `switching` also the pieces and
                    states the model itself reports.
    false change    `constant` and `null`: spread over windows of the fitted slope; a change of the modulation
                    reported by the model itself (`joint`: pieces; `switching`: states); and the calibrated test
                    (`trajectory.change_test`, 99 replicates, level 0.05) on `joint`.  The same test gives power
                    on the other designs.
    inclusion       decoy terms switched on by `joint` or chosen by `switching`; the true term on under `null`.

Expectations, written before the run
    1. `joint` has a smaller modulation error than `windows` and `switching` in every cell (it borrows strength
       over windows); the gap is largest at n = 100.
    2. `joint` against `smooth` and against `gam`: smaller error on `abrupt`, `recurring` and `sawtooth` (they
       cannot follow a jump without a small basis penalty), about equal on `ramp` and `constant`.
    3. `gam_vc1` is at least as good as `gam_vc4` (it is told the term), and `joint1` as `joint`.
    4. The calibrated test rejects in about 5% of `constant` and `null` datasets; the model's own pieces report a
       change more often than that (they are tuned for prediction, not calibrated).
    5. The changing background costs `joint` little: `abrupt` and `abrupt_static` have similar modulation error.
    Not known in advance: n = 100 (the validation data may be too small to tune two penalties), and whether the
    decoys are kept out.

Changed during development (54 datasets on the seeds 970000 + s, outputs e17_dev1.json and e17_dev2.json), before
any run on the seeds 980000 + s: the change test starts its unrestricted search from the restricted fit and treats
differences below 1e-6 as zero (a numerical difference of 5e-9 had been tested as a statistic); the comparator's
time basis was extended from 4-16 to 4-24 because the validation choice was 16, the largest offered, in most
datasets with n >= 250; the design `ramp_static` was added.  The estimator, its penalty grids and the measures were
not changed.

Usage: e17_joint_trajectories.py OUT.json WORKERS RLIBRARY SEED0 SEEDS REPLICATES [name=value ...]
       optional filters: designs=a,b  sizes=100,250  strengths=0.3
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

from common import GRID, cvine, fam, training_transform
from dvc_package.modulation import trajectory as tr

T, TEST = 24, 1000
ORDER = [0, 1, 2, 3]
COMPONENTS = (('linear', (0,)), ('quadratic', (0,)), ('linear', (1,)), ('product', (0, 1)))
LOWER_STATES = fam.default_states(('gaussian', 'clayton', 'gumbel', 'frank'), rotations=False)
LOWER = dict(switch=0.08, drift=0.0, activation=0.0)
SMOOTH = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, np.inf)            # times the number of observations per window
JUMP = (3.0, 6.0, 12.0, 24.0)
BASES = (4, 8, 12, 16, 20, 24)
DESIGNS = ('null', 'constant', 'abrupt', 'recurring', 'ramp', 'sawtooth', 'abrupt_static', 'ramp_static')
RUNNER = Path(__file__).resolve().parent/'e17_gam_edge.R'
ETA_MAX = cvine.ETA_MAX


def modulation(design, g):
    t = np.arange(T)
    if design == 'null':
        return np.zeros(T)
    if design == 'constant':
        return np.full(T, float(g))
    design = design.replace('_static', '')
    if design == 'abrupt':
        return g*((t >= 8) & (t < 16)).astype(float)
    if design == 'recurring':
        return g*(((t >= 2) & (t < 6)) | ((t >= 10) & (t < 14)) | ((t >= 18) & (t < 22))).astype(float)
    if design == 'ramp':
        return g*t/(T-1.0)
    if design == 'sawtooth':
        return g*np.clip((t-4)/11.0, 0.0, None)*(t < 16)
    raise ValueError(design)


def law(design, g):
    """The true vine, its baseline b(t) and modulation coefficient c(t) of the edge (2, 3 | 0, 1)."""
    t = np.arange(T)
    gauss = lambda values: [cvine.EdgeState('gaussian', float(v)) for v in values]
    if design.endswith('_static'):
        first = [gauss(np.full(T, 0.5)), [cvine.EdgeState('clayton0', 1.0) for _ in t], gauss(np.full(T, 0.45))]
        second = [gauss(np.full(T, 0.3)), gauss(np.full(T, 0.25))]
        base = np.full(T, 0.35)
    else:
        first = [gauss(np.full(T, 0.5)),
                 [cvine.EdgeState('clayton0', 1.0) if k < 14 else cvine.EdgeState('gumbel0', 1.5) for k in t],
                 gauss(np.where(t < 6, 0.3, 0.6))]
        second = [gauss(np.full(T, 0.3)), gauss(0.1+0.3*t/(T-1.0))]
        base = 0.2+0.3*((t >= 4) & (t < 19))+0.1*np.sin(2*np.pi*t/T)
    c = modulation(design, g)
    top = [[cvine.EdgeState('gaussian', components=(COMPONENTS[0],), beta=np.array([base[k], c[k]])) for k in t]]
    return cvine.DynamicCVine(list(ORDER), [first, second, top]), base, c


def lower_fit(windows):
    return cvine.fit(list(windows), order=ORDER, states=LOWER_STATES, max_order=2, max_depth=2, **LOWER)


def predictor(beta, features):
    """Fisher-z predictor (T, n) from coefficients (T, K+1) in the convention of `cvine.EdgeState.beta`."""
    return beta[:, :1]+np.einsum('tnk,tk->tn', features[:, :, :beta.shape[1]-1], beta[:, 1:])


def measures(eta, x, y, true_eta, zeta0):
    """Per-window measures of one arm from its predictor at the test observations (see the header)."""
    eta, true_eta = np.clip(eta, -ETA_MAX, ETA_MAX), np.clip(true_eta, -ETA_MAX, ETA_MAX)
    rho, true_rho = np.tanh(eta), np.tanh(true_eta)
    s = 1.0-rho*rho
    out = dict(score=(-0.5*np.log(s)-(rho*rho*(x*x+y*y)-2.0*rho*x*y)/(2.0*s)).mean(axis=1))
    for name, a, b in (('r', rho, true_rho), ('z', eta, true_eta)):
        difference = a-b
        out[f'base_{name}'] = difference.mean(axis=1)**2
        out[f'mod_{name}'] = difference.var(axis=1)
    centred = zeta0-zeta0.mean(axis=1, keepdims=True)
    out['slope'] = (centred*eta).mean(axis=1)/(centred*centred).mean(axis=1)
    out['strength'] = rho.std(axis=1)
    return {k: [round(float(v), 6) for v in values] for k, values in out.items()}


def clean(value):
    """Plain Python types for JSON; infinite penalties are written as the string 'inf'."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else ('inf' if value > 0 else '-inf' if value < 0 else 'nan')
    return value


def tune_joint(train, validation, k, n, search=True):
    """Validation choice of the penalties of the joint estimator (see the header).  Returns (trajectory, penalty)."""
    smooth = [n*s for s in SMOOTH]
    make = lambda sb, jb, sm, jm: tr.penalty(k, smooth=(sb, sm), jump=(jb, jm))
    if not search:                                                   # smooth trajectories without jumps
        best = tr.tune(train, validation, [tr.penalty(k, smooth=(sb, sm), jump=0.0) for sb in smooth for sm in smooth], search=False)
        return best[0], best[0].penalty
    best = tr.tune(train, validation, [make(s, j, s, j) for s in smooth for j in JUMP])
    if k:
        sb, jb = best[0].penalty.smooth[0], best[0].penalty.jump[0]
        again = tr.tune(train, validation, [make(sb, jb, s, j) for s in smooth for j in JUMP])
        best = again if again[1] > best[1] else best
        sm, jm = best[0].penalty.smooth[1], best[0].penalty.jump[1]
        again = tr.tune(train, validation, [make(s, j, sm, jm) for s in smooth for j in JUMP])
        best = again if again[1] > best[1] else best
    return best[0], best[0].penalty


def gam_arm(inputs, folder, library):
    """Run the comparator on the edge.  Returns (dict form -> predictor at the test observations, record)."""
    folder.mkdir(parents=True, exist_ok=True)
    for role, (u, v, features) in inputs.items():
        rows = np.concatenate([np.column_stack([v[t], u[t], np.full(len(u[t]), t/(T-1.0)), features[t]]) for t in range(T)])
        with gzip.open(folder/f'{role}.csv.gz', 'wt') as handle:
            np.savetxt(handle, rows, delimiter=',', header='u1,u2,time,z0,q0,z1,z01', comments='', fmt='%.8f')
    start = time.time()
    process = subprocess.run(['Rscript', str(RUNNER), str(folder.resolve()), str(Path(library).resolve()), ','.join(map(str, BASES))],
                             capture_output=True, text=True)
    record = dict(seconds=time.time()-start)
    out = {}
    if process.returncode != 0:
        record['failed'] = process.stderr[-600:]
    else:
        result = json.loads((folder/'gam_results.json').read_text())
        record.update(selected=result['selected'], versions=result['versions'],
                      configurations={k: dict(validation=v['validation'], converged=v['converged'], edf=v['edf'], seconds=v['seconds'],
                                              error=v['error']) for k, v in result['records'].items()})
        table = np.genfromtxt(gzip.open(folder/'test_par.csv.gz', 'rt'), delimiter=',', names=True)
        for form in result['selected']:
            out[f'gam_{form}'] = np.arctanh(np.clip(np.atleast_1d(table[form]), -0.999, 0.999)).reshape(T, TEST)
        if result['selected']:
            chosen = max(result['selected'], key=lambda form: result['records'][result['selected'][form]]['validation'])
            out['gam'] = out[f'gam_{chosen}']
            record['best_form'] = chosen
    shutil.rmtree(folder, ignore_errors=True)
    return out, record


def run(task):
    design, g, n, seed, library, scratch, replicates = task
    try:
        return _run(design, g, n, seed, library, scratch, replicates)
    except Exception:                                                # reported, never dropped
        return dict(design=design, g=g, n=n, seed=seed, error=traceback.format_exc()[-1500:])


def _run(design, g, n, seed, library, scratch, replicates):
    clock = {}
    start = time.time()
    rng = np.random.default_rng([seed, n, int(round(100*g)), DESIGNS.index(design)])
    truth, base, c = law(design, g)
    raw = {role: np.stack([cvine.simulate(truth, t, m, rng) for t in range(T)]) for role, m in (('train', n), ('validation', n), ('test', TEST))}
    data = dict(train=np.stack([cvine.pseudo_observations(x) for x in raw['train']]),
                validation=training_transform(raw['train'], raw['validation']), test=training_transform(raw['train'], raw['test']))
    lower = lower_fit(data['train'])
    inputs = {role: tr.edge_inputs(lower, list(data[role]), 2, 0, COMPONENTS) for role in data}
    train = tr.edge_data(*inputs['train'])
    validation = tr.edge_data(*inputs['validation'], centre=train.means)
    test = tr.edge_data(*inputs['test'], centre=False)
    features = np.stack(inputs['test'][2])                           # (T, TEST, 4), not centred
    zeta0 = ndtri(raw['test'][:, :, 0])                              # the true modulating coordinate
    true_eta = base[:, None]+c[:, None]*zeta0
    smallest, inside = tr.identifiability(train)
    out = dict(design=design, g=g, n=n, seed=seed, base=base.tolist(), c=c.tolist(), arms={}, selected={}, structure={},
               identifiability=dict(smallest_eigenvalue=float(smallest.min()), inside_share=[float(v) for v in inside]),
               lower_states=[[sorted({s.state for s in path}) for path in level] for level in lower.edges])
    clock['data'] = time.time()-start

    def add(name, eta):
        out['arms'][name] = measures(eta, test.x, test.y, true_eta, zeta0)

    add('oracle', true_eta)
    start = time.time()
    u, v, f = inputs['train']
    add('windows', predictor(np.array([cvine.fit_modulated_gaussian(u[t], v[t], f[t])[0] for t in range(T)]), features))
    fits = tr.switching_candidates(u, v, f, [(0,), (1,), (2,), (3,), (0, 1, 2, 3)], ['ind', 'gaussian'])
    best, plain = None, tr.edge_data(*inputs['validation'], centre=False)
    for options in GRID:
        theta, _, names = tr.switching_path(fits, **options)
        value = tr.score(theta, plain)
        if best is None or value > best[0]+1e-12:
            best = (value, theta, names, options)
    add('switching', predictor(best[1], features))
    out['selected']['switching'] = best[3]
    out['structure']['switching'] = dict(states=best[2], term_windows=[np.nonzero(best[1][:, 1+k])[0].tolist() for k in range(4)])
    clock['windows'] = time.time()-start

    start = time.time()
    empty = lambda role: tr.edge_data(inputs[role][0], inputs[role][1], [np.zeros((len(a), 0)) for a in inputs[role][0]])
    simple, pen = tune_joint(empty('train'), empty('validation'), 0, n)
    add('simplified', predictor(simple.coefficients(), features))
    out['selected']['simplified'] = dict(smooth=pen.smooth, jump=pen.jump)
    out['structure']['simplified'] = dict(pieces=[simple.pieces(0)])
    smooth, pen = tune_joint(train, validation, 4, n, search=False)
    add('smooth', predictor(smooth.coefficients(), features))
    out['selected']['smooth'] = dict(smooth=pen.smooth)
    joint, joint_penalty = tune_joint(train, validation, 4, n)
    add('joint', predictor(joint.coefficients(), features))
    out['selected']['joint'] = dict(smooth=joint_penalty.smooth, jump=joint_penalty.jump)
    out['structure']['joint'] = dict(pieces=[joint.pieces(p) for p in range(5)], changes=bool(joint.modulation_changes()), sweeps=joint.sweeps)
    first = lambda role: tr.edge_data(inputs[role][0], inputs[role][1], [a[:, :1] for a in inputs[role][2]],
                                      centre=True if role == 'train' else train.means[:, :1])
    joint1, pen = tune_joint(first('train'), first('validation'), 1, n)
    add('joint1', predictor(joint1.coefficients(), features))
    out['selected']['joint1'] = dict(smooth=pen.smooth, jump=pen.jump)
    out['structure']['joint1'] = dict(pieces=[joint1.pieces(p) for p in range(2)], changes=bool(joint1.modulation_changes()))
    clock['joint'] = time.time()-start

    if replicates:
        start = time.time()
        result = tr.change_test(list(data['train']), lower_fit, COMPONENTS, joint_penalty, replicates=replicates,
                                rng=np.random.default_rng([seed, n, 17]))
        out['change_test'] = dict(statistic=result['statistic'], p=result['p'], null_mean=float(result['null'].mean()),
                                  null_positive=float((result['null'] > 0).mean()), changes=bool(result['free'].modulation_changes()),
                                  pieces=[result['free'].pieces(p) for p in range(5)])
        clock['change_test'] = time.time()-start

    if library:
        folder = Path(tempfile.mkdtemp(prefix=f'{design}_g{g}_n{n}_s{seed}_', dir=scratch))
        predictions, record = gam_arm(inputs, folder, library)
        for name, eta in predictions.items():
            add(name, eta)
        out['gam'] = record
        clock['gam'] = record['seconds']
    out['seconds'] = {k: round(v, 2) for k, v in clock.items()}
    return clean(out)


if __name__ == '__main__':
    target, workers, library = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    seed0, seeds, replicates = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    options = dict(item.split('=') for item in sys.argv[7:])
    designs = options.get('designs', ','.join(DESIGNS)).split(',')
    sizes = [int(v) for v in options.get('sizes', '100,250,1000').split(',')]
    strengths = [float(v) for v in options.get('strengths', '0.3,0.6').split(',')]
    library = '' if library in ('none', '-') else library
    scratch = target.parent/(target.stem+'_gam')
    scratch.mkdir(parents=True, exist_ok=True)
    tasks = [(design, g, n, seed0+s, library, str(scratch), replicates)
             for design in designs for g in ((0.0,) if design == 'null' else strengths) for n in sizes for s in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, tasks, chunksize=1))
    target.write_text(json.dumps(rows))
    shutil.rmtree(scratch, ignore_errors=True)
    failed = [r for r in rows if 'error' in r]
    print(f'{len(rows)} datasets, {len(failed)} failed')
    for r in failed[:5]:
        print(r['design'], r['g'], r['n'], r['seed'], r['error'][-400:])
    seconds = {}
    for r in rows:
        for k, v in r.get('seconds', {}).items():
            seconds.setdefault((r['n'], k), []).append(v)
    for key in sorted(seconds):
        print(f'median seconds, n = {key[0]}, {key[1]}: {np.median(seconds[key]):.1f}')
    sys.exit(1 if failed else 0)
