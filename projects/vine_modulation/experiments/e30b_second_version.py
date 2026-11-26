"""E30b.  Development of the next version (A2): `vinemodel2.fit` on the datasets of E26, E28, E29 and E29b.

Every one of these datasets was seen before A2 was written.  This run is development evidence: the same
observations through the first version (saved outputs of E26 and E29) and through the second, to find defects
before A2 is frozen and to see what each change does.  It is not a test of A2; that is a challenge on fresh
seeds and laws after the freeze.  What is looked at, per law and n:
  - planted terms installed, and the held-out gain of the assembled vine, against the first version;
  - changes declared for planted terms that change, for planted constant terms, and for terms that are not
    planted, by the rule of the first version (every orientation) and by that of the second (every orientation
    and every joint model of a shared pair);
  - companion terms (the pair of a constant planted term with the hub of its modulator);
  - a constant triple under a fourth-order episode on its pair (law `nested`, the datasets of E28);
  - datasets with nothing planted; the whole-model check; blocks per pair under serial dependence.
Laws: those of `e26_challenge.LAWS` and `e29_validation_batch.LAWS`, `nested` (`e28_nested_terms.generate`; no
held-out role, so no scores) and `pairwise_scale` (`e29b_scale_control`).
Usage: e30b_second_version.py OUT.json LAW N G SEED WORKERS [REPLICATES=399] [option=value ...]
  options are passed to `vinemodel2.fit` (for ablations): estimate=score, joint=0, serial=none
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
import e28_nested_terms as e28
import e29_validation_batch as e29
from dvc_package.modulation import vinemodel2  # noqa: E402

T = laws.T


def generate(law, n, g, seed):
    if law == 'nested':
        spec, raw = e28.generate(n, seed)
        return spec, raw, dict(d=e28.D, max_order=4)
    if law == 'pairwise_scale':
        import e29b_scale_control as e29b
        spec, raw = e29b.generate(law, n, g, seed)
        return spec, raw, dict(d=50)
    if law in e29.LAWS and law not in ('dense', 'overlap', 'moving', 'serial', 'margins', 'fourth'):
        spec, raw = e29.generate(law, n, g, seed)
        return spec, raw, e29.LAWS[law]
    spec, raw = e26.generate(law, n, g, seed)
    return spec, raw, e26.LAWS[law]


def run(law, n, g, seed, workers, replicates=399, **options):
    start = time.time()
    spec, raw, settings = generate(law, n, g, seed)
    clock = dict(data=round(time.time()-start, 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        result = vinemodel2.fit(raw['train'], raw['validation'], raw['confirm'], raw.get('test'), replicates=replicates, seed=seed, pool=pool,
                                max_order=settings.get('max_order', 3), **options)
    clock.update(result['seconds'])
    planted, unplanted = e26.evaluate(spec, result, None, g)
    terms = []
    for e in result['edges']:
        for k, t in e.get('test', {}).get('terms', {}).items():
            terms.append(dict(subset=t['subset'], shape=e['components'][k][0], edge_pair=list(e['pair']), features=e['features'], install_z=e['install_z'],
                              p=t['p'], declared=t['declared'], p_orientations=t['p_orientations'], declared_orientations=t['declared_orientations'],
                              joint=t['joint'], preferred=t['directed'], orientations=[dict(pair=o['pair'], modulators=o['modulators'], features=o['features'],
                                                                                           gain=o['gain'], z=o['z'], shown=o['shown'], p=o['p'], vine=o['vine'], block=o['block'])
                                                                                      for o in t['orientations']]))
    r = np.array(result['serial']['r'])
    summary = dict(law=law, d=settings['d'], n=n, g=g, seed=seed, replicates=replicates, options={k: str(v) for k, v in options.items()},
                   order=result['order'][:12], block=result['block'], inflation=result['inflation'],
                   serial=dict(largest=float(r.max()), median=float(np.median(r)), blocks=sorted(set(result['serial']['blocks']))),
                   flags={kind: result['flags'].count(kind) for kind in ('pooled', 'aligned', 'window')},
                   components=result['components'], screened_edges=len(result['edges']), kept_edges=int(sum(e['kept'] for e in result['edges'])),
                   installed_edges=int(sum(e['installed'] for e in result['edges'])), scores=result.get('scores'), whole=result['whole'],
                   first_pass_edges=int(sum(e.get('confirm_z') is not None for e in result['edges'])),
                   planted=planted, unplanted=unplanted, terms=terms,
                   groups=[dict(pair=g_['pair'], modulators=g_['modulators'], members=len(g_['members']), p=g_.get('p'), block=g_.get('block')) for g_ in result['groups']],
                   seconds=clock)
    edges = [{k: v for k, v in e.items() if k not in ('coefficients', 'test')} for e in result['edges']]
    return summary, edges


def describe(s):
    planted = s['planted']
    installed = [r for r in planted if r['installed']]
    changing = [r for r in installed if r['shape'] != 'constant']
    constant = [r for r in installed if r['shape'] == 'constant']
    by = {tuple(sorted(t['subset'])): t for t in s['terms']}
    old = lambda rows: sum(bool(by.get(tuple(sorted(r['subset'])), {}).get('declared_orientations')) for r in rows)
    others = [u for u in s['unplanted'] if u['installed']]
    known = {tuple(sorted(r['subset'])) for r in planted}
    extra = [t for t in s['terms'] if tuple(sorted(t['subset'])) not in known]
    gain = None if s['scores'] is None else s['scores']['assembled']-s['scores']['simplified']
    print(f"{s['law']} n {s['n']} g {s['g']} seed {s['seed']}: blocks {s['serial']['blocks']}, largest inflation {s['inflation']:.2f}; edges screened "
          f"{s['screened_edges']}, kept {s['kept_edges']}, installed {s['installed_edges']} (on the validation windows {s.get('first_pass_edges', '-')}); "
          f"planted installed {len(installed)} of {len(planted)}; "
          f"changes declared: changing terms {sum(r['change'] for r in changing)} of {len(changing)} (first version's rule {old(changing)}), constant terms "
          f"{sum(r['change'] for r in constant)} of {len(constant)} ({old(constant)}); terms not planted installed {len(others)}, declared "
          f"{sum(t['declared'] for t in extra)} ({sum(t['declared_orientations'] for t in extra)}); joint models {len(s['groups'])}; whole model "
          f"{s['whole']['gain']:+.4f} (z {s['whole']['z']:.1f}{', simplified vine returned' if s['whole']['fallback'] else ''}); held-out gain "
          f"{'-' if gain is None else format(gain, '+.4f')}; minutes {sum(v for v in s['seconds'].values())/60:.1f}", flush=True)


if __name__ == '__main__':
    target, law, n, g, seed, workers = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    rest = sys.argv[7:]
    replicates = int(rest.pop(0)) if rest and '=' not in rest[0] else 399
    options = {}
    for item in rest:
        name, value = item.split('=', 1)
        options[name] = dict(estimate=str, joint=lambda v: bool(int(v)), serial=lambda v: None if v == 'none' else v, orientations=lambda v: bool(int(v)))[name](value)
    summary, edges = run(law, n, g, seed, workers, replicates, **options)
    target.write_text(json.dumps(dict(summary=summary, edges=edges), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    describe(summary)
