"""E80: score the fourth-order episodes of the law `quartets` and compare them with E31's law `nested`.

    e80_summary.py RESULTS_ROOT WORKED_JSON VERIFICATION_FOLDER

Reads RESULTS_ROOT/e80/e80_quartets_n*_g*_s*.json (primary, seeds 991800 to 991803), RESULTS_ROOT/e80/
e80p_quartets_n*_g0.6_s*.json (paired, E31's seeds 990000 to 990002) and RESULTS_ROOT/e31/e31_a2_nested_*.json (checked
against the hashes in v3/paper/evidence/worked.json).  Scoring as v3/paper/worked_examples.py (`challenge`) and
v3/paper/verify_challenge.py: per planted term, retained (`installed`), change declared (`change`), every active window
(`installed` and windows_on == windows_true), and the combined endpoint of Fig 3E (all three); per dataset, terms that
are not planted (retained, declared changing) and the held-out gain of the assembled over the simplified vine.
Writes e80_episodes.tsv, e80_datasets.tsv and e80_summary.json (expectations q1 to q6 and the decision rule of
E80_QUARTETS_ALONE.md).
"""
import csv
import hashlib
import json
import sys
from math import comb, sqrt
from pathlib import Path

SIZES = (300, 1000)
PRIMARY = (991800, 991801, 991802)
CONTROL = 991803
PAIRED = (990000, 990001, 990002)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k/n
    centre, half = (p+z*z/(2*n))/(1+z*z/n), z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return (round(centre-half, 4), round(centre+half, 4))


def mcnemar(only_a, only_b):
    """Two-sided exact test on the discordant pairs."""
    n = only_a+only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(only_a, only_b)+1))/2**n
    return min(1.0, 2*tail)


def load(path, expected_hash=None):
    raw = path.read_bytes()
    if expected_hash is not None and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"{path.name} differs from its receipt")
    s = json.loads(raw)['summary']
    if s['d'] != 50 or s['replicates'] != 399:
        raise ValueError(f"unexpected settings in {path.name}")
    return s, hashlib.sha256(raw).hexdigest()


def episodes(study, s):
    rows = []
    for p in s['planted']:
        exact = bool(p['installed'] and p['windows_on'] == p['windows_true'])
        rows.append(dict(study=study, law=s['law'], n=s['n'], seed=s['seed'], order=p['order'], shape=p['shape'], kind=p['kind'],
                         pair='-'.join(map(str, p['pair'])), modulators='-'.join(map(str, p['modulators'])),
                         subset='-'.join(map(str, p['subset'])), installed=int(bool(p['installed'])), changed=int(bool(p['change'])),
                         exact=int(exact), success=int(bool(p['installed'] and p['change'] and exact)),
                         windows_on=' '.join(map(str, p['windows_on'])), windows_true=' '.join(map(str, p['windows_true'])),
                         p_change='' if p['p'] is None else p['p']))
    return rows


def dataset(study, s, digest):
    terms = s['unplanted']
    return dict(study=study, law=s['law'], n=s['n'], g=s['g'], seed=s['seed'], planted_terms=len(s['planted']),
                fourth_order_episodes=sum(p['order'] == 4 and p['shape'] == 'a' for p in s['planted']),
                extra_installed=sum(bool(t['installed']) for t in terms), extra_changing=sum(bool(t['change']) for t in terms),
                test_gain=round(s['scores']['assembled']-s['scores']['simplified'], 6), simplified_returned=int(bool(s['whole']['fallback'])),
                minutes=round(sum(v for k, v in s['seconds'].items() if k != 'strict')/60.0, 1), sha256=digest)


def tally(rows):
    out = {}
    for key in ('installed', 'changed', 'exact', 'success'):
        k = sum(r[key] for r in rows)
        out[key] = dict(k=k, n=len(rows), wilson95=wilson(k, len(rows)))
    return out


def main(root, worked, folder):
    receipts = {(r['law'], r['n'], r['seed']): r['sha256'] for r in json.loads(Path(worked).read_text())['challenge']}
    studies = {}
    for n in SIZES:
        for seed in PRIMARY+(CONTROL,):
            g = 0.0 if seed == CONTROL else 0.6
            studies[('quartets', n, seed)] = load(root/'e80'/f'e80_quartets_n{n}_g{g}_s{seed}.json')
        for seed in PAIRED:
            path = root/'e80'/f'e80p_quartets_n{n}_g0.6_s{seed}.json'
            if path.exists():
                studies[('quartets_paired', n, seed)] = load(path)
        for seed in PAIRED+(990003,):
            g = 0.0 if seed == 990003 else 0.6
            studies[('nested', n, seed)] = load(root/'e31'/f'e31_a2_nested_n{n}_g{g}_s{seed}.json', receipts[('nested', n, seed)])
    episode_rows, dataset_rows = [], []
    for (study, n, seed), (s, digest) in sorted(studies.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        episode_rows += episodes(study, s)
        dataset_rows.append(dataset(study, s, digest))
    fourth = lambda study, n=None: [r for r in episode_rows if r['study'] == study and r['order'] == 4 and r['shape'] == 'a' and (n is None or r['n'] == n)]
    report = dict(studies={})
    for study in ('quartets', 'nested', 'quartets_paired'):
        if not fourth(study):
            continue
        report['studies'][study] = {str(n): tally(fourth(study, n)) for n in SIZES}
        report['studies'][study]['both'] = tally(fourth(study))
        planted = [d for d in dataset_rows if d['study'] == study and d['g'] > 0]
        controls = [d for d in dataset_rows if d['study'] == study and d['g'] == 0]
        report['studies'][study]['datasets'] = dict(
            planted=len(planted), gain_positive=sum(d['test_gain'] > 0 for d in planted), gain_mean=round(sum(d['test_gain'] for d in planted)/max(len(planted), 1), 4),
            gain_min=min((d['test_gain'] for d in planted), default=None), simplified_returned=sum(d['simplified_returned'] for d in planted),
            extra_installed=sum(d['extra_installed'] for d in planted), extra_changing=sum(d['extra_changing'] for d in planted),
            datasets_with_extra=sum(d['extra_installed'] > 0 for d in planted),
            controls=len(controls), control_installed=sum(d['extra_installed'] for d in controls), control_changing=sum(d['extra_changing'] for d in controls),
            minutes={str(n): [d['minutes'] for d in planted+controls if d['n'] == n] for n in SIZES})
    # third-order terms of `nested` (for context)
    third = [r for r in episode_rows if r['study'] == 'nested' and r['order'] == 3]
    report['nested_constant_triples'] = dict(installed=sum(r['installed'] for r in third), changed=sum(r['changed'] for r in third), n=len(third))
    # paired comparison: each quartet episode in E31's dataset (with triples) and in the same dataset without them
    if fourth('quartets_paired'):
        paired = {}
        index = {(r['n'], r['seed'], r['subset']): r for r in fourth('nested')}
        for key in ('installed', 'changed', 'exact', 'success'):
            both = only_nested = only_quartets = neither = 0
            for r in fourth('quartets_paired'):
                a, b = index[(r['n'], r['seed'], r['subset'])][key], r[key]
                both += a and b
                only_nested += a and not b
                only_quartets += b and not a
                neither += (not a) and (not b)
            paired[key] = dict(both=both, nested_only=only_nested, quartets_only=only_quartets, neither=neither,
                               mcnemar_p=round(mcnemar(only_nested, only_quartets), 4))
        report['paired'] = paired
    q = report['studies']['quartets']
    d = q['datasets']
    report['expectations'] = dict(
        q1_no_failure=len([x for x in dataset_rows if x['study'] == 'quartets']) == 8,
        q2_controls=d['control_changing'] == 0 and d['control_installed'] <= 2,
        q3_gain_all_planted=d['gain_positive'] == d['planted'] == 6,
        q4_retained_1000=q['1000']['installed']['k'] >= 9,
        q5_declared_1000=q['1000']['changed']['k'] >= 6,
        q6_extra_changing=d['extra_changing'] <= 1)
    k = q['both']['success']['k']
    report['decision'] = dict(combined_quartets=k, combined_nested=report['studies']['nested']['both']['success']['k'], of=q['both']['success']['n'],
                              conclusion='design' if k >= 12 else 'order' if k <= 8 else 'mixed')
    folder = Path(folder)
    for name, rows in (('e80_episodes.tsv', episode_rows), ('e80_datasets.tsv', dataset_rows)):
        with (folder/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    (folder/'e80_summary.json').write_text(json.dumps(report, indent=1, sort_keys=True)+'\n')
    return report


if __name__ == '__main__':
    print(json.dumps(main(Path(sys.argv[1]), sys.argv[2], sys.argv[3]), indent=1, sort_keys=True))
