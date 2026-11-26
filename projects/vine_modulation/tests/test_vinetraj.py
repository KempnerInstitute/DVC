"""Tests of `dvc_package.modulation.vinetraj`: candidate terms from the scan of one vine, trajectories on its edges."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dvc_package.modulation import cvine, margins, trajectory as tr, vinetraj  # noqa: E402


def hub_law(steps, strength, episode):
    """Five variables: a Gaussian star around variable 0; inside the episode the hub modulates the pair (3, 4)."""
    star = [[cvine.EdgeState('gaussian', 0.5) for _ in range(steps)] for _ in range(4)]
    second = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(3)]
    third = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(2)]
    top = [[cvine.EdgeState('gaussian', components=(('linear', (0,)),), beta=np.array([0.0, strength if t in episode else 0.0]))
            for t in range(steps)]]
    return cvine.DynamicCVine([0, 1, 2, 3, 4], [star, second, third, top])


def draw(law, steps, n, rng):
    raw = [cvine.simulate(law, t, n, rng) for t in range(steps)]
    train = [cvine.pseudo_observations(x) for x in raw]
    return train, [cvine.pseudo_observations(cvine.simulate(law, t, n, rng)) for t in range(steps)]


def test_screen_collect_and_columns():
    steps, n = 6, 500
    train, _ = draw(hub_law(steps, 0.7, range(2, 5)), steps, n, np.random.default_rng(1))
    model = cvine.fit(train, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], max_order=2, switch=0.05)
    candidates, threshold, count = vinetraj.screen(model, train, level=1e-3)
    assert count == 2*(3*1+2*2+1*3) and threshold > 3.0                      # two shapes for every (edge, earlier root)
    assert (3, 0) in candidates
    best = candidates[(3, 0)][0]
    assert best['component'] == ('linear', (0,)) and best['subset'] == (0, 3, 4) and (best['start'], best['stop']) == (2, 5)
    assert vinetraj.calibration_columns(model, 3, 0, [best['component']]) == [0]        # the hub: modulator and only parent
    assert [m for _, m in vinetraj.parents(model, 4)][0] == 0 and [m for _, m in vinetraj.parents(model, 1)] == [0]
    transforms, zeta = vinetraj.collect(model, train, [(3, 0)])
    u, v, features = tr.edge_inputs(model, train, 3, 0, [best['component']])
    assert np.allclose(transforms[(3, 0)][0][2], u[2]) and np.allclose(transforms[(3, 0)][1][2], v[2])
    assert np.allclose(zeta[2][:, 0], features[2][:, 0])


def test_fit_keeps_the_planted_term_and_tests_it():
    steps, n = 6, 500
    rng = np.random.default_rng(2)
    train, validation = draw(hub_law(steps, 0.7, range(2, 5)), steps, n, rng)
    confirm, _ = draw(hub_law(steps, 0.7, range(2, 5)), steps, n, rng)                  # used for the change test only
    with ProcessPoolExecutor(max_workers=2) as pool:
        result = vinetraj.fit(train, validation, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=9, seed=3, pool=pool,
                              confirm=confirm)
    assert result['order'] == [0, 1, 2, 3, 4]
    kept = [e for e in result['edges'] if e['kept']]
    assert len(kept) == 1
    edge = kept[0]
    assert (edge['level'], edge['edge'], tuple(edge['pair'])) == (3, 0, (3, 4)) and edge['subsets'][0] == [0, 3, 4]
    assert edge['on'][0] and [tuple(piece) for piece in edge['pieces'][1]] == [(2, 5)]
    assert edge['score'] > edge['base_score'] and edge['z'] > 5
    assert edge['test']['p'] == 1/10 and edge['test']['variables'] == [0, 3, 4]        # sub-vine: the hub and the pair
    alone = vinetraj.fit(train, validation, model=result['model'], replicates=0)        # no pool, no test: same decisions
    assert [e['kept'] for e in alone['edges']] == [e['kept'] for e in result['edges']] and 'test' not in alone['edges'][0]
    assert vinetraj.relevant(result['model'], 3, 0, [('linear', (0,))], [0]) == [0]


def test_fit_keeps_nothing_without_modulation():
    steps, n = 6, 500
    rng = np.random.default_rng(4)
    train, validation = draw(hub_law(steps, 0.0, range(0)), steps, n, rng)
    result = vinetraj.fit(train, validation, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=0, screen_level=0.05)
    assert not any(e['kept'] for e in result['edges'])


def test_fit_with_margins_over_all_windows():
    """The pipeline with margins estimated over all windows: same decisions, and the change test of the kept edge
    ranks its confirmation windows and its simulated datasets the same way."""
    steps, n = 6, 500
    rng = np.random.default_rng(2)
    law = hub_law(steps, 0.7, range(2, 5))
    raw = {role: [cvine.simulate(law, t, n, rng) for t in range(steps)] for role in ('train', 'validation', 'confirm')}
    data, flags = margins.choose(raw['train'], dict(validation=raw['validation']))
    assert flags == [True]*5
    result = vinetraj.fit(data['train'], data['validation'], order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=9, seed=3,
                          confirm=raw['confirm'], pooled=flags)
    kept = [e for e in result['edges'] if e['kept']]
    assert len(kept) == 1 and tuple(kept[0]['pair']) == (3, 4) and [tuple(piece) for piece in kept[0]['pieces'][1]] == [(2, 5)]
    assert kept[0]['test']['p'] == 1/10 and kept[0]['test']['variables'] == [0, 3, 4]
    try:
        vinetraj.fit(data['train'], data['validation'], model=result['model'], replicates=9, confirm=raw['confirm'], pooled=[True, True])
    except ValueError:
        pass
    else:
        raise AssertionError('a flag per variable is required')


def test_modulators_first_in_the_sub_vine_of_the_test():
    """The planted modulator is the third root.  By default the sub-vine of the test keeps the order of the vine;
    with ``modulators_first`` the modulator leads it, and the edge is still tested."""
    steps, n = 6, 500
    star = [[cvine.EdgeState('gaussian', 0.5) for _ in range(steps)] for _ in range(4)]
    second = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(3)]
    third = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(2)]
    top = [[cvine.EdgeState('gaussian', components=(('linear', (2,)),), beta=np.array([0.0, 0.7 if t in range(2, 5) else 0.0]))
            for t in range(steps)]]
    law = cvine.DynamicCVine([0, 1, 2, 3, 4], [star, second, third, top])
    rng = np.random.default_rng(5)
    train, validation = draw(law, steps, n, rng)
    confirm, _ = draw(law, steps, n, rng)
    plain = vinetraj.fit(train, validation, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=9, seed=3, confirm=confirm)
    first = vinetraj.fit(train, validation, model=plain['model'], replicates=9, seed=3, confirm=confirm, modulators_first=True)
    for result in (plain, first):
        kept = [e for e in result['edges'] if e['kept']]
        assert len(kept) == 1 and tuple(kept[0]['pair']) == (3, 4) and kept[0]['subsets'][0] == [2, 3, 4]
    a, b = [e for e in plain['edges'] if e['kept']][0]['test'], [e for e in first['edges'] if e['kept']][0]['test']
    assert a['variables'] == [0, 2, 3, 4] and b['variables'] == [2, 0, 3, 4]
    assert a['p'] == 1/10 and b['p'] == 1/10
