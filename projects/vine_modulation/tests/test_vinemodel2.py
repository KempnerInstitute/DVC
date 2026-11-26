"""Tests of `dvc_package.modulation.vinemodel2`, the second version of the pipeline: no calibration in the model, tests with the
profile estimate, joint tests of terms that share two variables, serial dependence per pair, the rule over the
orientations that show a term."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dvc_package.modulation import cvine, dynvine, halfsample as hs, vinemodel, vinemodel2  # noqa: E402
from test_vinetraj import hub_law  # noqa: E402


def raw(law, steps, n, rng):
    return [cvine.simulate(law, t, n, rng) for t in range(steps)]


def test_costs_per_edge_reduce_to_one_cost():
    """With the same autocorrelation for every variable the vine is the one fitted with the scaled costs that the
    first version passes for all edges; with a fast variable its edges keep their unscaled costs."""
    steps, n = 5, 400
    rng = np.random.default_rng(0)
    train = [cvine.pseudo_observations(x) for x in raw(hub_law(steps, 0.0, range(0)), steps, n, rng)]
    r = 0.8
    inflation = (1+r*r)/(1-r*r)
    old = cvine.fit(train, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian', 'clayton0'], max_order=2, switch=0.05*inflation,
                    activation=(inflation-1)/n)
    new = cvine.fit(train, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian', 'clayton0'], max_order=2, switch=0.05, serial=[r]*5)
    for a, b in zip(old.edges, new.edges):
        for x, y in zip(a, b):
            assert [s.state for s in x] == [s.state for s in y] and np.allclose([s.par for s in x], [s.par for s in y], equal_nan=True)
    plain = cvine.fit(train, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian', 'clayton0'], max_order=2, switch=0.05)
    mixed = cvine.fit(train, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian', 'clayton0'], max_order=2, switch=0.05, serial=[0.0, 0.9, 0.9, 0.9, 0.9])
    assert [[s.state for s in x] for x in mixed.edges[0]] == [[s.state for s in x] for x in plain.edges[0]]     # edges of the fast root
    try:
        cvine.fit(train, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], max_order=2, serial=[0.5]*3)
    except ValueError:
        pass
    else:
        raise AssertionError('one autocorrelation per variable')


def test_serial_profile_and_margins_per_variable():
    rng = np.random.default_rng(1)
    steps, n = 4, 800
    a = np.array([0.9, 0.5, 0.0])
    windows = []
    for t in range(steps):
        e = rng.normal(size=(n, 3))
        x = np.empty_like(e)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = a*x[i-1]+np.sqrt(1-a*a)*e[i]
        x[:, 2] += 0.8*t                                             # the fast variable drifts
        windows.append(x)
    r, blocks = vinemodel2.serial_profile(windows)
    assert np.all(np.abs(r[:2]-a[:2]) < 0.06) and r[2] == 0.0 and blocks[0] > blocks[1] > blocks[2] == 1
    assert vinemodel2.serial_profile(windows, None)[1] == [1, 1, 1] and vinemodel2.serial_profile(windows, [0.5, 0.5, 0.5])[1][0] == hs.block_of(0.5, n)
    data, kinds = vinemodel2.choose_margins(windows, dict(validation=windows), 0.05, True, blocks)
    assert kinds[2] in ('aligned', 'window') and kinds[0] == 'pooled' and set(data) == {'train', 'validation'}
    same, flags = vinemodel2.choose_margins(windows, None, 0.05, True, [1, 1, 1])
    assert flags[2] in ('aligned', 'window')


def test_groups_of():
    terms = [dict(name='planted', subset=[5, 7, 2], shape='own_linear', strength=9.0),
             dict(name='companion', subset=[0, 5, 7], shape='linear', strength=4.0),
             dict(name='alone', subset=[1, 3, 4], shape='quadratic', strength=5.0)]
    groups = vinemodel2.groups_of(terms)
    assert len(groups) == 1 and groups[0]['pair'] == (5, 7) and groups[0]['modulators'] == [2, 0]
    assert groups[0]['members'] == [('planted', (2,), 'own_linear'), ('companion', (0,), 'own_linear')]
    nested = [dict(name='triple', subset=[1, 2, 3], shape='linear', strength=5.0), dict(name='quadruple', subset=[1, 2, 3, 4], shape='product', strength=6.0)]
    groups = vinemodel2.groups_of(nested)
    assert [g['pair'] for g in groups] == [(1, 2), (1, 3), (2, 3)]
    assert groups[0]['members'] == [('quadruple', (3, 4), 'own_product'), ('triple', (3,), 'own_linear')] and groups[0]['modulators'] == [3, 4]
    two_shapes = [dict(name='l', subset=[1, 2, 3], shape='linear', strength=5.0), dict(name='q', subset=[1, 2, 3], shape='quadratic', strength=4.0)]
    assert len(vinemodel2.groups_of(two_shapes)) == 3 and vinemodel2.groups_of(two_shapes)[0]['members'][1] == ('q', (3,), 'own_quadratic')
    many = [dict(name=k, subset=[0, 1, 10+k], shape='linear', strength=float(k)) for k in range(6)]
    cut = vinemodel2.groups_of(many, max_group=3)
    assert len(cut) == 1 and [m[0] for m in cut[0]['members']] == [5, 4, 3] and cut[0]['modulators'] == [15, 14, 13]
    assert vinemodel2.groups_of(terms[2:]) == []


def test_second_version_on_the_planted_term():
    """Independent observations, one planted term: installed without a calibration, tested in every orientation,
    no joint model; the whole model is above the simplified vine."""
    steps, n = 6, 500
    rng = np.random.default_rng(2)
    law = hub_law(steps, 0.7, range(2, 5))
    train, validation, confirm, test = (raw(law, steps, m, rng) for m in (n, n, n, 800))
    result = vinemodel2.fit(train, validation, confirm, test, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=19, seed=3,
                            features='own', tolerance=0.0)
    assert result['flags'] == ['pooled']*5 and result['serial']['blocks'] == [1]*5 and result['block'] == 1 and result['inflation'] == 1.0
    installed = [e for e in result['edges'] if e['installed']]
    assert len(installed) == 1 and tuple(installed[0]['pair']) == (3, 4) and installed[0]['calibration'] is False
    edge = installed[0]
    assert [tuple(piece) for piece in edge['pieces'][1]] == [(2, 5)] and not result['vine'].calibrations
    term = edge['test']['terms'][0]
    assert len(term['orientations']) == 3 and term['joint'] == [] and term['p'] == term['p_orientations'] == max(o['p'] for o in term['orientations'])
    assert next(o['p'] for o in term['orientations'] if o['vine']) == 1/20 and result['groups'] == []
    assert result['whole']['gain'] > 0.01 and result['whole']['z'] > 3 and not result['whole']['fallback'] and result['whole']['windows'] == 'confirmation'
    assert edge['confirm_z'] > 3 and edge['confirm_gain'] > 0.01 and edge['install_z'] > 3
    without = vinemodel2.fit(train, validation, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], features='own')
    kept = [e for e in without['edges'] if e['installed']]
    assert len(kept) == 1 and kept[0]['confirm_z'] is None and without['whole']['windows'] == 'validation' and without['whole']['gain'] > 0.01
    assert result['scores']['assembled'] > result['scores']['simplified']+0.01
    first = vinemodel.fit(train, validation, confirm, test, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=19, seed=3, features='own')
    old = [e for e in first['edges'] if e['installed']][0]
    assert old['pieces'] == edge['pieces'] and np.allclose(old['coefficients'], edge['coefficients'], atol=0.05)
    # one term and no tolerance: the tests are those of the first version (same seeds, same halves, the profile estimate is the score estimate)
    assert [o['p'] for o in old['test']['terms'][0]['orientations']] == [o['p'] for o in term['orientations']]
    usual = vinemodel2.fit(train, validation, confirm, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=19, seed=3, features='own')
    again = [e for e in usual['edges'] if e['installed']][0]['test']['terms'][0]
    assert next(o['p'] for o in again['orientations'] if o['vine']) == 1/20          # an episode of 0.7 is far outside the tolerance
    assert all(a['p'] >= b['p'] for a, b in zip(again['orientations'], term['orientations']))
    nothing = vinemodel2.fit(*(raw(hub_law(steps, 0.0, range(0)), steps, n, rng) for _ in range(4)), order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'],
                             replicates=9, screen_level=0.05)
    assert not any(e['installed'] for e in nothing['edges']) and nothing['whole'] == dict(gain=0.0, z=0.0, fallback=False, windows='confirmation')
    for bad in (dict(features='other'), dict(margin='other'), dict(serial='other')):
        try:
            vinemodel2.fit(train, validation, **bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def companion_law(steps, c, rho):
    """Four variables: 0 a hub, 1 a modulator with Gaussian dependence rho[t] on the hub, 2 and 3 a pair whose
    Fisher-z correlation is c[t] times the own score of 1 = c[t] (rho[t] z_0 + sqrt(1 - rho[t]^2) e_1)."""
    star = [[cvine.EdgeState('gaussian', float(rho[t])) for t in range(steps)]] + [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(2)]
    level1 = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(2)]
    top = [[cvine.EdgeState('gaussian', components=(('linear', (0,)), ('linear', (1,))),
                            beta=np.array([0.0, c[t]*rho[t], c[t]*np.sqrt(1-rho[t]**2)])) for t in range(steps)]]
    return cvine.DynamicCVine([0, 1, 2, 3], [star, level1, top])


def test_joint_model_explains_the_companion_term():
    """A constant term whose modulator's dependence on its hub rises.  In the root order hub, leaf, modulator, leaf
    the edge of the two leaves has the hub and not the modulator among its conditioning variables: the vine holds
    the companion term (pair with the hub), whose coefficient rises, and the planted term on another edge.  The
    companion is rejected in the orientations of its own subset and not in the joint model of the pair."""
    steps, n = 8, 1500
    rho = np.where(np.arange(steps) < 4, 0.3, 0.85)
    law = companion_law(steps, np.full(steps, 0.7), rho)
    rng = np.random.default_rng(7)
    train, validation, confirm = (raw(law, steps, n, rng) for _ in range(3))
    result = vinemodel2.fit(train, validation, confirm, order=[0, 2, 1, 3], states=['ind', 'gaussian'], replicates=19, seed=1)
    terms = {tuple(term['subset']): term for e in result['edges'] if e['installed'] for term in e['test']['terms'].values()}
    assert (0, 2, 3) in terms and (1, 2, 3) in terms                 # the companion and the planted term
    companion, planted = terms[(0, 2, 3)], terms[(1, 2, 3)]
    assert len(result['groups']) == 1 and result['groups'][0]['pair'] == [2, 3] and sorted(result['groups'][0]['modulators']) == [0, 1]
    assert companion['p_orientations'] == 1/20                       # every orientation of its own subset rejects
    assert len(companion['joint']) == 1 and companion['joint'][0]['p'] > 0.1 and companion['p'] == companion['joint'][0]['p']
    assert not companion['declared'] and planted['p'] > 1/20 and not planted['declared']
    estimates = np.array(result['groups'][0]['estimates'])           # (T, 2): the modulator's own score, the hub's
    index = result['groups'][0]['modulators'].index(0)
    assert np.abs(estimates[:, index]).max() < 0.15 and np.all(np.abs(estimates[:, 1-index]-0.7) < 0.2)
    # an episode of the interaction itself is still found, in the orientations and in the joint model
    law = companion_law(steps, np.where((np.arange(steps) >= 2) & (np.arange(steps) < 6), 0.8, 0.0), np.full(steps, 0.6))
    train, validation, confirm = (raw(law, steps, n, rng) for _ in range(3))
    result = vinemodel2.fit(train, validation, confirm, order=[0, 2, 1, 3], states=['ind', 'gaussian'], replicates=19, seed=1)
    terms = {tuple(term['subset']): term for e in result['edges'] if e['installed'] for term in e['test']['terms'].values()}
    assert (1, 2, 3) in terms and terms[(1, 2, 3)]['p'] == 1/20 and all(g['p'] == 1/20 for g in terms[(1, 2, 3)]['joint'])


def even_law(steps, strength, episode):
    """`hub_law` with a modulation that is an even function of the hub's score, (z^2 - 1) / sqrt(2)."""
    star = [[cvine.EdgeState('gaussian', 0.5) for _ in range(steps)] for _ in range(4)]
    second = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(3)]
    third = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(2)]
    top = [[cvine.EdgeState('gaussian', components=(('quadratic', (0,)),), beta=np.array([0.0, strength if t in episode else 0.0]))
            for t in range(steps)]]
    return cvine.DynamicCVine([0, 1, 2, 3, 4], [star, second, third, top])


def test_orientations_that_show_the_term():
    """An episode of a modulation that is an even function of the modulator.  With the modulator in the pair there
    is no correlation to modulate: only the vine's orientation shows the term, and the two others do not veto the
    change, as they do under the rule over every orientation."""
    steps, n = 6, 800
    rng = np.random.default_rng(5)
    law = even_law(steps, 0.7, range(2, 5))
    train, validation, confirm = (raw(law, steps, n, rng) for _ in range(3))
    result = vinemodel2.fit(train, validation, confirm, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=19, seed=3)
    installed = [e for e in result['edges'] if e['installed']]
    assert len(installed) == 1 and tuple(installed[0]['pair']) == (3, 4)
    kept = [shape for (shape, _), on in zip(installed[0]['components'], installed[0]['on']) if on]
    assert len(kept) == 1 and kept[0].endswith('quadratic')
    term = next(iter(installed[0]['test']['terms'].values()))
    own = next(o for o in term['orientations'] if o['vine'])
    others = [o for o in term['orientations'] if not o['vine']]
    assert len(others) == 2 and not any(o['shown'] for o in others) and max(o['gain'] for o in others) < 1e-3 < own['gain']
    assert own['shown'] and own['z'] > 3 and own['p'] == 1/20 and term['p'] == 1/20 and term['declared']
    assert term['p_orientations'] == max(o['p'] for o in term['orientations']) > 1/20 and not term['declared_orientations']
    every = vinemodel2.fit(train, validation, confirm, order=[0, 1, 2, 3, 4], states=['ind', 'gaussian'], replicates=19, seed=3, show_z=None)
    old = next(iter([e for e in every['edges'] if e['installed']][0]['test']['terms'].values()))
    assert all(o['shown'] for o in old['orientations']) and old['p'] == old['p_orientations'] == term['p_orientations'] and not old['declared']
