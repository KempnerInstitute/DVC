"""Tests of `dvc_package.modulation.trajectory`: joint coefficient trajectories of one edge."""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dvc_package.modulation import cvine, margins, trajectory as tr  # noqa: E402


def planted(steps=24, n=250, g=0.6, seed=0, k=2, base=None, coefficient=None):
    """One edge: Gaussian copula with Fisher-z correlation base(t) + coefficient(t) z_0; other features are decoys."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    base = 0.2+0.3*((t >= 4) & (t < 19)) if base is None else np.asarray(base, dtype=float)
    coefficient = g*((t >= 8) & (t < 16)) if coefficient is None else np.asarray(coefficient, dtype=float)
    u, v, f = [], [], []
    for i in t:
        z, x, e = rng.normal(size=(n, k)), rng.normal(size=n), rng.normal(size=n)
        r = np.tanh(base[i]+coefficient[i]*z[:, 0])
        u.append(ndtr(r*x+np.sqrt(1-r*r)*e))
        v.append(ndtr(x))
        f.append(z)
    return u, v, f, base, coefficient


def star(steps=10, n=300, g=0.6, seed=0, episode=range(3, 7), raw=False):
    """Three variables: Gaussian star around variable 0, which modulates the pair (1, 2) inside the episode.
    Pseudo-observations from ranks within every window, or with ``raw`` the values themselves."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(steps):
        x = rng.normal(size=(n, 3))
        z0 = x[:, 0]
        r = np.tanh(0.3+(g if t in episode else 0.0)*z0)
        e1, e2 = x[:, 1].copy(), r*x[:, 1]+np.sqrt(1-r*r)*x[:, 2]
        x[:, 1], x[:, 2] = 0.5*z0+np.sqrt(0.75)*e1, 0.5*z0+np.sqrt(0.75)*e2
        out.append(x if raw else cvine.pseudo_observations(x))
    return out


def test_trajectory_derivatives():
    u, v, f, _, _ = planted(steps=5, n=80, k=2, seed=3)
    f[2] = f[2][:60]                                                # windows of unequal size
    u[2], v[2] = u[2][:60], v[2][:60]
    edge = tr.edge_data(u, v, f)
    assert edge.counts.tolist() == [80, 80, 60, 80, 80]
    theta = np.random.default_rng(0).normal(0, 0.3, (5, 3))
    gradient, hessian, loglik = tr._derivatives(theta, edge, ridge=0.01)

    def value(th):
        return -tr._terms(th, edge)[0].sum()+0.01*(edge.counts[:, None]*th[:, 1:]**2).sum()
    assert abs(loglik+value(theta)-0.01*(edge.counts[:, None]*theta[:, 1:]**2).sum()) < 1e-9
    for index in ((0, 0), (2, 1), (4, 2)):
        step = np.zeros_like(theta)
        step[index] = 1e-6
        numeric = (value(theta+step)-value(theta-step))/2e-6
        assert abs(numeric-gradient[index]) < 1e-5*(1+abs(numeric))
    assert np.all(np.linalg.eigvalsh(hessian) > 0)
    # expected information of the Fisher-z correlation is 1 + rho^2 per observation
    u, v, f, _, _ = planted(steps=1, n=200000, k=1, seed=4, base=[np.arctanh(0.5)], coefficient=[0.0])
    edge = tr.edge_data(u, v, [np.zeros((200000, 0))])
    theta = np.full((1, 1), np.arctanh(0.5))
    numeric = (tr._derivatives(theta+1e-4, edge)[0]-tr._derivatives(theta-1e-4, edge)[0])/2e-4
    assert abs(numeric[0, 0]/200000-1.25) < 0.02 and abs(tr._derivatives(theta, edge)[1][0, 0, 0]/200000-1.25) < 1e-9


def test_trajectory_pieces_dynamic_programme():
    """`best_pieces` is exact: it equals exhaustive search over all pieces and on/off patterns."""
    rng = np.random.default_rng(1)
    steps = 6
    labels = [np.r_[0, np.cumsum(cuts)] for cuts in itertools.product((0, 1), repeat=steps-1)]
    for trial in range(48):
        h = rng.uniform(20, 200, steps)
        m = rng.normal(0, 0.3, steps)+(rng.uniform() < 0.5)*0.5*(np.arange(steps) >= 3)
        smooth, order, jump = (0.0, 5.0, 50.0, np.inf)[trial % 4], 1+(trial//4) % 2, (0.5, 2.0, 6.0)[trial % 3]
        include = (0.0, 1.5)[(trial//8) % 2]
        for can_off in (False, True):
            piece, on = tr.best_pieces(h, m, smooth, order, jump, can_off, first_free=not can_off, include=include if can_off else 0.0)
            got = tr.quadratic_value(h, m, piece, on, smooth, order, jump, not can_off, include if can_off else 0.0)
            masks = itertools.product((False, True), repeat=steps) if can_off else [(True,)*steps]
            best = min(tr.quadratic_value(h, m, label, np.array(mask), smooth, order, jump, not can_off, include if can_off else 0.0)
                       for mask in masks for label in labels)
            assert got <= best+1e-8
    # `whole`: one piece over all windows, or off
    h, m = np.full(steps, 100.0), np.r_[np.zeros(3), np.full(3, 0.5)]
    piece, on = tr.best_pieces(h, m, np.inf, 1, 2.0, True, False, whole=True)
    assert on.all() and len(set(piece)) == 1
    assert not tr.best_pieces(h, 0.01*m, np.inf, 1, 2.0, True, False, whole=True)[1].any()
    assert tr.best_pieces(h, m, np.inf, 1, 2.0, True, False)[1].tolist() == [False]*3+[True]*3


def test_trajectory_untied_windows_equal_window_fits():
    """With smoothness 0 and no search the joint solver is the window-by-window fit of `cvine`."""
    u, v, f, _, _ = planted(steps=6, n=200, seed=5)
    edge = tr.edge_data(u, v, f)
    free = tr.fit(edge, tr.penalty(2, smooth=0.0, jump=0.0), search=False)
    reference = np.array([cvine.fit_modulated_gaussian(u[t], v[t], f[t])[0] for t in range(6)])
    assert np.abs(free.coefficients()-reference).max() < 5e-3
    plain = tr.edge_data(u, v, f, centre=False)

    def value(beta):
        return -tr._terms(beta, plain)[0].sum()+1e-3*(plain.counts[:, None]*beta[:, 1:]**2).sum()
    assert value(free.coefficients()) <= value(reference)+1e-6
    assert abs(tr.score(free.theta, edge)-tr.score(free.coefficients(), plain)) < 1e-12        # centring is a reparameterisation


def test_trajectory_recovers_pieces():
    u, v, f, base, coefficient = planted()
    edge = tr.edge_data(u, v, f)
    fitted = tr.fit(edge, tr.penalty(2, smooth=np.inf, jump=6.0))
    assert fitted.pieces(0) == [(0, 4), (4, 19), (19, 24)] and fitted.pieces(1) == [(8, 16)] and fitted.pieces(2) == []
    assert fitted.modulation_changes()
    assert np.abs(fitted.theta[8:16, 1]-0.6).max() < 0.08 and np.all(fitted.theta[:8, 1] == 0) and np.all(fitted.theta[:, 2] == 0)
    assert np.abs(fitted.theta[:, 0]-base).max() < 0.15
    # smooth trajectories (no search) follow a ramp; the joint fit has a smaller error than untied windows
    ramp = 0.6*np.arange(24)/23.0
    u, v, f, _, _ = planted(base=np.full(24, 0.3), coefficient=ramp, seed=7)
    edge = tr.edge_data(u, v, f)
    smooth = tr.fit(edge, tr.penalty(2, smooth=250*1e3, jump=0.0), search=False)
    untied = tr.fit(edge, tr.penalty(2, smooth=0.0, jump=0.0), search=False)
    assert np.sqrt(np.mean((smooth.theta[:, 1]-ramp)**2)) < 0.5*np.sqrt(np.mean((untied.theta[:, 1]-ramp)**2))
    # constant modulation: one piece; the restricted fit is not better than the free one
    u, v, f, _, _ = planted(coefficient=np.full(24, 0.4), seed=2)
    edge = tr.edge_data(u, v, f)
    pen = tr.penalty(2, smooth=np.inf, jump=6.0)
    free, constant = tr.fit(edge, pen), tr.fit(edge, tr.constant_modulation(pen))
    assert constant.pieces(1) == [(0, 24)] and constant.pieces(2) == [] and np.ptp(constant.theta[:, 1]) < 1e-12
    assert free.objective <= constant.objective+1e-6


def test_trajectory_search_against_exhaustive_pieces():
    """The search over pieces is local (one trajectory at a time).  On small problems it finds the pieces that
    exhaustive search finds."""
    steps = 4
    labels = [np.r_[0, np.cumsum(cuts)] for cuts in itertools.product((0, 1), repeat=steps-1)]
    hits = 0
    for seed in range(8):
        base = np.where(np.arange(steps) < 2, 0.1, 0.7) if seed % 2 else np.full(steps, 0.3)
        coefficient = np.array([0.0, 0.7, 0.7, 0.0]) if seed % 4 < 2 else np.zeros(steps)
        u, v, f, _, _ = planted(steps=steps, n=150, seed=40+seed, k=1, base=base, coefficient=coefficient)
        edge = tr.edge_data(u, v, f)
        pen = tr.penalty(1, smooth=(np.inf, 300.0)[seed % 2], jump=4.0, order=1+(seed//2) % 2)
        fitted = tr.fit(edge, pen)
        best, seen = np.inf, set()
        for label0, label1 in itertools.product(labels, labels):
            for mask in itertools.product((False, True), repeat=steps):
                key = (tuple(tr.runs(label0, np.ones(steps, dtype=bool))), tuple(tr.runs(label1, np.array(mask))))
                if key in seen:
                    continue
                seen.add(key)
                piece, on = np.stack([label0, label1], axis=1), np.stack([np.ones(steps, dtype=bool), np.array(mask)], axis=1)
                best = min(best, tr._structure_fit(edge, pen, piece, on, fitted.theta)[1])
        assert fitted.objective <= best+2.0                          # never far from the best pieces
        hits += fitted.objective <= best+1e-5
    assert hits >= 7


def test_switching_reference_equals_vine_fit():
    """`switching_path` on the transforms of an edge reproduces the states and coefficients `cvine.fit` selects."""
    train = star()
    states = ['ind', 'gaussian']
    components = (('linear', (0,)), ('quadratic', (0,)))
    model = cvine.fit(train, order=[0, 1, 2], states=states, allowed={(0, 1, 2): np.ones(10, dtype=bool)}, switch=0.05)
    lower = cvine.fit(train, order=[0, 1, 2], states=states, max_order=2, max_depth=1, switch=0.05)
    u, v, features = tr.edge_inputs(lower, train, 1, 0, components)
    fits = tr.switching_candidates(u, v, features, [(0,), (1,), (0, 1)], states)
    theta, choice, names = tr.switching_path(fits, switch=0.05)
    modulated = 0
    for t, state in enumerate(model.edges[1][0]):
        expected = np.zeros(3)
        if state.components:
            modulated += 1
            expected[0] = state.beta[0]
            for component, value in zip(state.components, state.beta[1:]):
                expected[1+components.index(component)] = value
        elif state.state == 'gaussian':
            expected[0] = np.arctanh(state.par)
        assert np.abs(theta[t]-expected).max() < 1e-10, (t, names[t])
    assert modulated >= 4                                         # the path objective keeps a state for several windows


def test_trajectory_edge_in_a_vine():
    """The fitted trajectory as an edge of a vine: `cvine.forward` gives the same log density as the estimator
    (centred features are a reparameterisation), and the change test runs on the whole vine."""
    train = star(steps=8, n=200, seed=3, episode=range(2, 6))
    states = ['ind', 'gaussian']
    components = (('linear', (0,)), ('quadratic', (0,)))

    def lower_fit(windows):
        return cvine.fit(list(windows), order=[0, 1, 2], states=states, max_order=2, max_depth=1, switch=0.05)
    lower = lower_fit(train)
    edge = tr.edge_data(*tr.edge_inputs(lower, train, 1, 0, components))
    pen = tr.penalty(2, smooth=np.inf, jump=5.0)
    fitted = tr.fit(edge, pen)
    assert fitted.pieces(1) == [(2, 6)]
    model = cvine.DynamicCVine([0, 1, 2], lower.edges+[[tr.as_states(fitted, components)]])
    for t, data in enumerate(train):
        assert abs(cvine.forward(model, t, data)['logc'][:, 1].mean()-tr.window_scores(fitted.theta, edge)[t]) < 1e-9
    result = tr.change_test(train, lower_fit, components, pen, replicates=5, rng=np.random.default_rng(0))
    assert result['statistic'] > 10 and result['p'] == 1/6 and result['free'].pieces(1) == [(2, 6)]
    assert result['free'].objective <= fitted.objective+1e-6          # started from the restricted fit, same pieces found
    assert result['constant'].pieces(1) in ([], [(0, 8)])
    quiet = star(steps=8, n=200, seed=4, episode=range(0, 8), g=0.0)       # no modulation: nothing to find
    result = tr.change_test(quiet, lower_fit, components, tr.penalty(2, smooth=np.inf, jump=20.0), replicates=3)
    assert result['statistic'] < 6 and result['free'].pieces(1) in ([], [(0, 8)])


def test_identifiability_checks():
    u, v, f, _, _ = planted(steps=6, n=300, seed=9, k=2)
    smallest, inside = tr.identifiability(tr.edge_data(u, v, f))
    assert smallest.min() > 0.7 and inside.min() > 0.95
    slow = [np.column_stack((a[:, 0], np.full(len(a), float(t))+0.01*a[:, 1])) for t, a in enumerate(f)]   # nearly constant in a window
    assert tr.identifiability(tr.edge_data(u, v, slow))[1][1] < 0.01
    collinear = [np.column_stack((a[:, 0], a[:, 0]+1e-4*a[:, 1])) for a in f]
    assert tr.identifiability(tr.edge_data(u, v, collinear))[0].max() < 1e-6


def test_calibration_identity_on_exact_transforms():
    """Exact transforms need no calibration: fitted means near zero, scales near one, transforms nearly unchanged.
    A window with too few observations keeps the identity.  The Jacobian term is a density of the computed transform."""
    rng = np.random.default_rng(5)
    steps, n = 4, 2000
    zeta = [rng.normal(size=(n, 2)) for _ in range(steps)]
    u = [ndtr(rng.normal(size=n)) for _ in range(steps)]
    v = [ndtr(rng.normal(size=n)) for _ in range(steps)]
    cal = tr.fit_calibration(u, v, zeta, degree=2)
    assert cal.beta.shape == (steps, 2, 6) and np.abs(cal.beta).max() < 0.1 and np.abs(cal.scale-1).max() < 0.05
    a, b = cal.apply(0, u[0], v[0], zeta[0])
    assert np.abs(ndtri(a)-ndtri(u[0])).mean() < 0.1 and np.abs(ndtri(b)-ndtri(v[0])).mean() < 0.1   # six coefficients of sd 0.022
    assert tr.fit_calibration(u, v, zeta, degree=1).beta.shape == (steps, 2, 3)
    assert tr.fit_calibration(u, v, zeta, degree=2, products=False, columns=[1]).beta.shape == (steps, 2, 3)
    few = tr.fit_calibration([u[0][:10]], [v[0][:10]], [zeta[0][:10]], degree=2)
    assert np.all(few.beta == 0) and np.all(few.scale == 1)
    assert np.allclose(few.apply(0, u[0][:10], v[0][:10], zeta[0][:10])[0], u[0][:10], atol=1e-9)
    assert np.allclose(few.log_jacobian(0, u[0][:10], v[0][:10], zeta[0][:10]), 0.0, atol=1e-9)
    held_out_zeta = [rng.normal(size=(n, 2)) for _ in range(steps)]     # an adequate vine: the gate switches both off
    gated, gain = cal.gated([ndtr(rng.normal(size=n)) for _ in range(steps)], [ndtr(rng.normal(size=n)) for _ in range(steps)], held_out_zeta)
    assert np.all(gain < 0) and np.all(gated.beta == 0) and np.all(gated.scale == 1)
    grid = (np.arange(400000)+0.5)/400000                             # integral of the density over the computed transform
    cal.beta[0, 0], cal.scale[0, 0] = [0.2, 0.1, -0.3, 0.15, 0.05, 0.2], 0.8
    cal.beta[0, 1], cal.scale[0, 1] = 0.0, 1.0
    density = np.exp(cal.log_jacobian(0, grid, np.full(len(grid), 0.5), np.tile(np.array([[0.7, -1.2]]), (len(grid), 1))))
    assert abs(density.mean()-1.0) < 2e-3


def test_calibration_recovers_true_transform():
    """Proposition 8 in the form used: a computed transform whose normal score is the true one shifted by a quadratic
    function of the coordinates and rescaled is mapped back to the true transform; a linear calibration is not."""
    rng = np.random.default_rng(6)
    steps, n = 3, 4000
    zeta = [rng.normal(size=(n, 2)) for _ in range(steps)]
    x = [rng.normal(size=n) for _ in range(steps)]
    y = [rng.normal(size=n) for _ in range(steps)]
    shift = lambda z: 0.3*z[:, 0]*z[:, 1]+0.2*(z[:, 0]**2-1)-0.15*z[:, 1]
    u = [ndtr(1.3*(a+shift(z))) for a, z in zip(x, zeta)]
    v = [ndtr(b-0.5*shift(z)) for b, z in zip(y, zeta)]
    cal = tr.fit_calibration(u, v, zeta, degree=2)
    for t in range(steps):
        a, b = cal.apply(t, u[t], v[t], zeta[t])
        assert np.corrcoef(ndtri(a), x[t])[0, 1] > 0.998 and np.abs(ndtri(a)-x[t]).mean() < 0.06
        assert np.corrcoef(ndtri(b), y[t])[0, 1] > 0.998 and np.abs(ndtri(b)-y[t]).mean() < 0.06
    a, _ = tr.fit_calibration(u, v, zeta, degree=1).apply(0, u[0], v[0], zeta[0])
    assert np.abs(ndtri(a)-x[0]).mean() > 0.15
    fresh = [rng.normal(size=(n, 2)) for _ in range(steps)]             # held-out observations: the gate keeps both
    gated, gain = cal.gated([ndtr(1.3*(rng.normal(size=n)+shift(z))) for z in fresh], [ndtr(rng.normal(size=n)-0.5*shift(z)) for z in fresh], fresh)
    assert gain[0] > 0.1 and gain[1] > 0.005 and np.array_equal(gated.beta, cal.beta) and np.array_equal(gated.scale, cal.scale)   # expected 0.245 and 0.024
    half, gain = cal.gated([ndtr(1.3*(rng.normal(size=n)+shift(z))) for z in fresh], [ndtr(rng.normal(size=n)) for z in fresh], fresh)
    assert gain[0] > 0.02 and gain[1] < 0 and np.array_equal(half.beta[:, 0], cal.beta[:, 0]) and np.all(half.beta[:, 1] == 0)


def _four_variable_law(steps, slope=0.6):
    """Simplified and parametric below the top edge in the order (0, 1, 2, 3) only; the top edge (2, 3 | 0, 1) has
    Fisher-z correlation 0.35 + slope zeta_0."""
    constant = lambda state, value: [cvine.EdgeState(state, float(value)) for _ in range(steps)]
    first = [constant('gaussian', 0.5), constant('clayton0', 1.0), constant('gaussian', 0.45)]
    second = [constant('gaussian', 0.3), constant('gaussian', 0.25)]
    top = [[cvine.EdgeState('gaussian', components=(('linear', (0,)),), beta=np.array([0.35, slope])) for _ in range(steps)]]
    return cvine.DynamicCVine([0, 1, 2, 3], [first, second, top])


def test_calibration_reduces_order_dependence():
    """With the two conditioning variables in the other order the trees below the top edge are misspecified.  The
    fitted correlation of the edge is then closer to the truth with calibrated transforms than without, and in the
    true order the calibration costs little."""
    steps, n = 4, 1500
    rng = np.random.default_rng(11)
    truth = _four_variable_law(steps)
    raw = [cvine.simulate(truth, t, n, rng) for t in range(steps)]
    target = [np.tanh(0.35+0.6*ndtri(x[:, 0])) for x in raw]
    components = (('linear', (0,)), ('linear', (1,)))
    states = ['ind', 'gaussian', 'clayton0', 'gumbel0', 'frank']
    pen = tr.penalty(2, smooth=np.inf, order=1)                        # coefficients constant over the windows
    error = {}
    for name, order in (('same', [0, 1, 2, 3]), ('other', [1, 0, 2, 3])):
        data = [cvine.pseudo_observations(x)[:, order] for x in raw]
        lower = cvine.fit(data, order=[0, 1, 2, 3], states=states, max_order=2, max_depth=2, switch=0.08)
        plain = tr.edge_inputs(lower, data, 2, 0, components)
        calibrated, cal = tr.edge_inputs_calibrated(lower, data, 2, 0, components, degree=2)
        assert cal.beta.shape == (steps, 2, 6)
        for kind, inputs in (('plain', plain), ('calibrated', calibrated)):
            edge = tr.edge_data(*inputs)
            fitted = tr.fit(edge, pen, search=False)
            rho = np.tanh(np.einsum('tnk,tk->tn', edge.design, fitted.theta))
            error[name, kind] = float(np.sqrt(np.mean((rho-np.array(target))**2)))
    assert error['other', 'plain'] > 1.5*error['same', 'plain']                    # the order matters without calibration
    assert error['other', 'calibrated'] < 0.75*error['other', 'plain']
    assert error['same', 'calibrated'] < 1.2*error['same', 'plain']


def test_change_test_with_calibration():
    train = star(steps=8, n=200, seed=3, episode=range(2, 6))
    components = (('linear', (0,)), ('quadratic', (0,)))

    def lower_fit(windows):
        return cvine.fit(list(windows), order=[0, 1, 2], states=['ind', 'gaussian'], max_order=2, max_depth=1, switch=0.05)
    result = tr.change_test(train, lower_fit, components, tr.penalty(2, smooth=np.inf, jump=5.0), replicates=3,
                            rng=np.random.default_rng(0), calibrate=2)
    assert result['statistic'] > 10 and 0 < result['p'] <= 1 and result['free'].pieces(1) == [(2, 6)]


def test_local_change_test():
    """An episode of modulation is found by the local test; a constant modulation is not."""
    u, v, f, _, _ = planted(steps=8, n=300, seed=21, k=1, base=np.full(8, 0.3), coefficient=0.6*((np.arange(8) >= 2) & (np.arange(8) < 6)))
    pen = tr.penalty(1, smooth=np.inf, jump=5.0)
    result = tr.local_change_test(tr.edge_data(u, v, f), pen, replicates=19, rng=np.random.default_rng(1))
    assert result['statistic'] > 20 and result['p'] == 1/20 and result['free'].pieces(1) == [(2, 6)]
    u, v, f, _, _ = planted(steps=8, n=300, seed=22, k=1, base=np.full(8, 0.3), coefficient=np.full(8, 0.6))
    result = tr.local_change_test(tr.edge_data(u, v, f), pen, replicates=19, rng=np.random.default_rng(2))
    assert result['p'] > 0.1 and result['constant'].pieces(1) == [(0, 8)]


def test_change_test_with_pooled_margins():
    """Flags that pool nothing leave the test as it is, simulated datasets included.  With margins over all windows
    the data and the simulated datasets are ranked over all windows, and the episode is still found."""
    raw = star(steps=8, n=200, seed=3, episode=range(2, 6), raw=True)
    components = (('linear', (0,)), ('quadratic', (0,)))
    pen = tr.penalty(2, smooth=np.inf, jump=5.0)

    def lower_fit(windows):
        return cvine.fit(list(windows), order=[0, 1, 2], states=['ind', 'gaussian'], max_order=2, max_depth=1, switch=0.05)
    within = margins.ranks(raw)
    assert all(np.array_equal(a, b) for a, b in zip(within, star(steps=8, n=200, seed=3, episode=range(2, 6))))
    plain = tr.change_test(within, lower_fit, components, pen, replicates=3, rng=np.random.default_rng(0))
    same = tr.change_test(within, lower_fit, components, pen, replicates=3, rng=np.random.default_rng(0), pooled=[False]*3)
    assert same['statistic'] == plain['statistic'] and np.array_equal(same['null'], plain['null'])
    flags = margins.stationary(raw)
    assert flags == [True, True, True]
    pooled = tr.change_test(margins.ranks(raw, flags), lower_fit, components, pen, replicates=3, rng=np.random.default_rng(0), pooled=flags)
    assert pooled['statistic'] > 10 and pooled['free'].pieces(1) == [(2, 6)]
    assert not np.array_equal(pooled['null'], plain['null'])                     # other pseudo-observations in the draws
    kinds = ['pooled', 'aligned', 'window']                                       # every kind of margin in one test
    mixed = tr.change_test(margins.ranks(raw, kinds), lower_fit, components, pen, replicates=3, rng=np.random.default_rng(0), pooled=kinds)
    assert mixed['statistic'] > 10 and mixed['free'].pieces(1) == [(2, 6)] and 0 < mixed['p'] <= 1
    try:
        tr.change_test(within, lower_fit, components, pen, replicates=1, pooled=[True, True])
    except ValueError:
        pass
    else:
        raise AssertionError('a flag per column is required')


def test_observed_curvature_matches_finite_differences():
    """The observed Hessian of the negative log-likelihood is the derivative of its gradient."""
    u, v, f, _, _ = planted(steps=3, n=60, seed=5, k=2)
    edge = tr.edge_data(u, v, f)
    rng = np.random.default_rng(1)
    theta = rng.normal(scale=0.3, size=(3, 3))
    observed = tr._observed(theta, edge)
    step = 1e-6
    for t in range(3):
        for j in range(3):
            shifted = theta.copy()
            shifted[t, j] += step
            numeric = (tr._derivatives(shifted, edge)[0][t]-tr._derivatives(theta, edge)[0][t])/step
            assert np.allclose(numeric, observed[t, :, j], atol=1e-3*(1+np.abs(observed[t]).max()))
    expected = tr._derivatives(theta, edge)[1]
    assert np.all(np.linalg.eigvalsh(expected) > 0)                      # the expected information is always positive


def test_fisher_scoring_stops_at_a_saddle_and_polish_leaves_it():
    """Observations near the origin with a zero mean product: at correlation zero the gradient vanishes and the
    observed curvature is negative.  Fisher scoring stays there (objective 0); `stationarity` shows it, and
    `fit(polish=True)` finds a lower objective.  So the coefficient fit for fixed pieces is not exact."""
    x = np.array([[0.3, -0.3, 0.2, -0.2, 0.1, -0.1]])
    y = np.array([[0.3, 0.3, -0.2, -0.2, 0.1, 0.1]])
    edge = tr.edge_data([ndtr(x[0])], [ndtr(y[0])], [np.zeros((6, 0))])
    assert abs(float((edge.x*edge.y).sum())) < 1e-12
    pen = tr.penalty(0, smooth=np.inf, order=1)
    stuck = tr.fit(edge, pen, search=False)
    report = tr.stationarity(edge, stuck)
    assert abs(stuck.objective) < 1e-12 and report['gradient'] < 1e-10 and report['curvature'] < -1.0
    better = tr.fit(edge, pen, search=False, polish=True)
    assert better.objective < stuck.objective-1.0 and tr.stationarity(edge, better)['curvature'] > -1e-6
    u, v, f, _, _ = planted(steps=6, n=250, seed=6, k=1)                 # an ordinary fit is a local minimum and is left alone
    edge = tr.edge_data(u, v, f)
    pen = tr.penalty(1, smooth=np.inf, jump=5.0)
    plain, polished = tr.fit(edge, pen), tr.fit(edge, pen, polish=True)
    report = tr.stationarity(edge, plain)
    assert report['curvature'] > 0 and report['decrement'] < 1e-6 and report['clipped'] == 0
    assert polished.objective == plain.objective and np.array_equal(polished.theta, plain.theta)


def _two_modulator_law(steps, first, second):
    """Four variables: 0 and 1 dependent (Gaussian, 0.6), independent of the pair's margins; the pair (2, 3) has
    Fisher-z correlation first[t] * (normal score of 0) + second[t] * (coordinate of 1 given 0)."""
    star = [[cvine.EdgeState('gaussian', 0.6) for _ in range(steps)]] + [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(2)]
    level1 = [[cvine.EdgeState('ind', 0.0) for _ in range(steps)] for _ in range(2)]
    top = [[cvine.EdgeState('gaussian', components=(('linear', (0,)), ('linear', (1,))), beta=np.array([0.0, first[t], second[t]]))
            for t in range(steps)]]
    return cvine.DynamicCVine([0, 1, 2, 3], [star, level1, top])


def test_own_score_features_and_null_simulation():
    """A modulation by the own score of the second root is one own-score term and two coordinate terms.
    `simulate_null` draws from the fitted model, calibration included."""
    steps, n = 3, 4000
    rng = np.random.default_rng(8)
    r = 0.6
    # modulation 0.7 * (own score of variable 1) = 0.7 * (r zeta_0 + sqrt(1 - r^2) zeta_1)
    law = _two_modulator_law(steps, np.full(steps, 0.7*r), np.full(steps, 0.7*np.sqrt(1-r*r)))
    data = [cvine.simulate(law, t, n, rng) for t in range(steps)]

    def lower_fit(windows):
        return cvine.fit(list(windows), order=[0, 1, 2, 3], states=['ind', 'gaussian'], max_order=2, max_depth=2, switch=0.05)
    lower = lower_fit(data)
    own = (('own_linear', (1,)),)
    u, v, f = tr.edge_inputs(lower, data, 2, 0, own)
    assert np.allclose(f[0][:, 0], ndtri(data[0][:, 1]))                                   # the variable's own normal score
    pen = tr.penalty(1, smooth=np.inf, order=1)
    one = tr.fit(tr.edge_data(u, v, f), pen, search=False)
    assert np.all(np.abs(one.theta[:, 1]-0.7) < 0.06)
    coordinates = tr.fit(tr.edge_data(*tr.edge_inputs(lower, data, 2, 0, (('linear', (1,)),))), pen, search=False)
    assert one.loglik > coordinates.loglik+0.02*steps*n                                  # the coordinate of 1 alone leaves a part out
    try:
        tr.component_features(own, np.zeros((5, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError('own-score terms need the own scores')
    # simulation from the fitted model with a calibration that is not the identity
    calibration = tr.fit_calibration(u, v, tr.edge_transforms(lower, data, 2, 0)[2], degree=2)
    calibration.beta[:, 0, 1], calibration.scale[:, 0] = 0.3, 0.8                          # mean of the later variable's score: 0.3 zeta_0
    sim = tr.simulate_null(lower, one, own, calibration, 1, 20000, np.random.default_rng(9))
    assert sim.shape == (20000, 4) and 0 < sim.min() and sim.max() < 1
    a, b, zeta = tr.edge_transforms(lower, [sim, sim, sim], 2, 0)
    raw = ndtri(a[1])
    slope = np.polyfit(zeta[1][:, 0], raw, 1)[0]
    assert abs(slope-(0.3+calibration.beta[1, 0, 1]-0.3)) < 0.05 and abs(np.std(raw-slope*zeta[1][:, 0])-0.8) < 0.05
    calibrated, other = calibration.apply(1, a[1], b[1], zeta[1])
    refit = tr.fit(tr.edge_data([calibrated], [other], [ndtri(sim[:, [1]])]), tr.penalty(1, smooth=np.inf, order=1), search=False)
    assert abs(refit.theta[0, 1]-one.theta[1, 1]) < 0.05                                   # the edge of the simulated data is the fitted one


def test_change_test_from_the_calibrated_model_and_for_single_terms():
    """Two terms: the first has an episode, the second is constant.  The test of the edge and the test of the
    first term see the change; the test of the second term alone does not.  The observed statistic does not depend
    on what the null is simulated from."""
    steps, n = 8, 400
    first = np.where((np.arange(steps) >= 2) & (np.arange(steps) < 6), 0.7, 0.0)
    law = _two_modulator_law(steps, first, np.full(steps, 0.5))
    rng = np.random.default_rng(12)
    data = [cvine.pseudo_observations(cvine.simulate(law, t, n, rng)) for t in range(steps)]
    components = (('linear', (0,)), ('linear', (1,)))
    pen = tr.penalty(2, smooth=np.inf, order=1, jump=6.0)

    def lower_fit(windows):
        return cvine.fit(list(windows), order=[0, 1, 2, 3], states=['ind', 'gaussian'], max_order=2, max_depth=2, switch=0.05)
    edge = tr.change_test(data, lower_fit, components, pen, replicates=3, rng=np.random.default_rng(0), calibrate=2)
    same = tr.change_test(data, lower_fit, components, pen, replicates=3, rng=np.random.default_rng(0), calibrate=2, null='calibrated')
    assert edge['statistic'] > 20 and same['statistic'] == edge['statistic'] and edge['free'].pieces(1) == [(2, 6)]
    assert 0 < same['p'] <= 1 and same['null'].shape == (3,)
    moving = tr.change_test(data, lower_fit, components, pen, replicates=3, rng=np.random.default_rng(1), calibrate=2, null='calibrated', terms=[0])
    still = tr.change_test(data, lower_fit, components, pen, replicates=3, rng=np.random.default_rng(2), calibrate=2, null='calibrated', terms=[1])
    assert moving['statistic'] > 20 and moving['constant'].pieces(2) == [(0, steps)]
    assert still['statistic'] < 3 and still['constant'].pieces(1) == [(2, 6)]             # the first term is free under this null
    own = (('own_linear', (0,)), ('own_linear', (1,)))
    result = tr.change_test(data, lower_fit, own, pen, replicates=2, rng=np.random.default_rng(3), calibrate=2, null='calibrated')
    assert result['statistic'] > 20
    for bad in (dict(null='other'), dict(terms=[2])):
        try:
            tr.change_test(data, lower_fit, components, pen, replicates=1, **bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    try:
        tr.change_test(data, lower_fit, own, pen, replicates=1)
    except ValueError:
        pass
    else:
        raise AssertionError('own-score terms cannot be simulated by the plain vine')
