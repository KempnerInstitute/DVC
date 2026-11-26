"""Laws of the targeted validation batch.  Three laws; each answers one objection to the
challenge of E26 (`e26_laws.py`), whose generator is used unchanged where it applies.

exposed     E26's `moving` law exposed a constant term to its hazard only when the term's modulator happened to
            sit in a block whose leaves get stronger (5 of 24 terms), and the vine's own orientation was the
            planted one for 4 of those 5, so the hazard was hardly tested.  Here every planted term has its
            modulator in such a block: the dependence of every modulator on its hub rises at window 6 (Kendall tau
            + 0.15 as in E26, or + 0.30 in the strong version) while the planted coefficient stays the same.  In
            the Gaussian analogue the coefficient of a description that is not invariant is proportional to
            b = sqrt(1 - r^2), r the correlation of the modulator with its hub: a rise of tau from 0.25 to 0.40
            lowers it by 12%, from 0.35 to 0.50 by 17%, from 0.25 to 0.55 by 30% and from 0.35 to 0.65 by 39%.
            Then, for every term, the coefficient changes in the descriptions in which that dependence enters (the planted pair with the modulator's coordinate, and
            the orientations in which a leaf of the pair modulates) and stays the same in the
            planted pair with the modulator's own score.  Six constant terms and two episodes per dataset; the
            leaves of the pairs are in blocks that do not change.
continuous  E26's `serial` law starts its autoregression again in every window and draws training, validation,
            confirmation and test observations as separate series.  Here one series runs through all windows
            (innovations of every variable: a Gaussian autoregression of order one, coefficient 0.8, never
            restarted), and every window is cut in time order into training, validation, confirmation and test
            stretches with `gap` observations left out between stretches.  The law of every observation is that of
            E18 (`e26_laws`, plan `e18`, tree background).
pairwise    A law whose log density has pairwise terms only and is not Gaussian, on a dense graph with cycles:
                log p(x) = - sum_i x_i^2 / 2 + sum_{i<j} b_ij (x_i x_j + lam a(x_i) a(x_j)) + constant,
            a(x) = x cut off at -1 and 1.  With lam = 1 (`pairwise`) the pair term of two variables is twice as
            strong where both are inside (-1, 1) as where one is outside: the dependence weakens in the tails.
            With lam = 0 the law is Gaussian (`gaussian`: the control of the control).  Every variable has six
            neighbours on a ring (three on each side) and the graph has 25 more random edges.  b: random signs,
            sizes between a half and one times a common scale, the scale set so that the largest absolute
            eigenvalue of the matrix b is `radius`; (1 + lam) radius stays below one, so the quadratic form is
            positive definite everywhere.  Given the others a variable has density proportional to
            exp(-x^2 / 2 + m x + s a(x)), m = sum_j b_ij x_j, s = lam sum_j b_ij a(x_j): three truncated normal
            pieces (below -1 and above 1 around m, inside around m + s) with explicit weights.  So the sampler is
            a Gibbs sampler with exact conditional draws; n chains run side by side, each from independent starts
            for `sweeps` sweeps, and the last state of each is one observation (independent observations).
            `moving`: from window 6 the b of every edge at a variable of every second ten (10-19, 30-39) is
            multiplied by 1.5 (pairwise terms change; nothing else).  No term of the log density has more than
            two variables.  In the sense of this project (the dependence of a pair given other variables varies
            with them) such a law can still have third-order structure: a third variable shifts a variable of
            the pair towards or away from the region where the pair's term is strong.
            Two earlier versions were dropped before the run because on a development seed the fitted vine
            found almost no dependence in them, so they tested nothing: terms - c x_i^2 x_j^2 / 2 (held-out log
            density of the simplified vine -0.10 to 0.02 nats per observation at n = 300) and the saturating
            term alone, without x_i x_j (0.68 at n = 1000; the Gaussian law with the same b: 6.35).
"""
import numpy as np
from scipy.special import ndtr, ndtri

import e26_laws as laws

T, BLOCK = laws.T, laws.BLOCK


def exposed_design(d, seed, g, constants=6, episodes=2, rise=0.15):
    """Background of E26's `moving` law; every modulator in a block whose leaves get stronger, every pair in
    blocks that do not change."""
    spec = laws.design(d, seed, 0.0, plan='e18', background='tree', moving=True)
    rng = np.random.default_rng([seed, d, 291])
    rising = set(spec['rising'])
    moving = [v for v in range(d) if v % BLOCK and v//BLOCK in rising]
    still = [v for v in range(d) if v % BLOCK and v//BLOCK not in rising]
    count = constants+episodes if g > 0 else 0
    modulators = [int(v) for v in rng.choice(moving, count, replace=False)]
    pairs = [int(v) for v in rng.choice(still, 2*count, replace=False)]
    planted = []
    for index in range(count):
        i, j = pairs[2*index], pairs[2*index+1]
        kind, shape = ('constant_leaf', 'constant') if index < constants else ('episode_leaf', 'a')
        planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(modulators[index],), shape=shape)]))
    spec['planted'] = planted
    spec['rise'] = float(rise)
    return spec


def autoregression(length, d, coefficient, rng):
    """Uniform innovations (length, d) whose normal scores follow one stationary Gaussian autoregression of order
    one per column."""
    e = rng.normal(size=(length, d))
    x = np.empty((length, d))
    x[0] = e[0]
    scale = np.sqrt(1.0-coefficient*coefficient)
    for i in range(1, length):
        x[i] = coefficient*x[i-1]+scale*e[i]
    return ndtr(x)


def continuous_sample(spec, sizes, g, coefficient, rng, gap=25):
    """One series through all windows.  sizes: {role: observations per window}, the roles in their order in time
    inside a window.  Returns {role: [window arrays]}."""
    d = spec['d']
    per_window = sum(sizes.values())+gap*len(sizes)
    series = autoregression(T*per_window, d, coefficient, rng)
    position = [0]

    def feed(n, d_, serial, rng_):
        out = series[position[0]:position[0]+n]
        position[0] += n+gap
        return out.copy()
    original = laws.innovations
    laws.innovations = feed                                         # `e26_laws.sample` takes its innovations from the series
    try:
        raw = {role: [] for role in sizes}
        for t in range(T):
            for role, n in sizes.items():
                raw[role].append(laws.sample(spec, n, t, g, rng, serial=coefficient))
    finally:
        laws.innovations = original
    assert position[0] == T*per_window
    return raw


def pairwise_design(d, seed, radius=0.45, extra=25, saturating=True):
    rng = np.random.default_rng([seed, d, 292])
    edges = {tuple(sorted((i, (i+step) % d))) for i in range(d) for step in (1, 2, 3)}
    while len(edges) < 3*d+extra:
        i, j = sorted(int(v) for v in rng.choice(d, 2, replace=False))
        edges.add((i, j))
    edges = sorted(edges)
    b = np.zeros((d, d))
    for i, j in edges:
        b[i, j] = b[j, i] = rng.choice((-1.0, 1.0))*rng.uniform(0.5, 1.0)
    b *= radius/np.abs(np.linalg.eigvalsh(b)).max()
    rising = [v for v in range(d) if (v//10) % 2 == 1]
    return dict(d=d, edges=edges, b=b, lam=1.0 if saturating else 0.0, rising=rising, planted=[], hubs=[], margin_kind=['fixed']*d)


def pairwise_matrix(spec, t, moving=False):
    """The matrix b of window t."""
    b = spec['b']
    if moving and t >= laws.BLOCK_CHANGE:
        scale = np.ones(b.shape)
        scale[spec['rising'], :] = 1.5
        scale[:, spec['rising']] = 1.5
        b = b*scale
    if (1.0+spec['lam'])*np.abs(np.linalg.eigvalsh(b)).max() >= 1.0:
        raise ValueError('the quadratic form of the pairwise terms must stay positive definite')
    return b


def pairwise_sample(spec, n, t, rng, moving=False, sweeps=150):
    """n independent observations of window t."""
    d, lam, b = spec['d'], spec['lam'], pairwise_matrix(spec, t, moving)
    x = rng.normal(size=(n, d))
    if lam == 0.0:
        for _ in range(sweeps):
            for i in range(d):
                x[:, i] = x@b[i]+rng.normal(size=n)
        return x
    a = np.clip(x, -1.0, 1.0)
    for _ in range(sweeps):
        for i in range(d):
            m, s = x@b[i], lam*(a@b[i])
            centre = m+s
            low, high = ndtr(-1.0-centre), ndtr(1.0-centre)
            left, right = ndtr(-1.0-m), ndtr(m-1.0)
            below, inside, above = np.exp(0.5*m*m-s)*left, np.exp(0.5*centre*centre)*(high-low), np.exp(0.5*m*m+s)*right
            u = rng.uniform(size=n)*(below+inside+above)
            v = np.clip(rng.uniform(size=n), 1e-12, 1.0-1e-12)
            value = np.where(u < below, m+ndtri(np.maximum(v*left, 1e-300)),
                             np.where(u < below+inside, centre+ndtri(low+v*(high-low)), m-ndtri(np.maximum(v*right, 1e-300))))
            x[:, i] = value
            a[:, i] = np.clip(value, -1.0, 1.0)
    return x
