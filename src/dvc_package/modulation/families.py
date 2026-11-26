"""Bivariate copula families in float64: log density, conditional distribution, batched fitting and sampling.

Conventions
-----------
A *state* is a family with a rotation, for example ``clayton180``.  Every function is vectorised: ``u`` and ``v`` are
arrays of pseudo-observations in (0, 1) and the parameters broadcast against them, so one call evaluates many pairs
(columns) and, for modulated copulas, a different parameter for every observation (rows).

``evaluate(state, u, v, par, nu)`` returns ``(log c(u, v), h)`` with ``h = P(U <= u | V = v)``: the first argument is
the variable and the second the conditioning variable.  ``h_reverse`` gives ``P(V <= v | U = u)``.

Rotations of an exchangeable base density c0:
    0:   c(u, v) = c0(u, v)
    90:  c(u, v) = c0(1 - u, v)
    180: c(u, v) = c0(1 - u, 1 - v)
    270: c(u, v) = c0(u, 1 - v)
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, ndtr, ndtri, stdtr, stdtrit

EPS = 1e-10
NU_GRID = (2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 12.0, 20.0, 40.0)
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def clip01(u, eps=EPS):
    return np.clip(u, eps, 1.0 - eps)


# ----------------------------------------------------------------------------------------------------------------
# Base families: (log density, P(U <= u | V = v)).
# ----------------------------------------------------------------------------------------------------------------
def _independence(u, v, par=None, nu=None):
    shape = np.broadcast(u, v).shape
    return np.zeros(shape), np.broadcast_to(u, shape).copy()


def _gaussian(u, v, rho, nu=None):
    x, y = ndtri(u), ndtri(v)
    s = 1.0 - rho * rho
    logc = -0.5 * np.log(s) - (rho * rho * (x * x + y * y) - 2.0 * rho * x * y) / (2.0 * s)
    return logc, ndtr((x - rho * y) / np.sqrt(s))


def _student(u, v, rho, nu):
    x, y = stdtrit(nu, u), stdtrit(nu, v)
    s = 1.0 - rho * rho
    logc = (gammaln((nu + 2.0) / 2.0) + gammaln(nu / 2.0) - 2.0 * gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(s)
            - (nu + 2.0) / 2.0 * np.log1p((x * x + y * y - 2.0 * rho * x * y) / (nu * s))
            + (nu + 1.0) / 2.0 * (np.log1p(x * x / nu) + np.log1p(y * y / nu)))
    h = stdtr(nu + 1.0, (x - rho * y) / np.sqrt((nu + y * y) * s / (nu + 1.0)))
    return logc, h


def _clayton(u, v, th, nu=None):
    lu, lv = np.log(u), np.log(v)
    a, b = -th * lu, -th * lv                                  # logs of u^-th and v^-th, both >= 0
    m = np.maximum(a, b)
    logs = m + np.log(np.exp(a - m) + np.exp(b - m) - np.exp(-m))   # log(u^-th + v^-th - 1)
    logc = np.log1p(th) - (th + 1.0) * (lu + lv) - (2.0 + 1.0 / th) * logs
    h = np.exp(-(th + 1.0) * lv - (1.0 / th + 1.0) * logs)
    return logc, h


def _gumbel(u, v, th, nu=None):
    x, y = -np.log(u), -np.log(v)
    lx, ly = np.log(x), np.log(y)
    logs = np.logaddexp(th * lx, th * ly)                      # log(x^th + y^th)
    s1 = np.exp(logs / th)
    logc = -s1 + (th - 1.0) * (lx + ly) + x + y + (2.0 / th - 2.0) * logs + np.log1p((th - 1.0) * np.exp(-logs / th))
    h = np.exp(-s1 + (1.0 / th - 1.0) * logs + (th - 1.0) * ly + y)
    return logc, h


def _frank(u, v, th, nu=None):
    th = np.where(np.abs(th) < 1e-4, np.where(th < 0, -1e-4, 1e-4), th)
    e, eu, ev = np.expm1(-th), np.expm1(-th * u), np.expm1(-th * v)
    den = e + eu * ev
    logc = np.log(-th * e) - th * (u + v) - 2.0 * np.log(np.abs(den))
    h = eu * np.exp(-th * v) / den
    return logc, h


def _joe(u, v, th, nu=None):
    l1u, l1v = np.log1p(-u), np.log1p(-v)
    a, b = np.exp(th * l1u), np.exp(th * l1v)
    s = a + b - a * b
    logs = np.log(s)
    logc = (1.0 / th - 2.0) * logs + (th - 1.0) * (l1u + l1v) + np.log(th - 1.0 + s)
    h = np.exp((1.0 / th - 1.0) * logs + (th - 1.0) * l1v) * (1.0 - a)
    return logc, h


BASE = {'ind': _independence, 'gaussian': _gaussian, 'student': _student, 'clayton': _clayton, 'gumbel': _gumbel,
        'frank': _frank, 'joe': _joe}
N_PARAMETERS = {'ind': 0, 'gaussian': 1, 'student': 2, 'clayton': 1, 'gumbel': 1, 'frank': 1, 'joe': 1}
ROTATED = ('clayton', 'gumbel', 'joe')
SWAP = {0: 0, 180: 180, 90: 270, 270: 90}


def split_state(state: str):
    """'clayton180' -> ('clayton', 180); families without rotation return rotation 0."""
    for family in ROTATED:
        if state.startswith(family):
            tail = state[len(family):]
            return family, int(tail) if tail else 0
    if state not in BASE:
        raise ValueError(f'unknown copula state {state!r}')
    return state, 0


def n_parameters(state: str) -> int:
    return N_PARAMETERS[split_state(state)[0]]


def evaluate(state, u, v, par=None, nu=None, eps=EPS):
    """Log density and P(U <= u | V = v) of the copula state at (u, v)."""
    family, rotation = split_state(state)
    u, v = clip01(np.asarray(u, dtype=np.float64), eps), clip01(np.asarray(v, dtype=np.float64), eps)
    base = BASE[family]
    if rotation == 0:
        logc, h = base(u, v, par, nu)
    elif rotation == 180:
        logc, h = base(1.0 - u, 1.0 - v, par, nu)
        h = 1.0 - h
    elif rotation == 90:
        logc, h = base(1.0 - u, v, par, nu)
        h = 1.0 - h
    elif rotation == 270:
        logc, h = base(u, 1.0 - v, par, nu)
    else:
        raise ValueError(f'unknown rotation {rotation}')
    return logc, clip01(h, eps)


def swap_state(state: str) -> str:
    """State of the copula of (V, U) when (U, V) has ``state``."""
    family, rotation = split_state(state)
    return f'{family}{SWAP[rotation]}' if family in ROTATED else family


def h_reverse(state, u, v, par=None, nu=None, eps=EPS):
    """P(V <= v | U = u) for the copula state of (U, V)."""
    return evaluate(swap_state(state), v, u, par, nu, eps)[1]


# ----------------------------------------------------------------------------------------------------------------
# Parameter domains and maps from the optimisation variable.
# ----------------------------------------------------------------------------------------------------------------
BOUNDS = {'gaussian': (-0.99, 0.99), 'student': (-0.99, 0.99), 'clayton': (np.log(0.01), np.log(40.0)),
          'gumbel': (np.log(0.005), np.log(24.0)), 'joe': (np.log(0.005), np.log(29.0)), 'frank': (1e-3, 45.0)}


def _natural(family, p, sign=1.0):
    if family in ('gaussian', 'student'):
        return p
    if family == 'clayton':
        return np.exp(p)
    if family in ('gumbel', 'joe'):
        return 1.0 + np.exp(p)
    if family == 'frank':
        return sign * p
    raise ValueError(family)


def golden_maximise(function, low, high, iterations=40):
    """Maximise a vectorised unimodal function of one variable on [low, high], one problem per array entry."""
    a, b = np.array(low, dtype=np.float64), np.array(high, dtype=np.float64)
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = function(c), function(d)
    for _ in range(iterations):
        left = fc > fd
        b = np.where(left, d, b)
        a = np.where(left, a, c)
        new_c, new_d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
        probe = np.where(left, new_c, new_d)
        f_probe = function(probe)
        c, d, fc, fd = (np.where(left, new_c, d), np.where(left, c, new_d),
                        np.where(left, f_probe, fd), np.where(left, fc, f_probe))
    best = 0.5 * (a + b)
    return best, function(best)


def default_states(families=('gaussian', 'student', 'clayton', 'gumbel', 'frank'), rotations=True):
    """Ordered state list: independence first, then each family (with its four rotations when ``rotations``)."""
    states = ['ind']
    for family in families:
        if family == 'ind':
            continue
        if family in ROTATED:
            states.extend(f'{family}{r}' for r in ((0, 180, 90, 270) if rotations else (0,)))
        else:
            states.append(family)
    return states


def fit_states(u, v, states, nu_grid=NU_GRID, eps=EPS, iterations=40):
    """Maximum-likelihood fit of every state to every pair (column).

    u, v: arrays (n, P).  Returns ``par`` (S, P), ``nu`` (S, P) and ``loglik`` (S, P), the mean log-likelihood per
    observation.  States that cannot represent the sign of the dependence (rotations of the opposite sign) get
    ``loglik = -inf``.  The sign is taken from the correlation of normal scores.
    """
    u, v = clip01(np.asarray(u, dtype=np.float64), eps), clip01(np.asarray(v, dtype=np.float64), eps)
    if u.ndim == 1:
        u, v = u[:, None], v[:, None]
    n, pairs = u.shape
    x, y = ndtri(u), ndtri(v)
    sxx, syy, sxy = (x * x).mean(0), (y * y).mean(0), (x * y).mean(0)
    positive = sxy >= 0.0
    par = np.full((len(states), pairs), np.nan)
    nus = np.full((len(states), pairs), np.nan)
    loglik = np.full((len(states), pairs), -np.inf)
    for index, state in enumerate(states):
        family, rotation = split_state(state)
        if family == 'ind':
            loglik[index] = 0.0
            continue
        low, high = BOUNDS[family]
        if family == 'gaussian':
            def objective(p):
                s = 1.0 - p * p
                return -0.5 * np.log(s) - (p * p * (sxx + syy) - 2.0 * p * sxy) / (2.0 * s)
            best, value = golden_maximise(objective, np.full(pairs, low), np.full(pairs, high), iterations)
            par[index], loglik[index] = best, value
        elif family == 'student':
            best_value = np.full(pairs, -np.inf)
            for nu in nu_grid:
                tx, ty = stdtrit(nu, u), stdtrit(nu, v)
                txx, tyy = tx * tx, ty * ty
                const = (gammaln((nu + 2.0) / 2.0) + gammaln(nu / 2.0) - 2.0 * gammaln((nu + 1.0) / 2.0)
                         + (nu + 1.0) / 2.0 * (np.log1p(txx / nu) + np.log1p(tyy / nu)).mean(0))
                sq, cross = txx + tyy, tx * ty

                def objective(p, sq=sq, cross=cross, const=const, nu=nu):
                    s = 1.0 - p * p
                    return const - 0.5 * np.log(s) - (nu + 2.0) / 2.0 * np.log1p((sq - 2.0 * p * cross) / (nu * s)).mean(0)
                best, value = golden_maximise(objective, np.full(pairs, low), np.full(pairs, high), iterations)
                better = value > best_value
                best_value = np.where(better, value, best_value)
                par[index] = np.where(better, best, par[index])
                nus[index] = np.where(better, nu, nus[index])
            loglik[index] = best_value
        elif family == 'frank':
            sign = np.where(positive, 1.0, -1.0)

            def objective(p):
                return _frank(u, v, sign * p)[0].mean(0)
            best, value = golden_maximise(objective, np.full(pairs, low), np.full(pairs, high), iterations)
            par[index], loglik[index] = sign * best, value
        else:
            allowed = positive if rotation in (0, 180) else ~positive
            if not allowed.any():
                continue
            base = BASE[family]
            uu = 1.0 - u[:, allowed] if rotation in (90, 180) else u[:, allowed]
            vv = 1.0 - v[:, allowed] if rotation in (180, 270) else v[:, allowed]

            def objective(p, uu=uu, vv=vv, base=base, family=family):
                return base(uu, vv, _natural(family, p))[0].mean(0)
            count = int(allowed.sum())
            best, value = golden_maximise(objective, np.full(count, low), np.full(count, high), iterations)
            par[index, allowed], loglik[index, allowed] = _natural(family, best), value
    return par, nus, loglik


# ----------------------------------------------------------------------------------------------------------------
# Sampling by inversion of the conditional distribution (bisection), for tests and synthetic laws.
# ----------------------------------------------------------------------------------------------------------------
def h_inverse(state, q, v, par=None, nu=None, iterations=60):
    """u with P(U <= u | V = v) = q."""
    q, v = np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)
    family, _ = split_state(state)
    if family == 'ind':
        return np.broadcast_to(q, np.broadcast(q, v).shape).copy()
    low, high = np.zeros(np.broadcast(q, v).shape), np.ones(np.broadcast(q, v).shape)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        below = evaluate(state, mid, v, par, nu, eps=1e-15)[1] < q
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    return 0.5 * (low + high)


def sample_pair(state, n, rng, par=None, nu=None):
    """n draws (u, v) from the copula state."""
    v, q = rng.uniform(size=n), rng.uniform(size=n)
    return h_inverse(state, q, v, par, nu), v
