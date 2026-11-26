"""Switching state paths for pair copulas over time windows (the per-edge objective of DVC-switch, batched).

For every edge the path of states minimises

    sum_t [ AIC_t(state_t) / (2 n_t) + activation * 1{state_t != independence} ]
    + sum_t [ switch * 1{state_t != state_{t-1}} + drift * distance(theta_t, theta_{t-1}) * 1{state_t == state_{t-1}} ]

which is the objective of ``dvc_package.time.switching_dynamic_cvine`` with ``normalize_cost=True``.  All edges
(columns) are solved together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from . import families as fam


@dataclass
class EdgePaths:
    """Selected state and parameters of P edges in T windows."""
    states: List[str]
    choice: np.ndarray          # (T, P) index into states
    par: np.ndarray             # (T, P) first parameter of the selected state (nan for independence)
    nu: np.ndarray              # (T, P) degrees of freedom when the state is Student-t
    loglik: np.ndarray          # (T, P) mean training log-likelihood per observation of the selected state
    objective: np.ndarray       # (P,) value of the path objective

    def names(self, t: int) -> List[str]:
        return [self.states[s] for s in self.choice[t]]


def parameter_distance(par, nu=None):
    """Mean absolute change of a state's parameters between consecutive windows: (T, S, P), zero in window 0."""
    delta = np.abs(np.diff(par, axis=0))
    count = np.isfinite(delta).astype(float)
    total = np.where(np.isfinite(delta), delta, 0.0)
    if nu is not None:
        delta_nu = np.abs(np.diff(nu, axis=0))
        count = count + np.isfinite(delta_nu)
        total = total + np.where(np.isfinite(delta_nu), delta_nu, 0.0)
    return np.concatenate((np.zeros((1,) + par.shape[1:]), total / np.maximum(count, 1.0)))


def viterbi(local, distance=None, switch=0.0, drift=0.0):
    """Minimum-cost state paths.  local: (T, S, P) costs (inf = unavailable); distance: (T, S, P) parameter change
    of each state from the previous window.  Returns choice (T, P) and the objective (P,)."""
    local = np.asarray(local, dtype=np.float64)
    steps, n_states, n_edges = local.shape
    index = np.arange(n_states)[:, None]
    dp = local[0].copy()
    back = np.zeros((steps, n_states, n_edges), dtype=np.int32)
    for t in range(1, steps):
        stay = dp + drift * distance[t] if (drift > 0.0 and distance is not None) else dp
        if n_states > 1:
            order = np.argsort(dp, axis=0, kind='stable')[:2]
            best = np.take_along_axis(dp, order[:1], axis=0)[0]
            second = np.take_along_axis(dp, order[1:2], axis=0)[0]
            is_best = order[0][None, :] == index
            other_cost = np.where(is_best, second[None, :], best[None, :]) + switch
            other_index = np.where(is_best, order[1][None, :], order[0][None, :])
            use_stay = stay <= other_cost
            back[t] = np.where(use_stay, index, other_index)
            dp = local[t] + np.where(use_stay, stay, other_cost)
        else:
            dp = local[t] + stay
    choice = np.zeros((steps, n_edges), dtype=np.int32)
    choice[-1] = np.argmin(dp, axis=0)
    objective = np.take_along_axis(dp, choice[-1][None, :], axis=0)[0]
    columns = np.arange(n_edges)
    for t in range(steps - 1, 0, -1):
        choice[t - 1] = back[t, choice[t], columns]
    return choice, objective


def local_costs(loglik, counts, states, activation=0.0, extra_parameters=None):
    """AIC per observation: loglik (T, S, P) mean log-likelihood, counts (T,) observations per window.
    ``activation``: a number, or one per edge (P,).  `viterbi` takes ``switch`` in the same two forms."""
    k = np.array([fam.n_parameters(s) if extra_parameters is None else extra_parameters[i]
                  for i, s in enumerate(states)], dtype=np.float64)
    cost = k[None, :, None] / np.asarray(counts, dtype=np.float64)[:, None, None] - loglik
    if np.any(np.asarray(activation) > 0.0):                          # one value, or one per edge (the last axis)
        active = np.array([s != 'ind' for s in states])
        cost = cost + activation * active[None, :, None]
    return cost


def fit_edge_paths(u_by_window: Sequence[np.ndarray], v_by_window: Sequence[np.ndarray], states: Sequence[str],
                   switch=0.05, drift=0.0, activation=0.0, nu_grid=fam.NU_GRID, eps=fam.EPS) -> EdgePaths:
    """Fit every state to every edge in every window and select the switching paths.

    u_by_window[t], v_by_window[t]: arrays (n_t, P) of the pair's two coordinates.
    """
    states = list(states)
    fits = [fam.fit_states(u, v, states, nu_grid=nu_grid, eps=eps) for u, v in zip(u_by_window, v_by_window)]
    par = np.stack([f[0] for f in fits])
    nu = np.stack([f[1] for f in fits])
    loglik = np.stack([f[2] for f in fits])
    counts = [np.asarray(u).shape[0] for u in u_by_window]
    cost = local_costs(loglik, counts, states, activation)
    choice, objective = viterbi(cost, parameter_distance(par, nu), switch=switch, drift=drift)
    take = lambda a: np.take_along_axis(a, choice[:, None, :], axis=1)[:, 0, :]
    return EdgePaths(states, choice, take(par), take(nu), take(loglik), objective)


def apply_states(states, choice, par, nu, u, v, reverse=False, eps=fam.EPS):
    """Evaluate each column's selected state.  choice, par, nu: (P,); u, v: (n, P).

    Returns (log density (n, P), h (n, P)) with h = P(U <= u | V = v), or P(V <= v | U = u) when ``reverse``.
    """
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    logc, h = np.zeros(u.shape), np.empty(u.shape)
    for s in np.unique(choice):
        columns = choice == s
        state = states[s]
        if reverse:
            lc, hh = fam.evaluate(fam.swap_state(state), v[:, columns], u[:, columns], par[columns], nu[columns], eps)
        else:
            lc, hh = fam.evaluate(state, u[:, columns], v[:, columns], par[columns], nu[columns], eps)
        logc[:, columns], h[:, columns] = lc, hh
    return logc, h
