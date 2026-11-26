"""Laws of the challenge on fresh data.  One generator with options; every law has blocks of
ten variables (a hub and nine leaves), 12 windows and planted modulations of the dependence of two leaves.

Planted modulations.  As in E18 the parts of two leaves i, j that the background does not explain (their uniform
innovations) are coupled by a Gaussian copula whose Fisher-z correlation is a sum of terms c(t) f(z): f is the
normal score of one modulator (third order) or the product of the normal scores of two (fourth order), and c(t)
is an episode, a constant or a linear rise.  A planted item has one pair and one or more terms.

Backgrounds.
  tree    E18: every block a star around its hub (Clayton, Gumbel, Student-t, Gaussian, Frank by block; Kendall tau
          0.25 or 0.35 per leaf), the hubs a Gaussian chain.  A simplified vine whose dependence graph is a tree.
  dense   not a tree.  The hubs form a D-vine with two trees and non-Gaussian pairs; a leaf depends on its own
          hub, on the next hub given its own (Kendall tau 0.2) and on the first leaf of its block given both
          (Kendall tau 0.2).  A simplified regular vine with three trees: no conditional copula of it varies, its
          dependence graph has cycles everywhere, and it is not a simplified C-vine in any root order.  The first
          leaf of a block carries no planted modulation.

Changes of the pairwise background over time (never an interaction):
  E18's     the chain of hubs gets stronger at window 3 and the leaves of block 1 at window 6 (planted variables
            avoid block 1);
  moving    the chain gets stronger at window 3, and the leaves of every second block (1, 3, ...) get stronger at
            window 6; planted variables do not avoid these blocks, so the pairwise dependence of planted variables
            on their hubs changes while their modulation need not.

Other options of `sample`:
  serial    the innovations of every variable follow a Gaussian autoregression of order one inside a window;
  margins   'moving': a third of the variables drift in location and scale, a third change the shape of their
            margin at window 6 (normal to log-normal), a third keep a normal margin.  Otherwise uniform margins.

Plans (planted items by kind):
  e18       3 leaf episodes, 2 hub episodes, 2 constant, 1 rise                    (the kinds of E18)
  constant  4 constant, 2 leaf episodes, 1 rise, 1 hub episode                    (for the moving background)
  overlap   two triples on one pair with episodes at different times; two pairs with one modulator; a triple with
            a fourth-order term on the same pair; a fourth-order term alone; two triples that share a leaf
  fourth    4 fourth-order episodes and 4 third-order ones (2 episodes, 1 constant, 1 rise)
"""
from functools import lru_cache

import numpy as np
from scipy.special import ndtr, ndtri

from common import fam
import e4_large

T, BLOCK = 12, 10
EPISODES = dict(a=range(4, 8), b=range(6, 10), early=range(2, 5), late=range(7, 10))
CHAIN_CHANGE, BLOCK_CHANGE = 3, 6
HUB_FAMILIES = ('clayton0', 'gumbel0', 'frank', 'student')
PLANS = dict(
    e18=(('episode_leaf', 3), ('episode_hub', 2), ('constant_leaf', 2), ('ramp_leaf', 1)),
    constant=(('constant_leaf', 4), ('episode_leaf', 2), ('ramp_leaf', 1), ('episode_hub', 1)),
    overlap=(('shared_pair', 1), ('shared_modulator', 1), ('nested_fourth', 1), ('fourth', 1), ('shared_leaf', 1)),
    fourth=(('fourth', 4), ('episode_leaf', 2), ('constant_leaf', 1), ('ramp_leaf', 1)))


@lru_cache(maxsize=None)
def frank_parameter(tau):
    """Parameter of the Frank copula with Kendall's tau (tau = 1 - 4 (1 - D(theta)) / theta, D the first Debye
    function), by bisection."""
    integral = getattr(np, 'trapezoid', None) or np.trapz

    def kendall(theta):
        grid = np.linspace(1e-9, theta, 4001)
        debye = integral(grid/np.expm1(grid), grid)/theta
        return 1.0-4.0*(1.0-debye)/theta
    low, high = 1e-3, 40.0
    for _ in range(60):
        middle = 0.5*(low+high)
        low, high = (middle, high) if kendall(middle) < tau else (low, middle)
    return 0.5*(low+high)


def parameter(state, tau):
    return frank_parameter(tau) if state == 'frank' else e4_large.tau_parameter(state, tau)


def course(shape, g, t):
    """Coefficient of a planted term in window t.  shape: 'constant', 'ramp' or the name of an episode."""
    if shape == 'constant':
        return g
    if shape == 'ramp':
        return g*t/(T-1.0)
    return g if t in EPISODES[shape] else 0.0


def design(d, seed, g, plan='e18', background='tree', moving=False, families=e4_large.FAMILIES):
    """Background and planted items.  planted: list of dict(kind, pair=(i, j), terms=[dict(modulators, shape)]);
    the subset of a term is the pair with its modulators."""
    rng = np.random.default_rng([seed, d, 261])
    blocks = d//BLOCK
    hubs = [b*BLOCK for b in range(blocks)]
    edges = {}
    for b in range(blocks):
        for leaf in range(b*BLOCK+1, (b+1)*BLOCK):
            edges[leaf] = (families[b % len(families)], float(rng.choice((0.25, 0.35))))
    rising = [b for b in range(blocks) if b % 2 == 1] if moving else [1]
    avoid = set(hubs) | (set() if moving else set(range(BLOCK, 2*BLOCK)))
    if background == 'dense':
        avoid |= {b*BLOCK+1 for b in range(blocks)}                 # every other leaf of the block is drawn after it
    used = set(avoid)
    leaves = lambda count, distinct_blocks=True: _leaves(rng, d, count, used, distinct_blocks)
    planted = []
    for kind, count in PLANS[plan]:
        for _ in range(count if g > 0 else 0):
            if kind == 'episode_hub':
                while True:
                    block = int(rng.choice([b for b in range(blocks) if b not in rising or moving]))
                    free = [v for v in range(block*BLOCK+1, (block+1)*BLOCK) if v not in used]
                    if len(free) >= 2 and not any(block*BLOCK in term['modulators'] for item in planted for term in item['terms']):
                        break
                i, j = sorted(int(v) for v in rng.choice(free, 2, replace=False))     # the earlier leaf of the block first
                used |= {i, j}
                planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(block*BLOCK,), shape='b')]))
            elif kind in ('episode_leaf', 'constant_leaf', 'ramp_leaf'):
                i, j, k = leaves(3)
                shape = dict(episode_leaf='a', constant_leaf='constant', ramp_leaf='ramp')[kind]
                planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(k,), shape=shape)]))
            elif kind == 'shared_pair':                          # two modulators of one pair, at different times
                i, j, k, m = leaves(4)
                planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(k,), shape='early'), dict(modulators=(m,), shape='late')]))
            elif kind == 'shared_modulator':                     # one modulator of two pairs, at different times
                i, j, k, a, b = leaves(5)
                planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(k,), shape='early')]))
                planted.append(dict(kind=kind, pair=(a, b), terms=[dict(modulators=(k,), shape='late')]))
            elif kind == 'nested_fourth':                        # a constant triple and a fourth-order episode on its pair
                i, j, k, m = leaves(4)
                planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(k,), shape='constant'), dict(modulators=(k, m), shape='a')]))
            elif kind == 'fourth':
                i, j, k, m = leaves(4)
                planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(k, m), shape='a')]))
            elif kind == 'shared_leaf':                          # the leaf j belongs to two modulated pairs
                i, j, k, a, m = leaves(5)
                planted.append(dict(kind=kind, pair=(i, j), terms=[dict(modulators=(k,), shape='early')]))
                planted.append(dict(kind=kind, pair=(j, a), terms=[dict(modulators=(m,), shape='late')]))
            else:
                raise ValueError(kind)
    spec = dict(d=d, hubs=hubs, edges=edges, planted=planted, background=background, moving=bool(moving), rising=rising)
    if background == 'dense':
        spec['hub_pairs'] = [HUB_FAMILIES[m % len(HUB_FAMILIES)] for m in range(blocks-1)]
        spec['second'] = {leaf: (('clayton0', 'gumbel0', 'frank')[leaf % 3], 0.2) for leaf in edges}       # with the next hub given the own
        spec['third'] = {leaf: (('gaussian', 'frank')[leaf % 2], 0.2) for leaf in edges if leaf % BLOCK >= 2}  # with the first leaf of the block
    elif background != 'tree':
        raise ValueError(background)
    rng_margin = np.random.default_rng([seed, d, 262])
    spec['margin_kind'] = [('drift', 'shape', 'fixed')[int(v)] for v in rng_margin.integers(0, 3, size=d)]
    spec['margin_sign'] = rng_margin.choice((-1.0, 1.0), size=d).tolist()
    return spec


def _leaves(rng, d, count, used, distinct_blocks=True):
    """``count`` unused leaves, of as many different blocks as there are (blocks with unused leaves); marks them
    as used."""
    free = [v for v in range(d) if v not in used]
    if len(free) < count:
        raise ValueError('not enough unused leaves for the planted items')
    needed = min(count, len({v//BLOCK for v in free})) if distinct_blocks else 1
    for _ in range(100000):
        chosen = [int(v) for v in rng.choice(free, count, replace=False)]
        if len({v//BLOCK for v in chosen}) >= needed:
            used.update(chosen)
            return chosen
    raise ValueError('no choice of leaves of different blocks was found')


def subsets(spec, g):
    """Every planted term: dict(kind, subset, order, pair, modulators, windows) with the windows in which its
    coefficient is not zero."""
    out = []
    for item in spec['planted']:
        for term in item['terms']:
            out.append(dict(kind=item['kind'], pair=tuple(item['pair']), modulators=tuple(term['modulators']), shape=term['shape'],
                            subset=tuple(sorted(set(item['pair']) | set(term['modulators']))), order=2+len(term['modulators']),
                            windows=[t for t in range(T) if course(term['shape'], g, t) != 0.0]))
    return out


def innovations(n, d, serial, rng):
    """Uniform innovations (n, d); with ``serial`` > 0 the normal scores of every column follow a stationary
    Gaussian autoregression of order one with that coefficient."""
    if not serial:
        return rng.uniform(size=(n, d))
    e = rng.normal(size=(n, d))
    x = np.empty((n, d))
    x[0] = e[0]
    scale = np.sqrt(1.0-serial*serial)
    for i in range(1, n):
        x[i] = serial*x[i-1]+scale*e[i]
    return ndtr(x)


def sample(spec, n, t, g, rng, static=False, serial=0.0, margins=None):
    """n observations of window t.  ``static``: no change of the pairwise background.  Returns values with uniform
    margins, or with ``margins='moving'`` on the scales described in the header."""
    d, hubs = spec['d'], spec['hubs']
    dense = spec['background'] == 'dense'
    u = np.empty((n, d))
    q = innovations(n, d, serial, rng)
    strong = (not static) and t >= CHAIN_CHANGE
    chain = 0.6 if strong else 0.4
    cond = {}                                                       # F(next hub | own hub) of every block, for the dense background
    for m, hub in enumerate(hubs):
        if m == 0:
            u[:, hub] = q[:, hub]
            continue
        before = hubs[m-1]
        if not dense:
            u[:, hub] = fam.h_inverse('gaussian', q[:, hub], u[:, before], chain)
            continue
        first = (spec['hub_pairs'][m-1], parameter(spec['hub_pairs'][m-1], 0.4 if strong else 0.3))
        nu = 4.0 if first[0] == 'student' else None
        w = q[:, hub]
        if m >= 2:                                                  # second tree of the hubs: (hub, hub before the last | last)
            earlier = (spec['hub_pairs'][m-2], parameter(spec['hub_pairs'][m-2], 0.4 if strong else 0.3))
            given = fam.evaluate(earlier[0], u[:, hubs[m-2]], u[:, before], earlier[1], 4.0 if earlier[0] == 'student' else None)[1]
            w = fam.h_inverse('frank', w, given, parameter('frank', 0.2))
        u[:, hub] = fam.h_inverse(first[0], w, u[:, before], first[1], nu)
    if dense:
        for m, hub in enumerate(hubs):
            other = hubs[m+1] if m+1 < len(hubs) else hubs[m-1]
            state = spec['hub_pairs'][min(m, len(hubs)-2)]
            cond[m] = fam.evaluate(state, u[:, other], u[:, hub], parameter(state, 0.4 if strong else 0.3), 4.0 if state == 'student' else None)[1]
    coupled = {}
    for item in spec['planted']:
        terms = [(term['modulators'], course(term['shape'], g, t)) for term in item['terms']]
        coupled[item['pair'][1]] = (item['pair'][0], [(mods, c) for mods, c in terms if c != 0.0], [v for mods, _ in terms for v in mods])
    seconds = set(coupled)
    done = set(hubs)
    pending = [v for v in range(d) if v not in done]
    while pending:
        later = []
        for v in pending:
            block = v//BLOCK
            # a coupled leaf needs its modulators, and its partner's innovation as it is after the partner's own coupling
            if v in coupled and not (all(m in done for m in coupled[v][2]) and (coupled[v][0] not in seconds or coupled[v][0] in done)):
                later.append(v)
                continue
            if dense and v in spec['third'] and block*BLOCK+1 not in done:   # the first leaf of the block comes first
                later.append(v)
                continue
            if v in coupled and coupled[v][1]:
                i, terms, _ = coupled[v]
                eta = np.zeros(n)
                for mods, c in terms:
                    eta += c*np.prod([ndtri(np.clip(u[:, k], 1e-10, 1-1e-10)) for k in mods], axis=0)
                rho = np.tanh(eta)
                q[:, v] = ndtr(rho*ndtri(q[:, i])+np.sqrt(1-rho**2)*ndtri(q[:, v]))
            state, tau = spec['edges'][v]
            if block in spec['rising'] and t >= BLOCK_CHANGE and not static:
                tau = tau+spec.get('rise', 0.15)                    # 0.15 in every law of E26; `rise` is set by E29 only
            own = (state, parameter(state, tau), 4.0 if state == 'student' else None)
            w = q[:, v]
            if dense:
                second = spec['second'][v]
                if v in spec['third']:                              # (v, first leaf of the block | own hub, next hub)
                    third, last = spec['third'][v], block*BLOCK+1
                    last_own = spec['edges'][last]
                    last_tau = last_own[1]+(spec.get('rise', 0.15) if block in spec['rising'] and t >= BLOCK_CHANGE and not static else 0.0)
                    step = fam.evaluate(last_own[0], u[:, last], u[:, block*BLOCK], parameter(last_own[0], last_tau), 4.0 if last_own[0] == 'student' else None)[1]
                    last_second = spec['second'][last]
                    given = fam.evaluate(last_second[0], step, cond[block], parameter(*last_second))[1]
                    w = fam.h_inverse(third[0], w, given, parameter(*third))
                w = fam.h_inverse(second[0], w, cond[block], parameter(*second))
            u[:, v] = fam.h_inverse(own[0], w, u[:, block*BLOCK], own[1], own[2])
            done.add(v)
        if len(later) == len(pending):
            raise RuntimeError('the planted items cannot be ordered')
        pending = later
    if margins != 'moving':
        return u
    x = ndtri(np.clip(u, 1e-10, 1-1e-10))
    for v in range(d):
        kind, sign = spec['margin_kind'][v], spec['margin_sign'][v]
        if kind == 'drift':                                         # location moves by 1.5 and the scale grows to 1.6 over the windows
            x[:, v] = sign*1.5*t/(T-1.0)+(1.0+0.6*t/(T-1.0))*x[:, v]
        elif kind == 'shape' and t >= BLOCK_CHANGE:                 # normal, then skewed (log-normal, to the right or to the left)
            x[:, v] = sign*(np.exp(0.8*sign*x[:, v])-1.0)/0.8       # increasing in x for either sign
    return x
