"""Laws of the challenge of the second version of the pipeline (E31): mechanisms that no run of this project has
used, and fresh seeds of two that it has.

Background as in E26 (`e26_laws.design`, tree): 50 variables in five blocks of a hub and nine leaves; the leaves
of a block follow their hub with one of five pair families (Kendall tau 0.25 or 0.35); the hubs are a Gaussian
chain whose correlation rises from 0.4 to 0.6 at window 3; the leaves of the blocks named `rising` get stronger
at window 6.  Twelve windows.  A planted item couples the innovations of two leaves; `terms` are its modulating
terms with a course over the windows (`e26_laws.course`: constant, the episode `a` of windows 4 to 7, a rise).
New here is how a term enters and what couples the pair:

step      Fisher-z correlation c * 0.75 * sign(z): the correlation is +0.42 where the modulator is above its
          median and -0.42 below.  Not linear in the modulator's score.  Three constant terms, three episodes.
ushape    Fisher-z correlation c (z^2 - 1) / sqrt(2): an even function of the modulator (the quadratic shape of
          the scan).  Three constant terms, three episodes.
clayton   the innovations of the pair follow a Clayton copula with Kendall tau 0.3 + 0.25 tanh(c z): a pair with
          a baseline dependence of a family that is not the Gaussian of a trajectory edge, and whose strength is
          modulated.  With c = 0 the pair keeps tau = 0.3 (pairwise only).  Three constant terms, three episodes.
sum       two modulators of one pair at the same time, c z_1 + c z_2 with one course: two terms that share a pair
          and are both real.  Two pairs with an episode of both terms, one pair with both constant, and one
          ordinary episode and one ordinary constant term.
hub       a hub whose leaves get stronger at window 6 (by 0.30, as in `exposed_strong`) modulates pairs of leaves
          of blocks that do not change, by its own score: four constant terms and two episodes, three per hub.
          Every leaf of such a hub is an indirect member: the pair with that leaf shows
          the planted strength times the dependence of the leaf on the hub, which is what rises.
mixed     the plan of E18 (three episodes among leaves, two episodes with the hub of the pair, two constant
          terms, one rise) on a series that runs through all windows, with an autoregression of order one whose
          coefficient differs between blocks: 0.95, 0.95, 0.6, 0.3 and 0.  The roles are cut in time order
          inside every window with a gap, as in the law `continuous` of E29.
exposed   fresh seeds of `exposed_strong` (E29): six constant terms and two episodes whose modulators are
          leaves of blocks that get stronger by 0.30.
nested    fresh seeds of the plan of E28: four constant triples with a fourth-order episode on the same pair and
          four ordinary constant triples.

`generate(law, n, g, seed)` returns (spec, observations by role, settings); roles: train, validation, test,
confirm.  With g = 0 nothing is planted (in `clayton` the pairs keep their constant dependence).
"""
import numpy as np
from scipy.special import ndtr, ndtri

from common import fam
import e26_laws as laws
import e29_laws

T, BLOCK = laws.T, laws.BLOCK
STEP, TAU0, AMPLITUDE = 0.75, 0.3, 0.25
TEST = 1000                                                          # held-out observations per window, as in E26
GAP = 25
laws.PLANS['e31_six'] = (('constant_leaf', 3), ('episode_leaf', 3))
laws.PLANS['e31_sum'] = (('shared_pair', 3), ('episode_leaf', 1), ('constant_leaf', 1))
laws.PLANS['nested'] = (('nested_fourth', 4), ('constant_leaf', 4))
LAWS = dict(step=dict(d=50), ushape=dict(d=50), clayton=dict(d=50), sum=dict(d=50), hub=dict(d=50), mixed=dict(d=50), exposed=dict(d=50),
            nested=dict(d=50, max_order=4))
SERIAL = (0.95, 0.95, 0.6, 0.3, 0.0)                                # law `mixed`: coefficient of the variables of every block


def design(law, d, seed, g):
    if law in ('step', 'ushape', 'clayton'):
        spec = laws.design(d, seed, g, plan='e31_six', background='tree')
        for item in spec['planted']:
            if law == 'clayton':
                item['copula'] = 'clayton0'
            for term in item['terms']:
                term['link'] = dict(step='step', ushape='quadratic', clayton='linear')[law]
        if law == 'clayton' and g == 0:                               # the pairs and their constant dependence stay
            pairs = laws.design(d, seed, 0.6, plan='e31_six', background='tree')['planted']
            spec['pairs'] = [tuple(item['pair']) for item in pairs]
    elif law == 'sum':
        spec = laws.design(d, seed, g, plan='e31_sum', background='tree')
        shared = [item for item in spec['planted'] if item['kind'] == 'shared_pair']
        for index, item in enumerate(shared):
            for term in item['terms']:
                term['shape'] = 'constant' if index == len(shared)-1 else 'a'
    elif law == 'hub':
        spec = laws.design(d, seed, 0.0, plan='e18', background='tree', moving=True)
        spec['rise'] = 0.30
        rng = np.random.default_rng([seed, d, 311])
        rising = sorted(spec['rising'])
        still = [v for v in range(d) if v % BLOCK and v//BLOCK not in rising]
        count = 6 if g > 0 else 0
        leaves = [int(v) for v in rng.choice(still, 2*count, replace=False)]
        planted = []
        for index in range(count):
            hub = rising[index % len(rising)]*BLOCK
            shape = 'a' if index >= 4 else 'constant'
            planted.append(dict(kind='episode_of_hub' if shape == 'a' else 'constant_of_hub', pair=(leaves[2*index], leaves[2*index+1]),
                                terms=[dict(modulators=(hub,), shape=shape)]))
        spec['planted'] = planted
    elif law == 'mixed':
        spec = laws.design(d, seed, g, plan='e18', background='tree')
        spec['serial'] = [SERIAL[(v//BLOCK) % len(SERIAL)] for v in range(d)]
    elif law == 'exposed':
        spec = e29_laws.exposed_design(d, seed, g, rise=0.30)
    elif law == 'nested':
        spec = laws.design(d, seed, g, plan='nested', background='tree')
    else:
        raise ValueError(law)
    spec['law'] = law
    return spec


def sample(spec, n, t, g, rng, q=None):
    """n observations of window t with uniform margins.  q: uniform innovations (n, d) to use (for a series)."""
    d, hubs = spec['d'], spec['hubs']
    q = rng.uniform(size=(n, d)) if q is None else np.array(q, dtype=np.float64)
    u = np.empty((n, d))
    chain = 0.6 if t >= laws.CHAIN_CHANGE else 0.4
    for m, hub in enumerate(hubs):
        u[:, hub] = q[:, hub] if m == 0 else fam.h_inverse('gaussian', q[:, hub], u[:, hubs[m-1]], chain)
    score = lambda k: ndtri(np.clip(u[:, k], 1e-10, 1-1e-10))
    coupled = {item['pair'][1]: item for item in spec['planted']}
    constant = {j: i for i, j in spec.get('pairs', [])}              # `clayton` with nothing planted
    done = set(hubs)
    pending = [v for v in range(d) if v not in done]
    while pending:
        later = []
        for v in pending:
            item = coupled.get(v)
            if item is not None:
                partner = item['pair'][0]
                needed = {m for term in item['terms'] for m in term['modulators']}
                if not (needed <= done and (partner not in coupled or partner in done)):
                    later.append(v)
                    continue
                eta = np.zeros(n)
                for term in item['terms']:
                    c = laws.course(term['shape'], g, t)
                    if c == 0.0:
                        continue
                    z = np.prod([score(k) for k in term['modulators']], axis=0)
                    link = term.get('link', 'linear')
                    eta += c*(z if link == 'linear' else STEP*np.sign(z) if link == 'step' else (z*z-1.0)/np.sqrt(2.0))
                if item.get('copula') == 'clayton0':
                    tau = np.clip(TAU0+AMPLITUDE*np.tanh(eta), 0.02, 0.75)
                    q[:, v] = fam.h_inverse('clayton0', q[:, v], q[:, partner], 2.0*tau/(1.0-tau))
                elif np.any(eta != 0.0):
                    rho = np.tanh(eta)
                    q[:, v] = ndtr(rho*ndtri(np.clip(q[:, partner], 1e-12, 1-1e-12))+np.sqrt(1-rho**2)*ndtri(np.clip(q[:, v], 1e-12, 1-1e-12)))
            elif v in constant:
                q[:, v] = fam.h_inverse('clayton0', q[:, v], q[:, constant[v]], 2.0*TAU0/(1.0-TAU0))
            state, tau = spec['edges'][v]
            if v//BLOCK in spec['rising'] and t >= laws.BLOCK_CHANGE:
                tau = tau+spec.get('rise', 0.15)
            u[:, v] = fam.h_inverse(state, q[:, v], u[:, (v//BLOCK)*BLOCK], laws.parameter(state, tau), 4.0 if state == 'student' else None)
            done.add(v)
        if len(later) == len(pending):
            raise RuntimeError('the planted items cannot be ordered')
        pending = later
    return u


def series(length, coefficients, rng):
    """Uniform innovations (length, d): the normal scores of column j follow a stationary Gaussian autoregression
    of order one with the coefficient coefficients[j]."""
    a = np.asarray(coefficients, dtype=np.float64)
    e = rng.normal(size=(length, a.size))
    x = np.empty_like(e)
    x[0] = e[0]
    scale = np.sqrt(1.0-a*a)
    for i in range(1, length):
        x[i] = a*x[i-1]+scale*e[i]
    return ndtr(x)


def generate(law, n, g, seed):
    settings = LAWS[law]
    d = settings['d']
    spec = design(law, d, seed, g)
    rng = np.random.default_rng([seed, d, n, 312])
    roles = (('train', n), ('validation', n), ('test', TEST), ('confirm', n))
    if law in ('exposed', 'nested'):
        raw = {role: [laws.sample(spec, m, t, g, rng) for t in range(T)] for role, m in roles}
    elif law == 'mixed':
        sizes = dict(train=n, validation=n, confirm=n, test=TEST)   # the order of the roles in time inside a window
        per_window = sum(sizes.values())+GAP*len(sizes)
        innovations = series(T*per_window, spec['serial'], rng)
        raw, position = {role: [] for role in sizes}, 0
        for t in range(T):
            for role, m in sizes.items():
                raw[role].append(sample(spec, m, t, g, rng, q=innovations[position:position+m]))
                position += m+GAP
    else:
        raw = {role: [sample(spec, m, t, g, rng) for t in range(T)] for role, m in roles}
    return spec, raw, settings
