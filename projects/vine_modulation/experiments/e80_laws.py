"""E80.  Fourth-order episodes without constant triples: the law `quartets` (design in `E80_QUARTETS_ALONE.md`).

The law `nested` of the challenge E31 (`e31_laws.py`) plants four pairs that each carry a constant third-order term
(modulator k) and a fourth-order episode (the product of the normal scores of k and a second modulator m, windows
4 to 7), and four ordinary constant triples on other pairs.  `quartets` is `nested` with every constant third-order
term removed: the same background, the same four pairs, modulators, episode and strength, and nothing else planted.
For a given seed the quartets are those of `nested` with that seed, and because `e26_laws.sample` draws all its
random numbers before it couples the planted pairs, the observations differ from those of `nested` only in the
coupling of the planted pairs.

generate(law, n, g, seed) follows `e31_laws.generate` for `nested`; `e31_laws.py` is not changed.
"""
import numpy as np

import e26_laws as laws
import e31_laws as e31

T = laws.T
LAWS = dict(quartets=dict(d=50, max_order=4))


def design(law, d, seed, g):
    """The design of `nested` for this seed with every term of one modulator removed."""
    if law != 'quartets':
        raise ValueError(law)
    spec = e31.design('nested', d, seed, g)
    planted = []
    for item in spec['planted']:
        terms = [term for term in item['terms'] if len(term['modulators']) > 1]
        if terms:
            planted.append(dict(item, kind='fourth_alone', terms=terms))
    spec['planted'] = planted
    spec['law'] = law
    return spec


def generate(law, n, g, seed):
    """(spec, observations by role, settings), roles train, validation, test and confirm, as in `e31_laws`."""
    settings = LAWS[law]
    d = settings['d']
    spec = design(law, d, seed, g)
    rng = np.random.default_rng([seed, d, n, 312])
    roles = (('train', n), ('validation', n), ('test', e31.TEST), ('confirm', n))
    raw = {role: [laws.sample(spec, m, t, g, rng) for t in range(T)] for role, m in roles}
    return spec, raw, settings
