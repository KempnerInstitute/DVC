"""Reference pipeline for scanning subsets, fitting modulated edges and testing changes.

1. Detection (`detect`).  Within-window ranks; a switching pair copula for every pair of variables; every
   (i, j | k) statistic in every window, combined over contiguous windows; a triple is a candidate only if all three
   choices of the conditioned pair reject at familywise 0.05 (Bonferroni over triples, shapes and intervals), and is
   confirmed only if it still rejects after conditioning on the variables most dependent on its members and on the
   third member of every stronger candidate that shares two variables.  The same for quadruples with the product
   of the two conditioning coordinates (all six choices of the pair; familywise 0.05 separately).
2. Model (`fit_models`).  A dynamic C-vine (root order by the DVC-switch rule) in which only confirmed interactions
   may be modelled, in the windows of their detected runs.  With validation data the switching penalties are chosen
   on it and the simplified candidates compete, so a modulated state is kept only if it improves validation score.
3. Readouts (`readout`).  Held-out log score of the simplified vine, of the model with its fitted correlations
   truncated at order 3 and at order 4 (orders on the correlation scale), fitted coefficients, and the variance of
   each fitted correlation by order.
4. Change (`change_tests`).  For every confirmed interaction with a run of at least four windows, the bootstrap
   test of `vinemod.change` (99 replicates).

Observations within a window are assumed independent (no serial-dependence correction in P1).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import change, cvine, pairscan
from . import families as fam

CONFIG = dict(
    version='P1',
    families=('gaussian', 'student', 'clayton', 'gumbel', 'frank'), rotations=True,
    pair_switch=0.05, pair_nu_grid=(3.0, 5.0, 10.0),
    shapes=('linear', 'quadratic'), level=0.05,
    explain_away_per_member=3, explain_away_per_member_fourth=2, fourth_order=True,
    model_grid=[dict(switch=s, drift=d, activation=a)
                for s, d, a in itertools.product((0.02, 0.08, 0.2), (0.0, 0.02), (0.0, 0.02))],
    model_switch=0.08, change_replicates=99,
)


def states(config=CONFIG):
    return fam.default_states(config['families'], rotations=config['rotations'])


@dataclass
class Detection:
    triples: List[dict]                 # confirmed: dict(subset, run=(start, stop), value)
    quadruples: List[dict]
    counts: Dict[str, int]
    steps: int
    layer: Optional[pairscan.PairLayer] = field(default=None, repr=False)

    def subsets(self):
        return [tuple(x['subset']) for x in self.triples + self.quadruples]

    def allowed(self):
        """{subset: mask over windows} for `cvine.fit`."""
        out = {}
        for item in self.triples + self.quadruples:
            mask = np.zeros(self.steps, dtype=bool)
            mask[item['run'][0]:item['run'][1]] = True
            out[tuple(item['subset'])] = mask
        return out

    def summary(self):
        return dict(triples=[dict(subset=list(x['subset']), run=list(x['run']), value=x['value']) for x in self.triples],
                    quadruples=[dict(subset=list(x['subset']), run=list(x['run']), value=x['value']) for x in self.quadruples],
                    counts=self.counts)


def detect(windows: Sequence[np.ndarray], config=CONFIG, workers=1) -> Detection:
    """Step 1.  windows: list of (n_t, d) pseudo-observations (within-window ranks)."""
    d = windows[0].shape[1]
    layer = pairscan.fit_pair_layer(windows, states=states(config), switch=config['pair_switch'],
                                    nu_grid=config['pair_nu_grid'], workers=workers)
    stat = pairscan.scan_all(layer, windows, shapes=config['shapes'], workers=workers)
    loose, _ = pairscan.any_orientation(stat, config['level'])
    found, _ = pairscan.candidates(stat, config['level'])
    checked, _ = pairscan.confirm(layer, windows, found, per_member=config['explain_away_per_member'], level=config['level'])
    triples = [dict(subset=tuple(c['triple']), run=tuple(c['interval']), value=c['value']) for c in checked if c['confirmed']]
    counts = dict(triples_any_choice=len(loose), triples_all_choices=len(found), triples_confirmed=len(triples))
    quadruples = []
    if config['fourth_order'] and d >= 4:
        entries, _ = pairscan.fourth_order_scan(layer, windows, level=config['level'], workers=workers)
        table = pairscan.quadruples(entries)
        checked4, _ = pairscan.confirm_fourth(layer, windows, table, per_member=config['explain_away_per_member_fourth'],
                                              level=config['level'])
        quadruples = [dict(subset=tuple(c['quadruple']), run=tuple(c['interval']), value=c['value'])
                      for c in checked4 if c['confirmed']]
        counts.update(quadruples_any_choice=len(table), quadruples_all_choices=len(checked4),
                      quadruples_confirmed=len(quadruples))
    return Detection(triples, quadruples, counts, len(windows), layer)


def fit_models(train: Sequence[np.ndarray], detection: Detection, validation: Optional[Sequence[np.ndarray]] = None,
               config=CONFIG, order=None):
    """Step 2.  Returns dict(simplified=(model, options), gated=(model, options)).

    Without validation data both use the fixed penalty ``model_switch``.  With it, each is the candidate of the
    penalty grid with the best validation score; the gated choice also includes every simplified candidate.
    """
    family_states = states(config)
    order = list(order) if order is not None else cvine.infer_order(list(train))
    allowed = detection.allowed()
    if validation is None:
        options = dict(switch=config['model_switch'], drift=0.0, activation=0.0)
        simple = cvine.fit(list(train), order=order, states=family_states, **options)
        gated = cvine.fit(list(train), order=order, states=family_states, allowed=allowed, **options) if allowed else simple
        return dict(simplified=(simple, options), gated=(gated, options))
    simple = None
    for options in config['model_grid']:
        model = cvine.fit(list(train), order=order, states=family_states, **options)
        value = cvine.score(model, list(validation), eps=1e-6)
        if simple is None or value > simple[2] + 1e-12:
            simple = (model, options, value)
    gated = simple                                                  # the simplified candidates compete
    if allowed:
        for options in config['model_grid']:
            model = cvine.fit(list(train), order=order, states=family_states, allowed=allowed, **options)
            value = cvine.score(model, list(validation), eps=1e-6)
            if value > gated[2] + 1e-12:
                gated = (model, options, value)
    return dict(simplified=simple[:2], gated=gated[:2])


def readout(models, test: Sequence[np.ndarray], detection: Detection, eps=1e-6):
    """Step 3.  Per-window held-out scores and, for every confirmed interaction, the fitted coefficients and the
    variance of the fitted correlation by order (mean over the windows in which it is modelled)."""
    simple, gated = models['simplified'][0], models['gated'][0]
    per_window = lambda model, max_order=None: [float(v.mean()) for v in cvine.log_density(model, list(test), eps, max_order)]
    out = dict(simplified=per_window(simple), order3=per_window(gated, 3), order4=per_window(gated, 4), full=per_window(gated),
               not_converged=int(gated.config.get('not_converged', 0)), interactions=[])
    for item in detection.triples + detection.quadruples:
        level, edge, members = change.carrier(gated.order, item['subset'])
        path = gated.edges[level][edge]
        windows = [t for t, state in enumerate(path) if state.components]
        shares = {}
        for t in windows:
            parts = cvine.coupling_decomposition(path[t])
            for key, value in parts.items():
                if key != 'mean':
                    shares[2 + len(key)] = shares.get(2 + len(key), 0.0) + value / len(windows)
        wanted = ('linear', members) if len(members) == 1 else ('product', members)
        coefficient = [[t, float(path[t].beta[1 + path[t].components.index(wanted)])] for t in windows
                       if wanted in path[t].components]
        out['interactions'].append(dict(subset=list(item['subset']), run=list(item['run']), modelled_windows=windows,
                                        coefficient=coefficient, variance_by_order={str(k): v for k, v in sorted(shares.items())}))
    return out


def change_tests(models, train: Sequence[np.ndarray], detection: Detection, config=CONFIG, rng=None, options=None):
    """Step 4.  One bootstrap change test per confirmed interaction whose run has at least four windows."""
    gated, chosen = models['gated']
    options = chosen if options is None else options
    allowed = detection.allowed()
    refit = lambda sim: cvine.fit(sim, order=gated.order, states=states(config), allowed=allowed, **options)
    out = []
    for item in detection.triples + detection.quadruples:
        run = list(range(item['run'][0], item['run'][1]))
        result = change.change_test(gated, list(train), item['subset'], run, replicates=config['change_replicates'],
                                    rng=rng, refit=refit)
        out.append(dict(subset=list(item['subset']), run=list(item['run']), result=result))
    return out
