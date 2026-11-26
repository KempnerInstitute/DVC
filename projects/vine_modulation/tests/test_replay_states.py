"""The state-space law of real_data/replay_states.py on a tiny example and on made-up moving replay."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("replay_states", Path(__file__).resolve().parent.parent / "real_data" / "replay_states.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def tiny_law():
    """Two positions on one map and no location; two cells."""
    law = rs.States.__new__(rs.States)
    law.rates = np.array([[40.0, 2.0], [3.0, 30.0], [8.0, 8.0]])
    law.walk = np.array([[0.7, 0.3], [0.3, 0.7]])
    law.places = 2
    law.factor = np.array([1.2, 0.8])
    law.gain_weight = np.full(len(rs.GAINS), 1.0 / len(rs.GAINS))
    law.first = np.array([0.4, 0.3, 0.3])
    law.mixture = np.array([0.5, 0.2, 0.3])
    law.leave, law.enter = 0.1, 0.2
    return law


def test_forward_pass_is_the_sum_over_every_path_of_states():
    law = tiny_law()
    rng = np.random.default_rng(1)
    y = (rng.random((3, 2, rs.STEPS)) < 0.3).astype(float)
    transition = law.transition()
    assert np.allclose(transition.sum(1), 1)
    found = law.evidence(y)
    for e in range(3):
        total = 0.0
        for gi, g in enumerate(rs.GAINS):
            p = -np.expm1(-g * law.factor[None, :] * law.rates * rs.DT)
            emit = np.array([[np.prod(np.where(y[e, :, t] > 0, p[s], 1 - p[s])) for s in range(3)] for t in range(rs.STEPS)])
            alpha = law.first * emit[0]
            for t in range(1, rs.STEPS):
                alpha = (alpha @ transition) * emit[t]
            total += law.gain_weight[gi] * alpha.sum()
        assert abs(found[e] - np.log(total)) < 1e-9
    paths = list(itertools.product(range(3), repeat=3))                         # a three-step check by explicit paths
    short = y[:, :, :3]
    law_short = tiny_law()
    saved = rs.STEPS
    try:
        rs.STEPS = 3
        brute = []
        for e in range(3):
            total = 0.0
            for gi, g in enumerate(rs.GAINS):
                p = -np.expm1(-g * law_short.factor[None, :] * law_short.rates * rs.DT)
                for path in paths:
                    weight = law_short.first[path[0]] * np.prod([transition[path[t - 1], path[t]] for t in range(1, 3)])
                    weight *= np.prod([np.prod(np.where(short[e, :, t] > 0, p[path[t]], 1 - p[path[t]])) for t in range(3)])
                    total += law_short.gain_weight[gi] * weight
            brute.append(np.log(total))
        assert np.allclose(law_short.evidence(short), brute, atol=1e-9)
    finally:
        rs.STEPS = saved


def test_leaving_a_cell_out_and_counting_it_silent():
    law = tiny_law()
    y = np.zeros((1, 2, rs.STEPS))
    y[0, 0, 3] = 1
    quiet = law.evidence(y, 1, silent=True)
    assert np.allclose(quiet, law.evidence(y))                                    # cell 1 was silent anyway
    without = law.evidence(y, 1, silent=False)
    p = law.predicted(y)
    assert np.allclose(p[:, 1], -np.expm1(quiet - without)) and 0 < p[0, 1] < 1


def made_up(rng, events, moving=0.6):
    """Twelve cells along one track; ripples replay a continuous walk (or nothing); cells spike in 10-ms steps."""
    bins = np.arange(40)
    centres = np.linspace(3, 36, 12)
    block = 0.3 + 15.0 * np.exp(-0.5 * ((bins[None, :] - centres[:, None]) / 2.5) ** 2)
    rates, which, walk = rs.positions([block], circular=False)
    places = len(which) - 1
    y = np.zeros((events, 12, rs.STEPS))
    for e in range(events):
        if rng.random() < moving:
            position, direction = rng.integers(places), rng.choice([-1, 1])
            path = np.clip(position + direction * np.arange(rs.STEPS), 0, places - 1)
        else:
            path = np.full(rs.STEPS, places)
        p = -np.expm1(-2.0 * rates[path] * rs.DT)                                   # (steps, cells)
        y[e] = (rng.random((rs.STEPS, 12)) < p).T
    return y, rates, walk, block


def test_fitted_law_finds_continuous_replay_and_predicts_held_out_cells_no_worse_than_the_stretch_law():
    """On made-up moving replay the stretch states already say which cells a path recruits; the gain from timing is
    small (about 0.025 per ripple with plain EM run to convergence), so the test asks only that the law not lose."""
    rng = np.random.default_rng(2)
    y, rates, walk, block = made_up(rng, 3000)
    law = rs.States(y[:2000], rates, walk)
    assert law.cycles < rs.CYCLES and law.mixture[0] > 0.6 and law.mixture[0] > law.mixture[2]
    x = (y.max(2) > 0).astype(np.int8)
    stretch_rates, _ = rs.content.content_states(block, np.zeros(40, dtype=int))
    stretch = rs.content.Content(x[:2000], stretch_rates, 0.1)
    score = {}
    for name, p in (("states", law.predicted(y[2000:])), ("stretch", rs.sequences.stretch_predicted(stretch, x[2000:]))):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        score[name] = (x[2000:] * np.log(p) + (1 - x[2000:]) * np.log1p(-p)).sum(1)
    difference = score["states"] - score["stretch"]
    assert difference.mean() > -2 * difference.std() / np.sqrt(len(difference))


def test_em_rounds_never_lower_the_log_probability_where_a_rate_is_zero():
    """25 November 2026: a zero map rate gave 0/0 in the factor update, every factor then fell round after round and the
    log probability fell with it (the first E72 run).  EM must never lower it."""
    rng = np.random.default_rng(4)
    y, rates, walk, _ = made_up(rng, 600)
    rates = rates.copy()
    rates[:-1][rates[:-1] < 1.0] = 0.0                                              # zero rates away from each field
    law = rs.States(y, rates, walk, cycles=0)
    values = np.array([law.round(y) for _ in range(30)])
    assert np.all(np.diff(values) > -1e-6 * abs(values[0])), np.diff(values).min()
    assert values[-1] - values[0] > 1.0
    assert np.isfinite(law.factor).all() and np.isclose(law.first.sum(), 1.0)
