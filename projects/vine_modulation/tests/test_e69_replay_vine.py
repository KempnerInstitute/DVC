"""The replay-based sequential law of experiments/e69_replay_vine.py on made-up populations."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("e69_replay_vine", Path(__file__).resolve().parent.parent / "experiments" / "e69_replay_vine.py")
e69 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e69)
content = e69.content

BINS = 40
GROUPS = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
CENTRES = np.array([4, 5, 6, 14, 15, 16, 24, 25, 26, 33, 34, 35])


def population():
    """Twelve place cells in four groups with neighbouring fields; overlapping pairs and trios within a group,
    separate ones across three groups (as in the recordings, the separate class is the larger one)."""
    import itertools
    fields = 0.2 + 10.0 * np.exp(-0.5 * ((np.arange(BINS)[None, :] - CENTRES[:, None]) / 2.0) ** 2)
    rates, kinds = content.content_states(fields, np.zeros(BINS, dtype=int))
    group = np.repeat(np.arange(4), 3)
    pairs_o = [p for p in itertools.combinations(range(12), 2) if group[p[0]] == group[p[1]]]
    pairs_s = [p for p in itertools.combinations(range(12), 2) if group[p[0]] != group[p[1]]]
    trios_s = [t for t in itertools.combinations(range(12), 3) if len(set(group[list(t)])) == 3]
    sets = {("pairs", "overlapping"): np.array(pairs_o), ("pairs", "separate"): np.array(pairs_s),
            ("trios", "overlapping"): GROUPS.copy(), ("trios", "separate"): np.array(trios_s)}
    return rates, kinds, sets


def law_of(rates, kinds, share):
    """A share of the ripples replays one bin; the others have no place, at strengths 1/2 to 2."""
    law = content.Content(np.zeros((1, len(rates)), dtype=np.int8), rates, 0.1, cycles=0)
    weight = np.zeros((len(content.GAINS), rates.shape[1]))
    weight[2, kinds == 0] = share / BINS
    weight[1:4, -1] = (1 - share) / 3
    law.weight, law.factor = weight, np.full(len(rates), 2.0)
    law.tables()
    return law


def recording(rng, events=(3000, 3000), shares=(0.2, 0.3)):
    rates, kinds, sets = population()
    laws = {"PRE": law_of(rates, kinds, shares[0]), "POST": law_of(rates, kinds, shares[1])}
    epoch = np.repeat(["PRE", "POST"], events)
    x = np.zeros((sum(events), len(rates)), dtype=np.int8)
    for rest, law in laws.items():
        rows = np.flatnonzero(epoch == rest)
        pick = rng.choice(law.weight.size, len(rows), p=law.weight.ravel())
        x[rows] = rng.random((len(rows), len(rates))) < law.table[pick]
    order = np.argsort(rates[:, kinds == 0].argmax(1), kind="stable")
    return dict(name="made_up", x=x, epoch=epoch, sets=sets, order=order, rates=rates, kinds=kinds, window=0.1, laws=laws)


def test_replay_offsets_give_the_replay_laws_own_probability_of_every_ripple():
    rng = np.random.default_rng(1)
    rec = recording(rng)
    law = rec["laws"]["POST"]
    f = e69.features(rec["x"], rec["sets"], rec["order"])
    o = e69.replay_logits(law, f["x"], rec["order"])
    chain = (f["x"] * o - np.logaddexp(0.0, o)).sum(1)
    assert np.allclose(chain, law.evidence(law.likelihood(rec["x"])), atol=1e-6)


def test_counts_follow_the_order_of_the_fields_and_the_classes():
    rng = np.random.default_rng(2)
    rec = recording(rng, events=(200, 200))
    order = rec["order"][::-1]                                            # any order
    f = e69.features(rec["x"], rec["sets"], order)
    rank = {int(c): k for k, c in enumerate(order)}
    for cls, short in (("overlapping", "o"), ("separate", "s")):
        pairs, trios = np.zeros_like(f["x"]), np.zeros_like(f["x"])
        for i, j in rec["sets"][("pairs", cls)]:
            first, last = sorted((i, j), key=rank.get)
            pairs[:, rank[last]] += rec["x"][:, first]
        for trio in rec["sets"][("trios", cls)]:
            a, b, last = sorted(trio, key=rank.get)
            trios[:, rank[last]] += rec["x"][:, a] * rec["x"][:, b]
        assert np.array_equal(f["x" + short], pairs) and np.array_equal(f["n" + short], trios)


def test_gradient_of_every_variant_is_the_slope_of_its_objective_and_widening_keeps_a_fit():
    rng = np.random.default_rng(3)
    rec = recording(rng, events=(300, 300))
    f = e69.features(rec["x"], rec["sets"], rec["order"])
    o = e69.replay_logits(rec["laws"]["PRE"], f["x"], rec["order"])
    u = (rec["epoch"] == "POST").astype(float)
    rows = np.arange(600)
    n = rec["x"].shape[1]
    for variant in e69.VARIANTS:
        objective, _, _, pairs = e69.problem(f, rows, o, u, variant, 0.3)
        theta = rng.normal(0, 0.2, n + pairs + len(e69.VARIANTS[variant]))
        value, gradient = objective(theta)
        for index in rng.choice(len(theta), min(10, len(theta)), replace=False):
            step = np.zeros(len(theta))
            step[index] = 1e-5
            slope = (objective(theta + step)[0] - objective(theta - step)[0]) / 2e-5
            assert abs(slope - gradient[index]) < 1e-4 * max(1.0, abs(slope)), (variant, index)
    theta = e69.fit(f, rows, o, u, "shared", 0.3)
    start = e69.widen(theta, n, "shared", "class")
    assert abs(e69.problem(f, rows, o, u, "class", 0.3)[0](start)[0] - e69.problem(f, rows, o, u, "shared", 0.3)[0](theta)[0]) < 1e-8


def test_class_change_answers_a_change_of_the_overlapping_trios_and_not_classes_that_start_apart_and_move_together():
    rng = np.random.default_rng(4)
    rec = recording(rng)
    found = {}
    for name, coefficients in (("class", dict(d_o=0.0, d_s=0.0, c_o=0.8, c_s=0.0)), ("apart", dict(d_o=0.8, d_s=0.0, c_o=0.6, c_s=0.6))):
        x = e69.generate(rec, coefficients, rng)
        scores, block, chosen = e69.held_out(rec, x)
        found[name] = e69.compared(scores, block)
        assert len(chosen["folds"]) == e69.FOLDS and all(c["tau"] in e69.TAUS for c in chosen["folds"])
    z = {name: found[name]["class_change"]["total"] / np.sqrt(found[name]["class_change"]["variance"]) for name in found}
    assert z["class"] > 2.0
    assert z["apart"] < 2.0
    assert found["apart"]["time_varying"]["total"] > 2 * np.sqrt(found["apart"]["time_varying"]["variance"])



def test_trio_scales_are_fit_on_training_ripples_only():
    training = np.array([0, 1])
    raw = {"no": np.array([[1., 3.], [1., 3.], [100., 200.]]),
           "ns": np.array([[0., 0.], [0., 0.], [15., 26.]])}
    scaled, units = e69.scale_trio_features(raw, training)
    assert units == {"no": 1., "ns": 1.}
    np.testing.assert_array_equal(scaled["ns"], raw["ns"])
    raw["no"][2] = 1e5
    again, new_units = e69.scale_trio_features(raw, training)
    assert new_units == units
    np.testing.assert_array_equal(again["no"][training], scaled["no"][training])


def test_common_count_keeps_the_equality_constraint_in_raw_pair_units():
    raw = {"no": np.array([[0., 1.], [0., 3.], [20., 0.]]),
           "ns": np.array([[0., 4.], [0., 8.], [1., 0.]])}
    train = np.array([0, 1])
    scaled, units = e69.scale_trio_features(raw, train, "common")
    np.testing.assert_allclose(units["no"], raw["ns"][train].std())
    assert units["no"] == units["ns"]
    np.testing.assert_allclose(scaled["no"]+scaled["ns"], (raw["no"]+raw["ns"])/units["no"])
    raw["ns"][2,0] = 100000
    assert e69.scale_trio_features(raw, train, "common")[1] == units


def test_equal_per_pair_plant_has_one_coefficient_on_raw_counts():
    rec = recording(np.random.default_rng(101))
    raw = e69.features(rec["x"], rec["sets"], rec["order"])
    units = {name: raw[name].std() for name in ("no","ns")}
    assert 0 < min(units.values()) < max(units.values())
    rate = .15 / max(units.values())
    manual = dict(e69.SCENARIOS["apart"], c_o=rate*units["no"], c_s=rate*units["ns"])
    direct = e69.generate(rec, manual, np.random.default_rng(203))
    calibrated = e69.generate(rec, e69.SCENARIOS["apart"], np.random.default_rng(203), equal_per_pair=True)
    np.testing.assert_array_equal(direct, calibrated)
    np.testing.assert_allclose(manual["c_o"]/units["no"], manual["c_s"]/units["ns"])
