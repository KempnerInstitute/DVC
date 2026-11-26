"""Exact conditional inference for a shared change of the three-way term (vinemod/binary_pooled.py)."""
import itertools
import math

import numpy as np

from dvc_package.modulation.binary_pooled import (PARITY, SplitFamily, efficient_weights, fitted_split, pooled_change, shared_shift,
                                   single_trio_pvalues, trio_tables)

BITS = (np.arange(8)[:, None] >> np.arange(3)) & 1


def law(single, pair, triple):
    """Probabilities of the eight patterns of a log-linear law with the given terms."""
    energy = BITS @ np.asarray(single, dtype=float)
    energy = energy + pair[0] * BITS[:, 0] * BITS[:, 1] + pair[1] * BITS[:, 0] * BITS[:, 2] + pair[2] * BITS[:, 1] * BITS[:, 2]
    energy = energy + triple * BITS.prod(1)
    p = np.exp(energy)
    return p / p.sum()


def test_parity_contrast_is_the_three_way_term():
    p = law([-1.0, -0.5, -2.0], [0.4, -0.3, 0.8], 0.7)
    assert abs(np.log(p) @ PARITY - 0.7) < 1e-12


def test_tables_match_a_direct_count_with_and_without_strata():
    rng = np.random.default_rng(0)
    x = (rng.random((400, 7)) < 0.3).astype(int)
    trios = np.array(list(itertools.combinations(range(7), 3)))
    plain = trio_tables(x, trios, batch=10)
    size, edges = x.sum(1), np.array([2, 4])
    stratified = trio_tables(x, trios, size, edges, batch=10)
    assert plain.shape == (35, 1, 8) and stratified.shape == (35, 3, 8)
    for t, (i, j, k) in enumerate(trios):
        code = x[:, i] + 2 * x[:, j] + 4 * x[:, k]
        assert np.array_equal(plain[t, 0], np.bincount(code, minlength=8))
        others = size - x[:, i] - x[:, j] - x[:, k]
        level = np.searchsorted(edges, others, side="right")
        for s in range(3):
            assert np.array_equal(stratified[t, s], np.bincount(code[level == s], minlength=8))
    assert np.array_equal(stratified.sum(1), plain[:, 0])


def enumerate_split(a, b, delta=0.0):
    odd, even = PARITY > 0, PARITY < 0
    lower = int(max(-b[odd].min(), -a[even].min()))
    upper = int(min(b[even].min(), a[odd].min()))
    us = np.arange(lower, upper + 1)
    logw = np.array([delta * u - sum(math.lgamma(b[k] + u * PARITY[k] + 1) + math.lgamma(a[k] - u * PARITY[k] + 1)
                                     for k in range(8)) for u in us])
    p = np.exp(logw - logw.max())
    p /= p.sum()
    mean = (p * us).sum()
    return mean, (p * (us - mean) ** 2).sum()


def test_family_moments_match_enumeration():
    rng = np.random.default_rng(1)
    a = rng.multinomial(60, law([-1, -1, -1], [0.3, 0.2, 0.1], 0.5), size=40)
    b = rng.multinomial(90, law([-0.5, -1.2, -0.8], [0.0, 0.6, -0.2], 0.5), size=40)
    family = SplitFamily(a, b, batch_cells=50)  # several batches
    for delta in (0.0, 0.8, -1.3):
        mean, variance = family.moments(delta)
        for r in range(40):
            m, v = enumerate_split(a[r], b[r], delta)
            assert abs(mean[r] - m) < 1e-9 and abs(variance[r] - v) < 1e-9


def test_conditional_score_is_unbiased_in_sparse_tables_whatever_the_lower_terms():
    # the two sets have different single-cell and pair terms and the same three-way term; tables are sparse
    rng = np.random.default_rng(2)
    p_a = law([-2.2, -1.8, -2.0], [0.5, 0.2, 0.9], 0.6)
    p_b = law([-1.2, -2.4, -1.5], [-0.2, 1.0, 0.3], 0.6)
    replicates = 20000
    a, b = rng.multinomial(300, p_a, size=replicates), rng.multinomial(700, p_b, size=replicates)
    family = SplitFamily(a, b)
    mean, variance = family.moments(0.0)
    score = -mean
    assert family.informative.mean() > 0.8
    assert abs(score.mean()) < 4 * score.std() / math.sqrt(replicates)
    # the conditional variance is the variance of the score
    assert abs(variance.mean() / score.var() - 1) < 0.05
    # the unconditional fit of no change is biased here: its score does not average to zero (4 standard errors)
    _, _, _, asymptotic = efficient_weights(a, b)
    assert abs(asymptotic.mean()) > 3 * asymptotic.std() / math.sqrt(replicates)


def test_shared_shift_recovers_a_planted_change():
    rng = np.random.default_rng(3)
    rows = 3000
    a = rng.multinomial(400, law([-1.5, -1.5, -1.5], [0.3, 0.3, 0.3], 0.2), size=rows)
    b = rng.multinomial(500, law([-1.3, -1.6, -1.4], [0.5, 0.1, 0.3], 0.2 + math.log(1.5)), size=rows)
    delta, information, score0, information0 = shared_shift(SplitFamily(a, b))
    assert abs(delta - math.log(1.5)) < 4 / math.sqrt(information)
    assert score0 > 0 and information0 > 0 and 1 / math.sqrt(information) < 0.03


def test_fitted_split_has_one_three_way_term_and_weights_sum_to_its_score():
    rng = np.random.default_rng(4)
    a = rng.multinomial(500, law([-1, -1, -1], [0.3, 0.2, 0.1], 0.1), size=30)
    b = rng.multinomial(800, law([-0.5, -1.2, -0.8], [0.0, 0.6, -0.2], 0.9), size=30)
    u, fa, fb = fitted_split(a, b)
    assert np.all(fa > 0) and np.all(fb > 0)
    assert np.allclose(np.log(fa) @ PARITY, np.log(fb) @ PARITY, atol=1e-8)
    assert np.allclose(fb[:, 7] - b[:, 7], u) and np.allclose(a[:, 7] - fa[:, 7], u)
    w_a, w_b, information, score = efficient_weights(a, b)
    assert np.allclose((a * w_a).sum(1) + (b * w_b).sum(1), score, atol=1e-8) and np.allclose(score, -u)
    assert np.allclose((fa * w_a ** 2).sum(1) + (fb * w_b ** 2).sum(1), information)
    assert np.allclose(information, 1 / ((1 / fa).sum(1) + (1 / fb).sum(1)))


def test_tables_without_information_are_ignored():
    a = np.array([[50, 5, 6, 0, 4, 0, 0, 0]])
    b = np.array([[80, 9, 7, 0, 6, 0, 0, 0]])
    family = SplitFamily(a, b)
    assert not family.informative[0]
    w_a, w_b, information, score = efficient_weights(a, b)
    assert information[0] == 0 and score[0] == 0 and not w_a.any() and not w_b.any()
    assert np.isnan(shared_shift(family)[0])


def test_pooled_test_holds_its_level_for_overlapping_trios_of_independent_cells():
    # all 220 trios of twelve independent cells; rates differ between the two sets
    rng = np.random.default_rng(5)
    trios = np.array(list(itertools.combinations(range(12), 3)))
    rates_a, rates_b = rng.uniform(0.08, 0.35, 12), rng.uniform(0.08, 0.35, 12)
    z, design, delta, se = [], [], [], []
    for _ in range(150):
        x_a = (rng.random((900, 12)) < rates_a).astype(int)
        x_b = (rng.random((1300, 12)) < rates_b).astype(int)
        fit = pooled_change(x_a, x_b, trios)
        z.append(fit.z), design.append(fit.design_effect), delta.append(fit.delta), se.append(fit.se)
    z, design, delta, se = map(np.array, (z, design, delta, se))
    assert 0.8 < np.median(design) < 1.25          # efficient scores of overlapping trios are uncorrelated
    assert abs(z.mean()) < 0.3 and 0.8 < z.std() < 1.2
    assert (np.abs(z) > 1.96).mean() < 0.12
    assert abs(delta.mean()) < 3 * delta.std() / math.sqrt(len(delta)) and 0.8 < se.mean() / delta.std() < 1.25


def test_pooled_test_with_strata_finds_a_planted_change_and_single_trios_do_not():
    rng = np.random.default_rng(6)
    # the shift is small on purpose: a three-way term of 0.2 shared by all 84 trios of nine cells would make
    # the all-active pattern the most likely one (84 * 0.2 exceeds the cost of nine active cells)
    cells, shift = 9, 0.05
    trios = np.array(list(itertools.combinations(range(cells), 3)))

    def sample(n, triple):
        # a log-linear law in which every trio has the three-way term `triple`, by Gibbs sampling
        x = (rng.random((n, cells)) < 0.3).astype(float)
        for _ in range(40):
            for i in range(cells):
                others = x.sum(1) - x[:, i]
                field = math.log(0.3 / 0.7) + triple * others * (others - 1) / 2
                x[:, i] = rng.random(n) < 1 / (1 + np.exp(-field))
        return x.astype(int)

    x_a, x_b = sample(20000, 0.0), sample(20000, shift)
    plain = pooled_change(x_a, x_b, trios)
    assert plain.z > 3 and 0.5 * shift < plain.delta < 2.5 * shift
    edges = np.array([2, 4])
    stratified = pooled_change(x_a, x_b, trios, x_a.sum(1), x_b.sum(1), edges)
    assert stratified.informative_rows > len(trios) and stratified.z > 2
    p = single_trio_pvalues(x_a[:2500], x_b[:2500], trios[:25], x_a[:2500].sum(1), x_b[:2500].sum(1), edges)
    assert p.shape == (25,) and np.all((p > 0) & (p <= 1))
    assert (p < 0.05 / 25).sum() <= 1


def test_pairs_the_split_of_the_joint_count_gives_the_change_of_the_log_odds_ratio():
    # the classical conditional analysis of 2 x 2 tables, one order below the trios
    rng = np.random.default_rng(7)

    def pair_law(single, coupling):
        energy = np.array([0.0, single[0], single[1], single[0] + single[1] + coupling])
        p = np.exp(energy)
        return p / p.sum()

    rows = 4000
    a = rng.multinomial(300, pair_law([-1.5, -2.0], 0.4), size=rows)
    b = rng.multinomial(450, pair_law([-1.0, -2.5], 0.4), size=rows)
    family = SplitFamily(a, b)
    mean, variance = family.moments(0.0)
    assert abs(mean.mean()) < 4 * mean.std() / math.sqrt(rows)          # exact under no change, sparse or not
    assert abs(variance.mean() / mean.var() - 1) < 0.08
    shifted = rng.multinomial(450, pair_law([-1.0, -2.5], 0.4 + math.log(1.5)), size=rows)
    delta, information, score0, information0 = shared_shift(SplitFamily(a, shifted))
    assert abs(delta - math.log(1.5)) < 4 / math.sqrt(information) and score0 > 0
    u, fa, fb = fitted_split(a[:50], shifted[:50])
    contrast = np.array([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(np.log(fa) @ contrast, np.log(fb) @ contrast, atol=1e-8)


def test_pooled_pair_change_holds_its_level_and_finds_a_planted_coupling():
    rng = np.random.default_rng(8)
    pairs = np.array(list(itertools.combinations(range(10), 2)))
    rates_a, rates_b = rng.uniform(0.08, 0.35, 10), rng.uniform(0.08, 0.35, 10)
    z = []
    for _ in range(120):
        x_a = (rng.random((900, 10)) < rates_a).astype(int)
        x_b = (rng.random((1300, 10)) < rates_b).astype(int)
        z.append(pooled_change(x_a, x_b, pairs, x_a.sum(1), x_b.sum(1), np.array([2, 4])).z)
    z = np.array(z)
    assert abs(z.mean()) < 0.3 and 0.8 < z.std() < 1.2 and (np.abs(z) > 1.96).mean() < 0.12

    def coupled(n, coupling):
        x = (rng.random((n, 10)) < 0.25).astype(float)
        for _ in range(30):
            for i in range(10):
                others = x.sum(1) - x[:, i]
                x[:, i] = rng.random(n) < 1 / (1 + np.exp(-(math.log(0.25 / 0.75) + coupling * others)))
        return x.astype(int)

    x_a, x_b = coupled(6000, 0.0), coupled(6000, 0.1)
    fit = pooled_change(x_a, x_b, pairs)
    assert fit.z > 4 and 0.05 < fit.delta < 0.2
    assert trio_tables(x_a, pairs).shape == (45, 1, 4)


def test_parts_combine_and_coinciding_events_are_not_counted_as_independent():
    from dvc_package.modulation.binary_pooled import combine, contrast, prepare
    rng = np.random.default_rng(9)
    trios = np.array(list(itertools.combinations(range(8), 3)))
    x_a, x_b = (rng.random((1500, 8)) < 0.25).astype(int), (rng.random((1800, 8)) < 0.3).astype(int)
    single = pooled_change(x_a, x_b, trios)
    part = prepare(x_a, x_b, trios)
    again = combine([part])
    assert abs(again.delta - single.delta) < 1e-12 and abs(again.se - single.se) < 1e-12
    assert abs((again.influence_a ** 2).sum() + (again.influence_b ** 2).sum() - again.se ** 2) < 1e-12
    # the same population entered twice: with every event alone the standard error would shrink by sqrt(2);
    # with coinciding events labelled it stays what it was
    alone = combine([part, part])
    labelled = combine([part, part], [np.arange(1500), np.arange(1500)], [np.arange(1800), np.arange(1800)])
    assert abs(alone.delta - single.delta) < 1e-9 and abs(labelled.delta - single.delta) < 1e-9
    assert abs(alone.se * math.sqrt(2) / single.se - 1) < 1e-6
    assert abs(labelled.se / single.se - 1) < 1e-3          # the factor G / (G - 1) of 1,500 and 1,800 clusters
    # blocks of 50 independent events as clusters: the same standard error up to the noise of 30 and 36 blocks,
    # and the standard error with every event alone is returned beside it
    blocks = combine([part], [np.arange(1500) // 50], [np.arange(1800) // 50], also={"events": (None, None)})
    assert 0.7 < blocks.se / single.se < 1.4 and abs(blocks.other["events"][0] / single.se - 1) < 1e-9
    assert abs(contrast(blocks, blocks, "events")["se"]) < 1e-12
    # two classes on the same events: the contrast of a class with itself is exactly zero with zero error,
    # and disjoint cells give nearly independent estimates
    same = contrast(again, combine([part]))
    assert same["delta"] == 0 and same["se"] < 1e-12
    first = combine([prepare(x_a, x_b, np.array(list(itertools.combinations(range(4), 3))))])
    second = combine([prepare(x_a, x_b, np.array(list(itertools.combinations(range(4, 8), 3))))])
    both = contrast(first, second)
    assert abs(both["se"] ** 2 / (first.se ** 2 + second.se ** 2) - 1) < 0.25 and abs(both["correlation"]) < 0.25


def test_log_probability_of_the_observed_split_matches_enumeration():
    rng = np.random.default_rng(10)
    a = rng.multinomial(80, law([-1, -1, -1], [0.3, 0.2, 0.1], 0.5), size=25)
    b = rng.multinomial(120, law([-0.5, -1.2, -0.8], [0.0, 0.6, -0.2], 0.5), size=25)
    family = SplitFamily(a, b, batch_cells=60)
    for delta in (0.0, 0.7, np.linspace(-1, 1, 25)):
        got = family.log_probability(delta)
        each = np.broadcast_to(delta, (25,))
        for r in range(25):
            odd, even = PARITY > 0, PARITY < 0
            lower = int(max(-b[r][odd].min(), -a[r][even].min()))
            upper = int(min(b[r][even].min(), a[r][odd].min()))
            us = np.arange(lower, upper + 1)
            logw = np.array([each[r] * u - sum(math.lgamma(b[r][k] + u * PARITY[k] + 1) + math.lgamma(a[r][k] - u * PARITY[k] + 1)
                                               for k in range(8)) for u in us])
            want = logw[us == 0][0] - (logw.max() + math.log(np.exp(logw - logw.max()).sum()))
            assert abs(got[r] - want) < 1e-9


def test_scan_agrees_with_the_two_set_estimate_and_finds_a_planted_boundary():
    from dvc_package.modulation.binary_pooled import block_information, bridge_p, scan_change, scan_rows
    rng = np.random.default_rng(11)
    # two blocks: the scan at its one boundary is the conditional split of the two-set family
    a = rng.multinomial(150, law([-1.2, -1.0, -1.4], [0.3, 0.2, 0.1], 0.3), size=60)
    b = rng.multinomial(200, law([-1.0, -1.3, -1.1], [0.1, 0.5, 0.0], 0.3), size=60)
    score, variance, single = scan_rows(np.stack((a, b), axis=1))
    mean, var = SplitFamily(a, b).moments(0.0)
    assert abs(score[0] + mean.sum()) < 1e-8 and abs(variance[0] - var.sum()) < 1e-8
    # six blocks with different margins, the term steps up after the fourth
    def block(n, rows, triple, seed):
        r = np.random.default_rng(seed)
        single_terms = r.uniform(-1.6, -0.8, 3)
        return r.multinomial(n, law(single_terms, r.uniform(0, 0.5, 3), triple), size=rows)
    tables = np.stack([block(250, 300, 0.2 if k < 4 else 0.9, 100 + k) for k in range(6)], axis=1)
    found = scan_change(tables)
    assert found["boundary"] == 4 and found["z_max"] > 5 and found["p_max"] < 0.001
    assert abs(found["step"][3] - 0.7) < 0.2
    flat = scan_change(np.stack([block(250, 300, 0.2, 200 + k) for k in range(6)], axis=1))
    assert flat["z_max"] < 3.2 and flat["p_max"] > 0.01
    # the blocks' shares of the information: equal blocks, equal shares
    info = block_information(scan_rows(np.stack([block(250, 200, 0.2, 7) for _ in range(4)], axis=1))[2])
    assert np.allclose(info / info.sum(), 0.25, atol=0.02)
    # one boundary in the middle: the largest |z| is |z| itself
    assert abs(bridge_p(1.96, [0.5]) - 0.05) < 0.005
    assert bridge_p(1.96, [0.2, 0.4, 0.6, 0.8]) > 0.1


def test_scan_holds_its_level_when_only_lower_order_terms_change_between_blocks():
    from dvc_package.modulation.binary_pooled import scan_change
    rng = np.random.default_rng(12)
    p_values = []
    for _ in range(120):
        tables = []
        for k in range(6):
            single_terms, pairs = rng.uniform(-1.8, -0.8, 3), rng.uniform(-0.2, 0.8, 3)
            tables.append(rng.multinomial(200, law(single_terms, pairs, 0.4), size=40))
        p_values.append(scan_change(np.stack(tables, axis=1))["p_max"])
    p_values = np.array(p_values)
    assert (p_values < 0.05).mean() < 0.11 and 0.3 < np.median(p_values) < 0.7
