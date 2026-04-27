from __future__ import annotations

from dataclasses import replace

import numpy as np

from scripts_ale_final.showcase_analysis_utils import (
    ShowcaseConfig,
    generate_window,
    showcase_truth_by_phase,
)


def _corr(x: np.ndarray, i: int, j: int) -> float:
    return float(np.corrcoef(x[:, i], x[:, j])[0, 1])


def _corr_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_fixed_phase3_preserves_pairwise_star_root():
    """Regression for the old phase-3 overwrite bug Ale found."""
    config = ShowcaseConfig(n_per_time=640)
    rng = np.random.default_rng(11)
    x_phase3 = generate_window(30, rng, config=config, variant="fixed_phase3")

    star_edges = [_corr(x_phase3, 0, leaf) for leaf in (1, 2, 3, 4)]
    assert min(star_edges) > 0.45


def test_contrast_harder_phase3_keeps_star_and_adds_pairwise_matched_triplets():
    config = replace(
        ShowcaseConfig(n_per_time=900),
        pair_leaves=(1, 2, 3),
        pair_rho=0.55,
        phase3_mode="multiplicative_triplets",
        triplet_blocks=((4, 5, 6), (7, 8, 9)),
        multiplicative_noise_std=0.10,
        tail_theta=3.5,
    )
    rng = np.random.default_rng(17)
    x_phase3 = generate_window(30, rng, config=config, variant="multiplicative_triplets")

    star_edges = [_corr(x_phase3, 0, leaf) for leaf in config.pair_leaves]
    triplet_pair_edges = [
        abs(_corr(x_phase3, i, j))
        for block in config.triplet_blocks
        for i, j in ((block[0], block[1]), (block[0], block[2]), (block[1], block[2]))
    ]
    product_signal = abs(_corr_vec(x_phase3[:, 4] * x_phase3[:, 5], x_phase3[:, 6]))

    assert min(star_edges) > 0.35
    assert max(triplet_pair_edges) < 0.15
    assert product_signal > 0.35


def test_contrast_harder_has_explicit_oracle_ground_truth():
    config = replace(
        ShowcaseConfig(n_per_time=128),
        pair_leaves=(1, 2, 3),
        pair_rho=0.55,
        phase3_mode="multiplicative_triplets",
        triplet_blocks=((4, 5, 6), (7, 8, 9)),
        multiplicative_noise_std=0.10,
        tail_theta=3.5,
    )
    truth = showcase_truth_by_phase(config, "multiplicative_triplets")

    pair_expected = -0.5 * len(config.pair_leaves) * np.log(1.0 - config.pair_rho ** 2)
    tail_lambda_expected = 2.0 ** (-1.0 / config.tail_theta)

    assert truth["independent"]["truth_tc_total"] == 0.0
    assert np.isclose(truth["pairwise-block"]["truth_tc_total"], pair_expected)
    assert truth["pairwise+higher-order"]["truth_tc_total"] > truth["pairwise-block"]["truth_tc_total"]
    assert truth["pairwise+higher-order"]["truth_tc_higher_oracle"] > 1.0
    assert truth["pairwise+higher-order"]["truth_pair_mi56"] > 0.0
    assert truth["tail-block"]["truth_tc_total"] > 1.0
    assert np.isclose(truth["tail-block"]["truth_tail_lambda_lower"], tail_lambda_expected)
