from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm

from dvc_package.core.objects import cop_par_obj
from dvc_package.core.param_copula import copulaccdf
from dvc_package.core.vine_factory import create_vine
from dvc_package.experiments.simulation_benchmarks import (
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
    _make_levelwise_cvine,
    _mean_copula_nll,
)


def test_gaussian_parametric_h_function_removes_conditioning_signal():
    rng = np.random.default_rng(42)
    rho = 0.55
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=20_000)
    uv = torch.tensor(norm.cdf(z), dtype=torch.float32)

    h_v_given_u = copulaccdf(cop_par_obj("gaussian", rho), uv)
    h_u_given_v = copulaccdf(cop_par_obj("gaussian", rho), uv[:, [1, 0]])

    corr_v_given_u_with_u = float(torch.corrcoef(torch.stack([h_v_given_u, uv[:, 0]]))[0, 1])
    corr_u_given_v_with_v = float(torch.corrcoef(torch.stack([h_u_given_v, uv[:, 1]]))[0, 1])

    assert abs(corr_v_given_u_with_u) < 0.03
    assert abs(corr_u_given_v_with_v) < 0.03


def test_parametric_dvine_matches_gaussian_ar1_ground_truth_tc():
    rng = np.random.default_rng(123)
    d = 4
    rho = 0.5
    n_train = 3_000
    n_test = 3_000
    corr = np.asarray([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    truth_tc = -0.5 * math.log(float(np.linalg.det(corr)))

    x_train = rng.multivariate_normal(np.zeros(d), corr, size=n_train).astype(np.float32)
    x_test = rng.multivariate_normal(np.zeros(d), corr, size=n_test).astype(np.float32)

    vine = create_vine("d-vine", d, families=["gaussian", "independence"], variable_order=list(range(d)))
    vine.fit(
        x_train,
        gen_dict={"param": True, "binning": False, "fitted": True},
        npc_dict={},
        par_dict={"param_families": ["gaussian", "independence"], "seed": 123},
        bin_dict={},
    )

    estimated_tc = -float(_mean_copula_nll(vine, x_test))
    selected_families = [[cop.family for cop in level] for level in vine.copulas]

    assert abs(estimated_tc - truth_tc) < 0.08
    assert selected_families[0] == ["gaussian", "gaussian", "gaussian"]
    assert selected_families[1] == ["independence", "independence"]
    assert selected_families[2] == ["independence"]


def test_parametric_cvine_recovers_non_gaussian_higher_tree_likelihood_gain():
    """A correctly specified non-Gaussian C-vine should beat its 1-truncated fit."""
    rng = np.random.default_rng(321)
    generator = _make_levelwise_cvine(
        3,
        order=[0, 1, 2],
        level_families=["student", "clayton"],
        level_thetas=[(0.55, 4.0), 2.0],
    )
    x = generator.sample(3_500).astype(np.float32)
    idx = rng.permutation(x.shape[0])
    tr = x[idx[:2_500]]
    te = x[idx[2_500:]]
    families = ["independence", "gaussian", "student", "clayton", "gumbel", "frank"]

    vine = _fit_parametric_vine(tr, families=families, optimize_structure=False, seed=321)
    trunc_vine = _fit_truncated_cvine_level0(tr, families=families, order=[0, 1, 2])

    full_nll = float(_mean_copula_nll(vine, te))
    trunc_nll = float(_mean_copula_nll(trunc_vine, te))
    selected_families = [[cop.family for cop in level] for level in vine.copulas]

    assert np.isfinite(full_nll)
    assert np.isfinite(trunc_nll)
    # The exact held-out gap varies with dependency versions and finite-sample
    # family selection, but a correctly specified full vine should improve
    # materially over its matched first-tree truncation.
    assert trunc_nll - full_nll > 0.25
    assert len(selected_families[1]) == 1
    assert selected_families[1][0] != "ind"
