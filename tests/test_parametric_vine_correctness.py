from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import norm

from dvc_package.core.objects import cop_par_obj
from dvc_package.core.param_copula import copulaccdf
from dvc_package.core.vine_factory import create_vine
from dvc_package.experiments.simulation_benchmarks import _mean_copula_nll


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
