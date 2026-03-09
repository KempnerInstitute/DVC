import numpy as np
import pytest
import torch

from dvc_package.core.cop_eval import cdf_grid_fun
from dvc_package.core.sampling import vine_copula_sample
from dvc_package.core.vine_factory import create_vine


def _npc_fit_kwargs():
    return dict(
        gen_dict={"param": False, "binning": False, "fitted": False},
        npc_dict={
            "opt_method": "LL1",
            "max_iter_phase1": 1,
            "max_iter_phase2": 1,
            "normal_iters_phase1": 5,
            "normal_iters_phase2": 5,
            "final_normalization_iters": 50,
            "batch_size": 1,
        },
        par_dict={},
        bin_dict={},
    )


@pytest.fixture(scope="module")
def fitted_nonparametric_vine():
    rng = np.random.default_rng(4)
    cov = np.array(
        [
            [1.0, 0.6, 0.25],
            [0.6, 1.0, 0.4],
            [0.25, 0.4, 1.0],
        ],
        dtype=np.float32,
    )
    x = rng.multivariate_normal(np.zeros(3), cov, size=24).astype(np.float32)
    vine = create_vine("c-vine", 3, knots=7)
    vine.fit(x, **_npc_fit_kwargs())
    return vine, x


def test_nonparametric_vine_fit_builds_kernel_edges(fitted_nonparametric_vine):
    vine, _x = fitted_nonparametric_vine
    assert vine.param is False
    assert len(vine.copulas) == 2
    assert [len(level) for level in vine.copulas] == [2, 1]
    assert torch.isfinite(vine.theta).all()
    assert hasattr(vine, "nonparametric_summary")
    assert vine.nonparametric_summary["n_edges"] == 3
    assert vine.nonparametric_summary["max_ccdf_monotone_violation"] <= 1e-5
    for level in vine.copulas:
        for cop in level:
            assert hasattr(cop, "opt_bw")
            assert cop.pd_grid_uv is not None
            assert cop.ccdf_grid is not None
            assert cop.cdf is not None
            assert torch.allclose(cop.cdf, cop.ccdf_grid)
            assert cop.validation["ccdf_monotone_violation"] <= 1e-5
            assert cop.validation["pdf_min"] >= 0.0


def test_nonparametric_vine_logpdf_and_sample_are_finite(fitted_nonparametric_vine):
    vine, x = fitted_nonparametric_vine
    logp = vine.logpdf(torch.tensor(x[:6], dtype=torch.float32))
    assert torch.isfinite(logp).all()

    samps = vine.sample(6)
    assert samps.shape == (6, 3)
    assert np.isfinite(samps).all()

    sample1, u, sample_pdf, sample_pds = vine_copula_sample(vine, 6)
    assert sample1.shape == (6, 3)
    assert u.shape == (6, 3)
    assert np.isfinite(sample1).all()
    assert np.isfinite(u).all()
    assert np.all((u > 0.0) & (u < 1.0))


def test_nonparametric_sampler_uniforms_are_not_collapsed(fitted_nonparametric_vine):
    vine, _x = fitted_nonparametric_vine
    _sample1, u, _sample_pdf, _sample_pds = vine_copula_sample(vine, 512)
    assert np.isfinite(u).all()
    assert np.all((u > 0.0) & (u < 1.0))
    assert np.all(np.abs(np.mean(u, axis=0) - 0.5) < 0.15)
    assert np.all((np.std(u, axis=0) > 0.15) & (np.std(u, axis=0) < 0.40))


def test_conditional_grid_for_independence_matches_second_axis_cdf():
    pdf = torch.ones((4, 4, 1), dtype=torch.float32)
    steps = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    cond_grid = cdf_grid_fun(pdf, torch.empty(0), steps, steps, 1)[:, :, 0]
    expected = torch.cumsum(steps, dim=0) / torch.sum(steps)
    expected = expected.unsqueeze(0).repeat(4, 1)
    assert torch.allclose(cond_grid, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("vine_type", "kwargs"),
    [
        ("d-vine", {"variable_order": [0, 2, 1]}),
        ("r-vine", {"r_matrix": np.array([[1, 1, 1], [0, 2, 2], [0, 0, 3]], dtype=np.int32)}),
    ],
)
def test_nonparametric_fit_supports_generic_vines(vine_type, kwargs):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3)).astype(np.float32)
    vine = create_vine(vine_type, 3, knots=7, **kwargs)
    vine.fit(x, **_npc_fit_kwargs())

    logp = vine.logpdf(torch.tensor(x[:6], dtype=torch.float32))
    assert torch.isfinite(logp).all()
    assert len(vine.copulas) == 2
    assert hasattr(vine, "_internal_ind_vine")
    assert len(vine._internal_ind_vine) == 2
    assert len(vine.flip_flag) == 2

    samples = vine.sample(24)
    assert samples.shape == (24, 3)
    assert np.isfinite(samples).all()
    sample_logp = vine.logpdf(torch.tensor(samples[:8], dtype=torch.float32))
    assert torch.isfinite(sample_logp).all()
    assert np.all(np.std(samples, axis=0) > 1e-3)
    _, u, _, _ = vine_copula_sample(vine, 128)
    assert np.isfinite(u).all()
    assert np.all((u > 0.0) & (u < 1.0))
    assert np.all(np.abs(np.mean(u, axis=0) - 0.5) < 0.2)
