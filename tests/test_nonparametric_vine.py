import numpy as np
import pytest
import torch
from scipy.stats import norm

from dvc_package.core.nonparametric_vine import _build_edge_input_pairs, _build_internal_edge_structure
from dvc_package.core.nonparametric_vine import evaluate_nonparametric_edge_h, evaluate_nonparametric_edge_pdf
from dvc_package.core.cop_eval import cdf_grid_fun
from dvc_package.core.sampling import vine_copula_sample
from dvc_package.core.utils_interpolation import interp_regular_nd_grid
from dvc_package.core.utils_prob import kernel_cdf
from dvc_package.core.vine_factory import create_vine
from dvc_package.core.vine_tree import flip_check_all


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


def _npc_fit_kwargs_stronger():
    return dict(
        gen_dict={"param": False, "binning": False, "fitted": False},
        npc_dict={
            "opt_method": "LL1",
            "max_iter_phase1": 3,
            "max_iter_phase2": 4,
            "normal_iters_phase1": 20,
            "normal_iters_phase2": 35,
            "final_normalization_iters": 75,
            "batch_size": 2,
            "validation_fraction": 0.2,
            "min_validation_improvement": 0.01,
        },
        par_dict={},
        bin_dict={},
    )


def _npc_fit_kwargs_with_space(data_space: str):
    kwargs = _npc_fit_kwargs_stronger()
    kwargs["npc_dict"] = dict(kwargs["npc_dict"])
    kwargs["npc_dict"]["data_space"] = data_space
    return kwargs


def _npc_fit_kwargs_null_adjusted():
    kwargs = _npc_fit_kwargs()
    kwargs["npc_dict"] = dict(kwargs["npc_dict"])
    kwargs["npc_dict"]["higher_tree_validation_margin"] = 0.025
    kwargs["npc_dict"]["higher_tree_null_adjusted_validation"] = True
    kwargs["npc_dict"]["higher_tree_null_permutations"] = 2
    kwargs["npc_dict"]["higher_tree_null_adjusted_margin"] = 0.0
    kwargs["npc_dict"]["higher_tree_null_seed"] = 123
    return kwargs


def _sample_clayton_normal_margins(n: int, theta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.gamma(shape=1.0 / float(theta), scale=1.0, size=n)
    e = rng.exponential(scale=1.0, size=(n, 2))
    u = (1.0 + e / v[:, None]) ** (-1.0 / float(theta))
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    return norm.ppf(u).astype(np.float32)


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


def test_conditional_grid_default_integrates_second_axis():
    pdf = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 1.0],
            [4.0, 3.0, 2.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)
    u1_steps = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    u2_steps = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float32)

    cond_grid = cdf_grid_fun(pdf, torch.empty(0), u1_steps, u2_steps, 1)[:, :, 0]
    expected = torch.cumsum(pdf[:, :, 0] * u2_steps.view(1, -1), dim=1)
    expected = expected / expected[:, -1:].clamp_min(1e-12)

    reverse_grid = cdf_grid_fun(pdf, torch.empty(0), u1_steps, u2_steps, 1, axis=0)[:, :, 0]
    expected_reverse = torch.cumsum(pdf[:, :, 0] * u1_steps.view(-1, 1), dim=0)
    expected_reverse = expected_reverse / expected_reverse[-1:, :].clamp_min(1e-12)

    assert torch.allclose(cond_grid, expected, atol=1e-6)
    assert torch.allclose(reverse_grid, expected_reverse, atol=1e-6)


def test_regular_grid_interpolation_keeps_point_axis_order():
    grid_vals_axis0 = torch.arange(5, dtype=torch.float32).view(5, 1).repeat(1, 5)
    grid_vals_axis1 = torch.arange(5, dtype=torch.float32).view(1, 5).repeat(5, 1)
    points = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    grid_min = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    grid_max = torch.tensor([1.0, 1.0], dtype=torch.float32)

    out_axis0 = interp_regular_nd_grid(points, grid_min, grid_max, grid_vals_axis0)
    out_axis1 = interp_regular_nd_grid(points, grid_min, grid_max, grid_vals_axis1)

    assert torch.allclose(out_axis0, torch.tensor([0.0, 4.0, 2.0, 2.0]))
    assert torch.allclose(out_axis1, torch.tensor([2.0, 2.0, 0.0, 4.0]))


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


def test_nonparametric_bivariate_independence_prefers_independence_edge():
    rng = np.random.default_rng(22)
    torch.manual_seed(22)
    x_train = rng.normal(size=(160, 2)).astype(np.float32)
    x_test = rng.normal(size=(160, 2)).astype(np.float32)

    vine = create_vine("c-vine", 2, knots=7)
    vine.fit(x_train, **_npc_fit_kwargs_stronger())

    edge = vine.copulas[0][0]
    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())

    assert edge.family in {"ind", "kercop"}
    assert edge.validation["selected_model"] in {"independence", "kernel"}
    assert "independence" in edge.validation["candidate_validation_nlls"]
    assert edge.validation["candidate_validation_nlls"]["independence"] == 0.0
    assert edge.validation["validation_nll"] > -0.05
    assert mean_logp > -0.35


def test_nonparametric_bivariate_gaussian_dependency_keeps_kernel_edge():
    rng = np.random.default_rng(23)
    torch.manual_seed(23)
    cov = np.array([[1.0, 0.7], [0.7, 1.0]], dtype=np.float32)
    x_train = rng.multivariate_normal(np.zeros(2), cov, size=192).astype(np.float32)
    x_test = rng.multivariate_normal(np.zeros(2), cov, size=192).astype(np.float32)

    vine = create_vine("c-vine", 2, knots=7)
    vine.fit(x_train, **_npc_fit_kwargs_stronger())

    edge = vine.copulas[0][0]
    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())

    assert edge.family == "kercop"
    assert edge.validation["selected_model"] == "kernel"
    assert edge.validation["validation_nll"] < -0.01
    assert mean_logp > -0.1


def test_nonparametric_bivariate_gaussian_prefers_archive_x_space_over_s_space():
    rng = np.random.default_rng(123)
    cov = np.array([[1.0, 0.65], [0.65, 1.0]], dtype=np.float32)
    x_train = rng.multivariate_normal(np.zeros(2), cov, size=224).astype(np.float32)
    x_test = rng.multivariate_normal(np.zeros(2), cov, size=224).astype(np.float32)

    torch.manual_seed(0)
    vine_s = create_vine("c-vine", 2, knots=7)
    vine_s.fit(x_train, **_npc_fit_kwargs_with_space("s"))
    logp_s = float(torch.mean(vine_s.logpdf(torch.tensor(x_test, dtype=torch.float32))).item())

    torch.manual_seed(0)
    vine_x = create_vine("c-vine", 2, knots=7)
    vine_x.fit(x_train, **_npc_fit_kwargs_with_space("x"))
    logp_x = float(torch.mean(vine_x.logpdf(torch.tensor(x_test, dtype=torch.float32))).item())

    edge_s = vine_s.copulas[0][0]
    edge_x = vine_x.copulas[0][0]

    assert edge_x.validation["data_space"] == "x"
    assert edge_s.validation["data_space"] == "s"
    assert logp_x > 0.10
    assert logp_x > logp_s + 0.10


def test_nonparametric_bivariate_clayton_kernel_only_has_positive_heldout_gain():
    torch.manual_seed(31)
    x_train = _sample_clayton_normal_margins(192, theta=1.5, seed=31)
    x_test = _sample_clayton_normal_margins(192, theta=1.5, seed=32)

    fit_kwargs = _npc_fit_kwargs_stronger()
    fit_kwargs["npc_dict"] = dict(fit_kwargs["npc_dict"])
    fit_kwargs["npc_dict"]["edge_estimators"] = ["kde"]
    fit_kwargs["npc_dict"]["allow_independence_fallback"] = False

    vine = create_vine("c-vine", 2, knots=7)
    vine.fit(x_train, **fit_kwargs)

    edge = vine.copulas[0][0]
    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())

    assert edge.family == "kercop"
    assert edge.validation["selected_model"] == "kernel"
    assert mean_logp > -0.1


def test_nonparametric_bivariate_clayton_prefers_archive_x_space_over_s_space():
    x_train = _sample_clayton_normal_margins(224, theta=1.5, seed=77)
    x_test = _sample_clayton_normal_margins(224, theta=1.5, seed=78)

    torch.manual_seed(0)
    vine_s = create_vine("c-vine", 2, knots=7)
    vine_s.fit(x_train, **_npc_fit_kwargs_with_space("s"))
    logp_s = float(torch.mean(vine_s.logpdf(torch.tensor(x_test, dtype=torch.float32))).item())

    torch.manual_seed(0)
    vine_x = create_vine("c-vine", 2, knots=7)
    vine_x.fit(x_train, **_npc_fit_kwargs_with_space("x"))
    logp_x = float(torch.mean(vine_x.logpdf(torch.tensor(x_test, dtype=torch.float32))).item())

    edge_x = vine_x.copulas[0][0]
    assert edge_x.validation["data_space"] == "x"
    assert logp_x > 0.10
    assert logp_x > logp_s + 0.10
    assert edge_x.validation["selected_bandwidth_label"] is not None


def test_nonparametric_kernel_only_fit_no_longer_crashes_when_kernel_loses():
    rng = np.random.default_rng(41)
    torch.manual_seed(41)
    x_train = rng.normal(size=(160, 2)).astype(np.float32)

    fit_kwargs = _npc_fit_kwargs_stronger()
    fit_kwargs["npc_dict"] = dict(fit_kwargs["npc_dict"])
    fit_kwargs["npc_dict"]["edge_estimators"] = ["kde"]
    fit_kwargs["npc_dict"]["allow_independence_fallback"] = False

    vine = create_vine("c-vine", 2, knots=7)
    vine.fit(x_train, **fit_kwargs)

    edge = vine.copulas[0][0]
    assert edge.family == "kercop"
    assert edge.validation["selected_model"] == "kernel"
    assert edge.validation["selected_bandwidth_label"] is not None


def test_nonparametric_independence_margin_can_flip_weak_gaussian_to_kernel():
    rng = np.random.default_rng(1)
    x_train = rng.multivariate_normal(
        np.zeros(2),
        np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32),
        size=100,
    ).astype(np.float32)

    default_kwargs = _npc_fit_kwargs_stronger()
    default_kwargs["npc_dict"] = dict(default_kwargs["npc_dict"])
    default_kwargs["npc_dict"]["min_validation_improvement"] = 0.02
    default_kwargs["npc_dict"]["data_space"] = "s"
    margin_kwargs = _npc_fit_kwargs_stronger()
    margin_kwargs["npc_dict"] = dict(margin_kwargs["npc_dict"])
    margin_kwargs["npc_dict"]["min_validation_improvement"] = 0.02
    margin_kwargs["npc_dict"]["independence_margin"] = 0.01
    margin_kwargs["npc_dict"]["prefer_kernel_on_tie"] = True
    margin_kwargs["npc_dict"]["data_space"] = "s"

    torch.manual_seed(7)
    vine_default = create_vine("c-vine", 2, knots=7)
    vine_default.fit(x_train, **default_kwargs)
    edge_default = vine_default.copulas[0][0]

    torch.manual_seed(7)
    vine_margin = create_vine("c-vine", 2, knots=7)
    vine_margin.fit(x_train, **margin_kwargs)
    edge_margin = vine_margin.copulas[0][0]

    assert edge_default.family == "ind"
    assert edge_default.validation["selected_model"] == "independence"
    assert edge_margin.family == "kercop"
    assert edge_margin.validation["selected_model"] == "kernel"
    assert edge_margin.validation["effective_kernel_threshold"] < edge_default.validation["effective_kernel_threshold"]


def test_nonparametric_default_depth_validation_applies_to_3d_vines():
    rng = np.random.default_rng(55)
    cov = np.array(
        [
            [1.0, 0.7, 0.15],
            [0.7, 1.0, 0.7],
            [0.15, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    x = rng.multivariate_normal(np.zeros(3), cov, size=192).astype(np.float32)

    vine = create_vine("d-vine", 3, knots=7, variable_order=[0, 1, 2])
    vine.fit(x, **_npc_fit_kwargs_stronger())

    summary = getattr(vine, "nonparametric_summary", {})
    assert summary.get("depth_selection_enabled") is True
    assert int(summary.get("selected_depth")) in {1, 2}
    assert float(summary.get("higher_tree_validation_margin")) == pytest.approx(0.05)


def test_nonparametric_default_higher_tree_safeguard_keeps_3d_null_near_zero():
    rng = np.random.default_rng(91)
    torch.manual_seed(91)
    x_train = rng.uniform(1e-6, 1.0 - 1e-6, size=(240, 3)).astype(np.float32)
    x_test = rng.uniform(1e-6, 1.0 - 1e-6, size=(120, 3)).astype(np.float32)

    vine = create_vine("d-vine", 3, knots=7, variable_order=[0, 1, 2], margin_types=["uniform"] * 3)
    vine.fit(x_train, **_npc_fit_kwargs())

    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())
    summary = getattr(vine, "nonparametric_summary", {})

    assert int(summary.get("selected_depth")) == 1
    assert abs(mean_logp) < 0.15


def test_nonparametric_default_higher_tree_safeguard_keeps_10d_null_at_depth1():
    rng = np.random.default_rng(193)
    torch.manual_seed(193)
    x_train = rng.uniform(1e-6, 1.0 - 1e-6, size=(300, 10)).astype(np.float32)
    x_test = rng.uniform(1e-6, 1.0 - 1e-6, size=(150, 10)).astype(np.float32)

    vine = create_vine("d-vine", 10, knots=7, variable_order=list(range(10)), margin_types=["uniform"] * 10)
    vine.fit(x_train, **_npc_fit_kwargs())

    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())
    summary = getattr(vine, "nonparametric_summary", {})

    assert int(summary.get("selected_depth")) == 1
    assert abs(mean_logp) < 0.25


def test_nonparametric_higher_tree_guard_preserves_dependent_chain_signal():
    rng = np.random.default_rng(92)
    torch.manual_seed(92)
    cov = np.array(
        [
            [1.0, 0.7, 0.2],
            [0.7, 1.0, 0.7],
            [0.2, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    x_train = rng.multivariate_normal(np.zeros(3), cov, size=240).astype(np.float32)
    x_test = rng.multivariate_normal(np.zeros(3), cov, size=120).astype(np.float32)

    vine = create_vine("d-vine", 3, knots=7, variable_order=[0, 1, 2])
    vine.fit(x_train, **_npc_fit_kwargs_stronger())

    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())
    summary = getattr(vine, "nonparametric_summary", {})

    assert int(summary.get("selected_depth")) >= 1
    assert mean_logp > 0.15


def test_nonparametric_optional_null_adjusted_guard_rejects_tree2_on_null_chain():
    rng = np.random.default_rng(187)
    torch.manual_seed(187)
    x_train = rng.uniform(1e-6, 1.0 - 1e-6, size=(240, 3)).astype(np.float32)
    x_test = rng.uniform(1e-6, 1.0 - 1e-6, size=(120, 3)).astype(np.float32)

    vine = create_vine("d-vine", 3, knots=7, variable_order=[0, 1, 2], margin_types=["uniform"] * 3)
    vine.fit(x_train, **_npc_fit_kwargs_null_adjusted())

    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())
    summary = getattr(vine, "nonparametric_summary", {})

    assert summary.get("higher_tree_null_adjusted_validation") is True
    assert int(summary.get("selected_depth")) == 1
    assert abs(mean_logp) < 0.2


def test_nonparametric_optional_null_adjusted_guard_keeps_dependent_depth1_signal():
    rng = np.random.default_rng(188)
    torch.manual_seed(188)
    cov = np.array(
        [
            [1.0, 0.7, 0.2],
            [0.7, 1.0, 0.7],
            [0.2, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    x_train = rng.multivariate_normal(np.zeros(3), cov, size=240).astype(np.float32)
    x_test = rng.multivariate_normal(np.zeros(3), cov, size=120).astype(np.float32)

    vine = create_vine("d-vine", 3, knots=7, variable_order=[0, 1, 2])
    vine.fit(x_train, **_npc_fit_kwargs_null_adjusted())

    logp = vine.logpdf(torch.tensor(x_test, dtype=torch.float32))
    mean_logp = float(torch.mean(logp).item())
    summary = getattr(vine, "nonparametric_summary", {})

    assert summary.get("higher_tree_null_adjusted_validation") is True
    assert int(summary.get("selected_depth")) >= 1
    assert mean_logp > 0.1


def test_nonparametric_logpdf_matches_fit_style_h_propagation():
    rng = np.random.default_rng(77)
    cov = np.array(
        [
            [1.0, 0.65, 0.25],
            [0.65, 1.0, 0.65],
            [0.25, 0.65, 1.0],
        ],
        dtype=np.float32,
    )
    x_train = rng.multivariate_normal(np.zeros(3), cov, size=192).astype(np.float32)
    x_test = rng.multivariate_normal(np.zeros(3), cov, size=96).astype(np.float32)

    vine = create_vine("d-vine", 3, knots=7, variable_order=[0, 1, 2])
    vine.fit(x_train, **_npc_fit_kwargs_stronger())

    points = torch.tensor(x_test, dtype=torch.float32)
    model_logp = vine.logpdf(points)

    device = points.device
    n, d = points.shape
    edge_refs = getattr(vine, "_internal_ind_vine", None)
    if edge_refs is None:
        edge_refs = _build_internal_edge_structure(vine, d)
        vine._internal_ind_vine = edge_refs

    u_state = torch.zeros((n, d, d), dtype=torch.float32, device=device)
    u_state_flip = torch.zeros_like(u_state)
    grid_u_ex = vine.grid_u.ex.detach().cpu().numpy()
    for i in range(d):
        cdf_query, _mar_s, _mar_p = kernel_cdf(
            np.asarray(vine.margin[i].ker),
            points[:, i].detach().cpu().numpy(),
            grid_u_ex,
        )
        u_state[:, 0, i] = torch.tensor(cdf_query, dtype=torch.float32, device=device)

    manual_logp = torch.zeros(n, dtype=torch.float32, device=device)
    for level in range(d - 1):
        if level >= len(vine.copulas) or not vine.copulas[level]:
            continue
        if level < len(vine.flip_flag):
            flip_flag1 = vine.flip_flag[level]
            ind_edge_rel1 = vine.ind_edge_rel[level]
        else:
            flip_flag1, ind_edge_rel1, _parent_all = flip_check_all(edge_refs, level, False, 1)
        point_u = _build_edge_input_pairs(
            state=u_state,
            state_flip=u_state_flip,
            edge_refs=edge_refs,
            level=level,
            device=device,
        )
        cops_now = vine.copulas[level]
        h_cops_now = getattr(vine, "_np_h_copulas", [])
        h_cops_level = h_cops_now[level] if level < len(h_cops_now) else []
        density_edges_seen = set()
        for j, ind_edge in enumerate(ind_edge_rel1):
            uv = point_u[:, :, ind_edge]
            if int(ind_edge) not in density_edges_seen:
                density_edges_seen.add(int(ind_edge))
                manual_logp = manual_logp + torch.log(
                    evaluate_nonparametric_edge_pdf(cops_now[ind_edge], uv, vine.grid_s).clamp_min(1e-12)
                )
            if level < d - 1:
                cop_h = h_cops_level[j] if j < len(h_cops_level) else cops_now[ind_edge]
                uv_h = uv[:, [1, 0]] if flip_flag1[j] else uv
                hval = evaluate_nonparametric_edge_h(cop_h, uv_h, vine.grid_s)
                if flip_flag1[j]:
                    u_state_flip[:, level + 1, ind_edge] = hval
                else:
                    u_state[:, level + 1, ind_edge] = hval

    assert torch.allclose(model_logp, manual_logp, atol=1e-5, rtol=1e-5)


def test_nonparametric_independence_h_propagates_target_given_root():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(120, 3)).astype(np.float32)
    vine = create_vine("c-vine", 3, knots=7, variable_order=[0, 1, 2])
    kwargs = _npc_fit_kwargs()
    kwargs["npc_dict"] = dict(kwargs["npc_dict"])
    kwargs["npc_dict"].update(
        {
            "allow_independence_fallback": True,
            "minimum_kernel_gain": 999.0,
            "validation_fraction": 0.2,
        }
    )
    vine.fit(x, **kwargs)

    assert vine.flip_flag[0] == [False, False]
    assert [getattr(cop, "family", None) for cop in vine.copulas[0]] == ["ind", "ind"]
    assert torch.allclose(vine.theta[:, 1, 0], vine.theta[:, 0, 1])
    assert torch.allclose(vine.theta[:, 1, 1], vine.theta[:, 0, 2])


def test_flip_check_keeps_both_dvine_conditionals_for_reused_edges():
    vine = create_vine("d-vine", 4, knots=7, variable_order=[0, 1, 2, 3])
    edge_refs = _build_internal_edge_structure(vine, 4)

    flip_flag, ind_edge_rel, parent_all = flip_check_all(edge_refs, 0, False, 1)

    assert ind_edge_rel == [0, 1, 1, 2]
    assert flip_flag == [True, False, True, False]
    assert parent_all == [[1], [1, 2], [2]]
