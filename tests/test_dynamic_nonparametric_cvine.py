import numpy as np

import torch

from dvc_package.core.sampling import vine_copula_sample
from dvc_package.time.nonparametric_dynamic_cvine import (
    JointDynamicNonparametricCVine,
    WindowedNonparametricCVine,
)


def _toy_windows(seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.4, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ],
        dtype=np.float32,
    )
    out = []
    for rho in [0.2, 0.5, 0.8]:
        cov = base.copy()
        cov[0, 1] = cov[1, 0] = rho
        out.append(rng.multivariate_normal(np.zeros(3), cov, size=20).astype(np.float32))
    return out


def test_windowed_and_joint_dynamic_nonparametric_cvine_fit_and_evaluate():
    windows = _toy_windows()

    windowed = WindowedNonparametricCVine(
        knots=7,
        npc_dict={
            "opt_method": "LL1",
            "max_iter_phase1": 1,
            "max_iter_phase2": 1,
            "normal_iters_phase1": 5,
            "normal_iters_phase2": 5,
            "final_normalization_iters": 50,
            "batch_size": 1,
        },
    )
    w_result = windowed.fit(windows)
    w_eval = w_result.evaluate(windows)
    assert np.isfinite(w_eval).all()
    assert len(w_result.vines_by_time) == len(windows)

    coupled_windowed = WindowedNonparametricCVine(
        knots=7,
        temporal_smoothing_bandwidth=0.25,
        temporal_smoothing_normalization_iters=20,
        npc_dict={
            "opt_method": "LL1",
            "max_iter_phase1": 1,
            "max_iter_phase2": 1,
            "normal_iters_phase1": 5,
            "normal_iters_phase2": 5,
            "final_normalization_iters": 20,
            "batch_size": 1,
        },
    )
    cw_result = coupled_windowed.fit(windows)
    cw_eval = cw_result.evaluate(windows)
    assert np.isfinite(cw_eval).all()
    assert cw_result.config["temporal_smoothing_bandwidth"] == 0.25
    assert any(
        ((getattr(cop, "validation", {}) or {}).get("dynamic_smoothing") == "temporal_log_density")
        for vine in cw_result.vines_by_time
        for level in vine.copulas
        for cop in level
        if getattr(cop, "family", "kercop") != "ind"
    )

    joint = JointDynamicNonparametricCVine(
        knots=7,
        trajectory_type="basis",
        trajectory_kwargs={"n_basis": 2},
        n_epochs=2,
        lr=5e-2,
        smoothness_penalty=5e-3,
        batch_size=1,
        normalization_iters=5,
        final_normalization_iters=50,
        density_smoothing_bandwidth=0.20,
    )
    j_result = joint.fit(windows)
    j_eval = j_result.evaluate(windows)
    assert np.isfinite(j_eval).all()
    assert len(j_result.edge_fits) == 3
    assert j_result.order == w_result.order
    assert j_result.config["density_smoothing_bandwidth"] == 0.20
    assert np.max(j_eval) - np.min(j_eval) > 1e-4
    statuses = {edge_fit.status for edge_fit in j_result.edge_fits}
    assert statuses & {"optimized", "target_bandwidth_fallback", "warm_start_fallback"}
    variation = []
    for edge_fit in j_result.edge_fits:
        bw = np.asarray(edge_fit.bandwidth_trajectory, dtype=np.float64)
        assert np.isfinite(bw).all()
        variation.append(float(np.max(bw) - np.min(bw)))
    assert max(variation) > 1e-3


def test_joint_dynamic_nonparametric_cvine_handles_unequal_window_sizes():
    windows = _toy_windows(seed=5)
    windows = [windows[0][:12], windows[1], windows[2][:28]]
    joint = JointDynamicNonparametricCVine(
        knots=7,
        trajectory_type="basis",
        trajectory_kwargs={"n_basis": 2},
        n_epochs=2,
        lr=5e-2,
        smoothness_penalty=5e-3,
        batch_size=1,
        normalization_iters=5,
        final_normalization_iters=50,
    )
    result = joint.fit(windows)
    eval_nll = result.evaluate(windows)
    assert np.isfinite(eval_nll).all()
    for edge_fit in result.edge_fits:
        bw = np.asarray(edge_fit.bandwidth_trajectory, dtype=np.float64)
        target_bw = np.asarray(edge_fit.target_bandwidth_trajectory, dtype=np.float64)
        assert np.isfinite(bw).all()
        assert np.isfinite(target_bw).all()


def test_dynamic_nonparametric_supports_generic_vine_structures_and_sampling():
    windows = _toy_windows(seed=9)

    d_model = JointDynamicNonparametricCVine(
        vine_type="d-vine",
        order=[0, 2, 1],
        knots=7,
        trajectory_type="basis",
        trajectory_kwargs={"n_basis": 2},
        n_epochs=2,
        lr=5e-2,
        smoothness_penalty=5e-3,
        batch_size=1,
        normalization_iters=5,
        final_normalization_iters=50,
    )
    d_result = d_model.fit(windows)
    assert d_result.config["vine_family"] == "d-vine"
    assert all(vine.vine_family == "d-vine" for vine in d_result.vines_by_time)

    vine0 = d_result.vines_by_time[0]
    samples = vine0.sample(24)
    assert samples.shape == (24, 3)
    assert np.isfinite(samples).all()
    logp = vine0.logpdf(torch.tensor(samples[:8], dtype=torch.float32))
    assert torch.isfinite(logp).all()
    _, u, _, _ = vine_copula_sample(vine0, 128)
    assert np.isfinite(u).all()
    assert np.all((u > 0.0) & (u < 1.0))

    r_model = JointDynamicNonparametricCVine(
        vine_type="r-vine",
        vine_kwargs={"r_matrix": np.array([[1, 1, 1], [0, 2, 2], [0, 0, 3]], dtype=np.int32)},
        knots=7,
        trajectory_type="basis",
        trajectory_kwargs={"n_basis": 2},
        n_epochs=2,
        lr=5e-2,
        smoothness_penalty=5e-3,
        batch_size=1,
        normalization_iters=5,
        final_normalization_iters=50,
    )
    r_result = r_model.fit(windows)
    assert r_result.config["vine_family"] == "r-vine"
    assert all(vine.vine_family == "r-vine" for vine in r_result.vines_by_time)
    r_vine0 = r_result.vines_by_time[0]
    r_samples = r_vine0.sample(24)
    assert r_samples.shape == (24, 3)
    assert np.isfinite(r_samples).all()


def test_dynamic_nonparametric_can_select_best_vine_type():
    windows = _toy_windows(seed=21)

    model = WindowedNonparametricCVine(
        vine_type="auto",
        knots=7,
        npc_dict={
            "opt_method": "LL1",
            "max_iter_phase1": 1,
            "max_iter_phase2": 1,
            "normal_iters_phase1": 5,
            "normal_iters_phase2": 5,
            "final_normalization_iters": 50,
            "batch_size": 1,
        },
        optimize_structure=False,
        selection_criterion="aic",
    )
    result = model.fit(windows)
    selected = result.config["selected_vine_type"]
    assert selected in {"c-vine", "d-vine", "r-vine"}
    assert all(vine.vine_family == selected for vine in result.vines_by_time)
