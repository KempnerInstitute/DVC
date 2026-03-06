import numpy as np

from dvc_package.core.objects import cop_par_obj
from dvc_package.time.regularized_cvine import (
    EdgeCandidate,
    RegularizedDynamicCVine,
    select_edge_candidate,
    solve_root_sequence,
)


def _make_hub_window(
    n_samples: int,
    n_variables: int,
    root: int,
    rho: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    root_signal = rng.normal(size=n_samples)
    out = np.zeros((n_samples, n_variables), dtype=np.float32)
    out[:, root] = root_signal
    noise_scale = np.sqrt(max(1.0 - rho ** 2, 1e-6))
    for j in range(n_variables):
        if j == root:
            continue
        out[:, j] = rho * root_signal + noise_scale * rng.normal(size=n_samples)
    out += 1e-3 * rng.normal(size=out.shape).astype(np.float32)
    return out


def _make_hub_switch_sequence(
    n_time_steps: int = 10,
    n_samples: int = 320,
    n_variables: int = 5,
    change_point: int = 5,
    rho: float = 0.85,
    seed: int = 7,
):
    windows = []
    true_roots = []
    for t in range(n_time_steps):
        root = 0 if t < change_point else 1
        true_roots.append(root)
        windows.append(
            _make_hub_window(
                n_samples=n_samples,
                n_variables=n_variables,
                root=root,
                rho=rho,
                seed=seed + t,
            )
        )
    return windows, np.asarray(true_roots, dtype=np.int32)


def test_solve_root_sequence_respects_switch_penalty():
    local_costs = np.array(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [0.9, 0.0],
            [1.0, 0.1],
        ],
        dtype=np.float64,
    )

    path_low, obj_low = solve_root_sequence(local_costs, switch_penalty=0.0)
    path_high, obj_high = solve_root_sequence(local_costs, switch_penalty=2.0)

    assert path_low == [0, 0, 1, 1]
    assert path_high in ([0, 0, 0, 0], [1, 1, 1, 1])
    assert obj_low <= obj_high


def test_select_edge_candidate_prefers_sticky_family_and_smoothed_theta():
    prev = cop_par_obj("gaussian", 0.25)
    candidates = [
        EdgeCandidate(family="gaussian", theta=0.45, raw_aic=10.2),
        EdgeCandidate(family="frank", theta=4.0, raw_aic=9.8),
    ]

    selected = select_edge_candidate(
        candidates,
        prev_copula=prev,
        family_switch_penalty=1.0,
        parameter_drift_penalty=0.5,
        parameter_smoothing=0.5,
    )

    assert selected.family == "gaussian"
    assert abs(float(selected.theta) - 0.35) < 1e-8
    assert not selected.family_switched
    assert abs(selected.parameter_distance - 0.10) < 1e-8


def test_regularized_dynamic_cvine_recovers_hub_switch():
    windows, true_roots = _make_hub_switch_sequence()
    change_point = int(np.where(np.diff(true_roots) != 0)[0][0] + 1)

    model = RegularizedDynamicCVine(
        families=["gaussian", "ind"],
        root_switch_penalty=0.25,
        family_switch_penalty=0.5,
        parameter_drift_penalty=1.0,
        parameter_smoothing=0.35,
        root_score_method="kendall_tau",
    )
    result = model.fit(windows)

    pred = np.asarray(result.root_sequence, dtype=np.int32)
    assert result.root_local_costs.shape == (len(windows), windows[0].shape[1])
    assert len(result.window_fits) == len(windows)
    assert np.mean(pred[:change_point] == 0) >= 0.8
    assert np.mean(pred[change_point:] == 1) >= 0.8

    switch_candidates = np.where(pred == 1)[0]
    assert switch_candidates.size > 0
    est_change = int(switch_candidates[0])
    assert abs(est_change - change_point) <= 1

    eval_nll = model.evaluate(windows)
    assert eval_nll.shape == (len(windows),)
    assert np.all(np.isfinite(eval_nll))
    assert result.total_family_switches() >= 0
    assert result.total_parameter_drift() >= 0.0
