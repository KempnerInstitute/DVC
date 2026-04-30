import numpy as np

from dvc_package.time import SwitchingDynamicCVine


def _make_on_off_gaussian_sequence(n_time_steps=8, n_samples=140, seed=17):
    rng = np.random.default_rng(seed)
    windows = []
    labels = []
    for t in range(n_time_steps):
        active = t >= n_time_steps // 2
        rho = 0.75 if active else 0.0
        x0 = rng.normal(size=n_samples)
        x1 = rho * x0 + np.sqrt(max(1.0 - rho**2, 1e-6)) * rng.normal(size=n_samples)
        windows.append(np.column_stack([x0, x1]).astype(np.float32))
        labels.append(int(active))
    return windows, np.asarray(labels, dtype=np.int32)


def test_switching_dynamic_cvine_detects_on_off_edge_state():
    windows, labels = _make_on_off_gaussian_sequence()
    model = SwitchingDynamicCVine(
        families=["gaussian", "independence"],
        order=[0, 1],
        family_switch_penalty=0.02,
    )
    result = model.fit(windows)

    assert result.order == [0, 1]
    assert result.total_family_switches() <= 2
    assert np.all(np.isfinite(result.mean_nlls()))
    assert np.all(np.isfinite(model.evaluate(windows)))

    theta = np.asarray(
        [0.0 if state.theta is None else float(np.asarray(state.theta).reshape(-1)[0]) for state in result.edge_fits[0].states],
        dtype=np.float64,
    )
    assert np.mean(np.abs(theta[labels == 1])) > np.mean(np.abs(theta[labels == 0])) + 0.35
