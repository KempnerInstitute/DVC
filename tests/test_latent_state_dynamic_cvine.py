import numpy as np

from dvc_package.time.latent_state_dynamic_cvine import LatentStateDynamicCVine


def _make_shared_latent_sequence(
    n_time_steps: int = 7,
    n_samples: int = 150,
    seed: int = 31,
):
    rng = np.random.default_rng(seed)
    windows = []
    true_rho = []
    for t in range(n_time_steps):
        rho = 0.10 + 0.65 * (t / max(n_time_steps - 1, 1))
        x0 = rng.normal(size=n_samples)
        x1 = rho * x0 + np.sqrt(max(1.0 - rho ** 2, 1e-6)) * rng.normal(size=n_samples)
        x2 = 0.9 * rho * x0 + np.sqrt(max(1.0 - (0.9 * rho) ** 2, 1e-6)) * rng.normal(size=n_samples)
        x = np.column_stack([x0, x1, x2]).astype(np.float32)
        x += 1e-3 * rng.normal(size=x.shape).astype(np.float32)
        windows.append(x)
        true_rho.append(rho)
    true_tau = (2.0 / np.pi) * np.arcsin(np.asarray(true_rho, dtype=np.float64))
    return windows, true_tau


def test_latent_state_dynamic_cvine_recovers_shared_temporal_trend():
    windows, true_tau = _make_shared_latent_sequence()
    model = LatentStateDynamicCVine(
        families=["gaussian", "ind"],
        order=[0, 1, 2],
        selection_n_basis=4,
        selection_smoothness_penalty=0.5,
        latent_dim=1,
        transition_penalty=1e-2,
        n_epochs=80,
        lr=2e-2,
    )
    result = model.fit(windows)

    edge_map = result.edge_fit_map()
    tau01 = np.asarray(edge_map[(0, 0, 1)].tau_trajectory, dtype=np.float64)
    tau02 = np.asarray(edge_map[(0, 0, 2)].tau_trajectory, dtype=np.float64)

    assert len(result.latent_states) == len(windows)
    assert np.corrcoef(tau01, true_tau)[0, 1] > 0.85
    assert np.corrcoef(tau02, true_tau)[0, 1] > 0.80
    assert np.mean(np.diff(tau01) >= -1e-3) > 0.7

    eval_nll = model.evaluate(windows)
    assert eval_nll.shape == (len(windows),)
    assert np.all(np.isfinite(eval_nll))
