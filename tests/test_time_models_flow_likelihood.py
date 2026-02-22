import numpy as np
import torch

from dvc_package.core.vine_factory import create_vine
from dvc_package.time.models import (
    DynamicEntropyEstimator,
    TimeBandwidthFlow,
    TimeDependentVine,
    create_time_dependent_vine,
)


def _make_time_data(T: int, N: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((T, N, d), dtype=np.float32)
    for t in range(T):
        rho = 0.2 + 0.6 * (t / max(T - 1, 1))
        C = np.eye(d, dtype=np.float64)
        C[0, 1] = C[1, 0] = rho
        if d > 2:
            C[1, 2] = C[2, 1] = 0.25
        out[t] = rng.multivariate_normal(np.zeros(d), C, size=N).astype(np.float32)
    return out


def test_time_dependent_vine_local_likelihood_path_runs():
    T, N, d = 6, 120, 4
    time_data = _make_time_data(T=T, N=N, d=d, seed=7)
    times = np.arange(T, dtype=np.float32)

    base_vine = create_vine("c-vine", d)
    n_edges = max(len(base_vine.ind_vine[0]), 1)
    flow = TimeBandwidthFlow(n_edges=n_edges, hidden_dims=[16], dropout_rate=0.0)
    model = TimeDependentVine(base_vine=base_vine, bandwidth_flow=flow, device="cpu")

    fit_info = model.fit_bandwidth_flow(
        time_data,
        time_points=times,
        val_fraction=0.2,
        n_epochs=10,
        lr=5e-3,
        batch_time_steps=3,
        seed=11,
    )
    assert fit_info["n_edges"] == n_edges
    assert len(fit_info["history"]) == 10

    x_eval = torch.tensor(time_data[2][:40], dtype=torch.float32)
    t_eval = torch.full((x_eval.shape[0],), 2.0, dtype=torch.float32)
    nll = model(x_eval, t_eval)

    assert nll.shape == (x_eval.shape[0],)
    assert torch.isfinite(nll).all()

    bw = model.get_bandwidths_over_time(torch.tensor(times, dtype=torch.float32))
    assert bw.shape == (T, n_edges)
    assert torch.all(bw > 0.0)


def test_factory_uses_level0_edge_count_and_sample_checks_time_length():
    d = 5
    base_vine = create_vine("c-vine", d)
    model = create_time_dependent_vine(base_vine, hidden_dims=[8], device="cpu")

    expected_edges = max(len(base_vine.ind_vine[0]), 1)
    assert model.bandwidth_flow.n_edges == expected_edges

    with np.testing.assert_raises(ValueError):
        model.sample(10, torch.linspace(0.0, 1.0, 9))


def test_dynamic_entropy_estimator_mi_plugin_is_finite_and_nonnegative():
    rng = np.random.default_rng(123)
    n = 400
    x = rng.normal(size=(n, 2))
    y = 0.8 * x + 0.2 * rng.normal(size=(n, 2))
    xy = np.concatenate([x, y], axis=1)

    base_vine = create_vine("c-vine", 4)
    model = create_time_dependent_vine(base_vine, hidden_dims=[8], device="cpu")
    est = DynamicEntropyEstimator(model, n_samples=128, n_time_points=8)

    mi = est._estimate_mi_from_samples(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(xy, dtype=torch.float32),
    )
    assert torch.isfinite(mi)
    assert float(mi) >= 0.0
    assert float(mi) > 0.05
