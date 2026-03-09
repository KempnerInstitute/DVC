import numpy as np
import torch

from dvc_package.core.vine_factory import create_vine
from dvc_package.time.models import TimeDependentVine, create_time_dependent_vine
from dvc_package.time.trajectory_models import BasisTrajectory, StateSpaceTrajectory


def _make_time_data(T: int, N: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((T, N, d), dtype=np.float32)
    for t in range(T):
        rho = 0.2 + 0.55 * (t / max(T - 1, 1))
        C = np.eye(d, dtype=np.float64)
        C[0, 1] = C[1, 0] = rho
        if d > 2:
            C[1, 2] = C[2, 1] = 0.2
        out[t] = rng.multivariate_normal(np.zeros(d), C, size=N).astype(np.float32)
    return out


def test_basis_trajectory_bounded_output_shape():
    model = BasisTrajectory(output_dim=3, n_basis=4, constraint="bounded", min_value=0.01, max_value=2.0)
    model.set_time_range(0.0, 9.0)
    t = torch.linspace(0.0, 9.0, 5)
    y = model(t)
    assert y.shape == (5, 3)
    assert torch.all(y >= 0.01)
    assert torch.all(y <= 2.0)


def test_state_space_trajectory_regularization_and_interpolation():
    model = StateSpaceTrajectory(
        output_dim=2,
        latent_dim=2,
        n_steps=6,
        transition_penalty=0.1,
        constraint="bounded",
        min_value=0.01,
        max_value=2.0,
    )
    model.set_reference_time_grid(torch.arange(6, dtype=torch.float32))
    y = model(torch.tensor([0.0, 2.5, 5.0], dtype=torch.float32))
    reg = model.regularization_loss()
    assert y.shape == (3, 2)
    assert torch.isfinite(y).all()
    assert torch.isfinite(reg)
    assert float(reg.detach().cpu()) >= 0.0


def test_time_dependent_vine_trains_with_state_space_bandwidth_trajectory():
    T, N, d = 5, 100, 4
    time_data = _make_time_data(T=T, N=N, d=d, seed=13)
    times = np.arange(T, dtype=np.float32)

    base_vine = create_vine("c-vine", d)
    model = create_time_dependent_vine(
        base_vine,
        trajectory_type="state_space",
        trajectory_kwargs={"latent_dim": 2, "transition_penalty": 1e-2, "n_steps": T},
        device="cpu",
    )

    assert isinstance(model.bandwidth_flow, StateSpaceTrajectory)
    fit_info = model.fit_bandwidth_flow(
        time_data,
        time_points=times,
        val_fraction=0.2,
        n_epochs=5,
        lr=5e-3,
        batch_time_steps=3,
        seed=17,
    )

    assert fit_info["n_edges"] == model.bandwidth_flow.output_dim
    bw = model.get_bandwidths_over_time(torch.tensor(times, dtype=torch.float32))
    assert bw.shape == (T, model.bandwidth_flow.output_dim)
    assert torch.all(torch.isfinite(bw))

    x_eval = torch.tensor(time_data[1][:20], dtype=torch.float32)
    t_eval = torch.full((x_eval.shape[0],), 1.0, dtype=torch.float32)
    nll = model(x_eval, t_eval)
    assert torch.isfinite(nll).all()
