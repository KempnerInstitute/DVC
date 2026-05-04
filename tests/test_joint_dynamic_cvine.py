import numpy as np
from scipy.stats import kendalltau

from dvc_package.time.joint_dynamic_cvine import JointDynamicCVine


def _make_smooth_sequence(
    n_time_steps: int = 8,
    n_samples: int = 140,
    seed: int = 11,
):
    rng = np.random.default_rng(seed)
    windows = []
    true_rho = []
    for t in range(n_time_steps):
        rho01 = 0.15 + 0.65 * (t / max(n_time_steps - 1, 1))
        rho02 = 0.45
        x0 = rng.normal(size=n_samples)
        x1 = rho01 * x0 + np.sqrt(max(1.0 - rho01 ** 2, 1e-6)) * rng.normal(size=n_samples)
        x2 = rho02 * x0 + np.sqrt(max(1.0 - rho02 ** 2, 1e-6)) * rng.normal(size=n_samples)
        x = np.column_stack([x0, x1, x2]).astype(np.float32)
        x += 1e-3 * rng.normal(size=x.shape).astype(np.float32)
        windows.append(x)
        true_rho.append(rho01)
    true_tau = (2.0 / np.pi) * np.arcsin(np.asarray(true_rho, dtype=np.float64))
    return windows, true_tau


def test_joint_dynamic_cvine_recovers_smooth_edge_trajectory():
    windows, true_tau = _make_smooth_sequence()
    model = JointDynamicCVine(
        families=["gaussian", "ind"],
        order=[0, 1, 2],
        n_basis=4,
        smoothness_penalty=0.5,
        ridge_penalty=1e-4,
        maxiter=40,
    )
    result = model.fit(windows)

    edge_map = result.edge_fit_map()
    edge01 = edge_map[(0, 0, 1)]
    pred_tau = np.asarray(edge01.tau_trajectory, dtype=np.float64)

    assert result.order == [0, 1, 2]
    assert edge01.family == "gaussian"
    assert pred_tau.shape == true_tau.shape
    assert np.corrcoef(pred_tau, true_tau)[0, 1] > 0.9

    eval_nll = model.evaluate(windows)
    assert eval_nll.shape == (len(windows),)
    assert np.all(np.isfinite(eval_nll))
    assert np.all(np.isfinite(result.mean_nlls()))


def test_joint_dynamic_cvine_is_smoother_than_windowwise_tau_control():
    windows, true_tau = _make_smooth_sequence(n_time_steps=7, n_samples=90, seed=19)
    model = JointDynamicCVine(
        families=["gaussian", "ind"],
        order=[0, 1, 2],
        n_basis=4,
        smoothness_penalty=1.0,
        ridge_penalty=1e-4,
        maxiter=35,
    )
    result = model.fit(windows)

    pred_tau = np.asarray(result.edge_fit_map()[(0, 0, 1)].tau_trajectory, dtype=np.float64)
    emp_tau = np.asarray(
        [0.0 if not np.isfinite(tau) else float(tau) for tau, _ in (kendalltau(x[:, 0], x[:, 1]) for x in windows)],
        dtype=np.float64,
    )

    pred_rmse = float(np.sqrt(np.mean((pred_tau - true_tau) ** 2)))
    emp_rmse = float(np.sqrt(np.mean((emp_tau - true_tau) ** 2)))
    pred_roughness = float(np.mean(np.diff(pred_tau, n=2) ** 2))
    emp_roughness = float(np.mean(np.diff(emp_tau, n=2) ** 2))

    assert pred_rmse <= emp_rmse + 0.02
    assert pred_roughness < emp_roughness


def test_joint_dynamic_cvine_frank_family_has_independence_floor():
    rng = np.random.default_rng(123)
    train = [rng.normal(size=(90, 3)).astype(np.float32) for _ in range(6)]
    test = [rng.normal(size=(40, 3)).astype(np.float32) for _ in range(6)]

    model = JointDynamicCVine(
        families=["ind", "gaussian", "frank"],
        order=[0, 1, 2],
        n_basis=3,
        smoothness_penalty=1.0,
        ridge_penalty=1e-4,
        maxiter=35,
    )
    result = model.fit(train)
    nll = result.evaluate(test)
    trunc = result.evaluate_truncated_level0(test)

    assert np.all(np.isfinite(nll))
    assert abs(float(np.mean(nll))) < 0.1
    assert abs(float(np.mean(trunc - nll))) < 0.1
