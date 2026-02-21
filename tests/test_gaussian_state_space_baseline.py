import numpy as np

from dvc_package.baselines.gaussian_state_space import (
    gaussian_copula_state_space_nll_fit_eval,
)


def _make_dynamic_gaussian_time_series(T: int, N: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = []
    for t in range(T):
        rho = 0.15 + 0.7 * (t / max(T - 1, 1))
        C = np.array(
            [
                [1.0, rho, 0.0],
                [rho, 1.0, 0.25],
                [0.0, 0.25, 1.0],
            ],
            dtype=np.float64,
        )
        data.append(rng.multivariate_normal(np.zeros(3), C, size=N).astype(np.float32))
    return data


def test_gaussian_state_space_fit_eval_shapes_and_finiteness():
    T = 8
    N = 140
    time_data = _make_dynamic_gaussian_time_series(T=T, N=N, seed=12)

    x_train = []
    x_test = []
    for t, x_t in enumerate(time_data):
        idx = np.random.default_rng(123 + t).permutation(x_t.shape[0])
        n_train = int(0.8 * x_t.shape[0])
        x_train.append(x_t[idx[:n_train]])
        x_test.append(x_t[idx[n_train:]])

    nll, fit = gaussian_copula_state_space_nll_fit_eval(x_train, x_test)

    assert len(nll) == T
    assert np.isfinite(np.asarray(nll, dtype=np.float64)).all()
    assert fit.process_variance >= 0.0
    assert fit.latent_fisher_z.shape[0] == T
    assert fit.observed_fisher_z.shape[0] == T
    assert len(fit.correlations) == T

    for R in fit.correlations:
        assert R.shape == (3, 3)
        assert np.allclose(np.diag(R), 1.0, atol=1e-6)
        eigvals = np.linalg.eigvalsh(R)
        assert np.min(eigvals) > 0.0
