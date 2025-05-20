import numpy as np
from src.DVC.objects import vine_obj_bin, margin_obj


_DEF_RHO = 0.6


def _build_gaussian_cov(d: int, rho: float) -> np.ndarray:
    cov = np.full((d, d), rho)
    np.fill_diagonal(cov, 1.0)
    return cov


def _fit_vine(d: int, n: int, param: bool) -> vine_obj_bin:
    cov = _build_gaussian_cov(d, _DEF_RHO)
    z = np.random.multivariate_normal(np.zeros(d), cov, size=n)

    margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(d)]
    families = ['gaussian'] if param else 'kercop'
    vine = vine_obj_bin('c-vine', families, d, margins, knots=30, method='random')

    gen_dict = {'param': param, 'binning': False, 'fitted': False}
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    vine.fit(z, gen_dict, npc_dict, par_dict, bin_dict)
    return vine


def _check_correlations(samples: np.ndarray, tol: float = 0.1):
    corr = np.corrcoef(samples, rowvar=False)
    d = corr.shape[0]
    for i in range(d):
        assert abs(corr[i, i] - 1.0) < tol
        for j in range(i + 1, d):
            assert abs(corr[i, j] - _DEF_RHO) < tol


def test_parametric_vine_correlation():
    vine = _fit_vine(d=5, n=2500, param=True)
    samples = vine.sample(3000)
    _check_correlations(samples)


def test_nonparametric_vine_correlation():
    vine = _fit_vine(d=5, n=2500, param=False)
    samples = vine.sample(3000)
    _check_correlations(samples)
