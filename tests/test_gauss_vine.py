import numpy as np
import torch
from src.DVC_pyolder.objects import vine_obj_bin, margin_obj
from src.DVC_pyolder.config import DEFAULT_CFG


def _build_gaussian_cov(d: int, rho: float):
    cov = np.full((d,d), rho)
    np.fill_diagonal(cov, 1.0)
    return cov


def _fit_gaussian_vine(d: int = 3, n: int = 2000):
    cov = _build_gaussian_cov(d, 0.6)
    z = np.random.multivariate_normal(np.zeros(d), cov, size=n)  # shape [n,d]
    # create margins (standard normal)
    margins = [margin_obj('norm', [0.0,1.0], True) for _ in range(d)]
    vine = vine_obj_bin('c-vine', ['gaussian'], d, margins, knots=30, method='random')

    gen_dict = {'param': True, 'binning': False, 'fitted': False}
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    vine.fit(z, gen_dict, npc_dict, par_dict, bin_dict)
    return vine


def test_fast_vs_legacy_sampler():
    vine = _fit_gaussian_vine(d=3, n=1500)
    cfg_fast = DEFAULT_CFG.copy()
    cfg_fast['sampler']['fast_parametric'] = True
    cfg_fast['sampler']['fast_nonparam'] = True

    x_fast = vine.sample(2000, cfg=cfg_fast)
    cfg_slow = DEFAULT_CFG.copy()
    cfg_slow['sampler']['fast_parametric'] = False
    cfg_slow['sampler']['fast_nonparam'] = False
    x_slow = vine.sample(2000, cfg=cfg_slow)

    # compare marginal means and std
    assert np.allclose(x_fast.mean(axis=0), x_slow.mean(axis=0), atol=0.05)
    assert np.allclose(x_fast.std(axis=0), x_slow.std(axis=0), atol=0.05)


def test_pdf_integrates():
    vine = _fit_gaussian_vine(d=3, n=1500)
    pts = torch.randn(1000, 3)
    pdf = vine.pdf(pts)
    assert torch.all(torch.isfinite(pdf))
    assert (pdf > 0).all() 