import numpy as np
import torch
import sys
from pathlib import Path

# Add repository root and TensorFlow-style src to PYTHONPATH so imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "DVC_tensorflow" / "src"))

from src.DVC.objects import vine_obj_bin, margin_obj
from classes.objects import vine_obj_bin as tf_vine_obj


_DEF_RHO = 0.5


def _build_cov(d, rho):
    cov = np.full((d, d), rho)
    np.fill_diagonal(cov, 1.0)
    return cov


# Differential entropy of multivariate normal with covariance matrix `cov`
def _mv_entropy(cov):
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * (d * (1.0 + np.log(2*np.pi)) + logdet)


def _fit_vine(lib="torch", d=4, n=300):
    cov = _build_cov(d, _DEF_RHO)
    z = np.random.multivariate_normal(np.zeros(d), cov, size=n)
    margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(d)]
    if lib == "torch":
        vine = vine_obj_bin('c-vine', ['gaussian'], d, margins, knots=10, method='random')
    else:
        vine = tf_vine_obj('c-vine', ['gaussian'], d, margins, knots=10, method='random')
    gen_dict = {
        'param': True,
        'binning': False,
        'fitted': False,
        'parallel': False,
        'vine_depth': d,
    }
    npc_dict = {}
    par_dict = {'param_families': ['gaussian']}
    bin_dict = {'n_bin': 1}
    vine.fit(torch.tensor(z, dtype=torch.float32), gen_dict, npc_dict, par_dict, bin_dict)
    return vine, cov


def _evaluate(vine, cov, ns=500):
    samples = vine.sample(ns)
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    corr = np.corrcoef(samples, rowvar=False)
    diff = np.linalg.norm(corr - cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov))), ord="fro")
    entropy = _mv_entropy(np.cov(samples, rowvar=False))
    true_entropy = _mv_entropy(cov)
    ent_diff = abs(entropy - true_entropy)
    return diff, ent_diff


def test_tensorflow_vs_torch_vine():
    vine_torch, cov = _fit_vine("torch")
    vine_tf, _ = _fit_vine("tf")
    diff_torch, ent_torch = _evaluate(vine_torch, cov)
    diff_tf, ent_tf = _evaluate(vine_tf, cov)
    assert diff_torch <= diff_tf + 1e-3
    assert ent_torch <= ent_tf + 1e-3
    
    
if __name__ == "__main__":
    test_tensorflow_vs_torch_vine() 