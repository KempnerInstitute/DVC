# Fitting vine copulas

- Select vine type (C-/D-/R-vine) and choose parametric vs nonparametric pair copulas.
- Prepare margins (e.g., Gaussian) and fit using DVC_pytorch.
- For nonparametric fitting, bandwidths are optimized via local likelihood on grids.

Example (PyTorch):

```python
import dvc.backends.pytorch as DVC
margins = [DVC.margin_obj('gaussian', [0.0, 1.0], True) for _ in range(d)]
vine = DVC.vine_obj_bin('c-vine', ['gaussian'], d, margins, 64, 'matrix')

gen = {'fitted': False, 'binning': False, 'parallel': True, 'param': True, 'vine_depth': d}
npc = {'opt_method': 'LL1', 'batch_paral': 1}
par = {'param_families': ['gaussian']}
bin_d = {'n_bin': 1}

vine.fit(X, gen, npc, par, bin_d)
```
