import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from DVC.config import load_config, DEFAULT_CFG
from DVC.objects import vine_obj_bin, margin_obj
from DVC.utils_prob import biv_norm
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# Load configuration
# ------------------------------------------------------------
CFG_PATH = Path(__file__).parent.parent / "configs" / "gauss_nd.yaml"
cfg = load_config(CFG_PATH if CFG_PATH.exists() else None)

# Data generation parameters
n_samples = cfg['data']['n_samples']
dim       = cfg['data']['dim']
rho       = cfg['data']['rho']

# ------------------------------------------------------------
# Synthetic Gaussian data
# ------------------------------------------------------------
cov_true = np.full((dim, dim), rho)
np.fill_diagonal(cov_true, 1.0)
data = np.random.multivariate_normal(np.zeros(dim), cov_true, size=n_samples)

# ------------------------------------------------------------
# Build vine object
# ------------------------------------------------------------
margins = [margin_obj('norm', [0.0, 1.0], True) for _ in range(dim)]

vine = vine_obj_bin(
    cfg['vine']['family'],
    ['gaussian'],
    dim,
    margins,
    knots=cfg['vine']['knots'],
    method=cfg['vine']['method']
)

# Fit dictionaries
gen_dict = {
    'param': cfg['general']['param'],
    'binning': cfg['general']['binning'],
    'fitted': False
}

npc_dict = cfg.get('npc', {})
par_dict = {'param_families': ['gaussian']}
bin_dict = {'n_bin': 1}

# Fit
vine.fit(data, gen_dict, npc_dict, par_dict, bin_dict, cfg)

# ------------------------------------------------------------
# Entropy and KL diagnostics
# ------------------------------------------------------------
normal = torch.distributions.Normal(0., 1.)

def gauss_entropy(d, det):
    return 0.5 * np.log((2*np.pi*np.e)**d * det)

# Monte-Carlo estimate of vine entropy
samples = vine.sample(5000, cfg)
logp    = vine.logpdf(torch.tensor(samples, dtype=torch.float32))
H_est   = -logp.mean().item()
H_true  = gauss_entropy(dim, np.linalg.det(cov_true))

kl = (logp - torch.tensor(multivariate_normal.logpdf(samples, mean=np.zeros(dim), cov=cov_true))).mean().item()

print(f"True Gaussian entropy : {H_true:.3f}")
print(f"Vine   entropy (MC)   : {H_est:.3f}")
print(f"KL(vine ‖ true)      : {kl:.3f}\n")

# ------------------------------------------------------------
# Visualise vine structure
# ------------------------------------------------------------
G = nx.Graph()
for lvl, edges in enumerate(vine.ind_vine):
    for e in edges:
        G.add_edge(e[0], e[1], level=lvl)

pos = nx.spring_layout(G, seed=0)
levels = [G[u][v]['level'] for u, v in G.edges()]
plt.figure(figsize=(4,4))
nx.draw(G, pos, with_labels=True, edge_color=levels, edge_cmap=plt.cm.viridis, width=2)
plt.title('Vine structure')
plt.show()

# ------------------------------------------------------------
# Plot first-level copula densities
# ------------------------------------------------------------
fig, axes = plt.subplots(1, min(3, len(vine.copulas[0])), figsize=(12,3))
for ax, cobj in zip(axes, vine.copulas[0][:3]):
    if hasattr(cobj, 'pd_grid_uv') and cobj.pd_grid_uv is not None:
        ax.imshow(cobj.pd_grid_uv.cpu().numpy(), origin='lower', cmap='magma')
    ax.axis('off')
plt.suptitle('First-level copula PDFs')
plt.show() 