###############################################
# src/DVC/config.py
###############################################

"""Central place for default settings and YAML config loading.

Usage
-----
>>> from DVC.config import load_config, DEFAULT_CFG
>>> cfg = load_config("my_experiment.yaml")
"""

import copy
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # Fallback – we will handle this gracefully below.

# ---------------------------------------------------------------------
# Default configuration dictionary – mutating in-place is discouraged.
# ---------------------------------------------------------------------
DEFAULT_CFG: Dict[str, Any] = {
    "optimizer": {
        # optimise all edges of a tree level in one batched pass
        "batch_edges": True,
        # loclik batches when evaluating the grid
        "batch_size": 5,
        # phase-1 MISE optimiser parameters
        "max_iter_phase1": 70,
        "lr_phase1": 0.10,
        "tol_phase1": 1e-5,
        # phase-2 (row/col normalised) parameters
        "max_iter_phase2": 100,
        "lr_phase2": 0.03,
        "tol_phase2": 5e-5,
    },
    "bandwidth": {
        # Available: "rule_of_thumb", "knn"
        "method": "rule_of_thumb",
        # only used if method=="knn"
        "knn_k": 10,
    },
    "npc": {
        # Local-likelihood variant: "LL1" (original), "LL2" (squared), etc.
        "opt_method": "LL1",
        # If true compute CDF gradient grid once and use it for h-function
        "grad_precompute": False
    },
    "sampler": {
        "fast_parametric": True,
        "fast_nonparam": True,
        "nspline": 200
    },
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _recursive_update(base: Dict[str, Any], override: Dict[str, Any]):
    """Recursively merge *override* into *base* (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _recursive_update(base[k], v)
        else:
            base[k] = v


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load a YAML config file and merge with :pydata:`DEFAULT_CFG`.

    If *path* is ``None`` or the file does not exist / cannot be parsed the
    default configuration is returned.
    """
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path is None:
        return cfg

    path = Path(path)
    if not path.is_file():
        print(f"[DVC] Config file '{path}' not found – falling back to defaults.")
        return cfg

    if yaml is None:
        print("[DVC] PyYAML not available – cannot read YAML configs. Using defaults.")
        return cfg

    try:
        with path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError("Top-level YAML object must be a mapping.")
        _recursive_update(cfg, user_cfg)
    except Exception as exc:  # pragma: no cover
        print(f"[DVC] Failed to parse config '{path}': {exc}. Using defaults.")
    return cfg 