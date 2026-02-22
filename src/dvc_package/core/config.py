##################################################
# src/DVC/config.py
##################################################
import copy
from pathlib import Path
from typing import Any, Dict, Union

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

DEFAULT_CFG: Dict[str, Any] = {
    "optimizer": {
        "batch_edges": True,
        "batch_size": 5,
        "max_iter_phase1": 70,
        "lr_phase1": 0.10,
        "tol_phase1": 1e-5,
        "max_iter_phase2": 100,
        "lr_phase2": 0.03,
        "tol_phase2": 5e-5,
        "jit": False,
        "max_edges_per_batch": None,
    },
    "bandwidth": {
        "method": "rule_of_thumb",
        "knn_k": 10,
    },
    "npc": {
        "opt_method": "LL1",
        "grad_precompute": False
    },
    "sampler": {
        "fast_parametric": True,
        "fast_nonparam": True,
        "nspline": 200
    },
}

def _recursive_update(base: Dict[str, Any], override: Dict[str, Any]):
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _recursive_update(base[k], v)
        else:
            base[k] = v

def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CFG)
    if path is None:
        return cfg

    path = Path(path)
    if not path.is_file():
        print(f"[DVC] Config file '{path}' not found - using defaults.")
        return cfg

    if yaml is None:
        print("[DVC] PyYAML not available - cannot read YAML configs. Using defaults.")
        return cfg

    try:
        with path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError("Top-level YAML object must be a mapping.")
        _recursive_update(cfg, user_cfg)
    except Exception as exc:
        print(f"[DVC] Failed to parse config '{path}': {exc}. Using defaults.")
    return cfg