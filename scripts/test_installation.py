#!/usr/bin/env python3
"""Verify the dynamic-vine-copulas installation.

Checks that the runtime dependencies and the public package modules import,
runs a tiny end-to-end sanity check, and reports CUDA availability.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path


RUNTIME_PACKAGES = [
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("sklearn", "scikit-learn"),
    ("yaml", "PyYAML"),
    ("click", "Click"),
    ("h5py", "h5py"),
]


PACKAGE_MODULES = [
    "dvc_package.core.param_copula",
    "dvc_package.core.vine_factory",
    "dvc_package.core.vine_model",
    "dvc_package.core.info_estimation",
    "dvc_package.optimization.structure",
    "dvc_package.time.models",
    "dvc_package.experiments.experiment_framework",
]


def _try_import(module_name: str, label: str | None = None) -> bool:
    label = label or module_name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "")
        suffix = f" {version}" if version else ""
        print(f"  [ok] {label}{suffix}")
        return True
    except ImportError as exc:
        print(f"  [missing] {label}: {exc}")
        return False


def _smoke_test() -> bool:
    try:
        import numpy as np
        import torch

        from dvc_package.core.vine_factory import create_vine
        from dvc_package.core.vine_model import evaluate_vine, fit_vine

        rng = np.random.default_rng(0)
        x = rng.multivariate_normal(
            mean=np.zeros(3),
            cov=np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]),
            size=200,
        ).astype(np.float32)

        families = ["independence", "gaussian", "clayton"]
        vine = create_vine("c-vine", vine_depth=3, families=families)
        fit_vine(
            vine,
            x,
            {"param": True, "binning": False, "fitted": False},
            {},
            {"param_families": families},
            {},
        )
        evaluate_vine(vine, torch.tensor(x[:8]))
        print("  [ok] minimal fit + evaluate")
        return True
    except Exception as exc:  # pragma: no cover - smoke check
        print(f"  [fail] minimal fit + evaluate: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    print("Dynamic Vine Copulas: installation check")
    print("-" * 40)
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")

    src_path = Path(__file__).resolve().parent.parent / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    print("\nRuntime packages:")
    runtime_ok = sum(_try_import(name, label) for name, label in RUNTIME_PACKAGES)

    print("\nPackage modules:")
    package_ok = sum(_try_import(name) for name in PACKAGE_MODULES)

    print("\nSmoke test:")
    smoke_ok = _smoke_test()

    print("\nCUDA:")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  [ok] available, device 0: {torch.cuda.get_device_name(0)}")
        else:
            print("  [info] not available; CPU mode")
    except Exception as exc:  # pragma: no cover
        print(f"  [fail] unable to query CUDA: {exc}")

    overall_ok = (
        runtime_ok == len(RUNTIME_PACKAGES)
        and package_ok == len(PACKAGE_MODULES)
        and smoke_ok
    )
    print("\nResult:", "ok" if overall_ok else "incomplete")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
