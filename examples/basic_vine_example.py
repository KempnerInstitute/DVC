#!/usr/bin/env python3
"""Minimal end-to-end vine-copula example.

Generates a small multivariate Gaussian dataset, fits parametric C-, D-, and
R-vines on the same data, evaluates the held-out copula log density, and
samples from each fitted model. Saves a 2x3 diagnostic figure under
``results/basic_vine_example/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running before `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dvc_package.core.vine_factory import create_vine
from dvc_package.core.vine_model import evaluate_vine, fit_vine, sample_vine


FAMILIES = ["independence", "gaussian", "clayton", "frank", "gumbel"]


def make_synthetic_data(
    n_samples: int = 1000,
    dim: int = 4,
    base_corr: float = 0.55,
    seed: int = 0,
) -> np.ndarray:
    """Sample a small multivariate-Gaussian dataset with banded correlations."""
    rng = np.random.default_rng(seed)
    cov = np.eye(dim, dtype=np.float64)
    for i in range(dim):
        for j in range(i + 1, dim):
            cov[i, j] = cov[j, i] = base_corr * (0.6 ** (j - i - 1))
    return rng.multivariate_normal(np.zeros(dim), cov, size=n_samples).astype(np.float32)


def fit_one_vine(vine_type: str, x_train: np.ndarray):
    """Fit a parametric vine of the requested type on x_train."""
    vine = create_vine(vine_type, vine_depth=x_train.shape[1], families=FAMILIES)
    fit_vine(
        vine,
        x_train,
        gen_dict={"param": True, "binning": False, "fitted": False},
        npc_dict={},
        par_dict={"param_families": FAMILIES},
        bin_dict={},
    )
    return vine


def held_out_log_density(vine, x_test: np.ndarray) -> float:
    points = torch.as_tensor(x_test, dtype=torch.float32)
    joint_pdf, _, _ = evaluate_vine(vine, points)
    return float(torch.log(joint_pdf.clamp_min(1e-12)).mean().item())


def main() -> None:
    out_dir = Path("results/basic_vine_example")
    out_dir.mkdir(parents=True, exist_ok=True)

    x = make_synthetic_data()
    n_train = int(0.8 * len(x))
    x_train, x_test = x[:n_train], x[n_train:]

    summaries = {}
    samples_by_type = {}
    for vine_type in ("c-vine", "d-vine", "r-vine"):
        vine = fit_one_vine(vine_type, x_train)
        log_dens = held_out_log_density(vine, x_test)
        summaries[vine_type] = log_dens
        samples_by_type[vine_type] = np.asarray(sample_vine(vine, n_train))
        print(f"  {vine_type}: held-out mean log f(x) = {log_dens:+.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for col, (vine_type, samples) in enumerate(samples_by_type.items()):
        axes[0, col].scatter(x_train[:, 0], x_train[:, 1], s=4, alpha=0.4, label="train")
        axes[0, col].set_title(f"{vine_type}: train (x0, x1)")
        axes[0, col].set_xlabel("x0")
        axes[0, col].set_ylabel("x1")

        axes[1, col].scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.4, color="C1", label="sampled")
        axes[1, col].set_title(f"{vine_type}: sampled (x0, x1)")
        axes[1, col].set_xlabel("x0")
        axes[1, col].set_ylabel("x1")

    fig_path = out_dir / "fitted_vs_sampled.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\nFigure: {fig_path}")
    print("Held-out log densities:", {k: f"{v:+.4f}" for k, v in summaries.items()})


if __name__ == "__main__":
    main()
