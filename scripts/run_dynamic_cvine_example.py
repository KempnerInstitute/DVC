#!/usr/bin/env python3
"""Run a compact example of jointly fitted dynamic C-vines.

This script generates a smooth time-varying dependence sequence, compares
windowed per-slice fitting against jointly parameterized and latent-state
dynamic C-vines, and saves a small summary plot plus JSON metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from dvc_package.experiments.simulation_benchmarks import _fit_parametric_vine, _mean_copula_nll
from dvc_package.time import JointDynamicCVine, LatentStateDynamicCVine


def _make_smooth_sequence(
    *,
    n_time_steps: int,
    n_samples_per_time: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    windows: List[np.ndarray] = []
    true_tau = np.zeros(n_time_steps, dtype=np.float64)

    for t in range(n_time_steps):
        tau_scale = t / max(n_time_steps - 1, 1)
        rho01 = 0.10 + 0.70 * tau_scale
        rho02 = 0.35 + 0.20 * np.sin(np.pi * tau_scale)
        x0 = rng.normal(size=n_samples_per_time)
        x1 = rho01 * x0 + np.sqrt(max(1.0 - rho01 ** 2, 1e-6)) * rng.normal(size=n_samples_per_time)
        x2 = rho02 * x0 + np.sqrt(max(1.0 - rho02 ** 2, 1e-6)) * rng.normal(size=n_samples_per_time)
        x3 = 0.5 * x1 - 0.35 * x2 + rng.normal(scale=0.8, size=n_samples_per_time)
        x = np.column_stack([x0, x1, x2, x3]).astype(np.float32)
        x += 1e-3 * rng.normal(size=x.shape).astype(np.float32)
        windows.append(x)
        true_tau[t] = (2.0 / np.pi) * np.arcsin(rho01)

    return windows, true_tau


def _train_test_split(windows: List[np.ndarray], seed: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    x_train: List[np.ndarray] = []
    x_test: List[np.ndarray] = []
    for t, x in enumerate(windows):
        n = int(x.shape[0])
        n_train = max(int(round(0.8 * n)), 10)
        idx = np.random.default_rng(seed + 97 * t).permutation(n)
        x_train.append(x[idx[:n_train]])
        x_test.append(x[idx[n_train:]])
    return x_train, x_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example: jointly fitted dynamic C-vines")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dynamic_cvine_example"))
    parser.add_argument("--n-time-steps", type=int, default=10)
    parser.add_argument("--n-samples-per-time", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    windows, true_tau = _make_smooth_sequence(
        n_time_steps=args.n_time_steps,
        n_samples_per_time=args.n_samples_per_time,
        seed=args.seed,
    )
    x_train, x_test = _train_test_split(windows, seed=args.seed)

    families = ["gaussian", "student", "independence"]

    windowed_nll = []
    empirical_tau = []
    for t, (tr, te) in enumerate(zip(x_train, x_test)):
        vine = _fit_parametric_vine(tr, families=families, optimize_structure=False, seed=args.seed + t)
        windowed_nll.append(float(_mean_copula_nll(vine, te)))
        tau_hat, _ = kendalltau(tr[:, 0], tr[:, 1])
        empirical_tau.append(0.0 if not np.isfinite(tau_hat) else float(tau_hat))

    joint = JointDynamicCVine(
        families=families,
        n_basis=3,
        smoothness_penalty=0.5,
        ridge_penalty=1e-3,
        maxiter=30,
    ).fit(x_train)
    latent = LatentStateDynamicCVine(
        families=families,
        order=joint.order,
        selection_n_basis=3,
        selection_smoothness_penalty=0.5,
        latent_dim=1,
        transition_penalty=5e-2,
        n_epochs=40,
        lr=2e-2,
    ).fit(x_train)

    joint_nll = joint.evaluate(x_test)
    latent_nll = latent.evaluate(x_test)

    # Prefer the leading root-edge that involves variables 0 and 1 when available.
    edge_key = (0, 0, 1) if (0, 0, 1) in joint.edge_fit_map() else sorted(joint.edge_fit_map().keys())[0]
    joint_tau = np.asarray(joint.edge_fit_map()[edge_key].tau_trajectory, dtype=np.float64)
    latent_tau = np.asarray(latent.edge_fit_map()[edge_key].tau_trajectory, dtype=np.float64)

    summary = {
        "seed": int(args.seed),
        "n_time_steps": int(args.n_time_steps),
        "n_samples_per_time": int(args.n_samples_per_time),
        "joint_order": list(joint.order),
        "latent_order": list(latent.order),
        "windowed_mean_nll": float(np.mean(windowed_nll)),
        "joint_mean_nll": float(np.mean(joint_nll)),
        "latent_mean_nll": float(np.mean(latent_nll)),
        "joint_improvement_over_windowed": float(np.mean(np.asarray(windowed_nll) - joint_nll)),
        "latent_improvement_over_windowed": float(np.mean(np.asarray(windowed_nll) - latent_nll)),
    }

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    time = np.arange(args.n_time_steps)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True)

    axes[0].plot(time, true_tau, color="black", linewidth=2.0, label="True $\\tau$")
    axes[0].plot(time, empirical_tau, color="#999999", linewidth=1.4, linestyle="--", label="Windowed emp. $\\tau$")
    axes[0].plot(time, joint_tau, color="#0072B2", linewidth=1.9, label="Joint DVC")
    axes[0].plot(time, latent_tau, color="#6A3D9A", linewidth=1.9, linestyle=(0, (3, 1.5)), label="Latent DVC")
    axes[0].set_title("Smooth edge trajectory")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Kendall $\\tau$")
    axes[0].legend(fontsize=7, frameon=True)

    axes[1].plot(time, windowed_nll, color="#999999", linewidth=1.7, linestyle="--", label="Windowed DVC")
    axes[1].plot(time, joint_nll, color="#0072B2", linewidth=1.9, label="Joint DVC")
    axes[1].plot(time, latent_nll, color="#6A3D9A", linewidth=1.9, linestyle=(0, (3, 1.5)), label="Latent DVC")
    axes[1].set_title("Held-out copula NLL")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("NLL")
    axes[1].legend(fontsize=7, frameon=True)

    fig.savefig(args.output_dir / "dynamic_cvine_example.pdf")
    fig.savefig(args.output_dir / "dynamic_cvine_example.png", dpi=300)
    plt.close(fig)

    print(f"Saved summary to {args.output_dir / 'summary.json'}")
    print(f"Saved figure to {args.output_dir / 'dynamic_cvine_example.pdf'}")


if __name__ == "__main__":
    main()
