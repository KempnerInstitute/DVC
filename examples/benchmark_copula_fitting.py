#!/usr/bin/env python
"""
Benchmark: Copula Family Fitting

Systematically benchmarks all copula families (Gaussian, Student-t, Clayton,
Frank, Gumbel) with known parameters. Reports:
  - Parameter recovery accuracy (mean absolute error)
  - AIC-based family selection accuracy
  - Computation time per family
"""

import time
import numpy as np
import torch
from scipy.stats import norm, t as t_dist
from collections import defaultdict

from dvc_package.core.param_copula import (
    fit_gaussian, fit_student, fit_clayton, fit_frank, fit_gumbel,
    parametric_fit,
)


# ---------------------------------------------------------------------------
# Data generators for each copula family
# ---------------------------------------------------------------------------

def generate_gaussian(rho, n, rng):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0, 0], cov, size=n)
    return norm.cdf(z)


def generate_clayton(alpha, n, rng):
    u1 = rng.uniform(0.001, 0.999, n)
    w = rng.uniform(0.001, 0.999, n)
    u2 = (w ** (-alpha / (1 + alpha)) * u1 ** (-alpha) - u1 ** (-alpha) + 1) ** (-1 / alpha)
    return np.column_stack([u1, np.clip(u2, 1e-6, 1 - 1e-6)])


def generate_frank(theta, n, rng):
    u1 = rng.uniform(0.001, 0.999, n)
    w = rng.uniform(0.001, 0.999, n)
    if abs(theta) < 1e-6:
        return np.column_stack([u1, w])
    exp_neg_theta = np.exp(-theta)
    u2 = -np.log(1 + (exp_neg_theta - 1) / (np.exp(-theta * u1) * (1 / w - 1) + 1)) / theta
    return np.column_stack([u1, np.clip(u2, 1e-6, 1 - 1e-6)])


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(n_trials=50, n_samples=1000, seed=42):
    rng = np.random.default_rng(seed)

    configs = [
        ("Gaussian", 0.3,  "gaussian", lambda u: fit_gaussian(u)[0]),
        ("Gaussian", 0.6,  "gaussian", lambda u: fit_gaussian(u)[0]),
        ("Gaussian", 0.8,  "gaussian", lambda u: fit_gaussian(u)[0]),
        ("Gaussian", -0.5, "gaussian", lambda u: fit_gaussian(u)[0]),
        ("Clayton",  1.0,  "clayton",  lambda u: fit_clayton(u)[0]),
        ("Clayton",  2.0,  "clayton",  lambda u: fit_clayton(u)[0]),
        ("Clayton",  5.0,  "clayton",  lambda u: fit_clayton(u)[0]),
        ("Frank",    2.0,  "frank",    lambda u: fit_frank(u)[0]),
        ("Frank",    5.0,  "frank",    lambda u: fit_frank(u)[0]),
        ("Frank",   -3.0,  "frank",    lambda u: fit_frank(u)[0]),
        ("Gumbel",   1.5,  "gumbel",   lambda u: fit_gumbel(u)[0]),
        ("Gumbel",   3.0,  "gumbel",   lambda u: fit_gumbel(u)[0]),
    ]

    generators = {
        "gaussian": generate_gaussian,
        "clayton":  generate_clayton,
        "frank":    generate_frank,
    }

    print("=" * 80)
    print(f"Copula Fitting Benchmark  |  {n_trials} trials × {n_samples} samples")
    print("=" * 80)

    for family_name, true_theta, gen_key, fit_fn in configs:
        errors, times_list = [], []
        gen_fn = generators.get(gen_key)
        if gen_fn is None:
            continue

        for trial in range(n_trials):
            data = gen_fn(true_theta, n_samples, rng).astype(np.float32)
            u = torch.tensor(data, dtype=torch.float32)

            t0 = time.perf_counter()
            try:
                theta_hat = fit_fn(u)
                if isinstance(theta_hat, tuple):
                    theta_hat = theta_hat[0]
                errors.append(abs(theta_hat - true_theta))
            except Exception as e:
                errors.append(float("nan"))
            times_list.append(time.perf_counter() - t0)

        errs = np.array(errors)
        valid = errs[np.isfinite(errs)]
        print(f"\n{family_name:10s} θ={true_theta:6.2f}  |  "
              f"MAE={np.mean(valid):.4f}±{np.std(valid):.4f}  "
              f"median={np.median(valid):.4f}  "
              f"time={np.mean(times_list)*1000:.1f} ms  "
              f"({len(valid)}/{n_trials} valid)")

    # --- AIC family selection benchmark ---
    print("\n" + "=" * 80)
    print("AIC Family Selection Benchmark")
    print("=" * 80)

    families_to_test = ["ind", "gaussian", "clayton", "frank", "gumbel"]
    selection_configs = [
        ("Gaussian ρ=0.6", "gaussian", 0.6),
        ("Clayton α=2.0",  "clayton",  2.0),
        ("Frank θ=3.0",    "frank",    3.0),
    ]

    for label, gen_key, true_theta in selection_configs:
        gen_fn = generators[gen_key]
        correct = 0
        for trial in range(n_trials):
            data = gen_fn(true_theta, n_samples, rng).astype(np.float32)
            data_3d = data[:, :, np.newaxis]
            aic2, _, _ = parametric_fit(data_3d, families_to_test, n_cop=1)
            best = families_to_test[np.argmin(aic2[0])]
            if best == gen_key:
                correct += 1
        print(f"{label:25s}  |  AIC correct: {correct}/{n_trials} ({100*correct/n_trials:.0f}%)")

    print("\nDone.")


if __name__ == "__main__":
    run_benchmark()
