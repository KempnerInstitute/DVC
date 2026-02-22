#!/usr/bin/env python
"""
Benchmark: Vine Entropy Estimation vs Analytical

Compares vine copula entropy estimates against closed-form Gaussian entropy
for multivariate Gaussian distributions with known covariance. Reports:
  - Absolute error vs analytical entropy
  - Convergence with increasing sample size
  - Scaling with dimension
"""

import time
import numpy as np
import torch
from scipy.stats import norm

from dvc_package.core.objects import vine_obj_bin, margin_obj
from dvc_package.core.vine_model import fit_vine, evaluate_vine, sample_vine
from dvc_package.core.info_estimation import vine_entropy, theoretic_mutual_information_AWGN


# ---------------------------------------------------------------------------
# Analytical Gaussian entropy / MI
# ---------------------------------------------------------------------------

def gaussian_entropy_nats(cov):
    """H(X) for multivariate Gaussian in nats."""
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("nan")
    return 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * logdet


def gaussian_entropy_bits(cov):
    """H(X) for multivariate Gaussian in bits."""
    return gaussian_entropy_nats(cov) / np.log(2)


def gaussian_mi_bits(rho):
    """MI = -0.5 log2(1-rho^2) for bivariate Gaussian."""
    return -0.5 * np.log2(1 - rho ** 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vine(d, families=None):
    """Create a fresh parametric vine for d variables."""
    if families is None:
        families = ["ind", "gaussian", "clayton"]
    margins = [margin_obj("norm", (0.0, 1.0)) for _ in range(d)]
    vine = vine_obj_bin("c-vine", families, d, margins, 30)
    return vine


def _fit_vine(data, families=None):
    """Fit a parametric vine to data (numpy array, shape [N, d])."""
    d = data.shape[1]
    vine = _make_vine(d, families)
    gen_dict = {"param": True, "binning": False, "fitted": False}
    npc_dict = {}
    par_dict = {"param_families": families or ["ind", "gaussian", "clayton"]}
    bin_dict = {}
    fit_vine(vine, data, gen_dict, npc_dict, par_dict, bin_dict)
    return vine


def _generate_gaussian(n, cov, rng):
    """Generate multivariate Gaussian data."""
    d = cov.shape[0]
    return rng.multivariate_normal(np.zeros(d), cov, size=n).astype(np.float32)


def _toeplitz_cov(d, rho):
    """Create a Toeplitz correlation matrix: C[i,j] = rho^|i-j|."""
    return np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])


# ---------------------------------------------------------------------------
# Benchmark 1: Entropy accuracy vs dimension
# ---------------------------------------------------------------------------

def bench_entropy_vs_dimension(n_samples=2000, n_trials=5, seed=42):
    rng = np.random.default_rng(seed)
    rho = 0.5
    dims = [2, 3, 4, 5]

    print("=" * 80)
    print(f"Entropy vs Dimension  |  rho={rho}  n={n_samples}  trials={n_trials}")
    print("=" * 80)

    for d in dims:
        cov = _toeplitz_cov(d, rho)
        H_true = gaussian_entropy_bits(cov)
        errors = []

        for trial in range(n_trials):
            data = _generate_gaussian(n_samples, cov, rng)
            vine = _fit_vine(data)

            info = {"alpha": 0.05, "cases": 500, "iterations": 8}
            try:
                H_est = vine_entropy(vine, info)
                errors.append(abs(H_est - H_true))
            except Exception as e:
                errors.append(float("nan"))

        errs = np.array(errors)
        valid = errs[np.isfinite(errs)]
        if len(valid) > 0:
            print(f"  d={d}  H_true={H_true:.3f} bits  "
                  f"MAE={np.mean(valid):.3f}±{np.std(valid):.3f}  "
                  f"({len(valid)}/{n_trials} valid)")
        else:
            print(f"  d={d}  H_true={H_true:.3f} bits  (no valid estimates)")


# ---------------------------------------------------------------------------
# Benchmark 2: Entropy convergence with sample size
# ---------------------------------------------------------------------------

def bench_entropy_convergence(d=3, rho=0.6, n_trials=5, seed=123):
    rng = np.random.default_rng(seed)
    sample_sizes = [200, 500, 1000, 2000, 5000]
    cov = _toeplitz_cov(d, rho)
    H_true = gaussian_entropy_bits(cov)

    print("\n" + "=" * 80)
    print(f"Entropy Convergence  |  d={d}  rho={rho}  H_true={H_true:.3f} bits")
    print("=" * 80)

    for n in sample_sizes:
        errors, times_list = [], []
        for trial in range(n_trials):
            data = _generate_gaussian(n, cov, rng)
            t0 = time.perf_counter()
            vine = _fit_vine(data)
            info = {"alpha": 0.05, "cases": min(n, 500), "iterations": 8}
            try:
                H_est = vine_entropy(vine, info)
                errors.append(abs(H_est - H_true))
            except Exception:
                errors.append(float("nan"))
            times_list.append(time.perf_counter() - t0)

        errs = np.array(errors)
        valid = errs[np.isfinite(errs)]
        if len(valid) > 0:
            print(f"  n={n:5d}  MAE={np.mean(valid):.3f}±{np.std(valid):.3f}  "
                  f"time={np.mean(times_list)*1000:.0f} ms  "
                  f"({len(valid)}/{n_trials} valid)")
        else:
            print(f"  n={n:5d}  (no valid estimates)")


# ---------------------------------------------------------------------------
# Benchmark 3: Mutual information recovery
# ---------------------------------------------------------------------------

def bench_mi_recovery(n_samples=2000, n_trials=5, seed=77):
    rng = np.random.default_rng(seed)
    rho_values = [0.2, 0.4, 0.6, 0.8]

    print("\n" + "=" * 80)
    print(f"Bivariate MI Recovery  |  n={n_samples}  trials={n_trials}")
    print("=" * 80)

    for rho in rho_values:
        cov = np.array([[1.0, rho], [rho, 1.0]])
        MI_true = gaussian_mi_bits(rho)
        errors = []

        for trial in range(n_trials):
            data = _generate_gaussian(n_samples, cov, rng)
            vine = _fit_vine(data)
            H_joint = gaussian_entropy_bits(cov)

            info = {"alpha": 0.05, "cases": 500, "iterations": 8}
            try:
                # MI = H(X) + H(Y) - H(X,Y)
                # For std-normal marginals, H(X) = H(Y) = 0.5*log2(2*pi*e)
                H_marginal = 0.5 * np.log2(2 * np.pi * np.e)
                H_est = vine_entropy(vine, info)
                MI_est = 2 * H_marginal - H_est
                errors.append(abs(MI_est - MI_true))
            except Exception:
                errors.append(float("nan"))

        errs = np.array(errors)
        valid = errs[np.isfinite(errs)]
        if len(valid) > 0:
            print(f"  rho={rho:.1f}  MI_true={MI_true:.4f} bits  "
                  f"MAE={np.mean(valid):.4f}±{np.std(valid):.4f}  "
                  f"({len(valid)}/{n_trials} valid)")
        else:
            print(f"  rho={rho:.1f}  MI_true={MI_true:.4f} bits  (no valid estimates)")


# ---------------------------------------------------------------------------
# Benchmark 4: AWGN theoretical MI
# ---------------------------------------------------------------------------

def bench_awgn_mi():
    print("\n" + "=" * 80)
    print("AWGN Theoretical MI Check")
    print("=" * 80)

    snr_db_values = [0, 5, 10, 20]
    for snr_db in snr_db_values:
        snr = 10 ** (snr_db / 10)
        power = snr
        noise = 1.0
        for dim in [1, 2, 4]:
            mi = theoretic_mutual_information_AWGN(power, noise, dim)
            expected = 0.5 * dim * np.log2(1 + snr)
            print(f"  SNR={snr_db:3d} dB  dim={dim}  MI={mi:.4f}  expected={expected:.4f}  "
                  f"match={'yes' if abs(mi - expected) < 1e-6 else 'NO'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark():
    bench_entropy_vs_dimension()
    bench_entropy_convergence()
    bench_mi_recovery()
    bench_awgn_mi()
    print("\nDone.")


if __name__ == "__main__":
    run_benchmark()
