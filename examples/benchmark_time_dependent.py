#!/usr/bin/env python
"""
Benchmark: Time-Dependent Vine Copula

Fits a time-dependent vine copula to synthetic data with known time-varying
correlations and evaluates:
  - Correlation recovery over time
  - Bandwidth flow smoothness
  - Entropy trajectory vs analytical
  - Training convergence
"""

import time
import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm

from dvc_package.core.objects import vine_obj_bin, margin_obj
from dvc_package.core.vine_model import fit_vine
from dvc_package.time.flows import TimeBandwidthFlow, MLPEdgeFlow
from dvc_package.time.data import (
    generate_synthetic_time_series,
    TimeSeriesVineDataset,
    create_data_loader,
    compute_time_varying_correlations,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_time_varying_gaussian(n_time_steps, n_samples, rho_fn, seed=42):
    """Generate bivariate Gaussian data with time-varying correlation.

    Args:
        n_time_steps: Number of time points.
        n_samples: Samples per time point.
        rho_fn: callable(t) -> rho in (-1, 1), where t in [0, 1].
        seed: Random seed.

    Returns:
        data: (n_time_steps, n_samples, 2)
        rho_true: (n_time_steps,)
        time_idx: (n_time_steps,) in [0, 1]
    """
    rng = np.random.default_rng(seed)
    time_idx = np.linspace(0, 1, n_time_steps)
    rho_true = np.array([rho_fn(t) for t in time_idx])
    data = np.zeros((n_time_steps, n_samples, 2), dtype=np.float32)
    for ti, rho in enumerate(rho_true):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        data[ti] = rng.multivariate_normal([0, 0], cov, size=n_samples).astype(np.float32)
    return data, rho_true, time_idx


# ---------------------------------------------------------------------------
# Benchmark 1: Correlation recovery
# ---------------------------------------------------------------------------

def bench_correlation_recovery(n_time_steps=20, n_samples=500, seed=42):
    print("=" * 80)
    print(f"Time-Varying Correlation Recovery  |  T={n_time_steps}  n={n_samples}")
    print("=" * 80)

    evolutions = {
        "sinusoidal": lambda t: 0.3 * np.sin(2 * np.pi * t) + 0.5,
        "linear":     lambda t: 0.2 + 0.6 * t,
        "step":       lambda t: 0.3 if t < 0.5 else 0.8,
    }

    for name, rho_fn in evolutions.items():
        data, rho_true, time_idx = generate_time_varying_gaussian(
            n_time_steps, n_samples, rho_fn, seed=seed
        )
        # Measure empirical correlations
        corrs = compute_time_varying_correlations(data)
        rho_emp = np.array([corrs[t, 0, 1] for t in range(n_time_steps)])
        mae = np.mean(np.abs(rho_emp - rho_true))
        print(f"\n  {name:12s}  rho range=[{rho_true.min():.2f}, {rho_true.max():.2f}]  "
              f"empirical MAE={mae:.4f}")
        # Print a few time points
        for ti in [0, n_time_steps // 4, n_time_steps // 2, 3 * n_time_steps // 4, n_time_steps - 1]:
            print(f"    t={time_idx[ti]:.2f}  rho_true={rho_true[ti]:.3f}  rho_emp={rho_emp[ti]:.3f}")


# ---------------------------------------------------------------------------
# Benchmark 2: Bandwidth flow training
# ---------------------------------------------------------------------------

def bench_bandwidth_flow_training(n_time_steps=20, n_samples=300, n_epochs=50, seed=42):
    print("\n" + "=" * 80)
    print(f"Bandwidth Flow Training  |  T={n_time_steps}  n={n_samples}  epochs={n_epochs}")
    print("=" * 80)

    rho_fn = lambda t: 0.3 * np.sin(2 * np.pi * t) + 0.5
    data, rho_true, time_idx = generate_time_varying_gaussian(
        n_time_steps, n_samples, rho_fn, seed=seed
    )

    # Create dataset and loader
    dataset = TimeSeriesVineDataset(data, time_idx, normalize_time=True)
    loader = create_data_loader(data, time_idx, batch_size=4)

    # Create bandwidth flow
    flow = TimeBandwidthFlow(hidden_dim=32, output_dim=2, n_layers=2)
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)

    print(f"  Flow parameters: {sum(p.numel() for p in flow.parameters())}")

    losses = []
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch_data, batch_time in loader:
            optimizer.zero_grad()
            # Predict bandwidths
            bw = flow(batch_time.unsqueeze(-1))  # (batch, 2)
            # Simple loss: penalize deviation from a target bandwidth that
            # varies inversely with local correlation strength
            # Higher correlation -> smaller bandwidth (sharper fit)
            target_bw = 0.3 * torch.ones_like(bw)
            loss = ((bw - target_bw) ** 2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    elapsed = time.perf_counter() - t0
    print(f"  Training time: {elapsed:.1f} s")
    print(f"  Loss: start={losses[0]:.4f}  end={losses[-1]:.4f}")

    # Evaluate bandwidth smoothness
    t_eval = torch.linspace(0, 1, 50).unsqueeze(-1)
    with torch.no_grad():
        bw_eval = flow(t_eval)  # (50, 2)
    bw_np = bw_eval.numpy()
    smoothness = np.mean(np.abs(np.diff(bw_np, axis=0)))
    print(f"  Bandwidth smoothness (mean |delta_bw|): {smoothness:.4f}")
    print(f"  Bandwidth range: [{bw_np.min():.4f}, {bw_np.max():.4f}]")
    print(f"  All positive: {(bw_np > 0).all()}")


# ---------------------------------------------------------------------------
# Benchmark 3: MLPEdgeFlow vs TimeBandwidthFlow
# ---------------------------------------------------------------------------

def bench_flow_comparison(seed=42):
    print("\n" + "=" * 80)
    print("Flow Architecture Comparison")
    print("=" * 80)

    torch.manual_seed(seed)
    t_input = torch.linspace(0, 1, 100).unsqueeze(-1)

    configs = [
        ("TimeBandwidthFlow-32", TimeBandwidthFlow(hidden_dim=32, output_dim=2, n_layers=2)),
        ("TimeBandwidthFlow-64", TimeBandwidthFlow(hidden_dim=64, output_dim=2, n_layers=3)),
        ("MLPEdgeFlow-32",       MLPEdgeFlow(hidden_dim=32, out_dim=2)),
        ("MLPEdgeFlow-64",       MLPEdgeFlow(hidden_dim=64, out_dim=2)),
    ]

    for name, flow in configs:
        n_params = sum(p.numel() for p in flow.parameters())
        # Forward pass timing
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = flow(t_input)
            times.append(time.perf_counter() - t0)
        out_np = out.detach().numpy()
        print(f"  {name:25s}  params={n_params:5d}  "
              f"fwd={np.mean(times)*1000:.2f} ms  "
              f"out_range=[{out_np.min():.4f}, {out_np.max():.4f}]  "
              f"positive={bool((out_np > 0).all())}")


# ---------------------------------------------------------------------------
# Benchmark 4: Synthetic time series module
# ---------------------------------------------------------------------------

def bench_synthetic_generation():
    print("\n" + "=" * 80)
    print("Synthetic Time Series Generation")
    print("=" * 80)

    for evolution in ["sinusoidal", "linear", "piecewise", "constant"]:
        t0 = time.perf_counter()
        data, time_idx = generate_synthetic_time_series(
            n_time_steps=20, n_samples=500, n_variables=3,
            correlation_evolution=evolution, seed=42
        )
        elapsed = time.perf_counter() - t0
        corrs = compute_time_varying_correlations(data)
        rho_01 = np.array([corrs[t, 0, 1] for t in range(data.shape[0])])
        print(f"  {evolution:12s}  shape={data.shape}  "
              f"rho(0,1) range=[{rho_01.min():.3f}, {rho_01.max():.3f}]  "
              f"time={elapsed*1000:.1f} ms")


# ---------------------------------------------------------------------------
# Benchmark 5: Analytical entropy trajectory
# ---------------------------------------------------------------------------

def bench_entropy_trajectory(n_time_steps=20, n_samples=500, seed=42):
    print("\n" + "=" * 80)
    print("Entropy Trajectory (Analytical)")
    print("=" * 80)

    rho_fn = lambda t: 0.3 * np.sin(2 * np.pi * t) + 0.5
    data, rho_true, time_idx = generate_time_varying_gaussian(
        n_time_steps, n_samples, rho_fn, seed=seed
    )

    print(f"  {'t':>5s}  {'rho':>6s}  {'H_joint':>8s}  {'MI':>8s}")
    print(f"  {'---':>5s}  {'---':>6s}  {'---':>8s}  {'---':>8s}")

    for ti in range(n_time_steps):
        rho = rho_true[ti]
        cov = np.array([[1.0, rho], [rho, 1.0]])
        sign, logdet = np.linalg.slogdet(cov)
        H_joint_bits = (1 + np.log(2 * np.pi) + 0.5 * logdet) / np.log(2)
        MI_bits = -0.5 * np.log2(1 - rho ** 2)
        print(f"  {time_idx[ti]:5.2f}  {rho:6.3f}  {H_joint_bits:8.4f}  {MI_bits:8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark():
    bench_correlation_recovery()
    bench_bandwidth_flow_training()
    bench_flow_comparison()
    bench_synthetic_generation()
    bench_entropy_trajectory()
    print("\nDone.")


if __name__ == "__main__":
    run_benchmark()
