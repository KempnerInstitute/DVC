#!/usr/bin/env python3
"""Compact example for the dynamic nonparametric C-vine models."""

from __future__ import annotations

import numpy as np

from dvc_package.time.nonparametric_dynamic_cvine import (
    JointDynamicNonparametricCVine,
    WindowedNonparametricCVine,
)


def main() -> None:
    rng = np.random.default_rng(11)
    base = np.array(
        [
            [1.0, 0.45, 0.2],
            [0.45, 1.0, 0.35],
            [0.2, 0.35, 1.0],
        ],
        dtype=np.float32,
    )
    windows = []
    for rho in [0.15, 0.35, 0.55, 0.75]:
        cov = base.copy()
        cov[0, 1] = cov[1, 0] = rho
        windows.append(rng.multivariate_normal(np.zeros(3), cov, size=28).astype(np.float32))

    windowed = WindowedNonparametricCVine(
        knots=7,
        npc_dict={
            "opt_method": "LL1",
            "max_iter_phase1": 1,
            "max_iter_phase2": 1,
            "normal_iters_phase1": 5,
            "normal_iters_phase2": 5,
            "final_normalization_iters": 50,
            "batch_size": 1,
        },
    )
    w_result = windowed.fit(windows)

    joint = JointDynamicNonparametricCVine(
        order=w_result.order,
        knots=7,
        trajectory_type="basis",
        trajectory_kwargs={"n_basis": 2},
        n_epochs=3,
        lr=5e-2,
        smoothness_penalty=5e-3,
        batch_size=1,
        normalization_iters=5,
        final_normalization_iters=50,
    )
    j_result = joint.fit(windows)

    print("windowed nonparametric C-vine")
    print("  order:", w_result.order)
    print("  mean NLL by time:", [round(v, 4) for v in w_result.mean_nll_by_time])
    print("joint dynamic nonparametric C-vine")
    print("  order:", j_result.order)
    print("  mean NLL by time:", [round(v, 4) for v in j_result.mean_nll_by_time])
    print("  edge statuses:")
    for edge_fit in j_result.edge_fits:
        bw = np.asarray(edge_fit.bandwidth_trajectory, dtype=np.float64)
        print(
            "   ",
            edge_fit.edge,
            edge_fit.status,
            "bandwidth range",
            round(float(np.max(bw) - np.min(bw)), 4),
        )


if __name__ == "__main__":
    main()
