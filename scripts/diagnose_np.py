#!/usr/bin/env python3
"""Diagnose nonparametric h-function independence-of-conditioning property.

For an AR(1) Gaussian, level-1 D-vine inputs (u_{0|1}, u_{2|1}) should be
independent. This script measures the empirical correlation as a sanity test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from dvc_package.core.vine_factory import create_vine
from dvc_package.core.vine_model import fit_vine


def main() -> None:
    rng = np.random.default_rng(42)
    d = 4
    n = 5000
    rho = 0.5
    C = np.array([[rho ** abs(i - j) for j in range(d)] for i in range(d)])
    x = rng.multivariate_normal(np.zeros(d), C, size=n).astype(np.float32)

    vine = create_vine(
        "d-vine",
        d,
        families=["independence", "gaussian"],
        variable_order=list(range(d)),
    )
    fit_vine(
        vine,
        x,
        {"param": False, "binning": False, "fitted": True},
        {
            "opt_method": "LL1",
            "max_iter_phase1": 4,
            "early_stop": True,
            "depth": 3,
            "knots": 30,
            "knn_k": 10,
            "bandwidth_method": "rule_of_thumb",
            "batch": 3,
            "normalization_iters": 50,
            "estimator": "ll",
            "data_space": "x",
            "select_depth": False,
        },
        {},
        {},
    )

    theta = vine.theta
    theta_flip = vine.theta_flip

    print("=== Pearson on margin pseudo-obs ===")
    for j in range(d):
        for k in range(j + 1, d):
            cor = float(torch.corrcoef(torch.stack([theta[:, 0, j], theta[:, 0, k]]))[0, 1])
            print(f"  cor(u_{j}, u_{k}) = {cor:+.3f}  (truth = {rho ** (k - j):.3f})")

    print("\n=== Independence-of-conditioning ===")
    from dvc_package.core.nonparametric_vine import (
        _build_edge_input_pairs,
        _propagated_inputs_for_level,
        evaluate_nonparametric_edge_h,
    )

    edge_refs = vine._internal_ind_vine
    point_u_l0 = _build_edge_input_pairs(vine.theta, vine.theta_flip, edge_refs, 0, torch.device("cpu"))
    flip_flags = vine.flip_flag[0]
    edge_indices = vine.ind_edge_rel[0]
    h_cops = vine._np_h_copulas[0]
    for j, (flip, edge_idx) in enumerate(zip(flip_flags, edge_indices)):
        edge = edge_refs[0][edge_idx]
        uv = point_u_l0[:, :, edge_idx]
        cop = h_cops[j]
        if flip:
            hval = evaluate_nonparametric_edge_h(cop, uv[:, [1, 0]], vine.grid_s)
            conditioner = uv[:, 1]
            target, given = edge[0], edge[1]
        else:
            hval = evaluate_nonparametric_edge_h(cop, uv, vine.grid_s)
            conditioner = uv[:, 0]
            target, given = edge[1], edge[0]
        cor = float(torch.corrcoef(torch.stack([hval, conditioner]))[0, 1])
        print(
            f"  Edge ({edge[0]},{edge[1]}): cor(h(u_{target}|u_{given}), u_{given}) = {cor:+.3f}  "
            "(should be ~0)"
        )

    for level in range(1, d - 1):
        print(f"\n=== Pearson at LEVEL {level} propagated inputs ===")
        point_u_l = _propagated_inputs_for_level(vine, x, edge_refs, level, torch.device("cpu"))
        for j in range(point_u_l.shape[2]):
            u1 = point_u_l[:, 0, j]
            u2 = point_u_l[:, 1, j]
            c = float(torch.corrcoef(torch.stack([u1, u2]))[0, 1])
            print(f"  L{level} edge{j}: cor(u_in1, u_in2) = {c:+.3f}  (truth = 0)")


if __name__ == "__main__":
    main()
