#!/usr/bin/env python3
"""Minimal example for the static nonparametric vine path."""

from __future__ import annotations

import numpy as np

from dvc_package.core.vine_factory import create_vine


def main() -> None:
    rng = np.random.default_rng(7)
    cov = np.array(
        [
            [1.0, 0.65, 0.35],
            [0.65, 1.0, 0.45],
            [0.35, 0.45, 1.0],
        ],
        dtype=np.float32,
    )
    x = rng.multivariate_normal(np.zeros(3), cov, size=64).astype(np.float32)

    vine = create_vine("c-vine", 3, knots=11)
    vine.fit(
        x,
        gen_dict={"param": False, "binning": False, "fitted": False},
        npc_dict={
            "opt_method": "LL1",
            "max_iter_phase1": 2,
            "max_iter_phase2": 2,
            "normal_iters_phase1": 10,
            "normal_iters_phase2": 10,
            "final_normalization_iters": 50,
            "batch_size": 2,
        },
        par_dict={},
        bin_dict={},
    )

    logp = vine.logpdf(x[:8])
    print("nonparametric vine fitted")
    print("  param flag:", vine.param)
    print("  copula levels:", [len(level) for level in vine.copulas])
    print("  mean logpdf on first 8 rows:", float(logp.mean()))


if __name__ == "__main__":
    main()
