#!/usr/bin/env python3
"""Four-phase detection showcase: $d=10$, $T=60$.

Phases (15 windows each):
1. Independent baseline.
2. Pairwise Gaussian star block rooted at $X_0$ with leaves $(X_1,\dots,X_4)$.
3. The star block remains, and a root-aligned higher-order C-vine is added on
   $(X_0, X_5, X_6)$ so that a genuine conditional edge is accessible to the
   fitted C-vine's higher tree levels.
4. Clayton lower-tail block on variables $X_0,\dots,X_3$; remainder independent.

For each phase boundary, the intent is that DVC's total-correlation decomposition
$\\TC = \\TC_{\\mathrm{pair}} + \\TC_{\\mathrm{higher}}$ should surface different
structures in each phase.  Results are saved to ``results/showcase/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.baselines.gaussian_state_space import (
    gaussian_copula_state_space_nll_fit_eval,
)
from dvc_package.baselines.nf_copula import nf_copula_nll_fit_eval
from dvc_package.experiments.simulation_benchmarks import (
    _embed_higher_order_vine,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
    _gaussian_copula_nll_fit_eval,
    _mean_copula_nll,
    _pseudo_obs_rank,
)


D = 10
T = 60
N_PER_TIME = 300
TRAIN_FRAC = 0.85
PHASE_BOUNDARIES = [0, 15, 30, 45, 60]  # phase 1 = [0,15), etc.
PHASES = ["independent", "pairwise-block", "pairwise+higher-order", "tail-block"]
FAMILIES = ["gaussian", "student", "clayton", "gumbel", "independence"]

OUT = Path("results/showcase")
OUT.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DVC four-phase showcase benchmark.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of independent benchmark seeds to aggregate.")
    parser.add_argument("--base-seed", type=int, default=2026, help="Base seed for the showcase.")
    parser.add_argument("--out", type=Path, default=OUT, help="Output directory for summary.json.")
    return parser.parse_args()


def _phase_of_window(t: int) -> int:
    for i in range(len(PHASE_BOUNDARIES) - 1):
        if PHASE_BOUNDARIES[i] <= t < PHASE_BOUNDARIES[i + 1]:
            return i
    return len(PHASES) - 1


def _gen_independent(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, d))


def _gen_pairwise_star_block(
    n: int,
    d: int,
    *,
    root_index: int,
    leaf_indices: list[int],
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian star block with conditional independence given the root."""
    x = rng.standard_normal((n, d))
    root = rng.standard_normal(n)
    x[:, root_index] = root
    scale = np.sqrt(max(1e-8, 1.0 - rho**2))
    for idx in leaf_indices:
        x[:, idx] = rho * root + scale * rng.standard_normal(n)
    return x


def _gen_pairwise_plus_triplet(
    n: int,
    d: int,
    pair_root: int,
    pair_leaves: list[int],
    rho: float,
    triplet: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Pairwise star block plus root-aligned higher-order C-vine triplet."""
    x = _gen_pairwise_star_block(
        n,
        d,
        root_index=pair_root,
        leaf_indices=pair_leaves,
        rho=rho,
        rng=rng,
    )
    return _embed_higher_order_vine(
        x,
        agents=triplet,
        rho=0.65,
        nu=4.5,
        rng=rng,
        eps=1e-6,
    )


def _gen_tail_block(
    n: int,
    d: int,
    block_indices: list[int],
    theta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Clayton-copula lower-tail block on ``block_indices`` via conditional CDF inversion."""
    from scipy.stats import norm

    x = rng.standard_normal((n, d))
    k = len(block_indices)
    u = np.zeros((n, k), dtype=np.float64)
    u[:, 0] = rng.random(n)
    for j in range(1, k):
        w = rng.random(n)
        u[:, j] = (1.0 + (w ** (-theta / (1.0 + theta)) - 1.0) * u[:, 0] ** (-theta)) ** (
            -1.0 / theta
        )
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    z = norm.ppf(u)
    for local_idx, global_idx in enumerate(block_indices):
        x[:, global_idx] = z[:, local_idx]
    return x


def _generate_window(t: int, rng: np.random.Generator) -> np.ndarray:
    phase = _phase_of_window(t)
    if phase == 0:
        return _gen_independent(N_PER_TIME, D, rng)
    if phase == 1:
        return _gen_pairwise_star_block(
            N_PER_TIME,
            D,
            root_index=0,
            leaf_indices=[1, 2, 3, 4],
            rho=0.7,
            rng=rng,
        )
    if phase == 2:
        return _gen_pairwise_plus_triplet(
            N_PER_TIME,
            D,
            pair_root=0,
            pair_leaves=[1, 2, 3, 4],
            rho=0.7,
            triplet=[0, 5, 6],
            rng=rng,
        )
    return _gen_tail_block(N_PER_TIME, D, block_indices=list(range(4)), theta=1.5, rng=rng)


def _run_window(x: np.ndarray, seed: int) -> Dict[str, float]:
    n = x.shape[0]
    split = int(round(TRAIN_FRAC * n))
    tr = x[:split]
    te = x[split:]

    out: Dict[str, float] = {}
    t0 = time.perf_counter()
    vine = _fit_parametric_vine(tr, families=FAMILIES, optimize_structure=False, seed=seed)
    out["dvc_time_s"] = time.perf_counter() - t0

    nll_dvc = _mean_copula_nll(vine, te)
    trunc_vine = _fit_truncated_cvine_level0(
        tr, families=FAMILIES, order=list(range(D))
    )
    nll_trunc = _mean_copula_nll(trunc_vine, te)
    nll_gauss = _gaussian_copula_nll_fit_eval(tr, te)

    out["nll_dvc"] = float(nll_dvc)
    out["nll_trunc_level0"] = float(nll_trunc)
    out["nll_gauss"] = float(nll_gauss)

    # TC components.  In nats:
    #   TC_total  = -E[log c_full(u)]  (=  -nll_dvc evaluated as expectation)
    #   TC_pair   = -E[log c_{1-trunc}(u)]
    #   TC_higher = TC_total - TC_pair
    out["tc_total_dvc"] = -out["nll_dvc"]
    out["tc_pair_dvc"] = -out["nll_trunc_level0"]
    out["tc_higher_dvc"] = out["tc_total_dvc"] - out["tc_pair_dvc"]
    out["tc_gauss"] = -out["nll_gauss"]

    try:
        out["nll_nf"] = float(
            nf_copula_nll_fit_eval(tr, te, n_epochs=40, hidden_dim=32, n_blocks=4, seed=seed)
        )
    except Exception:
        out["nll_nf"] = float("nan")
    out["tc_total_nf"] = -out["nll_nf"]

    return out


def _run_pairwise_mine(x_by_t: List[np.ndarray], pair: tuple[int, int]) -> List[float]:
    """Time-dependent MI estimate for one pair using MINE per window."""
    from dvc_package.baselines.mine import mine_mi_estimate

    out: List[float] = []
    for t, x in enumerate(x_by_t):
        try:
            mi = mine_mi_estimate(
                x[:, pair[0]], x[:, pair[1]], n_epochs=60, seed=2026 + t
            )
        except Exception:
            mi = float("nan")
        out.append(float(mi))
    return out


def _run_pairwise_gaussian_mi(x_by_t: List[np.ndarray], pair: tuple[int, int]) -> List[float]:
    """DVC-native pairwise MI estimate via Kendall-tau Gaussian approximation."""
    import pandas as pd

    out: List[float] = []
    for x in x_by_t:
        x_a = x[:, pair[0]]
        x_b = x[:, pair[1]]
        tau = float(pd.Series(x_a).corr(pd.Series(x_b), method="kendall"))
        if not np.isfinite(tau):
            out.append(float("nan"))
            continue
        tau = float(np.clip(tau, -0.999, 0.999))
        rho = np.clip(np.sin(np.pi * tau / 2.0), -0.999, 0.999)
        out.append(float(-0.5 * np.log(1.0 - rho**2)))
    return out


def _run_single_seed(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    data_by_t: List[np.ndarray] = []
    for t in range(T):
        data_by_t.append(_generate_window(t, rng))

    rows: List[Dict[str, float]] = []
    for t in range(T):
        res = _run_window(data_by_t[t], seed=seed + 1000 + 17 * t)
        res["t"] = t
        res["phase"] = _phase_of_window(t)
        res["phase_name"] = PHASES[res["phase"]]
        rows.append(res)

    x_train_by_t = [x[: int(round(TRAIN_FRAC * x.shape[0]))] for x in data_by_t]
    x_test_by_t = [x[int(round(TRAIN_FRAC * x.shape[0])) :] for x in data_by_t]
    try:
        ssm_seq, ssm_fit = gaussian_copula_state_space_nll_fit_eval(x_train_by_t, x_test_by_t)
        ssm_nll = [float(v) for v in ssm_seq]
        ssm_q = float(ssm_fit.process_variance)
    except Exception as exc:
        ssm_nll = [float("nan")] * T
        ssm_q = float("nan")
        print(f"SSM failed for seed {seed}: {exc}")
    for t, nll in enumerate(ssm_nll):
        rows[t]["nll_ssm"] = nll
        rows[t]["tc_total_ssm"] = -nll

    print(f"\nRunning MINE for pair (0, 1) [seed {seed}] ...")
    mine_01 = _run_pairwise_mine(data_by_t, pair=(0, 1))
    print(f"Running MINE for pair (5, 6) [seed {seed}] ...")
    mine_56 = _run_pairwise_mine(data_by_t, pair=(5, 6))
    dvc_mi_01 = _run_pairwise_gaussian_mi(data_by_t, pair=(0, 1))
    dvc_mi_56 = _run_pairwise_gaussian_mi(data_by_t, pair=(5, 6))
    for t in range(T):
        rows[t]["mine_mi_pair01"] = mine_01[t]
        rows[t]["mine_mi_pair56"] = mine_56[t]
        rows[t]["dvc_pair_mi01"] = dvc_mi_01[t]
        rows[t]["dvc_pair_mi56"] = dvc_mi_56[t]

    for phase_idx, phase_name in enumerate(PHASES):
        phase_rows = [r for r in rows if r["phase"] == phase_idx]
        tc_mean = np.mean([r["tc_total_dvc"] for r in phase_rows])
        higher_mean = np.mean([r["tc_higher_dvc"] for r in phase_rows])
        print(
            f"seed={seed} phase={phase_name:<22} "
            f"TC_total={tc_mean:+.3f} TC_higher={higher_mean:+.3f}"
        )

    return {"seed": seed, "ssm_process_variance": ssm_q, "rows": rows}


def _aggregate_runs(seed_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    numeric_keys = sorted(
        {
            key
            for run in seed_runs
            for row in run["rows"]
            for key, value in row.items()
            if isinstance(value, (int, float, np.floating))
        }
        - {"t", "phase"}
    )

    rows: List[Dict[str, Any]] = []
    for t in range(T):
        ref = seed_runs[0]["rows"][t]
        agg_row: Dict[str, Any] = {
            "t": int(ref["t"]),
            "phase": int(ref["phase"]),
            "phase_name": ref["phase_name"],
        }
        for key in numeric_keys:
            vals = np.asarray(
                [run["rows"][t].get(key, np.nan) for run in seed_runs],
                dtype=np.float64,
            )
            agg_row[key] = float(np.nanmean(vals))
            agg_row[f"{key}_std"] = float(np.nanstd(vals))
        rows.append(agg_row)

    ssm_vars = np.asarray([run["ssm_process_variance"] for run in seed_runs], dtype=np.float64)
    return {
        "d": D,
        "T": T,
        "n_per_time": N_PER_TIME,
        "phase_boundaries": PHASE_BOUNDARIES,
        "phase_names": PHASES,
        "n_seeds": len(seed_runs),
        "seeds": [int(run["seed"]) for run in seed_runs],
        "ssm_process_variance": float(np.nanmean(ssm_vars)),
        "ssm_process_variance_std": float(np.nanstd(ssm_vars)),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(args.base_seed + 97 * i) for i in range(args.n_seeds)]
    seed_runs = [_run_single_seed(seed) for seed in seeds]
    summary = _aggregate_runs(seed_runs)
    out_path = args.out / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
