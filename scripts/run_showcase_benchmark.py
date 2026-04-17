#!/usr/bin/env python3
"""Four-phase detection showcase: $d=10$, $T=60$.

Phases (15 windows each):
1. Independent baseline.
2. Pairwise Gaussian block on variables 1--5 ($\\rho \\approx 0.6$).
3. Pairwise block remains on variables 1--5; higher-order continuous-XOR triplet
   is superimposed on variables 6--8 (pairwise marginals of the triplet stay
   near zero but the joint is deterministic).
4. Clayton lower-tail block on variables 1--4; remainder independent.

For each phase boundary, the intent is that DVC's total-correlation decomposition
$\\TC = \\TC_{\\mathrm{pair}} + \\TC_{\\mathrm{higher}}$ should surface different
structures in each phase.  Results are saved to ``results/showcase/summary.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.baselines.gaussian_state_space import (
    gaussian_copula_state_space_nll_fit_eval,
)
from dvc_package.baselines.nf_copula import nf_copula_nll_fit_eval
from dvc_package.experiments.simulation_benchmarks import (
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


def _phase_of_window(t: int) -> int:
    for i in range(len(PHASE_BOUNDARIES) - 1):
        if PHASE_BOUNDARIES[i] <= t < PHASE_BOUNDARIES[i + 1]:
            return i
    return len(PHASES) - 1


def _gen_independent(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, d))


def _gen_pairwise_block(
    n: int, d: int, block_indices: list[int], rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Correlated Gaussian block on ``block_indices``, rest independent."""
    x = rng.standard_normal((n, d))
    k = len(block_indices)
    cov = np.full((k, k), rho)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, k)) @ L.T
    for local_idx, global_idx in enumerate(block_indices):
        x[:, global_idx] = z[:, local_idx]
    return x


def _gen_pairwise_plus_triplet(
    n: int,
    d: int,
    pair_block: list[int],
    rho: float,
    triplet: list[int],
    rng: np.random.Generator,
    *,
    noise_weight: float = 0.15,
) -> np.ndarray:
    """Gaussian pairwise block plus deterministic copula-level XOR triplet.

    Construction for the triplet (indices ``triplet = [a, b, c]``):
    ``u_a, u_b ~ Uniform(0,1)`` independent,
    ``u_c = (u_a + u_b) mod 1``, then ``x_i = \\Phi^{-1}(u_i)`` for ``i \\in {a,b,c}``.
    A small Gaussian jitter ``noise_weight * eps`` is added to ``x_c`` so that
    the pseudo-observations are unique almost surely, without destroying the
    deterministic copula-level relationship.

    All three pairwise marginals of the triplet are Uniform(0,1) and pairwise
    independent; only the \\emph{joint} of the three is deterministic.  Hence
    ``\\mathrm{MI}(X_c, X_b \\mid X_a)`` is large while all pairwise MIs are
    \\emph{zero}.
    """
    from scipy.stats import norm

    x = _gen_pairwise_block(n, d, pair_block, rho, rng)
    a, b, c = triplet
    ua = rng.random(n)
    ub = rng.random(n)
    uc = (ua + ub) % 1.0
    uc = np.clip(uc, 1e-6, 1.0 - 1e-6)
    ua = np.clip(ua, 1e-6, 1.0 - 1e-6)
    ub = np.clip(ub, 1e-6, 1.0 - 1e-6)
    xa = norm.ppf(ua)
    xb = norm.ppf(ub)
    xc = norm.ppf(uc) + noise_weight * rng.standard_normal(n)
    x[:, a] = xa
    x[:, b] = xb
    x[:, c] = xc
    return x


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
        return _gen_pairwise_block(N_PER_TIME, D, block_indices=list(range(5)), rho=0.6, rng=rng)
    if phase == 2:
        return _gen_pairwise_plus_triplet(
            N_PER_TIME,
            D,
            pair_block=list(range(5)),
            rho=0.6,
            triplet=[5, 6, 7],
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


def main() -> None:
    rng = np.random.default_rng(2026)
    data_by_t: List[np.ndarray] = []
    for t in range(T):
        data_by_t.append(_generate_window(t, rng))

    rows: List[Dict[str, float]] = []
    for t in range(T):
        res = _run_window(data_by_t[t], seed=1000 + 17 * t)
        res["t"] = t
        res["phase"] = _phase_of_window(t)
        res["phase_name"] = PHASES[res["phase"]]
        rows.append(res)
        if (t + 1) % 5 == 0 or t == T - 1:
            print(
                f"t={t+1}/{T} phase={PHASES[res['phase']]} "
                f"TC_total={res['tc_total_dvc']:.3f} "
                f"TC_pair={res['tc_pair_dvc']:.3f} "
                f"TC_higher={res['tc_higher_dvc']:.3f}"
            )

    # Gaussian state-space: evaluated as a sequence.
    x_train_by_t = [x[: int(round(TRAIN_FRAC * x.shape[0]))] for x in data_by_t]
    x_test_by_t = [x[int(round(TRAIN_FRAC * x.shape[0])) :] for x in data_by_t]
    try:
        ssm_seq, ssm_fit = gaussian_copula_state_space_nll_fit_eval(x_train_by_t, x_test_by_t)
        ssm_nll = [float(v) for v in ssm_seq]
        ssm_q = float(ssm_fit.process_variance)
    except Exception as exc:
        ssm_nll = [float("nan")] * T
        ssm_q = float("nan")
        print(f"SSM failed: {exc}")
    for t, nll in enumerate(ssm_nll):
        rows[t]["nll_ssm"] = nll
        rows[t]["tc_total_ssm"] = -nll

    # Pairwise MI via MINE for two representative pairs:
    #   - (0, 1): lives inside the pairwise block (phases 2, 3)
    #   - (5, 6): lives inside the higher-order triplet (phase 3 only)
    print("\nRunning MINE for pair (0, 1) ...")
    mine_01 = _run_pairwise_mine(data_by_t, pair=(0, 1))
    print("Running MINE for pair (5, 6) ...")
    mine_56 = _run_pairwise_mine(data_by_t, pair=(5, 6))
    for t in range(T):
        rows[t]["mine_mi_pair01"] = mine_01[t]
        rows[t]["mine_mi_pair56"] = mine_56[t]

    summary = {
        "d": D,
        "T": T,
        "n_per_time": N_PER_TIME,
        "phase_boundaries": PHASE_BOUNDARIES,
        "phase_names": PHASES,
        "ssm_process_variance": ssm_q,
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
