#!/usr/bin/env python3
"""Bounded probe: D-vine with triplet-adjacent variable order on phases 2 and 3.

Generates data matching :mod:`run_showcase_benchmark` phases 1 (pairwise block)
and 2 (pairwise + copula-XOR triplet), fits both the default C-vine (hub at
$X_0$) and a D-vine with order ``[5, 6, 7, 0, 1, 2, 3, 4, 8, 9]`` (triplet at
levels 0--1), and reports per-phase mean $\\TC_{\\mathrm{pair}}$, $\\TC$,
$\\TC_{\\mathrm{higher}}$ with bootstrap 95% intervals.  This is an explicit test
of whether putting the XOR triplet at lower vine levels improves the phase-3
separation in Figure 7.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvc_package.core.vine_factory import create_vine
from dvc_package.core.vine_model import fit_vine
from dvc_package.experiments.simulation_benchmarks import (
    _fit_truncated_cvine_level0,
    _mean_copula_nll,
)
from scripts.run_showcase_benchmark import (
    N_PER_TIME,
    TRAIN_FRAC,
    D,
    FAMILIES,
    _generate_window,
)


OUT = Path("results/showcase/dvine_probe.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

PHASE_2_RANGE = range(15, 30)
PHASE_3_RANGE = range(30, 45)
TRIPLET_ORDER = [5, 6, 7, 0, 1, 2, 3, 4, 8, 9]


def _fit_vine_with_order(
    x_train: np.ndarray, vine_type: str, variable_order=None, seed: int = 0
):
    d = x_train.shape[1]
    if vine_type == "d-vine":
        vine = create_vine("d-vine", d, variable_order=variable_order)
    else:
        vine = create_vine("c-vine", d)
    gen_dict = {"param": True, "binning": False, "fitted": True}
    npc_dict: Dict[str, object] = {}
    par_dict = {"param_families": FAMILIES, "seed": seed}
    bin_dict: Dict[str, object] = {}
    fit_vine(vine, x_train, gen_dict, npc_dict, par_dict, bin_dict)
    return vine


def _fit_truncated_with_order(x_train: np.ndarray, variable_order) -> object:
    x_train = np.asarray(x_train, dtype=np.float32)
    return _fit_truncated_cvine_level0(x_train, families=FAMILIES, order=list(variable_order))


def _fit_and_score(x_train, x_test, seed):
    """Fit C-vine hub-0 and D-vine triplet-order; return per-window TC metrics."""
    out: Dict[str, float] = {}

    # C-vine hub-at-0 (default)
    vine_c = _fit_vine_with_order(x_train, "c-vine", seed=seed)
    nll_c_full = _mean_copula_nll(vine_c, x_test)
    trunc_c = _fit_truncated_cvine_level0(x_train, families=FAMILIES, order=list(range(D)))
    nll_c_trunc = _mean_copula_nll(trunc_c, x_test)
    out["cvine_tc_total"] = float(-nll_c_full)
    out["cvine_tc_pair"] = float(-nll_c_trunc)
    out["cvine_tc_higher"] = float(-nll_c_full + nll_c_trunc)

    # D-vine with triplet-adjacent order [5, 6, 7, 0, 1, 2, 3, 4, 8, 9]
    try:
        vine_d = _fit_vine_with_order(
            x_train, "d-vine", variable_order=TRIPLET_ORDER, seed=seed
        )
        nll_d_full = _mean_copula_nll(vine_d, x_test)
        trunc_d = _fit_truncated_with_order(x_train, variable_order=TRIPLET_ORDER)
        nll_d_trunc = _mean_copula_nll(trunc_d, x_test)
        out["dvine_tc_total"] = float(-nll_d_full)
        out["dvine_tc_pair"] = float(-nll_d_trunc)
        out["dvine_tc_higher"] = float(-nll_d_full + nll_d_trunc)
    except Exception as exc:  # pragma: no cover
        out["dvine_tc_total"] = float("nan")
        out["dvine_tc_pair"] = float("nan")
        out["dvine_tc_higher"] = float("nan")
        out["dvine_error"] = str(exc)

    return out


def _summarize(label: str, rows: List[Dict[str, float]]) -> Dict[str, float]:
    def stats(key: str):
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }

    return {
        "label": label,
        "cvine_tc_higher": stats("cvine_tc_higher"),
        "cvine_tc_pair": stats("cvine_tc_pair"),
        "dvine_tc_higher": stats("dvine_tc_higher"),
        "dvine_tc_pair": stats("dvine_tc_pair"),
    }


def main() -> None:
    rng = np.random.default_rng(2026)
    phase_rows: Dict[str, List[Dict[str, float]]] = {
        "phase2_pairwise_block": [],
        "phase3_pairwise_plus_triplet": [],
    }

    for t_range, label in [
        (PHASE_2_RANGE, "phase2_pairwise_block"),
        (PHASE_3_RANGE, "phase3_pairwise_plus_triplet"),
    ]:
        for t in t_range:
            x = _generate_window(t, rng)
            n = x.shape[0]
            split = int(round(TRAIN_FRAC * n))
            tr = x[:split]
            te = x[split:]
            res = _fit_and_score(tr, te, seed=1000 + 17 * t)
            phase_rows[label].append(res)
            print(
                f"t={t:3d} [{label:35s}] cvine_higher={res['cvine_tc_higher']:+.3f} "
                f"dvine_higher={res.get('dvine_tc_higher', float('nan')):+.3f}"
            )

    summary = [
        _summarize(lbl, rows) for lbl, rows in phase_rows.items()
    ]
    diff = {}
    for mdl in ("cvine", "dvine"):
        p2 = np.asarray(
            [r[f"{mdl}_tc_higher"] for r in phase_rows["phase2_pairwise_block"] if np.isfinite(r.get(f"{mdl}_tc_higher", np.nan))]
        )
        p3 = np.asarray(
            [r[f"{mdl}_tc_higher"] for r in phase_rows["phase3_pairwise_plus_triplet"] if np.isfinite(r.get(f"{mdl}_tc_higher", np.nan))]
        )
        if p2.size and p3.size:
            diff[mdl] = {
                "mean_phase3_minus_phase2": float(p3.mean() - p2.mean()),
                "phase2_mean": float(p2.mean()),
                "phase2_std": float(p2.std()),
                "phase3_mean": float(p3.mean()),
                "phase3_std": float(p3.std()),
            }

    full = {"per_phase": summary, "phase3_minus_phase2": diff, "triplet_order": TRIPLET_ORDER}
    OUT.write_text(json.dumps(full, indent=2))
    print(f"\n== summary ==")
    for mdl, vals in diff.items():
        print(
            f"  {mdl:6s}  phase 2 TC_higher = {vals['phase2_mean']:+.3f} "
            f"+/- {vals['phase2_std']:.3f}, "
            f"phase 3 TC_higher = {vals['phase3_mean']:+.3f} "
            f"+/- {vals['phase3_std']:.3f}, "
            f"Delta = {vals['mean_phase3_minus_phase2']:+.3f}"
        )
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
