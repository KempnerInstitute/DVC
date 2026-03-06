"""Windowed dependence analysis for real-data sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.objects import cop_par_obj, vine_obj_bin
from ..core.param_copula import parametric_fit
from ..core.vine_factory import create_vine
from ..core.vine_model import fit_vine
from ..optimization.structure import optimize_vine_structure
from ..time import mean_copula_nll
from .allen_vbn import AllenVBNSessionData

logging.getLogger("DVC.vine").setLevel(logging.WARNING)


def _normalize_family_name(family: str) -> str:
    fam = str(family).lower().strip()
    if fam in {"independence", "independent"}:
        return "ind"
    if fam in {"gauss"}:
        return "gaussian"
    if fam in {"student-t", "t"}:
        return "student"
    return fam


def _pseudo_obs_rank(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x)
    n, d = x.shape
    out = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        col = x[:, j]
        ranks = np.argsort(np.argsort(col, kind="mergesort"), kind="mergesort").astype(np.float64) + 1.0
        out[:, j] = ranks / (n + 1.0)
    return np.clip(out, eps, 1.0 - eps).astype(np.float32)


def _fit_best_bivariate_copula(u_train: np.ndarray, families: Sequence[str]) -> cop_par_obj:
    u3 = np.asarray(u_train, dtype=np.float32)[:, :, None]
    aic2, theta_list, _ = parametric_fit(u3, families=list(families), n_cop=1)
    aic = np.asarray(aic2[0], dtype=np.float64)
    best_idx = int(np.nanargmin(aic))
    fam = _normalize_family_name(list(families)[best_idx])
    theta = theta_list[0][best_idx]
    return cop_par_obj(fam, theta)


def _build_cvine_edges(order: List[int]) -> List[List[List[int]]]:
    d = len(order)
    ind_vine: List[List[List[int]]] = []
    for level in range(d - 1):
        root = int(order[level])
        edges_level: List[List[int]] = []
        for j in range(level + 1, d):
            edges_level.append([root, int(order[j])])
        ind_vine.append(edges_level)
    return ind_vine


def _fit_parametric_vine(
    x_train: np.ndarray,
    *,
    families: Sequence[str],
    optimize_structure: bool,
    seed: int,
) -> vine_obj_bin:
    if optimize_structure:
        opt = optimize_vine_structure(
            x_train,
            vine_type="c-vine",
            method="sequential",
            criterion="kendall_tau",
            max_iterations=1,
            verbose=False,
        )
        vine = opt.best_vine
    else:
        vine = create_vine("c-vine", x_train.shape[1])

    gen_dict = {"param": True, "binning": False, "fitted": True}
    npc_dict: Dict[str, Any] = {}
    par_dict = {"param_families": list(families), "seed": int(seed)}
    bin_dict: Dict[str, Any] = {}
    fit_vine(vine, x_train, gen_dict, npc_dict, par_dict, bin_dict)
    return vine


def _cvine_order_from_vine(vine: vine_obj_bin, d: int) -> List[int]:
    roots: List[int] = []
    remaining = set(range(d))
    for level in range(max(d - 1, 0)):
        if level >= len(vine.ind_vine) or not vine.ind_vine[level]:
            break
        root = int(vine.ind_vine[level][0][0])
        if root not in roots:
            roots.append(root)
            remaining.discard(root)
    if remaining:
        roots.extend(sorted(remaining))
    return roots[:d]


def _fit_truncated_cvine_level0(
    x_train: np.ndarray,
    *,
    families: Sequence[str],
    order: Optional[List[int]] = None,
) -> vine_obj_bin:
    x_train = np.asarray(x_train, dtype=np.float32)
    d = int(x_train.shape[1])
    if order is None:
        order = list(range(d))

    fams = sorted({_normalize_family_name(f) for f in families} | {"ind"})
    vine = create_vine("c-vine", d, families=fams)
    vine.ind_vine = _build_cvine_edges(order)
    u = _pseudo_obs_rank(x_train)

    copulas: List[List[cop_par_obj]] = []
    level0: List[cop_par_obj] = []
    for edge in vine.ind_vine[0]:
        i, j = int(edge[0]), int(edge[1])
        level0.append(_fit_best_bivariate_copula(u[:, [i, j]], families=families))
    copulas.append(level0)

    for level in range(1, d - 1):
        copulas.append([cop_par_obj("ind", None) for _ in vine.ind_vine[level]])

    vine.copulas = copulas
    vine.param = True
    vine.fitted = True
    return vine


def _mean_abs_corr(x: np.ndarray) -> float:
    R = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.eye(R.shape[0], dtype=bool)
    vals = np.abs(R[mask])
    return float(np.mean(vals)) if vals.size else 0.0


@dataclass
class WindowedDependenceResult:
    session_id: int
    mouse_id: str
    experience_level: str
    stimulus_table_name: str
    region_names: List[str]
    region_unit_counts: Dict[str, int]
    window_size: int
    stride: int
    response_window: List[float]
    window_start_index: List[int]
    window_center_index: List[int]
    window_center_time: List[float]
    dvc_nll: List[float]
    truncated_level0_nll: List[float]
    nll_gap_truncated_level0: List[float]
    mean_abs_corr: List[float]
    change_fraction: List[float]
    rewarded_fraction: List[float]
    omitted_fraction: List[float]
    mean_gap: float
    std_gap: float
    positive_gap_fraction: float
    gap_change_corr: float
    n_presentations_selected: int


def analyze_session_windows(
    session: AllenVBNSessionData,
    *,
    families: Optional[Sequence[str]] = None,
    window_size: int = 120,
    stride: int = 60,
    train_fraction: float = 0.7,
    seed: int = 0,
    optimize_structure: bool = True,
) -> WindowedDependenceResult:
    families = list(families or ["independence", "gaussian", "frank"])
    x = np.log1p(np.asarray(session.presentation_matrix, dtype=np.float32))
    presentations = session.presentations.reset_index(drop=True)
    n = x.shape[0]
    if n < window_size:
        raise ValueError(f"Need at least {window_size} presentations, got {n}")

    starts = list(range(0, n - window_size + 1, stride))
    rng = np.random.default_rng(seed)
    dvc_nll: List[float] = []
    trunc_nll: List[float] = []
    gap: List[float] = []
    mean_abs_corr: List[float] = []
    centers: List[int] = []
    center_times: List[float] = []
    change_fraction: List[float] = []
    rewarded_fraction: List[float] = []
    omitted_fraction: List[float] = []

    for start in starts:
        stop = start + window_size
        x_window = np.asarray(x[start:stop], dtype=np.float32)
        x_window = x_window + 1e-3 * rng.standard_normal(x_window.shape).astype(np.float32)
        n_train = max(20, int(round(window_size * float(train_fraction))))
        n_train = min(n_train, window_size - 20)
        x_train = x_window[:n_train]
        x_test = x_window[n_train:]

        vine = _fit_parametric_vine(
            x_train,
            families=families,
            optimize_structure=optimize_structure,
            seed=seed,
        )
        order = _cvine_order_from_vine(vine, d=x_train.shape[1])
        trunc_vine = _fit_truncated_cvine_level0(
            x_train,
            families=families,
            order=order,
        )

        dvc_val = float(mean_copula_nll(vine, x_test))
        trunc_val = float(mean_copula_nll(trunc_vine, x_test))
        dvc_nll.append(dvc_val)
        trunc_nll.append(trunc_val)
        gap.append(trunc_val - dvc_val)
        mean_abs_corr.append(_mean_abs_corr(x_window))

        center = start + window_size // 2
        centers.append(center)
        center_times.append(float(presentations.iloc[center]["start_time"]))

        window_df = presentations.iloc[start:stop]
        if "is_change" in window_df.columns:
            change_fraction.append(float(np.mean(window_df["is_change"].to_numpy(dtype=np.float64) > 0.5)))
        else:
            change_fraction.append(0.0)
        if "rewarded" in window_df.columns:
            rewarded_fraction.append(float(np.mean(window_df["rewarded"].to_numpy(dtype=np.float64) > 0.5)))
        else:
            rewarded_fraction.append(0.0)
        if "omitted" in window_df.columns:
            omitted_fraction.append(float(np.mean(window_df["omitted"].to_numpy(dtype=np.float64) > 0.5)))
        else:
            omitted_fraction.append(0.0)

    gap_arr = np.asarray(gap, dtype=np.float64)
    change_arr = np.asarray(change_fraction, dtype=np.float64)
    finite_gap = gap_arr[np.isfinite(gap_arr)]
    if gap_arr.size >= 2 and np.nanstd(change_arr) > 0 and np.nanstd(gap_arr) > 0:
        gap_change_corr = float(np.corrcoef(gap_arr, change_arr)[0, 1])
    else:
        gap_change_corr = float("nan")

    return WindowedDependenceResult(
        session_id=int(session.session_id),
        mouse_id=str(session.mouse_id),
        experience_level=str(session.experience_level),
        stimulus_table_name=str(session.stimulus_table_name),
        region_names=list(session.region_names),
        region_unit_counts=dict(session.summary.region_unit_counts),
        window_size=int(window_size),
        stride=int(stride),
        response_window=[0.0, 0.25],
        window_start_index=[int(v) for v in starts],
        window_center_index=[int(v) for v in centers],
        window_center_time=[float(v) for v in center_times],
        dvc_nll=[float(v) for v in dvc_nll],
        truncated_level0_nll=[float(v) for v in trunc_nll],
        nll_gap_truncated_level0=[float(v) for v in gap],
        mean_abs_corr=[float(v) for v in mean_abs_corr],
        change_fraction=[float(v) for v in change_fraction],
        rewarded_fraction=[float(v) for v in rewarded_fraction],
        omitted_fraction=[float(v) for v in omitted_fraction],
        mean_gap=float(np.mean(finite_gap)) if finite_gap.size else float("nan"),
        std_gap=float(np.std(finite_gap)) if finite_gap.size else float("nan"),
        positive_gap_fraction=float(np.mean(finite_gap > 0.0)) if finite_gap.size else float("nan"),
        gap_change_corr=gap_change_corr,
        n_presentations_selected=int(session.presentation_matrix.shape[0]),
    )


def result_to_dict(result: WindowedDependenceResult) -> Dict[str, Any]:
    return asdict(result)


def cohort_summary_table(results: Sequence[WindowedDependenceResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        mean_abs_corr = np.asarray(result.mean_abs_corr, dtype=np.float64)
        rows.append(
            {
                "session_id": int(result.session_id),
                "mouse_id": str(result.mouse_id),
                "experience_level": str(result.experience_level),
                "stimulus_table_name": str(result.stimulus_table_name),
                "n_regions": int(len(result.region_names)),
                "n_presentations_selected": int(result.n_presentations_selected),
                "mean_gap": float(result.mean_gap),
                "std_gap": float(result.std_gap),
                "positive_gap_fraction": float(result.positive_gap_fraction),
                "gap_change_corr": float(result.gap_change_corr),
                "mean_abs_corr_mean": float(np.mean(mean_abs_corr)) if mean_abs_corr.size else float("nan"),
                "mean_abs_corr_std": float(np.std(mean_abs_corr)) if mean_abs_corr.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_cohort_results(results: Sequence[WindowedDependenceResult]) -> Dict[str, Any]:
    table = cohort_summary_table(results)
    session_rows = table.to_dict(orient="records")

    experience_summary: Dict[str, Dict[str, float]] = {}
    if not table.empty and "experience_level" in table.columns:
        for exp, group in table.groupby("experience_level", dropna=False):
            label = str(exp) if str(exp) else "unknown"
            experience_summary[label] = {
                "n_sessions": int(len(group)),
                "mean_gap_mean": float(group["mean_gap"].mean()),
                "mean_gap_std": float(group["mean_gap"].std(ddof=0)),
                "mean_abs_corr_mean": float(group["mean_abs_corr_mean"].mean()),
                "mean_abs_corr_std": float(group["mean_abs_corr_mean"].std(ddof=0)),
                "positive_gap_fraction_mean": float(group["positive_gap_fraction"].mean()),
            }

    paired_rows: List[Dict[str, Any]] = []
    if not table.empty:
        for mouse_id, group in table.groupby("mouse_id", dropna=False):
            fam = group[group["experience_level"] == "Familiar"]
            nov = group[group["experience_level"] == "Novel"]
            if fam.empty or nov.empty:
                continue
            fam_row = fam.iloc[0]
            nov_row = nov.iloc[0]
            paired_rows.append(
                {
                    "mouse_id": str(mouse_id),
                    "familiar_session_id": int(fam_row["session_id"]),
                    "novel_session_id": int(nov_row["session_id"]),
                    "delta_mean_gap_novel_minus_familiar": float(
                        nov_row["mean_gap"] - fam_row["mean_gap"]
                    ),
                    "delta_mean_abs_corr_novel_minus_familiar": float(
                        nov_row["mean_abs_corr_mean"] - fam_row["mean_abs_corr_mean"]
                    ),
                }
            )

    paired_summary: Dict[str, float] = {}
    if paired_rows:
        paired = pd.DataFrame(paired_rows)
        paired_summary = {
            "n_mice": int(len(paired)),
            "delta_mean_gap_mean": float(paired["delta_mean_gap_novel_minus_familiar"].mean()),
            "delta_mean_gap_std": float(paired["delta_mean_gap_novel_minus_familiar"].std(ddof=0)),
            "delta_mean_abs_corr_mean": float(
                paired["delta_mean_abs_corr_novel_minus_familiar"].mean()
            ),
            "delta_mean_abs_corr_std": float(
                paired["delta_mean_abs_corr_novel_minus_familiar"].std(ddof=0)
            ),
        }

    return {
        "n_sessions": int(len(table)),
        "n_mice_with_pairs": int(len(paired_rows)),
        "session_table": session_rows,
        "experience_summary": experience_summary,
        "paired_mouse_deltas": paired_rows,
        "paired_summary": paired_summary,
    }


def plot_pilot_summary(
    results: Sequence[WindowedDependenceResult],
    *,
    out_path: str,
) -> str:
    if not results:
        raise ValueError("Need at least one result to plot")

    fig, axes = plt.subplots(
        len(results),
        2,
        figsize=(10.0, 3.0 * len(results)),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, result in enumerate(results):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]
        x = np.asarray(result.window_center_index, dtype=np.int64)
        gap = np.asarray(result.nll_gap_truncated_level0, dtype=np.float64)
        corr = np.asarray(result.mean_abs_corr, dtype=np.float64)
        change = np.asarray(result.change_fraction, dtype=np.float64)

        ax_left.plot(x, corr, color="#1f77b4", linewidth=1.8, label="Mean abs corr")
        ax_left.set_ylabel("Mean abs corr")
        ax_left.set_xlabel("Presentation index")
        ax_left.set_title(
            f"Session {result.session_id} ({result.experience_level or 'unknown'})",
            fontsize=10,
        )
        ax_left_t = ax_left.twinx()
        ax_left_t.plot(x, change, color="#d95f02", linewidth=1.2, linestyle=":", label="Change fraction")
        ax_left_t.set_ylabel("Change fraction")

        ax_right.plot(x, gap, color="#6a3d9a", linewidth=1.8)
        ax_right.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
        ax_right.set_xlabel("Presentation index")
        ax_right.set_ylabel("Higher-order gap")
        ax_right.set_title(
            f"mean gap={result.mean_gap:+.3f} ± {result.std_gap:.3g}",
            fontsize=10,
        )
        label = (
            f"positive={result.positive_gap_fraction:.1%}\n"
            f"corr(gap, change)={result.gap_change_corr:+.2f}"
        )
        ax_right.text(
            0.03,
            0.97,
            label,
            transform=ax_right.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
        )

    out = str(out_path)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_cohort_summary(
    results: Sequence[WindowedDependenceResult],
    *,
    out_path: str,
) -> str:
    table = cohort_summary_table(results)
    if table.empty:
        raise ValueError("Need at least one result to plot")

    levels = ["Familiar", "Novel"]
    xpos = {"Familiar": 0.0, "Novel": 1.0}
    colors = {"Familiar": "#1f77b4", "Novel": "#d95f02"}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    metrics = [
        ("mean_gap", "Mean higher-order gap (nats)"),
        ("mean_abs_corr_mean", "Mean abs corr"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for mouse_id, group in table.groupby("mouse_id", dropna=False):
            group = group.sort_values("experience_level")
            xs: List[float] = []
            ys: List[float] = []
            for level in levels:
                sub = group[group["experience_level"] == level]
                if sub.empty:
                    continue
                xs.append(xpos[level])
                ys.append(float(sub.iloc[0][metric]))
            if len(xs) == 2:
                ax.plot(xs, ys, color="0.75", linewidth=1.0, zorder=1)
            for level in levels:
                sub = group[group["experience_level"] == level]
                if sub.empty:
                    continue
                ax.scatter(
                    xpos[level],
                    float(sub.iloc[0][metric]),
                    color=colors[level],
                    edgecolor="white",
                    linewidth=0.5,
                    s=40,
                    zorder=2,
                )

        grouped = table.groupby("experience_level")[metric].mean()
        for level in levels:
            if level not in grouped:
                continue
            ax.hlines(
                float(grouped[level]),
                xpos[level] - 0.18,
                xpos[level] + 0.18,
                color="black",
                linewidth=2.0,
                zorder=3,
            )

        ax.set_xticks([xpos[level] for level in levels])
        ax.set_xticklabels(levels)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")

    axes[0].set_title("Allen VBN cohort: higher-order residual", fontsize=10)
    axes[1].set_title("Allen VBN cohort: pairwise correlation", fontsize=10)

    out = str(out_path)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
