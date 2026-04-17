"""Real-world finance benchmark around crisis periods.

This module adds a practical stress-test for Dynamic Vine Copulas (DVC):
- multi-asset daily returns
- rolling-window held-out copula NLL
- crisis-period diagnostics (NLL-gap spikes, tail dependence, family switches)

Data sources supported:
- Stooq CSV endpoint (indices/ETFs/etc.)
- FRED CSV endpoint
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import urllib.parse

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from ...baselines.gaussian_state_space import gaussian_copula_state_space_nll_fit_eval
from ...baselines.nf_copula import nf_copula_nll_fit_eval
from ..simulation_benchmarks import (
    _empirical_tail_dependence,
    _fit_parametric_vine,
    _fit_truncated_cvine_level0,
    _gaussian_copula_nll_fit_eval,
    _glasso_gaussian_copula_nll_fit_eval,
    _mean_copula_nll,
    _normalize_family_name,
    _pseudo_obs_rank,
    _taildeps_from_copula,
    _tvgl_gaussian_copula_nll_fit_eval,
)


DEFAULT_FAMILY_CODEBOOK = ["ind", "gaussian", "student", "clayton", "frank", "gumbel", "joe"]
GAP_FIELDS = {
    "nll_gap": "Gaussian copula",
    "nll_gap_truncated_level0": "1-truncated C-vine",
    "nll_gap_glasso": "Graphical Lasso",
    "nll_gap_tvgl": "TVGL (Frobenius)",
    "nll_gap_state_space": "Gaussian SSM",
    "nll_gap_nf_copula": "NF copula (Real-NVP)",
}


@dataclass(frozen=True)
class CrisisPeriod:
    name: str
    start: str
    end: str


def _default_assets() -> List[Dict[str, str]]:
    """Default multi-asset panel with known regime shifts around crises."""
    return [
        {"name": "SPY", "source": "stooq", "symbol": "spy.us", "price_column": "Close"},
        {"name": "TLT", "source": "stooq", "symbol": "tlt.us", "price_column": "Close"},
        {"name": "GLD", "source": "stooq", "symbol": "gld.us", "price_column": "Close"},
        {"name": "USO", "source": "stooq", "symbol": "uso.us", "price_column": "Close"},
        {"name": "UUP", "source": "stooq", "symbol": "uup.us", "price_column": "Close"},
        {"name": "VIX", "source": "fred", "symbol": "VIXCLS"},
    ]


def _default_crisis_periods() -> List[CrisisPeriod]:
    return [
        CrisisPeriod(name="GFC", start="2008-09-01", end="2009-06-30"),
        CrisisPeriod(name="COVID", start="2020-02-15", end="2020-06-30"),
    ]


def _sanitize_token(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return out.strip("_") or "series"


def _stooq_url(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={urllib.parse.quote(symbol.lower())}&i=d"


def _fred_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={urllib.parse.quote(series_id)}"


def _load_price_series(
    asset: Dict[str, Any],
    data_dir: Path,
    refresh_data: bool = False,
    min_points: int = 200,
) -> pd.Series:
    name = str(asset.get("name") or asset.get("symbol") or "asset")
    source = str(asset.get("source", "stooq")).strip().lower()
    symbol = str(asset.get("symbol", "")).strip()
    if not symbol:
        raise ValueError(f"Asset '{name}' is missing symbol")

    data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = data_dir / f"{_sanitize_token(source)}_{_sanitize_token(symbol)}.csv"

    raw: Optional[pd.DataFrame] = None
    if cache_path.exists() and not refresh_data:
        raw = pd.read_csv(cache_path)
    else:
        if source == "stooq":
            url = _stooq_url(symbol)
        elif source == "fred":
            url = _fred_url(symbol)
        else:
            raise ValueError(f"Unsupported data source '{source}' for asset '{name}'")

        try:
            raw = pd.read_csv(url)
            raw.to_csv(cache_path, index=False)
        except Exception as exc:
            if cache_path.exists():
                raw = pd.read_csv(cache_path)
            else:
                raise RuntimeError(
                    f"Failed to download {name} ({source}:{symbol}) and no cache exists. "
                    f"Original error: {exc}"
                ) from exc

    if raw is None or raw.empty:
        raise ValueError(f"No rows for asset '{name}' ({source}:{symbol})")

    if source == "stooq":
        date_col = str(asset.get("date_column", "Date"))
        value_col = str(asset.get("price_column", "Close"))
    else:
        date_col = str(asset.get("date_column", "DATE"))
        value_col = str(asset.get("price_column", symbol))
        if date_col not in raw.columns:
            # FRED CSVs are often `observation_date,<series_id>` rather than `DATE`.
            for cand in ["observation_date", "DATE", "date", "Date"]:
                if cand in raw.columns:
                    date_col = cand
                    break
            if date_col not in raw.columns and len(raw.columns) >= 1:
                date_col = str(raw.columns[0])
        if value_col not in raw.columns and len(raw.columns) >= 2:
            value_col = str(raw.columns[1])

    if date_col not in raw.columns:
        raise ValueError(f"Missing date column '{date_col}' in {name} ({source}:{symbol})")
    if value_col not in raw.columns:
        raise ValueError(f"Missing value column '{value_col}' in {name} ({source}:{symbol})")

    df = raw[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    df = df[df[value_col] > 0.0]
    if len(df) < min_points:
        raise ValueError(
            f"Asset '{name}' has only {len(df)} positive observations (< {min_points}) after cleaning"
        )
    series = pd.Series(df[value_col].to_numpy(dtype=np.float64), index=df[date_col], name=name)
    return series


def _build_price_panel(
    assets: List[Dict[str, Any]],
    data_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    refresh_data: bool,
    min_assets: int,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    panel: Dict[str, pd.Series] = {}
    failures: Dict[str, str] = {}
    for asset in assets:
        name = str(asset.get("name") or asset.get("symbol") or "asset")
        try:
            panel[name] = _load_price_series(asset, data_dir=data_dir, refresh_data=refresh_data)
        except Exception as exc:
            failures[name] = str(exc)

    if len(panel) < int(min_assets):
        raise RuntimeError(
            f"Could not build panel with at least {min_assets} assets. "
            f"Loaded={len(panel)} failures={len(failures)}"
        )

    prices = pd.concat(panel.values(), axis=1, join="inner")
    prices.columns = list(panel.keys())
    prices = prices.sort_index()
    if start_date is not None:
        prices = prices.loc[pd.to_datetime(start_date):]
    if end_date is not None:
        prices = prices.loc[:pd.to_datetime(end_date)]
    prices = prices.dropna()

    if prices.empty:
        raise RuntimeError("Aligned price panel is empty after date filtering/intersection.")

    return prices, failures


def _rank_gaussianize(df: pd.DataFrame) -> pd.DataFrame:
    x = df.to_numpy(dtype=np.float64)
    u = _pseudo_obs_rank(x).astype(np.float64)
    z = norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def _rolling_splits(
    x: np.ndarray,
    dates: pd.DatetimeIndex,
    window: int,
    stride: int,
    train_fraction: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[pd.Timestamp]]:
    if window < 20:
        raise ValueError("window must be >= 20")
    if not (0.5 <= train_fraction < 1.0):
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    n = int(x.shape[0])
    if n < window:
        raise ValueError(f"Not enough observations ({n}) for window={window}")

    x_train: List[np.ndarray] = []
    x_test: List[np.ndarray] = []
    end_dates: List[pd.Timestamp] = []

    for end in range(window, n + 1, max(int(stride), 1)):
        block = x[end - window:end]
        split = int(round(train_fraction * window))
        split = max(min(split, window - 5), 5)
        tr = block[:split]
        te = block[split:]
        if tr.shape[0] < 5 or te.shape[0] < 5:
            continue
        x_train.append(np.asarray(tr, dtype=np.float64))
        x_test.append(np.asarray(te, dtype=np.float64))
        end_dates.append(pd.Timestamp(dates[end - 1]))

    if not x_train:
        raise ValueError("No valid rolling windows were generated.")
    return x_train, x_test, end_dates


def _tvgl_nll_sequence(
    x_train_by_time: List[np.ndarray],
    x_test_by_time: List[np.ndarray],
    *,
    alpha: float,
    beta: float,
    max_iter: int,
    step_size: float,
    eps: float,
    mode: str = "causal_rolling",
    history_windows: Optional[int] = 24,
) -> List[float]:
    """Evaluate TVGL NLL sequence either full-sequence or causal rolling."""
    mode = str(mode).lower().strip()
    if mode in {"full", "full_sequence", "noncausal"}:
        return _tvgl_gaussian_copula_nll_fit_eval(
            x_train_by_time,
            x_test_by_time,
            alpha=alpha,
            beta=beta,
            max_iter=max_iter,
            step_size=step_size,
            eps=eps,
        )

    out: List[float] = []
    hist = None if history_windows is None else int(max(history_windows, 1))
    for t in range(len(x_train_by_time)):
        start = 0 if hist is None else max(0, t - hist + 1)
        tr_hist = x_train_by_time[start:t + 1]
        te_hist = x_test_by_time[start:t + 1]
        try:
            seq = _tvgl_gaussian_copula_nll_fit_eval(
                tr_hist,
                te_hist,
                alpha=alpha,
                beta=beta,
                max_iter=max_iter,
                step_size=step_size,
                eps=eps,
            )
            out.append(float(seq[-1]) if len(seq) else float("nan"))
        except Exception:
            # Fallback for very short/ill-conditioned histories.
            try:
                out.append(float(_gaussian_copula_nll_fit_eval(tr_hist[-1], te_hist[-1])))
            except Exception:
                out.append(float("nan"))
    return out


def _state_space_nll_sequence(
    x_train_by_time: List[np.ndarray],
    x_test_by_time: List[np.ndarray],
    *,
    mode: str = "causal_rolling",
    history_windows: Optional[int] = 24,
) -> Tuple[List[float], float]:
    """Evaluate Gaussian state-space NLL sequence with optional causal rolling."""
    mode = str(mode).lower().strip()
    if mode in {"full", "full_sequence", "noncausal"}:
        seq, fit = gaussian_copula_state_space_nll_fit_eval(x_train_by_time, x_test_by_time)
        return [float(x) for x in seq], float(fit.process_variance)

    out: List[float] = []
    q_last = float("nan")
    hist = None if history_windows is None else int(max(history_windows, 1))
    for t in range(len(x_train_by_time)):
        start = 0 if hist is None else max(0, t - hist + 1)
        tr_hist = x_train_by_time[start:t + 1]
        te_hist = x_test_by_time[start:t + 1]
        try:
            seq, fit = gaussian_copula_state_space_nll_fit_eval(tr_hist, te_hist)
            out.append(float(seq[-1]) if len(seq) else float("nan"))
            q_last = float(fit.process_variance)
        except Exception:
            out.append(float("nan"))
    return out, q_last


def _dates_to_iso(dates: List[pd.Timestamp]) -> List[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]


def _period_mask(dates: List[pd.Timestamp], start: str, end: str) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (d >= pd.to_datetime(start)) & (d <= pd.to_datetime(end))


def _build_crisis_summary_rows(
    dates: List[pd.Timestamp],
    gaps: Dict[str, np.ndarray],
    crisis_periods: List[CrisisPeriod],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not dates:
        return rows

    period_masks = {cp.name: _period_mask(dates, cp.start, cp.end) for cp in crisis_periods}
    union_mask = np.zeros(len(dates), dtype=bool)
    for m in period_masks.values():
        union_mask = union_mask | m
    calm_mask = ~union_mask

    for key, baseline in GAP_FIELDS.items():
        arr = np.asarray(gaps.get(key, []), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue

        rows.append(
            {
                "baseline": baseline,
                "scope": "overall",
                "mean_gap": float(np.nanmean(arr)),
                "positive_fraction": float(np.nanmean(arr > 0.0)),
            }
        )

        if np.any(calm_mask):
            rows.append(
                {
                    "baseline": baseline,
                    "scope": "calm",
                    "mean_gap": float(np.nanmean(arr[calm_mask])),
                    "positive_fraction": float(np.nanmean(arr[calm_mask] > 0.0)),
                }
            )

        if np.any(union_mask):
            rows.append(
                {
                    "baseline": baseline,
                    "scope": "all_crises",
                    "mean_gap": float(np.nanmean(arr[union_mask])),
                    "positive_fraction": float(np.nanmean(arr[union_mask] > 0.0)),
                }
            )

        for cp in crisis_periods:
            m = period_masks[cp.name]
            if not np.any(m):
                continue
            rows.append(
                {
                    "baseline": baseline,
                    "scope": cp.name,
                    "mean_gap": float(np.nanmean(arr[m])),
                    "positive_fraction": float(np.nanmean(arr[m] > 0.0)),
                }
            )
    return rows


def _set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _shade_crises(ax: plt.Axes, crisis_periods: List[CrisisPeriod]) -> None:
    for cp in crisis_periods:
        ax.axvspan(
            pd.to_datetime(cp.start),
            pd.to_datetime(cp.end),
            color="#d62728",
            alpha=0.10,
            linewidth=0,
        )


def _plot_nll_gaps(
    out_png: Path,
    dates: List[pd.Timestamp],
    gaps: Dict[str, np.ndarray],
    crisis_periods: List[CrisisPeriod],
) -> None:
    _set_plot_style()
    d = pd.DatetimeIndex(dates)

    series_specs = [
        ("nll_gap", "Gaussian copula", "black", "-"),
        ("nll_gap_truncated_level0", "1-truncated C-vine", "#1f77b4", "--"),
        ("nll_gap_glasso", "Graphical Lasso", "#2ca02c", ":"),
        ("nll_gap_tvgl", "TVGL (Frobenius)", "#d62728", "-."),
        ("nll_gap_state_space", "Gaussian SSM", "#9467bd", (0, (3, 1, 1, 1))),
    ]

    fig, ax = plt.subplots(figsize=(12.0, 4.5))
    for key, label, color, ls in series_specs:
        arr = np.asarray(gaps.get(key, []), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        ax.plot(d, arr, label=label, color=color, linewidth=1.8, linestyle=ls)
    _shade_crises(ax, crisis_periods)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_ylabel("NLL gap (baseline - DVC)")
    ax.set_xlabel("Window end date")
    ax.set_title("Held-out copula NLL gap trajectories")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_tail_trajectory(
    out_png: Path,
    dates: List[pd.Timestamp],
    tail_emp_upper: List[float],
    tail_emp_lower: List[float],
    tail_fit_upper: List[float],
    tail_fit_lower: List[float],
    crisis_periods: List[CrisisPeriod],
) -> None:
    _set_plot_style()
    d = pd.DatetimeIndex(dates)
    fig, ax = plt.subplots(figsize=(12.0, 4.5))

    ax.plot(d, tail_emp_upper, color="#1f77b4", linewidth=1.8, label="Empirical upper tail")
    ax.plot(d, tail_fit_upper, color="#1f77b4", linewidth=1.8, linestyle="--", label="Fitted upper tail")
    ax.plot(d, tail_emp_lower, color="#d62728", linewidth=1.8, label="Empirical lower tail")
    ax.plot(d, tail_fit_lower, color="#d62728", linewidth=1.8, linestyle="--", label="Fitted lower tail")
    _shade_crises(ax, crisis_periods)

    ax.set_ylabel("Mean tail dependence")
    ax.set_xlabel("Window end date")
    ax.set_title("Tail dependence trajectories (level-0 edges)")
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _plot_family_heatmap(
    out_png: Path,
    dates: List[pd.Timestamp],
    family_codes: List[List[int]],
    edge_labels: List[str],
    codebook: List[str],
    title: str = "DVC level-0 selected copula families",
) -> None:
    if not family_codes:
        return
    mat = np.asarray(family_codes, dtype=np.int64).T  # edges x time
    if mat.size == 0:
        return

    _set_plot_style()
    cmap = ListedColormap(sns.color_palette("tab10", n_colors=len(codebook)))
    bounds = np.arange(len(codebook) + 1) - 0.5
    norm_ = BoundaryNorm(bounds, cmap.N)

    fig_h = max(3.0, 0.48 * mat.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm_)
    ax.set_yticks(np.arange(len(edge_labels)))
    ax.set_yticklabels(edge_labels)

    n_t = mat.shape[1]
    n_ticks = min(8, max(2, n_t))
    tick_idx = np.linspace(0, n_t - 1, num=n_ticks).astype(int)
    tick_lbl = [pd.Timestamp(dates[i]).strftime("%Y-%m-%d") for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right")

    ax.set_xlabel("Window end date")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks(range(len(codebook)))
    cbar.set_ticklabels(codebook)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _run_finance_scenario(
    output_dir: Path,
    seed: int,
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    name = str(scenario.get("name", "multi_asset_crisis")).strip()
    if not name:
        raise ValueError("Finance scenario name cannot be empty.")

    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    assets = scenario.get("assets", _default_assets())
    if not isinstance(assets, list) or not assets:
        raise ValueError("Scenario must provide a non-empty 'assets' list.")

    crisis_periods_raw = scenario.get("crisis_periods")
    if crisis_periods_raw is None:
        crisis_periods = _default_crisis_periods()
    else:
        crisis_periods = [
            CrisisPeriod(
                name=str(cp.get("name", f"crisis_{i+1}")),
                start=str(cp["start"]),
                end=str(cp["end"]),
            )
            for i, cp in enumerate(crisis_periods_raw)
            if isinstance(cp, dict) and "start" in cp and "end" in cp
        ]

    prices, failures = _build_price_panel(
        assets=assets,
        data_dir=data_dir,
        start_date=scenario.get("start_date"),
        end_date=scenario.get("end_date"),
        refresh_data=bool(scenario.get("refresh_data", False)),
        min_assets=int(scenario.get("min_assets", 5)),
    )
    prices.to_csv(data_dir / f"{name}_prices.csv")

    returns = np.log(prices).diff().dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.shape[0] < 200:
        raise RuntimeError(f"Insufficient returns after cleaning: {returns.shape[0]} rows")
    returns.to_csv(data_dir / f"{name}_log_returns.csv")

    # Rank-Gaussianized data for stable copula likelihood fitting.
    x_df = _rank_gaussianize(returns)
    x = x_df.to_numpy(dtype=np.float64)
    col_names = list(x_df.columns)
    dates = x_df.index

    window = int(scenario.get("window", 60))
    stride = int(scenario.get("stride", 5))
    train_fraction = float(scenario.get("train_fraction", 0.8))
    x_train, x_test, end_dates = _rolling_splits(
        x=x,
        dates=dates,
        window=window,
        stride=stride,
        train_fraction=train_fraction,
    )

    families_raw = [
        _normalize_family_name(f)
        for f in scenario.get(
            "families",
            ["gaussian", "student", "clayton", "gumbel", "independence"],
        )
    ]
    families = [("ind" if f in {"independence", "independent"} else f) for f in families_raw]
    # Frank/Joe are currently numerically brittle on short rolling windows.
    if bool(scenario.get("drop_brittle_families", True)):
        families = [f for f in families if f not in {"frank", "joe"}]
    if "ind" not in families:
        families.append("ind")
    families = list(dict.fromkeys(families))
    fam_to_code = {fam: i for i, fam in enumerate(DEFAULT_FAMILY_CODEBOOK)}
    q_upper = float(scenario.get("tail_q_upper", 0.90))
    q_lower = float(scenario.get("tail_q_lower", 0.10))
    q_upper = min(max(q_upper, 0.60), 0.99)
    q_lower = min(max(q_lower, 0.01), 0.40)
    optimize_structure = bool(scenario.get("optimize_structure", True))
    temporal_mode = str(scenario.get("temporal_baseline_mode", "causal_rolling"))
    temporal_hist = scenario.get("temporal_baseline_history_windows", 24)

    dvc_nll: List[float] = []
    gauss_nll: List[float] = []
    trunc_nll: List[float] = []
    glasso_nll: List[float] = []
    nf_copula_nll: List[float] = []

    tau_mean_abs: List[float] = []
    tail_emp_upper: List[float] = []
    tail_emp_lower: List[float] = []
    tail_fit_upper: List[float] = []
    tail_fit_lower: List[float] = []
    level0_family_codes: List[List[int]] = []
    edge_labels: List[str] = []
    root_history_idx: List[int] = []
    root_history_name: List[str] = []
    x_train_used: List[np.ndarray] = []
    x_test_used: List[np.ndarray] = []
    end_dates_used: List[pd.Timestamp] = []
    skipped_windows = 0
    skipped_window_errors: List[Dict[str, str]] = []

    for t, (tr, te, dt_end) in enumerate(zip(x_train, x_test, end_dates)):
        try:
            vine = _fit_parametric_vine(
                tr,
                families=families,
                optimize_structure=optimize_structure,
                seed=int(seed) + 37 * t,
            )
            order = getattr(vine, "variable_order", None)
            if order and len(order) >= 1:
                root_idx = int(order[0])
            elif getattr(vine, "ind_vine", None) and vine.ind_vine and vine.ind_vine[0]:
                root_idx = int(vine.ind_vine[0][0][0])
            else:
                root_idx = -1
            root_history_idx.append(root_idx)
            root_history_name.append(col_names[root_idx] if 0 <= root_idx < len(col_names) else "unknown")
            dvc_nll.append(_mean_copula_nll(vine, te))
            gauss_nll.append(_gaussian_copula_nll_fit_eval(tr, te))
            trunc_vine = _fit_truncated_cvine_level0(tr, families=families, order=list(range(len(col_names))))
            trunc_nll.append(_mean_copula_nll(trunc_vine, te))
            glasso_nll.append(
                _glasso_gaussian_copula_nll_fit_eval(
                    tr,
                    te,
                    alpha=float(scenario.get("glasso_alpha", 0.02)),
                )
            )
            try:
                nf_nll_value = nf_copula_nll_fit_eval(
                    tr,
                    te,
                    n_epochs=int(scenario.get("nf_copula_epochs", 50)),
                    hidden_dim=int(scenario.get("nf_copula_hidden_dim", 32)),
                    n_blocks=int(scenario.get("nf_copula_n_blocks", 4)),
                    seed=int(seed) + 7 * t,
                )
            except Exception:
                nf_nll_value = float("nan")
            nf_copula_nll.append(nf_nll_value)

            level0_edges = vine.ind_vine[0] if getattr(vine, "ind_vine", None) else []
            level0_cops = vine.copulas[0] if getattr(vine, "copulas", None) else []

            if not edge_labels and level0_edges:
                edge_labels = [f"{col_names[int(i)]}-{col_names[int(j)]}" for i, j in level0_edges]

            taus_t = []
            emp_u_t = []
            emp_l_t = []
            fit_u_t = []
            fit_l_t = []
            code_t: List[int] = []

            for e_idx, (i_raw, j_raw) in enumerate(level0_edges):
                i, j = int(i_raw), int(j_raw)
                # Tail metrics on full rolling window for stability (NLL remains held-out).
                pair = np.concatenate([tr[:, [i, j]], te[:, [i, j]]], axis=0)
                u_pair = _pseudo_obs_rank(pair)

                tau_ij = float(pd.Series(pair[:, 0]).corr(pd.Series(pair[:, 1]), method="kendall"))
                if np.isfinite(tau_ij):
                    taus_t.append(abs(tau_ij))

                emp_u_t.append(_empirical_tail_dependence(u_pair, q=q_upper, tail="upper"))
                emp_l_t.append(_empirical_tail_dependence(u_pair, q=q_lower, tail="lower"))

                if e_idx < len(level0_cops):
                    cop = level0_cops[e_idx]
                    fam = _normalize_family_name(getattr(cop, "family", "ind"))
                    code_t.append(int(fam_to_code.get(fam, 0)))
                    lam_l, lam_u = _taildeps_from_copula(cop)
                    fit_l_t.append(float(lam_l))
                    fit_u_t.append(float(lam_u))
                else:
                    code_t.append(0)
                    fit_l_t.append(0.0)
                    fit_u_t.append(0.0)

            tau_mean_abs.append(float(np.nanmean(taus_t)) if taus_t else float("nan"))
            tail_emp_upper.append(float(np.nanmean(emp_u_t)) if emp_u_t else float("nan"))
            tail_emp_lower.append(float(np.nanmean(emp_l_t)) if emp_l_t else float("nan"))
            tail_fit_upper.append(float(np.nanmean(fit_u_t)) if fit_u_t else float("nan"))
            tail_fit_lower.append(float(np.nanmean(fit_l_t)) if fit_l_t else float("nan"))
            level0_family_codes.append(code_t)
            x_train_used.append(tr)
            x_test_used.append(te)
            end_dates_used.append(dt_end)
        except Exception as exc:
            skipped_windows += 1
            skipped_window_errors.append(
                {
                    "window_end_date": pd.Timestamp(dt_end).strftime("%Y-%m-%d"),
                    "error": str(exc),
                }
            )
            continue

    if len(x_train_used) < 5:
        raise RuntimeError(
            f"Finance benchmark failed: only {len(x_train_used)} valid windows "
            f"(skipped={skipped_windows}). Last errors: {skipped_window_errors[-3:]}"
        )

    try:
        tvgl_nll = _tvgl_nll_sequence(
            x_train_used,
            x_test_used,
            alpha=float(scenario.get("tvgl_alpha", 0.02)),
            beta=float(scenario.get("tvgl_beta", 1.0)),
            max_iter=int(scenario.get("tvgl_max_iter", 200)),
            step_size=float(scenario.get("tvgl_step_size", 0.05)),
            eps=float(scenario.get("tvgl_eps", 1e-4)),
            mode=temporal_mode,
            history_windows=None if temporal_hist is None else int(temporal_hist),
        )
    except Exception:
        tvgl_nll = [float("nan")] * len(dvc_nll)

    try:
        ssm_nll, ssm_q = _state_space_nll_sequence(
            x_train_used,
            x_test_used,
            mode=temporal_mode,
            history_windows=None if temporal_hist is None else int(temporal_hist),
        )
    except Exception:
        ssm_nll = [float("nan")] * len(dvc_nll)
        ssm_q = float("nan")

    dvc_arr = np.asarray(dvc_nll, dtype=np.float64)
    gaps = {
        "nll_gap": np.asarray(gauss_nll, dtype=np.float64) - dvc_arr,
        "nll_gap_truncated_level0": np.asarray(trunc_nll, dtype=np.float64) - dvc_arr,
        "nll_gap_glasso": np.asarray(glasso_nll, dtype=np.float64) - dvc_arr,
        "nll_gap_tvgl": np.asarray(tvgl_nll, dtype=np.float64) - dvc_arr,
        "nll_gap_state_space": np.asarray(ssm_nll, dtype=np.float64) - dvc_arr,
        "nll_gap_nf_copula": np.asarray(nf_copula_nll, dtype=np.float64) - dvc_arr,
    }

    crisis_summary = _build_crisis_summary_rows(end_dates_used, gaps, crisis_periods)
    crisis_union = np.zeros(len(end_dates_used), dtype=bool)
    for cp in crisis_periods:
        crisis_union = crisis_union | _period_mask(end_dates_used, cp.start, cp.end)

    outperformance_flags: Dict[str, bool] = {}
    for key, label in GAP_FIELDS.items():
        arr = gaps.get(key, np.asarray([], dtype=np.float64))
        if arr.size == 0:
            outperformance_flags[label] = False
            continue
        if np.any(crisis_union):
            outperformance_flags[label] = bool(np.nanmean(arr[crisis_union]) > 0.0)
        else:
            outperformance_flags[label] = bool(np.nanmean(arr) > 0.0)

    nll_df = pd.DataFrame(
        {
            "date": _dates_to_iso(end_dates_used),
            "dvc_nll": np.asarray(dvc_nll, dtype=np.float64),
            "gaussian_nll": np.asarray(gauss_nll, dtype=np.float64),
            "truncated_level0_nll": np.asarray(trunc_nll, dtype=np.float64),
            "glasso_gaussian_nll": np.asarray(glasso_nll, dtype=np.float64),
            "tvgl_gaussian_nll": np.asarray(tvgl_nll, dtype=np.float64),
            "state_space_gaussian_nll": np.asarray(ssm_nll, dtype=np.float64),
            "nf_copula_nll": np.asarray(nf_copula_nll, dtype=np.float64),
            "nll_gap": gaps["nll_gap"],
            "nll_gap_truncated_level0": gaps["nll_gap_truncated_level0"],
            "nll_gap_glasso": gaps["nll_gap_glasso"],
            "nll_gap_tvgl": gaps["nll_gap_tvgl"],
            "nll_gap_state_space": gaps["nll_gap_state_space"],
            "nll_gap_nf_copula": gaps["nll_gap_nf_copula"],
            "tau_mean_abs_level0": np.asarray(tau_mean_abs, dtype=np.float64),
            "tail_emp_upper_q95_mean": np.asarray(tail_emp_upper, dtype=np.float64),
            "tail_emp_lower_q05_mean": np.asarray(tail_emp_lower, dtype=np.float64),
            "tail_fit_upper_mean": np.asarray(tail_fit_upper, dtype=np.float64),
            "tail_fit_lower_mean": np.asarray(tail_fit_lower, dtype=np.float64),
        }
    )
    nll_df.to_csv(data_dir / f"{name}_rolling_metrics.csv", index=False)
    pd.DataFrame(crisis_summary).to_csv(data_dir / f"{name}_crisis_summary.csv", index=False)

    _plot_nll_gaps(
        plots_dir / f"{name}_nll_gap_panel.png",
        end_dates_used,
        gaps=gaps,
        crisis_periods=crisis_periods,
    )
    _plot_tail_trajectory(
        plots_dir / f"{name}_tail_dependence_panel.png",
        end_dates_used,
        tail_emp_upper=tail_emp_upper,
        tail_emp_lower=tail_emp_lower,
        tail_fit_upper=tail_fit_upper,
        tail_fit_lower=tail_fit_lower,
        crisis_periods=crisis_periods,
    )
    root_unique = sorted(set(root_history_name))
    if len(root_unique) <= 1:
        _plot_family_heatmap(
            plots_dir / f"{name}_family_heatmap.png",
            end_dates_used,
            family_codes=level0_family_codes,
            edge_labels=edge_labels,
            codebook=DEFAULT_FAMILY_CODEBOOK,
            title="DVC level-0 selected copula families",
        )
    else:
        edge_labels_pos = [f"edge_pos_{i + 1}" for i in range(len(level0_family_codes[0]))] if level0_family_codes else []
        _plot_family_heatmap(
            plots_dir / f"{name}_family_heatmap.png",
            end_dates_used,
            family_codes=level0_family_codes,
            edge_labels=edge_labels_pos,
            codebook=DEFAULT_FAMILY_CODEBOOK,
            title="DVC level-0 families (edge position; root varies over time)",
        )

    return {
        "name": name,
        "assets": [str(c) for c in col_names],
        "n_assets": int(len(col_names)),
        "n_raw_days": int(prices.shape[0]),
        "n_returns_days": int(returns.shape[0]),
        "start_date_effective": str(prices.index.min().date()),
        "end_date_effective": str(prices.index.max().date()),
        "window": int(window),
        "stride": int(stride),
        "train_fraction": float(train_fraction),
        "window_end_dates": _dates_to_iso(end_dates_used),
        "n_windows_skipped": int(skipped_windows),
        "skipped_window_errors": skipped_window_errors[:50],
        "download_failures": failures,
        "crisis_periods": [{"name": cp.name, "start": cp.start, "end": cp.end} for cp in crisis_periods],
        "dvc_nll": np.asarray(dvc_nll, dtype=np.float64).tolist(),
        "gaussian_nll": np.asarray(gauss_nll, dtype=np.float64).tolist(),
        "truncated_level0_nll": np.asarray(trunc_nll, dtype=np.float64).tolist(),
        "glasso_gaussian_nll": np.asarray(glasso_nll, dtype=np.float64).tolist(),
        "tvgl_gaussian_nll": np.asarray(tvgl_nll, dtype=np.float64).tolist(),
        "state_space_gaussian_nll": np.asarray(ssm_nll, dtype=np.float64).tolist(),
        "nf_copula_nll": np.asarray(nf_copula_nll, dtype=np.float64).tolist(),
        "nll_gap": gaps["nll_gap"].tolist(),
        "nll_gap_truncated_level0": gaps["nll_gap_truncated_level0"].tolist(),
        "nll_gap_glasso": gaps["nll_gap_glasso"].tolist(),
        "nll_gap_tvgl": gaps["nll_gap_tvgl"].tolist(),
        "nll_gap_state_space": gaps["nll_gap_state_space"].tolist(),
        "nll_gap_nf_copula": gaps["nll_gap_nf_copula"].tolist(),
        "state_space_process_variance": ssm_q,
        "tau_mean_abs_level0": np.asarray(tau_mean_abs, dtype=np.float64).tolist(),
        "tail_emp_upper_q95_mean": np.asarray(tail_emp_upper, dtype=np.float64).tolist(),
        "tail_emp_lower_q05_mean": np.asarray(tail_emp_lower, dtype=np.float64).tolist(),
        "tail_fit_upper_mean": np.asarray(tail_fit_upper, dtype=np.float64).tolist(),
        "tail_fit_lower_mean": np.asarray(tail_fit_lower, dtype=np.float64).tolist(),
        "level0_family_codes": level0_family_codes,
        "family_codebook": list(DEFAULT_FAMILY_CODEBOOK),
        "level0_edge_labels": edge_labels,
        "root_history_idx": root_history_idx,
        "root_history_name": root_history_name,
        "root_unique": root_unique,
        "root_change_count": int(sum(1 for i in range(1, len(root_history_idx)) if root_history_idx[i] != root_history_idx[i - 1])),
        "tail_q_upper": float(q_upper),
        "tail_q_lower": float(q_lower),
        "families_used": families,
        "optimize_structure": bool(optimize_structure),
        "temporal_baseline_mode": temporal_mode,
        "temporal_baseline_history_windows": temporal_hist,
        "crisis_summary": crisis_summary,
        "outperformance_flags": outperformance_flags,
    }


def run_finance_crisis_benchmark_suite(
    output_dir: Path,
    seed: int,
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one or more finance crisis scenarios and write artifacts under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scenarios:
        scenarios = [{"name": "multi_asset_daily_returns_crises"}]

    results: Dict[str, Any] = {
        "experiment_type": "finance_crisis_benchmarks",
        "seed": int(seed),
        "scenarios": {},
    }

    summary_rows: List[Dict[str, Any]] = []
    for idx, sc in enumerate(scenarios):
        payload = _run_finance_scenario(output_dir=output_dir, seed=int(seed) + 101 * idx, scenario=sc)
        sc_name = str(payload["name"])
        results["scenarios"][sc_name] = payload

        row = {
            "scenario": sc_name,
            "n_assets": payload.get("n_assets"),
            "n_windows": len(payload.get("window_end_dates", [])),
        }
        for gap_key in GAP_FIELDS:
            arr = np.asarray(payload.get(gap_key, []), dtype=np.float64)
            if arr.size == 0:
                continue
            row[f"{gap_key}_mean"] = float(np.nanmean(arr))
            row[f"{gap_key}_std"] = float(np.nanstd(arr))
            row[f"{gap_key}_positive_fraction"] = float(np.nanmean(arr > 0.0))
        summary_rows.append(row)

    results["summary_table"] = summary_rows
    return results


__all__ = ["run_finance_crisis_benchmark_suite"]
