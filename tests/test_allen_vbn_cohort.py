from __future__ import annotations

import pytest

from dvc_package.real_data.windowed_analysis import (
    WindowedDependenceResult,
    cohort_summary_table,
    summarize_cohort_results,
)


def _fake_result(
    *,
    session_id: int,
    mouse_id: str,
    experience_level: str,
    mean_gap: float,
    mean_abs_corr: list[float],
) -> WindowedDependenceResult:
    return WindowedDependenceResult(
        session_id=session_id,
        mouse_id=mouse_id,
        experience_level=experience_level,
        stimulus_table_name="stimulus",
        region_names=["A", "B", "C"],
        region_unit_counts={"A": 30, "B": 24, "C": 22},
        window_size=100,
        stride=100,
        response_window=[0.0, 0.25],
        window_start_index=[0, 100],
        window_center_index=[50, 150],
        window_center_time=[1.0, 2.0],
        dvc_nll=[-3.0, -2.9],
        truncated_level0_nll=[0.5, 0.6],
        nll_gap_truncated_level0=[mean_gap, mean_gap],
        mean_abs_corr=mean_abs_corr,
        change_fraction=[0.1, 0.2],
        rewarded_fraction=[0.0, 0.0],
        omitted_fraction=[0.0, 0.0],
        mean_gap=mean_gap,
        std_gap=0.0,
        positive_gap_fraction=1.0 if mean_gap > 0 else 0.0,
        gap_change_corr=0.0,
        n_presentations_selected=200,
    )


def test_cohort_summary_table_and_pair_deltas() -> None:
    results = [
        _fake_result(
            session_id=1,
            mouse_id="m1",
            experience_level="Familiar",
            mean_gap=10.0,
            mean_abs_corr=[0.2, 0.4],
        ),
        _fake_result(
            session_id=2,
            mouse_id="m1",
            experience_level="Novel",
            mean_gap=12.5,
            mean_abs_corr=[0.5, 0.7],
        ),
        _fake_result(
            session_id=3,
            mouse_id="m2",
            experience_level="Familiar",
            mean_gap=9.0,
            mean_abs_corr=[0.1, 0.3],
        ),
        _fake_result(
            session_id=4,
            mouse_id="m2",
            experience_level="Novel",
            mean_gap=8.5,
            mean_abs_corr=[0.15, 0.35],
        ),
    ]

    table = cohort_summary_table(results)
    assert len(table) == 4
    assert set(table["experience_level"]) == {"Familiar", "Novel"}
    assert float(table.loc[table["session_id"] == 2, "mean_abs_corr_mean"].iloc[0]) == 0.6

    summary = summarize_cohort_results(results)
    assert summary["n_sessions"] == 4
    assert summary["n_mice_with_pairs"] == 2
    assert summary["experience_summary"]["Familiar"]["n_sessions"] == 2
    assert summary["experience_summary"]["Novel"]["n_sessions"] == 2
    assert summary["paired_summary"]["n_mice"] == 2
    assert summary["paired_summary"]["delta_mean_gap_mean"] == pytest.approx(1.0)
    assert summary["paired_summary"]["delta_mean_abs_corr_mean"] == pytest.approx(0.175)
