import numpy as np

from dvc_package.experiments.simulation_benchmarks import run_simulation_benchmark_suite


def test_hub_switch_reports_structure_recovery(tmp_path):
    results = run_simulation_benchmark_suite(
        output_dir=tmp_path,
        seed=123,
        scenarios=[
            {
                "name": "hub_switch",
                "n_time_steps": 12,
                "n_samples_per_time": 140,
                "n_variables": 6,
                "hub_a": 0,
                "hub_b": 1,
                "rho_hub": 0.75,
            }
        ],
    )

    payload = results["scenarios"]["hub_switch"]
    assert payload["root_recovery_accuracy"] >= 0.80
    assert payload["change_point_abs_error_dvc"] <= 1
    assert payload["regularized_root_recovery_accuracy"] >= 0.80
    assert payload["change_point_abs_error_regularized_dvc"] <= 1
    assert payload["corr_hub_recovery_accuracy"] >= 0.75
    assert "nll_improvement_regularized_over_dvc" in payload


def test_agent_interactions_expose_higher_order_signal_and_ranking_metrics(tmp_path):
    results = run_simulation_benchmark_suite(
        output_dir=tmp_path,
        seed=321,
        scenarios=[
            {
                "name": "agent_interaction_episodes",
                "n_time_steps": 18,
                "n_samples_per_time": 180,
                "n_agents": 6,
                "rho_pairwise": 0.7,
                "rho_higher": 0.5,
                "nu_higher": 3.0,
            }
        ],
    )

    payload = results["scenarios"]["agent_interaction_episodes"]
    dvc_metrics = payload["method_detection_metrics"]["DVC"]
    reg_metrics = payload["method_detection_metrics"]["Regularized DVC"]

    assert dvc_metrics["auroc"] is not None
    assert dvc_metrics["average_precision"] is not None
    assert dvc_metrics["auroc"] >= 0.90
    assert reg_metrics["auroc"] is not None
    assert reg_metrics["average_precision"] is not None
    assert payload["order_classification_accuracy"] >= 0.60
    assert payload["regularized_order_classification_accuracy"] >= 0.60

    pairwise_mean = payload["tc_higher_pairwise_mean"]
    higher_mean = payload["tc_higher_higher_order_mean"]
    mixed_mean = payload["tc_higher_mixed_mean"]
    reg_pairwise_mean = payload["regularized_tc_higher_pairwise_mean"]
    reg_higher_mean = payload["regularized_tc_higher_higher_order_mean"]
    reg_mixed_mean = payload["regularized_tc_higher_mixed_mean"]

    assert np.isfinite(pairwise_mean)
    assert np.isfinite(higher_mean)
    assert np.isfinite(mixed_mean)
    assert np.isfinite(reg_pairwise_mean)
    assert np.isfinite(reg_higher_mean)
    assert np.isfinite(reg_mixed_mean)
    assert higher_mean > pairwise_mean
    assert mixed_mean > pairwise_mean
    assert reg_higher_mean > reg_pairwise_mean
    assert reg_mixed_mean > reg_pairwise_mean


def test_higher_order_only_switch_keeps_pairwise_signal_flat(tmp_path):
    results = run_simulation_benchmark_suite(
        output_dir=tmp_path,
        seed=123,
        scenarios=[
            {
                "name": "higher_order_only_switch",
                "n_time_steps": 8,
                "n_samples_per_time": 3000,
            }
        ],
    )

    payload = results["scenarios"]["higher_order_only_switch"]

    assert payload["pairwise_abs_corr_shift"] <= 0.02
    # The corrected parametric estimator no longer supports the older sign-based
    # claim on this XOR-style stress test, but the metric should remain finite
    # and materially different from zero while pairwise statistics stay flat.
    assert np.isfinite(payload["higher_order_regime_contrast"])
    assert abs(payload["higher_order_regime_contrast"]) > 1.0
    assert np.isfinite(payload["tc_higher_higher_order_mean"])
    assert np.isfinite(payload["tc_higher_independence_mean"])
    assert np.mean(payload["nll_gap_truncated_level0"]) > 1.0


def test_dynamic_tail_df_reports_joint_dynamic_improvements(tmp_path):
    results = run_simulation_benchmark_suite(
        output_dir=tmp_path,
        seed=123,
        scenarios=[
            {
                "name": "dynamic_tail_df",
                "n_time_steps": 4,
                "n_samples_per_time": 80,
                "n_variables": 4,
                "rho": 0.6,
                "nu_low": 3.0,
                "nu_high": 30.0,
                "schedule": "piecewise",
            }
        ],
    )

    payload = results["scenarios"]["dynamic_tail_df"]
    for key in [
        "joint_dynamic_dvc_nll",
        "latent_state_dvc_nll",
        "nll_improvement_joint_over_dvc",
        "nll_improvement_latent_over_dvc",
        "tail_true_upper",
        "tail_true_lower",
    ]:
        arr = np.asarray(payload[key], dtype=np.float64)
        assert arr.shape == (4,)
        assert np.all(np.isfinite(arr))

    assert payload["tail_true_upper"][0] > payload["tail_true_upper"][-1]
    assert payload["windowed_nonparametric_config"]["temporal_smoothing_bandwidth"] > 0.0
    assert payload["joint_nonparametric_config"]["density_smoothing_bandwidth"] > 0.0
    assert payload["joint_dynamic_order"] == payload["latent_state_order"]
    assert np.mean(payload["nll_improvement_joint_over_dvc"]) > 0.0
    assert np.mean(payload["nll_improvement_latent_over_dvc"]) > 0.0
