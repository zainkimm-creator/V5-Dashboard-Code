from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import pytest
import backend.validation.studies as studies
from backend.validation.paper_inputs import GROUP_A, excitation_schedule


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "validation_summary" / "excitation_summary_v41.json"
DRIFT_SUMMARY_PATH = ROOT / "reports" / "validation_summary" / "drift_summary_all_plants.json"
ALL_PARAMETER_NAMES = "kt_UW,kt_Nip,kt_RW,kf_UW,kf_Nip,kf_RW,EA"

# These lock the shape of generated study artifacts. They are outputs, not
# inputs, so on a fresh clone they do not exist yet - run the excitation and
# drift studies first (see README) to enable these checks.
pytestmark = pytest.mark.skipif(
    not (SUMMARY_PATH.exists() and DRIFT_SUMMARY_PATH.exists()),
    reason="study artifacts not generated yet (reports/validation_summary/)",
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for child in value.values()
            for key in _all_mapping_keys(child)
        }
    if isinstance(value, list):
        return {
            key
            for child in value
            for key in _all_mapping_keys(child)
        }
    return set()


def test_excitation_summary_uses_corrected_all_ten_plant_protocol():
    payload = _load(SUMMARY_PATH)
    assert payload["calculation_version"] == studies.EXCITATION_CACHE_VERSION
    assert payload["plant_scope"] == "all_plants"
    assert payload["plant_count"] == 10
    assert len(set(payload["plant_ids"])) == 10
    assert len(payload["comparison_rows"]) == 6
    # 6 types x 10 plants x (1 noise-free seed + 3 dual-channel seeds). Fig. 2(a)
    # pools the dual-channel condition over "10 plants x 3 seeds"; noise-free is a
    # 10-plant median at a single seed.
    assert len(payload["raw_rows"]) == 6 * 10 * (1 + len(studies.EXCITATION_SN_SEEDS))

    settings = payload["run_metadata"]["run_settings"]
    assert settings["controller_sample_time_s"] == pytest.approx(0.001)
    assert settings["NF_log_sample_time_s"] == pytest.approx(0.005)
    assert settings["SN_log_sample_time_s"] == pytest.approx(0.020)
    # v5 reports the excitation SN column at the dual-channel condition
    # behind the 50 Hz working cutoff, not tension-only at 100 Hz.
    assert settings["sensor_noise_lpf_hz"] == pytest.approx(50.0)
    assert settings["sn_measurement_condition"] == "dual_channel"
    assert settings["sensor_noise_velocity_percent"] == pytest.approx(0.3)
    assert settings["sysid_estimator"] == "paper_eq8_weighted_pem_trf"
    assert settings["controller_integral_time"] == "per_plant_auto_Ti"
    assert settings["steady_velocity_baseline"] == "nominal_tension_target_with_current_line_speed"
    assert settings["velocity_correction_clamp"] == "none"
    assert settings["EV1_identification"] == "continuous_midrun_speed_step_segment_split"

    cell_counts = Counter((row["strategy"], row["condition"]) for row in payload["raw_rows"])
    assert {key: value for key, value in cell_counts.items() if key[1] == "NF"}
    assert all(value == 10 for key, value in cell_counts.items() if key[1] == "NF")
    assert all(
        value == 10 * len(studies.EXCITATION_SN_SEEDS)
        for key, value in cell_counts.items()
        if key[1] == "SN"
    )
    for row in payload["raw_rows"]:
        assert row["value_status"] == "computed_raw"
        assert row["controller_sample_time_s"] == pytest.approx(0.001)
        assert row["estimator"] == "paper_eq8_weighted_pem_trf"
        assert all(float(value) >= 0.1 for value in row["operating_point_TI_s"].split(","))
        assert row["high_ea_kp_cap_enabled"] is False
        assert row["noise_affects_controller"] is True
        assert row["ev1_segment_split"] is (row["strategy"] == "EV1")
        expected_mode = "paper_aligned_filtered_measurement" if row["condition"] == "SN" else "paper_aligned"
        assert row["controller_mode"] == expected_mode
        # v5 SN is the dual-channel condition behind the 50 Hz working cutoff.
        expected_lpf = 50.0 if row["condition"] == "SN" else None
        assert row["sensor_lpf_hz"] == expected_lpf

    et3m_rows = [row for row in payload["raw_rows"] if row["strategy"] == "ET3M"]
    assert et3m_rows
    assert all(row["operating_point_speed_multipliers"] == "0.5,1,2" for row in et3m_rows)


def test_excitation_corrected_noisy_ranking_is_computed_from_raw_runs():
    payload = _load(SUMMARY_PATH)
    raw_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in payload["raw_rows"]:
        raw_by_cell[(row["strategy"], row["condition"])].append(
            float(row["dashboard_MARE_theta_percent"])
        )

    comparison = {row["strategy"]: row for row in payload["comparison_rows"]}
    for strategy, row in comparison.items():
        for condition in ("NF", "SN"):
            assert row[f"dashboard_{condition}_percent"] == pytest.approx(
                statistics.median(raw_by_cell[(strategy, condition)])
            )

    assert comparison["ET1"]["dashboard_SN_percent"] > comparison["EV1"]["dashboard_SN_percent"]
    assert payload["best_noisy_excitation"] == "EV1"
    assert comparison["EV1"]["dashboard_SN_percent"] == min(
        row["dashboard_SN_percent"] for row in payload["comparison_rows"]
    )


def test_drift_summary_is_fresh_independent_simulation_with_complete_run_accounting():
    payload = _load(DRIFT_SUMMARY_PATH)
    assert payload["cache_key"]["version"] == studies.DRIFT_CACHE_VERSION
    assert payload["data_source"] == "dashboard_simulation"
    assert payload["reference_data_usage"] == "comparison_only"
    assert payload["cache_key"]["data_source"] == "dashboard_simulation"
    assert payload["drift_excitation_fraction_of_Tref"] == pytest.approx(0.2)
    assert payload["drift_sensor_noise_seeds"] == [0, 1, 2]
    assert payload["controller_tracks_drift"] is False
    # 13 scenarios = the paper's ten (five EA legs, two friction legs, two J
    # legs, one combined) plus the three friction axis ticks the paper draws but
    # does not publish, which drift_reference.json marks as dashboard-only.
    assert len(payload["comparison_rows"]) == 13
    assert len(payload["raw_rows"]) == 520
    ea_rows = [row for row in payload["comparison_rows"] if row["drift_family"] == "EA"]
    # Fig. 4a: "(a) axial stiffness EA (+/-10-30% and +50%)" - five legs, and the
    # published band/mean is the statistic over those five.
    assert [row["drift_percent"] for row in ea_rows] == [-30.0, -10.0, 10.0, 30.0, 50.0]
    combined_rows = [row for row in payload["comparison_rows"] if row["drift_family"] == "combined"]
    assert len(combined_rows) == 1
    assert combined_rows[0]["paper_NF_percent"] == pytest.approx(29.6)
    assert combined_rows[0]["paper_SN_percent"] is None
    friction_rows = [
        row for row in payload["comparison_rows"] if row["drift_family"] == "f"
    ]
    assert [row["drift_percent"] for row in friction_rows] == [-30.0, -15.0, 0.0, 15.0, 30.0]
    assert [row["drift_case"] for row in friction_rows] == [
        "f -30%",
        "f -15%",
        "f 0%",
        "f +15%",
        "f +30%",
    ]

    settings = payload["run_metadata"]["run_settings"]
    # The record length is the paper schedule's, not a constant: ledger campaign
    # 2 is a group-A campaign, and E_Toggle there is a 17 s record with a 2 s
    # settle and edges at 2/7/12 s. The previous hardcoded 7 s truncated the
    # record before the second and third toggle edges.
    schedule = excitation_schedule("E_Toggle", GROUP_A, 0)
    assert schedule.duration_s == pytest.approx(17.0)
    assert settings["duration_s"] == pytest.approx(schedule.duration_s)
    assert settings["settle_s"] == pytest.approx(schedule.settle_s)
    assert settings["excitation"] == "E_Toggle"
    assert settings["campaign_group"] == GROUP_A
    assert settings["pem_initial_scale"] == pytest.approx(1.01)
    assert settings["controller_sample_time_s"] == pytest.approx(0.001)
    assert settings["log_sample_time_s_NF"] == pytest.approx(0.020)
    assert settings["log_sample_time_s_SN"] == pytest.approx(0.020)
    # The drift campaign deliberately keeps its tension-only acquisition at
    # 100 Hz; it is not the representative dual-channel condition.
    assert settings["sensor_lpf_hz_SN"] == pytest.approx(100.0)
    assert settings["estimator"] == "paper_eq8_weighted_pem_trf"
    assert settings["tension_integral_time"] == "per_plant_auto_Ti"
    assert settings["steady_velocity_baseline"] == "nominal_tension_current_line_speed"
    assert settings["velocity_correction_clamp"] is None
    assert settings["metric_truth"] == "pre_drift_baseline"
    assert settings["metric_parameters"] == ALL_PARAMETER_NAMES.split(",")
    assert settings["data_source"] == "dashboard_simulation"
    assert settings["reference_data_usage"] == "comparison_only"

    forbidden_source_keys = {
        "source_sha256",
        "source_path",
        "source_total_row_count",
        "selected_source_row_count",
        "source_row_index",
    }
    assert forbidden_source_keys.isdisjoint(_all_mapping_keys(payload))
    serialized = json.dumps(payload).lower()
    assert "author_raw_replay" not in serialized
    assert "author_phase3_drift_results_csv" not in serialized

    counts = Counter((row["drift_case"], row["condition"]) for row in payload["raw_rows"])
    for comparison_row in payload["comparison_rows"]:
        case = comparison_row["drift_case"]
        assert counts[(case, "NF")] == 10
        assert counts[(case, "SN")] == 30
        assert Counter(
            row["seed"]
            for row in payload["raw_rows"]
            if row["drift_case"] == case and row["condition"] == "NF"
        ) == {0: 10}
        assert Counter(
            row["seed"]
            for row in payload["raw_rows"]
            if row["drift_case"] == case and row["condition"] == "SN"
        ) == {0: 10, 1: 10, 2: 10}
        assert comparison_row["valid_run_count_NF"] == comparison_row["expected_run_count_NF"] == 10
        assert comparison_row["valid_run_count_SN"] == comparison_row["expected_run_count_SN"] == 30
        assert comparison_row["failed_run_count_NF"] == 0
        assert comparison_row["failed_run_count_SN"] == 0

    for row in payload["raw_rows"]:
        assert row["value_status"] == "computed_raw"
        assert row["samples"] > 0
        assert row["controller_tracks_drift"] is False
        assert row["controller_sample_time_s"] == pytest.approx(0.001)
        assert row["log_sample_time_s"] == pytest.approx(0.020)
        assert row["operating_point_TI_s"] >= 0.1
        assert row["estimator"] == "paper_eq8_weighted_pem_trf"
        assert row["optimizer_success"] is True
        assert row["optimizer_status"] > 0
        assert row["optimizer_nfev"] > 0
        assert row["optimizer_cost"] is None or row["optimizer_cost"] >= 0
        assert row["optimizer_optimality"] is None or row["optimizer_optimality"] >= 0
        residual_rmse = row.get("weighted_residual_rmse", row.get("normalized_residual_rmse"))
        assert residual_rmse is None or residual_rmse >= 0
        for parameter in ALL_PARAMETER_NAMES.split(","):
            assert row[f"estimate_{parameter}"] > 0
            assert math.isfinite(float(row[f"relative_error_{parameter}_percent"]))
        assert row["metric_truth"] == "pre_drift_baseline"
        assert row["active_parameters"] == ALL_PARAMETER_NAMES
        assert row["metric_variant"] == "paper_eq7_one_step_pem_pre_drift_baseline_all7_mare"
        if row["condition"] == "NF":
            assert row["seed"] == 0
        else:
            assert row["seed"] in (0, 1, 2)


def test_drift_dashboard_medians_and_reference_differences_are_honest():
    payload = _load(DRIFT_SUMMARY_PATH)
    raw_by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in payload["raw_rows"]:
        value = row["dashboard_MARE_theta_percent"]
        if value is not None:
            raw_by_cell[(row["drift_case"], row["condition"])].append(float(value))

    nonzero_differences = {"NF": [], "SN": []}
    for row in payload["comparison_rows"]:
        case = row["drift_case"]
        for condition in ("NF", "SN"):
            recomputed = statistics.median(raw_by_cell[(case, condition)])
            dashboard = float(row[f"dashboard_{condition}_percent"])
            assert dashboard == pytest.approx(recomputed)
            assert row[f"raw_dashboard_{condition}_percent"] == pytest.approx(recomputed)
            assert row[f"displayed_dashboard_{condition}_percent"] == pytest.approx(recomputed)
            paper_value = row[f"paper_{condition}_percent"]
            difference_value = row[f"difference_{condition}_percent"]
            if paper_value is None:
                assert difference_value is None
                assert row[f"paper_value_status_{condition}"] == "unpublished_axis_tick_no_phase3_result"
                assert row["__provenance"][f"difference_{condition}_percent"] == "not_available_unpublished_reference"
            else:
                paper = float(paper_value)
                difference = float(difference_value)
                assert difference == pytest.approx(dashboard - paper)
                nonzero_differences[condition].append(abs(difference))
                assert row[f"paper_value_status_{condition}"] == "published_phase3_median"
        assert row["display_adjustment_type"] == "none_independent_simulation"
        assert row["__provenance"]["dashboard_NF_percent"] == "dashboard_simulation_median"
        assert row["__provenance"]["dashboard_SN_percent"] == "dashboard_simulation_median"

    assert any(value > 1e-9 for value in nonzero_differences["NF"])
    assert any(value > 1e-9 for value in nonzero_differences["SN"])
