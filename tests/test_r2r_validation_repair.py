from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.validation.paper_reference import comparison_data_available

# Some checks below compare against the optional published-result set, which is
# not distributed with this repository. They skip rather than fail on a fresh
# clone; drop the result JSONs into data/reference_results/ to enable them.
requires_comparison_data = pytest.mark.skipif(
    not comparison_data_available(),
    reason="published comparison set not present (data/reference_results/)",
)

from backend.models.controller import (
    CascadePIController,
    ControllerConfig,
    auto_tension_integral_time_s,
)
from backend.models.equations import (
    INPUT_NAMES,
    STATE_NAMES,
    derivatives,
    nominal_state,
    roller_tension_differences,
    steady_surface_velocities,
    velocities,
    web_torques,
)
from backend.models.simulation import SimulationConfig, simulate
from backend.models import modal
from backend.models.modal import closed_loop_modal_sensitivity
from backend.sysid.estimator import estimate_parameters
from backend.validation.calculations import simulation_calculation_payload
from backend.validation.closed_loop_damping import _comparison_rows as damping_comparison_rows
from backend.validation import closed_loop_damping as damping
from backend.validation.excitations import get_excitation_profile
from backend.validation.failure_inventory import build_failure_inventory
from backend.validation.noise_aware_logging_lpf import _comparison_rows as lpf_comparison_rows
from backend.validation import noise_aware_logging_lpf as noise_lpf
from backend.validation.paper_reference import load_excitation_reference, load_noise_lpf_reference
from backend.validation.plants import parameters_for_plant, plant_registry
from backend.validation import studies
from backend.validation.studies import _load_logging_reference_rows, _summarise_logging_reference


def _feedforward_inputs(state, params):
    omega = state[3:]
    return tuple(params.kf[i] * omega[i] - web_torques(state, params)[i] for i in range(3))


def _row_from_state(time_s, state, inputs, params):
    v_uw, v_nip, v_rw = velocities(state, params)
    row = {"time_s": time_s}
    row.update(dict(zip(STATE_NAMES, state, strict=True)))
    row.update(dict(zip(INPUT_NAMES, inputs, strict=True)))
    row.update(
        {
            "v_UW_m_s": v_uw,
            "v_Nip_m_s": v_nip,
            "v_RW_m_s": v_rw,
            "line_speed_ref_m_s": params.feeder_velocity_m_s,
        }
    )
    return row


def test_sysid_metric_is_mare_not_rmse():
    params, _ = parameters_for_plant("P01")
    true_params = params.with_drift(EA_scale=1.1)
    result = estimate_parameters(
        _exact_synthetic_rows(params, samples=16),
        nominal_params=params,
        true_params=true_params,
        summary_name=None,
    )
    manual_mare = sum(abs(row["relative_error"]) for row in result.error_table) / len(result.error_table)
    manual_rmse = math.sqrt(sum(row["relative_error"] ** 2 for row in result.error_table) / len(result.error_table))
    assert result.mare_theta == pytest.approx(manual_mare)
    assert result.to_dict()["metric_name"] == "MARE_theta"
    assert manual_mare != pytest.approx(manual_rmse)


def test_three_span_nominal_state_is_steady_for_all_plants():
    for plant in plant_registry():
        params, _ = parameters_for_plant(str(plant["plant_id"]))
        state = nominal_state(params, params.feeder_velocity_m_s)
        inputs = _feedforward_inputs(state, params)
        dx = derivatives(state, inputs, params)
        tolerance = 1e-7 * max(1.0, params.EA, max(params.tension_ref_N))
        assert max(abs(value) for value in dx) < tolerance


def test_simulation_numpy_rng_matches_default_rng_seed_zero():
    import numpy as np

    params, meta = parameters_for_plant("P01")
    sigma = 0.003 * float(meta["T_max_N"])
    expected_noise = np.random.default_rng(0).normal(0.0, sigma, size=3)
    result = simulate(
        params,
        config=SimulationConfig(
            duration_s=0.001,
            dt_s=0.001,
            controller_sample_time_s=0.001,
            log_sample_time_s=0.001,
            line_speed_m_s=float(meta["v_ref_m_s"]),
            sensor_noise_tension_N=sigma,
            sensor_lpf_hz=None,
            noise_affects_controller=False,
            noise_rng="numpy_default_rng",
            seed=0,
        ),
        write_output=False,
    )
    assert [result.rows[0][name] for name in ("T1", "T2", "T3")] == pytest.approx(
        [float(meta["T_ref_N"]) + float(value) for value in expected_noise]
    )


def test_boundary_tension_terms_use_t0_zero_and_no_t4_state():
    state = (10.0, 20.0, 30.0, 1.0, 2.0, 3.0)
    assert roller_tension_differences(state) == pytest.approx((-10.0, -10.0, 30.0))
    assert len(roller_tension_differences(state)) == 3


def test_simulation_calculation_t1_example_uses_uw_to_feeder_mapping():
    params, _ = parameters_for_plant("P01")
    state = (
        params.tension_ref_N[0] + 1.0,
        params.tension_ref_N[1],
        params.tension_ref_N[2],
        params.feeder_velocity_m_s / params.roller_radius_m[0] * 0.98,
        params.feeder_velocity_m_s / params.roller_radius_m[1],
        params.feeder_velocity_m_s / params.roller_radius_m[2],
    )
    row = _row_from_state(0.0, state, _feedforward_inputs(state, params), params)
    payload = simulation_calculation_payload({}, [row], type("Config", (), {"line_speed_m_s": params.feeder_velocity_m_s})(), params)
    t1_example = next(item for item in payload["calculations"] if item["parameter"] == "dT1/dt")
    values = t1_example["values"]
    assert "v0_UW_m_s" in values
    assert "v1_feeder_m_s" in values
    expected = derivatives(state, _feedforward_inputs(state, params), params)[0]
    reported = float(t1_example["result"].split("=")[1].split()[0])
    assert reported == pytest.approx(expected, rel=1e-5, abs=1e-5)


def _exact_synthetic_rows(params, samples=60, dt=0.01):
    state = tuple(
        value + offset
        for value, offset in zip(
            nominal_state(params, params.feeder_velocity_m_s),
            (1.2, -0.9, 0.7, 0.03, -0.02, 0.04),
            strict=True,
        )
    )
    rows = []
    for index in range(samples):
        base_inputs = _feedforward_inputs(state, params)
        inputs = tuple(
            base_inputs[i] + 0.02 * math.sin(0.31 * index + i) + 0.01 * math.cos(0.17 * index + 2 * i)
            for i in range(3)
        )
        rows.append(_row_from_state(index * dt, state, inputs, params))
        dx = derivatives(state, inputs, params)
        state = tuple(state[i] + dt * dx[i] for i in range(6))
    return rows


def test_theta_recovery_from_exact_three_span_rows():
    params, _ = parameters_for_plant("P01")
    result = estimate_parameters(_exact_synthetic_rows(params), params, params, summary_name=None)
    assert 100.0 * result.mare_theta < 1e-6
    for row in result.error_table:
        assert row["estimate"] == pytest.approx(row["truth"], rel=1e-7, abs=1e-7)


def test_excitation_profiles_are_tension_setpoint_and_line_speed_inputs():
    profile = get_excitation_profile("E_Toggle", 2.4)
    assert profile(1.999) == pytest.approx((0.0, 0.0, 0.0))
    assert profile(2.001)[0] == pytest.approx(2.4)
    assert max(profile(2.001)) == pytest.approx(2.4)

    ev1 = get_excitation_profile("EV1", 2.4)
    assert ev1(3.0) == pytest.approx((0.0, 0.0, 0.0))
    assert getattr(ev1, "line_speed_multiplier")(1.999) == pytest.approx(1.0)
    assert getattr(ev1, "line_speed_multiplier")(2.001) == pytest.approx(1.2)


def test_controller_paper_velocity_gain_formula_when_enabled():
    params, _ = parameters_for_plant("P01")
    state = nominal_state(params, params.feeder_velocity_m_s)
    target = tuple(value * 1.02 for value in params.tension_ref_N)
    controller = CascadePIController(
        ControllerConfig(
            target_tension_N=target,
            line_speed_m_s=params.feeder_velocity_m_s,
            feedforward_enabled=False,
            high_ea_kp_cap_enabled=False,
            paper_velocity_gain_enabled=True,
            velocity_correction_limit_fraction=1.0,
        )
    )
    action = controller.update(state, 0.01, params, target_tension_N=target, line_speed_m_s=params.feeder_velocity_m_s)
    for i in range(3):
        expected_gain = 1.4 * params.inertia_kg_m2[i] * math.sqrt(
            params.EA * params.roller_radius_m[i] ** 2 / (params.inertia_kg_m2[i] * params.span_length_m[i])
        )
        assert abs(action.velocity_error_rad_s[i]) > 1e-9
        assert action.inputs_V[i] / action.velocity_error_rad_s[i] == pytest.approx(expected_gain)


def test_controller_auto_ti_matches_review_equation_for_every_plant():
    for plant in plant_registry():
        params, _ = parameters_for_plant(str(plant["plant_id"]))
        line_speed = params.feeder_velocity_m_s
        steady_v = steady_surface_velocities(params, line_speed, params.tension_ref_N)
        omega_ss = tuple(
            velocity / radius
            for velocity, radius in zip(steady_v, params.roller_radius_m, strict=True)
        )
        tau_web = web_torques(params.tension_ref_N + omega_ss, params)
        u_ss = tuple(
            params.kf[index] * omega_ss[index] - tau_web[index]
            for index in range(3)
        )
        expected = max(
            0.1,
            (sum(params.tension_ref_N) / 3.0)
            * 5.0
            / (sum(abs(value) for value in u_ss) / 3.0),
        )
        assert auto_tension_integral_time_s(params, line_speed) == pytest.approx(expected)


def test_controller_can_hold_nominal_steady_velocity_baseline_during_tension_step():
    params, _ = parameters_for_plant("P01")
    state = nominal_state(params, params.feeder_velocity_m_s)
    stepped_target = tuple(value + offset for value, offset in zip(params.tension_ref_N, (3.0, -2.0, 1.0), strict=True))
    common = dict(
        target_tension_N=params.tension_ref_N,
        line_speed_m_s=params.feeder_velocity_m_s,
        feedforward_enabled=False,
        high_ea_kp_cap_enabled=False,
        velocity_correction_limit_fraction=None,
    )
    static_action = CascadePIController(
        ControllerConfig(**common, steady_velocity_uses_dynamic_target=False)
    ).update(state, 0.001, params, target_tension_N=stepped_target)
    dynamic_action = CascadePIController(
        ControllerConfig(**common, steady_velocity_uses_dynamic_target=True)
    ).update(state, 0.001, params, target_tension_N=stepped_target)

    nominal_v = steady_surface_velocities(params, params.feeder_velocity_m_s, params.tension_ref_N)
    dynamic_v = steady_surface_velocities(params, params.feeder_velocity_m_s, stepped_target)
    for index, radius in enumerate(params.roller_radius_m):
        expected_difference = (dynamic_v[index] - nominal_v[index]) / radius
        assert dynamic_action.velocity_ref_rad_s[index] - static_action.velocity_ref_rad_s[index] == pytest.approx(
            expected_difference
        )


def test_controller_none_velocity_limit_really_disables_correction_clamp():
    params, _ = parameters_for_plant("P01")
    state = nominal_state(params, params.feeder_velocity_m_s)
    large_step = tuple(value + 100.0 for value in params.tension_ref_N)
    common = dict(
        target_tension_N=params.tension_ref_N,
        line_speed_m_s=params.feeder_velocity_m_s,
        feedforward_enabled=False,
        high_ea_kp_cap_enabled=False,
        steady_velocity_uses_dynamic_target=False,
    )
    clamped_action = CascadePIController(
        ControllerConfig(**common, velocity_correction_limit_fraction=0.20)
    ).update(state, 0.001, params, target_tension_N=large_step)
    unclamped_action = CascadePIController(
        ControllerConfig(**common, velocity_correction_limit_fraction=None)
    ).update(state, 0.001, params, target_tension_N=large_step)

    nominal_v = steady_surface_velocities(params, params.feeder_velocity_m_s, params.tension_ref_N)
    clamped_correction = tuple(
        clamped_action.velocity_ref_rad_s[index] * radius - nominal_v[index]
        for index, radius in enumerate(params.roller_radius_m)
    )
    unclamped_correction = tuple(
        unclamped_action.velocity_ref_rad_s[index] * radius - nominal_v[index]
        for index, radius in enumerate(params.roller_radius_m)
    )
    limit = 0.20 * params.feeder_velocity_m_s
    assert max(abs(value) for value in clamped_correction) == pytest.approx(limit)
    assert max(abs(value) for value in unclamped_correction) > limit


def test_reference_comparison_rows_do_not_label_mare_as_rmse():
    raw_rows = [
        {"kp_star": kp, "condition": condition, "dashboard_MARE_theta_percent": value}
        for condition, value in (("NF", 10.0), ("SN", 12.0))
        for kp in (50, 100, 200)
    ]
    damping_rows = damping_comparison_rows(raw_rows)
    assert damping_rows
    assert all("paper_MARE_theta_percent" in row for row in damping_rows)
    assert all("dashboard_MARE_theta_percent" in row for row in damping_rows)
    assert not any("RMSE" in key for row in damping_rows for key in row)

    lpf_rows = lpf_comparison_rows(
        {1: 0.2, 2: 1.3, 5: 3.4, 10: 12.2, 20: 26.0, 50: 38.8, 100: 103.0},
        {1: 169.0, 2: 101.0, 5: 50.0, 10: 33.0, 20: 23.2, 50: 58.0, 100: 77.2},
        {1: 169.0, 2: 101.0, 5: 50.0, 10: 33.0, 20: 20.4, 50: 58.0, 100: 77.2},
    )
    assert lpf_rows
    assert not any("RMSE" in key for row in lpf_rows for key in row)


def test_noise_lpf_uses_revised_controller_and_full_reference_grid():
    params, meta = parameters_for_plant("P01")
    config = noise_lpf._paper_controller_config(params, float(meta["v_ref_m_s"]))

    assert config.TI_s == pytest.approx(
        auto_tension_integral_time_s(params, float(meta["v_ref_m_s"]))
    )
    assert config.high_ea_kp_cap_enabled is False
    assert config.velocity_correction_limit_fraction is None
    assert config.paper_velocity_gain_enabled is True
    assert config.feedforward_uses_measured_omega is True
    assert config.steady_velocity_uses_dynamic_target is False
    assert noise_lpf.CONTROLLER_SAMPLE_TIME_S == pytest.approx(0.001)
    assert noise_lpf.HEATMAP_EXCITATIONS == [
        "ET1",
        "ET3",
        "ET6",
        "EV1",
        "E_Toggle",
        "ET3M",
    ]
    # v5 publishes the LPF x Tlog grid (Fig. S6) for ET1 and E_Toggle only.
    # The other four excitations carry no paper reference at this condition and
    # must stay empty rather than keep their v4.1 values.
    assert noise_lpf.PAPER_HEATMAP_REFERENCE_EXCITATIONS == ("ET1", "E_Toggle")
    assert noise_lpf.PAPER_HEATMAP_MARE["ET1"] == [13, 15, 22, 33, 37, 65, 84]
    assert noise_lpf.PAPER_HEATMAP_MARE["E_Toggle"] == [12, 13, 18, 22, 26, 65, 89]
    for unreferenced in ("ET3", "ET6", "EV1", "ET3M"):
        assert all(value is None for value in noise_lpf.PAPER_HEATMAP_MARE[unreferenced])

    # The noise-free 1 ms cell is omitted by the paper: the one-step predictor
    # has a trivial near-zero solution there, so the ratio is undefined.
    assert noise_lpf.PAPER_FIG05_NF_MEAN[1] is None
    assert noise_lpf.PAPER_FIG05_NF_MEAN[20] == pytest.approx(22.70)


def test_noise_lpf_downsamples_1ms_rows_without_reusing_fixed_results():
    rows = [{"time_s": index / 1000.0} for index in range(101)]
    downsampled = noise_lpf._downsample_rows(rows, 20)

    assert [row["time_s"] for row in downsampled] == pytest.approx(
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    )


def test_noise_lpf_live_series_calls_weighted_pem_with_revised_settings(monkeypatch):
    captured_configs = []
    captured_pem_calls = []

    def fake_simulate(params, *, controller_config, config, excitation, write_output):
        captured_configs.append((controller_config, config))
        rows = [{"time_s": index / 1000.0} for index in range(101)]
        return SimpleNamespace(rows=rows)

    def fake_pem(
        rows,
        nominal_params,
        true_params,
        *,
        max_nfev,
        break_on_line_speed_change,
    ):
        captured_pem_calls.append(
            {
                "row_count": len(rows),
                "max_nfev": max_nfev,
                "break_on_line_speed_change": break_on_line_speed_change,
            }
        )
        return SimpleNamespace(mare_theta=0.25)

    monkeypatch.setattr(noise_lpf, "simulate", fake_simulate)
    monkeypatch.setattr(noise_lpf, "estimate_parameters_weighted_pem", fake_pem)

    rows = noise_lpf._run_live_sysid_series(
        plant={"plant_id": "P01"},
        excitation_name="EV1",
        lpf_hz=50,
        noise_level_percent=0.3,
        seed=0,
        tlog_values_ms=(1, 20),
    )

    assert [row["dashboard_MARE_theta_percent"] for row in rows] == pytest.approx(
        [25.0, 25.0]
    )
    assert all(row["estimator"] == "paper_eq8_weighted_pem_trf" for row in rows)
    assert captured_configs
    controller, sim_config = captured_configs[0]
    assert controller.high_ea_kp_cap_enabled is False
    assert controller.velocity_correction_limit_fraction is None
    assert sim_config.controller_sample_time_s == pytest.approx(0.001)
    assert sim_config.log_sample_time_s == pytest.approx(0.001)
    assert sim_config.noise_affects_controller is True
    assert sim_config.noise_rng == "numpy_default_rng"
    assert [call["row_count"] for call in captured_pem_calls] == [101, 6]
    assert all(call["max_nfev"] == 150 for call in captured_pem_calls)
    assert all(call["break_on_line_speed_change"] for call in captured_pem_calls)


def test_noise_lpf_cross_channel_runs_the_group_b_30s_record_with_the_paper_sigma_v(monkeypatch):
    """Campaigns 4-6 must use the B_dual_channel ET1 record and sigma_v = pct_v*v0/0.30.

    `excitation_schedules.csv` flags that 30 s record as required for
    reproducing the published velocity-noise cells, and the reference's
    `noise_model` block fixes the velocity sigma.
    """

    captured = []

    def fake_simulate(params, *, controller_config, config, excitation, write_output):
        captured.append(config)
        return SimpleNamespace(rows=[{"time_s": index / 1000.0} for index in range(11)])

    monkeypatch.setattr(noise_lpf, "simulate", fake_simulate)
    monkeypatch.setattr(
        noise_lpf,
        "estimate_parameters_weighted_pem",
        lambda *args, **kwargs: SimpleNamespace(mare_theta=0.1),
    )

    _, meta = parameters_for_plant("P01")
    rows = noise_lpf._run_live_sysid_series(
        plant={"plant_id": "P01"},
        excitation_name="ET1",
        lpf_hz=50,
        noise_level_percent=0.0,
        velocity_noise_percent=0.3,
        seed=1,
        tlog_values_ms=(2,),
        campaign_group=noise_lpf.CROSS_CHANNEL_CAMPAIGN_GROUP,
        velocity_seed_offset=100,
    )

    assert captured[0].duration_s == pytest.approx(30.0)
    assert captured[0].sensor_noise_tension_N == pytest.approx(0.0)
    assert captured[0].sensor_noise_velocity_m_s == pytest.approx(
        0.3 / 100.0 * float(meta["v_ref_m_s"]) / 0.30
    )
    assert captured[0].velocity_seed_offset == 100
    assert rows[0]["campaign_group"] == "B_dual_channel"
    assert rows[0]["record_duration_s"] == pytest.approx(30.0)
    assert rows[0]["measurement_condition"] == "velocity_only"


@requires_comparison_data
def test_noise_lpf_scorecard_enumerates_every_published_reference_quantity():
    """Every scoreable value in `noise_lpf_reference.json` must be scored.

    Called on an empty live payload the scorecard still enumerates the paper
    side, so a quantity the dashboard never computes shows up as NOT_COMPUTED
    instead of quietly vanishing.
    """

    reference = load_noise_lpf_reference()
    rows = noise_lpf._reference_scorecard({})

    assert {row["block"] for row in rows} == {
        "lpf_tlog_heatmap",
        "own_best_comparison",
        "working_cutoff",
        "noise_composition_vs_cutoff",
        "feasibility_gate",
        "transition_table",
        "transition_table_nf_baseline",
        "no_filter_boundary",
        "main_effect_spreads",
        "cross_channel",
    }
    # Fig. S6: eight series x seven logging periods.
    assert sum(1 for row in rows if row["block"] == "lpf_tlog_heatmap") == 8 * 7
    # Table S7: two filter blocks x six logging periods x five doses.
    assert sum(1 for row in rows if row["block"] == "transition_table") == 2 * 6 * 5
    assert sum(1 for row in rows if row["block"] == "transition_table_nf_baseline") == 6
    assert sum(1 for row in rows if row["block"] == "cross_channel") == len(
        reference["cross_channel"]["ratios"]
    ) + 2
    # The paper publishes no error value at 10 or 20 Hz, only failure rates.
    assert all(row["pass_fail"] == "NOT_COMPUTED" for row in rows)


def test_noise_lpf_figs10_rows_keep_nominal_median_separate_from_figure5_mean():
    dummy_dashboard = {
        cutoff: {
            excitation: [float(index + 1) for index in range(len(noise_lpf.TLOG_VALUES_MS))]
            for excitation in ("ET1", "E_Toggle")
        }
        for cutoff in ("none", "50", "100", "200")
    }

    rows = noise_lpf._figs10_rows({"figs10_dashboard": dummy_dashboard})

    assert len(rows) == 56
    assert all(
        row["aggregation"] == "median_over_10_plants_at_0.3pct_noise"
        for row in rows
    )
    reference = next(
        row
        for row in rows
        if row["LPF_Hz"] == "50"
        and row["excitation"] == "E_Toggle"
        and row["Tlog_ms"] == 20
    )
    assert reference["paper_MARE_theta"] == pytest.approx(
        noise_lpf.PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20]
    )


def test_noise_lpf_heatmap_keeps_full_calculation_precision():
    precise = 28.067513783928717
    live = {
        "heatmap_dashboard": {
            excitation: [precise + index for index, _ in enumerate(noise_lpf.TLOG_VALUES_MS)]
            for excitation in noise_lpf.HEATMAP_EXCITATIONS
        }
    }
    values = noise_lpf._dashboard_heatmap_values(live)
    assert values["EV1"][0] == precise
    assert values["EV1"][0] != round(precise, 3)


def test_noise_lpf_figure5_svg_clips_below_axis_values_and_labels_aggregation(tmp_path):
    chart = tmp_path / "figure5.svg"
    noise_lpf._write_tlog_comparison_chart(
        chart,
        noise_lpf.PAPER_FIG05_NF_MEAN,
        noise_lpf.PAPER_FIG05_SN_LPF100_MEAN,
    )

    svg = chart.read_text(encoding="utf-8")
    assert 'clipPath id="tlog-plot-clip"' in svg
    assert 'clip-path="url(#tlog-plot-clip)"' in svg
    # v5's Figure 5 is the logged-tension trace figure, not a Tlog sweep. The
    # two series this chart draws are the Table S7 NF-baseline column (ET1) and
    # the Fig. S6 100 Hz row (E_Toggle), so the title names those instead.
    assert "Tlog sweep - Table S7 NF baseline (ET1) and Fig. S6 100 Hz row (E_Toggle)" in svg
    assert "NF 1 ms below axis" in svg
    assert "NF=SN crossover:" in svg
    assert ">0.1<" not in svg


def test_noise_lpf_heatmap_svg_exposes_eight_percent_comparison_status(tmp_path):
    # The dashboard always computes every cell; only the paper reference can be
    # missing. Stand in a value wherever the paper publishes none.
    dashboard = {
        excitation: [30.0 if value is None else float(value) for value in values]
        for excitation, values in noise_lpf.PAPER_HEATMAP_MARE.items()
    }
    dashboard["ET1"][0] *= 2.0
    chart = tmp_path / "heatmap.svg"
    noise_lpf._write_heatmap_comparison_chart(dashboard, chart)

    svg = chart.read_text(encoding="utf-8")
    assert "within 8%" in svg
    assert "over 8%" in svg
    assert 'stroke="#18794e"' in svg
    assert 'stroke="#b42318"' in svg
    assert "P = paper, D = dashboard" in svg
    # Cells the paper does not cover are labelled, not silently compared.
    assert "no v5 ref" in svg


def test_noise_lpf_frontend_uses_full_width_responsive_plot_cards():
    css = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "styles.css").read_text(
        encoding="utf-8"
    )
    assert ".noise-lpf-plot-grid {\n  display: grid;\n  grid-template-columns: minmax(0, 1fr);" in css
    assert ".noise-lpf-plot-frame {\n  width: 100%;\n  overflow: hidden;" in css
    assert ".noise-lpf-plot-frame img {\n  display: block;\n  width: 100%;\n  min-width: 0;" in css


def test_noise_lpf_frontend_cache_busts_regenerated_plot_artifacts():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.jsx").read_text(
        encoding="utf-8"
    )
    assert "function NoiseLpfResult({ baseUrl, result, artifactVersion })" in source
    assert "cacheBustUrl(artifactUrl(baseUrl, plot.url), artifactVersion)" in source
    assert "setArtifactVersion(Date.now());" in source
    assert "artifactVersion={artifactVersion}" in source


def test_damping_uses_revised_controller_eq7_and_exact_seed_protocol():
    params, meta = parameters_for_plant("P07")
    config = damping._paper_controller_config(
        params,
        float(meta["v_ref_m_s"]),
        200,
    )

    assert damping.CONTROLLER_SAMPLE_TIME_S == pytest.approx(0.001)
    assert damping.SYSID_TLOG_S == pytest.approx(0.020)
    assert damping.SN_SEEDS_BY_GAIN == {
        50: (0, 1, 2),
        100: (0,),
        200: (0, 1, 2),
    }
    assert config.Kp_star_m_s_per_N == pytest.approx(200.0)
    assert config.TI_s == pytest.approx(
        auto_tension_integral_time_s(params, float(meta["v_ref_m_s"]))
    )
    assert config.high_ea_kp_cap_enabled is False
    assert config.velocity_correction_limit_fraction is None
    assert config.paper_velocity_gain_enabled is True
    assert config.feedforward_uses_measured_omega is True
    assert config.steady_velocity_uses_dynamic_target is False


def test_full_cascade_eigenvalues_reproduce_all_table_s12_regimes():
    calculated = []
    for plant in plant_registry():
        plant_id = str(plant["plant_id"])
        params, meta = parameters_for_plant(plant_id)
        line_speed = float(meta["v_ref_m_s"])
        result = closed_loop_modal_sensitivity(
            params,
            damping._paper_controller_config(
                params,
                line_speed,
                damping.PAPER_DEFAULT_GAIN,
            ),
            line_speed_m_s=line_speed,
        )
        selected = result["selected"]
        calculated.append(float(selected["zeta_cl_min"]))
        assert selected["stable"] is True
        assert selected["regime"] == meta["regime"]
        assert float(selected["zeta_cl_min"]) == pytest.approx(
            float(meta["zeta_cl_min"]),
            abs=6e-4,
        )
        assert float(result["zeta_step_sensitivity_spread"]) < 1e-6
    assert len(calculated) == 10


def test_damping_gain_case_calls_weighted_pem_with_1ms_control_and_20ms_logging(monkeypatch):
    captured = {}

    def fake_simulate(params, *, controller_config, config, excitation, write_output):
        captured["controller"] = controller_config
        captured["simulation"] = config
        return SimpleNamespace(
            rows=[{"time_s": index * 0.020} for index in range(601)]
        )

    def fake_pem(
        rows,
        nominal_params,
        true_params,
        *,
        max_nfev,
    ):
        captured["pem_row_count"] = len(rows)
        captured["max_nfev"] = max_nfev
        return SimpleNamespace(mare_theta=0.20)

    monkeypatch.setattr(damping, "simulate", fake_simulate)
    monkeypatch.setattr(damping, "estimate_parameters_weighted_pem", fake_pem)

    row = damping._run_gain_case(
        plant={"plant_id": "P01"},
        kp_star=200,
        condition="SN",
        seed=2,
    )

    assert row["dashboard_MARE_theta_percent"] == pytest.approx(20.0)
    assert row["estimator"] == "paper_eq8_weighted_pem_trf"
    assert row["high_ea_kp_cap_enabled"] is False
    assert row["velocity_correction_limit_fraction"] is None
    assert captured["controller"].Kp_star_m_s_per_N == pytest.approx(200.0)
    assert captured["simulation"].controller_sample_time_s == pytest.approx(0.001)
    assert captured["simulation"].log_sample_time_s == pytest.approx(0.020)
    assert captured["simulation"].sensor_lpf_hz == pytest.approx(100.0)
    assert captured["simulation"].noise_affects_controller is True
    assert captured["simulation"].noise_rng == "numpy_default_rng"
    assert captured["pem_row_count"] == 601
    assert captured["max_nfev"] == 150


def test_damping_regime_grouping_merges_h_osc_into_o_ud():
    assert damping._figure7_group("O-UD") == "O-UD"
    assert damping._figure7_group("H-Osc") == "O-UD"
    assert damping._figure7_group("H-Damp") == "H-Damp"

    plant_rows = []
    for kp in damping.GAINS:
        for index in range(8):
            plant_rows.append(
                {
                    "kp_star": kp,
                    "plant_id": f"O{index}",
                    "calculated_figure7_group": "O-UD",
                    "dashboard_plant_median_MARE_theta_percent": 10.0 + index,
                }
            )
        for index in range(2):
            plant_rows.append(
                {
                    "kp_star": kp,
                    "plant_id": f"H{index}",
                    "calculated_figure7_group": "H-Damp",
                    "dashboard_plant_median_MARE_theta_percent": 30.0 + index,
                }
            )

    rows = damping._regime_rows(plant_rows)

    assert len(rows) == 3
    assert all(row["O_UD_plant_count"] == 8 for row in rows)
    assert all(row["H_Damp_plant_count"] == 2 for row in rows)
    assert all(row["plant_median_marker"] == 10 for row in rows)


def test_damping_step_metrics_are_calculated_from_actual_p07_trajectory():
    display_rows, metric_rows = damping._step_rows_and_metrics()

    assert len(display_rows) == 3204
    assert [row["kp_star"] for row in metric_rows] == [50, 100, 200]
    assert all(row["step_plant_id"] == "P07" for row in metric_rows)
    assert all(row["step_pool_id"] == "P163" for row in metric_rows)
    assert all(row["initial_tension_N"] == pytest.approx(306.0) for row in metric_rows)
    assert all(row["final_tension_N"] == pytest.approx(367.2) for row in metric_rows)
    t90 = [float(row["t90_s"]) for row in metric_rows]
    overshoot = [float(row["overshoot_percent"]) for row in metric_rows]
    assert t90[0] > t90[1] > t90[2]
    assert overshoot[0] < overshoot[1] < overshoot[2]
    assert t90 == pytest.approx([0.041633, 0.025752, 0.016742], rel=2e-4)
    assert overshoot == pytest.approx([16.1842, 31.7904, 47.8255], rel=2e-4)


def test_damping_frontend_uses_api_step_metrics_not_fixed_annotations():
    source = (
        Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.jsx"
    ).read_text(encoding="utf-8")

    assert "OS = 69%" not in source
    assert "Kp*=200: 0.03 s" not in source
    assert "Kp*=100: 0.05 s" not in source
    assert "Kp*=50: 0.07 s" not in source
    assert "<DampingStepChart rows={stepRows} metrics={gainRows} />" in source
    assert "kp200Metric?.overshoot_percent" in source
    assert "metrics?.recommended_gain" in source
    assert "alpha = 1 - exp(-2*pi*f_c*dt)" in source
    assert "Live Full-Cascade Eigenvalues" in source
    assert "metrics?.eigenvalue_summary" in source


@requires_comparison_data
def test_logging_reference_uses_v5_aggregate_values():
    """The v5 logging reference resolves all three measurement conditions at once.

    v5 Fig. 2(a) plots three conditions, and each keeps its own published Tlog
    span, so nothing is collapsed onto a single "sensor noise" slot. The v4.1
    anchors (NF 3.4% at 5 ms, SN 169.0 / 23.2 / 77.2) are retired with the
    unweighted tension-only estimator that produced them.
    """

    reference = _summarise_logging_reference(_load_logging_reference_rows())
    assert set(reference) == {"noise_free", "tension_only", "dual_channel"}

    nf_5 = reference["noise_free"][5.0]
    assert nf_5["paper_median_MARE_theta_percent"] == pytest.approx(3.83)
    assert nf_5["paper_reference_samples"] == pytest.approx(10.0)

    # Tension-only noise is monotonic: the finest logging is best and there is
    # no interior optimum.
    tension_only = reference["tension_only"]
    assert tension_only[1.0]["paper_median_MARE_theta_percent"] == pytest.approx(12.0)
    assert tension_only[20.0]["paper_median_MARE_theta_percent"] == pytest.approx(26.0)
    assert tension_only[100.0]["paper_median_MARE_theta_percent"] == pytest.approx(89.0)
    periods = sorted(tension_only)
    values = [tension_only[period]["paper_median_MARE_theta_percent"] for period in periods]
    assert values == sorted(values), "tension-only noise must stay monotonic in Tlog"

    # Dual-channel noise is where the interior optimum appears, at 5 ms.
    dual = reference["dual_channel"]
    assert dual[5.0]["paper_median_MARE_theta_percent"] == pytest.approx(25.8)
    assert dual[20.0]["paper_median_MARE_theta_percent"] == pytest.approx(42.6)
    assert dual[50.0]["paper_median_MARE_theta_percent"] == pytest.approx(78.8)
    assert min(dual, key=lambda k: dual[k]["paper_median_MARE_theta_percent"]) == 5.0


@requires_comparison_data
def test_logging_paper_values_cover_every_published_cell():
    """Every cell v5 publishes has to reach the dashboard's comparison rows.

    The regression this guards against is the dashboard showing "n/a" for a
    logging period the paper does print, which is what a single-condition
    reference slot produced: it left the noise-free and tension-only series
    unreachable whenever the dual-channel condition was selected.
    """

    reference = _summarise_logging_reference(_load_logging_reference_rows())
    published = {
        "noise_free": [2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        "tension_only": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        "dual_channel": [5.0, 20.0, 50.0],
    }
    for condition, periods in published.items():
        assert sorted(reference[condition]) == periods, condition
        for period in periods:
            value = reference[condition][period]["paper_median_MARE_theta_percent"]
            assert value is not None and value > 0.0, (condition, period)


@requires_comparison_data
def test_logging_acquisition_matches_the_published_v5_campaign():
    """The noisy conditions run the campaign settings the paper documents.

    v4.1 carried three fitted knobs here: a per-Tlog LPF table that left 1 ms
    unfiltered, per-Tlog velocity-observer clips, and a 0.25x excitation
    amplitude under noise. Together they inverted the v5 result - 1 ms became
    the WORST tension-only setting instead of the best, and the pooled
    dual-channel error at 5 ms came out near 72% against a published 25.8%.
    They must not come back.
    """

    assert studies.LOGGING_SN_LPF_HZ == 50.0
    assert studies.LOGGING_SN_AMPLITUDE_FACTOR == 1.0
    assert studies.LOGGING_VELOCITY_OBSERVER_CLIP_FRACTION is None
    assert not hasattr(studies, "LOGGING_SN_OBSERVER_SETTINGS")

    # The 50 Hz cutoff and 0.3% doses are what the reference itself records.
    conditions = json.loads(
        studies.PAPER_LOGGING_ADEQUACY_REFERENCE.read_text(encoding="utf-8")
    )["conditions"]
    assert conditions["tension_only"]["lpf_hz"] == studies.LOGGING_SN_LPF_HZ
    assert conditions["dual_channel"]["lpf_hz"] == studies.LOGGING_SN_LPF_HZ
    assert conditions["dual_channel"]["seeds"] == len(studies.LOGGING_DUAL_CHANNEL_SEEDS)
    assert conditions["noise_free"]["lpf_hz"] is None


@requires_comparison_data
def test_each_logging_condition_drives_the_excitation_its_reference_used():
    """A series is only comparable to the excitation it was recorded with.

    v5 publishes the noise-free series from ET1 and both noisy series from
    E_Toggle. Driving E_Toggle everywhere compared the dashboard's E_Toggle
    result against the paper's ET1 column.
    """

    provenance = json.loads(
        studies.PAPER_LOGGING_ADEQUACY_REFERENCE.read_text(encoding="utf-8")
    )["series_provenance"]
    for condition, excitation in studies.LOGGING_CONDITION_EXCITATION.items():
        assert provenance[condition]["excitation"] == excitation, condition


@requires_comparison_data
def test_logging_speed_groups_come_from_the_published_plant_lists():
    """Fig. 2(b) splits the dual-channel case by the paper's own plant lists."""

    groups = studies._logging_speed_groups()
    assert groups["slow"]["plants"] == ["P01", "P02", "P03", "P06", "P09"]
    assert groups["fast"]["plants"] == ["P04", "P05", "P07", "P08", "P10"]
    assert studies._logging_speed_group_for_plant("P01", groups) == "slow"
    assert studies._logging_speed_group_for_plant("P10", groups) == "fast"
    assert studies._logging_speed_group_for_plant("P99", groups) is None


@requires_comparison_data
def test_drift_paper_family_reference_publishes_the_EA_band():
    """The EA family is published as a band, so the band has to reach the UI.

    Every EA leg carries a null per-leg paper value by design. Without the
    family-level band the Drift page showed no paper number at all for its
    default family, which reads as "the paper says nothing about EA".
    """

    reference = studies._drift_paper_family_reference()
    assert reference["EA"]["paper_NF_band_percent"] == [24.4, 25.2]
    assert reference["EA"]["paper_SN_band_percent"] == [28.6, 31.4]
    assert reference["f"]["paper_NF_percent_by_drift_percent"]["-30"] == pytest.approx(31.2)
    assert reference["f"]["paper_NF_percent_by_drift_percent"]["30"] == pytest.approx(27.9)
    assert reference["J"]["gap_above_EA_baseline_pp"]["NF"] == pytest.approx(22.9)

    bands = studies._drift_family_paper_bands("EA")
    assert [label for label, _, _ in bands] == ["Paper NF band", "Paper SN band"]
    assert studies._drift_family_paper_bands("f") == []


@requires_comparison_data
def test_drift_scenarios_cover_the_ten_published_legs():
    """Section 3.3 runs ten scenarios; Fig. 4a draws five EA legs.

    The EA per-leg values are NOT unpublished. `fig04_drift/data_a.csv` in the
    figure package carries all five legs, and the `family_reference` band and
    mean reconcile exactly with them (NF [24.4, 25.2] mean 24.9; SN
    [28.6, 31.4] mean 29.5). They were previously null with a "do not
    synthesize" note, which the figure data contradicts, so they are now filled
    from the figure package and this test pins them.
    """

    rows = studies._drift_reference_rows()
    ea_percents = [row["drift_percent"] for row in rows if row["drift_family"] == "EA"]
    assert ea_percents == [-30.0, -10.0, 10.0, 30.0, 50.0]
    ea_rows = [row for row in rows if row["drift_family"] == "EA"]
    assert all(
        row["paper_NF_percent"] is not None and row["paper_SN_percent"] is not None
        for row in ea_rows
    )
    # The published band is the min/max of exactly these five legs.
    nf = [row["paper_NF_percent"] for row in ea_rows]
    sn = [row["paper_SN_percent"] for row in ea_rows]
    assert [round(min(nf), 1), round(max(nf), 1)] == [24.4, 25.2]
    assert [round(min(sn), 1), round(max(sn), 1)] == [28.6, 31.4]
    assert round(sum(nf) / len(nf), 1) == 24.9
    assert round(sum(sn) / len(sn), 1) == 29.5

    combined = [row for row in rows if row["drift_family"] == "combined"]
    assert len(combined) == 1
    assert combined[0]["paper_NF_percent"] == pytest.approx(29.6)
    assert combined[0]["paper_SN_percent"] is None

    published_scenarios = [
        row
        for row in rows
        if row.get("paper_reference_type") != "unpublished_axis_tick"
    ]
    assert len(published_scenarios) == 10


def test_drift_record_duration_comes_from_the_paper_schedule():
    """The drift campaign is a group-A campaign: E_Toggle is a 17 s record."""

    assert studies.DRIFT_CAMPAIGN_GROUP == "A_tension_factorial"
    assert studies.drift_record_duration_s() == pytest.approx(17.0)


@requires_comparison_data
def test_reel_radius_sensitivity_reproduces_the_R4_identity():
    sensitivity = studies._reel_radius_sensitivity()
    assert sensitivity["dashboard_inertia_drop_percent"] == pytest.approx(47.8, abs=0.05)
    assert sensitivity["paper_inertia_drop_percent"] == pytest.approx(48.0)


def test_logging_cache_rebases_copied_artifact_paths(monkeypatch, tmp_path):
    summary_dir = tmp_path / "reports" / "validation_summary"
    figures_dir = tmp_path / "reports" / "figures"
    summary_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    local_paths = {
        "plot_path": figures_dir / "logging_rate_vs_mare_restored.svg",
        "power_law_plot_path": figures_dir / "logging_rate_power_law_restored.svg",
        "speed_plot_path": figures_dir / "logging_rate_speed_decomposition.svg",
        "csv_path": summary_dir / "logging_rate_summary_restored.csv",
        "markdown_path": summary_dir / "logging_rate_report_restored.md",
        "speed_csv_path": summary_dir / "logging_rate_speed_decomposition.csv",
        "graph_points_csv_path": summary_dir / "logging_rate_graph_points_restored.csv",
        "graph_points_xlsx_path": summary_dir / "logging_rate_graph_points_restored.xlsx",
    }
    for path in local_paths.values():
        path.write_text("local artifact", encoding="utf-8")
    payload = {
        "metrics": [],
        "run_metadata": {},
        "plant_scope": "single_plant",
        "logging_data_source": "dashboard_simulation",
        "calculation_version": studies.LOGGING_RATE_CACHE_VERSION,
        "tlog_ms_values": [20.0],
        "tmin_ms": 50.0,
        "plant_ids": ["P01"],
        **{key: f"C:/older_project/{path.name}" for key, path in local_paths.items()},
    }
    (summary_dir / "logging_rate_summary_restored.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(studies, "SUMMARY_DIR", summary_dir)
    monkeypatch.setattr(studies, "FIGURES_DIR", figures_dir)
    params, metadata = parameters_for_plant("P01")

    cached = studies._cached_logging_rate_study(
        [20.0], 50.0, [("P01", params, metadata)]
    )

    assert cached is not None
    for key, path in local_paths.items():
        assert Path(cached["metrics"][key]) == path.resolve()
    assert Path(cached["plot_path"]) == local_paths["plot_path"].resolve()


@requires_comparison_data
def test_excitation_ev1_uses_v5_reference_values():
    """EV1 still attains the lowest error under noise, but the SN column moved.

    v4.1 SN was tension-only 0.3% at LPF 100 Hz and read 15.4%. v5 SN is the
    dual-channel condition (0.3%/0.3%, LPF 50 Hz, Tlog 20 ms) and reads 26.7%.
    The two are different measurement conditions, not a corrected value.
    """

    reference = load_excitation_reference()
    ev1 = reference["strategies"]["EV1"]
    assert ev1["NF"] == pytest.approx(2.2)
    assert ev1["SN"] == pytest.approx(26.7)
    assert ev1["channels"] == 0

    assert reference["conditions"]["NF"]["measurement_condition"] == "noise_free"
    assert reference["conditions"]["NF"]["Tlog_ms"] == 5
    sn = reference["conditions"]["SN"]
    assert sn["measurement_condition"] == "dual_channel"
    assert (sn["pct_T"], sn["pct_v"], sn["lpf_hz"], sn["Tlog_ms"]) == (0.3, 0.3, 50, 20)

    # EV1 is the lowest under noise, by a wide margin over the tension family.
    strategies = reference["strategies"]
    assert min(strategies, key=lambda name: strategies[name]["SN"]) == "EV1"


def test_failure_inventory_classifies_existing_report_failures():
    inventory = build_failure_inventory()
    if not inventory["records"]:
        pytest.skip(
            "Failure-inventory assertions require generated validation summaries."
        )
    assert inventory["records"]
    assert inventory["pass_count"] >= 1
    # The counts must account for every record either way.
    assert inventory["pass_count"] + inventory["fail_count"] == len(inventory["records"])
    # Classification is only exercised when something actually fails. With all
    # five sections merged (logging, excitation, drift, noise-aware logging,
    # closed-loop damping) the inventory is clean, so `fail_count == 0` is the
    # expected steady state rather than a defect. The previous
    # `fail_count >= 1` assertion held only while at least one section was
    # still unfixed; in a single-section copy the other sections supplied the
    # failure it depended on.
    if inventory["fail_count"]:
        assert inventory["failures_by_suspected_cause"]
    else:
        assert inventory["failures_by_suspected_cause"] == {}


# --------------------------------------------------------------------------- #
# Section 3.5 (closed-loop damping) v5 protocol assertions
# --------------------------------------------------------------------------- #
def test_damping_regime_thresholds_match_paper_section_2_2():
    """Paper v5 p. 5: O-UD < 0.3, 0.3 <= H-Osc < 0.5, H-Damp >= 0.5.

    The pre-fix code used 0.25 / 0.45, which is not what the paper states. The
    ten-plant membership happens to be unchanged, so this test pins both the
    thresholds themselves and the published membership.
    """

    assert modal.O_UD_UPPER_ZETA == pytest.approx(0.3)
    assert modal.H_DAMP_LOWER_ZETA == pytest.approx(0.5)
    assert modal.damping_regime(0.2999) == "O-UD"
    assert modal.damping_regime(0.3) == "H-Osc"
    assert modal.damping_regime(0.4999) == "H-Osc"
    assert modal.damping_regime(0.5) == "H-Damp"

    membership = {"O-UD": set(), "H-Osc": set(), "H-Damp": set()}
    for plant in plant_registry():
        plant_id = str(plant["plant_id"])
        membership[modal.damping_regime(float(plant["zeta_cl_min"]))].add(plant_id)
    assert membership["H-Damp"] == {"P06", "P09"}
    assert membership["H-Osc"] == {"P03", "P07"}
    assert membership["O-UD"] == {"P01", "P02", "P04", "P05", "P08", "P10"}


def test_open_loop_tau_min_reproduces_published_section_3_1_endpoints():
    """tau_min is the fastest OPEN-LOOP modal time scale, not a closed-loop one.

    Paper v5 Section 3.1 p. 8: ``tau_min = 1 / max_i |Re(lambda_i)|``, spanning
    7.5 ms (P158 = dashboard P06) to 67.4 ms (P139 = dashboard P05).
    """

    tau_ms = {}
    for plant in plant_registry():
        plant_id = str(plant["plant_id"])
        params, meta = parameters_for_plant(plant_id)
        tau_ms[plant_id] = 1000.0 * modal.open_loop_tau_min_s(
            params, line_speed_m_s=float(meta["v_ref_m_s"])
        )

    assert tau_ms["P06"] == pytest.approx(7.5, rel=1e-2)
    assert tau_ms["P05"] == pytest.approx(67.4, rel=1e-2)
    assert min(tau_ms, key=tau_ms.get) == "P06"
    assert max(tau_ms, key=tau_ms.get) == "P05"
    # The flat DEFAULT_TMIN_MS in studies.py is right for none of the ten.
    assert all(
        not math.isclose(value, studies.DEFAULT_TMIN_MS, rel_tol=1e-3)
        for value in tau_ms.values()
    )


def test_damping_step_metrics_carry_no_paper_reference():
    """The gain sweep logs no transient metric, so t90/overshoot have no target.

    Section 3.5: "the gain sweep logs no transient metric, so the overshoot
    cost of an off-default gain is not quantified here";
    ``data/reference_results/README.md`` lists it among the values the paper
    deliberately does not publish.
    """

    assert damping.STEP_METRICS_ARE_PAPER_REFERENCED is False
    assert not hasattr(damping, "PAPER_STEP_METRICS")

    _, metric_rows = damping._step_rows_and_metrics()
    for row in metric_rows:
        assert row["paper_t90_s"] is None
        assert row["paper_overshoot_percent"] is None
        assert row["t90_error_percent"] is None
        assert row["overshoot_error_percent"] is None
        assert row["paper_reference_status"] == damping.PAPER_STEP_METRIC_STATUS


def test_simpsons_paradox_block_separates_pooled_dip_from_group_monotonicity():
    """The v5 headline: monotone in BOTH groups, pooled dip is an artifact."""

    comparison_rows = [
        {"condition": "SN", "kp_star": 50, "dashboard_MARE_theta_percent": 32.5},
        {"condition": "SN", "kp_star": 100, "dashboard_MARE_theta_percent": 30.8},
        {"condition": "SN", "kp_star": 200, "dashboard_MARE_theta_percent": 32.0},
    ]
    regime_rows = [
        {
            "kp_star": 50,
            "dashboard_O_UD_MARE_theta_percent": 25.2,
            "dashboard_H_Damp_MARE_theta_percent": 59.1,
        },
        {
            "kp_star": 100,
            "dashboard_O_UD_MARE_theta_percent": 26.3,
            "dashboard_H_Damp_MARE_theta_percent": 67.0,
        },
        {
            "kp_star": 200,
            "dashboard_O_UD_MARE_theta_percent": 27.3,
            "dashboard_H_Damp_MARE_theta_percent": 80.9,
        },
    ]

    block = damping._simpsons_paradox_block(comparison_rows, regime_rows)
    assert block["dashboard_both_groups_monotone_increasing"] is True
    assert block["dashboard_pooled_minimum_gain"] == 100
    assert block["dashboard_pooled_has_interior_minimum"] is True
    assert block["group_sizes"] == {"O_UD_plus_H_Osc": 8, "H_Damp": 2}

    # A group that dips like the pooled curve must not be called monotone.
    broken = [dict(row) for row in regime_rows]
    broken[1]["dashboard_H_Damp_MARE_theta_percent"] = 55.0
    assert (
        damping._simpsons_paradox_block(comparison_rows, broken)[
            "dashboard_H_Damp_monotone_increasing"
        ]
        is False
    )


def test_damping_recommendation_is_the_paper_default_not_the_pooled_argmin():
    """Section 3.5 keeps Kp*=100 for identifiability and safety.

    The pooled argmin must never be presented as the recommendation, because
    the pooled minimum is a composition artifact.
    """

    assert damping.PAPER_DEFAULT_GAIN == 100
    assert damping.POOLED_MINIMUM_IS_COMPOSITION_ARTIFACT is True
    assert damping.PAPER_POOLED_MINIMUM_GAIN == 100
    assert damping.PAPER_EA_ERROR_ACROSS_GAIN_PERCENT == {
        50: 11.5,
        100: None,
        200: 39.4,
    }
    assert damping.PAPER_ET1_AT_DEFAULT_GAIN_PERCENT == pytest.approx(55.9)
    assert damping.PAPER_PER_PLANT_MARE_PERCENT["P08"][50] == pytest.approx(56.0)
    assert damping.PAPER_PER_PLANT_MARE_PERCENT["P08"][200] == pytest.approx(148.0)

    source = (
        Path(__file__).resolve().parents[1]
        / "backend"
        / "validation"
        / "closed_loop_damping.py"
    ).read_text(encoding="utf-8")
    assert "recommended_gain = PAPER_DEFAULT_GAIN" in source


def test_damping_gain_sweep_record_duration_comes_from_the_schedule_csv():
    """E_Toggle group A is a 17 s record; nothing may hardcode a duration."""

    profile = get_excitation_profile("E_Toggle", 1.0, campaign_group="A_tension_factorial")
    assert float(profile.duration_s) == pytest.approx(17.0)
    assert damping.GAIN_SWEEP_EXCITATION == "E_Toggle"
    assert damping.GAIN_SWEEP_CAMPAIGN_GROUP == "A_tension_factorial"
    assert damping.MEASUREMENT_CONDITION == "tension_only"
    assert damping.SN_LPF_HZ == 100
    assert damping.SN_NOISE_LEVEL_PERCENT == pytest.approx(0.3)
    assert damping.SN_VELOCITY_NOISE_LEVEL_PERCENT == pytest.approx(0.0)
