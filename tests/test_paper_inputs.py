"""The two paper CSVs are the source of truth for the logging and excitation sections.

These tests read `data/model_inputs/ten_plant_parameters.csv` and
`data/model_inputs/excitation_schedules.csv` directly and assert that what
the dashboard simulates is what those files specify -- plant parameters, the
derived campaign-default controller settings, every excitation edge, the
campaign-group record lengths, and the seed conventions.
"""

from __future__ import annotations

import csv
import math

import pytest

from backend.models.controller import auto_tension_integral_time_s
from backend.models.equations import steady_surface_velocities, web_torques
from backend.models.simulation import SimulationConfig
from backend.validation import paper_inputs
from backend.validation.excitations import excitation_names, get_excitation_profile
from backend.validation.plants import PROFESSOR_PLANT_DETAILS, parameters_for_plant
from backend.validation.studies import (
    LOGGING_CONDITION_CAMPAIGN_GROUP,
    LOGGING_CONDITION_EXCITATION,
)


def _csv_plant_rows() -> dict[str, dict[str, str]]:
    with paper_inputs.TEN_PLANT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return {row["plant_id"]: row for row in csv.DictReader(handle)}


def _csv_schedule_rows() -> list[dict[str, str]]:
    with paper_inputs.EXCITATION_SCHEDULE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #
# ten_plant_parameters.csv
# --------------------------------------------------------------------------- #
def test_both_paper_input_files_are_vendored():
    assert paper_inputs.TEN_PLANT_CSV.exists()
    assert paper_inputs.EXCITATION_SCHEDULE_CSV.exists()


def test_plant_registry_covers_the_fixed_ten_plant_subset():
    expected = (
        "P001", "P049", "P053", "P060", "P139",
        "P158", "P163", "P177", "P186", "P189",
    )
    assert paper_inputs.ten_plant_pool_ids() == expected
    assert tuple(d["pool_id"] for d in PROFESSOR_PLANT_DETAILS.values()) == expected


@pytest.mark.parametrize("display_id", sorted(PROFESSOR_PLANT_DETAILS))
def test_plant_parameters_match_the_input_csv(display_id):
    details = PROFESSOR_PLANT_DETAILS[display_id]
    row = _csv_plant_rows()[details["pool_id"]]

    assert details["EA_N"] == pytest.approx(float(row["EA_N"]))
    assert details["v_ref_m_s"] == pytest.approx(float(row["v0_mps"]))
    assert details["T_ref_N"] == pytest.approx(float(row["T_ref_N"]))
    assert details["T_max_N"] == pytest.approx(float(row["T_max_N"]))
    assert details["regime"] == row["regime_class"]
    assert details["span_length_m"] == pytest.approx(
        (float(row["L1_m"]), float(row["L2_m"]), float(row["L3_m"]))
    )
    assert details["roller_radius_m"] == pytest.approx(
        (float(row["R_UW_m"]), float(row["R_Nip_m"]), float(row["R_RW_m"]))
    )
    assert details["inertia_kg_m2"] == pytest.approx(
        (float(row["J_UW_kgm2"]), float(row["J_Nip_kgm2"]), float(row["J_RW_kgm2"]))
    )
    assert details["viscous_friction"] == pytest.approx(
        (
            float(row["f_UW_Nms_per_rad"]),
            float(row["f_Nip_Nms_per_rad"]),
            float(row["f_RW_Nms_per_rad"]),
        )
    )


@pytest.mark.parametrize("display_id", sorted(PROFESSOR_PLANT_DETAILS))
def test_campaign_default_controller_settings_match_the_paper_csv(display_id):
    """omega_n, K_vel = 1.4 J omega_n, omega_ss, u_ss and auto_Ti are the CSV's."""

    details = PROFESSOR_PLANT_DETAILS[display_id]
    row = _csv_plant_rows()[details["pool_id"]]
    params, _meta = parameters_for_plant(display_id)
    v0 = float(row["v0_mps"])

    omega_n = tuple(
        math.sqrt(
            params.EA
            * params.roller_radius_m[i] ** 2
            / (params.inertia_kg_m2[i] * params.span_length_m[i])
        )
        for i in range(3)
    )
    assert omega_n == pytest.approx(
        (
            float(row["omega_n_UW_rad_s"]),
            float(row["omega_n_Nip_rad_s"]),
            float(row["omega_n_RW_rad_s"]),
        ),
        rel=1e-5,
    )

    k_vel = tuple(1.4 * params.inertia_kg_m2[i] * omega_n[i] for i in range(3))
    assert k_vel == pytest.approx(
        (float(row["K_vel_UW"]), float(row["K_vel_Nip"]), float(row["K_vel_RW"])),
        rel=1e-5,
    )

    steady_v = steady_surface_velocities(params, v0, params.tension_ref_N)
    omega_ss = tuple(steady_v[i] / params.roller_radius_m[i] for i in range(3))
    assert omega_ss == pytest.approx(
        (
            float(row["omega_ss_UW_rad_s"]),
            float(row["omega_ss_Nip_rad_s"]),
            float(row["omega_ss_RW_rad_s"]),
        ),
        rel=1e-4,
    )

    tau_web = web_torques(tuple(params.tension_ref_N) + omega_ss, params)
    u_ss = tuple(params.kf[i] * omega_ss[i] - tau_web[i] for i in range(3))
    assert u_ss == pytest.approx(
        (float(row["u_ss_UW_Nm"]), float(row["u_ss_Nip_Nm"]), float(row["u_ss_RW_Nm"])),
        rel=1e-4,
        abs=1e-4,
    )

    assert auto_tension_integral_time_s(params, v0) == pytest.approx(
        float(row["T_I_s"]), rel=1e-4
    )


# --------------------------------------------------------------------------- #
# excitation_schedules.csv
# --------------------------------------------------------------------------- #
def test_every_csv_edge_is_reproduced_by_the_profile():
    """Each row of the schedule table is a step the built profile actually takes."""

    amplitude = 10.0
    scale = amplitude / paper_inputs.REFERENCE_STEP_PERCENT
    channel_index = {"span1": 0, "span2": 1, "span3": 2}
    checked = 0
    for row in _csv_schedule_rows():
        schedule = paper_inputs.excitation_schedule(
            row["excitation"], row["campaign_group"], int(float(row["record_index"]))
        )
        profile = paper_inputs.build_excitation(schedule, amplitude)
        edge_time = float(row["edge_time_s"])
        before, after = edge_time - 1e-6, edge_time + 1e-6
        if row["edge_channel"] == paper_inputs.LINE_SPEED_CHANNEL:
            multiplier = getattr(profile, "line_speed_multiplier")
            assert multiplier(before) == pytest.approx(
                1.0 + float(row["value_from_pct_of_setpoint"]) / 100.0
            )
            assert multiplier(after) == pytest.approx(
                1.0 + float(row["value_to_pct_of_setpoint"]) / 100.0
            )
        else:
            index = channel_index[row["edge_channel"]]
            assert profile(before)[index] == pytest.approx(
                scale * float(row["value_from_pct_of_setpoint"])
            )
            assert profile(after)[index] == pytest.approx(
                scale * float(row["value_to_pct_of_setpoint"])
            )
        checked += 1
    assert checked == len(_csv_schedule_rows())


@pytest.mark.parametrize(
    ("name", "campaign_group", "duration_s"),
    [
        ("ET1", paper_inputs.GROUP_A, 7.0),
        ("ET1", paper_inputs.GROUP_B, 30.0),
        ("ET3", paper_inputs.GROUP_A, 17.0),
        ("ET6", paper_inputs.GROUP_A, 32.0),
        ("ET3M", paper_inputs.GROUP_A, 17.0),
        ("E_Toggle", paper_inputs.GROUP_A, 17.0),
        ("E_Toggle", paper_inputs.GROUP_C, 16.0),
        ("EV1", paper_inputs.GROUP_A, 12.0),
    ],
)
def test_record_lengths_are_the_published_ones(name, campaign_group, duration_s):
    schedule = paper_inputs.excitation_schedule(name, campaign_group)
    assert schedule.duration_s == pytest.approx(duration_s)
    assert getattr(
        get_excitation_profile(name, 1.0, campaign_group=campaign_group), "duration_s"
    ) == pytest.approx(duration_s)


def test_group_b_falls_back_to_group_a_for_the_five_identical_types():
    """Only ET1 differs between the tension factorial and the dual grids."""

    for name in excitation_names():
        resolved = paper_inputs.excitation_schedule(name, paper_inputs.GROUP_B)
        expected = paper_inputs.GROUP_B if name == "ET1" else paper_inputs.GROUP_A
        assert resolved.campaign_group == expected


def test_e_toggle_is_the_staggered_schedule_not_an_in_phase_square_wave():
    """One channel is introduced per episode; span 3 steps once and never toggles."""

    profile = get_excitation_profile("E_Toggle", 10.0)
    assert profile(1.0) == pytest.approx((0.0, 0.0, 0.0))
    assert profile(4.0) == pytest.approx((10.0, 0.0, 0.0))
    assert profile(9.0) == pytest.approx((0.0, 10.0, 0.0))
    assert profile(14.0) == pytest.approx((10.0, 0.0, 10.0))
    # Six edges in a 17 s record, not a ~1 Hz square wave.
    assert len(paper_inputs.excitation_schedule("E_Toggle").edges) == 6


def test_ev1_steps_line_speed_only():
    schedule = paper_inputs.excitation_schedule("EV1")
    assert schedule.excites_line_speed
    assert schedule.tension_edges == ()
    profile = get_excitation_profile("EV1", 10.0)
    assert profile(11.0) == pytest.approx((0.0, 0.0, 0.0))
    multiplier = getattr(profile, "line_speed_multiplier")
    assert multiplier(1.999) == pytest.approx(1.0)
    assert multiplier(2.001) == pytest.approx(1.2)


def test_et3m_logs_three_records_at_the_published_speed_multipliers():
    records = paper_inputs.excitation_records("ET3M")
    assert len(records) == 3
    assert [record.v_ref_multiplier for record in records] == [0.5, 1.0, 2.0]
    # Every record carries the ET3 edge pattern.
    et3_edges = paper_inputs.excitation_schedule("ET3").describe()
    for record in records:
        assert record.describe() == et3_edges


# --------------------------------------------------------------------------- #
# seed conventions
# --------------------------------------------------------------------------- #
def test_seed_conventions_match_the_csv_notes():
    assert paper_inputs.velocity_seed(0) == 100
    assert paper_inputs.velocity_seed(2) == 102
    assert [paper_inputs.et3m_record_seed(0, i) for i in range(3)] == [0, 17, 34]


def test_velocity_seed_offset_gives_the_velocity_channel_its_own_stream():
    shared = SimulationConfig()
    assert shared.velocity_seed_offset is None
    offset = SimulationConfig(velocity_seed_offset=paper_inputs.COMPOSITE_SEED_V_OFFSET)
    assert offset.velocity_seed_offset == 100


# --------------------------------------------------------------------------- #
# study wiring
# --------------------------------------------------------------------------- #
def test_logging_conditions_are_bound_to_a_published_campaign_grid():
    assert set(LOGGING_CONDITION_CAMPAIGN_GROUP) == set(LOGGING_CONDITION_EXCITATION)
    assert LOGGING_CONDITION_CAMPAIGN_GROUP["dual_channel"] == paper_inputs.GROUP_B
    for condition, name in LOGGING_CONDITION_EXCITATION.items():
        group = LOGGING_CONDITION_CAMPAIGN_GROUP[condition]
        # Resolves without raising, which is what the study depends on.
        assert paper_inputs.excitation_schedule(name, group).duration_s > 0.0
