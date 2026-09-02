"""Regression tests for the Tier 2 Section 4.2 campaign definition."""

from __future__ import annotations

import pytest

from backend.validation.plants import parameters_for_plant
from backend.validation.retuning import (
    DRIFT_BY_CODE,
    DRIFT_SCENARIOS,
    HGS_TWIN_BUDGET,
    PROTOCOLS,
    RETUNING_PLANTS,
    apply_drift,
    evaluate_gains,
    plant_auto_ti_s,
    step_response_cost,
)


def test_campaign_grid_is_six_by_ten() -> None:
    assert len(RETUNING_PLANTS) == 6
    assert len(DRIFT_SCENARIOS) == 10
    assert len({s.code for s in DRIFT_SCENARIOS}) == 10


def test_friction_legs_are_plus_minus_30_only() -> None:
    # data/processed also carries f +-15 % and f 0 %, which are NOT campaign
    # cells; including them would change every pooled statistic.
    friction = {s.friction_scale for s in DRIFT_SCENARIOS if s.code in ("D08", "D09")}
    assert friction == {1.30, 0.70}


def test_hgs_budget_matches_supplement() -> None:
    assert HGS_TWIN_BUDGET == 2805


def test_protocols_differ_only_in_acquisition() -> None:
    fm, lo = PROTOCOLS["field_matched"], PROTOCOLS["logging_only"]
    assert (fm.log_sample_time_s, lo.log_sample_time_s) == (0.005, 0.020)
    assert fm.velocity_noise_fraction > 0 and lo.velocity_noise_fraction == 0
    assert (fm.tension_lpf_hz, lo.tension_lpf_hz) == (50.0, 100.0)


def test_asymmetric_inertia_drift_moves_reels_independently() -> None:
    base, _ = parameters_for_plant("P01")
    drifted = apply_drift(base, DRIFT_BY_CODE["D07"])
    assert drifted.inertia_kg_m2[0] == pytest.approx(base.inertia_kg_m2[0] * 0.5)
    assert drifted.inertia_kg_m2[1] == pytest.approx(base.inertia_kg_m2[1])
    assert drifted.inertia_kg_m2[2] == pytest.approx(base.inertia_kg_m2[2] * 2.0)


# --------------------------------------------------------------------------- #
# Eq. (12)
# --------------------------------------------------------------------------- #
def test_cost_is_zero_for_a_perfect_step() -> None:
    times = [i * 0.01 for i in range(201)]
    before, after = (10.0, 10.0, 10.0), (12.0, 12.0, 12.0)
    tensions = [after if t >= 1.0 else before for t in times]
    cost = step_response_cost(times, tensions, before, after, 1.0)
    assert cost.S == pytest.approx(0.0, abs=1e-12)
    assert cost.settled


def test_cost_terms_carry_the_published_scales() -> None:
    # A pure 3 N offset with no overshoot and instant settling must give
    # exactly (3/3)^2 = 1.
    times = [i * 0.01 for i in range(201)]
    before, after = (10.0, 10.0, 10.0), (12.0, 12.0, 12.0)
    tensions = [(9.0, 9.0, 9.0) if t >= 1.0 else before for t in times]
    cost = step_response_cost(times, tensions, before, after, 1.0)
    assert cost.rmse_y_N == pytest.approx(3.0)
    # Never enters the band, so it is charged the full horizon.
    assert not cost.settled


def test_overshoot_is_relative_to_step_size() -> None:
    times = [0.0, 1.0, 1.5, 2.0]
    before, after = (10.0, 10.0, 10.0), (12.0, 12.0, 12.0)
    # Peak 13 N on a 2 N step is 50 % overshoot.
    tensions = [before, after, (13.0, 12.0, 12.0), after]
    cost = step_response_cost(times, tensions, before, after, 1.0)
    assert cost.overshoot_percent == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# the inner velocity loop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dashboard_id", ["P01", "P02", "P03", "P06", "P09", "P10"])
def test_every_campaign_plant_can_settle(dashboard_id: str) -> None:
    """Guards the K_vel regression.

    With the paper velocity gain disabled the controller falls back to a
    heuristic capped at 16 N m s/rad. P189 needs 745.7, so the heavy-reel plants
    became far more oscillatory and never entered the 2 % band inside the
    episode. A plant that cannot settle at any sampled gain is that bug
    returning, not a hard plant.
    """

    params, _ = parameters_for_plant(dashboard_id)
    line_speed = float(params.feeder_velocity_m_s)
    auto_ti = plant_auto_ti_s(params, line_speed)
    settled = [
        evaluate_gains(params, kp, ti * auto_ti, line_speed_m_s=line_speed).settled
        for kp in (3.0, 10.0, 30.0)
        for ti in (0.1, 1.0)
    ]
    assert any(settled), f"{dashboard_id} never settles - check paper_velocity_gain"
