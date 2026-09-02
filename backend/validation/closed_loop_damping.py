"""Closed-loop damping and live SysID gain-sweep validation."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models.controller import ControllerConfig, auto_tension_integral_time_s
from backend.models.modal import closed_loop_modal_sensitivity, open_loop_modal_analysis
from backend.models.simulation import SimulationConfig, simulate
from backend.sysid.estimator import estimate_parameters_one_step_pem, estimate_parameters_weighted_pem
from backend.validation.excitations import get_excitation_profile
from backend.validation.plants import parameters_for_plant, plant_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "reports" / "validation_summary"
CACHE_VERSION = "closed_loop_damping_v5_group_monotonic_taumin_ea_et1_v10_fig06paperrefs_20260818"

DT_S = 0.001
CONTROLLER_SAMPLE_TIME_S = 0.001
SYSID_TLOG_S = 0.020
SN_LPF_HZ = 100
SN_NOISE_LEVEL_PERCENT = 0.3
SN_VELOCITY_NOISE_LEVEL_PERCENT = 0.0
MEASUREMENT_CONDITION = "tension_only"
GAIN_SWEEP_EXCITATION = "E_Toggle"
GAIN_SWEEP_CAMPAIGN_GROUP = "A_tension_factorial"
GAINS = (50, 100, 200)
PAPER_DEFAULT_GAIN = 100
EIGENVALUE_REFERENCE_TOLERANCE_PERCENT = 0.5
SN_SEEDS_BY_GAIN = {50: (0, 1, 2), 100: (0,), 200: (0, 1, 2)}

# v5 gain-sweep reference (paper Section 3.5, Fig. 6). Every value was
# recomputed under the operating-point-weighted PEM, and the conclusion
# inverted: error grows MONOTONICALLY with gain in both damping groups. The
# v4.1 numbers, which fell with gain and made K_p* = 200 look competitive at
# 18.6%, are retired.
#
# Campaign condition: tension-only sensor noise, E_Toggle, T_log = 20 ms,
# pct_T = 0.3%, LPF 100 Hz. This is deliberately NOT the representative
# dual-channel condition of Table 1, where E_Toggle reads 42.6%.
#
# The running text prints the pooled median at all three gains but only the
# endpoint values for the two damping groups. The group value at K_p* = 100 is
# NOT unpublished, however: Fig. 6 plots all three gains as grouped bars, and
# `fig06_kp/data_bars.csv` in the figure package carries them. The 50 and 200
# entries below round to that file exactly (25.199 -> 25.2, 59.127 -> 59.1,
# 27.345 -> 27.3, 80.871 -> 80.9), as does the pooled series in PAPER_SN_MARE
# (32.514 -> 32.5, 30.784 -> 30.8, 32.019 -> 32.0), which identifies that file
# as the source of this table. The K_p* = 100 group values are therefore taken
# from it rather than left null.
#
# PAPER_NF_MARE keeps `None` at K_p* = 100: Fig. 6 is the sensor-noise figure
# and carries no noise-free series, so that value really is unpublished.
PAPER_NF_MARE = {
    50: 21.0,
    100: None,
    200: 32.0,
}
PAPER_SN_MARE = {
    50: 32.5,
    100: 30.8,
    200: 32.0,
}
PAPER_O_UD_MARE = {
    50: 25.2,
    100: 26.25492806405652,  # fig06_kp/data_bars.csv, bar_O_UD_pct
    200: 27.3,
}
PAPER_H_DAMP_MARE = {
    50: 59.1,
    100: 67.27995992343594,  # fig06_kp/data_bars.csv, bar_H_Damp_pct
    200: 80.9,
}
H_DAMP_TO_O_UD_RATIO = {
    50: 2.3,
    100: 2.6,
    200: 3.0,
}

# The pooled median dips at K_p* = 100, but that valley is a Simpson's-paradox
# composition artifact of the imbalanced 8:2 group mix. Neither group shares it.
# (Section 3.5 and the Fig. 6 caption; `simpsons_paradox` in the reference JSON.)
POOLED_MINIMUM_IS_COMPOSITION_ARTIFACT = True
PAPER_POOLED_MINIMUM_GAIN = 100
PAPER_POOLED_FLAT_BAND_PERCENT = (30.0, 33.0)

# Floors printed in Section 3.5, last paragraph ("no gain setting brings the
# pooled median ... below about 30%; even the more identifiable eight-plant
# group never falls below its 25.2% floor, and the highly damped pair stays
# above 59%").
PAPER_POOLED_FLOOR_PERCENT = 30.0
PAPER_O_UD_FLOOR_PERCENT = 25.2
PAPER_H_DAMP_FLOOR_PERCENT = 59.0

# Section 3.5, "A per-roller versus EA trade-off": the EA error climbs from
# 11.5% at the lowest gain to 39.4% at the highest, still under the tension-only
# sensor noise of the gain sweep.
PAPER_EA_ERROR_ACROSS_GAIN_PERCENT = {50: 11.5, 100: None, 200: 39.4}

# Section 3.5, "Gain tuning does not replace excitation design" plus footnote 3:
# ET1 reads 55.9% at the default gain, at the gain-sweep condition. ET1 itself
# was never swept over gain, so this is a single-cell comparison.
PAPER_ET1_AT_DEFAULT_GAIN_PERCENT = 55.9
ET1_EXCITATION = "ET1"

# Section 3.5, "What the gain sweep does show ...": P177 (dashboard P08)
# degrades from 56% at K_p* = 50 to 148% at K_p* = 200, and is the clipped
# off-scale triangle in Fig. 6.
PAPER_PER_PLANT_MARE_PERCENT = {"P08": {50: 56.0, 100: None, 200: 148.0}}

# Section 3.1, p. 8: tau_min = 1 / max_i |Re(lambda_i)| on the OPEN-LOOP plant,
# spanning 7.5 ms (P158 = P06) to 67.4 ms (P139 = P05).
PAPER_TAU_MIN_ENDPOINTS_MS = {"P06": 7.5, "P05": 67.4}

PAPER_GAIN_CONCLUSION = (
    "Error grows monotonically with K_p* in both damping groups; a higher gain buys no "
    "noise robustness. K_p* = 100 is kept for identifiability and safety, because it "
    "preserves the EA estimate that the highest gains destroy (11.5% -> 39.4%)."
)

STEP_PLANT_ID = "P07"
STEP_TIME_S = 1.0
STEP_RATIO = 0.20
STEP_DURATION_S = 4.0
# The paper publishes NO transient metric for the gain sweep. Section 3.5: "the
# gain sweep logs no transient metric, so the overshoot cost of an off-default
# gain is not quantified here", and `data/reference_results/README.md` lists it
# under "Values the paper deliberately does not publish". The v4.1-era t90 and
# overshoot targets that used to sit here had no traceable source and were
# removed; the step trajectory is still simulated, but only as an unreferenced
# dashboard diagnostic.
STEP_METRICS_ARE_PAPER_REFERENCED = False
PAPER_STEP_METRIC_STATUS = "not_published_gain_sweep_logs_no_transient_metric"


def _write_summary(name: str, payload: Mapping[str, object]) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / name
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return str(path)


def _write_rows_csv(
    name: str,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return str(path)


def _cached_closed_loop_damping_study() -> dict[str, object] | None:
    summary_path = SUMMARY_DIR / "closed_loop_damping_summary.json"
    csv_path = SUMMARY_DIR / "closed_loop_damping_dashboard_vs_paper_comparison.csv"
    if not summary_path.exists() or not csv_path.exists():
        return None
    try:
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metrics.get("cache_version") != CACHE_VERSION:
        return None
    required = (
        "raw_rows",
        "comparison_rows",
        "gain_rows",
        "regime_rows",
        "eigenvalue_rows",
        "eigenvalue_summary",
        "step_response_rows",
        "validation_status",
        "simpsons_paradox",
        "tau_min_rows",
        "ea_error_rows",
        "et1_comparison_row",
        "per_plant_comparison_rows",
        "floor_rows",
    )
    if any(key not in metrics for key in required):
        return None
    return {
        "study": "closed_loop_damping",
        "metrics": metrics,
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
    }


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else None


def _percent_error(reference: float, calculated: float | None) -> float | None:
    if calculated is None or not math.isfinite(calculated):
        return None
    return abs(reference - calculated) / abs(reference) * 100.0


def _paper_controller_config(params, line_speed_m_s: float, kp_star: int) -> ControllerConfig:
    return ControllerConfig(
        target_tension_N=params.tension_ref_N,
        line_speed_m_s=line_speed_m_s,
        Kp_star_m_s_per_N=float(kp_star),
        TI_s=auto_tension_integral_time_s(params, line_speed_m_s),
        high_ea_kp_cap_enabled=False,
        feedforward_uses_measured_omega=True,
        paper_velocity_gain_enabled=True,
        velocity_correction_limit_fraction=None,
        steady_velocity_uses_dynamic_target=False,
    )


def _figure7_group(regime: object) -> str:
    return "H-Damp" if str(regime) == "H-Damp" else "O-UD"


def _modal_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Calculate full-cascade eigenvalues without using paper damping targets."""

    summary_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    for kp_star in GAINS:
        for plant in plant_registry():
            plant_id = str(plant["plant_id"])
            params, meta = parameters_for_plant(plant_id)
            line_speed = float(meta["v_ref_m_s"])
            sensitivity = closed_loop_modal_sensitivity(
                params,
                _paper_controller_config(params, line_speed, kp_star),
                line_speed_m_s=line_speed,
            )
            selected = sensitivity["selected"]
            calculated_zeta = float(selected["zeta_cl_min"])
            reference_zeta = (
                float(meta["zeta_cl_min"])
                if kp_star == PAPER_DEFAULT_GAIN
                else None
            )
            reference_error = (
                _percent_error(reference_zeta, calculated_zeta)
                if reference_zeta is not None
                else None
            )
            summary_rows.append(
                {
                    "plant_id": plant_id,
                    "source_pool_id": str(meta["pool_id"]),
                    "kp_star": kp_star,
                    "calculated_zeta_cl_min": calculated_zeta,
                    "calculated_regime": str(selected["regime"]),
                    "reference_zeta_cl_min_at_kp100": reference_zeta,
                    "reference_regime_at_kp100": (
                        str(meta["regime"])
                        if kp_star == PAPER_DEFAULT_GAIN
                        else None
                    ),
                    "reference_error_percent": reference_error,
                    "stable": bool(selected["stable"]),
                    "unstable_mode_count": int(selected["unstable_mode_count"]),
                    "spectral_abscissa_per_s": float(
                        selected["spectral_abscissa_per_s"]
                    ),
                    "oscillatory_pair_count": int(
                        selected["oscillatory_pair_count"]
                    ),
                    "equilibrium_residual_max_abs": float(
                        selected["equilibrium_residual_max_abs"]
                    ),
                    "jacobian_relative_step": float(selected["relative_step"]),
                    "numerical_iteration_count": len(sensitivity["iterations"]),
                    "zeta_step_sensitivity_spread": float(
                        sensitivity["zeta_step_sensitivity_spread"]
                    ),
                    "calculation_source": "live_full_cascade_eigenvalue_jacobian",
                }
            )
            for mode_index, mode in enumerate(selected["modes"], start=1):
                mode_rows.append(
                    {
                        "plant_id": plant_id,
                        "kp_star": kp_star,
                        "mode_index": mode_index,
                        **mode,
                    }
                )
    return summary_rows, mode_rows


def _parameter_relative_error_percent(sysid: Any, parameter: str) -> float | None:
    """Return |relative error| in percent for one identified parameter."""

    for row in getattr(sysid, "error_table", ()) or ():
        if str(row.get("parameter")) == parameter:
            value = row.get("relative_error")
            if value is None or not math.isfinite(float(value)):
                return None
            return abs(float(value)) * 100.0
    return None


def _tau_min_rows() -> list[dict[str, object]]:
    """Calculate the paper's per-plant ``tau_min`` from the OPEN-LOOP plant.

    Section 3.1 (p. 8) defines ``tau_min = 1 / max_i |Re(lambda_i)|`` and the
    abstract/Contribution C1 call it "the fastest open-loop modal time scale".
    Only the two endpoints are published (7.5 ms for P158, 67.4 ms for P139),
    so only those two rows carry a paper comparison; the other eight are
    calculated values with no printed target.

    `backend/validation/studies.py` currently applies a flat
    ``DEFAULT_TMIN_MS = 50.0`` to all ten plants, which is wrong for nine of
    them; these rows are the per-plant replacement.
    """

    rows: list[dict[str, object]] = []
    for plant in plant_registry():
        plant_id = str(plant["plant_id"])
        params, meta = parameters_for_plant(plant_id)
        analysis = open_loop_modal_analysis(
            params, line_speed_m_s=float(meta["v_ref_m_s"])
        )
        calculated_ms = float(analysis["tau_min_ms"])
        paper_ms = PAPER_TAU_MIN_ENDPOINTS_MS.get(plant_id)
        rows.append(
            {
                "plant_id": plant_id,
                "source_pool_id": str(meta["pool_id"]),
                "calculated_tau_min_ms": calculated_ms,
                "paper_tau_min_ms": paper_ms,
                "tau_min_error_percent": (
                    None if paper_ms is None else _percent_error(paper_ms, calculated_ms)
                ),
                "paper_reference_status": (
                    "paper_reference_endpoint"
                    if paper_ms is not None
                    else "not_published_only_the_range_endpoints_are_printed"
                ),
                "tau_min_over_Tlog_at_20ms": calculated_ms / (SYSID_TLOG_S * 1000.0),
                "nf_logging_guideline_met": (
                    calculated_ms / (SYSID_TLOG_S * 1000.0) >= 5.0
                ),
                "equilibrium_residual_max_abs": float(
                    analysis["equilibrium_residual_max_abs"]
                ),
                "calculation_source": "live_open_loop_six_state_eigenvalue_jacobian",
                "formula": str(analysis["formula"]),
            }
        )
    return rows


def _run_gain_case(
    *,
    plant: Mapping[str, Any],
    kp_star: int,
    condition: str,
    seed: int,
    excitation: str = GAIN_SWEEP_EXCITATION,
) -> dict[str, object]:
    plant_id = str(plant["plant_id"])
    params, meta = parameters_for_plant(plant_id)
    line_speed = float(meta["v_ref_m_s"])
    amplitude = float(meta["recommended_excitation_amplitude_V"])
    profile = get_excitation_profile(
        excitation, amplitude, campaign_group=GAIN_SWEEP_CAMPAIGN_GROUP
    )
    noisy = condition == "SN"
    noise_sigma = (
        SN_NOISE_LEVEL_PERCENT / 100.0 * float(meta["T_max_N"])
        if noisy
        else 0.0
    )
    try:
        record_duration_s = getattr(profile, "duration_s", None)
        if record_duration_s is None:
            raise ValueError(
                f"excitation {excitation!r} carries no record duration; the "
                "duration must come from excitation_schedules.csv, never "
                "from a hardcoded constant"
            )
        sim = simulate(
            params,
            controller_config=_paper_controller_config(params, line_speed, kp_star),
            config=SimulationConfig(
                duration_s=float(record_duration_s),
                dt_s=DT_S,
                controller_sample_time_s=CONTROLLER_SAMPLE_TIME_S,
                log_sample_time_s=SYSID_TLOG_S,
                line_speed_m_s=line_speed,
                sensor_noise_tension_N=noise_sigma,
                sensor_noise_omega_rad_s=0.0,
                sensor_lpf_hz=SN_LPF_HZ if noisy else None,
                noise_affects_controller=True,
                noise_rng="numpy_default_rng",
                seed=int(seed),
            ),
            excitation=profile,
            write_output=False,
        )
        sysid = estimate_parameters_weighted_pem(
            sim.rows,
            nominal_params=params,
            true_params=params,
            max_nfev=150,
        )
        mare_theta = float(sysid.mare_theta)
        mare_percent = 100.0 * mare_theta
        ea_error_percent = _parameter_relative_error_percent(sysid, "EA")
        record_duration = float(record_duration_s)
        samples = len(sim.rows)
        status = "ok"
    except Exception as exc:
        mare_theta = None
        mare_percent = None
        ea_error_percent = None
        record_duration = None
        samples = 0
        status = f"failed:{type(exc).__name__}:{exc}"
    return {
        "kp_star": int(kp_star),
        "condition": condition,
        "seed": int(seed),
        "plant_id": plant_id,
        "excitation": excitation,
        "campaign_group": GAIN_SWEEP_CAMPAIGN_GROUP,
        "record_duration_s": record_duration,
        "measurement_condition": (
            MEASUREMENT_CONDITION if noisy else "noise_free"
        ),
        "pct_T": SN_NOISE_LEVEL_PERCENT if noisy else 0.0,
        "pct_v": SN_VELOCITY_NOISE_LEVEL_PERCENT,
        "tension_lpf_hz": SN_LPF_HZ if noisy else None,
        "velocity_lpf_hz": None,
        "EA_relative_error_percent": ea_error_percent,
        "source_pool_id": str(meta.get("pool_id", "")),
        "reference_regime_at_kp100": str(meta["regime"]),
        "reference_zeta_cl_min_at_kp100": float(meta["zeta_cl_min"]),
        "dashboard_MARE_theta_percent": mare_percent,
        "MARE_theta": mare_theta,
        "samples": samples,
        "sensor_noise_sigma_N": noise_sigma,
        "controller_sample_time_s": CONTROLLER_SAMPLE_TIME_S,
        "logging_sample_time_s": SYSID_TLOG_S,
        "controller_integral_time_s": auto_tension_integral_time_s(
            params,
            line_speed,
        ),
        "high_ea_kp_cap_enabled": False,
        "velocity_correction_limit_fraction": None,
        "noise_affects_controller": True,
        "noise_rng": "numpy.default_rng(seed)",
        "estimator": "paper_eq8_weighted_pem_trf",
        "source": "fresh_dashboard_simulation",
        "value_status": status,
    }


def _comparison_rows(
    raw_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected_counts = {
        ("NF", 50): 10,
        ("NF", 100): 10,
        ("NF", 200): 10,
        ("SN", 50): 30,
        ("SN", 100): 10,
        ("SN", 200): 30,
    }
    for condition, paper_values in (("NF", PAPER_NF_MARE), ("SN", PAPER_SN_MARE)):
        for kp_star in GAINS:
            values = [
                float(row["dashboard_MARE_theta_percent"])
                for row in raw_rows
                if row["condition"] == condition
                and int(row["kp_star"]) == kp_star
                and row.get("value_status", "ok") == "ok"
                and row.get("dashboard_MARE_theta_percent") is not None
            ]
            dashboard = _median(values)
            # The paper does not print a group value at K_p* = 100; a missing
            # reference means "not published", so the row carries the dashboard
            # value with no comparison rather than a fabricated target.
            raw_paper = paper_values[kp_star]
            paper = None if raw_paper is None else float(raw_paper)
            error_percent = None if paper is None else _percent_error(paper, dashboard)
            rows.append(
                {
                    "condition": condition,
                    "kp_star": kp_star,
                    "paper_MARE_theta_percent": paper,
                    "dashboard_MARE_theta_percent": dashboard,
                    "validation_error_percent": (
                        round(error_percent, 3)
                        if error_percent is not None
                        else None
                    ),
                    "pass_fail": (
                        "NO_REFERENCE"
                        if paper is None
                        else "PASS"
                        if error_percent is not None and error_percent <= 15.0
                        else "CHECK"
                    ),
                    "paper_reference_status": (
                        "not_published_at_this_gain" if paper is None else "paper_reference"
                    ),
                    "valid_run_count": len(values),
                    "expected_run_count": expected_counts[(condition, kp_star)],
                    "aggregation": "median_over_requested_plant_seed_rows",
                    "dashboard_source": "fresh_simulation_eq8_weighted_pem",
                }
            )
    return rows


def _plant_median_rows(
    raw_rows: Sequence[Mapping[str, object]],
    eigenvalue_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kp_star in GAINS:
        for plant in plant_registry():
            plant_id = str(plant["plant_id"])
            matching = [
                row
                for row in raw_rows
                if row["condition"] == "SN"
                and int(row["kp_star"]) == kp_star
                and row["plant_id"] == plant_id
                and row.get("value_status") == "ok"
                and row.get("dashboard_MARE_theta_percent") is not None
            ]
            values = [
                float(row["dashboard_MARE_theta_percent"])
                for row in matching
            ]
            eigenvalue = next(
                row
                for row in eigenvalue_rows
                if row["plant_id"] == plant_id
                and int(row["kp_star"]) == PAPER_DEFAULT_GAIN
            )
            calculated_regime = str(eigenvalue["calculated_regime"])
            rows.append(
                {
                    "kp_star": kp_star,
                    "plant_id": plant_id,
                    "calculated_zeta_cl_min_at_kp100": float(
                        eigenvalue["calculated_zeta_cl_min"]
                    ),
                    "calculated_regime_at_kp100": calculated_regime,
                    "calculated_figure7_group": _figure7_group(
                        calculated_regime
                    ),
                    "reference_zeta_cl_min_at_kp100": float(
                        plant["zeta_cl_min"]
                    ),
                    "reference_regime_at_kp100": str(plant["regime"]),
                    "dashboard_plant_median_MARE_theta_percent": _median(values),
                    "valid_seed_count": len(values),
                    "expected_seed_count": len(SN_SEEDS_BY_GAIN[kp_star]),
                }
            )
    return rows


def _regime_rows(
    plant_median_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kp_star in GAINS:
        grouped: dict[str, list[float]] = {"O-UD": [], "H-Damp": []}
        for row in plant_median_rows:
            if int(row["kp_star"]) != kp_star:
                continue
            value = row.get("dashboard_plant_median_MARE_theta_percent")
            if value is None:
                continue
            grouped[str(row["calculated_figure7_group"])].append(float(value))
        dashboard_o_ud = _median(grouped["O-UD"])
        dashboard_h_damp = _median(grouped["H-Damp"])
        ratio = (
            dashboard_h_damp / dashboard_o_ud
            if dashboard_h_damp is not None
            and dashboard_o_ud is not None
            and dashboard_o_ud != 0.0
            else None
        )
        rows.append(
            {
                "kp_star": kp_star,
                "paper_O_UD_MARE_theta_percent": PAPER_O_UD_MARE[kp_star],
                "paper_H_Damp_MARE_theta_percent": PAPER_H_DAMP_MARE[kp_star],
                "dashboard_O_UD_MARE_theta_percent": dashboard_o_ud,
                "dashboard_H_Damp_MARE_theta_percent": dashboard_h_damp,
                "paper_H_Damp_to_O_UD_ratio": H_DAMP_TO_O_UD_RATIO[kp_star],
                "dashboard_H_Damp_to_O_UD_ratio": (
                    round(ratio, 6) if ratio is not None else None
                ),
                "O_UD_plant_count": len(grouped["O-UD"]),
                "H_Damp_plant_count": len(grouped["H-Damp"]),
                "plant_median_marker": len(grouped["O-UD"]) + len(grouped["H-Damp"]),
                "aggregation": (
                    "median_of_per_plant_SN_seed_medians_grouped_by_"
                    "calculated_kp100_full_cascade_eigenvalue_regime"
                ),
            }
        )
    return rows


def _et1_comparison_row(
    et1_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare the single published ET1 cell at the default gain (55.9%).

    Section 3.5, footnote 3: "the 55.9% figure is the ET1 value at the default
    K_p* = 100, read at the gain-sweep condition (tension-only sensor noise at
    0.3% of Tmax, LPF 100 Hz, Tlog = 20 ms), and ET1 itself was not swept over
    gain." The paper does not state which seeds that cell pools, so both the
    single-seed and the three-seed pooled medians are reported and the seed
    convention is flagged as inferred, not sourced.
    """

    usable = [
        row
        for row in et1_rows
        if row.get("value_status") == "ok"
        and row.get("dashboard_MARE_theta_percent") is not None
    ]
    pooled = _median(
        [float(row["dashboard_MARE_theta_percent"]) for row in usable]
    )
    single_seed = _median(
        [
            float(row["dashboard_MARE_theta_percent"])
            for row in usable
            if int(row["seed"]) == 0
        ]
    )
    return {
        "excitation": ET1_EXCITATION,
        "kp_star": PAPER_DEFAULT_GAIN,
        "condition": "SN",
        "paper_MARE_theta_percent": PAPER_ET1_AT_DEFAULT_GAIN_PERCENT,
        "dashboard_MARE_theta_percent_pooled_seeds": pooled,
        "dashboard_MARE_theta_percent_seed0": single_seed,
        "validation_error_percent_pooled_seeds": _percent_error(
            PAPER_ET1_AT_DEFAULT_GAIN_PERCENT, pooled
        ),
        "validation_error_percent_seed0": _percent_error(
            PAPER_ET1_AT_DEFAULT_GAIN_PERCENT, single_seed
        ),
        "valid_run_count": len(usable),
        "seed_convention": "inferred_seeds_0_1_2; the paper does not state the ET1 seed pool",
        "paper_reference_status": "paper_reference",
        "note": (
            "ET1 was not swept over gain; this is the one published ET1 cell at "
            "the gain-sweep condition."
        ),
    }


def _ea_error_rows(
    raw_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Median |EA relative error| per gain under the gain-sweep noise condition.

    Section 3.5: "the EA error climbs from 11.5% to 39.4% across the gain
    range", still under the tension-only sensor noise of the gain sweep. Only
    the two endpoints are printed.
    """

    rows: list[dict[str, object]] = []
    for kp_star in GAINS:
        values = [
            float(row["EA_relative_error_percent"])
            for row in raw_rows
            if row["condition"] == "SN"
            and int(row["kp_star"]) == kp_star
            and row.get("value_status") == "ok"
            and row.get("EA_relative_error_percent") is not None
        ]
        dashboard = _median(values)
        paper = PAPER_EA_ERROR_ACROSS_GAIN_PERCENT[kp_star]
        rows.append(
            {
                "kp_star": kp_star,
                "paper_EA_error_percent": paper,
                "dashboard_EA_error_percent": dashboard,
                "validation_error_percent": (
                    None if paper is None else _percent_error(paper, dashboard)
                ),
                "paper_reference_status": (
                    "paper_reference_endpoint"
                    if paper is not None
                    else "not_published_only_the_range_endpoints_are_printed"
                ),
                "valid_run_count": len(values),
                "aggregation": "median_abs_EA_relative_error_over_SN_plant_seed_rows",
            }
        )
    return rows


def _per_plant_comparison_rows(
    plant_median_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare the one per-plant value the paper prints (P177 = dashboard P08)."""

    rows: list[dict[str, object]] = []
    for plant_id, by_gain in PAPER_PER_PLANT_MARE_PERCENT.items():
        for kp_star in GAINS:
            match = next(
                (
                    row
                    for row in plant_median_rows
                    if str(row["plant_id"]) == plant_id
                    and int(row["kp_star"]) == kp_star
                ),
                None,
            )
            dashboard = (
                None
                if match is None
                else match.get("dashboard_plant_median_MARE_theta_percent")
            )
            paper = by_gain[kp_star]
            rows.append(
                {
                    "plant_id": plant_id,
                    "source_pool_id": "P177" if plant_id == "P08" else "",
                    "kp_star": kp_star,
                    "paper_plant_median_MARE_theta_percent": paper,
                    "dashboard_plant_median_MARE_theta_percent": dashboard,
                    "validation_error_percent": (
                        None
                        if paper is None or dashboard is None
                        else _percent_error(paper, float(dashboard))
                    ),
                    "paper_reference_status": (
                        "paper_reference" if paper is not None else "not_published"
                    ),
                    "note": (
                        "Fig. 6 clipped off-scale triangle at K_p* = 200"
                        if kp_star == 200
                        else ""
                    ),
                }
            )
    return rows


def _floor_rows(
    comparison_rows: Sequence[Mapping[str, object]],
    regime_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Check the three floors Section 3.5 states across the whole sweep."""

    pooled = [
        float(row["dashboard_MARE_theta_percent"])
        for row in comparison_rows
        if row["condition"] == "SN"
        and row.get("dashboard_MARE_theta_percent") is not None
    ]
    o_ud = [
        float(row["dashboard_O_UD_MARE_theta_percent"])
        for row in regime_rows
        if row.get("dashboard_O_UD_MARE_theta_percent") is not None
    ]
    h_damp = [
        float(row["dashboard_H_Damp_MARE_theta_percent"])
        for row in regime_rows
        if row.get("dashboard_H_Damp_MARE_theta_percent") is not None
    ]
    return [
        {
            "floor": "pooled_median_never_below_percent",
            "paper_value_percent": PAPER_POOLED_FLOOR_PERCENT,
            "dashboard_minimum_percent": min(pooled) if pooled else None,
            "holds": bool(pooled) and min(pooled) >= PAPER_POOLED_FLOOR_PERCENT,
            "paper_wording": "no gain setting brings the pooled median below about 30%",
        },
        {
            "floor": "O_UD_plus_H_Osc_floor_percent",
            "paper_value_percent": PAPER_O_UD_FLOOR_PERCENT,
            "dashboard_minimum_percent": min(o_ud) if o_ud else None,
            "holds": bool(o_ud) and min(o_ud) >= PAPER_O_UD_FLOOR_PERCENT,
            "paper_wording": (
                "the more identifiable eight-plant group never falls below its 25.2% floor"
            ),
        },
        {
            "floor": "H_Damp_always_above_percent",
            "paper_value_percent": PAPER_H_DAMP_FLOOR_PERCENT,
            "dashboard_minimum_percent": min(h_damp) if h_damp else None,
            "holds": bool(h_damp) and min(h_damp) >= PAPER_H_DAMP_FLOOR_PERCENT,
            "paper_wording": "the highly damped pair stays above 59%",
        },
    ]


def _monotone_increasing(values: Sequence[float | None]) -> bool:
    finite = [value for value in values if value is not None]
    if len(finite) != len(values) or len(finite) < 2:
        return False
    return all(
        float(finite[index]) < float(finite[index + 1])
        for index in range(len(finite) - 1)
    )


def _simpsons_paradox_block(
    comparison_rows: Sequence[Mapping[str, object]],
    regime_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Decompose the pooled curve into its two damping groups.

    This is the v5 headline: the pooled dip at K_p* = 100 is a composition
    artifact of the imbalanced 8:2 group mix, and NEITHER group shares it.
    """

    pooled = {
        int(row["kp_star"]): row["dashboard_MARE_theta_percent"]
        for row in comparison_rows
        if row["condition"] == "SN"
    }
    o_ud = {
        int(row["kp_star"]): row["dashboard_O_UD_MARE_theta_percent"]
        for row in regime_rows
    }
    h_damp = {
        int(row["kp_star"]): row["dashboard_H_Damp_MARE_theta_percent"]
        for row in regime_rows
    }
    o_ud_monotone = _monotone_increasing([o_ud[kp] for kp in GAINS])
    h_damp_monotone = _monotone_increasing([h_damp[kp] for kp in GAINS])
    pooled_values = {kp: pooled[kp] for kp in GAINS if pooled.get(kp) is not None}
    pooled_minimum_gain = (
        min(pooled_values, key=pooled_values.get) if pooled_values else None
    )
    pooled_has_interior_minimum = (
        pooled_minimum_gain is not None
        and pooled_minimum_gain not in (GAINS[0], GAINS[-1])
    )
    band_low, band_high = PAPER_POOLED_FLAT_BAND_PERCENT
    return {
        "paper_claim": (
            "Error grows monotonically with K_p* in BOTH damping groups; the "
            "pooled dip at K_p* = 100 is a Simpson's-paradox composition "
            "artifact of the imbalanced 8:2 group mix."
        ),
        "paper_source": "Section 3.5 and the Fig. 6 caption; simpsons_paradox in the reference JSON",
        "group_sizes": {"O_UD_plus_H_Osc": 8, "H_Damp": 2},
        "dashboard_pooled_percent_by_gain": pooled,
        "dashboard_O_UD_percent_by_gain": o_ud,
        "dashboard_H_Damp_percent_by_gain": h_damp,
        "dashboard_O_UD_monotone_increasing": o_ud_monotone,
        "dashboard_H_Damp_monotone_increasing": h_damp_monotone,
        "dashboard_both_groups_monotone_increasing": o_ud_monotone and h_damp_monotone,
        "paper_pooled_minimum_gain": PAPER_POOLED_MINIMUM_GAIN,
        "dashboard_pooled_minimum_gain": pooled_minimum_gain,
        "dashboard_pooled_has_interior_minimum": pooled_has_interior_minimum,
        "pooled_minimum_is_composition_artifact": POOLED_MINIMUM_IS_COMPOSITION_ARTIFACT,
        "paper_pooled_flat_band_percent": list(PAPER_POOLED_FLAT_BAND_PERCENT),
        "dashboard_pooled_inside_flat_band": all(
            band_low <= float(value) <= band_high for value in pooled_values.values()
        )
        if pooled_values
        else False,
        "reading_rule": (
            "Do not read the pooled minimum as an accuracy result. The pooled "
            "curve is reported only so the artifact can be shown; the group "
            "curves carry the conclusion."
        ),
    }


def _make_step_profile(amplitude_N: float):
    def profile(time_s: float) -> tuple[float, float, float]:
        return (
            float(amplitude_N) if time_s >= STEP_TIME_S else 0.0,
            0.0,
            0.0,
        )

    profile.duration_s = STEP_DURATION_S
    return profile


def _interpolated_crossing_time(
    rows: Sequence[Mapping[str, float]],
    target_N: float,
) -> float | None:
    for previous, current in zip(rows, rows[1:]):
        y0 = float(previous["T1"])
        y1 = float(current["T1"])
        if y0 < target_N <= y1:
            t0 = float(previous["time_s"])
            t1 = float(current["time_s"])
            if math.isclose(y0, y1, abs_tol=1e-15):
                return t1
            fraction = (target_N - y0) / (y1 - y0)
            return t0 + fraction * (t1 - t0)
    return None


def _settling_time(
    rows: Sequence[Mapping[str, float]],
    final_N: float,
    band_N: float,
) -> float | None:
    post = [row for row in rows if float(row["time_s"]) >= STEP_TIME_S]
    for index, row in enumerate(post):
        if all(
            abs(float(candidate["T1"]) - final_N) <= band_N
            for candidate in post[index:]
        ):
            return float(row["time_s"]) - STEP_TIME_S
    return None


def _step_rows_and_metrics() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    params, meta = parameters_for_plant(STEP_PLANT_ID)
    line_speed = float(meta["v_ref_m_s"])
    initial_N = float(meta["T_ref_N"])
    amplitude_N = STEP_RATIO * initial_N
    final_N = initial_N + amplitude_N
    profile = _make_step_profile(amplitude_N)
    display_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    reference_written = False

    for kp_star in GAINS:
        sim = simulate(
            params,
            controller_config=_paper_controller_config(params, line_speed, kp_star),
            config=SimulationConfig(
                duration_s=STEP_DURATION_S,
                dt_s=DT_S,
                controller_sample_time_s=CONTROLLER_SAMPLE_TIME_S,
                log_sample_time_s=DT_S,
                line_speed_m_s=line_speed,
                sensor_noise_tension_N=0.0,
                sensor_noise_omega_rad_s=0.0,
                sensor_lpf_hz=None,
                noise_affects_controller=True,
                noise_rng="numpy_default_rng",
                seed=0,
            ),
            excitation=profile,
            write_output=False,
        )
        post = [row for row in sim.rows if float(row["time_s"]) >= STEP_TIME_S]
        peak_row = max(post, key=lambda row: float(row["T1"]))
        peak_N = float(peak_row["T1"])
        peak_time_s = float(peak_row["time_s"])
        overshoot_percent = max(0.0, (peak_N - final_N) / amplitude_N * 100.0)
        t10_cross = _interpolated_crossing_time(
            sim.rows,
            initial_N + 0.10 * amplitude_N,
        )
        t90_cross = _interpolated_crossing_time(
            sim.rows,
            initial_N + 0.90 * amplitude_N,
        )
        t90_s = (
            t90_cross - STEP_TIME_S
            if t90_cross is not None
            else None
        )
        rise_time_s = (
            t90_cross - t10_cross
            if t90_cross is not None and t10_cross is not None
            else None
        )
        settling_time_s = _settling_time(
            sim.rows,
            final_N,
            0.02 * amplitude_N,
        )
        metric_rows.append(
            {
                "kp_star": kp_star,
                "step_plant_id": STEP_PLANT_ID,
                "step_pool_id": str(meta["pool_id"]),
                "step_time_s": STEP_TIME_S,
                "step_ratio": STEP_RATIO,
                "initial_tension_N": initial_N,
                "final_tension_N": final_N,
                "peak_tension_N": peak_N,
                "peak_time_s": peak_time_s,
                "peak_normalized_tension": peak_N / initial_N,
                "t90_s": t90_s,
                "rise_time_10_90_s": rise_time_s,
                "settling_time_2pct_step_s": settling_time_s,
                "overshoot_percent": overshoot_percent,
                "paper_t90_s": None,
                "paper_overshoot_percent": None,
                "t90_error_percent": None,
                "overshoot_error_percent": None,
                "paper_reference_status": PAPER_STEP_METRIC_STATUS,
                "response_source": "fresh_P07_1ms_step_simulation",
            }
        )
        for row in sim.rows[::5]:
            display_rows.append(
                {
                    "kp_star": kp_star,
                    "time_s": round(float(row["time_s"]), 6),
                    "normalized_tension": float(row["T1"]) / initial_N,
                    "normalized_reference": float(row["T1_ref_N"]) / initial_N,
                }
            )
            if not reference_written:
                display_rows.append(
                    {
                        "kp_star": "Tref",
                        "time_s": round(float(row["time_s"]), 6),
                        "normalized_tension": float(row["T1_ref_N"]) / initial_N,
                        "normalized_reference": float(row["T1_ref_N"]) / initial_N,
                    }
                )
        reference_written = True

    return display_rows, metric_rows


def _gain_rows(
    comparison_rows: Sequence[Mapping[str, object]],
    step_metric_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kp_star in GAINS:
        nf = next(
            row
            for row in comparison_rows
            if row["condition"] == "NF" and int(row["kp_star"]) == kp_star
        )
        sn = next(
            row
            for row in comparison_rows
            if row["condition"] == "SN" and int(row["kp_star"]) == kp_star
        )
        step = next(
            row for row in step_metric_rows if int(row["kp_star"]) == kp_star
        )
        rows.append(
            {
                "kp_star": kp_star,
                "response_behavior": "fresh simulated gain sweep",
                "t90_s": step["t90_s"],
                "rise_time_10_90_s": step["rise_time_10_90_s"],
                "settling_time_2pct_step_s": step["settling_time_2pct_step_s"],
                "overshoot_percent": step["overshoot_percent"],
                "peak_time_s": step["peak_time_s"],
                "peak_normalized_tension": step["peak_normalized_tension"],
                "NF_dashboard_MARE_theta_percent": nf[
                    "dashboard_MARE_theta_percent"
                ],
                "SN_dashboard_MARE_theta_percent": sn[
                    "dashboard_MARE_theta_percent"
                ],
            }
        )
    return rows


def _acceptance_checks(
    raw_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    regime_rows: Sequence[Mapping[str, object]],
    gain_rows: Sequence[Mapping[str, object]],
    eigenvalue_rows: Sequence[Mapping[str, object]],
    tau_min_rows: Sequence[Mapping[str, object]],
    simpsons: Mapping[str, object],
    floor_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    finite_rows = [
        row
        for row in raw_rows
        if row.get("value_status") == "ok"
        and row.get("dashboard_MARE_theta_percent") is not None
    ]
    expected_raw_count = 100
    exact_settings = bool(finite_rows) and all(
        math.isclose(
            float(row["controller_sample_time_s"]),
            CONTROLLER_SAMPLE_TIME_S,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(row["logging_sample_time_s"]),
            SYSID_TLOG_S,
            abs_tol=1e-12,
        )
        and row["high_ea_kp_cap_enabled"] is False
        and row["velocity_correction_limit_fraction"] is None
        and row["noise_rng"] == "numpy.default_rng(seed)"
        and row["estimator"] == "paper_eq8_weighted_pem_trf"
        for row in finite_rows
    )
    regime_complete = all(
        int(row["O_UD_plant_count"]) == 8
        and int(row["H_Damp_plant_count"]) == 2
        for row in regime_rows
    )
    baseline_eigenvalues = [
        row
        for row in eigenvalue_rows
        if int(row["kp_star"]) == PAPER_DEFAULT_GAIN
    ]
    eigenvalue_complete = (
        len(eigenvalue_rows) == len(GAINS) * len(plant_registry())
        and all(bool(row["stable"]) for row in eigenvalue_rows)
        and all(
            math.isfinite(float(row["calculated_zeta_cl_min"]))
            for row in eigenvalue_rows
        )
    )
    baseline_reference_match = (
        len(baseline_eigenvalues) == len(plant_registry())
        and all(
            row.get("reference_error_percent") is not None
            and float(row["reference_error_percent"])
            <= EIGENVALUE_REFERENCE_TOLERANCE_PERCENT
            and row["calculated_regime"] == row["reference_regime_at_kp100"]
            for row in baseline_eigenvalues
        )
    )
    maximum_numerical_spread = max(
        float(row["zeta_step_sensitivity_spread"])
        for row in eigenvalue_rows
    )
    t90 = [float(row["t90_s"]) for row in gain_rows]
    overshoot = [float(row["overshoot_percent"]) for row in gain_rows]
    sn = {
        int(row["kp_star"]): float(row["dashboard_MARE_theta_percent"])
        for row in comparison_rows
        if row["condition"] == "SN"
        and row.get("dashboard_MARE_theta_percent") is not None
    }
    raw_sn_minimum_gain = min(sn, key=sn.get) if sn else None
    paper_comparison_passes = sum(
        1 for row in comparison_rows if row["pass_fail"] == "PASS"
    )
    return [
        {
            "category": "calculation_integrity",
            "criterion": "All requested gain-sweep rows are finite",
            "passed": len(finite_rows) == expected_raw_count,
            "evidence": f"Finite rows={len(finite_rows)}/{expected_raw_count}.",
        },
        {
            "category": "calculation_integrity",
            "criterion": "Every row uses 1 ms control, 20 ms logging, auto-Ti, no cap/clamp, and Eq. (8) weighted PEM",
            "passed": exact_settings,
            "evidence": (
                "controller=1 ms; Tlog=20 ms; auto-Ti; high-EA cap=false; "
                "velocity clamp=None; RNG=numpy.default_rng(seed); "
                "estimator=paper_eq8_weighted_pem_trf."
            ),
        },
        {
            "category": "calculation_integrity",
            "criterion": "All 30 full-cascade eigenvalue analyses are finite and stable",
            "passed": eigenvalue_complete,
            "evidence": (
                f"Calculated rows={len(eigenvalue_rows)}/30; "
                f"stable rows={sum(1 for row in eigenvalue_rows if row['stable'])}; "
                "no Table S12 zeta enters the Jacobian."
            ),
        },
        {
            "category": "paper_comparison",
            "criterion": "All ten calculated Kp*=100 damping ratios and regimes match rounded Table S12",
            "passed": baseline_reference_match,
            "evidence": (
                f"Rows={len(baseline_eigenvalues)}/10; tolerance="
                f"{EIGENVALUE_REFERENCE_TOLERANCE_PERCENT:.1f}%; maximum "
                f"central-difference spread={maximum_numerical_spread:.3e}."
            ),
        },
        {
            "category": "calculation_integrity",
            "criterion": "Figure 7 calculated-eigenvalue grouping contains O-UD=8 and H-Damp=2 plants",
            "passed": regime_complete,
            "evidence": (
                "Kp*=100 eigenvalue regimes are calculated live; H-Osc P03/P07 "
                "are merged into the paper Figure 7 O-UD display group."
            ),
        },
        {
            "category": "calculation_integrity",
            "criterion": "Step metrics come from the actual P07 +20% 1 ms trajectory",
            "passed": all(
                row.get("response_behavior") == "fresh simulated gain sweep"
                and row.get("t90_s") is not None
                and row.get("overshoot_percent") is not None
                for row in gain_rows
            ),
            "evidence": (
                f"Plant={STEP_PLANT_ID}; step={STEP_RATIO * 100:.0f}% at "
                f"{STEP_TIME_S:.1f} s; dense sampling={DT_S * 1000:.0f} ms."
            ),
        },
        {
            "category": "scientific_trend",
            "criterion": "t90 decreases as Kp* increases",
            "passed": t90[0] > t90[1] > t90[2],
            "evidence": f"t90={dict(zip(GAINS, t90, strict=True))}.",
        },
        {
            "category": "scientific_trend",
            "criterion": "Step overshoot increases as Kp* increases",
            "passed": overshoot[0] < overshoot[1] < overshoot[2],
            "evidence": (
                f"overshoot={dict(zip(GAINS, overshoot, strict=True))}."
            ),
        },
        {
            "category": "paper_comparison",
            "criterion": (
                "v5 headline: SN error grows monotonically with Kp* in BOTH damping groups"
            ),
            "passed": bool(simpsons["dashboard_both_groups_monotone_increasing"]),
            "evidence": (
                f"O-UD union H-Osc={simpsons['dashboard_O_UD_percent_by_gain']} "
                f"(monotone={simpsons['dashboard_O_UD_monotone_increasing']}); "
                f"H-Damp={simpsons['dashboard_H_Damp_percent_by_gain']} "
                f"(monotone={simpsons['dashboard_H_Damp_monotone_increasing']}). "
                "Paper: 25.2->27.3% and 59.1->80.9%."
            ),
        },
        {
            "category": "paper_comparison",
            "criterion": (
                "The pooled SN minimum is interior (at Kp*=100) while neither group "
                "shares it - the Simpson's-paradox composition artifact"
            ),
            "passed": bool(
                simpsons["dashboard_pooled_has_interior_minimum"]
                and simpsons["dashboard_pooled_minimum_gain"]
                == PAPER_POOLED_MINIMUM_GAIN
                and simpsons["dashboard_both_groups_monotone_increasing"]
            ),
            "evidence": (
                f"Pooled={simpsons['dashboard_pooled_percent_by_gain']}; minimum at "
                f"Kp*={simpsons['dashboard_pooled_minimum_gain']} (paper "
                f"{PAPER_POOLED_MINIMUM_GAIN}); pooled inside the published "
                f"{PAPER_POOLED_FLAT_BAND_PERCENT[0]:.0f}-"
                f"{PAPER_POOLED_FLAT_BAND_PERCENT[1]:.0f}% flat band="
                f"{simpsons['dashboard_pooled_inside_flat_band']}."
            ),
        },
        {
            "category": "paper_comparison",
            "criterion": "The three Section 3.5 floors hold across the sweep",
            "passed": all(bool(row["holds"]) for row in floor_rows),
            "evidence": "; ".join(
                f"{row['floor']}: dashboard min="
                + (
                    "n/a"
                    if row["dashboard_minimum_percent"] is None
                    else f"{float(row['dashboard_minimum_percent']):.1f}%"
                )
                + f" vs floor {row['paper_value_percent']}% -> {row['holds']}"
                for row in floor_rows
            ),
        },
        {
            "category": "paper_comparison",
            "criterion": (
                "Open-loop tau_min reproduces the two published Section 3.1 endpoints "
                "(P06/P158 = 7.5 ms, P05/P139 = 67.4 ms) within 1%"
            ),
            "passed": all(
                row["tau_min_error_percent"] is not None
                and float(row["tau_min_error_percent"]) <= 1.0
                for row in tau_min_rows
                if row["paper_tau_min_ms"] is not None
            )
            and sum(1 for row in tau_min_rows if row["paper_tau_min_ms"] is not None)
            == len(PAPER_TAU_MIN_ENDPOINTS_MS),
            "evidence": "; ".join(
                f"{row['plant_id']}({row['source_pool_id']}): "
                f"{float(row['calculated_tau_min_ms']):.2f} ms vs "
                f"{row['paper_tau_min_ms']} ms"
                for row in tau_min_rows
                if row["paper_tau_min_ms"] is not None
            ),
        },
        {
            "category": "paper_comparison",
            "criterion": "At least four of six NF/SN medians are within 15% of the paper campaign",
            "passed": paper_comparison_passes >= 4,
            "evidence": (
                f"Within-tolerance rows={paper_comparison_passes}/"
                f"{len(comparison_rows)}."
            ),
        },
    ]


def closed_loop_damping_study(prefer_cache: bool = True) -> dict[str, object]:
    """Run the Closed-loop damping validation and paper comparison."""

    if prefer_cache:
        cached = _cached_closed_loop_damping_study()
        if cached is not None:
            return cached

    eigenvalue_rows, eigenvalue_mode_rows = _modal_rows()
    tau_min_rows = _tau_min_rows()

    jobs: list[dict[str, object]] = []
    for kp_star in GAINS:
        for plant in plant_registry():
            jobs.append(
                {
                    "plant": plant,
                    "kp_star": kp_star,
                    "condition": "NF",
                    "seed": 0,
                }
            )
            for seed in SN_SEEDS_BY_GAIN[kp_star]:
                jobs.append(
                    {
                        "plant": plant,
                        "kp_star": kp_star,
                        "condition": "SN",
                        "seed": seed,
                    }
                )
    # Section 3.5 footnote 3: the one ET1 cell published at the default gain.
    # The paper does not state the seed pool, so three seeds are run and both
    # the pooled and single-seed medians are reported.
    et1_jobs = [
        {
            "plant": plant,
            "kp_star": PAPER_DEFAULT_GAIN,
            "condition": "SN",
            "seed": seed,
            "excitation": ET1_EXCITATION,
        }
        for plant in plant_registry()
        for seed in (0, 1, 2)
    ]

    all_rows: list[dict[str, object]] = []
    all_jobs = jobs + et1_jobs
    with ThreadPoolExecutor(max_workers=min(10, len(all_jobs))) as executor:
        futures = [executor.submit(_run_gain_case, **job) for job in all_jobs]
        for future in as_completed(futures):
            all_rows.append(future.result())
    raw_rows = [
        row for row in all_rows if row["excitation"] == GAIN_SWEEP_EXCITATION
    ]
    et1_rows = [row for row in all_rows if row["excitation"] == ET1_EXCITATION]
    et1_rows.sort(key=lambda row: (str(row["plant_id"]), int(row["seed"])))
    raw_rows.sort(
        key=lambda row: (
            int(row["kp_star"]),
            str(row["condition"]),
            str(row["plant_id"]),
            int(row["seed"]),
        )
    )

    comparison_rows = _comparison_rows(raw_rows)
    plant_median_rows = _plant_median_rows(raw_rows, eigenvalue_rows)
    regime_rows = _regime_rows(plant_median_rows)
    ea_error_rows = _ea_error_rows(raw_rows)
    per_plant_comparison_rows = _per_plant_comparison_rows(plant_median_rows)
    et1_comparison_row = _et1_comparison_row(et1_rows)
    floor_rows = _floor_rows(comparison_rows, regime_rows)
    simpsons = _simpsons_paradox_block(comparison_rows, regime_rows)
    step_response_rows, step_metric_rows = _step_rows_and_metrics()
    gain_rows = _gain_rows(comparison_rows, step_metric_rows)
    acceptance_checks = _acceptance_checks(
        raw_rows,
        comparison_rows,
        regime_rows,
        gain_rows,
        eigenvalue_rows,
        tau_min_rows,
        simpsons,
        floor_rows,
    )
    integrity_failed = any(
        not bool(row["passed"])
        and row.get("category") == "calculation_integrity"
        for row in acceptance_checks
    )
    # A row the paper deliberately does not publish (`NO_REFERENCE`) is not a
    # failure: `data/reference_results/README.md` lists the two damping-group
    # values at K_p* = 100 and the noise-free per-gain table among the values
    # the paper omits by authorial choice. Only a real out-of-tolerance
    # comparison ("CHECK") downgrades the status.
    any_failed = (
        any(not bool(row["passed"]) for row in acceptance_checks)
        or any(row.get("pass_fail") == "CHECK" for row in comparison_rows)
    )
    validation_status = (
        "FAIL"
        if integrity_failed
        else ("CHECK" if any_failed else "PASS")
    )
    sn_comparisons = {
        int(row["kp_star"]): float(row["dashboard_MARE_theta_percent"])
        for row in comparison_rows
        if row["condition"] == "SN"
        and row.get("dashboard_MARE_theta_percent") is not None
    }
    raw_sn_minimum_gain = min(sn_comparisons, key=sn_comparisons.get)
    regime_worst_case = {
        int(row["kp_star"]): max(
            float(row["dashboard_O_UD_MARE_theta_percent"]),
            float(row["dashboard_H_Damp_MARE_theta_percent"]),
        )
        for row in regime_rows
    }
    # The paper default is NOT the argmin of the pooled curve: the pooled dip at
    # K_p* = 100 is a composition artifact, and the paper keeps 100 "for
    # identifiability and safety, not noise averaging" (Section 3.5). The
    # argmin is still reported so the artifact can be shown, but it is not a
    # recommendation.
    recommended_gain = PAPER_DEFAULT_GAIN
    baseline_eigenvalue_rows = [
        row
        for row in eigenvalue_rows
        if int(row["kp_star"]) == PAPER_DEFAULT_GAIN
    ]
    eigenvalue_match_count = sum(
        1
        for row in baseline_eigenvalue_rows
        if row.get("reference_error_percent") is not None
        and float(row["reference_error_percent"])
        <= EIGENVALUE_REFERENCE_TOLERANCE_PERCENT
        and row["calculated_regime"] == row["reference_regime_at_kp100"]
    )

    metrics: dict[str, object] = {
        "workflow": "closed-loop damping",
        "cache_version": CACHE_VERSION,
        "title": "Closed-loop damping and SysID accuracy",
        "source_protocol": "docs/closed_loop_damping_validation.md",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "validation_status": validation_status,
        "recommended_gain": recommended_gain,
        "raw_SN_minimum_gain": raw_sn_minimum_gain,
        "paper_default_gain": PAPER_DEFAULT_GAIN,
        "recommendation_basis": (
            "identifiability and safety, following Section 3.5. The pooled-SN "
            "argmin is reported separately as raw_SN_minimum_gain and is a "
            "Simpson's-paradox composition artifact, not an accuracy result; "
            "the eigenvalue regime is a descriptor and does not select the gain."
        ),
        "provenance": {
            "paper_used_for_calculation": False,
            "paper_reference_files": [
                "data/reference_results/closed_loop_damping_reference.json",
                "data/model_inputs/excitation_schedules.csv",
                "data/model_inputs/ten_plant_parameters.csv",
            ],
            "paper_sections": ["3.1 (tau_min)", "3.5", "Fig. 6", "2.2 (regime thresholds)"],
            "note": (
                "Every dashboard number here is recomputed from the model. Paper "
                "values enter only as comparison targets."
            ),
        },
        "measurement_condition": MEASUREMENT_CONDITION,
        "pct_T": SN_NOISE_LEVEL_PERCENT,
        "pct_v": SN_VELOCITY_NOISE_LEVEL_PERCENT,
        "tension_lpf_hz": SN_LPF_HZ,
        "velocity_lpf_hz": None,
        "simpsons_paradox": simpsons,
        "floor_rows": floor_rows,
        "ea_error_rows": ea_error_rows,
        "per_plant_comparison_rows": per_plant_comparison_rows,
        "et1_comparison_row": et1_comparison_row,
        "tau_min_rows": tau_min_rows,
        "tau_min_note": (
            "tau_min = 1 / max_i |Re(lambda_i)| on the OPEN-LOOP plant Jacobian "
            "(Section 3.1, p. 8). backend/validation/studies.py uses a flat "
            "DEFAULT_TMIN_MS = 50.0 for all ten plants; these per-plant values "
            "are the sourced replacement."
        ),
        "step_metric_reference_status": PAPER_STEP_METRIC_STATUS,
        "regime_worst_case_MARE_theta_percent": regime_worst_case,
        "eigenvalue_summary": {
            "method": "continuous full-cascade 9-state central-difference Jacobian",
            "calculated_row_count": len(eigenvalue_rows),
            "baseline_reference_match_count": eigenvalue_match_count,
            "baseline_reference_row_count": len(baseline_eigenvalue_rows),
            "all_modes_stable": all(bool(row["stable"]) for row in eigenvalue_rows),
            "maximum_zeta_step_sensitivity_spread": max(
                float(row["zeta_step_sensitivity_spread"])
                for row in eigenvalue_rows
            ),
            "reference_tolerance_percent": EIGENVALUE_REFERENCE_TOLERANCE_PERCENT,
        },
        "settings": {
            "kp_star_values": list(GAINS),
            "plant_count": len(plant_registry()),
            "NF_seed_count_per_gain": 1,
            "SN_seeds_by_gain": {
                str(kp): list(seeds)
                for kp, seeds in SN_SEEDS_BY_GAIN.items()
            },
            "plant_integration_step_ms": DT_S * 1000.0,
            "controller_period_ms": CONTROLLER_SAMPLE_TIME_S * 1000.0,
            "Tlog_ms": SYSID_TLOG_S * 1000.0,
            "LPF_Hz": SN_LPF_HZ,
            "sensor_noise_percent": SN_NOISE_LEVEL_PERCENT,
            "excitation": "E_Toggle tension-setpoint profile",
            "controller_integral_time": "per_plant_auto_Ti",
            "high_ea_kp_cap_enabled": False,
            "velocity_correction_limit_fraction": None,
            "noise_affects_controller": True,
            "noise_rng": "numpy.default_rng(seed)",
            "estimator": "paper_eq8_weighted_pem_trf",
            "plant_groups": ["O-UD (including H-Osc)", "H-Damp"],
            "regime_source": "calculated_kp100_full_cascade_eigenvalues",
            "eigenvalue_method": "continuous_9_state_full_cascade_jacobian",
            "eigenvalue_numerical_relative_steps": [1e-4, 1e-5, 1e-6, 1e-7],
            "paper_default_gain": PAPER_DEFAULT_GAIN,
            "recommendation_rule": "minimum_fresh_pooled_SN_MARE_only",
            "metric": "100*mean(abs((theta_hat-theta_true)/theta_true))",
            "step_plant_id": STEP_PLANT_ID,
            "step_time_s": STEP_TIME_S,
            "step_ratio": STEP_RATIO,
            "step_dense_sample_ms": DT_S * 1000.0,
        },
        "raw_rows": raw_rows,
        "eigenvalue_rows": eigenvalue_rows,
        "eigenvalue_mode_rows": eigenvalue_mode_rows,
        "plant_median_rows": plant_median_rows,
        "gain_rows": gain_rows,
        "step_metric_rows": step_metric_rows,
        "step_response_rows": step_response_rows,
        "comparison_rows": comparison_rows,
        "paper_dashboard_rows": comparison_rows,
        "sysid_error_rows": [
            {
                "kp_star": kp_star,
                "NF_MARE_theta_percent": next(
                    row["dashboard_MARE_theta_percent"]
                    for row in comparison_rows
                    if row["condition"] == "NF"
                    and int(row["kp_star"]) == kp_star
                ),
                "SN_MARE_theta_percent": next(
                    row["dashboard_MARE_theta_percent"]
                    for row in comparison_rows
                    if row["condition"] == "SN"
                    and int(row["kp_star"]) == kp_star
                ),
            }
            for kp_star in GAINS
        ],
        "regime_rows": regime_rows,
        "acceptance_checks": acceptance_checks,
        "recommendation": (
            f"Kp*={PAPER_DEFAULT_GAIN} is the default, kept for identifiability and "
            "safety rather than for noise averaging: it preserves the EA estimate "
            "that the highest gains destroy. The fresh pooled SN median is lowest "
            f"at Kp*={raw_sn_minimum_gain}, but that dip is a Simpson's-paradox "
            "composition artifact of the 8:2 group mix and must not be read as an "
            "accuracy result. The calculated damping regime is a descriptor of "
            "plant diversity, not a SysID predictor."
        ),
        "paper_conclusion": PAPER_GAIN_CONCLUSION,
    }

    csv_path = _write_rows_csv(
        "closed_loop_damping_dashboard_vs_paper_comparison.csv",
        comparison_rows,
        (
            "condition",
            "kp_star",
            "paper_MARE_theta_percent",
            "dashboard_MARE_theta_percent",
            "validation_error_percent",
            "pass_fail",
            "valid_run_count",
            "expected_run_count",
            "aggregation",
            "dashboard_source",
        ),
    )
    _write_rows_csv(
        "closed_loop_damping_live_plant_runs.csv",
        raw_rows,
        (
            "kp_star",
            "condition",
            "seed",
            "plant_id",
            "source_pool_id",
            "reference_regime_at_kp100",
            "reference_zeta_cl_min_at_kp100",
            "dashboard_MARE_theta_percent",
            "MARE_theta",
            "EA_relative_error_percent",
            "excitation",
            "campaign_group",
            "record_duration_s",
            "measurement_condition",
            "pct_T",
            "pct_v",
            "tension_lpf_hz",
            "samples",
            "sensor_noise_sigma_N",
            "controller_sample_time_s",
            "logging_sample_time_s",
            "controller_integral_time_s",
            "high_ea_kp_cap_enabled",
            "velocity_correction_limit_fraction",
            "noise_affects_controller",
            "estimator",
            "source",
            "value_status",
        ),
    )
    _write_rows_csv(
        "closed_loop_damping_step_metrics.csv",
        step_metric_rows,
        tuple(step_metric_rows[0].keys()),
    )
    _write_rows_csv(
        "closed_loop_damping_live_eigenvalues.csv",
        eigenvalue_rows,
        tuple(eigenvalue_rows[0].keys()),
    )
    _write_rows_csv(
        "closed_loop_damping_live_eigenvalue_modes.csv",
        eigenvalue_mode_rows,
        tuple(eigenvalue_mode_rows[0].keys()),
    )
    _write_rows_csv(
        "closed_loop_damping_open_loop_tau_min.csv",
        tau_min_rows,
        tuple(tau_min_rows[0].keys()),
    )
    _write_rows_csv(
        "closed_loop_damping_group_decomposition.csv",
        regime_rows,
        tuple(regime_rows[0].keys()),
    )
    _write_rows_csv(
        "closed_loop_damping_ea_error_by_gain.csv",
        ea_error_rows,
        tuple(ea_error_rows[0].keys()),
    )
    summary_path = _write_summary("closed_loop_damping_summary.json", metrics)
    return {
        "study": "closed_loop_damping",
        "metrics": metrics,
        "csv_path": csv_path,
        "summary_path": summary_path,
    }
