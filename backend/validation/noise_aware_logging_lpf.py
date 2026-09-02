"""Noise-aware logging and anti-alias LPF validation."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Any

from backend.models.controller import ControllerConfig, auto_tension_integral_time_s
from backend.models.simulation import SimulationConfig, simulate
from backend.sysid.estimator import estimate_parameters_one_step_pem, estimate_parameters_weighted_pem
from backend.validation.excitations import GROUP_A, GROUP_B, get_excitation_profile
from backend.validation.paper_inputs import COMPOSITE_SEED_V_OFFSET, excitation_schedule
from backend.validation.paper_reference import load_noise_lpf_reference
from backend.validation.plants import parameters_for_plant, plant_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "reports" / "validation_summary"
NOISE_LPF_BRIEF = PROJECT_ROOT / "docs" / "noise_aware_logging_lpf_validation.md"
LIVE_CACHE_PATH = SUMMARY_DIR / "noiseLpf_live_dashboard_values.json"
# v8: the excitation profiles moved to `excitation_schedules.csv`
# (`backend/validation/paper_inputs.py`) after the v7 cache was written, so the
# v7 E_Toggle cells were still produced by the retired hand-transcribed profile.
# v8 also adds campaign 1 in full (six cutoffs, not three), Table S7, the
# dual-channel cells and the S6 cross-channel campaigns 4-6.
LIVE_CACHE_VERSION = "noise_lpf_campaign1_tableS7_figS6_S6xchannel_lpfv_ts1_numpy_rng_v9_20260814"

DT_S = 0.001
TLOG_VALUES_MS = [1, 2, 5, 10, 20, 50, 100]
LIVE_SENSOR_NOISE_OMEGA_RAD_S = 0.0
LIVE_EXCITATION_AMPLITUDE_MULTIPLIER = 1.0
CONTROLLER_SAMPLE_TIME_S = 0.001
NOMINAL_NOISE_LEVEL_PERCENT = 0.3
FIG06_NOISE_LEVELS_PERCENT = (0.02, 0.05, 0.1, 0.3, 0.5)
TRACE_PLANT_ID = "P07"
TRACE_DURATION_S = 4.0

# ---------------------------------------------------------------------------
# Campaign wiring, read off `data/reference_results/experiment_ledger_v5.json`
# and `excitation_schedules.csv`.
#
# Campaign 1 (full factorial, group A) is the grid behind Fig. S6, Table S7,
# the feasibility gate and the main-effect spreads:
#   10 plants x 6 excitations x 7 Tlog x (NF + 5 noise levels x 6 LPF cutoffs)
#
# Campaigns 4/5/6 (cross-channel, supplement Section S6) run ET1 on the
# `B_dual_channel` schedule, whose ET1 record is 30 s rather than group A's 7 s.
# The schedule CSV flags that record as REQUIRED for reproducing the published
# R13/R15 velocity-noise cells, so every cross-channel number below is run on it.
# ---------------------------------------------------------------------------
FACTORIAL_CAMPAIGN_GROUP = GROUP_A
CROSS_CHANNEL_CAMPAIGN_GROUP = GROUP_B

# Campaign 1 cutoff axis. `None` is the unfiltered leg.
FACTORIAL_LPF_CUTOFFS: tuple[int | None, ...] = (None, 10, 20, 50, 100, 200)

# Campaigns 4 and 5: 10 plants x ET1 x 3 Tlog x (NF + 4 levels x 3 seeds).
CROSS_CHANNEL_TLOG_VALUES_MS = (2, 5, 50)
CROSS_CHANNEL_DOSES_PERCENT = (0.05, 0.1, 0.3, 0.5)
CROSS_CHANNEL_SEEDS = (0, 1, 2)
# Supplement Section S6, Fig. S5(c): the extended low-level velocity grid.
EXTENDED_LOW_VELOCITY_DOSES_PERCENT = (0.005, 0.01, 0.02)
# Campaign 6: composite noise on both channels at Tlog = 5 ms.
COMPOSITE_TLOG_MS = 5
# The reference gives no cutoff for campaigns 4-6. 50 Hz is the paper's working
# cutoff (Section 3.4) and the field-matched protocol runs "LPF 50/50 Hz"
# (Table S8), so 50 Hz on both channels is the inferred setting. Flagged as an
# inference, not a published value.
CROSS_CHANNEL_LPF_HZ = 50

# Dual-channel cells of `noise_composition_vs_cutoff`. The condition block of
# `logging_rate_v5_reference.json` fixes it: E_Toggle, pct_T = pct_v = 0.3 %,
# LPF 50 Hz, pooled over 10 plants x 3 seeds.
DUAL_CHANNEL_SEEDS = (0, 1, 2)
DUAL_CHANNEL_PCT_T = 0.3
DUAL_CHANNEL_PCT_V = 0.3
# The dual-channel cutoff pair is written "LPF 50/50 Hz" (tension/velocity) in
# Table S8, and the ledger gives the dual-channel campaigns an `LPF_v` axis
# rather than a single cutoff, so the second level of
# `noise_composition_vs_cutoff` raises the velocity cutoff and leaves the
# tension cutoff at the working 50 Hz.
DUAL_CHANNEL_TENSION_LPF_HZ = 50
DUAL_CHANNEL_VELOCITY_CUTOFFS_HZ = (50, 100)

# `sigma_v = pct_v * v_max` with `v_max = v0 / 0.30` (reference `noise_model`).
VELOCITY_FULL_SCALE_FRACTION = 0.30

# Fan-out width. Purely a scheduling choice: every job is seeded and therefore
# independent of the worker that runs it.
NOISE_LPF_MAX_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))

# ---------------------------------------------------------------------------
# v5 reference values (paper Section 3.4, supplement Section S7).
#
# All tension-only at pct_T = 0.3% of T_max, 10 plants, single seed, pooled
# median. Every number here was recomputed under the operating-point-weighted
# PEM; the v4.1 tables that used to sit in this block are retired.
#
# What v5 publishes, and what it does not:
#   - Fig. S6 gives the LPF x Tlog grid for ET1 and E_Toggle only.
#   - Table S7 gives the noise-free baseline per Tlog (ET1), with 1 ms omitted
#     because the one-step predictor has a trivial near-zero solution there.
#   - There is NO six-excitation heatmap at this condition. The four remaining
#     excitations therefore carry no paper reference and are reported as such
#     rather than being filled in with v4.1 numbers.
# ---------------------------------------------------------------------------

# Supplement Table S7, NF-baseline column (ET1). 1 ms is deliberately absent.
PAPER_NF_BASELINE_MEDIAN = {
    2: 0.68,
    5: 3.83,
    10: 8.05,
    20: 22.70,
    50: 58.24,
    100: 81.06,
}
PAPER_FIG05_NF_MEAN = {1: None, **PAPER_NF_BASELINE_MEDIAN}

# Supplement Fig. S6, 100 Hz row, E_Toggle.
PAPER_FIG05_SN_LPF100_MEAN = dict(zip(TLOG_VALUES_MS, [19, 14, 19, 27, 31, 62, 89]))

# Supplement Fig. S6, 50 Hz and 100 Hz rows, E_Toggle.
PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN = dict(zip(TLOG_VALUES_MS, [12, 13, 18, 22, 26, 65, 89]))
PAPER_FIGS10_ETOGGLE_LPF100_MEDIAN = dict(PAPER_FIG05_SN_LPF100_MEAN)

# Supplement Fig. S6 in full. The 20 Hz cutoff is not shown because most of its
# runs fail to converge, so quoting an error there would be survivorship bias.
PAPER_FIGS10_MARE = {
    "none": {
        "ET1": [291, 145, 66, 64, 64, 70, 83],
        "E_Toggle": [156, 77, 42, 55, 54, 63, 88],
    },
    "50": {
        "ET1": [13, 15, 22, 33, 37, 65, 84],
        "E_Toggle": [PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[tlog] for tlog in TLOG_VALUES_MS],
    },
    "100": {
        "ET1": [42, 37, 29, 42, 56, 67, 83],
        "E_Toggle": [PAPER_FIGS10_ETOGGLE_LPF100_MEDIAN[tlog] for tlog in TLOG_VALUES_MS],
    },
    "200": {
        "ET1": [117, 90, 46, 56, 62, 69, 83],
        "E_Toggle": [61, 44, 31, 41, 43, 61, 89],
    },
}

PAPER_NF_MARE = PAPER_FIG05_NF_MEAN
PAPER_SN_LPF_50_MARE = PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN

HEATMAP_EXCITATIONS = ["ET1", "ET3", "ET6", "EV1", "E_Toggle", "ET3M"]
HEATMAP_TLOG_VALUES_MS = [1, 2, 5, 10, 20, 50, 100]

# Only the two excitations Fig. S6 actually plots carry a v5 reference. `None`
# means "the paper publishes no value for this cell" - do not synthesize one.
PAPER_HEATMAP_REFERENCE_EXCITATIONS = ("ET1", "E_Toggle")
PAPER_HEATMAP_MARE = {
    "ET1": list(PAPER_FIGS10_MARE["50"]["ET1"]),
    "ET3": [None] * len(HEATMAP_TLOG_VALUES_MS),
    "ET6": [None] * len(HEATMAP_TLOG_VALUES_MS),
    "EV1": [None] * len(HEATMAP_TLOG_VALUES_MS),
    "E_Toggle": list(PAPER_FIGS10_MARE["50"]["E_Toggle"]),
    "ET3M": [None] * len(HEATMAP_TLOG_VALUES_MS),
}

# The anti-alias filter is a FEASIBILITY GATE, not the top influence factor.
# The v4.1 ranking (LPF > Tlog > excitation > noise level) is retired: among
# runs that converge at a usable cutoff, the median main-effect spread is
# Tlog 74 pp > noise amplitude 30 pp ~ excitation 29 pp > cutoff 9 pp, and the
# paper states the factors interact too strongly for any single additive
# ranking to hold.
#
# The cell below is E_Toggle at Tlog = 20 ms under tension-only noise at 0.3%
# of T_max. The 50 Hz and 100 Hz entries are the paper's headline pair; the
# unfiltered and 200 Hz entries are read from supplement Fig. S6.
#
# 10 Hz and 20 Hz publish NO error value on purpose. Quoting one would be
# survivorship bias: it would average only the runs that happened to survive a
# cutoff that fails on most of them.
LPF_SWEEP = [
    {
        "LPF": "No LPF",
        "cutoff_hz": "none",
        "paper_result": "54% median MARE_theta - converges, but the worst of any setting",
        "paper_MARE_theta": 54.0,
        "dashboard_MARE_theta": None,
        "convergence_failure_rate_percent": 0.0,
        "interpretation": (
            "Unfiltered logging converges every time. The gate is an over-aggressive low "
            "cutoff, not the absence of a filter - but the noise stays in the signal, so "
            "this is the highest-error setting."
        ),
    },
    {
        "LPF": "10 Hz",
        "cutoff_hz": 10,
        "paper_result": "100% convergence failure",
        "paper_MARE_theta": None,
        "dashboard_MARE_theta": None,
        "convergence_failure_rate_percent": 100.0,
        "interpretation": (
            "The filter destroys the dynamics rather than the noise: it suppresses the "
            "oscillatory modes near the natural frequencies, and the cascade inner loop "
            "adds phase lag the filter cannot track. No error value is published."
        ),
    },
    {
        "LPF": "20 Hz",
        "cutoff_hz": 20,
        "paper_result": "70% convergence failure on E_Toggle runs",
        "paper_MARE_theta": None,
        "dashboard_MARE_theta": None,
        "convergence_failure_rate_percent": 70.0,
        "interpretation": (
            "Still inside the convergence gate. No error value is published, deliberately: "
            "averaging the survivors would understate the damage."
        ),
    },
    {
        "LPF": "50 Hz",
        "cutoff_hz": 50,
        "paper_result": "25.9% median MARE_theta",
        "paper_MARE_theta": 25.9,
        "dashboard_MARE_theta": None,
        "convergence_failure_rate_percent": 0.0,
        "interpretation": (
            "The working cutoff. Not a lower bound to be exceeded - this is the best "
            "setting measured, and raising it makes things worse."
        ),
    },
    {
        "LPF": "100 Hz",
        "cutoff_hz": 100,
        "paper_result": "30.8% median MARE_theta",
        "paper_MARE_theta": 30.8,
        "dashboard_MARE_theta": None,
        "convergence_failure_rate_percent": 0.0,
        "interpretation": (
            "Converges without failure but is measurably worse than 50 Hz. v4.1 used this "
            "cutoff and called >= 50 Hz mandatory; v5 shows 50 Hz is the optimum, not a floor."
        ),
    },
    {
        "LPF": "200 Hz",
        "cutoff_hz": 200,
        "paper_result": "43% median MARE_theta",
        "paper_MARE_theta": 43.0,
        "dashboard_MARE_theta": None,
        "convergence_failure_rate_percent": 0.0,
        "interpretation": "Too permissive: more of the noise band reaches the estimator.",
    },
]

# Median main-effect spreads among runs that converge at a usable cutoff.
MAIN_EFFECT_SPREAD_PP = {
    "Tlog": 74,
    "noise_amplitude": 30,
    "excitation": 29,
    "lpf_cutoff": 9,
}


def _write_summary(name: str, payload: Mapping[str, object]) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _median(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite_values)) if finite_values else math.nan


def _mean(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.mean(finite_values)) if finite_values else math.nan


def _safe_key(value: object) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace("%", "pct")


def _downsample_rows(rows: Sequence[Mapping[str, float]], tlog_ms: int) -> list[Mapping[str, float]]:
    stride = int(round(float(tlog_ms) / (DT_S * 1000.0)))
    if stride <= 0:
        raise ValueError("Tlog must be at least the plant integration step")
    return list(rows[::stride])


def _paper_controller_config(params, line_speed_m_s: float, kp_star: float = 100.0) -> ControllerConfig:
    return ControllerConfig(
        target_tension_N=params.tension_ref_N,
        line_speed_m_s=line_speed_m_s,
        Kp_star_m_s_per_N=kp_star,
        TI_s=auto_tension_integral_time_s(params, line_speed_m_s),
        high_ea_kp_cap_enabled=False,
        feedforward_uses_measured_omega=True,
        paper_velocity_gain_enabled=True,
        velocity_correction_limit_fraction=None,
        steady_velocity_uses_dynamic_target=False,
    )


def _run_live_sysid_series(
    *,
    plant: Mapping[str, Any],
    excitation_name: str,
    lpf_hz: int | None,
    noise_level_percent: float,
    seed: int,
    tlog_values_ms: Sequence[int],
    campaign_group: str = FACTORIAL_CAMPAIGN_GROUP,
    velocity_noise_percent: float = 0.0,
    velocity_lpf_hz: int | None | str = "follow_tension",
    velocity_seed_offset: int | None = None,
) -> list[dict[str, object]]:
    plant_id = str(plant["plant_id"])
    params, meta = parameters_for_plant(plant_id)
    amplitude = float(meta.get("recommended_excitation_amplitude_V", 0.2 * float(meta.get("T_ref_N", 1.0))))
    amplitude *= LIVE_EXCITATION_AMPLITUDE_MULTIPLIER
    tension_noisy = float(noise_level_percent) > 0.0
    velocity_noisy = float(velocity_noise_percent) > 0.0
    noisy = tension_noisy or velocity_noisy
    noise_sigma = float(noise_level_percent) / 100.0 * float(meta["T_max_N"]) if tension_noisy else 0.0
    # sigma_v = pct_v * v_max with v_max = v0 / 0.30 (reference `noise_model`).
    # The simulator converts this surface-speed error to a per-roller angular
    # sigma of sigma_v / R, which is the paper's injection rule.
    nominal_line_speed = float(meta.get("v_ref_m_s", params.feeder_velocity_m_s))
    velocity_sigma = (
        float(velocity_noise_percent) / 100.0 * (nominal_line_speed / VELOCITY_FULL_SCALE_FRACTION)
        if velocity_noisy
        else 0.0
    )
    speed_multipliers = (0.5, 1.0, 2.0) if excitation_name == "ET3M" else (1.0,)
    per_tlog_mare: dict[int, list[float]] = {int(value): [] for value in tlog_values_ms}
    per_tlog_samples: dict[int, int] = {int(value): 0 for value in tlog_values_ms}
    per_tlog_failures: dict[int, list[str]] = {int(value): [] for value in tlog_values_ms}
    record_duration_s = float(
        excitation_schedule(
            "ET3" if excitation_name == "ET3M" else excitation_name, campaign_group
        ).duration_s
    )

    for speed_multiplier in speed_multipliers:
        line_speed = nominal_line_speed * float(speed_multiplier)
        profile = get_excitation_profile(
            "ET3" if excitation_name == "ET3M" else excitation_name,
            amplitude,
            campaign_group=campaign_group,
        )
        try:
            sim = simulate(
                params,
                controller_config=_paper_controller_config(params, line_speed),
                config=SimulationConfig(
                    duration_s=float(getattr(profile, "duration_s", 12.0)),
                    dt_s=DT_S,
                    controller_sample_time_s=CONTROLLER_SAMPLE_TIME_S,
                    log_sample_time_s=DT_S,
                    line_speed_m_s=line_speed,
                    sensor_noise_tension_N=noise_sigma,
                    sensor_noise_omega_rad_s=LIVE_SENSOR_NOISE_OMEGA_RAD_S if noisy else 0.0,
                    sensor_noise_velocity_m_s=velocity_sigma,
                    sensor_lpf_hz=lpf_hz if noisy else None,
                    # `velocity_lpf_hz=None` falls back to `sensor_lpf_hz` in the
                    # simulator, which is what "follow_tension" means here. An
                    # explicit number decouples the two cutoffs, which is the
                    # LPF_v axis the dual-channel campaigns sweep.
                    velocity_lpf_hz=None if velocity_lpf_hz == "follow_tension" else velocity_lpf_hz,
                    velocity_seed_offset=velocity_seed_offset,
                    noise_affects_controller=True,
                    noise_rng="numpy_default_rng",
                    seed=seed,
                    output_name=(
                        f"noise_lpf_{plant_id}_{_safe_key(excitation_name)}_"
                        f"{noise_level_percent:g}pct_{_safe_key(lpf_hz if lpf_hz is not None else 'none')}_"
                        f"v{speed_multiplier:g}.csv"
                    ),
                ),
                excitation=profile,
                write_output=False,
            )
        except Exception as exc:
            for tlog_ms in tlog_values_ms:
                per_tlog_failures[int(tlog_ms)].append(f"simulation:{type(exc).__name__}:{exc}")
            continue

        for tlog_ms in tlog_values_ms:
            try:
                sysid_rows = _downsample_rows(sim.rows, int(tlog_ms))
                sysid = estimate_parameters_weighted_pem(
                    sysid_rows,
                    nominal_params=params,
                    true_params=params,
                    max_nfev=150,
                    break_on_line_speed_change=excitation_name == "EV1",
                )
                per_tlog_mare[int(tlog_ms)].append(100.0 * float(sysid.mare_theta))
                per_tlog_samples[int(tlog_ms)] += len(sysid_rows)
            except Exception as exc:
                per_tlog_failures[int(tlog_ms)].append(f"sysid:{type(exc).__name__}:{exc}")

    if velocity_noisy and tension_noisy:
        measurement_condition = "dual_channel"
    elif velocity_noisy:
        measurement_condition = "velocity_only"
    elif tension_noisy:
        measurement_condition = "tension_only"
    else:
        measurement_condition = "noise_free"

    rows: list[dict[str, object]] = []
    for tlog_ms in tlog_values_ms:
        values = per_tlog_mare[int(tlog_ms)]
        complete = len(values) == len(speed_multipliers)
        rows.append(
            {
                "plant_id": plant_id,
                "excitation": excitation_name,
                "campaign_group": campaign_group,
                "record_duration_s": record_duration_s,
                "Tlog_ms": int(tlog_ms),
                "lpf_hz": lpf_hz if lpf_hz is not None else "none",
                "condition": "SN" if noisy else "NF",
                "measurement_condition": measurement_condition,
                "noise_level_percent": float(noise_level_percent),
                "velocity_noise_percent": float(velocity_noise_percent),
                "velocity_sigma_m_s": velocity_sigma,
                "velocity_lpf_hz": lpf_hz if velocity_lpf_hz == "follow_tension" else velocity_lpf_hz,
                "velocity_seed_offset": velocity_seed_offset,
                "noise_sigma_N": noise_sigma,
                "seed": int(seed),
                "dashboard_MARE_theta_percent": _mean(values) if complete else None,
                "samples": per_tlog_samples[int(tlog_ms)],
                "operating_point_count": len(speed_multipliers),
                "operating_point_valid_count": len(values),
                "controller_sample_time_s": CONTROLLER_SAMPLE_TIME_S,
                "controller_integral_time": "per_plant_auto_Ti",
                "high_ea_kp_cap_enabled": False,
                "velocity_correction_limit_fraction": None,
                "noise_affects_controller": True,
                "noise_rng": "numpy.default_rng(seed)",
                "estimator": "paper_eq8_weighted_pem_trf",
                "status": "ok" if complete else "failed:" + "|".join(per_tlog_failures[int(tlog_ms)]),
            }
        )
    return rows


def _simulate_trace_rows() -> list[dict[str, object]]:
    params, meta = parameters_for_plant(TRACE_PLANT_ID)
    amplitude = 0.2 * float(meta["T_ref_N"])
    profile = get_excitation_profile("ET1", amplitude)
    line_speed = float(meta["v_ref_m_s"])
    modes = (
        ("NF", 0.0, None),
        ("SN_no_LPF", 1.0, None),
        ("SN_LPF50", 1.0, 50),
    )
    rows: list[dict[str, object]] = []
    for mode, noise_percent, cutoff in modes:
        sim = simulate(
            params,
            controller_config=_paper_controller_config(params, line_speed),
            config=SimulationConfig(
                duration_s=TRACE_DURATION_S,
                dt_s=DT_S,
                controller_sample_time_s=CONTROLLER_SAMPLE_TIME_S,
                log_sample_time_s=DT_S,
                line_speed_m_s=line_speed,
                sensor_noise_tension_N=noise_percent / 100.0 * float(meta["T_max_N"]),
                sensor_noise_omega_rad_s=0.0,
                sensor_lpf_hz=cutoff,
                noise_affects_controller=True,
                noise_rng="numpy_default_rng",
                seed=0,
            ),
            excitation=profile,
            write_output=False,
        )
        for row in sim.rows[::5]:
            rows.append(
                {
                    "mode": mode,
                    "plant_id": TRACE_PLANT_ID,
                    "time_s": float(row["time_s"]),
                    "normalized_tension": float(row["T1"]) / float(meta["T_ref_N"]),
                    "normalized_reference": float(row["T1_ref_N"]) / float(meta["T_ref_N"]),
                    "noise_level_percent": noise_percent,
                    "lpf_hz": cutoff if cutoff is not None else "none",
                }
            )
    return rows


def _run_executor(job_count: int):
    """Return the executor used to fan the campaign out.

    Every job is fully determined by its arguments and its own seeded RNG, so
    the results do not depend on how the work is distributed. The simulator's
    RK4 loop is pure Python and therefore GIL-bound, so processes are used when
    they are available and threads are the fallback.
    """

    workers = max(1, min(NOISE_LPF_MAX_WORKERS, job_count))
    if workers > 1:
        try:
            return ProcessPoolExecutor(max_workers=workers)
        except (OSError, ValueError, ImportError):  # pragma: no cover - platform guard
            pass
    return ThreadPoolExecutor(max_workers=workers)


def _compute_live_dashboard_values(
    prefer_cache: bool = True,
) -> dict[str, object]:
    if prefer_cache and LIVE_CACHE_PATH.exists():
        try:
            cached = json.loads(LIVE_CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("cache_version") == LIVE_CACHE_VERSION:
                return cached
        except (OSError, json.JSONDecodeError):
            pass

    plants = plant_registry()
    jobs_by_key: dict[tuple[object, ...], dict[str, object]] = {}

    def add_job(
        plant: Mapping[str, Any],
        excitation_name: str,
        noise_level_percent: float,
        lpf_hz: int | None,
        tlog_values_ms: Sequence[int],
        seed: int = 0,
        *,
        campaign_group: str = FACTORIAL_CAMPAIGN_GROUP,
        velocity_noise_percent: float = 0.0,
        velocity_lpf_hz: int | None | str = "follow_tension",
        velocity_seed_offset: int | None = None,
    ) -> None:
        key = (
            str(plant["plant_id"]),
            excitation_name,
            campaign_group,
            float(noise_level_percent),
            float(velocity_noise_percent),
            str(lpf_hz if lpf_hz is not None else "none"),
            str(velocity_lpf_hz),
            velocity_seed_offset,
            int(seed),
        )
        if key not in jobs_by_key:
            jobs_by_key[key] = {
                "plant": plant,
                "excitation_name": excitation_name,
                "campaign_group": campaign_group,
                "noise_level_percent": float(noise_level_percent),
                "velocity_noise_percent": float(velocity_noise_percent),
                "velocity_lpf_hz": velocity_lpf_hz,
                "velocity_seed_offset": velocity_seed_offset,
                "lpf_hz": lpf_hz,
                "seed": int(seed),
                "tlog_values_ms": set(),
            }
        jobs_by_key[key]["tlog_values_ms"].update(int(value) for value in tlog_values_ms)

    for plant in plants:
        # ------------------------------------------------------------------
        # Campaign 1, in full (ledger: 10 plants x 6 excitations x 7 Tlog x
        # (NF + 5 noise levels x 6 LPF cutoffs)). Fig. S6, Table S7, the
        # feasibility gate and the main-effect spreads all read off this grid,
        # so it is run as one factorial rather than as four ad-hoc slices.
        # ------------------------------------------------------------------
        for excitation in HEATMAP_EXCITATIONS:
            add_job(plant, excitation, 0.0, None, TLOG_VALUES_MS)
            for noise_level in FIG06_NOISE_LEVELS_PERCENT:
                for cutoff in FACTORIAL_LPF_CUTOFFS:
                    add_job(plant, excitation, noise_level, cutoff, TLOG_VALUES_MS)

        # ------------------------------------------------------------------
        # Dual-channel cells of `noise_composition_vs_cutoff`. Condition from
        # `logging_rate_v5_reference.json`: E_Toggle, pct_T = pct_v = 0.3 %,
        # 10 plants x 3 seeds, LPF 50 Hz.
        #
        # The cutoff the 100 Hz cell raises is the VELOCITY cutoff. The ledger
        # gives the dual-channel campaigns (7-9, 12-15) an `LPF_v` axis of their
        # own and Table S8 writes the dual-channel setting as a pair, "LPF
        # 50/50 Hz", so 50/100 is the paper's second level. Measured: 50/50
        # 42.47 %, 50/100 47.15 %, 100/50 54.23 %, 100/100 59.84 % against the
        # published 42.6 % and 47.0 %.
        # ------------------------------------------------------------------
        for velocity_cutoff in DUAL_CHANNEL_VELOCITY_CUTOFFS_HZ:
            for seed in DUAL_CHANNEL_SEEDS:
                add_job(
                    plant,
                    "E_Toggle",
                    DUAL_CHANNEL_PCT_T,
                    DUAL_CHANNEL_TENSION_LPF_HZ,
                    TLOG_VALUES_MS,
                    seed,
                    velocity_noise_percent=DUAL_CHANNEL_PCT_V,
                    velocity_lpf_hz=velocity_cutoff,
                    velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
                )

        # ------------------------------------------------------------------
        # Campaigns 4/5/6 - supplement Section S6 cross-channel comparison.
        # ET1 on the `B_dual_channel` schedule (30 s record), 3 Tlog values,
        # four matched doses x three seeds per channel.
        # ------------------------------------------------------------------
        add_job(
            plant, "ET1", 0.0, None, CROSS_CHANNEL_TLOG_VALUES_MS,
            campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
        )
        for seed in CROSS_CHANNEL_SEEDS:
            for dose in CROSS_CHANNEL_DOSES_PERCENT:
                # campaign 4: tension-only
                add_job(
                    plant, "ET1", dose, CROSS_CHANNEL_LPF_HZ, CROSS_CHANNEL_TLOG_VALUES_MS, seed,
                    campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
                )
                # campaign 5: velocity-only, single stream (seed_T is NaN on
                # those rows, so the velocity channel is the run's own stream)
                add_job(
                    plant, "ET1", 0.0, CROSS_CHANNEL_LPF_HZ, CROSS_CHANNEL_TLOG_VALUES_MS, seed,
                    campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
                    velocity_noise_percent=dose,
                )
                # campaign 6: composite, independent streams (seed_v = seed_T + 100)
                add_job(
                    plant, "ET1", dose, CROSS_CHANNEL_LPF_HZ, (COMPOSITE_TLOG_MS,), seed,
                    campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
                    velocity_noise_percent=dose,
                    velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
                )
                # campaign 6 control: matched-seed velocity-only, i.e. the same
                # velocity realisation the composite run saw
                add_job(
                    plant, "ET1", 0.0, CROSS_CHANNEL_LPF_HZ, (COMPOSITE_TLOG_MS,), seed,
                    campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
                    velocity_noise_percent=dose,
                    velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
                )
            for dose in EXTENDED_LOW_VELOCITY_DOSES_PERCENT:
                add_job(
                    plant, "ET1", 0.0, CROSS_CHANNEL_LPF_HZ, CROSS_CHANNEL_TLOG_VALUES_MS, seed,
                    campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
                    velocity_noise_percent=dose,
                )

    raw_rows: list[dict[str, object]] = []
    jobs = []
    for job in jobs_by_key.values():
        jobs.append({**job, "tlog_values_ms": sorted(job["tlog_values_ms"])})
    with _run_executor(len(jobs)) as executor:
        futures = [executor.submit(_run_live_sysid_series, **job) for job in jobs]
        for future in as_completed(futures):
            raw_rows.extend(future.result())

    def matching_rows(
        *,
        excitation: str,
        tlog: int,
        condition: str | None = None,
        lpf_hz: int | str,
        noise_level_percent: float | None = None,
        velocity_noise_percent: float | None = 0.0,
        campaign_group: str = FACTORIAL_CAMPAIGN_GROUP,
        seed: int | None = 0,
        velocity_seed_offset: int | None | str = None,
        velocity_lpf_hz: object = "any",
    ) -> list[Mapping[str, object]]:
        def close(row_key: str, wanted: float) -> bool:
            return math.isclose(float(row[row_key]), float(wanted), rel_tol=0.0, abs_tol=1e-12)

        selected = []
        for row in raw_rows:
            if row["excitation"] != excitation or int(row["Tlog_ms"]) != int(tlog):
                continue
            if str(row.get("campaign_group")) != campaign_group:
                continue
            if condition is not None and row["condition"] != condition:
                continue
            if str(row["lpf_hz"]) != str(lpf_hz):
                continue
            if noise_level_percent is not None and not close("noise_level_percent", noise_level_percent):
                continue
            if velocity_noise_percent is not None and not close(
                "velocity_noise_percent", velocity_noise_percent
            ):
                continue
            if seed is not None and int(row["seed"]) != int(seed):
                continue
            if velocity_seed_offset != "any" and row.get("velocity_seed_offset") != velocity_seed_offset:
                continue
            if velocity_lpf_hz != "any" and row.get("velocity_lpf_hz") != velocity_lpf_hz:
                continue
            selected.append(row)
        return selected

    def values_for(**kwargs: object) -> list[float]:
        return [
            float(row["dashboard_MARE_theta_percent"])
            for row in matching_rows(**kwargs)  # type: ignore[arg-type]
            if row["status"] == "ok"
        ]

    # Table S7's NF-baseline column is the ET1 noise-free MEDIAN over the ten
    # plants - not an E_Toggle mean. The v7 cache compared an E_Toggle mean
    # against it.
    dashboard_nf = {
        str(tlog): _median(
            values_for(excitation="ET1", tlog=tlog, condition="NF", lpf_hz="none", noise_level_percent=0.0)
        )
        for tlog in TLOG_VALUES_MS
    }
    dashboard_nf_etoggle = {
        str(tlog): _median(
            values_for(excitation="E_Toggle", tlog=tlog, condition="NF", lpf_hz="none", noise_level_percent=0.0)
        )
        for tlog in TLOG_VALUES_MS
    }
    # Fig. S6 cells are pooled medians, so the 100 Hz E_Toggle row is a median.
    dashboard_fig05_sn_lpf_100 = {
        str(tlog): _median(
            values_for(
                excitation="E_Toggle",
                tlog=tlog,
                condition="SN",
                lpf_hz=100,
                noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
            )
        )
        for tlog in TLOG_VALUES_MS
    }
    dashboard_sn_lpf_50 = {
        str(tlog): _median(
            values_for(
                excitation="E_Toggle",
                tlog=tlog,
                condition="SN",
                lpf_hz=50,
                noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
            )
        )
        for tlog in TLOG_VALUES_MS
    }
    dashboard_sn_lpf_100 = {
        str(tlog): _median(
            values_for(
                excitation="E_Toggle",
                tlog=tlog,
                condition="SN",
                lpf_hz=100,
                noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
            )
        )
        for tlog in TLOG_VALUES_MS
    }
    lpf_sweep_dashboard = {
        str(cutoff if cutoff is not None else "none"): _median(
            values_for(
                excitation="E_Toggle",
                tlog=20,
                condition="SN",
                lpf_hz=cutoff if cutoff is not None else "none",
                noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
            )
        )
        for cutoff in (None, 10, 20, 50, 100, 200)
    }
    def failure_rate(rows: Sequence[Mapping[str, object]]) -> float:
        if not rows:
            return math.nan
        failed = [
            row
            for row in rows
            if row["status"] != "ok"
            or row["dashboard_MARE_theta_percent"] is None
            or not math.isfinite(float(row["dashboard_MARE_theta_percent"]))
        ]
        return 100.0 * len(failed) / len(rows)

    lpf_failure_rates = {}
    for cutoff in (None, 10, 20, 50, 100, 200):
        key = str(cutoff if cutoff is not None else "none")
        lpf_failure_rates[key] = failure_rate(
            matching_rows(
                excitation="E_Toggle",
                tlog=20,
                condition="SN",
                lpf_hz=key,
                noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
            )
        )

    # Feasibility gate pooled over the six excitations and the whole campaign-1
    # grid, which is the scope the paper quotes its 67-100 % band on.
    six_excitation_failure_rates = {}
    for cutoff in FACTORIAL_LPF_CUTOFFS:
        key = str(cutoff if cutoff is not None else "none")
        pooled = [
            row
            for row in raw_rows
            if str(row.get("campaign_group")) == FACTORIAL_CAMPAIGN_GROUP
            and str(row["lpf_hz"]) == key
            and row["condition"] == "SN"
            and float(row["velocity_noise_percent"]) == 0.0
            and int(row["seed"]) == 0
        ]
        six_excitation_failure_rates[key] = failure_rate(pooled)

    # Fig. S6 condition: pooled median over the ten plants at pct_T = 0.3 %.
    # The five-level mean the v7 cache used is a different quantity and is kept
    # beside it rather than compared against Fig. S6.
    heatmap_dashboard: dict[str, list[float]] = {}
    heatmap_five_level_mean: dict[str, list[float]] = {}
    for excitation in HEATMAP_EXCITATIONS:
        heatmap_dashboard[excitation] = [
            _median(
                values_for(
                    excitation=excitation,
                    tlog=tlog,
                    condition="SN",
                    lpf_hz=50,
                    noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
                )
            )
            for tlog in HEATMAP_TLOG_VALUES_MS
        ]
        heatmap_five_level_mean[excitation] = [
            _mean(
                values_for(
                    excitation=excitation,
                    tlog=tlog,
                    condition="SN",
                    lpf_hz=50,
                    noise_level_percent=None,
                )
            )
            for tlog in HEATMAP_TLOG_VALUES_MS
        ]

    figs10_dashboard: dict[str, dict[str, list[float]]] = {}
    for cutoff in (None, 50, 100, 200):
        key = str(cutoff if cutoff is not None else "none")
        figs10_dashboard[key] = {}
        for excitation in ("ET1", "E_Toggle"):
            figs10_dashboard[key][excitation] = [
                _median(
                    values_for(
                        excitation=excitation,
                        tlog=tlog,
                        condition="SN",
                        lpf_hz=cutoff if cutoff is not None else "none",
                        noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
                    )
                )
                for tlog in TLOG_VALUES_MS
            ]

    # ----------------------------------------------------------------- #
    # Supplement Table S7: NF -> SN transition, ET1, tension-only.
    # Each cell is the ratio of the SN pooled median to the NF pooled
    # median at the same Tlog. 1 ms is omitted by the paper because the NF
    # baseline there is a trivial near-zero fit.
    # ----------------------------------------------------------------- #
    transition_dashboard: dict[str, dict[str, object]] = {}
    for cutoff, key in ((None, "unfiltered"), (50, "lpf_50hz")):
        block: dict[str, object] = {}
        for tlog in TLOG_VALUES_MS:
            nf_median = float(dashboard_nf[str(tlog)])
            ratios = {}
            for dose in FIG06_NOISE_LEVELS_PERCENT:
                sn_median = _median(
                    values_for(
                        excitation="ET1",
                        tlog=tlog,
                        condition="SN",
                        lpf_hz=cutoff if cutoff is not None else "none",
                        noise_level_percent=dose,
                    )
                )
                ratios[f"{dose:g}"] = (
                    sn_median / nf_median
                    if math.isfinite(sn_median) and math.isfinite(nf_median) and nf_median > 0.0
                    else math.nan
                )
            block[str(tlog)] = {
                "NF_baseline_percent": nf_median,
                "SN_median_percent": {
                    f"{dose:g}": _median(
                        values_for(
                            excitation="ET1",
                            tlog=tlog,
                            condition="SN",
                            lpf_hz=cutoff if cutoff is not None else "none",
                            noise_level_percent=dose,
                        )
                    )
                    for dose in FIG06_NOISE_LEVELS_PERCENT
                },
                "ratios": ratios,
            }
        transition_dashboard[key] = block

    # ----------------------------------------------------------------- #
    # Dual-channel cells of `noise_composition_vs_cutoff`: E_Toggle,
    # pct_T = pct_v = 0.3 %, pooled over 10 plants x 3 seeds.
    # ----------------------------------------------------------------- #
    # Keyed by the velocity cutoff; the tension cutoff stays at the working
    # 50 Hz in both legs.
    dual_channel_dashboard: dict[str, dict[str, float]] = {}
    for velocity_cutoff in DUAL_CHANNEL_VELOCITY_CUTOFFS_HZ:
        dual_channel_dashboard[str(velocity_cutoff)] = {
            str(tlog): _median(
                [
                    value
                    for seed in DUAL_CHANNEL_SEEDS
                    for value in values_for(
                        excitation="E_Toggle",
                        tlog=tlog,
                        condition="SN",
                        lpf_hz=DUAL_CHANNEL_TENSION_LPF_HZ,
                        noise_level_percent=DUAL_CHANNEL_PCT_T,
                        velocity_noise_percent=DUAL_CHANNEL_PCT_V,
                        velocity_lpf_hz=velocity_cutoff,
                        velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
                        seed=seed,
                    )
                ]
            )
            for tlog in TLOG_VALUES_MS
        }
    dual_channel_dashboard["tension_lpf_hz"] = {  # type: ignore[assignment]
        str(cutoff): float(DUAL_CHANNEL_TENSION_LPF_HZ)
        for cutoff in DUAL_CHANNEL_VELOCITY_CUTOFFS_HZ
    }

    # ----------------------------------------------------------------- #
    # Main-effect spreads among the runs that converge at a usable cutoff.
    # Each cell is the pooled median over the ten plants; the spread of one
    # factor is max - min across its levels holding the other three fixed,
    # and the reported figure is the median of those spreads.
    # ----------------------------------------------------------------- #
    usable_cutoffs = [cutoff for cutoff in FACTORIAL_LPF_CUTOFFS if cutoff is not None and cutoff >= 50]
    factor_levels = {
        "excitation": list(HEATMAP_EXCITATIONS),
        "Tlog": list(TLOG_VALUES_MS),
        "noise_amplitude": list(FIG06_NOISE_LEVELS_PERCENT),
        "lpf_cutoff": list(usable_cutoffs),
    }
    cube: dict[tuple[str, int, float, int], float] = {}
    for excitation in factor_levels["excitation"]:
        for tlog in factor_levels["Tlog"]:
            for dose in factor_levels["noise_amplitude"]:
                for cutoff in factor_levels["lpf_cutoff"]:
                    cube[(excitation, int(tlog), float(dose), int(cutoff))] = _median(
                        values_for(
                            excitation=excitation,
                            tlog=tlog,
                            condition="SN",
                            lpf_hz=cutoff,
                            noise_level_percent=dose,
                        )
                    )

    def _main_effect_spread(factor: str) -> dict[str, object]:
        others = [name for name in factor_levels if name != factor]
        spreads: list[float] = []
        def combos(index: int, fixed: dict[str, object]):
            if index == len(others):
                cell_values = []
                for level in factor_levels[factor]:
                    key_map = {**fixed, factor: level}
                    value = cube[(
                        str(key_map["excitation"]),
                        int(key_map["Tlog"]),
                        float(key_map["noise_amplitude"]),
                        int(key_map["lpf_cutoff"]),
                    )]
                    if math.isfinite(value):
                        cell_values.append(value)
                if len(cell_values) == len(factor_levels[factor]):
                    spreads.append(max(cell_values) - min(cell_values))
                return
            for level in factor_levels[others[index]]:
                combos(index + 1, {**fixed, others[index]: level})

        combos(0, {})
        return {
            "median_spread_pp": _median(spreads),
            "cell_count": len(spreads),
            "level_count": len(factor_levels[factor]),
        }

    main_effect_dashboard = {factor: _main_effect_spread(factor) for factor in factor_levels}

    # ----------------------------------------------------------------- #
    # Supplement Section S6, campaigns 4/5/6 - the cross-channel block.
    # ET1 on the 30 s `B_dual_channel` record.
    # ----------------------------------------------------------------- #
    def cross_values(
        tlog: int,
        *,
        pct_T: float,
        pct_v: float,
        seeds: Sequence[int],
        velocity_seed_offset: int | None,
    ) -> list[float]:
        return [
            value
            for seed in seeds
            for value in values_for(
                excitation="ET1",
                tlog=tlog,
                lpf_hz=CROSS_CHANNEL_LPF_HZ if (pct_T or pct_v) else "none",
                noise_level_percent=pct_T,
                velocity_noise_percent=pct_v,
                campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
                velocity_seed_offset=velocity_seed_offset,
                seed=seed,
            )
        ]

    cross_channel_dashboard: dict[str, object] = {
        "campaign_group": CROSS_CHANNEL_CAMPAIGN_GROUP,
        "record_duration_s": float(
            excitation_schedule("ET1", CROSS_CHANNEL_CAMPAIGN_GROUP).duration_s
        ),
        "lpf_hz": CROSS_CHANNEL_LPF_HZ,
        "doses_percent": list(CROSS_CHANNEL_DOSES_PERCENT),
        "seeds": list(CROSS_CHANNEL_SEEDS),
        "ratios": {},
        "pooled_medians": {},
        "per_dose": {},
        "noise_free": {
            str(tlog): _median(
                values_for(
                    excitation="ET1",
                    tlog=tlog,
                    condition="NF",
                    lpf_hz="none",
                    noise_level_percent=0.0,
                    campaign_group=CROSS_CHANNEL_CAMPAIGN_GROUP,
                )
            )
            for tlog in CROSS_CHANNEL_TLOG_VALUES_MS
        },
    }
    for tlog in CROSS_CHANNEL_TLOG_VALUES_MS:
        tension_pool = [
            value
            for dose in CROSS_CHANNEL_DOSES_PERCENT
            for value in cross_values(
                tlog, pct_T=dose, pct_v=0.0, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None
            )
        ]
        velocity_pool = [
            value
            for dose in CROSS_CHANNEL_DOSES_PERCENT
            for value in cross_values(
                tlog, pct_T=0.0, pct_v=dose, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None
            )
        ]
        tension_median = _median(tension_pool)
        velocity_median = _median(velocity_pool)
        cross_channel_dashboard["pooled_medians"][str(tlog)] = {
            "tension_only": tension_median,
            "velocity_only": velocity_median,
            "tension_n": len(tension_pool),
            "velocity_n": len(velocity_pool),
        }
        cross_channel_dashboard["ratios"][str(tlog)] = (
            velocity_median / tension_median if tension_median > 0.0 else math.nan
        )
        cross_channel_dashboard["per_dose"][str(tlog)] = {
            f"{dose:g}": {
                "tension_only": _median(
                    cross_values(tlog, pct_T=dose, pct_v=0.0, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None)
                ),
                "velocity_only": _median(
                    cross_values(tlog, pct_T=0.0, pct_v=dose, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None)
                ),
            }
            for dose in CROSS_CHANNEL_DOSES_PERCENT
        }

    # Fig. S5(c): the extended low-level velocity grid.
    cross_channel_dashboard["extended_low_velocity"] = {
        f"{dose:g}": {
            str(tlog): _median(
                cross_values(
                    tlog, pct_T=0.0, pct_v=dose, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None
                )
            )
            for tlog in CROSS_CHANNEL_TLOG_VALUES_MS
        }
        for dose in EXTENDED_LOW_VELOCITY_DOSES_PERCENT
    }

    # Campaign 6: composite noise and its matched-seed velocity-only control.
    # The ledger says the composite grid is two levels; the paper does not say
    # which two, so every dose is reported and the ratio is given per dose and
    # pooled over all four.
    composite_per_dose = {}
    for dose in CROSS_CHANNEL_DOSES_PERCENT:
        tension = _median(
            cross_values(
                COMPOSITE_TLOG_MS, pct_T=dose, pct_v=0.0, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None
            )
        )
        velocity_control = _median(
            cross_values(
                COMPOSITE_TLOG_MS,
                pct_T=0.0,
                pct_v=dose,
                seeds=CROSS_CHANNEL_SEEDS,
                velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
            )
        )
        velocity_campaign5 = _median(
            cross_values(
                COMPOSITE_TLOG_MS, pct_T=0.0, pct_v=dose, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None
            )
        )
        composite = _median(
            cross_values(
                COMPOSITE_TLOG_MS,
                pct_T=dose,
                pct_v=dose,
                seeds=CROSS_CHANNEL_SEEDS,
                velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
            )
        )
        composite_per_dose[f"{dose:g}"] = {
            "tension_only": tension,
            "velocity_only_matched_seed_control": velocity_control,
            "velocity_only_campaign5": velocity_campaign5,
            "composite": composite,
            "composite_over_sum": composite / (tension + velocity_control)
            if (tension + velocity_control) > 0.0
            else math.nan,
            "composite_over_velocity": composite / velocity_control if velocity_control > 0.0 else math.nan,
        }
    pooled_tension = _median(
        [
            value
            for dose in CROSS_CHANNEL_DOSES_PERCENT
            for value in cross_values(
                COMPOSITE_TLOG_MS, pct_T=dose, pct_v=0.0, seeds=CROSS_CHANNEL_SEEDS, velocity_seed_offset=None
            )
        ]
    )
    pooled_velocity_control = _median(
        [
            value
            for dose in CROSS_CHANNEL_DOSES_PERCENT
            for value in cross_values(
                COMPOSITE_TLOG_MS,
                pct_T=0.0,
                pct_v=dose,
                seeds=CROSS_CHANNEL_SEEDS,
                velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
            )
        ]
    )
    pooled_composite = _median(
        [
            value
            for dose in CROSS_CHANNEL_DOSES_PERCENT
            for value in cross_values(
                COMPOSITE_TLOG_MS,
                pct_T=dose,
                pct_v=dose,
                seeds=CROSS_CHANNEL_SEEDS,
                velocity_seed_offset=COMPOSITE_SEED_V_OFFSET,
            )
        ]
    )
    cross_channel_dashboard["composite"] = {
        "Tlog_ms": COMPOSITE_TLOG_MS,
        "per_dose": composite_per_dose,
        "pooled_over_four_doses": {
            "tension_only": pooled_tension,
            "velocity_only_matched_seed_control": pooled_velocity_control,
            "composite": pooled_composite,
            "composite_over_sum": pooled_composite / (pooled_tension + pooled_velocity_control)
            if (pooled_tension + pooled_velocity_control) > 0.0
            else math.nan,
            "composite_over_velocity": pooled_composite / pooled_velocity_control
            if pooled_velocity_control > 0.0
            else math.nan,
        },
        "level_pair_note": (
            "experiment_ledger_v5.json campaign 6 runs two levels; the paper does "
            "not publish which two, so every dose is reported"
        ),
    }

    payload = {
        "cache_version": LIVE_CACHE_VERSION,
        "value_source": "live_simulation_sysid_cache",
        "settings": {
            "plant_count": len(plants),
            "plants": [str(plant["plant_id"]) for plant in plants],
            "dt_s": DT_S,
            "duration_s": "profile_specific",
            "controller_sample_time_s": CONTROLLER_SAMPLE_TIME_S,
            "controller_integral_time": "per_plant_auto_Ti",
            "high_ea_kp_cap_enabled": False,
            "velocity_correction_limit_fraction": None,
            "noise_affects_controller": True,
            "noise_rng": "numpy.default_rng(seed)",
            "noise_seeds": [0],
            "dual_channel_seeds": list(DUAL_CHANNEL_SEEDS),
            "cross_channel_seeds": list(CROSS_CHANNEL_SEEDS),
            "figure6_noise_levels_percent": list(FIG06_NOISE_LEVELS_PERCENT),
            "factorial_lpf_cutoffs_hz": ["none" if value is None else value for value in FACTORIAL_LPF_CUTOFFS],
            "factorial_campaign_group": FACTORIAL_CAMPAIGN_GROUP,
            "cross_channel_campaign_group": CROSS_CHANNEL_CAMPAIGN_GROUP,
            "cross_channel_record_duration_s": float(
                excitation_schedule("ET1", CROSS_CHANNEL_CAMPAIGN_GROUP).duration_s
            ),
            "cross_channel_lpf_hz": CROSS_CHANNEL_LPF_HZ,
            "cross_channel_lpf_source": "inferred_not_published",
            "velocity_seed_offset_dual": COMPOSITE_SEED_V_OFFSET,
            "sensor_noise_omega_rad_s": LIVE_SENSOR_NOISE_OMEGA_RAD_S,
            "velocity_noise_rule": "sigma_v = pct_v * v0 / 0.30, per-roller sigma_v / R",
            "excitation_amplitude_multiplier": LIVE_EXCITATION_AMPLITUDE_MULTIPLIER,
            "metric": "100*mean(abs((theta_hat-theta_true)/theta_true))",
            "estimator": "paper_eq8_weighted_pem_trf",
        },
        "dashboard_nf_mare": dashboard_nf,
        "dashboard_nf_mare_etoggle": dashboard_nf_etoggle,
        "dashboard_fig05_sn_lpf_100_mean": dashboard_fig05_sn_lpf_100,
        "dashboard_sn_lpf_50_mare": dashboard_sn_lpf_50,
        "dashboard_sn_lpf_100_mare": dashboard_sn_lpf_100,
        "lpf_sweep_dashboard": lpf_sweep_dashboard,
        "lpf_failure_rates": lpf_failure_rates,
        "six_excitation_failure_rates": six_excitation_failure_rates,
        "heatmap_dashboard": heatmap_dashboard,
        "heatmap_five_level_mean": heatmap_five_level_mean,
        "figs10_dashboard": figs10_dashboard,
        "transition_dashboard": transition_dashboard,
        "dual_channel_dashboard": dual_channel_dashboard,
        "main_effect_dashboard": main_effect_dashboard,
        "cross_channel_dashboard": cross_channel_dashboard,
        "trace_rows": _simulate_trace_rows(),
        "raw_rows": raw_rows,
    }
    LIVE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _live_curve(live: Mapping[str, object], key: str) -> dict[int, float]:
    values = live.get(key, {})
    if not isinstance(values, Mapping):
        return {}
    return {int(float(tlog)): float(value) for tlog, value in values.items()}


def _build_lpf_rows(
    live: Mapping[str, object],
) -> list[dict[str, float | str | None]]:
    dashboard = live.get("lpf_sweep_dashboard", {})
    failures = live.get("lpf_failure_rates", {})
    if not isinstance(dashboard, Mapping):
        dashboard = {}
    if not isinstance(failures, Mapping):
        failures = {}
    key_by_label = {
        "No LPF": "none",
        "10 Hz": "10",
        "20 Hz": "20",
        "50 Hz": "50",
        "100 Hz": "100",
        "200 Hz": "200",
    }
    def finite_or_none(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    rows: list[dict[str, float | str | None]] = []
    for row in LPF_SWEEP:
        live_key = key_by_label.get(str(row["LPF"]))
        if live_key is None:
            continue
        rows.append(
            {
                **row,
                "dashboard_MARE_theta": finite_or_none(dashboard.get(live_key)),
                "convergence_failure_rate_percent": float(failures.get(live_key, row["convergence_failure_rate_percent"])),
                "dashboard_source": "live_simulation_sysid_cache",
            }
        )
    # 200 Hz is part of LPF_SWEEP itself now, so there is no separate row to
    # append; the v4.1 duplicate that used to live here is gone.
    return rows


def _lpf_alpha(cutoff_hz: int | None, dt_s: float = DT_S) -> float | str:
    if cutoff_hz is None:
        return "not applied"
    omega_dt = 2.0 * math.pi * cutoff_hz * dt_s
    return 1.0 - math.exp(-omega_dt)


def _percent_error(paper: float, dashboard: float) -> float:
    return abs(paper - dashboard) / abs(paper) * 100.0


def _pass_fail(error_percent: float, *, tolerance_percent: float = 15.0) -> str:
    return "PASS" if error_percent <= tolerance_percent else "FAIL"


def _monotonic_increasing(values: Sequence[float]) -> bool:
    return all(next_value >= value for value, next_value in zip(values, values[1:]))


def _is_u_shaped(values: Sequence[float]) -> bool:
    best_index = min(range(len(values)), key=lambda idx: values[idx])
    if best_index == 0 or best_index == len(values) - 1:
        return False
    left = values[: best_index + 1]
    right = values[best_index:]
    return all(a >= b for a, b in zip(left, left[1:])) and all(a <= b for a, b in zip(right, right[1:]))


def _log_scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    log_value = math.log10(max(value, 1e-9))
    return out_lo + (log_value - math.log10(lo)) * (out_hi - out_lo) / (math.log10(hi) - math.log10(lo))


def _linear_scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if abs(hi - lo) < 1e-12:
        return 0.5 * (out_lo + out_hi)
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def _log_x_crossover(
    first: Mapping[int, float | None],
    second: Mapping[int, float | None],
) -> tuple[float, float] | None:
    """Return the first first=second crossover using paper-style interpolation.

    A logging period where either series has no published value is skipped: the
    paper omits the noise-free 1 ms cell, and a crossover cannot be interpolated
    across a gap.
    """

    for left_tlog, right_tlog in zip(TLOG_VALUES_MS, TLOG_VALUES_MS[1:]):
        if any(
            series.get(tlog) is None
            for series in (first, second)
            for tlog in (left_tlog, right_tlog)
        ):
            continue
        left_diff = float(first[left_tlog]) - float(second[left_tlog])
        right_diff = float(first[right_tlog]) - float(second[right_tlog])
        if left_diff == 0.0:
            return float(left_tlog), float(first[left_tlog])
        if left_diff * right_diff < 0.0:
            fraction = -left_diff / (right_diff - left_diff)
            log_tlog = math.log10(float(left_tlog)) + fraction * (
                math.log10(float(right_tlog)) - math.log10(float(left_tlog))
            )
            value = float(first[left_tlog]) + fraction * (
                float(first[right_tlog]) - float(first[left_tlog])
            )
            return 10.0**log_tlog, value
    return None


def _comparison_rows(
    dashboard_nf: Mapping[int, float],
    dashboard_sn_lpf_50: Mapping[int, float],
    dashboard_sn_lpf_100: Mapping[int, float],
) -> list[dict[str, float | str]]:
    cases: list[dict[str, float | str]] = [
        {
            "case": 1,
            "case_label": "SN 1 ms",
            "condition": "SN",
            "Tlog_ms": 1,
            "LPF": "50 Hz",
            "paper_MARE_theta": PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[1],
            "dashboard_MARE_theta": dashboard_sn_lpf_50[1],
            "paper_note": "Figure S10 nominal-noise median",
        },
        {
            "case": 2,
            "case_label": "SN 10 ms",
            "condition": "SN",
            "Tlog_ms": 10,
            "LPF": "50 Hz",
            "paper_MARE_theta": PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[10],
            "dashboard_MARE_theta": dashboard_sn_lpf_50[10],
            "paper_note": "Figure S10 nominal-noise median",
        },
        {
            "case": 3,
            "case_label": "SN 20 ms / 50 Hz",
            "condition": "SN",
            "Tlog_ms": 20,
            "LPF": "50 Hz",
            "paper_MARE_theta": PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20],
            "dashboard_MARE_theta": dashboard_sn_lpf_50[20],
            "paper_note": "Figure S10 nominal-noise median",
        },
        {
            "case": 4,
            "case_label": "SN 20 ms / 100 Hz",
            "condition": "SN",
            "Tlog_ms": 20,
            "LPF": "100 Hz",
            "paper_MARE_theta": PAPER_FIGS10_ETOGGLE_LPF100_MEDIAN[20],
            "dashboard_MARE_theta": dashboard_sn_lpf_100[20],
            "paper_note": "Figure S10 nominal-noise median",
        },
        {
            "case": 5,
            "case_label": "SN 100 ms",
            "condition": "SN",
            "Tlog_ms": 100,
            "LPF": "50 Hz",
            "paper_MARE_theta": PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[100],
            "dashboard_MARE_theta": dashboard_sn_lpf_50[100],
            "paper_note": "Figure S10 nominal-noise median",
        },
        {
            "case": 6,
            "case_label": "NF 5 ms",
            "condition": "NF",
            "Tlog_ms": 5,
            "LPF": "No noise",
            "paper_MARE_theta": PAPER_FIG05_NF_MEAN[5],
            "dashboard_MARE_theta": dashboard_nf[5],
            "paper_note": "Table S7 NF baseline, ET1, median over 10 plants",
        },
        {
            "case": 7,
            "case_label": "NF 50 ms",
            "condition": "NF",
            "Tlog_ms": 50,
            "LPF": "No noise",
            "paper_MARE_theta": PAPER_FIG05_NF_MEAN[50],
            "dashboard_MARE_theta": dashboard_nf[50],
            "paper_note": "Table S7 NF baseline, ET1, median over 10 plants",
        },
    ]
    for row in cases:
        # A key case can reference a cell the paper does not publish - the
        # noise-free 1 ms fit is trivially near zero and is omitted. Report the
        # dashboard value on its own rather than comparing against nothing.
        raw_paper = row["paper_MARE_theta"]
        if raw_paper is None:
            row["error_percent"] = None
            row["pass_fail"] = "NO_REFERENCE"
            continue
        paper = float(raw_paper)
        dashboard = float(row["dashboard_MARE_theta"])
        error = _percent_error(paper, dashboard)
        row["error_percent"] = round(error, 2)
        row["pass_fail"] = _pass_fail(error)
    return cases


def _filter_config_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for label, cutoff in (("No LPF", None), ("LPF 50 Hz", 50), ("LPF 100 Hz", 100), ("LPF 200 Hz", 200)):
        rows.append({"setting": label, "cutoff_hz": "none" if cutoff is None else cutoff, "dt_ms": DT_S * 1000.0, "alpha": _lpf_alpha(cutoff)})
    return rows


def _dashboard_heatmap_values(live: Mapping[str, object]) -> dict[str, list[float]]:
    values = live.get("heatmap_dashboard", {})
    if not isinstance(values, Mapping):
        return {excitation: [math.nan for _ in HEATMAP_TLOG_VALUES_MS] for excitation in HEATMAP_EXCITATIONS}
    return {
        excitation: [float(value) for value in values.get(excitation, [])]
        for excitation in HEATMAP_EXCITATIONS
    }


def _heatmap_rows(dashboard_values: Mapping[str, Sequence[float]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for excitation in HEATMAP_EXCITATIONS:
        for idx, tlog in enumerate(HEATMAP_TLOG_VALUES_MS):
            paper = PAPER_HEATMAP_MARE[excitation][idx]
            dashboard = float(dashboard_values[excitation][idx])
            # v5 publishes Fig. S6 for ET1 and E_Toggle only. The other four
            # excitations have no reference at this condition, so the row
            # carries the dashboard value with no comparison.
            error = None if paper is None else _percent_error(paper, dashboard)
            rows.append(
                {
                    "excitation": excitation,
                    "Tlog_ms": tlog,
                    "aggregation": "median_over_10_plants_at_0.3pct_noise_LPF50",
                    "paper_MARE_theta": paper,
                    "dashboard_MARE_theta": dashboard,
                    "error_percent": None if error is None else round(error, 2),
                    "pass_fail": (
                        "NO_REFERENCE"
                        if error is None
                        else _pass_fail(error, tolerance_percent=8.0)
                    ),
                    "paper_reference_status": (
                        "not_published_for_this_excitation" if paper is None else "paper_reference"
                    ),
                }
            )
    return rows


def _figs10_rows(live: Mapping[str, object]) -> list[dict[str, float | str]]:
    dashboard = live.get("figs10_dashboard", {})
    if not isinstance(dashboard, Mapping):
        return []
    rows: list[dict[str, float | str]] = []
    for cutoff_key in ("none", "50", "100", "200"):
        cutoff_values = dashboard.get(cutoff_key, {})
        if not isinstance(cutoff_values, Mapping):
            continue
        for excitation in ("ET1", "E_Toggle"):
            values = cutoff_values.get(excitation, [])
            if not isinstance(values, Sequence):
                continue
            for index, tlog in enumerate(TLOG_VALUES_MS):
                paper = float(PAPER_FIGS10_MARE[cutoff_key][excitation][index])
                calculated = float(values[index])
                error = _percent_error(paper, calculated)
                rows.append(
                    {
                        "LPF_Hz": cutoff_key,
                        "excitation": excitation,
                        "Tlog_ms": tlog,
                        "aggregation": "median_over_10_plants_at_0.3pct_noise",
                        "paper_MARE_theta": paper,
                        "dashboard_MARE_theta": calculated,
                        "error_percent": round(error, 2),
                        "pass_fail": _pass_fail(error),
                    }
                )
    return rows


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _scored(
    rows: list[dict[str, object]],
    *,
    block: str,
    quantity: str,
    paper: object,
    dashboard: object,
    unit: str,
    tolerance_percent: float,
    condition: str,
    source: str,
    note: str | None = None,
) -> None:
    """Append one scoreable reference quantity to the scorecard."""

    paper_value = _finite_or_none(paper)
    dashboard_value = _finite_or_none(dashboard)
    if paper_value is None:
        status = "NOT_PUBLISHED"
        error = None
    elif dashboard_value is None:
        status = "NOT_COMPUTED"
        error = None
    elif paper_value == 0.0:
        error = abs(dashboard_value)
        status = "PASS" if error <= 1e-9 else "FAIL"
    else:
        error = _percent_error(paper_value, dashboard_value)
        status = _pass_fail(error, tolerance_percent=tolerance_percent)
    rows.append(
        {
            "block": block,
            "quantity": quantity,
            "condition": condition,
            "unit": unit,
            "paper_value": paper_value,
            "dashboard_value": dashboard_value,
            "delta": None if (paper_value is None or dashboard_value is None) else dashboard_value - paper_value,
            "error_percent": None if error is None else round(error, 2),
            "tolerance_percent": tolerance_percent,
            "pass_fail": status,
            "paper_source": source,
            "note": note,
        }
    )


def _transition_rows(live: Mapping[str, object]) -> list[dict[str, object]]:
    """Supplement Table S7: NF -> SN transition ratios under ET1, tension-only."""

    reference = load_noise_lpf_reference().get("transition_table", {})
    dashboard = live.get("transition_dashboard", {})
    if not isinstance(dashboard, Mapping):
        return []
    rows: list[dict[str, object]] = []
    for block_key in ("unfiltered", "lpf_50hz"):
        live_block = dashboard.get(block_key, {})
        for entry in reference[block_key]:
            tlog = int(entry["Tlog_ms"])
            cell = live_block.get(str(tlog), {}) if isinstance(live_block, Mapping) else {}
            rows.append(
                {
                    "filter": block_key,
                    "Tlog_ms": tlog,
                    "excitation": reference["excitation"],
                    "measurement_condition": reference["measurement_condition"],
                    "paper_NF_baseline_percent": entry["NF_baseline_percent"],
                    "dashboard_NF_baseline_percent": _finite_or_none(cell.get("NF_baseline_percent")),
                    "paper_ratios": entry["ratios"],
                    "dashboard_ratios": {
                        key: _finite_or_none(value)
                        for key, value in (cell.get("ratios", {}) or {}).items()
                    },
                }
            )
    return rows


def _cross_channel_rows(live: Mapping[str, object]) -> list[dict[str, object]]:
    """Supplement Section S6: velocity-over-tension ratios and the composite."""

    reference = load_noise_lpf_reference().get("cross_channel", {})
    dashboard = live.get("cross_channel_dashboard", {})
    if not isinstance(dashboard, Mapping):
        return []
    ratios = dashboard.get("ratios", {})
    medians = dashboard.get("pooled_medians", {})
    rows: list[dict[str, object]] = []
    for entry in reference["ratios"]:
        tlog = str(int(entry["Tlog_ms"]))
        cell = medians.get(tlog, {}) if isinstance(medians, Mapping) else {}
        dashboard_ratio = _finite_or_none(ratios.get(tlog)) if isinstance(ratios, Mapping) else None
        paper_ratio = float(entry["velocity_over_tension"])
        rows.append(
            {
                "Tlog_ms": int(tlog),
                "excitation": reference["excitation"],
                "campaign_group": dashboard.get("campaign_group"),
                "record_duration_s": dashboard.get("record_duration_s"),
                "paper_velocity_over_tension": paper_ratio,
                "dashboard_velocity_over_tension": dashboard_ratio,
                "dashboard_tension_only_median_percent": _finite_or_none(cell.get("tension_only")),
                "dashboard_velocity_only_median_percent": _finite_or_none(cell.get("velocity_only")),
                "pooled_n_per_channel": cell.get("tension_n"),
                "error_percent": None
                if dashboard_ratio is None
                else round(_percent_error(paper_ratio, dashboard_ratio), 2),
                "pass_fail": "NOT_COMPUTED"
                if dashboard_ratio is None
                else _pass_fail(_percent_error(paper_ratio, dashboard_ratio), tolerance_percent=20.0),
            }
        )
    return rows


def _main_effect_rows(live: Mapping[str, object]) -> list[dict[str, object]]:
    """Median main-effect spreads among runs that converge at a usable cutoff."""

    reference = load_noise_lpf_reference().get("main_effect_spreads_pp", {})
    dashboard = live.get("main_effect_dashboard", {})
    if not isinstance(dashboard, Mapping):
        return []
    rows: list[dict[str, object]] = []
    for factor in ("Tlog", "noise_amplitude", "excitation", "lpf_cutoff"):
        cell = dashboard.get(factor, {})
        value = _finite_or_none(cell.get("median_spread_pp")) if isinstance(cell, Mapping) else None
        paper = float(reference[factor])
        rows.append(
            {
                "factor": factor,
                "paper_median_spread_pp": paper,
                "dashboard_median_spread_pp": value,
                "combination_count": cell.get("cell_count") if isinstance(cell, Mapping) else None,
                "level_count": cell.get("level_count") if isinstance(cell, Mapping) else None,
                "delta_pp": None if value is None else round(value - paper, 2),
            }
        )
    dashboard_order = [
        row["factor"]
        for row in sorted(
            (row for row in rows if row["dashboard_median_spread_pp"] is not None),
            key=lambda row: -float(row["dashboard_median_spread_pp"]),
        )
    ]
    for row in rows:
        row["paper_rank"] = ["Tlog", "noise_amplitude", "excitation", "lpf_cutoff"].index(row["factor"]) + 1
        row["dashboard_rank"] = (
            dashboard_order.index(row["factor"]) + 1 if row["factor"] in dashboard_order else None
        )
    return rows


def _reference_scorecard(live: Mapping[str, object]) -> list[dict[str, object]]:
    """Score every quantity `noise_lpf_reference.json` publishes for this section."""

    reference = load_noise_lpf_reference()
    rows: list[dict[str, object]] = []

    # --- supplement Fig. S6, the LPF x Tlog grid -------------------------- #
    figs10 = live.get("figs10_dashboard", {})
    for series in reference["lpf_tlog_heatmap"]["series"]:
        key = "none" if series["lpf_hz"] is None else str(series["lpf_hz"])
        excitation = str(series["excitation"])
        values = figs10.get(key, {}).get(excitation, []) if isinstance(figs10, Mapping) else []
        for index, tlog in enumerate(reference["lpf_tlog_heatmap"]["Tlog_grid_ms"]):
            _scored(
                rows,
                block="lpf_tlog_heatmap",
                quantity=f"MARE_theta {excitation} LPF {series['label']} Tlog {tlog} ms",
                paper=series["values"][index],
                dashboard=values[index] if index < len(values) else None,
                unit="percent",
                tolerance_percent=15.0,
                condition="tension_only pct_T=0.3, median over 10 plants, seed 0",
                source="supplement Fig. S6",
            )

    own_best = reference["lpf_tlog_heatmap"]["own_best_comparison"]
    for label, cutoff_key in (("ET1_at_50Hz", "50"), ("ET1_at_100Hz", "100")):
        series = figs10.get(cutoff_key, {}).get("ET1", []) if isinstance(figs10, Mapping) else []
        finite = [value for value in series if _finite_or_none(value) is not None]
        best_value = min(finite) if finite else None
        best_tlog = TLOG_VALUES_MS[series.index(best_value)] if best_value is not None else None
        _scored(
            rows,
            block="own_best_comparison",
            quantity=f"{label} best MARE_theta",
            paper=own_best[label]["MARE_percent"],
            dashboard=best_value,
            unit="percent",
            tolerance_percent=15.0,
            condition="tension_only pct_T=0.3, ET1",
            source="supplement Fig. S6 caption",
        )
        _scored(
            rows,
            block="own_best_comparison",
            quantity=f"{label} best Tlog",
            paper=own_best[label]["Tlog_ms"],
            dashboard=best_tlog,
            unit="ms",
            tolerance_percent=0.0,
            condition="tension_only pct_T=0.3, ET1",
            source="supplement Fig. S6 caption",
        )

    # --- working cutoff and noise composition ----------------------------- #
    sweep = live.get("lpf_sweep_dashboard", {})
    dual = live.get("dual_channel_dashboard", {})
    composition = reference["noise_composition_vs_cutoff"]
    _scored(
        rows,
        block="working_cutoff",
        quantity="E_Toggle Tlog 20 ms, tension-only, 50 Hz",
        paper=reference["working_cutoff"]["MARE_at_50Hz_percent"],
        dashboard=sweep.get("50") if isinstance(sweep, Mapping) else None,
        unit="percent",
        tolerance_percent=15.0,
        condition="tension_only pct_T=0.3, median over 10 plants",
        source="paper Section 3.4",
    )
    _scored(
        rows,
        block="working_cutoff",
        quantity="E_Toggle Tlog 20 ms, tension-only, 100 Hz",
        paper=reference["working_cutoff"]["MARE_at_100Hz_percent"],
        dashboard=sweep.get("100") if isinstance(sweep, Mapping) else None,
        unit="percent",
        tolerance_percent=15.0,
        condition="tension_only pct_T=0.3, median over 10 plants",
        source="paper Section 3.4",
    )
    dual_50 = dual.get("50", {}).get("20") if isinstance(dual, Mapping) else None
    dual_100 = dual.get("100", {}).get("20") if isinstance(dual, Mapping) else None
    _scored(
        rows,
        block="noise_composition_vs_cutoff",
        quantity="E_Toggle Tlog 20 ms, dual-channel, LPF 50/50",
        paper=composition["dual_channel_50Hz_percent"],
        dashboard=dual_50,
        unit="percent",
        tolerance_percent=15.0,
        condition="dual_channel pct_T=pct_v=0.3, LPF_T/LPF_v = 50/50 Hz, median over 10 plants x 3 seeds",
        source="paper Section 3.4 / Fig. 2(a)",
    )
    _scored(
        rows,
        block="noise_composition_vs_cutoff",
        quantity="E_Toggle Tlog 20 ms, dual-channel, LPF 50/100",
        paper=composition["dual_channel_100Hz_percent"],
        dashboard=dual_100,
        unit="percent",
        tolerance_percent=15.0,
        condition="dual_channel pct_T=pct_v=0.3, LPF_T/LPF_v = 50/100 Hz, median over 10 plants x 3 seeds",
        source="paper Section 3.4",
    )
    tension_50 = _finite_or_none(sweep.get("50") if isinstance(sweep, Mapping) else None)
    tension_100 = _finite_or_none(sweep.get("100") if isinstance(sweep, Mapping) else None)
    dual_50_value = _finite_or_none(dual_50)
    dual_100_value = _finite_or_none(dual_100)
    _scored(
        rows,
        block="noise_composition_vs_cutoff",
        quantity="noise-composition effect",
        paper=composition["noise_composition_effect_pp"],
        dashboard=None
        if dual_50_value is None or tension_50 is None
        else dual_50_value - tension_50,
        unit="pp",
        tolerance_percent=25.0,
        condition="dual-channel minus tension-only at 50 Hz",
        source="paper Section 3.4",
    )
    _scored(
        rows,
        block="noise_composition_vs_cutoff",
        quantity="cutoff effect",
        paper=composition["cutoff_effect_pp"],
        dashboard=None
        if dual_100_value is None or dual_50_value is None
        else dual_100_value - dual_50_value,
        unit="pp",
        tolerance_percent=25.0,
        condition="LPF_v 100 Hz minus LPF_v 50 Hz, dual-channel",
        source="paper Section 3.4",
    )

    # --- feasibility gate -------------------------------------------------- #
    cell_failures = live.get("lpf_failure_rates", {})
    pooled_failures = live.get("six_excitation_failure_rates", {})
    for entry in reference["feasibility_gate"]["per_cutoff"]:
        key = "none" if entry["lpf_hz"] is None else str(entry["lpf_hz"])
        _scored(
            rows,
            block="feasibility_gate",
            quantity=f"convergence failure at {key} Hz",
            paper=entry["convergence_failure_percent"],
            dashboard=cell_failures.get(key) if isinstance(cell_failures, Mapping) else None,
            unit="percent",
            tolerance_percent=15.0,
            condition=str(entry.get("scope", "E_Toggle Tlog 20 ms, pct_T=0.3, 10 plants")),
            source="paper Section 3.4 / supplement Section S7",
        )
    band = reference["feasibility_gate"]["divergence_below_floor_percent"]
    pooled_10 = pooled_failures.get("10") if isinstance(pooled_failures, Mapping) else None
    pooled_20 = pooled_failures.get("20") if isinstance(pooled_failures, Mapping) else None
    _scored(
        rows,
        block="feasibility_gate",
        quantity="divergence below the floor, lower end (20 Hz pooled)",
        paper=band[0],
        dashboard=pooled_20,
        unit="percent",
        tolerance_percent=15.0,
        condition="pooled over the six excitations, whole campaign-1 grid",
        source="paper Section 3.4",
    )
    _scored(
        rows,
        block="feasibility_gate",
        quantity="divergence below the floor, upper end (10 Hz pooled)",
        paper=band[1],
        dashboard=pooled_10,
        unit="percent",
        tolerance_percent=15.0,
        condition="pooled over the six excitations, whole campaign-1 grid",
        source="paper Section 3.4",
    )

    # --- Table S7 transition ---------------------------------------------- #
    transition = live.get("transition_dashboard", {})
    for block_key, label in (("unfiltered", "no filter"), ("lpf_50hz", "50 Hz")):
        live_block = transition.get(block_key, {}) if isinstance(transition, Mapping) else {}
        for entry in reference["transition_table"][block_key]:
            tlog = int(entry["Tlog_ms"])
            cell = live_block.get(str(tlog), {}) if isinstance(live_block, Mapping) else {}
            if block_key == "unfiltered":
                _scored(
                    rows,
                    block="transition_table_nf_baseline",
                    quantity=f"NF baseline Tlog {tlog} ms",
                    paper=entry["NF_baseline_percent"],
                    dashboard=cell.get("NF_baseline_percent"),
                    unit="percent",
                    tolerance_percent=15.0,
                    condition="ET1 noise-free, median over 10 plants",
                    source="supplement Table S7",
                )
            for dose, paper_ratio in entry["ratios"].items():
                _scored(
                    rows,
                    block="transition_table",
                    quantity=f"{label} SN/NF ratio, Tlog {tlog} ms, pct_T {dose}",
                    paper=paper_ratio,
                    dashboard=(cell.get("ratios", {}) or {}).get(dose),
                    unit="x",
                    tolerance_percent=25.0,
                    condition="ET1 tension-only, ratio of pooled medians",
                    source="supplement Table S7",
                )

    # --- no-filter boundary ------------------------------------------------ #
    boundary = reference["no_filter_boundary"]
    unfiltered_20 = ((transition.get("unfiltered", {}) or {}).get("20", {}) or {}).get("ratios", {})
    unfiltered_50 = ((transition.get("unfiltered", {}) or {}).get("50", {}) or {}).get("ratios", {})
    filtered_20 = ((transition.get("lpf_50hz", {}) or {}).get("20", {}) or {}).get("ratios", {})
    _scored(
        rows,
        block="no_filter_boundary",
        quantity="unfiltered ratio at 20 ms (band low)",
        paper=boundary["unfiltered_ratio_at_20ms"][0],
        dashboard=min((value for value in unfiltered_20.values() if _finite_or_none(value) is not None), default=None),
        unit="x",
        tolerance_percent=25.0,
        condition="ET1 tension-only, across the five doses",
        source="paper Section 3.4 / supplement Section S7",
    )
    _scored(
        rows,
        block="no_filter_boundary",
        quantity="unfiltered ratio at 20 ms (band high)",
        paper=boundary["unfiltered_ratio_at_20ms"][1],
        dashboard=max((value for value in unfiltered_20.values() if _finite_or_none(value) is not None), default=None),
        unit="x",
        tolerance_percent=25.0,
        condition="ET1 tension-only, across the five doses",
        source="paper Section 3.4 / supplement Section S7",
    )
    _scored(
        rows,
        block="no_filter_boundary",
        quantity="unfiltered ratio at 50 ms",
        paper=boundary["unfiltered_ratio_at_50ms"],
        dashboard=_median([value for value in unfiltered_50.values() if _finite_or_none(value) is not None]),
        unit="x",
        tolerance_percent=25.0,
        condition="ET1 tension-only, median across the five doses",
        source="supplement Table S7",
    )
    _scored(
        rows,
        block="no_filter_boundary",
        quantity="50 Hz ratio at 20 ms (band low)",
        paper=boundary["filtered_ratio_at_20ms"][0],
        dashboard=min((value for value in filtered_20.values() if _finite_or_none(value) is not None), default=None),
        unit="x",
        tolerance_percent=25.0,
        condition="ET1 tension-only, across the five doses",
        source="supplement Table S7",
    )
    _scored(
        rows,
        block="no_filter_boundary",
        quantity="50 Hz ratio at 20 ms (band high)",
        paper=boundary["filtered_ratio_at_20ms"][1],
        dashboard=max((value for value in filtered_20.values() if _finite_or_none(value) is not None), default=None),
        unit="x",
        tolerance_percent=25.0,
        condition="ET1 tension-only, across the five doses",
        source="supplement Table S7",
    )

    # --- main-effect spreads ----------------------------------------------- #
    main_effects = live.get("main_effect_dashboard", {})
    for factor, paper_value in reference["main_effect_spreads_pp"].items():
        if not isinstance(paper_value, (int, float)):
            continue
        cell = main_effects.get(factor, {}) if isinstance(main_effects, Mapping) else {}
        _scored(
            rows,
            block="main_effect_spreads",
            quantity=f"median main-effect spread, {factor}",
            paper=paper_value,
            dashboard=cell.get("median_spread_pp") if isinstance(cell, Mapping) else None,
            unit="pp",
            tolerance_percent=25.0,
            condition="runs that converge at a usable cutoff (>= 50 Hz)",
            source="supplement Section S7",
        )

    # --- cross-channel ------------------------------------------------------ #
    cross = live.get("cross_channel_dashboard", {})
    ratios = cross.get("ratios", {}) if isinstance(cross, Mapping) else {}
    for entry in reference["cross_channel"]["ratios"]:
        tlog = str(int(entry["Tlog_ms"]))
        _scored(
            rows,
            block="cross_channel",
            quantity=f"velocity/tension error ratio at Tlog {tlog} ms",
            paper=entry["velocity_over_tension"],
            dashboard=ratios.get(tlog) if isinstance(ratios, Mapping) else None,
            unit="x",
            tolerance_percent=20.0,
            condition="ET1 group B (30 s), 4 doses x 3 seeds x 10 plants per channel",
            source="paper Section 3.4 / supplement Section S6",
        )
    composite = cross.get("composite", {}) if isinstance(cross, Mapping) else {}
    pooled = composite.get("pooled_over_four_doses", {}) if isinstance(composite, Mapping) else {}
    _scored(
        rows,
        block="cross_channel",
        quantity="composite / (tension + velocity) at Tlog 5 ms",
        paper=reference["cross_channel"]["composite"]["composite_over_sum_of_channels"],
        dashboard=pooled.get("composite_over_sum") if isinstance(pooled, Mapping) else None,
        unit="ratio",
        tolerance_percent=25.0,
        condition="ET1 group B, matched doses, matched-seed velocity control",
        source="supplement Section S6",
        note=(
            "campaign 6 runs two dose levels and the paper does not say which two; "
            "this row pools all four"
        ),
    )
    extended = cross.get("extended_low_velocity", {}) if isinstance(cross, Mapping) else {}
    low = reference["cross_channel"]["extreme_low_dose"]
    _scored(
        rows,
        block="cross_channel",
        quantity=f"velocity-only MARE at pct_v {low['pct_v']} %, Tlog {low['Tlog_ms']} ms",
        paper=low["median_MARE_percent"],
        dashboard=(extended.get(f"{low['pct_v']:g}", {}) or {}).get(str(low["Tlog_ms"]))
        if isinstance(extended, Mapping)
        else None,
        unit="percent",
        tolerance_percent=25.0,
        condition="ET1 group B, velocity-only, 3 seeds x 10 plants",
        source="supplement Section S6 / Fig. S5(c)",
    )
    return rows


def _write_tlog_comparison_chart(
    path: Path,
    dashboard_nf: Mapping[int, float],
    dashboard_sn_lpf_100: Mapping[int, float],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 680
    left, right, top, bottom = 90, 42, 184, 86
    x_lo, x_hi = 1.0, 100.0
    # Match the author's Figure 5 visible range. The two near-zero NF/1 ms
    # values are deliberately shown as below-axis observations, not clamped.
    y_lo, y_hi = 0.5, 1500.0
    series = [
        ("Paper Table S7 NF baseline (ET1)", PAPER_FIG05_NF_MEAN, "#1f77b4", "circle", "5 6"),
        ("Dashboard NF median (ET1)", dashboard_nf, "#1f77b4", "circle", ""),
        ("Paper Fig. S6 100 Hz (E_Toggle)", PAPER_FIG05_SN_LPF100_MEAN, "#d62728", "square", "5 6"),
        ("Dashboard 100 Hz median (E_Toggle)", dashboard_sn_lpf_100, "#d62728", "square", ""),
    ]
    grid: list[str] = []
    for tick in TLOG_VALUES_MS:
        x = _log_scale(float(tick), x_lo, x_hi, left, width - right)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#d8dcd6" stroke-width="1"/>')
        grid.append(f'<text x="{x:.1f}" y="{height-bottom+26}" font-size="13" font-family="Arial" text-anchor="middle" fill="#243033">{tick}</text>')
    for tick in (1, 10, 100, 1000):
        y = _log_scale(float(tick), y_lo, y_hi, height - bottom, top)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e4e5dd" stroke-width="1"/>')
        grid.append(f'<text x="{left-12}" y="{y+4:.1f}" font-size="13" font-family="Arial" text-anchor="end" fill="#243033">{int(tick)}</text>')

    marks: list[str] = []
    for idx, (label, values, color, marker, dash) in enumerate(series):
        # A series may have no published value at some Tlog - the paper omits
        # the noise-free 1 ms cell because its fit is trivially near zero. Skip
        # those points rather than plotting a fabricated one.
        coords = [
            (
                _log_scale(float(tlog), x_lo, x_hi, left, width - right),
                _log_scale(float(values[tlog]), y_lo, y_hi, height - bottom, top),
            )
            for tlog in TLOG_VALUES_MS
            if values.get(tlog) is not None
        ]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marks.append(f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash_attr} points="{points}" />')
        for x, y in coords:
            if marker == "square":
                marks.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="{color}" />')
            else:
                marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" />')

    best_tlog = min(TLOG_VALUES_MS, key=lambda tlog: float(dashboard_sn_lpf_100[tlog]))
    optimum_x = _log_scale(float(best_tlog), x_lo, x_hi, left, width - right)
    optimum_y = _log_scale(float(dashboard_sn_lpf_100[best_tlog]), y_lo, y_hi, height - bottom, top)
    marks.append(f'<circle cx="{optimum_x:.1f}" cy="{optimum_y:.1f}" r="7" fill="#d62728" stroke="#fff" stroke-width="2"/>')

    paper_crossover = _log_x_crossover(PAPER_FIG05_NF_MEAN, PAPER_FIG05_SN_LPF100_MEAN)
    dashboard_crossover = _log_x_crossover(dashboard_nf, dashboard_sn_lpf_100)
    crossover_marks: list[str] = []
    crossover_labels: list[str] = []
    for label, crossover, color, dash in (
        ("paper", paper_crossover, "#343a3c", "5 5"),
        ("dashboard", dashboard_crossover, "#8c5b16", ""),
    ):
        if crossover is None:
            continue
        cross_tlog, cross_value = crossover
        cross_x = _log_scale(cross_tlog, x_lo, x_hi, left, width - right)
        cross_y = _log_scale(cross_value, y_lo, y_hi, height - bottom, top)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        crossover_marks.append(
            f'<line x1="{cross_x:.1f}" y1="{top}" x2="{cross_x:.1f}" y2="{height-bottom}" '
            f'stroke="{color}" stroke-width="1.5"{dash_attr} opacity="0.75"/>'
            f'<circle cx="{cross_x:.1f}" cy="{cross_y:.1f}" r="7" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        crossover_labels.append(f"{label} {cross_tlog:.0f} ms")

    below_axis_x = _log_scale(1.0, x_lo, x_hi, left, width - right)
    # The paper omits the noise-free 1 ms cell: the one-step predictor has a
    # trivially near-zero solution there, so it is off-scale rather than
    # missing. The dashboard still computes it, and it sits below the axis.
    dashboard_nf_1ms = dashboard_nf.get(1)
    below_axis_dashboard = (
        "not calculated" if dashboard_nf_1ms is None else f"{float(dashboard_nf_1ms):.2e}%"
    )
    below_axis_annotation = (
        f'<path d="M {below_axis_x-7:.1f} {height-bottom-12:.1f} L {below_axis_x+7:.1f} {height-bottom-12:.1f} '
        f'L {below_axis_x:.1f} {height-bottom-2:.1f} Z" fill="#1f77b4"/>'
        f'<text x="{below_axis_x+16:.1f}" y="{height-bottom-12:.1f}" font-size="12" font-family="Arial" fill="#1f5f91">'
        f'NF 1 ms below axis: paper not published (trivial near-zero fit), '
        f'dashboard {below_axis_dashboard}</text>'
    )

    legend_entries = [
        (left + 22, 58, "Paper NF median (ET1, Table S7)", "#1f77b4", "circle", "5 6"),
        (left + 390, 58, "Dashboard NF median (ET1)", "#1f77b4", "circle", ""),
        (left + 22, 84, "Paper SN median, LPF100 (E_Toggle, Fig. S6)", "#d62728", "square", "5 6"),
        (left + 390, 84, "Dashboard SN median, LPF100 (E_Toggle)", "#d62728", "square", ""),
    ]
    legend: list[str] = [
        f'<rect x="{left}" y="44" width="{width-left-right}" height="88" rx="6" fill="#fffdf8" stroke="#d4d0c1"/>'
    ]
    for x, y, label, color, marker, dash in legend_entries:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        legend.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x+28:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        if marker == "square":
            legend.append(f'<rect x="{x+10:.1f}" y="{y-4:.1f}" width="8" height="8" fill="{color}" />')
        else:
            legend.append(f'<circle cx="{x+14:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" />')
        legend.append(f'<text x="{x+40:.1f}" y="{y+4:.1f}" font-size="13" font-family="Arial" fill="#243033">{escape(label)}</text>')
    legend.append(f'<circle cx="{left+36}" cy="113" r="6" fill="#d62728" stroke="#fff" stroke-width="2"/>')
    legend.append(f'<text x="{left+54}" y="117" font-size="13" font-family="Arial" font-weight="700" fill="#8f1d1d">SN median optimum: {best_tlog} ms, MARE_theta {float(dashboard_sn_lpf_100[best_tlog]):.1f}%</text>')

    crossover_summary = "NF=SN crossover: " + " | ".join(crossover_labels)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><clipPath id="tlog-plot-clip"><rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}"/></clipPath></defs>
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="32" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">Tlog sweep - Table S7 NF baseline (ET1) and Fig. S6 100 Hz row (E_Toggle)</text>
{''.join(legend)}
<text x="{left}" y="158" font-size="13" font-family="Arial" font-weight="700" fill="#52605c">{escape(crossover_summary)}</text>
{''.join(grid)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<text x="{width/2}" y="{height-24}" font-size="15" font-family="Arial" text-anchor="middle" fill="#243033">Tlog (ms), log scale</text>
<text x="25" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 25 {height/2})" fill="#243033">MARE_theta (%), log scale</text>
<g clip-path="url(#tlog-plot-clip)">{''.join(marks)}{''.join(crossover_marks)}</g>
{below_axis_annotation}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _write_grouped_bar_chart(rows: Sequence[Mapping[str, float | str]], path: Path, *, title: str, label_key: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1040, 540
    left, right, top, bottom = 86, 44, 64, 124
    finite_values = [
        float(row[key])
        for row in rows
        for key in ("paper_MARE_theta", "dashboard_MARE_theta")
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    max_value = max(finite_values) if finite_values else 1.0
    max_value = max(max_value * 1.16, 1.0)
    group_width = (width - left - right) / max(1, len(rows))
    bar_width = group_width * 0.30
    paper_color = "#2f6f73"
    dashboard_color = "#b35f2e"
    bars: list[str] = []
    for idx, row in enumerate(rows):
        group_x = left + idx * group_width
        label = str(row[label_key])
        for offset, key, color in ((0.18, "paper_MARE_theta", paper_color), (0.52, "dashboard_MARE_theta", dashboard_color)):
            if row.get(key) is None:
                continue
            value = float(row[key])
            if not math.isfinite(value):
                continue
            x = group_x + group_width * offset
            y = height - bottom - (value / max_value) * (height - bottom - top)
            h = height - bottom - y
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" rx="3" fill="{color}" />')
            bars.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" font-size="12" font-family="Arial" text-anchor="middle" fill="#243033">{value:.4g}</text>')
        bars.append(f'<text x="{group_x + group_width * 0.5:.1f}" y="{height - bottom + 26}" font-size="12" font-family="Arial" text-anchor="middle" fill="#243033">{escape(label)}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="32" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">{escape(title)}</text>
<g transform="translate({left + 520},24)"><rect width="18" height="4" y="8" fill="{paper_color}" /><text x="26" y="14" font-size="14" font-family="Arial" fill="#243033">Paper</text></g>
<g transform="translate({left + 610},24)"><rect width="18" height="4" y="8" fill="{dashboard_color}" /><text x="26" y="14" font-size="14" font-family="Arial" fill="#243033">Dashboard</text></g>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<text x="26" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 26 {height/2})" fill="#243033">MARE_theta (%)</text>
{''.join(bars)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _write_heatmap_comparison_chart(dashboard_values: Mapping[str, Sequence[float]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1040
    left, top = 150, 104
    cell_w, cell_h = 104, 58
    height = top + cell_h * len(HEATMAP_EXCITATIONS) + 108
    referenced = [
        value
        for values in PAPER_HEATMAP_MARE.values()
        for value in values
        if value is not None
    ]
    max_value = max(referenced)
    min_value = min(referenced)

    def color_for(value: float) -> str:
        progress = max(0.0, min(1.0, (value - min_value) / max(1e-9, max_value - min_value)))
        dark, mid, light = (0, 104, 55), (72, 181, 97), (224, 243, 188)
        if progress < 0.5:
            p = progress / 0.5
            rgb = tuple(int(dark[i] + (mid[i] - dark[i]) * p) for i in range(3))
        else:
            p = (progress - 0.5) / 0.5
            rgb = tuple(int(mid[i] + (light[i] - mid[i]) * p) for i in range(3))
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    cells: list[str] = []
    for row_index, excitation in enumerate(HEATMAP_EXCITATIONS):
        y = top + row_index * cell_h
        cells.append(f'<text x="{left-18}" y="{y + cell_h * 0.58:.1f}" font-size="14" font-family="Arial" text-anchor="end" fill="#243033">{escape(excitation)}</text>')
        for col_index, tlog in enumerate(HEATMAP_TLOG_VALUES_MS):
            x = left + col_index * cell_w
            paper = PAPER_HEATMAP_MARE[excitation][col_index]
            dashboard = float(dashboard_values[excitation][col_index])
            if paper is None:
                # v5 publishes Fig. S6 for ET1 and E_Toggle only. The dashboard
                # value still stands on its own; there is simply nothing to
                # compare it against.
                cells.append(f'<rect x="{x+1:.1f}" y="{y+1:.1f}" width="{cell_w-2:.1f}" height="{cell_h-2:.1f}" rx="3" fill="{color_for(dashboard)}" stroke="#98a2a5" stroke-width="2" stroke-dasharray="4 3"/>')
                cells.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 23:.1f}" font-size="11" font-family="Arial" text-anchor="middle" fill="#5b6769">no v5 ref</text>')
                cells.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 42:.1f}" font-size="12" font-family="Arial" font-weight="700" text-anchor="middle" fill="#101a16">D {dashboard:.1f}</text>')
                continue
            comparison_error = _percent_error(paper, dashboard)
            comparison_passes = comparison_error <= 8.0
            comparison_color = "#18794e" if comparison_passes else "#b42318"
            comparison_label = "OK" if comparison_passes else "!"
            cells.append(f'<rect x="{x+1:.1f}" y="{y+1:.1f}" width="{cell_w-2:.1f}" height="{cell_h-2:.1f}" rx="3" fill="{color_for(dashboard)}" stroke="{comparison_color}" stroke-width="3"/>')
            cells.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 23:.1f}" font-size="12" font-family="Arial" text-anchor="middle" fill="#101a16">P {paper:.1f}</text>')
            cells.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 42:.1f}" font-size="12" font-family="Arial" font-weight="700" text-anchor="middle" fill="#101a16">D {dashboard:.1f}</text>')
            cells.append(f'<circle cx="{x+cell_w-12:.1f}" cy="{y+12:.1f}" r="9" fill="{comparison_color}"/><text x="{x+cell_w-12:.1f}" y="{y+16:.1f}" font-size="10" font-family="Arial" font-weight="700" text-anchor="middle" fill="#fff">{comparison_label}</text>')
    headers = [
        f'<text x="{left + col_index * cell_w + cell_w / 2:.1f}" y="{top - 18}" font-size="14" font-family="Arial" text-anchor="middle" fill="#243033">{tlog}</text>'
        for col_index, tlog in enumerate(HEATMAP_TLOG_VALUES_MS)
    ]
    axis_y = top + cell_h * len(HEATMAP_EXCITATIONS) / 2
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="34" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">Paper vs Dashboard Mean MARE Heatmap</text>
<text x="{left + cell_w * len(HEATMAP_TLOG_VALUES_MS) / 2:.1f}" y="62" font-size="14" font-family="Arial" text-anchor="middle" fill="#52605c">Tlog (ms)</text>
<g transform="translate({width-310},28)"><rect width="18" height="12" rx="2" fill="none" stroke="#18794e" stroke-width="3"/><text x="26" y="11" font-size="12" font-family="Arial" fill="#243033">within 8%</text><rect x="112" width="18" height="12" rx="2" fill="none" stroke="#b42318" stroke-width="3"/><text x="138" y="11" font-size="12" font-family="Arial" fill="#243033">over 8%</text></g>
<text x="34" y="{axis_y:.1f}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 34 {axis_y:.1f})" fill="#243033">Excitation type</text>
{''.join(headers)}
{''.join(cells)}
<text x="{left}" y="{height-34}" font-size="13" font-family="Arial" fill="#52605c">P = paper, D = dashboard; mean across 10 plants and five noise levels. Dashed inner box = corrected EV1/20 ms reference cell.</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _write_trace_comparison_chart(
    trace_rows: Sequence[Mapping[str, object]],
    path: Path,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 700
    left, right, top, bottom = 88, 36, 54, 66
    panel_gap = 24
    panel_h = (height - top - bottom - 2 * panel_gap) / 3.0
    displayed = [row for row in trace_rows if float(row["time_s"]) <= 4.0]
    if not displayed:
        raise ValueError("Logged-tension chart requires simulated trace rows")
    finite_tensions = [float(row["normalized_tension"]) for row in displayed]
    finite_refs = [float(row["normalized_reference"]) for row in displayed]
    y_lo = min([0.9, *finite_tensions, *finite_refs]) - 0.03
    y_hi = max([1.3, *finite_tensions, *finite_refs]) + 0.03
    modes = [
        ("NF", "Noise-free simulation", "(a)"),
        ("SN_no_LPF", "1% noise, no LPF", "(b)"),
        ("SN_LPF50", "1% noise, LPF 50 Hz", "(c)"),
    ]
    content: list[str] = []
    for panel_index, (mode, label, tag) in enumerate(modes):
        panel_top = top + panel_index * (panel_h + panel_gap)
        panel_bottom = panel_top + panel_h
        mode_rows = [row for row in displayed if str(row["mode"]) == mode]
        for tick in (0, 1, 2, 3, 4):
            x = _linear_scale(float(tick), 0.0, 4.0, left, width - right)
            content.append(f'<line x1="{x:.1f}" y1="{panel_top:.1f}" x2="{x:.1f}" y2="{panel_bottom:.1f}" stroke="#e3e4dd" stroke-width="1"/>')
            if panel_index == 2:
                content.append(f'<text x="{x:.1f}" y="{panel_bottom+24:.1f}" font-size="13" font-family="Arial" text-anchor="middle" fill="#243033">{tick}</text>')
        y_ticks = [y_lo + index * (y_hi - y_lo) / 5.0 for index in range(6)]
        for y_tick in y_ticks:
            y = _linear_scale(y_tick, y_lo, y_hi, panel_bottom, panel_top)
            content.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ebebe4" stroke-width="1"/>')
            content.append(f'<text x="{left-12}" y="{y+4:.1f}" font-size="12" font-family="Arial" text-anchor="end" fill="#243033">{y_tick:.2f}</text>')
        reference_points = " ".join(
            f"{_linear_scale(float(row['time_s']), 0.0, 4.0, left, width-right):.1f},"
            f"{_linear_scale(float(row['normalized_reference']), y_lo, y_hi, panel_bottom, panel_top):.1f}"
            for row in mode_rows
        )
        tension_points = " ".join(
            f"{_linear_scale(float(row['time_s']), 0.0, 4.0, left, width-right):.1f},"
            f"{_linear_scale(float(row['normalized_tension']), y_lo, y_hi, panel_bottom, panel_top):.1f}"
            for row in mode_rows
        )
        content.append(f'<polyline fill="none" stroke="#777777" stroke-width="2" stroke-dasharray="5 6" points="{reference_points}" />')
        content.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.2" points="{tension_points}" />')
        content.append(f'<rect x="{left+8}" y="{panel_top+7:.1f}" width="28" height="22" fill="#fffdf8" stroke="#c8c8c0"/><text x="{left+22}" y="{panel_top+23:.1f}" font-size="14" font-family="Arial" font-weight="700" text-anchor="middle" fill="#111">{tag}</text>')
        content.append(f'<rect x="{width-right-180}" y="{panel_top+8:.1f}" width="166" height="34" fill="#fffdf8" stroke="#c8c8c0"/><text x="{width-right-97}" y="{panel_top+29:.1f}" font-size="12" font-family="Arial" text-anchor="middle" fill="#52605c">{escape(label)}</text>')
        content.append(f'<rect x="{left}" y="{panel_top:.1f}" width="{width-left-right}" height="{panel_h:.1f}" fill="none" stroke="#445" stroke-width="1.4"/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="30" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">Fresh Simulated Logged Tension Traces - {TRACE_PLANT_ID}</text>
<g transform="translate({width-right-310},22)"><line x1="0" y1="7" x2="24" y2="7" stroke="#777" stroke-width="2.2" stroke-dasharray="5 6"/><text x="32" y="11" font-size="13" font-family="Arial" fill="#243033">Tension reference</text></g>
<g transform="translate({width-right-162},22)"><line x1="0" y1="7" x2="24" y2="7" stroke="#1f77b4" stroke-width="2.2"/><text x="32" y="11" font-size="13" font-family="Arial" fill="#243033">Measured T1</text></g>
<text x="26" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 26 {height/2})" fill="#243033">Normalized tension, T/Tref,0</text>
<text x="{width/2}" y="{height-20}" font-size="15" font-family="Arial" text-anchor="middle" fill="#243033">Time (s)</text>
{''.join(content)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def noise_aware_logging_lpf_validation(
    prefer_cache: bool = True,
) -> dict[str, object]:
    """Return the Noise-aware logging validation dataset, plots, and pass/fail decision."""

    live_values = _compute_live_dashboard_values(prefer_cache=prefer_cache)
    dashboard_nf = _live_curve(live_values, "dashboard_nf_mare")
    dashboard_fig05_sn_lpf_100 = _live_curve(live_values, "dashboard_fig05_sn_lpf_100_mean")
    dashboard_sn_lpf_50 = _live_curve(live_values, "dashboard_sn_lpf_50_mare")
    dashboard_sn_lpf_100 = _live_curve(live_values, "dashboard_sn_lpf_100_mare")
    dashboard_heatmap = _dashboard_heatmap_values(live_values)
    heatmap_table = _heatmap_rows(dashboard_heatmap)
    figs10_table = _figs10_rows(live_values)
    lpf_rows = _build_lpf_rows(live_values)
    comparison_table = _comparison_rows(dashboard_nf, dashboard_sn_lpf_50, dashboard_sn_lpf_100)
    transition_table = _transition_rows(live_values)
    cross_channel_table = _cross_channel_rows(live_values)
    main_effect_table = _main_effect_rows(live_values)
    scorecard = _reference_scorecard(live_values)
    filter_configurations = _filter_config_rows()
    tlog_rows = [
        {
            "Tlog_ms": tlog,
            "aggregation": "median over 10 plants (Table S7 NF baseline is ET1, Fig. S6 100 Hz row is E_Toggle)",
            "paper_NF_no_LPF_MARE_theta": PAPER_FIG05_NF_MEAN.get(tlog),
            "dashboard_NF_no_LPF_MARE_theta": dashboard_nf[tlog],
            "paper_SN_0.3pct_LPF100_MARE_theta": PAPER_FIG05_SN_LPF100_MEAN[tlog],
            "dashboard_SN_0.3pct_LPF100_MARE_theta": dashboard_fig05_sn_lpf_100[tlog],
            "dashboard_source": "live_simulation_sysid_cache",
        }
        for tlog in TLOG_VALUES_MS
    ]

    nf_values = [dashboard_nf[tlog] for tlog in TLOG_VALUES_MS]
    fig05_sn_values = [dashboard_fig05_sn_lpf_100[tlog] for tlog in TLOG_VALUES_MS]
    sn_50_values = [dashboard_sn_lpf_50[tlog] for tlog in TLOG_VALUES_MS]
    best_sn_tlog = min(TLOG_VALUES_MS, key=lambda tlog: dashboard_sn_lpf_50[tlog])
    best_fig05_sn_tlog = min(TLOG_VALUES_MS, key=lambda tlog: dashboard_fig05_sn_lpf_100[tlog])
    lpf_10 = next(row for row in lpf_rows if row["LPF"] == "10 Hz")
    lpf_20 = next(row for row in lpf_rows if row["LPF"] == "20 Hz")
    lpf_50 = next(row for row in lpf_rows if row["LPF"] == "50 Hz")
    lpf_100 = next(row for row in lpf_rows if row["LPF"] == "100 Hz")
    raw_rows = live_values.get("raw_rows", [])
    valid_raw_rows = [
        row
        for row in raw_rows
        if isinstance(row, Mapping)
        and row.get("status") == "ok"
        and row.get("dashboard_MARE_theta_percent") is not None
        and math.isfinite(float(row["dashboard_MARE_theta_percent"]))
    ] if isinstance(raw_rows, list) else []
    trace_rows = live_values.get("trace_rows", [])
    def valid_count(
        *,
        excitation: str,
        tlog: int,
        condition: str,
        lpf_hz: int | str,
        noise_level_percent: float | None = None,
    ) -> int:
        return sum(
            1
            for row in valid_raw_rows
            if row["excitation"] == excitation
            and int(row["Tlog_ms"]) == int(tlog)
            and row["condition"] == condition
            and str(row["lpf_hz"]) == str(lpf_hz)
            # campaign 1 only: the cross-channel group and the velocity-noise
            # legs are separate campaigns with their own coverage.
            and str(row.get("campaign_group")) == FACTORIAL_CAMPAIGN_GROUP
            and float(row.get("velocity_noise_percent", 0.0)) == 0.0
            and int(row.get("seed", 0)) == 0
            and (
                noise_level_percent is None
                or math.isclose(
                    float(row["noise_level_percent"]),
                    float(noise_level_percent),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        )

    figure5_nf_coverage = {
        tlog: valid_count(
            excitation="E_Toggle",
            tlog=tlog,
            condition="NF",
            lpf_hz="none",
            noise_level_percent=0.0,
        )
        for tlog in TLOG_VALUES_MS
    }
    figure5_sn_coverage = {
        tlog: valid_count(
            excitation="E_Toggle",
            tlog=tlog,
            condition="SN",
            lpf_hz=100,
            noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
        )
        for tlog in TLOG_VALUES_MS
    }
    figure6_coverage = {
        f"{excitation}:{tlog}": valid_count(
            excitation=excitation,
            tlog=tlog,
            condition="SN",
            lpf_hz=50,
        )
        for excitation in HEATMAP_EXCITATIONS
        for tlog in HEATMAP_TLOG_VALUES_MS
    }
    figs10_coverage = {
        f"{cutoff}:{excitation}:{tlog}": valid_count(
            excitation=excitation,
            tlog=tlog,
            condition="SN",
            lpf_hz=cutoff,
            noise_level_percent=NOMINAL_NOISE_LEVEL_PERCENT,
        )
        for cutoff in ("none", 50, 100, 200)
        for excitation in ("ET1", "E_Toggle")
        for tlog in TLOG_VALUES_MS
    }
    # Every run below the 50 Hz floor is expected to diverge - that is the
    # finding, not a defect - so those rows are assessed by their failure rate
    # instead of being required to be finite.
    expected_low_cutoff_rows = [
        row
        for row in raw_rows
        if isinstance(row, Mapping) and str(row.get("lpf_hz")) in {"10", "20"}
    ] if isinstance(raw_rows, list) else []
    expected_low_cutoff_ids = {id(row) for row in expected_low_cutoff_rows}
    core_rows = [
        row
        for row in raw_rows
        if isinstance(row, Mapping) and id(row) not in expected_low_cutoff_ids
    ] if isinstance(raw_rows, list) else []
    finite_core_rows = [
        row
        for row in valid_raw_rows
        if id(row) not in expected_low_cutoff_ids
    ]
    complete_core_coverage = (
        len(core_rows) > 0 and len(finite_core_rows) == len(core_rows)
    )
    figure5_complete = (
        all(count == 10 for count in figure5_nf_coverage.values())
        and all(count == 10 for count in figure5_sn_coverage.values())
    )
    figure6_complete = (
        len(heatmap_table)
        == len(HEATMAP_EXCITATIONS) * len(HEATMAP_TLOG_VALUES_MS)
        and all(count == 50 for count in figure6_coverage.values())
    )
    figs10_complete = all(count == 10 for count in figs10_coverage.values())
    # Fig. S6(c): the 100 Hz E_Toggle row bottoms out at 2 ms, not in the v4.1
    # 10-20 ms window. The window is taken from the paper row itself so that it
    # follows the reference rather than a transcribed constant.
    paper_fig_s6_100_best_tlog = min(
        TLOG_VALUES_MS, key=lambda tlog: PAPER_FIGS10_ETOGGLE_LPF100_MEDIAN[tlog]
    )
    paper_fig_s6_50_best_tlog = min(
        TLOG_VALUES_MS, key=lambda tlog: PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[tlog]
    )
    figure5_paper_range_match = best_fig05_sn_tlog == paper_fig_s6_100_best_tlog
    figure6_best = min(
        heatmap_table,
        key=lambda row: float(row["dashboard_MARE_theta"]),
    )
    key_case_pass_count = sum(
        1 for row in comparison_table if row.get("pass_fail") == "PASS"
    )
    figure6_pass_count = sum(
        1 for row in heatmap_table if row.get("pass_fail") == "PASS"
    )
    figure6_referenced_cells = sum(
        1 for row in heatmap_table if row.get("pass_fail") != "NO_REFERENCE"
    )
    figs10_pass_count = sum(
        1 for row in figs10_table if row.get("pass_fail") == "PASS"
    )
    cross_channel_pass_count = sum(
        1 for row in cross_channel_table if row.get("pass_fail") == "PASS"
    )
    cross_channel_settings = live_values.get("cross_channel_dashboard", {})
    cross_channel_group = (
        str(cross_channel_settings.get("campaign_group"))
        if isinstance(cross_channel_settings, Mapping)
        else None
    )
    cross_channel_duration_s = (
        float(cross_channel_settings.get("record_duration_s", math.nan))
        if isinstance(cross_channel_settings, Mapping)
        else math.nan
    )
    cross_channel_ratios = [
        row["dashboard_velocity_over_tension"]
        for row in cross_channel_table
        if row.get("dashboard_velocity_over_tension") is not None
    ]
    cross_channel_monotonic = (
        len(cross_channel_ratios) == 3
        and cross_channel_ratios[0] > cross_channel_ratios[1] > cross_channel_ratios[2]
        and cross_channel_ratios[2] < 2.0
    )
    cross_channel_ratio_text = ", ".join(
        f"{row['Tlog_ms']} ms {float(row['dashboard_velocity_over_tension']):.2f}x"
        for row in cross_channel_table
        if row.get("dashboard_velocity_over_tension") is not None
    )
    main_effect_text = ", ".join(
        f"{row['factor']} {row['dashboard_median_spread_pp']:.1f} pp"
        for row in main_effect_table
        if row.get("dashboard_median_spread_pp") is not None
    )
    main_effect_ranking_match = bool(main_effect_table) and all(
        row.get("dashboard_rank") is not None for row in main_effect_table
    ) and (
        next(row["dashboard_rank"] for row in main_effect_table if row["factor"] == "Tlog") == 1
        and next(row["dashboard_rank"] for row in main_effect_table if row["factor"] == "lpf_cutoff") == 4
    )
    scorecard_computed = sum(
        1 for row in scorecard if row.get("pass_fail") in {"PASS", "FAIL"}
    )
    scorecard_missing = sum(1 for row in scorecard if row.get("pass_fail") == "NOT_COMPUTED")
    scorecard_pass = sum(1 for row in scorecard if row.get("pass_fail") == "PASS")

    acceptance = [
        {"category": "calculation_integrity", "criterion": "All core Figure 5/Figure 6/Figure S10 simulations and Eq. (8) weighted fits are finite", "evidence": f"Finite core rows={len(finite_core_rows)}/{len(core_rows)}; low-cutoff sensitivity rows are assessed separately.", "pass_fail": "PASS" if complete_core_coverage else "FAIL"},
        {"category": "calculation_integrity", "criterion": "All finite calculated rows use paper Eq. (8) weighted one-step PEM", "evidence": f"{len(valid_raw_rows)} finite rows identify with paper_eq8_weighted_pem_trf.", "pass_fail": "PASS" if valid_raw_rows and all(row.get("estimator") == "paper_eq8_weighted_pem_trf" for row in valid_raw_rows) else "FAIL"},
        {"category": "calculation_integrity", "criterion": "Figure 5 and Figure S10 have complete ten-plant coverage", "evidence": f"Figure5 complete={figure5_complete}; FigureS10 complete={figs10_complete}.", "pass_fail": "PASS" if figure5_complete and figs10_complete else "FAIL"},
        {"category": "scientific_trend", "criterion": "NF Figure 5 mean increases with Tlog", "evidence": f"Dashboard NF means are {[round(value, 3) for value in nf_values]}.", "pass_fail": "PASS" if _monotonic_increasing(nf_values) else "FAIL"},
        {"category": "scientific_trend", "criterion": "SN Figure 5 curve is U-shaped", "evidence": f"Dashboard SN/LPF100 means are {[round(value, 3) for value in fig05_sn_values]}; best Tlog is {best_fig05_sn_tlog} ms.", "pass_fail": "PASS" if _is_u_shaped(fig05_sn_values) else "FAIL"},
        {"category": "paper_comparison", "criterion": "The 100 Hz E_Toggle minimum falls at the logging period Fig. S6(c) puts it at", "evidence": f"Paper minimum {paper_fig_s6_100_best_tlog} ms; recalculated minimum {best_fig05_sn_tlog} ms.", "pass_fail": "PASS" if figure5_paper_range_match else "FAIL"},
        {"category": "paper_comparison", "criterion": "All seven configured Figure 5 key comparisons are within tolerance", "evidence": f"Within-tolerance key cases={key_case_pass_count}/{len(comparison_table)}.", "pass_fail": "PASS" if key_case_pass_count == len(comparison_table) else "FAIL"},
        {"category": "scientific_trend", "criterion": "Under tension-only noise the finest logging is best - no interior optimum (Fig. S6(b); logging_rate_v5_reference notes)", "evidence": f"Paper 50 Hz E_Toggle minimum {paper_fig_s6_50_best_tlog} ms; recalculated {best_sn_tlog} ms.", "pass_fail": "PASS" if best_sn_tlog == paper_fig_s6_50_best_tlog else "FAIL"},
        {"category": "scientific_trend", "criterion": "LPF 10/20 Hz reproduce the expected convergence failures", "evidence": f"Failure rates: 10 Hz={float(lpf_10['convergence_failure_rate_percent']):.1f}%, 20 Hz={float(lpf_20['convergence_failure_rate_percent']):.1f}%, 50 Hz={float(lpf_50['convergence_failure_rate_percent']):.1f}%.", "pass_fail": "PASS" if math.isclose(float(lpf_10["convergence_failure_rate_percent"]), 100.0, abs_tol=1e-12) and math.isclose(float(lpf_20["convergence_failure_rate_percent"]), 70.0, abs_tol=1e-12) and math.isclose(float(lpf_50["convergence_failure_rate_percent"]), 0.0, abs_tol=1e-12) else "FAIL"},
        {"category": "scientific_trend", "criterion": "LPF 50 Hz achieves stable convergence", "evidence": f"50 Hz failure rate is {float(lpf_50['convergence_failure_rate_percent']):.1f}%.", "pass_fail": "PASS" if float(lpf_50["convergence_failure_rate_percent"]) == 0.0 else "FAIL"},
        {"category": "paper_comparison", "criterion": "50 Hz beats 100 Hz at the working cell (Section 3.4: 25.9% vs 30.8%) - raising the cutoff past 50 Hz brings no benefit", "evidence": f"LPF50={float(lpf_50['dashboard_MARE_theta']):.3f}%, LPF100={float(lpf_100['dashboard_MARE_theta']):.3f}%.", "pass_fail": "PASS" if float(lpf_50["dashboard_MARE_theta"]) <= float(lpf_100["dashboard_MARE_theta"]) else "FAIL"},
        {"category": "calculation_integrity", "criterion": "Figure 6 contains all six excitations, 42 cells, and 50 runs per cell", "evidence": f"Cells={len(heatmap_table)}; minimum cell coverage={min(figure6_coverage.values()) if figure6_coverage else 0}/50.", "pass_fail": "PASS" if figure6_complete else "FAIL"},
        {"category": "paper_comparison", "criterion": "The heatmap compares only the two excitations Fig. S6 publishes (ET1, E_Toggle); the other four carry no v5 reference", "evidence": f"Referenced excitations={list(PAPER_HEATMAP_REFERENCE_EXCITATIONS)}; calculated best={figure6_best['excitation']}/{figure6_best['Tlog_ms']}ms.", "pass_fail": "PASS" if all(PAPER_HEATMAP_MARE[name][0] is not None for name in PAPER_HEATMAP_REFERENCE_EXCITATIONS) and all(PAPER_HEATMAP_MARE[name][0] is None for name in HEATMAP_EXCITATIONS if name not in PAPER_HEATMAP_REFERENCE_EXCITATIONS) else "FAIL"},
        {"category": "paper_comparison", "criterion": "Every heatmap cell the paper publishes is within the configured 8% comparison tolerance (Fig. S6 plots ET1 and E_Toggle only; the other four excitations carry no reference and are not scored)", "evidence": f"Within-tolerance referenced cells={figure6_pass_count}/{figure6_referenced_cells}; unreferenced cells={len(heatmap_table)-figure6_referenced_cells}.", "pass_fail": "PASS" if figure6_referenced_cells and figure6_pass_count == figure6_referenced_cells else "FAIL"},
        {"category": "paper_comparison", "criterion": "All 56 Figure S10 cells are within the configured 15% comparison tolerance", "evidence": f"Within-tolerance Figure S10 cells={figs10_pass_count}/{len(figs10_table)}.", "pass_fail": "PASS" if figs10_pass_count == len(figs10_table) else "FAIL"},
        {"category": "calculation_integrity", "criterion": "Logged-tension graph uses simulated rows", "evidence": f"Trace rows={len(trace_rows) if isinstance(trace_rows, list) else 0}, modes=NF/SN_no_LPF/SN_LPF50.", "pass_fail": "PASS" if isinstance(trace_rows, list) and len(trace_rows) > 0 else "FAIL"},
        {"category": "calculation_integrity", "criterion": "The cross-channel campaigns run ET1 on the 30 s B_dual_channel record the schedule CSV flags as the reproduction gate", "evidence": f"Cross-channel group={cross_channel_group}, record duration={cross_channel_duration_s} s (schedule CSV: 30 s).", "pass_fail": "PASS" if cross_channel_group == CROSS_CHANNEL_CAMPAIGN_GROUP and cross_channel_duration_s == 30.0 else "FAIL"},
        {"category": "paper_comparison", "criterion": "All three published cross-channel velocity/tension ratios are within 20%", "evidence": f"Within tolerance {cross_channel_pass_count}/{len(cross_channel_table)}: " + ", ".join(f"{row['Tlog_ms']}ms dash={row['dashboard_velocity_over_tension']:.2f}x paper={row['paper_velocity_over_tension']}x" for row in cross_channel_table if row['dashboard_velocity_over_tension'] is not None) + ".", "pass_fail": "PASS" if cross_channel_table and cross_channel_pass_count == len(cross_channel_table) else "FAIL"},
        {"category": "scientific_trend", "criterion": "Cross-channel asymmetry decays with the logging period and reaches near-parity at 50 ms", "evidence": f"Recalculated ratios {cross_channel_ratio_text}.", "pass_fail": "PASS" if cross_channel_monotonic else "FAIL"},
        {"category": "paper_comparison", "criterion": "The main-effect ranking is Tlog first and the cutoff last (the v4.1 LPF-first ranking is retired)", "evidence": f"Recalculated median spreads: {main_effect_text}.", "pass_fail": "PASS" if main_effect_ranking_match else "FAIL"},
        {"category": "paper_comparison", "criterion": "Every quantity noise_lpf_reference.json publishes for this section is computed", "evidence": f"Scored {scorecard_computed}/{len(scorecard)}; not computed {scorecard_missing}.", "pass_fail": "PASS" if scorecard and scorecard_missing == 0 else "FAIL"},
    ]
    passed = sum(1 for row in acceptance if row["pass_fail"] == "PASS")
    failed = len(acceptance) - passed
    integrity_failed = any(
        row["pass_fail"] == "FAIL"
        and row.get("category") == "calculation_integrity"
        for row in acceptance
    )
    validation_status = (
        "FAIL"
        if integrity_failed
        else ("CHECK" if failed else "PASS")
    )

    tlog_plot_path = _write_tlog_comparison_chart(
        FIGURES_DIR / "noiseLpf_tlog_paper_vs_dashboard.svg",
        dashboard_nf,
        dashboard_fig05_sn_lpf_100,
    )
    lpf_plot_path = _write_grouped_bar_chart(lpf_rows, FIGURES_DIR / "noiseLpf_lpf_paper_vs_dashboard.svg", title="Noise-aware logging Paper vs Dashboard LPF Cutoff", label_key="LPF")
    comparison_plot_path = _write_grouped_bar_chart(comparison_table, FIGURES_DIR / "noiseLpf_key_cases_paper_vs_dashboard.svg", title="Noise-aware logging Key Cases Paper vs Dashboard", label_key="case_label")
    heatmap_plot_path = _write_heatmap_comparison_chart(dashboard_heatmap, FIGURES_DIR / "noiseLpf_heatmap_paper_vs_dashboard.svg")
    trace_plot_path = _write_trace_comparison_chart(
        trace_rows if isinstance(trace_rows, list) else [],
        FIGURES_DIR / "noiseLpf_traces_paper_vs_dashboard.svg",
    )

    metrics = {
        "workflow": "noise-aware logging LPF",
        "status": validation_status,
        "pass_count": passed,
        "fail_count": failed,
        "best_SN_LPF50_Tlog_ms": best_sn_tlog,
        "best_SN_LPF50_MARE_theta": dashboard_sn_lpf_50[best_sn_tlog],
        "best_Figure5_SN_LPF100_Tlog_ms": best_fig05_sn_tlog,
        "best_Figure5_SN_LPF100_MARE_theta": dashboard_fig05_sn_lpf_100[best_fig05_sn_tlog],
        "recommended_Tlog_ms": "10-20",
        "recommended_LPF_Hz": "50-100",
        "sensor_noise_percent_full_scale": NOMINAL_NOISE_LEVEL_PERCENT,
        "figure6_noise_levels_percent": list(FIG06_NOISE_LEVELS_PERCENT),
        "dt_ms": DT_S * 1000.0,
        "controller_period_ms": CONTROLLER_SAMPLE_TIME_S * 1000.0,
        "controller_integral_time": "per_plant_auto_Ti",
        "high_ea_kp_cap_enabled": False,
        "velocity_correction_limit_fraction": None,
        "noise_rng": "numpy.default_rng(seed)",
        "estimator": "paper_eq8_weighted_pem_trf",
        "raw_row_count": len(raw_rows) if isinstance(raw_rows, list) else 0,
        "finite_raw_row_count": len(valid_raw_rows),
        "figure5_nf_coverage": figure5_nf_coverage,
        "figure5_sn_coverage": figure5_sn_coverage,
        "figure6_min_cell_coverage": min(figure6_coverage.values()) if figure6_coverage else 0,
        "figureS10_min_cell_coverage": min(figs10_coverage.values()) if figs10_coverage else 0,
        "expected_low_cutoff_failure_rows": len(expected_low_cutoff_rows) - sum(
            1 for row in expected_low_cutoff_rows if row.get("status") == "ok"
        ),
        "figure5_paper_range_match": figure5_paper_range_match,
        "trace_plant_id": TRACE_PLANT_ID,
        "trace_duration_s": TRACE_DURATION_S,
        "noise_seeds_used": live_values.get("settings", {}).get("noise_seeds", [0]) if isinstance(live_values.get("settings"), Mapping) else [0],
        "dashboard_value_source": "live_simulation_sysid_cache",
        "reference_scorecard_total": len(scorecard),
        "reference_scorecard_computed": scorecard_computed,
        "reference_scorecard_not_computed": scorecard_missing,
        "reference_scorecard_within_tolerance": scorecard_pass,
        "cross_channel_campaign_group": cross_channel_group,
        "cross_channel_record_duration_s": cross_channel_duration_s,
        "cross_channel_ratios": cross_channel_ratio_text,
        "main_effect_spreads": main_effect_text,
    }

    calculations = [
        {
            "title": "LPF alpha at 50 Hz",
            "parameter": "alpha",
            "summary": "First-order EMA coefficient used by the anti-alias filter.",
            "formula": "alpha = 1 - exp(-2*pi*fc*dt)",
            "steps": ["Use fc = 50 Hz.", "Use dt = 0.001 s.", "Compute 2*pi*50*0.001 = 0.314159.", "Evaluate 1 - exp(-0.314159).", "The coefficient is approximately 0.2696."],
            "values": {"fc_Hz": 50, "dt_s": DT_S, "alpha": _lpf_alpha(50)},
            "substitution": "alpha = 1 - exp(-0.314159) = 0.269597",
            "result": "y_f[k] = 0.2696*y_meas[k] + 0.7304*y_f[k-1]",
        },
        {
            "title": "Paper vs dashboard error",
            "parameter": "Error(%)",
            "summary": (
                "Example uses the Figure S10 nominal-noise median for "
                "E_Toggle, Tlog 20 ms, LPF 50 Hz."
            ),
            "formula": "Error(%) = |paper - dashboard| / paper * 100",
            "steps": [
                f"Paper MARE_theta is {PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20]:.6f}%.",
                f"Dashboard MARE_theta is {float(dashboard_sn_lpf_50[20]):.6g}%.",
                "Take the absolute difference.",
                f"Divide by {PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20]:.6f} and multiply by 100.",
                "Compare against the 15% validation tolerance.",
            ],
            "values": {
                "paper_MARE_theta_percent": PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20],
                "dashboard_MARE_theta_percent": dashboard_sn_lpf_50[20],
                "error_percent": _percent_error(
                    PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20],
                    dashboard_sn_lpf_50[20],
                ),
            },
            "substitution": (
                f"Error = |{PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20]:.6f} - "
                f"{float(dashboard_sn_lpf_50[20]):.6g}| / "
                f"{PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20]:.6f} * 100"
            ),
            "result": (
                "PASS"
                if _percent_error(
                    PAPER_FIGS10_ETOGGLE_LPF50_MEDIAN[20],
                    dashboard_sn_lpf_50[20],
                )
                <= 15.0
                else "CHECK"
            ),
        },
    ]

    observed_conclusion = [
        (
            f"The recalculated Figure 5 NF mean curve is "
            f"{'monotonic increasing' if _monotonic_increasing(nf_values) else 'not monotonic increasing'} "
            "with logging period."
        ),
        (
            f"The recalculated Figure 5 SN/LPF100 mean curve has its minimum at "
            f"{best_fig05_sn_tlog} ms and "
            f"{'does' if _is_u_shaped(fig05_sn_values) else 'does not'} form a U-shape."
        ),
        (
            f"At nominal 0.3% noise, the recalculated E_Toggle/LPF50 plant median "
            f"is lowest at {best_sn_tlog} ms."
        ),
        (
            f"The complete Figure 6 recalculation contains {len(heatmap_table)} cells; "
            f"its minimum is {figure6_best['excitation']} at "
            f"{figure6_best['Tlog_ms']} ms."
        ),
        (
            "Paper values remain comparison-only; dashboard values come from fresh "
            "simulation followed by paper Eq. (8) weighted one-step PEM."
        ),
    ]

    payload: dict[str, object] = {
        "study": "noise-aware-logging-lpf",
        "source_document": str(NOISE_LPF_BRIEF),
        "metrics": metrics,
        "input_summary": {
            "plant_count": len(plant_registry()),
            "plant_integration_step_ms": DT_S * 1000.0,
            "control_period_ms": CONTROLLER_SAMPLE_TIME_S * 1000.0,
            "logging_periods_ms": list(TLOG_VALUES_MS),
            "nominal_sensor_noise_percent_full_scale": NOMINAL_NOISE_LEVEL_PERCENT,
            "figure6_noise_levels_percent_full_scale": list(FIG06_NOISE_LEVELS_PERCENT),
            "lpf_cutoffs_Hz": ["none", 10, 20, 50, 100, 200],
            "figure5_excitation": "E_Toggle",
            "figure6_excitations": list(HEATMAP_EXCITATIONS),
            "figureS10_excitations": ["ET1", "E_Toggle"],
            "controller_integral_time": "per_plant_auto_Ti",
            "high_ea_kp_cap_enabled": False,
            "velocity_correction_limit_fraction": None,
            "noise_affects_controller": True,
            "noise_rng": "numpy.default_rng(seed)",
            "estimator": "paper_eq8_weighted_pem_trf",
            "per_run_metric": "100*mean(abs((theta_hat-theta_true)/theta_true))",
            "figure5_aggregation": "mean_over_10_plants",
            "figure6_aggregation": "mean_over_10_plants_and_5_noise_levels",
            "figureS10_aggregation": "median_over_10_plants_at_0.3pct_noise",
            "trace_plant_id": TRACE_PLANT_ID,
            "trace_duration_s": TRACE_DURATION_S,
            "primary_output": "tension channels y = [T1, T2, T3]^T",
        },
        "tlog_sweep": tlog_rows,
        "lpf_sweep": lpf_rows,
        "live_dashboard_cache_path": str(LIVE_CACHE_PATH),
        "live_dashboard_settings": live_values.get("settings", {}),
        "live_dashboard_raw_rows_count": len(live_values.get("raw_rows", [])) if isinstance(live_values.get("raw_rows"), list) else None,
        "heatmap_comparison": heatmap_table,
        "heatmap_five_level_mean": live_values.get("heatmap_five_level_mean", {}),
        "figs10_comparison": figs10_table,
        "transition_table_comparison": transition_table,
        "cross_channel_comparison": cross_channel_table,
        "cross_channel_detail": live_values.get("cross_channel_dashboard", {}),
        "dual_channel_dashboard": live_values.get("dual_channel_dashboard", {}),
        "main_effect_comparison": main_effect_table,
        "six_excitation_failure_rates": live_values.get("six_excitation_failure_rates", {}),
        "reference_scorecard": scorecard,
        "comparison_table": comparison_table,
        "trace_rows": trace_rows,
        "filter_configurations": filter_configurations,
        "acceptance_criteria": acceptance,
        "recommendation": (
            f"Use the recalculated minimum as evidence: Figure 5 SN/LPF100 "
            f"{best_fig05_sn_tlog} ms; nominal E_Toggle/LPF50 {best_sn_tlog} ms. "
            "The v5 guidance is f_c >= 50 Hz with T_log = 5-20 ms under dual-channel "
            "noise. 50 Hz is the working cutoff, not a floor to exceed: raising it to "
            "100 Hz measurably worsens the fit. Cutoffs at or below 20 Hz are a "
            "convergence gate rather than a filter choice. Under tension-only noise "
            "there is no interior optimum at all - the finest logging is best. Retain "
            "these ranges only where the fresh trend and convergence evidence support them."
        ),
        "conclusion": observed_conclusion,
        "plots": {
            "tlog_sweep": {"title": "Figure 5 paper vs dashboard mean Tlog sweep", "path": tlog_plot_path},
            "lpf_cutoff": {"title": "Paper vs dashboard LPF cutoff", "path": lpf_plot_path},
            "paper_vs_dashboard": {"title": "Paper vs dashboard key cases", "path": comparison_plot_path},
            "heatmap": {"title": "Figure 6 paper vs dashboard five-level mean heatmap", "path": heatmap_plot_path},
            "logged_tension": {
                "title": f"Fresh simulated {TRACE_PLANT_ID} logged-tension traces",
                "path": trace_plot_path,
            },
        },
        "plot_path": tlog_plot_path,
        "calculation_summary": (
            "All dashboard MARE values are recomputed from ten-plant simulations "
            "and paper Eq. (8) weighted one-step PEM. Figure 5 means, Figure 6 means, and "
            "Figure S10 medians are reported separately instead of being mixed."
        ),
        "calculations": calculations,
    }
    payload["summary_path"] = _write_summary("noiseLpf_noise_aware_logging_summary.json", payload)
    return payload

