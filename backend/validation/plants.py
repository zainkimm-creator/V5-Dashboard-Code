"""Plant presets, built from the paper's authoritative ten-plant table.

Every physical parameter the simulator uses (EA, span lengths, roller radii,
inertias, viscous friction, nominal line speed, set-point and full-scale
tension) is read from `data/model_inputs/ten_plant_parameters.csv`, the file
that ships with the paper reply package. It is extracted deterministically from
the frozen `plant_pool_v3.json` pool plus `run_full_sweep.SELECTED`, so it -- not
this module -- is the source of truth.

The dashboard's display ids ``P01``..``P10`` map onto the pool ids in file
order (``P001``, ``P049``, ``P053``, ``P060``, ``P139``, ``P158``, ``P163``,
``P177``, ``P186``, ``P189``).

Two values are *not* in the CSV and stay transcribed here:

- ``overshoot_percent`` -- a reported step-response metric, not a plant
  parameter.
- ``zeta_cl_min`` -- kept at the precision the paper prints, because the
  closed-loop damping section compares against the printed figure. The CSV
  carries the same quantity at full precision and is checked against it on
  load, so a real disagreement fails loudly instead of drifting.

The CSV also carries derived quantities the dashboard recomputes rather than
reads (``omega_n``, ``K_vel = 1.4 J omega_n``, ``omega_ss``, ``u_ss``,
``T_I_s``); those are verified against the file by
`tests/test_paper_inputs.py` so the controller stays on the campaign default.
"""

from __future__ import annotations

import math
from typing import Any

from backend.models.equations import R2RParameters

from .paper_inputs import load_ten_plant_parameters

DEFAULT_PLANT_ID = "P01"

# Step-response overshoot as printed for each plant, and zeta_CL,min at the
# printed precision. Keyed by pool id so the mapping survives a reordering of
# the ten-plant subset.
_PAPER_PRINTED_METRICS: dict[str, dict[str, float]] = {
    "P001": {"zeta_cl_min": 0.151, "overshoot_percent": 45.8},
    "P049": {"zeta_cl_min": 0.237, "overshoot_percent": 45.3},
    "P053": {"zeta_cl_min": 0.35, "overshoot_percent": 28.8},
    "P060": {"zeta_cl_min": 0.201, "overshoot_percent": 41.0},
    "P139": {"zeta_cl_min": 0.144, "overshoot_percent": 50.1},
    "P158": {"zeta_cl_min": 0.515, "overshoot_percent": 9.8},
    "P163": {"zeta_cl_min": 0.357, "overshoot_percent": 31.8},
    "P177": {"zeta_cl_min": 0.179, "overshoot_percent": 55.3},
    "P186": {"zeta_cl_min": 0.53, "overshoot_percent": 16.1},
    "P189": {"zeta_cl_min": 0.21, "overshoot_percent": 51.9},
}

# Tolerance on the printed-vs-CSV zeta_CL,min check: the printed values carry
# three decimals, so half an ulp of that is the most they may disagree by.
_ZETA_PRINT_TOLERANCE = 5e-4


def _build_plant_details() -> dict[str, dict[str, Any]]:
    """Return the display-id keyed plant table built from the paper CSV."""

    rows = load_ten_plant_parameters()
    details: dict[str, dict[str, Any]] = {}
    for index, (pool_id, row) in enumerate(rows.items(), start=1):
        display_id = f"P{index:02d}"
        printed = _PAPER_PRINTED_METRICS.get(pool_id)
        if printed is None:
            raise KeyError(
                f"plant {pool_id} has no printed zeta_CL,min / overshoot entry; the "
                "ten-plant subset changed and the transcribed metrics must follow"
            )
        csv_zeta = float(row["zeta_CL_min"])
        if not math.isclose(
            csv_zeta, printed["zeta_cl_min"], rel_tol=0.0, abs_tol=_ZETA_PRINT_TOLERANCE
        ):
            raise ValueError(
                f"plant {pool_id}: zeta_CL,min {csv_zeta} in ten_plant_parameters.csv "
                f"disagrees with the printed {printed['zeta_cl_min']}"
            )
        details[display_id] = {
            "pool_id": pool_id,
            "material": str(row["material"]),
            "scale": str(row["scale"]),
            "regime": str(row["regime_class"]),
            "EA_N": float(row["EA_N"]),
            "v_ref_m_s": float(row["v0_mps"]),
            "v_max_m_s": float(row["v_max_mps"]),
            "T_ref_N": float(row["T_ref_N"]),
            "T_max_N": float(row["T_max_N"]),
            "zeta_cl_min": printed["zeta_cl_min"],
            "zeta_cl_min_full_precision": csv_zeta,
            "overshoot_percent": printed["overshoot_percent"],
            "roller_radius_m": (
                float(row["R_UW_m"]),
                float(row["R_Nip_m"]),
                float(row["R_RW_m"]),
            ),
            "inertia_kg_m2": (
                float(row["J_UW_kgm2"]),
                float(row["J_Nip_kgm2"]),
                float(row["J_RW_kgm2"]),
            ),
            "viscous_friction": (
                float(row["f_UW_Nms_per_rad"]),
                float(row["f_Nip_Nms_per_rad"]),
                float(row["f_RW_Nms_per_rad"]),
            ),
            "span_length_m": (
                float(row["L1_m"]),
                float(row["L2_m"]),
                float(row["L3_m"]),
            ),
            # Campaign-default derived quantities, carried for cross-checks and
            # for reporting; the controller recomputes them from the physics.
            "omega_n_rad_s": (
                float(row["omega_n_UW_rad_s"]),
                float(row["omega_n_Nip_rad_s"]),
                float(row["omega_n_RW_rad_s"]),
            ),
            "K_vel_reference": (
                float(row["K_vel_UW"]),
                float(row["K_vel_Nip"]),
                float(row["K_vel_RW"]),
            ),
            "T_I_reference_s": float(row["T_I_s"]),
        }
    return details


PROFESSOR_PLANT_DETAILS: dict[str, dict[str, Any]] = _build_plant_details()


def _plant_rows() -> list[dict[str, Any]]:
    return [
        {"plant_id": plant_id, "plant": plant_id, **details}
        for plant_id, details in PROFESSOR_PLANT_DETAILS.items()
    ]


def _recommended_excitation_amplitude(t_ref_n: float) -> float:
    return round(0.2 * float(t_ref_n), 6)


def plant_registry() -> list[dict[str, Any]]:
    """Return display-ready plant metadata from the paper's ten-plant table."""

    plants = []
    for row in _plant_rows():
        plant_id = str(row["plant_id"])
        ea_n = float(row["EA_N"])
        t_ref_n = float(row["T_ref_N"])
        t_max_n = float(row["T_max_N"])
        plants.append(
            {
                "plant": plant_id,
                "plant_id": plant_id,
                "pool_id": row["pool_id"],
                "label": f"{plant_id} | {row['material']} {row['scale']} | EA={ea_n:g} N",
                "EA_N": ea_n,
                "v_ref_m_s": float(row["v_ref_m_s"]),
                "v_max_m_s": float(row["v_max_m_s"]),
                "T_ref_N": t_ref_n,
                "T_max_N": t_max_n,
                "sensor_noise_sigma_N": round(0.003 * t_max_n, 6),
                "material": row["material"],
                "scale": row["scale"],
                "regime": row["regime"],
                "zeta_cl_min": float(row["zeta_cl_min"]),
                "overshoot_percent": float(row["overshoot_percent"]),
                "recommended_excitation_amplitude_V": _recommended_excitation_amplitude(t_ref_n),
                "baseline_range_compatible": True,
                "roller_radius_m": list(row["roller_radius_m"]),
                "span_length_m": list(row["span_length_m"]),
                "inertia_kg_m2": list(row["inertia_kg_m2"]),
                "viscous_friction": list(row["viscous_friction"]),
                "omega_n_rad_s": list(row["omega_n_rad_s"]),
                "K_vel_reference": list(row["K_vel_reference"]),
                "T_I_reference_s": float(row["T_I_reference_s"]),
                "process_noise_b": 0.0,
                "parameter_source": "data/model_inputs/ten_plant_parameters.csv",
                "simulation_note": "Paper ten-plant table supplies EA, v0, T_ref, T_max, R, J, f, and L.",
            }
        )
    return plants


def get_plant(plant_id: str | None = None) -> dict[str, Any]:
    """Return one plant of the fixed ten-plant subset by id."""

    selected_id = (plant_id or DEFAULT_PLANT_ID).strip()
    for plant in plant_registry():
        if plant["plant_id"] == selected_id:
            return plant
    valid = ", ".join(plant["plant_id"] for plant in plant_registry())
    raise ValueError(f"Unknown plant_id '{selected_id}'. Valid plants: {valid}.")


def parameters_for_plant(plant_id: str | None = None) -> tuple[R2RParameters, dict[str, Any]]:
    """Return model parameters with full paper-resolved plant details applied."""

    plant = get_plant(plant_id)
    params = R2RParameters(
        span_length_m=tuple(plant["span_length_m"]),
        roller_radius_m=tuple(plant["roller_radius_m"]),
        inertia_kg_m2=tuple(plant["inertia_kg_m2"]),
        tension_ref_N=(float(plant["T_ref_N"]),) * 3,
        process_noise_b=float(plant["process_noise_b"]),
        kf_UW=float(plant["viscous_friction"][0]),
        kf_Nip=float(plant["viscous_friction"][1]),
        kf_RW=float(plant["viscous_friction"][2]),
        EA=float(plant["EA_N"]),
        feeder_velocity_m_s=float(plant["v_ref_m_s"]),
    )
    return params, {
        **plant,
        "applied_parameters": {
            "EA_N": params.EA,
            "v_ref_m_s": params.feeder_velocity_m_s,
            "T_ref_N": plant["T_ref_N"],
            "T_max_N": plant["T_max_N"],
            "sensor_noise_sigma_N": plant["sensor_noise_sigma_N"],
            "roller_radius_m": list(params.roller_radius_m),
            "span_length_m": list(params.span_length_m),
            "inertia_kg_m2": list(params.inertia_kg_m2),
            "viscous_friction": list(params.kf),
            "process_noise_b": params.process_noise_b,
        },
    }
