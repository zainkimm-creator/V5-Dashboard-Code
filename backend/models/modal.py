"""Closed-loop modal analysis for the deployed three-span cascade.

The paper supplement defines modal damping as
``zeta_i = -Re(lambda_i) / abs(lambda_i)``.  Its ``A - B Kp C`` relation is a
single-loop illustration, so this module linearizes the active nonlinear plant
plus the complete outer-tension/inner-velocity cascade instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .controller import CascadePIController, ControllerConfig
from .equations import (
    R2RParameters,
    derivatives,
    steady_surface_velocities,
    web_torques,
)


CASCADE_STATE_NAMES = (
    "T1",
    "T2",
    "T3",
    "omega_UW",
    "omega_Nip",
    "omega_RW",
    "outer_integral_1",
    "outer_integral_2",
    "outer_integral_3",
)
DEFAULT_JACOBIAN_RELATIVE_STEP = 1e-6
JACOBIAN_SENSITIVITY_STEPS = (1e-4, 1e-5, 1e-6, 1e-7)


"""Regime thresholds on ``zeta_CL,min``, quoted verbatim from the paper.

Main text Section 2.2 (v5, p. 5): "P158 and P186 are highly damped (H-Damp,
zeta_CL,min >= 0.5); P001, P049, P060, P139, P177, and P189 exhibit observable
underdamped modes (O-UD, zeta_CL,min < 0.3); and P053 and P163 fall in the
hybrid-oscillatory regime (H-Osc, 0.3 <= zeta_CL,min < 0.5)."  The same
thresholds are carried in ``closed_loop_damping_reference.json``
(``damping_groups.thresholds``).
"""
O_UD_UPPER_ZETA = 0.3
H_DAMP_LOWER_ZETA = 0.5


def damping_regime(zeta_cl_min: float) -> str:
    """Return the paper's three-regime label for a damping ratio."""

    value = float(zeta_cl_min)
    if not math.isfinite(value):
        raise ValueError("zeta_cl_min must be finite")
    if value < O_UD_UPPER_ZETA:
        return "O-UD"
    if value < H_DAMP_LOWER_ZETA:
        return "H-Osc"
    return "H-Damp"


def _nominal_augmented_state(
    params: R2RParameters,
    controller_config: ControllerConfig,
    line_speed_m_s: float,
) -> np.ndarray:
    target = tuple(float(value) for value in controller_config.target_tension_N)
    surface_velocity = steady_surface_velocities(params, line_speed_m_s, target)
    omega = tuple(
        velocity / radius
        for velocity, radius in zip(
            surface_velocity, params.roller_radius_m, strict=True
        )
    )
    return np.asarray(target + omega + (0.0, 0.0, 0.0), dtype=float)


def _cascade_vector_field(
    augmented_state: Sequence[float],
    params: R2RParameters,
    controller_config: ControllerConfig,
    line_speed_m_s: float,
) -> np.ndarray:
    y = np.asarray(augmented_state, dtype=float)
    if y.shape != (9,) or not np.all(np.isfinite(y)):
        raise ValueError("augmented cascade state must contain nine finite values")
    controller = CascadePIController(controller_config)
    controller.tension_integral_N_s = [float(value) for value in y[6:9]]
    action = controller.update(
        y[:6],
        0.0,
        params,
        target_tension_N=controller_config.target_tension_N,
        line_speed_m_s=line_speed_m_s,
    )
    target = np.asarray(controller_config.target_tension_N, dtype=float)
    polarity = np.asarray((-1.0, 1.0, 1.0), dtype=float)
    integral_derivative = polarity * (target - y[:3])
    return np.concatenate(
        (
            np.asarray(derivatives(y[:6], action.inputs_V, params), dtype=float),
            integral_derivative,
        )
    )


def _central_jacobian(
    function,
    point: np.ndarray,
    relative_step: float,
) -> np.ndarray:
    if relative_step <= 0.0 or not math.isfinite(relative_step):
        raise ValueError("relative_step must be finite and positive")
    columns: list[np.ndarray] = []
    for index in range(point.size):
        step = relative_step * max(1.0, abs(float(point[index])))
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((function(plus) - function(minus)) / (2.0 * step))
    return np.column_stack(columns)


def closed_loop_modal_analysis(
    params: R2RParameters,
    controller_config: ControllerConfig,
    *,
    line_speed_m_s: float | None = None,
    relative_step: float = DEFAULT_JACOBIAN_RELATIVE_STEP,
) -> dict[str, object]:
    """Calculate continuous full-cascade eigenvalues and damping ratios.

    The nine-state Jacobian includes the six physical states and the three
    outer-loop PI integrals.  Paper/reference damping values are not accepted as
    inputs and cannot affect the calculated result.
    """

    line_speed = float(
        params.feeder_velocity_m_s
        if line_speed_m_s is None
        else line_speed_m_s
    )
    equilibrium = _nominal_augmented_state(params, controller_config, line_speed)
    vector_field = lambda state: _cascade_vector_field(
        state, params, controller_config, line_speed
    )
    equilibrium_residual = vector_field(equilibrium)
    jacobian = _central_jacobian(vector_field, equilibrium, relative_step)
    eigenvalues = np.linalg.eigvals(jacobian)
    magnitudes = np.abs(eigenvalues)
    valid = magnitudes > 1e-9
    damping_ratios = -np.real(eigenvalues) / np.maximum(magnitudes, 1e-30)
    if not np.any(valid):
        raise ValueError("closed-loop Jacobian has no nonzero eigenvalues")
    minimum_damping = float(np.min(damping_ratios[valid]))
    modes = [
        {
            "real_per_s": float(value.real),
            "imag_per_s": float(value.imag),
            "magnitude_per_s": float(abs(value)),
            "damping_ratio": float(zeta),
            "stable": bool(value.real < 0.0),
            "oscillatory": bool(abs(value.imag) > 1e-8),
        }
        for value, zeta in sorted(
            zip(eigenvalues, damping_ratios, strict=True),
            key=lambda item: (item[0].real, abs(item[0].imag)),
            reverse=True,
        )
    ]
    return {
        "method": "continuous_full_cascade_central_difference_jacobian",
        "state_names": list(CASCADE_STATE_NAMES),
        "state_count": len(CASCADE_STATE_NAMES),
        "relative_step": float(relative_step),
        "equilibrium_residual_max_abs": float(
            np.max(np.abs(equilibrium_residual))
        ),
        "zeta_cl_min": minimum_damping,
        "regime": damping_regime(minimum_damping),
        "spectral_abscissa_per_s": float(np.max(np.real(eigenvalues))),
        "stable": bool(np.all(np.real(eigenvalues) < 0.0)),
        "unstable_mode_count": int(np.sum(np.real(eigenvalues) >= 0.0)),
        "oscillatory_pair_count": int(np.sum(np.imag(eigenvalues) > 1e-8)),
        "modes": modes,
    }


def open_loop_modal_analysis(
    params: R2RParameters,
    *,
    line_speed_m_s: float | None = None,
    relative_step: float = DEFAULT_JACOBIAN_RELATIVE_STEP,
) -> dict[str, object]:
    """Calculate the six-state OPEN-LOOP plant eigenvalues and ``tau_min``.

    The paper defines the logging-adequacy time scale as

        tau_min = 1 / max_i |Re(lambda_i)|

    and calls it "the fastest **open-loop** modal time scale" (v5 abstract and
    Contribution C1; the formula is printed in Section 3.1, p. 8, where the
    range is quoted as 7.5 ms for P158 to 67.4 ms for P139).  It is therefore a
    property of the plant Jacobian alone -- the controller, its gains and the
    three PI integrator states must NOT enter it.  Using the nine-state
    closed-loop cascade Jacobian instead inflates tau_min by roughly 1.5-1.8x
    and moves the maximum onto the wrong plant.
    """

    line_speed = float(
        params.feeder_velocity_m_s if line_speed_m_s is None else line_speed_m_s
    )
    target = tuple(float(value) for value in params.tension_ref_N)
    surface_velocity = steady_surface_velocities(params, line_speed, target)
    omega = tuple(
        velocity / radius
        for velocity, radius in zip(
            surface_velocity, params.roller_radius_m, strict=True
        )
    )
    equilibrium = np.asarray(target + omega, dtype=float)
    steady_torque = web_torques(tuple(equilibrium), params)
    steady_input = tuple(
        params.kf[index] * omega[index] - steady_torque[index] for index in range(3)
    )
    vector_field = lambda state: np.asarray(
        derivatives(tuple(state), steady_input, params), dtype=float
    )
    equilibrium_residual = vector_field(equilibrium)
    jacobian = _central_jacobian(vector_field, equilibrium, relative_step)
    eigenvalues = np.linalg.eigvals(jacobian)
    fastest_rate = float(np.max(np.abs(np.real(eigenvalues))))
    if fastest_rate <= 0.0:
        raise ValueError("open-loop Jacobian has no decaying mode")
    return {
        "method": "continuous_six_state_open_loop_central_difference_jacobian",
        "state_names": list(CASCADE_STATE_NAMES[:6]),
        "relative_step": float(relative_step),
        "equilibrium_residual_max_abs": float(np.max(np.abs(equilibrium_residual))),
        "fastest_decay_rate_per_s": fastest_rate,
        "tau_min_s": 1.0 / fastest_rate,
        "tau_min_ms": 1000.0 / fastest_rate,
        "spectral_abscissa_per_s": float(np.max(np.real(eigenvalues))),
        "stable": bool(np.all(np.real(eigenvalues) < 0.0)),
        "formula": "tau_min = 1 / max_i |Re(lambda_i)| on the open-loop plant Jacobian",
        "source": "paper1_isa_v5 Section 3.1 (p. 8) and Contribution C1",
    }


def open_loop_tau_min_s(
    params: R2RParameters,
    *,
    line_speed_m_s: float | None = None,
) -> float:
    """Return the paper's ``tau_min`` in seconds for one plant."""

    return float(
        open_loop_modal_analysis(params, line_speed_m_s=line_speed_m_s)["tau_min_s"]
    )


def closed_loop_modal_sensitivity(
    params: R2RParameters,
    controller_config: ControllerConfig,
    *,
    line_speed_m_s: float | None = None,
    relative_steps: Sequence[float] = JACOBIAN_SENSITIVITY_STEPS,
) -> dict[str, object]:
    """Repeat the modal calculation across numerical Jacobian step sizes."""

    analyses = [
        closed_loop_modal_analysis(
            params,
            controller_config,
            line_speed_m_s=line_speed_m_s,
            relative_step=float(step),
        )
        for step in relative_steps
    ]
    damping_values = [float(row["zeta_cl_min"]) for row in analyses]
    selected_index = min(
        range(len(analyses)),
        key=lambda index: abs(
            float(analyses[index]["relative_step"])
            - DEFAULT_JACOBIAN_RELATIVE_STEP
        ),
    )
    return {
        "selected": analyses[selected_index],
        "iterations": [
            {
                "relative_step": float(row["relative_step"]),
                "zeta_cl_min": float(row["zeta_cl_min"]),
                "spectral_abscissa_per_s": float(
                    row["spectral_abscissa_per_s"]
                ),
                "stable": bool(row["stable"]),
            }
            for row in analyses
        ],
        "zeta_step_sensitivity_spread": max(damping_values)
        - min(damping_values),
    }
