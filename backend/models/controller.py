"""Cascade PI plus feedforward controller for the R2R model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .equations import R2RParameters, steady_surface_velocities, validate_vector, web_torques


@dataclass(frozen=True)
class ControllerConfig:
    """Controller gains and setpoints with explicit units."""

    target_tension_N: tuple[float, float, float] = (42.0, 44.0, 43.0)
    line_speed_m_s: float = 1.0
    Kp_star_m_s_per_N: float = 100.0
    TI_s: float = 2.00
    velocity_Kp_Nm_per_rad_s: float = 0.0
    velocity_TI_s: float = 0.20
    integral_limit_N_s: float = 200.0
    velocity_correction_limit_fraction: float | None = 0.20
    feedforward_enabled: bool = True
    feedforward_uses_measured_omega: bool = False
    high_ea_kp_cap_enabled: bool = True
    paper_velocity_gain_enabled: bool = False
    velocity_gain_alpha: float = 1.4
    steady_velocity_uses_dynamic_target: bool = True
    # Compute each roller's steady-state velocity from the MEASURED upstream
    # tension instead of its commanded one.
    #
    # v_Nip = v_f (EA - T1)/(EA - T2) and v_RW = v_Nip (EA - T2)/(EA - T3): every
    # roller's equilibrium speed depends on the span above it. Using the target
    # for that upstream term means that while T1 is still travelling to its new
    # setpoint the nip is held at the wrong speed, so span 2 and span 3 are
    # disturbed by a step they were never commanded. Feeding the measured value
    # back is the "measured-tension feedforward" the paper credits with
    # compensating inter-span coupling in real time (Sections 2.3, 3.3, 4.1).
    #
    # Default off: every existing study was calibrated against the target-based
    # form, and this changes closed-loop behaviour.
    steady_velocity_uses_measured_upstream: bool = False

    def __post_init__(self) -> None:
        if len(self.target_tension_N) != 3:
            raise ValueError("target_tension_N must contain exactly 3 values")
        if self.TI_s <= 0 or self.velocity_TI_s <= 0:
            raise ValueError("PI integral times must be positive")
        if self.integral_limit_N_s <= 0:
            raise ValueError("integral_limit_N_s must be positive")
        if self.velocity_gain_alpha <= 0:
            raise ValueError("velocity_gain_alpha must be positive")
        if (
            self.velocity_correction_limit_fraction is not None
            and self.velocity_correction_limit_fraction <= 0
        ):
            raise ValueError("velocity_correction_limit_fraction must be positive or None")


def auto_tension_integral_time_s(
    params: R2RParameters,
    line_speed_m_s: float | None = None,
    *,
    settle_time_s: float = 5.0,
) -> float:
    """Return the professor-review per-plant ``auto_Ti`` setting.

    ``T_I = mean(T_ref) * settle_time / mean(abs(u_ss))`` with a 0.1 s floor,
    where ``u_ss`` is the direct motor torque required at the nominal operating
    point.  This is a controller setting; it does not use any paper result data.
    """

    if settle_time_s <= 0:
        raise ValueError("settle_time_s must be positive")
    line_speed = float(
        params.feeder_velocity_m_s if line_speed_m_s is None else line_speed_m_s
    )
    steady_v = steady_surface_velocities(params, line_speed, params.tension_ref_N)
    omega_ss = tuple(
        velocity / radius
        for velocity, radius in zip(steady_v, params.roller_radius_m, strict=True)
    )
    steady_state = params.tension_ref_N + omega_ss
    tau_web = web_torques(steady_state, params)
    u_ss = tuple(
        params.kf[index] * omega_ss[index] - tau_web[index]
        for index in range(3)
    )
    mean_tension = sum(params.tension_ref_N) / 3.0
    mean_abs_torque = sum(abs(value) for value in u_ss) / 3.0
    return max(0.1, mean_tension * settle_time_s / max(mean_abs_torque, 1e-12))


@dataclass
class ControlAction:
    """Controller output and diagnostics for one sample."""

    inputs_V: tuple[float, float, float]
    velocity_ref_rad_s: tuple[float, float, float]
    tension_error_N: tuple[float, float, float]
    velocity_error_rad_s: tuple[float, float, float]
    feedforward_torque_Nm: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "inputs_V": list(self.inputs_V),
            "inputs_Nm": list(self.inputs_V),
            "velocity_ref_rad_s": list(self.velocity_ref_rad_s),
            "tension_error_N": list(self.tension_error_N),
            "velocity_error_rad_s": list(self.velocity_error_rad_s),
            "feedforward_torque_Nm": list(self.feedforward_torque_Nm),
        }


@dataclass
class CascadePIController:
    """Cascade PI controller with outer tension loop and inner velocity loop."""

    config: ControllerConfig = field(default_factory=ControllerConfig)
    tension_integral_N_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_integral_rad: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def reset(self) -> None:
        self.tension_integral_N_s = [0.0, 0.0, 0.0]
        self.velocity_integral_rad = [0.0, 0.0, 0.0]

    def update(
        self,
        state: Sequence[float],
        dt_s: float,
        params: R2RParameters | None = None,
        target_tension_N: Sequence[float] | None = None,
        line_speed_m_s: float | None = None,
    ) -> ControlAction:
        """Compute held motor-torque inputs for the next controller interval."""

        active_params = params or R2RParameters()
        x = validate_vector(state, 6, "state")
        measured_tension = x[:3]
        measured_omega = x[3:]
        target = validate_vector(
            target_tension_N or self.config.target_tension_N,
            3,
            "target_tension_N",
        )
        line_speed = float(line_speed_m_s if line_speed_m_s is not None else self.config.line_speed_m_s)
        tension_error = tuple(
            target[i] - measured_tension[i] for i in range(3)
        )
        polarity = (-1.0, 1.0, 1.0)
        for i in range(3):
            self.tension_integral_N_s[i] += polarity[i] * tension_error[i] * dt_s
            self.tension_integral_N_s[i] = max(
                -self.config.integral_limit_N_s,
                min(self.config.integral_limit_N_s, self.tension_integral_N_s[i]),
            )

        pi_out = tuple(
            polarity[i] * tension_error[i] + self.tension_integral_N_s[i] / self.config.TI_s
            for i in range(3)
        )
        l_eff = active_params.span_length_m
        effective_kp_star = self.config.Kp_star_m_s_per_N
        if self.config.high_ea_kp_cap_enabled:
            if active_params.EA >= 400000.0:
                effective_kp_star = min(effective_kp_star, 5.0)
            elif active_params.EA >= 300000.0:
                effective_kp_star = min(effective_kp_star, 20.0)
        correction = tuple(
            (l_eff[i] / active_params.EA) * effective_kp_star * pi_out[i]
            for i in range(3)
        )
        if self.config.velocity_correction_limit_fraction is not None:
            correction_limit = max(
                1e-6,
                self.config.velocity_correction_limit_fraction * max(line_speed, 1e-6),
            )
            correction = tuple(
                max(-correction_limit, min(correction_limit, value))
                for value in correction
            )

        steady_target = (
            target
            if self.config.steady_velocity_uses_dynamic_target
            else self.config.target_tension_N
        )
        if self.config.steady_velocity_uses_measured_upstream:
            # Own span keeps its commanded value (that is the thing being
            # driven); the upstream span uses what the sensor actually reads.
            ea = active_params.EA
            v_feed = float(line_speed)
            v_uw = v_feed * (1.0 - steady_target[0] / ea)
            v_nip = v_feed * (ea - measured_tension[0]) / max(ea - steady_target[1], 1e-9)
            v_rw = v_nip * (ea - measured_tension[1]) / max(ea - steady_target[2], 1e-9)
            steady_v = (v_uw, v_nip, v_rw)
        else:
            steady_v = steady_surface_velocities(active_params, line_speed, steady_target)
        v_ref_m_s = tuple(steady_v[i] + correction[i] for i in range(3))
        velocity_ref = tuple(
            v_ref_m_s[i] / active_params.roller_radius_m[i] for i in range(3)
        )
        velocity_error = tuple(velocity_ref[i] - measured_omega[i] for i in range(3))

        tau_web = web_torques(x, active_params)
        feedforward_omega = measured_omega if self.config.feedforward_uses_measured_omega else velocity_ref
        feedforward_torque = tuple(
            active_params.kf[i] * feedforward_omega[i] - tau_web[i]
            if self.config.feedforward_enabled
            else 0.0
            for i in range(3)
        )
        if self.config.velocity_Kp_Nm_per_rad_s > 0.0:
            velocity_gains = (self.config.velocity_Kp_Nm_per_rad_s,) * 3
        elif self.config.paper_velocity_gain_enabled:
            velocity_gains = tuple(
                self.config.velocity_gain_alpha
                * active_params.inertia_kg_m2[i]
                * (
                    active_params.EA
                    * active_params.roller_radius_m[i]
                    * active_params.roller_radius_m[i]
                    / (active_params.inertia_kg_m2[i] * active_params.span_length_m[i])
                )
                ** 0.5
                for i in range(3)
            )
        else:
            auto_gain = min(16.0, max(1.35, 0.105 * (sum(target) / 3.0)))
            velocity_gain = max(32.0, auto_gain) if active_params.EA >= 300000.0 else auto_gain
            velocity_gains = (velocity_gain,) * 3
        torque_cmd = tuple(
            velocity_gains[i] * velocity_error[i]
            + feedforward_torque[i]
            for i in range(3)
        )
        return ControlAction(
            inputs_V=torque_cmd,
            velocity_ref_rad_s=velocity_ref,
            tension_error_N=tension_error,
            velocity_error_rad_s=velocity_error,
            feedforward_torque_Nm=feedforward_torque,
        )
