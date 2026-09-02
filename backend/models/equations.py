"""R2R governing equations with explicit state, input, parameter, and unit names.

State vector:
    x = [T1, T2, T3, omega_UW, omega_Nip, omega_RW]

Input vector:
    u = [u_UW, u_Nip, u_RW]

Units:
    Tension T_i: N
    Angular velocity omega_i: rad/s
    Roller command input u_i: torque command in N*m
    Web speed v_i = R_i * omega_i: m/s
    Elastic modulus-area product EA: N
    SysID ratio kt_i = R_i^2/J_i: 1/kg
    SysID ratio kf_i = f_i/J_i: 1/s
    Viscous friction f_i: N*m*s/rad
    Roller inertia J_i: kg*m^2
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Iterable, Mapping, Sequence

STATE_NAMES = ("T1", "T2", "T3", "omega_UW", "omega_Nip", "omega_RW")
INPUT_NAMES = ("u_UW", "u_Nip", "u_RW")
PARAMETER_NAMES = ("kt_UW", "kt_Nip", "kt_RW", "kf_UW", "kf_Nip", "kf_RW", "EA")
ROLLER_NAMES = ("UW", "Nip", "RW")


def _tuple3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    result = tuple(float(v) for v in values)
    if not all(isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class R2RParameters:
    """Physical parameters for the paper-based R2R validation model."""

    span_length_m: tuple[float, float, float] = (0.80, 0.80, 1.60)
    roller_radius_m: tuple[float, float, float] = (0.050, 0.050, 0.050)
    inertia_kg_m2: tuple[float, float, float] = (0.075, 0.055, 0.090)
    tension_ref_N: tuple[float, float, float] = (42.0, 44.0, 43.0)
    tension_relaxation_hz: float = 0.05
    process_noise_b: float = 0.0
    kt_UW: float = 1.0
    kt_Nip: float = 1.0
    kt_RW: float = 1.0
    kf_UW: float = 0.010
    kf_Nip: float = 0.012
    kf_RW: float = 0.011
    EA: float = 4200.0
    feeder_velocity_m_s: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "span_length_m", _tuple3(self.span_length_m, "span_length_m"))
        object.__setattr__(self, "roller_radius_m", _tuple3(self.roller_radius_m, "roller_radius_m"))
        object.__setattr__(self, "inertia_kg_m2", _tuple3(self.inertia_kg_m2, "inertia_kg_m2"))
        object.__setattr__(self, "tension_ref_N", _tuple3(self.tension_ref_N, "tension_ref_N"))
        positive_fields = {
            "span_length_m": self.span_length_m,
            "roller_radius_m": self.roller_radius_m,
            "inertia_kg_m2": self.inertia_kg_m2,
            "kt_UW": (self.kt_UW,),
            "kt_Nip": (self.kt_Nip,),
            "kt_RW": (self.kt_RW,),
            "kf_UW": (self.kf_UW,),
            "kf_Nip": (self.kf_Nip,),
            "kf_RW": (self.kf_RW,),
            "EA": (self.EA,),
            "feeder_velocity_m_s": (self.feeder_velocity_m_s,),
        }
        for field_name, values in positive_fields.items():
            if any(v <= 0 or not isfinite(v) for v in values):
                raise ValueError(f"{field_name} values must be finite and positive")
        if self.tension_relaxation_hz < 0 or not isfinite(self.tension_relaxation_hz):
            raise ValueError("tension_relaxation_hz must be finite and non-negative")
        if self.process_noise_b < 0 or not isfinite(self.process_noise_b):
            raise ValueError("process_noise_b must be finite and non-negative")

    @property
    def kt(self) -> tuple[float, float, float]:
        return (self.kt_UW, self.kt_Nip, self.kt_RW)

    @property
    def kf(self) -> tuple[float, float, float]:
        return (self.kf_UW, self.kf_Nip, self.kf_RW)

    @property
    def paper_kt(self) -> tuple[float, float, float]:
        """Paper Eq. (6) ratio parameters `k_t,i = R_i^2/J_i`."""

        return tuple(
            radius * radius / inertia
            for radius, inertia in zip(self.roller_radius_m, self.inertia_kg_m2, strict=True)
        )

    @property
    def paper_kf(self) -> tuple[float, float, float]:
        """Paper Eq. (6) ratio parameters `k_f,i = f_i/J_i`."""

        return tuple(
            friction / inertia for friction, inertia in zip(self.kf, self.inertia_kg_m2, strict=True)
        )

    def sysid_values(self) -> dict[str, float]:
        kt_values = self.paper_kt
        kf_values = self.paper_kf
        return {
            "kt_UW": kt_values[0],
            "kt_Nip": kt_values[1],
            "kt_RW": kt_values[2],
            "kf_UW": kf_values[0],
            "kf_Nip": kf_values[1],
            "kf_RW": kf_values[2],
            "EA": float(self.EA),
        }

    def with_sysid_values(self, values: Mapping[str, float]) -> "R2RParameters":
        allowed = {name: float(value) for name, value in values.items() if name in PARAMETER_NAMES}
        return replace(self, **allowed)

    def with_drift(
        self,
        *,
        EA_scale: float = 1.0,
        friction_scale: float = 1.0,
        inertia_scale: float = 1.0,
    ) -> "R2RParameters":
        return replace(
            self,
            EA=self.EA * EA_scale,
            kf_UW=self.kf_UW * friction_scale,
            kf_Nip=self.kf_Nip * friction_scale,
            kf_RW=self.kf_RW * friction_scale,
            inertia_kg_m2=tuple(j * inertia_scale for j in self.inertia_kg_m2),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_vector(values: Sequence[float], expected: int, name: str) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} values")
    result = tuple(float(v) for v in values)
    if not all(isfinite(v) for v in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def velocities(state: Sequence[float], params: R2RParameters) -> tuple[float, float, float]:
    """Return roller surface velocities `(v_UW, v_Nip, v_RW)` in m/s."""

    x = validate_vector(state, 6, "state")
    _, _, _, omega_uw, omega_nip, omega_rw = x
    return (
        params.roller_radius_m[0] * omega_uw,
        params.roller_radius_m[1] * omega_nip,
        params.roller_radius_m[2] * omega_rw,
    )


def steady_surface_velocities(
    params: R2RParameters,
    line_speed_m_s: float | None = None,
    tensions: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """Return steady roller surface speeds for the v4.1 span equations."""

    v_feeder = float(line_speed_m_s if line_speed_m_s is not None else params.feeder_velocity_m_s)
    t1, t2, t3 = validate_vector(tensions or params.tension_ref_N, 3, "tensions")
    ea = params.EA
    v_uw = v_feeder * (1.0 - t1 / ea)
    v_nip = v_feeder * (ea - t1) / max(ea - t2, 1e-9)
    v_rw = v_nip * (ea - t2) / max(ea - t3, 1e-9)
    return (v_uw, v_nip, v_rw)


def tension_state_pairs(state: Sequence[float]) -> tuple[tuple[float, float], ...]:
    """Return convective tension pairs for the v4.1 topology.

    The three state tensions are the UW-to-feeder, feeder-to-nip, and
    nip-to-RW spans. The velocity-master feeder sits between the first and
    second spans.
    """

    t1, t2, t3, *_ = validate_vector(state, 6, "state")
    return ((0.0, t1), (t1, t2), (t2, t3))


def span_velocity_pairs(
    state: Sequence[float], params: R2RParameters
) -> tuple[tuple[float, float], ...]:
    """Return `(v_previous, v_current)` pairs for the v4.1 span chain."""

    v_uw, v_nip, v_rw = velocities(state, params)
    v_feeder = params.feeder_velocity_m_s
    return ((v_uw, v_feeder), (v_feeder, v_nip), (v_nip, v_rw))


def roller_tension_differences(state: Sequence[float]) -> tuple[float, float, float]:
    """Return the v4.1 equivalent tension force term for each roller.

    These are the tension terms in the surface-speed equation before multiplying
    by `R_i^2/J_i`: UW is braked by the first span, Nip is assisted by the
    feeder-to-nip span and braked by the nip-to-RW span, and RW is assisted by
    the final span. There is no live downstream fourth tension state.
    """

    t1, t2, t3, *_ = validate_vector(state, 6, "state")
    return (-t1, t2 - t3, t3)


def tension_derivatives(state: Sequence[float], params: R2RParameters) -> tuple[float, float, float]:
    """Return `[dT1/dt, dT2/dt, dT3/dt]` in N/s using paper Eq. (1).

    Topology:
        UW -- T1 -- feeder(v_ref) -- T2 -- Nip -- T3 -- RW
    """

    x = validate_vector(state, 6, "state")
    t1, t2, t3, *_ = x
    v_uw, v_nip, v_rw = velocities(x, params)
    v_feeder = params.feeder_velocity_m_s
    l1, l2, l3 = params.span_length_m
    d_t1 = (params.EA / l1) * (v_feeder - v_uw) - (v_feeder / l1) * t1
    d_t2 = (params.EA / l2) * (v_nip - v_feeder) + (t1 * v_feeder - t2 * v_nip) / l2
    d_t3 = (params.EA / l3) * (v_rw - v_nip) + (t2 * v_nip - t3 * v_rw) / l3
    return (d_t1, d_t2, d_t3)


def web_torques(state: Sequence[float], params: R2RParameters) -> tuple[float, float, float]:
    """Return the Eq. (2) web-tension torque term `R_i(T_{i+1}-T_i)`."""

    return tuple(
        radius * tension_delta
        for radius, tension_delta in zip(
            params.roller_radius_m, roller_tension_differences(state), strict=True
        )
    )


def roller_velocity_derivatives(
    state: Sequence[float],
    inputs: Sequence[float],
    params: R2RParameters,
) -> tuple[float, float, float]:
    """Return `[domega_UW/dt, domega_Nip/dt, domega_RW/dt]` via paper Eq. (2).

    The v4.1 simulator commands torque directly. `inputs` are N*m, not voltages.
    """

    x = validate_vector(state, 6, "state")
    u = validate_vector(inputs, 3, "inputs")
    t1, t2, t3, omega_uw, omega_nip, omega_rw = x
    r_uw, r_nip, r_rw = params.roller_radius_m
    j_uw, j_nip, j_rw = params.inertia_kg_m2
    f_uw, f_nip, f_rw = params.kf
    dw_uw = (u[0] - t1 * r_uw - f_uw * omega_uw) / j_uw
    dw_nip = (u[1] + t2 * r_nip - t3 * r_nip - f_nip * omega_nip) / j_nip
    dw_rw = (u[2] + t3 * r_rw - f_rw * omega_rw) / j_rw
    return (dw_uw, dw_nip, dw_rw)


def derivatives(
    state: Sequence[float],
    inputs: Sequence[float],
    params: R2RParameters | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return the complete state derivative `dx/dt` in state-vector order."""

    active_params = params or R2RParameters()
    return tension_derivatives(state, active_params) + roller_velocity_derivatives(
        state, inputs, active_params
    )


def rk4_step(
    state: Sequence[float],
    inputs: Sequence[float],
    dt_s: float,
    params: R2RParameters | None = None,
) -> tuple[float, ...]:
    """Advance the R2R state by one RK4 step."""

    active_params = params or R2RParameters()
    x = validate_vector(state, 6, "state")
    u = validate_vector(inputs, 3, "inputs")

    def add_scaled(base: Sequence[float], slope: Sequence[float], scale: float) -> tuple[float, ...]:
        return tuple(base[i] + scale * slope[i] for i in range(6))

    k1 = derivatives(x, u, active_params)
    k2 = derivatives(add_scaled(x, k1, 0.5 * dt_s), u, active_params)
    k3 = derivatives(add_scaled(x, k2, 0.5 * dt_s), u, active_params)
    k4 = derivatives(add_scaled(x, k3, dt_s), u, active_params)
    return tuple(x[i] + (dt_s / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(6))


def nominal_state(params: R2RParameters | None = None, line_speed_m_s: float = 1.0) -> tuple[float, ...]:
    """Return a nominal state at target tensions and uniform line speed."""

    active_params = params or R2RParameters()
    steady_velocities = steady_surface_velocities(active_params, line_speed_m_s)
    omegas = tuple(
        velocity / radius
        for velocity, radius in zip(steady_velocities, active_params.roller_radius_m, strict=True)
    )
    return active_params.tension_ref_N + omegas


def as_named_dict(values: Iterable[float], names: Sequence[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values)}
