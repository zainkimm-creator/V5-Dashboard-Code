"""Measurement conditions and the v5 two-channel sensor-noise model.

The paper reports every result under one of three measurement conditions rather
than the earlier noise-free / sensor-noise pair:

- ``noise_free``    -- the commissioning reference, no sensor noise.
- ``tension_only``  -- noise on the load-cell channel alone.
- ``dual_channel``  -- noise on tension *and* velocity, which is what a real
  line has. This condition is what produces the interior logging optimum;
  tension noise alone never creates one.

Noise model, unchanged from v4.3 but now stated for both channels in one place:

    sigma_T = pct_T * T_max          load-cell full scale
    sigma_v = pct_v * v_max          v_max = v0 / 0.30

``sigma_v`` is a surface-speed error injected per roller as the *same* speed
error, so a roller of radius ``R`` sees ``sigma_v / R`` rad/s.

Note that ``T_max`` defines the noise model only. The estimator's weight matrix
uses the controller set-point ``T_ref`` instead -- the two are different
quantities and must not be interchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# v_max is referenced to the line speed through the paper's fixed ratio.
VELOCITY_FULL_SCALE_RATIO = 0.30

# The measured tension-noise grid, as a percentage of T_max.
NOISE_GRID_PERCENT = (0.02, 0.05, 0.1, 0.3, 0.5)

# The representative dual-channel dose used for the headline results.
NOMINAL_TENSION_NOISE_PERCENT = 0.3
NOMINAL_VELOCITY_NOISE_PERCENT = 0.3


class MeasurementCondition(str, Enum):
    """The three conditions every v5 result is scoped to."""

    NOISE_FREE = "noise_free"
    TENSION_ONLY = "tension_only"
    DUAL_CHANNEL = "dual_channel"

    @property
    def label(self) -> str:
        return {
            MeasurementCondition.NOISE_FREE: "Noise-free",
            MeasurementCondition.TENSION_ONLY: "Tension-only",
            MeasurementCondition.DUAL_CHANNEL: "Dual-channel",
        }[self]

    @property
    def short_label(self) -> str:
        return {
            MeasurementCondition.NOISE_FREE: "NF",
            MeasurementCondition.TENSION_ONLY: "T-only",
            MeasurementCondition.DUAL_CHANNEL: "Dual",
        }[self]


def velocity_full_scale_m_s(line_speed_m_s: float) -> float:
    """Return ``v_max = v0 / 0.30`` for the velocity-noise model."""

    return float(line_speed_m_s) / VELOCITY_FULL_SCALE_RATIO


@dataclass(frozen=True)
class SensorNoise:
    """Resolved sensor sigmas for one measurement condition."""

    condition: MeasurementCondition
    tension_percent: float
    velocity_percent: float
    tension_sigma_N: float
    velocity_sigma_m_s: float
    tension_full_scale_N: float
    velocity_full_scale_m_s: float

    @property
    def enabled(self) -> bool:
        return self.condition is not MeasurementCondition.NOISE_FREE

    def to_dict(self) -> dict[str, object]:
        return {
            "measurement_condition": self.condition.value,
            "measurement_condition_label": self.condition.label,
            "tension_noise_percent_full_scale": self.tension_percent,
            "velocity_noise_percent_full_scale": self.velocity_percent,
            "tension_sigma_N": self.tension_sigma_N,
            "velocity_sigma_m_s": self.velocity_sigma_m_s,
            "T_max_N": self.tension_full_scale_N,
            "v_max_m_s": self.velocity_full_scale_m_s,
        }


def resolve_sensor_noise(
    condition: MeasurementCondition | str,
    *,
    tension_max_N: float,
    line_speed_m_s: float,
    tension_percent: float = NOMINAL_TENSION_NOISE_PERCENT,
    velocity_percent: float | None = None,
    tension_sigma_override_N: float | None = None,
    velocity_sigma_override_m_s: float | None = None,
) -> SensorNoise:
    """Resolve the sigmas a measurement condition implies for one plant.

    ``velocity_percent`` defaults to ``tension_percent`` so that the common
    ``0.3 %/0.3 %`` dual-channel dose needs only one number, matching how the
    paper quotes it.
    """

    resolved = MeasurementCondition(condition)
    v_max = velocity_full_scale_m_s(line_speed_m_s)
    pct_t = float(tension_percent)
    pct_v = float(tension_percent if velocity_percent is None else velocity_percent)

    if resolved is MeasurementCondition.NOISE_FREE:
        pct_t = 0.0
        pct_v = 0.0
    elif resolved is MeasurementCondition.TENSION_ONLY:
        pct_v = 0.0

    sigma_t = (
        float(tension_sigma_override_N)
        if tension_sigma_override_N is not None and resolved is not MeasurementCondition.NOISE_FREE
        else pct_t / 100.0 * float(tension_max_N)
    )
    sigma_v = (
        float(velocity_sigma_override_m_s)
        if velocity_sigma_override_m_s is not None
        and resolved is MeasurementCondition.DUAL_CHANNEL
        else pct_v / 100.0 * v_max
    )
    return SensorNoise(
        condition=resolved,
        tension_percent=pct_t,
        velocity_percent=pct_v,
        tension_sigma_N=sigma_t,
        velocity_sigma_m_s=sigma_v,
        tension_full_scale_N=float(tension_max_N),
        velocity_full_scale_m_s=v_max,
    )
