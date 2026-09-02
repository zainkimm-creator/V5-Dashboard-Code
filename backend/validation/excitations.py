"""Tension-setpoint excitation profiles used by SysID validation studies.

Every profile is now reconstructed from `data/model_inputs/
excitation_schedules.csv`, the edge-level schedule table that ships with the
paper (Supplementary Table S1). Nothing here transcribes step times by hand: a
schedule change in the paper package propagates by replacing that file.

The reconstruction was checked against the previous hand-written profiles and
reproduces all six group-A records bit-for-bit, and it additionally makes the
two campaign variants reachable, which transcription had missed:

- ``B_dual_channel`` ET1 is a **30 s** record, not 7 s. Group B is the grid the
  published velocity-noise (dual-channel) cells were run on, and ET1 is the one
  type whose record length differs between the groups.
- ``C_retuning_field_matched`` is the main-text section 4 protocol: E_Toggle,
  settle 1 s, 16 s record, edges at 1/6/11 s.
"""

from __future__ import annotations

from typing import Callable

from .paper_inputs import (
    DEFAULT_CAMPAIGN_GROUP,
    GROUP_A,
    GROUP_B,
    GROUP_C,
    build_excitation,
    excitation_record_count,
    excitation_records,
    excitation_schedule,
)

ExcitationProfile = Callable[[float], tuple[float, float, float]]

# Report/display order for the six published types. The membership of this tuple
# is validated against the schedule table on first use.
_NAMES = ("ET1", "ET3", "ET6", "ET3M", "E_Toggle", "EV1")

__all__ = [
    "ExcitationProfile",
    "excitation_names",
    "get_excitation_profile",
    "excitation_record_count",
    "excitation_records",
    "excitation_schedule",
    "GROUP_A",
    "GROUP_B",
    "GROUP_C",
    "DEFAULT_CAMPAIGN_GROUP",
]


def excitation_names() -> tuple[str, ...]:
    return _NAMES


def get_excitation_profile(
    name: str,
    amplitude_V: float = 0.75,
    *,
    campaign_group: str = DEFAULT_CAMPAIGN_GROUP,
    record_index: int = 0,
) -> ExcitationProfile:
    """Return a tension setpoint delta profile by paper-study name.

    `amplitude_V` is retained as a backwards-compatible argument name, but the
    value is now a tension increment in N. The simulator adds this delta to
    `T_ref`; it is no longer added to actuator command.

    `campaign_group` selects the published grid the record belongs to and
    defaults to the tension factorial. Group B falls back to group A for the
    five types whose records are identical between the two.

    `record_index` selects one record of a multi-record type (ET3M logs three,
    at line speeds v0 x {0.5, 1.0, 2.0}).
    """

    normalized = name.strip()
    if normalized not in _NAMES:
        raise ValueError(f"unknown excitation profile {name!r}; expected one of {_NAMES}")
    schedule = excitation_schedule(normalized, campaign_group, record_index)
    return build_excitation(schedule, float(amplitude_V))
