"""Simulation inputs, read from the machine-readable files under data/model_inputs.

Two files define what is simulated:

- ``ten_plant_parameters.csv`` -- one row per plant of the fixed ten-plant
  subset, carrying its geometry, inertias, friction and controller timing.
- ``excitation_schedules.csv`` -- edge-level excitation schedules on the
  campaign integration grid (dt = 1 ms).

Both are the source of truth for this dashboard: the plant registry and the
excitation profiles are built from them rather than from transcribed
constants, so a schedule or parameter change propagates by replacing the
file.

Excitation reconstruction
-------------------------
Each CSV row is one reference change: `edge_time_s`, `edge_channel`
(``span1..span3`` for tension, ``v_line`` for line speed), and the level either
side of the edge as a percentage of that channel's setpoint. A channel's level
at time ``t`` is therefore the ``value_to`` of the latest edge at or before
``t``, and zero before the first edge. Every published excitation is recovered
from that one rule -- there is no per-type special casing.

Campaign groups
---------------
``A_tension_factorial`` is the default. ``B_dual_channel`` carries only ET1
(30 s instead of 7 s, the bit-exact reproduction gate for the published
velocity-noise cells); the other five types are identical to group A and fall
back to it. ``C_retuning_field_matched`` is the main-text section 4 protocol
(E_Toggle, settle 1 s, 16 s record, edges at 1/6/11 s).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PAPER_INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "model_inputs"
TEN_PLANT_CSV = PAPER_INPUT_DIR / "ten_plant_parameters.csv"
EXCITATION_SCHEDULE_CSV = PAPER_INPUT_DIR / "excitation_schedules.csv"

GROUP_A = "A_tension_factorial"
GROUP_B = "B_dual_channel"
GROUP_C = "C_retuning_field_matched"
DEFAULT_CAMPAIGN_GROUP = GROUP_A

TENSION_CHANNELS = ("span1", "span2", "span3")
LINE_SPEED_CHANNEL = "v_line"

# Every tension step in the campaign is +20 % of that channel's setpoint, and the
# callers hand us that increment directly (``amplitude`` N = 0.2 * T_ref). Edge
# levels are stored as a percentage of the setpoint, so this is the conversion
# back to the caller's unit.
REFERENCE_STEP_PERCENT = 20.0

# Seed conventions, verified on the campaign CSVs and restated in the package
# README: the velocity channel of a dual/composite run is an independent noise
# stream offset from the tension seed, and each ET3M record steps the tension
# seed by 17.
COMPOSITE_SEED_V_OFFSET = 100
ET3M_SEED_STRIDE = 17


# --------------------------------------------------------------------------- #
# ten_plant_parameters.csv
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_ten_plant_parameters() -> dict[str, dict[str, Any]]:
    """Return the ten-plant table keyed by pool id (``P001`` ...)."""

    if not TEN_PLANT_CSV.exists():
        raise FileNotFoundError(
            f"missing authoritative plant table {TEN_PLANT_CSV}; it ships with the "
            "paper reply package and must be vendored under data/reference_results/"
        )
    rows: dict[str, dict[str, Any]] = {}
    with TEN_PLANT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                text = (value or "").strip()
                try:
                    parsed[key] = float(text)
                except ValueError:
                    parsed[key] = text
            rows[str(raw["plant_id"])] = parsed
    if len(rows) != 10:
        raise ValueError(f"expected 10 plants in {TEN_PLANT_CSV.name}, found {len(rows)}")
    return rows


def ten_plant_pool_ids() -> tuple[str, ...]:
    """Return the fixed ten-plant subset in file order."""

    return tuple(load_ten_plant_parameters())


# --------------------------------------------------------------------------- #
# excitation_schedules.csv
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScheduleEdge:
    """One reference change on one channel."""

    time_s: float
    channel: str
    direction: str
    value_from_percent: float
    value_to_percent: float


@dataclass(frozen=True)
class ExcitationSchedule:
    """One excitation record as published: timing grid plus its edge list."""

    excitation: str
    campaign_group: str
    record_index: int
    duration_s: float
    settle_s: float
    episode_s: float
    dt_s: float
    Ts_s: float
    v_ref_multiplier: float
    amplitude_rule: str
    code_ref: str
    edges: tuple[ScheduleEdge, ...] = field(default_factory=tuple)

    @property
    def tension_edges(self) -> tuple[ScheduleEdge, ...]:
        return tuple(edge for edge in self.edges if edge.channel in TENSION_CHANNELS)

    @property
    def line_speed_edges(self) -> tuple[ScheduleEdge, ...]:
        return tuple(edge for edge in self.edges if edge.channel == LINE_SPEED_CHANNEL)

    @property
    def excites_line_speed(self) -> bool:
        return bool(self.line_speed_edges)

    def edge_times(self) -> tuple[float, ...]:
        return tuple(sorted({edge.time_s for edge in self.edges}))

    def describe(self) -> str:
        parts = [
            f"{edge.time_s:g}: {edge.channel} {'^' if edge.direction == 'UP' else 'v'}"
            for edge in sorted(self.edges, key=lambda e: (e.time_s, e.channel))
        ]
        return "; ".join(parts)


@lru_cache(maxsize=1)
def _load_schedule_index() -> dict[tuple[str, str], tuple[ExcitationSchedule, ...]]:
    """Return every published schedule keyed by (excitation, campaign group)."""

    if not EXCITATION_SCHEDULE_CSV.exists():
        raise FileNotFoundError(
            f"missing authoritative excitation schedules {EXCITATION_SCHEDULE_CSV}; it "
            "ships with the paper reply package and must be vendored under "
            "data/reference_results/"
        )
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    with EXCITATION_SCHEDULE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row["excitation"]).strip(),
                str(row["campaign_group"]).strip(),
                int(float(row["record_index"])),
            )
            grouped.setdefault(key, []).append(row)

    index: dict[tuple[str, str], list[ExcitationSchedule]] = {}
    for (excitation, group, record_index), rows in grouped.items():
        head = rows[0]
        edges = tuple(
            sorted(
                (
                    ScheduleEdge(
                        time_s=float(row["edge_time_s"]),
                        channel=str(row["edge_channel"]).strip(),
                        direction=str(row["edge_direction"]).strip().upper(),
                        value_from_percent=float(row["value_from_pct_of_setpoint"]),
                        value_to_percent=float(row["value_to_pct_of_setpoint"]),
                    )
                    for row in rows
                ),
                key=lambda edge: (edge.time_s, edge.channel),
            )
        )
        schedule = ExcitationSchedule(
            excitation=excitation,
            campaign_group=group,
            record_index=record_index,
            duration_s=float(head["record_duration_s"]),
            settle_s=float(head["settle_s"]),
            episode_s=float(head["episode_s"]),
            dt_s=float(head["dt_s"]),
            Ts_s=float(head["Ts_s"]),
            v_ref_multiplier=float(head["v_ref_multiplier"]),
            amplitude_rule=str(head["amplitude_rule"]).strip(),
            code_ref=str(head["code_ref"]).strip(),
            edges=edges,
        )
        index.setdefault((excitation, group), []).append(schedule)

    return {
        key: tuple(sorted(records, key=lambda record: record.record_index))
        for key, records in index.items()
    }


def excitation_records(
    name: str, campaign_group: str = DEFAULT_CAMPAIGN_GROUP
) -> tuple[ExcitationSchedule, ...]:
    """Return every record of one excitation in one campaign group.

    Group B carries only ET1; for the other five types it is identical to
    group A, so the lookup falls back rather than failing.
    """

    index = _load_schedule_index()
    key = (name.strip(), campaign_group.strip())
    if key in index:
        return index[key]
    if campaign_group.strip() != DEFAULT_CAMPAIGN_GROUP:
        fallback = (name.strip(), DEFAULT_CAMPAIGN_GROUP)
        if fallback in index:
            return index[fallback]
    known = sorted({excitation for excitation, _ in index})
    raise KeyError(
        f"no schedule for excitation {name!r} in campaign group "
        f"{campaign_group!r}; known excitations: {', '.join(known)}"
    )


def excitation_schedule(
    name: str,
    campaign_group: str = DEFAULT_CAMPAIGN_GROUP,
    record_index: int = 0,
) -> ExcitationSchedule:
    """Return one record of one excitation."""

    records = excitation_records(name, campaign_group)
    for record in records:
        if record.record_index == record_index:
            return record
    raise KeyError(
        f"excitation {name!r} in {campaign_group!r} has {len(records)} record(s); "
        f"record_index {record_index} is out of range"
    )


def excitation_record_count(
    name: str, campaign_group: str = DEFAULT_CAMPAIGN_GROUP
) -> int:
    """Return how many records one excitation logs (ET3M logs three)."""

    return len(excitation_records(name, campaign_group))


def _channel_level_percent(edges: Sequence[ScheduleEdge], channel: str, t_s: float) -> float:
    """Return the held level of one channel at ``t`` as a percent of setpoint."""

    level = 0.0
    for edge in edges:
        if edge.channel != channel:
            continue
        if t_s + 1e-12 >= edge.time_s:
            level = edge.value_to_percent
        else:
            break
    return level


def build_excitation(
    schedule: ExcitationSchedule, amplitude_N: float
) -> Callable[[float], tuple[float, float, float]]:
    """Return the tension-delta callable for one published record.

    The returned callable carries ``duration_s`` and, when the record steps the
    line speed, ``line_speed_multiplier`` -- the two attributes the simulator
    reads off an excitation.
    """

    edges = schedule.edges
    scale = float(amplitude_N) / REFERENCE_STEP_PERCENT

    def excitation(t_s: float) -> tuple[float, float, float]:
        return tuple(  # type: ignore[return-value]
            scale * _channel_level_percent(edges, channel, t_s)
            for channel in TENSION_CHANNELS
        )

    setattr(excitation, "duration_s", schedule.duration_s)
    setattr(excitation, "schedule", schedule)
    if schedule.excites_line_speed:

        def line_speed_multiplier(t_s: float) -> float:
            return 1.0 + _channel_level_percent(edges, LINE_SPEED_CHANNEL, t_s) / 100.0

        setattr(excitation, "line_speed_multiplier", line_speed_multiplier)
    return excitation


# --------------------------------------------------------------------------- #
# seed conventions
# --------------------------------------------------------------------------- #
def velocity_seed(tension_seed: int) -> int:
    """Return ``seed_v = seed_T + 100`` for a dual/composite run."""

    return int(tension_seed) + COMPOSITE_SEED_V_OFFSET


def et3m_record_seed(base_seed: int, record_index: int) -> int:
    """Return ``seed_T = base + 17 * i`` for ET3M record ``i``."""

    return int(base_seed) + ET3M_SEED_STRIDE * int(record_index)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def paper_input_provenance() -> dict[str, Any]:
    """Return a run-metadata block naming the files the run was driven from."""

    def _stat(path: Path) -> dict[str, Any]:
        return {
            "file": path.name,
            "present": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
        }

    schedules = _load_schedule_index()
    return {
        "source": "data/model_inputs",
        "ten_plant_parameters": _stat(TEN_PLANT_CSV),
        "excitation_schedules": _stat(EXCITATION_SCHEDULE_CSV),
        "plant_ids": list(ten_plant_pool_ids()),
        "schedule_keys": sorted(f"{name}/{group}" for name, group in schedules),
        "seed_conventions": {
            "composite_seed_v_offset": COMPOSITE_SEED_V_OFFSET,
            "et3m_seed_stride": ET3M_SEED_STRIDE,
        },
    }


def schedule_summary_rows() -> list[dict[str, Any]]:
    """Return one row per published record, for reports and tests."""

    rows: list[dict[str, Any]] = []
    for (name, group), records in sorted(_load_schedule_index().items()):
        for record in records:
            rows.append(
                {
                    "excitation": name,
                    "campaign_group": group,
                    "record_index": record.record_index,
                    "duration_s": record.duration_s,
                    "settle_s": record.settle_s,
                    "episode_s": record.episode_s,
                    "v_ref_multiplier": record.v_ref_multiplier,
                    "edge_count": len(record.edges),
                    "edges": record.describe(),
                }
            )
    return rows


def assert_matches_reference(
    reference: Mapping[str, Mapping[str, float]], *, tolerance: float = 1e-9
) -> None:
    """Raise if a caller's transcribed durations disagree with the CSV."""

    for name, expected in reference.items():
        schedule = excitation_schedule(name)
        got = schedule.duration_s
        want = float(expected["duration_s"])
        if not math.isclose(got, want, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"excitation {name}: duration {got} s disagrees with the "
                f"authoritative schedule {want} s"
            )
