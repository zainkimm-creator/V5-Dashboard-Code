"""Validation study orchestration."""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from backend.models.controller import ControllerConfig, auto_tension_integral_time_s
from backend.models.equations import PARAMETER_NAMES, R2RParameters
from backend.models.simulation import SimulationConfig, simulate
from backend.models.noise import resolve_sensor_noise, velocity_full_scale_m_s
from backend.sysid.estimator import (
    CANONICAL_BOUND_DECADE,
    CANONICAL_INITIAL_SCALE,
    MeasurementCondition,
    estimate_parameters,
    estimate_parameters_one_step_pem,
    estimate_parameters_weighted_pem,
)
from backend.validation.excitations import excitation_names, get_excitation_profile
from backend.validation.paper_inputs import (
    COMPOSITE_SEED_V_OFFSET,
    GROUP_A,
    GROUP_B,
    build_excitation,
    et3m_record_seed,
    excitation_records,
    excitation_schedule,
    paper_input_provenance,
)
from backend.validation.paper_reference import (
    load_drift_reference,
    load_excitation_reference,
    load_logging_power_law_reference,
)
from backend.validation.plotting import write_bar_chart, write_line_chart

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "reports" / "validation_summary"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PAPER_LOGGING_ADEQUACY_REFERENCE = PROJECT_ROOT / "data" / "reference_results" / "logging_rate_v5_reference.json"
DEFAULT_TLOG_MS_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
DEFAULT_TMIN_MS = 50.0
# v5 tension-only E_Toggle medians behind the 50 Hz working cutoff
# (supplement Fig. S6b). The v4.1 anchors 169.0 / 23.2 / 77.2 are retired:
# they came from the unweighted tension-only estimator at LPF 100 Hz.
LOGGING_SN_TEXT_REFERENCE_BY_TLOG_MS = {
    1.0: 12.0,
    2.0: 13.0,
    5.0: 18.0,
    10.0: 22.0,
    20.0: 26.0,
    50.0: 65.0,
    100.0: 89.0,
}
LOGGING_RATE_CACHE_VERSION = "logging_v5_paper_csv_inputs_20260813"
# v5 Fig. 2(a) plots epsilon_theta vs Tlog under three measurement conditions.
# The v4.1 noise-free / sensor-noise pair is superseded: tension-only and
# dual-channel behave differently, and only the dual-channel case has an
# interior optimum, so collapsing them hides the headline result.
LOGGING_CONDITIONS = ("noise_free", "tension_only", "dual_channel")
LOGGING_CONDITION_LABELS = {
    "noise_free": "Noise-free",
    "tension_only": "Tension-only",
    "dual_channel": "Dual-channel",
}
LOGGING_CONDITION_SHORT_LABELS = {
    "noise_free": "NF",
    "tension_only": "T-only",
    "dual_channel": "Dual",
}
# The reference records three seeds for the dual-channel campaign and one for
# the other two conditions.
LOGGING_DUAL_CHANNEL_SEEDS = (0, 1, 2)
# Each v5 series comes from a single excitation (reference `series_provenance`).
# Driving E_Toggle and comparing against an ET1 series would compare two
# different campaigns.
LOGGING_CONDITION_EXCITATION = {
    "noise_free": "ET1",
    "tension_only": "E_Toggle",
    "dual_channel": "E_Toggle",
}
# Campaign grid each condition's record comes from. The dual-channel cells were
# published from group B; for E_Toggle that grid is identical to group A, so the
# lookup resolves back to A and only the provenance differs.
LOGGING_CONDITION_CAMPAIGN_GROUP = {
    "noise_free": GROUP_A,
    "tension_only": GROUP_A,
    "dual_channel": GROUP_B,
}
LOGGING_CONDITION_COLORS = {
    "noise_free": "#2f6f73",
    "tension_only": "#b35f2e",
    "dual_channel": "#5954a4",
}
EXCITATION_CACHE_VERSION = "excitation_v5_paper_csv_inputs_3seed_sn_20260813"
# Record length is a property of the excitation schedule, not of the noise
# condition: `excitation_schedules.csv` gives ET1 7 s and E_Toggle 17 s, and
# the whole record including the settle window enters the estimator. These two
# constants are retained only as fallbacks for profiles that carry no duration.
LOGGING_NF_DURATION_S = 12.0
LOGGING_SN_DURATION_S = 12.0
# The v5 campaign runs both noisy conditions behind one fixed 50 Hz anti-alias
# cutoff (reference `conditions`). The v4.1 code carried a per-Tlog LPF table
# (None / 200 / 100 / 12 / 50 / 10 / 20 Hz) plus per-Tlog velocity-observer
# clips, and quartered the excitation amplitude under noise. Those three knobs
# were fitted to the earlier unweighted campaign and they invert the v5 result:
# leaving the finest logging periods unfiltered while cutting drive amplitude
# 4x makes 1 ms the WORST tension-only setting, whereas the paper reports it as
# the best and the curve as monotonic. They are retired here.
LOGGING_SN_LPF_HZ = 50.0
LOGGING_SN_AMPLITUDE_FACTOR = 1.0
LOGGING_VELOCITY_OBSERVER_CLIP_FRACTION = None
# v5 reports the excitation SN column at the dual-channel condition behind a
# 50 Hz filter. 50 Hz is the working cutoff; 100 Hz measurably worsens the fit.
EXCITATION_SN_LPF_HZ = 50.0
# The SN column is the dual-channel condition, and paper Fig. 2(a) states that
# condition is pooled over "10 plants x 3 seeds". Table 1 does not restate a seed
# count, so applying the same convention to its SN column is an inference -- but
# a single seed demonstrably is not enough there: EV1 reads 31.4 % on seed 0
# alone against 26.8 % pooled over three, where the paper prints 26.7 %.
EXCITATION_SN_SEEDS = (0, 1, 2)
# The velocity half of the dual-channel dose, as a percentage of
# v_max = v0 / 0.30. The paper quotes the pair as 0.3%/0.3%.
EXCITATION_SN_VELOCITY_PERCENT = 0.3
METADATA_FIELDS = (
    "run_timestamp_utc",
    "plant_scope",
    "code_version",
    "value_status",
    "run_settings_json",
)
GRAPH_POINT_FIELDS = (
    *METADATA_FIELDS,
    "group",
    "plant_id",
    "T_log_ms",
    "tau_ratio",
    "dashboard_MARE_percent",
    "paper_MARE_percent",
    "difference_MARE_percent",
)
REFERENCE_POWER_LAW_VALUES = load_logging_power_law_reference().get("values", [])
SENSOR_NOISE_FULL_SCALE_PERCENT = 0.3
SENSOR_NOISE_TENSION_FULL_SCALE_N = 50.0
SENSOR_NOISE_OMEGA_FULL_SCALE_RAD_S = 20.0
SENSOR_NOISE_TENSION_N = SENSOR_NOISE_TENSION_FULL_SCALE_N * SENSOR_NOISE_FULL_SCALE_PERCENT / 100.0
SENSOR_NOISE_OMEGA_RAD_S = 0.0


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    version = result.stdout.strip()
    return version or "uncommitted"


def _run_metadata(
    study: str,
    *,
    plant_scope: str,
    run_settings: Mapping[str, object] | None = None,
    value_status: str = "computed_raw",
) -> dict[str, object]:
    return {
        "study": study,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plant_scope": plant_scope,
        "code_version": _code_version(),
        "run_settings": dict(run_settings or {}),
        "value_status": value_status,
    }


def _metadata_row_fields(metadata: Mapping[str, object], value_status: str | None = None) -> dict[str, object]:
    return {
        "run_timestamp_utc": metadata.get("run_timestamp_utc"),
        "plant_scope": metadata.get("plant_scope"),
        "code_version": metadata.get("code_version"),
        "value_status": value_status or metadata.get("value_status"),
        "run_settings_json": json.dumps(metadata.get("run_settings", {}), sort_keys=True),
    }


def _bounded_velocity_observer_rows(
    rows: Sequence[Mapping[str, float]],
    params: R2RParameters,
    clip_fraction: float | None,
) -> list[dict[str, float]]:
    """Blend logged speed with speed inferred from noisy tension finite differences."""

    adjusted = [dict(row) for row in rows]
    if clip_fraction is None or clip_fraction <= 0.0 or len(adjusted) < 2:
        return adjusted

    ea = float(params.EA)
    l1, l2, l3 = params.span_length_m
    vf_default = float(params.feeder_velocity_m_s)
    derived: list[tuple[float, float, float]] = []

    def clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    for index in range(len(rows) - 1):
        row = rows[index]
        next_row = rows[index + 1]
        dt = float(next_row["time_s"]) - float(row["time_s"])
        if dt <= 0.0:
            continue
        vf = float(row.get("line_speed_ref_m_s", vf_default))
        limit = max(1e-9, float(clip_fraction) * max(abs(vf), 1e-9))
        t1 = float(row["T1"])
        t2 = float(row["T2"])
        t3 = float(row["T3"])
        d1 = (float(next_row["T1"]) - t1) / dt
        d2 = (float(next_row["T2"]) - t2) / dt
        d3 = (float(next_row["T3"]) - t3) / dt

        v_uw_from_t = vf - (l1 / ea) * (d1 + vf * t1 / l1)
        v_nip_from_t = (l2 * d2 - (t1 - ea) * vf) / max(ea - t2, 1e-9)
        v_rw_from_t = (l3 * d3 - (t2 - ea) * v_nip_from_t) / max(ea - t3, 1e-9)
        base = (
            float(row["v_UW_m_s"]),
            float(row["v_Nip_m_s"]),
            float(row["v_RW_m_s"]),
        )
        inferred = (v_uw_from_t, v_nip_from_t, v_rw_from_t)
        derived.append(tuple(base[i] + clamp(inferred[i] - base[i], limit) for i in range(3)))

    if not derived:
        return adjusted

    radii = params.roller_radius_m
    for index, (v_uw, v_nip, v_rw) in enumerate(derived):
        adjusted[index]["v_UW_m_s"] = v_uw
        adjusted[index]["v_Nip_m_s"] = v_nip
        adjusted[index]["v_RW_m_s"] = v_rw
        adjusted[index]["omega_UW"] = v_uw / radii[0]
        adjusted[index]["omega_Nip"] = v_nip / radii[1]
        adjusted[index]["omega_RW"] = v_rw / radii[2]
    v_uw, v_nip, v_rw = derived[-1]
    adjusted[-1]["v_UW_m_s"] = v_uw
    adjusted[-1]["v_Nip_m_s"] = v_nip
    adjusted[-1]["v_RW_m_s"] = v_rw
    adjusted[-1]["omega_UW"] = v_uw / radii[0]
    adjusted[-1]["omega_Nip"] = v_nip / radii[1]
    adjusted[-1]["omega_RW"] = v_rw / radii[2]
    return adjusted


def _write_summary(name: str, payload: Mapping[str, object]) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _write_rows_csv(name: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return str(path)


def _case_group(case_name: object) -> str:
    """Return the short condition tag used in graph-point exports.

    v5 has three conditions, so a two-valued NF/SN tag can no longer identify a
    row: tension-only and dual-channel are different campaigns with different
    conclusions and must not share a label.
    """

    return LOGGING_CONDITION_SHORT_LABELS.get(str(case_name), str(case_name))


def _graph_point_rows(
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        dashboard_mare = row.get("MARE_theta_percent")
        paper_mare = row.get("paper_numerical_MARE_theta_percent")
        difference_mare = None
        if dashboard_mare is not None and paper_mare is not None:
            difference_mare = float(dashboard_mare) - float(paper_mare)
        result.append(
            {
                **(_metadata_row_fields(metadata or {}, str(row.get("value_status") or "computed_raw")) if metadata else {}),
                "group": _case_group(row.get("case")),
                "plant_id": row.get("plant_id"),
                "T_log_ms": row.get("Tlog_ms"),
                "tau_ratio": row.get("tau_ratio"),
                "dashboard_MARE_percent": dashboard_mare,
                "paper_MARE_percent": paper_mare,
                "difference_MARE_percent": difference_mare,
            }
        )
    return result


def _write_graph_points_csv(
    name: str,
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> str:
    return _write_rows_csv(name, _graph_point_rows(rows, metadata), GRAPH_POINT_FIELDS)


def _excel_column_name(index: int) -> str:
    name = ""
    active = index
    while active:
        active, remainder = divmod(active - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(reference: str, value: object) -> str:
    if value is None:
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return f'<c r="{reference}"><v>{float(value):.12g}</v></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _write_graph_points_xlsx(
    name: str,
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / name
    graph_rows = _graph_point_rows(rows, metadata)
    sheet_rows = [dict(zip(GRAPH_POINT_FIELDS, GRAPH_POINT_FIELDS, strict=True)), *graph_rows]
    last_row = max(1, len(sheet_rows))
    last_col = _excel_column_name(len(GRAPH_POINT_FIELDS))
    xml_rows: list[str] = []
    for row_index, row in enumerate(sheet_rows, start=1):
        cells = []
        for col_index, field in enumerate(GRAPH_POINT_FIELDS, start=1):
            cells.append(_xlsx_cell(f"{_excel_column_name(col_index)}{row_index}", row.get(field)))
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    worksheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="A1:{last_col}{last_row}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="1" width="12" customWidth="1"/>
    <col min="2" max="2" width="12" customWidth="1"/>
    <col min="3" max="5" width="14" customWidth="1"/>
  </cols>
  <sheetData>{''.join(xml_rows)}</sheetData>
  <autoFilter ref="A1:{last_col}{last_row}"/>
</worksheet>"""
    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Graph points" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    workbook_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    root_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
    return str(path)


def _artifact_payload(
    metrics: object,
    plot_path: str | None,
    summary_path: str | None,
    csv_path: str | None = None,
    markdown_path: str | None = None,
) -> dict[str, object]:
    payload = {
        "metrics": metrics,
        "plot_path": plot_path,
        "summary_path": summary_path,
        "csv_path": csv_path,
    }
    if markdown_path:
        payload["markdown_path"] = markdown_path
    return payload


def _format_tlog_for_name(tlog_ms: float) -> str:
    if float(tlog_ms).is_integer():
        return f"{int(tlog_ms)}"
    return f"{tlog_ms:g}".replace(".", "p")


def _normalise_tlog_values(tlog_ms_values: Sequence[float] | None) -> list[float]:
    values = list(DEFAULT_TLOG_MS_VALUES if tlog_ms_values is None else tlog_ms_values)
    if not values:
        raise ValueError("At least one Tlog value is required.")
    normalized: list[float] = []
    for value in values:
        tlog_ms = float(value)
        if not math.isfinite(tlog_ms):
            raise ValueError("Tlog values must be finite.")
        if tlog_ms < 1.0 or tlog_ms > 200.0:
            raise ValueError("Tlog values must stay within the 1-200 ms PLC logging range.")
        normalized.append(tlog_ms)
    return normalized


# Preference order when the paper publishes a point for more than one excitation
# at the same (condition, Tlog). Each v5 condition series comes from a SINGLE
# excitation so the curve stays internally consistent: noise-free from ET1
# (supplement Table S7), the two noisy conditions from E_Toggle (Fig. S6b and
# Fig. 2(a) respectively).
LOGGING_REFERENCE_EXCITATION_PREFERENCE = ("E_Toggle", "ET1")


def _load_logging_reference_rows() -> list[dict[str, float | str]]:
    """Return the published v5 logging-adequacy rows for all three conditions.

    Every row keeps its own condition tag. Nothing is collapsed onto a single
    "sensor noise" slot: v5 publishes a complete 2-100 ms noise-free series, a
    complete 1-100 ms tension-only series, and three dual-channel points, and
    the dashboard has to be able to show each of them beside its own result.
    """

    if not PAPER_LOGGING_ADEQUACY_REFERENCE.exists():
        return []

    payload = json.loads(PAPER_LOGGING_ADEQUACY_REFERENCE.read_text(encoding="utf-8"))

    # Keep the most-preferred excitation per (condition, Tlog); the paper
    # publishes some points only for ET1 and others only for E_Toggle.
    best: dict[tuple[str, float], tuple[int, dict[str, float | str]]] = {}
    for row in payload.get("values", []):
        case_name = str(row.get("case", ""))
        if case_name not in LOGGING_CONDITIONS:
            continue
        value = row.get("MARE_theta_percent")
        if value is None:
            # The paper omits this cell deliberately (a trivial near-zero fit or
            # an undefined ratio). Do not fabricate a point for it.
            continue
        excitation = str(row.get("excitation", "E_Toggle"))
        try:
            rank = LOGGING_REFERENCE_EXCITATION_PREFERENCE.index(excitation)
        except ValueError:
            rank = len(LOGGING_REFERENCE_EXCITATION_PREFERENCE)
        tlog_ms = float(row["Tlog_ms"])
        key = (case_name, tlog_ms)
        if key in best and best[key][0] <= rank:
            continue
        best[key] = (
            rank,
            {
                "case": case_name,
                "measurement_condition": case_name,
                "excitation": excitation,
                "plant_id": "ALL",
                "Tlog_ms": tlog_ms,
                "tau_ratio": tlog_ms / DEFAULT_TMIN_MS,
                "MARE_theta_percent": float(value),
                "paper_reference_samples": float(row.get("plant_count", 10)),
                "paper_value_status": str(row.get("value_status", "paper_reported")),
            },
        )
    return [
        entry
        for _, entry in sorted(
            best.values(),
            key=lambda item: (LOGGING_CONDITIONS.index(str(item[1]["case"])), item[1]["Tlog_ms"]),
        )
    ]


def _logging_reference_annotations() -> dict[str, object]:
    """Return the published points that are annotations rather than curve points.

    The noise-free E_Toggle pair is the important case: the 1 ms cell is an
    essentially exact fit that falls off a logarithmic axis, so Fig. 2(a) starts
    the noise-free curve at 2 ms. That cell is off-scale, not missing, and the
    dashboard has to say so rather than leave a blank.
    """

    if not PAPER_LOGGING_ADEQUACY_REFERENCE.exists():
        return {}
    payload = json.loads(PAPER_LOGGING_ADEQUACY_REFERENCE.read_text(encoding="utf-8"))
    return {
        "E_Toggle_noise_free_anchors": payload.get("E_Toggle_noise_free_anchors", {}),
        "interior_optimum": payload.get("interior_optimum", {}),
        "series_provenance": payload.get("series_provenance", {}),
        "speed_decomposition": payload.get("speed_decomposition", {}),
        "organizing_variable": payload.get("organizing_variable", {}),
        "conditions": payload.get("conditions", {}),
        "notes": payload.get("notes", []),
        "source": payload.get("source", ""),
    }


def _logging_speed_groups() -> dict[str, dict[str, object]]:
    """Return the paper's slow/fast plant split for the dual-channel case.

    Fig. 2(b) exists because the pooled dual-channel valley represents neither
    sub-group. The plant membership is published, so the dashboard splits its
    own runs the same way instead of inventing a threshold.
    """

    decomposition = _logging_reference_annotations().get("speed_decomposition", {})
    groups: dict[str, dict[str, object]] = {}
    for key, label in (("slow", "Slow plants (v0 <= 0.5 m/s)"), ("fast", "Fast plants (v0 >= 2.0 m/s)")):
        entry = decomposition.get(key) or {}
        groups[key] = {
            "key": key,
            "label": label,
            "definition": str(entry.get("definition", "")),
            "plants": [str(plant) for plant in entry.get("plants", [])],
            "note": str(entry.get("note", "")),
            "paper_band_percent": (
                entry.get("E_Toggle_band_percent_at_Tlog_le_5ms")
                or entry.get("plateau_percent_at_Tlog_le_5ms")
            ),
            "paper_band_meaning": (
                "own optimum band at Tlog <= 5 ms"
                if entry.get("E_Toggle_band_percent_at_Tlog_le_5ms")
                else "plateau at Tlog <= 5 ms"
            ),
        }
    return groups


def _logging_speed_group_for_plant(
    plant_id: str,
    speed_groups: Mapping[str, Mapping[str, object]],
) -> str | None:
    """Return "slow", "fast", or None for a plant id.

    Membership comes from the paper's published lists rather than from a
    threshold applied to the dashboard's own speeds, so the decomposition is
    the same split the paper drew.
    """

    for group_key, group in speed_groups.items():
        if str(plant_id) in {str(plant) for plant in group.get("plants", [])}:
            return group_key
    return None


def _summarise_logging_reference(
    reference_rows: Sequence[Mapping[str, float | str]],
) -> dict[str, dict[float, dict[str, float]]]:
    groups: dict[tuple[str, float], list[float]] = {}
    for row in reference_rows:
        case_name = str(row["case"])
        tlog_ms = float(row["Tlog_ms"])
        mare = float(row["MARE_theta_percent"])
        groups.setdefault((case_name, tlog_ms), []).append(mare)

    reference: dict[str, dict[float, dict[str, float]]] = {
        condition: {} for condition in LOGGING_CONDITIONS
    }
    for (case_name, tlog_ms), values in groups.items():
        reference[case_name][tlog_ms] = {
            "paper_median_MARE_theta_percent": statistics.median(values),
            "paper_mean_MARE_theta_percent": sum(values) / len(values),
            "paper_reference_samples": max(
                float(row.get("paper_reference_samples", 1.0))
                for row in reference_rows
                if str(row["case"]) == case_name and math.isclose(float(row["Tlog_ms"]), tlog_ms)
            ),
        }
    return reference


def _fit_power_law(
    label: str,
    points: Sequence[tuple[float, float]],
    *,
    source: str,
    case_name: str,
    branch: str,
) -> dict[str, float | str | None]:
    filtered = [
        (float(x_value), float(y_value))
        for x_value, y_value in points
        if x_value is not None
        and y_value is not None
        and math.isfinite(float(x_value))
        and math.isfinite(float(y_value))
        and float(x_value) > 0.0
        and float(y_value) > 0.0
    ]
    if len(filtered) < 2:
        return {
            "label": label,
            "source": source,
            "case": case_name,
            "branch": branch,
            "a": None,
            "alpha": None,
            "r_square": None,
            "points": float(len(filtered)),
        }

    log_x = [math.log(point[0]) for point in filtered]
    log_y = [math.log(point[1]) for point in filtered]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    variance_x = sum((x_value - mean_x) ** 2 for x_value in log_x)
    if variance_x <= 1e-12:
        return {
            "label": label,
            "source": source,
            "case": case_name,
            "branch": branch,
            "a": None,
            "alpha": None,
            "r_square": None,
            "points": float(len(filtered)),
        }

    alpha = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(log_x, log_y, strict=True))
    alpha /= variance_x
    intercept = mean_y - alpha * mean_x
    predictions = [intercept + alpha * x_value for x_value in log_x]
    residual_sum = sum((actual - predicted) ** 2 for actual, predicted in zip(log_y, predictions, strict=True))
    total_sum = sum((actual - mean_y) ** 2 for actual in log_y)
    r_square = 1.0 - residual_sum / total_sum if total_sum > 1e-12 else 1.0
    coefficient = math.exp(intercept)
    return {
        "label": label,
        "source": source,
        "case": case_name,
        "branch": branch,
        "a": coefficient,
        "alpha": alpha,
        "r_square": r_square,
        "points": float(len(filtered)),
    }


def _fit_power_law_with_reference_alpha(
    label: str,
    points: Sequence[tuple[float, float]],
    *,
    source: str,
    case_name: str,
    branch: str,
) -> dict[str, float | str | None]:
    """Fit coefficient a with the paper-reported alpha and R2."""

    reference = REFERENCE_POWER_LAW_VALUES[case_name]
    alpha = float(reference["alpha"])
    r_square = float(reference["r_square"])
    filtered = [
        (float(x_value), float(y_value))
        for x_value, y_value in points
        if x_value is not None
        and y_value is not None
        and math.isfinite(float(x_value))
        and math.isfinite(float(y_value))
        and float(x_value) > 0.0
        and float(y_value) > 0.0
    ]
    if len(filtered) < 2:
        return {
            "label": label,
            "source": source,
            "case": case_name,
            "branch": branch,
            "a": None,
            "alpha": alpha,
            "r_square": r_square,
            "points": float(len(filtered)),
            "fit_note": "paper_reported_alpha_r2",
        }

    log_intercept = sum(math.log(y_value) - alpha * math.log(x_value) for x_value, y_value in filtered) / len(filtered)
    return {
        "label": label,
        "source": source,
        "case": case_name,
        "branch": branch,
        "a": math.exp(log_intercept),
        "alpha": alpha,
        "r_square": r_square,
        "points": float(len(filtered)),
        "fit_note": "paper_reported_alpha_r2",
    }


def _curve_metadata(
    label: str,
    points: Sequence[tuple[float, float]],
    *,
    source: str,
    case_name: str,
    branch: str,
    fit_note: str,
) -> dict[str, float | str | None]:
    return {
        "label": label,
        "source": source,
        "case": case_name,
        "branch": branch,
        "a": None,
        "alpha": None,
        "r_square": None,
        "points": float(len(_sorted_positive_points(points))),
        "fit_note": fit_note,
    }


def _log_interpolate_positive(
    x_value: float,
    left_x: float,
    left_y: float,
    right_x: float,
    right_y: float,
) -> float:
    if min(x_value, left_x, left_y, right_x, right_y) <= 0.0:
        raise ValueError("Log interpolation values must be positive.")
    if math.isclose(left_x, right_x):
        return float(left_y)
    fraction = (math.log(float(x_value)) - math.log(float(left_x))) / (
        math.log(float(right_x)) - math.log(float(left_x))
    )
    return math.exp(math.log(float(left_y)) + fraction * (math.log(float(right_y)) - math.log(float(left_y))))


def _logging_sensor_noise_text_reference_percent(tlog_ms: float) -> float:
    tlog = float(tlog_ms)
    low_tlog = 1.0
    optimum_tlog = 20.0
    high_tlog = 100.0
    low_value = LOGGING_SN_TEXT_REFERENCE_BY_TLOG_MS[low_tlog]
    optimum_value = LOGGING_SN_TEXT_REFERENCE_BY_TLOG_MS[optimum_tlog]
    high_value = LOGGING_SN_TEXT_REFERENCE_BY_TLOG_MS[high_tlog]
    if tlog <= optimum_tlog:
        return _log_interpolate_positive(tlog, low_tlog, low_value, optimum_tlog, optimum_value)
    return _log_interpolate_positive(tlog, optimum_tlog, optimum_value, high_tlog, high_value)


def _logging_sensor_noise_text_reference_points(
    tlog_ms_values: Sequence[float],
    tmin_ms: float,
) -> list[tuple[float, float]]:
    return [
        (float(tlog_ms) / float(tmin_ms), _logging_sensor_noise_text_reference_percent(float(tlog_ms)))
        for tlog_ms in tlog_ms_values
    ]


def _fit_line_points(fit: Mapping[str, float | str | None], points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    coefficient = fit.get("a")
    alpha = fit.get("alpha")
    if coefficient is None or alpha is None:
        return []
    filtered_x = [float(x_value) for x_value, y_value in points if x_value > 0.0 and y_value > 0.0]
    if not filtered_x:
        return []
    x_min = min(filtered_x)
    x_max = max(filtered_x)
    if math.isclose(x_min, x_max):
        return []
    result = []
    for index in range(36):
        fraction = index / 35.0
        x_value = math.exp(math.log(x_min) + fraction * (math.log(x_max) - math.log(x_min)))
        y_value = float(coefficient) * (x_value ** float(alpha))
        result.append((x_value, y_value))
    return result


def _sorted_positive_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted(
        [
            (float(x_value), float(y_value))
            for x_value, y_value in points
            if x_value > 0.0
            and y_value > 0.0
            and math.isfinite(float(x_value))
            and math.isfinite(float(y_value))
        ],
        key=lambda point: point[0],
    )


def _log_tick_values(log_lo: float, log_hi: float, *, limit: int = 9) -> list[float]:
    """Return readable 1/2/5 log ticks inside a log10 domain."""

    ticks: list[float] = []
    start_power = math.floor(log_lo) - 1
    end_power = math.ceil(log_hi) + 1
    for power in range(start_power, end_power + 1):
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * (10.0**power)
            if log_lo <= math.log10(value) <= log_hi:
                ticks.append(value)

    if len(ticks) <= limit:
        return ticks

    step = max(1, math.ceil(len(ticks) / limit))
    sampled = ticks[::step]
    if ticks[-1] not in sampled:
        sampled.append(ticks[-1])
    return sampled


def _format_axis_tick(value: float) -> str:
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if value >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _write_logging_log_chart(
    series: Sequence[tuple[str, Sequence[tuple[float, float]], str, str]],
    path: Path,
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    bands: Sequence[tuple[str, Sequence[float], str]] = (),
    annotations: Sequence[str] = (),
) -> str:
    """Draw a log-log MARE-vs-Tlog figure with one entry per plotted series.

    Each series is ``(label, points, color, dash)``. A dashed stroke marks the
    paper series and a solid stroke the dashboard's own result, so the reader
    can tell at a glance which is which without reading the legend twice.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 640
    left, right, top, bottom = 92, 400, 74, 150
    plotted = [
        point
        for _, values, _, _ in series
        for point in values
        if point[0] > 0.0 and point[1] > 0.0
    ]
    band_values = [
        float(bound)
        for _, bounds, _ in bands
        if bounds is not None
        for bound in bounds
        if float(bound) > 0.0
    ]
    if not plotted:
        return ""

    log_x_values = [math.log10(point[0]) for point in plotted]
    log_y_values = [math.log10(point[1]) for point in plotted] + [
        math.log10(value) for value in band_values
    ]
    x_lo, x_hi = min(log_x_values), max(log_x_values)
    y_lo, y_hi = min(log_y_values), max(log_y_values)
    x_pad = max(0.08, (x_hi - x_lo) * 0.06)
    y_pad = max(0.10, (y_hi - y_lo) * 0.10)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad

    def scale_x(x_value: float) -> float:
        return left + (math.log10(x_value) - x_lo) * (width - left - right) / (x_hi - x_lo)

    def scale_y(y_value: float) -> float:
        return height - bottom - (math.log10(y_value) - y_lo) * (height - top - bottom) / (y_hi - y_lo)

    x_axis_y = height - bottom
    plot_right = width - right
    grid_elements: list[str] = []
    for tick in _log_tick_values(x_lo, x_hi, limit=8):
        x_pos = scale_x(tick)
        grid_elements.append(
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{x_axis_y}" stroke="#dedbce" stroke-width="1" />'
        )
        grid_elements.append(
            f'<text x="{x_pos:.1f}" y="{x_axis_y + 24}" font-size="12" font-family="Arial" text-anchor="middle" fill="#34463f">{_format_axis_tick(tick)}</text>'
        )
    for tick in _log_tick_values(y_lo, y_hi, limit=8):
        y_pos = scale_y(tick)
        grid_elements.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{plot_right}" y2="{y_pos:.1f}" stroke="#dedbce" stroke-width="1" />'
        )
        grid_elements.append(
            f'<text x="{left - 10}" y="{y_pos + 4:.1f}" font-size="12" font-family="Arial" text-anchor="end" fill="#34463f">{_format_axis_tick(tick)}</text>'
        )

    elements: list[str] = []
    legend_rows: list[str] = []
    legend_index = 0
    for band_label, bounds, band_color in bands:
        if bounds is None or len(bounds) != 2:
            continue
        lo, hi = sorted(float(bound) for bound in bounds)
        if lo <= 0.0:
            continue
        y_top, y_bottom = scale_y(hi), scale_y(lo)
        elements.append(
            f'<rect x="{left}" y="{y_top:.1f}" width="{plot_right - left:.1f}" '
            f'height="{max(2.0, y_bottom - y_top):.1f}" fill="{band_color}" fill-opacity="0.16" />'
        )
        legend_rows.append(
            f'<g transform="translate({plot_right + 24},{top + 8 + legend_index * 26})">'
            f'<rect width="18" height="10" y="3" fill="{band_color}" fill-opacity="0.3" stroke="{band_color}" stroke-width="1.2" />'
            f'<text x="28" y="13" font-size="13" font-family="Arial" fill="#243033">{escape(band_label)}</text></g>'
        )
        legend_index += 1

    for label, values, color, dash in series:
        points = [(x, y) for x, y in values if x > 0.0 and y > 0.0]
        if not points:
            continue
        points.sort()
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if len(points) > 1:
            path_data = " ".join(f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in points)
            elements.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash_attr} points="{path_data}" />'
            )
        for x, y in points:
            elements.append(
                f'<circle cx="{scale_x(x):.1f}" cy="{scale_y(y):.1f}" r="4.6" fill="{color}" stroke="#fff" stroke-width="1.4" />'
            )
        legend_rows.append(
            f'<g transform="translate({plot_right + 24},{top + 8 + legend_index * 26})">'
            f'<line x1="0" y1="8" x2="20" y2="8" stroke="{color}" stroke-width="3"{dash_attr} />'
            f'<text x="28" y="13" font-size="13" font-family="Arial" fill="#243033">{escape(label)}</text></g>'
        )
        legend_index += 1

    # Wrap footnotes to the drawing width; an unwrapped line runs off the SVG
    # and is silently clipped by the viewer.
    wrapped: list[str] = []
    max_chars = max(40, int((width - left - 24) / 6.1))
    for note in annotations:
        if not note.strip():
            continue
        line = ""
        for word in note.split():
            candidate = f"{line} {word}".strip()
            if len(candidate) > max_chars and line:
                wrapped.append(line)
                line = word
            else:
                line = candidate
        if line:
            wrapped.append(line)
    note_elements = [
        f'<text x="{left}" y="{x_axis_y + 62 + index * 16}" font-size="12" font-family="Arial" fill="#6f6a5d">{escape(line)}</text>'
        for index, line in enumerate(wrapped)
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="30" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">{escape(title)}</text>
<text x="{left}" y="52" font-size="13" font-family="Arial" fill="#6f6a5d">{escape(subtitle)}</text>
{''.join(grid_elements)}
<line x1="{left}" y1="{x_axis_y}" x2="{plot_right}" y2="{x_axis_y}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{x_axis_y}" stroke="#445" stroke-width="1.5"/>
<text x="{(left + plot_right)/2}" y="{x_axis_y + 46}" font-size="15" font-family="Arial" text-anchor="middle" fill="#243033">{escape(x_label)}</text>
<text x="24" y="{(top + x_axis_y)/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 24 {(top + x_axis_y)/2})" fill="#243033">{escape(y_label)}</text>
{''.join(elements)}
{''.join(legend_rows)}
{''.join(note_elements)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _write_logging_power_law_chart(
    scatter_series: Mapping[str, Sequence[tuple[float, float]]],
    fit_series: Mapping[str, tuple[Mapping[str, float | str | None], Sequence[tuple[float, float]]]],
    path: Path,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 560
    left, right, top, bottom = 92, 320, 58, 76
    colors = {
        "dashboard_plant_noise_free": "#2f6f73",
        "dashboard_plant_sensor_noise": "#b35f2e",
        "dashboard_noise_free": "#2f6f73",
        "dashboard_sensor_noise": "#b35f2e",
        "paper_noise_free": "#5954a4",
        "paper_sensor_noise": "#7b6b3a",
    }
    all_points = [
        point
        for values in scatter_series.values()
        for point in values
        if point[0] > 0.0 and point[1] > 0.0
    ]
    all_fit_points = [
        point
        for _, values in fit_series.values()
        for point in values
        if point[0] > 0.0 and point[1] > 0.0
    ]
    points = all_points + all_fit_points
    if not points:
        return ""

    log_x_values = [math.log10(point[0]) for point in points]
    log_y_values = [math.log10(point[1]) for point in points]
    x_lo, x_hi = min(log_x_values), max(log_x_values)
    y_lo, y_hi = min(log_y_values), max(log_y_values)
    x_pad = max(0.08, (x_hi - x_lo) * 0.06)
    y_pad = max(0.08, (y_hi - y_lo) * 0.08)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad

    def scale_x(x_value: float) -> float:
        return left + (math.log10(x_value) - x_lo) * (width - left - right) / (x_hi - x_lo)

    def scale_y(y_value: float) -> float:
        return height - bottom - (math.log10(y_value) - y_lo) * (height - top - bottom) / (y_hi - y_lo)

    x_axis_y = height - bottom
    plot_right = width - right
    x_ticks = _log_tick_values(x_lo, x_hi, limit=8)
    y_ticks = _log_tick_values(y_lo, y_hi, limit=8)
    grid_elements: list[str] = []
    for tick in x_ticks:
        x_pos = scale_x(tick)
        grid_elements.append(
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{x_axis_y}" stroke="#dedbce" stroke-width="1" />'
        )
        grid_elements.append(
            f'<line x1="{x_pos:.1f}" y1="{x_axis_y}" x2="{x_pos:.1f}" y2="{x_axis_y + 6}" stroke="#445" stroke-width="1" />'
        )
        grid_elements.append(
            f'<text x="{x_pos:.1f}" y="{x_axis_y + 24}" font-size="12" font-family="Arial" text-anchor="middle" fill="#34463f">{_format_axis_tick(tick)}</text>'
        )
    for tick in y_ticks:
        y_pos = scale_y(tick)
        grid_elements.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{plot_right}" y2="{y_pos:.1f}" stroke="#dedbce" stroke-width="1" />'
        )
        grid_elements.append(
            f'<line x1="{left - 6}" y1="{y_pos:.1f}" x2="{left}" y2="{y_pos:.1f}" stroke="#445" stroke-width="1" />'
        )
        grid_elements.append(
            f'<text x="{left - 10}" y="{y_pos + 4:.1f}" font-size="12" font-family="Arial" text-anchor="end" fill="#34463f">{_format_axis_tick(tick)}</text>'
        )

    elements: list[str] = []
    for key, values in scatter_series.items():
        color = colors.get(key, "#33435a")
        if key.startswith("dashboard_plant"):
            radius = 2.3
            opacity = 0.3
        elif key.startswith("dashboard"):
            radius = 5.0
            opacity = 0.96
        else:
            radius = 2.5
            opacity = 0.38
        for x_value, y_value in values:
            if x_value <= 0.0 or y_value <= 0.0:
                continue
            elements.append(
                f'<circle cx="{scale_x(x_value):.1f}" cy="{scale_y(y_value):.1f}" r="{radius}" fill="{color}" opacity="{opacity}" />'
            )

    legend_rows: list[str] = []
    legend_y = top + 8
    for index, (key, (fit, values)) in enumerate(fit_series.items()):
        color = colors.get(key, "#33435a")
        path_data = " ".join(
            f"{scale_x(x_value):.1f},{scale_y(y_value):.1f}"
            for x_value, y_value in values
            if x_value > 0.0 and y_value > 0.0
        )
        if path_data:
            elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{path_data}" />')
        y_pos = legend_y + index * 28
        legend_rows.append(
            f'<g transform="translate({width-right+24},{y_pos})">'
            f'<rect width="18" height="4" y="8" fill="{color}" />'
            f'<text x="28" y="14" font-size="13" font-family="Arial" fill="#243033">{str(fit["label"])}</text>'
            f'</g>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="30" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">Fig. S2 | Noise-free tau_min-normalized power law</text>
<text x="{left}" y="50" font-size="13" font-family="Arial" fill="#6f6a5d">The sensor-noise branch is deliberately not overlaid</text>
{''.join(grid_elements)}
<line x1="{left}" y1="{x_axis_y}" x2="{width-right}" y2="{x_axis_y}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{x_axis_y}" stroke="#445" stroke-width="1.5"/>
<text x="{(left + width - right)/2}" y="{height-18}" font-size="15" font-family="Arial" text-anchor="middle" fill="#243033">Tlog/tau_min (log scale)</text>
<text x="24" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 24 {height/2})" fill="#243033">MARE_theta % (log scale)</text>
{''.join(elements)}
{''.join(legend_rows)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _write_logging_report(
    payload: Mapping[str, object],
    plot_path: str,
    power_law_plot_path: str,
    csv_path: str,
    speed_plot_path: str | None = None,
) -> str:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = SUMMARY_DIR / "logging_rate_report_restored.md"
    fits = payload.get("power_law_fits", [])
    fit_rows = []
    if isinstance(fits, Sequence):
        for fit in fits:
            if not isinstance(fit, Mapping):
                continue
            alpha = fit.get("alpha")
            r_square = fit.get("r_square")
            coefficient = fit.get("a")
            fit_rows.append(
                "| {label} | {case} | {source} | {branch} | {a} | {alpha} | {r2} | {points} |".format(
                    label=fit.get("label", ""),
                    case=fit.get("case", ""),
                    source=fit.get("source", ""),
                    branch=fit.get("branch", ""),
                    a="n/a" if coefficient is None else f"{float(coefficient):.6g}",
                    alpha="n/a" if alpha is None else f"{float(alpha):.6g}",
                    r2="n/a" if r_square is None else f"{float(r_square):.6g}",
                    points=fit.get("points", ""),
                )
            )

    content = [
        "# Logging Adequacy Report",
        "",
        f"- Tlog values: {', '.join(str(value) for value in payload.get('tlog_ms_values', []))} ms",
        f"- tmin / tau_min: {payload.get('tmin_ms')} ms",
        f"- Measurement conditions: {', '.join(str(item) for item in payload.get('measurement_conditions', []))}",
        f"- Dual-channel optimum Tlog: {payload.get('best_noisy_Tlog_ms')} ms",
        f"- Dual-channel optimum MARE_theta: {payload.get('best_noisy_MARE_theta_percent')}%",
        f"- Dual-channel optimum Tlog/tmin: {payload.get('best_noisy_tau_ratio')}",
        f"- Dual-channel optimum tau_min/Tlog: {payload.get('best_noisy_tau_min_over_tlog')}",
        f"- Tension-only best Tlog: {payload.get('best_tension_only_Tlog_ms')} ms "
        "(no interior optimum; the finest setting always wins)",
        f"- CSV summary: `{Path(csv_path).relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Fig. 2(a) | Three measurement conditions",
        "",
        f"![Logging period under three measurement conditions](../figures/{Path(plot_path).name})",
        "",
        *(
            [
                "## Fig. 2(b) | Dual-channel speed decomposition",
                "",
                f"![Dual-channel decomposed by line speed](../figures/{Path(speed_plot_path).name})",
                "",
                "The pooled valley represents neither sub-group and is never reported without this split.",
                "",
            ]
            if speed_plot_path
            else []
        ),
        "## Fig. S2 | Noise-free power law",
        "",
        f"![Noise-free tau_min-normalized power law](../figures/{Path(power_law_plot_path).name})",
        "",
        str(payload.get("power_law_sensor_noise_caveat", "")),
        "",
        "## Power-Law Fits",
        "",
        "| Fit | Case | Source | Branch | a | alpha | R^2 | Points |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        *fit_rows,
        "",
        "Model form: `MARE_theta_percent = a * (Tlog/tau_min)^alpha`.",
        "The power law is fitted on the noise-free branch only, per Fig. S2.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return str(path)


def _reference_for_tlog(
    reference: Mapping[str, Mapping[float, Mapping[str, float]]],
    case_name: str,
    tlog_ms: float,
) -> Mapping[str, float] | None:
    case_reference = reference.get(case_name, {})
    for reference_tlog, values in case_reference.items():
        if math.isclose(reference_tlog, tlog_ms, rel_tol=0.0, abs_tol=1e-9):
            return values
    return None


def _predict_power_law_mare_percent(
    fit: Mapping[str, float | str | None],
    tau_ratio: float,
) -> float | None:
    coefficient = fit.get("a")
    alpha = fit.get("alpha")
    if coefficient is None or alpha is None or tau_ratio <= 0.0:
        return None
    return float(coefficient) * (float(tau_ratio) ** float(alpha))


def _rebase_cached_logging_artifact_paths(payload: dict[str, object]) -> bool:
    """Resolve copied-cache artifacts inside this project, never the source project."""

    artifact_roots = {
        "plot_path": FIGURES_DIR,
        "power_law_plot_path": FIGURES_DIR,
        "speed_plot_path": FIGURES_DIR,
        "csv_path": SUMMARY_DIR,
        "markdown_path": SUMMARY_DIR,
        "speed_csv_path": SUMMARY_DIR,
        "graph_points_csv_path": SUMMARY_DIR,
        "graph_points_xlsx_path": SUMMARY_DIR,
    }
    rebased_paths: dict[str, str] = {}
    for key, root in artifact_roots.items():
        source_value = payload.get(key)
        filename = Path(str(source_value or "")).name
        if not filename:
            return False
        local_path = (root / filename).resolve()
        if not local_path.is_file():
            return False
        rebased_paths[key] = str(local_path)
    payload.update(rebased_paths)
    return True


def _cached_logging_rate_study(
    values: Sequence[float],
    tmin_ms: float,
    active_plant_runs: Sequence[tuple[str, R2RParameters, Mapping[str, Any]]],
) -> dict[str, object] | None:
    summary_paths = (
        [
            SUMMARY_DIR / "logging_rate_summary_all_plants_restored.json",
            SUMMARY_DIR / "logging_rate_summary_restored.json",
        ]
        if len(active_plant_runs) > 1
        else [SUMMARY_DIR / "logging_rate_summary_restored.json"]
    )
    for summary_path in summary_paths:
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        required_keys = (
            "metrics",
            "run_metadata",
            "plot_path",
            "csv_path",
            "markdown_path",
            "power_law_plot_path",
            "speed_plot_path",
            "speed_csv_path",
            "graph_points_csv_path",
            "graph_points_xlsx_path",
        )
        if any(key not in payload for key in required_keys):
            continue
        if payload.get("plant_scope") != ("all_plants" if len(active_plant_runs) > 1 else "single_plant"):
            continue
        if payload.get("logging_data_source") != "dashboard_simulation":
            continue
        if payload.get("calculation_version") != LOGGING_RATE_CACHE_VERSION:
            continue

        payload_tlogs = [float(value) for value in payload.get("tlog_ms_values", [])]
        if len(payload_tlogs) != len(values) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(sorted(payload_tlogs), sorted(float(value) for value in values), strict=True)
        ):
            continue
        if not math.isclose(float(payload.get("tmin_ms", 0.0)), float(tmin_ms), rel_tol=0.0, abs_tol=1e-9):
            continue

        cached_plant_ids = [str(plant_id) for plant_id in payload.get("plant_ids", [])]
        active_plant_ids = [str(plant_id) for plant_id, _, _ in active_plant_runs]
        if cached_plant_ids != active_plant_ids:
            continue
        if not _rebase_cached_logging_artifact_paths(payload):
            continue

        artifact = _artifact_payload(
            payload,
            str(payload.get("plot_path") or ""),
            str(summary_path),
            str(payload.get("csv_path") or ""),
            str(payload.get("markdown_path") or ""),
        )
        artifact["power_law_plot_path"] = payload["power_law_plot_path"]
        artifact["speed_plot_path"] = payload["speed_plot_path"]
        artifact["speed_csv_path"] = payload["speed_csv_path"]
        artifact["graph_points_csv_path"] = payload["graph_points_csv_path"]
        artifact["graph_points_xlsx_path"] = payload["graph_points_xlsx_path"]
        return artifact
    return None


def logging_rate_study(
    tlog_ms_values: Sequence[float] | None = None,
    tmin_ms: float = DEFAULT_TMIN_MS,
    params: R2RParameters | None = None,
    plant_runs: Sequence[tuple[str, R2RParameters, Mapping[str, Any]]] | None = None,
    prefer_cache: bool = False,
) -> dict[str, object]:
    """Sweep logging rates and compare SysID MARE for noise-free and noisy sensors."""

    if not math.isfinite(float(tmin_ms)) or float(tmin_ms) <= 0.0:
        raise ValueError("tmin_ms must be finite and positive.")
    active_tmin_ms = float(tmin_ms)
    values = _normalise_tlog_values(tlog_ms_values)
    if plant_runs:
        active_plant_runs = [
            (str(plant_id), plant_params, dict(plant_meta))
            for plant_id, plant_params, plant_meta in plant_runs
        ]
    else:
        active_params = params or R2RParameters()
        active_plant_runs = [
            (
                "selected",
                active_params,
                {
                    "plant_id": "selected",
                    "label": "Selected plant",
                    "EA_N": active_params.EA,
                },
            )
        ]
    if not active_plant_runs:
        raise ValueError("At least one plant is required.")

    if prefer_cache:
        cached = _cached_logging_rate_study(values, active_tmin_ms, active_plant_runs)
        if cached is not None:
            return cached

    is_aggregate = len(active_plant_runs) > 1
    aggregation = "median" if is_aggregate else "single"
    run_metadata = _run_metadata(
        "logging-rate",
        plant_scope="all_plants" if is_aggregate else "single_plant",
        run_settings={
            "assumed_tmin_ms": active_tmin_ms,
            "tau_min_source": "configured_value",
            "sensor_noise_lpf_hz": LOGGING_SN_LPF_HZ,
            "sensor_noise_full_scale_percent": SENSOR_NOISE_FULL_SCALE_PERCENT,
            "measurement_conditions": list(LOGGING_CONDITIONS),
            "sensor_noise_channels": "tension_only_and_dual_channel",
            "sensor_noise_omega_rad_s": SENSOR_NOISE_OMEGA_RAD_S,
            "velocity_noise_percent_full_scale": EXCITATION_SN_VELOCITY_PERCENT,
            "dual_channel_seeds": list(LOGGING_DUAL_CHANNEL_SEEDS),
            "tlog_ms_values": values,
            "sysid_estimator": "paper_eq8_weighted_pem_trf",
            "controller_mode": "paper_aligned",
            "controller_integral_time": "per_plant_auto_Ti",
            "controller_sample_time_s": 0.001,
            "noise_affects_controller": True,
            "noise_free_duration_s": LOGGING_NF_DURATION_S,
            "sensor_noise_duration_s": LOGGING_SN_DURATION_S,
            "sensor_noise_amplitude_factor": LOGGING_SN_AMPLITUDE_FACTOR,
            "velocity_observer_clip_fraction": LOGGING_VELOCITY_OBSERVER_CLIP_FRACTION,
            "condition_excitation": dict(LOGGING_CONDITION_EXCITATION),
            "condition_campaign_group": dict(LOGGING_CONDITION_CAMPAIGN_GROUP),
            "record_duration_source": "excitation_schedules.csv",
            "dual_channel_velocity_seed_offset": COMPOSITE_SEED_V_OFFSET,
            "paper_inputs": paper_input_provenance(),
        },
    )
    reference_rows = _load_logging_reference_rows()
    reference = _summarise_logging_reference(reference_rows)
    reference_annotations = _logging_reference_annotations()
    speed_groups = _logging_speed_groups()
    paper_points_by_condition = {
        condition: [
            (float(row["tau_ratio"]), float(row["MARE_theta_percent"]))
            for row in reference_rows
            if row["case"] == condition
        ]
        for condition in LOGGING_CONDITIONS
    }
    # Fig. S2 is the tau_min-normalized NOISE-FREE power law. The sensor-noise
    # branch is deliberately not fitted or overlaid: under sensor noise the error
    # does not collapse onto the power law (fitted exponent +0.41, R^2 ~ 0.11),
    # so an SN line would contradict its own trend.
    paper_reference_fits = {
        "noise_free": _fit_power_law_with_reference_alpha(
            "Paper NF (MARE)",
            paper_points_by_condition["noise_free"],
            source="paper_reference",
            case_name="noise_free",
            branch="all_runs",
        ),
    }

    def _median_or_none(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
        values_for_key = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]
        return statistics.median(values_for_key) if values_for_key else None

    def _reference_fields(
        case_name: str,
        tlog_ms: float,
        mare_percent: float | None,
        tau_ratio_override: float | None = None,
    ) -> dict[str, float | str | bool | None]:
        tau_ratio = float(tau_ratio_override) if tau_ratio_override is not None else tlog_ms / active_tmin_ms
        effective_tmin_ms = tlog_ms / tau_ratio if tau_ratio > 0.0 else active_tmin_ms
        tau_min_over_tlog = 1.0 / tau_ratio if tau_ratio > 0.0 else None
        reference_row = _reference_for_tlog(reference, case_name, tlog_ms)
        # Only the noise-free branch has a published power law (Fig. S2).
        reference_fit = paper_reference_fits.get(case_name) or {}
        reference_power_law_mare = (
            _predict_power_law_mare_percent(reference_fit, tau_ratio) if reference_fit else None
        )
        paper_median = float(reference_row["paper_median_MARE_theta_percent"]) if reference_row else None
        paper_mean = float(reference_row["paper_mean_MARE_theta_percent"]) if reference_row else None
        return {
            "tmin_ms": effective_tmin_ms,
            "assumed_tmin_ms": effective_tmin_ms,
            "tau_min_source": "configured_value",
            "tau_ratio": tau_ratio,
            "tau_min_over_tlog": tau_min_over_tlog,
            "nf_guideline_met": tau_min_over_tlog >= 5.0 if tau_min_over_tlog is not None else False,
            "reference_power_law_MARE_theta_percent": reference_power_law_mare,
            "reference_power_law_MARE_theta": (
                reference_power_law_mare / 100.0 if reference_power_law_mare is not None else None
            ),
            "reference_power_law_a": float(reference_fit["a"]) if reference_fit.get("a") is not None else None,
            "reference_power_law_alpha": (
                float(reference_fit["alpha"]) if reference_fit.get("alpha") is not None else None
            ),
            "paper_numerical_MARE_theta_percent": paper_median,
            "paper_numerical_MARE_theta": paper_median / 100.0 if paper_median is not None else None,
            "paper_numerical_source": "paper_reference_median" if paper_median is not None else None,
            "paper_numerical_difference_MARE_theta_percent": (
                mare_percent - paper_median if mare_percent is not None and paper_median is not None else None
            ),
            "measurement_condition": case_name,
            "measurement_condition_label": LOGGING_CONDITION_LABELS.get(case_name, case_name),
            "paper_value_status": (
                "published"
                if paper_median is not None
                else "off_scale_below_0p01_percent"
                if case_name == "noise_free" and math.isclose(tlog_ms, 1.0, rel_tol=0.0, abs_tol=1e-9)
                else "not_published_at_this_Tlog"
            ),
            # v5 recommends Tlog = 5-20 ms under DUAL-CHANNEL noise. Tension-only
            # noise creates no interior optimum at all; its best setting is the
            # finest 1 ms one, so the window claim does not apply to it.
            "supports_5_20ms_window": bool(
                case_name == "dual_channel"
                and any(
                    math.isclose(tlog_ms, value, rel_tol=0.0, abs_tol=1e-9)
                    for value in (5.0, 10.0, 20.0)
                )
            ),
            "paper_median_MARE_theta_percent": paper_median,
            "paper_mean_MARE_theta_percent": paper_mean,
            "paper_delta_median_MARE_theta_percent": (
                mare_percent - paper_median if mare_percent is not None and paper_median is not None else None
            ),
            "paper_abs_delta_median_MARE_theta_percent": (
                abs(mare_percent - paper_median)
                if mare_percent is not None and paper_median is not None
                else None
            ),
            "paper_reference_samples": float(reference_row["paper_reference_samples"]) if reference_row else None,
        }

    # One job per (plant, condition, Tlog, seed). Only the dual-channel case is
    # seeded more than once: velocity noise is what creates the interior optimum,
    # so a single realization is not enough to place it.
    jobs: list[tuple[str, R2RParameters, Mapping[str, Any], str, float, int]] = []
    for plant_id, plant_params, plant_meta in active_plant_runs:
        for case_name in LOGGING_CONDITIONS:
            seeds = LOGGING_DUAL_CHANNEL_SEEDS if case_name == "dual_channel" else (0,)
            for tlog_ms in values:
                for seed in seeds:
                    jobs.append((plant_id, plant_params, plant_meta, case_name, tlog_ms, seed))

    def _run_logging_job(
        job: tuple[str, R2RParameters, Mapping[str, Any], str, float, int],
    ) -> dict[str, float | str | bool | None]:
        plant_id, plant_params, plant_meta, case_name, tlog_ms, seed = job
        tlog_name = _format_tlog_for_name(tlog_ms)
        safe_plant_id = "".join(char if char.isalnum() else "_" for char in plant_id)
        plant_noise_sigma = float(plant_meta.get("sensor_noise_sigma_N", SENSOR_NOISE_TENSION_N))
        line_speed_m_s = float(plant_meta.get("v_ref_m_s", plant_params.feeder_velocity_m_s))
        sensor_noise = resolve_sensor_noise(
            case_name,
            tension_max_N=float(plant_meta.get("T_max_N", SENSOR_NOISE_TENSION_FULL_SCALE_N)),
            line_speed_m_s=line_speed_m_s,
            tension_percent=SENSOR_NOISE_FULL_SCALE_PERCENT,
            velocity_percent=EXCITATION_SN_VELOCITY_PERCENT,
            # The per-plant tension sigma is already calibrated in the plant
            # registry; only the velocity channel is derived from v_max here.
            tension_sigma_override_N=plant_noise_sigma,
        )
        active_noise_tension = sensor_noise.tension_sigma_N
        active_noise_velocity = sensor_noise.velocity_sigma_m_s
        active_sensor_lpf_hz = None if case_name == "noise_free" else LOGGING_SN_LPF_HZ
        observer_clip_fraction = LOGGING_VELOCITY_OBSERVER_CLIP_FRACTION
        base_excitation_amplitude = float(
            plant_meta.get("recommended_excitation_amplitude_V", 0.2 * float(plant_meta.get("T_ref_N", 1.0)))
        )
        excitation_amplitude = (
            base_excitation_amplitude
            if case_name == "noise_free"
            else LOGGING_SN_AMPLITUDE_FACTOR * base_excitation_amplitude
        )
        excitation_name = LOGGING_CONDITION_EXCITATION[case_name]
        campaign_group = LOGGING_CONDITION_CAMPAIGN_GROUP[case_name]
        excitation_profile = get_excitation_profile(
            excitation_name,
            excitation_amplitude,
            campaign_group=campaign_group,
        )
        excitation_schedule_record = getattr(excitation_profile, "schedule", None)
        controller_config = ControllerConfig(
            line_speed_m_s=line_speed_m_s,
            target_tension_N=plant_params.tension_ref_N,
            TI_s=auto_tension_integral_time_s(plant_params, line_speed_m_s),
            # The paper runs every plant at K_p* = 100. Leaving this flag at its
            # default (True) silently drops K_p* to 20 for EA >= 300 kN and to 5
            # for EA >= 400 kN, which is five of the ten plants -- a different
            # controller from both the paper and the excitation campaign, which
            # disables the cap explicitly.
            high_ea_kp_cap_enabled=False,
            feedforward_uses_measured_omega=True,
            paper_velocity_gain_enabled=True,
            velocity_correction_limit_fraction=None,
            steady_velocity_uses_dynamic_target=False,
        )
        fallback_duration_s = LOGGING_NF_DURATION_S if case_name == "noise_free" else LOGGING_SN_DURATION_S
        config = SimulationConfig(
            duration_s=float(getattr(excitation_profile, "duration_s", fallback_duration_s)),
            controller_sample_time_s=0.001,
            log_sample_time_s=tlog_ms / 1000.0,
            line_speed_m_s=line_speed_m_s,
            sensor_noise_tension_N=active_noise_tension,
            sensor_noise_velocity_m_s=active_noise_velocity,
            sensor_noise_omega_rad_s=SENSOR_NOISE_OMEGA_RAD_S,
            sensor_lpf_hz=float(active_sensor_lpf_hz) if active_sensor_lpf_hz is not None else None,
            # The plant runs closed-loop on the same filtered measurement the
            # identification sees, as in the Excitation and Drift campaigns.
            noise_affects_controller=True,
            output_name=f"logging_rate_{safe_plant_id}_{case_name}_{tlog_name}ms_seed{seed}.csv",
            seed=seed,
            # Dual-channel runs draw the velocity channel from its own generator
            # at seed_v = seed_T + 100 (`excitation_schedules.csv`, "Seed
            # conventions"). Sharing one interleaved stream makes the tension
            # realisation depend on whether the velocity channel is switched on,
            # so the tension-only and dual-channel cells of the same seed stop
            # being the same tension experiment.
            velocity_seed_offset=(
                COMPOSITE_SEED_V_OFFSET if case_name == "dual_channel" else None
            ),
        )
        try:
            sim = simulate(
                plant_params,
                controller_config=controller_config,
                config=config,
                excitation=excitation_profile,
                output_dir=DATA_DIR,
            )
            sysid_rows = _bounded_velocity_observer_rows(
                sim.rows,
                plant_params,
                float(observer_clip_fraction) if observer_clip_fraction is not None else None,
            )
            sysid = estimate_parameters_weighted_pem(
                sysid_rows,
                nominal_params=plant_params,
                true_params=plant_params,
            )
            mare_theta: float | None = sysid.mare_theta
            mare_percent: float | None = 100.0 * sysid.mare_theta
            samples = float(len(sim.rows))
            value_status = "computed_raw"
        except Exception as exc:
            mare_theta = None
            mare_percent = None
            samples = 0.0
            value_status = f"simulation_unstable:{type(exc).__name__}"
        row: dict[str, float | str | bool | None] = {
            "case": case_name,
            "plant_id": plant_id,
            "plant_label": str(plant_meta.get("label", plant_id)),
            "EA_N": float(plant_meta.get("EA_N", plant_params.EA)),
            "Tlog_ms": float(tlog_ms),
            "seed": seed,
            "MARE_theta": mare_theta,
            "MARE_theta_percent": mare_percent,
            "samples": samples,
            "excitation": excitation_name,
            "excitation_campaign_group": (
                excitation_schedule_record.campaign_group
                if excitation_schedule_record is not None
                else campaign_group
            ),
            "excitation_record_duration_s": float(config.duration_s),
            "velocity_seed": (
                seed + COMPOSITE_SEED_V_OFFSET if case_name == "dual_channel" else None
            ),
            "sensor_noise_sigma_N": active_noise_tension,
            "sensor_noise_velocity_sigma_m_s": active_noise_velocity,
            "sensor_lpf_hz": active_sensor_lpf_hz,
            "velocity_observer_clip_fraction": observer_clip_fraction,
            "v_ref_m_s": line_speed_m_s,
            "speed_group": _logging_speed_group_for_plant(plant_id, speed_groups),
            "value_status": value_status,
        }
        row.update(_reference_fields(case_name, float(tlog_ms), mare_percent))
        return row

    interpolated_tlog_values: list[float] = []
    if len(jobs) > 1:
        plant_metric_rows = []
        with ThreadPoolExecutor(max_workers=min(10, len(jobs))) as executor:
            futures = [executor.submit(_run_logging_job, job) for job in jobs]
            for future in as_completed(futures):
                plant_metric_rows.append(future.result())
    else:
        plant_metric_rows = [_run_logging_job(job) for job in jobs]

    case_order = {case_name: index for index, case_name in enumerate(LOGGING_CONDITIONS)}
    plant_order = {plant_id: index for index, (plant_id, _, _) in enumerate(active_plant_runs)}
    plant_metric_rows.sort(
        key=lambda row: (
            case_order[str(row["case"])],
            float(row["Tlog_ms"]),
            plant_order.get(str(row["plant_id"]), 999),
            int(row.get("seed") or 0),
        )
    )
    for metric_row in plant_metric_rows:
        metric_row.update(_metadata_row_fields(run_metadata, str(metric_row.get("value_status") or "computed_raw")))

    def _aggregate_metric_row(
        case_name: str,
        tlog_ms: float,
        grouped_rows: Sequence[Mapping[str, object]],
        *,
        speed_group: str | None = None,
    ) -> dict[str, float | str | bool | None]:
        median_mare = _median_or_none(grouped_rows, "MARE_theta")
        median_mare_percent = _median_or_none(grouped_rows, "MARE_theta_percent")
        median_samples = _median_or_none(grouped_rows, "samples")
        median_tau_ratio = _median_or_none(grouped_rows, "tau_ratio")
        finite_mare_percent_values = [
            float(row["MARE_theta_percent"])
            for row in grouped_rows
            if row.get("MARE_theta_percent") is not None
            and math.isfinite(float(row["MARE_theta_percent"]))
        ]
        plant_ids = sorted({str(row["plant_id"]) for row in grouped_rows})
        metric_row: dict[str, float | str | bool | None] = {
            "case": case_name,
            "plant_id": "ALL",
            "aggregation": "median",
            "plant_count": float(len(plant_ids)),
            "run_count": float(len(grouped_rows)),
            "valid_run_count": float(len(finite_mare_percent_values)),
            "failed_run_count": float(len(grouped_rows) - len(finite_mare_percent_values)),
            "plant_ids": ", ".join(plant_ids),
            "seed_count": float(len({int(row.get("seed") or 0) for row in grouped_rows})),
            "Tlog_ms": float(tlog_ms),
            "MARE_theta": median_mare,
            "MARE_theta_percent": median_mare_percent,
            "MARE_theta_percent_min": min(finite_mare_percent_values) if finite_mare_percent_values else None,
            "MARE_theta_percent_max": max(finite_mare_percent_values) if finite_mare_percent_values else None,
            "samples": median_samples,
            "data_source": "dashboard_simulation",
        }
        if speed_group is not None:
            metric_row["speed_group"] = speed_group
        metric_row.update(_reference_fields(case_name, float(tlog_ms), median_mare_percent, median_tau_ratio))
        return metric_row

    metrics: list[dict[str, float | str | bool | None]] = []
    for case_name in LOGGING_CONDITIONS:
        for tlog_ms in values:
            grouped_rows = [
                row
                for row in plant_metric_rows
                if row["case"] == case_name and math.isclose(float(row["Tlog_ms"]), tlog_ms, rel_tol=0.0, abs_tol=1e-9)
            ]
            if not grouped_rows:
                continue
            if is_aggregate or len(grouped_rows) > 1:
                metric_row = _aggregate_metric_row(case_name, float(tlog_ms), grouped_rows)
            else:
                metric_row = {**grouped_rows[0], "aggregation": "single", "plant_count": 1.0}
            metric_row.update(_metadata_row_fields(run_metadata, str(metric_row.get("value_status") or "computed_raw")))
            metrics.append(metric_row)

    # Fig. 2(b): the dual-channel case split by line speed. The pooled valley
    # represents neither sub-group, so the paper never reports it without this
    # decomposition and neither does the dashboard.
    speed_metrics: dict[str, list[dict[str, float | str | bool | None]]] = {}
    for group_key in ("slow", "fast"):
        group_rows: list[dict[str, float | str | bool | None]] = []
        for tlog_ms in values:
            grouped_rows = [
                row
                for row in plant_metric_rows
                if row["case"] == "dual_channel"
                and row.get("speed_group") == group_key
                and math.isclose(float(row["Tlog_ms"]), tlog_ms, rel_tol=0.0, abs_tol=1e-9)
            ]
            if not grouped_rows:
                continue
            group_row = _aggregate_metric_row(
                "dual_channel", float(tlog_ms), grouped_rows, speed_group=group_key
            )
            group_row.update(
                _metadata_row_fields(run_metadata, str(group_row.get("value_status") or "computed_raw"))
            )
            group_rows.append(group_row)
        speed_metrics[group_key] = group_rows

    def _condition_points(rows, condition, x_key, y_key):
        return [
            (float(row[x_key]), float(row[y_key]))
            for row in rows
            if row["case"] == condition
            and row.get(x_key) is not None
            and row.get(y_key) is not None
            and math.isfinite(float(row[x_key]))
            and math.isfinite(float(row[y_key]))
        ]

    dashboard_points_by_condition = {
        condition: _condition_points(metrics, condition, "tau_ratio", "MARE_theta_percent")
        for condition in LOGGING_CONDITIONS
    }

    # Fig. S2 fits the noise-free branch only. The sensor-noise branches are
    # carried as numerical curves, never as a power law: the reference states
    # explicitly that an SN fit would contradict its own trend.
    dashboard_noise_free_power_law_fit = _fit_power_law(
        "Median NF" if is_aggregate else "Dashboard NF",
        dashboard_points_by_condition["noise_free"],
        source="dashboard_run",
        case_name="noise_free",
        branch="all_plant_median" if is_aggregate else "selected_Tlog",
    )
    power_law_fit_series = {
        "dashboard_noise_free": (
            dashboard_noise_free_power_law_fit,
            _sorted_positive_points(dashboard_points_by_condition["noise_free"]),
        ),
        "paper_noise_free": (
            paper_reference_fits["noise_free"],
            _fit_line_points(paper_reference_fits["noise_free"], paper_points_by_condition["noise_free"]),
        ),
    }
    numerical_curve_series = {}
    for condition in LOGGING_CONDITIONS:
        short_label = LOGGING_CONDITION_SHORT_LABELS[condition]
        numerical_curve_series[f"dashboard_{condition}"] = (
            _curve_metadata(
                f"{'Median' if is_aggregate else 'Dashboard'} {short_label}",
                dashboard_points_by_condition[condition],
                source="dashboard_run",
                case_name=condition,
                branch="numerical_rows",
                fit_note="numerical_logging_curve",
            ),
            _sorted_positive_points(dashboard_points_by_condition[condition]),
        )
        paper_curve_points = _condition_points(
            metrics, condition, "tau_ratio", "paper_numerical_MARE_theta_percent"
        )
        numerical_curve_series[f"paper_{condition}"] = (
            _curve_metadata(
                f"Paper {short_label} (MARE)",
                paper_curve_points,
                source="paper_reference",
                case_name=condition,
                branch="numerical_rows",
                fit_note="numerical_logging_curve",
            ),
            paper_curve_points,
        )
    power_law_fits = [fit for fit, _ in power_law_fit_series.values()]
    numerical_curve_fits = [fit for fit, _ in numerical_curve_series.values()]

    # The headline "best noisy setting" is the dual-channel one: tension-only
    # noise creates no interior optimum, so its minimum is always the finest
    # logging period and says nothing about a sweet spot.
    dual_channel_rows = [
        row
        for row in metrics
        if row["case"] == "dual_channel" and row.get("MARE_theta") is not None
    ]
    tension_only_rows = [
        row
        for row in metrics
        if row["case"] == "tension_only" and row.get("MARE_theta") is not None
    ]
    best_noisy = (
        min(dual_channel_rows, key=lambda row: float(row["MARE_theta"]))
        if dual_channel_rows
        else min(metrics, key=lambda row: float(row["MARE_theta"] or math.inf))
    )
    best_tension_only = (
        min(tension_only_rows, key=lambda row: float(row["MARE_theta"])) if tension_only_rows else None
    )
    noisy_reference_power_law_rows = [
        row for row in dual_channel_rows if row.get("reference_power_law_MARE_theta") is not None
    ]
    best_noisy_reference_power_law = (
        min(noisy_reference_power_law_rows, key=lambda row: float(row["reference_power_law_MARE_theta"]))
        if noisy_reference_power_law_rows
        else None
    )
    supports_window = float(best_noisy["Tlog_ms"]) in (5.0, 10.0, 20.0)

    # Fig. 2(a): epsilon_theta vs Tlog under the three measurement conditions,
    # dashboard (solid) against paper (dashed).
    condition_chart_series = []
    for condition in LOGGING_CONDITIONS:
        label = LOGGING_CONDITION_LABELS[condition]
        color = LOGGING_CONDITION_COLORS[condition]
        condition_chart_series.append(
            (
                f"Dashboard {label}",
                _condition_points(metrics, condition, "Tlog_ms", "MARE_theta_percent"),
                color,
                "",
            )
        )
        condition_chart_series.append(
            (
                f"Paper {label}",
                _condition_points(metrics, condition, "Tlog_ms", "paper_numerical_MARE_theta_percent"),
                color,
                "7 5",
            )
        )
    noise_free_anchor = (
        reference_annotations.get("E_Toggle_noise_free_anchors", {}).get("Tlog_1ms", {}) or {}
    )
    condition_annotations = [
        "Each paper series comes from a single excitation: noise-free from ET1 (Table S7), "
        "tension-only and dual-channel from E_Toggle (Fig. S6b and Fig. 2a).",
        "Noise-free at Tlog = 1 ms is off the logarithmic axis, not missing: "
        + str(noise_free_anchor.get("status", "an essentially exact fit below 0.01%.")),
        "Tension-only has no interior optimum; its best setting is the finest 1 ms one. "
        "The interior optimum belongs to the dual-channel case and is produced by the velocity channel.",
    ]
    plot_path = _write_logging_log_chart(
        condition_chart_series,
        FIGURES_DIR / "logging_rate_vs_mare_restored.svg",
        title="Fig. 2(a) | Logging period under three measurement conditions",
        subtitle="Dashboard result (solid) against the published v5 values (dashed)",
        x_label="Tlog (ms, log scale)",
        y_label="MARE_theta % (log scale)",
        annotations=condition_annotations,
    )

    # Fig. 2(b): the dual-channel case decomposed by line speed.
    speed_chart_series = []
    speed_bands = []
    for group_key, group_color in (("slow", "#2f6f73"), ("fast", "#b35f2e")):
        group = speed_groups.get(group_key, {})
        group_label = "Slow plants" if group_key == "slow" else "Fast plants"
        speed_chart_series.append(
            (
                f"Dashboard {group_label}",
                [
                    (float(row["Tlog_ms"]), float(row["MARE_theta_percent"]))
                    for row in speed_metrics.get(group_key, [])
                    if row.get("MARE_theta_percent") is not None
                    and math.isfinite(float(row["MARE_theta_percent"]))
                ],
                group_color,
                "",
            )
        )
        band = group.get("paper_band_percent")
        if isinstance(band, list | tuple) and len(band) == 2:
            speed_bands.append(
                (f"Paper {group_key} band ({group.get('paper_band_meaning', 'band')})", list(band), group_color)
            )
    speed_chart_series.append(
        (
            "Dashboard pooled (all 10 plants)",
            [
                (float(row["Tlog_ms"]), float(row["MARE_theta_percent"]))
                for row in metrics
                if row["case"] == "dual_channel"
                and row.get("MARE_theta_percent") is not None
                and math.isfinite(float(row["MARE_theta_percent"]))
            ],
            "#5954a4",
            "3 4",
        )
    )
    speed_plot_path = _write_logging_log_chart(
        speed_chart_series,
        FIGURES_DIR / "logging_rate_speed_decomposition.svg",
        title="Fig. 2(b) | Dual-channel case decomposed by line speed",
        subtitle="The pooled valley represents neither sub-group, so it is never read on its own",
        x_label="Tlog (ms, log scale)",
        y_label="MARE_theta % (log scale)",
        bands=speed_bands,
        annotations=[
            f"Slow: {speed_groups.get('slow', {}).get('definition', '')} | "
            f"{', '.join(speed_groups.get('slow', {}).get('plants', []))}",
            f"Fast: {speed_groups.get('fast', {}).get('definition', '')} | "
            f"{', '.join(speed_groups.get('fast', {}).get('plants', []))}",
            str(speed_groups.get("fast", {}).get("note", "")),
        ],
    )
    csv_fields = [
        *METADATA_FIELDS,
        "case",
        "measurement_condition",
        "measurement_condition_label",
        "plant_id",
        "aggregation",
        "plant_count",
        "run_count",
        "valid_run_count",
        "failed_run_count",
        "seed_count",
        "plant_ids",
        "Tlog_ms",
        "tmin_ms",
        "assumed_tmin_ms",
        "tau_min_source",
        "tau_ratio",
        "tau_min_over_tlog",
        "nf_guideline_met",
        "MARE_theta",
        "MARE_theta_percent",
        "reference_power_law_MARE_theta",
        "reference_power_law_MARE_theta_percent",
        "reference_power_law_a",
        "reference_power_law_alpha",
        "paper_numerical_MARE_theta",
        "paper_numerical_MARE_theta_percent",
        "paper_numerical_source",
        "paper_numerical_difference_MARE_theta_percent",
        "samples",
        "sensor_noise_sigma_N",
        "sensor_noise_velocity_sigma_m_s",
        "sensor_lpf_hz",
        "velocity_observer_clip_fraction",
        "supports_5_20ms_window",
        "paper_value_status",
        "paper_median_MARE_theta_percent",
        "paper_mean_MARE_theta_percent",
        "paper_delta_median_MARE_theta_percent",
        "paper_abs_delta_median_MARE_theta_percent",
        "paper_reference_samples",
    ]
    csv_path = _write_rows_csv("logging_rate_summary_restored.csv", metrics, csv_fields)
    graph_points_csv_path = _write_graph_points_csv("logging_rate_graph_points_restored.csv", metrics, run_metadata)
    graph_points_xlsx_path = _write_graph_points_xlsx("logging_rate_graph_points_restored.xlsx", metrics, run_metadata)
    speed_rows = [row for group_rows in speed_metrics.values() for row in group_rows]
    speed_csv_path = (
        _write_rows_csv(
            "logging_rate_speed_decomposition.csv",
            speed_rows,
            ["speed_group", *csv_fields],
        )
        if speed_rows
        else None
    )
    power_law_plot_path = _write_logging_power_law_chart(
        {},
        power_law_fit_series,
        FIGURES_DIR / "logging_rate_power_law_restored.svg",
    )
    logging_data_source = "dashboard_simulation"
    payload = {
        "study": "logging-rate",
        "calculation_version": LOGGING_RATE_CACHE_VERSION,
        "metrics": metrics,
        "run_metadata": run_metadata,
        "plant_scope": "all_plants" if is_aggregate else "single_plant",
        "aggregation": aggregation,
        "plant_count": len(active_plant_runs),
        "plant_ids": [plant_id for plant_id, _, _ in active_plant_runs],
        "plant_metrics": plant_metric_rows,
        "tmin_ms": active_tmin_ms,
        "tlog_ms_values": values,
        "logging_data_source": logging_data_source,
        "interpolated_tlog_ms_values": interpolated_tlog_values,
        "sensor_noise_full_scale_percent": SENSOR_NOISE_FULL_SCALE_PERCENT,
        "sensor_noise_full_scale_values": {
            "tension_N": SENSOR_NOISE_TENSION_FULL_SCALE_N,
            "omega_rad_s": SENSOR_NOISE_OMEGA_FULL_SCALE_RAD_S,
            "noise_tension_N": SENSOR_NOISE_TENSION_N,
            "noise_omega_rad_s": SENSOR_NOISE_OMEGA_RAD_S,
            "noise_channels": "tension_only_and_dual_channel",
            "velocity_percent_full_scale": EXCITATION_SN_VELOCITY_PERCENT,
        },
        "measurement_conditions": list(LOGGING_CONDITIONS),
        "measurement_condition_labels": dict(LOGGING_CONDITION_LABELS),
        "speed_metrics": speed_metrics,
        "speed_groups": speed_groups,
        "paper_reference": reference_annotations,
        "paper_reference_rows": reference_rows,
        "figure_map": {
            "numerical": "v5 Fig. 2(a) - three measurement conditions",
            "speed": "v5 Fig. 2(b) - dual-channel decomposed by line speed",
            "power_law": "v5 Fig. S2 - noise-free tau_min-normalized power law",
        },
        "power_law_scope": "noise_free_only",
        "power_law_sensor_noise_caveat": (
            "The sensor-noise branch is deliberately not overlaid on the power law: under "
            "tension-only sensor noise the error does not collapse onto it (fitted exponent "
            "+0.41, R^2 ~ 0.11), so an SN fit would contradict its own trend."
        ),
        "best_tension_only_Tlog_ms": (
            best_tension_only["Tlog_ms"] if best_tension_only is not None else None
        ),
        "best_tension_only_MARE_theta_percent": (
            best_tension_only["MARE_theta_percent"] if best_tension_only is not None else None
        ),
        "tension_only_has_interior_optimum": False,
        "tlog_range_note": "PLC logging period Tlog is operator-configurable from 1 ms to 200 ms.",
        "tau_min_formula": "tau_min = 1 / max_i(|Re(lambda_i)|)",
        "nf_guideline_tau_min_over_tlog": 5.0,
        "best_noisy_Tlog_ms": best_noisy["Tlog_ms"],
        "best_noisy_MARE_theta": best_noisy["MARE_theta"],
        "best_noisy_MARE_theta_percent": best_noisy["MARE_theta_percent"],
        "best_noisy_tau_ratio": best_noisy["tau_ratio"],
        "best_noisy_tau_min_over_tlog": best_noisy["tau_min_over_tlog"],
        "best_noisy_reference_power_law_Tlog_ms": (
            best_noisy_reference_power_law["Tlog_ms"] if best_noisy_reference_power_law else None
        ),
        "best_noisy_reference_power_law_MARE_theta": (
            best_noisy_reference_power_law["reference_power_law_MARE_theta"]
            if best_noisy_reference_power_law
            else None
        ),
        "best_noisy_reference_power_law_MARE_theta_percent": (
            best_noisy_reference_power_law["reference_power_law_MARE_theta_percent"]
            if best_noisy_reference_power_law
            else None
        ),
        "supports_noisy_optimum_in_5_20ms_window": supports_window,
        "power_law_model": "MARE_theta_percent = a * (Tlog/tmin)^alpha",
        "reference_power_law_values": REFERENCE_POWER_LAW_VALUES,
        "power_law_fits": power_law_fits,
        "numerical_curve_fits": numerical_curve_fits,
        "power_law_plot_path": power_law_plot_path,
        "plot_path": plot_path,
        "speed_plot_path": speed_plot_path,
        "csv_path": csv_path,
        "speed_csv_path": speed_csv_path,
        "graph_points_csv_path": graph_points_csv_path,
        "graph_points_xlsx_path": graph_points_xlsx_path,
    }
    markdown_path = _write_logging_report(
        payload, plot_path, power_law_plot_path, csv_path, speed_plot_path
    )
    payload["markdown_path"] = markdown_path
    summary_path = _write_summary("logging_rate_summary_restored.json", payload)
    if is_aggregate:
        _write_summary("logging_rate_summary_all_plants_restored.json", payload)
    artifact = _artifact_payload(payload, plot_path, summary_path, csv_path, markdown_path)
    artifact["graph_points_csv_path"] = graph_points_csv_path
    artifact["graph_points_xlsx_path"] = graph_points_xlsx_path
    artifact["speed_plot_path"] = speed_plot_path
    artifact["speed_csv_path"] = speed_csv_path
    artifact["power_law_plot_path"] = power_law_plot_path
    return artifact


def _cached_excitation_study(
    active_plant_runs: Sequence[tuple[str, R2RParameters, Mapping[str, Any]]],
) -> dict[str, object] | None:
    if len(active_plant_runs) > 1:
        summary_paths = [SUMMARY_DIR / "excitation_summary_all_plants_v41.json", SUMMARY_DIR / "excitation_summary_v41.json"]
    else:
        summary_paths = [SUMMARY_DIR / f"excitation_summary_v41_{active_plant_runs[0][0]}.json"]
    for summary_path in summary_paths:
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        required_keys = ("comparison_rows", "raw_rows", "run_metadata", "plot_path", "csv_path", "raw_csv_path")
        if any(key not in payload for key in required_keys):
            continue
        if payload.get("plant_scope") != ("all_plants" if len(active_plant_runs) > 1 else "single_plant"):
            continue
        if payload.get("calculation_version") != EXCITATION_CACHE_VERSION:
            continue

        cached_plant_ids = [str(plant_id) for plant_id in payload.get("plant_ids", [])]
        active_plant_ids = [str(plant_id) for plant_id, _, _ in active_plant_runs]
        if cached_plant_ids != active_plant_ids:
            continue

        artifact_paths = [
            Path(str(payload["plot_path"])),
            Path(str(payload["csv_path"])),
            Path(str(payload["raw_csv_path"])),
        ]
        try:
            for artifact_path in artifact_paths:
                artifact_path.resolve().relative_to(PROJECT_ROOT)
                if not artifact_path.exists():
                    raise FileNotFoundError(str(artifact_path))
        except (OSError, ValueError):
            continue

        artifact = _artifact_payload(
            payload,
            str(payload.get("plot_path") or ""),
            str(summary_path),
            str(payload.get("csv_path") or ""),
        )
        artifact["raw_csv_path"] = payload["raw_csv_path"]
        return artifact
    return None


def excitation_study(
    params: R2RParameters | None = None,
    plant_runs: Sequence[tuple[str, R2RParameters, Mapping[str, Any]]] | None = None,
    prefer_cache: bool = False,
) -> dict[str, object]:
    """Compare excitation strategies with one backend simulation/SysID path."""

    active_params = params or R2RParameters()
    active_plant_runs = (
        [(str(plant_id), plant_params, dict(plant_meta)) for plant_id, plant_params, plant_meta in plant_runs]
        if plant_runs
        else [("selected", active_params, {"plant_id": "selected", "label": "Selected plant", "sensor_noise_sigma_N": SENSOR_NOISE_TENSION_N})]
    )
    if not active_plant_runs:
        raise ValueError("At least one plant is required.")

    if prefer_cache:
        cached = _cached_excitation_study(active_plant_runs)
        if cached is not None:
            return cached

    is_aggregate = len(active_plant_runs) > 1
    run_metadata = _run_metadata(
        "excitation",
        plant_scope="all_plants" if is_aggregate else "single_plant",
        run_settings={
            "duration_s": "profile_specific",
            "controller_sample_time_s": 0.001,
            "NF_log_sample_time_s": 0.005,
            "SN_log_sample_time_s": 0.020,
            "sensor_noise_lpf_hz": EXCITATION_SN_LPF_HZ,
            "sensor_noise_velocity_percent": EXCITATION_SN_VELOCITY_PERCENT,
            "sn_measurement_condition": "dual_channel",
            "sensor_noise_full_scale_percent": SENSOR_NOISE_FULL_SCALE_PERCENT,
            "metric_formula": "100*mean(abs(relative_error_i))",
            "sensor_noise_observer_settings": "none",
            "NF_controller_mode": "paper_aligned",
            "SN_controller_mode": "paper_aligned_filtered_measurement",
            "sensor_noise_scope": "filtered_measurement_to_controller_and_sysid",
            "sysid_estimator": "paper_eq8_weighted_pem_trf",
            "controller_integral_time": "per_plant_auto_Ti",
            "steady_velocity_baseline": "nominal_tension_target_with_current_line_speed",
            "velocity_correction_clamp": "none",
            "EV1_identification": "continuous_midrun_speed_step_segment_split",
            "required_valid_plants_per_condition": 10,
            "NF_campaign_group": GROUP_A,
            "SN_campaign_group": GROUP_B,
            "record_duration_source": "excitation_schedules.csv",
            "SN_velocity_seed_offset": COMPOSITE_SEED_V_OFFSET,
            "ET3M_record_seed_rule": "seed_T = base + 17*i",
            "paper_inputs": paper_input_provenance(),
        },
    )
    reference = load_excitation_reference()
    paper_by_strategy = reference.get("strategies", {})
    strategy_channels = {
        "ET1": 1,
        "ET3": 3,
        "ET6": 3,
        "ET3M": 3,
        "E_Toggle": 3,
        "EV1": "0 (velocity-only)",
    }
    raw_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    def run_one(
        plant_run: tuple[str, R2RParameters, Mapping[str, Any]],
        strategy: str,
        condition: str,
        seed: int = 0,
    ) -> dict[str, object]:
        plant_id, plant_params, plant_meta = plant_run
        # v5 reports the excitation SN column at the DUAL-CHANNEL condition
        # (tension and velocity, 0.3%/0.3%, LPF 50 Hz, Tlog 20 ms). v4.1 used
        # tension-only noise at 100 Hz, which is a different measurement
        # condition, not a different value for the same one.
        noise_tension = 0.0
        noise_velocity = 0.0
        if condition == "SN":
            noise_tension = float(plant_meta.get("sensor_noise_sigma_N", SENSOR_NOISE_TENSION_N))
            noise_velocity = (
                EXCITATION_SN_VELOCITY_PERCENT
                / 100.0
                * velocity_full_scale_m_s(float(plant_params.feeder_velocity_m_s))
            )
        active_sensor_lpf_hz = EXCITATION_SN_LPF_HZ if condition == "SN" else None
        observer_clip_fraction = None
        paper_controller_enabled = True
        high_ea_kp_cap_enabled = False
        default_tension = statistics.mean(float(value) for value in plant_params.tension_ref_N)
        amplitude = float(
            plant_meta.get(
                "recommended_excitation_amplitude_V",
                0.2 * float(plant_meta.get("T_ref_N", default_tension)),
            )
        )
        # The SN column is the dual-channel condition, and the dual-channel cells
        # were published from campaign group B. Only ET1 differs between the two
        # grids -- 30 s there against 7 s in the tension factorial -- so this is
        # the one place the record length depends on the noise condition, and it
        # does so because the campaign does, not because the noise does.
        campaign_group = GROUP_B if condition == "SN" else GROUP_A
        # Records, line-speed multipliers and per-record seeds all come from the
        # schedule table: ET3M logs three records at v0 x {0.5, 1, 2} with
        # seed_T = base + 17*i, everything else logs one record at v0.
        records = excitation_records(strategy, campaign_group)
        base_line_speed = float(plant_meta.get("v_ref_m_s", plant_params.feeder_velocity_m_s))
        try:
            operating_point_mare: list[float] = []
            operating_point_ti_s: list[float] = []
            # ET3M logs its three operating points as separate experiments and
            # identifies them in ONE joint fit through the multi-condition cost
            # (paper eq. 11), each condition carrying its own operating-point
            # weighting W^(c). Averaging three independent per-operating-point
            # fits is a different estimator and reads about 4 pp low under noise.
            joint_conditions: list[MeasurementCondition] = []
            samples = 0.0
            speed_multipliers = tuple(record.v_ref_multiplier for record in records)
            record_seeds = tuple(
                et3m_record_seed(seed, record.record_index) if len(records) > 1 else seed
                for record in records
            )
            for record, record_seed in zip(records, record_seeds, strict=True):
                speed_multiplier = record.v_ref_multiplier
                line_speed = base_line_speed * speed_multiplier
                ti_s = auto_tension_integral_time_s(plant_params, line_speed)
                operating_point_ti_s.append(ti_s)
                excitation_profile = build_excitation(record, amplitude)
                speed_suffix = f"_v{speed_multiplier:g}" if len(records) > 1 else ""
                sim = simulate(
                    plant_params,
                    controller_config=ControllerConfig(
                        line_speed_m_s=line_speed,
                        target_tension_N=plant_params.tension_ref_N,
                        TI_s=ti_s,
                        high_ea_kp_cap_enabled=high_ea_kp_cap_enabled,
                        feedforward_uses_measured_omega=paper_controller_enabled,
                        paper_velocity_gain_enabled=paper_controller_enabled,
                        velocity_correction_limit_fraction=None,
                        steady_velocity_uses_dynamic_target=False,
                    ),
                    config=SimulationConfig(
                        duration_s=float(getattr(excitation_profile, "duration_s", 6.0)),
                        controller_sample_time_s=0.001,
                        log_sample_time_s=0.005 if condition == "NF" else 0.020,
                        line_speed_m_s=line_speed,
                        sensor_noise_tension_N=noise_tension,
                        sensor_noise_velocity_m_s=noise_velocity,
                        sensor_lpf_hz=float(active_sensor_lpf_hz) if active_sensor_lpf_hz is not None else None,
                        noise_affects_controller=True,
                        output_name=f"excitation_{plant_id}_{condition}_{strategy}{speed_suffix}.csv",
                        seed=record_seed,
                        velocity_seed_offset=(
                            COMPOSITE_SEED_V_OFFSET if condition == "SN" else None
                        ),
                    ),
                    excitation=excitation_profile,
                    output_dir=DATA_DIR,
                )
                sysid_rows = _bounded_velocity_observer_rows(
                    sim.rows,
                    plant_params,
                    float(observer_clip_fraction) if observer_clip_fraction is not None else None,
                )
                sysid = estimate_parameters_weighted_pem(
                    sysid_rows,
                    nominal_params=plant_params,
                    true_params=plant_params,
                    max_nfev=150,
                    break_on_line_speed_change=strategy == "EV1",
                )
                operating_point_mare.append(float(sysid.mare_theta))
                joint_conditions.append(
                    MeasurementCondition(
                        rows=sysid_rows,
                        line_speed_m_s=line_speed,
                        params=plant_params,
                        label=f"v{speed_multiplier:g}",
                    )
                )
                samples += float(len(sim.rows))
            if len(joint_conditions) > 1:
                joint = estimate_parameters_weighted_pem(
                    joint_conditions,
                    nominal_params=plant_params,
                    true_params=plant_params,
                    max_nfev=150,
                    break_on_line_speed_change=strategy == "EV1",
                )
                mare_theta = float(joint.mare_theta)
            else:
                mare_theta = float(operating_point_mare[0])
            mare_percent = 100.0 * mare_theta
            value_status = "computed_raw"
        except (TypeError, KeyError, AttributeError):
            raise
        except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
            mare_percent = None
            mare_theta = None
            samples = 0.0
            value_status = f"simulation_unstable:{type(exc).__name__}:{exc}"
        return {
            **_metadata_row_fields(run_metadata, "computed_raw"),
            "strategy": strategy,
            "excitation": strategy,
            "condition": condition,
            "plant_id": plant_id,
            "plant_label": str(plant_meta.get("label", plant_id)),
            "EA_N": float(plant_meta.get("EA_N", plant_params.EA)),
            "dashboard_MARE_theta_percent": mare_percent,
            "MARE_theta": mare_theta,
            "samples": samples,
            "excitation_amplitude_V": amplitude,
            "excitation_amplitude_N": amplitude,
            "sensor_noise_tension_N": noise_tension,
            "sensor_noise_velocity_m_s": noise_velocity,
            "measurement_condition": "dual_channel" if condition == "SN" else "noise_free",
            "sensor_lpf_hz": active_sensor_lpf_hz,
            "controller_sample_time_s": 0.001,
            "operating_point_TI_s": ",".join(f"{value:.9g}" for value in operating_point_ti_s),
            "steady_velocity_baseline": "nominal_tension_current_line_speed",
            "velocity_correction_limit_fraction": None,
            "velocity_observer_clip_fraction": observer_clip_fraction,
            "high_ea_kp_cap_enabled": high_ea_kp_cap_enabled,
            "controller_mode": "paper_aligned_filtered_measurement" if condition == "SN" else "paper_aligned",
            "noise_affects_controller": True,
            "estimator": "paper_eq8_weighted_pem_trf",
            "pem_initial_scale": 1.5,
            "pem_lower_scale": 0.1,
            "pem_upper_scale": 10.0,
            "pem_max_nfev": 150,
            "noise_rng": "random.Random(seed).gauss",
            "ev1_segment_split": strategy == "EV1",
            "metric_formula": "100*mean(abs(relative_error_i))",
            "operating_point_speed_multipliers": ",".join(f"{value:g}" for value in speed_multipliers),
            "campaign_group": campaign_group,
            "resolved_campaign_group": records[0].campaign_group,
            "record_count": float(len(records)),
            "record_duration_s": ",".join(f"{record.duration_s:g}" for record in records),
            "record_seeds": ",".join(str(value) for value in record_seeds),
            "velocity_seed_offset": COMPOSITE_SEED_V_OFFSET if condition == "SN" else None,
            "value_status": value_status,
        }

    for strategy in excitation_names():
        condition_values: dict[str, float] = {}
        condition_valid_counts: dict[str, int] = {}
        for condition in ("NF", "SN"):
            # Fig. 2(a): the dual-channel condition is a median over 10 plants x
            # 3 seeds; noise-free is a 10-plant median at a single seed.
            condition_seeds = EXCITATION_SN_SEEDS if condition == "SN" else (0,)
            plant_results = [
                run_one(plant_run, strategy, condition, seed)
                for seed in condition_seeds
                for plant_run in active_plant_runs
            ]
            raw_rows.extend(plant_results)
            finite_values = [
                float(row["dashboard_MARE_theta_percent"])
                for row in plant_results
                if row.get("dashboard_MARE_theta_percent") is not None
                and math.isfinite(float(row["dashboard_MARE_theta_percent"]))
            ]
            condition_valid_counts[condition] = len(finite_values)
            expected_results = len(active_plant_runs) * len(condition_seeds)
            if is_aggregate and len(finite_values) != expected_results:
                failed_details = ", ".join(
                    f"{row['plant_id']}={row['value_status']}"
                    for row in plant_results
                    if row.get("dashboard_MARE_theta_percent") is None
                    or not math.isfinite(float(row["dashboard_MARE_theta_percent"]))
                )
                raise RuntimeError(
                    f"Excitation {strategy}/{condition} requires all {expected_results} results "
                    f"({len(active_plant_runs)} plants x {len(condition_seeds)} seeds); "
                    f"received {len(finite_values)} finite results. Failed: {failed_details}"
                )
            condition_values[condition] = statistics.median(finite_values) if finite_values else math.nan

        paper = paper_by_strategy.get(strategy, {})
        paper_nf = float(paper["NF"]) if "NF" in paper else None
        paper_sn = float(paper["SN"]) if "SN" in paper else None
        row = {
            **_metadata_row_fields(run_metadata, "computed_raw"),
            "strategy": strategy,
            "excitation": strategy,
            "channels": strategy_channels.get(strategy, ""),
            "analysis_scope": "all_plants" if is_aggregate else "single_plant",
            "plant_count": float(len(active_plant_runs)),
            "valid_plant_count_NF": float(condition_valid_counts["NF"]),
            "failed_plant_count_NF": float(len(active_plant_runs) - condition_valid_counts["NF"]),
            "raw_dashboard_NF_percent": condition_values["NF"],
            "dashboard_NF_percent": condition_values["NF"],
            "displayed_dashboard_NF_percent": condition_values["NF"],
            "paper_NF_percent": paper_nf,
            "difference_NF_percent": condition_values["NF"] - paper_nf if paper_nf is not None else None,
            "raw_dashboard_SN_percent": condition_values["SN"],
            "valid_plant_count_SN": float(condition_valid_counts["SN"]),
            "failed_plant_count_SN": float(len(active_plant_runs) - condition_valid_counts["SN"]),
            "dashboard_SN_percent": condition_values["SN"],
            "displayed_dashboard_SN_percent": condition_values["SN"],
            "paper_SN_percent": paper_sn,
            "difference_SN_percent": condition_values["SN"] - paper_sn if paper_sn is not None else None,
            "display_adjustment_type": "none_independent_simulation",
            "metric_formula": "100*mean(abs(relative_error_i))",
            "value_status": "computed_raw",
            "__provenance": {
                "raw_dashboard_NF_percent": "dashboard_simulation",
                "dashboard_NF_percent": "dashboard_simulation_median",
                "displayed_dashboard_NF_percent": "dashboard_simulation_median",
                "raw_dashboard_SN_percent": "dashboard_simulation",
                "dashboard_SN_percent": "dashboard_simulation_median",
                "displayed_dashboard_SN_percent": "dashboard_simulation_median",
                "paper_NF_percent": "paper_reference",
                "paper_SN_percent": "paper_reference",
                "difference_NF_percent": "computed_dashboard_minus_reference",
                "difference_SN_percent": "computed_dashboard_minus_reference",
            },
        }
        comparison_rows.append(row)

    noisy = comparison_rows
    best_noisy = min(noisy, key=lambda row: float(row["dashboard_SN_percent"]))
    multi_channel_best = str(best_noisy["strategy"]) in {"ET3", "ET6", "ET3M", "E_Toggle"}
    plot_rows = [
        {"label": f"{row['strategy']}\nNF", "value": float(row["dashboard_NF_percent"])}
        for row in comparison_rows
    ] + [
        {"label": f"{row['strategy']}\nSN", "value": float(row["dashboard_SN_percent"])}
        for row in comparison_rows
    ]
    artifact_suffix = "" if is_aggregate else f"_{active_plant_runs[0][0]}"
    plot_path = write_bar_chart(
        plot_rows,
        FIGURES_DIR / f"excitation_vs_mare_v41{artifact_suffix}.svg",
        title="Excitation Type vs SysID MARE",
        category_key="label",
        value_key="value",
        y_label="MARE_theta (%)",
    )
    csv_fields = (
        *METADATA_FIELDS,
        "strategy",
        "excitation",
        "channels",
        "analysis_scope",
        "plant_count",
        "valid_plant_count_NF",
        "failed_plant_count_NF",
        "raw_dashboard_NF_percent",
        "dashboard_NF_percent",
        "displayed_dashboard_NF_percent",
        "paper_NF_percent",
        "difference_NF_percent",
        "valid_plant_count_SN",
        "failed_plant_count_SN",
        "raw_dashboard_SN_percent",
        "dashboard_SN_percent",
        "displayed_dashboard_SN_percent",
        "paper_SN_percent",
        "difference_SN_percent",
        "display_adjustment_type",
        "metric_formula",
    )
    csv_path = _write_rows_csv(
        f"excitation_summary_dashboard_vs_paper_v41{artifact_suffix}.csv",
        comparison_rows,
        csv_fields,
    )
    raw_csv_fields = (
        *METADATA_FIELDS,
        "strategy",
        "excitation",
        "condition",
        "plant_id",
        "plant_label",
        "EA_N",
        "dashboard_MARE_theta_percent",
        "MARE_theta",
        "samples",
        "excitation_amplitude_V",
        "excitation_amplitude_N",
        "sensor_noise_tension_N",
        "sensor_noise_velocity_m_s",
        "measurement_condition",
        "sensor_lpf_hz",
        "velocity_observer_clip_fraction",
        "high_ea_kp_cap_enabled",
        "controller_mode",
        "noise_affects_controller",
        "metric_formula",
        "operating_point_speed_multipliers",
        "controller_sample_time_s",
        "operating_point_TI_s",
        "steady_velocity_baseline",
        "velocity_correction_limit_fraction",
        "estimator",
        "pem_initial_scale",
        "pem_lower_scale",
        "pem_upper_scale",
        "pem_max_nfev",
        "noise_rng",
        "ev1_segment_split",
        "campaign_group",
        "resolved_campaign_group",
        "record_count",
        "record_duration_s",
        "record_seeds",
        "velocity_seed_offset",
    )
    raw_csv_path = _write_rows_csv(
        f"excitation_plant_runs_v41{artifact_suffix}.csv",
        raw_rows,
        raw_csv_fields,
    )
    payload = {
        "study": "excitation",
        "calculation_version": EXCITATION_CACHE_VERSION,
        "metrics": comparison_rows,
        "comparison_rows": comparison_rows,
        "raw_rows": raw_rows,
        "run_metadata": run_metadata,
        "plant_scope": "all_plants" if is_aggregate else "single_plant",
        "plant_count": len(active_plant_runs),
        "plant_ids": [plant_id for plant_id, _, _ in active_plant_runs],
        "best_noisy_excitation": best_noisy["strategy"],
        "supports_multi_channel_or_toggle_under_noise": multi_channel_best,
        "plot_path": plot_path,
        "csv_path": csv_path,
        "raw_csv_path": raw_csv_path,
    }
    summary_path = _write_summary(f"excitation_summary_v41{artifact_suffix}.json", payload)
    if is_aggregate:
        _write_summary("excitation_summary_all_plants_v41.json", payload)
    artifact = _artifact_payload(payload, plot_path, summary_path, csv_path)
    artifact["raw_csv_path"] = raw_csv_path
    return artifact


def _fixed_scenario_scale(final_scale: float) -> callable:
    """Return the paper Section 3.3 fixed perturbed-plant scale."""

    def scale(_t_s: float) -> float:
        return final_scale

    return scale


# Paper Fig. 4a draws the EA family over five legs: "(a) axial stiffness EA
# (+/-10-30% and +50%)". `drift_reference.json` carries only the four +/-10/30
# rows, but its own `family_reference.EA.legs` field says "+/-10%, +/-30% and
# +50% (five legs)" and the published EA band/mean is the statistic over those
# five. The +50% leg is therefore reconstructed here with *null* paper cells
# (the family is published as a band, never per leg), so the dashboard band is
# taken over the same five legs as the paper's.
EA_DRIFT_PERCENT_LEVELS = (-30.0, -10.0, 10.0, 30.0, 50.0)
F_DRIFT_PERCENT_LEVELS = (-30.0, -15.0, 0.0, 15.0, 30.0)
DRIFT_EXCITATION_TENSION_FRACTION = 0.20
DRIFT_SENSOR_NOISE_SEEDS = (0, 1, 2)
DRIFT_CACHE_VERSION = "drift_v5_weighted_pem_tlog20_schedule17s_ea50_combined_perroller_fig04paperrefs_20260818"
DRIFT_LOG_SAMPLE_TIME_S = 0.020
# Ledger campaign 2 ("Parameter drift", 400 runs) sits in ledger group A, whose
# excitation records are the `A_tension_factorial` rows of
# excitation_schedules.csv. E_Toggle there is a 17 s record (settle 2 s,
# edges at 2/7/12 s), not the 7 s ET1 record that was hardcoded before.
DRIFT_CAMPAIGN_GROUP = GROUP_A
DRIFT_EXCITATION_NAME = "E_Toggle"


def drift_record_duration_s() -> float:
    """Return the drift campaign's record length from the paper schedule CSV."""

    return float(
        excitation_schedule(DRIFT_EXCITATION_NAME, DRIFT_CAMPAIGN_GROUP, 0).duration_s
    )


def _signed_percent_label(percent: float) -> str:
    if math.isclose(percent, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "0%"
    return f"{percent:+.0f}%"


def _symmetric_reference_rows(
    family: str,
    percent_levels: Sequence[float],
    nf_values: Mapping[float, float],
    sn_values: Mapping[float, float],
    notes: Mapping[float, str],
) -> list[dict[str, object]]:
    return [
        {
            "drift_case": f"{family} {_signed_percent_label(percent)}",
            "drift_percent": percent,
            "drift_family": family,
            "drift_kind": family,
            "paper_NF_percent": float(nf_values[percent]),
            "paper_SN_percent": float(sn_values[percent]),
            "plot_only": percent == 0.0,
            "paper_note": str(notes[percent]),
        }
        for percent in percent_levels
    ]


def _drift_reference_rows() -> list[dict[str, object]]:
    reference = load_drift_reference()
    rows: list[dict[str, object]] = []
    for family in ("EA", "f"):
        family_rows: list[dict[str, object]] = []
        for item in reference.get(family, []):
            family_rows.append(
                {
                    "drift_case": item["drift_case"],
                    "drift_percent": float(item["drift_percent"]),
                    "drift_family": family,
                    "drift_kind": family,
                    "paper_NF_percent": (
                        None
                        if item.get("paper_NF_percent") is None
                        else float(item["paper_NF_percent"])
                    ),
                    "paper_SN_percent": (
                        None
                        if item.get("paper_SN_percent") is None
                        else float(item["paper_SN_percent"])
                    ),
                    "plot_only": bool(item.get("plot_only", False)),
                    "paper_note": str(item.get("note", "")),
                    "paper_reference_type": str(item.get("reference_type", "paper_reference")),
                }
            )
        if family == "EA":
            family_rows.extend(_missing_ea_band_legs(family_rows, reference))
            family_rows.sort(key=lambda row: float(row["drift_percent"]))
        rows.extend(family_rows)
    for item in reference.get("J", []):
        rows.append(
            {
                "drift_case": item["drift_case"],
                "drift_percent": float(item["drift_percent"]),
                "drift_family": str(item.get("drift_family", "J")),
                "drift_kind": str(item.get("drift_kind", "J_asymmetric")),
                "J_UW_percent": float(item["J_UW_percent"]),
                "J_RW_percent": float(item["J_RW_percent"]),
                "paper_NF_percent": float(item["paper_NF_percent"]),
                "paper_SN_percent": float(item["paper_SN_percent"]),
                "paper_note": str(item.get("note", "")),
                "paper_reference_type": str(item.get("reference_type", "paper_reference")),
            }
        )
    combined = _combined_drift_reference_row(reference)
    if combined is not None:
        rows.append(combined)
    return rows


# The three-parameter scenario of Section 3.3 ("The tenth scenario moves all
# three at once (EA+20%, J_UW-20%, f+15%) and lands at 29.6% (NF)"). It is the
# tenth of the campaign's ten scenarios and is carried in
# `drift_reference.json` under `family_reference.combined`, which prints the
# scenario string and the NF median; the SN cell is not published.
_COMBINED_DRIFT_SCALES = {"EA": 1.20, "J_UW": 0.80, "f": 1.15}


def _missing_ea_band_legs(
    rows: Sequence[Mapping[str, object]], reference: Mapping[str, object]
) -> list[dict[str, object]]:
    """Return the EA legs Fig. 4a draws that the reference JSON does not list.

    Paper Fig. 4a caption: "(a) axial stiffness EA (+/-10-30% and +50%)", and
    `family_reference.EA.legs` repeats "five legs". The JSON's per-leg array
    stops at +30% because every EA cell is `null` by design (the family is
    published as a band, not per leg), so the missing +50% leg is reconstructed
    with null paper cells rather than dropped - otherwise the dashboard band is
    taken over four legs where the paper's is taken over five.
    """

    present = {
        round(float(row["drift_percent"]), 6)
        for row in rows
        if str(row.get("drift_family")) == "EA"
    }
    template = next(
        (dict(item) for item in reference.get("EA", []) if item.get("drift_percent") is not None),
        {},
    )
    missing: list[dict[str, object]] = []
    for percent in EA_DRIFT_PERCENT_LEVELS:
        if round(float(percent), 6) in present:
            continue
        missing.append(
            {
                "drift_case": f"EA {_signed_percent_label(float(percent))}",
                "drift_percent": float(percent),
                "drift_family": "EA",
                "drift_kind": "EA",
                "paper_NF_percent": None,
                "paper_SN_percent": None,
                "plot_only": False,
                "paper_note": str(
                    template.get(
                        "note",
                        "Within the published EA band; the paper reports the family as a band, not per leg.",
                    )
                ),
                "paper_reference_type": "published_band_only",
            }
        )
    return sorted(missing, key=lambda row: float(row["drift_percent"]))


def _combined_drift_reference_row(reference: Mapping[str, object]) -> dict[str, object] | None:
    """Return the tenth (three-parameter) drift scenario as a comparison row."""

    combined = (reference.get("family_reference") or {}).get("combined")
    if not isinstance(combined, Mapping):
        return None
    paper_nf = combined.get("paper_NF_percent")
    paper_sn = combined.get("paper_SN_percent")
    return {
        "drift_case": str(combined.get("scenario", "EA +20%, J -20%, f +15%")),
        "drift_percent": None,
        "drift_family": "combined",
        "drift_kind": "combined",
        "EA_percent": 100.0 * (_COMBINED_DRIFT_SCALES["EA"] - 1.0),
        "J_UW_percent": 100.0 * (_COMBINED_DRIFT_SCALES["J_UW"] - 1.0),
        "J_RW_percent": 0.0,
        "f_percent": 100.0 * (_COMBINED_DRIFT_SCALES["f"] - 1.0),
        "paper_NF_percent": None if paper_nf is None else float(paper_nf),
        "paper_SN_percent": None if paper_sn is None else float(paper_sn),
        "plot_only": False,
        "paper_note": (
            "Section 3.3, tenth scenario: EA, J_UW and f move together. The paper "
            "prints the noise-free median only."
        ),
        "paper_reference_type": "paper_reference",
    }


def _drift_paper_family_reference() -> dict[str, dict[str, object]]:
    """Return the published v5 drift numbers the per-leg rows cannot carry.

    Section 3.3 does not print a paper median for every leg. The EA family is
    published as a *band* across its five legs, and the friction family prints a
    noise-free pair only. Those numbers exist in the paper, so the dashboard has
    to show them somewhere; per-leg cells are the wrong place, because inventing
    a per-leg value out of a band would misreport the paper. They are returned
    per family instead, together with the acquisition condition they were taken
    under.
    """

    reference = load_drift_reference()
    families = reference.get("family_reference", {})
    acquisition = reference.get("acquisition_condition", {})
    ea = families.get("EA", {})
    friction = families.get("f", {})
    inertia = families.get("J", {})
    combined = families.get("combined", {})

    def _band(values: object) -> list[float] | None:
        if not isinstance(values, list | tuple) or len(values) != 2:
            return None
        return [float(values[0]), float(values[1])]

    return {
        "EA": {
            "family": "EA",
            "published_as": "band",
            "legs": str(ea.get("legs", "")),
            "paper_NF_band_percent": _band(ea.get("paper_NF_band_percent")),
            "paper_SN_band_percent": _band(ea.get("paper_SN_band_percent")),
            "paper_NF_mean_percent": ea.get("paper_NF_mean_percent"),
            "paper_SN_mean_percent": ea.get("paper_SN_mean_percent"),
            "note": str(ea.get("note", "")),
            "acquisition_condition": acquisition,
        },
        "f": {
            "family": "f",
            "published_as": "endpoint_pair",
            "legs": str(friction.get("legs", "")),
            "paper_NF_percent_by_drift_percent": {
                "-30": friction.get("paper_NF_percent_at_minus_30"),
                "30": friction.get("paper_NF_percent_at_plus_30"),
            },
            "note": str(friction.get("note", "")),
            "acquisition_condition": acquisition,
        },
        "J": {
            "family": "J",
            "published_as": "per_leg",
            "gap_above_EA_baseline_pp": inertia.get("gap_above_EA_baseline_pp", {}),
            "note": str(inertia.get("note", "")),
            "acquisition_condition": acquisition,
        },
        "combined": {
            "family": "combined",
            "published_as": "single_scenario",
            "scenario": str(combined.get("scenario", "")),
            "paper_NF_percent": combined.get("paper_NF_percent"),
            "paper_SN_percent": combined.get("paper_SN_percent"),
        },
        "__source": str(reference.get("source", "")),
        "__notes": list(reference.get("notes", [])),
    }


def _drift_dashboard_family_summary(
    comparison_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute, from this run, the family statistics the paper publishes.

    Section 3.3 does not report EA per leg; it reports the band and the mean
    across the five EA legs, and quotes the J-EA gap against that mean. Those
    are dashboard-side statistics of the freshly simulated legs, compared with
    the published band afterwards.
    """

    reference = _drift_paper_family_reference()
    ea_rows = [row for row in comparison_rows if row.get("drift_family") == "EA"]
    summary: dict[str, object] = {}

    def _values(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        ]

    ea_stats: dict[str, object] = {"leg_count": len(ea_rows)}
    for condition, band_key, mean_key in (
        ("NF", "paper_NF_band_percent", "paper_NF_mean_percent"),
        ("SN", "paper_SN_band_percent", "paper_SN_mean_percent"),
    ):
        values = _values(ea_rows, f"dashboard_{condition}_percent")
        paper_band = reference["EA"].get(band_key)
        paper_mean = reference["EA"].get(mean_key)
        ea_stats[condition] = {
            "dashboard_band_percent": [min(values), max(values)] if values else None,
            "dashboard_mean_percent": statistics.mean(values) if values else None,
            "dashboard_spread_pp": (max(values) - min(values)) if values else None,
            "paper_band_percent": paper_band,
            "paper_mean_percent": paper_mean,
            "difference_mean_pp": (
                None
                if not values or paper_mean is None
                else statistics.mean(values) - float(paper_mean)
            ),
        }
    summary["EA"] = ea_stats

    # J - EA gap: paper +22.9 pp (NF), +20.5 pp (SN) for the extreme leg above
    # the five-leg EA mean.
    extreme = next(
        (
            row
            for row in comparison_rows
            if row.get("drift_kind") == "J_asymmetric"
            and float(row.get("J_RW_percent") or 0.0) >= 100.0
        ),
        None,
    )
    paper_gap = reference["J"].get("gap_above_EA_baseline_pp") or {}
    gap: dict[str, object] = {}
    for condition in ("NF", "SN"):
        ea_mean = ea_stats[condition]["dashboard_mean_percent"]
        extreme_value = None if extreme is None else extreme.get(f"dashboard_{condition}_percent")
        dashboard_gap = (
            None
            if ea_mean is None or extreme_value is None or not math.isfinite(float(extreme_value))
            else float(extreme_value) - float(ea_mean)
        )
        paper_value = paper_gap.get(condition)
        gap[condition] = {
            "dashboard_gap_pp": dashboard_gap,
            "paper_gap_pp": None if paper_value is None else float(paper_value),
            "difference_pp": (
                None
                if dashboard_gap is None or paper_value is None
                else dashboard_gap - float(paper_value)
            ),
        }
    summary["J_minus_EA_gap"] = gap
    return summary


def _reel_radius_sensitivity() -> dict[str, object]:
    """Recompute the Section 3.3 reel-depletion identity J_reel ~ R^4."""

    reference = load_drift_reference().get("reel_radius_sensitivity", {})
    radius_drop_percent = float(reference.get("radius_drop_percent", 15.0))
    ratio = (1.0 - radius_drop_percent / 100.0) ** 4
    computed_drop = 100.0 * (1.0 - ratio)
    paper_drop = reference.get("inertia_drop_percent")
    return {
        "radius_drop_percent": radius_drop_percent,
        "inertia_ratio": ratio,
        "dashboard_inertia_drop_percent": computed_drop,
        "paper_inertia_drop_percent": None if paper_drop is None else float(paper_drop),
        "difference_pp": (
            None if paper_drop is None else computed_drop - float(paper_drop)
        ),
        "relation": str(reference.get("relation", "J_reel proportional to R^4")),
        "value_status": "computed_derived_identity",
        "note": str(reference.get("note", "")),
    }


_ROLLER_NAMES = ("UW", "Nip", "RW")


def _drift_per_roller_rows(
    raw_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return the Section 3.3 per-roller stiffness-error diagnostic.

    The paper defines the per-roller error as |khat_t,m - kt,m| / kt,m and
    reports its median over the runs of a cell. Every term is already recorded
    per run in `raw_rows`, so this is a decomposition of the same runs, not a
    second campaign.
    """

    reference = load_drift_reference().get("per_roller_reference", {})
    extreme_nf = reference.get("extreme_J_noise_free", {})
    extreme_sn = reference.get("extreme_J_sensor_noise", {})
    nip_change = reference.get("nip_change_under_extreme_J", {})

    paper_cells: dict[tuple[str, str, str], float] = {}
    for roller in _ROLLER_NAMES:
        value = extreme_nf.get(f"{roller}_percent")
        if value is not None:
            paper_cells[("UW -50%, RW +100%", "NF", roller)] = float(value)
        value = extreme_sn.get(f"{roller}_percent")
        if value is not None:
            paper_cells[("UW -50%, RW +100%", "SN", roller)] = float(value)
    pre_drift_nip = nip_change.get("pre_drift_percent")
    if pre_drift_nip is not None:
        paper_cells[("f 0%", "NF", "Nip")] = float(pre_drift_nip)

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in raw_rows:
        case = str(row.get("drift_case"))
        condition = str(row.get("condition"))
        for roller in _ROLLER_NAMES:
            value = row.get(f"relative_error_kt_{roller}_percent")
            if value is None or not math.isfinite(float(value)):
                continue
            grouped.setdefault((case, condition, roller), []).append(abs(float(value)))

    rows: list[dict[str, object]] = []
    for (case, condition, roller), values in sorted(grouped.items()):
        paper_value = paper_cells.get((case, condition, roller))
        dashboard_value = statistics.median(values)
        rows.append(
            {
                "drift_case": case,
                "condition": condition,
                "roller": roller,
                "dashboard_kt_error_percent": dashboard_value,
                "paper_kt_error_percent": paper_value,
                "difference_percent": (
                    None if paper_value is None else dashboard_value - paper_value
                ),
                "run_count": len(values),
                "metric": "median |kt_hat_m - kt_m| / kt_m x 100 against the pre-drift baseline",
                "paper_value_status": (
                    "published_phase3_median" if paper_value is not None else "not_published"
                ),
                "value_status": "computed_raw",
            }
        )
    return rows


def _drift_per_roller_summary(
    per_roller_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the qualitative per-roller claims the paper makes, recomputed."""

    reference = load_drift_reference().get("per_roller_reference", {})

    def _worst(case: str, condition: str) -> str | None:
        candidates = [
            row
            for row in per_roller_rows
            if row["drift_case"] == case and row["condition"] == condition
        ]
        if not candidates:
            return None
        return str(max(candidates, key=lambda row: float(row["dashboard_kt_error_percent"]))["roller"])

    paper_worst = (reference.get("pre_drift_worst_roller") or {}).get("tension_only_acquisition")
    dashboard_worst = _worst("f 0%", "NF")
    return {
        "pre_drift_worst_roller": {
            "dashboard": dashboard_worst,
            "paper": paper_worst,
            "agrees": None if paper_worst is None else dashboard_worst == str(paper_worst),
            "pre_drift_case": "f 0%",
            "condition": "NF",
            "note": str((reference.get("pre_drift_worst_roller") or {}).get("note", "")),
        },
        "extreme_J_worst_roller_NF": _worst("UW -50%, RW +100%", "NF"),
        "extreme_J_worst_roller_SN": _worst("UW -50%, RW +100%", "SN"),
        "note": str(reference.get("note", "")),
    }


def _drift_case_function(row: Mapping[str, object], duration_s: float):
    _ = duration_s  # retained for API compatibility; Section 3.3 cases are fixed, not ramped.
    drift_kind = str(row["drift_kind"])
    if drift_kind == "EA":
        final_scale = 1.0 + float(row["drift_percent"]) / 100.0
        ea_scale = _fixed_scenario_scale(final_scale)

        def drift(t_s: float, base: R2RParameters) -> R2RParameters:
            return base.with_drift(EA_scale=ea_scale(t_s))

        return drift, {"EA_scale": final_scale, "f_scale": 1.0, "J_scale": 1.0, "J_UW_scale": 1.0, "J_RW_scale": 1.0}

    if drift_kind == "f":
        final_scale = 1.0 + float(row["drift_percent"]) / 100.0
        friction_scale = _fixed_scenario_scale(final_scale)

        def drift(t_s: float, base: R2RParameters) -> R2RParameters:
            return base.with_drift(friction_scale=friction_scale(t_s))

        return drift, {"EA_scale": 1.0, "f_scale": final_scale, "J_scale": 1.0, "J_UW_scale": 1.0, "J_RW_scale": 1.0}

    if drift_kind == "J":
        final_scale = 1.0 + float(row["drift_percent"]) / 100.0
        inertia_scale = _fixed_scenario_scale(final_scale)

        def drift(t_s: float, base: R2RParameters) -> R2RParameters:
            return base.with_drift(inertia_scale=inertia_scale(t_s))

        return drift, {"EA_scale": 1.0, "f_scale": 1.0, "J_scale": final_scale, "J_UW_scale": final_scale, "J_RW_scale": final_scale}

    if drift_kind == "J_asymmetric":
        uw_final_scale = 1.0 + float(row["J_UW_percent"]) / 100.0
        rw_final_scale = 1.0 + float(row["J_RW_percent"]) / 100.0
        uw_scale = _fixed_scenario_scale(uw_final_scale)
        rw_scale = _fixed_scenario_scale(rw_final_scale)

        def drift(t_s: float, base: R2RParameters) -> R2RParameters:
            return replace(
                base,
                inertia_kg_m2=(
                    base.inertia_kg_m2[0] * uw_scale(t_s),
                    base.inertia_kg_m2[1],
                    base.inertia_kg_m2[2] * rw_scale(t_s),
                ),
            )

        return drift, {"EA_scale": 1.0, "f_scale": 1.0, "J_scale": None, "J_UW_scale": uw_final_scale, "J_RW_scale": rw_final_scale}

    if drift_kind == "combined":
        ea_scale = _COMBINED_DRIFT_SCALES["EA"]
        friction_scale = _COMBINED_DRIFT_SCALES["f"]
        uw_scale = _COMBINED_DRIFT_SCALES["J_UW"]

        def drift(_t_s: float, base: R2RParameters) -> R2RParameters:
            moved = base.with_drift(EA_scale=ea_scale, friction_scale=friction_scale)
            return replace(
                moved,
                inertia_kg_m2=(
                    moved.inertia_kg_m2[0] * uw_scale,
                    moved.inertia_kg_m2[1],
                    moved.inertia_kg_m2[2],
                ),
            )

        return drift, {
            "EA_scale": ea_scale,
            "f_scale": friction_scale,
            "J_scale": None,
            "J_UW_scale": uw_scale,
            "J_RW_scale": 1.0,
        }

    raise ValueError(f"Unsupported drift case {drift_kind!r}")


def _drift_final_truth(base: R2RParameters, row: Mapping[str, object]) -> R2RParameters:
    drift_kind = str(row["drift_kind"])
    if drift_kind == "EA":
        return base.with_drift(EA_scale=1.0 + float(row["drift_percent"]) / 100.0)
    if drift_kind == "f":
        return base.with_drift(friction_scale=1.0 + float(row["drift_percent"]) / 100.0)
    if drift_kind == "J":
        return base.with_drift(inertia_scale=1.0 + float(row["drift_percent"]) / 100.0)
    if drift_kind == "J_asymmetric":
        uw_final_scale = 1.0 + float(row["J_UW_percent"]) / 100.0
        rw_final_scale = 1.0 + float(row["J_RW_percent"]) / 100.0
        return replace(
            base,
            inertia_kg_m2=(base.inertia_kg_m2[0] * uw_final_scale, base.inertia_kg_m2[1], base.inertia_kg_m2[2] * rw_final_scale),
        )
    if drift_kind == "combined":
        moved = base.with_drift(
            EA_scale=_COMBINED_DRIFT_SCALES["EA"], friction_scale=_COMBINED_DRIFT_SCALES["f"]
        )
        return replace(
            moved,
            inertia_kg_m2=(
                moved.inertia_kg_m2[0] * _COMBINED_DRIFT_SCALES["J_UW"],
                moved.inertia_kg_m2[1],
                moved.inertia_kg_m2[2],
            ),
        )
    raise ValueError(f"Unsupported drift case {drift_kind!r}")


def _drift_metric_truth(base: R2RParameters, row: Mapping[str, object]) -> R2RParameters:
    """Return the pre-drift baseline used by the paper's sensitivity metric.

    Section 3.3 measures the apparent post-drift identification error relative
    to the fixed pre-drift parameter vector. The comparison-only campaign file
    independently confirms that convention through the extreme-J per-roller
    errors (for example, UW ``kt`` changes when ``J_UW`` is perturbed).
    """

    _ = row
    return base


def _write_drift_sysid_chart(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    title: str = "Drift vs Median SysID Error",
    paper_bands: Sequence[tuple[str, Sequence[float], str]] = (),
) -> str:
    """Draw one drift family.

    ``paper_bands`` carries the published values a per-leg line cannot show. The
    EA family is reported in the paper as a band across its five legs rather
    than per leg, so it is drawn as a shaded horizontal band instead of being
    left off the figure entirely.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1040, 560
    left, right, top, bottom = 82, 34, 66, 104
    categories = [str(row["drift_case"]) for row in rows]
    value_keys = ("dashboard_NF_percent", "dashboard_SN_percent", "paper_NF_percent", "paper_SN_percent")
    plotted_values = [
        float(row[key])
        for row in rows
        for key in value_keys
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    active_bands = [
        (label, [float(bounds[0]), float(bounds[1])], color)
        for label, bounds, color in paper_bands
        if bounds is not None and len(bounds) == 2
    ]
    plotted_values.extend(bound for _, bounds, _ in active_bands for bound in bounds)
    max_value = max(plotted_values) if plotted_values else 1.0
    y_hi = max(50.0, max_value * 1.12)

    def x_at(index: int) -> float:
        if len(categories) == 1:
            return left + (width - left - right) / 2.0
        return left + index * ((width - left - right) / (len(categories) - 1))

    def y_at(value: float) -> float:
        return (height - bottom) - (value / y_hi) * (height - bottom - top)

    series = [
        ("Dashboard NF", "dashboard_NF_percent", "#2f6f73", "", 0.0),
        ("Dashboard SN", "dashboard_SN_percent", "#d97732", "", 0.0),
        ("Paper NF", "paper_NF_percent", "#3b82f6", "8 5", -4.0),
        ("Paper SN", "paper_SN_percent", "#a855b8", "2 5", 4.0),
    ]
    series = [series_item for series_item in series if any(row.get(series_item[1]) is not None for row in rows)]
    elements: list[str] = []
    legend: list[str] = []
    band_elements: list[str] = []
    for band_index, (band_label, (band_lo, band_hi), band_color) in enumerate(active_bands):
        y_top = y_at(max(band_lo, band_hi))
        y_bottom = y_at(min(band_lo, band_hi))
        band_elements.append(
            f'<rect x="{left}" y="{y_top:.1f}" width="{width-right-left}" height="{max(2.0, y_bottom - y_top):.1f}" '
            f'fill="{band_color}" fill-opacity="0.18" />'
        )
        for edge in (y_top, y_bottom):
            band_elements.append(
                f'<line x1="{left}" y1="{edge:.1f}" x2="{width-right}" y2="{edge:.1f}" stroke="{band_color}" '
                f'stroke-width="1.6" stroke-dasharray="6 4" />'
            )
        band_elements.append(
            f'<text x="{width-right-8}" y="{y_top-6:.1f}" font-size="12" font-family="Arial" text-anchor="end" '
            f'fill="{band_color}">{escape(band_label)} {min(band_lo, band_hi):.1f}-{max(band_lo, band_hi):.1f}%</text>'
        )
        legend.append(
            f'<g transform="translate({left + (len(series) + band_index) * 220},32)">'
            f'<rect x="0" y="3" width="34" height="10" fill="{band_color}" fill-opacity="0.28" stroke="{band_color}" stroke-width="1.2"/>'
            f'<text x="44" y="13" font-size="13" font-family="Arial" fill="#243033">{escape(band_label)}</text></g>'
        )
    for series_index, (label, key, color, dash, x_offset) in enumerate(series):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        current_segment: list[tuple[float, float]] = []
        segments: list[list[tuple[float, float]]] = []
        for index, row in enumerate(rows):
            value = row.get(key)
            if value is None or not math.isfinite(float(value)):
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                continue
            current_segment.append((x_at(index) + x_offset, y_at(float(value))))
        if current_segment:
            segments.append(current_segment)
        for coords in segments:
            if len(coords) > 1:
                points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
                elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash_attr} points="{points}" />')
            for x, y in coords:
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.8" fill="{color}" stroke="#fff" stroke-width="1.5" />')
        legend_x = left + series_index * 220
        legend.append(
            f'<g transform="translate({legend_x},32)"><line x1="0" y1="8" x2="34" y2="8" stroke="{color}" stroke-width="3"{dash_attr}/>'
            f'<text x="44" y="13" font-size="13" font-family="Arial" fill="#243033">{escape(label)}</text></g>'
        )

    x_labels = []
    for index, label in enumerate(categories):
        x = x_at(index)
        x_labels.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+6}" stroke="#445" />')
        x_labels.append(
            f'<text x="{x:.1f}" y="{height-bottom+28}" font-size="12" font-family="Arial" text-anchor="middle" fill="#243033">{escape(label)}</text>'
        )

    y_grid = []
    for tick in range(0, int(y_hi) + 1, 10):
        y = y_at(float(tick))
        y_grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dedbce" stroke-width="1" />')
        y_grid.append(
            f'<text x="{left-12}" y="{y+4:.1f}" font-size="12" font-family="Arial" text-anchor="end" fill="#243033">{tick}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="24" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">{escape(title)}</text>
{''.join(legend)}
{''.join(y_grid)}
{''.join(band_elements)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<text x="{width/2}" y="{height-24}" font-size="15" font-family="Arial" text-anchor="middle" fill="#243033">Drift case</text>
<text x="24" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 24 {height/2})" fill="#243033">Median SysID error MARE_theta (%)</text>
{''.join(x_labels)}
{''.join(elements)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _drift_family_paper_bands(family: str) -> list[tuple[str, list[float], str]]:
    """Return the published bands to overlay on one drift family's chart."""

    reference = _drift_paper_family_reference().get(family) or {}
    bands: list[tuple[str, list[float], str]] = []
    for label, key, color in (
        ("Paper NF band", "paper_NF_band_percent", "#3b82f6"),
        ("Paper SN band", "paper_SN_band_percent", "#a855b8"),
    ):
        bounds = reference.get(key)
        if isinstance(bounds, list | tuple) and len(bounds) == 2:
            bands.append((label, [float(bounds[0]), float(bounds[1])], color))
    return bands


def _write_drift_grouped_bar_chart(rows: Sequence[Mapping[str, object]], path: Path, title: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1040, 560
    left, right, top, bottom = 82, 34, 66, 116
    categories = [str(row["drift_case"]) for row in rows]
    series = [
        ("Dashboard NF", "dashboard_NF_percent", "#2f6f73"),
        ("Dashboard SN", "dashboard_SN_percent", "#d97732"),
        ("Paper NF", "paper_NF_percent", "#5b8bd9"),
        ("Paper SN", "paper_SN_percent", "#a855b8"),
    ]
    plotted_values = [
        float(row[key])
        for row in rows
        for _, key, _ in series
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    max_value = max(plotted_values) if plotted_values else 1.0
    y_hi = max(50.0, math.ceil(max_value * 1.14 / 10.0) * 10.0)
    chart_width = width - left - right
    chart_height = height - bottom - top
    group_width = chart_width / max(1, len(categories))
    bar_gap = 8.0
    bar_width = min(44.0, max(20.0, (group_width - 58.0) / len(series)))

    def y_at(value: float) -> float:
        return (height - bottom) - (value / y_hi) * chart_height

    legend = []
    for series_index, (label, _, color) in enumerate(series):
        legend_x = left + series_index * 220
        legend.append(
            f'<g transform="translate({legend_x},32)"><rect x="0" y="1" width="28" height="14" rx="3" fill="{color}"/>'
            f'<text x="40" y="13" font-size="13" font-family="Arial" fill="#243033">{escape(label)}</text></g>'
        )

    y_grid = []
    for tick in range(0, int(y_hi) + 1, 10):
        y = y_at(float(tick))
        y_grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dedbce" stroke-width="1" />')
        y_grid.append(
            f'<text x="{left-12}" y="{y+4:.1f}" font-size="12" font-family="Arial" text-anchor="end" fill="#243033">{tick}</text>'
        )

    bars = []
    x_labels = []
    for index, row in enumerate(rows):
        group_left = left + index * group_width
        group_center = group_left + group_width / 2.0
        total_bar_width = len(series) * bar_width + (len(series) - 1) * bar_gap
        start_x = group_center - total_bar_width / 2.0
        x_labels.append(f'<line x1="{group_center:.1f}" y1="{height-bottom}" x2="{group_center:.1f}" y2="{height-bottom+6}" stroke="#445" />')
        x_labels.append(
            f'<text x="{group_center:.1f}" y="{height-bottom+28}" font-size="12" font-family="Arial" text-anchor="middle" fill="#243033">{escape(str(row["drift_case"]))}</text>'
        )
        for series_index, (_, key, color) in enumerate(series):
            raw_value = row.get(key)
            value = math.nan if raw_value is None else float(raw_value)
            x = start_x + series_index * (bar_width + bar_gap)
            if not math.isfinite(value):
                bars.append(
                    f'<text x="{x + bar_width / 2.0:.1f}" y="{height-bottom-8:.1f}" font-size="10" '
                    f'font-family="Arial" text-anchor="middle" fill="#8a3b2e">n/a</text>'
                )
                continue
            y = y_at(value)
            bar_height = height - bottom - y
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{color}" />'
                f'<text x="{x + bar_width / 2.0:.1f}" y="{y - 7:.1f}" font-size="11" font-family="Arial" text-anchor="middle" fill="#243033">{value:.1f}</text>'
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f2"/>
<text x="{left}" y="24" font-size="20" font-family="Arial" font-weight="700" fill="#1f2a2d">{escape(title)}</text>
{''.join(legend)}
{''.join(y_grid)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#445" stroke-width="1.5"/>
<text x="{width/2}" y="{height-24}" font-size="15" font-family="Arial" text-anchor="middle" fill="#243033">J drift case</text>
<text x="24" y="{height/2}" font-size="15" font-family="Arial" text-anchor="middle" transform="rotate(-90 24 {height/2})" fill="#243033">Median SysID error MARE_theta (%)</text>
{''.join(x_labels)}
{''.join(bars)}
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return str(path)


def _drift_cache_key(active_plant_runs: Sequence[tuple[str, R2RParameters, Mapping[str, Any]]]) -> dict[str, object]:
    return {
        "version": DRIFT_CACHE_VERSION,
        "plant_ids": [plant_id for plant_id, _, _ in active_plant_runs],
        "plant_count": len(active_plant_runs),
        "excitation_fraction": DRIFT_EXCITATION_TENSION_FRACTION,
        "sensor_noise_seeds": list(DRIFT_SENSOR_NOISE_SEEDS),
        "EA_levels": list(EA_DRIFT_PERCENT_LEVELS),
        "f_levels": list(F_DRIFT_PERCENT_LEVELS),
        "controller_tracks_drift": False,
        "excitation": DRIFT_EXCITATION_NAME,
        "campaign_group": DRIFT_CAMPAIGN_GROUP,
        "duration_s": drift_record_duration_s(),
        "controller_sample_time_s": 0.001,
        "log_sample_time_s_NF": DRIFT_LOG_SAMPLE_TIME_S,
        "log_sample_time_s_SN": DRIFT_LOG_SAMPLE_TIME_S,
        "sensor_lpf_hz_SN": 100.0,
        "estimator": "paper_eq8_weighted_pem_trf",
        "metric_truth": "pre_drift_baseline",
        "data_source": "dashboard_simulation",
    }


def _cached_drift_study(expected_cache_key: Mapping[str, object] | None = None) -> dict[str, object] | None:
    if expected_cache_key is not None and int(expected_cache_key.get("plant_count", 0)) > 1:
        summary_paths = [SUMMARY_DIR / "drift_summary_all_plants.json", SUMMARY_DIR / "drift_summary.json"]
    elif expected_cache_key is not None and expected_cache_key.get("plant_ids"):
        plant_id = str(list(expected_cache_key["plant_ids"])[0])
        summary_paths = [SUMMARY_DIR / f"drift_summary_{plant_id}.json"]
    else:
        summary_paths = [SUMMARY_DIR / "drift_summary.json"]
    for summary_path in summary_paths:
        if not summary_path.exists():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required_keys = (
            "comparison_rows",
            "family_rows",
            "per_roller_rows",
            "per_roller_summary",
            "dashboard_family_summary",
            "reel_radius_sensitivity",
            "raw_rows",
            "run_metadata",
            "drift_EA_plot_path",
            "drift_f_plot_path",
            "drift_J_plot_path",
            "drift_combined_plot_path",
            "csv_path",
            "raw_csv_path",
        )
        if any(key not in payload for key in required_keys):
            continue
        if expected_cache_key is not None and payload.get("cache_key") != dict(expected_cache_key):
            continue
        # The published family reference is read straight from the paper JSON, so
        # it is refreshed rather than trusted from an older cached summary. The
        # family charts are redrawn for the same reason: they are pure functions
        # of the cached rows plus the paper reference, so a change in how the
        # paper is drawn must not need a 400-run recalculation to appear.
        payload["paper_family_reference"] = _drift_paper_family_reference()
        cached_comparison_rows = [dict(row) for row in payload.get("comparison_rows", [])]
        cached_family_rows = {
            family: [row for row in cached_comparison_rows if row.get("drift_family") == family]
            for family in ("EA", "f", "J", "combined")
        }
        redraw_plant_ids = list(payload.get("cache_key", {}).get("plant_ids", []))
        redraw_suffix = "" if len(redraw_plant_ids) != 1 else f"_{redraw_plant_ids[0]}"
        if cached_comparison_rows:
            payload["family_rows"] = cached_family_rows
            payload["drift_EA_plot_path"] = _write_drift_sysid_chart(
                cached_family_rows["EA"],
                FIGURES_DIR / f"drift_sysid_error_EA{redraw_suffix}.svg",
                "EA Drift vs Median SysID Error",
                paper_bands=_drift_family_paper_bands("EA"),
            )
            payload["drift_f_plot_path"] = _write_drift_sysid_chart(
                cached_family_rows["f"],
                FIGURES_DIR / f"drift_sysid_error_f{redraw_suffix}.svg",
                "f Drift vs Median SysID Error",
                paper_bands=_drift_family_paper_bands("f"),
            )
            payload["drift_J_plot_path"] = _write_drift_grouped_bar_chart(
                cached_family_rows["J"],
                FIGURES_DIR / f"drift_sysid_error_J{redraw_suffix}.svg",
                "J Drift vs Median SysID Error",
            )
            payload["drift_combined_plot_path"] = _write_drift_grouped_bar_chart(
                cached_family_rows["combined"],
                FIGURES_DIR / f"drift_sysid_error_combined{redraw_suffix}.svg",
                "Combined Drift (EA/J/f) vs Median SysID Error",
            )
            payload["plot_path"] = payload["drift_EA_plot_path"]
        artifact_paths = [
            Path(str(payload.get("plot_path") or payload["drift_EA_plot_path"])),
            Path(str(payload["drift_EA_plot_path"])),
            Path(str(payload["drift_f_plot_path"])),
            Path(str(payload["drift_J_plot_path"])),
            Path(str(payload["drift_combined_plot_path"])),
            Path(str(payload["csv_path"])),
            Path(str(payload["raw_csv_path"])),
        ]
        try:
            for artifact_path in artifact_paths:
                artifact_path.resolve().relative_to(PROJECT_ROOT)
                if not artifact_path.exists():
                    raise FileNotFoundError(str(artifact_path))
        except (OSError, ValueError):
            comparison_rows = [dict(row) for row in payload["comparison_rows"]]
            family_rows = {
                family: [row for row in comparison_rows if row.get("drift_family") == family]
                for family in ("EA", "f", "J", "combined")
            }
            payload["family_rows"] = family_rows
            cached_plant_ids = list(payload.get("cache_key", {}).get("plant_ids", []))
            artifact_suffix = "" if len(cached_plant_ids) != 1 else f"_{cached_plant_ids[0]}"
            payload["drift_EA_plot_path"] = _write_drift_sysid_chart(
                family_rows["EA"],
                FIGURES_DIR / f"drift_sysid_error_EA{artifact_suffix}.svg",
                "EA Drift vs Median SysID Error",
                paper_bands=_drift_family_paper_bands("EA"),
            )
            payload["drift_f_plot_path"] = _write_drift_sysid_chart(
                family_rows["f"],
                FIGURES_DIR / f"drift_sysid_error_f{artifact_suffix}.svg",
                "f Drift vs Median SysID Error",
                paper_bands=_drift_family_paper_bands("f"),
            )
            payload["drift_J_plot_path"] = _write_drift_grouped_bar_chart(
                family_rows["J"],
                FIGURES_DIR / f"drift_sysid_error_J{artifact_suffix}.svg",
                "J Drift vs Median SysID Error",
            )
            payload["drift_combined_plot_path"] = _write_drift_grouped_bar_chart(
                family_rows["combined"],
                FIGURES_DIR / f"drift_sysid_error_combined{artifact_suffix}.svg",
                "Combined Drift (EA/J/f) vs Median SysID Error",
            )
            payload["plot_path"] = payload["drift_EA_plot_path"]
            if comparison_rows:
                payload["csv_path"] = _write_rows_csv(
                    f"drift_dashboard_vs_paper_comparison{artifact_suffix}.csv",
                    comparison_rows,
                    [key for key in comparison_rows[0].keys() if key != "__provenance"],
                )
            cached_raw_rows = [dict(row) for row in payload.get("raw_rows", [])]
            if cached_raw_rows:
                payload["raw_csv_path"] = _write_rows_csv(
                    f"drift_simulated_plant_runs{artifact_suffix}.csv",
                    cached_raw_rows,
                    [key for key in cached_raw_rows[0].keys() if key != "__provenance"],
                )
            _write_summary(summary_path.name, payload)
        plot_path = str(payload.get("plot_path") or payload["drift_EA_plot_path"])
        csv_path = str(payload.get("csv_path") or "")
        artifact = _artifact_payload(payload, plot_path, str(summary_path), csv_path)
        artifact["drift_EA_plot_path"] = payload["drift_EA_plot_path"]
        artifact["drift_f_plot_path"] = payload["drift_f_plot_path"]
        artifact["drift_J_plot_path"] = payload["drift_J_plot_path"]
        artifact["drift_combined_plot_path"] = payload["drift_combined_plot_path"]
        artifact["raw_csv_path"] = payload["raw_csv_path"]
        artifact["per_roller_csv_path"] = payload.get("per_roller_csv_path")
        return artifact
    return None


def _drift_active_parameter_names(drift_family: str) -> tuple[str, ...]:
    _ = drift_family
    # The professor review defines MARE_theta over all seven identified
    # parameters for every experiment; it is not a family-subset score.
    return tuple(PARAMETER_NAMES)


def _active_mare_theta_percent(error_table: Sequence[Mapping[str, object]], drift_family: str) -> tuple[float, tuple[str, ...]]:
    active_names = _drift_active_parameter_names(drift_family)
    errors = [
        float(row["relative_error"])
        for row in error_table
        if str(row["parameter"]) in active_names
    ]
    if not errors:
        raise ValueError(f"No active SysID errors found for drift family {drift_family!r}.")
    return 100.0 * statistics.mean(abs(error) for error in errors), active_names


def _noise_augmented_mare_theta_percent(
    noisy_estimates: Mapping[str, float],
    noise_free_estimates: Mapping[str, float],
    truth: R2RParameters,
    active_names: Sequence[str],
) -> float:
    truth_values = truth.sysid_values()
    absolute_terms = []
    for name in active_names:
        truth_value = abs(float(truth_values[name]))
        denom = truth_value if truth_value > 1e-12 else 1.0
        systematic_error = (float(noise_free_estimates[name]) - float(truth_values[name])) / denom
        stochastic_delta = (float(noisy_estimates[name]) - float(noise_free_estimates[name])) / denom
        absolute_terms.append(abs(systematic_error + stochastic_delta))
    if not absolute_terms:
        raise ValueError("No active estimates found for noise-augmented drift MARE.")
    return 100.0 * statistics.mean(absolute_terms)


def drift_study(
    params: R2RParameters | None = None,
    plant_runs: Sequence[tuple[str, R2RParameters, Mapping[str, Any]]] | None = None,
    prefer_cache: bool = False,
) -> dict[str, object]:
    """Compare drift vs median SysID error under NF and SN, with paper references."""

    active_params = params or R2RParameters()
    # Record length is read from the paper's edge-level schedule table, never
    # hardcoded: ledger campaign 2 is a group-A campaign and its E_Toggle record
    # is 17 s (settle 2 s; edges at 2/7/12 s). The previous hardcoded 7 s cut the
    # record off before the second and third toggle edges, so the campaign was
    # in effect running ET1 under the E_Toggle label.
    duration_s = drift_record_duration_s()
    active_plant_runs = (
        [(str(plant_id), plant_params, dict(plant_meta)) for plant_id, plant_params, plant_meta in plant_runs]
        if plant_runs
        else [("selected", active_params, {"plant_id": "selected", "sensor_noise_sigma_N": 0.15})]
    )
    if not active_plant_runs:
        raise ValueError("At least one plant is required.")
    run_metadata = _run_metadata(
        "drift",
        plant_scope="all_plants" if len(active_plant_runs) > 1 else "single_plant",
        run_settings={
            "duration_s": duration_s,
            "record_duration_source": (
                "data/model_inputs/excitation_schedules.csv "
                f"[{DRIFT_EXCITATION_NAME}, {DRIFT_CAMPAIGN_GROUP}, record 0]"
            ),
            "excitation": DRIFT_EXCITATION_NAME,
            "campaign_group": DRIFT_CAMPAIGN_GROUP,
            "settle_s": float(
                excitation_schedule(DRIFT_EXCITATION_NAME, DRIFT_CAMPAIGN_GROUP, 0).settle_s
            ),
            "drift_excitation_fraction_of_Tref": DRIFT_EXCITATION_TENSION_FRACTION,
            "drift_sensor_noise_seeds": list(DRIFT_SENSOR_NOISE_SEEDS),
            "controller_tracks_drift": False,
            "controller_sample_time_s": 0.001,
            "log_sample_time_s_NF": DRIFT_LOG_SAMPLE_TIME_S,
            "log_sample_time_s_SN": DRIFT_LOG_SAMPLE_TIME_S,
            "sensor_lpf_hz_SN": 100.0,
            "estimator": "paper_eq8_weighted_pem_trf",
            # Paper Section 2.4: theta_init = 1.01 * theta, reels symmetrized,
            # box constrained one decade either side. These mirror the estimator
            # defaults the study actually calls with; the previous 1.5 was a
            # stale transcription that never matched the code.
            "pem_initial_scale": CANONICAL_INITIAL_SCALE,
            "pem_parameter_scale_bounds": [1.0 / CANONICAL_BOUND_DECADE, CANONICAL_BOUND_DECADE],
            "pem_max_nfev": 150,
            "tension_integral_time": "per_plant_auto_Ti",
            "steady_velocity_baseline": "nominal_tension_current_line_speed",
            "velocity_correction_clamp": None,
            "metric_truth": "pre_drift_baseline",
            "metric_parameters": list(PARAMETER_NAMES),
            "data_source": "dashboard_simulation",
            "reference_data_usage": "comparison_only",
        },
        value_status="computed_raw",
    )
    cache_key = _drift_cache_key(active_plant_runs)
    if prefer_cache and plant_runs:
        cached = _cached_drift_study(cache_key)
        if cached is not None:
            return cached

    reference_rows = _drift_reference_rows()
    comparison_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for reference_row in reference_rows:
        if bool(reference_row.get("plot_only")):
            continue
        drift_fn, scale_meta = _drift_case_function(reference_row, duration_s)
        dashboard_by_condition: dict[str, float] = {}
        run_counts_by_condition: dict[str, dict[str, int]] = {}
        def run_plant_condition(
            plant_run: tuple[str, R2RParameters, Mapping[str, Any]],
            condition: str,
            seed: int,
        ) -> tuple[float, dict[str, object], dict[str, float] | None]:
            plant_id, plant_params, plant_meta = plant_run
            final_truth = _drift_metric_truth(plant_params, reference_row)
            noise_sigma = float(plant_meta.get("sensor_noise_sigma_N", SENSOR_NOISE_TENSION_N))
            config = SimulationConfig(
                duration_s=duration_s,
                controller_sample_time_s=0.001,
                log_sample_time_s=DRIFT_LOG_SAMPLE_TIME_S,
                line_speed_m_s=float(plant_meta.get("v_ref_m_s", plant_params.feeder_velocity_m_s)),
                sensor_noise_tension_N=0.0 if condition == "NF" else noise_sigma,
                sensor_noise_omega_rad_s=0.0,
                sensor_lpf_hz=None if condition == "NF" else 100.0,
                controller_tracks_drift=False,
                output_name=f"drift_{plant_id}_{condition}_seed{seed}_{str(reference_row['drift_case']).replace(' ', '_').replace('%', 'pct')}.csv",
                seed=seed,
            )
            excitation_amplitude = DRIFT_EXCITATION_TENSION_FRACTION * float(plant_meta.get("T_ref_N", 12.0))
            active_names = _drift_active_parameter_names(str(reference_row["drift_family"]))
            try:
                sim = simulate(
                    plant_params,
                    controller_config=ControllerConfig(
                        line_speed_m_s=float(plant_meta.get("v_ref_m_s", plant_params.feeder_velocity_m_s)),
                        target_tension_N=plant_params.tension_ref_N,
                        TI_s=auto_tension_integral_time_s(
                            plant_params,
                            float(plant_meta.get("v_ref_m_s", plant_params.feeder_velocity_m_s)),
                        ),
                        high_ea_kp_cap_enabled=False,
                        feedforward_uses_measured_omega=True,
                        paper_velocity_gain_enabled=True,
                        velocity_correction_limit_fraction=None,
                        steady_velocity_uses_dynamic_target=False,
                    ),
                    config=config,
                    excitation=get_excitation_profile("E_Toggle", excitation_amplitude),
                    drift=drift_fn,
                    output_dir=DATA_DIR,
                )
                sysid = estimate_parameters_weighted_pem(
                    sim.rows,
                    nominal_params=plant_params,
                    true_params=final_truth,
                    max_nfev=150,
                )
                raw_error_percent = 100.0 * sysid.mare_theta
                error_percent = raw_error_percent
                all_parameter_mare = 100.0 * sysid.mare_theta
                estimates = dict(sysid.estimates)
                relative_error_percent = {
                    str(item["parameter"]): 100.0 * float(item["relative_error"])
                    for item in sysid.error_table
                }
                diagnostics = dict(sysid.diagnostics)
                samples = float(len(sim.rows))
                value_status = "computed_raw"
            except (TypeError, KeyError, AttributeError):
                raise
            except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
                raw_error_percent = math.nan
                error_percent = math.nan
                all_parameter_mare = math.nan
                estimates = {}
                relative_error_percent = {}
                diagnostics = {}
                samples = 0.0
                value_status = f"simulation_unstable:{type(exc).__name__}:{exc}"
            metric_variant = "paper_eq7_one_step_pem_pre_drift_baseline_all7_mare"
            raw_row = {
                **_metadata_row_fields(run_metadata, "computed_raw"),
                "drift_case": reference_row["drift_case"],
                "drift_family": reference_row["drift_family"],
                "drift_kind": reference_row["drift_kind"],
                "condition": condition,
                "plant_id": plant_id,
                "dashboard_MARE_theta_percent": error_percent if math.isfinite(error_percent) else None,
                "raw_estimator_MARE_theta_percent": raw_error_percent if math.isfinite(raw_error_percent) else None,
                "all_parameter_MARE_theta_percent": all_parameter_mare if math.isfinite(all_parameter_mare) else None,
                "active_parameters": ",".join(active_names),
                "controller_tracks_drift": False,
                "controller_sample_time_s": 0.001,
                "log_sample_time_s": DRIFT_LOG_SAMPLE_TIME_S,
                "operating_point_TI_s": auto_tension_integral_time_s(
                    plant_params,
                    float(plant_meta.get("v_ref_m_s", plant_params.feeder_velocity_m_s)),
                ),
                "estimator": "paper_eq8_weighted_pem_trf",
                "optimizer_success": diagnostics.get("success"),
                "optimizer_status": diagnostics.get("status"),
                "optimizer_nfev": diagnostics.get("nfev"),
                "optimizer_njev": diagnostics.get("njev"),
                "optimizer_cost": diagnostics.get("cost"),
                "optimizer_optimality": diagnostics.get("optimality"),
                "normalized_residual_rmse": diagnostics.get("normalized_residual_rmse"),
                "optimizer_lower_bound_parameters": ",".join(
                    str(name) for name in diagnostics.get("lower_bound_parameters", [])
                ),
                "optimizer_upper_bound_parameters": ",".join(
                    str(name) for name in diagnostics.get("upper_bound_parameters", [])
                ),
                **{
                    f"estimate_{name}": estimates.get(name)
                    for name in PARAMETER_NAMES
                },
                **{
                    f"relative_error_{name}_percent": relative_error_percent.get(name)
                    for name in PARAMETER_NAMES
                },
                "metric_truth": "pre_drift_baseline",
                "samples": samples,
                "seed": seed,
                "excitation_fraction_of_Tref": DRIFT_EXCITATION_TENSION_FRACTION,
                "excitation_amplitude_V": excitation_amplitude,
                "excitation_amplitude_N": excitation_amplitude,
                "metric_variant": metric_variant,
                "value_status": value_status,
                **scale_meta,
            }
            return error_percent, raw_row, estimates if estimates else None

        for condition in ("NF", "SN"):
            plant_errors = []
            condition_seeds = (0,) if condition == "NF" else DRIFT_SENSOR_NOISE_SEEDS
            max_workers = min(10, max(1, len(active_plant_runs) * len(condition_seeds)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(run_plant_condition, plant_run, condition, seed)
                    for seed in condition_seeds
                    for plant_run in active_plant_runs
                ]
                for future in as_completed(futures):
                    error_percent, raw_row, _ = future.result()
                    if math.isfinite(error_percent):
                        plant_errors.append(error_percent)
                    raw_rows.append(raw_row)
            dashboard_by_condition[condition] = statistics.median(plant_errors) if plant_errors else math.nan
            expected_run_count = len(active_plant_runs) * len(condition_seeds)
            run_counts_by_condition[condition] = {
                "valid": len(plant_errors),
                "failed": expected_run_count - len(plant_errors),
                "expected": expected_run_count,
            }

        drift_family = str(reference_row["drift_family"])
        base_iteration_note = (
            f"NF uses one deterministic simulation per plant. SN uses {len(DRIFT_SENSOR_NOISE_SEEDS)} "
            f"noise seeds per plant and reports the median across plant/seed results. Drift excitation "
            f"amplitude is {DRIFT_EXCITATION_TENSION_FRACTION:.3f}*T_ref for every plant."
        )
        base_iteration_note += (
            " Both conditions use T_log=20 ms; SN adds 0.3% T_max tension noise and a 100 Hz LPF. "
            "The estimator independently minimizes paper Eq. (7) one-step tension residuals with TRF. "
            "MARE_theta is the arithmetic mean of the seven absolute relative parameter errors against "
            "the pre-drift baseline. Reference CSV values are used only after calculation for comparison."
        )
        if (
            reference_row.get("paper_NF_percent") is None
            and reference_row.get("paper_SN_percent") is None
        ):
            base_iteration_note += (
                " The paper shows this friction level only as an axis tick and publishes no Phase-3 "
                "NF/SN value, so the dashboard result is reported without an invented paper comparison."
            )

        paper_nf = (
            None
            if reference_row.get("paper_NF_percent") is None
            else float(reference_row["paper_NF_percent"])
        )
        paper_sn = (
            None
            if reference_row.get("paper_SN_percent") is None
            else float(reference_row["paper_SN_percent"])
        )
        row_value_status = (
            "computed_raw"
            if all(counts["failed"] == 0 for counts in run_counts_by_condition.values())
            else "computed_partial_nonfinite_runs_reported"
        )
        comparison_rows.append(
            {
                **_metadata_row_fields(run_metadata, row_value_status),
                "drift_case": reference_row["drift_case"],
                "drift_percent": reference_row["drift_percent"],
                "drift_family": reference_row["drift_family"],
                "drift_kind": reference_row["drift_kind"],
                "J_UW_percent": reference_row.get("J_UW_percent"),
                "J_RW_percent": reference_row.get("J_RW_percent"),
                "raw_dashboard_NF_percent": dashboard_by_condition["NF"],
                "dashboard_NF_percent": dashboard_by_condition["NF"],
                "displayed_dashboard_NF_percent": dashboard_by_condition["NF"],
                "paper_NF_percent": paper_nf,
                "difference_NF_percent": (
                    None
                    if paper_nf is None
                    else dashboard_by_condition["NF"] - paper_nf
                ),
                "raw_dashboard_SN_percent": dashboard_by_condition["SN"],
                "dashboard_SN_percent": dashboard_by_condition["SN"],
                "displayed_dashboard_SN_percent": dashboard_by_condition["SN"],
                "paper_SN_percent": paper_sn,
                "difference_SN_percent": None if paper_sn is None else dashboard_by_condition["SN"] - paper_sn,
                "valid_run_count_NF": run_counts_by_condition["NF"]["valid"],
                "failed_run_count_NF": run_counts_by_condition["NF"]["failed"],
                "expected_run_count_NF": run_counts_by_condition["NF"]["expected"],
                "valid_run_count_SN": run_counts_by_condition["SN"]["valid"],
                "failed_run_count_SN": run_counts_by_condition["SN"]["failed"],
                "expected_run_count_SN": run_counts_by_condition["SN"]["expected"],
                "display_adjustment_type": "none_independent_simulation",
                "value_status": row_value_status,
                "paper_reference_type": reference_row.get("paper_reference_type", "paper_reference"),
                # v5 can publish one condition of a leg and not the other: the
                # friction legs print a noise-free pair with no matching SN
                # value, and the EA family is published as a band rather than
                # per leg. The status is therefore resolved per condition, with
                # the row-level key kept as a summary for existing consumers.
                "paper_value_status": (
                    "published_phase3_median"
                    if paper_nf is not None and paper_sn is not None
                    else "partially_published"
                    if paper_nf is not None or paper_sn is not None
                    else "unpublished_axis_tick_no_phase3_result"
                ),
                "paper_value_status_NF": (
                    "published_phase3_median"
                    if paper_nf is not None
                    else "unpublished_axis_tick_no_phase3_result"
                ),
                "paper_value_status_SN": (
                    "published_phase3_median"
                    if paper_sn is not None
                    else "unpublished_axis_tick_no_phase3_result"
                ),
                "dashboard_metric_equation": "drift_MARE_theta = mean_i(abs((theta_hat_i,d-theta_i,0)/theta_i,0)) x 100 over all seven identified parameters, evaluated against the pre-drift baseline",
                "paper_note": reference_row["paper_note"],
                "dashboard_iteration_note": base_iteration_note,
                "__provenance": {
                    "raw_dashboard_NF_percent": "dashboard_simulation",
                    "raw_dashboard_SN_percent": "dashboard_simulation",
                    "dashboard_NF_percent": "dashboard_simulation_median",
                    "dashboard_SN_percent": "dashboard_simulation_median",
                    "displayed_dashboard_NF_percent": "dashboard_simulation_median",
                    "displayed_dashboard_SN_percent": "dashboard_simulation_median",
                    "paper_NF_percent": str(reference_row.get("paper_reference_type", "paper_reference")),
                    "paper_SN_percent": str(reference_row.get("paper_reference_type", "paper_reference")),
                    "difference_NF_percent": (
                        "not_available_unpublished_reference"
                        if paper_nf is None
                        else "computed_dashboard_minus_reference"
                    ),
                    "difference_SN_percent": (
                        "not_available_unpublished_reference"
                        if paper_sn is None
                        else "computed_dashboard_minus_reference"
                    ),
                },
            }
        )

    dominant = max(
        (row for row in comparison_rows if row["drift_family"] != "combined"),
        key=lambda row: float(row["dashboard_SN_percent"]),
    )
    family_rows: dict[str, list[dict[str, object]]] = {
        family: [row for row in comparison_rows if row["drift_family"] == family]
        for family in ("EA", "f", "J", "combined")
    }
    chart_rows: dict[str, list[dict[str, object]]] = {}
    for family in ("EA", "f", "J"):
        chart_rows[family] = family_rows[family]
    artifact_suffix = "" if len(active_plant_runs) > 1 else f"_{active_plant_runs[0][0]}"
    drift_EA_plot_path = _write_drift_sysid_chart(
        chart_rows["EA"],
        FIGURES_DIR / f"drift_sysid_error_EA{artifact_suffix}.svg",
        "EA Drift vs Median SysID Error",
        paper_bands=_drift_family_paper_bands("EA"),
    )
    drift_f_plot_path = _write_drift_sysid_chart(
        chart_rows["f"],
        FIGURES_DIR / f"drift_sysid_error_f{artifact_suffix}.svg",
        "f Drift vs Median SysID Error",
        paper_bands=_drift_family_paper_bands("f"),
    )
    drift_J_plot_path = _write_drift_grouped_bar_chart(
        family_rows["J"],
        FIGURES_DIR / f"drift_sysid_error_J{artifact_suffix}.svg",
        "J Drift vs Median SysID Error",
    )
    drift_combined_plot_path = _write_drift_grouped_bar_chart(
        family_rows["combined"],
        FIGURES_DIR / f"drift_sysid_error_combined{artifact_suffix}.svg",
        "Combined Drift (EA/J/f) vs Median SysID Error",
    )
    plot_path = drift_EA_plot_path
    csv_fields = (
        *METADATA_FIELDS,
        "drift_case",
        "drift_percent",
        "drift_family",
        "drift_kind",
        "J_UW_percent",
        "J_RW_percent",
        "raw_dashboard_NF_percent",
        "dashboard_NF_percent",
        "displayed_dashboard_NF_percent",
        "paper_NF_percent",
        "difference_NF_percent",
        "raw_dashboard_SN_percent",
        "dashboard_SN_percent",
        "displayed_dashboard_SN_percent",
        "paper_SN_percent",
        "difference_SN_percent",
        "valid_run_count_NF",
        "failed_run_count_NF",
        "expected_run_count_NF",
        "valid_run_count_SN",
        "failed_run_count_SN",
        "expected_run_count_SN",
        "display_adjustment_type",
        "paper_reference_type",
        "paper_value_status",
        "paper_note",
        "dashboard_iteration_note",
    )
    csv_path = _write_rows_csv(
        f"drift_dashboard_vs_paper_comparison{artifact_suffix}.csv",
        comparison_rows,
        csv_fields,
    )
    raw_csv_fields = (
        *METADATA_FIELDS,
        "drift_case",
        "drift_family",
        "drift_kind",
        "condition",
        "plant_id",
        "dashboard_MARE_theta_percent",
        "raw_estimator_MARE_theta_percent",
        "all_parameter_MARE_theta_percent",
        "active_parameters",
        "controller_tracks_drift",
        "controller_sample_time_s",
        "log_sample_time_s",
        "operating_point_TI_s",
        "estimator",
        "optimizer_success",
        "optimizer_status",
        "optimizer_nfev",
        "optimizer_njev",
        "optimizer_cost",
        "optimizer_optimality",
        "normalized_residual_rmse",
        "optimizer_lower_bound_parameters",
        "optimizer_upper_bound_parameters",
        *(f"estimate_{name}" for name in PARAMETER_NAMES),
        *(f"relative_error_{name}_percent" for name in PARAMETER_NAMES),
        "metric_truth",
        "samples",
        "seed",
        "excitation_fraction_of_Tref",
        "excitation_amplitude_V",
        "excitation_amplitude_N",
        "metric_variant",
        "EA_scale",
        "f_scale",
        "J_scale",
        "J_UW_scale",
        "J_RW_scale",
    )
    raw_csv_path = _write_rows_csv(
        f"drift_simulated_plant_runs{artifact_suffix}.csv",
        raw_rows,
        raw_csv_fields,
    )
    per_roller_rows = _drift_per_roller_rows(raw_rows)
    per_roller_csv_path = _write_rows_csv(
        f"drift_per_roller_diagnostic{artifact_suffix}.csv",
        per_roller_rows,
        (
            "drift_case",
            "condition",
            "roller",
            "dashboard_kt_error_percent",
            "paper_kt_error_percent",
            "difference_percent",
            "run_count",
            "metric",
            "paper_value_status",
            "value_status",
        ),
    )
    payload = {
        "study": "drift",
        "metrics": comparison_rows,
        "comparison_rows": comparison_rows,
        "family_rows": family_rows,
        "paper_family_reference": _drift_paper_family_reference(),
        "dashboard_family_summary": _drift_dashboard_family_summary(comparison_rows),
        "per_roller_rows": per_roller_rows,
        "per_roller_summary": _drift_per_roller_summary(per_roller_rows),
        "per_roller_csv_path": per_roller_csv_path,
        "reel_radius_sensitivity": _reel_radius_sensitivity(),
        "raw_rows": raw_rows,
        "run_metadata": run_metadata,
        "dominant_drift_case_dashboard_SN": dominant["drift_case"],
        "supports_J_drift_dominance": str(dominant["drift_family"]) == "J",
        "cache_key": cache_key,
        "dashboard_iteration_note": (
            f"Dashboard drift values are recalculated from the current run: NF is the median across plants; SN is the "
            f"median across plants and seeds {list(DRIFT_SENSOR_NOISE_SEEDS)}. The drift-study excitation is "
            f"{DRIFT_EXCITATION_TENSION_FRACTION:.3f}*T_ref, matching the paper's stated default tension-step amplitude. "
            "Both conditions use T_log=20 ms; the controller runs at 1 ms with per-plant auto_Ti. Paper Eq. (7) "
            "nonlinear one-step tension PEM/TRF identifies every freshly simulated trajectory. Controller parameters "
            "remain fixed at their pre-drift values while each run uses one fixed perturbed plant. "
            "For every family, MARE_theta is the arithmetic mean across all seven identified parameters, as defined in "
            "the professor review. Reference CSV values are comparison-only and never feed the dashboard calculation."
            " Per-row valid/failed/expected run counts are reported explicitly; a median excludes only runs whose "
            "physical state became non-finite. The friction sweep is freshly calculated at -30%, -15%, 0%, +15%, "
            "and +30%; the paper supplies comparison medians only at the two endpoints, so intermediate paper cells "
            "remain explicitly unpublished. Raw output persists Eq. (7) optimizer, bound, estimate, and per-parameter "
            "error diagnostics."
        ),
        "drift_excitation_fraction_of_Tref": DRIFT_EXCITATION_TENSION_FRACTION,
        "drift_sensor_noise_seeds": list(DRIFT_SENSOR_NOISE_SEEDS),
        "controller_tracks_drift": False,
        "data_source": "dashboard_simulation",
        "reference_data_usage": "comparison_only",
        "plot_path": plot_path,
        "drift_EA_plot_path": drift_EA_plot_path,
        "drift_f_plot_path": drift_f_plot_path,
        "drift_J_plot_path": drift_J_plot_path,
        "drift_combined_plot_path": drift_combined_plot_path,
        "csv_path": csv_path,
        "raw_csv_path": raw_csv_path,
    }
    summary_filename = f"drift_summary{artifact_suffix}.json"
    summary_path = _write_summary(summary_filename, payload)
    if len(active_plant_runs) > 1:
        _write_summary("drift_summary_all_plants.json", payload)
    artifact = _artifact_payload(payload, plot_path, summary_path, csv_path)
    artifact["raw_csv_path"] = raw_csv_path
    artifact["drift_EA_plot_path"] = drift_EA_plot_path
    artifact["drift_f_plot_path"] = drift_f_plot_path
    artifact["drift_J_plot_path"] = drift_J_plot_path
    artifact["drift_combined_plot_path"] = drift_combined_plot_path
    artifact["per_roller_csv_path"] = per_roller_csv_path
    return artifact




