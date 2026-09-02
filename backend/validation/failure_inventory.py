"""Failure inventory helpers for validation report triage."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "reports" / "validation_summary"


def _status_value(row: Mapping[str, Any]) -> str | None:
    for key in ("pass_fail", "status", "validation_status"):
        value = row.get(key)
        if value is not None:
            return str(value)
    if "passed" in row:
        return "PASS" if bool(row["passed"]) else "FAIL"
    return None


def _passed(status: str | None) -> bool | None:
    if status is None:
        return None
    normalized = status.upper()
    if normalized == "PASS" or normalized == "OK" or normalized == "TRUE":
        return True
    if normalized.startswith("OK"):
        return True
    if normalized in {"FAIL", "CHECK", "FALSE"}:
        return False
    if normalized.startswith("FAILED"):
        return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _classify(row: Mapping[str, Any], module: str) -> str:
    text = " ".join(str(value) for value in row.values()).lower()
    if "rmse_theta" in text or "rmse_θ" in text:
        return "wrong validation metric"
    if "computed_raw" in text and "paper_reference" in text:
        return "paper-match gap after live computation"
    if "lpf" in text or "noise" in text or "excitation" in text or "line_speed" in text:
        return "incorrect excitation or line-speed input"
    if "t0" in text or "t4" in text or "boundary" in text:
        return "incorrect boundary handling T0=0 and T4=0"
    if "topology" in text or "span" in text:
        return "wrong 3-span topology reconstruction"
    if "theta" in text or "mare" in text:
        return "wrong theta parameter identification"
    if module in {"closed_loop_damping", "noise_aware_logging_lpf"}:
        return "paper-match gap after live computation"
    return "unclassified validation mismatch"


def _comparison_error(row: Mapping[str, Any]) -> float | None:
    direct = _safe_float(row.get("error_percent") or row.get("validation_error_percent"))
    if direct is not None:
        return direct
    for paper_key, dashboard_key in (
        ("paper_MARE_theta_percent", "dashboard_MARE_theta_percent"),
        ("paper_NF_percent", "dashboard_NF_percent"),
        ("paper_SN_percent", "dashboard_SN_percent"),
        ("paper_MARE_theta", "dashboard_MARE_theta"),
    ):
        paper = _safe_float(row.get(paper_key))
        dashboard = _safe_float(row.get(dashboard_key))
        if paper is not None and dashboard is not None and abs(paper) > 1e-12:
            return abs(paper - dashboard) / abs(paper) * 100.0
    return None


def _inventory_record(module: str, source: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
    status = _status_value(row)
    passed = _passed(status)
    if passed is None:
        return None
    return {
        "module": module,
        "source": source,
        "test_id": row.get("criterion")
        or row.get("case_label")
        or row.get("strategy")
        or row.get("condition")
        or row.get("drift_case")
        or source,
        "status": "PASS" if passed else "FAIL",
        "raw_status": status,
        "error_percent": _comparison_error(row),
        "suspected_cause": _classify(row, module),
        "evidence": {
            key: row.get(key)
            for key in (
                "criterion",
                "case_label",
                "condition",
                "Tlog_ms",
                "LPF",
                "kp_star",
                "strategy",
                "drift_case",
                "paper_MARE_theta",
                "dashboard_MARE_theta",
                "paper_MARE_theta_percent",
                "dashboard_MARE_theta_percent",
                "paper_NF_percent",
                "dashboard_NF_percent",
                "paper_SN_percent",
                "dashboard_SN_percent",
            )
            if key in row
        },
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_json_rows(path: Path, module: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if not payload:
        return []
    rows: list[dict[str, Any]] = []
    for key in ("acceptance_criteria", "comparison_table", "comparison_rows"):
        values = payload.get(key)
        if isinstance(values, list):
            for row in values:
                if isinstance(row, Mapping):
                    record = _inventory_record(module, f"{path.name}:{key}", row)
                    if record is not None:
                        rows.append(record)
    return rows


def _collect_csv_rows(path: Path, module: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record = _inventory_record(module, path.name, row)
                if record is not None:
                    rows.append(record)
    except OSError:
        return []
    return rows


def build_failure_inventory(summary_dir: Path | None = None) -> dict[str, Any]:
    """Return a pass/fail inventory with suspected-cause labels."""

    active_dir = summary_dir or SUMMARY_DIR
    module_files = {
        "noise_aware_logging_lpf": (
            "noiseLpf_noise_aware_logging_summary.json",
            "noiseLpf_noise_aware_logging_summary.csv",
        ),
        "closed_loop_damping": (
            "closed_loop_damping_summary.json",
            "closed_loop_damping_dashboard_vs_paper_comparison.csv",
        ),
        "excitation": (
            "excitation_summary.json",
            "excitation_summary_dashboard_vs_paper.csv",
        ),
        "drift": (
            "drift_summary.json",
            "drift_dashboard_vs_paper_comparison.csv",
        ),
    }
    records: list[dict[str, Any]] = []
    for module, filenames in module_files.items():
        module_records: list[dict[str, Any]] = []
        for filename in filenames:
            path = active_dir / filename
            if not path.exists():
                continue
            if path.suffix.lower() == ".json":
                module_records.extend(_collect_json_rows(path, module))
            elif path.suffix.lower() == ".csv":
                if not module_records:
                    module_records.extend(_collect_csv_rows(path, module))
        seen: set[tuple[object, ...]] = set()
        for record in module_records:
            evidence = record.get("evidence", {})
            key = (
                record.get("module"),
                record.get("test_id"),
                evidence.get("condition") if isinstance(evidence, Mapping) else None,
                evidence.get("Tlog_ms") if isinstance(evidence, Mapping) else None,
                evidence.get("LPF") if isinstance(evidence, Mapping) else None,
                evidence.get("kp_star") if isinstance(evidence, Mapping) else None,
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(record)

    pass_count = sum(1 for row in records if row["status"] == "PASS")
    fail_count = sum(1 for row in records if row["status"] == "FAIL")
    by_cause: dict[str, int] = {}
    for row in records:
        if row["status"] != "FAIL":
            continue
        cause = str(row["suspected_cause"])
        by_cause[cause] = by_cause.get(cause, 0) + 1
    return {
        "summary_dir": str(active_dir),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "records": records,
        "failures_by_suspected_cause": by_cause,
    }
