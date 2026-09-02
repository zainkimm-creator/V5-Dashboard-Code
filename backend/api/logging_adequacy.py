"""Focused FastAPI backend for logging adequacy validation only."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.validation.plants import DEFAULT_PLANT_ID, parameters_for_plant, plant_registry
from backend.validation.studies import DEFAULT_TLOG_MS_VALUES, logging_rate_study


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALL_PLANTS_ID = "ALL"

app = FastAPI(
    title="R2R Logging Adequacy API",
    version="0.2.0",
    description="Backend API for the logging adequacy validation section.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(PROJECT_ROOT)), name="artifacts")


class LoggingAdequacyRequest(BaseModel):
    plant_id: str | None = DEFAULT_PLANT_ID
    tlog_ms_values: list[float] | None = None
    tmin_ms: float = Field(default=50.0, gt=0)


def _artifact_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return f"/artifacts/{rel.as_posix()}"


def _attach_urls(payload: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "csv_path",
        "plot_path",
        "summary_path",
        "markdown_path",
        "power_law_plot_path",
        "graph_points_csv_path",
        "graph_points_xlsx_path",
    ):
        if key in payload:
            payload[key.replace("_path", "_url")] = _artifact_url(payload.get(key))
    return payload


def _plant_params_or_400(plant_id: str | None) -> tuple[object, dict[str, Any]]:
    try:
        return parameters_for_plant(plant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _is_all_plants(plant_id: str | None) -> bool:
    return str(plant_id or "").strip().upper() in {ALL_PLANTS_ID, "__ALL__", "*"}


def _all_plant_runs() -> tuple[list[tuple[str, object, dict[str, Any]]], dict[str, Any]]:
    runs: list[tuple[str, object, dict[str, Any]]] = []
    plants = plant_registry()
    for plant in plants:
        params, plant_payload = parameters_for_plant(str(plant["plant_id"]))
        runs.append((str(plant["plant_id"]), params, plant_payload))
    return runs, {
        "plant_id": ALL_PLANTS_ID,
        "label": "All 10 plants | median result",
        "plant_count": len(plants),
        "plant_ids": [str(plant["plant_id"]) for plant in plants],
        "aggregation": "median",
    }


def _assert_finite_numbers(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise HTTPException(status_code=422, detail=f"{label} produced a non-finite value.")
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_numbers(item, label)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _assert_finite_numbers(item, label)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    all_plants = {
        "plant_id": ALL_PLANTS_ID,
        "label": "All 10 plants | median result",
        "plant_count": len(plant_registry()),
        "aggregation": "median",
    }
    return {
        "default_plant_id": ALL_PLANTS_ID,
        "single_plant_default_id": DEFAULT_PLANT_ID,
        "plants": [all_plants, *plant_registry()],
        "tlog_ms_options": DEFAULT_TLOG_MS_VALUES,
        "default_tmin_ms": 50.0,
        "routes": ["POST /validate/logging-adequacy"],
    }


@app.post("/validate/logging-adequacy")
def logging_adequacy_route(request: LoggingAdequacyRequest) -> dict[str, object]:
    if _is_all_plants(request.plant_id):
        plant_runs, plant = _all_plant_runs()
        study_args = {"plant_runs": plant_runs}
    else:
        params, plant = _plant_params_or_400(request.plant_id)
        study_args = {"params": params}
    try:
        payload = logging_rate_study(request.tlog_ms_values, tmin_ms=request.tmin_ms, **study_args)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Logging adequacy became numerically invalid: {exc}") from exc

    _assert_finite_numbers(payload, "Logging adequacy")
    payload["plant"] = plant
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("power_law_plot_path"):
        payload["power_law_plot_path"] = metrics["power_law_plot_path"]
    return _attach_urls(payload)
