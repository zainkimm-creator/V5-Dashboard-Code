"""FastAPI routes for simulation, SysID, and validation workflows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.models.equations import R2RParameters
from backend.models.simulation import DEFAULT_DATA_DIR, SimulationConfig, simulate
from backend.sysid.estimator import estimate_parameters, load_rows_from_csv
from backend.validation.calculations import (
    drift_calculation_payload,
    excitation_calculation_payload,
    logging_rate_calculation_payload,
    simulation_calculation_payload,
    sysid_calculation_payload,
)
from backend.validation.excitations import excitation_names, get_excitation_profile
from backend.validation.plants import DEFAULT_PLANT_ID, parameters_for_plant, plant_registry
from backend.validation.noise_aware_logging_lpf import noise_aware_logging_lpf_validation
from backend.validation.closed_loop_damping import closed_loop_damping_study
from backend.validation.studies import (
    drift_study,
    excitation_study,
    logging_rate_study,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
ALL_PLANTS_ID = "ALL"

app = FastAPI(
    title="R2R System-Identification Dashboard API",
    version="0.1.0",
    description="Backend API for R2R simulation, SysID, and validation studies.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(PROJECT_ROOT)), name="artifacts")


class SimulationRequest(BaseModel):
    plant_id: str | None = DEFAULT_PLANT_ID
    duration_s: float = Field(default=4.0, gt=0)
    dt_ms: float = Field(default=1.0, gt=0)
    # T_s = dt = 1 ms in all campaigns; the multirate axis is T_log alone.
    controller_sample_time_ms: float = Field(default=1.0, gt=0)
    log_sample_time_ms: float = Field(default=10.0, gt=0)
    line_speed_m_s: float | None = Field(default=None, gt=0)
    excitation: str = "ET3"
    excitation_amplitude_V: float = 0.08
    sensor_noise_tension_N: float = Field(default=0.0, ge=0)
    sensor_noise_omega_rad_s: float = Field(default=0.0, ge=0)
    seed: int = 7
    output_name: str = "api_simulation.csv"


class SysIDRequest(BaseModel):
    plant_id: str | None = DEFAULT_PLANT_ID
    csv_path: str | None = None
    duration_s: float = Field(default=4.0, gt=0)
    log_sample_time_ms: float = Field(default=10.0, gt=0)
    excitation: str = "E_Toggle"
    excitation_amplitude_V: float = 0.08
    sensor_noise_tension_N: float = Field(default=0.0, ge=0)
    sensor_noise_omega_rad_s: float = Field(default=0.0, ge=0)


class LoggingRateRequest(BaseModel):
    plant_id: str | None = DEFAULT_PLANT_ID
    tlog_ms_values: list[float] | None = None
    tmin_ms: float = Field(default=50.0, gt=0)


class EmptyRequest(BaseModel):
    plant_id: str | None = DEFAULT_PLANT_ID
    force_rerun: bool = False


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
        "xlsx_path",
        "plot_path",
        "summary_path",
        "markdown_path",
        "power_law_plot_path",
        "speed_plot_path",
        "speed_csv_path",
        "graph_points_csv_path",
        "graph_points_xlsx_path",
        "raw_csv_path",
        "drift_EA_plot_path",
        "drift_f_plot_path",
        "drift_J_plot_path",
        "drift_combined_plot_path",
        "per_roller_csv_path",
    ):
        if key in payload:
            payload[key.replace("_path", "_url")] = _artifact_url(payload.get(key))
    return payload


def _attach_nested_plot_urls(payload: dict[str, Any]) -> dict[str, Any]:
    plots = payload.get("plots")
    if isinstance(plots, dict):
        for item in plots.values():
            if isinstance(item, dict) and "path" in item:
                item["url"] = _artifact_url(item.get("path"))
    return payload


def _preview_rows(rows: Sequence[dict[str, float]], limit: int = 8) -> list[dict[str, float]]:
    if len(rows) <= limit:
        return list(rows)
    half = max(1, limit // 2)
    return list(rows[:half]) + list(rows[-half:])


def _plant_params_or_400(plant_id: str | None) -> tuple[R2RParameters, dict[str, Any]]:
    try:
        return parameters_for_plant(plant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _is_all_plants(plant_id: str | None) -> bool:
    return str(plant_id or "").strip().upper() in {ALL_PLANTS_ID, "__ALL__", "*"}


def _all_plant_runs() -> tuple[list[tuple[str, R2RParameters, dict[str, Any]]], dict[str, Any]]:
    runs: list[tuple[str, R2RParameters, dict[str, Any]]] = []
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


def _run_or_422(label: str, fn: Any) -> Any:
    try:
        return fn()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"{label} became numerically invalid: {exc}") from exc


def _assert_finite_numbers(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise HTTPException(
                status_code=422,
                detail=f"{label} produced a non-finite value. Reduce excitation amplitude or select a baseline-range plant.",
            )
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite_numbers(item, label)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _assert_finite_numbers(item, label)


def _safe_upload_path(filename: str | None) -> Path:
    safe_name = Path(filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    return UPLOAD_DIR / safe_name


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def dashboard_redirect() -> RedirectResponse:
    return RedirectResponse("http://127.0.0.1:5199/")


@app.get("/plants")
def plants_route() -> dict[str, object]:
    return {
        "default_plant_id": DEFAULT_PLANT_ID,
        "plants": plant_registry(),
        "note": "Plant presets use the professor-resolved P01-P10 values: EA, v_ref, T_ref, T_max, R, J, f, and L. The supplement b_i typo is treated as viscous friction f_i.",
    }


@app.post("/simulate")
def simulate_route(request: SimulationRequest) -> dict[str, object]:
    params, plant = _plant_params_or_400(request.plant_id)
    config = SimulationConfig(
        duration_s=request.duration_s,
        dt_s=request.dt_ms / 1000.0,
        controller_sample_time_s=request.controller_sample_time_ms / 1000.0,
        log_sample_time_s=request.log_sample_time_ms / 1000.0,
        line_speed_m_s=request.line_speed_m_s or float(plant["v_ref_m_s"]),
        sensor_noise_tension_N=request.sensor_noise_tension_N,
        sensor_noise_omega_rad_s=request.sensor_noise_omega_rad_s,
        seed=request.seed,
        output_name=request.output_name,
    )
    result = _run_or_422(
        "Simulation",
        lambda: simulate(
            params,
            config=config,
            excitation=get_excitation_profile(request.excitation, request.excitation_amplitude_V),
            output_dir=DEFAULT_DATA_DIR,
        ),
    )
    _assert_finite_numbers(result.metrics, "Simulation metrics")
    payload: dict[str, Any] = {
        "metrics": result.metrics,
        "csv_path": result.csv_path,
        "xlsx_path": result.xlsx_path,
        "plot_path": None,
        "plant": plant,
        "equation_sample": result.equation_sample,
        "preview_rows": _preview_rows(result.rows),
    }
    payload.update(simulation_calculation_payload(result.metrics, result.rows, config, params))
    return _attach_urls(payload)


@app.post("/sysid")
def sysid_route(request: SysIDRequest) -> dict[str, object]:
    params, plant = _plant_params_or_400(request.plant_id)
    csv_path = request.csv_path
    if csv_path:
        rows = load_rows_from_csv(csv_path)
    else:
        sim = _run_or_422(
            "SysID source simulation",
            lambda: simulate(
                params,
                config=SimulationConfig(
                    duration_s=request.duration_s,
                    log_sample_time_s=request.log_sample_time_ms / 1000.0,
                    line_speed_m_s=float(plant["v_ref_m_s"]),
                    sensor_noise_tension_N=request.sensor_noise_tension_N,
                    sensor_noise_omega_rad_s=request.sensor_noise_omega_rad_s,
                    output_name="api_sysid_source.csv",
                ),
                excitation=get_excitation_profile(request.excitation, request.excitation_amplitude_V),
                output_dir=DEFAULT_DATA_DIR,
            ),
        )
        _assert_finite_numbers(sim.metrics, "SysID source simulation metrics")
        rows = sim.rows
        csv_path = sim.csv_path
    result = _run_or_422(
        "SysID estimation",
        lambda: estimate_parameters(rows, params, params, summary_name="api_sysid_result.json"),
    )
    _assert_finite_numbers(result.to_dict(), "SysID result")
    metrics = {
        "MARE_theta": result.mare_theta,
        "metric_name": "MARE_theta",
        "samples": len(rows),
    }
    payload: dict[str, Any] = {
        "metrics": metrics,
        "estimates": result.estimates,
        "error_table": result.error_table,
        "plant": plant,
        "csv_path": csv_path,
        "plot_path": None,
        "summary_path": result.summary_path,
    }
    payload.update(sysid_calculation_payload(metrics, result.error_table))
    return _attach_urls(payload)


@app.post("/validate/logging-rate")
def logging_rate_route(request: LoggingRateRequest) -> dict[str, object]:
    if _is_all_plants(request.plant_id):
        plant_runs, plant = _all_plant_runs()
        study_args = {"plant_runs": plant_runs, "prefer_cache": True}
    else:
        params, plant = _plant_params_or_400(request.plant_id)
        study_args = {"params": params}
    payload = _run_or_422(
        "Logging-rate study",
        lambda: logging_rate_study(request.tlog_ms_values, tmin_ms=request.tmin_ms, **study_args),
    )
    _assert_finite_numbers(payload, "Logging-rate study")
    payload["plant"] = plant
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict) and metrics.get("power_law_plot_path"):
        payload["power_law_plot_path"] = metrics["power_law_plot_path"]
    payload.update(logging_rate_calculation_payload(payload))
    return _attach_urls(payload)


@app.post("/validate/logging-adequacy")
def logging_adequacy_route(request: LoggingRateRequest) -> dict[str, object]:
    return logging_rate_route(request)


@app.post("/validate/excitation")
def excitation_route(_: EmptyRequest | None = None) -> dict[str, object]:
    request = _ or EmptyRequest()
    if _is_all_plants(request.plant_id):
        plant_runs, plant = _all_plant_runs()
        payload = _run_or_422(
            "Excitation study",
            lambda: excitation_study(plant_runs=plant_runs, prefer_cache=not request.force_rerun),
        )
    else:
        params, plant = _plant_params_or_400(request.plant_id)
        payload = _run_or_422(
            "Excitation study",
            lambda: excitation_study(
                plant_runs=[(str(request.plant_id), params, plant)],
                prefer_cache=not request.force_rerun,
            ),
        )
    _assert_finite_numbers(payload, "Excitation study")
    payload["plant"] = plant
    payload.update(excitation_calculation_payload(payload))
    return _attach_urls(payload)


@app.post("/validate/drift")
def drift_route(_: EmptyRequest | None = None) -> dict[str, object]:
    request = _ or EmptyRequest()
    if _is_all_plants(request.plant_id):
        plant_runs, plant = _all_plant_runs()
        payload = _run_or_422(
            "Drift study",
            lambda: drift_study(plant_runs=plant_runs, prefer_cache=not request.force_rerun),
        )
    else:
        params, plant = _plant_params_or_400(request.plant_id)
        payload = _run_or_422(
            "Drift study",
            lambda: drift_study(
                plant_runs=[(str(request.plant_id), params, plant)],
                prefer_cache=not request.force_rerun,
            ),
        )
    _assert_finite_numbers(payload, "Drift study")
    payload["plant"] = plant
    payload.update(drift_calculation_payload(payload))
    return _attach_urls(payload)


@app.post("/validate/noise-aware-logging-lpf")
def noiseLpf_route(_: EmptyRequest | None = None) -> dict[str, object]:
    request = _ or EmptyRequest()
    _, plant = _plant_params_or_400(request.plant_id)
    payload = _run_or_422("Noise-aware logging validation", noise_aware_logging_lpf_validation)
    _assert_finite_numbers(payload, "Noise-aware logging validation")
    payload["plant"] = plant
    _attach_nested_plot_urls(payload)
    return _attach_urls(payload)


@app.post("/validate/closed-loop-damping")
def damping_route() -> dict[str, object]:
    payload = _run_or_422("Closed-loop damping study", closed_loop_damping_study)
    _assert_finite_numbers(payload, "Closed-loop damping study")
    return _attach_urls(payload)


@app.post("/validate/retuning")
def retuning_route() -> dict[str, object]:
    """Section 4 adaptive retuning - precomputed campaign results.

    Unlike the live sections this reads the offline campaign's checkpoints
    (~3.7 h of GPU optimisation cannot run inside a request); provenance and
    caveats travel in the payload so the frontend can label it honestly.
    """

    from backend.validation.retuning_results import retuning_results_payload

    payload = _run_or_422("Retuning results", retuning_results_payload)
    _assert_finite_numbers(payload, "Retuning results")
    return payload


@app.post("/validate/retuning-tier1")
def retuning_tier1_route() -> dict[str, object]:
    """Live Tier-1 verification: recompute every published Section 4.2
    statistic from the figure package's raw per-run costs (<1 s, no optimiser)
    and check the dashboard's own reference file against the paper."""

    from backend.validation.paper_reference import comparison_data_available
    from backend.validation.retuning_tier1 import (
        DEFAULT_FIGURE_PACKAGE,
        run_tier1,
    )

    if not comparison_data_available():
        raise HTTPException(
            status_code=422,
            detail=(
                "Tier-1 verification compares this dashboard against a "
                "published result set that is not distributed with this "
                "repository. Add the reference JSONs to "
                "data/reference_results/ to enable it."
            ),
        )
    if not DEFAULT_FIGURE_PACKAGE.exists():
        raise HTTPException(
            status_code=422,
            detail=(
                "v5 figure package not found beside the dashboard "
                f"(expected {DEFAULT_FIGURE_PACKAGE}); Tier-1 verification "
                "needs its raw per-run costs."
            ),
        )
    result = _run_or_422("Tier-1 retuning verification", run_tier1)
    return {
        "status": result.status,
        "passed": result.passed,
        "failed": result.failed,
        "unverifiable": result.unverifiable,
        "method_stats": result.method_stats,
        "checks": [check.to_row() for check in result.checks],
    }


@app.get("/metadata")
def metadata() -> dict[str, object]:
    all_plants = {
        "plant_id": ALL_PLANTS_ID,
        "label": "All 10 plants | median result",
        "plant_count": len(plant_registry()),
        "aggregation": "median",
    }
    return {
        "excitation_profiles": list(excitation_names()),
        "default_plant_id": ALL_PLANTS_ID,
        "single_plant_default_id": DEFAULT_PLANT_ID,
        "plants": [all_plants, *plant_registry()],
        "tlog_ms_options": [1, 2, 5, 10, 20, 50, 100],
        "default_tmin_ms": 50.0,
        "routes": [
            "POST /simulate",
            "POST /sysid",
            "GET /plants",
            "POST /validate/logging-rate",
            "POST /validate/logging-adequacy",
            "POST /validate/excitation",
            "POST /validate/noise-aware-logging-lpf",
            "POST /validate/closed-loop-damping",
            "POST /validate/drift",
            "POST /upload",
        ],
    }


@app.post("/upload")
async def upload_data_file(file: UploadFile = File(...)) -> dict[str, object]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = _safe_upload_path(file.filename)
    contents = await file.read()
    destination.write_bytes(contents)
    rel_path = destination.relative_to(PROJECT_ROOT)
    return {
        "filename": destination.name,
        "bytes": len(contents),
        "path": str(destination),
        "url": f"/artifacts/{rel_path.as_posix()}",
    }

