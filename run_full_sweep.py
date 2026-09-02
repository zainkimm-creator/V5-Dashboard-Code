"""Full-sweep raw-data export in the 27-column template format.

Grid is campaign 1 (group A full factorial), exactly as recorded in
`backend/validation/noise_aware_logging_lpf.py`:

    10 plants x 6 excitations x 7 Tlog x (NF + 5 noise levels x 6 LPF cutoffs)
    = 13,020 rows, single seed.

This is a read-only consumer of the existing simulate + weighted-PEM path. It
does not change the estimator or the section pipeline; it only stops discarding
the per-parameter detail that `_run_live_sysid_series` averages away.

Column sources (see FULL_SWEEP_NOTES.md for the audit):
  eRMSE            mean(|relative_error|) over the 7 parameters == MARE_theta.
                   Verified against the template: mean matches 11/11 sample
                   rows to double precision, RMS matches 0/11. The header name
                   says RMS, the values are a mean.
  OS_max           100 * max_overshoot_N / excitation amplitude. Verified exact
                   on ET1 / ET3 / E_Toggle against the template.
  zeta_CL_min      ten_plant_parameters.csv `zeta_CL_min` (full precision, not
                   the paper-printed 3-figure value).
  zeta_OL_min      OPEN GAP - left empty. Not traceable to any source; see
                   FULL_SWEEP_NOTES.md.
  n_iterations     scipy least_squares `njev`.
  FIM_*            eigenvalues of F = J^T W^2 J, exposed additively in
                   estimator.py alongside the existing log10_kappa_fisher.
  regime           ten_plant_parameters.csv `regime_pool` (e.g. "UD"), not
                   `regime_paper` ("O-UD").
  plant_id         the pool id (P001...), not the registry id (P01...).

ET3M runs three operating points (0.5x, 1.0x, 2.0x line speed). Every numeric
column for ET3M is the mean over those three fits, matching how the existing
pipeline pools MARE_theta.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.simulation import SimulationConfig, simulate  # noqa: E402
from backend.sysid.estimator import estimate_parameters_weighted_pem  # noqa: E402
from backend.validation.excitations import get_excitation_profile  # noqa: E402
from backend.validation.noise_aware_logging_lpf import (  # noqa: E402
    CONTROLLER_SAMPLE_TIME_S,
    DT_S,
    FACTORIAL_CAMPAIGN_GROUP,
    FACTORIAL_LPF_CUTOFFS,
    FIG06_NOISE_LEVELS_PERCENT,
    TLOG_VALUES_MS,
    _downsample_rows,
    _paper_controller_config,
)
from backend.validation.plants import parameters_for_plant, plant_registry  # noqa: E402

HEADER = [
    "plant_id", "T_log", "noise_type", "noise_level", "seed", "excitation",
    "lpf_cutoff", "eRMSE", "converged", "n_iterations", "final_cost", "OS_max",
    "zeta_CL_min", "zeta_OL_min", "regime", "err_kt_uw", "err_kt_nip",
    "err_kt_rw", "err_kf_uw", "err_kf_nip", "err_kf_rw", "err_EA", "FIM_cond",
    "FIM_logdet", "FIM_lambda_min", "material", "scale",
]

# error_table order is PARAMETER_NAMES = (kt_UW, kt_Nip, kt_RW, kf_UW, kf_Nip,
# kf_RW, EA), which maps 1:1 onto the template's err_* columns.
ERR_COLUMNS = [
    "err_kt_uw", "err_kt_nip", "err_kt_rw",
    "err_kf_uw", "err_kf_nip", "err_kf_rw", "err_EA",
]

EXCITATIONS = ("ET1", "ET3", "ET6", "ET3M", "EV1", "E_Toggle")
SEED = 0

# `plant_registry()` exposes the paper-printed zeta (0.151, 3 figures) and
# `regime_paper` ("O-UD"). The template wants the full-precision zeta
# (0.151079) and `regime_pool` ("UD"), which only the CSV carries.
PLANT_CSV = PROJECT_ROOT / "data" / "model_inputs" / "ten_plant_parameters.csv"


def _plant_csv():
    with PLANT_CSV.open(encoding="utf-8-sig") as handle:
        return {row["plant_id"]: row for row in csv.DictReader(handle)}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_cell(plant_pool_id, excitation, noise_level_percent, lpf_hz):
    """One (plant, excitation, noise, LPF) cell -> one row per Tlog value."""
    plant = next(p for p in plant_registry() if p["pool_id"] == plant_pool_id)
    csv_row = _plant_csv()[plant_pool_id]
    params, meta = parameters_for_plant(plant["plant_id"])
    amplitude = float(meta["recommended_excitation_amplitude_V"])
    nominal_speed = float(meta.get("v_ref_m_s", params.feeder_velocity_m_s))
    noisy = float(noise_level_percent) > 0.0
    noise_sigma = float(noise_level_percent) / 100.0 * float(meta["T_max_N"]) if noisy else 0.0

    # ET3M is the multi-operating-point variant; it reuses the ET3 schedule.
    speed_multipliers = (0.5, 1.0, 2.0) if excitation == "ET3M" else (1.0,)
    schedule_name = "ET3" if excitation == "ET3M" else excitation

    per_tlog = {int(t): [] for t in TLOG_VALUES_MS}
    overshoots = []

    for multiplier in speed_multipliers:
        line_speed = nominal_speed * float(multiplier)
        profile = get_excitation_profile(
            schedule_name, amplitude, campaign_group=FACTORIAL_CAMPAIGN_GROUP
        )
        try:
            sim = simulate(
                params,
                controller_config=_paper_controller_config(params, line_speed),
                config=SimulationConfig(
                    duration_s=float(getattr(profile, "duration_s", 12.0)),
                    dt_s=DT_S,
                    controller_sample_time_s=CONTROLLER_SAMPLE_TIME_S,
                    log_sample_time_s=DT_S,
                    line_speed_m_s=line_speed,
                    sensor_noise_tension_N=noise_sigma,
                    sensor_noise_omega_rad_s=0.0,
                    sensor_lpf_hz=lpf_hz if noisy else None,
                    noise_affects_controller=True,
                    noise_rng="numpy_default_rng",
                    seed=SEED,
                ),
                excitation=profile,
                write_output=False,
            )
        except Exception:
            continue

        # OS_max: peak tension excursion above setpoint, as a percent of the
        # excitation amplitude. Verified exact on ET1 / ET3 / E_Toggle.
        overshoots.append(100.0 * float(sim.metrics["max_overshoot_N"]) / amplitude)

        for tlog_ms in TLOG_VALUES_MS:
            try:
                result = estimate_parameters_weighted_pem(
                    _downsample_rows(sim.rows, int(tlog_ms)),
                    nominal_params=params,
                    true_params=params,
                    max_nfev=150,
                    break_on_line_speed_change=excitation == "EV1",
                )
                per_tlog[int(tlog_ms)].append(result)
            except Exception:
                pass

    os_max = _mean(overshoots)
    rows = []
    for tlog_ms in TLOG_VALUES_MS:
        results = per_tlog[int(tlog_ms)]
        row = {
            "plant_id": plant_pool_id,
            "T_log": float(tlog_ms) / 1000.0,
            "noise_type": "SN" if noisy else "NF",
            "noise_level": float(noise_level_percent),
            "seed": SEED,
            "excitation": excitation,
            "lpf_cutoff": -1 if lpf_hz is None else int(lpf_hz),
            "OS_max": os_max,
            "zeta_CL_min": float(csv_row["zeta_CL_min"]),
            "zeta_OL_min": "",  # open gap - see module docstring
            "regime": str(csv_row["regime_pool"]),
            "material": str(csv_row["material"]),
            "scale": str(csv_row["scale"]),
        }
        if not results or len(results) != len(speed_multipliers):
            row["converged"] = 0
            for key in ("eRMSE", "n_iterations", "final_cost", "FIM_cond",
                        "FIM_logdet", "FIM_lambda_min", *ERR_COLUMNS):
                row[key] = ""
            rows.append(row)
            continue

        diags = [r.diagnostics for r in results]
        row["eRMSE"] = _mean([float(r.mare_theta) for r in results])
        row["converged"] = 1 if all(d.get("success") for d in diags) else 0
        row["n_iterations"] = _mean([d.get("njev") for d in diags])
        row["final_cost"] = _mean([d.get("cost") for d in diags])
        row["FIM_cond"] = _mean([d.get("fisher_cond") for d in diags])
        row["FIM_logdet"] = _mean([d.get("fisher_logdet") for d in diags])
        row["FIM_lambda_min"] = _mean([d.get("fisher_lambda_min") for d in diags])
        for idx, column in enumerate(ERR_COLUMNS):
            row[column] = _mean([float(r.error_table[idx]["relative_error"]) for r in results])
        rows.append(row)
    return rows


def _worker(job):
    return run_cell(*job)


def build_jobs(plant_ids):
    jobs = []
    for pool_id in plant_ids:
        for excitation in EXCITATIONS:
            jobs.append((pool_id, excitation, 0.0, None))  # the NF leg
            for level in FIG06_NOISE_LEVELS_PERCENT:
                for cutoff in FACTORIAL_LPF_CUTOFFS:
                    jobs.append((pool_id, excitation, float(level), cutoff))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plants", default="all",
                        help="comma-separated pool ids (e.g. P001), or 'all'")
    parser.add_argument("--out", default="raw_data_full_sweep.csv")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()

    registry = plant_registry()
    all_ids = [str(p["pool_id"]) for p in registry]
    plant_ids = all_ids if args.plants == "all" else [
        p.strip() for p in args.plants.split(",") if p.strip()
    ]
    unknown = [p for p in plant_ids if p not in all_ids]
    if unknown:
        raise SystemExit(f"unknown plant pool ids: {unknown}; known: {all_ids}")

    jobs = build_jobs(plant_ids)
    expected = len(jobs) * len(TLOG_VALUES_MS)
    print(f"plants={len(plant_ids)} cells={len(jobs)} expected_rows={expected} "
          f"workers={args.workers}", flush=True)

    started = time.time()
    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, job) for job in jobs]
        for future in as_completed(futures):
            rows.extend(future.result())
            done += 1
            if done % 100 == 0 or done == len(jobs):
                rate = done / max(1e-9, time.time() - started)
                remaining = (len(jobs) - done) / rate if rate else 0.0
                print(f"  {done}/{len(jobs)} cells  {len(rows)} rows  "
                      f"eta {remaining:6.1f}s", flush=True)

    # Deterministic order regardless of completion order.
    order = {name: i for i, name in enumerate(EXCITATIONS)}
    rows.sort(key=lambda r: (r["plant_id"], r["excitation"] and order[r["excitation"]],
                             r["noise_level"], r["lpf_cutoff"], r["T_log"]))

    out_path = PROJECT_ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - started
    failed = sum(1 for r in rows if r["converged"] == 0)
    print(f"\nwrote {len(rows)} rows -> {out_path}")
    print(f"elapsed {elapsed:.1f}s   non-converged rows: {failed} "
          f"({100.0 * failed / max(1, len(rows)):.2f}%)")


if __name__ == "__main__":
    main()
