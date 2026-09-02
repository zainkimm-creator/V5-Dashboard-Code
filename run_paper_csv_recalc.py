"""Re-run the logging-rate and excitation sections from the paper's CSV inputs.

Both sections are driven by `data/model_inputs/excitation_schedules.csv`
and `data/model_inputs/ten_plant_parameters.csv`. This regenerates every
figure, CSV, XLSX and JSON artefact for the two sections over all ten plants and
prints the dashboard-vs-paper comparison.
"""

from __future__ import annotations

import time

from backend.api.main import _all_plant_runs
from backend.validation.paper_inputs import paper_input_provenance
from backend.validation.studies import excitation_study, logging_rate_study


def main() -> None:
    provenance = paper_input_provenance()
    print("paper inputs:")
    for key in ("ten_plant_parameters", "excitation_schedules"):
        entry = provenance[key]
        print(f"  {entry['file']}: present={entry['present']} bytes={entry['bytes']}")
    print(f"  schedules: {', '.join(provenance['schedule_keys'])}")

    plant_runs, _ = _all_plant_runs()
    print(f"\nplants: {len(plant_runs)} -> {', '.join(pid for pid, _, _ in plant_runs)}")

    started = time.time()
    print("\n=== logging-rate study ===", flush=True)
    logging_artifact = logging_rate_study(plant_runs=plant_runs, prefer_cache=False)
    logging_metrics = logging_artifact["metrics"]
    print(f"calculation_version={logging_metrics['calculation_version']}")
    print(f"elapsed={time.time() - started:.1f}s")
    for row in logging_metrics["metrics"]:
        if row.get("aggregation") not in {"all_plant_median", "median"}:
            continue
        paper = row.get("paper_median_MARE_theta_percent")
        delta = row.get("paper_delta_median_MARE_theta_percent")
        print(
            f"  {str(row['case']):<13} Tlog={float(row['Tlog_ms']):6.1f} ms  "
            f"dash={float(row['MARE_theta_percent']):7.3f}%  "
            f"paper={paper if paper is None else f'{float(paper):6.2f}%'}  "
            f"delta={delta if delta is None else f'{float(delta):+6.2f} pp'}"
        )

    started = time.time()
    print("\n=== excitation study ===", flush=True)
    excitation_artifact = excitation_study(plant_runs=plant_runs, prefer_cache=False)
    excitation_metrics = excitation_artifact["metrics"]
    print(f"calculation_version={excitation_metrics['calculation_version']}")
    print(f"elapsed={time.time() - started:.1f}s")
    for row in excitation_metrics["comparison_rows"]:
        print(
            f"  {str(row['strategy']):<9} "
            f"NF dash={float(row['dashboard_NF_percent']):7.3f}% "
            f"paper={float(row['paper_NF_percent']):5.2f}% "
            f"delta={float(row['difference_NF_percent']):+6.2f} pp   |   "
            f"SN dash={float(row['dashboard_SN_percent']):7.3f}% "
            f"paper={float(row['paper_SN_percent']):5.2f}% "
            f"delta={float(row['difference_SN_percent']):+6.2f} pp"
        )

    print("\nartefacts:")
    for label, artifact in (("logging", logging_artifact), ("excitation", excitation_artifact)):
        for key in ("plot_path", "csv_path", "raw_csv_path", "summary_path", "report_path"):
            value = artifact.get(key)
            if value:
                print(f"  {label:<10} {key:<12} {value}")


if __name__ == "__main__":
    main()
