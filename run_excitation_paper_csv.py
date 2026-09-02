"""Re-run the excitation section over all ten plants from the paper CSV inputs."""

from __future__ import annotations

import time

from backend.api.main import _all_plant_runs
from backend.validation.paper_inputs import paper_input_provenance
from backend.validation.studies import excitation_study


def main() -> None:
    provenance = paper_input_provenance()
    print(f"schedules: {', '.join(provenance['schedule_keys'])}", flush=True)
    plant_runs, _ = _all_plant_runs()
    print(f"plants: {len(plant_runs)}", flush=True)

    started = time.time()
    artifact = excitation_study(plant_runs=plant_runs, prefer_cache=False)
    payload = artifact["metrics"]
    print(f"calculation_version={payload['calculation_version']}", flush=True)
    print(f"elapsed={time.time() - started:.1f}s", flush=True)

    for row in payload["comparison_rows"]:
        print(
            f"  {str(row['strategy']):<9} "
            f"NF dash={float(row['dashboard_NF_percent']):7.3f}% "
            f"paper={float(row['paper_NF_percent']):5.2f}% "
            f"delta={float(row['difference_NF_percent']):+6.2f} pp   |   "
            f"SN dash={float(row['dashboard_SN_percent']):7.3f}% "
            f"paper={float(row['paper_SN_percent']):5.2f}% "
            f"delta={float(row['difference_SN_percent']):+6.2f} pp",
            flush=True,
        )

    print("\nrecord provenance:", flush=True)
    seen: set[tuple[str, str]] = set()
    for row in payload["raw_rows"]:
        key = (str(row["strategy"]), str(row["condition"]))
        if key in seen:
            continue
        seen.add(key)
        print(
            f"  {key[0]:<9} {key[1]:<3} group={row['resolved_campaign_group']:<22} "
            f"records={int(row['record_count'])} durations={row['record_duration_s']:<12} "
            f"seeds={row['record_seeds']:<10} v_seed_offset={row['velocity_seed_offset']}",
            flush=True,
        )

    for key in ("plot_path", "csv_path", "raw_csv_path", "summary_path"):
        if artifact.get(key):
            print(f"{key}: {artifact[key]}", flush=True)


if __name__ == "__main__":
    main()
