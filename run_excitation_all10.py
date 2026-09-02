from backend.api.main import _all_plant_runs
from backend.validation.studies import excitation_study


plant_runs, _ = _all_plant_runs()
result = excitation_study(plant_runs=plant_runs, prefer_cache=False)
metrics = result["metrics"]

print(f"calculation_version={metrics['calculation_version']}")
for row in metrics["comparison_rows"]:
    print(
        row["strategy"],
        f"NF={row['dashboard_NF_percent']:.6f}",
        f"NF_valid={int(row['valid_plant_count_NF'])}/10",
        f"SN={row['dashboard_SN_percent']:.6f}",
        f"SN_valid={int(row['valid_plant_count_SN'])}/10",
    )
print(f"summary_path={result['summary_path']}")
print(f"csv_path={result['csv_path']}")
print(f"raw_csv_path={result['raw_csv_path']}")
