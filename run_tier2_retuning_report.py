#!/usr/bin/env python3
"""Aggregate the Tier 2 campaign checkpoints into the Section 4.2 comparison.

Reads the per-cell JSON written by ``run_tier2_retuning_campaign.py`` and
produces the tables the paper reports: per-method distribution, the paired win
rates, and the protocol comparison. Paper values are printed alongside for
comparison only.

    .venv/bin/python run_tier2_retuning_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.validation.retuning import ANCHORED_5, FAST_EXCEPTION  # noqa: E402
from backend.validation.retuning_tier1 import percentile_linear  # noqa: E402

OUT_DIR = ROOT / "reports" / "section4_tier2"

# Method order and the published field-matched medians (comparison only).
METHOD_ORDER = ["CS-BO", "WS-BO", "HGS-only", "HGS+BO5", "HGS+BO10"]
PAPER_MEDIAN = {"CS-BO": 0.407, "WS-BO": 0.407, "HGS-only": 0.357,
                "HGS+BO5": 0.357, "HGS+BO10": 0.357}
PAPER_REAL_EVALS = {"CS-BO": 30, "WS-BO": 30, "HGS-only": 0,
                    "HGS+BO5": 5, "HGS+BO10": 10}


def load_cells(cell_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for path in sorted(cell_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            errors.append(f"{path.stem}: {payload.get('error', 'unknown')}")
            continue
        rows.extend(payload.get("rows", []))
    return rows, errors


def describe(values: list[float]) -> dict[str, float]:
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if not finite:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "P5": float("nan"), "P95": float("nan")}
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "median": percentile_linear(finite, 50.0),
        "P5": percentile_linear(finite, 5.0),
        "P95": percentile_linear(finite, 95.0),
    }


def per_cell_median(rows: list[dict], method: str) -> dict[tuple[str, str], float]:
    """Median over BO seeds for each (plant, drift) - the pairing unit."""

    buckets: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["method"] == method:
            buckets.setdefault((r["pool_id"], r["drift"]), []).append(
                float(r["final_best_cost"])
            )
    return {k: percentile_linear(v, 50.0) for k, v in buckets.items() if v}


def win_rate(rows: list[dict], challenger: str, baseline: str,
             plants: tuple[str, ...] | None = None) -> tuple[float, int]:
    """Fraction of paired (plant, drift) cells where challenger costs less."""

    a = per_cell_median(rows, challenger)
    b = per_cell_median(rows, baseline)
    keys = sorted(set(a) & set(b))
    if plants is not None:
        keys = [k for k in keys if k[0] in plants]
    if not keys:
        return float("nan"), 0
    wins = sum(1 for k in keys if a[k] < b[k])
    return 100.0 * wins / len(keys), len(keys)


def build_report(rows: list[dict], errors: list[str]) -> tuple[str, dict]:
    protocols = sorted({r["protocol"] for r in rows})
    lines = ["# Section 4.2 Tier 2 - retuning campaign run locally", ""]
    if errors:
        lines += [f"**{len(errors)} cell(s) failed** - see `cells/` for detail.", ""]

    summary: dict = {"protocols": {}, "errors": errors}

    for protocol in protocols:
        sub = [r for r in rows if r["protocol"] == protocol]
        cells = len({(r["pool_id"], r["drift"]) for r in sub})
        lines += [f"## Protocol: `{protocol}`", "",
                  f"{cells} plant x drift cells, {len(sub)} method runs.", "",
                  "| Method | Real evals | n | Mean | Median | P5 | P95 | Paper median |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        stats: dict[str, dict] = {}
        for method in METHOD_ORDER:
            values = [float(r["final_best_cost"]) for r in sub if r["method"] == method]
            s = describe(values)
            stats[method] = s
            paper = PAPER_MEDIAN[method] if protocol == "field_matched" else None
            lines.append(
                f"| {method} | {PAPER_REAL_EVALS[method]} | {s['n']} | "
                f"{s['mean']:.3f} | {s['median']:.3f} | {s['P5']:.3f} | "
                f"{s['P95']:.3f} | {'-' if paper is None else f'{paper:.3f}'} |"
            )

        anchored, n_anchored = win_rate(sub, "HGS-only", "CS-BO", ANCHORED_5)
        pooled, n_pooled = win_rate(sub, "HGS-only", "CS-BO")
        fast, n_fast = win_rate(sub, "HGS-only", "CS-BO", (FAST_EXCEPTION,))
        few_shot, n_few = win_rate(sub, "HGS+BO5", "CS-BO")
        lines += ["", "### Paired win rates (HGS-only vs CS-BO(30), lower cost wins)", "",
                  "| Scope | n pairs | Win rate | Paper |", "|---|---:|---:|---:|",
                  f"| anchored-5 slow plants | {n_anchored} | {anchored:.1f}% | "
                  f"{'58%' if protocol == 'field_matched' else '-'} |",
                  f"| pooled 6 plants | {n_pooled} | {pooled:.1f}% | - |",
                  f"| {FAST_EXCEPTION} (fast exception) | {n_fast} | {fast:.1f}% | "
                  f"{'0%' if protocol == 'field_matched' else '-'} |",
                  f"| HGS+BO(5) vs CS-BO(30) | {n_few} | {few_shot:.1f}% | "
                  f"{'100%' if protocol == 'field_matched' else '-'} |",
                  ""]
        if protocol == "logging_only":
            lines += ["Paper reports a 5 % HGS-only win rate under this protocol; "
                      "the conclusion is expected to flip against `field_matched`.", ""]

        summary["protocols"][protocol] = {
            "cells": cells,
            "stats": stats,
            "win_rates": {
                "anchored5_hgs_vs_csbo": anchored,
                "pooled_hgs_vs_csbo": pooled,
                "fast_plant_hgs_vs_csbo": fast,
                "hgsbo5_vs_csbo": few_shot,
            },
        }

    if len(protocols) == 2:
        fm = summary["protocols"].get("field_matched", {}).get("win_rates", {})
        lo = summary["protocols"].get("logging_only", {}).get("win_rates", {})
        lines += ["## Protocol flip", "",
                  "The paper's central Section 4 claim is that matching the "
                  "identification protocol to the field noise is what makes the "
                  "twin usable.", "",
                  "| Protocol | HGS-only win rate, anchored-5 | Paper |",
                  "|---|---:|---:|",
                  f"| field_matched | {fm.get('anchored5_hgs_vs_csbo', float('nan')):.1f}% | 58% |",
                  f"| logging_only | {lo.get('anchored5_hgs_vs_csbo', float('nan')):.1f}% | 5% |",
                  ""]

    return "\n".join(lines) + "\n", summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    cell_dir = args.out_dir / "cells"
    if not cell_dir.exists():
        print(f"no checkpoints at {cell_dir}; run the campaign first")
        return 1

    rows, errors = load_cells(cell_dir)
    if not rows:
        print(f"no successful cells in {cell_dir} ({len(errors)} errors)")
        return 1

    runs_path = args.out_dir / "tier2_method_runs.csv"
    fields = ["protocol", "pool_id", "drift", "method", "real_evals", "seed",
              "final_best_cost", "best_kp", "best_ti_scale", "twin_mare_percent"]
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    report, summary = build_report(rows, errors)
    (args.out_dir / "tier2_retuning_report.md").write_text(report, encoding="utf-8")
    (args.out_dir / "tier2_retuning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(report)
    print(f"wrote {runs_path}")
    print(f"wrote {args.out_dir/'tier2_retuning_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
