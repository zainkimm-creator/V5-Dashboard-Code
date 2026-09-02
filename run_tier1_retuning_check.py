#!/usr/bin/env python3
"""Run the Tier 1 Section 4.2 check and write the report.

    .venv/bin/python run_tier1_retuning_check.py
    .venv/bin/python run_tier1_retuning_check.py --figure-package /path/to/figure_package_v5

Exits non-zero if any check fails, so it can gate a build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.validation.retuning_tier1 import (  # noqa: E402
    DEFAULT_FIGURE_PACKAGE,
    PAPER_METHODS,
    PAPER_REAL_EVALS,
    format_report,
    run_tier1,
    write_outputs,
)

OUT_DIR = Path(__file__).resolve().parent / "reports" / "validation_summary"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-package", type=Path, default=DEFAULT_FIGURE_PACKAGE,
                        help="path to figure_package_v5")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    print(f"figure package : {args.figure_package}")
    result = run_tier1(args.figure_package)

    print("\nRecomputed from 540 raw per-run costs:\n")
    print(f"  {'Method':<12} {'evals':>5} {'n':>4} {'mean':>7} {'median':>7} "
          f"{'P5':>7} {'P95':>7}")
    for name in PAPER_METHODS:
        s = result.method_stats[name]
        print(f"  {name:<12} {PAPER_REAL_EVALS[name]:>5} {int(s['n']):>4} "
              f"{s['mean']:>7.3f} {s['median']:>7.3f} {s['P5']:>7.3f} {s['P95']:>7.3f}")

    print("\nChecks:\n")
    groups: dict[str, list] = {}
    for check in result.checks:
        groups.setdefault(check.group, []).append(check)
    for group, checks in groups.items():
        failed = [c for c in checks if c.status == "FAIL"]
        unver = [c for c in checks if c.status == "UNVERIFIABLE"]
        if unver:
            mark, detail = "--", f"{len(unver)} unverifiable"
        elif failed:
            mark, detail = "!!", f"{len(failed)}/{len(checks)} FAILED"
        else:
            mark, detail = "ok", f"{len(checks)}/{len(checks)} pass"
        print(f"  [{mark}] {group:<38} {detail}")
        for c in failed:
            print(f"         FAIL {c.name}: expected {c.expected!r}, got {c.actual!r}")

    written = write_outputs(result, args.out_dir)
    report_path = args.out_dir / "tier1_retuning_report.md"
    report_path.write_text(format_report(result), encoding="utf-8")
    written["report_md"] = str(report_path)

    summary_path = args.out_dir / "tier1_retuning_check.json"
    summary_path.write_text(json.dumps({
        "section": "4.2",
        "tier": 1,
        "scope": "reporting layer only; no optimizer run",
        "figure_package": str(args.figure_package),
        "status": result.status,
        "passed": result.passed,
        "failed": result.failed,
        "unverifiable": result.unverifiable,
        "method_stats": result.method_stats,
        "checks": [c.to_row() for c in result.checks],
        "artifacts": written,
    }, indent=2), encoding="utf-8")
    written["summary_json"] = str(summary_path)

    print(f"\n{result.status}: {result.passed} passed, {result.failed} failed, "
          f"{result.unverifiable} unverifiable")
    print("\nWrote:")
    for key, path in written.items():
        print(f"  {key:<18} {path}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
