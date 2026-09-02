"""Tier 1 Section 4 check: reproduce the published retuning statistics.

Section 4.2 of paper1_isa_v5 compares five PI-retuning strategies under the
field-matched SysID protocol. The v5 figure package ships the raw per-run costs
behind Fig. 7 and Fig. S7, so every printed statistic can be recomputed without
running the optimizer. That is what this module does.

The point is to validate the *reporting layer* before any compute is spent:
seed pooling, the percentile convention, and the method-name mapping between
the three places a number can live. Three independent sources are cross-checked
against each other:

1. ``figS07_tailrisk/data.csv``   - 540 raw per-run final costs (the ground truth)
2. ``fig07_budget/data_b_final.csv`` - the package's own median/IQR summary
3. ``data/reference_results/retuning_reference.json`` - what the dashboard will read

Source 3 is the one that matters operationally: it is comparison-only input to
the dashboard, so if it disagrees with the paper every Section 4 verdict the
dashboard ever prints is silently wrong. Nothing here feeds a calculation.

What this module deliberately does NOT claim
--------------------------------------------
The distributed ``data.csv`` carries ``method`` and ``final_best_cost`` only.
The ``plant`` and ``drift`` columns live in the original project tree
(``research/R16_dual_noise_sweep/``), so the paired win rates (58 % anchored-5,
5 % logging-only, 0 % on P189, 100 % for HGS+BO(5)), the sim-to-real gaps and
the whole logging-only campaign are **not reproducible from this package**.
They are reported as ``UNVERIFIABLE``, not as passes.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paper_reference import load_retuning_reference

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Optional: a package of per-run costs to verify published statistics against.
# Not distributed here - drop one in to enable the Tier-1 route.
DEFAULT_FIGURE_PACKAGE = PROJECT_ROOT / "data" / "reference_results" / "figure_package"

# Method keys differ between the figure package and the paper/reference JSON.
# Everything below is keyed on the paper name; this maps the package's spelling.
PACKAGE_TO_PAPER = {
    "CS-BO": "CS-BO(30)",
    "WS-BO": "WS-BO(30)",
    "HGS-only": "HGS-only",
    "HGS+BO5": "HGS+BO(5)",
    "HGS+BO10": "HGS+BO(10)",
}
PAPER_METHODS = tuple(PACKAGE_TO_PAPER.values())

# Convergence-curve column stems in fig07_budget/data_a_convergence.csv.
CONVERGENCE_COLUMNS = {
    "CS-BO(30)": "CS_BO",
    "WS-BO(30)": "WS_BO",
    "HGS+BO(5)": "HGS_BO5",
    "HGS+BO(10)": "HGS_BO10",
}

# Paper Table 2 (real-plant evaluation budget) and the pooled run counts of
# Table 3 / Table S9. The HGS family is deterministic given the twin, so it
# carries one seed (n = 60) against the stochastic methods' three (n = 180).
PAPER_REAL_EVALS = {
    "CS-BO(30)": 30,
    "WS-BO(30)": 30,
    "HGS-only": 0,
    "HGS+BO(5)": 5,
    "HGS+BO(10)": 10,
}
PAPER_N = {
    "CS-BO(30)": 180,
    "WS-BO(30)": 180,
    "HGS-only": 60,
    "HGS+BO(5)": 60,
    "HGS+BO(10)": 60,
}

# Paper Table S9, printed to three decimals.
PAPER_TABLE_S9 = {
    "CS-BO(30)": {"mean": 0.687, "median": 0.407, "P5": 0.109, "P95": 2.634},
    "WS-BO(30)": {"mean": 0.689, "median": 0.407, "P5": 0.110, "P95": 2.642},
    "HGS-only": {"mean": 0.695, "median": 0.357, "P5": 0.109, "P95": 2.688},
    "HGS+BO(5)": {"mean": 0.692, "median": 0.357, "P5": 0.109, "P95": 2.688},
    "HGS+BO(10)": {"mean": 0.690, "median": 0.357, "P5": 0.109, "P95": 2.688},
}

# Main-text §4.2 convergence claims: the warm start is ~12x worse than cold
# start at five evaluations, and the two are level by thirty.
PAPER_CONVERGENCE_CLAIMS = (
    ("WS-BO(30) median at 5 real evals", "WS-BO(30)", 5, 10.7),
    ("CS-BO(30) median at 5 real evals", "CS-BO(30)", 5, 0.878),
    ("CS-BO(30) median at 30 real evals", "CS-BO(30)", 30, 0.407),
    ("WS-BO(30) median at 30 real evals", "WS-BO(30)", 30, 0.407),
)

# figS07 README: the two dashed reference lines are the mean of the five
# per-method percentiles.
PAPER_REFERENCE_LINES = {"P5_reference_line": 0.108983, "P95_reference_line": 2.668184}

# The package's own assert gates. figS07 uses 2e-3, fig07 uses 1e-3; we adopt
# the tighter one for package-internal comparisons.
PACKAGE_TOL = 1.0e-3
# Paper values are printed to three decimals, so half a unit in the last place
# is the only defensible tolerance against them.
PRINTED_TOL = 5.0e-4


@dataclass
class Check:
    """One comparison with an explicit verdict."""

    group: str
    name: str
    expected: float | int | str | None
    actual: float | int | str | None
    tolerance: float | None
    status: str  # PASS | FAIL | UNVERIFIABLE
    note: str = ""

    @property
    def delta(self) -> float | None:
        if isinstance(self.expected, (int, float)) and isinstance(self.actual, (int, float)):
            return float(self.actual) - float(self.expected)
        return None

    def to_row(self) -> dict[str, Any]:
        delta = self.delta
        return {
            "group": self.group,
            "check": self.name,
            "expected": self.expected,
            "actual": self.actual,
            "delta": "" if delta is None else f"{delta:.6g}",
            "tolerance": "" if self.tolerance is None else f"{self.tolerance:.1e}",
            "status": self.status,
            "note": self.note,
        }


@dataclass
class Tier1Result:
    method_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    convergence: list[dict[str, Any]] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def unverifiable(self) -> int:
        return sum(1 for c in self.checks if c.status == "UNVERIFIABLE")

    @property
    def status(self) -> str:
        return "FAIL" if self.failed else "PASS"


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def percentile_linear(values: Sequence[float], q: float) -> float:
    """`numpy.percentile` default (linear interpolation between order statistics).

    Reimplemented so the check does not inherit the convention it is testing
    from the same library the figure package used. `q` is in percent.
    """

    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def describe(values: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": percentile_linear(values, 50.0),
        "P5": percentile_linear(values, 5.0),
        "P95": percentile_linear(values, 95.0),
        "min": min(values),
        "max": max(values),
    }


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"figure-package file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_per_run_costs(package_dir: Path) -> dict[str, list[float]]:
    """figS07_tailrisk/data.csv -> per-method final costs, keyed on paper names."""

    rows = _read_csv(package_dir / "figS07_tailrisk" / "data.csv")
    out: dict[str, list[float]] = {name: [] for name in PAPER_METHODS}
    for row in rows:
        paper_name = PACKAGE_TO_PAPER.get(row["method"].strip())
        if paper_name is None:
            raise ValueError(f"unmapped method in data.csv: {row['method']!r}")
        out[paper_name].append(float(row["final_best_cost"]))
    return out


def load_package_annotations(package_dir: Path) -> dict[str, dict[str, float]]:
    """figS07_tailrisk/data_annotations.csv -> the package's own summary."""

    rows = _read_csv(package_dir / "figS07_tailrisk" / "data_annotations.csv")
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        method = row["method"].strip()
        key = PACKAGE_TO_PAPER.get(method, method)  # "ALL" passes through
        out.setdefault(key, {})[row["statistic"].strip()] = float(row["value"])
    return out


def load_final_summary(package_dir: Path) -> dict[str, dict[str, float]]:
    """fig07_budget/data_b_final.csv -> the independently derived bar-panel table."""

    rows = _read_csv(package_dir / "fig07_budget" / "data_b_final.csv")
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        out[PACKAGE_TO_PAPER[row["method"].strip()]] = {
            "real_evals": float(row["real_evals"]),
            "median": float(row["median_final_best_cost"]),
            "q25": float(row["q25"]),
            "q75": float(row["q75"]),
            "n": float(row["n_runs"]),
        }
    return out


def load_convergence(package_dir: Path) -> list[dict[str, Any]]:
    """fig07_budget/data_a_convergence.csv -> median best cost vs eval count."""

    rows = _read_csv(package_dir / "fig07_budget" / "data_a_convergence.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {"eval_count": int(row["eval_count"])}
        for paper_name, stem in CONVERGENCE_COLUMNS.items():
            raw = row.get(f"{stem}_median", "")
            entry[paper_name] = float(raw) if raw not in ("", None) else None
        out.append(entry)
    return sorted(out, key=lambda r: r["eval_count"])


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def _numeric_check(
    group: str,
    name: str,
    expected: float,
    actual: float | None,
    tol: float,
    note: str = "",
) -> Check:
    if actual is None:
        return Check(group, name, expected, None, tol, "FAIL", note or "value missing")
    ok = abs(actual - expected) <= tol
    return Check(group, name, expected, actual, tol, "PASS" if ok else "FAIL", note)


def _at_most_check(
    group: str,
    name: str,
    limit: float,
    actual: float | None,
    note: str = "",
) -> Check:
    """One-sided check: the paper claims a bound, not an equality.

    "No worse than" is satisfied by being better, so an equality test would
    reject the very result the paper reports.
    """

    if actual is None:
        return Check(group, name, f"<= {limit:g}", None, None, "FAIL", "value missing")
    return Check(group, name, f"<= {limit:g}", actual, None,
                 "PASS" if actual <= limit else "FAIL", note)


def run_tier1(package_dir: Path | None = None) -> Tier1Result:
    """Recompute every reproducible Section 4.2 statistic and check it."""

    pkg = Path(package_dir) if package_dir else DEFAULT_FIGURE_PACKAGE
    result = Tier1Result()

    per_run = load_per_run_costs(pkg)
    annotations = load_package_annotations(pkg)
    final_summary = load_final_summary(pkg)
    convergence = load_convergence(pkg)
    reference = load_retuning_reference()
    # Tier 1 exists to check published statistics; with no published set to
    # check against there is nothing for it to do.
    if not reference.get("methods"):
        raise ValueError(
            "Tier-1 verification compares against the optional published "
            "result set, which is not distributed with this repository. "
            "Place the reference JSONs in data/reference_results/ to enable it."
        )
    ref_methods = {m["method"]: m for m in reference["methods"]}

    result.method_stats = {name: describe(vals) for name, vals in per_run.items()}
    result.convergence = convergence
    add = result.checks.append

    # -- A. run counts: the seed structure is the thing most easily got wrong -
    total = sum(len(v) for v in per_run.values())
    add(_numeric_check("A. run counts", "total pooled runs", 540, total, 0,
                       "(2 stochastic x 3 seeds + 3 deterministic x 1) x 60"))
    for name in PAPER_METHODS:
        add(_numeric_check("A. run counts", f"n [{name}]", PAPER_N[name],
                           len(per_run[name]), 0))

    # -- B. raw data vs the package's own summary ---------------------------
    for name in PAPER_METHODS:
        got = result.method_stats[name]
        want = annotations[name]
        for stat in ("mean", "median", "P5", "P95"):
            add(_numeric_check("B. recomputed vs package summary",
                               f"{stat} [{name}]", want[stat], got[stat], PACKAGE_TOL))

    # -- C. two independent package derivations must agree ------------------
    # data_b_final.csv is produced from the campaign dump by fig07's extractor;
    # data.csv by figS07's. They should land on the same medians.
    for name in PAPER_METHODS:
        add(_numeric_check("C. fig07 vs figS07 cross-check",
                           f"median [{name}]", final_summary[name]["median"],
                           result.method_stats[name]["median"], PACKAGE_TOL,
                           "independent extractors, same campaign"))
        add(_numeric_check("C. fig07 vs figS07 cross-check",
                           f"real_evals [{name}]", PAPER_REAL_EVALS[name],
                           final_summary[name]["real_evals"], 0))

    # -- D. recomputed values vs the paper's printed tables ------------------
    for name in PAPER_METHODS:
        got = result.method_stats[name]
        for stat, want in PAPER_TABLE_S9[name].items():
            add(_numeric_check("D. recomputed vs paper Table S9",
                               f"{stat} [{name}]", want, round(got[stat], 3),
                               PRINTED_TOL, "paper prints 3 dp"))

    # -- E. the dashboard's own reference file vs the paper ------------------
    # This is the check with operational consequences: retuning_reference.json
    # is what the dashboard will compare against.
    json_keys = {"median": "field_matched_median_S", "mean": "mean_S",
                 "P5": "p5_S", "P95": "p95_S"}
    for name in PAPER_METHODS:
        entry = ref_methods.get(name)
        if entry is None:
            add(Check("E. retuning_reference.json", f"method present [{name}]",
                      name, None, None, "FAIL", "missing from reference JSON"))
            continue
        add(_numeric_check("E. retuning_reference.json", f"real_evals [{name}]",
                           PAPER_REAL_EVALS[name], entry.get("real_evals"), 0))
        add(_numeric_check("E. retuning_reference.json", f"n [{name}]",
                           PAPER_N[name], entry.get("n"), 0))
        for stat, json_key in json_keys.items():
            add(_numeric_check("E. retuning_reference.json", f"{stat} [{name}]",
                               round(result.method_stats[name][stat], 3),
                               entry.get(json_key), PRINTED_TOL,
                               "reference JSON vs recomputed raw data"))

    # -- F. convergence-curve claims from the main text ----------------------
    by_eval = {row["eval_count"]: row for row in convergence}
    for label, method, evals, want in PAPER_CONVERGENCE_CLAIMS:
        row = by_eval.get(evals)
        actual = row.get(method) if row else None
        # The text quotes these to 3 s.f., so scale the tolerance to the value.
        tol = max(PRINTED_TOL, abs(want) * 5.0e-3)
        add(_numeric_check("F. convergence claims (paper §4.2)", label, want, actual, tol))

    # The HGS arms are flat: the twin's optimum is applied directly, so every
    # eval count carries the same median.
    for name in ("HGS+BO(5)", "HGS+BO(10)"):
        values = [row[name] for row in convergence if row.get(name) is not None]
        spread = max(values) - min(values) if values else None
        add(_numeric_check("F. convergence claims (paper §4.2)",
                           f"{name} curve is flat (max-min)", 0.0, spread, 1e-9,
                           "HGS transfer is deterministic"))

    # -- G. the headline equalities and the budget arithmetic ----------------
    hgs_medians = [result.method_stats[m]["median"] for m in
                   ("HGS-only", "HGS+BO(5)", "HGS+BO(10)")]
    add(_numeric_check("G. headline claims",
                       "HGS family shares one median (max-min)", 0.0,
                       max(hgs_medians) - min(hgs_medians), 1e-9,
                       "few-shot BO adds no median gain"))
    median_gap = (result.method_stats["HGS-only"]["median"]
                  - result.method_stats["CS-BO(30)"]["median"])
    add(_at_most_check("G. headline claims",
                       "HGS-only median - CS-BO(30) median (no worse => <= 0)",
                       0.0, median_gap,
                       "0 real evals vs 30; negative means HGS-only is ahead"))
    add(_numeric_check("G. headline claims",
                       "real-eval reduction, HGS-only vs CS-BO(30) (%)", 100.0,
                       100.0 * (30 - 0) / 30, 1e-9))
    add(Check("G. headline claims", "v4.1 '83.3% (5 vs 30)' headline is retired",
              "retired", reference.get("v41_claim_retired", "")[:60] + "...",
              None, "PASS" if reference.get("v41_claim_retired") else "FAIL",
              "v5 routes HGS-only on parity, not superiority"))

    # -- H. the two dashed reference lines of Fig. S7 ------------------------
    for stat, want in PAPER_REFERENCE_LINES.items():
        key = "P5" if stat.startswith("P5") else "P95"
        mean_of_methods = statistics.fmean(
            result.method_stats[m][key] for m in PAPER_METHODS
        )
        add(_numeric_check("H. Fig. S7 reference lines", stat, want,
                           mean_of_methods, PACKAGE_TOL,
                           "mean of the five per-method percentiles"))
        add(_numeric_check("H. Fig. S7 reference lines", f"{stat} (package)",
                           annotations["ALL"][stat], mean_of_methods, PACKAGE_TOL))

    # -- I. what this package cannot settle ----------------------------------
    for name, note in (
        ("HGS-only vs CS-BO(30) win rate, anchored-5 (58%)",
         "needs plant/drift columns; not in the distributed data.csv"),
        ("HGS-only win rate under logging-only protocol (5%)",
         "logging-only campaign dump is not shipped"),
        ("HGS-only vs CS-BO on P189 (0%, 2.53 vs 2.49)",
         "per-plant identity not in the distributed data.csv"),
        ("HGS+BO(5) beats cold start in every paired run (100%)",
         "pairing requires plant/drift columns"),
        ("sim-to-real gap, field-matched (-1.16%) / logging-only (-16.2%)",
         "twin-vs-plant cost pairs are not shipped"),
        ("P189 re-identification (293% -> 64.6% error, cost doubles)",
         "supplement S10 per-plant dump is not shipped"),
    ):
        add(Check("I. not reproducible from this package", name, None, None,
                  None, "UNVERIFIABLE", note))

    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_outputs(result: Tier1Result, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    stats_path = out_dir / "tier1_retuning_method_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "real_evals", "n", "mean", "median", "P5", "P95",
                         "min", "max"])
        for name in PAPER_METHODS:
            s = result.method_stats[name]
            writer.writerow([name, PAPER_REAL_EVALS[name], int(s["n"]),
                             f"{s['mean']:.6f}", f"{s['median']:.6f}",
                             f"{s['P5']:.6f}", f"{s['P95']:.6f}",
                             f"{s['min']:.6f}", f"{s['max']:.6f}"])
    written["method_stats_csv"] = str(stats_path)

    conv_path = out_dir / "tier1_retuning_convergence.csv"
    with conv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["eval_count", *CONVERGENCE_COLUMNS])
        for row in result.convergence:
            writer.writerow([row["eval_count"],
                             *("" if row.get(m) is None else f"{row[m]:.6f}"
                               for m in CONVERGENCE_COLUMNS)])
    written["convergence_csv"] = str(conv_path)

    checks_path = out_dir / "tier1_retuning_checks.csv"
    rows = [c.to_row() for c in result.checks]
    with checks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written["checks_csv"] = str(checks_path)

    return written


def format_report(result: Tier1Result) -> str:
    lines = [
        "# Section 4.2 Tier 1 - published retuning statistics reproduced",
        "",
        f"**Status: {result.status}** - {result.passed} passed, "
        f"{result.failed} failed, {result.unverifiable} unverifiable.",
        "",
        "Recomputed from the v5 figure package's raw per-run costs "
        "(`figS07_tailrisk/data.csv`, 540 rows). No optimizer was run; this "
        "validates the reporting layer only.",
        "",
        "## Recomputed distribution of the retuning cost S",
        "",
        "| Method | Real evals | n | Mean | Median | P5 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in PAPER_METHODS:
        s = result.method_stats[name]
        lines.append(
            f"| {name} | {PAPER_REAL_EVALS[name]} | {int(s['n'])} | "
            f"{s['mean']:.3f} | {s['median']:.3f} | {s['P5']:.3f} | {s['P95']:.3f} |"
        )

    groups: dict[str, list[Check]] = {}
    for check in result.checks:
        groups.setdefault(check.group, []).append(check)

    lines += ["", "## Checks", ""]
    for group, checks in groups.items():
        failed = [c for c in checks if c.status == "FAIL"]
        unver = [c for c in checks if c.status == "UNVERIFIABLE"]
        if unver:
            verdict = f"{len(unver)} unverifiable"
        else:
            verdict = "all pass" if not failed else f"{len(failed)} FAILED"
        lines.append(f"### {group} - {verdict}")
        lines.append("")
        if unver:
            lines += ["| Item | Why |", "|---|---|"]
            lines += [f"| {c.name} | {c.note} |" for c in unver]
        else:
            lines += ["| Check | Expected | Actual | Status |",
                      "|---|---:|---:|---|"]
            for c in (failed or checks):
                exp = c.expected if isinstance(c.expected, str) else (
                    "" if c.expected is None else f"{float(c.expected):.6g}")
                act = c.actual if isinstance(c.actual, str) else (
                    "" if c.actual is None else f"{float(c.actual):.6g}")
                lines.append(f"| {c.name} | {exp} | {act} | {c.status} |")
            if not failed and len(checks) > 1:
                lines.append(f"| _({len(checks)} checks, all within tolerance)_ | | | |")
        lines.append("")

    return "\n".join(lines) + "\n"
