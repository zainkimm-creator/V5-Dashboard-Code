"""Section 4 (adaptive retuning) results for the dashboard.

Unlike the five live sections, the retuning campaign cannot be recomputed inside
an HTTP request: one campaign is 120 cells x ~3,000 evaluations of 16 s of
closed-loop physics (~3.7 h on the GPU). This module therefore follows the
pattern the project's own Section-4 guide prescribes: the heavy optimiser runs
offline (``run_tier2_retuning_campaign.py``) and the dashboard *reads the saved
outputs* - the per-cell checkpoints, the per-cell achievable bounds, and the
paper reference JSON (comparison-only, as everywhere else).

Honesty contract carried in the payload itself:

* ``provenance`` names the evaluation model (the papers never state the test
  signal; ours is the reconstruction that survived a 23-model falsification
  battery), the cost weights (published w_os = 2), the run date and backend.
* ``caveats`` lists what is reconstructed vs. published, and the two open
  author questions the comparison still hinges on.
* Diagnostic campaigns (retracted signal readings, the w_os = 0 run) are
  reported under ``diagnostic_runs`` so they cannot be mistaken for the
  headline result.
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from .paper_reference import REFERENCE_DIR, load_retuning_reference
from .retuning import RETUNING_PLANTS
from .retuning_tier1 import percentile_linear as pct

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"

# The campaign shown as the headline result.
CANONICAL_CAMPAIGN = "section4_tier2_stepseq"

METHOD_ORDER = ["CS-BO", "WS-BO", "HGS-only", "HGS+BO5", "HGS+BO10"]
METHOD_LABEL = {"CS-BO": "CS-BO(30)", "WS-BO": "WS-BO(30)", "HGS-only": "HGS-only",
                "HGS+BO5": "HGS+BO(5)", "HGS+BO10": "HGS+BO(10)"}
REAL_EVALS = {"CS-BO": 30, "WS-BO": 30, "HGS-only": 0, "HGS+BO5": 5, "HGS+BO10": 10}
POOL_TO_DASH = dict(RETUNING_PLANTS)

# The BO-configuration study (``run_bo_config_study.py``): 540 extra CS/WS-BO
# runs isolating which unprinted skopt setting moves the Section 4.2 ordering.
BO_CONFIG_STUDY = REPORTS_DIR / "bo_config_study" / "results.json"

# What each studied variant changes, and the hypothesis it was built to test.
# Mirrors the VARIANTS table of run_bo_config_study.py.
BO_VARIANTS = [
    ("V-wide", "CS-BO",
     "Five-fold wider log box",
     "Kp* 0.5-1000, TI/auto 0.01-1000, log-uniform",
     "Does a wider search space slow late convergence into the paper's shape?"),
    ("V-linabs", "CS-BO",
     "Linear-uniform priors, absolute gains",
     "Kp* 1-300 linear, T_I 0.1-100 s linear",
     "The unsophisticated default: wastes evaluations in the flat high region."),
    ("V-cluster", "WS-BO",
     "Warm start seeded with a cluster",
     "SysID-mode point plus four +-30% perturbations, no random init",
     "The only mechanism that keeps the first five evaluations pinned at the seed."),
]

# Diagnostic campaigns, with why each is not the headline.
DIAGNOSTIC_RUNS = [
    ("section4_tier2_cpu", "single-channel step, w_os=2",
     "early evaluation-signal reading; uniformly ~3x too easy"),
    ("section4_tier2_record", "16 s E_Toggle record, w_os=2",
     "signal retracted: the record is the SysID excitation, not the cost test"),
    ("section4_tier2_wos0", "16 s record, w_os=0",
     "diagnostic only - zeroing the published overshoot weight"),
]


def resolve_campaign(campaign: str = "") -> str:
    """Pick the campaign directory to display.

    The canonical name is preferred, but a reviewer who ran the campaign into a
    different ``--out-dir`` should still see their results rather than an empty
    panel, so fall back to whichever ``section4_tier2*`` directory holds the
    most completed cells.
    """

    name = campaign or CANONICAL_CAMPAIGN
    if (REPORTS_DIR / name / "cells").is_dir():
        return name
    candidates = []
    for path in sorted(REPORTS_DIR.glob("section4_tier2*/cells")):
        completed = len(list(path.glob("*.json")))
        if completed:
            candidates.append((completed, path.parent.name))
    if candidates:
        return max(candidates)[1]
    return name


def _load_rows(campaign: str, protocol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((REPORTS_DIR / campaign / "cells").glob(f"{protocol}__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "ok":
            rows.extend(payload.get("rows", []))
    return rows


def _describe(values: list[float]) -> dict[str, float]:
    return {"n": len(values), "mean": st.fmean(values),
            "median": pct(values, 50.0), "p5": pct(values, 5.0),
            "p95": pct(values, 95.0)}


def _per_cell_median(rows, method) -> dict[tuple[str, str], float]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["method"] == method:
            buckets[(r["pool_id"], r["drift"])].append(float(r["final_best_cost"]))
    return {k: pct(v, 50.0) for k, v in buckets.items()}


def _win_rate(rows, challenger, baseline, plants=None) -> dict[str, Any]:
    a, b = _per_cell_median(rows, challenger), _per_cell_median(rows, baseline)
    keys = sorted(set(a) & set(b))
    if plants is not None:
        keys = [k for k in keys if k[0] in plants]
    wins = sum(1 for k in keys if a[k] < b[k])
    return {"wins": wins, "pairs": len(keys),
            "percent": 100.0 * wins / len(keys) if keys else None}


def _protocol_payload(rows, reference_methods, bounds) -> dict[str, Any]:
    methods = []
    for m in METHOD_ORDER:
        vals = [float(r["final_best_cost"]) for r in rows if r["method"] == m]
        if not vals:
            continue
        stats = _describe(vals)
        ref = reference_methods.get(METHOD_LABEL[m], {})
        methods.append({
            "method": METHOD_LABEL[m], "real_evals": REAL_EVALS[m], **stats,
            "paper_median": ref.get("field_matched_median_S"),
            "paper_mean": ref.get("mean_S"),
            "paper_p5": ref.get("p5_S"), "paper_p95": ref.get("p95_S"),
        })

    anchored5 = tuple(p for p, _ in RETUNING_PLANTS if p != "P189")
    win_rates = {
        "hgs_vs_csbo_anchored5": _win_rate(rows, "HGS-only", "CS-BO", anchored5),
        "hgs_vs_csbo_pooled": _win_rate(rows, "HGS-only", "CS-BO"),
        "hgs_vs_csbo_p189": _win_rate(rows, "HGS-only", "CS-BO", ("P189",)),
        "hgsbo5_vs_csbo": _win_rate(rows, "HGS+BO5", "CS-BO"),
    }

    per_plant = []
    for pool, dash in RETUNING_PLANTS:
        cs = [float(r["final_best_cost"]) for r in rows
              if r["method"] == "CS-BO" and r["pool_id"] == pool]
        hg = [float(r["final_best_cost"]) for r in rows
              if r["method"] == "HGS-only" and r["pool_id"] == pool]
        mare = [float(r["twin_mare_percent"]) for r in rows
                if r["method"] == "HGS-only" and r["pool_id"] == pool
                and r.get("twin_mare_percent") is not None]
        cell_bounds = [v for k, v in bounds.items() if k.startswith(f"{pool}|")]
        if not cs:
            continue
        per_plant.append({
            "pool_id": pool, "dashboard_id": dash,
            "twin_mare_percent": pct(mare, 50.0) if mare else None,
            "cs_bo_median": pct(cs, 50.0),
            "hgs_only_median": pct(hg, 50.0) if hg else None,
            "bound_median": pct(cell_bounds, 50.0) if cell_bounds else None,
        })
    return {"methods": methods, "win_rates": win_rates, "per_plant": per_plant}


def _convergence(rows) -> dict[str, Any]:
    ours = {}
    for m in ("CS-BO", "WS-BO"):
        tr = [r["trajectory"] for r in rows if r["method"] == m and r.get("trajectory")]
        if not tr:
            continue
        n = max(len(t) for t in tr)
        ours[m] = [pct([t[k] for t in tr if len(t) > k], 50.0) for k in range(n)]
    paper = {"CS-BO": [], "WS-BO": []}
    ref_csv = REFERENCE_DIR / "fig7a_convergence_reference.csv"
    if ref_csv.exists():
        for r in csv.DictReader(ref_csv.open(encoding="utf-8-sig")):
            paper["CS-BO"].append(float(r["CS_BO_median"]))
            paper["WS-BO"].append(float(r["WS_BO_median"]))
    return {
        "evals": list(range(1, 31)),
        "ours_cs": ours.get("CS-BO", [])[:30], "ours_ws": ours.get("WS-BO", [])[:30],
        "paper_cs": paper["CS-BO"][:30], "paper_ws": paper["WS-BO"][:30],
        "paper_source": "fig7a_convergence_reference.csv (figure package, comparison only)",
    }


def _variant_row(key, label, method, space, hypothesis, finals, traj5, traj30,
                 hgs, bounds, hgs_pooled_median) -> dict[str, Any]:
    """One configuration's line in the sensitivity table.

    ``finals`` maps (pool_id, drift) -> the per-cell median final cost of this
    baseline; the win rate is HGS-only beating it on strict inequality, over
    the cells both sides cover.
    """

    keys = sorted(set(finals) & set(hgs))
    wins = sum(1 for k in keys if hgs[k] < finals[k])
    ratios = [finals[k] / bounds[f"{k[0]}|{k[1]}"]
              for k in keys if bounds.get(f"{k[0]}|{k[1]}")]
    median30 = pct(traj30, 50.0) if traj30 else None
    return {
        "key": key,
        "label": label,
        "baseline_method": method,
        "search_space": space,
        "hypothesis": hypothesis,
        "median_at_5": pct(traj5, 50.0) if traj5 else None,
        "median_at_30": median30,
        "final_over_bound": pct(ratios, 50.0) if ratios else None,
        "cs_bo_over_hgs": (median30 / hgs_pooled_median
                           if median30 and hgs_pooled_median else None),
        "hgs_win_percent": 100.0 * wins / len(keys) if keys else None,
        "n_runs": len(traj30),
        "n_pairs": len(keys),
        "reference_only": False,
        "is_default": key == "V-base",
    }


def bo_config_sensitivity(campaign: str = CANONICAL_CAMPAIGN,
                          study_path: Path | None = None) -> dict[str, Any]:
    """How far the Section 4.2 ordering moves under unprinted BO settings.

    The paper never prints its ``gp_minimize`` ``dimensions`` list. Our default
    (log-uniform priors on Kp* and the T_I scale) converges CS-BO(30) onto the
    per-cell achievable bound, so the twin transfer cannot win a strict pair and
    the published ordering inverts. This block reports our configuration beside
    the three the study varied and the paper's own numbers, so the panel can
    show the *band* the headline sits in rather than asserting one end of it.

    Degrades to ``available: False`` where the study has not been run.
    """

    campaign = resolve_campaign(campaign)
    path = BO_CONFIG_STUDY if study_path is None else study_path
    if not path.exists():
        return {
            "available": False,
            "variants": [],
            "reason": (
                f"No BO-configuration study at {path}. Run it first: "
                ".venv/bin/python run_bo_config_study.py"
            ),
        }

    fm_rows = _load_rows(campaign, "field_matched")
    bounds_path = REPORTS_DIR / campaign / "cell_bounds.json"
    bounds = (json.loads(bounds_path.read_text(encoding="utf-8"))
              if bounds_path.exists() else {})
    hgs = _per_cell_median(fm_rows, "HGS-only")
    hgs_pooled = [float(r["final_best_cost"]) for r in fm_rows
                  if r["method"] == "HGS-only"]
    hgs_pooled_median = pct(hgs_pooled, 50.0) if hgs_pooled else None

    # Our own configuration, read off the canonical campaign's CS-BO runs.
    base_traj = [r["trajectory"] for r in fm_rows
                 if r["method"] == "CS-BO" and r.get("trajectory")]
    variants = [_variant_row(
        "V-base", "Dashboard default", "CS-BO",
        "Kp* 0.5-300, TI/auto 0.02-100, log-uniform",
        "Log-uniform priors over both gains - the sophisticated default.",
        _per_cell_median(fm_rows, "CS-BO"),
        [t[4] for t in base_traj if len(t) > 4],
        [t[29] for t in base_traj if len(t) > 29],
        hgs, bounds, hgs_pooled_median)]

    study = json.loads(path.read_text(encoding="utf-8"))
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in study:
        by_variant[row.get("variant")].append(row)

    for key, method, label, space, hypothesis in BO_VARIANTS:
        runs = [r for r in by_variant.get(key, []) if r.get("running_best")]
        if not runs:
            continue
        cells: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in runs:
            cells[(r["pool"], r["drift"])].append(float(r["running_best"][29]))
        variants.append(_variant_row(
            key, label, method, space, hypothesis,
            {k: pct(v, 50.0) for k, v in cells.items()},
            [float(r["running_best"][4]) for r in runs],
            [float(r["running_best"][29]) for r in runs],
            hgs, bounds, hgs_pooled_median))

    reference = load_retuning_reference()
    ref_methods = {m["method"]: m for m in reference.get("methods", [])}
    paper_win = next((w["percent"] for w in reference.get("win_rates", [])
                      if "anchored" in w["scope"] or "five slow plants" in w["scope"]),
                     None)
    paper_cs = ref_methods.get("CS-BO(30)", {}).get("field_matched_median_S")
    paper_hgs = ref_methods.get("HGS-only", {}).get("field_matched_median_S")
    variants.append({
        "key": "paper",
        "label": "Paper (configuration unprinted)",
        "baseline_method": "CS-BO",
        "search_space": "not stated in main text or supplement (author question Q2)",
        "hypothesis": "The published Table 3 / Fig. 7a ordering.",
        # Their Fig. 7a CS-BO median at five real evaluations.
        "median_at_5": reference.get("warm_start", {}).get("at_5_evals", {}).get("CS_BO"),
        "median_at_30": paper_cs,
        # Their per-cell achievable bounds are not published, so the ratio our
        # local rows carry has no paper counterpart.
        "final_over_bound": None,
        "cs_bo_over_hgs": paper_cs / paper_hgs if paper_cs and paper_hgs else None,
        "hgs_win_percent": paper_win,
        "n_runs": ref_methods.get("CS-BO(30)", {}).get("n"),
        "n_pairs": None,
        "reference_only": True,
        "is_default": False,
    })

    local = [v["hgs_win_percent"] for v in variants
             if not v["reference_only"] and v["hgs_win_percent"] is not None]
    return {
        "available": True,
        "source": "reports/bo_config_study/results.json (540 runs, 142.5 min GPU)",
        "question": (
            "Which unprinted BO setting decides whether the twin transfer beats "
            "cold-start BO?"
        ),
        "finding": (
            "Box width does not move it and warm-start seeding does not move it; "
            "the sampling prior does. Log-uniform priors converge CS-BO(30) onto "
            "the achievable bound (ratio ~1.00) and the twin wins 0% of cells; "
            "linear-uniform priors on absolute gains leave it short and the twin "
            "wins ~42%, against the paper's 58%. Nothing else changed - not the "
            "plant, the twin, the cost, or the transfer."
        ),
        "band": {
            "min_percent": min(local) if local else None,
            "max_percent": max(local) if local else None,
            "paper_percent": paper_win,
        },
        "variants": variants,
    }


def _diagnostic_summaries() -> list[dict[str, Any]]:
    out = []
    for campaign, model, note in DIAGNOSTIC_RUNS:
        path = REPORTS_DIR / campaign / "tier2_method_runs.csv"
        if not path.exists():
            continue
        cs, hg = [], []
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r.get("protocol") != "field_matched":
                continue
            if r["method"] == "CS-BO":
                cs.append(float(r["final_best_cost"]))
            elif r["method"] == "HGS-only":
                hg.append(float(r["final_best_cost"]))
        if cs:
            out.append({"campaign": campaign, "cost_model": model, "note": note,
                        "cs_bo_median": pct(cs, 50.0),
                        "hgs_only_median": pct(hg, 50.0) if hg else None})
    return out


def retuning_results_payload(campaign: str = CANONICAL_CAMPAIGN) -> dict[str, Any]:
    """Everything the Section-4 dashboard page renders, in one payload."""

    campaign = resolve_campaign(campaign)
    cell_dir = REPORTS_DIR / campaign / "cells"
    if not cell_dir.exists():
        raise ValueError(
            f"No campaign checkpoints at {cell_dir}. Run the offline campaign "
            "first: .venv/bin/python run_tier2_retuning_campaign.py "
            "--backend jax --eval-model stepseq-clean-true-clamp"
        )

    reference = load_retuning_reference()
    ref_methods = {m["method"]: m for m in reference.get("methods", [])}

    bounds_path = REPORTS_DIR / campaign / "cell_bounds.json"
    bounds = (json.loads(bounds_path.read_text(encoding="utf-8"))
              if bounds_path.exists() else {})

    fm_rows = _load_rows(campaign, "field_matched")
    lo_rows = _load_rows(campaign, "logging_only")
    if not fm_rows:
        raise ValueError(f"campaign {campaign} has no completed field-matched cells")

    payload: dict[str, Any] = {
        "campaign": campaign,
        "provenance": {
            "evaluation_model": "stepseq-clean-true-clamp",
            "evaluation_model_meaning": (
                "sequential per-channel +20% steps, one per 5 s episode, held; "
                "deterministic loop; cost on true span tensions; published "
                "cost weights (w_os = 2)"
            ),
            "evaluation_model_status": (
                "RECONSTRUCTED - the papers do not state the test signal the "
                "retuning cost is measured on; this reading survived a "
                "23-model falsification battery scored against the paper's "
                "own published statistics"
            ),
            "run_date": "2026-08-20",
            "backend": "JAX / RTX 5070 Ti, 120 cells, 0 failures, 219.8 min",
            "cells": {"field_matched": len({(r['pool_id'], r['drift']) for r in fm_rows}),
                      "logging_only": len({(r['pool_id'], r['drift']) for r in lo_rows})},
            "reference_data_usage": "comparison_only_never_dashboard_calculation_input",
        },
        "caveats": [
            "The evaluation signal is reconstructed, not published: main-text "
            "Sec. 2.5 defines the metrics of Eq. (12) but no document states "
            "the experiment they are measured on (author question Q1).",
            "The BO search space and priors are unprinted; defensible choices "
            "move the HGS-vs-CS-BO win rate between 0% and 42% with nothing "
            "else changing (author question Q2).",
            "Our CS-BO(30) converges to the per-cell achievable bound "
            "(ratio 0.999), so strict-inequality win rates read ~0% here even "
            "where costs are near-identical.",
            "Residual model-fidelity gap: our achievable floor sits ~25% above "
            "the paper's, concentrated in the hard plants (P189: 3.29 vs ~2.5).",
        ],
        "protocols": {
            "field_matched": _protocol_payload(fm_rows, ref_methods, bounds),
            "logging_only": _protocol_payload(lo_rows, ref_methods, {}),
        },
        "convergence": _convergence(fm_rows),
        "bo_config_sensitivity": bo_config_sensitivity(campaign),
        "diagnostic_runs": _diagnostic_summaries(),
        "reports": [
            {"label": "Final report (PDF)",
             "url": "/artifacts/reports/section4_final_report.pdf"},
            {"label": "Gap re-examination round 2 (PDF)",
             "url": "/artifacts/reports/section4_gap_round2.pdf"},
            {"label": "Reproduction session report (PDF)",
             "url": "/artifacts/reports/section4_reproduction_report.pdf"},
            {"label": "Interactive figures",
             "url": "/artifacts/reports/section4_figures.html"},
        ],
    }
    return payload
