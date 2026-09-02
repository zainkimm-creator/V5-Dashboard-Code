#!/usr/bin/env python3
"""Run the Tier 2 Section 4.2 retuning campaign.

Six plants x ten drift scenarios x two identification protocols, five retuning
strategies each. Cells are independent, so they are farmed across processes and
each one is checkpointed the moment it finishes: a long run that is interrupted
resumes instead of starting over.

    .venv/bin/python run_tier2_retuning_campaign.py                 # full budget
    .venv/bin/python run_tier2_retuning_campaign.py --smoke         # tiny budget
    .venv/bin/python run_tier2_retuning_campaign.py --protocol field_matched
    .venv/bin/python run_tier2_retuning_campaign.py --resume        # keep existing cells
"""

from __future__ import annotations

import os

# Each cell is a separate process, so the BLAS inside it must stay single
# threaded. Left alone, every worker spawns its own thread pool and the machine
# oversubscribes: a pilot run put 8 workers at ~300 % CPU each and drove the load
# average past 50 while doing less work than one core would have. This has to run
# before numpy/scipy/skopt are imported anywhere, hence the position.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.validation.retuning import (  # noqa: E402
    DRIFT_SCENARIOS,
    HGS_TWIN_BUDGET,
    PROTOCOLS,
    RETUNING_PLANTS,
    run_cell,
)

# The dashboard reads this directory by name, so a plain run of this script
# must land here or the panel will not find it.
CANONICAL_EVAL_MODEL = "stepseq-clean-true-clamp"
OUT_DIR = ROOT / "reports" / "section4_tier2_stepseq"
CELL_DIR = OUT_DIR / "cells"

SMOKE_HGS = {"coarse": 36, "lhs": 20, "fine": 36, "dt_bo": 5}


def cell_key(pool_id: str, drift_code: str, protocol: str) -> str:
    return f"{protocol}__{pool_id}__{drift_code}"


def _run_one(args: tuple) -> tuple[str, dict]:
    """Worker entry point. Returns (key, payload) and never raises: one bad cell
    must not take the campaign down with it."""

    pool_id, drift_code, protocol, hgs_kwargs, bo_seeds, backend = args
    wos = os.environ.get("RETUNING_W_OS")
    if wos is not None:
        import backend.validation.retuning as _R
        _R.COST_OVERSHOOT_WEIGHT_OVERRIDE = float(wos)
    em = os.environ.get("RETUNING_EVAL_MODEL")
    if em is not None:
        import backend.validation.retuning as _R
        _R.EVAL_MODEL_OVERRIDE = em
    key = cell_key(pool_id, drift_code, protocol)
    started = time.time()
    try:
        runs = run_cell(pool_id, drift_code, protocol,
                        bo_seeds=tuple(bo_seeds), hgs_kwargs=hgs_kwargs,
                        backend=backend)
        rows = [r.to_row(pool_id=pool_id, drift=drift_code, protocol=protocol,
                         trajectory=r.trajectory) for r in runs]
        return key, {"status": "ok", "wall_s": time.time() - started, "rows": rows}
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        import traceback
        return key, {"status": "error", "wall_s": time.time() - started,
                     "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc()[-2000:], "rows": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int,
                        default=max(1, min(22, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), action="append",
                        help="restrict to one protocol (repeatable)")
    parser.add_argument("--plants", type=str, default=None,
                        help="comma-separated pool ids, e.g. P001,P189")
    parser.add_argument("--drifts", type=str, default=None,
                        help="comma-separated drift codes, e.g. D01,D07")
    parser.add_argument("--bo-seeds", type=int, default=3)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny HGS budget to exercise the pipeline")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells that already have a checkpoint")
    parser.add_argument("--eval-model", type=str, default=CANONICAL_EVAL_MODEL,
                        help="evaluation-model key; defaults to "
                             f"{CANONICAL_EVAL_MODEL}, the model the dashboard "
                             "reports. Pass another key only to explore.")
    parser.add_argument("--overshoot-weight", type=float, default=None,
                        help="override Eq.(12) w_os (published value is 2); "
                             "results produced with this MUST be labelled")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy",
                        help="simulator backend; jax runs the grid search on GPU")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    cell_dir = args.out_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    protocols = args.protocol or sorted(PROTOCOLS)
    pools = ([p.strip() for p in args.plants.split(",")] if args.plants
             else [pool for pool, _ in RETUNING_PLANTS])
    drifts = ([d.strip() for d in args.drifts.split(",")] if args.drifts
              else [s.code for s in DRIFT_SCENARIOS])
    if args.eval_model is not None:
        import backend.validation.retuning as _R
        _R.EVAL_MODEL_OVERRIDE = args.eval_model
        os.environ["RETUNING_EVAL_MODEL"] = args.eval_model
        print(f"evaluation model: {args.eval_model}")
    if args.overshoot_weight is not None:
        import backend.validation.retuning as _R
        _R.COST_OVERSHOOT_WEIGHT_OVERRIDE = args.overshoot_weight
        os.environ["RETUNING_W_OS"] = str(args.overshoot_weight)
        print(f"!! overshoot weight overridden to {args.overshoot_weight} "
              f"(published value is 2)")
    hgs_kwargs = SMOKE_HGS if args.smoke else {}
    bo_seeds = tuple(range(args.bo_seeds))

    jobs = []
    skipped = 0
    for protocol in protocols:
        for pool_id in pools:
            for drift_code in drifts:
                key = cell_key(pool_id, drift_code, protocol)
                if args.resume and (cell_dir / f"{key}.json").exists():
                    skipped += 1
                    continue
                jobs.append((pool_id, drift_code, protocol, hgs_kwargs,
                             bo_seeds, args.backend))

    twin_evals = (sum(SMOKE_HGS.values()) if args.smoke else HGS_TWIN_BUDGET)
    print(f"cells        : {len(jobs)} to run, {skipped} already checkpointed")
    print(f"grid         : {len(pools)} plants x {len(drifts)} drifts x "
          f"{len(protocols)} protocols")
    print(f"HGS budget   : ~{twin_evals} twin evals/cell"
          f"{' (SMOKE)' if args.smoke else ''}")
    print(f"BO seeds     : {args.bo_seeds} (stochastic arms)")
    print(f"backend      : {args.backend}")
    print(f"workers      : {args.workers}")
    print(f"out          : {args.out_dir}", flush=True)
    if not jobs:
        print("nothing to do")
        return 0

    started = time.time()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for future in as_completed(futures):
            key, payload = future.result()
            (cell_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
            done += 1
            if payload["status"] != "ok":
                failed += 1
                print(f"  [{done}/{len(jobs)}] {key} ERROR {payload['error']}",
                      flush=True)
            else:
                elapsed = time.time() - started
                rate = elapsed / done
                eta = rate * (len(jobs) - done)
                print(f"  [{done}/{len(jobs)}] {key} ok "
                      f"({payload['wall_s']:.0f}s)  elapsed {elapsed/60:.1f}m  "
                      f"eta {eta/60:.1f}m", flush=True)

    total = time.time() - started
    print(f"\ncampaign finished in {total/60:.1f} min: "
          f"{done - failed} ok, {failed} failed")
    print(f"checkpoints: {cell_dir}")
    print("next: .venv/bin/python run_tier2_retuning_report.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
