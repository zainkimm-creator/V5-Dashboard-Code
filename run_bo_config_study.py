#!/usr/bin/env python3
"""BO-configuration study: can any unprinted search setting reproduce the
paper's convergence signature?

Background. With the stepseq evaluation model the CS-BO convergence curve
matches the paper's almost exactly through eval 15 and then plateaus at OUR
achievable floor, while theirs keeps descending; and the paper's WS-BO sits
flat at ~10.7 for five evaluations (stuck at the seed) where ours escapes by
eval 3. This study tests the two remaining mechanisms the papers leave
unprinted:

  V-wide    CS-BO in a much wider log box (Kp* 0.5-1000, TI/auto 0.01-1000):
            does late convergence slow to the paper's shape (still descending
            at 30) and does the twin then win pairs?
  V-linabs  CS-BO with LINEAR-uniform priors on absolute gains
            (Kp* 1-300 linear, T_I 0.1-100 s linear) - the unsophisticated
            default choice; wastes evaluations in the flat high region.
  V-cluster WS-BO seeded with a CLUSTER (the SysID-mode point plus four +-30%
            perturbations, no random inits) - the only mechanism that keeps
            the first five evaluations pinned near the seed like the paper's.

Runs on the stepseq evaluation model (published cost, w_os=2), all 60
field-matched cells, 3 seeds for CS variants. Win rates use the existing
stepseq campaign's HGS-only per-cell costs and the per-cell achievable bounds.

    .venv/bin/python run_bo_config_study.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "bo_config_study"
PLANTS = [("P001", "P01"), ("P049", "P02"), ("P053", "P03"),
          ("P158", "P06"), ("P186", "P09"), ("P189", "P10")]
DRIFTS = ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10"]

VARIANTS = {
    # name: (method, kp_lo, kp_hi, kp_prior, ti_mode, ti_lo, ti_hi, ti_prior, seeding)
    "V-wide":    ("CS", 0.5, 1000.0, "log-uniform", "scale", 0.01, 1000.0, "log-uniform", "random5"),
    "V-linabs":  ("CS", 1.0, 300.0, "uniform",     "abs",   0.1,  100.0,  "uniform",     "random5"),
    "V-cluster": ("WS", 0.5, 300.0, "log-uniform", "scale", 0.02, 100.0,  "log-uniform", "cluster"),
}


def run_one(job):
    variant, pool, dash, drift, seed = job
    import numpy as np

    import backend.validation.retuning as R
    from backend.validation.retuning_eval_models import EvalModel, ModelEvaluator
    from backend.validation.plants import parameters_for_plant
    from skopt import gp_minimize
    from skopt.space import Real

    method, kp_lo, kp_hi, kp_prior, ti_mode, ti_lo, ti_hi, ti_prior, seeding = VARIANTS[variant]

    base, _ = parameters_for_plant(dash)
    v0 = float(base.feeder_velocity_m_s)
    auto = R.plant_auto_ti_s(base, v0)
    drifted = R.apply_drift(base, R.DRIFT_BY_CODE[drift])
    ev = ModelEvaluator(drifted, v0, EvalModel("stepseq", False, False, "true", 200.0))

    def cost(x):
        kp, tiv = float(x[0]), float(x[1])
        ti = tiv * auto if ti_mode == "scale" else tiv
        s = float(ev.cost_only(np.array([kp]), np.array([ti]))[0])
        return min(s, 1e6) if math.isfinite(s) else 1e6

    space = [Real(kp_lo, kp_hi, prior=kp_prior),
             Real(ti_lo, ti_hi, prior=ti_prior)]

    if seeding == "cluster":
        # SysID-mode seed plus four +-30% perturbations - five seeded points,
        # NO random initial design (n_initial_points is consumed by x0).
        seed_ti = auto if ti_mode == "abs" else 1.0
        rng = np.random.default_rng(seed)
        x0 = [[100.0, seed_ti]]
        for _ in range(4):
            x0.append([float(np.clip(100.0 * rng.uniform(0.7, 1.3), kp_lo, kp_hi)),
                       float(np.clip(seed_ti * rng.uniform(0.7, 1.3), ti_lo, ti_hi))])
        n_initial = 1
    else:
        x0, n_initial = None, 5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gp_minimize(cost, space, n_calls=30, n_initial_points=n_initial,
                          random_state=seed, x0=x0, acq_func="EI")
    vals = [float(v) for v in res.func_vals]
    running = []
    best = float("inf")
    for v in vals:
        best = min(best, v)
        running.append(best)
    return {"variant": variant, "pool": pool, "drift": drift, "seed": seed,
            "running_best": running, "final": running[-1]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(v, pool, dash, drift, seed)
            for v in VARIANTS
            for pool, dash in PLANTS
            for drift in DRIFTS
            for seed in ((0, 1, 2) if VARIANTS[v][0] == "CS" else (0, 1, 2))]
    print(f"jobs: {len(jobs)}  ({len(VARIANTS)} variants x 60 cells x 3 seeds)")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 60 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {el/60:.1f}m  eta {(el/done*(len(jobs)-done))/60:.1f}m",
                      flush=True)
    (OUT / "results.json").write_text(json.dumps(results), encoding="utf-8")
    print(f"done in {(time.time()-t0)/60:.1f} min -> {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
