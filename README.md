# R2R Live SysID Dashboard — retuning re-run build

A simulation and system-identification dashboard for a roll-to-roll web line.
It simulates a pool of ten heterogeneous plants under closed-loop cascade
control, logs them through a configurable acquisition chain (logging period,
sensor noise, anti-alias filter), identifies the plant parameters with a
weighted one-step prediction-error method, and reports how identification
accuracy responds to each design choice.

Everything the dashboard shows is **recomputed from the model** on the machine
it runs on. Nothing is read back from stored results.

## What it contains

| Section | Question it answers |
|---|---|
| Simulation / Plants / SysID | The forward model, the ten-plant pool, and a single identification run |
| Logging rate | How fine must the logging period be before identification degrades? |
| Excitation | Which excitation profile identifies the plant best? |
| Noise-aware logging (LPF) | How does the anti-alias cutoff interact with sensor noise? |
| Closed-loop damping | Does a stiffer controller buy any noise robustness? |
| Retuning | Can a digital twin replace real-plant retuning runs after drift? |
| Drift | How does identification hold up as the plant drifts? |

## Requirements

- Python 3.11
- Node 18 or newer (22 tested)
- Linux or Windows. No GPU is required; JAX is used opportunistically for the
  retuning campaign and falls back to CPU.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

cd frontend && npm ci && cd ..
```

`requirements-lock.txt` is a `pip freeze` of a verified environment; use it
instead of `requirements-dev.txt` to reproduce exact versions.

## Running

```bash
./start_dashboard.sh
# override the ports if they are taken:
BACKEND_PORT=8020 FRONTEND_PORT=5200 ./start_dashboard.sh
```

- Dashboard: <http://127.0.0.1:5198/>
- API: <http://127.0.0.1:8014/> — health at `/health`, OpenAPI at `/docs`
- Logs: `logs/backend.log`, `logs/frontend.log`

Both services also run standalone:

```bash
.venv/bin/python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8014
cd frontend && npm run dev -- --port 5198 --strictPort
```

## Tests

```bash
.venv/bin/python -m pytest -q
cd frontend && npm run build
```

A number of tests skip on a fresh clone. They fall into two groups, and both
are expected:

- tests that check **generated study artifacts** (`reports/…`) — run the study
  first and they activate;
- tests that compare against an **optional published-result set** — see below.

## Repository layout

```
backend/
  api/          FastAPI routes
  models/       plant physics, modal analysis, controller, noise
  sysid/        the weighted one-step PEM estimator
  validation/   one module per study section
frontend/src/   the React dashboard (single-page)
configs/        default run configuration
data/
  model_inputs/ the model definition - REQUIRED (see below)
  processed/    generated intermediates (git-ignored)
reports/        generated study outputs (git-ignored)
tests/          pytest suite
```

## Data

### `data/model_inputs/` — required

Three files define the simulated system. They ship with the repository and the
code will not run without them.

| File | What it defines |
|---|---|
| `ten_plant_parameters.csv` | One row per plant: geometry, inertias, friction, span lengths, reel radii, tension setpoints, controller integral time |
| `excitation_schedules.csv` | Edge-level excitation schedules on the 1 ms integration grid |
| `model_parameters.json` | Physics constants, plant characteristics, state-matrix structure, cascade controller settings |

The plant registry and every excitation profile are built from these files
rather than from constants in the source, so changing a plant or a schedule is
a matter of editing the file.

### `data/reference_results/` — optional, not distributed

Several sections can display a column of previously published result values
beside their own, so a reader can see the difference. **That comparison set is
not included in this repository.** With it absent — the normal case — every
study computes and reports its own numbers alone, comparison columns read `—`,
and the Tier-1 verification route returns a 422 explaining why.

To enable the comparison display, create `data/reference_results/` and place
the result JSONs in it; `backend/validation/paper_reference.py` lists the file
names it looks for. Comparison values are never inputs to a calculation.

## Re-running the retuning section on this machine

**This build ships no precomputed retuning results.** The Retuning panel is
empty until you run the campaign, and everything it then shows is computed on
your machine from the model.

```bash
./run_retuning_campaign.sh
```

That is the whole procedure. It runs all 120 cells (6 plants x 10 drift
scenarios x 2 identification protocols, five retuning strategies each),
aggregates them, and writes the checkpoints the dashboard reads. Then start the
dashboard and open the Retuning section.

### What to expect

| | |
|---|---|
| Hardware | **No GPU required.** The default backend is `numpy`. Pass `--backend jax` to use a GPU if one is present. |
| Cost per cell | roughly 10–20 minutes of single-core physics |
| Wall time | about `(120 x 15 min) / workers` — near 1.5 h on 24 cores, most of a day on 4 |
| Workers | one per core, less one, unless you set `WORKERS=` |
| Interruptions | every cell is checkpointed as it finishes and the run defaults to `--resume`, so Ctrl-C and re-run continues where it stopped |
| Re-running | a completed campaign is a no-op; delete `reports/section4_tier2_stepseq/` to force a fresh run |

Useful variations — anything you pass is forwarded to the campaign script:

```bash
WORKERS=8 ./run_retuning_campaign.sh                   # cap the worker count
./run_retuning_campaign.sh --protocol field_matched    # one protocol (halves the work)
./run_retuning_campaign.sh --plants P001,P189          # a two-plant subset
./run_retuning_campaign.sh --smoke                     # ~2 min/cell, proves the pipeline
```

`--smoke` shrinks the twin search budget rather than the grid. It is the fastest
way to confirm the pipeline works end to end before committing to the full run;
its numbers are not comparable to a full campaign.

### Where the results land

`reports/section4_tier2_stepseq/cells/` — one JSON per cell, named
`<protocol>__<plant>__<drift>.json`. The dashboard reads that directory by
name; if you direct the campaign elsewhere with `--out-dir`, the panel falls
back to whichever `reports/section4_tier2*` directory holds the most completed
cells, so your results still appear.

The campaign defaults to the `stepseq-clean-true-clamp` evaluation model, which
is the model the dashboard labels its results with. Override it with
`--eval-model` only to explore alternatives — the panel will still be labelled
`stepseq-clean-true-clamp`, so results from another model are not comparable.

## Generating the other study outputs

The five live sections compute inside the HTTP request and need no batch step.

Other batch runners:

| Script | What it does |
|---|---|
| `run_full_sweep.py` | Full-factorial raw-data export in the 27-column template |
| `run_excitation_all10.py` | Excitation study over all ten plants |
| `run_paper_csv_recalc.py` | Regenerates logging-rate and excitation artifacts from the model inputs |
| `run_bo_config_study.py` | Sensitivity of the retuning result to the optimiser's configuration |
| `run_tier1_retuning_check.py` | Verification pass; needs the optional comparison set |
