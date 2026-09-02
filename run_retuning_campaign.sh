#!/usr/bin/env bash
# Run the full Section 4 retuning campaign ON THE GPU, then aggregate it.
#
# This is the one command a reviewer needs on a Linux box with an NVIDIA GPU:
#
#   ./run_retuning_campaign.sh                            # full campaign, GPU
#   WORKERS=4 ./run_retuning_campaign.sh                  # fewer GPU workers
#   ./run_retuning_campaign.sh --protocol field_matched   # one protocol only
#   ./run_retuning_campaign.sh --smoke                    # ~25 s/cell pipeline check
#   ALLOW_CPU=1 ./run_retuning_campaign.sh                # fall back to CPU
#
# Extra arguments are passed straight through to the campaign script.
#
# Everything is computed from the model on this machine. Nothing is read back
# from stored results.
#
# Runtime. 120 cells (6 plants x 10 drifts x 2 identification protocols, five
# retuning strategies each). On the GPU a cell is roughly 4x faster than on a
# CPU core, and the workers share one device, so expect a few hours for the
# full campaign. Every cell is checkpointed as it finishes and the run defaults
# to --resume, so an interrupted run continues instead of starting over.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "No .venv found. Create it first:" >&2
    echo "  python3.11 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-gpu.txt" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. CUDA libraries.
#
# jax[cuda12] ships its CUDA runtime inside the venv as nvidia/*/lib, and does
# NOT add those directories to the loader path. Without this, JAX reports
# "Could not find cuda drivers on your machine, GPU will not be used" and
# silently runs on the CPU at roughly a quarter of the speed - on a machine
# whose GPU is sitting idle. Set it before Python starts.
# ---------------------------------------------------------------------------
SITE_NVIDIA="$("$PYTHON" - <<'PY'
import os, sysconfig
p = os.path.join(sysconfig.get_paths()["purelib"], "nvidia")
print(p if os.path.isdir(p) else "")
PY
)"
if [[ -z "$SITE_NVIDIA" ]]; then
    # lib64 layouts put it beside purelib rather than in it
    SITE_NVIDIA="$(find "$ROOT/.venv" -maxdepth 4 -type d -name nvidia 2>/dev/null | head -1)"
fi
if [[ -n "$SITE_NVIDIA" && -d "$SITE_NVIDIA" ]]; then
    CUDA_LIBS="$(find "$SITE_NVIDIA" -maxdepth 2 -type d -name lib 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:-}"
fi

# ---------------------------------------------------------------------------
# 2. GPU memory across worker processes.
#
# Each cell runs in its own process and each one initialises JAX. JAX
# preallocates ~75 % of the device by default, so the second worker would fail
# to allocate anything. Growing on demand lets the workers share the card, and
# matters doubly on a shared HPC node where someone else already holds memory.
# ---------------------------------------------------------------------------
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-}"

# ---------------------------------------------------------------------------
# 3. Preflight: confirm the GPU is really being used before spending hours.
# ---------------------------------------------------------------------------
echo "Checking the GPU..."
BACKEND="$("$PYTHON" - <<'PY' 2>/dev/null
import warnings; warnings.filterwarnings("ignore")
try:
    import jax
    print(jax.default_backend())
except Exception as exc:                       # noqa: BLE001
    print(f"error:{type(exc).__name__}")
PY
)"

if [[ "$BACKEND" != "gpu" ]]; then
    echo >&2
    echo "  JAX is not using the GPU (backend reported: ${BACKEND:-none})." >&2
    echo >&2
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  nvidia-smi sees:" >&2
        nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader >&2 || true
        echo >&2
        echo "  So the driver is fine and this is a Python-side problem. Usually:" >&2
        echo "    .venv/bin/python -m pip install -r requirements-gpu.txt" >&2
        echo "  installs the CUDA build of JAX. Check what it reports with:" >&2
        echo "    .venv/bin/python -c 'import jax; print(jax.devices())'" >&2
    else
        echo "  nvidia-smi is not on PATH - this machine has no usable NVIDIA" >&2
        echo "  driver, or you are on a login node. On an HPC cluster, request" >&2
        echo "  a GPU node first (e.g. srun --gres=gpu:1 ... or qsub -l ngpus=1)." >&2
    fi
    echo >&2
    echo "  To run on the CPU instead (about 4x slower per cell):" >&2
    echo "    ALLOW_CPU=1 ./run_retuning_campaign.sh" >&2
    echo >&2
    if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
        exit 1
    fi
    echo "  ALLOW_CPU=1 set - continuing on the CPU." >&2
    BACKEND_FLAG="numpy"
else
    "$PYTHON" - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import jax
for d in jax.devices():
    print(f"  GPU ready: {d.device_kind} (id {d.id})")
PY
    BACKEND_FLAG="jax"
fi

# ---------------------------------------------------------------------------
# 4. Worker count.
#
# On the GPU the workers share one device, so more of them stops helping well
# before the core count - they queue on the card and each holds a CUDA context.
# Six is the shape the published campaign used. On the CPU, one per core less
# one.
# ---------------------------------------------------------------------------
if [[ -z "${WORKERS:-}" ]]; then
    CORES="$(nproc 2>/dev/null || echo 4)"
    if [[ "$BACKEND_FLAG" == "jax" ]]; then
        WORKERS=$(( CORES < 6 ? (CORES > 1 ? CORES - 1 : 1) : 6 ))
    else
        WORKERS=$(( CORES > 2 ? CORES - 1 : 1 ))
    fi
fi

echo "=============================================================="
echo " Section 4 retuning campaign"
echo "   backend : $BACKEND_FLAG"
echo "   workers : $WORKERS"
echo "   output  : reports/section4_tier2_stepseq/cells"
echo "   resume  : on (finished cells are skipped)"
echo "=============================================================="

"$PYTHON" run_tier2_retuning_campaign.py \
    --backend "$BACKEND_FLAG" --workers "$WORKERS" --resume "$@"

# The aggregation step has to read the same directory the campaign wrote. Most
# pass-through flags mean nothing to it, so forward only --out-dir.
REPORT_ARGS=()
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--out-dir" ]]; then
        REPORT_ARGS=(--out-dir "$arg")
    elif [[ "$arg" == --out-dir=* ]]; then
        REPORT_ARGS=(--out-dir "${arg#--out-dir=}")
    fi
    prev="$arg"
done

echo
echo "Aggregating the campaign into the comparison tables..."
"$PYTHON" run_tier2_retuning_report.py "${REPORT_ARGS[@]+"${REPORT_ARGS[@]}"}" || {
    echo "Campaign checkpoints are written; only the aggregation step failed." >&2
    exit 1
}

echo
echo "Done. Start the dashboard and open the Retuning section:"
echo "  ./start_dashboard.sh      then  http://127.0.0.1:5198/#retuning"
