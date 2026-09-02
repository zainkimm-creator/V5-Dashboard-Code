#!/usr/bin/env bash
# Run the full Section 4 retuning campaign on this machine, then aggregate it.
#
# This is the one command a reviewer needs. It computes every cell from the
# model - nothing is read back from stored results - and writes the per-cell
# checkpoints the dashboard's Retuning panel reads.
#
#   ./run_retuning_campaign.sh                  # full 60-cell campaign, both protocols
#   WORKERS=8 ./run_retuning_campaign.sh        # cap the worker count
#   ./run_retuning_campaign.sh --protocol field_matched   # one protocol only
#
# Any extra arguments are passed straight through to the campaign script.
#
# Runtime. The work is 120 cells (6 plants x 10 drifts x 2 protocols), each
# roughly 10-20 minutes of single-core physics. Wall time is therefore about
# (120 x 15 min) / WORKERS: near 1.5 hours on 24 cores, most of a working day
# on 4. Every cell is checkpointed the moment it finishes and the run defaults
# to --resume, so an interrupted run continues where it stopped rather than
# starting over. Re-running after a completed campaign is a no-op.
#
# No GPU is required: the default backend is numpy. Pass --backend jax to use
# one if the machine has it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "No .venv found. Create it first:" >&2
    echo "  python3.11 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt" >&2
    exit 1
fi

# One process per core, but leave a core free so the box stays usable. The
# campaign pins each worker's BLAS to a single thread, so workers do not
# oversubscribe each other.
if [[ -z "${WORKERS:-}" ]]; then
    CORES="$(nproc 2>/dev/null || echo 4)"
    WORKERS=$(( CORES > 2 ? CORES - 1 : 1 ))
fi

echo "=============================================================="
echo " Section 4 retuning campaign"
echo "   workers : $WORKERS"
echo "   output  : reports/section4_tier2_stepseq/cells"
echo "   resume  : on (finished cells are skipped)"
echo "=============================================================="

"$PYTHON" run_tier2_retuning_campaign.py --workers "$WORKERS" --resume "$@"

echo
echo "Aggregating the campaign into the comparison tables..."
"$PYTHON" run_tier2_retuning_report.py || {
    echo "Campaign checkpoints are written; only the aggregation step failed." >&2
    exit 1
}

echo
echo "Done. Start the dashboard and open the Retuning section:"
echo "  ./start_dashboard.sh      then  http://127.0.0.1:5198/#retuning"
