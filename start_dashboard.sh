#!/usr/bin/env bash
# Linux launcher for the R2R Live SysID Dashboard (Windows equivalent: start_version_2.ps1).
# Starts the FastAPI backend and the Vite dev server, waits on both, and stops
# both on Ctrl-C. Override ports with BACKEND_PORT / FRONTEND_PORT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BACKEND_PORT="${BACKEND_PORT:-8014}"
FRONTEND_PORT="${FRONTEND_PORT:-5198}"

PYTHON="$ROOT/.venv/bin/python"
VITE="$ROOT/frontend/node_modules/.bin/vite"

if [[ ! -x "$PYTHON" ]]; then
    echo "No .venv found. Create it first:" >&2
    echo "  python3.11 -m venv .venv" >&2
    echo "  .venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt" >&2
    exit 1
fi
if [[ ! -x "$VITE" ]]; then
    echo "frontend deps missing. Run: (cd frontend && npm ci)" >&2
    exit 1
fi

mkdir -p "$ROOT/logs"

PIDS=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Both services are exec'd directly rather than through a wrapper (npm run dev),
# so $! is the real process and the trap above can actually stop it.
"$PYTHON" -m uvicorn backend.api.main:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" \
    >"$ROOT/logs/backend.log" 2>&1 &
PIDS+=($!)

sleep 2

# vite takes its root as a positional argument (there is no --root flag), so
# run it from the frontend directory. `exec` makes the subshell become vite,
# which keeps $! pointing at the real process for cleanup().
(
    cd "$ROOT/frontend"
    VITE_API_BASE_URL="http://127.0.0.1:${BACKEND_PORT}" \
        exec "$VITE" --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort
) >"$ROOT/logs/frontend.log" 2>&1 &
PIDS+=($!)

echo "Dashboard : http://127.0.0.1:${FRONTEND_PORT}/"
echo "Backend   : http://127.0.0.1:${BACKEND_PORT}/  (health: /health, docs: /docs)"
echo "Logs      : logs/backend.log  logs/frontend.log"
echo "Ctrl-C to stop both."
wait
