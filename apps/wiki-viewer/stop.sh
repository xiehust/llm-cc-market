#!/usr/bin/env bash
# Stop the background wiki-viewer started by ./start.sh.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${WIKI_VIEWER_PORT:-5175}"
PID_FILE="$APP_DIR/.wiki-viewer.pid"

stopped=false

# Kill the recorded process and any children (tsx may spawn a node child).
if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    pkill -TERM -P "$PID" 2>/dev/null || true
    kill -TERM "$PID" 2>/dev/null || true
    stopped=true
    echo "Stopped wiki-viewer (pid $PID)."
  fi
  rm -f "$PID_FILE"
fi

# Safety net: kill whatever is still listening on the port.
PORT_PIDS="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
if [ -n "$PORT_PIDS" ]; then
  echo "Killing process(es) still on port $PORT: $PORT_PIDS"
  # shellcheck disable=SC2086
  kill -TERM $PORT_PIDS 2>/dev/null || true
  stopped=true
fi

[ "$stopped" = true ] && echo "Done." || echo "wiki-viewer was not running."
