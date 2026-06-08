#!/usr/bin/env bash
# Start the wiki-viewer in the background on 0.0.0.0:5175 (override with
# WIKI_VIEWER_HOST / WIKI_VIEWER_PORT). Builds the frontend if needed.
#
#   ./start.sh            # start (build only if dist/ is missing)
#   ./start.sh --build    # force a fresh build, then start
#
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

HOST="${WIKI_VIEWER_HOST:-0.0.0.0}"
PORT="${WIKI_VIEWER_PORT:-5175}"
PID_FILE="$APP_DIR/.wiki-viewer.pid"
LOG_FILE="$APP_DIR/.wiki-viewer.log"

FORCE_BUILD=false
[ "${1:-}" = "--build" ] && FORCE_BUILD=true

# Already running?
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "wiki-viewer already running (pid $(cat "$PID_FILE")). Use ./stop.sh first."
  exit 0
fi

# Dependencies.
if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install
fi

# Build the frontend if missing or forced.
if [ "$FORCE_BUILD" = true ] || [ ! -f dist/index.html ]; then
  echo "Building frontend..."
  npm run build
fi

echo "Starting wiki-viewer on http://$HOST:$PORT ..."
WIKI_VIEWER_HOST="$HOST" WIKI_VIEWER_PORT="$PORT" \
  nohup "$APP_DIR/node_modules/.bin/tsx" "$APP_DIR/src/server/serve.ts" \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

# Give it a moment and confirm it came up.
sleep 1.5
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Started (pid $(cat "$PID_FILE")). Logs: $LOG_FILE"
  echo "Reachable at http://$HOST:$PORT"
else
  echo "Failed to start. Last log lines:"
  tail -n 20 "$LOG_FILE" || true
  rm -f "$PID_FILE"
  exit 1
fi
