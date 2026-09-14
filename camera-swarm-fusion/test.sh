#!/usr/bin/env bash
set -Eeuo pipefail

# --- Config ---
ROOT="/home/admin/iiot_project/c920s_v4l2"
VENV="$HOME/iiot_env/bin/activate"

# Camera / detection flags
DEVICE="${DEVICE:-8}"
FPS="${FPS:-15}"
DURATION="${DURATION:-40}"
CALIB_REL="${CALIB_REL:-calib/c920s_1920x1080_simple.yml}"
OUT_REL="${OUT_REL:-data/sessions}"
DICT="${DICT:-4x4_50}"
MARKER_LEN_M="${MARKER_LEN_M:-0.0476}"

# Listener (optional) — lives in the robot project
LISTENER_FILE="${LISTENER_FILE:-/home/admin/iiot_project/robot/listener.py}"
PORT="${PORT:-5000}"

# MQTT / Data team flags (only used if PUBLISH=1)
BROKER_IP="${BROKER_IP:-192.168.1.76}"
DEVICE_ID="${DEVICE_ID:-CameraPi}"
CLIENT_TYPE="${CLIENT_TYPE:-CAMERA}"

# --- Helpers ---
kill_port() {
  if command -v ss >/dev/null 2>&1 && ss -ltnp | grep -q ":$PORT "; then
    PIDS=$(ss -ltnp | awk -v p=":$PORT" '$4 ~ p {print $6}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)
    for pid in $PIDS; do kill -9 "$pid" || true; done
    echo "Cleared port $PORT."
  else
    echo "Port $PORT is free."
  fi
}

start_listener() {
  if [ -f "$LISTENER_FILE" ]; then
    echo "Starting listener: $LISTENER_FILE on 0.0.0.0:$PORT ..."
    python3 "$LISTENER_FILE" &
    LISTENER_PID=$!
    trap 'kill $LISTENER_PID 2>/dev/null || true' INT TERM EXIT
  else
    echo "No listener.py found at $LISTENER_FILE; skipping listener."
  fi
}

# --- Main ---
kill_port

# Activate venv if available
if [ -f "$VENV" ]; then
  # shellcheck disable=SC1090
  source "$VENV"
  echo "Activated venv: $VENV"
else
  echo "WARNING: venv not found at $VENV — continuing with system Python."
fi

cd "$ROOT"

# Ensure project root is importable (helps editable runs)
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Build base CLI args (capture/detection)
ARGS=(
  --device "$DEVICE"
  --fps "$FPS"
  --duration "$DURATION"
  --calib "$CALIB_REL"
  --out "$OUT_REL"
  --dict "$DICT"
  --marker-length-m "$MARKER_LEN_M"
)

# Optional publishing flags (enabled with PUBLISH=1)
if [ "${PUBLISH:-0}" = "1" ]; then
  echo "Publishing is ENABLED (PUBLISH=1). Broker: $BROKER_IP"
  ARGS+=( --broker-ip "$BROKER_IP" --device-id "$DEVICE_ID" --client-type "$CLIENT_TYPE" )
else
  echo "Publishing is DISABLED (omit PUBLISH=1 to run local-only)."
fi

# Start listener in background if present
start_listener

# Run the CLI as a module from project root
echo "Running: python -m iiot_pipeline.cli ${ARGS[*]}"
python -m iiot_pipeline.cli "${ARGS[@]}"
exit


