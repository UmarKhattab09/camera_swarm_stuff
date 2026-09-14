#!/usr/bin/env bash
set -euo pipefail
DEV=${1:-/dev/video0}

# Focus: manual
v4l2-ctl -d "$DEV" -c focus_auto=0 || true
v4l2-ctl -d "$DEV" -c focus_absolute=120 || true   # tweak while previewing

# Exposure: manual
v4l2-ctl -d "$DEV" -c exposure_auto=1 || true      # 1=manual
v4l2-ctl -d "$DEV" -c exposure_absolute=200 || true

# White balance: manual
v4l2-ctl -d "$DEV" -c white_balance_temperature_auto=0 || true
v4l2-ctl -d "$DEV" -c white_balance_temperature=4500 || true

# Optional gentle defaults
v4l2-ctl -d "$DEV" -c sharpness=128 -c brightness=128 -c contrast=128 -c saturation=128 || true

echo "Locked camera controls on $DEV."
