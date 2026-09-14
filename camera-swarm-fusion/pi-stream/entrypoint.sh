#!/usr/bin/env bash
# Lightweight ffmpeg launcher that streams V4L2 video to RTP based on env vars.
set -euo pipefail

log() {
  printf '[%s] %s\n' "${CAMERA_NAME:-camera}" "$1"
}

: "${DEST_IP:?DEST_IP must be set}"
: "${DEST_PORT:?DEST_PORT must be set}"
VIDEO_DEVICE="${VIDEO_DEVICE:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
CAMERA_NAME="${CAMERA_NAME:-pi-cam}"
INPUT_FORMAT="${INPUT_FORMAT:-mjpeg}"

case "${INPUT_FORMAT,,}" in
  mjpeg)
    FF_INPUT_FORMAT="mjpeg"
    V4L2_PIXFMT="MJPG"
    ;;
  yuyv|yuyv422)
    FF_INPUT_FORMAT="yuyv422"
    V4L2_PIXFMT="YUYV"
    ;;
  *)
    log "Unsupported INPUT_FORMAT=${INPUT_FORMAT}. Use 'mjpeg' or 'yuyv'."
    exit 2
    ;;
esac

log "Starting ffmpeg pipeline from ${VIDEO_DEVICE} to ${DEST_IP}:${DEST_PORT}"
log "Requested capture ${WIDTH}x${HEIGHT} @ ${FPS}fps, input_format=${FF_INPUT_FORMAT}"

log "Camera advertised formats/resolutions (${VIDEO_DEVICE}):"
v4l2-ctl -d "${VIDEO_DEVICE}" --list-formats-ext || {
  log "Failed to query V4L2 formats for ${VIDEO_DEVICE}"
  exit 3
}

log "Applying V4L2 capture format: pixelformat=${V4L2_PIXFMT} ${WIDTH}x${HEIGHT}"
v4l2-ctl -d "${VIDEO_DEVICE}" --set-fmt-video=width="${WIDTH}",height="${HEIGHT}",pixelformat="${V4L2_PIXFMT}" || {
  log "Failed to set requested V4L2 format"
  exit 4
}

V4L2_FMT="$(v4l2-ctl -d "${VIDEO_DEVICE}" --get-fmt-video)"
log "Active V4L2 format:\n${V4L2_FMT}"

if ! printf '%s\n' "${V4L2_FMT}" | grep -q "Width/Height[[:space:]]*:[[:space:]]*${WIDTH}/${HEIGHT}"; then
  log "Camera is NOT honoring requested resolution ${WIDTH}x${HEIGHT}. Refusing to stream."
  exit 5
fi

if ! printf '%s\n' "${V4L2_FMT}" | grep -q "Pixel Format[[:space:]]*:[[:space:]]*'${V4L2_PIXFMT}'"; then
  log "Camera is NOT honoring requested pixel format ${V4L2_PIXFMT}. Refusing to stream."
  exit 6
fi

log "V4L2 capture verified at ${WIDTH}x${HEIGHT} ${V4L2_PIXFMT}"
log "Running ffmpeg capture command"

exec ffmpeg \
  -f v4l2 \
  -input_format "${FF_INPUT_FORMAT}" \
  -video_size "${WIDTH}x${HEIGHT}" \
  -framerate "${FPS}" \
  -i "${VIDEO_DEVICE}" \
  -an \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -preset ultrafast \
  -tune zerolatency \
  -payload_type 96 \
  -f rtp "rtp://${DEST_IP}:${DEST_PORT}"
