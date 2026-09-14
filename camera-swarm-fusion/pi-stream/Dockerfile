# Simple Raspberry Pi camera streaming container image.
# Builds on arm64 Debian slim and installs ffmpeg + v4l-utils only.
FROM arm64v8/debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/pi-stream

COPY entrypoint.sh /usr/local/bin/start-stream.sh
RUN chmod +x /usr/local/bin/start-stream.sh

ENV VIDEO_DEVICE=/dev/video0 \
    WIDTH=1280 \
    HEIGHT=720 \
    FPS=30

ENTRYPOINT ["/usr/local/bin/start-stream.sh"]
