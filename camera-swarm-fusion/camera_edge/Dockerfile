# Lightweight image for camera_edge -- the Swarm-managed detection/pose/MQTT
# container. It never imports cv2.VideoCapture or references /dev/video*;
# its only dependency on the physical camera is the Unix socket exposed by
# camera_host (bind-mounted in, not baked into this image). No devices:,
# privileged:, or device_cgroup_rules: are needed anywhere for this image.
#
# python:3.11-slim is a multi-arch image (amd64/arm64/armv7), so this same
# Dockerfile builds on a dev machine and on the Pi.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# opencv-contrib-python-headless still soft-depends on libGL/libglib at
# import time on some base images; install the small runtime libs instead
# of pulling in the full desktop OpenCV stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-edge.txt /app/requirements-edge.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-edge.txt

COPY camera_edge /app/camera_edge
COPY camera_fusion /app/camera_fusion
COPY iiot_pipeline /app/iiot_pipeline
COPY calib /app/calib

CMD ["python3", "-m", "camera_edge"]
