# Camera Sensor Fusion Docker Setup

This repository now includes a minimal Dockerized workflow for two components:

1. **pi-stream/** – a reusable Raspberry Pi camera streaming container that wraps `ffmpeg`.
2. **middleware/** – the Python data fusion container that runs `python -m camera_fusion.launch configs/cam1.json configs/cam2.json`.

Both services keep configuration in environment files or mounted folders so that adding new cameras only requires duplicating configuration, not editing code.

---

## Recommended Folder Layout

```
camera_sensor_fusion/
+-- pi-stream/            # Pi camera streaming container assets
|   +-- Dockerfile        # arm64 Debian + ffmpeg build
|   +-- docker-compose.yml
|   +-- entrypoint.sh     # builds the ffmpeg command from env vars
|   +-- .env.cam1         # sample env for camera 1
|   +-- .env.cam2         # sample env for camera 2
+-- middleware/           # Middleware container assets
|   +-- Dockerfile        # python:3.11-slim + OpenCV deps
|   +-- docker-compose.yml
+-- configs/              # JSON configs mounted read-only
+-- calib/                # Camera calibration files (mounted)
+-- data/                 # Output/telemetry data (mounted)
+-- logs/                 # Runtime logs mounted from container
+-- scripts/              # Any helper scripts you already have
+-- camera_fusion/        # Python source copied into middleware image
```

Feel free to add more per-camera `.env` files under `pi-stream/` (e.g., `.env.cam3`) as you scale the system.

---

## Pi Camera Streaming Container

1. **Select the correct env file** on each Pi:
   - Camera 1: `cp pi-stream/.env.cam1 pi-stream/.env`
   - Camera 2: `cp pi-stream/.env.cam2 pi-stream/.env`
   - Edit the copied `.env` if you need to update width/height/FPS/device.
2. **Build and start** (run on each Pi with its own USB camera):
   ```sh
   cd ~/camera_sensor_fusion/pi-stream
   docker compose build
   docker compose up -d
   ```
3. **Logs:** `docker compose logs -f`
4. **Stop/restart:**
   - Stop: `docker compose down`
   - Restart only container: `docker compose restart`

Environment variables exported via `.env`:
- `CAMERA_NAME` – used for log prefix/container identification.
- `VIDEO_DEVICE` – usually `/dev/video0`.
- `DEST_IP` / `DEST_PORT` – middleware target.
- `WIDTH`, `HEIGHT`, `FPS` – video format negotiated with the camera.

The container runs with host networking and maps `/dev/video0` directly so RTP packets leave the Pi exactly as before.

---

## Middleware Container

Middleware runs on the processing machine that receives both RTP streams.

1. **Build and start**:
   ```sh
   cd ~/camera_sensor_fusion/middleware
   docker compose build
   docker compose up -d
   ```
2. **Logs:** `docker compose logs -f`
3. **Stop:** `docker compose down`

Mounted volumes (relative to repo root):
- `../configs -> /app/configs`
- `../calib -> /app/calib`
- `../data -> /app/data`
- `../logs -> /app/logs`

These ensure configuration and outputs persist outside the container and can be edited without rebuilding.

---

## Adding More Cameras

1. Copy `pi-stream/.env.cam1` to a new file (e.g., `.env.cam3`).
2. Update `CAMERA_NAME`, `DEST_PORT`, and any camera-specific settings.
3. Deploy the same `pi-stream` folder to the new Pi, copy the env file to `.env`, and run `docker compose up -d`.

No changes are required in the middleware container unless you need to pass additional config JSON files into the `camera_fusion.launch` command.

---

## Common Pitfalls & Notes

- **/dev/video0 permissions:** Ensure the Pi user is in the `video` group. If access is denied, either add `privileged: true` in `pi-stream/docker-compose.yml` temporarily or fix udev permissions.
- **Host networking:** Both Compose stacks use `network_mode: host` for low-latency RTP. This only works on Linux-based Docker hosts (including Raspberry Pi OS). Avoid running these Compose files on Docker Desktop for Mac/Windows.
- **UDP/RTP ports:** Confirm that ports 5000 and 5001 are free on the middleware host before starting the container. Use `sudo lsof -iUDP:5000` if needed.
- **Volume paths:** The middleware Compose file uses `../configs`, `../calib`, `../data`, and `../logs`. Run Compose from the `middleware/` directory so the relative paths resolve correctly, and ensure those folders exist.
- **ffmpeg availability:** The Pi image installs upstream Debian `ffmpeg`. If your Pi requires MMAL or special codecs, install extra packages or swap the base image to `balenalib` variants.
- **OpenCV/GStreamer dependencies:** The middleware Dockerfile pulls in `libgl1`, `libglib2.0-0`, and GStreamer plugins so `opencv-python` can open RTP streams. Add more plugins if your pipeline needs them.
- **Rebuilding after code changes:** Any changes under `camera_fusion/` require rerunning `docker compose build` in `middleware/` so the updated code is copied into the image.
- **Time sync:** Low-latency fusion assumes synchronized clocks. Use `chrony` or `systemd-timesyncd` on all machines.

---

## Exact Command Reference

### Middleware machine (192.168.1.113)
```sh
cd ~/camera_sensor_fusion/middleware
# Build image (repeat when Python deps/code change)
docker compose build
# Start middleware service
docker compose up -d
# Tail logs
docker compose logs -f
# Stop the service
docker compose down
```

### cam1 Pi (192.168.1.80)
```sh
cd ~/camera_sensor_fusion/pi-stream
cp .env.cam1 .env    # edit .env if needed
docker compose build
docker compose up -d
docker compose logs -f
docker compose down   # stop when necessary
```

### cam2 Pi (192.168.1.92)
```sh
cd ~/camera_sensor_fusion/pi-stream
cp .env.cam2 .env    # edit .env if needed
docker compose build
docker compose up -d
docker compose logs -f
docker compose down
```

With this setup, each Raspberry Pi streams RTP/H.264 exactly as before, while the middleware continues to run the existing `camera_sensor_fusion` processing pipeline inside a reproducible container.
