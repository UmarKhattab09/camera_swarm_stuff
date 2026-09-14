# Camera Edge Redesign Brief

## What Changed

The camera pipeline is now split into two layers, following the same
pattern `cylissama/imu_node_smr` used to solve this for IMU sensors
(see that repo's `EDGE_REDESIGN.md`).

1. Pi-local hardware service (`camera_host/`)
   - Runs on the Raspberry Pi host, NOT under Docker Swarm
   - Owns `/dev/video0` (or whatever `CAMERA_DEVICE` is configured) exclusively
     via OpenCV's `cv2.VideoCapture`
   - Starts at boot with `systemd`
   - Exposes a local Unix-socket API at `/run/camera-hw/camera.sock`
   - Sits idle (not grabbing frames) until a session is started, then
     captures at the requested rate until the session is stopped

2. Swarm-managed edge container (`camera_edge/`)
   - Runs without `privileged: true` and with no `devices:` grant
   - Never imports `cv2.VideoCapture` or references `/dev/video*`
   - Talks to the Pi-local hardware service over the Unix socket
   - Starts and stops camera sessions
   - Pulls frames from the host service
   - Runs the existing ArUco detection + PnP pose estimation, unchanged
   - Publishes results to MQTT and writes CSV output

## Why This Fixes Swarm

The previous single-process design (`camera_fusion.run`, still available
for non-Swarm single-Pi deployments) opens `/dev/video0` directly inside
whatever process runs it. Under `docker stack deploy`, Swarm services have
no equivalent of `docker run --device`: `devices:` is silently ignored,
and `privileged:` / `device_cgroup_rules:` are rejected outright with
`"Additional property ... is not allowed"`. The repo's earlier
`swarm_front/` attempt worked around this with `privileged: true`, which
grants far more host access than the camera alone requires.

The new edge container is hardware-agnostic. It only assumes:

- a local socket exists at `/run/camera-hw/camera.sock`
- the host camera service is running
- the host camera service can provide session control and frames

That makes the container fully compatible with Docker Swarm.

## What The Pi-Local Service Does

The host service (`camera_host/manager.py`, modeled on
`imu_host/manager.py`):

- connects to the camera at startup and stays connected/idle
- starts and stops capture sessions
- actively grabs frames at the requested rate only while a session is active
- JPEG-encodes each frame and exposes it via the API
- retries/reconnects if the camera disappears or a read fails

Recommended host startup:

```bash
python -m camera_host
```

Recommended boot-time management:

- install the `systemd` unit at `deploy/systemd/camera-hw.service`
- enable and start it with `systemctl`

## What The Container Must Do To Activate The Camera

The container no longer activates the camera by opening `/dev/video0`
directly. Instead, it activates the workflow through the host service API:

1. wait for the host service to become ready
   - `GET /v1/readyz`
2. start a session
   - `POST /v1/session/start`
3. provide a session request payload like:

```json
{
  "session_id": "cam1-83",
  "fps": 15,
  "width": 1920,
  "height": 1080
}
```

4. connect to the frame stream
   - `GET /v1/frame/stream`
5. for each frame received:
   - decode the JPEG bytes
   - run ArUco detection + PnP pose estimation (unchanged, reused code)
   - publish to MQTT
   - write to CSV
6. optionally stop the session on shutdown
   - `POST /v1/session/stop`

In this repo, that behavior is implemented by:

- host service: `camera_host/`
- edge container agent: `camera_edge/`

## Local API Contract

The Pi-local service exposes these endpoints:

- `GET /v1/healthz`
- `GET /v1/readyz` -- reports whether the camera is actually connected/openable
- `GET /v1/status`
- `POST /v1/session/start`
- `POST /v1/session/stop`
- `GET /v1/frame/latest` -- most recent frame as raw JPEG bytes
  (`Content-Type: image/jpeg`), with `X-Frame-Idx` and `X-Capture-Time-Ms`
  headers
- `GET /v1/frame/stream` -- NDJSON stream, one JSON object per line, for
  lower latency than polling `/v1/frame/latest`

Example stream sample (before `image_b64` is decoded back to JPEG bytes
by `camera_edge/service_client.py`):

```json
{
  "session_id": "cam1-83",
  "source": "camera-hw-service",
  "frame_idx": 42,
  "capture_time_ms": 1711111111111,
  "image_b64": "<base64 JPEG bytes>"
}
```

## MQTT Output

The edge container keeps the existing payload layout produced by
`camera_fusion.output.MqttOutput` and the existing CSV columns produced by
`iiot_pipeline.services.csv_writer.CsvWriter`:

`frame_idx, capture_time, recorded_at, marker_id, rvec_x, rvec_y, rvec_z, tvec_x, tvec_y, tvec_z, length_m, image_path`

MQTT topic convention (`{client_type}/{device_id}`) and row format are
unchanged, so `wku-smr-lab-middleware`'s ingestion keeps working without
modification.

The host service generates:

- `frame_idx`
- `capture_time_ms`
- the JPEG frame bytes

The edge container adds:

- detection + pose estimation (reusing `camera_fusion.detect` and
  `iiot_pipeline.strategies.localize_pnp.PnPLocalize` unchanged)
- `recorded_at` (wall-clock time when the row is written)
- MQTT forwarding and CSV writing (reusing `camera_fusion.output` unchanged)

## Container Runtime Requirements

The container should:

- mount `/run/camera-hw` from the host
- read `CAMERA_SOCKET_PATH`
- wait for the host service to be ready
- request a session start
- reconnect if the stream drops
- keep health state updated
- publish to MQTT
- never require `/dev/video0`
- never require privileged mode

## Useful Environment Variables

Container-side (`camera_edge`):

- `DEVICE_ID`
- `CAMERA_NAME`
- `CAMERA_SOCKET_PATH`
- `CAMERA_SESSION_ID`
- `CAMERA_FPS`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`
- `CAMERA_CALIBRATION_PATH`
- `CAMERA_ARUCO_DICT`
- `CAMERA_MARKER_LENGTH_M`, `CAMERA_MARKER_LENGTHS_M`
- `CAMERA_TARGET_IDS`
- `CAMERA_AUTO_START_SESSION`
- `CAMERA_STOP_SESSION_ON_EXIT`
- `CAMERA_USE_FRAME_STREAM`
- `MQTT_BROKER_IP`, `MQTT_BROKER_PORT`
- `CLIENT_TYPE`
- `CAMERA_CSV_PATH`
- `CAMERA_EDGE_HEALTH_PATH`

Host-side (`camera_host`):

- `CAMERA_SOCKET_PATH`
- `CAMERA_SERVICE_BACKEND` (`real` or `fake`)
- `CAMERA_DEVICE`
- `CAMERA_SERVICE_FPS`, `CAMERA_SERVICE_WIDTH`, `CAMERA_SERVICE_HEIGHT`
- `CAMERA_JPEG_QUALITY`
- `CAMERA_SERVICE_START_ON_BOOT`

## Deployment Notes

### Docker Compose

Use the socket mount:

```yaml
volumes:
  - ./data:/app/data
  - /run/camera-hw:/run/camera-hw
```

Do not use:

- `privileged: true`
- `devices:`
- `/dev/video0`

### Docker Swarm

Your swarm service should:

- run only on Pi nodes that have `camera_host` installed and running
  (pinned via `node.labels.camera == true`)
- bind-mount `/run/camera-hw`
- use a container healthcheck

## Recommended Bring-Up Order

1. Install host dependencies:

```bash
pip install -r requirements-host.txt
```

2. Start the Pi-local service:

```bash
python -m camera_host
```

3. Verify the socket exists:

```bash
ls -l /run/camera-hw/camera.sock
```

4. Confirm `/v1/healthz` and `/v1/readyz` respond.

5. Build and run the edge container as a plain Docker container (not yet
   Swarm):

```bash
docker compose up -d --build
```

6. Confirm it connects to the host socket, starts a session, and MQTT
   messages appear on the broker with the expected payload format.

7. Only then attempt Swarm:

```bash
docker stack deploy -c swarm.yml camera
```

## Health And Troubleshooting

If the container is not streaming:

1. Check the host service is running (`systemctl status camera-hw` on the Pi)
2. Check `/run/camera-hw/camera.sock` exists
3. Check the host service can see the camera (`GET /v1/readyz`)
4. Check MQTT broker connectivity
5. Check container health output:

```bash
python -m camera_edge healthcheck
```

If the camera is missing or not ready:

- `GET /v1/readyz` should report not ready
- `GET /v1/status` should show the last error
- the edge container should keep retrying instead of crashing

## Files To Know

- `camera_host/__main__.py`
- `camera_host/manager.py`
- `camera_host/server.py`
- `camera_edge/__main__.py`
- `camera_edge/agent.py`
- `camera_edge/service_client.py`
- `camera_fusion/detect.py`, `iiot_pipeline/strategies/localize_pnp.py` (reused, unchanged)
- `camera_fusion/output.py` (reused, unchanged)
- `deploy/systemd/camera-hw.service`
- `docker-compose.yml`
- `swarm.yml`

## Short Summary

The Raspberry Pi host now owns the physical camera.
The container now owns detection, pose estimation, forwarding, and
integration. To activate the camera from the container, the container
must start a session through the host service API, then consume the
frame stream and run the existing detection/pose/MQTT pipeline on it.
The single-process `camera_fusion.run` path still works unchanged for
non-Swarm, single-Pi deployments.
