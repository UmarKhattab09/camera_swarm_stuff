# C920s + V4L2 ArUco Pipeline  
*(Facade + Strategy Design Patterns)*

This repository captures frames from a Logitech C920 (or any V4L2 USB camera), **undistorts** them using a calibration file, detects **ArUco markers**, estimates pose using PnP, and saves results to a per-session folder.  

The design applies two object-oriented design patterns:  
- **Facade** a single entrypoint to run the full pipeline  
- **Strategy** pluggable steps for capture, preprocessing, undistortion, detection, and localization  

---

## Quick Start

### 1. Find your camera index
```bash
v4l2-ctl --list-devices
```
Look for the *HD Pro Webcam C920* block (e.g. `/dev/video8`).

---

### 2. Activate environment
```bash
source .venv/bin/activate
```

---

### 3. Run a session
Example (USB C920 on `/dev/video8`, 1080p @ 15fps, ArUco 4x4_50, marker side = 14.45 cm):

```bash
python3 cli.py --device 8 --fps 15 --duration 10   --calib calib/c920s_1920x1080_simple.yml   --out data/sessions   --dict 4x4_50 --marker-length-m 0.1445
```

After it finishes, open the newest folder under `data/sessions/`.

---

## Multi-camera service (new)

This repo now includes a clean, multi-camera runner in the `camera_fusion` package.
Each camera runs in its own process with an isolated config and output folder prefix.

### Quick Start (new system)

1. **Find your camera device:**
   ```bash
   v4l2-ctl --list-devices
   ```
   Note the device path (e.g., `/dev/video8`) or index (e.g., `8`).

2. **Edit a config file** (e.g., `configs/cam1.json`):
   ```json
   {
     "camera_name": "cam1",
     "device": 8,
     "target_ids": [1, 2, 3]
   }
   ```

3. **Run it:**
   ```bash
   python -m camera_fusion.run --config configs/cam1.json
   ```

4. **Check outputs:**
   ```bash
   ls data/sessions/cam1_session_*/
   ```

Each camera session folder now also includes baseline ArUco metrics files:
- `metrics_frames.csv`: per-frame detection count, success/partial ratio, primary-marker pose, and latency measurements
- `metrics_summary.json`: aggregate success rate (`success_rate_all3`), pose stability stddevs, and latency summary stats

To quickly print the newest metrics file paths for cam1/cam2:
```bash
python scripts/show_latest_metrics.py
```
Add `--show-summary` to print the latest `metrics_summary.json` content.

### One camera

Run with a config file:
```bash
python -m camera_fusion.run --config configs/cam1.json
```

Override any config values from the CLI (no code edits needed):

```bash
python -m camera_fusion.run --config configs/cam1.json --device 8 --fps 10 --target-ids 1 2 3
```

### Multiple cameras (separate terminals)

```bash
python -m camera_fusion.run --config configs/cam1.json
python -m camera_fusion.run --config configs/cam2.json
```

### Multiple cameras (single launcher)

```bash
python -m camera_fusion.launch configs/cam1.json configs/cam2.json
```

Press `Ctrl+C` once to stop all cameras cleanly.

## Docker Swarm deployment (host/edge split)

`camera_fusion.run` (above) still works unchanged for a single Pi running
outside Docker -- keep using it for that case. If you need to run under
Docker Swarm, use the new `camera_host/` + `camera_edge/` split instead:
Swarm services cannot be granted access to host devices like
`/dev/video0` (`devices:`, `privileged:`, and `device_cgroup_rules:` are
all rejected or silently ignored by `docker stack deploy`), so the camera
is owned by a small hardware service running directly on the Pi
(`camera_host/`, started via `systemd`, not Docker), and the Swarm
container (`camera_edge/`) talks to it over a Unix socket instead of
opening the camera itself.

```bash
# On the Pi, outside Docker:
pip install -r requirements-host.txt
python -m camera_host

# Then, locally or in Swarm, once /run/camera-hw/camera.sock exists:
docker compose up -d --build        # local, non-Swarm test
docker stack deploy -c swarm.yml camera   # Swarm
```

See [CAMERA_EDGE_REDESIGN.md](CAMERA_EDGE_REDESIGN.md) for the full
architecture, API contract, environment variables, and bring-up order.

## Option B: Live Streaming (Jetson → Dell via RTP/UDP)

For distributed setups where **Jetson captures** from USB cameras and **Dell processes** the streams over the network.

### On Jetson (capture & stream)

Stream camera 1 (e.g., `/dev/video8`) to Dell IP `192.168.1.100` on port 5000:

```bash
gst-launch-1.0 v4l2src device=/dev/video8 ! \
  video/x-raw,width=1280,height=720,framerate=15/1 ! \
  videoconvert ! video/x-raw,format=I420 ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 ! \
  rtph264pay ! \
  udpsink host=192.168.1.100 port=5000 sync=false
```

Stream camera 2 (e.g., `/dev/video4`) to same Dell on port 5002:

```bash
gst-launch-1.0 v4l2src device=/dev/video4 ! \
  video/x-raw,width=1280,height=720,framerate=15/1 ! \
  videoconvert ! video/x-raw,format=I420 ! \
  x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 ! \
  rtph264pay ! \
  udpsink host=192.168.1.100 port=5002 sync=false
```

*(Replace `192.168.1.100` with Dell's actual IP address.)*

### On Dell (receive & process)

Use streaming configs (already configured in `configs/cam1_stream.json` and `configs/cam2_stream.json`):

```bash
python -m camera_fusion.launch configs/cam1_stream.json configs/cam2_stream.json
```

This runs two workers concurrently:
- `cam1_stream`: listens on UDP port 5000 (from Jetson cam1)
- `cam2_stream`: listens on UDP port 5002 (from Jetson cam2)

Press `Ctrl+C` to stop.

**Important:**
- Jetson and Dell must be on the same network
- Ports must match: Jetson sends to 5000/5002 → Dell listens on 5000/5002
- Dell IP in Jetson's `udpsink host=` must be correct

### Optional: YAML configs


Configs are JSON by default. To use YAML instead:

```bash
pip install pyyaml
```

Then create `configs/cam1.yaml`:
```yaml
camera_name: cam1
device: 8
fps: 15
target_ids: [1, 2, 3]
reference_id: 0  # Floor/world marker
```

### Reference frame / world coordinate system

To compute robot marker poses relative to a fixed world frame (e.g., a floor marker):

1. **Place a reference ArUco marker** on the floor or fixed location
2. **Set `reference_id`** to that marker's ID in your config:
   ```json
   {
     "reference_id": 0,
     "target_ids": [0, 1, 2]
   }
   ```
3. **Include reference ID in `target_ids`** so it gets detected

When the reference marker is visible, the CSV will include:
- Camera-relative poses (`rvec_x/y/z`, `tvec_x/y/z`)
- Reference-relative poses (`ref_rvec_x/y/z`, `ref_tvec_x/y/z`)
- `ref_visible` flag (1 if reference seen, 0 otherwise)

When reference marker is NOT visible in a frame:
- `ref_visible = 0`
- Reference-relative fields are NaN
- Camera-relative poses still logged

**Use case:** Track robot joint markers relative to floor coordinate system instead of camera.

### Dry run (no physical camera)

```bash
python -m camera_fusion.run --config configs/cam1.json --dry-run --max-frames 5 --no-detect --no-save-frames
```

### Config file options

All config files (JSON or YAML) support these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `camera_name` | string | `"cam"` | Unique identifier for logging and output folder prefix |
| `device` | int or string | `0` | Camera index (e.g., `8`) or path (e.g., `"/dev/video8"`) |
| `fps` | int | `15` | Target frame rate |
| `width` | int | `1920` | Frame width |
| `height` | int | `1080` | Frame height |
| `calibration_path` | string | `"calib/c920s_1920x1080_simple.yml"` | Path to calibration file |
| `session_root` | string | `"data/sessions"` | Root directory for session outputs |
| `duration_sec` | float | `30.0` | Max session duration (seconds) |
| `aruco_dict` | string | `"4x4_50"` | ArUco dictionary (accepts `4x4_50`, `DICT_4X4_50`, `4X4_50`, etc.) |
| `marker_length_m` | float | `0.035` | Marker side length in meters - measure the **outer black square** edge-to-edge (NOT diagonal, NOT inner pattern) |
| `target_ids` | list of int | `null` | Filter to specific marker IDs (e.g., `[1, 2, 3]`); `null` = all |
| `reference_id` | int or null | `null` | ID of reference/world marker (e.g., floor marker). When set, computes poses of other markers relative to this reference frame |
| `marker_lengths_m` | map | `null` | Per-marker lengths override (e.g., `{ "0": 0.150, "1": 0.050 }`). Keys may be strings in JSON |
| `no_detect` | bool | `false` | Skip detection (capture only) |
| `dry_run` | bool | `false` | Use synthetic frames (no physical camera) |
| `max_frames` | int or null | `null` | Stop after N frames |
| `save_annotated` | bool | `true` | Save frames with ArUco markers drawn |
| `save_frames` | bool | `true` | Save undistorted original frames |
| `expected_marker_count` | int | `3` | Expected number of visible markers per frame for success-rate metrics |
| `metrics_enabled` | bool | `true` | Enable per-camera `metrics_frames.csv` and `metrics_summary.json` in each session folder |
| `metrics_flush_every_n_frames` | int | `30` | Flush metrics CSV to disk every N frames for interruption robustness |
| `metrics_primary_marker_strategy` | string | `"min_id"` | Primary marker choice for pose stability logging (`min_id` supported) |
| `lightglue` | object or null | `null` | LightGlue fallback configuration (see below) |

### LightGlue Fallback (Advanced)

When ArUco detection fails to find expected markers (e.g., due to occlusion, motion blur, lighting), the LightGlue fallback can recover missing markers using SuperPoint feature matching and homography-based corner estimation.

**Features:**
- Template-based marker re-acquisition using SuperPoint+LightGlue matching
- Temporal tracking with optical flow for recently-seen markers
- Homography-based corner recovery with RANSAC
- Optional corner refinement with sub-pixel accuracy
- Debug visualization with color-coded marker sources

**Configuration:**

Add a `lightglue` section to your config:

```json
{
  "camera_name": "cam1",
  "device": 0,
  "target_ids": [0, 1, 2, 3],
  "lightglue": {
    "enabled": true,
    "device": "cpu",
    "template_dir": "templates/markers",
    "min_inliers": 4,
    "max_age_frames": 5,
    "roi_expand_px": 50,
    "debug_save": true,
    "corner_refine": true,
    "match_threshold": 0.2
  }
}
```

**LightGlue Config Options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable LightGlue fallback (requires PyTorch + LightGlue) |
| `device` | string | `"cpu"` | PyTorch device: `"cpu"`, `"cuda"`, or `"cuda:0"` |
| `template_dir` | string | `"templates/markers"` | Directory containing marker template images (id_<ID>.png) |
| `min_inliers` | int | `4` | Minimum RANSAC inliers required for valid homography |
| `max_age_frames` | int | `5` | Max frames since last detection to attempt tracking |
| `roi_expand_px` | int | `50` | ROI expansion in pixels for optical flow tracking |
| `debug_save` | bool | `false` | Save debug frames to session folder |
| `corner_refine` | bool | `true` | Refine corners with cv2.cornerSubPix |
| `match_threshold` | float | `0.2` | LightGlue matching threshold |

For a two-camera setup, keep the template images in separate folders:

- `templates/markers/cam1/id_<MARKER_ID>.png`
- `templates/markers/cam2/id_<MARKER_ID>.png`

Each folder should contain the marker template PNGs for that camera, not a single combined scene image.

**Setup:**

1. **Install dependencies:**
   ```bash
   # CPU version
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install lightglue
   
   # OR CUDA version
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install lightglue
   ```
   
   For Jetson platforms, see `requirements-jetson.txt` for platform-specific instructions.

2. **Generate marker templates:**
   ```bash
   python lightglue/scripts/create_marker_templates.py \
     --marker-ids 0 1 2 3 \
     --dict 4x4_50 \
     --output-dir templates/markers/cam1 \
     --size 400
   python lightglue/scripts/create_marker_templates.py \
     --marker-ids 0 1 2 3 \
     --dict 4x4_50 \
     --output-dir templates/markers/cam2 \
     --size 400
   ```
   
   This creates PNG templates for each camera, such as `templates/markers/cam1/id_0.png` and `templates/markers/cam2/id_0.png`.

   If you already have camera-specific template images, place them in the matching camera folder and keep the `id_<MARKER_ID>.png` naming convention.

3. **Update the camera configs:**
   - `configs/cam1.json` should point `lightglue.template_dir` at `templates/markers/cam1`
   - `configs/cam2.json` should point `lightglue.template_dir` at `templates/markers/cam2`

4. **Run the middleware stack:**
   ```bash
   cd middleware
   docker compose up --build
   ```

  The middleware compose file now mounts `../templates` into the container so template changes are picked up on the next run, and the middleware image installs `torch` + `lightglue` during build.

5. **Run with fallback enabled:**
   ```bash
   python -m camera_fusion.run --config configs/cam_lightglue_example.json
   ```

   The pi-stream container does not use LightGlue directly; it only needs to keep streaming the camera feed. The fallback runs in the middleware container that consumes those streams.

**Visualization:**

When LightGlue fallback is enabled, annotated frames show markers color-coded by detection source:
- **Green**: ArUco detection (normal)
- **Cyan**: LightGlue tracking (optical flow)
- **Magenta**: LightGlue re-acquire (template matching)

Marker labels show source: `ID (LG:track)` or `ID (LG:reacquire)`

**Testing:**

Test the fallback on a saved frame:

```bash
python lightglue/scripts/test_lightglue_fallback.py \
  --frame data/sessions/cam1_session_20260204_120000/frames/frame_000010.png \
  --templates templates/markers \
  --marker-ids 0 1 2 3 \
  --device cpu
```

**Graceful Degradation:**

If PyTorch or LightGlue are not installed, the system logs a warning and continues with ArUco-only detection (no crashes).

**Performance Notes:**

- CPU inference: ~100-500ms per missing marker
- CUDA inference: ~20-100ms per missing marker
- Tracking is faster (~10-30ms) than template re-acquisition
- Enable `debug_save` only for debugging (increases storage)

### CLI overrides

Any config option can be overridden from the command line:

```bash
python -m camera_fusion.run --config configs/cam1.json \
  --camera-name cam_left \
  --device /dev/video8 \
  --fps 20 \
  --width 1280 \
  --height 720 \
  --calib calib/other.yml \
  --out data/custom_sessions \
  --duration 60.0 \
  --dict 5x5_100 \
  --marker-length-m 0.05 \
  --target-ids 10 11 12 \
  --no-detect \
  --dry-run \
  --max-frames 100 \
  --no-save-frames \
  --no-save-annotated
```

### Graceful shutdown

Press `Ctrl+C` (SIGINT) to stop cleanly. The worker will:
- Release the camera
- Close output files
- Write session summary to logs

When using the multi-camera launcher, `Ctrl+C` stops all processes.

### Troubleshooting

- **Camera fails to open**: Check `device` index/path and ensure no other process is using it.
- **ArUco detection fails**: Install `opencv-contrib-python` (not just `opencv-python`).
- **YAML config errors**: Install PyYAML: `pip install pyyaml`.
- **Device string `/dev/videoX`**: Automatically parsed to integer index `X` with V4L2 backend.
- **Logs not appearing**: Check `session_root/<camera_name>_session_*/logs/session.log`.
- **Multiple instances conflict**: Ensure each camera has a unique `camera_name` and `device`.
- **Inaccurate pose estimation**: Measure `marker_length_m` precisely. Use a ruler to measure the outer black square edge-to-edge (e.g., if marker is 35mm wide, use `0.035`). Do NOT measure diagonally or just the inner pattern.

## Outputs

### Multi-camera service outputs

Each camera worker creates a unique session folder:

```
data/sessions/<camera_name>_session_YYYYMMDD_HHMMSS/
  frames/             # undistorted originals (no drawings)
  annotated/          # frames with ArUco markers + axes drawn
  detections.csv      # frame_idx, recorded_at, marker_id, rvec_*, tvec_*, [ref_visible, ref_rvec_*, ref_tvec_*], length_m, image_path
  metrics_frames.csv  # per-frame baseline metrics (counts, success, pose, timing)
  metrics_summary.json# aggregate baseline metrics (rates, stddev stability, latency)
  logs/session.log    # per-camera logs with [camera_name] prefix
  config.json         # snapshot of config used for this session
```

**CSV columns:**
- Without reference: `frame_idx, recorded_at, marker_id, rvec_x/y/z, tvec_x/y/z, length_m, image_path`
- With reference: adds `ref_visible, ref_rvec_x/y/z, ref_tvec_x/y/z` (poses relative to reference marker)

Example with two cameras:
```
data/sessions/
  cam1_session_20260202_143012/
  cam2_session_20260202_143012/
```

### Legacy CLI outputs

```
data/sessions/<aruco_session_YYYYMMDD_HHMMSS>/
  frames/        # undistorted originals (no drawings)
  annotated/     # frames with detections (boxes + axes + text)
  detections.csv # ts_iso, frame_idx, marker_id, rvec_*, tvec_*, img_path
  logs/session.log
  config.json
```

---

## Legacy CLI Flags

| Flag | Description |
|------|-------------|
| `--device` | V4L2 index (see `v4l2-ctl --list-devices`). |
| `--fps` / `--duration` | Capture rate & session length. |
| `--calib` | Camera calibration `.yml` (must match resolution). |
| `--out` | Root folder for session outputs. |
| `--dict` | ArUco dictionary (e.g., `4x4_50`, `4x4_100`, `5x5_50`, …). |
| `--marker-length-m` | Marker side length in meters (needed for pose & axes). Use `0` to skip pose estimation (IDs only). |

