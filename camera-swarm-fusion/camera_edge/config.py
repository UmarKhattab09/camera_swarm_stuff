from __future__ import annotations

from dataclasses import dataclass
import os
import socket
import time


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_target_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(v) for v in raw.split(",") if v.strip()]


def _parse_marker_lengths(raw: str | None) -> dict[int, float] | None:
    if not raw:
        return None
    lengths: dict[int, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        marker_id, length = pair.split(":", 1)
        lengths[int(marker_id)] = float(length)
    return lengths or None


@dataclass
class EdgeAgentConfig:
    socket_path: str
    session_id: str
    fps: int
    width: int
    height: int
    reconnect_delay_s: float
    calibration_path: str
    aruco_dict: str
    marker_length_m: float
    marker_lengths_m: dict[int, float] | None
    target_ids: list[int] | None
    mqtt_broker_ip: str
    mqtt_broker_port: int
    device_id: str
    client_type: str
    camera_name: str
    csv_path: str | None
    health_path: str
    stale_after_s: int
    start_session_on_boot: bool
    stop_session_on_exit: bool
    use_frame_stream: bool
    poll_interval_s: float

    @classmethod
    def from_env(cls) -> "EdgeAgentConfig":
        hostname = socket.gethostname()
        timestamp = int(time.time())
        csv_path = os.getenv("CAMERA_CSV_PATH", "/app/data/detections.csv")
        return cls(
            socket_path=os.getenv("CAMERA_SOCKET_PATH", "/run/camera-hw/camera.sock"),
            session_id=os.getenv("CAMERA_SESSION_ID", f"{hostname}-{timestamp}"),
            fps=max(1, int(os.getenv("CAMERA_FPS", "15"))),
            width=int(os.getenv("CAMERA_WIDTH", "1920")),
            height=int(os.getenv("CAMERA_HEIGHT", "1080")),
            reconnect_delay_s=float(os.getenv("CAMERA_RECONNECT_DELAY_S", "2.0")),
            calibration_path=os.getenv("CAMERA_CALIBRATION_PATH", "calib/c920s_1920x1080_simple.yml"),
            aruco_dict=os.getenv("CAMERA_ARUCO_DICT", "4x4_50"),
            marker_length_m=float(os.getenv("CAMERA_MARKER_LENGTH_M", "0.035")),
            marker_lengths_m=_parse_marker_lengths(os.getenv("CAMERA_MARKER_LENGTHS_M")),
            target_ids=_parse_target_ids(os.getenv("CAMERA_TARGET_IDS")),
            mqtt_broker_ip=os.getenv("MQTT_BROKER_IP", "127.0.0.1"),
            mqtt_broker_port=int(os.getenv("MQTT_BROKER_PORT", "1883")),
            device_id=os.getenv("DEVICE_ID", "CameraPi"),
            client_type=os.getenv("CLIENT_TYPE", "CAMERA"),
            camera_name=os.getenv("CAMERA_NAME", hostname),
            csv_path=csv_path if csv_path else None,
            health_path=os.getenv("CAMERA_EDGE_HEALTH_PATH", "/tmp/camera-edge-health.json"),
            stale_after_s=max(5, int(os.getenv("CAMERA_EDGE_STALE_AFTER_S", "30"))),
            start_session_on_boot=_env_flag("CAMERA_AUTO_START_SESSION", True),
            stop_session_on_exit=_env_flag("CAMERA_STOP_SESSION_ON_EXIT", False),
            use_frame_stream=_env_flag("CAMERA_USE_FRAME_STREAM", True),
            poll_interval_s=float(os.getenv("CAMERA_POLL_INTERVAL_S", "0.05")),
        )
