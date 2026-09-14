from dataclasses import dataclass
import os


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_device(name: str, default: str) -> int | str:
    raw = os.getenv(name, default)
    return int(raw) if raw.isdigit() else raw


@dataclass
class HostServiceConfig:
    socket_path: str = "/run/camera-hw/camera.sock"
    backend: str = "real"
    device: int | str = "/dev/video0"
    fps: int = 15
    width: int = 1920
    height: int = 1080
    jpeg_quality: int = 85
    startup_session: bool = False
    startup_session_id: str = "boot-session"
    reconnect_delay_s: float = 2.0
    stream_queue_size: int = 8

    @classmethod
    def from_env(cls) -> "HostServiceConfig":
        return cls(
            socket_path=os.getenv("CAMERA_SOCKET_PATH", "/run/camera-hw/camera.sock"),
            backend=os.getenv("CAMERA_SERVICE_BACKEND", "real"),
            device=_env_device("CAMERA_DEVICE", "/dev/video0"),
            fps=max(1, int(os.getenv("CAMERA_SERVICE_FPS", "15"))),
            width=int(os.getenv("CAMERA_SERVICE_WIDTH", "1920")),
            height=int(os.getenv("CAMERA_SERVICE_HEIGHT", "1080")),
            jpeg_quality=max(1, min(100, int(os.getenv("CAMERA_JPEG_QUALITY", "85")))),
            startup_session=_env_flag("CAMERA_SERVICE_START_ON_BOOT", False),
            startup_session_id=os.getenv("CAMERA_SERVICE_STARTUP_SESSION_ID", "boot-session"),
            reconnect_delay_s=float(os.getenv("CAMERA_RECONNECT_DELAY_S", "2.0")),
            stream_queue_size=max(2, int(os.getenv("CAMERA_STREAM_QUEUE_SIZE", "8"))),
        )
