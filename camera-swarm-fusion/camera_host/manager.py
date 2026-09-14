from __future__ import annotations

import base64
from collections.abc import Callable
import queue
import threading
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import cv2

if TYPE_CHECKING:
    from camera_fusion.frame_source import FrameSource


HardwareFactory = Callable[[], "FrameSource"]


class ServiceError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


class CameraHardwareService:
    """Owns the physical camera and runs the idle/active capture loop.

    Mirrors imu_host.manager.IMUHardwareService: the capture loop sits idle
    (not touching the camera) until a session is started, then grabs frames
    at the requested rate until the session is stopped. This keeps the
    camera free 24/7 unless something actually needs frames.
    """

    def __init__(
        self,
        backend_name: str,
        hardware_factory: HardwareFactory,
        jpeg_quality: int = 85,
        reconnect_delay_s: float = 2.0,
        stream_queue_size: int = 8,
    ):
        self.backend_name = backend_name
        self._hardware_factory = hardware_factory
        self._jpeg_quality = jpeg_quality
        self._reconnect_delay_s = reconnect_delay_s
        self._stream_queue_size = stream_queue_size

        self._source = None
        self._source_lock = threading.Lock()
        self._state = "starting"
        self._session_id: str | None = None
        self._fps = 15
        self._width: int | None = None
        self._height: int | None = None
        self._session_active = False
        self._frame_idx = 0
        self._last_frame_jpeg: bytes | None = None
        self._last_frame_idx: int | None = None
        self._last_capture_time_ms: int | None = None
        self._last_error_code: str | None = None
        self._last_error_message: str | None = None
        self._subscribers: dict[int, "queue.Queue[dict]"] = {}
        self._subscriber_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="camera-hw")
        self._next_stream_id = 0

    def start(self) -> None:
        self._ensure_connected()
        self._worker.start()

    def close(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=5)

    def status(self) -> dict:
        return {
            "state": self._state,
            "backend": self.backend_name,
            "camera_connected": self._source is not None,
            "session_active": self._session_active,
            "session_id": self._session_id,
            "fps": self._fps,
            "width": self._width,
            "height": self._height,
            "last_frame_idx": self._last_frame_idx,
            "last_capture_time_ms": self._last_capture_time_ms,
            "last_error_code": self._last_error_code,
            "last_error_message": self._last_error_message,
            "subscriber_count": len(self._subscribers),
        }

    def latest_frame(self) -> dict | None:
        if self._last_frame_jpeg is None:
            return None
        return {
            "jpeg": self._last_frame_jpeg,
            "frame_idx": self._last_frame_idx,
            "capture_time_ms": self._last_capture_time_ms,
            "session_id": self._session_id,
        }

    def register_stream(self) -> tuple[int, "queue.Queue[dict]"]:
        with self._subscriber_lock:
            stream_id = self._next_stream_id
            self._next_stream_id += 1
            stream_queue: "queue.Queue[dict]" = queue.Queue(maxsize=self._stream_queue_size)
            self._subscribers[stream_id] = stream_queue
            return stream_id, stream_queue

    def unregister_stream(self, stream_id: int) -> None:
        with self._subscriber_lock:
            self._subscribers.pop(stream_id, None)

    def start_session(self, session_id: str, fps=None, width=None, height=None) -> dict:
        session_id = session_id or str(uuid4())
        if self._session_active:
            raise ServiceError(
                "SESSION_ALREADY_ACTIVE",
                f"Session '{self._session_id}' is already active.",
                retryable=False,
                status_code=409,
            )

        if not self._ensure_connected():
            raise ServiceError(
                "CAMERA_NOT_READY",
                "The camera is not connected yet.",
                retryable=True,
                status_code=503,
            )

        self._fps = max(1, int(fps or self._fps))
        self._width = width
        self._height = height
        self._session_id = session_id
        self._session_active = True
        self._frame_idx = 0
        self._state = "running"
        self._wake_event.set()
        return self.status()

    def stop_session(self) -> dict:
        if not self._session_active:
            raise ServiceError(
                "NO_ACTIVE_SESSION",
                "There is no active session to stop.",
                retryable=False,
                status_code=409,
            )

        self._session_active = False
        self._session_id = None
        self._state = "ready" if self._source is not None else "degraded"
        self._wake_event.set()
        return self.status()

    def _set_error(self, code: str, message: str) -> None:
        self._last_error_code = code
        self._last_error_message = message
        self._state = "degraded"

    def _clear_error(self) -> None:
        self._last_error_code = None
        self._last_error_message = None

    def _ensure_connected(self) -> bool:
        with self._source_lock:
            if self._source is not None:
                return True

            try:
                source = self._hardware_factory()
                source.start()
                self._source = source
                self._state = "ready"
                self._clear_error()
                return True
            except ServiceError:
                raise
            except Exception as error:
                self._source = None
                self._set_error("CAMERA_NOT_FOUND", str(error))
                return False

    def _drop_connection(self, code: str, message: str) -> None:
        with self._source_lock:
            if self._source is not None:
                try:
                    self._source.stop()
                except Exception:
                    pass
            self._source = None
        self._set_error(code, message)

    def _emit_frame(self, image, capture_time_ms: int) -> None:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return

        jpeg_bytes = buf.tobytes()
        self._frame_idx += 1
        self._last_frame_jpeg = jpeg_bytes
        self._last_frame_idx = self._frame_idx
        self._last_capture_time_ms = capture_time_ms

        payload = {
            "session_id": self._session_id,
            "source": "camera-hw-service",
            "frame_idx": self._frame_idx,
            "capture_time_ms": capture_time_ms,
            "image_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
        }

        with self._subscriber_lock:
            queues = list(self._subscribers.values())

        for stream_queue in queues:
            try:
                stream_queue.put_nowait(payload)
            except queue.Full:
                try:
                    _ = stream_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    stream_queue.put_nowait(payload)
                except queue.Full:
                    continue

    def _run(self) -> None:
        next_read_at = time.perf_counter()

        while not self._stop_event.is_set():
            if not self._session_active:
                self._wake_event.wait(timeout=0.25)
                self._wake_event.clear()
                next_read_at = time.perf_counter()
                continue

            if not self._ensure_connected():
                time.sleep(self._reconnect_delay_s)
                continue

            interval_s = 1.0 / max(1, self._fps)

            try:
                with self._source_lock:
                    if self._source is None:
                        raise RuntimeError("camera disconnected")
                    frame_data = self._source.read()
            except Exception as error:
                self._drop_connection("READ_FAILED", str(error))
                time.sleep(self._reconnect_delay_s)
                continue

            if frame_data is None:
                self._drop_connection("READ_FAILED", "camera returned no frame")
                time.sleep(self._reconnect_delay_s)
                continue

            image, timestamp_ns, _frame_id = frame_data
            self._emit_frame(image, timestamp_ns // 1_000_000)

            next_read_at += interval_s
            sleep_s = next_read_at - time.perf_counter()
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)
            else:
                next_read_at = time.perf_counter()


def build_hardware_factory(backend_name: str, device, fps: int, width: int, height: int) -> HardwareFactory:
    backend = (backend_name or "real").strip().lower()

    if backend == "real":
        from camera_fusion.frame_source import DeviceCameraSource

        return lambda: DeviceCameraSource(device, fps, width, height)

    if backend == "fake":
        from .fake_source import FakeCameraSource

        return lambda: FakeCameraSource(fps, width, height)

    raise ValueError(f"Unsupported camera backend '{backend_name}'")
