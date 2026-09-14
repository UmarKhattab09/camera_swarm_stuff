from __future__ import annotations

from contextlib import contextmanager
import base64
import http.client
import json
import socket
from typing import Generator, Iterable


class ServiceError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 10.0):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not None:
            self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def _raise_for_error(status: int, raw: bytes) -> None:
    decoded = json.loads(raw.decode("utf-8")) if raw else {}
    error_payload = decoded.get("error", {})
    raise ServiceError(
        error_payload.get("code", "HTTP_ERROR"),
        error_payload.get("message", f"HTTP {status}"),
        bool(error_payload.get("retryable", False)),
        status,
    )


class LocalCameraServiceClient:
    """Talks to camera_host over its Unix-socket HTTP API.

    Mirrors imu_edge.service_client.LocalIMUServiceClient, adapted for the
    binary /v1/frame/latest response and NDJSON+base64 /v1/frame/stream.
    """

    def __init__(self, socket_path: str, timeout: float = 10.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def _connection(self) -> UnixHTTPConnection:
        return UnixHTTPConnection(self.socket_path, timeout=self.timeout)

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload)
            headers["Content-Type"] = "application/json"

        conn = self._connection()
        try:
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            raw = response.read()
        finally:
            conn.close()

        if response.status >= 400:
            _raise_for_error(response.status, raw)

        return json.loads(raw.decode("utf-8")) if raw else {}

    def health(self) -> dict:
        return self._request_json("GET", "/v1/healthz")

    def readiness(self) -> dict:
        return self._request_json("GET", "/v1/readyz")

    def status(self) -> dict:
        return self._request_json("GET", "/v1/status")

    def start_session(self, session_id: str, fps: int, width: int, height: int) -> dict:
        return self._request_json(
            "POST",
            "/v1/session/start",
            {"session_id": session_id, "fps": fps, "width": width, "height": height},
        )

    def stop_session(self) -> dict:
        return self._request_json("POST", "/v1/session/stop", {})

    def frame_latest(self) -> dict:
        conn = self._connection()
        try:
            conn.request("GET", "/v1/frame/latest")
            response = conn.getresponse()
            raw = response.read()
            if response.status >= 400:
                _raise_for_error(response.status, raw)

            return {
                "jpeg": raw,
                "frame_idx": int(response.getheader("X-Frame-Idx", "0")),
                "capture_time_ms": int(response.getheader("X-Capture-Time-Ms", "0")),
            }
        finally:
            conn.close()

    @contextmanager
    def stream(self) -> Generator[Iterable[dict], None, None]:
        conn = self._connection()
        conn.request("GET", "/v1/frame/stream")
        response = conn.getresponse()
        if response.status >= 400:
            raw = response.read()
            conn.close()
            _raise_for_error(response.status, raw)

        try:
            yield self._iter_stream(response)
        finally:
            response.close()
            conn.close()

    def _iter_stream(self, response) -> Generator[dict, None, None]:
        while True:
            line = response.fp.readline()
            if not line:
                break

            payload = json.loads(line.decode("utf-8"))
            payload["jpeg"] = base64.b64decode(payload.pop("image_b64"))
            yield payload
