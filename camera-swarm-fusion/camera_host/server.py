from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
import json
import os
import queue
import socketserver
from urllib.parse import urlparse

from .manager import ServiceError


class ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, socket_path: str, handler_cls, service):
        self.socket_path = socket_path
        self.service = service
        super().__init__(socket_path, handler_cls)

    def server_close(self):
        super().server_close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


class CameraRequestHandler(BaseHTTPRequestHandler):
    server: ThreadingUnixHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return None

    def do_GET(self):
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/v1/healthz":
                return self._send_json(HTTPStatus.OK, {"status": "ok"})
            if parsed.path == "/v1/readyz":
                status = self.server.service.status()
                ready = status["camera_connected"]
                return self._send_json(
                    HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                    {"ready": ready, "status": status},
                )
            if parsed.path == "/v1/status":
                return self._send_json(HTTPStatus.OK, {"status": self.server.service.status()})
            if parsed.path == "/v1/frame/latest":
                return self._send_latest_frame()
            if parsed.path == "/v1/frame/stream":
                return self._stream_frames()
            raise ServiceError(
                "NOT_FOUND",
                f"No route exists for '{parsed.path}'.",
                retryable=False,
                status_code=404,
            )
        except ServiceError as error:
            return self._send_error(error)

    def do_POST(self):
        parsed = urlparse(self.path)

        try:
            payload = self._read_json_body()
            if parsed.path == "/v1/session/start":
                session_id = str(payload.get("session_id") or "")
                if not session_id:
                    raise ServiceError(
                        "INVALID_REQUEST",
                        "session_id is required.",
                        retryable=False,
                        status_code=400,
                    )
                status = self.server.service.start_session(
                    session_id,
                    payload.get("fps"),
                    payload.get("width"),
                    payload.get("height"),
                )
                return self._send_json(HTTPStatus.OK, {"status": status})
            if parsed.path == "/v1/session/stop":
                status = self.server.service.stop_session()
                return self._send_json(HTTPStatus.OK, {"status": status})
            raise ServiceError(
                "NOT_FOUND",
                f"No route exists for '{parsed.path}'.",
                retryable=False,
                status_code=404,
            )
        except ServiceError as error:
            return self._send_error(error)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}

        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ServiceError(
                "INVALID_JSON",
                f"Request body could not be decoded: {error}",
                retryable=False,
                status_code=400,
            ) from error

    def _send_json(self, status_code, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, error: ServiceError) -> None:
        self._send_json(error.status_code, {"error": error.to_dict()})

    def _send_latest_frame(self) -> None:
        frame = self.server.service.latest_frame()
        if frame is None:
            raise ServiceError(
                "NO_FRAME_AVAILABLE",
                "No frame has been captured yet.",
                retryable=True,
                status_code=404,
            )

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame["jpeg"])))
        self.send_header("X-Frame-Idx", str(frame["frame_idx"]))
        self.send_header("X-Capture-Time-Ms", str(frame["capture_time_ms"]))
        if frame["session_id"]:
            self.send_header("X-Session-Id", str(frame["session_id"]))
        self.end_headers()
        self.wfile.write(frame["jpeg"])

    def _stream_frames(self) -> None:
        status = self.server.service.status()
        if not status["session_active"]:
            raise ServiceError(
                "NO_ACTIVE_SESSION",
                "No active camera session is running.",
                retryable=True,
                status_code=409,
            )

        stream_id, stream_queue = self.server.service.register_stream()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        try:
            while True:
                try:
                    sample = stream_queue.get(timeout=5)
                except queue.Empty:
                    status = self.server.service.status()
                    if not status["session_active"]:
                        break
                    continue

                line = json.dumps(sample).encode("utf-8") + b"\n"
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return None
        finally:
            self.server.service.unregister_stream(stream_id)


def run_server(socket_path: str, service) -> None:
    socket_dir = os.path.dirname(socket_path)
    if socket_dir:
        os.makedirs(socket_dir, exist_ok=True)
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    server = ThreadingUnixHTTPServer(socket_path, CameraRequestHandler, service)

    try:
        server.serve_forever()
    finally:
        server.server_close()
