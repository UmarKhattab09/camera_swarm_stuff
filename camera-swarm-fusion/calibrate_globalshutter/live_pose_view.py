#!/usr/bin/env python3
"""
Live camera view with ArUco marker pose overlaid (distance + rotation angle),
streamed as MJPEG over HTTP so it can be watched from a headless Pi.

Usage (on the Pi):

  python3 live_pose_view.py --device 0 --width 1920 --height 1080 \
      --calib calib/gs_front_1920x1080.yml --marker-length-m 0.035 \
      --dict 4x4_50

Then on your desktop:
  - Browser: http://<PI_IP>:8091/stream.mjpg
  - VLC: Media -> Open Network Stream -> http://<PI_IP>:8091/stream.mjpg

Overlay shows, per detected marker:
  - 3D axes drawn at the marker (drawFrameAxes)
  - Distance from camera to marker, in meters (norm of tvec)
  - Total rotation angle from camera-facing orientation, in degrees
    (angle-magnitude of rvec, via Rodrigues)
"""
import argparse
import http.server
import socketserver
import threading
import time

import cv2
import numpy as np

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}

latest_jpeg = None
latest_lock = threading.Lock()


def load_calib(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    if K is None or dist is None:
        raise SystemExit(f"Could not read camera_matrix/dist_coeffs from {path}")
    return K, dist


class MjpegHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream.mjpg":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                with latest_lock:
                    jpeg = latest_jpeg
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        pass  # quiet


def run_server(port):
    handler = MjpegHandler
    with socketserver.ThreadingTCPServer(("0.0.0.0", port), handler) as httpd:
        httpd.serve_forever()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--calib", required=True)
    p.add_argument("--marker-length-m", type=float, default=0.035)
    p.add_argument("--dict", default="4x4_50", choices=list(ARUCO_DICTS.keys()))
    p.add_argument("--port", type=int, default=8091)
    args = p.parse_args()

    K, dist = load_calib(args.calib)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera device {args.device}")

    server_thread = threading.Thread(target=run_server, args=(args.port,), daemon=True)
    server_thread.start()
    print(f"Streaming at http://0.0.0.0:{args.port}/stream.mjpg -- Ctrl+C to stop")

    global latest_jpeg
    half_len = args.marker_length_m / 2.0
    obj_points = np.array([
        [-half_len,  half_len, 0],
        [ half_len,  half_len, 0],
        [ half_len, -half_len, 0],
        [-half_len, -half_len, 0],
    ], dtype=np.float32)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            corners, ids, _ = detector.detectMarkers(frame)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, marker_id in enumerate(ids.flatten()):
                    ok_pnp, rvec, tvec = cv2.solvePnP(
                        obj_points, corners[i][0], K, dist
                    )
                    if not ok_pnp:
                        continue

                    cv2.drawFrameAxes(frame, K, dist, rvec, tvec, args.marker_length_m * 0.75)

                    distance_m = float(np.linalg.norm(tvec))
                    angle_deg = float(np.degrees(np.linalg.norm(rvec)))

                    corner_pt = tuple(corners[i][0][0].astype(int))
                    text = f"id={marker_id} dist={distance_m:.3f}m angle={angle_deg:.1f}deg"
                    cv2.putText(frame, text, (corner_pt[0], corner_pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "no marker detected", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                with latest_lock:
                    latest_jpeg = buf.tobytes()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
