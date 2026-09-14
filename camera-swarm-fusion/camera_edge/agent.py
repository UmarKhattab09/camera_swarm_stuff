from __future__ import annotations

import logging
from pathlib import Path
import signal
import time

import cv2
import numpy as np

from camera_fusion.detect import build_detector, detect_markers
from camera_fusion.output import CsvOutput, MqttOutput
from iiot_pipeline.services.calib import load_calib
from iiot_pipeline.services.storage import SessionStorage
from iiot_pipeline.strategies.localize_pnp import PnPLocalize

from .config import EdgeAgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("camera_edge")
from .health import EdgeHealthTracker
from .service_client import LocalCameraServiceClient, ServiceError


class EdgeAgent:
    """Swarm-managed detection/pose/MQTT agent.

    Never touches /dev/video* or cv2.VideoCapture -- it only talks to
    camera_host over the Unix socket, then reuses the existing ArUco
    detection (camera_fusion.detect) and PnP pose estimation
    (iiot_pipeline.strategies.localize_pnp.PnPLocalize) unchanged, and
    publishes through the existing CsvOutput/MqttOutput sinks so the
    detections.csv schema and MQTT payload/topic stay exactly what the
    middleware already expects.
    """

    def __init__(self, config: EdgeAgentConfig):
        self.config = config
        self.client = LocalCameraServiceClient(config.socket_path)
        self.health = EdgeHealthTracker(config.health_path)
        self._stop_requested = False

        K, dist, _ = load_calib(config.calibration_path)
        self._localize = PnPLocalize(K, dist, config.marker_length_m)
        self._detector_state = build_detector(config.aruco_dict)
        self._target_ids = set(config.target_ids) if config.target_ids else None

    def install_signal_handlers(self) -> None:
        def _handle_signal(_signum, _frame):
            self._stop_requested = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def run(self) -> None:
        self.install_signal_handlers()
        logger.info(
            "starting camera_edge camera_name=%s device_id=%s broker=%s:%s calibration=%s",
            self.config.camera_name, self.config.device_id,
            self.config.mqtt_broker_ip, self.config.mqtt_broker_port,
            self.config.calibration_path,
        )

        csv_output = CsvOutput()
        mqtt_output = MqttOutput(
            broker_ip=self.config.mqtt_broker_ip,
            broker_port=self.config.mqtt_broker_port,
            device_id=self.config.device_id,
            client_type=self.config.client_type,
            camera_name=self.config.camera_name,
        )

        storage = SessionStorage(self.config.session_root, name=f"{self.config.camera_name}_session")
        session_dir = Path(storage.begin())
        self._frames_dir = storage.frames_dir
        self._storage = storage
        logger.info("session directory: %s", session_dir)

        storage.write_manifest({
            "camera_name": self.config.camera_name,
            "device_id": self.config.device_id,
            "calibration_path": self.config.calibration_path,
            "marker_length_m": self.config.marker_length_m,
            "aruco_dict": self.config.aruco_dict,
            "fps": self.config.fps,
            "width": self.config.width,
            "height": self.config.height,
            "session_id": self.config.session_id,
        })

        csv_output.open(session_dir)
        mqtt_output.open(session_dir)

        try:
            while not self._stop_requested:
                try:
                    self._wait_until_ready()
                    if self.config.start_session_on_boot:
                        self._start_session()

                    if self.config.use_frame_stream:
                        self._consume_stream(csv_output, mqtt_output)
                    else:
                        self._poll_latest(csv_output, mqtt_output)
                except ServiceError as error:
                    logger.warning("service error: %s %s", error.code, error.message)
                    self.health.update(state="degraded", error_code=error.code, error_message=error.message)
                    time.sleep(self.config.reconnect_delay_s)
                except Exception as error:
                    logger.exception("unexpected failure in edge loop")
                    self.health.update(state="degraded", error_code="EDGE_FAILURE", error_message=str(error))
                    time.sleep(self.config.reconnect_delay_s)
        finally:
            csv_output.close()
            mqtt_output.close()

        if self.config.stop_session_on_exit:
            try:
                self.client.stop_session()
            except ServiceError:
                pass

    def _wait_until_ready(self) -> None:
        while not self._stop_requested:
            try:
                ready = self.client.readiness()
            except OSError as error:
                self.health.update(state="waiting", error_code="SERVICE_UNAVAILABLE", error_message=str(error))
                time.sleep(self.config.reconnect_delay_s)
                continue
            except ServiceError as error:
                self.health.update(state="waiting", error_code=error.code, error_message=error.message)
                time.sleep(self.config.reconnect_delay_s)
                continue

            if ready.get("ready"):
                return

            time.sleep(self.config.reconnect_delay_s)

    def _start_session(self) -> None:
        try:
            self.client.start_session(
                self.config.session_id, self.config.fps, self.config.width, self.config.height
            )
            logger.info("session started: %s", self.config.session_id)
        except ServiceError as error:
            if error.code != "SESSION_ALREADY_ACTIVE":
                raise
            logger.info("session already active: %s", self.config.session_id)

    def _consume_stream(self, csv_output: CsvOutput, mqtt_output: MqttOutput) -> None:
        with self.client.stream() as stream:
            for sample in stream:
                if self._stop_requested:
                    break
                self._process_frame(sample, csv_output, mqtt_output)

    def _poll_latest(self, csv_output: CsvOutput, mqtt_output: MqttOutput) -> None:
        last_frame_idx = None
        while not self._stop_requested:
            frame = self.client.frame_latest()
            if frame["frame_idx"] != last_frame_idx:
                last_frame_idx = frame["frame_idx"]
                sample = {
                    "jpeg": frame["jpeg"],
                    "frame_idx": frame["frame_idx"],
                    "capture_time_ms": frame["capture_time_ms"],
                }
                self._process_frame(sample, csv_output, mqtt_output)
            time.sleep(self.config.poll_interval_s)

    def _process_frame(self, sample: dict, csv_output: CsvOutput, mqtt_output: MqttOutput) -> None:
        image = cv2.imdecode(np.frombuffer(sample["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return

        dets = detect_markers(image, self._detector_state)
        if self.config.marker_lengths_m:
            poses = self._localize.estimate_with_lengths(
                dets, self.config.marker_lengths_m, default_length=self.config.marker_length_m
            )
        else:
            poses = self._localize.estimate(dets)

        pose_map = {
            d.marker_id: poses[i] for i, d in enumerate(dets) if i < len(poses) and poses[i] is not None
        }

        capture_time = sample["capture_time_ms"] / 1000.0
        ts_unix = time.time()

        image_path = None
        if self.config.save_frames:
            frame_filename = f"f{sample['frame_idx']:06d}.jpg"
            frame_path = self._frames_dir / frame_filename
            cv2.imwrite(str(frame_path), image)
            try:
                image_path = str(frame_path.relative_to("/app"))
            except ValueError:
                image_path = str(frame_path)

        if dets and self.config.save_annotated:
            draw = image.copy()
            h, w = draw.shape[:2]
            ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
            txt = f"#{sample['frame_idx']} {ts_iso} {w}x{h}"
            cv2.putText(draw, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            for det in dets:
                color = (0, 255, 0)  # green -- camera_edge has no LightGlue fallback, always aruco source
                corners_int = det.corners.astype(np.int32)
                cv2.polylines(draw, [corners_int], True, color, 2)
                center = corners_int.reshape(-1, 2).mean(axis=0)
                org = (int(center[0]), int(center[1]))
                cv2.putText(draw, str(det.marker_id), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            for det in dets:
                pose = pose_map.get(det.marker_id)
                if pose is None:
                    continue
                try:
                    cv2.drawFrameAxes(
                        draw, self._localize.K, self._localize.dist,
                        pose.rvec, pose.tvec,
                        max(0.01, self.config.marker_length_m * 0.5),
                    )
                except Exception:
                    logger.warning("drawFrameAxes failed for marker_id=%s", det.marker_id)

            self._storage.save_annotated(sample["frame_idx"], draw)

        for det in dets:
            if self._target_ids is not None and det.marker_id not in self._target_ids:
                continue

            pose = pose_map.get(det.marker_id)
            rvec = pose.rvec if pose is not None else None
            tvec = pose.tvec if pose is not None else None

            length_m = self.config.marker_length_m
            if self.config.marker_lengths_m and det.marker_id in self.config.marker_lengths_m:
                length_m = self.config.marker_lengths_m[det.marker_id]

            for out in (csv_output, mqtt_output):
                out.write_detection(
                    ts_unix,
                    sample["frame_idx"],
                    det.marker_id,
                    rvec,
                    tvec,
                    image_path,
                    length_m=length_m,
                    capture_time=capture_time,
                )

        self.health.update(
            state="streaming",
            session_id=self.config.session_id,
            last_capture_time_ms=sample["capture_time_ms"],
            last_frame_idx=sample["frame_idx"],
            detections=len(dets),
        )
        logger.info(
            "frame=%s dets=%s saved=%s",
            sample["frame_idx"], len(dets), image_path if image_path else "none",
        )
