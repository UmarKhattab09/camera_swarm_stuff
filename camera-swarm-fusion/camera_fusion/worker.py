from __future__ import annotations

import time
import threading
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from iiot_pipeline.ip_types import Frame, Detection
from iiot_pipeline.services.calib import load_calib
from iiot_pipeline.services.storage import SessionStorage
from iiot_pipeline.strategies.preprocess import ColorFrame
from iiot_pipeline.strategies.undistort_simple import SimpleUndistort
from iiot_pipeline.strategies.localize_pnp import PnPLocalize

from .capture import BaseCapture, SyntheticCapture, USBOpenCVCapture
from .config import CameraConfig
from .detect import build_detector, detect_markers
from .frame_source import FrameSource, DeviceCameraSource, RTPStreamSource
from .logging_utils import add_file_handler, setup_logger
from .metrics_logger import CameraMetricsLogger
from .output import CsvOutput, MqttOutput, OutputSink
from .transforms import compute_relative_pose


@dataclass
class SessionSummary:
    session_path: str
    frames_processed: int
    csv_path: str
    log_path: str
    avg_fps: float
    errors: int


class NoUndistort:
    def apply(self, f: Frame) -> Frame:
        return f


class NoLocalize:
    def estimate(self, detections) -> list:
        return []


class CameraWorker:
    def __init__(
        self,
        config: CameraConfig,
        logger=None,
        outputs: Optional[list[OutputSink]] = None,
        capture: Optional[BaseCapture] = None,
        frame_source: Optional[FrameSource] = None,
    ):
        self.config = config
        self.logger = logger or setup_logger(config.camera_name)
        
        # Default output with reference support if reference_id is set
        if outputs is None:
            use_ref = config.reference_id is not None
            outputs = [CsvOutput(use_reference=use_ref)]
            if config.publish:
                outputs.append(
                    MqttOutput(
                        broker_ip=config.broker_ip,
                        broker_port=config.broker_port,
                        device_id=config.device_id,
                        client_type=config.client_type,
                        camera_name=config.camera_name,
                        use_reference=use_ref,
                        logger=self.logger,
                    )
                )
        
        self.outputs = outputs
        self.capture = capture  # Kept for backward compatibility
        self.frame_source = frame_source
        self._stop_event = threading.Event()

        if config.target_ids is None:
            self.target_ids = None
        else:
            self.target_ids = {int(t) for t in config.target_ids}

    def stop(self) -> None:
        self._stop_event.set()

    def _build_capture(self) -> BaseCapture:
        """DEPRECATED: Use _build_frame_source instead."""
        if self.capture is not None:
            return self.capture
        if self.config.dry_run:
            return SyntheticCapture(self.config.fps, self.config.width, self.config.height)
        return USBOpenCVCapture(
            self.config.device,
            self.config.fps,
            self.config.width,
            self.config.height,
        )

    def _build_frame_source(self) -> FrameSource:
        """Build a FrameSource for input frames."""
        if self.frame_source is not None:
            return self.frame_source
        
        # Determine source type from config
        source_config = self.config.source
        if source_config is None or source_config.type == "v4l2":
            # Default: local USB camera
            return DeviceCameraSource(
                self.config.device,
                self.config.fps,
                self.config.width,
                self.config.height,
            )
        elif source_config.type == "rtp_h264_udp":
            # RTP stream over UDP
            if source_config.port is None:
                raise ValueError("source.port must be set for rtp_h264_udp type")
            return RTPStreamSource(
                source_config.port,
                self.config.fps,
                self.config.width,
                self.config.height,
            )
        else:
            raise ValueError(f"Unsupported source type: {source_config.type}")

    def run(self) -> SessionSummary:
        storage = SessionStorage(self.config.session_root, name=f"{self.config.camera_name}_session")
        session_path = storage.begin()
        storage.write_manifest(self.config.as_dict())

        metrics_logger: Optional[CameraMetricsLogger] = None
        if self.config.metrics_enabled:
            metrics_logger = CameraMetricsLogger(
                storage.session_dir,
                flush_every_n_frames=self.config.metrics_flush_every_n_frames,
            )

        log_file = str(Path(storage.logs_dir) / "session.log")
        add_file_handler(self.logger, self.config.camera_name, log_file)

        for out in self.outputs:
            out.open(Path(storage.session_dir))

        frame_source = self._build_frame_source()
        pre = ColorFrame()

        if self.config.dry_run:
            und = NoUndistort()
            loc = NoLocalize()
        else:
            K, dist, _ = load_calib(self.config.calibration_path)
            loc = PnPLocalize(K, dist, self.config.marker_length_m)
            und = SimpleUndistort(self.config.calibration_path) if self.config.apply_undistort else NoUndistort()

        detector_state = build_detector(self.config.aruco_dict)
        
        # Initialize LightGlue fallback if enabled
        lightglue_fallback = None
        if self.config.lightglue is not None and self.config.lightglue.enabled:
            try:
                from .fallback import LightGlueFallback
                self.config.lightglue.debug_matches_dir = str(Path(storage.session_dir) / "debug_lightglue_matches")
                lightglue_fallback = LightGlueFallback(
                    self.config.lightglue,
                    self.config.aruco_dict
                )
                self.logger.info("LightGlue fallback initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LightGlue fallback: {e}")
                lightglue_fallback = None

        self.logger.info("session started: %s", session_path)
        self.logger.info("config: %s", self.config.as_dict())

        condition_label = os.getenv("CONDITION_LABEL", "default")
        system_mode = (
            "hybrid_aruco_lightglue"
            if (lightglue_fallback is not None and self.target_ids is not None)
            else "aruco_only"
        )

        frame_source.start()
        t0 = time.time()
        frames = 0
        errors = 0

        try:
            while True:
                frame_loop_start = time.perf_counter()

                if self._stop_event.is_set():
                    break
                if self.config.duration_sec and (time.time() - t0) >= self.config.duration_sec:
                    break
                if self.config.max_frames and frames >= self.config.max_frames:
                    break

                frame_data = frame_source.read()
                if frame_data is None:
                    errors += 1
                    continue
                
                frame_img, timestamp_ns, frame_id = frame_data
                capture_time = timestamp_ns / 1_000_000_000.0
                ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                raw_frame = Frame(frame_id, ts_iso, frame_img, capture_time=capture_time)

                f = pre.apply(raw_frame)
                if self.config.apply_undistort:
                    f = und.apply(f)

                dets = []
                poses = []
                pose_map = {}
                debug_info_list = []
                aruco_detect_ms = 0.0
                aruco_detected_ids = set()
                required_ids = set(self.target_ids) if self.target_ids is not None else set()
                missing_target_ids = set()
                fallback_stats = {
                    "fallback_triggered": False,
                    "fallback_success": False,
                    "raw_match_count": 0,
                    "ransac_inlier_count": 0,
                    "inlier_ratio": 0.0,
                    "superpoint_ms": 0.0,
                    "lightglue_ms": 0.0,
                    "ransac_ms": 0.0,
                    "fallback_total_ms": 0.0,
                    "failure_reason": None,
                }
                image_path = ""
                annotated_path = ""
                
                if not self.config.no_detect and not self.config.dry_run:
                    aruco_start = time.perf_counter()

                    # Step 1: Run ArUco detection
                    dets = detect_markers(f.image, detector_state)
                    aruco_detected_ids = {d.marker_id for d in dets}
                    
                    # Step 2: Build detected_dict for fallback
                    detected_dict = {d.marker_id: d.corners for d in dets}
                    missing_target_ids = required_ids - aruco_detected_ids
                    
                    # Step 3: Attempt LightGlue recovery for missing markers
                    if lightglue_fallback is not None and len(missing_target_ids) > 0:
                        detected_dict, debug_info_list = lightglue_fallback.recover_missing(
                            f.image,
                            detected_dict,
                            required_ids
                        )
                        fallback_stats = lightglue_fallback.get_last_frame_stats()
                        
                        # Rebuild dets list from updated detected_dict
                        dets = [
                            Detection(mid, corners, None, None)
                            for mid, corners in detected_dict.items()
                        ]
                    
                    # Step 4: Continue with existing pose estimation
                    if self.config.marker_lengths_m:
                        poses = loc.estimate_with_lengths(
                            dets,
                            self.config.marker_lengths_m,
                            default_length=self.config.marker_length_m,
                        )
                    else:
                        poses = loc.estimate(dets)

                    pose_map = {
                        d.marker_id: poses[i]
                        for i, d in enumerate(dets)
                        if i < len(poses) and poses[i] is not None
                    }

                    aruco_detect_ms = (time.perf_counter() - aruco_start) * 1000.0

                if dets and self.config.save_annotated:
                    draw = raw_frame.image.copy() if not self.config.apply_undistort else f.image.copy()
                    h, w = draw.shape[:2]
                    txt = f"#{f.idx} {f.ts_iso} {w}x{h}"
                    cv2.putText(
                        draw,
                        txt,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Build debug info lookup
                    debug_sources = {}
                    for info in debug_info_list:
                        debug_sources[info.marker_id] = info.source
                    
                    # Draw detected markers with color coding by source
                    for det in dets:
                        source = debug_sources.get(det.marker_id, "aruco")
                        
                        # Color code: green=aruco, cyan=lg_track, magenta=lg_reacquire
                        if source == "aruco":
                            color = (0, 255, 0)  # Green
                        elif source == "lg_track":
                            color = (255, 255, 0)  # Cyan
                        else:  # lg_reacquire
                            color = (255, 0, 255)  # Magenta
                        
                        # Draw marker corners
                        corners_int = det.corners.astype(np.int32)
                        cv2.polylines(draw, [corners_int], True, color, 2)
                        
                        # Draw marker ID
                        # Draw marker ID
                        center = corners_int.reshape(-1, 2).mean(axis=0)
                        org = (int(center[0]), int(center[1]))

                        label = f"{det.marker_id}"
                        if source != "aruco":
                            label += f" ({source})"

                        cv2.putText(
                            draw,
                            label,
                            org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                            cv2.LINE_AA,
                        )


                    # Draw pose axes
                    for det in dets:
                        pose = pose_map.get(det.marker_id)
                        if pose is None:
                            continue
                        try:
                            cv2.drawFrameAxes(
                                draw,
                                loc.K,
                                loc.dist,
                                pose.rvec,
                                pose.tvec,
                                max(0.01, self.config.marker_length_m * 0.5),
                            )
                        except Exception:
                            pass

                    annotated_path = storage.save_annotated(f.idx, draw)
                    
                    # Save debug frame if LightGlue debug enabled
                    if lightglue_fallback is not None and self.config.lightglue.debug_save:
                        debug_path = Path(storage.session_dir) / "debug_lightglue"
                        debug_path.mkdir(exist_ok=True)
                        debug_file = debug_path / f"frame_{f.idx:06d}.png"
                        cv2.imwrite(str(debug_file), draw)

                if self.config.save_frames:
                    storage.save_frame(raw_frame)
                    image_path = str(storage.last_path or "")

                # Compute reference-relative poses if reference marker is present
                ref_visible = False
                ref_pose_map = {}
                
                if self.config.reference_id is not None and self.config.reference_id in pose_map:
                    ref_pose = pose_map[self.config.reference_id]
                    if ref_pose is not None and ref_pose.rvec is not None and ref_pose.tvec is not None:
                        ref_visible = True
                    else:
                        ref_visible = False
                    
                    for marker_id, pose in pose_map.items():
                        if marker_id == self.config.reference_id:
                            continue
                        if pose is None or pose.rvec is None or pose.tvec is None:
                            continue
                        try:
                            ref_rvec, ref_tvec = compute_relative_pose(
                                ref_pose.rvec, ref_pose.tvec,
                                pose.rvec, pose.tvec
                            )
                            ref_pose_map[marker_id] = (ref_rvec, ref_tvec)
                        except Exception as e:
                            self.logger.warning(
                                "Failed to compute relative pose for marker %d: %s",
                                marker_id, e
                            )

                for det in dets:
                    if self.target_ids is not None and det.marker_id not in self.target_ids:
                        continue
                    pose = pose_map.get(det.marker_id)
                    rvec = pose.rvec if pose is not None else None
                    tvec = pose.tvec if pose is not None else None
                    ts_unix = time.time()

                    length_m = self.config.marker_length_m
                    if self.config.marker_lengths_m and det.marker_id in self.config.marker_lengths_m:
                        length_m = self.config.marker_lengths_m[det.marker_id]
                    
                    # Get reference-relative pose if available
                    ref_rvec, ref_tvec = None, None
                    if det.marker_id in ref_pose_map:
                        ref_rvec, ref_tvec = ref_pose_map[det.marker_id]

                    for out in self.outputs:
                        out.write_detection(
                            ts_unix,
                            f.idx,
                            det.marker_id,
                            rvec,
                            tvec,
                            storage.last_path,
                            ref_visible=ref_visible,
                            ref_rvec=ref_rvec,
                            ref_tvec=ref_tvec,
                            length_m=length_m,
                            capture_time=raw_frame.capture_time,
                        )

                self.logger.info(
                    "frame=%d saved=%s dets=%d",
                    f.idx,
                    storage.last_path,
                    len(dets),
                )

                total_frame_ms = (time.perf_counter() - frame_loop_start) * 1000.0
                if metrics_logger is not None:
                    final_detected_target_ids = {d.marker_id for d in dets if d.marker_id in required_ids}
                    missing_target_ids_final = required_ids - final_detected_target_ids

                    aruco_success = len(missing_target_ids) == 0
                    fallback_triggered = len(missing_target_ids) > 0
                    fallback_success = bool(fallback_stats.get("fallback_success", False))

                    detection_source = "none"
                    if fallback_success:
                        detection_source = "lightglue_assisted"
                    elif aruco_success:
                        detection_source = "aruco"

                    pose_recovered = any(
                        (mid in pose_map and pose_map[mid] is not None)
                        for mid in final_detected_target_ids
                    )

                    failure_reason = fallback_stats.get("failure_reason")
                    if aruco_success:
                        final_status = "aruco_success"
                    elif fallback_success:
                        final_status = "fallback_success"
                    elif failure_reason == "insufficient_inliers":
                        final_status = "fallback_triggered_insufficient_inliers"
                    elif fallback_triggered:
                        final_status = "fallback_triggered_no_match"
                    else:
                        final_status = "total_failure"

                    metrics_logger.log_frame(
                        {
                            "frame_id": f.idx,
                            "timestamp": f.ts_iso,
                            "camera_name": self.config.camera_name,
                            "expected_marker_count": len(required_ids),
                            "required_target_ids": sorted(list(required_ids)),
                            "detected_ids": sorted(list(final_detected_target_ids)),
                            "missing_target_ids": sorted(list(missing_target_ids_final)),
                            "condition_label": condition_label,
                            "system_mode": system_mode,
                            "aruco_success": aruco_success,
                            "fallback_triggered": fallback_triggered,
                            "fallback_success": fallback_success,
                            "detection_source": detection_source,
                            "raw_match_count": int(fallback_stats.get("raw_match_count", 0)),
                            "ransac_inlier_count": int(fallback_stats.get("ransac_inlier_count", 0)),
                            "inlier_ratio": float(fallback_stats.get("inlier_ratio", 0.0)),
                            "pose_recovered": pose_recovered,
                            "aruco_detect_ms": aruco_detect_ms,
                            "superpoint_ms": float(fallback_stats.get("superpoint_ms", 0.0)),
                            "lightglue_ms": float(fallback_stats.get("lightglue_ms", 0.0)),
                            "ransac_ms": float(fallback_stats.get("ransac_ms", 0.0)),
                            "fallback_total_ms": float(fallback_stats.get("fallback_total_ms", 0.0)),
                            "pipeline_total_ms": total_frame_ms,
                            "final_status": final_status,
                            "image_path": image_path,
                            "annotated_path": annotated_path,
                        }
                    )

                frames += 1

        finally:
            if metrics_logger is not None:
                try:
                    metrics_logger.finalize()
                except Exception:
                    self.logger.exception("failed to finalize metrics logger")

            try:
                frame_source.stop()
            except Exception:
                pass

            for out in self.outputs:
                try:
                    out.close()
                except Exception:
                    pass

        avg = frames / max(1e-6, (time.time() - t0))
        self.logger.info("summary frames=%d avg_fps=%.2f errors=%d", frames, avg, errors)

        csv_path = str(Path(storage.session_dir) / "detections.csv")
        return SessionSummary(
            str(session_path),
            frames,
            csv_path,
            log_file,
            avg,
            errors,
        )
