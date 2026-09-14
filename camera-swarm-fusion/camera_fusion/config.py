from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import os
import socket


@dataclass
class SourceConfig:
    """Configuration for frame source (camera, RTP stream, etc.)."""
    
    type: str = "v4l2"  # "v4l2", "rtp_h264_udp"
    port: Optional[int] = None  # For RTP streams: UDP port
    
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LightGlueConfig:
    enabled: bool = False
    device: str = "cpu"  # "cpu", "cuda", or "cuda:0"
    template_dir: str = "templates/markers"
    min_inliers: int = 4
    max_age_frames: int = 5
    roi_expand_px: int = 50
    debug_save: bool = False
    corner_refine: bool = True
    match_threshold: float = 0.2
    debug_matches_dir: Optional[str] = None
    # Safety features
    verify_id: bool = True  # Verify ArUco ID after recovery
    max_fallback_markers_per_frame: int = 2  # Limit fallback attempts per frame
    reacquire_interval_frames: int = 5  # Min frames between reacquire attempts per marker
    prefer_roi_matching: bool = True  # Use ROI if last_corners exist

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraConfig:
    camera_name: str = "cam"
    device: int | str = 0
    fps: int = 15
    width: int = 1920
    height: int = 1080
    calibration_path: str = "calib/c920s_1920x1080_simple.yml"
    session_root: str = "data/sessions"
    duration_sec: float = 30.0
    aruco_dict: str = "4x4_50"
    marker_length_m: float = 0.035
    target_ids: Optional[list[int]] = None
    reference_id: Optional[int] = None
    marker_lengths_m: Optional[dict[int, float]] = None
    no_detect: bool = False
    dry_run: bool = False
    max_frames: Optional[int] = None
    save_annotated: bool = True
    save_frames: bool = True
    apply_undistort: bool = True
    publish: bool = False
    broker_ip: str = "192.168.1.76"
    broker_port: int = 1883
    device_id: str = "CameraPi"
    client_type: str = "CAMERA"
    expected_marker_count: int = 3
    metrics_enabled: bool = True
    metrics_flush_every_n_frames: int = 30
    metrics_primary_marker_strategy: str = "min_id"
    lightglue: Optional[LightGlueConfig] = None
    source: Optional[SourceConfig] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_overrides(self, **kwargs: Any) -> "CameraConfig":
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self


def _normalize_target_ids(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    if isinstance(value, (int, float)):
        return [int(value)]
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping")
    return data


def load_config(path: str | Path) -> CameraConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    if p.suffix.lower() in {".yaml", ".yml"}:
        raw = _load_yaml(p)
    else:
        with p.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a JSON/YAML object")

    cfg = CameraConfig()
    cfg.camera_name = str(raw.get("camera_name", cfg.camera_name))
    cfg.device = raw.get("device", cfg.device)
    cfg.fps = int(raw.get("fps", cfg.fps))
    cfg.width = int(raw.get("width", cfg.width))
    cfg.height = int(raw.get("height", cfg.height))
    cfg.calibration_path = str(raw.get("calibration_path", cfg.calibration_path))
    cfg.session_root = str(raw.get("session_root", cfg.session_root))
    cfg.duration_sec = float(raw.get("duration_sec", cfg.duration_sec))
    cfg.aruco_dict = str(raw.get("aruco_dict", cfg.aruco_dict))
    cfg.marker_length_m = float(raw.get("marker_length_m", cfg.marker_length_m))
    cfg.target_ids = _normalize_target_ids(raw.get("target_ids", cfg.target_ids))
    cfg.reference_id = raw.get("reference_id", cfg.reference_id)
    if cfg.reference_id is not None:
        cfg.reference_id = int(cfg.reference_id)
    lengths_raw = raw.get("marker_lengths_m", cfg.marker_lengths_m)
    if lengths_raw is not None:
        if not isinstance(lengths_raw, dict):
            raise ValueError("marker_lengths_m must be a mapping of marker_id -> length_m")
        cfg.marker_lengths_m = {int(k): float(v) for k, v in lengths_raw.items()}
    cfg.no_detect = bool(raw.get("no_detect", cfg.no_detect))
    cfg.dry_run = bool(raw.get("dry_run", cfg.dry_run))
    cfg.max_frames = raw.get("max_frames", cfg.max_frames)
    if cfg.max_frames is not None:
        cfg.max_frames = int(cfg.max_frames)
    cfg.save_annotated = bool(raw.get("save_annotated", cfg.save_annotated))
    cfg.save_frames = bool(raw.get("save_frames", cfg.save_frames))
    cfg.apply_undistort = bool(raw.get("apply_undistort", cfg.apply_undistort))
    cfg.publish = bool(raw.get("publish", cfg.publish))
    cfg.broker_ip = str(raw.get("broker_ip", cfg.broker_ip))
    cfg.broker_port = int(raw.get("broker_port", cfg.broker_port))
    env_device_id = os.getenv("DEVICE_ID")
    raw_device_id = raw.get("device_id")

    if env_device_id is not None:
        cfg.device_id = str(env_device_id)
    elif raw_device_id is not None:
        cfg.device_id = str(raw_device_id)
    else:
        cfg.device_id = str(cfg.device_id)
    cfg.client_type = str(raw.get("client_type", cfg.client_type))
    cfg.expected_marker_count = int(raw.get("expected_marker_count", cfg.expected_marker_count))
    cfg.metrics_enabled = bool(raw.get("metrics_enabled", cfg.metrics_enabled))
    cfg.metrics_flush_every_n_frames = int(
        raw.get("metrics_flush_every_n_frames", cfg.metrics_flush_every_n_frames)
    )
    cfg.metrics_primary_marker_strategy = str(
        raw.get("metrics_primary_marker_strategy", cfg.metrics_primary_marker_strategy)
    )
    
    # Load lightglue config if present
    lg_raw = raw.get("lightglue")
    if lg_raw is not None and isinstance(lg_raw, dict):
        lg_cfg = LightGlueConfig()
        lg_cfg.enabled = bool(lg_raw.get("enabled", lg_cfg.enabled))
        lg_cfg.device = str(lg_raw.get("device", lg_cfg.device))
        lg_cfg.template_dir = str(lg_raw.get("template_dir", lg_cfg.template_dir))
        lg_cfg.min_inliers = int(lg_raw.get("min_inliers", lg_cfg.min_inliers))
        lg_cfg.max_age_frames = int(lg_raw.get("max_age_frames", lg_cfg.max_age_frames))
        lg_cfg.roi_expand_px = int(lg_raw.get("roi_expand_px", lg_cfg.roi_expand_px))
        lg_cfg.debug_save = bool(lg_raw.get("debug_save", lg_cfg.debug_save))
        lg_cfg.corner_refine = bool(lg_raw.get("corner_refine", lg_cfg.corner_refine))
        lg_cfg.match_threshold = float(lg_raw.get("match_threshold", lg_cfg.match_threshold))
        debug_matches_dir = lg_raw.get("debug_matches_dir", lg_cfg.debug_matches_dir)
        lg_cfg.debug_matches_dir = str(debug_matches_dir) if debug_matches_dir is not None else None
        lg_cfg.verify_id = bool(lg_raw.get("verify_id", lg_cfg.verify_id))
        lg_cfg.max_fallback_markers_per_frame = int(lg_raw.get("max_fallback_markers_per_frame", lg_cfg.max_fallback_markers_per_frame))
        lg_cfg.reacquire_interval_frames = int(lg_raw.get("reacquire_interval_frames", lg_cfg.reacquire_interval_frames))
        lg_cfg.prefer_roi_matching = bool(lg_raw.get("prefer_roi_matching", lg_cfg.prefer_roi_matching))
        cfg.lightglue = lg_cfg
    
    # Load source config if present
    src_raw = raw.get("source")
    if src_raw is not None and isinstance(src_raw, dict):
        src_cfg = SourceConfig()
        src_cfg.type = str(src_raw.get("type", src_cfg.type))
        src_cfg.port = src_raw.get("port", src_cfg.port)
        if src_cfg.port is not None:
            src_cfg.port = int(src_cfg.port)
        cfg.source = src_cfg
    
    return cfg
