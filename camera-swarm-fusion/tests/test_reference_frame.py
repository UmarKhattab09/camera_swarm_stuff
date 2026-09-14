import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from camera_fusion.config import CameraConfig
from camera_fusion.worker import CameraWorker
from iiot_pipeline.ip_types import Detection, Frame, Pose


class FakeCapture:
    """Minimal capture stub that returns a single black frame."""

    def __init__(self):
        self.idx = 0

    def start(self) -> None:
        return None

    def next_frame(self):
        self.idx += 1
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        ts = "2026-02-02T10:00:00"
        return Frame(self.idx, ts, img)

    def stop(self) -> None:
        return None


class NoOpUndistort:
    def __init__(self, *_args, **_kwargs):
        return None

    def apply(self, f: Frame) -> Frame:
        return f


def _read_csv_rows(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows = [line.strip().split(",") for line in f if line.strip()]
    return header, rows


def test_worker_with_reference_marker(tmp_path: Path):
    """
    Validate reference-frame math with simple translations.

    Camera frame:
        ref marker at t = [0, 0, 1]
        target marker at t = [1, 0, 1]

    Then:
        T_ref_cam = inverse(T_cam_ref)
        T_ref_target = T_ref_cam @ T_cam_target
        Expected target translation in ref frame: [1, 0, 0]
    """
    cfg = CameraConfig(
        camera_name="refcam",
        session_root=str(tmp_path),
        dry_run=False,
        max_frames=1,
        save_frames=False,
        save_annotated=False,
        reference_id=0,
        target_ids=[0, 1],
        marker_length_m=0.05,
    )

    worker = CameraWorker(cfg, capture=FakeCapture())

    mock_dets = [
        Detection(0, np.zeros((4, 2)), None, None),
        Detection(1, np.zeros((4, 2)), None, None),
    ]

    mock_poses = [
        Pose(np.zeros((3, 1)), np.array([[0.0], [0.0], [1.0]])),
        Pose(np.zeros((3, 1)), np.array([[1.0], [0.0], [1.0]])),
    ]

    with patch("camera_fusion.worker.SimpleUndistort", NoOpUndistort), patch(
        "camera_fusion.worker.load_calib",
        return_value=(np.eye(3), np.zeros((5, 1)), (640, 480)),
    ), patch("camera_fusion.detect.detect_markers", return_value=mock_dets), patch(
        "camera_fusion.worker.PnPLocalize.estimate", return_value=mock_poses
    ):
        summary = worker.run()

    csv_path = Path(summary.csv_path)
    header, rows = _read_csv_rows(csv_path)

    assert "ref_visible" in header
    ref_visible_idx = header.index("ref_visible")
    ref_tvec_x_idx = header.index("ref_tvec_x")
    ref_tvec_y_idx = header.index("ref_tvec_y")
    ref_tvec_z_idx = header.index("ref_tvec_z")
    marker_id_idx = header.index("marker_id")

    # Find the row for marker_id=1 (target marker)
    target_row = next(r for r in rows if r[marker_id_idx] == "1")

    # Expect reference visible and relative tvec = [1, 0, 0]
    assert target_row[ref_visible_idx] == "1"
    assert math.isclose(float(target_row[ref_tvec_x_idx]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(target_row[ref_tvec_y_idx]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(target_row[ref_tvec_z_idx]), 0.0, abs_tol=1e-6)


def test_worker_reference_marker_missing(tmp_path: Path):
    """
    If reference marker is NOT detected, we should still log detections.
    `ref_visible` should be 0 and reference-relative fields should be NaN.
    """
    cfg = CameraConfig(
        camera_name="missingrefcam",
        session_root=str(tmp_path),
        dry_run=False,
        max_frames=1,
        save_frames=False,
        save_annotated=False,
        reference_id=0,
        target_ids=[1],
        marker_length_m=0.05,
    )

    worker = CameraWorker(cfg, capture=FakeCapture())

    mock_dets = [
        Detection(1, np.zeros((4, 2)), None, None),
    ]

    mock_poses = [
        Pose(np.zeros((3, 1)), np.array([[0.5], [0.0], [1.0]])),
    ]

    with patch("camera_fusion.worker.SimpleUndistort", NoOpUndistort), patch(
        "camera_fusion.worker.load_calib",
        return_value=(np.eye(3), np.zeros((5, 1)), (640, 480)),
    ), patch("camera_fusion.detect.detect_markers", return_value=mock_dets), patch(
        "camera_fusion.worker.PnPLocalize.estimate", return_value=mock_poses
    ):
        summary = worker.run()

    csv_path = Path(summary.csv_path)
    header, rows = _read_csv_rows(csv_path)

    ref_visible_idx = header.index("ref_visible")
    ref_tvec_x_idx = header.index("ref_tvec_x")
    marker_id_idx = header.index("marker_id")

    target_row = next(r for r in rows if r[marker_id_idx] == "1")

    assert target_row[ref_visible_idx] == "0"
    assert math.isnan(float(target_row[ref_tvec_x_idx]))
