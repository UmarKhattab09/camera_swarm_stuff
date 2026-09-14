import csv
import json
import logging
from pathlib import Path

import numpy as np

from iiot_pipeline.config import RunConfig
from iiot_pipeline.factory import StrategyFactory
from iiot_pipeline.facade import CameraPipelineFacade
from iiot_pipeline.ip_types import Frame
from iiot_pipeline.services.storage import SessionStorage
from iiot_pipeline.strategies import localize_pnp as localize_mod


class FakeCapture:
    def __init__(self, frames):
        """Queue deterministic frames to emulate a camera."""
        self.frames = list(frames)
        self.started = False
        self.stopped = False

    def start(self):
        """Mark the fake capture as started."""
        self.started = True

    def next_frame(self):
        """Return the next frame or None when depleted."""
        if self.frames:
            return self.frames.pop(0)
        return None

    def stop(self):
        """Mark the fake capture as stopped."""
        self.stopped = True


class RecordingPublisher:
    def __init__(self):
        """Collect published lines for assertions."""
        self.lines = []

    def publish(self, line):
        """Store published CSV payloads."""
        self.lines.append(line)


def test_pipeline_integration_full_flow(tmp_path, monkeypatch):
    """Run the entire pipeline stack and validate outputs & publishing."""
    cfg = RunConfig(
        fps=10,
        sessionRoot=str(tmp_path),
        durationSec=0.05,
        calibrationPath="calib/c920s_1920x1080_simple.yml",
        markerLengthM=0.12,
        device=1,
        width=640,
        height=480,
    )

    _cap, pre, und, det, loc = StrategyFactory.from_config(cfg)

    frames = [
        Frame(1, "ts_1", np.ones((4, 4, 3), dtype=np.uint8) * 10),
        Frame(2, "ts_2", np.ones((4, 4, 3), dtype=np.uint8) * 20),
    ]
    capture = FakeCapture(frames)
    publisher = RecordingPublisher()
    storage = SessionStorage(cfg.sessionRoot, name="integration_session")

    logger = logging.getLogger("integration-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    corners = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=np.float32)
    ids = np.array([[25]], dtype=np.int32)
    detect_seq = [
        (corners.copy(), ids.copy(), []),
        ([], None, []),
    ]

    fake_detector_cls = type(det._detector)

    def detect_markers(self, image):
        if detect_seq:
            c, i, rej = detect_seq.pop(0)
            return list(c), i, rej
        return [], None, []

    monkeypatch.setattr(fake_detector_cls, "detectMarkers", detect_markers, raising=False)

    def fake_pose(corners_in, marker_length, K, dist):
        rvec = np.array([[[0.1, 0.2, 0.3]]])
        tvec = np.array([[[0.4, 0.5, 0.6]]])
        return rvec, tvec, None

    monkeypatch.setattr(localize_mod.cv2.aruco, "estimatePoseSingleMarkers", fake_pose)

    times = iter([0.0, 0.01, 0.02, 0.03, 0.06, 0.07])

    def fake_time():
        try:
            return next(times)
        except StopIteration:
            return 0.07

    monkeypatch.setattr("iiot_pipeline.facade.time.time", fake_time)

    facade = CameraPipelineFacade(
        capture, pre, und, det, loc,
        storage, logger,
        publisher=publisher,
    )

    summary = facade.run_session(cfg)

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    session_dir = Path(summary.sessionPath)
    manifest = json.loads(session_dir.joinpath("config.json").read_text())
    assert manifest["fps"] == cfg.fps
    assert manifest["durationSec"] == cfg.durationSec

    frames_dir = session_dir / "frames"
    annotated_dir = session_dir / "annotated"
    logs_dir = session_dir / "logs"
    assert summary.framesProcessed == 2
    assert frames_dir.exists()
    assert len(list(frames_dir.glob("*.jpg"))) == 2
    assert annotated_dir.exists()
    assert len(list(annotated_dir.glob("*_aruco.jpg"))) == 1
    assert logs_dir.joinpath("session.log").exists()

    csv_path = Path(summary.csvPath)
    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "frame_idx",
        "recorded_at",
        "marker_id",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "tvec_x",
        "tvec_y",
        "tvec_z",
        "image_path",
    ]
    assert rows[1][0] == "1"
    assert rows[1][1] == "0.020000"
    assert rows[1][2] == "25"
    assert rows[1][3:6] == ["0.1", "0.2", "0.3"]
    assert rows[1][6:9] == ["0.5", "0.6", "0.4"]

    assert publisher.lines[0] == ",".join(rows[0])
    assert publisher.lines[1] == ",".join(rows[1])
