from pathlib import Path
from unittest.mock import patch
import logging
import numpy as np

from iiot_pipeline.config import RunConfig
from iiot_pipeline.facade import CameraPipelineFacade
from iiot_pipeline.ip_types import Frame, Detection, Pose


class DummyCapture:
    def __init__(self, frames):
        """Seed the fake capture object with predetermined frames."""
        self.frames = list(frames)
        self.started = False
        self.stopped = False

    def start(self):
        """Mark capture as started."""
        self.started = True

    def next_frame(self):
        """Return the next queued frame or None when exhausted."""
        if self.frames:
            return self.frames.pop(0)
        return None

    def stop(self):
        """Mark capture as stopped."""
        self.stopped = True


class DummyTransform:
    def apply(self, frame):
        """Return the frame unchanged (identity transform)."""
        return frame


class DummyDetector:
    def __init__(self, detections):
        """Initialize with a list of detection sequences to return."""
        self._dets = list(detections)

    def detect(self, frame):
        """Pop and return the next detection batch."""
        if self._dets:
            return self._dets.pop(0)
        return []


class DummyLocalizer:
    def __init__(self, pose_sequences):
        """Provide canned pose results for each detection batch."""
        self._poses = list(pose_sequences)
        self.K = np.eye(3)
        self.dist = np.zeros((5, 1))

    def estimate(self, detections):
        """Return the next pose batch, ignoring detections."""
        if self._poses:
            return self._poses.pop(0)
        return []


class DummyStorage:
    def __init__(self, root: Path):
        """Initialize storage rooted at a temporary directory."""
        self.root = root
        self.session_dir = None
        self.frames_dir = None
        self.annotated_dir = None
        self.logs_dir = None
        self.last_path = ""
        self.saved = []

    def begin(self):
        """Create the session directories expected by the facade."""
        self.session_dir = self.root / "session"
        self.frames_dir = self.session_dir / "frames"
        self.annotated_dir = self.session_dir / "annotated"
        self.logs_dir = self.session_dir / "logs"
        for folder in (self.frames_dir, self.annotated_dir, self.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)
        return str(self.session_dir)

    def write_manifest(self, meta):
        """Record the configuration metadata for assertions."""
        self.meta = meta

    def save_frame(self, frame):
        """Track saved frame paths for verification."""
        self.last_path = str(self.frames_dir / f"f{frame.idx:06d}.jpg")
        self.saved.append(("frame", frame.idx))

    def save_annotated(self, idx, image):
        """Track annotated frame paths for verification."""
        path = str(self.annotated_dir / f"f{idx:06d}_aruco.jpg")
        self.saved.append(("annotated", idx))
        return path


class DummyPublisher:
    def __init__(self):
        """Collect published lines so the test can assert on them."""
        self.lines = []

    def publish(self, msg):
        """Store the message line."""
        self.lines.append(msg)


class RecordingCsvWriter:
    HEADER = [
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
    instances = []

    def __init__(self, path):
        """Remember every writer instance so tests can inspect rows."""
        RecordingCsvWriter.instances.append(self)
        self.path = path
        self.rows = []
        self.opened = False
        self.closed = False

    def open(self):
        """Simulate opening the CSV file."""
        self.opened = True

    def append(self, *row):
        """Record appended rows for later inspection."""
        self.rows.append(row)

    def close(self):
        """Simulate closing the writer."""
        self.closed = True

    @classmethod
    def to_csv_line(cls, *values):
        """Return a deterministic CSV-esque payload for publisher assertions."""
        return "|".join(str(v) for v in values)


def test_facade_filters_targets_and_publishes_once(tmp_path):
    """Ensure target filtering, CSV writes, and publisher behavior work together."""
    RecordingCsvWriter.instances.clear()
    frames = [
        Frame(1, "ts1", np.zeros((4, 4, 3), dtype=np.uint8)),
        Frame(2, "ts2", np.zeros((4, 4, 3), dtype=np.uint8)),
    ]
    detections = [
        [
            Detection(5, np.zeros((4, 2)), None, None),
            Detection(7, np.zeros((4, 2)), None, None),
        ],
        [],
    ]
    poses = [
        [
            Pose(np.array([[1, 0, 0]]), np.array([[0, 0, 1]])),
            Pose(np.array([[0, 1, 0]]), np.array([[0, 1, 0]])),
        ],
        [],
    ]

    capture = DummyCapture(frames)
    pre = DummyTransform()
    und = DummyTransform()
    detector = DummyDetector(detections)
    localizer = DummyLocalizer(poses)
    storage = DummyStorage(tmp_path)
    publisher = DummyPublisher()

    cfg = RunConfig(durationSec=0.02, markerLengthM=0.1, targetIds=[7])
    logger = logging.Logger("test-facade")
    logger.addHandler(logging.NullHandler())

    time_values = iter([0.0, 0.001, 0.002, 0.003, 0.012, 0.021, 0.03])

    def fake_time():
        try:
            return next(time_values)
        except StopIteration:
            return 0.03

    facade = CameraPipelineFacade(
        capture,
        pre,
        und,
        detector,
        localizer,
        storage,
        logger,
        publisher=publisher,
        target_ids=[7],
    )

    with patch("iiot_pipeline.facade.CsvWriter", RecordingCsvWriter), patch(
        "iiot_pipeline.facade.time.time", side_effect=fake_time
    ):
        summary = facade.run_session(cfg)

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    assert capture.started is True
    assert capture.stopped is True
    assert summary.framesProcessed == 2
    assert summary.errors >= 0

    writer = RecordingCsvWriter.instances[0]
    assert len(writer.rows) == 1
    assert writer.rows[0][2] == 7
    assert ("annotated", 1) in storage.saved
    assert publisher.lines[0] == ",".join(RecordingCsvWriter.HEADER)
    assert len(publisher.lines) == 2  # header + one record
