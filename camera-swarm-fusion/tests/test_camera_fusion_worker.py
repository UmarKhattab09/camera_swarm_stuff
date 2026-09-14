from pathlib import Path

from camera_fusion.config import CameraConfig
from camera_fusion.worker import CameraWorker


def test_worker_dry_run_creates_session(tmp_path: Path):
    cfg = CameraConfig(
        camera_name="drycam",
        session_root=str(tmp_path),
        dry_run=True,
        max_frames=2,
        save_frames=False,
        save_annotated=False,
        no_detect=True,
    )

    worker = CameraWorker(cfg)
    summary = worker.run()

    assert summary.frames_processed == 2
    assert Path(summary.session_path).exists()
    assert Path(summary.csv_path).exists()
    assert Path(summary.log_path).exists()
