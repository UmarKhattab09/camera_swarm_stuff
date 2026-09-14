from pathlib import Path
import sys

import numpy as np
import pytest

from iiot_pipeline import cli
from iiot_pipeline.ip_types import Detection, Pose


@pytest.mark.system
def test_cli_system_run_creates_outputs(tmp_path, monkeypatch, capsys):
    """Run the CLI entrypoint to confirm sessions and output artifacts exist."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)

    args = [
        "prog",
        "--fps",
        "5",
        "--duration",
        "0.02",
        "--device",
        "1",
        "--width",
        "320",
        "--height",
        "240",
        "--out",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.delenv("PUBLISH", raising=False)

    cli.main()
    out = capsys.readouterr().out
    assert "SessionSummary" in out or "sessionPath" in out

    sessions = list(Path(tmp_path).glob("aruco_session_*"))
    assert sessions, "No session directory created by CLI run"
    session_dir = sessions[0]
    frames_dir = session_dir / "frames"
    logs_dir = session_dir / "logs"

    assert (session_dir / "detections.csv").exists()
    assert frames_dir.exists() and any(frames_dir.iterdir())
    assert (logs_dir / "session.log").exists()


def test_cli_system_run_with_publish(tmp_path, monkeypatch, capsys):
    """System test: enable --publish and ensure MQTT client receives payloads."""
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)

    class RecordingClient:
        CAMERA = "CAMERA"

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.lines = []
            RecordingClient.last_instance = self

        def publish(self, line):
            self.lines.append(line)

        def close(self):
            pass

    RecordingClient.last_instance = None
    monkeypatch.setattr("iiot_pipeline.cli.DataClient", RecordingClient)
    monkeypatch.setattr(
        "iiot_pipeline.strategies.detect_aruco.ArucoDetect.detect",
        lambda self, frame: [
            Detection(42, np.zeros((4, 2)), None, None),
        ],
    )
    monkeypatch.setattr(
        "iiot_pipeline.strategies.localize_pnp.PnPLocalize.estimate",
        lambda self, dets: [Pose(np.array([[0.1, 0.2, 0.3]]), np.array([[0.4, 0.5, 0.6]]))],
    )

    args = [
        "prog",
        "--fps",
        "5",
        "--duration",
        "0.02",
        "--device",
        "1",
        "--width",
        "320",
        "--height",
        "240",
        "--out",
        str(tmp_path),
        "--publish",
        "--dict",
        "4x4_50",
        "--marker-length-m",
        "0.05",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()

    out = capsys.readouterr().out
    assert "SessionSummary" in out

    client = RecordingClient.last_instance
    assert client is not None
    assert client.kwargs["client_type"] == RecordingClient.CAMERA
    assert client.lines, "Publisher should receive at least header+payload"

    sessions = list(Path(tmp_path).glob("aruco_session_*"))
    assert sessions, "Expected a session directory when publishing is enabled"
