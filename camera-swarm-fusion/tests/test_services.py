import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from iiot_pipeline.ip_types import Frame
from iiot_pipeline.services.storage import SessionStorage
from iiot_pipeline.services.csv_writer import CsvWriter
from iiot_pipeline.services.calib import load_calib


def test_session_storage_creates_dirs_and_manifest(tmp_path):
    """SessionStorage should create directories, save frames, and emit config."""
    storage = SessionStorage(tmp_path, name="demo")
    session_dir = Path(storage.begin())
    assert (session_dir / "frames").exists()
    assert (session_dir / "annotated").exists()
    assert (session_dir / "logs").exists()

    frame = Frame(1, "ts", np.zeros((2, 2, 3), dtype=np.uint8))
    storage.save_frame(frame)
    assert storage.last_path.endswith("f000001.jpg")
    annotated_path = storage.save_annotated(frame.idx, frame.image)
    assert annotated_path.endswith("_aruco.jpg")

    storage.write_manifest({"name": "demo"})
    manifest = json.loads((session_dir / "config.json").read_text())
    assert manifest["name"] == "demo"


def test_csv_writer_persists_rows_and_formats_lines(tmp_path):
    """CsvWriter should emit proper rows and CSV line formatting."""
    csv_path = tmp_path / "detections.csv"
    writer = CsvWriter(str(csv_path))
    writer.open()
    writer.append(1.234567, 1, 2, [1, 2, 3], [4, 5, 6], "/tmp/img.jpg")
    writer.append(2.0, 3, 4, None, [7], "/tmp/other.jpg")
    writer.close()

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("frame_idx")
    assert lines[1].startswith("1,nan,1.234567")
    assert lines[2].split(",")[-1] == "/tmp/other.jpg"

    inline = CsvWriter.to_csv_line(3.0, 5, 9, None, None, "img")
    assert inline.startswith("5,nan,3.000000")


def test_load_calib_reads_expected_nodes():
    """load_calib must pull the expected nodes from cv2.FileStorage."""
    fs = MagicMock()
    matrix_node = MagicMock()
    matrix_node.mat.return_value = "K"
    dist_node = MagicMock()
    dist_node.mat.return_value = "D"
    width_node = MagicMock()
    width_node.real.return_value = 1280
    height_node = MagicMock()
    height_node.real.return_value = 720

    fs.getNode.side_effect = [matrix_node, dist_node, width_node, height_node]

    with patch("iiot_pipeline.services.calib.cv2.FileStorage", return_value=fs):
        K, dist, size = load_calib("calib.yml")

    assert K == "K"
    assert dist == "D"
    assert size == (1280, 720)
    fs.release.assert_called_once()
