import json
from pathlib import Path

from camera_fusion.config import CameraConfig, load_config


def test_load_config_json(tmp_path: Path):
    cfg_path = tmp_path / "cam.json"
    cfg_path.write_text(
        json.dumps(
            {
                "camera_name": "camA",
                "device": 2,
                "fps": 20,
                "width": 640,
                "height": 480,
                "aruco_dict": "4x4_50",
                "marker_length_m": 0.05,
                "marker_lengths_m": {"0": 0.15, "1": 0.05},
                "target_ids": [1, 2],
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.camera_name == "camA"
    assert cfg.device == 2
    assert cfg.fps == 20
    assert cfg.width == 640
    assert cfg.height == 480
    assert cfg.target_ids == [1, 2]
    assert cfg.marker_lengths_m == {0: 0.15, 1: 0.05}

    cfg.apply_overrides(camera_name="camB", fps=10)
    assert cfg.camera_name == "camB"
    assert cfg.fps == 10


def test_config_defaults():
    cfg = CameraConfig()
    assert cfg.session_root
