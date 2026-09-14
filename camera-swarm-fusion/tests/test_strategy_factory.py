from unittest.mock import patch

from iiot_pipeline.config import RunConfig
from iiot_pipeline.factory import StrategyFactory


@patch("iiot_pipeline.factory.PnPLocalize")
@patch("iiot_pipeline.factory.ArucoDetect")
@patch("iiot_pipeline.factory.load_calib")
@patch("iiot_pipeline.factory.SimpleUndistort")
@patch("iiot_pipeline.factory.GrayscaleFrame")
@patch("iiot_pipeline.factory.ColorFrame")
@patch("iiot_pipeline.factory.USBWebcamCapture")
def test_strategy_factory_configures_components(
    mock_cap,
    mock_color,
    mock_gray,
    mock_und,
    mock_load_calib,
    mock_detect,
    mock_localize,
):
    """StrategyFactory should construct every strategy with correct arguments."""
    cfg = RunConfig(
        fps=24,
        grayscale=True,
        calibrationPath="calib.yml",
        device=2,
        arucoDict="5x5_50",
        markerLengthM=0.04,
    )
    cfg.width = 800
    cfg.height = 600
    mock_load_calib.return_value = ("K", "dist", (cfg.width, cfg.height))

    StrategyFactory.from_config(cfg)

    mock_cap.assert_called_once_with(
        device_index=cfg.device, requested_fps=cfg.fps, w=cfg.width, h=cfg.height
    )
    mock_gray.assert_called_once()
    mock_color.assert_not_called()
    mock_und.assert_called_once_with(cfg.calibrationPath)
    mock_load_calib.assert_called_once_with(cfg.calibrationPath)
    mock_detect.assert_called_once_with(cfg.arucoDict)
    mock_localize.assert_called_once_with("K", "dist", cfg.markerLengthM)


@patch("iiot_pipeline.factory.ColorFrame")
@patch("iiot_pipeline.factory.USBWebcamCapture")
def test_factory_falls_back_to_default_resolution(mock_cap, mock_color):
    """Factory should use default resolution when config lacks width/height."""
    cfg = RunConfig(grayscale=False)
    cap, pre, *_ = StrategyFactory.from_config(cfg)
    mock_cap.assert_called_once_with(
        device_index=cfg.device, requested_fps=cfg.fps, w=1920, h=1080
    )
    assert pre == mock_color.return_value
