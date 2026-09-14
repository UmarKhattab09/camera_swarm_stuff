from unittest.mock import MagicMock, patch

import numpy as np

from iiot_pipeline.ip_types import Frame, Detection
from iiot_pipeline.strategies import capture_usb as capture_mod
from iiot_pipeline.strategies import preprocess as preprocess_mod
from iiot_pipeline.strategies import undistort_simple as undistort_mod
from iiot_pipeline.strategies import detect_aruco as detect_mod
from iiot_pipeline.strategies import localize_pnp as localize_mod
from iiot_pipeline.strategies.capture_usb import USBWebcamCapture
from iiot_pipeline.strategies.preprocess import ColorFrame, GrayscaleFrame
from iiot_pipeline.strategies.undistort_simple import SimpleUndistort
from iiot_pipeline.strategies.detect_aruco import ArucoDetect
from iiot_pipeline.strategies.localize_pnp import PnPLocalize


@patch("iiot_pipeline.strategies.capture_usb.time.strftime", return_value="ts")
@patch("iiot_pipeline.strategies.capture_usb.cv2.VideoCapture")
def test_usb_capture_reads_frames(mock_cap_class, mock_strftime):
    """USB capture should configure the camera and yield Frame objects."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, "image")
    mock_cap_class.return_value = mock_cap

    capture = USBWebcamCapture(device_index=1, requested_fps=20, w=640, h=480)
    capture.start()
    frame = capture.next_frame()
    capture.stop()

    mock_cap_class.assert_called_once_with(1, capture_mod.cv2.CAP_V4L2)
    mock_cap.set.assert_any_call(capture_mod.cv2.CAP_PROP_FRAME_WIDTH, 640)
    mock_cap.set.assert_any_call(capture_mod.cv2.CAP_PROP_FRAME_HEIGHT, 480)
    mock_cap.set.assert_any_call(capture_mod.cv2.CAP_PROP_FPS, 20)
    assert frame.image == "image"
    assert frame.idx == 1
    assert frame.ts_iso == "ts"
    assert frame.capture_time is not None
    mock_cap.release.assert_called_once()


def test_color_frame_is_noop():
    """ColorFrame should leave the image untouched."""
    frame = Frame(1, "ts", np.zeros((1, 1, 3)))
    assert ColorFrame().apply(frame) is frame


@patch("iiot_pipeline.strategies.preprocess.cv2.cvtColor", return_value="gray")
def test_grayscale_frame_converts(mock_cvt):
    """GrayscaleFrame should call cv2.cvtColor with the correct flag."""
    frame = Frame(1, "ts", np.zeros((1, 1, 3)))
    out = GrayscaleFrame().apply(frame)
    assert out.image == "gray"
    mock_cvt.assert_called_once_with(frame.image, preprocess_mod.cv2.COLOR_BGR2GRAY)


@patch("iiot_pipeline.strategies.undistort_simple.load_calib")
def test_simple_undistort_calls_cv2(mock_load_calib):
    """SimpleUndistort must build the remap matrices and remap frames."""
    mock_load_calib.return_value = (
        np.eye(3),
        np.zeros((5, 1)),
        (640, 480),
    )
    frame = Frame(1, "ts", "img")

    with patch.object(
        undistort_mod.cv2, "getOptimalNewCameraMatrix", return_value=("newK", None)
    ) as mock_opt, patch.object(
        undistort_mod.cv2, "initUndistortRectifyMap", return_value=("map1", "map2")
    ) as mock_maps, patch.object(
        undistort_mod.cv2, "remap", return_value="undistorted"
    ) as mock_remap:
        und = SimpleUndistort("calib.yml")
        result = und.apply(frame)

    mock_opt.assert_called_once()
    mock_maps.assert_called_once()
    mock_remap.assert_called_once_with(
        frame.image, "map1", "map2", undistort_mod.cv2.INTER_LINEAR
    )
    assert result.image == "undistorted"


def test_aruco_detect_uses_new_detector():
    """When available, ArucoDetect should use the ArucoDetector API."""
    detector = ArucoDetect("4x4_50")
    fake_detector = MagicMock()
    fake_detector.detectMarkers.return_value = (
        [np.zeros((4, 2))],
        np.array([[42]], dtype=np.int32),
        [],
    )
    detector._detector = fake_detector
    frame = Frame(1, "ts", np.zeros((2, 2)))

    results = detector.detect(frame)
    assert len(results) == 1
    assert results[0].marker_id == 42
    fake_detector.detectMarkers.assert_called_once_with(frame.image)


@patch.object(detect_mod.cv2.aruco, "detectMarkers")
def test_aruco_detect_falls_back_to_module(mock_detect):
    """Older OpenCV code path should call cv2.aruco.detectMarkers."""
    mock_detect.return_value = (
        [np.zeros((4, 2))],
        np.array([[7]], dtype=np.int32),
        [],
    )
    detector = ArucoDetect("4x4_50")
    detector._detector = None
    frame = Frame(1, "ts", np.zeros((2, 2)))

    results = detector.detect(frame)
    assert len(results) == 1
    mock_detect.assert_called_once()


@patch.object(localize_mod.cv2.aruco, "estimatePoseSingleMarkers")
def test_pnp_localize_returns_pose(mock_estimate):
    """PnPLocalize should wrap cv2's pose estimate output into Pose objects."""
    mock_estimate.return_value = (
        np.array([[[1.0, 0.0, 0.0]]]),
        np.array([[[0.0, 0.0, 1.0]]]),
        None,
    )
    localizer = PnPLocalize("K", "dist", 0.1)
    det = Detection(1, np.zeros((4, 2)), None, None)

    poses = localizer.estimate([det])
    assert len(poses) == 1
    assert poses[0].tvec[0, 2] == 1.0
    mock_estimate.assert_called_once()


def test_pnp_localize_returns_empty_when_disabled():
    """No poses should be returned when marker length is non-positive."""
    localizer = PnPLocalize("K", "dist", 0.0)
    det = Detection(1, np.zeros((4, 2)), None, None)
    assert localizer.estimate([det]) == []
