from .strategies.capture_usb import USBWebcamCapture
from .strategies.preprocess import ColorFrame, GrayscaleFrame
from .strategies.undistort_simple import SimpleUndistort
from .strategies.detect_aruco import ArucoDetect
from .strategies.localize_pnp import PnPLocalize
from .services.calib import load_calib

class StrategyFactory:
    @staticmethod
    def from_config(config):
        # Camera (use width/height from config if present; fall back to 1920x1080)
        cap = USBWebcamCapture(
            device_index=config.device,
            requested_fps=config.fps,
            w=getattr(config, "width", 1920),
            h=getattr(config, "height", 1080),
        )

        # Preprocess
        pre = GrayscaleFrame() if config.grayscale else ColorFrame()

        # Undistortion
        und = SimpleUndistort(config.calibrationPath)

        # Detection and localization
        K, dist, _ = load_calib(config.calibrationPath)
        det = ArucoDetect(config.arucoDict)
        loc = PnPLocalize(K, dist, config.markerLengthM)

        return cap, pre, und, det, loc

