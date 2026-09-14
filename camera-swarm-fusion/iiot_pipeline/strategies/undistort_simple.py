import cv2
from ..ip_types import Frame
from ..services.calib import load_calib

class SimpleUndistort:
    def __init__(self, calib_path: str, alpha: float = 0.0):
        K, dist, (w, h) = load_calib(calib_path)
        self.K, self.dist = K, dist
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w, h), cv2.CV_16SC2)

    def apply(self, f: Frame) -> Frame:
        und = cv2.remap(f.image, self.map1, self.map2, cv2.INTER_LINEAR)
        return Frame(f.idx, f.ts_iso, und)

