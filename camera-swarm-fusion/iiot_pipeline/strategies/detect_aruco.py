import cv2
from ..ip_types import Frame, Detection

def get_dict(name: str):
    """
    ArUco-only dictionary resolver (no AprilTag).
    Falls back to 4x4_50 if name not recognized.
    Works on OpenCV 4.12 (getPredefinedDictionary) and older (Dictionary_get).
    """
    key = (name or "").strip().lower()
    table = {
        "4x4_50":  cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_50":  cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_50":  cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
        "7x7_50":  cv2.aruco.DICT_7X7_50,
        "7x7_100": cv2.aruco.DICT_7X7_100,
    }
    code = table.get(key, cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "getPredefinedDictionary"):           # OpenCV >= 4.7
        return cv2.aruco.getPredefinedDictionary(code)
    elif hasattr(cv2.aruco, "Dictionary_get"):                   # Older OpenCV
        return cv2.aruco.Dictionary_get(code)
    else:                                                        # Very old fallback
        return cv2.aruco.Dictionary(code)

def _make_params():
    """Create detector parameters across OpenCV versions."""
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()

class ArucoDetect:
    """
    Strategy: detect ArUco markers in a frame.
    Returns a list[Detection] with (marker_id, corners, rvec=None, tvec=None).
    Pose is estimated later by the Localize strategy.
    """
    def __init__(self, dict_name: str = "4x4_50"):
        self.dictionary = get_dict(dict_name)
        self.params = _make_params()
        self._detector = None
        # Prefer the newer ArucoDetector API if present
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, f: Frame) -> list[Detection]:
        if self._detector is not None:
            corners, ids, _rej = self._detector.detectMarkers(f.image)
        else:
            corners, ids, _rej = cv2.aruco.detectMarkers(
                f.image, self.dictionary, parameters=self.params
            )

        dets: list[Detection] = []
        if ids is not None and len(ids) > 0:
            for i, mid in enumerate(ids.flatten()):
                dets.append(Detection(int(mid), corners[i], None, None))
        return dets
