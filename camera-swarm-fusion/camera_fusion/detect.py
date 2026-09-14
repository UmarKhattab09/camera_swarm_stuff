import cv2
from typing import Any, Tuple

from iiot_pipeline.ip_types import Detection
from iiot_pipeline.strategies.detect_aruco import get_dict as _get_dict
from iiot_pipeline.strategies.detect_aruco import _make_params as _make_params


DetectorState = Tuple[Any, Any, Any]


def build_detector(dict_name: str) -> DetectorState:
    name = (dict_name or "").strip()
    if name.upper().startswith("DICT_"):
        name = name[5:]
    name = name.lower()
    dictionary = _get_dict(name)
    params = _make_params()
    detector = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, params, detector


def detect_markers(image, detector_state: DetectorState) -> list[Detection]:
    dictionary, params, detector = detector_state
    if detector is not None:
        corners, ids, _rej = detector.detectMarkers(image)
    else:
        corners, ids, _rej = cv2.aruco.detectMarkers(
            image, dictionary, parameters=params
        )

    dets: list[Detection] = []
    if ids is not None and len(ids) > 0:
        for i, mid in enumerate(ids.flatten()):
            dets.append(Detection(int(mid), corners[i], None, None))
    return dets
