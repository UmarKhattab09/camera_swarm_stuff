import sys
import types
import numpy as np


def _register_cv2_stub():
    """Install a lightweight cv2 stub so tests can run without OpenCV."""
    if "cv2" in sys.modules:
        return

    cv2_stub = types.ModuleType("cv2")
    cv2_stub.CAP_V4L2 = 0
    cv2_stub.CAP_PROP_FRAME_WIDTH = 3
    cv2_stub.CAP_PROP_FRAME_HEIGHT = 4
    cv2_stub.CAP_PROP_FPS = 5
    cv2_stub.CAP_PROP_FOURCC = 9
    cv2_stub.CV_16SC2 = 6
    cv2_stub.INTER_LINEAR = 1
    cv2_stub.COLOR_BGR2GRAY = 7
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.LINE_AA = 10

    class _FakeVideoCapture:
        def __init__(self, index, backend=None):
            self.index = index
            self.backend = backend
            self.props = {}
            self.read_return_value = (True, np.zeros((1, 1, 3), dtype=np.uint8))
            self.released = False

        def set(self, prop, value):
            self.props[prop] = value

        def read(self):
            return self.read_return_value

        def release(self):
            self.released = True

    cv2_stub.VideoCapture = _FakeVideoCapture
    cv2_stub.VideoWriter_fourcc = lambda *letters: 0
    cv2_stub.cvtColor = lambda image, code: image
    def _write_text(img, *args, **kwargs):
        return img
    cv2_stub.putText = _write_text
    cv2_stub.drawFrameAxes = lambda *args, **kwargs: None

    def _imwrite(path, image):
        from pathlib import Path
        data = getattr(image, "tobytes", lambda: b"")()
        Path(path).write_bytes(data if isinstance(data, (bytes, bytearray)) else b"img")
        return True

    cv2_stub.imwrite = _imwrite
    cv2_stub.getOptimalNewCameraMatrix = (
        lambda K, dist, size, alpha: (K, (0, 0, size[0], size[1]))
    )
    cv2_stub.initUndistortRectifyMap = (
        lambda K, dist, _rect, newK, size, code: (np.zeros(size), np.zeros(size))
    )
    cv2_stub.remap = lambda image, map1, map2, mode: image

    class _FakeNode:
        def __init__(self, data):
            self._data = data

        def mat(self):
            return np.array(self._data)

        def real(self):
            return float(self._data)

    class _FakeFileStorage:
        def __init__(self, path, mode):
            self.nodes = {
                "camera_matrix": np.eye(3),
                "dist_coeffs": np.zeros((5, 1)),
                "image_width": 640,
                "image_height": 480,
            }

        def getNode(self, key):
            return _FakeNode(self.nodes[key])

        def release(self):
            pass

    cv2_stub.FileStorage = _FakeFileStorage
    cv2_stub.FILE_STORAGE_READ = 0

    cv2_stub.IMWRITE_JPEG_QUALITY = 1
    cv2_stub.IMREAD_COLOR = 1

    def _imencode(ext, image, params=None):
        data = getattr(image, "tobytes", lambda: b"")()
        return True, np.frombuffer(data, dtype=np.uint8) if data else np.zeros((0,), dtype=np.uint8)

    def _imdecode(buf, flags):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    cv2_stub.imencode = _imencode
    cv2_stub.imdecode = _imdecode

    aruco_stub = types.ModuleType("cv2.aruco")
    aruco_stub.DICT_4X4_50 = 0
    aruco_stub.DICT_4X4_100 = 1
    aruco_stub.DICT_5X5_50 = 2
    aruco_stub.DICT_5X5_100 = 3
    aruco_stub.DICT_6X6_50 = 4
    aruco_stub.DICT_6X6_100 = 5
    aruco_stub.DICT_7X7_50 = 6
    aruco_stub.DICT_7X7_100 = 7

    def _dict_lookup(code):
        return f"dict:{code}"

    aruco_stub.getPredefinedDictionary = _dict_lookup
    aruco_stub.Dictionary_get = _dict_lookup
    aruco_stub.DetectorParameters_create = lambda: {"source": "create"}
    aruco_stub.DetectorParameters = lambda: {"source": "ctor"}
    aruco_stub.drawDetectedMarkers = lambda *args, **kwargs: None

    class _FakeArucoDetector:
        def __init__(self, dictionary, params):
            self.dictionary = dictionary
            self.params = params
            self.calls = []

        def detectMarkers(self, image):
            self.calls.append(image)
            return [], None, []

    aruco_stub.ArucoDetector = _FakeArucoDetector
    aruco_stub.detectMarkers = lambda image, dictionary, parameters=None: ([], None, [])

    def _estimate_pose(corners, marker_length, K, dist):
        count = len(corners)
        return (
            np.zeros((count, 1, 3)),
            np.zeros((count, 1, 3)),
            None,
        )

    aruco_stub.estimatePoseSingleMarkers = _estimate_pose
    cv2_stub.aruco = aruco_stub

    sys.modules["cv2"] = cv2_stub


def _register_facade_sdk_stub():
    """Stub out the facade_sdk client so CLI/system tests can import it."""
    if "facade_sdk" in sys.modules:
        return

    module = types.ModuleType("facade_sdk")

    class _DummyClient:
        def __init__(self, **_):
            self.published = []

        def publish(self, line):
            self.published.append(line)

        def close(self):
            pass

    module.Client = _DummyClient
    sys.modules["facade_sdk"] = module


_register_cv2_stub()
_register_facade_sdk_stub()
