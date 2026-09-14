from __future__ import annotations

import time

import numpy as np


class FakeCameraSource:
    """In-process synthetic frame source.

    Used by the "fake" backend (tests, and dev-machine dry runs when no
    physical camera is attached) so camera_host's session/reconnect logic
    can be exercised without real hardware -- mirrors imu_host's FakeIMU.
    """

    def __init__(self, fps: int, width: int, height: int):
        self.fps = fps
        self.width = width or 640
        self.height = height or 480
        self._frame_id = 0
        self._last = 0.0

    def start(self) -> None:
        self._last = time.time()
        self._frame_id = 0

    def read(self) -> tuple[np.ndarray, int, int]:
        now = time.time()
        if self.fps > 0:
            wait = max(0.0, (1.0 / self.fps) - (now - self._last))
            if wait > 0:
                time.sleep(wait)
        self._last = time.time()
        self._frame_id += 1
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        timestamp_ns = int(time.time() * 1_000_000_000)
        return image, timestamp_ns, self._frame_id

    def stop(self) -> None:
        return None
