import time
import re
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from iiot_pipeline.ip_types import Frame


class BaseCapture(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def next_frame(self) -> Frame | None: ...

    @abstractmethod
    def stop(self) -> None: ...


class USBOpenCVCapture(BaseCapture):
    def __init__(self, device: int | str, fps: int, width: int, height: int):
        self.device = device
        self.fps = fps
        self.width = width
        self.height = height
        self.cap: Any = None
        self.idx = 0

    def start(self) -> None:
        if isinstance(self.device, int):
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            dev_str = str(self.device)
            match = re.match(r"^/dev/video(\d+)$", dev_str)
            if match:
                dev_idx = int(match.group(1))
                self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(dev_str)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.device}")

    def next_frame(self) -> Frame | None:
        ok, img = self.cap.read()
        if not ok:
            return None
        self.idx += 1
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        return Frame(self.idx, ts, img)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()


class SyntheticCapture(BaseCapture):
    def __init__(self, fps: int, width: int, height: int):
        self.fps = fps
        self.width = width
        self.height = height
        self.idx = 0
        self._last = 0.0

    def start(self) -> None:
        self._last = time.time()

    def next_frame(self) -> Frame | None:
        now = time.time()
        if self.fps > 0:
            wait = max(0.0, (1.0 / self.fps) - (now - self._last))
            if wait > 0:
                time.sleep(wait)
        self._last = time.time()
        self.idx += 1
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        return Frame(self.idx, ts, img)

    def stop(self) -> None:
        return None

