from dataclasses import dataclass
from typing import Any

@dataclass
class Frame:
    idx: int
    ts_iso: str
    image: Any  # numpy array
    capture_time: float | None = None

@dataclass
class Detection:
    marker_id: int
    corners: Any  # (4,2) ndarray
    rvec: Any | None
    tvec: Any | None

@dataclass
class Pose:
    rvec: Any
    tvec: Any

