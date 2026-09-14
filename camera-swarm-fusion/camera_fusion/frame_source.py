"""Frame source abstraction for camera input.

Provides a unified interface for different frame sources:
- Device cameras (USB via V4L2)
- RTP streams (video over network)
- Synthetic test frames
"""

from __future__ import annotations

import time
import re
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class FrameSource(ABC):
    """Abstract base class for frame sources.
    
    A frame source provides frames from various inputs: USB cameras, RTP streams, etc.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the frame source. Called before any read() calls."""
        ...

    @abstractmethod
    def read(self) -> tuple[np.ndarray, int, int] | None:
        """Read next frame.
        
        Returns:
            (frame_bgr, timestamp_ns, frame_id) tuple or None if no frame available.
            - frame_bgr: BGR image as ndarray
            - timestamp_ns: timestamp in nanoseconds since epoch
            - frame_id: sequential frame number (1-indexed)
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the frame source and release resources."""
        ...


class DeviceCameraSource(FrameSource):
    """USB camera source using OpenCV's V4L2 interface.
    
    Wraps cv2.VideoCapture to provide a unified FrameSource interface.
    """

    def __init__(self, device: int | str, fps: int, width: int, height: int):
        self.device = device
        self.fps = fps
        self.width = width
        self.height = height
        self.cap: Any = None
        self.frame_id = 0

    def start(self) -> None:
        """Open the camera device."""
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
        
        self.frame_id = 0

    def read(self) -> tuple[np.ndarray, int, int] | None:
        """Read next frame from camera."""
        if self.cap is None:
            return None
        
        ok, img = self.cap.read()
        if not ok:
            return None

        capture_time = time.time()
        
        self.frame_id += 1
        timestamp_ns = int(capture_time * 1_000_000_000)
        
        return (img, timestamp_ns, self.frame_id)

    def stop(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RTPStreamSource(FrameSource):
    """RTP H.264 stream source using OpenCV's GStreamer backend.
    
    Receives H.264 encoded video over RTP/UDP and decodes to BGR frames.
    """

    def __init__(self, port: int, fps: int, width: int, height: int):
        """Initialize RTP stream source.
        
        Args:
            port: UDP port to listen on for RTP packets
            fps: Expected frames per second (informational)
            width: Expected frame width (informational)
            height: Expected frame height (informational)
        """
        self.port = port
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_id = 0
        self.cap: Any = None
        self._shape_logged = False

    def start(self) -> None:
        """Open the RTP stream via GStreamer backend."""
        # Build GStreamer pipeline
        gst_pipeline = (
            f"udpsrc port={self.port} "
            f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000,packetization-mode=1,width={self.width},height={self.height},framerate={self.fps}/1" ! '
            f"rtph264depay ! "
            f"h264parse ! "
            f"avdec_h264 ! "
            f"video/x-raw,format=I420,width={self.width},height={self.height},pixel-aspect-ratio=1/1 ! "
            f"videoconvert ! "
            f"videoscale ! "
            f"video/x-raw,format=BGR,width={self.width},height={self.height},pixel-aspect-ratio=1/1 ! "
            f"appsink caps=video/x-raw,format=BGR,width={self.width},height={self.height} drop=true max-buffers=1 sync=false"
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open RTP stream on port {self.port}. "
                f"Ensure GStreamer is installed: "
                f"sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"
            )
        
        self.frame_id = 0

    def read(self) -> tuple[np.ndarray, int, int] | None:
        """Read next frame from RTP stream.
        
        Returns:
            (frame_bgr, timestamp_ns, frame_id) tuple or None if no frame available.
        """
        if self.cap is None:
            return None
        
        ok, img = self.cap.read()
        if not ok:
            return None

        if not self._shape_logged:
            print(f"[RTPStreamSource:{self.port}] frame.shape={img.shape}")
            self._shape_logged = True

        capture_time = time.time()
        
        self.frame_id += 1
        timestamp_ns = int(capture_time * 1_000_000_000)
        
        return (img, timestamp_ns, self.frame_id)

    def stop(self) -> None:
        """Release RTP stream resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


