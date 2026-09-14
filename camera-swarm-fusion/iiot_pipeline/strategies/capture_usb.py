import cv2, time
from ..ip_types import Frame

class USBWebcamCapture:
    def __init__(self, device_index=0, requested_fps=15, w=1920, h=1080):
        self.dev, self.fps, self.w, self.h = device_index, requested_fps, w, h
        self.cap = None
        self.idx = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)

    def next_frame(self) -> Frame | None:
        ok, img = self.cap.read()
        if not ok: return None
        capture_time = time.time()
        self.idx += 1
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        return Frame(self.idx, ts, img, capture_time=capture_time)

    def stop(self):
        if self.cap is not None:
            self.cap.release()

