from abc import ABC, abstractmethod
import cv2
from ..ip_types import Frame

class PreprocessStrategy(ABC):
    @abstractmethod
    def apply(self, f: Frame) -> Frame: ...

class ColorFrame(PreprocessStrategy):
    def apply(self, f: Frame) -> Frame:
        return f

class GrayscaleFrame(PreprocessStrategy):
    def apply(self, f: Frame) -> Frame:
        g = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        return Frame(f.idx, f.ts_iso, g)

