import cv2, numpy as np
from typing import Tuple

def load_calib(path: str) -> Tuple[np.ndarray, np.ndarray, tuple[int,int]]:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    w = int(fs.getNode("image_width").real()); h = int(fs.getNode("image_height").real())
    fs.release()
    return K, dist, (w, h)

