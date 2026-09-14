#!/usr/bin/env python3
import cv2, glob
import numpy as np
from pathlib import Path

# ==== CONFIG ====
# Inner corners (10 across, 7 down)
BOARD_COLS = 10
BOARD_ROWS = 7
# Edge length of one square in meters (20.64 mm)
SQUARE_SIZE_M = 0.02064

INPUT_GLOB = str((Path(__file__).resolve().parents[2] / "data" / "raw" / "*.jpg"))
OUT_YML = Path(__file__).resolve().parents[2] / "calib" / "c920s_1920x1080.yml"
# ===============

pattern_size = (BOARD_COLS, BOARD_ROWS)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

# 3D board points for each image (Z=0)
objp = np.zeros((BOARD_ROWS*BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

objpoints = []  # 3d points in world space
imgpoints = []  # 2d points in image plane

images = sorted(glob.glob(INPUT_GLOB))
if not images:
    raise SystemExit("No images found in data/selected/*.jpg")

img_shape = None
bad = []
for f in images:
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    if img is None:
        bad.append(f); continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None: img_shape = gray.shape[::-1]
    ok, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        bad.append(f); continue
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners)

print(f"Detected boards in {len(imgpoints)} / {len(images)} images.")
if bad:
    from pathlib import Path as _P
    print("No corners found in:", ", ".join(_P(b).name for b in bad))

if len(imgpoints) < 10:
    print("Warning: few valid detections; consider collecting more/better images.")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None,
    flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3,
)

print("\n=== Calibration Results ===")
print("RMS reprojection error:", ret)
print("K:\n", K)
print("dist:", dist.ravel())

OUT_YML.parent.mkdir(parents=True, exist_ok=True)
fs = cv2.FileStorage(str(OUT_YML), cv2.FILE_STORAGE_WRITE)
fs.write("image_width", img_shape[0])
fs.write("image_height", img_shape[1])
fs.write("camera_matrix", K)
fs.write("dist_coeffs", dist)
fs.release()
print("Saved calibration to", OUT_YML)
