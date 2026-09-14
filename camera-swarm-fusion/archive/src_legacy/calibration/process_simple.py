#!/usr/bin/env python3
import cv2, glob, numpy as np
from pathlib import Path

# --- Your board ---
BOARD_COLS, BOARD_ROWS = 10, 7        # inner corners
SQUARE_SIZE_M = 0.02064               # 20.64 mm

# --- IO ---
INPUT_GLOB = str((Path(__file__).resolve().parents[2] / "data" / "raw" / "*.jpg"))
OUT_YML    = Path(__file__).resolve().parents[2] / "calib" / "c920s_1920x1080_simple.yml"

# --- Prepare object pattern points (Z=0) ---
pattern_size = (BOARD_COLS, BOARD_ROWS)
objp = np.zeros((BOARD_ROWS*BOARD_COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

objpoints, imgpoints, imgs_used = [], [], []
images = sorted(glob.glob(INPUT_GLOB))
if not images:
    raise SystemExit("No images in data/rawd/*.jpg")

img_shape = None
bad = []
for f in images:
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    if img is None:
        bad.append(f); continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None: img_shape = gray.shape[::-1]  # (w,h)
    ok, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        bad.append(f); continue
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    objpoints.append(objp); imgpoints.append(corners); imgs_used.append(f)

print(f"Detected boards in {len(imgpoints)} / {len(images)} images.")
if bad:
    from pathlib import Path as _P
    print("No corners found in:", ", ".join(_P(b).name for b in bad))

if len(imgpoints) < 10:
    print("Warning: few detections; consider more/better shots.")

# --- First pass: classic 5-param model (NO rational extras) ---
flags = 0  # (lets k1,k2,p1,p2,k3 vary)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None, flags=flags
)

def per_view_errors():
    errs = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        e = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        errs.append(float(e))
    return np.array(errs)

errs = per_view_errors()
print(f"\nFirst pass RMS (OpenCV): {ret:.4f}")
print(f"Per-image mean reprojection error: mean={errs.mean():.4f}, median={np.median(errs):.4f}")

# --- Drop worst 20% outliers and recalibrate ---
keep = int(max(10, round(len(imgpoints)*0.8)))
keep_idx = np.argsort(errs)[:keep]
obj2 = [objpoints[i] for i in keep_idx]
img2 = [imgpoints[i] for i in keep_idx]

ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
    obj2, img2, img_shape, None, None, flags=flags
)
print(f"\nRecalibrated on best {keep}/{len(imgpoints)} images.")
print(f"RMS reprojection error: {ret2:.4f}")
print("K:\n", K2)
print("dist:", dist2.ravel())

OUT_YML.parent.mkdir(parents=True, exist_ok=True)
fs = cv2.FileStorage(str(OUT_YML), cv2.FILE_STORAGE_WRITE)
fs.write("image_width", img_shape[0])
fs.write("image_height", img_shape[1])
fs.write("camera_matrix", K2)
fs.write("dist_coeffs", dist2)
fs.release()
print("Saved calibration to", OUT_YML)
