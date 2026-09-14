#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

# === Paths ===
CALIB_FILE = Path("calib/c920s_1920x1080_simple.yml")
INPUT_DIR = Path("data/raw")
OUT_DIR = Path("data/undistorted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load calibration ===
fs = cv2.FileStorage(str(CALIB_FILE), cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("dist_coeffs").mat()
fs.release()

print("Loaded camera matrix:\n", K)
print("Loaded dist coeffs:", dist.ravel())

# === Function: draw grid ===
def draw_grid(img, step=100, color=(0, 255, 0)):
    h, w = img.shape[:2]
    grid = img.copy()
    for x in range(0, w, step):
        cv2.line(grid, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(grid, (0, y), (w, y), color, 1)
    return grid

# === Process images ===
images = sorted(INPUT_DIR.glob("*.jpg"))
if not images:
    raise SystemExit(f"No images found in {INPUT_DIR}")

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print("Failed to read:", img_path)
        continue

    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, K, dist, None, new_K)

    # Overlay grid on both
    img_grid = draw_grid(img)
    undist_grid = draw_grid(undistorted)

    # Side by side comparison
    both = np.hstack((img_grid, undist_grid))

    out_path = OUT_DIR / f"{img_path.stem}_undist_grid.jpg"
    cv2.imwrite(str(out_path), both)
    print("Saved:", out_path)

print("\nDone! Check results in:", OUT_DIR)

