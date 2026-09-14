import cv2
from pathlib import Path

CALIB = Path(__file__).resolve().parents[2] / "calib" / "c920s_1920x1080.yml"
IMG   = Path(__file__).resolve().parents[2] / "data" / "selected" / "sample.jpg"  # change me

fs = cv2.FileStorage(str(CALIB), cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("dist_coeffs").mat()
fs.release()

img = cv2.imread(str(IMG))
if img is None:
    raise SystemExit(f"Could not read {IMG}")
h, w = img.shape[:2]
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
und = cv2.undistort(img, K, dist, None, newK)

cv2.imshow("orig", img)
cv2.imshow("undistorted", und)
cv2.waitKey(0)
