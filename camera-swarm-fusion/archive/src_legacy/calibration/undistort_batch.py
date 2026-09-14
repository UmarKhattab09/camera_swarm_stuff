import glob, cv2, argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Undistort all images from data/raw into data/undistorted/<name>")
parser.add_argument("name", nargs="?", default="checkerboard_test", help="output subfolder name under data/undistorted/")
parser.add_argument("--alpha", type=float, default=0.0, help="0.0=crop to valid; 0.5 some FOV; 1.0 max FOV (may add borders)")
args = parser.parse_args()

root = Path(__file__).resolve().parents[2]
CALIB   = root / "calib" / "c920s_1920x1080_simple.yml"
RAW_DIR = root / "data" / "raw"
OUT_DIR = root / "data" / "undistorted" / args.name
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load K, dist
fs = cv2.FileStorage(str(CALIB), cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("dist_coeffs").mat()
w = int(fs.getNode("image_width").real()); h = int(fs.getNode("image_height").real())
fs.release()

# Undistortion maps
newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), args.alpha)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), cv2.CV_16SC2)

# Collect files
files = []
for ext in ("*.jpg","*.jpeg","*.png"):
    files.extend(glob.glob(str(RAW_DIR / ext)))
files.sort()
if not files:
    raise SystemExit(f"No images found in {RAW_DIR}")

# Process
count = 0
skipped_size = 0
for f in files:
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    if img is None:
        print("Skipping unreadable:", f); continue
    ih, iw = img.shape[:2]
    if (iw, ih) != (w, h):
        skipped_size += 1
        print("Skipping (wrong size):", f, f"(got {iw}x{ih}, need {w}x{h})")
        continue
    und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    outp = OUT_DIR / (Path(f).stem + "_und.jpg")
    cv2.imwrite(str(outp), und)
    count += 1

print("Undistorted", count, "images ->", OUT_DIR)
if skipped_size:
    print("Skipped", skipped_size, "images due to size mismatch.")
