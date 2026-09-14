#!/usr/bin/env python3
import cv2, time
from pathlib import Path

# ==== CONFIG ====
DEVICE_INDEX = 8          # USB camera index
W, H, FPS = 1920, 1080, 30
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "selected"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 20           # How many images to capture
DELAY_BETWEEN = 1.0       # Seconds between captures
# =====================

cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_AUTO_WB, 1)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera")

# Ask camera what FPS it really applied
reported_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Requested FPS: {FPS}, Camera reports: {reported_fps}")

print(f"Starting automatic capture for calibration ({NUM_IMAGES} images)...")

count = 0
last_time = time.time()
frame_count = 0
measured_fps = 0.0

while count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    # Estimate measured FPS over 1s windows
    frame_count += 1
    now = time.time()
    if now - last_time >= 1.0:
        measured_fps = frame_count / (now - last_time)
        last_time = now
        frame_count = 0

    # Overlay info
    overlay_text = f"{W}x{H} ({count}/{NUM_IMAGES}) | Req: {FPS} | Rep: {reported_fps:.1f} | Meas: {measured_fps:.1f}"
    preview = frame.copy()
    cv2.putText(preview, overlay_text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Calibration Capture", preview)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord(' '):  # manual capture
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = OUT_DIR / f"cal_{ts}.jpg"
        cv2.putText(frame, overlay_text, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(fname), frame)
        print("Saved", fname)
        count += 1

    # Automatic capture
    if count < NUM_IMAGES:
        ts = time.time()
        while time.time() - ts < DELAY_BETWEEN:
            cv2.imshow("Calibration Capture", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                count = NUM_IMAGES
                break
        fname = OUT_DIR / f"cal_{count:03d}.jpg"
        cv2.putText(frame, overlay_text, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(fname), frame)
        print("Saved", fname)
        count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Finished capturing {NUM_IMAGES} images. Run process.py next.")

