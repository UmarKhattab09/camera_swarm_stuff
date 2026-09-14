import cv2, time
from pathlib import Path

DEVICE_INDEX = 8;
W, H, FPS = 1920, 1080, 30
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera")

print("Press SPACE to capture, 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    cv2.putText(view, f"{W}x{H}@{FPS}", (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("Capture (space to save)", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = OUT_DIR / f"cal_{ts}.jpg"
        cv2.imwrite(str(fname), frame)
        print("Saved", fname)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
