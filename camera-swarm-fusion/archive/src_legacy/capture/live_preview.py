import cv2

DEVICE_INDEX = 0
W, H, FPS = 1920, 1080, 30

cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Live (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
