import time
import csv
import io
import paho.mqtt.client as mqtt

BROKER_IP = "192.168.1.76"   # your broker
TOPIC = "camera/CameraPi"    # must match handle_camera

def make_mock_message(recorded_at, frame_idx, marker_idx, rvec, tvec, img_path):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        f"{recorded_at:.6f}",
        frame_idx,
        marker_idx,
        *rvec,
        *tvec,
        img_path
    ])
    return buf.getvalue().strip()

def main():
    client = mqtt.Client()
    client.connect(BROKER_IP, 1883, 60)

    print("Publishing 5 mock camera messages...")
    for i in range(5):
        recorded_at = time.time()
        frame_idx = 100 + i
        marker_idx = 42
        rvec = [0.1 * i, 0.2 * i, 0.3 * i]
        tvec = [1.0 * i, 2.0 * i, 3.0 * i]
        img_path = f"data/sessions/mock/frame_{frame_idx}.jpg"

        msg = make_mock_message(recorded_at, frame_idx, marker_idx, rvec, tvec, img_path)
        print(" ->", msg)
        client.publish(TOPIC, msg)
        time.sleep(1)

    client.disconnect()
    print("Done.")

if __name__ == "__main__":
    main()

