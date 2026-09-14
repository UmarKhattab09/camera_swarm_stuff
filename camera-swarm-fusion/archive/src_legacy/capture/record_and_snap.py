#!/usr/bin/env python3
import cv2, time, json, os, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Record video and save still frames at a fixed rate.")
    ap.add_argument("--device", type=int, default=8, help="V4L2 device index (default 8 for your C920)")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--video-fps", type=int, default=30, help="video recording fps")
    ap.add_argument("--snap-fps",  type=float, default=10.0, help="save frame images at this fps (>=10)")
    ap.add_argument("--name", default="session", help="base name for the new session folder")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until you press q)")
    ap.add_argument("--no-gui", action="store_true", help="run without preview window")
    ap.add_argument("--codec", default="MJPG", choices=["MJPG","XVID"], help="AVI codec for video")
    args = ap.parse_args()

    # Project root: this file lives at c920s_v4l2/src/capture/, so parents[2] is c920s_v4l2/
    root = Path(__file__).resolve().parents[2]

    # New session directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_name = f"{args.name}_{ts}"
    session_dir  = root / "data" / "sessions" / session_name
    video_dir    = session_dir / "video"
    frames_dir   = session_dir / "frames"
    for d in (video_dir, frames_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Save metadata for reproducibility
    meta = {
        "device": args.device,
        "width": args.width,
        "height": args.height,
        "video_fps": args.video_fps,
        "snap_fps": args.snap_fps,
        "codec": args.codec,
        "start_time": ts,
    }
    with open(session_dir / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Open camera
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # C920 delivers MJPEG nicely
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.video_fps)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.device}")

    # Prepare VideoWriter (AVI)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    video_path = video_dir / f"{session_name}.avi"
    writer = cv2.VideoWriter(str(video_path), fourcc, float(args.video_fps), (args.width, args.height))
    if not writer.isOpened():
        # Fallback to the other codec
        alt = "XVID" if args.codec=="MJPG" else "MJPG"
        fourcc = cv2.VideoWriter_fourcc(*alt)
        video_path = video_dir / f"{session_name}_{alt}.avi"
        writer = cv2.VideoWriter(str(video_path), fourcc, float(args.video_fps), (args.width, args.height))
        if not writer.isOpened():
            cap.release()
            raise SystemExit("Failed to open VideoWriter with MJPG or XVID. Try lower resolution/fps or install codecs.")

    print("Recording to:", video_path)
    print("Saving stills in:", frames_dir)
    print("Controls: press 'q' to stop.")

    frame_idx = 0
    next_snap_t = 0.0
    snap_period = 1.0 / max(1e-6, args.snap_fps)
    t0 = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame_idx += 1

        # Overlay frame number in the top-left corner
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        # Write video frame
        writer.write(frame)

        # Save stills at snap_fps
        t = time.monotonic() - t0
        if t >= next_snap_t:
            out_img = frames_dir / f"f{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_img), frame)
            next_snap_t += snap_period

        # Show preview unless headless
        if not args.no_gui:
            cv2.imshow("recording (q=quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        # Duration stop
        if args.duration > 0 and t >= args.duration:
            break

    cap.release()
    writer.release()
    if not args.no_gui:
        cv2.destroyAllWindows()

    # convenience: update a 'latest' symlink to this session
    latest = (root / "data" / "sessions" / "latest")
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(session_dir)
    except Exception:
        pass

    print("Done.")
    print("Video:", video_path)
    print("Frames:", frames_dir)

if __name__ == "__main__":
    main()
