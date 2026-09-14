import argparse
import signal
import sys

from .config import CameraConfig, load_config
from .worker import CameraWorker


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a single camera worker")
    ap.add_argument("--config", required=True, help="Path to JSON/YAML config")

    ap.add_argument("--camera-name")
    ap.add_argument("--device")
    ap.add_argument("--fps", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--calib")
    ap.add_argument("--out")
    ap.add_argument("--duration", type=float)
    ap.add_argument("--dict")
    ap.add_argument("--marker-length-m", type=float)
    ap.add_argument("--target-ids", nargs="+", type=int)
    ap.add_argument("--reference-id", type=int)
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--no-save-frames", action="store_true")
    ap.add_argument("--save-annotated", action="store_true")
    ap.add_argument("--no-save-annotated", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--broker-ip")
    ap.add_argument("--broker-port", type=int)
    ap.add_argument("--device-id")
    ap.add_argument(
        "--client-type",
        choices=["CAMERA", "IMU", "AI", "ROBOT"],
    )

    return ap


def _apply_args(cfg: CameraConfig, args: argparse.Namespace) -> CameraConfig:
    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    save_frames = None
    if args.save_frames:
        save_frames = True
    if args.no_save_frames:
        save_frames = False

    save_annotated = None
    if args.save_annotated:
        save_annotated = True
    if args.no_save_annotated:
        save_annotated = False

    publish = None
    if args.publish:
        publish = True
    if args.no_publish:
        publish = False

    cfg.apply_overrides(
        camera_name=args.camera_name,
        device=device,
        fps=args.fps,
        width=args.width,
        height=args.height,
        calibration_path=args.calib,
        session_root=args.out,
        duration_sec=args.duration,
        aruco_dict=args.dict,
        marker_length_m=args.marker_length_m,
        target_ids=args.target_ids,
        reference_id=args.reference_id,
        no_detect=args.no_detect if args.no_detect else None,
        dry_run=args.dry_run if args.dry_run else None,
        max_frames=args.max_frames,
        save_frames=save_frames,
        save_annotated=save_annotated,
        publish=publish,
        broker_ip=args.broker_ip,
        broker_port=args.broker_port,
        device_id=args.device_id,
        client_type=args.client_type,
    )
    return cfg


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_args(cfg, args)

    worker = CameraWorker(cfg)

    def _handle_signal(_sig, _frame):
        worker.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    summary = worker.run()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
