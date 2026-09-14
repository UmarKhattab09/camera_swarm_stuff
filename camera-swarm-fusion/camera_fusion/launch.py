import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Launch multiple camera workers concurrently",
        epilog="Example: python -m camera_fusion.launch configs/cam1_stream.json configs/cam2_stream.json"
    )
    ap.add_argument("configs", nargs="+", help="List of config files (e.g., cam1.json cam2.json)")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between spawns (sec)")
    ap.add_argument("--publish", action="store_true", help="Enable MQTT publishing for all workers")
    ap.add_argument("--no-publish", action="store_true", help="Disable MQTT publishing for all workers")
    ap.add_argument("--broker-ip", help="MQTT broker IP for all workers")
    ap.add_argument("--broker-port", type=int, help="MQTT broker port for all workers")
    ap.add_argument("--device-id", help="Base device ID for all workers")
    ap.add_argument(
        "--client-type",
        choices=["CAMERA", "IMU", "AI", "ROBOT"],
        help="Data Team client type for all workers",
    )

    args = ap.parse_args()

    procs: list[subprocess.Popen] = []

    def _terminate_all():
        """Gracefully terminate all worker processes."""
        for p in procs:
            if p.poll() is None:
                p.terminate()

    def _handle_signal(_sig, _frame):
        """Signal handler for SIGINT and SIGTERM."""
        print("\n[launch.py] Received stop signal, terminating all workers...")
        _terminate_all()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    # Start all workers
    for i, cfg in enumerate(args.configs, 1):
        cfg_path = Path(cfg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")
        cmd = [sys.executable, "-m", "camera_fusion.run", "--config", cfg]
        if args.publish:
            cmd.append("--publish")
        if args.no_publish:
            cmd.append("--no-publish")
        if args.broker_ip:
            cmd.extend(["--broker-ip", args.broker_ip])
        if args.broker_port is not None:
            cmd.extend(["--broker-port", str(args.broker_port)])
        if args.device_id:
            cmd.extend(["--device-id", args.device_id])
        if args.client_type:
            cmd.extend(["--client-type", args.client_type])
        print(f"[launch.py] Starting worker {i}/{len(args.configs)}: {cfg_path.name}")
        procs.append(subprocess.Popen(cmd))
        if args.delay > 0:
            time.sleep(args.delay)

    print(f"[launch.py] All {len(procs)} worker(s) started. Press Ctrl+C to stop.")

    try:
        exit_codes = [p.wait() for p in procs]
        max_code = max(exit_codes) if exit_codes else 0
        if max_code != 0:
            print(f"[launch.py] One or more workers exited with error code {max_code}")
        return max_code
    except KeyboardInterrupt:
        _terminate_all()
        return 130


if __name__ == "__main__":
    sys.exit(main())
