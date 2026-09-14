#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _latest_session(session_root: Path, camera_name: str) -> Path | None:
    candidates = sorted(
        session_root.glob(f"{camera_name}_session_*"),
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _print_camera_metrics(session_root: Path, camera_name: str, show_summary: bool) -> None:
    session = _latest_session(session_root, camera_name)
    if session is None:
        print(f"[{camera_name}] no session folders found under: {session_root}")
        return

    frames_csv = session / "metrics_frames.csv"
    summary_json = session / "metrics_summary.json"
    log_file = session / "logs" / "session.log"

    print(f"[{camera_name}] latest session: {session}")
    print(f"  metrics_frames.csv: {frames_csv} ({'OK' if frames_csv.exists() else 'MISSING'})")
    print(f"  metrics_summary.json: {summary_json} ({'OK' if summary_json.exists() else 'MISSING'})")
    print(f"  session.log: {log_file} ({'OK' if log_file.exists() else 'MISSING'})")

    if show_summary and summary_json.exists():
        try:
            data = json.loads(summary_json.read_text(encoding="utf-8"))
            print("  summary:")
            print(json.dumps(data, indent=2))
        except Exception as exc:
            print(f"  failed to read summary json: {exc}")


def _print_camera_one_line(session_root: Path, camera_name: str) -> None:
    session = _latest_session(session_root, camera_name)
    if session is None:
        print(f"[{camera_name}] no session")
        return

    summary_json = session / "metrics_summary.json"
    if not summary_json.exists():
        print(f"[{camera_name}] latest={session.name} summary=MISSING")
        return

    try:
        data = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[{camera_name}] latest={session.name} summary=ERROR ({exc})")
        return

    total = int(data.get("total_frames", 0))
    aruco_rate = float(data.get("aruco_success_rate", 0.0))
    hybrid_rate = float(data.get("hybrid_success_rate", 0.0))
    recovery_rate = float(data.get("recovery_rate", 0.0))
    fallback_count = int(data.get("fallback_triggered_count", 0))
    pipeline = data.get("pipeline_latency_ms", {}) or {}
    pipe_mean = float(pipeline.get("mean", 0.0))
    pipe_p50 = float(pipeline.get("median", 0.0))
    pipe_max = float(pipeline.get("max", 0.0))
    fallback_mean = float(data.get("mean_fallback_latency_ms", 0.0))

    print(
        f"[{camera_name}] latest={session.name} frames={total} "
        f"aruco={aruco_rate:.3f} hybrid={hybrid_rate:.3f} recovery={recovery_rate:.3f} "
        f"fallback_n={fallback_count} pipe_ms(mean/p50/max)={pipe_mean:.1f}/{pipe_p50:.1f}/{pipe_max:.1f} "
        f"fallback_ms_mean={fallback_mean:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show latest metrics files for camera sessions"
    )
    parser.add_argument(
        "--session-root",
        default="data/sessions",
        help="Session root directory (default: data/sessions)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["cam1", "cam2"],
        help="Camera names to check (default: cam1 cam2)",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Print the full metrics_summary.json content",
    )
    parser.add_argument(
        "--one-line",
        action="store_true",
        help="Print one-line summary per camera from latest metrics_summary.json",
    )
    args = parser.parse_args()

    session_root = Path(args.session_root)
    if not session_root.exists():
        print(f"Session root does not exist: {session_root}")
        return 1

    for camera_name in args.cameras:
        if args.one_line:
            _print_camera_one_line(session_root, camera_name)
            continue
        _print_camera_metrics(session_root, camera_name, args.show_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
