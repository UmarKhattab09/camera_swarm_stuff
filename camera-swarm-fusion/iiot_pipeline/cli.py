import argparse
import logging
import os

from iiot_pipeline.config import RunConfig
from iiot_pipeline.factory import StrategyFactory
from iiot_pipeline.services.storage import SessionStorage
from iiot_pipeline.facade import CameraPipelineFacade

# Data Team SDK import (supports either published name)
try:
    from package.client import Client as DataClient
except ImportError:
    from facade_sdk import Client as DataClient


def main():
    ap = argparse.ArgumentParser()

    # Capture / detection
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--calib", default="calib/c920s_1920x1080_simple.yml")
    ap.add_argument("--out", default="data/sessions")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--dict", default="4x4_50")
    ap.add_argument("--marker-length-m", type=float, default=0.035)
    ap.add_argument("--device", type=int, default=8)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--no-detect", action="store_true")
    #adding list of marker id to detect
    ap.add_argument(
        "--target-ids",
        nargs="+",
        type=int,
        help="List of ArUco marker IDs to filter (optional).",
    )

    # Broker / identity for Data Team SDK
    ap.add_argument("--broker-ip", default="192.168.1.76")
    ap.add_argument("--broker-port", type=int, default=1883)
    ap.add_argument("--device-id", default="CameraPi")
    ap.add_argument(
        "--client-type",
        default="CAMERA",
        choices=["CAMERA", "IMU", "AI", "ROBOT"],
        help="Data Team client type",
    )

    # NEW: toggle publishing (opt-in)
    ap.add_argument(
        "--publish",
        action="store_true",
        help="Enable MQTT publishing (or set env PUBLISH=1).",
    )

    args = ap.parse_args()

    config = RunConfig(
        fps=args.fps,
        grayscale=args.grayscale,
        calibrationPath=args.calib,
        sessionRoot=args.out,
        durationSec=args.duration,
        arucoDict=args.dict,
        markerLengthM=args.marker_length_m,
        device=args.device,
        targetIds = args.target_ids,
    )

    cap, pre, und, det, loc = StrategyFactory.from_config(config)
    storage = SessionStorage(config.sessionRoot, name="aruco_session")

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # Construct the SDK client only when publishing; otherwise use a no-op
    should_publish = args.publish or os.getenv("PUBLISH") == "1"

    if should_publish:
        logger.info("[iiot_pipeline] Publishing ENABLED (broker=%s)", args.broker_ip)
        client_type_val = getattr(DataClient, args.client_type, args.client_type)
        data_client = DataClient(
            broker_ip=args.broker_ip,
            client_type=client_type_val,
            device_id=args.device_id,
            auto_connect=True,
            broker_port=args.broker_port,
            timeout=60,
        )
    else:
        logger.info("[iiot_pipeline] Publishing DISABLED; using NullPublisher (no MQTT connection).")

        class NullPublisher:
            def publish(self, *_, **__):
                pass

            def close(self):
                pass

        data_client = NullPublisher()

    facade = CameraPipelineFacade(
        cap, pre, und, det, loc,
        storage, logger,
        publisher=data_client,
        target_ids=config.targetIds, 
    )

    summary = facade.run_session(config)
    print(summary)


if __name__ == "__main__":
    main()

