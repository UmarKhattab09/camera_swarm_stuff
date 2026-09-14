# iiot_pipeline/facade.py
import os, time, logging, json
import cv2
import numpy as np
import io

from .config import RunConfig, SessionSummary
from .services.storage import SessionStorage
from .services.csv_writer import CsvWriter


class CameraPipelineFacade:
    def __init__(
        self,
        cap,
        pre,
        und,
        det,
        loc,
        storage: SessionStorage,
        logger: logging.Logger,
        publisher=None  # Data Team SDK client; if None, no publishing
    ):
        self.cap = cap
        self.pre = pre
        self.und = und
        self.det = det
        self.loc = loc
        self.store = storage
        self.log = logger
        self.publisher = publisher

    def _to_list(self, arr):
        if arr is None:
            return None
        try:
            return np.asarray(arr).astype(float).reshape(-1).tolist()
        except Exception:
            return None

    def run_session(self, config: RunConfig) -> SessionSummary:
        # create session folders + manifest
        session_path = self.store.begin()
        self.store.write_manifest(vars(config))

        # file logger
        log_file = os.path.join(self.store.logs_dir, "session.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(fmt)
        self.log.addHandler(fh)
        self.log.propagate = False

        self.log.info("session started: %s", session_path)
        self.log.info("config: %s", vars(config))

        csv = CsvWriter(str(self.store.session_dir / "detections.csv"))
        csv.open()

        self.cap.start()
        t0 = time.time()
        frames = 0
        errors = 0

        try:
            while True:
                if config.durationSec and (time.time() - t0) >= config.durationSec:
                    break

                f = self.cap.next_frame()
                if f is None:
                    errors += 1
                    continue

                # preprocess + undistort (clean image)
                f = self.pre.apply(f)
                f = self.und.apply(f)

                # detect -> poses
                dets = self.det.detect(f)
                poses = self.loc.estimate(dets)

                # --- Save annotated copy only when there are detections ---
                if dets:
                    draw = f.image.copy()

                    # overlay text (frame#, ts, WxH)
                    h, w = draw.shape[:2]
                    txt = f"#{f.idx} {f.ts_iso} {w}x{h}"
                    cv2.putText(
                        draw,
                        txt,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # boxes + ids
                    ids = np.array([d.marker_id for d in dets], dtype=np.int32).reshape(-1, 1)
                    corners = [d.corners for d in dets]
                    try:
                        cv2.aruco.drawDetectedMarkers(draw, corners, ids)
                    except Exception:
                        pass

                    # axes using PnP result
                    for i, pose in enumerate(poses):
                        try:
                            cv2.drawFrameAxes(
                                draw,
                                self.loc.K,
                                self.loc.dist,
                                pose.rvec,
                                pose.tvec,
                                max(0.01, config.markerLengthM * 0.5),
                            )
                        except Exception:
                            pass

                    self.store.save_annotated(f.idx, draw)
                # ----------------------------------------------------------

                # always save the UNDISTORTED ORIGINAL
                self.store.save_frame(f)

                # CSV rows: (marker_id, rvec, tvec) per detection
                for i, det in enumerate(dets):
                    rvec = poses[i].rvec if i < len(poses) else None
                    tvec = poses[i].tvec if i < len(poses) else None
                    csv.append(f.ts_iso, f.idx, det.marker_id, rvec, tvec, self.store.last_path)

                    # Publish detection as JSON string (if publisher configured)
                    if self.publisher:
                        payload = {
                            "source": "camera",
                            "device_id": getattr(self.publisher, "device_id", None),
                            "client_type": "CAMERA",
                            "ts_epoch": time.time(),
                            "ts_iso": f.ts_iso,
                            "frame_idx": int(f.idx),
                            "marker_id": int(det.marker_id),
                            "pose": {
                                "rvec": self._to_list(rvec),
                                "tvec": self._to_list(tvec),
                            },
                            "image_path": str(self.store.last_path),
                        }
                        try:
                            #self.publisher.publish(
                            #output = io.StringIO()
                            #writer = csv.writer(output)
                            #writer.writerrow([
                             #   payload["ts_epoch"],
                              #  payload["frame_idx"],
                               # payload["marker_id"],
                                #*payload["pose"]["rvec"],
                                #*payload["pose"]["tvec"],
                                #payload["image_path"]
                            #])
                            #csv_line = output.getvalue().strip()
                            #self.publisher.publish(csv_line)
                            self.publisher.publish(json.dumps(payload))
                        except Exception as e:
                            self.log.warning("Data publish failed: %s", e)

                self.log.info(
                    "frame=%d saved=%s dets=%d",
                    f.idx,
                    self.store.last_path,
                    len(dets),
                )
                self.publisher.publish(csv)
                frames += 1

        finally:
            self.cap.stop()
            csv.close()

        avg = frames / max(1e-6, (time.time() - t0))
        self.log.info("summary frames=%d avg_fps=%.2f errors=%d", frames, avg, errors)
        return SessionSummary(
            str(session_path),
            frames,
            str(self.store.session_dir / "detections.csv"),
            str(self.store.logs_dir / "session.log"),
            avg,
            errors,
        )

