import os, time, logging
import cv2
import numpy as np

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
        publisher=None,  # Data Team SDK client; if None, no publishing
        target_ids = None, 
    ):
        self.cap = cap
        self.pre = pre
        self.und = und
        self.det = det
        self.loc = loc
        self.store = storage
        self.log = logger
        self.publisher = publisher
        self._published_header = False

        if target_ids is None:
            self.target_ids = None
        elif isinstance(target_ids, (int, np.integer)):
            self.target_ids = {int(target_ids)}
        elif isinstance(target_ids, (list, tuple, set)):
            self.target_ids = {int(tid) for tid in target_ids}
        else:
            raise TypeError("target_ids must be int, list, tuple, set, or None")

    def _publish_header_once(self):
        """Publish CSV header line once per session."""
        if self.publisher and not self._published_header:
            try:
                self.publisher.publish(",".join(CsvWriter.HEADER))
                self._published_header = True
            except Exception as e:
                self.log.warning("Header publish failed: %s", e)

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

        csv_local = CsvWriter(str(self.store.session_dir / "detections.csv"))
        csv_local.open()

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

                # CSV rows: one per detection
                for i, det in enumerate(dets):
                    if self.target_ids is not None and det.marker_id not in self.target_ids:
                        continue
                    rvec = poses[i].rvec if i < len(poses) else None
                    tvec = poses[i].tvec if i < len(poses) else None
                    ts_unix = time.time()
                    capture_time = f.capture_time

                    # local file append (DB schema order)
                    csv_local.append(
                        ts_unix,
                        f.idx,
                        det.marker_id,
                        rvec,
                        tvec,
                        self.store.last_path,
                        capture_time=capture_time,
                    )

                    # publish line as CSV string (DB schema order)
                    if self.publisher:
                        try:
                            self._publish_header_once()
                            line = CsvWriter.to_csv_line(
                                ts_unix,
                                f.idx,
                                det.marker_id,
                                rvec,
                                tvec,
                                str(self.store.last_path),
                                capture_time=capture_time,
                            )
                            self.publisher.publish(line)
                        except Exception as e:
                            self.log.warning("Data publish failed: %s", e)

                self.log.info(
                    "frame=%d saved=%s dets=%d",
                    f.idx,
                    self.store.last_path,
                    len(dets),
                )
                frames += 1

        finally:
            try:
                self.cap.stop()
            except Exception:
                pass
            try:
                csv_local.close()
            except Exception:
                pass

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

