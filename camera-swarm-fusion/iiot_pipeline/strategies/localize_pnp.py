import cv2, numpy as np
from ..ip_types import Detection, Pose

class PnPLocalize:
    def __init__(self, K, dist, marker_length_m: float):
        self.K, self.dist, self.L = K, dist, marker_length_m

    def _estimate_single_marker_pose(self, corners, marker_length: float):
        aruco_mod = getattr(cv2, "aruco", None)
        estimate_fn = getattr(aruco_mod, "estimatePoseSingleMarkers", None)

        if estimate_fn is not None:
            rvec, tvec, _ = estimate_fn([corners], marker_length, self.K, self.dist)
            return rvec[0], tvec[0]

        half = marker_length / 2.0
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)

        solvepnp_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.K,
            self.dist,
            flags=solvepnp_flag,
        )
        if not ok:
            return None, None
        return rvec.reshape(1, 3), tvec.reshape(1, 3)

    def estimate(self, detections: list[Detection]) -> list[Pose]:
        poses = []
        if self.L <= 0 or not detections: return poses
        # Estimate per marker (you can aggregate later)
        for det in detections:
            rvec, tvec = self._estimate_single_marker_pose(det.corners, self.L)
            if rvec is None or tvec is None:
                poses.append(Pose(None, None))
            else:
                poses.append(Pose(rvec, tvec))
        return poses

    def estimate_with_lengths(
        self,
        detections: list[Detection],
        length_map: dict[int, float] | None,
        default_length: float | None = None,
    ) -> list[Pose]:
        poses = []
        if not detections:
            return poses
        default_len = self.L if default_length is None else default_length
        if default_len <= 0 and not length_map:
            return poses

        for det in detections:
            length = default_len
            if length_map and det.marker_id in length_map:
                length = length_map[det.marker_id]
            if length <= 0:
                poses.append(Pose(None, None))
                continue
            rvec, tvec = self._estimate_single_marker_pose(det.corners, length)
            if rvec is None or tvec is None:
                poses.append(Pose(None, None))
            else:
                poses.append(Pose(rvec, tvec))
        return poses

