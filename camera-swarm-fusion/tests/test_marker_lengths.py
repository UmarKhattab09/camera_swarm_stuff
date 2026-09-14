import numpy as np
from unittest.mock import patch

from iiot_pipeline.ip_types import Detection
from iiot_pipeline.strategies.localize_pnp import PnPLocalize


def test_per_marker_lengths_scale_translation():
    """
    If two markers are identical in image geometry but have different physical sizes,
    the estimated translation should scale proportionally with marker length.
    """
    det1 = Detection(1, np.zeros((4, 2)), None, None)
    det2 = Detection(2, np.zeros((4, 2)), None, None)

    length_map = {1: 0.05, 2: 0.10}

    localizer = PnPLocalize(K="K", dist="dist", marker_length_m=0.05)

    def fake_estimate_pose(corners, length, K, dist):
        # Return tvec proportional to length for deterministic scaling
        rvec = np.array([[[0.0, 0.0, 0.0]]])
        tvec = np.array([[[length, 0.0, 0.0]]])
        return rvec, tvec, None

    with patch("iiot_pipeline.strategies.localize_pnp.cv2.aruco.estimatePoseSingleMarkers", side_effect=fake_estimate_pose):
        poses = localizer.estimate_with_lengths([det1, det2], length_map, default_length=0.05)

    assert len(poses) == 2
    t1 = poses[0].tvec.reshape(-1)
    t2 = poses[1].tvec.reshape(-1)

    # Expect translation to scale with length: 0.10 is 2x 0.05
    assert np.isclose(t1[0], 0.05)
    assert np.isclose(t2[0], 0.10)
    assert np.isclose(t2[0] / t1[0], 2.0)
