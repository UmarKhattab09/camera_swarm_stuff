"""SE(3) transformation utilities for ArUco pose handling."""

import numpy as np
import cv2
from typing import Tuple, Optional


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert rotation vector and translation vector to 4x4 transformation matrix.
    
    Args:
        rvec: Rotation vector (3,) or (3,1)
        tvec: Translation vector (3,) or (3,1)
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    rvec = np.array(rvec).reshape(3)
    tvec = np.array(tvec).reshape(3)
    
    R, _ = cv2.Rodrigues(rvec)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec
    
    return T


def matrix_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 4x4 transformation matrix to rotation vector and translation vector.
    
    Args:
        T: 4x4 homogeneous transformation matrix
    
    Returns:
        (rvec, tvec) where rvec is (3,1) and tvec is (3,1)
    """
    R = T[:3, :3]
    tvec = T[:3, 3].reshape(3, 1)
    
    rvec, _ = cv2.Rodrigues(R)
    
    return rvec, tvec


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 homogeneous transformation matrix.
    
    For SE(3): T^-1 = [R^T, -R^T * t; 0, 1]
    
    Args:
        T: 4x4 transformation matrix
    
    Returns:
        4x4 inverted transformation matrix
    """
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    
    R_T = R.T
    T_inv[:3, :3] = R_T
    T_inv[:3, 3] = -R_T @ t
    
    return T_inv


def compute_relative_pose(
    rvec_ref: np.ndarray,
    tvec_ref: np.ndarray,
    rvec_target: np.ndarray,
    tvec_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pose of target relative to reference frame.
    
    Given:
        T_cam_ref: pose of reference marker in camera frame
        T_cam_target: pose of target marker in camera frame
    
    Compute:
        T_ref_target = inv(T_cam_ref) @ T_cam_target
    
    Args:
        rvec_ref: Rotation vector of reference marker (camera frame)
        tvec_ref: Translation vector of reference marker (camera frame)
        rvec_target: Rotation vector of target marker (camera frame)
        tvec_target: Translation vector of target marker (camera frame)
    
    Returns:
        (rvec_rel, tvec_rel): Pose of target in reference frame
    """
    T_cam_ref = rvec_tvec_to_matrix(rvec_ref, tvec_ref)
    T_cam_target = rvec_tvec_to_matrix(rvec_target, tvec_target)
    
    T_ref_cam = invert_transform(T_cam_ref)
    T_ref_target = T_ref_cam @ T_cam_target
    
    rvec_rel, tvec_rel = matrix_to_rvec_tvec(T_ref_target)
    
    return rvec_rel, tvec_rel
