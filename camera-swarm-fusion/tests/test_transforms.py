import numpy as np
import pytest

from camera_fusion.transforms import (
    rvec_tvec_to_matrix,
    matrix_to_rvec_tvec,
    invert_transform,
    compute_relative_pose,
)


def test_rvec_tvec_to_matrix():
    """Test conversion from rvec/tvec to 4x4 matrix."""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([1.0, 2.0, 3.0])
    
    T = rvec_tvec_to_matrix(rvec, tvec)
    
    assert T.shape == (4, 4)
    assert np.allclose(T[3, :], [0, 0, 0, 1])
    assert np.allclose(T[:3, 3], tvec)
    
    # Rotation matrix should be orthonormal
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_matrix_to_rvec_tvec():
    """Test conversion from 4x4 matrix to rvec/tvec."""
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    
    rvec, tvec = matrix_to_rvec_tvec(T)
    
    assert rvec.shape == (3, 1)
    assert tvec.shape == (3, 1)
    assert np.allclose(tvec.flatten(), [1.0, 2.0, 3.0])
    assert np.allclose(rvec.flatten(), [0.0, 0.0, 0.0], atol=1e-6)


def test_roundtrip_conversion():
    """Test that rvec/tvec -> matrix -> rvec/tvec gives same result."""
    rvec_orig = np.array([0.1, 0.2, 0.3])
    tvec_orig = np.array([1.0, 2.0, 3.0])
    
    T = rvec_tvec_to_matrix(rvec_orig, tvec_orig)
    rvec_back, tvec_back = matrix_to_rvec_tvec(T)
    
    assert np.allclose(rvec_back.flatten(), rvec_orig, atol=1e-6)
    assert np.allclose(tvec_back.flatten(), tvec_orig, atol=1e-6)


def test_invert_transform_identity():
    """Test that inverting identity gives identity."""
    T = np.eye(4)
    T_inv = invert_transform(T)
    
    assert np.allclose(T_inv, np.eye(4))


def test_invert_transform_roundtrip():
    """Test that T @ inv(T) = I."""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([1.0, 2.0, 3.0])
    
    T = rvec_tvec_to_matrix(rvec, tvec)
    T_inv = invert_transform(T)
    
    result = T @ T_inv
    assert np.allclose(result, np.eye(4), atol=1e-6)


def test_compute_relative_pose_identity():
    """Test relative pose when both markers are at origin."""
    rvec_ref = np.zeros(3)
    tvec_ref = np.zeros(3)
    rvec_target = np.zeros(3)
    tvec_target = np.zeros(3)
    
    rvec_rel, tvec_rel = compute_relative_pose(
        rvec_ref, tvec_ref, rvec_target, tvec_target
    )
    
    assert np.allclose(rvec_rel.flatten(), 0.0, atol=1e-6)
    assert np.allclose(tvec_rel.flatten(), 0.0, atol=1e-6)


def test_compute_relative_pose_translation_only():
    """Test relative pose with pure translation."""
    # Reference at origin
    rvec_ref = np.zeros(3)
    tvec_ref = np.zeros(3)
    
    # Target translated by [1, 0, 0] in camera frame
    rvec_target = np.zeros(3)
    tvec_target = np.array([1.0, 0.0, 0.0])
    
    rvec_rel, tvec_rel = compute_relative_pose(
        rvec_ref, tvec_ref, rvec_target, tvec_target
    )
    
    # In reference frame, target should be at [1, 0, 0]
    assert np.allclose(rvec_rel.flatten(), 0.0, atol=1e-6)
    assert np.allclose(tvec_rel.flatten(), [1.0, 0.0, 0.0], atol=1e-6)


def test_compute_relative_pose_with_rotation():
    """Test relative pose with rotation."""
    # Reference at origin
    rvec_ref = np.zeros(3)
    tvec_ref = np.zeros(3)
    
    # Target rotated 90 degrees around Z axis
    rvec_target = np.array([0.0, 0.0, np.pi / 2])
    tvec_target = np.zeros(3)
    
    rvec_rel, tvec_rel = compute_relative_pose(
        rvec_ref, tvec_ref, rvec_target, tvec_target
    )
    
    # Relative rotation should be same as target rotation
    assert np.allclose(rvec_rel.flatten(), rvec_target, atol=1e-6)
    assert np.allclose(tvec_rel.flatten(), 0.0, atol=1e-6)


def test_compute_relative_pose_offset_reference():
    """Test relative pose when reference is not at origin."""
    # Reference at [1, 0, 0]
    rvec_ref = np.zeros(3)
    tvec_ref = np.array([1.0, 0.0, 0.0])
    
    # Target at [2, 0, 0]
    rvec_target = np.zeros(3)
    tvec_target = np.array([2.0, 0.0, 0.0])
    
    rvec_rel, tvec_rel = compute_relative_pose(
        rvec_ref, tvec_ref, rvec_target, tvec_target
    )
    
    # In reference frame, target should be at [1, 0, 0]
    assert np.allclose(rvec_rel.flatten(), 0.0, atol=1e-6)
    assert np.allclose(tvec_rel.flatten(), [1.0, 0.0, 0.0], atol=1e-6)
