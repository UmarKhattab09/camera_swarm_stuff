#!/usr/bin/env python3
"""Test script for LightGlue fallback functionality.

This script:
1. Loads a saved frame from a session
2. Simulates missing ArUco detections
3. Attempts recovery using LightGlue fallback
4. Verifies at least one marker can be recovered
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from camera_fusion.config import LightGlueConfig
from camera_fusion.fallback import LightGlueFallback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_frame(frame_path: str) -> np.ndarray:
    """Load a test frame from disk."""
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not load frame: {frame_path}")
    logger.info(f"Loaded test frame: {frame_path} (shape: {img.shape})")
    return img


def simulate_missing_markers(
    all_marker_ids: set[int],
    remove_count: int = 1
) -> tuple[dict[int, np.ndarray], set[int]]:
    """Simulate some markers being missing.
    
    Args:
        all_marker_ids: Set of all expected marker IDs
        remove_count: Number of markers to simulate as missing
    
    Returns:
        (detected_dict, expected_ids) where detected_dict is empty
    """
    detected_dict = {}  # Simulate all markers missing initially
    expected_ids = all_marker_ids
    logger.info(f"Simulating missing markers: {expected_ids}")
    return detected_dict, expected_ids


def test_fallback_recovery(
    frame_path: str,
    template_dir: str,
    marker_ids: set[int],
    device: str = "cpu"
) -> bool:
    """Test LightGlue fallback recovery.
    
    Args:
        frame_path: Path to test frame image
        template_dir: Directory containing marker templates
        marker_ids: Set of marker IDs to test
        device: Device to use ("cpu" or "cuda")
    
    Returns:
        True if at least one marker was recovered
    """
    logger.info("="*60)
    logger.info("Testing LightGlue Fallback Recovery")
    logger.info("="*60)
    
    # Load test frame
    frame = load_test_frame(frame_path)
    
    # Create LightGlue config
    lg_config = LightGlueConfig(
        enabled=True,
        device=device,
        template_dir=template_dir,
        min_inliers=4,
        max_age_frames=5,
        roi_expand_px=50,
        debug_save=False,
        corner_refine=True,
        match_threshold=0.2
    )
    
    # Initialize fallback
    logger.info("Initializing LightGlue fallback...")
    try:
        fallback = LightGlueFallback(lg_config, aruco_cfg="4x4_50")
    except Exception as e:
        logger.error(f"Failed to initialize LightGlue fallback: {e}")
        logger.error("Make sure PyTorch and LightGlue are installed:")
        logger.error("  pip install torch")
        logger.error("  pip install lightglue")
        return False
    
    if not fallback.enabled:
        logger.error("LightGlue fallback is not enabled (dependencies missing?)")
        return False
    
    logger.info(f"Loaded {len(fallback.templates)} templates")
    if len(fallback.templates) == 0:
        logger.error(f"No templates found in {template_dir}")
        logger.error("Create templates named: id_<MARKER_ID>.png")
        return False
    
    # Simulate missing markers
    detected_dict, expected_ids = simulate_missing_markers(marker_ids)
    
    # Attempt recovery
    logger.info("Attempting to recover missing markers...")
    updated_dict, debug_info = fallback.recover_missing(
        frame,
        detected_dict,
        expected_ids
    )
    
    # Check results
    recovered_ids = set(updated_dict.keys()) - set(detected_dict.keys())
    success = len(recovered_ids) > 0
    
    logger.info("="*60)
    logger.info("Recovery Results:")
    logger.info("="*60)
    logger.info(f"Expected markers: {sorted(expected_ids)}")
    logger.info(f"Initially detected: {sorted(detected_dict.keys())}")
    logger.info(f"Recovered markers: {sorted(recovered_ids)}")
    logger.info(f"Total after recovery: {sorted(updated_dict.keys())}")
    
    logger.info("\nDebug Info:")
    for info in debug_info:
        logger.info(f"  Marker {info.marker_id}: source={info.source}, "
                   f"inliers={info.inliers}, quality={info.match_quality:.3f}")
    
    if success:
        logger.info("\n✓ SUCCESS: Recovered at least one marker!")
        
        # Visualize results
        vis_frame = frame.copy()
        for marker_id, corners in updated_dict.items():
            source = next((d.source for d in debug_info if d.marker_id == marker_id), "unknown")
            
            # Color code
            if source == "aruco":
                color = (0, 255, 0)
            elif source == "lightglue_track":
                color = (255, 255, 0)
            else:
                color = (255, 0, 255)
            
            corners_int = corners.astype(np.int32)
            cv2.polylines(vis_frame, [corners_int], True, color, 2)
            
            center = corners_int.mean(axis=0).astype(np.int32)
            cv2.putText(
                vis_frame,
                f"{marker_id} ({source})",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Save visualization
        output_path = Path(frame_path).parent / "test_lightglue_result.png"
        cv2.imwrite(str(output_path), vis_frame)
        logger.info(f"Saved visualization to: {output_path}")
        
    else:
        logger.error("\n✗ FAILED: Could not recover any markers")
        logger.error("Check that:")
        logger.error("  1. Templates match the markers in the test frame")
        logger.error("  2. Template directory contains id_<MARKER_ID>.png files")
        logger.error("  3. Markers are visible in the test frame")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Test LightGlue fallback for ArUco marker recovery"
    )
    parser.add_argument(
        "--frame",
        type=str,
        required=True,
        help="Path to test frame image (e.g., from data/sessions/*/frames/frame_000001.png)"
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="templates/markers",
        help="Directory containing marker templates (default: templates/markers)"
    )
    parser.add_argument(
        "--marker-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Marker IDs to test (default: 0 1 2 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0"],
        help="Device to use for inference (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Run test
    success = test_fallback_recovery(
        frame_path=args.frame,
        template_dir=args.templates,
        marker_ids=set(args.marker_ids),
        device=args.device
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
