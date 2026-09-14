#!/usr/bin/env python3
"""Validate LightGlue fallback on saved session frames.

This script loads frames from a session folder, runs ArUco detection with
LightGlue fallback, and outputs detection statistics and debug overlays.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from camera_fusion.config import LightGlueConfig
from camera_fusion.detect import build_detector, detect_markers
from camera_fusion.fallback import LightGlueFallback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_session_config(session_dir: Path) -> dict:
    """Load session configuration."""
    config_path = session_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with config_path.open('r') as f:
        return json.load(f)


def validate_session(
    session_dir: str,
    lightglue_config: LightGlueConfig,
    output_dir: str = None,
    max_frames: int = None
) -> dict:
    """Validate LightGlue fallback on session frames.
    
    Args:
        session_dir: Path to session directory
        lightglue_config: LightGlue configuration
        output_dir: Optional output directory for debug overlays
        max_frames: Maximum number of frames to process
    
    Returns:
        Statistics dictionary
    """
    session_path = Path(session_dir)
    
    # Load session config
    try:
        session_config = load_session_config(session_path)
    except Exception as e:
        logger.error(f"Failed to load session config: {e}")
        return {}
    
    # Get expected marker IDs from config
    expected_ids = set(session_config.get("target_ids", []))
    if not expected_ids:
        logger.warning("No target_ids in session config, using all detected IDs")
    
    # Get ArUco dict
    aruco_dict = session_config.get("aruco_dict", "4x4_50")
    
    # Initialize detectors
    detector_state = build_detector(aruco_dict)
    
    # Initialize LightGlue fallback
    fallback = None
    if lightglue_config.enabled:
        try:
            fallback = LightGlueFallback(lightglue_config, aruco_dict)
            if fallback.enabled:
                logger.info("LightGlue fallback initialized")
            else:
                logger.warning("LightGlue fallback failed to initialize")
                fallback = None
        except Exception as e:
            logger.error(f"Failed to initialize LightGlue fallback: {e}")
            fallback = None
    
    # Find frames
    frames_dir = session_path / "frames"
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return {}
    
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        logger.error(f"No frames found in {frames_dir}")
        return {}
    
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    logger.info(f"Processing {len(frame_files)} frames from {session_path.name}")
    
    # Statistics
    stats = {
        "total_frames": len(frame_files),
        "aruco_detections": defaultdict(int),
        "lg_track_recoveries": defaultdict(int),
        "lg_reacquire_recoveries": defaultdict(int),
        "total_detections": defaultdict(int),
        "detection_rate": {},
        "frames_with_all_markers": 0,
        "frames_with_fallback_used": 0,
    }
    
    # Create output directory for debug overlays
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug overlays will be saved to: {output_path}")
    
    # Process frames
    for frame_idx, frame_file in enumerate(frame_files):
        # Load frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            logger.warning(f"Failed to load frame: {frame_file}")
            continue
        
        # Run ArUco detection
        dets = detect_markers(frame, detector_state)
        detected_dict = {d.marker_id: d.corners for d in dets}
        
        # Count ArUco detections
        for marker_id in detected_dict.keys():
            stats["aruco_detections"][marker_id] += 1
        
        # Run LightGlue fallback if enabled and we have expected IDs
        debug_info_list = []
        if fallback and expected_ids:
            detected_dict, debug_info_list = fallback.recover_missing(
                frame,
                detected_dict,
                expected_ids
            )
        
        # Count recoveries by source
        fallback_used = False
        for info in debug_info_list:
            if info.source == "lg_track":
                stats["lg_track_recoveries"][info.marker_id] += 1
                fallback_used = True
            elif info.source == "lg_reacquire":
                stats["lg_reacquire_recoveries"][info.marker_id] += 1
                fallback_used = True
        
        if fallback_used:
            stats["frames_with_fallback_used"] += 1
        
        # Count total detections
        for marker_id in detected_dict.keys():
            stats["total_detections"][marker_id] += 1
        
        # Check if all expected markers detected
        if expected_ids and set(detected_dict.keys()) >= expected_ids:
            stats["frames_with_all_markers"] += 1
        
        # Create debug overlay
        if output_dir:
            debug_frame = frame.copy()
            
            # Build source lookup
            debug_sources = {}
            for info in debug_info_list:
                debug_sources[info.marker_id] = info.source
            
            # Draw markers with color coding
            for marker_id, corners in detected_dict.items():
                source = debug_sources.get(marker_id, "aruco")
                
                # Color code
                if source == "aruco":
                    color = (0, 255, 0)  # Green
                elif source == "lg_track":
                    color = (255, 255, 0)  # Cyan
                else:  # lg_reacquire
                    color = (255, 0, 255)  # Magenta
                
                # Draw corners
                corners_int = corners.astype(np.int32)
                cv2.polylines(debug_frame, [corners_int], True, color, 2)
                
                # Draw ID and source
                center = corners_int.mean(axis=0).astype(np.int32)
                label = f"{marker_id} ({source})"
                cv2.putText(
                    debug_frame,
                    label,
                    tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            
            # Add frame info
            info_text = f"Frame {frame_idx}/{len(frame_files)}"
            cv2.putText(
                debug_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Save
            output_file = output_path / f"debug_frame_{frame_idx:06d}.png"
            cv2.imwrite(str(output_file), debug_frame)
        
        # Progress
        if (frame_idx + 1) % 10 == 0:
            logger.info(f"Processed {frame_idx + 1}/{len(frame_files)} frames")
    
    # Compute detection rates
    for marker_id in stats["total_detections"]:
        stats["detection_rate"][marker_id] = stats["total_detections"][marker_id] / stats["total_frames"]
    
    return stats


def print_statistics(stats: dict):
    """Print validation statistics."""
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    print(f"\nTotal frames processed: {stats['total_frames']}")
    print(f"Frames with all expected markers: {stats['frames_with_all_markers']}")
    print(f"Frames where fallback was used: {stats['frames_with_fallback_used']}")
    
    print("\nPer-Marker Detection Statistics:")
    print("-"*70)
    print(f"{'Marker ID':<12} {'ArUco':<10} {'LG Track':<12} {'LG Reacq':<12} {'Total':<10} {'Rate':<8}")
    print("-"*70)
    
    all_marker_ids = sorted(set(stats["total_detections"].keys()))
    for marker_id in all_marker_ids:
        aruco = stats["aruco_detections"].get(marker_id, 0)
        lg_track = stats["lg_track_recoveries"].get(marker_id, 0)
        lg_reacq = stats["lg_reacquire_recoveries"].get(marker_id, 0)
        total = stats["total_detections"].get(marker_id, 0)
        rate = stats["detection_rate"].get(marker_id, 0.0)
        
        print(f"{marker_id:<12} {aruco:<10} {lg_track:<12} {lg_reacq:<12} {total:<10} {rate*100:>6.1f}%")
    
    print("-"*70)
    
    # Summary
    if stats["frames_with_fallback_used"] > 0:
        print(f"\nFallback Impact:")
        print(f"  Frames where fallback helped: {stats['frames_with_fallback_used']}")
        
        total_recoveries = sum(stats["lg_track_recoveries"].values()) + sum(stats["lg_reacquire_recoveries"].values())
        total_track = sum(stats["lg_track_recoveries"].values())
        total_reacq = sum(stats["lg_reacquire_recoveries"].values())
        
        print(f"  Total recoveries: {total_recoveries}")
        print(f"    Via tracking: {total_track}")
        print(f"    Via reacquire: {total_reacq}")
    else:
        print("\nNo fallback recoveries (all markers detected by ArUco)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate LightGlue fallback on saved session frames"
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Path to session directory (e.g., data/sessions/cam1_session_20260204_120000)"
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="templates/markers",
        help="Template directory (default: templates/markers)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "cuda:0"],
        help="PyTorch device (default: cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for debug overlays (optional)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process (default: all)"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable LightGlue fallback (baseline ArUco only)"
    )
    parser.add_argument(
        "--verify-id",
        action="store_true",
        default=True,
        help="Enable ID verification (default: enabled)"
    )
    parser.add_argument(
        "--no-verify-id",
        action="store_false",
        dest="verify_id",
        help="Disable ID verification"
    )
    
    args = parser.parse_args()
    
    # Create LightGlue config
    lg_config = LightGlueConfig(
        enabled=not args.no_fallback,
        device=args.device,
        template_dir=args.templates,
        verify_id=args.verify_id,
        min_inliers=4,
        max_age_frames=5,
        roi_expand_px=50,
        max_fallback_markers_per_frame=2,
        reacquire_interval_frames=5,
        prefer_roi_matching=True,
        debug_save=False,
        corner_refine=True,
        match_threshold=0.2
    )
    lg_config.debug_matches_dir = str(Path(args.output) / "debug_lightglue_matches") if args.output else str(Path(args.session) / "debug_lightglue_matches")
    
    # Run validation
    stats = validate_session(
        args.session,
        lg_config,
        args.output,
        args.max_frames
    )
    
    if not stats:
        logger.error("Validation failed")
        return 1
    
    # Print results
    print_statistics(stats)
    
    if args.output:
        print(f"\nDebug overlays saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
