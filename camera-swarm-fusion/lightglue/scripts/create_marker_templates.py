#!/usr/bin/env python3
"""Helper script to create marker templates for LightGlue fallback.

This script generates ArUco marker templates and saves them as PNG images
that can be used by the LightGlue fallback system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def get_aruco_dict(dict_name: str):
    """Get OpenCV ArUco dictionary."""
    name = (dict_name or "").strip()
    if name.upper().startswith("DICT_"):
        name = name[5:]
    name = name.lower()
    
    # Map common names to OpenCV constants
    dict_map = {
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "4x4_250": cv2.aruco.DICT_4X4_250,
        "4x4_1000": cv2.aruco.DICT_4X4_1000,
        "5x5_50": cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "5x5_250": cv2.aruco.DICT_5X5_250,
        "5x5_1000": cv2.aruco.DICT_5X5_1000,
        "6x6_50": cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
        "6x6_250": cv2.aruco.DICT_6X6_250,
        "6x6_1000": cv2.aruco.DICT_6X6_1000,
        "7x7_50": cv2.aruco.DICT_7X7_50,
        "7x7_100": cv2.aruco.DICT_7X7_100,
        "7x7_250": cv2.aruco.DICT_7X7_250,
        "7x7_1000": cv2.aruco.DICT_7X7_1000,
    }
    
    if name not in dict_map:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    
    return cv2.aruco.getPredefinedDictionary(dict_map[name])


def create_marker_template(
    marker_id: int,
    aruco_dict,
    size_px: int = 200,
    border_bits: int = 1
) -> np.ndarray:
    """Create a marker template image.
    
    Args:
        marker_id: ArUco marker ID
        aruco_dict: OpenCV ArUco dictionary
        size_px: Size of the marker in pixels
        border_bits: Border size in bits (default 1 for standard ArUco)
    
    Returns:
        Marker image (grayscale)
    """
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        marker_id,
        size_px,
        borderBits=border_bits
    )
    return marker_img


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker templates for LightGlue fallback"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="templates/markers",
        help="Output directory for templates (default: templates/markers)"
    )
    parser.add_argument(
        "--marker-ids",
        type=int,
        nargs="+",
        required=True,
        help="Marker IDs to generate (e.g., 0 1 2 3)"
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="4x4_50",
        help="ArUco dictionary (default: 4x4_50)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=400,
        help="Template size in pixels (default: 400)"
    )
    parser.add_argument(
        "--border-bits",
        type=int,
        default=1,
        help="Border size in bits (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get ArUco dictionary
    try:
        aruco_dict = get_aruco_dict(args.dict)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    print(f"Generating templates for ArUco dictionary: {args.dict}")
    print(f"Output directory: {output_dir}")
    print(f"Marker IDs: {args.marker_ids}")
    print(f"Template size: {args.size}px")
    print()
    
    # Generate templates
    for marker_id in args.marker_ids:
        template = create_marker_template(
            marker_id,
            aruco_dict,
            args.size,
            args.border_bits
        )
        
        output_path = output_dir / f"id_{marker_id}.png"
        cv2.imwrite(str(output_path), template)
        print(f"✓ Created template: {output_path}")
    
    print()
    print(f"Successfully generated {len(args.marker_ids)} templates!")
    print()
    print("Usage:")
    print(f"  1. Configure lightglue.template_dir = '{output_dir}' in your config")
    print(f"  2. Enable lightglue.enabled = true")
    print(f"  3. Run your camera fusion application")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
