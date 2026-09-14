#!/usr/bin/env python3
"""
Batch-annotate a folder of saved images with detected ArUco marker borders
and ID labels -- no camera, no live feed, just processes existing .jpg/.png
files and saves annotated copies you can open and look at.

Usage:
    python3 annotate_images.py --input-dir calibimages --output-dir calibimages_annotated --dict 4x4_50
"""
import argparse
import glob
import os

import cv2


ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_250": cv2.aruco.DICT_6X6_250,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dict", default="4x4_50", choices=list(ARUCO_DICTS.keys()))
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.input_dir, "*.png")))
    if not files:
        raise SystemExit(f"No images found in {args.input_dir}")

    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"  could not read {os.path.basename(f)} (skipped)")
            continue

        corners, ids, _ = detector.detectMarkers(img)
        n_markers = 0
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            n_markers = len(ids)

        out_path = os.path.join(args.output_dir, os.path.basename(f))
        cv2.imwrite(out_path, img)
        print(f"{os.path.basename(f)}: {n_markers} markers found -> saved {out_path}")

    print(f"\nDone. Annotated images saved to {args.output_dir}")


if __name__ == "__main__":
    main()