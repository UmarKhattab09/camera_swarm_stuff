#!/usr/bin/env python3
import cv2, glob, csv, argparse
from pathlib import Path

def get_dict(name: str):
    name = name.strip().lower()
    table = {
        "4x4_50":  cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_50":  cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_50":  cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
        "7x7_50":  cv2.aruco.DICT_7X7_50,
        "7x7_100": cv2.aruco.DICT_7X7_100,
        "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    key = name if name in table else "4x4_50"
    return cv2.aruco.Dictionary_get(table[key]), key

def main():
    ap = argparse.ArgumentParser(description="Undistort frames then detect ArUco markers (batch).")
    ap.add_argument("--in", dest="in_dir",
                    default=None,
                    help="Input frames dir (default: data/sessions/latest/frames)")
    ap.add_argument("--out-name", default=None,
                    help="Output folder name under data/aruco/ (default: derived from input)")
    ap.add_argument("--dict", default="4x4_50",
                    help="ArUco dictionary: 4x4_50, 4x4_100, 5x5_100, apriltag_36h11, ...")
    ap.add_argument("--marker-length", type=float, default=0.0,
                    help="Marker side length in meters for pose (0=skip pose)")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="Undistort alpha (0.0=crop valid; 0.5 some FOV; 1.0 max FOV w/ borders)")
    ap.add_argument("--calib", default=None,
                    help="Calibration file path (default: calib/c920s_1920x1080_simple.yml)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]

    # Resolve paths
    in_dir = Path(args.in_dir) if args.in_dir else (root / "data" / "sessions" / "latest" / "frames")
    calib  = Path(args.calib) if args.calib else (root / "calib" / "c920s_1920x1080_simple.yml")

    if not in_dir.is_dir():
        raise SystemExit(f"Input dir not found: {in_dir}")
    if not calib.exists():
        raise SystemExit(f"Calibration file not found: {calib}")

    # Output folder name
    if args.out_name:
        out_name = args.out_name
    else:
        # derive from session path (parent of frames dir)
        session = in_dir.parent if in_dir.name == "frames" else in_dir
        out_name = f"aruco_{session.parent.name}_{session.name}".replace("/", "_")

    out_root = root / "data" / "aruco" / out_name
    out_annot = out_root / "annotated"
    out_root.mkdir(parents=True, exist_ok=True)
    out_annot.mkdir(parents=True, exist_ok=True)

    # Load calibration
    fs = cv2.FileStorage(str(calib), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    w = int(fs.getNode("image_width").real()); h = int(fs.getNode("image_height").real())
    fs.release()
    if K is None or dist is None or w == 0 or h == 0:
        raise SystemExit("Failed to read K/dist/size from calibration file.")

    # Undistortion maps
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), args.alpha)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), cv2.CV_16SC2)

    # Collect input frames
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files = []
    for e in exts: files.extend(glob.glob(str(in_dir / e)))
    files.sort()
    if not files:
        raise SystemExit(f"No images found in {in_dir}")

    # ArUco setup (OpenCV 4.6-compatible API)
    dictionary, dict_used = get_dict(args.dict)
    params = cv2.aruco.DetectorParameters_create()

    # CSV of detections
    csv_path = out_root / "detections.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_file","marker_id","rvec_x","rvec_y","rvec_z","tvec_x","tvec_y","tvec_z"])

    num_imgs = len(files)
    total_dets = 0

    for idx, f in enumerate(files, 1):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            print("Skipping unreadable:", f); continue
        ih, iw = img.shape[:2]
        if (iw, ih) != (w, h):
            print("Skipping (wrong size):", f, f"(got {iw}x{ih}, need {w}x{h})")
            continue

        und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

        corners, ids, _ = cv2.aruco.detectMarkers(und, dictionary, parameters=params)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(und, corners, ids)

            if args.marker_length > 0:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, args.marker_length, K, dist
                )
                # Draw pose axes and write CSV
                for i, mid in enumerate(ids.flatten()):
                    rvec = rvecs[i].reshape(-1)
                    tvec = tvecs[i].reshape(-1)
                    try:
                        cv2.drawFrameAxes(und, K, dist, rvec, tvec, args.marker_length*0.5)
                    except Exception:
                        pass
                    writer.writerow([Path(f).name, int(mid), *rvec.tolist(), *tvec.tolist()])
                    total_dets += 1
            else:
                # Write CSV rows with NaNs for pose
                import math
                for mid in ids.flatten():
                    writer.writerow([Path(f).name, int(mid), *([float("nan")]*6)])
                    total_dets += 1

        # Save annotated image
        outp = out_annot / (Path(f).stem + "_aruco.jpg")
        cv2.imwrite(str(outp), und)

        if idx % 50 == 0 or idx == num_imgs:
            print(f"[{idx}/{num_imgs}] processed")

    csv_file.close()

    # Write a small manifest
    with open(out_root / "manifest.txt", "w") as mf:
        mf.write(f"input={in_dir}\n")
        mf.write(f"calib={calib}\n")
        mf.write(f"dictionary={dict_used}\n")
        mf.write(f"marker_length_m={args.marker_length}\n")
        mf.write(f"alpha={args.alpha}\n")
        mf.write(f"images={num_imgs}\n")
        mf.write(f"detections={total_dets}\n")

    print("Done.")
    print("Annotated frames ->", out_annot)
    print("Detections CSV   ->", csv_path)

if __name__ == "__main__":
    main()
