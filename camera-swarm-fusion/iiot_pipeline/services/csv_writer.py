import csv
import numpy as np
import io

class CsvWriter:
    # Match mock.py header order
    HEADER = [
        "frame_idx",
        "capture_time",
        "recorded_at", "marker_id",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x", "tvec_y", "tvec_z",
        "image_path"
    ]

    HEADER_WITH_LENGTH = [
        "frame_idx",
        "capture_time",
        "recorded_at", "marker_id",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x", "tvec_y", "tvec_z",
        "length_m",
        "image_path",
    ]

    HEADER_WITH_REF = [
        "frame_idx",
        "capture_time",
        "recorded_at", "marker_id",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x", "tvec_y", "tvec_z",
        "ref_visible",
        "ref_rvec_x", "ref_rvec_y", "ref_rvec_z",
        "ref_tvec_x", "ref_tvec_y", "ref_tvec_z",
        "image_path"
    ]

    HEADER_WITH_REF_AND_LENGTH = [
        "frame_idx",
        "capture_time",
        "recorded_at", "marker_id",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x", "tvec_y", "tvec_z",
        "ref_visible",
        "ref_rvec_x", "ref_rvec_y", "ref_rvec_z",
        "ref_tvec_x", "ref_tvec_y", "ref_tvec_z",
        "length_m",
        "image_path",
    ]

    def __init__(self, csv_path: str, use_reference: bool = False, use_length: bool = False):
        self.csv_path = csv_path
        self.use_reference = use_reference
        self.use_length = use_length
        self._opened = False
        self._fh = None
        self._w = None

    def open(self):
        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._fh)
        if self.use_reference and self.use_length:
            header = self.HEADER_WITH_REF_AND_LENGTH
        elif self.use_reference:
            header = self.HEADER_WITH_REF
        elif self.use_length:
            header = self.HEADER_WITH_LENGTH
        else:
            header = self.HEADER
        self._w.writerow(header)
        self._opened = True

    def _vec3(self, vec):
        if vec is None:
            return [float("nan")] * 3
        a = np.array(vec).reshape(-1).tolist()
        if len(a) < 3:
            a += [float("nan")] * (3 - len(a))
        return a[:3]

    def _map_tvec_axes(self, tvec_xyz):
        x, y, z = tvec_xyz
        return [y, z, x]

    def append(
        self,
        ts_unix,
        frame_idx,
        marker_id,
        rvec,
        tvec,
        img_path,
        ref_visible=None,
        ref_rvec=None,
        ref_tvec=None,
        length_m=None,
        capture_time=None,
    ):
        r = self._vec3(rvec)
        t = self._map_tvec_axes(self._vec3(tvec))
        capture_value = float("nan") if capture_time is None else float(capture_time)
        
        if self.use_reference:
            ref_r = self._vec3(ref_rvec)
            ref_t = self._vec3(ref_tvec)
            ref_vis = 1 if ref_visible else 0
            row = [
                frame_idx,
                f"{capture_value:.6f}",
                f"{ts_unix:.6f}", marker_id,
                *r, *t,
                ref_vis,
                *ref_r, *ref_t,
            ]
            if self.use_length:
                row.extend([length_m])
            row.append(img_path)
            self._w.writerow(row)
        else:
            row = [
                frame_idx,
                f"{capture_value:.6f}",
                f"{ts_unix:.6f}", marker_id,
                *r, *t,
            ]
            if self.use_length:
                row.extend([length_m])
            row.append(img_path)
            self._w.writerow(row)

    @classmethod
    def to_csv_line(
        cls,
        ts_unix,
        frame_idx,
        marker_id,
        rvec,
        tvec,
        img_path,
        length_m=None,
        capture_time=None,
    ):
        def _vec3_local(vec):
            if vec is None:
                return [float("nan"), float("nan"), float("nan")]
            a = np.array(vec).reshape(-1).tolist()
            if len(a) < 3:
                a += [float("nan")] * (3 - len(a))
            return a[:3]

        def _map_tvec_axes_local(tvec_xyz):
            x, y, z = tvec_xyz
            return [y, z, x]

        r = _vec3_local(rvec)
        t = _map_tvec_axes_local(_vec3_local(tvec))
        capture_value = float("nan") if capture_time is None else float(capture_time)
        buf = io.StringIO()
        w = csv.writer(buf)
        row = [
            frame_idx,
            f"{capture_value:.6f}",
            f"{ts_unix:.6f}", marker_id,
            *r, *t,
        ]
        if length_m is not None:
            row.extend([length_m])
        row.append(img_path)
        w.writerow(row)
        return buf.getvalue().strip()

    def close(self):
        if self._opened and self._fh:
            self._fh.close()
            self._opened = False
            self._fh = None
            self._w = None
