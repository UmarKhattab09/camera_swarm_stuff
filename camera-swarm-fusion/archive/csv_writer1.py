import csv
import numpy as np

class CsvWriter:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._opened = False

    def open(self):
        self._fh = open(self.csv_path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow([
            "ts_iso","frame_idx","marker_id",
            "rvec_x","rvec_y","rvec_z",
            "tvec_x","tvec_y","tvec_z",
            "img_path"
        ])
        self._opened = True

    def _vec3(self, vec):
        if vec is None:
            return [float("nan")] * 3
        a = np.array(vec).reshape(-1).tolist()
        # ensure length 3
        if len(a) < 3:
            a += [float("nan")] * (3 - len(a))
        return a[:3]

    # NEW signature: pass marker_id and rvec/tvec separately
    def append(self, ts, frame_idx, marker_id, rvec, tvec, img_path):
        r = self._vec3(rvec)
        t = self._vec3(tvec)
        self._w.writerow([ts, frame_idx, marker_id, *r, *t, img_path])

    def close(self):
        if self._opened:
            self._fh.close()
            self._opened = False
