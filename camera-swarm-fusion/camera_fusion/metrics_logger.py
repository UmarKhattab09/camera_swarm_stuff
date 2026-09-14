from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median, pstdev
from typing import Any


class CameraMetricsLogger:
    """Structured per-frame and per-session evaluation logging for hybrid detection."""

    FIELDNAMES = [
        "frame_id",
        "timestamp",
        "camera_name",
        "expected_marker_count",
        "required_target_ids",
        "detected_ids",
        "missing_target_ids",
        "condition_label",
        "system_mode",
        "aruco_success",
        "fallback_triggered",
        "fallback_success",
        "detection_source",
        "raw_match_count",
        "ransac_inlier_count",
        "inlier_ratio",
        "pose_recovered",
        "aruco_detect_ms",
        "superpoint_ms",
        "lightglue_ms",
        "ransac_ms",
        "fallback_total_ms",
        "pipeline_total_ms",
        "final_status",
        "image_path",
        "annotated_path",
    ]

    def __init__(self, session_dir: str | Path, flush_every_n_frames: int = 30):
        self.session_dir = Path(session_dir)
        self.flush_every_n_frames = max(1, int(flush_every_n_frames))

        self.csv_path = self.session_dir / "metrics_frames.csv"
        self.summary_path = self.session_dir / "metrics_summary.json"

        self._csv_fp = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_fp, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

        self._rows_since_flush = 0
        self._finalized = False

        self.total_frames = 0
        self.aruco_success_count = 0
        self.fallback_triggered_count = 0
        self.fallback_success_count = 0
        self.total_failure_count = 0

        self._pipeline_total_ms_values: list[float] = []
        self._fallback_total_ms_values: list[float] = []
        self._superpoint_ms_values: list[float] = []
        self._lightglue_ms_values: list[float] = []
        self._ransac_ms_values: list[float] = []
        self._inlier_ratio_values: list[float] = []

        self.counts_by_final_status: dict[str, int] = defaultdict(int)
        self.counts_by_detection_source: dict[str, int] = defaultdict(int)

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / float(len(values))

    @staticmethod
    def _safe_median(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(median(values))

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(pstdev(values))

    @staticmethod
    def _safe_max(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(max(values))

    def log_frame(self, row: dict[str, Any]) -> None:
        row_out = dict(row)

        # Normalize boolean-ish fields to explicit 0/1 values for easy analysis.
        for key in ("aruco_success", "fallback_triggered", "fallback_success", "pose_recovered"):
            row_out[key] = 1 if bool(row_out.get(key, False)) else 0

        row_out["raw_match_count"] = int(row_out.get("raw_match_count", 0))
        row_out["ransac_inlier_count"] = int(row_out.get("ransac_inlier_count", 0))
        row_out["expected_marker_count"] = int(row_out.get("expected_marker_count", 0))

        for key in ("required_target_ids", "detected_ids", "missing_target_ids"):
            value = row_out.get(key, [])
            if isinstance(value, str):
                row_out[key] = value
            else:
                row_out[key] = json.dumps(value)

        for key in (
            "inlier_ratio",
            "aruco_detect_ms",
            "superpoint_ms",
            "lightglue_ms",
            "ransac_ms",
            "fallback_total_ms",
            "pipeline_total_ms",
        ):
            row_out[key] = float(row_out.get(key, 0.0))

        row_out.setdefault("image_path", "")
        row_out.setdefault("annotated_path", "")
        row_out.setdefault("condition_label", "default")
        row_out.setdefault("system_mode", "aruco_only")
        row_out.setdefault("detection_source", "none")
        row_out.setdefault("final_status", "total_failure")

        self._writer.writerow(row_out)

        self.total_frames += 1
        if row_out["aruco_success"] == 1:
            self.aruco_success_count += 1
        if row_out["fallback_triggered"] == 1:
            self.fallback_triggered_count += 1
        if row_out["fallback_success"] == 1:
            self.fallback_success_count += 1
        if row_out["final_status"] == "total_failure":
            self.total_failure_count += 1

        self._pipeline_total_ms_values.append(row_out["pipeline_total_ms"])

        if row_out["fallback_triggered"] == 1:
            self._fallback_total_ms_values.append(row_out["fallback_total_ms"])
            self._superpoint_ms_values.append(row_out["superpoint_ms"])
            self._lightglue_ms_values.append(row_out["lightglue_ms"])
            self._ransac_ms_values.append(row_out["ransac_ms"])
            self._inlier_ratio_values.append(row_out["inlier_ratio"])

        self.counts_by_final_status[str(row_out["final_status"])] += 1
        self.counts_by_detection_source[str(row_out["detection_source"])] += 1

        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every_n_frames:
            self._csv_fp.flush()
            self._rows_since_flush = 0

    def _build_summary(self) -> dict[str, Any]:
        total = self.total_frames
        aruco_success_rate = (self.aruco_success_count / total) if total > 0 else 0.0
        hybrid_success_rate = (
            (self.aruco_success_count + self.fallback_success_count) / total
            if total > 0
            else 0.0
        )
        recovery_rate = (
            self.fallback_success_count / self.fallback_triggered_count
            if self.fallback_triggered_count > 0
            else 0.0
        )

        return {
            "total_frames": total,
            "aruco_success_count": self.aruco_success_count,
            "aruco_full_success_count": self.aruco_success_count,
            "fallback_triggered_count": self.fallback_triggered_count,
            "fallback_success_count": self.fallback_success_count,
            "total_failure_count": self.total_failure_count,
            "aruco_success_rate": aruco_success_rate,
            "hybrid_success_rate": hybrid_success_rate,
            "recovery_rate": recovery_rate,
            "pipeline_latency_ms": {
                "mean": self._safe_mean(self._pipeline_total_ms_values),
                "median": self._safe_median(self._pipeline_total_ms_values),
                "std": self._safe_std(self._pipeline_total_ms_values),
                "max": self._safe_max(self._pipeline_total_ms_values),
            },
            "mean_fallback_latency_ms": self._safe_mean(self._fallback_total_ms_values),
            "mean_superpoint_ms": self._safe_mean(self._superpoint_ms_values),
            "mean_lightglue_ms": self._safe_mean(self._lightglue_ms_values),
            "mean_ransac_ms": self._safe_mean(self._ransac_ms_values),
            "mean_inlier_ratio": self._safe_mean(self._inlier_ratio_values),
            "counts_by_final_status": {
                key: value for key, value in sorted(self.counts_by_final_status.items())
            },
            "counts_by_detection_source": {
                key: value for key, value in sorted(self.counts_by_detection_source.items())
            },
        }

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        try:
            self._csv_fp.flush()
        finally:
            self._csv_fp.close()

        summary = self._build_summary()
        with self.summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
