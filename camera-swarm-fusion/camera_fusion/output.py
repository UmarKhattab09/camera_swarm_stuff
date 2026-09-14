from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from iiot_pipeline.services.csv_writer import CsvWriter


class OutputSink(ABC):
    @abstractmethod
    def open(self, session_dir: Path) -> None: ...

    @abstractmethod
    def write_detection(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
        ref_visible: bool = False,
        ref_rvec = None,
        ref_tvec = None,
        length_m = None,
        capture_time: float | None = None,
    ) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class CsvOutput(OutputSink):
    def __init__(self, filename: str = "detections.csv", use_reference: bool = False, use_length: bool = True):
        self.filename = filename
        self.use_reference = use_reference
        self.use_length = use_length
        self._writer: Optional[CsvWriter] = None

    def open(self, session_dir: Path) -> None:
        path = session_dir / self.filename
        self._writer = CsvWriter(
            str(path), use_reference=self.use_reference, use_length=self.use_length
        )
        self._writer.open()

    def write_detection(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
        ref_visible: bool = False,
        ref_rvec = None,
        ref_tvec = None,
        length_m = None,
        capture_time: float | None = None,
    ) -> None:
        if self._writer is None:
            return
        self._writer.append(
            ts_unix, frame_idx, marker_id, rvec, tvec, image_path,
            ref_visible=ref_visible, ref_rvec=ref_rvec, ref_tvec=ref_tvec,
            length_m=length_m,
            capture_time=capture_time,
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class NullOutput(OutputSink):
    def open(self, session_dir: Path) -> None:
        return None

    def write_detection(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
        ref_visible: bool = False,
        ref_rvec = None,
        ref_tvec = None,
        length_m = None,
        capture_time: float | None = None,
    ) -> None:
        return None

    def close(self) -> None:
        return None


class MqttOutput(OutputSink):
    def __init__(
        self,
        broker_ip: str,
        broker_port: int,
        device_id: str,
        client_type: str,
        camera_name: str,
        use_reference: bool = False,
        use_length: bool = True,
        logger=None,
    ):
        self.broker_ip = broker_ip
        self.broker_port = broker_port
        self.device_id = device_id
        self.client_type = client_type
        self.camera_name = camera_name
        self.use_reference = use_reference
        self.use_length = use_length
        self.logger = logger
        self._client = None
        self._header_published = False

    def _log_warning(self, msg: str, *args) -> None:
        if self.logger is not None:
            self.logger.warning(msg, *args)

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

    def _build_row(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
        ref_visible: bool,
        ref_rvec,
        ref_tvec,
        length_m,
        capture_time: float | None,
    ):
        r = self._vec3(rvec)
        t = self._map_tvec_axes(self._vec3(tvec))
        img = image_path if image_path is not None else ""
        capture_value = float("nan") if capture_time is None else float(capture_time)

        if self.use_reference:
            ref_r = self._vec3(ref_rvec)
            ref_t = self._vec3(ref_tvec)
            row = [
                frame_idx,
                f"{capture_value:.6f}",
                f"{ts_unix:.6f}", marker_id,
                *r, *t,
                1 if ref_visible else 0,
                *ref_r, *ref_t,
            ]
            if self.use_length:
                row.append(length_m)
            row.append(img)
            row.append(self.device_id)
            return row

        row = [
            frame_idx,
            f"{capture_value:.6f}",
            f"{ts_unix:.6f}", marker_id,
            *r, *t,
        ]
        if self.use_length:
            row.append(length_m)
        row.append(img)
        row.append(self.device_id)
        return row

    def _row_to_csv_line(self, row) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(row)
        return buf.getvalue().strip()

    def _header(self) -> list[str]:
        if self.use_reference and self.use_length:
            base = CsvWriter.HEADER_WITH_REF_AND_LENGTH
        elif self.use_reference:
            base = CsvWriter.HEADER_WITH_REF
        elif self.use_length:
            base = CsvWriter.HEADER_WITH_LENGTH
        else:
            base = CsvWriter.HEADER
        return [*base, "device_id"]

    def open(self, session_dir: Path) -> None:
        try:
            try:
                from package.client import Client as DataClient
            except ImportError:
                from facade_sdk import Client as DataClient

            client_type_val = getattr(DataClient, self.client_type, self.client_type)
            effective_device_id = f"{self.device_id}_{self.camera_name}"
            self._client = DataClient(
                broker_ip=self.broker_ip,
                client_type=client_type_val,
                device_id=effective_device_id,
                auto_connect=True,
                broker_port=self.broker_port,
                timeout=60,
            )
        except Exception as exc:
            self._log_warning("MQTT publisher disabled; init failed: %s", exc)
            self._client = None

    def write_detection(
        self,
        ts_unix: float,
        frame_idx: int,
        marker_id: int,
        rvec,
        tvec,
        image_path: Optional[str],
        ref_visible: bool = False,
        ref_rvec = None,
        ref_tvec = None,
        length_m = None,
        capture_time: float | None = None,
    ) -> None:
        if self._client is None:
            return

        try:
            if not self._header_published:
                self._client.publish(",".join(self._header()))
                self._header_published = True

            row = self._build_row(
                ts_unix,
                frame_idx,
                marker_id,
                rvec,
                tvec,
                image_path,
                ref_visible,
                ref_rvec,
                ref_tvec,
                length_m,
                capture_time,
            )
            self._client.publish(self._row_to_csv_line(row))
        except Exception as exc:
            self._log_warning("MQTT publish failed: %s", exc)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
