import logging
from typing import Optional


class CameraNameFilter(logging.Filter):
    def __init__(self, camera_name: str):
        super().__init__()
        self.camera_name = camera_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.camera = self.camera_name
        return True


def setup_logger(camera_name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"camera_fusion.{camera_name}")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [%(camera)s] %(message)s"
        )
        handler.setFormatter(fmt)
        handler.addFilter(CameraNameFilter(camera_name))
        logger.addHandler(handler)

    return logger


def add_file_handler(logger: logging.Logger, camera_name: str, log_path: str) -> None:
    handler = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(camera)s] %(message)s")
    handler.setFormatter(fmt)
    handler.addFilter(CameraNameFilter(camera_name))
    logger.addHandler(handler)
