import json
import os
import time


class EdgeHealthTracker:
    def __init__(self, path: str):
        self.path = path

    def update(self, **values) -> None:
        payload = {"updated_at_ms": int(time.time_ns() / 1e6)}
        payload.update(values)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as health_file:
            json.dump(payload, health_file)

    def read(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as health_file:
            return json.load(health_file)
