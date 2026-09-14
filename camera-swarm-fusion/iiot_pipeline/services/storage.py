from pathlib import Path
import json, cv2

class SessionStorage:
    def __init__(self, root: str, name: str = "session"):
        self.root = Path(root)
        self.session_dir = None
        self.frames_dir = None
        self.annotated_dir = None  # NEW
        self.logs_dir = None
        self.last_path = None
        self.name = name

    def begin(self) -> str:
        from time import strftime
        sid = f"{self.name}_{strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = self.root / sid
        self.frames_dir = self.session_dir / "frames"
        self.annotated_dir = self.session_dir / "annotated"   # NEW
        self.logs_dir = self.session_dir / "logs"
        for d in (self.frames_dir, self.annotated_dir, self.logs_dir):   # NEW
            d.mkdir(parents=True, exist_ok=True)
        return str(self.session_dir)

    def save_frame(self, f):
        """Save the UNDISTORTED ORIGINAL (no drawings)."""
        p = self.frames_dir / f"f{f.idx:06d}.jpg"
        cv2.imwrite(str(p), f.image)
        self.last_path = str(p)

    # NEW: save annotated copy (with drawings)
    def save_annotated(self, idx: int, image):
        p = self.annotated_dir / f"f{idx:06d}_aruco.jpg"
        cv2.imwrite(str(p), image)
        return str(p)

    def write_manifest(self, meta: dict):
        with open(self.session_dir / "config.json", "w") as fp:
            json.dump(meta, fp, indent=2)
